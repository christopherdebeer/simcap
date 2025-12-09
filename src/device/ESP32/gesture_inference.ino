/**
 * SIMCAP Gesture Inference - ESP32 Firmware
 * 
 * TinyML gesture classification using TensorFlow Lite Micro
 * 
 * Hardware:
 *   - ESP32-S3 or ESP32-C3 (recommended: S3 for more RAM)
 *   - MPU9250 or LSM9DS1 9-DoF IMU
 * 
 * Dependencies:
 *   - TensorFlow Lite Micro for ESP32
 *   - Wire.h (I2C)
 * 
 * Build:
 *   1. Copy gesture_model.h from ml/models/ to this directory
 *   2. Install TFLite Micro library
 *   3. Upload via Arduino IDE or PlatformIO
 */

#include <Wire.h>
#include "gesture_model.h"

// Uncomment the appropriate TFLite includes based on your setup
// For Arduino:
// #include <TensorFlowLite_ESP32.h>
// For ESP-IDF:
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// ============================================================================
// Configuration
// ============================================================================

#define SAMPLE_RATE_HZ 50
#define INFERENCE_INTERVAL_MS (1000 / SAMPLE_RATE_HZ)
#define INFERENCE_STRIDE 25  // Run inference every N samples

// IMU I2C address (adjust for your sensor)
#define IMU_ADDRESS 0x68  // MPU9250 default

// BLE configuration
#define BLE_DEVICE_NAME "SIMCAP-ESP32"
#define BLE_SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define BLE_GESTURE_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// ============================================================================
// TensorFlow Lite Micro Setup
// ============================================================================

#ifdef USE_TFLITE

constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

#endif

// ============================================================================
// Sliding Window Buffer
// ============================================================================

float window_buffer[GESTURE_MODEL_WINDOW_SIZE][GESTURE_MODEL_NUM_FEATURES];
int buffer_write_idx = 0;
int sample_count = 0;
bool buffer_full = false;

// ============================================================================
// Current State
// ============================================================================

int current_gesture = 0;
float current_confidence = 0.0f;
unsigned long last_sample_time = 0;
unsigned long last_inference_time = 0;

// ============================================================================
// IMU Functions
// ============================================================================

struct IMUData {
    float ax, ay, az;  // Accelerometer (m/s²)
    float gx, gy, gz;  // Gyroscope (deg/s)
    float mx, my, mz;  // Magnetometer (µT)
};

IMUData imu_data;

void initIMU() {
    Wire.begin();
    
    // Initialize IMU (example for MPU9250)
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(0x6B);  // PWR_MGMT_1
    Wire.write(0x00);  // Wake up
    Wire.endTransmission(true);
    
    // Configure accelerometer (±2g)
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(0x1C);  // ACCEL_CONFIG
    Wire.write(0x00);
    Wire.endTransmission(true);
    
    // Configure gyroscope (±250 deg/s)
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(0x1B);  // GYRO_CONFIG
    Wire.write(0x00);
    Wire.endTransmission(true);
    
    Serial.println("IMU initialized");
}

void readIMU() {
    // Read accelerometer
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(0x3B);  // ACCEL_XOUT_H
    Wire.endTransmission(false);
    Wire.requestFrom(IMU_ADDRESS, 6, true);
    
    int16_t ax_raw = (Wire.read() << 8) | Wire.read();
    int16_t ay_raw = (Wire.read() << 8) | Wire.read();
    int16_t az_raw = (Wire.read() << 8) | Wire.read();
    
    // Convert to m/s² (assuming ±2g range)
    imu_data.ax = ax_raw / 16384.0f * 9.81f;
    imu_data.ay = ay_raw / 16384.0f * 9.81f;
    imu_data.az = az_raw / 16384.0f * 9.81f;
    
    // Read gyroscope
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(0x43);  // GYRO_XOUT_H
    Wire.endTransmission(false);
    Wire.requestFrom(IMU_ADDRESS, 6, true);
    
    int16_t gx_raw = (Wire.read() << 8) | Wire.read();
    int16_t gy_raw = (Wire.read() << 8) | Wire.read();
    int16_t gz_raw = (Wire.read() << 8) | Wire.read();
    
    // Convert to deg/s (assuming ±250 deg/s range)
    imu_data.gx = gx_raw / 131.0f;
    imu_data.gy = gy_raw / 131.0f;
    imu_data.gz = gz_raw / 131.0f;
    
    // Magnetometer (placeholder - depends on your sensor)
    imu_data.mx = 0;
    imu_data.my = 0;
    imu_data.mz = 0;
}

// ============================================================================
// Inference Functions
// ============================================================================

void addSampleToBuffer(const IMUData& data) {
    // Normalize using training statistics
    window_buffer[buffer_write_idx][0] = (data.ax - GESTURE_MEAN[0]) / GESTURE_STD[0];
    window_buffer[buffer_write_idx][1] = (data.ay - GESTURE_MEAN[1]) / GESTURE_STD[1];
    window_buffer[buffer_write_idx][2] = (data.az - GESTURE_MEAN[2]) / GESTURE_STD[2];
    window_buffer[buffer_write_idx][3] = (data.gx - GESTURE_MEAN[3]) / GESTURE_STD[3];
    window_buffer[buffer_write_idx][4] = (data.gy - GESTURE_MEAN[4]) / GESTURE_STD[4];
    window_buffer[buffer_write_idx][5] = (data.gz - GESTURE_MEAN[5]) / GESTURE_STD[5];
    window_buffer[buffer_write_idx][6] = (data.mx - GESTURE_MEAN[6]) / GESTURE_STD[6];
    window_buffer[buffer_write_idx][7] = (data.my - GESTURE_MEAN[7]) / GESTURE_STD[7];
    window_buffer[buffer_write_idx][8] = (data.mz - GESTURE_MEAN[8]) / GESTURE_STD[8];
    
    buffer_write_idx = (buffer_write_idx + 1) % GESTURE_MODEL_WINDOW_SIZE;
    sample_count++;
    
    if (sample_count >= GESTURE_MODEL_WINDOW_SIZE) {
        buffer_full = true;
    }
}

#ifdef USE_TFLITE

void initModel() {
    model = tflite::GetModel(gesture_model_tflite);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        return;
    }
    
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddRelu();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMean();  // For GlobalAveragePooling
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed!");
        return;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.printf("Model loaded. Input shape: [%d, %d, %d]\n",
                  input->dims->data[0], input->dims->data[1], input->dims->data[2]);
    Serial.printf("Tensor arena used: %d bytes\n", interpreter->arena_used_bytes());
}

int runInference() {
    if (!buffer_full || interpreter == nullptr) {
        return -1;
    }
    
    unsigned long start_time = micros();
    
    // Copy window buffer to input tensor (circular buffer unwrap)
    int read_idx = buffer_write_idx;  // Start from oldest sample
    for (int i = 0; i < GESTURE_MODEL_WINDOW_SIZE; i++) {
        for (int j = 0; j < GESTURE_MODEL_NUM_FEATURES; j++) {
            input->data.f[i * GESTURE_MODEL_NUM_FEATURES + j] = window_buffer[read_idx][j];
        }
        read_idx = (read_idx + 1) % GESTURE_MODEL_WINDOW_SIZE;
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return -1;
    }
    
    // Find max probability
    int max_idx = 0;
    float max_prob = output->data.f[0];
    for (int i = 1; i < GESTURE_MODEL_NUM_CLASSES; i++) {
        if (output->data.f[i] > max_prob) {
            max_prob = output->data.f[i];
            max_idx = i;
        }
    }
    
    current_gesture = max_idx;
    current_confidence = max_prob;
    last_inference_time = micros() - start_time;
    
    return max_idx;
}

#else

// Fallback: Simple centroid-based classification (no TFLite)
// Uses pre-computed cluster centroids from training

float CENTROIDS[10][6] = {
    {-0.30f, 0.56f, 0.11f, 0.14f, -0.02f, -0.11f},   // Cluster 0 - fist
    {0.96f, 0.32f, 1.18f, -0.05f, 0.29f, 0.34f},    // Cluster 1 - thumbs_up
    {0.12f, 0.62f, -0.83f, -0.09f, -0.16f, -0.02f}, // Cluster 2 - peace
    {0.25f, 0.57f, 1.01f, 2.29f, 0.91f, 0.00f},     // Cluster 3 - rest
    {0.29f, 1.05f, -0.65f, 0.15f, -0.08f, -0.05f},  // Cluster 4 - open_palm
    {-0.00f, -0.61f, 0.15f, -0.10f, 0.02f, -0.02f}, // Cluster 5 - index_up
    {-0.01f, -0.59f, 0.17f, 0.15f, 0.08f, -0.02f},  // Cluster 6 - grab
    {-0.52f, 1.25f, -0.43f, -0.07f, -0.12f, -0.03f},// Cluster 7 - pinch
    {-0.53f, 0.98f, -0.63f, -0.59f, -0.28f, -0.03f},// Cluster 8 - ok_sign
    {-0.12f, -0.75f, 0.35f, -0.04f, 0.01f, 0.02f}   // Cluster 9 - rest
};

void initModel() {
    Serial.println("Using centroid-based classification (no TFLite)");
}

int runInference() {
    if (!buffer_full) {
        return -1;
    }
    
    unsigned long start_time = micros();
    
    // Calculate mean features over window
    float features[6] = {0};
    for (int i = 0; i < GESTURE_MODEL_WINDOW_SIZE; i++) {
        features[0] += window_buffer[i][0];  // ax
        features[1] += window_buffer[i][1];  // ay
        features[2] += window_buffer[i][2];  // az
        features[3] += window_buffer[i][3];  // gx
        features[4] += window_buffer[i][4];  // gy
        features[5] += window_buffer[i][5];  // gz
    }
    for (int i = 0; i < 6; i++) {
        features[i] /= GESTURE_MODEL_WINDOW_SIZE;
    }
    
    // Find nearest centroid
    float min_dist = 1e9;
    int best_cluster = 0;
    
    for (int c = 0; c < 10; c++) {
        float dist = 0;
        for (int i = 0; i < 6; i++) {
            float d = features[i] - CENTROIDS[c][i];
            dist += d * d;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }
    
    current_gesture = best_cluster;
    current_confidence = 1.0f / (1.0f + min_dist);  // Convert distance to confidence
    last_inference_time = micros() - start_time;
    
    return best_cluster;
}

#endif

// ============================================================================
// BLE Functions (optional)
// ============================================================================

#ifdef USE_BLE
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

BLEServer* pServer = nullptr;
BLECharacteristic* pGestureCharacteristic = nullptr;
bool deviceConnected = false;

class ServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        Serial.println("BLE client connected");
    }
    
    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        Serial.println("BLE client disconnected");
        pServer->startAdvertising();
    }
};

void initBLE() {
    BLEDevice::init(BLE_DEVICE_NAME);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());
    
    BLEService* pService = pServer->createService(BLE_SERVICE_UUID);
    
    pGestureCharacteristic = pService->createCharacteristic(
        BLE_GESTURE_UUID,
        BLECharacteristic::PROPERTY_READ |
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pGestureCharacteristic->addDescriptor(new BLE2902());
    
    pService->start();
    
    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(BLE_SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->start();
    
    Serial.println("BLE advertising started");
}

void notifyGesture() {
    if (deviceConnected && pGestureCharacteristic != nullptr) {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "{\"g\":%d,\"c\":%.2f,\"t\":%lu}",
                 current_gesture, current_confidence, last_inference_time);
        pGestureCharacteristic->setValue(buffer);
        pGestureCharacteristic->notify();
    }
}
#endif

// ============================================================================
// Main Setup and Loop
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("SIMCAP Gesture Inference - ESP32");
    Serial.println("========================================\n");
    
    // Initialize IMU
    initIMU();
    
    // Initialize ML model
    initModel();
    
    #ifdef USE_BLE
    initBLE();
    #endif
    
    Serial.println("\nSetup complete. Starting inference loop...\n");
}

void loop() {
    unsigned long current_time = millis();
    
    // Sample at fixed rate
    if (current_time - last_sample_time >= INFERENCE_INTERVAL_MS) {
        last_sample_time = current_time;
        
        // Read IMU
        readIMU();
        
        // Add to buffer
        addSampleToBuffer(imu_data);
        
        // Run inference every STRIDE samples
        if (sample_count % INFERENCE_STRIDE == 0 && buffer_full) {
            int gesture = runInference();
            
            if (gesture >= 0) {
                Serial.printf("Gesture: %s (%.1f%%) [%lu µs]\n",
                             GESTURE_LABELS[gesture],
                             current_confidence * 100.0f,
                             last_inference_time);
                
                #ifdef USE_BLE
                notifyGesture();
                #endif
            }
        }
    }
}
