/**
 * Core telemetry types for SIMCAP
 *
 * These types define the structure of sensor data throughout the system.
 */

/** 3D vector with x, y, z components */
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/** Quaternion for 3D rotation representation */
export interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

/** Euler angles in degrees */
export interface EulerAngles {
  roll: number;   // Rotation around X axis
  pitch: number;  // Rotation around Y axis
  yaw: number;    // Rotation around Z axis
}

/** Raw sensor reading in LSB (Least Significant Bits) from device */
export interface RawTelemetry {
  ax: number;  // Accelerometer X (LSB)
  ay: number;  // Accelerometer Y (LSB)
  az: number;  // Accelerometer Z (LSB)
  gx: number;  // Gyroscope X (LSB)
  gy: number;  // Gyroscope Y (LSB)
  gz: number;  // Gyroscope Z (LSB)
  mx: number;  // Magnetometer X (LSB)
  my: number;  // Magnetometer Y (LSB)
  mz: number;  // Magnetometer Z (LSB)
  t: number;   // Timestamp (ms since connection)
}

/** Telemetry converted to physical units */
export interface PhysicalTelemetry {
  accel: Vector3;      // Acceleration in g's
  gyro: Vector3;       // Angular velocity in deg/s
  mag: Vector3;        // Magnetic field in μT
  timestamp: number;   // Timestamp in ms
}

/** Processed telemetry with orientation */
export interface ProcessedTelemetry extends RawTelemetry {
  orientation?: Quaternion;
  euler?: EulerAngles;
}

/** Telemetry sample stored in session (extends raw with optional computed fields) */
export interface TelemetrySample extends RawTelemetry {
  orientation?: Quaternion;
  fingerMagnet?: {
    detected: boolean;
    confidence: number;
  };
}
