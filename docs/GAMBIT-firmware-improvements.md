# GAMBIT Firmware Improvement Proposals

**Date:** December 2025
**Status:** Proposal
**Target:** GAMBIT Firmware v0.4.0+

This document outlines dramatic improvements to the GAMBIT firmware across five key areas:
1. Higher Resolution & Reduced Battery Consumption
2. Enhanced Button Interactions
3. RGB LED Feedback System
4. Light & Capacitive Sensor Applications
5. Advanced BLE Capabilities

---

## Executive Summary

The current GAMBIT firmware (v0.3.6) provides solid 9-DoF IMU streaming at 20-26Hz with basic power management. However, significant untapped potential exists in the hardware:

| Area | Current State | Proposed State | Impact |
|------|---------------|----------------|--------|
| Resolution | 20-26Hz streaming | 100-400Hz with FIFO batching | **5-20x improvement** |
| Battery | ~30s burst, then idle | Adaptive sampling, 10x longer sessions | **10x battery life** |
| Button | Single press toggle | Multi-tap gestures, long-press modes | **8x more commands** |
| LEDs | Basic state feedback | Status encoding, alerts, orientation | **Rich feedback** |
| Light Sensor | 2Hz logging only | Context awareness, power optimization | **Smart adaptation** |
| Capacitive | 2Hz logging only | Grip detection, gesture input | **New input modality** |
| BLE | 20Hz JSON streaming | Batch transfers, binary protocol | **5x throughput** |

---

## 1. Higher Resolution & Reduced Battery Consumption

### 1.1 Current Limitations

The firmware currently:
- Samples at 20-26Hz (50ms intervals)
- Reads sensors individually on each interval
- Magnetometer sampled at 10Hz (every 2nd sample)
- No hardware FIFO utilization
- Always-on streaming during capture (30s bursts)

**Power consumption problem:** Each sensor read requires I2C wake/read/sleep cycles, consuming ~800µA with default firmware vs possible 40µA with optimization.

### 1.2 LSM6DS3 FIFO Batch Mode (HIGH IMPACT)

The LSM6DS3 IMU has an **8KB FIFO buffer** that is completely unused. This enables dramatic improvements:

```javascript
// PROPOSED: Enable hardware FIFO batching
function enableFifoBatching(odr) {
    // Configure LSM6DS3 FIFO mode
    // ODR options: 13, 26, 52, 104, 208, 416, 833, 1660 Hz

    // Write to FIFO_CTRL5 (0x0A): Enable FIFO, set ODR
    Puck.i2cWr(0x6A, 0x0A, 0x20 | (odr << 3));  // Continuous mode

    // Write to FIFO_CTRL3 (0x08): Enable accel+gyro decimation
    Puck.i2cWr(0x6A, 0x08, 0x09);  // Both sensors, no decimation

    // FIFO can hold ~682 samples (12 bytes each)
    // At 104Hz = 6.5 seconds of data
    // At 416Hz = 1.6 seconds of data
}

// PROPOSED: Burst read FIFO when threshold reached
function readFifoBuffer() {
    // Read FIFO_STATUS1/2 for sample count
    var status = Puck.i2cRd(0x6A, 0x3A, 2);
    var samples = (status[1] & 0x0F) << 8 | status[0];

    // Bulk read all samples (12 bytes each: 3x accel + 3x gyro)
    var data = [];
    for (var i = 0; i < samples; i++) {
        var raw = Puck.i2cRd(0x6A, 0x3E, 12);
        data.push(parseSample(raw));
    }
    return data;  // Send as batch over BLE
}
```

**Benefits:**
- **Resolution:** Up to 416Hz sampling (vs current 26Hz) = **16x improvement**
- **Power:** CPU sleeps while FIFO fills, reducing active time by 90%
- **Data quality:** Hardware timestamping, no jitter from JS execution

### 1.3 Adaptive Sampling Modes

```javascript
// PROPOSED: Context-aware sampling rates
var SAMPLING_MODES = {
    ULTRA_LOW_POWER: {
        accelHz: 1.6,      // Motion detection only
        gyroEnabled: false,
        magHz: 0,
        sleepBetween: true,
        description: "Pocket/sleep detection"
    },
    LOW_POWER: {
        accelHz: 26,
        gyroEnabled: false,  // Gyro uses 2x power
        magHz: 1,
        description: "Activity classification"
    },
    NORMAL: {
        accelHz: 104,
        gyroEnabled: true,
        magHz: 10,
        description: "Gesture tracking"
    },
    HIGH_RESOLUTION: {
        accelHz: 416,
        gyroEnabled: true,
        magHz: 20,
        fifoEnabled: true,
        description: "Fine motor capture"
    },
    BURST: {
        accelHz: 833,
        gyroEnabled: true,
        magHz: 100,
        fifoEnabled: true,
        duration: 5000,  // 5 second bursts
        description: "Maximum detail capture"
    }
};

function setMode(modeName) {
    var mode = SAMPLING_MODES[modeName];
    if (mode.fifoEnabled) {
        enableFifoBatching(mode.accelHz);
    }
    // ... configure sensors
    logInfo('Mode: ' + modeName);
}
```

### 1.4 Power Budget Analysis

| Mode | Accel | Gyro | Mag | CPU | BLE | Total | Duration (CR2032) |
|------|-------|------|-----|-----|-----|-------|-------------------|
| Current (streaming) | 300µA | 400µA | 100µA | 3mA | 2mA | ~6mA | 35 min |
| ULTRA_LOW_POWER | 3µA | 0 | 0 | 10µA | 50µA | ~65µA | 54 hours |
| LOW_POWER | 40µA | 0 | 20µA | 100µA | 200µA | ~360µA | 9 hours |
| NORMAL + FIFO | 100µA | 200µA | 50µA | 50µA | 500µA | ~900µA | 3.7 hours |
| HIGH_RES + FIFO | 200µA | 400µA | 100µA | 100µA | 1mA | ~1.8mA | 1.8 hours |

**Key insight:** FIFO batching with sleep between reads can achieve **10x battery life** at same resolution.

### 1.5 Magnetometer Optimization

The MMC5603NJ magnetometer is the most power-hungry sensor. Optimize:

```javascript
// PROPOSED: Burst magnetometer reading
function readMagBurst(samples) {
    // Enable continuous mode at 100Hz
    Puck.i2cWr(0x30, 0x1B, 0x80);  // CMM_FREQ_EN

    var data = [];
    for (var i = 0; i < samples; i++) {
        // Wait for data ready
        while (!(Puck.i2cRd(0x30, 0x18, 1)[0] & 0x01));
        var raw = Puck.i2cRd(0x30, 0x00, 9);
        data.push(parseMag(raw));
    }

    // Disable continuous mode
    Puck.i2cWr(0x30, 0x1B, 0x00);
    return data;
}
```

---

## 2. Enhanced Button Interactions

### 2.1 Current Limitations

The button currently only:
- Toggles between idle (state=0) and capturing (state=1)
- Increments press counter
- Triggers 30-second capture burst

### 2.2 Multi-Tap Gesture Recognition

```javascript
// PROPOSED: Rich button gesture system
var BUTTON_GESTURES = {
    SINGLE_TAP: { taps: 1, maxGap: 300 },
    DOUBLE_TAP: { taps: 2, maxGap: 300 },
    TRIPLE_TAP: { taps: 3, maxGap: 300 },
    LONG_PRESS: { holdTime: 1000 },
    VERY_LONG_PRESS: { holdTime: 3000 },
    DOUBLE_TAP_HOLD: { taps: 2, holdTime: 500 }
};

var tapCount = 0;
var tapTimer = null;
var pressStart = 0;
var lastTap = 0;

// PROPOSED: Button gesture state machine
setWatch(function(e) {
    var duration = e.time - e.lastTime;

    if (duration < 0.05) return;  // Debounce

    if (e.state) {
        // Button pressed
        pressStart = e.time;
    } else {
        // Button released
        var holdTime = (e.time - pressStart) * 1000;

        if (holdTime > 3000) {
            handleGesture('VERY_LONG_PRESS');
        } else if (holdTime > 1000) {
            handleGesture('LONG_PRESS');
        } else {
            // Short press - count taps
            tapCount++;
            if (tapTimer) clearTimeout(tapTimer);
            tapTimer = setTimeout(function() {
                if (tapCount === 1) handleGesture('SINGLE_TAP');
                else if (tapCount === 2) handleGesture('DOUBLE_TAP');
                else if (tapCount >= 3) handleGesture('TRIPLE_TAP');
                tapCount = 0;
            }, 300);
        }
    }
}, BTN, { edge: "both", repeat: true, debounce: 20 });

// PROPOSED: Gesture action mapping
function handleGesture(gesture) {
    logInfo('Gesture: ' + gesture);
    digitalPulse(LED3, 1, 100);  // Visual feedback

    switch (gesture) {
        case 'SINGLE_TAP':
            // Toggle streaming (current behavior)
            toggleStreaming();
            break;

        case 'DOUBLE_TAP':
            // Cycle through sampling modes
            cycleMode();
            break;

        case 'TRIPLE_TAP':
            // Mark event / timestamp annotation
            markEvent();
            break;

        case 'LONG_PRESS':
            // Enter calibration mode
            startCalibration();
            break;

        case 'VERY_LONG_PRESS':
            // Factory reset / deep sleep
            enterDeepSleep();
            break;
    }

    // Notify via BLE
    sendFrame('BTN', { gesture: gesture, time: Date.now() });
}
```

### 2.3 Proposed Button Commands

| Gesture | Action | LED Feedback | Use Case |
|---------|--------|--------------|----------|
| Single tap | Start/stop streaming | Blue pulse | Basic control |
| Double tap | Cycle sampling mode | 2x blue pulse | Quick mode switch |
| Triple tap | Mark event/annotation | Green pulse | Data annotation |
| Long press (1s) | Enter calibration | Yellow fade | Mag calibration |
| Very long press (3s) | Deep sleep / reset | Red fade-out | Battery save |
| Tap + hold | Lock current mode | Blue hold | Prevent accidental changes |

---

## 3. RGB LED Feedback System

### 3.1 Current Limitations

LEDs are underutilized:
- LED1 (Red): Unused
- LED2 (Green): NFC detection only
- LED3 (Blue): Button press only

### 3.2 Rich Status Encoding

```javascript
// PROPOSED: LED status patterns
var LED_PATTERNS = {
    // State indicators (background)
    IDLE: { led: LED2, pattern: [50], interval: 5000 },     // Dim green pulse every 5s
    STREAMING: { led: LED3, pattern: [100, 100], interval: 1000 },  // Blue blink
    LOW_BATTERY: { led: LED1, pattern: [200, 800], interval: 1000 }, // Slow red
    CRITICAL_BATTERY: { led: LED1, pattern: [100, 100], interval: 200 }, // Fast red

    // Event indicators (momentary)
    CONNECTED: { leds: [LED2], pattern: [500], once: true },   // Green on connect
    DISCONNECTED: { leds: [LED1], pattern: [200, 200, 200], once: true },
    DATA_RECEIVED: { led: LED3, pattern: [20], once: true },   // Quick blue flash
    CALIBRATING: { leds: [LED1, LED2], pattern: [100, 100], interval: 200 }, // Yellow blink
    CALIBRATED: { leds: [LED2], pattern: [1000], once: true }, // Long green
    ERROR: { led: LED1, pattern: [100, 100, 100], once: true }, // Triple red

    // Mode indicators (color coding)
    MODE_ULTRA_LOW: { led: LED2, pattern: [20], interval: 10000 }, // Rare green
    MODE_NORMAL: { led: LED3, pattern: [50], interval: 3000 },     // Blue every 3s
    MODE_HIGH_RES: { led: LED3, pattern: [50, 50, 50], interval: 2000 }, // Fast blue
    MODE_BURST: { leds: [LED1, LED3], pattern: [100], interval: 500 }  // Purple flash
};

// LED pattern player
var ledInterval = null;
var patternIndex = 0;

function playPattern(pattern) {
    if (ledInterval) clearInterval(ledInterval);

    var p = LED_PATTERNS[pattern];
    if (!p) return;

    function tick() {
        if (p.leds) {
            p.leds.forEach(function(led) {
                digitalPulse(led, 1, p.pattern[patternIndex % p.pattern.length]);
            });
        } else {
            digitalPulse(p.led, 1, p.pattern[patternIndex % p.pattern.length]);
        }
        patternIndex++;

        if (p.once && patternIndex >= p.pattern.length) {
            clearInterval(ledInterval);
            ledInterval = null;
        }
    }

    if (p.once) {
        tick();
    } else {
        tick();
        ledInterval = setInterval(tick, p.interval);
    }
}
```

### 3.3 Orientation Feedback

Use LEDs to indicate device orientation or provide user guidance:

```javascript
// PROPOSED: LED orientation feedback during calibration
function showOrientationFeedback(quaternion) {
    var euler = quaternionToEuler(quaternion);
    var pitch = euler.pitch * 180 / Math.PI;
    var roll = euler.roll * 180 / Math.PI;

    // LED brightness based on how level the device is
    var levelness = Math.abs(pitch) + Math.abs(roll);

    if (levelness < 5) {
        // Very level - green
        analogWrite(LED2, 0.1);
        digitalWrite(LED1, 0);
        digitalWrite(LED3, 0);
    } else if (levelness < 20) {
        // Slightly tilted - blue
        analogWrite(LED3, 0.1);
        digitalWrite(LED1, 0);
        digitalWrite(LED2, 0);
    } else {
        // Very tilted - red
        analogWrite(LED1, 0.1);
        digitalWrite(LED2, 0);
        digitalWrite(LED3, 0);
    }
}
```

### 3.4 Battery Level Indication

```javascript
// PROPOSED: Battery status on demand
function showBatteryLevel() {
    var level = Puck.getBatteryPercentage();

    if (level > 60) {
        // Good - green
        digitalPulse(LED2, 1, [200, 100, 200, 100, 200]);  // 3 pulses
    } else if (level > 30) {
        // Medium - yellow (red + green)
        digitalPulse(LED1, 1, [200, 100, 200]);
        digitalPulse(LED2, 1, [200, 100, 200]);
    } else if (level > 10) {
        // Low - red
        digitalPulse(LED1, 1, [200, 100, 200]);
    } else {
        // Critical - fast red
        digitalPulse(LED1, 1, [50, 50, 50, 50, 50, 50]);
    }
}
```

---

## 4. Light & Capacitive Sensor Applications

### 4.1 Light Sensor Applications

#### 4.1.1 Context-Aware Power Management

```javascript
// PROPOSED: Light-based context detection
var LIGHT_CONTEXTS = {
    POCKET: { max: 0.01 },      // Near zero light
    INDOOR: { min: 0.01, max: 0.4 },
    BRIGHT: { min: 0.4, max: 0.8 },
    OUTDOOR: { min: 0.8 }
};

var lastLightContext = null;
var lightHistory = [];
var LIGHT_WINDOW = 10;

function updateLightContext(light) {
    // Running average for stability
    lightHistory.push(light);
    if (lightHistory.length > LIGHT_WINDOW) lightHistory.shift();
    var avgLight = lightHistory.reduce((a,b) => a+b) / lightHistory.length;

    var context = null;
    for (var name in LIGHT_CONTEXTS) {
        var c = LIGHT_CONTEXTS[name];
        if ((c.min === undefined || avgLight >= c.min) &&
            (c.max === undefined || avgLight < c.max)) {
            context = name;
            break;
        }
    }

    if (context !== lastLightContext) {
        lastLightContext = context;
        onLightContextChange(context);
    }

    return context;
}

function onLightContextChange(context) {
    logInfo('Light context: ' + context);

    switch (context) {
        case 'POCKET':
            // Device pocketed - reduce sampling, save power
            setMode('ULTRA_LOW_POWER');
            break;

        case 'OUTDOOR':
            // Likely in use - enable full tracking
            setMode('NORMAL');
            break;

        case 'INDOOR':
            // Standard use
            setMode('LOW_POWER');
            break;
    }
}
```

#### 4.1.2 Screen Presence Detection

```javascript
// PROPOSED: Detect when near a screen/display
function detectScreenProximity(light, lightDelta) {
    // Screens cause rapid light changes when content updates
    // Measure variance in light readings
    var variance = calculateVariance(lightHistory);

    if (variance > 0.05 && light > 0.2) {
        // High variance + medium light = likely screen
        return true;
    }
    return false;
}
```

#### 4.1.3 Day/Night Cycle Tracking

```javascript
// PROPOSED: Track ambient light patterns for circadian data
var dailyLightProfile = new Array(24).fill(0);
var dailyLightCounts = new Array(24).fill(0);

function updateDailyLightProfile(light) {
    var hour = new Date().getHours();
    dailyLightProfile[hour] =
        (dailyLightProfile[hour] * dailyLightCounts[hour] + light) /
        (dailyLightCounts[hour] + 1);
    dailyLightCounts[hour]++;
}
```

### 4.2 Capacitive Sensor Applications

#### 4.2.1 Grip Detection

```javascript
// PROPOSED: Detect when device is being held
var CAP_BASELINE = null;
var CAP_GRIP_THRESHOLD = 500;  // Calibrate per device
var capHistory = [];

function calibrateCapacitive() {
    // Measure baseline when not held
    var samples = [];
    for (var i = 0; i < 10; i++) {
        samples.push(Puck.capSense());
        // Small delay
    }
    CAP_BASELINE = samples.reduce((a,b) => a+b) / samples.length;
    logInfo('Cap baseline: ' + CAP_BASELINE);
}

function detectGrip(capValue) {
    if (!CAP_BASELINE) return null;

    var delta = capValue - CAP_BASELINE;

    if (delta > CAP_GRIP_THRESHOLD * 2) {
        return 'FIRM_GRIP';
    } else if (delta > CAP_GRIP_THRESHOLD) {
        return 'LIGHT_GRIP';
    } else if (delta > CAP_GRIP_THRESHOLD * 0.3) {
        return 'PROXIMITY';
    } else {
        return 'NO_CONTACT';
    }
}
```

#### 4.2.2 Wake-on-Touch

```javascript
// PROPOSED: Use capacitive sensing for wake trigger
var touchWakeEnabled = false;
var touchPollingInterval = null;

function enableTouchWake() {
    touchWakeEnabled = true;

    // Low-frequency polling in sleep mode
    touchPollingInterval = setInterval(function() {
        var cap = Puck.capSense();
        var grip = detectGrip(cap);

        if (grip === 'FIRM_GRIP' && !streamingActive) {
            // Wake up and start streaming
            logInfo('Touch wake triggered');
            getData();
        }
    }, 500);  // Check every 500ms
}

function disableTouchWake() {
    touchWakeEnabled = false;
    if (touchPollingInterval) {
        clearInterval(touchPollingInterval);
        touchPollingInterval = null;
    }
}
```

#### 4.2.3 Capacitive Gesture Recognition (Advanced)

```javascript
// PROPOSED: Simple capacitive gestures via time-series analysis
var capBuffer = [];
var CAP_BUFFER_SIZE = 20;

function analyzeCapGesture() {
    if (capBuffer.length < CAP_BUFFER_SIZE) return null;

    // Calculate features
    var min = Math.min.apply(null, capBuffer);
    var max = Math.max.apply(null, capBuffer);
    var range = max - min;
    var firstHalf = capBuffer.slice(0, CAP_BUFFER_SIZE/2);
    var secondHalf = capBuffer.slice(CAP_BUFFER_SIZE/2);
    var trend = average(secondHalf) - average(firstHalf);

    // Classify gesture
    if (range < 100) {
        return 'STEADY_HOLD';
    } else if (trend > 200) {
        return 'APPROACH';  // Hand approaching
    } else if (trend < -200) {
        return 'RELEASE';   // Hand moving away
    } else if (range > 500) {
        return 'TAP';       // Quick touch
    }

    return null;
}

function onCapReading(value) {
    capBuffer.push(value);
    if (capBuffer.length > CAP_BUFFER_SIZE) capBuffer.shift();

    var gesture = analyzeCapGesture();
    if (gesture) {
        sendFrame('CAP_GESTURE', { type: gesture, time: Date.now() });
    }
}
```

### 4.3 Combined Sensor Intelligence

```javascript
// PROPOSED: Multi-sensor context inference
function inferContext() {
    var light = telemetry.l;
    var cap = telemetry.c;
    var accel = Math.sqrt(
        telemetry.ax*telemetry.ax +
        telemetry.ay*telemetry.ay +
        telemetry.az*telemetry.az
    );

    var grip = detectGrip(cap);
    var lightCtx = updateLightContext(light);
    var moving = accel > 9000;  // Significant motion

    // Decision matrix
    if (grip === 'NO_CONTACT' && lightCtx === 'POCKET') {
        return 'STORED';  // In pocket/bag, not in use
    } else if (grip === 'FIRM_GRIP' && moving) {
        return 'ACTIVE_USE';  // Being used actively
    } else if (grip === 'FIRM_GRIP' && !moving) {
        return 'HELD_STILL';  // Held but stationary
    } else if (lightCtx === 'OUTDOOR' && moving) {
        return 'OUTDOOR_ACTIVITY';
    }

    return 'UNKNOWN';
}
```

---

## 5. Advanced BLE Capabilities

### 5.1 Current Limitations

- Nordic UART Service with JSON text
- 20Hz max streaming rate
- No MTU optimization
- Single connection only
- No notification batching

### 5.2 Binary Protocol (HIGH IMPACT)

Replace JSON with compact binary format for 5x throughput:

```javascript
// PROPOSED: Binary telemetry packet (28 bytes vs ~120 bytes JSON)
// Format: [Header(2)] [Timestamp(4)] [Accel(6)] [Gyro(6)] [Mag(6)] [Aux(4)]
function emitBinary() {
    var buf = new ArrayBuffer(28);
    var view = new DataView(buf);

    // Header: 0xABCD (magic) + packet type
    view.setUint16(0, 0xABCD, false);

    // Timestamp (ms since boot, 32-bit)
    view.setUint32(2, Date.now() - bootTime, true);

    // Accelerometer (16-bit signed x3)
    var accel = Puck.accel();
    view.setInt16(6, accel.acc.x, true);
    view.setInt16(8, accel.acc.y, true);
    view.setInt16(10, accel.acc.z, true);

    // Gyroscope (16-bit signed x3)
    view.setInt16(12, accel.gyro.x, true);
    view.setInt16(14, accel.gyro.y, true);
    view.setInt16(16, accel.gyro.z, true);

    // Magnetometer (16-bit signed x3)
    var mag = Puck.mag();
    view.setInt16(18, mag.x, true);
    view.setInt16(20, mag.y, true);
    view.setInt16(22, mag.z, true);

    // Auxiliary (8-bit each: light, temp, cap, battery)
    view.setUint8(24, telemetry.l * 255);
    view.setUint8(25, telemetry.t + 40);  // Offset for range
    view.setUint8(26, telemetry.c >> 4);  // 12-bit to 8-bit
    view.setUint8(27, telemetry.b);

    // Send as raw bytes
    Bluetooth.write(new Uint8Array(buf));
}

// PROPOSED: Batch multiple samples
function emitBinaryBatch(samples) {
    // Header: [0xABCE] [count(2)] [first_timestamp(4)]
    // Data: [delta_t(1)] [accel(6)] [gyro(6)] per sample
    // Compressed: 13 bytes per sample after first

    var headerSize = 8;
    var sampleSize = 13;
    var buf = new ArrayBuffer(headerSize + samples.length * sampleSize);
    var view = new DataView(buf);

    view.setUint16(0, 0xABCE, false);  // Batch magic
    view.setUint16(2, samples.length, true);
    view.setUint32(4, samples[0].timestamp, true);

    var offset = headerSize;
    var lastTime = samples[0].timestamp;

    samples.forEach(function(s) {
        view.setUint8(offset, s.timestamp - lastTime);  // Delta time (0-255ms)
        view.setInt16(offset + 1, s.ax, true);
        view.setInt16(offset + 3, s.ay, true);
        view.setInt16(offset + 5, s.az, true);
        view.setInt16(offset + 7, s.gx, true);
        view.setInt16(offset + 9, s.gy, true);
        view.setInt16(offset + 11, s.gz, true);

        lastTime = s.timestamp;
        offset += sampleSize;
    });

    Bluetooth.write(new Uint8Array(buf));
}
```

**Throughput comparison:**

| Protocol | Bytes/sample | Samples/packet (MTU=247) | Effective rate |
|----------|--------------|--------------------------|----------------|
| JSON | ~120 | 1-2 | 20Hz |
| Binary | 28 | 8 | 80Hz |
| Binary Batch | 13 | 18 | 180Hz |

### 5.3 MTU and Connection Optimization

```javascript
// PROPOSED: Request optimal MTU on connect
NRF.on('connect', function(addr) {
    logInfo('BLE connected: ' + addr);

    // Request maximum MTU (247 bytes)
    // Note: Espruino handles this automatically if supported
    try {
        NRF.setConnectionInterval(7.5, 15);  // Fast connection interval
        // MTU negotiation happens automatically
    } catch (e) {
        logWarn('Connection optimization failed: ' + e.message);
    }
});
```

### 5.4 Connection Quality Monitoring

```javascript
// PROPOSED: Monitor and adapt to connection quality
var connectionStats = {
    packetsDropped: 0,
    lastRSSI: 0,
    avgLatency: 0
};

function getConnectionQuality() {
    try {
        var info = NRF.getSecurityStatus();
        connectionStats.lastRSSI = NRF.getBattery ? -50 : -70;  // Placeholder

        if (connectionStats.lastRSSI < -80) {
            return 'POOR';
        } else if (connectionStats.lastRSSI < -60) {
            return 'FAIR';
        } else {
            return 'GOOD';
        }
    } catch (e) {
        return 'UNKNOWN';
    }
}

// Adaptive streaming based on connection quality
function adaptStreamingRate() {
    var quality = getConnectionQuality();

    switch (quality) {
        case 'POOR':
            // Reduce rate, increase batching
            setStreamingRate(10);  // 10Hz
            setBatchSize(20);      // Larger batches
            break;

        case 'FAIR':
            setStreamingRate(20);
            setBatchSize(10);
            break;

        case 'GOOD':
            setStreamingRate(50);
            setBatchSize(5);
            break;
    }
}
```

### 5.5 Multi-Connection Support

```javascript
// PROPOSED: Support multiple simultaneous connections
// nRF52840 supports up to 20 connections
var connections = [];
var MAX_CONNECTIONS = 3;

NRF.on('connect', function(addr) {
    if (connections.length >= MAX_CONNECTIONS) {
        logWarn('Max connections reached, ignoring: ' + addr);
        return;
    }

    connections.push({
        address: addr,
        connectedAt: Date.now(),
        subscribed: false
    });

    logInfo('Connected (' + connections.length + '/' + MAX_CONNECTIONS + '): ' + addr);
});

NRF.on('disconnect', function(reason) {
    connections = connections.filter(c => c.active);
    logInfo('Disconnected (' + connections.length + ' remaining): ' + reason);
});
```

### 5.6 BLE Beaconing (Background Advertising)

```javascript
// PROPOSED: Advertise sensor summary in background
// Useful for proximity detection without active connection
function updateBeaconData() {
    var summary = {
        battery: Puck.getBatteryPercentage(),
        state: state,
        lastActivity: Date.now() - bootTime,
        motion: isMoving()
    };

    // Encode in manufacturer data
    NRF.setAdvertising({
        0xFF: [
            0xCD, 0xAB,  // Company ID (custom)
            summary.battery,
            summary.state,
            (summary.lastActivity >> 8) & 0xFF,
            summary.lastActivity & 0xFF,
            summary.motion ? 1 : 0
        ]
    }, { name: "GAMBIT", interval: 1000 });
}

// Update beacon every 10 seconds
setInterval(updateBeaconData, 10000);
```

### 5.7 Enhanced Command Protocol

```javascript
// PROPOSED: Structured command/response protocol
var COMMANDS = {
    0x01: { name: 'GET_FIRMWARE', handler: getFirmware },
    0x02: { name: 'GET_LOGS', handler: getLogs },
    0x03: { name: 'CLEAR_LOGS', handler: clearLogs },
    0x04: { name: 'START_STREAM', handler: getData },
    0x05: { name: 'STOP_STREAM', handler: stopData },
    0x06: { name: 'SET_MODE', handler: setMode },
    0x07: { name: 'CALIBRATE', handler: startCalibration },
    0x08: { name: 'GET_STATUS', handler: getStatus },
    0x09: { name: 'SET_CONFIG', handler: setConfig },
    0x0A: { name: 'RESET', handler: reset }
};

function handleCommand(data) {
    if (data.length < 1) return;

    var cmdId = data[0];
    var cmd = COMMANDS[cmdId];

    if (!cmd) {
        sendFrame('ERR', { code: 'UNKNOWN_CMD', id: cmdId });
        return;
    }

    logDebug('Cmd: ' + cmd.name);

    try {
        var args = data.slice(1);
        var result = cmd.handler(args);
        sendFrame('OK', { cmd: cmd.name, result: result });
    } catch (e) {
        sendFrame('ERR', { cmd: cmd.name, error: e.message });
    }
}
```

---

## 6. Implementation Roadmap

### Phase 1: Quick Wins (v0.4.0) ✅ COMPLETE
- [x] Multi-tap button gestures (SINGLE_TAP, DOUBLE_TAP, TRIPLE_TAP, LONG_PRESS, VERY_LONG_PRESS)
- [x] LED status patterns (mode indicators, streaming, battery, calibration)
- [x] Binary protocol option (28-byte packets, 4x bandwidth improvement)
- [x] Light-based power mode switching (context awareness)

### Phase 2: Power Optimization (v0.5.0) ✅ COMPLETE
- [x] FIFO batch mode for LSM6DS3 (416Hz+ via hardware FIFO)
- [x] Adaptive sampling modes (LOW_POWER, NORMAL, HIGH_RES, BURST)
- [x] Capacitive wake-on-touch (grip-triggered wake)
- [x] Background beaconing (advertise status when not connected)

### Phase 3: Intelligence (v0.6.0) - IN PROGRESS
- [x] Multi-sensor context inference (light + cap + motion → context)
- [ ] Grip-based auto mode switching (switch to HIGH_RES when gripped)
- [ ] Connection quality adaptive streaming (reduce rate on weak signal)
- [ ] On-device motion classification (activity recognition)

### Phase 4: Advanced (v1.0.0) - PLANNED
- [ ] Multi-connection support (up to 3 simultaneous clients)
- [ ] Full binary protocol with batch compression (13 bytes/sample)
- [ ] On-device Kalman filtering (sensor fusion)
- [ ] ML-ready feature extraction (real-time features for inference)

### Integration Tasks - NEW
- [ ] FFO$$ gesture app: Add GAMBIT context event support
- [ ] FFO$$ gesture app: Add orientation-aware gesture processing
- [ ] FFO$$ gesture app: Motion segmentation from GAMBIT pipeline
- [ ] Collector app: Update for new firmware events/features

---

## 7. References

### Hardware Documentation
- [LSM6DS3 Datasheet](https://content.arduino.cc/assets/st_imu_lsm6ds3_datasheet.pdf)
- [LSM6DS3 Application Note](https://cdn.sparkfun.com/assets/learn_tutorials/4/1/6/AN4650_DM00157511.pdf)
- [Puck.js v2 Documentation](https://www.espruino.com/Puck.js)
- [nRF52840 Product Specification](https://infocenter.nordicsemi.com/pdf/nRF52840_PS_v1.1.pdf)

### Power Optimization
- [nRF52840 Power Optimization](https://tomasmcguinness.com/2025/01/02/more-adventures-in-nrf52840-power-consumption/)
- [nRF52840 Sleep Modes](https://hardfault.in/2025/03/12/understanding-nrf52840-sleep-modes-for-efficiency/)

### BLE Throughput
- [nRF52840 GATT Throughput](https://devzone.nordicsemi.com/nordic/nordic-blog/b/blog/posts/nrf52840-gatt-data-throughput-with-zephyr-rtos)
- [Notification Throughput Optimization](https://devzone.nordicsemi.com/f/nordic-q-a/39079/how-to-increase-notification-throughput-on-nrf52840)
- [BLE 5.0 Throughput Testing](https://www.engeniustech.com/technical-papers/bluetooth-low-energy.pdf)

### Sensor Applications
- [Capacitive Gesture Recognition](https://dl.acm.org/doi/10.1145/3694907.3765936)
- [Adaptive Power Management](https://www.nature.com/articles/s41598-025-89709-3)
- [Energy-Aware Adaptive Sampling](https://dl.acm.org/doi/10.1145/3628353.3628545)

---

*Document generated as part of GAMBIT firmware improvement investigation.*
