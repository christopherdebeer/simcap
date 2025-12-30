// ===== GAMBIT Firmware Configuration =====
var FIRMWARE_INFO = {
    id: "GAMBIT",
    name: "GAMBIT IMU Telemetry",
    version: "0.4.2",
    features: ["imu", "magnetometer", "environmental", "streaming", "logging", "framing", "gestures", "modes", "context", "mag-degauss"],
    author: "SIMCAP"
};

// Track boot time for uptime calculation
var bootTime = Date.now();

// ===== Length-Prefixed Framing Protocol =====
// Format: \x02TYPE:LENGTH\nPAYLOAD\x03
// STX (0x02) = Start of frame
// ETX (0x03) = End of frame
// This enables robust parsing on the receiver side
function sendFrame(type, payload) {
    var json = JSON.stringify(payload);
    var frame = '\x02' + type + ':' + json.length + '\n' + json + '\x03';
    Bluetooth.print(frame);
    // Track packet stats if available
    if (typeof trackPacket === 'function') {
        trackPacket(frame.length);
    }
}

// ===== Device Logging System =====
var LOG_MAX_ENTRIES = 50;  // Rolling window size (~2-3KB total)
var logBuffer = [];
var logIndex = 0;

// Log levels: E=Error, W=Warn, I=Info, D=Debug
// Note: console.log is suppressed during streaming to prevent BLE interleaving
var streamingActive = false;

function deviceLog(level, msg) {
    var entry = {
        i: logIndex++,              // sequence number
        t: Date.now() - bootTime,   // ms since boot
        l: level,                   // level code
        m: String(msg).substring(0, 80)  // truncate long messages
    };

    logBuffer.push(entry);

    // Rolling window - remove oldest when full
    while (logBuffer.length > LOG_MAX_ENTRIES) {
        logBuffer.shift();
    }

    // Only emit to console when NOT streaming (prevents BLE interleaving)
    if (!streamingActive) {
        console.log("[" + level + "] " + entry.m);
    }
}

// Convenience logging functions
function logError(msg) { deviceLog('E', msg); }
function logWarn(msg) { deviceLog('W', msg); }
function logInfo(msg) { deviceLog('I', msg); }
function logDebug(msg) { deviceLog('D', msg); }

// Get logs as JSON (called from loader)
// Optional 'since' parameter to get only newer entries
function getLogs(since) {
    var filtered = (since !== undefined) ?
        logBuffer.filter(function(e) { return e.i > since; }) :
        logBuffer;
    var response = {
        count: filtered.length,
        total: logBuffer.length,
        nextIndex: logIndex,
        entries: filtered
    };
    sendFrame('LOGS', response);
    return response;
}

// Clear all logs
function clearLogs() {
    var count = logBuffer.length;
    logBuffer = [];
    logIndex = 0;
    logInfo('Logs cleared (' + count + ' entries)');
    var response = { cleared: count };
    sendFrame('LOGS_CLEARED', response);
    return response;
}

// Get log statistics
function getLogStats() {
    var stats = {
        entries: logBuffer.length,
        maxEntries: LOG_MAX_ENTRIES,
        nextIndex: logIndex,
        memUsed: process.memory().usage,
        memTotal: process.memory().total,
        uptime: Date.now() - bootTime
    };
    sendFrame('LOG_STATS', stats);
    return stats;
}

// Return firmware information for compatibility checking
function getFirmware() {
    var uptimeMs = Date.now() - bootTime;
    var mem = process.memory();
    var info = Object.assign({}, FIRMWARE_INFO, {
        uptime: uptimeMs,
        memUsed: mem.usage,
        memTotal: mem.total,
        logCount: logBuffer.length,
        mode: currentMode,
        context: currentContext
    });
    sendFrame('FW', info);
    return info;
}

// ===== Magnetometer SET/RESET (Degauss) =====
// The MMC5603NJ magnetometer can develop large null field offsets (up to ±100µT)
// due to temperature changes or exposure to strong magnetic fields (>32 Gauss).
// The SET/RESET operation clears residual magnetization and establishes a known
// reference state. This should be performed at boot and when calibration is requested.
//
// MMC5603NJ Register 0x1B (Internal Control 0):
//   Bit 3: Do_Set   - Triggers SET operation (375ns pulse)
//   Bit 4: Do_Reset - Triggers RESET operation (375ns pulse)
//   Bit 5: Auto_SR_en - Enables automatic SET/RESET (handled by Espruino)
//
// Note: Wait at least 1ms between SET/RESET and next operation (tSR from datasheet)

function degaussMag() {
    try {
        // Perform SET operation (bit 3 = 0x08)
        Puck.magWr(0x1B, 0x08);
        // Wait 1ms for SET to complete (tSR)
        // Note: We can't use blocking delay in Espruino, so we schedule RESET
    } catch (e) {
        logError('Mag SET failed: ' + e.message);
        return false;
    }

    // Schedule RESET operation after 2ms delay
    setTimeout(function() {
        try {
            // Perform RESET operation (bit 4 = 0x10)
            Puck.magWr(0x1B, 0x10);
            logInfo('Magnetometer degaussed (SET/RESET complete)');
            sendFrame('MAG_DEGAUSS', { success: true, timestamp: Date.now() });
        } catch (e) {
            logError('Mag RESET failed: ' + e.message);
            sendFrame('MAG_DEGAUSS', { success: false, error: e.message });
        }
    }, 2);

    return true;
}

// ===== Sampling Modes =====
var MODES = {
    LOW_POWER: {
        name: 'low_power',
        accelHz: 26,
        magEvery: 5,      // Every 5th sample (5Hz)
        lightEvery: 20,   // Every 20th sample (1.3Hz)
        battEvery: 200    // Every 200th sample
    },
    NORMAL: {
        name: 'normal',
        accelHz: 26,
        magEvery: 2,      // Every 2nd sample (13Hz)
        lightEvery: 10,   // Every 10th sample (2.6Hz)
        battEvery: 100
    },
    HIGH_RES: {
        name: 'high_res',
        accelHz: 52,
        magEvery: 1,      // Every sample
        lightEvery: 5,
        battEvery: 100
    },
    BURST: {
        name: 'burst',
        accelHz: 104,
        magEvery: 1,
        lightEvery: 10,
        battEvery: 200
    }
};

var currentMode = 'NORMAL';
var modeConfig = MODES.NORMAL;

function setMode(modeName) {
    if (!MODES[modeName]) {
        logWarn('Unknown mode: ' + modeName);
        return false;
    }
    currentMode = modeName;
    modeConfig = MODES[modeName];
    logInfo('Mode: ' + modeName);

    // Send mode change event
    sendFrame('MODE', { mode: modeName, config: modeConfig });

    // Update LED pattern
    showModeIndicator();

    return true;
}

function cycleMode() {
    var modeNames = Object.keys(MODES);
    var idx = modeNames.indexOf(currentMode);
    var nextIdx = (idx + 1) % modeNames.length;
    setMode(modeNames[nextIdx]);
}

function getMode() {
    return { mode: currentMode, config: modeConfig };
}

// ===== LSM6DS3 FIFO Batch Mode =====
// High-resolution mode using hardware FIFO for 400Hz+ sampling
// FIFO can buffer samples while BLE transmits, enabling 16x resolution increase

var LSM6DS3_ADDR = 0x6A;  // Default I2C address

// LSM6DS3 FIFO Registers
var LSM6DS3_FIFO_CTRL1 = 0x06;  // FIFO threshold low
var LSM6DS3_FIFO_CTRL2 = 0x07;  // FIFO threshold high
var LSM6DS3_FIFO_CTRL3 = 0x08;  // FIFO decimation settings
var LSM6DS3_FIFO_CTRL4 = 0x09;  // Additional decimation
var LSM6DS3_FIFO_CTRL5 = 0x0A;  // FIFO mode and ODR
var LSM6DS3_FIFO_STATUS1 = 0x3A; // Number of unread samples (low)
var LSM6DS3_FIFO_STATUS2 = 0x3B; // Number of unread samples (high) + flags
var LSM6DS3_FIFO_DATA_OUT_L = 0x3E; // FIFO data output low
var LSM6DS3_FIFO_DATA_OUT_H = 0x3F; // FIFO data output high

// FIFO Mode constants
var FIFO_MODE_BYPASS = 0x00;
var FIFO_MODE_FIFO = 0x01;
var FIFO_MODE_CONTINUOUS = 0x06;

// FIFO ODR (Output Data Rate) - for 416Hz set to 0x05
var FIFO_ODR_12_5 = 0x01;
var FIFO_ODR_26 = 0x02;
var FIFO_ODR_52 = 0x03;
var FIFO_ODR_104 = 0x04;
var FIFO_ODR_208 = 0x05;
var FIFO_ODR_416 = 0x06;
var FIFO_ODR_833 = 0x07;
var FIFO_ODR_1660 = 0x08;

var fifoEnabled = false;
var fifoThreshold = 64;  // Samples before interrupt/read

// Write a register to LSM6DS3
function lsm6Write(reg, val) {
    I2C1.writeTo(LSM6DS3_ADDR, [reg, val]);
}

// Read a register from LSM6DS3
function lsm6Read(reg) {
    I2C1.writeTo(LSM6DS3_ADDR, reg);
    return I2C1.readFrom(LSM6DS3_ADDR, 1)[0];
}

// Read multiple bytes from LSM6DS3
function lsm6ReadMulti(reg, count) {
    I2C1.writeTo(LSM6DS3_ADDR, reg);
    return I2C1.readFrom(LSM6DS3_ADDR, count);
}

// Configure FIFO for high-resolution batch mode
function configureFifo(enabled, odr) {
    if (!enabled) {
        // Disable FIFO - bypass mode
        lsm6Write(LSM6DS3_FIFO_CTRL5, FIFO_MODE_BYPASS);
        fifoEnabled = false;
        logInfo('FIFO disabled');
        return;
    }

    odr = odr || FIFO_ODR_416;  // Default 416Hz

    // Set FIFO threshold (number of 16-bit words)
    // Each 6-axis sample = 12 bytes = 6 words
    var thresholdWords = fifoThreshold * 6;
    lsm6Write(LSM6DS3_FIFO_CTRL1, thresholdWords & 0xFF);
    lsm6Write(LSM6DS3_FIFO_CTRL2, (thresholdWords >> 8) & 0x0F);

    // Configure decimation: accel=1, gyro=1 (no decimation)
    // FIFO_CTRL3: [7:5]=reserved, [4:3]=DEC_DS4_FIFO, [2:0]=DEC_DS3_FIFO
    // For accel: DEC_FIFO_XL[2:0] at bits [2:0]
    // For gyro: DEC_FIFO_GYRO[2:0] at bits [5:3]
    lsm6Write(LSM6DS3_FIFO_CTRL3, 0x09);  // Gyro dec=1, Accel dec=1

    // FIFO_CTRL4: No step counter or external sensors
    lsm6Write(LSM6DS3_FIFO_CTRL4, 0x00);

    // FIFO_CTRL5: Set mode (continuous) and ODR
    // [7:6] = reserved
    // [5:3] = ODR_FIFO
    // [2:0] = FIFO_MODE
    var ctrl5 = (odr << 3) | FIFO_MODE_CONTINUOUS;
    lsm6Write(LSM6DS3_FIFO_CTRL5, ctrl5);

    fifoEnabled = true;
    var hzLookup = { 1: 12.5, 2: 26, 3: 52, 4: 104, 5: 208, 6: 416, 7: 833, 8: 1660 };
    logInfo('FIFO enabled @ ' + (hzLookup[odr] || odr) + 'Hz (batch=' + fifoThreshold + ')');
}

// Get number of samples available in FIFO
function getFifoCount() {
    var status1 = lsm6Read(LSM6DS3_FIFO_STATUS1);
    var status2 = lsm6Read(LSM6DS3_FIFO_STATUS2);
    var count = ((status2 & 0x0F) << 8) | status1;
    return Math.floor(count / 6);  // Convert words to 6-axis samples
}

// Read a batch of samples from FIFO
// Returns array of {ax, ay, az, gx, gy, gz} objects
function readFifoBatch(maxSamples) {
    var available = getFifoCount();
    var toRead = Math.min(available, maxSamples || 64);

    if (toRead === 0) return [];

    var samples = [];
    var bytesPerSample = 12;  // 6 axes * 2 bytes each
    var totalBytes = toRead * bytesPerSample;

    // Read all data at once (more efficient)
    var data = lsm6ReadMulti(LSM6DS3_FIFO_DATA_OUT_L, totalBytes);

    for (var i = 0; i < toRead; i++) {
        var offset = i * bytesPerSample;
        // FIFO order: Gyro X, Y, Z then Accel X, Y, Z (little-endian 16-bit)
        var gx = (data[offset + 1] << 8) | data[offset];
        var gy = (data[offset + 3] << 8) | data[offset + 2];
        var gz = (data[offset + 5] << 8) | data[offset + 4];
        var ax = (data[offset + 7] << 8) | data[offset + 6];
        var ay = (data[offset + 9] << 8) | data[offset + 8];
        var az = (data[offset + 11] << 8) | data[offset + 10];

        // Convert to signed 16-bit
        if (gx > 32767) gx -= 65536;
        if (gy > 32767) gy -= 65536;
        if (gz > 32767) gz -= 65536;
        if (ax > 32767) ax -= 65536;
        if (ay > 32767) ay -= 65536;
        if (az > 32767) az -= 65536;

        samples.push({ ax: ax, ay: ay, az: az, gx: gx, gy: gy, gz: gz });
    }

    return samples;
}

// High-resolution streaming using FIFO batch mode
// Collects samples at 416Hz and sends in batches
var fifoStreamInterval = null;
var fifoSampleBuffer = [];
var fifoSampleCount = 0;
var fifoTargetCount = null;
var fifoStreamStart = null;

function startFifoStream(count, odr) {
    if (fifoStreamInterval) {
        logWarn('FIFO stream already active');
        return;
    }

    odr = odr || FIFO_ODR_416;
    logInfo('Starting FIFO stream @ ' + (odr === FIFO_ODR_416 ? 416 : odr) + 'Hz');

    // Initialize I2C for direct register access
    I2C1.setup({ scl: D19, sda: D20, bitrate: 400000 });

    // Enable accelerometer and gyroscope at high rate
    Puck.accelOn(1660);  // Max rate

    // Configure and enable FIFO
    configureFifo(true, odr);

    fifoSampleCount = 0;
    fifoTargetCount = count || null;
    fifoStreamStart = Date.now();
    streamingActive = true;

    // Read FIFO every 50ms (collects ~20 samples at 416Hz)
    fifoStreamInterval = setInterval(function() {
        var batch = readFifoBatch(64);
        if (batch.length === 0) return;

        // Add magnetometer to first sample in batch
        var mag = Puck.mag();
        batch[0].mx = mag.x;
        batch[0].my = mag.y;
        batch[0].mz = mag.z;

        // Add timestamp to each sample (interpolated)
        var now = Date.now() - bootTime;
        var interval = 1000 / (odr === FIFO_ODR_416 ? 416 : 104);
        for (var i = 0; i < batch.length; i++) {
            batch[i].t = now - ((batch.length - 1 - i) * interval);
            batch[i].n = fifoSampleCount + i;
        }

        fifoSampleCount += batch.length;

        // Send batch frame
        sendFrame('FIFO', {
            samples: batch,
            count: batch.length,
            total: fifoSampleCount
        });

        // Check if we've reached target
        if (fifoTargetCount && fifoSampleCount >= fifoTargetCount) {
            stopFifoStream();
        }
    }, 50);

    sendFrame('STREAM_START', {
        mode: 'FIFO',
        hz: odr === FIFO_ODR_416 ? 416 : 104,
        count: count,
        batch: true,
        time: Date.now() - bootTime
    });
}

function stopFifoStream() {
    if (!fifoStreamInterval) return;

    var duration = Date.now() - fifoStreamStart;
    logInfo('FIFO stream stopped - ' + fifoSampleCount + ' samples in ' + duration + 'ms');
    logInfo('Effective rate: ' + Math.round(fifoSampleCount * 1000 / duration) + 'Hz');

    clearInterval(fifoStreamInterval);
    fifoStreamInterval = null;

    // Disable FIFO and accelerometer
    configureFifo(false);
    Puck.accelOff();

    streamingActive = false;

    sendFrame('STREAM_STOP', {
        mode: 'FIFO',
        samples: fifoSampleCount,
        duration: duration,
        effectiveHz: Math.round(fifoSampleCount * 1000 / duration),
        time: Date.now() - bootTime
    });

    fifoSampleCount = 0;
    fifoTargetCount = null;
    fifoStreamStart = null;
}

// ===== Context Awareness =====
var CONTEXTS = {
    UNKNOWN: 'unknown',
    STORED: 'stored',      // In pocket/bag (dark, no grip)
    HELD: 'held',          // Being held (grip detected)
    ACTIVE: 'active',      // Active use (held + moving)
    TABLE: 'table'         // On surface (no grip, some light)
};

var currentContext = CONTEXTS.UNKNOWN;
var lightBaseline = null;
var capBaseline = null;
var contextHistory = [];
var CONTEXT_WINDOW = 5;

// Calibrate sensors for context detection
function calibrateContext() {
    logInfo('Calibrating context sensors...');

    // Take baseline readings (assume device is not held)
    var lightSum = 0;
    var capSum = 0;
    var samples = 10;

    for (var i = 0; i < samples; i++) {
        lightSum += Puck.light();
        capSum += Puck.capSense();
    }

    lightBaseline = lightSum / samples;
    capBaseline = capSum / samples;

    logInfo('Context calibrated: light=' + lightBaseline.toFixed(3) + ' cap=' + capBaseline.toFixed(0));
    sendFrame('CAL', { type: 'context', light: lightBaseline, cap: capBaseline });

    // Flash green to confirm
    digitalPulse(LED2, 1, [100, 50, 100]);
}

function detectContext(light, cap, accelMag) {
    // Thresholds
    var DARK_THRESHOLD = 0.02;
    var CAP_GRIP_THRESHOLD = 500;
    var MOTION_THRESHOLD = 9500; // > 1g motion

    var isDark = light < DARK_THRESHOLD;
    var isGripped = capBaseline ? (cap - capBaseline > CAP_GRIP_THRESHOLD) : false;
    var isMoving = accelMag > MOTION_THRESHOLD;

    var newContext;

    if (isDark && !isGripped) {
        newContext = CONTEXTS.STORED;
    } else if (isGripped && isMoving) {
        newContext = CONTEXTS.ACTIVE;
    } else if (isGripped) {
        newContext = CONTEXTS.HELD;
    } else if (!isDark && !isGripped) {
        newContext = CONTEXTS.TABLE;
    } else {
        newContext = CONTEXTS.UNKNOWN;
    }

    // Use hysteresis - require consistent readings
    contextHistory.push(newContext);
    if (contextHistory.length > CONTEXT_WINDOW) {
        contextHistory.shift();
    }

    // Only change context if majority agree
    var counts = {};
    contextHistory.forEach(function(c) {
        counts[c] = (counts[c] || 0) + 1;
    });

    var majority = null;
    var maxCount = 0;
    for (var c in counts) {
        if (counts[c] > maxCount) {
            maxCount = counts[c];
            majority = c;
        }
    }

    if (majority && maxCount >= 3 && majority !== currentContext) {
        var oldContext = currentContext;
        currentContext = majority;
        logInfo('Context: ' + oldContext + ' -> ' + currentContext);
        sendFrame('CTX', { context: currentContext, from: oldContext });

        // Auto-adjust mode based on context (if auto-mode enabled)
        if (autoModeEnabled) {
            autoAdjustMode(currentContext);
        }
    }

    return currentContext;
}

// ===== Auto Mode Switching =====
var autoModeEnabled = true;  // Can be disabled for manual control

function autoAdjustMode(context) {
    var targetMode = null;

    switch (context) {
        case CONTEXTS.STORED:
            // In pocket/bag - minimize power
            targetMode = 'LOW_POWER';
            break;

        case CONTEXTS.HELD:
            // Being held but not moving - anticipate use, increase resolution
            targetMode = 'HIGH_RES';
            break;

        case CONTEXTS.ACTIVE:
            // Active use - normal is sufficient for most gestures
            targetMode = 'NORMAL';
            break;

        case CONTEXTS.TABLE:
            // On surface - low power but ready for pickup
            targetMode = 'LOW_POWER';
            break;

        default:
            // Unknown - keep current mode
            return;
    }

    if (targetMode && targetMode !== currentMode) {
        logInfo('Auto mode: ' + context + ' -> ' + targetMode);
        setMode(targetMode);
    }
}

function setAutoMode(enabled) {
    autoModeEnabled = enabled;
    logInfo('Auto mode: ' + (enabled ? 'enabled' : 'disabled'));
    sendFrame('AUTO_MODE', { enabled: enabled });
}

function getAutoMode() {
    return autoModeEnabled;
}

// ===== LED Patterns =====
var ledTimer = null;

function stopLedPattern() {
    if (ledTimer) {
        clearInterval(ledTimer);
        ledTimer = null;
    }
    digitalWrite(LED1, 0);
    digitalWrite(LED2, 0);
    digitalWrite(LED3, 0);
}

function showModeIndicator() {
    stopLedPattern();

    switch (currentMode) {
        case 'LOW_POWER':
            digitalPulse(LED2, 1, 50); // Quick green
            break;
        case 'NORMAL':
            digitalPulse(LED3, 1, 100); // Blue
            break;
        case 'HIGH_RES':
            digitalPulse(LED3, 1, [50, 50, 50]); // Double blue
            break;
        case 'BURST':
            digitalPulse(LED1, 1, 50);
            digitalPulse(LED3, 1, 50); // Purple (red + blue)
            break;
    }
}

function showBatteryLevel() {
    var level = Puck.getBatteryPercentage();
    stopLedPattern();

    if (level > 60) {
        // Good - triple green
        digitalPulse(LED2, 1, [100, 50, 100, 50, 100]);
    } else if (level > 30) {
        // Medium - yellow (red + green)
        digitalPulse(LED1, 1, [100, 50, 100]);
        digitalPulse(LED2, 1, [100, 50, 100]);
    } else if (level > 10) {
        // Low - red
        digitalPulse(LED1, 1, [100, 50, 100]);
    } else {
        // Critical - fast red
        digitalPulse(LED1, 1, [30, 30, 30, 30, 30, 30]);
    }

    logInfo('Battery: ' + level + '%');
}

function showStreamingIndicator(on) {
    stopLedPattern();
    if (on) {
        // Periodic blue pulse during streaming
        ledTimer = setInterval(function() {
            digitalPulse(LED3, 1, 20);
        }, 2000);
    }
}

function showError() {
    digitalPulse(LED1, 1, [50, 50, 50, 50, 50]); // Fast red flashes
}

function showSuccess() {
    digitalPulse(LED2, 1, [200, 100, 200]); // Double green
}

// ===== Button Gesture System =====
var tapCount = 0;
var tapTimer = null;
var pressStart = 0;
var TAP_WINDOW = 300;  // ms between taps
var LONG_PRESS = 1000; // ms for long press
var VERY_LONG_PRESS = 3000;

// State for telemetry
var state = 1;
var pressCount = 0;
var lastGesture = null;

function handleGesture(gesture) {
    lastGesture = gesture;
    logInfo('Gesture: ' + gesture);

    // Visual feedback
    switch (gesture) {
        case 'SINGLE_TAP':
            digitalPulse(LED3, 1, 100);
            break;
        case 'DOUBLE_TAP':
            digitalPulse(LED3, 1, [50, 50, 50]);
            break;
        case 'TRIPLE_TAP':
            digitalPulse(LED2, 1, 100);
            break;
        case 'LONG_PRESS':
            digitalPulse(LED2, 1, [100, 100, 100, 100]);
            break;
        case 'VERY_LONG_PRESS':
            digitalPulse(LED1, 1, 500);
            break;
    }

    // Send gesture event frame
    sendFrame('BTN', {
        gesture: gesture,
        time: Date.now() - bootTime,
        pressCount: pressCount
    });

    // Execute gesture action
    switch (gesture) {
        case 'SINGLE_TAP':
            // Toggle streaming
            if (interval) {
                stopData();
            } else {
                getData();
            }
            break;

        case 'DOUBLE_TAP':
            // Cycle through modes
            cycleMode();
            break;

        case 'TRIPLE_TAP':
            // Mark event / annotation
            sendFrame('MARK', {
                time: Date.now() - bootTime,
                sampleCount: sampleCount
            });
            logInfo('Event marked');
            break;

        case 'LONG_PRESS':
            // Show battery / calibrate
            if (!capBaseline) {
                calibrateContext();
            } else {
                showBatteryLevel();
            }
            break;

        case 'VERY_LONG_PRESS':
            // Deep sleep / power save
            logInfo('Entering deep sleep...');
            stopData();
            setMode('LOW_POWER');
            sendFrame('SLEEP', { time: Date.now() - bootTime });
            break;
    }
}

// Button handler with gesture recognition
function setupButtonHandler() {
    setWatch(function(e) {
        var now = Date.now();

        if (e.state) {
            // Button pressed
            pressStart = now;
        } else {
            // Button released
            pressCount++;
            var holdTime = now - pressStart;

            if (holdTime > VERY_LONG_PRESS) {
                handleGesture('VERY_LONG_PRESS');
                tapCount = 0;
                if (tapTimer) clearTimeout(tapTimer);
            } else if (holdTime > LONG_PRESS) {
                handleGesture('LONG_PRESS');
                tapCount = 0;
                if (tapTimer) clearTimeout(tapTimer);
            } else {
                // Short press - count as tap
                tapCount++;

                if (tapTimer) clearTimeout(tapTimer);
                tapTimer = setTimeout(function() {
                    if (tapCount === 1) {
                        handleGesture('SINGLE_TAP');
                    } else if (tapCount === 2) {
                        handleGesture('DOUBLE_TAP');
                    } else if (tapCount >= 3) {
                        handleGesture('TRIPLE_TAP');
                    }
                    tapCount = 0;
                }, TAP_WINDOW);
            }
        }
    }, BTN, { edge: "both", repeat: true, debounce: 30 });
}

// ===== Binary Protocol =====
// Binary telemetry format (28 bytes vs ~120 bytes JSON = ~4x smaller)
// Header: [0xAB, 0xCD] (magic bytes)
// Data:   [ax:2][ay:2][az:2][gx:2][gy:2][gz:2][mx:2][my:2][mz:2][t:4][flags:1][aux:3]
// flags: [mode:2][ctx:3][grip:1][hasLight:1][hasBatt:1]
// aux:   [light:1][battery:1][temp:1] (only when available)

// Binary protocol is now the canonical/only telemetry format (v0.4.0+)
// JSON frames (sendFrame) are only used for control messages
var useBinaryProtocol = true;  // Always true - binary is canonical

// Pre-allocate binary buffer to avoid GC pressure during streaming
// Creating new Uint8Array on every sample causes memory fragmentation
var binaryBuf = new Uint8Array(28);
var binaryView = new DataView(binaryBuf.buffer);

function setBinaryProtocol(enabled) {
    // Binary is always enabled - this is kept for API compatibility
    if (!enabled) {
        logWarn('JSON telemetry deprecated - binary protocol is canonical');
    }
    useBinaryProtocol = true;  // Force true
    return true;
}

function emitBinary() {
    sampleCount++;

    // Read accelerometer + gyroscope
    var accel = Puck.accel();
    var ax = accel.acc.x;
    var ay = accel.acc.y;
    var az = accel.acc.z;
    var gx = accel.gyro.x;
    var gy = accel.gyro.y;
    var gz = accel.gyro.z;

    // Calculate accel magnitude for context detection
    var accelMag = Math.sqrt(ax*ax + ay*ay + az*az);

    // Timestamp
    var t = streamStartTime ? Date.now() - streamStartTime : 0;

    // Read magnetometer on schedule
    var mx = telemetry.mx || 0;
    var my = telemetry.my || 0;
    var mz = telemetry.mz || 0;
    var temp = telemetry.temp || 0;

    if (sampleCount % modeConfig.magEvery === 0) {
        var mag = Puck.mag();
        mx = mag.x;
        my = mag.y;
        mz = mag.z;
        temp = Puck.magTemp();
        telemetry.mx = mx;
        telemetry.my = my;
        telemetry.mz = mz;
        telemetry.temp = temp;
    }

    // Read ambient sensors on schedule
    var light = 0;
    var hasLight = 0;
    var hasBatt = 0;
    var batt = 0;

    if (sampleCount % modeConfig.lightEvery === 0) {
        light = Math.round(Puck.light() * 255);
        hasLight = 1;
        var cap = Puck.capSense();
        telemetry.l = light / 255;
        telemetry.c = cap;
        detectContext(telemetry.l, cap, accelMag);
        if (capBaseline) {
            telemetry.grip = (cap - capBaseline > 300) ? 1 : 0;
        }
    }

    if (sampleCount % modeConfig.battEvery === 0) {
        batt = Puck.getBatteryPercentage();
        hasBatt = 1;
        telemetry.b = batt;
    }

    // Build flags byte
    // [mode:2][ctx:3][grip:1][hasLight:1][hasBatt:1]
    var modeCode = {'L':0,'N':1,'H':2,'B':3}[currentMode.charAt(0)] || 1;
    var ctxCode = {'u':0,'s':1,'h':2,'a':3,'t':4}[currentContext.charAt(0)] || 0;
    var gripCode = telemetry.grip === 1 ? 1 : 0;
    var flags = (modeCode << 6) | (ctxCode << 3) | (gripCode << 2) | (hasLight << 1) | hasBatt;

    // Use pre-allocated binary buffer (28 bytes) to avoid GC pressure
    // Magic header
    binaryBuf[0] = 0xAB;
    binaryBuf[1] = 0xCD;

    // IMU data (16-bit signed integers, little-endian)
    binaryView.setInt16(2, ax, true);
    binaryView.setInt16(4, ay, true);
    binaryView.setInt16(6, az, true);
    binaryView.setInt16(8, gx, true);
    binaryView.setInt16(10, gy, true);
    binaryView.setInt16(12, gz, true);
    binaryView.setInt16(14, mx, true);
    binaryView.setInt16(16, my, true);
    binaryView.setInt16(18, mz, true);

    // Timestamp (32-bit unsigned, little-endian)
    binaryView.setUint32(20, t, true);

    // Flags and auxiliary data
    binaryBuf[24] = flags;
    binaryBuf[25] = light;
    binaryBuf[26] = batt;
    binaryBuf[27] = Math.max(0, Math.min(255, temp + 40)); // Offset for signed temp

    // Send raw binary
    Bluetooth.write(binaryBuf);

    return {ax:ax, ay:ay, az:az, gx:gx, gy:gy, gz:gz, mx:mx, my:my, mz:mz, t:t};
}

// ===== State and Telemetry =====
var telemetry = {
    // IMU sensors (always present)
    ax: null,
    ay: null,
    az: null,
    gx: null,
    gy: null,
    gz: null,
    mx: null,
    my: null,
    mz: null,
    // Timestamp (ms since stream start)
    t: 0,
    // Environmental sensors (sampled at lower rates)
    l: null,      // Light sensor
    c: null,      // Capacitive sensor
    b: null,      // Battery percentage
    temp: null,   // Temperature from magnetometer
    // Device state
    s: state,
    n: pressCount,
    // Mode and context (v0.4.0)
    mode: 'N',
    ctx: 'u',
    grip: null
};

// Battery optimization: Track sample count to reduce expensive sensor polling
var sampleCount = 0;

function emit() {
    sampleCount++;

    // Read accelerometer + gyroscope (efficient - single I2C read)
    var accel = Puck.accel();
    telemetry.ax = accel.acc.x;
    telemetry.ay = accel.acc.y;
    telemetry.az = accel.acc.z;
    telemetry.gx = accel.gyro.x;
    telemetry.gy = accel.gyro.y;
    telemetry.gz = accel.gyro.z;

    // Calculate accel magnitude for context detection
    var accelMag = Math.sqrt(
        telemetry.ax * telemetry.ax +
        telemetry.ay * telemetry.ay +
        telemetry.az * telemetry.az
    );

    // Timestamp (ms since stream start)
    telemetry.t = streamStartTime ? Date.now() - streamStartTime : 0;

    // BATTERY OPTIMIZATION: Read expensive sensors based on mode config
    if (sampleCount % modeConfig.magEvery === 0) {
        var mag = Puck.mag();
        telemetry.mx = mag.x;
        telemetry.my = mag.y;
        telemetry.mz = mag.z;
        telemetry.temp = Puck.magTemp();
    }

    // BATTERY OPTIMIZATION: Read ambient sensors based on mode config
    if (sampleCount % modeConfig.lightEvery === 0) {
        telemetry.l = Puck.light();
        telemetry.c = Puck.capSense();

        // Update context detection
        detectContext(telemetry.l, telemetry.c, accelMag);

        // Grip detection for telemetry
        if (capBaseline) {
            telemetry.grip = (telemetry.c - capBaseline > 300) ? 1 : 0;
        }
    }

    // BATTERY OPTIMIZATION: Read battery based on mode config
    if (sampleCount % modeConfig.battEvery === 0) {
        telemetry.b = Puck.getBatteryPercentage();
    }

    telemetry.s = state;
    telemetry.n = pressCount;
    telemetry.mode = currentMode.charAt(0); // First letter: L/N/H/B
    telemetry.ctx = currentContext.charAt(0); // First letter

    // Note: This function is deprecated for streaming - emitBinary() is used instead
    // This is kept for compatibility with getData() single-sample requests
    return telemetry;
}

var interval;
var streamTimeout;
var streamCount = null;  // Target sample count (null = unlimited)
var streamStartTime = null;

// getData(count, intervalMs) - Start streaming telemetry
// count: optional number of samples to collect before auto-stop
// intervalMs: optional sample interval in ms (default based on mode)
function getData(count, intervalMs) {
    // Default interval based on current mode
    if (!intervalMs || intervalMs < 10) {
        intervalMs = Math.round(1000 / modeConfig.accelHz);
    }

    var hz = Math.round(1000 / intervalMs);

    // If already streaming, this acts as keepalive
    if (interval) {
        // Keepalive - refresh timeout for unlimited streaming, don't log
        if (!streamCount) {
            if (streamTimeout) {
                clearTimeout(streamTimeout);
            }
            streamTimeout = setTimeout(function(){
                stopData();
            }, 30000);
        }
        return; // Don't restart streaming or emit extra sample
    }

    // Starting new stream - log before enabling streaming flag
    if (count && count > 0) {
        logInfo('Stream: ' + count + ' @ ' + hz + 'Hz [' + currentMode + ']');
    } else {
        logInfo('Stream: ' + hz + 'Hz (30s timeout) [' + currentMode + ']');
    }

    sampleCount = 0;
    streamCount = (count && count > 0) ? count : null;
    streamStartTime = Date.now();
    streamingActive = true;  // Suppress console.log during streaming
    state = 1;

    // CRITICAL: Keep accelerometer/gyroscope ON for stable readings
    var accelRate = modeConfig.accelHz;
    // Puck.accelOn() only accepts specific rates: 1660, 833, 416, 208, 104, 52, 26, 12.5, 1.6 Hz
    if (accelRate >= 104) accelRate = 104;
    else if (accelRate >= 52) accelRate = 52;
    else accelRate = 26;
    Puck.accelOn(accelRate);

    // Show streaming indicator
    showStreamingIndicator(true);

    // Only set timeout for unlimited streaming (no count specified)
    if (!streamCount) {
        if (streamTimeout) {
            clearTimeout(streamTimeout);
        }
        streamTimeout = setTimeout(function(){
            stopData();
        }, 30000);
    }

    // Start streaming - binary protocol only
    interval = setInterval(function() {
        emitBinary();

        // Auto-stop if we've reached the target count
        if (streamCount && sampleCount >= streamCount) {
            stopData();
        }
    }, intervalMs);

    sendFrame('STREAM_START', {
        mode: currentMode,
        hz: hz,
        count: streamCount,
        binary: true,  // Always binary
        time: Date.now() - bootTime
    });

    return emitBinary();  // Return first sample
}

// Stop streaming
function stopData() {
    streamingActive = false;  // Re-enable console.log
    state = 0;

    if (interval) {
        var duration = Date.now() - streamStartTime;
        logInfo('Stream stopped - ' + sampleCount + ' samples in ' + duration + 'ms');
        clearInterval(interval);
        interval = null;

        // Turn off accelerometer/gyroscope to save power
        Puck.accelOff();

        // Stop streaming indicator
        showStreamingIndicator(false);

        sendFrame('STREAM_STOP', {
            samples: sampleCount,
            duration: duration,
            time: Date.now() - bootTime
        });
    }
    if (streamTimeout) {
        clearTimeout(streamTimeout);
        streamTimeout = null;
    }
    streamCount = null;
    streamStartTime = null;
}


// ===== NFC Detection =====
function setupNFC() {
    NRF.nfcURL("webble://simcap.parc.land/src/web/GAMBIT/");
    NRF.on('NFCon', function() {
        logInfo('NFC field detected');
        digitalPulse(LED2, 1, 500);
        console.log('nfc_field : [1]');
        NRF.setAdvertising({
            0x183e: [1],
        }, { name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version });
    });
    NRF.on('NFCoff', function() {
        logDebug('NFC field removed');
        digitalPulse(LED2, 1, 200);
        console.log('nfc_field : [0]');
        NRF.setAdvertising({
            0x183e: [0],
        }, { name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version });
    });
}

// ===== Connection Quality Monitoring =====
var connStats = {
    rssi: null,           // Signal strength (dBm)
    connected: false,     // Connection status
    connectTime: null,    // When connected (ms since boot)
    packetsSent: 0,       // Total packets sent
    byteSent: 0,          // Total bytes sent
    lastActivity: null,   // Last activity timestamp
    reconnects: 0         // Number of reconnections
};

// Track connection events
function setupConnectionHandlers() {
    NRF.on('connect', function(addr) {
        connStats.connected = true;
        connStats.connectTime = Date.now() - bootTime;
        connStats.lastActivity = connStats.connectTime;
        if (connStats.packetsSent > 0) {
            connStats.reconnects++;
        }
        logInfo('BLE connected: ' + (addr || 'unknown'));
        sendFrame('CONN', { connected: true, addr: addr, time: connStats.connectTime });
    });

    NRF.on('disconnect', function() {
        connStats.connected = false;
        logInfo('BLE disconnected after ' + ((Date.now() - bootTime) - (connStats.connectTime || 0)) + 'ms');
        sendFrame('CONN', { connected: false, time: Date.now() - bootTime });
        // Stop streaming if client disconnects
        stopData();
    });
}

// Get current RSSI (signal strength)
function updateRssi() {
    try {
        var rssi = NRF.getSecurityStatus().rssi;
        if (rssi) {
            connStats.rssi = rssi;
        }
    } catch (e) {
        // RSSI not available when not connected
    }
    return connStats.rssi;
}

// Get connection quality statistics
function getConnStats() {
    updateRssi();
    var now = Date.now() - bootTime;
    var duration = connStats.connected && connStats.connectTime ?
        now - connStats.connectTime : 0;
    var stats = {
        connected: connStats.connected,
        rssi: connStats.rssi,
        rssiQuality: getRssiQuality(connStats.rssi),
        duration: duration,
        packetsSent: connStats.packetsSent,
        bytesSent: connStats.byteSent,
        reconnects: connStats.reconnects,
        avgPacketRate: duration > 0 ? Math.round(connStats.packetsSent * 1000 / duration) : 0
    };
    sendFrame('CONN_STATS', stats);
    return stats;
}

// Classify RSSI into quality levels
function getRssiQuality(rssi) {
    if (rssi === null) return 'unknown';
    if (rssi >= -50) return 'excellent';
    if (rssi >= -60) return 'good';
    if (rssi >= -70) return 'fair';
    if (rssi >= -80) return 'weak';
    return 'poor';
}

// ===== Adaptive Streaming =====
// Automatically adjust streaming rate based on connection quality
var adaptiveStreamingEnabled = false;
var adaptiveCheckInterval = null;
var lastAdaptiveCheck = 0;

function enableAdaptiveStreaming() {
    adaptiveStreamingEnabled = true;
    logInfo('Adaptive streaming enabled');

    // Check connection quality every 2 seconds
    adaptiveCheckInterval = setInterval(checkAndAdaptStreaming, 2000);
}

function disableAdaptiveStreaming() {
    adaptiveStreamingEnabled = false;
    if (adaptiveCheckInterval) {
        clearInterval(adaptiveCheckInterval);
        adaptiveCheckInterval = null;
    }
    logInfo('Adaptive streaming disabled');
}

function checkAndAdaptStreaming() {
    if (!streamingActive || !adaptiveStreamingEnabled) return;

    updateRssi();
    var quality = getRssiQuality(connStats.rssi);

    // Track packet delivery rate
    var now = Date.now() - bootTime;
    var expectedPackets = (now - lastAdaptiveCheck) / (1000 / modeConfig.accelHz);
    var actualPackets = connStats.packetsSent - (connStats.lastAdaptivePackets || 0);
    var deliveryRate = expectedPackets > 0 ? actualPackets / expectedPackets : 1;

    connStats.lastAdaptivePackets = connStats.packetsSent;
    lastAdaptiveCheck = now;

    // Adapt based on quality and delivery rate
    var targetMode = null;

    if (quality === 'poor' || deliveryRate < 0.5) {
        // Poor connection - reduce rate significantly
        if (currentMode !== 'LOW_POWER') {
            targetMode = 'LOW_POWER';
            logWarn('Weak signal - reducing to LOW_POWER mode');
        }
    } else if (quality === 'weak' || deliveryRate < 0.8) {
        // Weak connection - use normal rate
        if (currentMode === 'HIGH_RES' || currentMode === 'BURST') {
            targetMode = 'NORMAL';
            logWarn('Fair signal - reducing to NORMAL mode');
        }
    } else if (quality === 'excellent' && deliveryRate > 0.95) {
        // Excellent connection - can use higher rate if needed
        // Don't auto-increase, just allow manual increase
    }

    if (targetMode && targetMode !== currentMode) {
        setMode(targetMode);
        sendFrame('ADAPTIVE', {
            quality: quality,
            rssi: connStats.rssi,
            deliveryRate: deliveryRate,
            newMode: targetMode
        });
    }
}

// Track outgoing packets
function trackPacket(bytes) {
    connStats.packetsSent++;
    connStats.byteSent += bytes || 0;
    connStats.lastActivity = Date.now() - bootTime;
}

// ===== Background Beaconing =====
// Advertise device status even when not connected
var beaconEnabled = false;
var beaconInterval = null;
var beaconIntervalMs = 5000;  // 5 second update interval

function enableBeaconing(intervalMs) {
    if (beaconInterval) {
        clearInterval(beaconInterval);
    }

    beaconIntervalMs = intervalMs || 5000;
    beaconEnabled = true;
    logInfo('Beaconing enabled @ ' + (beaconIntervalMs / 1000) + 's');

    // Update advertising data periodically
    beaconInterval = setInterval(updateBeacon, beaconIntervalMs);
    updateBeacon(); // Initial update
}

function disableBeaconing() {
    if (beaconInterval) {
        clearInterval(beaconInterval);
        beaconInterval = null;
    }
    beaconEnabled = false;
    logInfo('Beaconing disabled');

    // Reset to default advertising
    try {
        NRF.setAdvertising({}, {
            name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version
        });
    } catch (e) {
        logError('Failed to reset advertising: ' + e.message);
    }
}

function updateBeacon() {
    if (!beaconEnabled) return;

    try {
        var batt = Puck.getBatteryPercentage();
        var temp = E.getTemperature();
        var status = streamingActive ? 1 : (connStats.connected ? 2 : 0);
        // Status: 0=idle, 1=streaming, 2=connected but not streaming

        // Create manufacturer-specific data
        // Format: [length, type=0xFF, company_id_low, company_id_high, ...data]
        // Using 0xFFFF as company ID (test/development)
        var mfgData = [
            11,       // Length (including type and company ID)
            0xFF,     // Type: Manufacturer Specific Data
            0xFF, 0xFF,  // Company ID: 0xFFFF (test)
            0x47,     // 'G' for GAMBIT
            0x01,     // Protocol version
            status,   // Device status
            batt,     // Battery percentage
            Math.round(temp) + 40,  // Temperature (offset for unsigned)
            currentMode.charCodeAt(0),  // Mode: L/N/H/B
            currentContext.charCodeAt(0)  // Context: u/s/h/a/t
        ];

        NRF.setAdvertising([
            {},  // Include standard advertising
            mfgData
        ], {
            name: "GAMBIT " + batt + "% " + currentMode.charAt(0),
            interval: 500  // 500ms advertising interval
        });

    } catch (e) {
        logWarn('Beacon update failed: ' + e.message);
    }
}

// Get beacon status
function getBeaconStatus() {
    var status = {
        enabled: beaconEnabled,
        interval: beaconIntervalMs
    };
    sendFrame('BEACON_STATUS', status);
    return status;
}

// ===== Wake-on-Touch =====
var wakeOnTouchEnabled = false;
var wakeOnTouchInterval = null;

function enableWakeOnTouch() {
    if (!capBaseline) {
        logWarn('Wake-on-touch requires calibration first');
        return false;
    }

    wakeOnTouchEnabled = true;
    logInfo('Wake-on-touch enabled');

    // Poll capacitive sensor at low frequency
    wakeOnTouchInterval = setInterval(function() {
        if (streamingActive) return; // Already streaming

        var cap = Puck.capSense();
        var delta = cap - capBaseline;

        if (delta > 500) { // Firm grip detected
            logInfo('Wake-on-touch triggered');
            digitalPulse(LED3, 1, 100);
            getData(); // Start streaming
        }
    }, 500); // Check every 500ms

    sendFrame('WAKE', { enabled: true });
    return true;
}

function disableWakeOnTouch() {
    wakeOnTouchEnabled = false;
    if (wakeOnTouchInterval) {
        clearInterval(wakeOnTouchInterval);
        wakeOnTouchInterval = null;
    }
    logInfo('Wake-on-touch disabled');
    sendFrame('WAKE', { enabled: false });
    return false;
}

// ===== Exported API =====
// These functions can be called from the client via BLE:
// - getFirmware() - Get firmware info
// - getLogs(since) - Get device logs
// - clearLogs() - Clear logs
// - getLogStats() - Get log statistics
// - getData(count, intervalMs) - Start streaming
// - stopData() - Stop streaming
// - setMode(modeName) - Set sampling mode: LOW_POWER, NORMAL, HIGH_RES, BURST
// - getMode() - Get current mode
// - cycleMode() - Cycle to next mode
// - calibrateContext() - Calibrate light/cap sensors
// - showBatteryLevel() - Show battery via LEDs
// - setBinaryProtocol(enabled) - Enable/disable binary telemetry protocol
// - enableWakeOnTouch() - Enable wake-on-touch (requires calibration)
// - disableWakeOnTouch() - Disable wake-on-touch
// - degaussMag() - Clear magnetometer null field offset via SET/RESET

// ===== Initialization =====
// IMPORTANT: All setup code is wrapped in functions and called here at the VERY END.
// This prevents "Interrupted processing event" errors during upload because
// Espruino executes code as it's received - any blocking calls mid-file will
// interrupt the upload of remaining code.
function init() {
    logInfo('Boot: GAMBIT v' + FIRMWARE_INFO.version);

    // Setup all handlers first (these are quick)
    setupButtonHandler();
    setupNFC();
    setupConnectionHandlers();

    // Degauss magnetometer to clear any null field offset
    // This establishes a known reference state after power-up
    degaussMag();

    // Set Bluetooth appearance and flags for sensor device
    try {
        NRF.setAdvertising([
            {}, // include original advertising packet
            [
                2, 1, 6,           // Bluetooth flags (General Discoverable, BR/EDR Not Supported)
                3, 0x19, 0x40, 0x05 // Appearance: Generic Sensor (0x0540)
            ]
        ], { name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version });
        logInfo('BLE advertising configured');
    } catch (e) {
        logError('BLE init failed: ' + e.message);
    }

    // Log initial memory state
    var mem = process.memory();
    logDebug('Memory: ' + mem.usage + '/' + mem.total);

    // Log battery level at boot
    var batt = Puck.getBatteryPercentage();
    if (batt < 20) {
        logWarn('Low battery: ' + batt + '%');
    } else {
        logInfo('Battery: ' + batt + '%');
    }

    // Initialize mode
    setMode('NORMAL');

    console.log("SIMCAP GAMBIT v" + FIRMWARE_INFO.version + " initialized");
    digitalPulse(LED2, 1, 200); // Green flash to indicate ready
}

// ===== FINAL INITIALIZATION =====
// This MUST be the last line of the file.
// The setTimeout ensures init() runs AFTER the entire file is parsed and uploaded.
// See: https://www.espruino.com/Troubleshooting
setTimeout(init, 100);
