// ===== GAMBIT Firmware Configuration =====
var FIRMWARE_INFO = {
    id: "GAMBIT",
    name: "GAMBIT IMU Telemetry",
    version: "0.4.0",
    features: ["imu", "magnetometer", "environmental", "streaming", "logging", "framing", "gestures", "modes", "context"],
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

        // Auto-adjust mode based on context
        if (currentContext === CONTEXTS.STORED && currentMode !== 'LOW_POWER') {
            setMode('LOW_POWER');
        } else if (currentContext === CONTEXTS.ACTIVE && currentMode === 'LOW_POWER') {
            setMode('NORMAL');
        }
    }

    return currentContext;
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

// ===== State and Telemetry =====
var telemetry = {
    ax: null,
    ay: null,
    az: null,
    gx: null,
    gy: null,
    gz: null,
    mx: null,
    my: null,
    mz: null,
    l: null,
    t: null,
    c: null,
    s: state,
    n: pressCount,
    b: null,
    // New fields for v0.4.0
    mode: null,
    ctx: null,
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

    // BATTERY OPTIMIZATION: Read expensive sensors based on mode config
    if (sampleCount % modeConfig.magEvery === 0) {
        var mag = Puck.mag();
        telemetry.mx = mag.x;
        telemetry.my = mag.y;
        telemetry.mz = mag.z;
        telemetry.t = Puck.magTemp();
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

    // Send telemetry using framing protocol
    sendFrame('T', telemetry);
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

    // Start streaming
    interval = setInterval(function() {
        emit();

        // Auto-stop if we've reached the target count
        if (streamCount && sampleCount >= streamCount) {
            stopData();
        }
    }, intervalMs);

    sendFrame('STREAM_START', {
        mode: currentMode,
        hz: hz,
        count: streamCount,
        time: Date.now() - bootTime
    });

    return emit();
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

// ===== Initialization =====
function init() {
    logInfo('Boot: GAMBIT v' + FIRMWARE_INFO.version);

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

// Initialize
init();

// ===== BLE Connection Events =====
NRF.on('connect', function(addr) {
    logInfo('BLE connected: ' + addr);
    sendFrame('CONN', { connected: true, addr: addr });
});

NRF.on('disconnect', function(reason) {
    logInfo('BLE disconnected: ' + reason);
    // Stop streaming if client disconnects
    stopData();
});

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
