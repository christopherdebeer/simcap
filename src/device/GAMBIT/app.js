// ===== GAMBIT Firmware Configuration =====
var FIRMWARE_INFO = {
    id: "GAMBIT",
    name: "GAMBIT IMU Telemetry",
    version: "0.3.4",
    features: ["imu", "magnetometer", "environmental", "streaming", "logging", "framing"],
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
        logCount: logBuffer.length
    });
    sendFrame('FW', info);
    return info;
}

// Initialize Bluetooth advertising with proper device name and appearance
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

    console.log("SIMCAP GAMBIT v" + FIRMWARE_INFO.version + " initialized");
    digitalPulse(LED2, 1, 200); // Green flash to indicate ready
}

// ===== State and Telemetry =====
var state = 1;
var pressCount = 0;

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
    b: null,
}

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

    // BATTERY OPTIMIZATION: Read expensive sensors less frequently
    // Magnetometer is power-hungry (requires sensor wake-up) - read every 2nd sample (10Hz instead of 20Hz)
    if (sampleCount % 2 === 0) {
        var mag = Puck.mag();
        telemetry.mx = mag.x;
        telemetry.my = mag.y;
        telemetry.mz = mag.z;
        telemetry.t = Puck.magTemp(); // Temperature from magnetometer - read together
    }

    // BATTERY OPTIMIZATION: Read ambient sensors every 10th sample (2Hz instead of 20Hz)
    if (sampleCount % 10 === 0) {
        telemetry.l = Puck.light();
        telemetry.c = Puck.capSense();
    }

    // BATTERY OPTIMIZATION: Read battery only every 100th sample (0.2Hz - every 5 seconds)
    if (sampleCount % 100 === 0) {
        telemetry.b = Puck.getBatteryPercentage();
    }

    telemetry.s = state;
    telemetry.n = pressCount;

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
// intervalMs: optional sample interval in ms (default 50ms = 20Hz)
function getData(count, intervalMs) {
    // Default interval is 50ms (20Hz)
    if (!intervalMs || intervalMs < 10) {
        intervalMs = 50;
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
        logInfo('Stream: ' + count + ' @ ' + hz + 'Hz');
    } else {
        logInfo('Stream: ' + hz + 'Hz (30s timeout)');
    }
    
    sampleCount = 0;
    streamCount = (count && count > 0) ? count : null;
    streamStartTime = Date.now();
    streamingActive = true;  // Suppress console.log during streaming

    // Only set timeout for unlimited streaming (no count specified)
    // Fixed-count collection auto-stops when complete, no timeout needed
    if (!streamCount) {
        // Clear any existing timeout
        if (streamTimeout) {
            clearTimeout(streamTimeout);
        }
        
        // BATTERY OPTIMIZATION: Auto-stop after 30 seconds to prevent accidental battery drain
        // Only for unlimited streaming - fixed-count collection doesn't need this
        streamTimeout = setTimeout(function(){
            stopData();  // stopData re-enables logging, then logs
        }, 30000);
    }

    // Start streaming
    interval = setInterval(function() {
        emit();
        
        // Auto-stop if we've reached the target count
        if (streamCount && sampleCount >= streamCount) {
            stopData();  // stopData re-enables logging, then logs
        }
    }, intervalMs);

    return emit();
}

// Stop streaming
function stopData() {
    streamingActive = false;  // Re-enable console.log
    if (interval) {
        logInfo('Stream stopped - ' + sampleCount + ' samples');
        clearInterval(interval);
        interval = null;
    }
    if (streamTimeout) {
        clearTimeout(streamTimeout);
        streamTimeout = null;
    }
    streamCount = null;
    streamStartTime = null;
}


//NFC Detection
NRF.nfcURL("webble://simcap.parc.land/src/web/GAMBIT/");
NRF.on('NFCon', function() {
    logInfo('NFC field detected');
    digitalPulse(LED2, 1, 500);//flash on green light
    console.log('nfc_field : [1]');
    NRF.setAdvertising({
        0x183e: [1],
    }, { name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version });
});
NRF.on('NFCoff', function() {
    logDebug('NFC field removed');
    digitalPulse(LED2, 1, 200);//flash on green light
    console.log('nfc_field : [0]');
    NRF.setAdvertising({
        0x183e: [0],
    }, { name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version });
});

// Initialize Bluetooth name and appearance
init();

// //Movement Sensor
// require("puckjsv2-accel-movement").on();
// var idleTimeout;
// Puck.on('accel',function(a) {
//     // digitalWrite(LED1,1); //turn on red light
//   if (idleTimeout) clearTimeout(idleTimeout);
//   else
//     if (state === 1) {
//         console.log('movement : 1');
//         NRF.setAdvertising({
//             0x182e: [1],
//         });
//     }
//     idleTimeout = setTimeout(function() {
//         idleTimeout = undefined;
//         // digitalWrite(LED1,0);//turn off red light
//         if (state === 1) {
//             console.log('movement : 0');
//             NRF.setAdvertising({
//                 0x182e: [0],
//             });
//         }
//     },500);  
// });


// //Magnetic Field Sensor
// require("puckjsv2-mag-level").on();
// Puck.on('field',function(m) {
//     digitalPulse(LED2, 1, 200);//flash green light
//     if (state === 1) {
//         console.log('magnetic_field : [' + m.state + ']');
//         NRF.setAdvertising({
//             0x183a: [m.state],
//         });
//     }
// });

//Button Press
//Turn Off/On MQTT Advertising
var pressCount = 0;
setWatch(function() {
    pressCount++;
    state = (pressCount+1)%2;
    logInfo('Button press #' + pressCount + ' -> state=' + state);
    if ((pressCount+1)%2) digitalPulse(LED3,1,1500); //long flash blue light
    else
        digitalPulse(LED3,1,100); //short flash blue light
    getData();
    // console.log('button_press_count : [' + pressCount + ']');
    // console.log('button_state : [' + (pressCount+1) + ']');
    // console.log('state: ' + state); 
    // NRF.setAdvertising({
    //     0xFFFF : [pressCount],
    //     0x183c: [((pressCount+1)%2)],
    // });
}, BTN, { edge:"rising", repeat:true, debounce:50 });

// ===== BLE Connection Events =====
NRF.on('connect', function(addr) {
    logInfo('BLE connected: ' + addr);
});

NRF.on('disconnect', function(reason) {
    logInfo('BLE disconnected: ' + reason);
    // Stop streaming if client disconnects
    stopData();
});
