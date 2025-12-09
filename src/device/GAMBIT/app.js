// ===== GAMBIT Firmware Configuration =====
var FIRMWARE_INFO = {
    id: "GAMBIT",
    name: "GAMBIT IMU Telemetry",
    version: "0.1.1",
    features: ["imu", "magnetometer", "environmental", "streaming"],
    author: "SIMCAP"
};

// Track boot time for uptime calculation
var bootTime = Date.now();

// Return firmware information for compatibility checking
function getFirmware() {
    var uptimeMs = Date.now() - bootTime;
    var info = Object.assign({}, FIRMWARE_INFO, { uptime: uptimeMs });
    console.log("\nFIRMWARE" + JSON.stringify(info));
    return info;
}

// Initialize Bluetooth advertising with proper device name and appearance
function init() {
    // Set Bluetooth appearance and flags for sensor device
    NRF.setAdvertising([
        {}, // include original advertising packet
        [
            2, 1, 6,           // Bluetooth flags (General Discoverable, BR/EDR Not Supported)
            3, 0x19, 0x40, 0x05 // Appearance: Generic Sensor (0x0540)
        ]
    ], { name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version });

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
    // Magnetometer is power-hungry (requires sensor wake-up) - read every 5th sample (10Hz instead of 20Hz)
    if (sampleCount % 5 === 0) {
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

    console.log("\nGAMBIT" + JSON.stringify(telemetry));
    return telemetry;
}

var interval;
var streamTimeout;

function getData() {
    // Reset sample counter if starting new stream
    if (!interval) {
        sampleCount = 0;
    }

    // CRITICAL: Always refresh the 30-second timeout when getData() is called
    // This enables keepalive mechanism (web app calls getData() every 25s to prevent timeout)
    if (streamTimeout) {
        clearTimeout(streamTimeout);
    }

    // BATTERY OPTIMIZATION: Auto-stop after 30 seconds to prevent accidental battery drain
    streamTimeout = setTimeout(function(){
        clearInterval(interval);
        interval = null;
        streamTimeout = null;
    }, 30000);

    // Start streaming at 20Hz (50ms interval) if not already running
    if (!interval) {
        interval = setInterval(emit, 50);
    }

    return emit();
}

// Optional: Stop streaming manually
function stopData() {
    if (interval) {
        clearInterval(interval);
        interval = null;
    }
    if (streamTimeout) {
        clearTimeout(streamTimeout);
        streamTimeout = null;
    }
}


//NFC Detection
NRF.nfcURL("https://simcap.parc.land");
NRF.on('NFCon', function() {
    digitalPulse(LED2, 1, 500);//flash on green light
    console.log('nfc_field : [1]');
    NRF.setAdvertising({
        0x183e: [1],
    }, { name: "SIMCAP GAMBIT v" + FIRMWARE_INFO.version });
});
NRF.on('NFCoff', function() {
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
    if ((pressCount+1)%2) digitalPulse(LED3,1,1500); //long flash blue light
    else
        digitalPulse(LED3,1,100); //short flash blue light
    getData()
    // console.log('button_press_count : [' + pressCount + ']');
    // console.log('button_state : [' + (pressCount+1) + ']');
    // console.log('state: ' + state); 
    // NRF.setAdvertising({
    //     0xFFFF : [pressCount],
    //     0x183c: [((pressCount+1)%2)],
    // });
}, BTN, { edge:"rising", repeat:true, debounce:50 });