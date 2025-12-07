/**
 * SIMCAP MOUSE - BLE HID Mouse Firmware
 *
 * Converts Puck.js IMU data into mouse movements.
 * Pairs as a Bluetooth HID mouse with any device.
 *
 * Controls:
 * - Tilt device to move cursor (accelerometer)
 * - Button press: Left click
 * - Double press: Right click
 * - Hold button + tilt: Scroll mode
 *
 * Hardware: Espruino Puck.js v2
 */

var mouse = require("ble_hid_mouse");

// Configuration
var CONFIG = {
    sensitivity: 3,      // Mouse movement sensitivity (1-10)
    deadzone: 500,       // Ignore small tilts (accelerometer units)
    scrollSensitivity: 2,
    doubleClickTime: 300, // ms between clicks for double-click
    invertX: false,
    invertY: false
};

// State
var state = {
    enabled: true,
    scrollMode: false,
    lastClickTime: 0,
    clickCount: 0,
    buttonHeld: false
};

// Initialize HID services
function init() {
    NRF.setServices(undefined, { hid: mouse.report });

    // Set HID appearance for better compatibility (Windows 11)
    NRF.setAdvertising([
        {}, // include original advertising packet
        [
            2, 1, 6,           // Bluetooth flags
            3, 3, 0x12, 0x18,  // HID Service UUID
            3, 0x19, 0xc2, 0x03 // Appearance: Mouse (0x03C2)
        ]
    ], { name: "SIMCAP Mouse" });

    console.log("SIMCAP Mouse initialized");
    flashLED(LED2, 3); // Green flash to indicate ready
}

// Flash LED pattern
function flashLED(led, times) {
    var count = 0;
    var interval = setInterval(function() {
        digitalPulse(led, 1, 100);
        count++;
        if (count >= times) clearInterval(interval);
    }, 200);
}

// Convert accelerometer to mouse movement
function accelToMouse(accel) {
    var x = accel.x;
    var y = accel.y;

    // Apply deadzone
    if (Math.abs(x) < CONFIG.deadzone) x = 0;
    if (Math.abs(y) < CONFIG.deadzone) y = 0;

    // Scale and apply sensitivity
    x = Math.round((x / 8192) * CONFIG.sensitivity * 10);
    y = Math.round((y / 8192) * CONFIG.sensitivity * 10);

    // Clamp to valid range (-127 to 127)
    x = Math.max(-127, Math.min(127, x));
    y = Math.max(-127, Math.min(127, y));

    // Apply inversion
    if (CONFIG.invertX) x = -x;
    if (CONFIG.invertY) y = -y;

    return { x: x, y: y };
}

// Main movement loop
var moveInterval;
function startMovement() {
    if (moveInterval) return;

    moveInterval = setInterval(function() {
        if (!state.enabled) return;

        var accel = Puck.accel();
        var move = accelToMouse(accel.acc);

        if (state.scrollMode) {
            // Scroll mode: use Y movement for scrolling
            if (move.y !== 0) {
                var scroll = Math.round(move.y / 5);
                mouse.send(0, 0, 0, scroll);
            }
        } else {
            // Normal mode: move cursor
            if (move.x !== 0 || move.y !== 0) {
                mouse.send(move.x, move.y, 0);
            }
        }
    }, 20); // 50Hz update rate
}

function stopMovement() {
    if (moveInterval) {
        clearInterval(moveInterval);
        moveInterval = null;
    }
}

// Button handling
function onButtonPress() {
    var now = Date.now();
    state.buttonHeld = true;

    // Check for double-click
    if (now - state.lastClickTime < CONFIG.doubleClickTime) {
        state.clickCount++;
    } else {
        state.clickCount = 1;
    }
    state.lastClickTime = now;

    // Start scroll mode after short hold
    setTimeout(function() {
        if (state.buttonHeld) {
            state.scrollMode = true;
            digitalPulse(LED3, 1, 100); // Blue flash for scroll mode
        }
    }, 300);
}

function onButtonRelease() {
    state.buttonHeld = false;

    if (state.scrollMode) {
        state.scrollMode = false;
    } else if (state.clickCount === 1) {
        // Single click: left button
        mouse.send(0, 0, mouse.BUTTONS.LEFT);
        setTimeout(function() {
            mouse.send(0, 0, 0); // Release
        }, 50);
        digitalPulse(LED1, 1, 50); // Red flash
    } else if (state.clickCount >= 2) {
        // Double click: right button
        mouse.send(0, 0, mouse.BUTTONS.RIGHT);
        setTimeout(function() {
            mouse.send(0, 0, 0); // Release
        }, 50);
        digitalPulse(LED3, 1, 50); // Blue flash
        state.clickCount = 0;
    }
}

// NFC tap to toggle enabled state
NRF.nfcURL("https://christopherdebeer.github.io/simcap/src/web/loader/");
NRF.on('NFCon', function() {
    state.enabled = !state.enabled;
    if (state.enabled) {
        flashLED(LED2, 2);
        startMovement();
    } else {
        flashLED(LED1, 2);
        stopMovement();
    }
});

// Button watchers
setWatch(onButtonPress, BTN, { edge: "rising", repeat: true, debounce: 50 });
setWatch(onButtonRelease, BTN, { edge: "falling", repeat: true, debounce: 50 });

// Telemetry for debugging (can be read via BLE console)
function getTelemetry() {
    var accel = Puck.accel();
    return {
        acc: accel.acc,
        enabled: state.enabled,
        scrollMode: state.scrollMode,
        battery: Puck.getBatteryPercentage()
    };
}

// Initialize
init();
startMovement();

// Export for console access
global.CONFIG = CONFIG;
global.state = state;
global.getTelemetry = getTelemetry;
