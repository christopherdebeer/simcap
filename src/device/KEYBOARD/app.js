/**
 * SIMCAP KEYBOARD - BLE HID Keyboard Firmware
 *
 * Converts Puck.js gestures into keyboard/media key presses.
 * Pairs as a Bluetooth HID keyboard with any device.
 *
 * Modes:
 * - Macro Mode: Button triggers configurable key sequence
 * - Gesture Mode: Tilt directions map to arrow keys
 * - Media Mode: Control playback with gestures
 *
 * Hardware: Espruino Puck.js v2
 */

var kb = require("ble_hid_keyboard");
var controls = require("ble_hid_controls");

// Configuration
var CONFIG = {
    mode: "macro",       // "macro", "gesture", or "media"
    macroKey: "a",       // Default macro key
    macroModifier: 0,    // Modifier flags (shift=2, ctrl=1, alt=4)
    gestureThreshold: 3000, // Accelerometer threshold for gesture detection
    repeatDelay: 200     // Gesture repeat delay (ms)
};

// Key mappings for gesture mode
var GESTURE_KEYS = {
    up: kb.KEY.UP,
    down: kb.KEY.DOWN,
    left: kb.KEY.LEFT,
    right: kb.KEY.RIGHT
};

// Modifier constants
var MODIFIERS = {
    NONE: 0,
    CTRL: kb.MODIFY.CTRL,
    SHIFT: kb.MODIFY.SHIFT,
    ALT: kb.MODIFY.ALT,
    GUI: kb.MODIFY.GUI,        // Windows/Command key
    CTRL_SHIFT: kb.MODIFY.CTRL | kb.MODIFY.SHIFT,
    CTRL_ALT: kb.MODIFY.CTRL | kb.MODIFY.ALT
};

// Preset macros
var MACROS = {
    copy: { key: kb.KEY.C, mod: MODIFIERS.CTRL },
    paste: { key: kb.KEY.V, mod: MODIFIERS.CTRL },
    cut: { key: kb.KEY.X, mod: MODIFIERS.CTRL },
    undo: { key: kb.KEY.Z, mod: MODIFIERS.CTRL },
    redo: { key: kb.KEY.Y, mod: MODIFIERS.CTRL },
    save: { key: kb.KEY.S, mod: MODIFIERS.CTRL },
    selectAll: { key: kb.KEY.A, mod: MODIFIERS.CTRL },
    find: { key: kb.KEY.F, mod: MODIFIERS.CTRL },
    newTab: { key: kb.KEY.T, mod: MODIFIERS.CTRL },
    closeTab: { key: kb.KEY.W, mod: MODIFIERS.CTRL },
    switchTab: { key: kb.KEY.TAB, mod: MODIFIERS.CTRL },
    enter: { key: kb.KEY.ENTER, mod: MODIFIERS.NONE },
    escape: { key: kb.KEY.ESC, mod: MODIFIERS.NONE },
    space: { key: kb.KEY[" "], mod: MODIFIERS.NONE }
};

// State
var state = {
    enabled: true,
    lastGesture: null,
    lastGestureTime: 0,
    buttonHeld: false,
    pressCount: 0
};

// Initialize HID services
function init() {
    // Combine keyboard and media control reports
    NRF.setServices(undefined, { hid: kb.report });

    // Set HID appearance for Windows 11 compatibility
    NRF.setAdvertising([
        {},
        [
            2, 1, 6,           // Bluetooth flags
            3, 3, 0x12, 0x18,  // HID Service UUID
            3, 0x19, 0xc1, 0x03 // Appearance: Keyboard (0x03C1)
        ]
    ], { name: "SIMCAP Keys" });

    console.log("SIMCAP Keyboard initialized");
    console.log("Mode:", CONFIG.mode);
    flashLED(LED2, 3);
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

// Send a key tap
function sendKey(key, modifier) {
    modifier = modifier || 0;
    kb.tap(key, modifier, function() {
        // Key released
    });
}

// Send a macro by name
function sendMacro(name) {
    var macro = MACROS[name];
    if (macro) {
        sendKey(macro.key, macro.mod);
        digitalPulse(LED2, 1, 50);
    } else {
        console.log("Unknown macro:", name);
    }
}

// Type a string
function typeString(str) {
    var i = 0;
    function typeNext() {
        if (i >= str.length) return;
        var char = str[i++];
        var key = kb.KEY[char.toUpperCase()];
        var mod = (char === char.toUpperCase() && char !== char.toLowerCase()) ?
                  kb.MODIFY.SHIFT : 0;
        if (key) {
            kb.tap(key, mod, typeNext);
        } else {
            typeNext();
        }
    }
    typeNext();
}

// Detect gesture from accelerometer
function detectGesture(accel) {
    var x = accel.x;
    var y = accel.y;
    var threshold = CONFIG.gestureThreshold;

    if (Math.abs(x) > threshold || Math.abs(y) > threshold) {
        if (Math.abs(x) > Math.abs(y)) {
            return x > 0 ? "right" : "left";
        } else {
            return y > 0 ? "down" : "up";
        }
    }
    return null;
}

// Handle gesture in gesture mode
function handleGesture(gesture) {
    var now = Date.now();

    // Debounce repeated gestures
    if (gesture === state.lastGesture &&
        now - state.lastGestureTime < CONFIG.repeatDelay) {
        return;
    }

    state.lastGesture = gesture;
    state.lastGestureTime = now;

    var key = GESTURE_KEYS[gesture];
    if (key) {
        sendKey(key, 0);
        digitalPulse(LED3, 1, 30);
    }
}

// Handle gesture in media mode
function handleMediaGesture(gesture) {
    var now = Date.now();

    if (gesture === state.lastGesture &&
        now - state.lastGestureTime < CONFIG.repeatDelay * 2) {
        return;
    }

    state.lastGesture = gesture;
    state.lastGestureTime = now;

    switch (gesture) {
        case "up":
            controls.volumeUp();
            break;
        case "down":
            controls.volumeDown();
            break;
        case "left":
            controls.prev();
            break;
        case "right":
            controls.next();
            break;
    }
    digitalPulse(LED3, 1, 30);
}

// Main gesture detection loop
var gestureInterval;
function startGestureDetection() {
    if (gestureInterval) return;

    gestureInterval = setInterval(function() {
        if (!state.enabled) return;
        if (CONFIG.mode === "macro") return; // No gesture detection in macro mode

        var accel = Puck.accel();
        var gesture = detectGesture(accel.acc);

        if (gesture) {
            if (CONFIG.mode === "gesture") {
                handleGesture(gesture);
            } else if (CONFIG.mode === "media") {
                handleMediaGesture(gesture);
            }
        } else {
            state.lastGesture = null;
        }
    }, 50); // 20Hz
}

function stopGestureDetection() {
    if (gestureInterval) {
        clearInterval(gestureInterval);
        gestureInterval = null;
    }
}

// Button handling
function onButtonPress() {
    state.buttonHeld = true;
    state.pressCount++;

    if (CONFIG.mode === "macro") {
        // Execute macro on press
        var key = CONFIG.macroKey;
        if (typeof key === "string") {
            if (MACROS[key]) {
                sendMacro(key);
            } else {
                sendKey(kb.KEY[key.toUpperCase()] || kb.KEY.A, CONFIG.macroModifier);
            }
        }
        digitalPulse(LED1, 1, 50);
    } else if (CONFIG.mode === "media") {
        // Play/pause on button press
        controls.playpause();
        digitalPulse(LED2, 1, 50);
    }
}

function onButtonRelease() {
    state.buttonHeld = false;
}

// Cycle through modes
function cycleMode() {
    var modes = ["macro", "gesture", "media"];
    var idx = modes.indexOf(CONFIG.mode);
    CONFIG.mode = modes[(idx + 1) % modes.length];

    console.log("Mode:", CONFIG.mode);

    // LED feedback for mode
    switch (CONFIG.mode) {
        case "macro":
            flashLED(LED1, 1); // Red
            break;
        case "gesture":
            flashLED(LED2, 1); // Green
            break;
        case "media":
            flashLED(LED3, 1); // Blue
            break;
    }
}

// NFC tap to cycle modes
NRF.nfcURL("https://christopherdebeer.github.io/simcap/src/web/loader/");
NRF.on('NFCon', function() {
    cycleMode();
});

// Long press to toggle enabled
var longPressTimer;
setWatch(function() {
    longPressTimer = setTimeout(function() {
        state.enabled = !state.enabled;
        if (state.enabled) {
            flashLED(LED2, 2);
            startGestureDetection();
        } else {
            flashLED(LED1, 2);
            stopGestureDetection();
        }
    }, 1000);
    onButtonPress();
}, BTN, { edge: "rising", repeat: true, debounce: 50 });

setWatch(function() {
    if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
    }
    onButtonRelease();
}, BTN, { edge: "falling", repeat: true, debounce: 50 });

// Telemetry
function getTelemetry() {
    return {
        mode: CONFIG.mode,
        enabled: state.enabled,
        pressCount: state.pressCount,
        battery: Puck.getBatteryPercentage()
    };
}

// Initialize
init();
startGestureDetection();

// Export for console access
global.CONFIG = CONFIG;
global.MACROS = MACROS;
global.MODIFIERS = MODIFIERS;
global.state = state;
global.getTelemetry = getTelemetry;
global.sendKey = sendKey;
global.sendMacro = sendMacro;
global.typeString = typeString;
global.cycleMode = cycleMode;
