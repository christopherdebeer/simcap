# GAMBIT Collector Modules

This directory contains the modular JavaScript architecture for the GAMBIT Collector web application.

## Architecture Overview

The collector has been refactored from a monolithic 3000-line HTML file into a clean modular architecture:

```
collector.html (38K)           ← Main HTML/CSS
├── collector-app.js (15K)     ← Application coordinator
└── modules/
    ├── state.js               ← Application state
    ├── logger.js              ← Logging utility
    ├── calibration-ui.js      ← Calibration controls (WITH STREAMING FIX)
    ├── telemetry-handler.js   ← Sensor data processing
    ├── connection-manager.js  ← Device connection
    └── recording-controls.js  ← Recording controls
```

## Module Descriptions

### `state.js`
- Centralized application state management
- Exports `state` object and `resetSession()` function
- Contains: connection status, recording state, session data, labels, current labels

### `logger.js`
- Timestamped logging to UI
- Exports: `initLogger()`, `log()`
- Maintains last 50 log messages

### `calibration-ui.js` ⭐
- Calibration UI controls and step execution
- **Contains the critical streaming fix**
- Exports: `initCalibration()`, `updateCalibrationStatus()`, `initCalibrationUI()`, `runCalibrationStep()`
- **Fix**: `runCalibrationStep()` now automatically starts/stops streaming as needed

### `telemetry-handler.js`
- Processes incoming sensor data
- Applies calibration corrections, IMU fusion, Kalman filtering
- Decorates telemetry with processed fields
- Exports: `onTelemetry()`, `setDependencies()`, `resetIMU()`

### `connection-manager.js`
- GAMBIT device connection/disconnection
- Registers event handlers for firmware, errors, disconnection
- Exports: `connect()`, `disconnect()`, `toggleConnection()`, `initConnectionUI()`

### `recording-controls.js`
- Recording start/stop/clear operations
- Manages streaming state
- Exports: `startRecording()`, `stopRecording()`, `clearSession()`, `initRecordingUI()`

### `collector-app.js`
- Main application coordinator
- Initializes all modules and wires dependencies
- Contains UI update logic and label management
- Entry point for the application

## Key Changes

### The Calibration Streaming Fix

**Problem**: The original `runCalibrationStep()` function would register a data handler and wait for samples, but never actually started data streaming from the device. This resulted in the "Failed: insufficient data" error.

**Solution**: The new implementation in `calibration-ui.js`:

```javascript
async function runCalibrationStep(stepName, durationMs, sampleHandler, completionHandler) {
    // ... connection check ...

    let streamingStartedByUs = false;

    try {
        // FIX: Ensure streaming is active before collecting data
        if (!state.recording) {
            await state.gambitClient.startStreaming();
            streamingStartedByUs = true;
        }

        // Collect data...

        // FIX: Stop streaming if we started it (and not recording)
        if (streamingStartedByUs && !state.recording) {
            state.gambitClient.stopStreaming();
        }
    } catch (error) {
        // Clean up streaming if needed
    }
}
```

This ensures:
1. Streaming is active during calibration
2. Streaming is properly stopped after calibration
3. Doesn't interfere with existing recording sessions
4. Handles errors gracefully

## Benefits of Modular Architecture

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Modules can be tested independently
3. **Readability**: Clear separation of concerns
4. **Reusability**: Modules can be reused in other applications
5. **Debuggability**: Easier to locate and fix issues
6. **Size**: HTML file reduced from 119K to 38K (68% reduction)

## Usage

The modules use ES6 module syntax and are loaded via:

```html
<script type="module" src="./collector-app.js"></script>
```

All modules are automatically initialized when the DOM is ready.

## Dependencies

External dependencies (loaded via script tags in HTML):
- `gambit-client.js` - GAMBIT device client
- `calibration.js` - Environmental calibration algorithms
- `hand-model.js` - Hand model and pose estimation
- `filters.js` - Kalman filter and IMU fusion

## Development

When modifying the modules:

1. **State changes**: Modify `state.js`
2. **Calibration logic**: Modify `calibration-ui.js`
3. **Data processing**: Modify `telemetry-handler.js`
4. **Connection handling**: Modify `connection-manager.js`
5. **UI updates**: Modify `collector-app.js`

All modules export their public API, making it easy to understand what functionality is available.

## Testing

To test the calibration fix:

1. Connect to GAMBIT device
2. Navigate to calibration panel
3. Click "Start Earth Field Calibration" **without** starting recording first
4. Verify: Data is collected and calibration completes successfully
5. Verify: Progress bar shows 0-100%
6. Verify: Quality metrics are displayed
7. Verify: No "Failed: insufficient data" error

## Migration Notes

All functionality from the original monolithic implementation has been preserved:
- Device connection/disconnection
- Data recording and streaming
- Real-time calibration and filtering
- Label management (multi-label system)
- Data export
- Calibration wizard
- GitHub upload
- Collapsible sections
- Custom labels

The refactoring is purely architectural - no features were removed or changed (except for the calibration streaming bug fix).
