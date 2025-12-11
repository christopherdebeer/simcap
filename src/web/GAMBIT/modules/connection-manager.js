/**
 * Connection Manager
 * Handles GAMBIT device connection/disconnection
 */

import { state } from './state.js';
import { log } from './logger.js';
import { onTelemetry } from './telemetry-handler.js';

let updateUI = null;
let updateCalibrationStatus = null;

/**
 * Set callbacks for UI updates
 * @param {Function} uiCallback - Function to update UI
 * @param {Function} calibrationCallback - Function to update calibration status
 */
export function setCallbacks(uiCallback, calibrationCallback) {
    updateUI = uiCallback;
    updateCalibrationStatus = calibrationCallback;
}

/**
 * Connect to GAMBIT device
 * @returns {Promise<boolean>} True if connected successfully
 */
export async function connect() {
    if (state.connected) {
        // Already connected, disconnect first
        disconnect();
        return false;
    }

    log('Connecting...');

    try {
        state.gambitClient = new GambitClient({
            debug: true,
            autoKeepalive: true  // Enable keepalive to prevent 30s firmware timeout during recording
        });

        // Register data handler
        state.gambitClient.on('data', onTelemetry);

        // Handle firmware info
        state.gambitClient.on('firmware', (info) => {
            console.log('[GAMBIT] Firmware info:', info);
            // Check compatibility with minimum version 0.1.0
            const compat = state.gambitClient.checkCompatibility('0.1.0');
            if (!compat.compatible) {
                log(`Incompatible firmware: ${compat.reason}`);
                setTimeout(() => state.gambitClient.disconnect(), 3000);
                return;
            }
            log(`Firmware: ${info.name} v${info.version}`);
        });

        // Handle disconnection
        state.gambitClient.on('disconnect', () => {
            console.log('[GAMBIT] Device disconnected');
            state.connected = false;
            state.recording = false;
            log('Connection closed');
            if (updateUI) updateUI();
            if (updateCalibrationStatus) updateCalibrationStatus();
        });

        // Handle errors
        state.gambitClient.on('error', (err) => {
            console.error('[GAMBIT] Error:', err);
            log(`Error: ${err.message}`);
        });

        // Attempt connection
        await state.gambitClient.connect();

        state.connected = true;
        log('Connected!');
        if (updateUI) updateUI();
        if (updateCalibrationStatus) updateCalibrationStatus();

        return true;

    } catch (e) {
        console.error('[GAMBIT] Connection error:', e);
        log(`Connection failed: ${e.message || e.toString() || 'Unknown error'}`);
        if (state.gambitClient) {
            state.gambitClient.disconnect();
            state.gambitClient = null;
        }
        return false;
    }
}

/**
 * Disconnect from GAMBIT device
 */
export function disconnect() {
    console.log('[GAMBIT] Disconnecting...');
    if (state.gambitClient) {
        state.gambitClient.disconnect();
        state.gambitClient = null;
    }
    state.connected = false;
    log('Disconnected');
    if (updateUI) updateUI();
    if (updateCalibrationStatus) updateCalibrationStatus();
}

/**
 * Toggle connection state
 * @returns {Promise<boolean>} New connection state
 */
export async function toggleConnection() {
    if (state.connected) {
        disconnect();
        return false;
    } else {
        return await connect();
    }
}

/**
 * Initialize connection UI controls
 * @param {HTMLElement} connectButton - Connect/disconnect button element
 */
export function initConnectionUI(connectButton) {
    if (connectButton) {
        connectButton.addEventListener('click', toggleConnection);
    }
}
