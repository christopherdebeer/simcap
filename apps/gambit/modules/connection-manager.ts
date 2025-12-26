/**
 * Connection Manager
 * Handles GAMBIT device connection/disconnection
 */

import { state } from './state.js';
import { log } from './logger.js';
import { onTelemetry } from './telemetry-handler.js';
import { GambitClient } from '../gambit-client';

// ===== Type Definitions =====

type UpdateUICallback = () => void;
type UpdateCalibrationCallback = () => void;
type ConnectCallback = () => void;
type DisconnectCallback = () => void;

interface FirmwareInfo {
  name: string;
  version: string;
}

interface CompatibilityResult {
  compatible: boolean;
  reason?: string;
}

// ===== Module State =====

let updateUI: UpdateUICallback | null = null;
let updateCalibrationStatus: UpdateCalibrationCallback | null = null;
let onConnectCallback: ConnectCallback | null = null;
let onDisconnectCallback: DisconnectCallback | null = null;

// ===== Callback Setup =====

/**
 * Set callbacks for UI updates
 * @param uiCallback - Function to update UI
 * @param calibrationCallback - Function to update calibration status
 * @param connectCallback - Function called on successful connection
 * @param disconnectCallback - Function called on disconnection
 */
export function setCallbacks(
  uiCallback: UpdateUICallback | null,
  calibrationCallback: UpdateCalibrationCallback | null,
  connectCallback: ConnectCallback | null = null,
  disconnectCallback: DisconnectCallback | null = null
): void {
    updateUI = uiCallback;
    updateCalibrationStatus = calibrationCallback;
    onConnectCallback = connectCallback;
    onDisconnectCallback = disconnectCallback;
}

// ===== Connection Functions =====

/**
 * Connect to GAMBIT device
 * @returns True if connected successfully
 */
export async function connect(): Promise<boolean> {
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
        state.gambitClient!.on('data', onTelemetry);

        // Handle firmware info
        state.gambitClient!.on('firmware', (info: FirmwareInfo) => {
            console.log('[GAMBIT] Firmware info:', info);
            // Store firmware version for session metadata
            state.firmwareVersion = info.version;
            // Check compatibility with minimum version 0.1.0
            const compat: CompatibilityResult = state.gambitClient!.checkCompatibility('0.1.0');
            if (!compat.compatible) {
                log(`Incompatible firmware: ${compat.reason}`);
                setTimeout(() => state.gambitClient!.disconnect(), 3000);
                return;
            }
            log(`Firmware: ${info.name} v${info.version}`);
        });

        // Handle disconnection
        state.gambitClient!.on('disconnect', () => {
            console.log('[GAMBIT] Device disconnected');
            state.connected = false;
            state.recording = false;
            log('Connection closed');
            if (updateUI) updateUI();
            if (updateCalibrationStatus) updateCalibrationStatus();
            if (onDisconnectCallback) onDisconnectCallback();
        });

        // Handle errors
        state.gambitClient!.on('error', (err: Error) => {
            console.error('[GAMBIT] Error:', err);
            log(`Error: ${err.message}`);
        });

        // Handle button gestures (v0.4.0+)
        state.gambitClient!.on('button', (event) => {
            console.log('[GAMBIT] Button gesture:', event.gesture);
            log(`Button: ${event.gesture}`);
        });

        // Handle mode changes (v0.4.0+)
        state.gambitClient!.on('mode', (event) => {
            console.log('[GAMBIT] Mode changed:', event.mode);
            state.samplingMode = event.mode;
            log(`Mode: ${event.mode} (${event.config.accelHz}Hz)`);
            if (updateUI) updateUI();
        });

        // Handle context changes (v0.4.0+)
        state.gambitClient!.on('context', (event) => {
            console.log('[GAMBIT] Context changed:', event.context);
            state.deviceContext = event.context;
            log(`Context: ${event.from} → ${event.context}`);
            if (updateUI) updateUI();
        });

        // Handle stream events (v0.4.0+)
        state.gambitClient!.on('streamStart', (event) => {
            if (event) {
                console.log('[GAMBIT] Stream started:', event.mode, event.hz + 'Hz');
            }
        });

        state.gambitClient!.on('streamStop', (event) => {
            if (event) {
                console.log('[GAMBIT] Stream stopped:', event.samples, 'samples');
            }
        });

        // Handle event markers (v0.4.0+)
        state.gambitClient!.on('mark', (event) => {
            console.log('[GAMBIT] Event marked at sample:', event.sampleCount);
            log(`Marker at sample ${event.sampleCount}`);
            // Store marker in session if recording
            if (state.recording && state.sessionData) {
                if (!state.sessionData.markers) {
                    state.sessionData.markers = [];
                }
                state.sessionData.markers.push({
                    time: event.time,
                    sampleCount: event.sampleCount
                });
            }
        });

        // Handle calibration events (v0.4.0+)
        state.gambitClient!.on('calibration', (event) => {
            console.log('[GAMBIT] Calibration:', event);
            log(`Calibrated: light=${event.light?.toFixed(3)}, cap=${event.cap?.toFixed(0)}`);
        });

        // Attempt connection
        await state.gambitClient!.connect();

        state.connected = true;
        log('Connected!');
        if (updateUI) updateUI();
        if (updateCalibrationStatus) updateCalibrationStatus();
        if (onConnectCallback) onConnectCallback();

        return true;

    } catch (e) {
        const error = e as Error;
        console.error('[GAMBIT] Connection error:', e);
        log(`Connection failed: ${error.message || error.toString() || 'Unknown error'}`);
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
export function disconnect(): void {
    console.log('[GAMBIT] Disconnecting...');
    if (state.gambitClient) {
        state.gambitClient.disconnect();
        state.gambitClient = null;
    }
    state.connected = false;
    log('Disconnected');
    if (updateUI) updateUI();
    if (updateCalibrationStatus) updateCalibrationStatus();
    if (onDisconnectCallback) onDisconnectCallback();
}

/**
 * Toggle connection state
 * @returns New connection state
 */
export async function toggleConnection(): Promise<boolean> {
    if (state.connected) {
        disconnect();
        return false;
    } else {
        return await connect();
    }
}

// ===== UI Initialization =====

/**
 * Initialize connection UI controls
 * @param connectButton - Connect/disconnect button element
 */
export function initConnectionUI(connectButton: HTMLElement | null): void {
    if (connectButton) {
        connectButton.addEventListener('click', toggleConnection);
    }
}

// ===== Default Export =====

export default {
    setCallbacks,
    connect,
    disconnect,
    toggleConnection,
    initConnectionUI
};
