/**
 * Recording Controls
 * Manages data recording start/stop/clear operations
 */

import { state, resetSession } from './state.js';
import { log } from './logger.js';

let updateUI = null;
let closeCurrentLabel = null;

/**
 * Set callbacks
 * @param {Function} uiCallback - Function to update UI
 * @param {Function} labelCallback - Function to close current label
 */
export function setCallbacks(uiCallback, labelCallback) {
    updateUI = uiCallback;
    closeCurrentLabel = labelCallback;
}

/**
 * Start recording
 * @returns {Promise<boolean>} True if recording started successfully
 */
export async function startRecording() {
    if (!state.gambitClient || !state.connected) {
        log('Error: Not connected to device');
        return false;
    }

    try {
        state.recording = true;
        state.currentLabelStart = state.sessionData.length;
        log('Recording started');

        await state.gambitClient.startStreaming();
        log('Data collection active');

        if (updateUI) updateUI();
        return true;
    } catch (e) {
        console.error('[GAMBIT] Failed to start recording:', e);
        log('Error: Failed to start data collection');
        state.recording = false;
        if (updateUI) updateUI();
        return false;
    }
}

/**
 * Stop recording
 * @returns {Promise<void>}
 */
export async function stopRecording() {
    if (closeCurrentLabel) closeCurrentLabel();

    state.recording = false;
    state.paused = false;
    state.currentLabelStart = null;
    log('Recording stopped');

    if (state.gambitClient && state.connected) {
        try {
            await state.gambitClient.stopStreaming();
            log('Data streaming stopped');
        } catch (e) {
            console.error('[GAMBIT] Failed to stop streaming:', e);
        }
    }

    if (updateUI) updateUI();
}

/**
 * Pause recording (keeps streaming but stops storing data)
 */
export function pauseRecording() {
    if (!state.recording || state.paused) {
        return;
    }

    // Close current label segment before pausing
    if (closeCurrentLabel) closeCurrentLabel();

    state.paused = true;
    log('Recording paused');

    if (updateUI) updateUI();
}

/**
 * Resume recording from pause
 */
export function resumeRecording() {
    if (!state.recording || !state.paused) {
        return;
    }

    state.paused = false;
    state.currentLabelStart = state.sessionData.length;
    log('Recording resumed');

    if (updateUI) updateUI();
}

/**
 * Clear session data
 * @param {boolean} confirm - Whether to show confirmation dialog
 * @returns {boolean} True if cleared
 */
export function clearSession(confirm = true) {
    if (confirm && !window.confirm('Clear all session data?')) {
        return false;
    }

    resetSession();
    log('Session cleared');
    if (updateUI) updateUI();
    return true;
}

/**
 * Toggle pause/resume
 */
export function togglePause() {
    if (state.paused) {
        resumeRecording();
    } else {
        pauseRecording();
    }
}

/**
 * Initialize recording UI controls
 * @param {Object} buttons - Button elements {start, pause, stop, clear}
 */
export function initRecordingUI(buttons) {
    if (!buttons) return;

    if (buttons.start) {
        buttons.start.addEventListener('click', startRecording);
    }

    if (buttons.pause) {
        buttons.pause.addEventListener('click', togglePause);
    }

    if (buttons.stop) {
        buttons.stop.addEventListener('click', stopRecording);
    }

    if (buttons.clear) {
        buttons.clear.addEventListener('click', () => clearSession(true));
    }
}
