/**
 * Recording Controls
 * Manages data recording start/stop/clear operations
 */

import { state, resetSession, getSessionJSON } from './state.js';
import { log } from './logger.js';

// ===== Type Definitions =====

export interface RecordingButtons {
  start?: HTMLElement | null;
  pause?: HTMLElement | null;
  stop?: HTMLElement | null;
  clear?: HTMLElement | null;
}

type UpdateUICallback = () => void;
type CloseLabelCallback = () => void;

// ===== Module State =====

let updateUI: UpdateUICallback | null = null;
let closeCurrentLabel: CloseLabelCallback | null = null;

// ===== Callback Setup =====

/**
 * Set callbacks
 * @param uiCallback - Function to update UI
 * @param labelCallback - Function to close current label
 */
export function setCallbacks(
  uiCallback: UpdateUICallback | null,
  labelCallback: CloseLabelCallback | null
): void {
    updateUI = uiCallback;
    closeCurrentLabel = labelCallback;
}

// ===== Recording Functions =====

/**
 * Start recording
 * @returns True if recording started successfully
 */
export async function startRecording(): Promise<boolean> {
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
 */
export async function stopRecording(): Promise<void> {
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

    // Log session data to console for easy access
    if (state.sessionData.length > 0) {
        const sessionJSON = getSessionJSON();
        console.log('=== SESSION DATA ===');
        console.log(sessionJSON);
        console.log('=== END SESSION DATA ===');
        log(`Session data logged to console (${state.sessionData.length} samples, ${state.labels.length} labels)`);
    }

    if (updateUI) updateUI();
}

/**
 * Pause recording (keeps streaming but stops storing data)
 */
export function pauseRecording(): void {
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
export function resumeRecording(): void {
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
 * @param confirm - Whether to show confirmation dialog
 * @returns True if cleared
 */
export function clearSession(confirm: boolean = true): boolean {
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
export function togglePause(): void {
    if (state.paused) {
        resumeRecording();
    } else {
        pauseRecording();
    }
}

// ===== UI Initialization =====

/**
 * Initialize recording UI controls
 * @param buttons - Button elements {start, pause, stop, clear}
 */
export function initRecordingUI(buttons: RecordingButtons | null): void {
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

// ===== Default Export =====

export default {
    setCallbacks,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    clearSession,
    togglePause,
    initRecordingUI
};
