/**
 * Recording Controls
 * Manages data recording start/stop/clear operations
 *
 * Supports two modes:
 * 1. Standard mode: Data accumulates in memory, uploaded on explicit button click
 * 2. Streaming mode (local dev): Data written continuously to filesystem during recording
 */

import { state, resetSession, getSessionJSON } from './state.js';
import { log } from './logger.js';
import {
  StreamingWriter,
  getStreamingWriter,
  isLocalMode,
  type StreamingProgress
} from '../shared/streaming-writer.js';
import type { TelemetrySample } from '@core/types';

// ===== Type Definitions =====

export interface RecordingButtons {
  start?: HTMLElement | null;
  pause?: HTMLElement | null;
  stop?: HTMLElement | null;
  clear?: HTMLElement | null;
}

export interface StreamingConfig {
  /** Enable streaming mode (auto-detected if not specified) */
  enabled?: boolean;
  /** Samples per chunk (default: 500) */
  samplesPerChunk?: number;
  /** Max time between flushes in ms (default: 30000) */
  maxFlushInterval?: number;
  /** Metadata to include in exports */
  metadata?: Record<string, unknown>;
}

type UpdateUICallback = () => void;
type CloseLabelCallback = () => void;
type StreamingProgressCallback = (progress: StreamingProgress) => void;

// ===== Module State =====

let updateUI: UpdateUICallback | null = null;
let closeCurrentLabel: CloseLabelCallback | null = null;
let onStreamingProgress: StreamingProgressCallback | null = null;

// Streaming writer instance
let streamingWriter: StreamingWriter | null = null;
let streamingEnabled = false;
let streamingAutoDetected = false;

// ===== Callback Setup =====

/**
 * Set callbacks
 * @param uiCallback - Function to update UI
 * @param labelCallback - Function to close current label
 * @param streamingCallback - Function to receive streaming progress updates
 */
export function setCallbacks(
  uiCallback: UpdateUICallback | null,
  labelCallback: CloseLabelCallback | null,
  streamingCallback?: StreamingProgressCallback | null
): void {
    updateUI = uiCallback;
    closeCurrentLabel = labelCallback;
    onStreamingProgress = streamingCallback ?? null;
}

// ===== Streaming Mode =====

/**
 * Initialize streaming mode
 * Call this during app initialization to detect and configure streaming
 */
export async function initStreaming(config: StreamingConfig = {}): Promise<void> {
    // Auto-detect local mode if not explicitly configured
    if (config.enabled === undefined) {
        streamingAutoDetected = true;
        streamingEnabled = await isLocalMode();
    } else {
        streamingAutoDetected = false;
        streamingEnabled = config.enabled;
    }

    if (streamingEnabled) {
        streamingWriter = getStreamingWriter();

        // Update writer config
        (streamingWriter as any).config = {
            ...(streamingWriter as any).config,
            samplesPerChunk: config.samplesPerChunk ?? 500,
            maxFlushInterval: config.maxFlushInterval ?? 30000,
            metadata: config.metadata ?? {},
            onProgress: (progress: StreamingProgress) => {
                if (onStreamingProgress) {
                    onStreamingProgress(progress);
                }
            },
            onError: (error: Error) => {
                log(`Streaming error: ${error.message}`);
                console.error('[StreamingWriter] Error:', error);
            }
        };

        log(`Streaming mode: ${streamingAutoDetected ? 'auto-detected' : 'enabled'} (local filesystem)`);
    } else {
        log('Streaming mode: disabled (will upload on demand)');
    }
}

/**
 * Check if streaming mode is enabled
 */
export function isStreamingEnabled(): boolean {
    return streamingEnabled;
}

/**
 * Get streaming writer state
 */
export function getStreamingState(): {
    enabled: boolean;
    autoDetected: boolean;
    isActive: boolean;
    chunksWritten: number;
    samplesWritten: number;
    pendingSamples: number;
} {
    const writerState = streamingWriter?.getState();
    return {
        enabled: streamingEnabled,
        autoDetected: streamingAutoDetected,
        isActive: writerState?.isActive ?? false,
        chunksWritten: writerState?.chunksWritten ?? 0,
        samplesWritten: writerState?.samplesWritten ?? 0,
        pendingSamples: writerState?.pendingSamples ?? 0
    };
}

/**
 * Add a sample to the streaming writer (called from telemetry handler)
 */
export function addStreamingSample(sample: TelemetrySample): void {
    if (streamingEnabled && streamingWriter && state.recording && !state.paused) {
        streamingWriter.addSample(sample);
    }
}

/**
 * Update labels in the streaming writer
 */
export function updateStreamingLabels(): void {
    if (streamingEnabled && streamingWriter) {
        streamingWriter.updateLabels(state.labels);
    }
}

// ===== Recording Functions =====

/**
 * Start recording
 * @param metadata - Optional metadata to include in session export
 * @returns True if recording started successfully
 */
export async function startRecording(metadata?: Record<string, unknown>): Promise<boolean> {
    if (!state.gambitClient || !state.connected) {
        log('Error: Not connected to device');
        return false;
    }

    try {
        state.recording = true;
        state.currentLabelStart = state.sessionData.length;
        log('Recording started');

        // Start streaming session if enabled
        if (streamingEnabled && streamingWriter) {
            await streamingWriter.startSession(metadata);
            log('Streaming session started (local filesystem)');
        }

        await state.gambitClient.startStreaming();
        log('Data collection active');

        if (updateUI) updateUI();
        return true;
    } catch (e) {
        console.error('[GAMBIT] Failed to start recording:', e);
        log('Error: Failed to start data collection');
        state.recording = false;

        // Cancel streaming session on error
        if (streamingEnabled && streamingWriter) {
            streamingWriter.cancelSession();
        }

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

    // Finalize streaming session if enabled
    if (streamingEnabled && streamingWriter) {
        try {
            // Update labels before finalizing
            streamingWriter.updateLabels(state.labels);

            const manifest = await streamingWriter.finalizeSession();
            if (manifest) {
                log(`Streaming session finalized: ${manifest.totalSamples} samples in ${manifest.totalChunks} chunks`);
                console.log('[StreamingWriter] Session manifest:', manifest);
            }
        } catch (e) {
            console.error('[GAMBIT] Failed to finalize streaming session:', e);
            log(`Streaming finalization error: ${(e as Error).message}`);
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
    initStreaming,
    isStreamingEnabled,
    getStreamingState,
    addStreamingSample,
    updateStreamingLabels,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    clearSession,
    togglePause,
    initRecordingUI
};
