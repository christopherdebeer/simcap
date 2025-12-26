/**
 * Application State Management
 * Central state for GAMBIT Collector
 */

import type {
  TelemetrySample,
  LabelSegment,
  FingerState,
  GeomagneticLocation
} from '@core/types';
import type { GambitClient } from '../gambit-client';

// Re-export GambitClient type for consumers
export type { GambitClient };

// ===== Type Definitions =====

export interface FingerStates {
  thumb: FingerState;
  index: FingerState;
  middle: FingerState;
  ring: FingerState;
  pinky: FingerState;
}

export type MotionState = 'static' | 'dynamic';
export type CalibrationState = 'none' | 'mag' | 'gyro';

export interface CurrentLabels {
  pose: string | null;
  fingers: FingerStates;
  motion: MotionState;
  calibration: CalibrationState;
  custom: string[];
}

// Custom label definitions are simple strings (label names)
export type CustomLabelDefinition = string;

/** Sampling mode from firmware v0.4.0+ */
export type SamplingMode = 'LOW_POWER' | 'NORMAL' | 'HIGH_RES' | 'BURST' | null;

/** Device context from firmware v0.4.0+ */
export type DeviceContext = 'unknown' | 'stored' | 'held' | 'active' | 'table' | null;

/** Event marker from triple-tap */
export interface EventMarker {
  time: number;
  sampleCount: number;
}

/** Extended session data with markers */
export interface SessionDataExtended {
  samples?: TelemetrySample[];
  markers?: EventMarker[];
}

export interface AppState {
  connected: boolean;
  recording: boolean;
  paused: boolean;
  sessionData: TelemetrySample[] & SessionDataExtended;
  labels: LabelSegment[];
  currentLabelStart: number | null;
  gambitClient: GambitClient | null;
  firmwareVersion: string | null;
  geomagneticLocation: GeomagneticLocation | null;
  currentLabels: CurrentLabels;
  customLabelDefinitions: CustomLabelDefinition[];
  activeCustomLabels: string[];
  // v0.4.0+ state
  samplingMode: SamplingMode;
  deviceContext: DeviceContext;
}

// ===== Initial State =====

const initialFingerStates: FingerStates = {
  thumb: 'unknown',
  index: 'unknown',
  middle: 'unknown',
  ring: 'unknown',
  pinky: 'unknown'
};

const initialCurrentLabels: CurrentLabels = {
  pose: null,
  fingers: { ...initialFingerStates },
  motion: 'static',
  calibration: 'none',
  custom: []
};

// ===== Exported State =====

export const state: AppState = {
  connected: false,
  recording: false,
  paused: false,
  sessionData: [],
  labels: [],
  currentLabelStart: null,
  gambitClient: null,
  firmwareVersion: null,
  geomagneticLocation: null,
  currentLabels: { ...initialCurrentLabels },
  customLabelDefinitions: [],
  activeCustomLabels: [],
  // v0.4.0+ state
  samplingMode: null,
  deviceContext: null
};

// ===== State Functions =====

/**
 * Reset session data while preserving connection and calibration
 */
export function resetSession(): void {
  state.sessionData = [];
  state.labels = [];
  state.currentLabelStart = null;
  state.paused = false;
  state.currentLabels = {
    pose: null,
    fingers: { ...initialFingerStates },
    motion: 'static',
    calibration: 'none',
    custom: []
  };
  // Note: customLabelDefinitions, activeCustomLabels, samplingMode, deviceContext are preserved
}

/**
 * Get a snapshot of the current state (for debugging)
 */
export function getStateSnapshot(): Readonly<AppState> {
  return { ...state };
}

/**
 * Check if we're ready to record (connected but not currently recording)
 */
export function canStartRecording(): boolean {
  return state.connected && !state.recording;
}

/**
 * Check if we can pause/resume (must be recording)
 */
export function canTogglePause(): boolean {
  return state.recording;
}

/**
 * Get session segment data for a specific label
 * @param labelIndex - Index of the label in state.labels
 * @returns Object with segment samples and label info, or null if invalid
 */
export function getSessionSegment(labelIndex: number): { samples: TelemetrySample[]; label: LabelSegment } | null {
  if (labelIndex < 0 || labelIndex >= state.labels.length) {
    return null;
  }
  const label = state.labels[labelIndex];
  const samples = state.sessionData.slice(label.startIndex, label.endIndex + 1);
  return { samples, label };
}

/**
 * Get formatted session segment as JSON string
 * @param labelIndex - Index of the label in state.labels
 * @returns JSON string of the segment, or null if invalid
 */
export function getSessionSegmentJSON(labelIndex: number): string | null {
  const segment = getSessionSegment(labelIndex);
  if (!segment) return null;
  return JSON.stringify({
    label: segment.label,
    samples: segment.samples,
    sampleCount: segment.samples.length
  }, null, 2);
}

/**
 * Get full session data as JSON string
 * @returns JSON string of the entire session
 */
export function getSessionJSON(): string {
  return JSON.stringify({
    samples: state.sessionData,
    labels: state.labels,
    sampleCount: state.sessionData.length,
    labelCount: state.labels.length,
    exportedAt: new Date().toISOString()
  }, null, 2);
}

// ===== Default Export =====

export default {
  state,
  resetSession,
  getStateSnapshot,
  canStartRecording,
  canTogglePause,
  getSessionSegment,
  getSessionSegmentJSON,
  getSessionJSON
};
