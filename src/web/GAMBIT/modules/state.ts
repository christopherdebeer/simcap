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

export interface AppState {
  connected: boolean;
  recording: boolean;
  paused: boolean;
  sessionData: TelemetrySample[];
  labels: LabelSegment[];
  currentLabelStart: number | null;
  gambitClient: GambitClient | null;
  firmwareVersion: string | null;
  geomagneticLocation: GeomagneticLocation | null;
  currentLabels: CurrentLabels;
  customLabelDefinitions: CustomLabelDefinition[];
  activeCustomLabels: string[];
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
  activeCustomLabels: []
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
  // Note: customLabelDefinitions and activeCustomLabels are preserved
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

// ===== Default Export =====

export default {
  state,
  resetSession,
  getStateSnapshot,
  canStartRecording,
  canTogglePause
};
