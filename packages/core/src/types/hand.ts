/**
 * Hand and Finger Types
 *
 * Types for finger labeling, tracking, and hand pose representation.
 * Differentiates between labeling (discrete states) and tracking (continuous values).
 *
 * @module core/types/hand
 */

import type { Vector3 } from './geometry';

// ===== Finger Names =====

/** Names of the five fingers */
export type FingerName = 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';

/** Array of finger names in order */
export const FINGER_NAMES: readonly FingerName[] = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const;

// ===== Finger Labeling (Discrete States for Annotation) =====

/**
 * Finger flexion label for data annotation.
 * Used when labeling recorded sessions with discrete finger states.
 *
 * - extended: Finger fully straight (0 flexion)
 * - curled: Finger partially bent/hooked (intermediate flexion)
 * - flexed: Finger fully bent/closed (full flexion)
 * - unknown: State not determined
 */
export type FingerLabel = 'extended' | 'curled' | 'flexed' | 'unknown';

/**
 * Numeric codes for finger states in compact notation.
 * Used in finger code strings (e.g., "01210" = thumb extended, index curled, etc.)
 */
export const FINGER_STATE_CODES = {
  extended: '0',
  curled: '1',
  flexed: '2',
  unknown: '?',
} as const;

/**
 * Map from numeric code to FingerLabel
 */
export const CODE_TO_FINGER_STATE: Record<string, FingerLabel> = {
  '0': 'extended',
  '1': 'curled',
  '2': 'flexed',
  '?': 'unknown',
};

/**
 * Labels for all five fingers.
 * Used in session labeling and data annotation.
 */
export interface FingerLabels {
  thumb: FingerLabel;
  index: FingerLabel;
  middle: FingerLabel;
  ring: FingerLabel;
  pinky: FingerLabel;
}

/** Default finger labels (all unknown) */
export const DEFAULT_FINGER_LABELS: FingerLabels = {
  thumb: 'unknown',
  index: 'unknown',
  middle: 'unknown',
  ring: 'unknown',
  pinky: 'unknown',
};

// ===== Finger Flexion (Continuous Values for Visualization) =====

/**
 * Continuous flexion values for visualization.
 * 0.0 = fully extended, 1.0 = fully flexed.
 */
export interface FingerFlexion {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
}

/** Default finger flexion (all extended) */
export const DEFAULT_FINGER_FLEXION: FingerFlexion = {
  thumb: 0,
  index: 0,
  middle: 0,
  ring: 0,
  pinky: 0,
};

// ===== Finger Position Tracking (6-DOF State) =====

/**
 * Finger position and velocity state for tracking.
 * Used in Kalman filter and particle filter for finger tracking.
 */
export interface FingerTrackingState {
  /** Position X (mm from sensor) */
  x: number;
  /** Position Y (mm from sensor) */
  y: number;
  /** Position Z (mm from sensor) */
  z: number;
  /** Velocity X (mm/s) */
  vx: number;
  /** Velocity Y (mm/s) */
  vy: number;
  /** Velocity Z (mm/s) */
  vz: number;
}

/**
 * Tracking state for all five fingers.
 * Used in multi-finger Kalman filter.
 */
export interface HandTrackingState {
  thumb: FingerTrackingState;
  index: FingerTrackingState;
  middle: FingerTrackingState;
  ring: FingerTrackingState;
  pinky: FingerTrackingState;
}

// ===== Hand Pose (3D Positions) =====

/**
 * Hand pose with 3D position for each finger.
 * Used in particle filter pose estimation.
 */
export interface HandPose {
  thumb: Vector3;
  index: Vector3;
  middle: Vector3;
  ring: Vector3;
  pinky: Vector3;
}

// ===== Magnet Configuration =====

/**
 * Magnetic moment configuration for a single finger magnet.
 */
export interface FingerMagnetConfig {
  /** Magnetic moment vector (units: A*m^2 or equivalent) */
  moment: Vector3;
  /** Optional: distance from fingertip (mm) */
  offset?: number;
}

/**
 * Magnet configuration for all fingers.
 * Optional per finger (not all fingers may have magnets).
 */
export interface HandMagnetConfig {
  thumb?: FingerMagnetConfig;
  index?: FingerMagnetConfig;
  middle?: FingerMagnetConfig;
  ring?: FingerMagnetConfig;
  pinky?: FingerMagnetConfig;
}

// ===== Magnet Detection State =====

/** Magnet detection confidence levels */
export type MagnetDetectionStatus = 'none' | 'possible' | 'likely' | 'confirmed';

/**
 * State of magnet detection from sensor data.
 */
export interface MagnetDetectionState {
  /** Detection status */
  status: MagnetDetectionStatus;
  /** Confidence score (0-1) */
  confidence: number;
  /** Whether a magnet is currently detected */
  detected: boolean;
  /** Whether baseline has been established */
  baselineEstablished: boolean;
  /** Baseline residual magnitude (uT) */
  baselineResidual: number;
  /** Current deviation from baseline (uT) */
  deviation: number;
}

// ===== Backward Compatibility =====

/**
 * @deprecated Use FingerLabel instead
 */
export type FingerState = FingerLabel;

/**
 * @deprecated Use FingerLabels instead
 */
export type FingerStates = FingerLabels;
