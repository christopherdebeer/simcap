/**
 * Session Data Types
 *
 * Types for recorded session structure, labels, and calibration data.
 *
 * @module core/types/session
 */

import type { TelemetrySample } from './telemetry';
import type { Vector3, Matrix3x3 } from './geometry';
import type { FingerLabels, FingerLabel } from './hand';
import type { DeviceInfo } from './device';

// Re-export DeviceInfo for convenience (FingerLabel/FingerLabels come from ./hand)

// ===== Motion Labels =====

/** Motion type for labeling (static vs dynamic) */
export type MotionLabel = 'static' | 'dynamic';

/** Calibration type for labeling */
export type CalibrationLabel = 'none' | 'mag' | 'gyro';

// ===== Label Segment =====

/**
 * Label segment within a session.
 * Defines annotations for a range of samples.
 */
export interface LabelSegment {
  /** Start sample index (inclusive) */
  startIndex: number;
  /** End sample index (inclusive) */
  endIndex: number;
  /** Pose name (e.g., "fist", "open", "pointing") */
  pose?: string;
  /** Finger state labels */
  fingers?: FingerLabels;
  /** Motion state during segment */
  motion?: MotionLabel;
  /** Calibration activity during segment */
  calibration?: CalibrationLabel;
  /** Custom labels (user-defined) */
  custom?: string[];
}

// ===== Calibration Data =====

/**
 * Magnetometer calibration data (hard/soft iron correction).
 */
export interface MagCalibration {
  /** Hard iron offset (uT) - subtracted from raw readings */
  hardIron: Vector3;
  /** Soft iron correction matrix - multiplied with corrected readings */
  softIron: Matrix3x3;
  /** When calibration was performed */
  timestamp: string;
}

/**
 * Gyroscope bias calibration data.
 */
export interface GyroBias {
  /** Bias X (LSB or deg/s) */
  x: number;
  /** Bias Y (LSB or deg/s) */
  y: number;
  /** Bias Z (LSB or deg/s) */
  z: number;
  /** Number of samples used for calibration */
  sampleCount: number;
}

/**
 * Combined calibration data for a session.
 */
export interface CalibrationData {
  /** Magnetometer calibration */
  mag?: MagCalibration;
  /** Gyroscope bias calibration */
  gyroBias?: GyroBias;
}

// ===== Geomagnetic Reference =====

/**
 * Geomagnetic field reference for a location.
 * Used for orientation estimation and Earth field subtraction.
 */
export interface GeomagneticLocation {
  /** City name */
  city: string;
  /** Country name */
  country: string;
  /** Latitude (degrees) */
  lat: number;
  /** Longitude (degrees) */
  lon: number;
  /** Magnetic declination (degrees, positive = east) */
  declination: number;
  /** Magnetic inclination (degrees, positive = down) */
  inclination: number;
  /** Total field intensity (uT) */
  intensity: number;
  /** Horizontal component (uT) */
  horizontal: number;
  /** Vertical component (uT, positive = down) */
  vertical: number;
}

/**
 * Subset of GeomagneticLocation for filter configuration.
 * Use toGeomagneticReference() to convert from full location.
 */
export interface GeomagneticReference {
  /** Horizontal component (uT) */
  horizontal: number;
  /** Vertical component (uT) */
  vertical: number;
  /** Magnetic declination (degrees) */
  declination: number;
}

/**
 * Convert full location to filter reference.
 */
export function toGeomagneticReference(loc: GeomagneticLocation): GeomagneticReference {
  return {
    horizontal: loc.horizontal,
    vertical: loc.vertical,
    declination: loc.declination,
  };
}

// ===== Session Metadata =====

/**
 * Session recording metadata.
 */
export interface SessionMetadata {
  /** When recording started (ISO string) */
  recordedAt: string;
  /** Recording duration (seconds) */
  duration: number;
  /** Total sample count */
  sampleCount: number;
  /** Sample rate (Hz) */
  sampleRate?: number;
  /** Device identifier */
  device?: string;
  /** Firmware version */
  firmwareVersion?: string;
  /** Subject identifier (anonymized) */
  subjectId?: string;
  /** Recording environment description */
  environment?: string;
  /** Which hand was used */
  hand?: 'left' | 'right';
  /** Session type/purpose */
  sessionType?: string;
  /** Magnet configuration used */
  magnetType?: string;
  /** Additional notes */
  notes?: string;
}

// ===== Complete Session =====

/**
 * Complete session data structure.
 * This is the full schema for stored session JSON files.
 */
export interface SessionData {
  /** Schema version */
  version: string;
  /** Device information */
  device: DeviceInfo;
  /** Calibration data */
  calibration: CalibrationData;
  /** Geomagnetic location (optional) */
  geomagneticLocation?: GeomagneticLocation;
  /** Telemetry samples */
  samples: TelemetrySample[];
  /** Label segments */
  labels: LabelSegment[];
  /** Session metadata */
  metadata?: SessionMetadata;
}

// ===== Backward Compatibility =====

// Note: FingerState and FingerStates are deprecated aliases exported from ./hand
// They are re-exported at the index level for backward compatibility

/**
 * @deprecated Use MotionLabel instead
 */
export type MotionType = MotionLabel;
