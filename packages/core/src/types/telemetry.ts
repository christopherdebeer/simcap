/**
 * Telemetry Types - Full Processing Pipeline
 *
 * Defines the structure of sensor data as it flows through the 8-stage processing pipeline:
 *
 *   Stage 0: Raw Telemetry (LSB from device)
 *   Stage 1: Unit Conversion (g, deg/s, uT)
 *   Stage 2: Motion Detection (isMoving, std deviation)
 *   Stage 3: Gyroscope Bias Calibration
 *   Stage 4: IMU Sensor Fusion (orientation via Madgwick AHRS)
 *   Stage 5: Magnetometer Calibration (hard/soft iron, Earth field)
 *   Stage 6: Magnetic Residual (Earth field subtracted)
 *   Stage 7: Magnet Detection (finger magnet presence)
 *   Stage 8: Kalman Filtering (smoothed magnetic field)
 *
 * IMPORTANT: Raw values are ALWAYS preserved. Converted/computed fields are ADDED.
 *
 * @module core/types/telemetry
 */

import type { Vector3, Quaternion, EulerAngles } from './geometry';

// Re-export geometry types for backward compatibility
export type { Vector3, Quaternion, EulerAngles } from './geometry';

// ============================================================================
// STAGE 0: RAW TELEMETRY (FROM DEVICE)
// ============================================================================

/**
 * Raw sensor reading in LSB (Least Significant Bits) from device.
 * This is the unprocessed data directly from the Puck.js IMU sensors.
 *
 * Sensor specifications:
 * - Accelerometer: LSM6DS3, ±2g range, 8192 LSB/g
 * - Gyroscope: LSM6DS3, ±245 dps range, 114.28 LSB/dps
 * - Magnetometer: MMC5603NJ, ±30 gauss range, 1024 LSB/gauss
 */
export interface RawTelemetry {
  /** Accelerometer X (LSB, range: ±16384) */
  ax: number;
  /** Accelerometer Y (LSB, range: ±16384) */
  ay: number;
  /** Accelerometer Z (LSB, range: ±16384) */
  az: number;
  /** Gyroscope X (LSB, range: ±28000) */
  gx: number;
  /** Gyroscope Y (LSB, range: ±28000) */
  gy: number;
  /** Gyroscope Z (LSB, range: ±28000) */
  gz: number;
  /** Magnetometer X (LSB, range: ±30720) */
  mx: number;
  /** Magnetometer Y (LSB, range: ±30720) */
  my: number;
  /** Magnetometer Z (LSB, range: ±30720) */
  mz: number;
  /** Timestamp (ms since connection) */
  t: number;
}

// ============================================================================
// STAGE 1: UNIT CONVERSION
// ============================================================================

/**
 * Fields added after unit conversion.
 * Raw LSB values converted to physical units.
 */
export interface UnitConvertedFields {
  /** Time delta since last sample (seconds) */
  dt?: number;
  /** Acceleration X (g) */
  ax_g?: number;
  /** Acceleration Y (g) */
  ay_g?: number;
  /** Acceleration Z (g) */
  az_g?: number;
  /** Angular velocity X (degrees/second) */
  gx_dps?: number;
  /** Angular velocity Y (degrees/second) */
  gy_dps?: number;
  /** Angular velocity Z (degrees/second) */
  gz_dps?: number;
  /** Magnetic field X (uT, aligned to accel frame) */
  mx_ut?: number;
  /** Magnetic field Y (uT, aligned to accel frame) */
  my_ut?: number;
  /** Magnetic field Z (uT, aligned to accel frame) */
  mz_ut?: number;
}

/**
 * Telemetry converted to physical units.
 * Alternative structure using Vector3 instead of individual fields.
 */
export interface PhysicalTelemetry {
  /** Acceleration vector (g) */
  accel: Vector3;
  /** Angular velocity vector (deg/s) */
  gyro: Vector3;
  /** Magnetic field vector (uT) */
  mag: Vector3;
  /** Timestamp (ms) */
  timestamp: number;
}

// ============================================================================
// STAGE 2: MOTION DETECTION
// ============================================================================

/**
 * Fields added after motion detection.
 * Detects whether device is stationary or moving.
 */
export interface MotionDetectionFields {
  /** Whether device is currently moving */
  isMoving?: boolean;
  /** Standard deviation of acceleration magnitude (LSB) */
  accelStd?: number;
  /** Standard deviation of angular velocity magnitude (LSB) */
  gyroStd?: number;
}

/**
 * Motion detector state (returned by MotionDetector class).
 */
export interface MotionDetectorState {
  isMoving: boolean;
  accelStd: number;
  gyroStd: number;
}

// ============================================================================
// STAGE 3: GYROSCOPE BIAS CALIBRATION
// ============================================================================

/**
 * Fields added after gyroscope bias calibration.
 */
export interface GyroBiasFields {
  /** Whether gyroscope bias has been calibrated */
  gyroBiasCalibrated?: boolean;
}

/**
 * Gyroscope bias calibration state.
 */
export interface GyroBiasCalibrationState {
  /** Whether calibration is complete */
  calibrated: boolean;
  /** Number of stationary samples collected */
  stationaryCount: number;
  /** Bias values (rad/s or deg/s depending on context) */
  bias?: Vector3;
}

// ============================================================================
// STAGE 4: IMU SENSOR FUSION (AHRS)
// ============================================================================

/**
 * Fields added after IMU sensor fusion (Madgwick AHRS).
 * Provides orientation estimation from accelerometer/gyroscope/magnetometer.
 */
export interface OrientationFields {
  /** Quaternion W component */
  orientation_w?: number;
  /** Quaternion X component */
  orientation_x?: number;
  /** Quaternion Y component */
  orientation_y?: number;
  /** Quaternion Z component */
  orientation_z?: number;
  /** Euler roll angle (degrees) */
  euler_roll?: number;
  /** Euler pitch angle (degrees) */
  euler_pitch?: number;
  /** Euler yaw angle (degrees) */
  euler_yaw?: number;
  /** AHRS magnetic residual X (uT) */
  ahrs_mag_residual_x?: number;
  /** AHRS magnetic residual Y (uT) */
  ahrs_mag_residual_y?: number;
  /** AHRS magnetic residual Z (uT) */
  ahrs_mag_residual_z?: number;
  /** AHRS magnetic residual magnitude (uT) */
  ahrs_mag_residual_magnitude?: number;
}

/**
 * Processed telemetry with orientation (simplified).
 * @deprecated Use DecoratedTelemetry for full pipeline data
 */
export interface ProcessedTelemetry extends RawTelemetry {
  orientation?: Quaternion;
  euler?: EulerAngles;
}

// ============================================================================
// STAGE 5: MAGNETOMETER CALIBRATION
// ============================================================================

/**
 * Fields added after magnetometer calibration.
 * Includes hard iron correction, soft iron correction, and Earth field estimation.
 */
export interface MagCalibrationFields {
  /** Hard iron corrected magnetic field X (uT) */
  iron_mx?: number;
  /** Hard iron corrected magnetic field Y (uT) */
  iron_my?: number;
  /** Hard iron corrected magnetic field Z (uT) */
  iron_mz?: number;
  /** Whether calibration is ready for use */
  mag_cal_ready?: boolean;
  /** Calibration confidence (0-1) */
  mag_cal_confidence?: number;
  /** Mean residual after calibration (uT) */
  mag_cal_mean_residual?: number;
  /** Estimated Earth field magnitude (uT) */
  mag_cal_earth_magnitude?: number;
  /** Whether hard iron calibration is complete */
  mag_cal_hard_iron?: boolean;
  /** Whether soft iron calibration is complete */
  mag_cal_soft_iron?: boolean;
}

// ============================================================================
// STAGE 6: MAGNETIC RESIDUAL
// ============================================================================

/**
 * Fields added after magnetic residual calculation.
 * Residual = Measured field - Expected Earth field (in device frame).
 * Represents magnetic anomalies (primarily finger magnets).
 */
export interface MagResidualFields {
  /** Magnetic residual X (uT) */
  residual_mx?: number;
  /** Magnetic residual Y (uT) */
  residual_my?: number;
  /** Magnetic residual Z (uT) */
  residual_mz?: number;
  /** Magnetic residual magnitude (uT) */
  residual_magnitude?: number;
}

// ============================================================================
// STAGE 7: MAGNET DETECTION
// ============================================================================

/** Magnet detection status levels */
export type MagnetStatus = 'none' | 'possible' | 'likely' | 'confirmed';

/**
 * Fields added after magnet detection.
 * Detects presence of finger magnets based on magnetic residual.
 */
export interface MagnetDetectionFields {
  /** Detection status (none/possible/likely/confirmed) */
  magnet_status?: MagnetStatus;
  /** Detection confidence (0-1) */
  magnet_confidence?: number;
  /** Whether magnet is detected (status != 'none') */
  magnet_detected?: boolean;
  /** Whether baseline has been established */
  magnet_baseline_established?: boolean;
  /** Baseline residual magnitude (uT) */
  magnet_baseline_residual?: number;
  /** Current deviation from baseline (uT) */
  magnet_deviation?: number;
}

// ============================================================================
// STAGE 8: KALMAN FILTERING
// ============================================================================

/**
 * Fields added after Kalman filtering.
 * Smoothed/denoised magnetic field for tracking.
 */
export interface KalmanFilteredFields {
  /** Kalman-filtered magnetic field X (uT) */
  filtered_mx?: number;
  /** Kalman-filtered magnetic field Y (uT) */
  filtered_my?: number;
  /** Kalman-filtered magnetic field Z (uT) */
  filtered_mz?: number;
}

// ============================================================================
// COMPLETE DECORATED TELEMETRY
// ============================================================================

/**
 * Fully decorated telemetry with all pipeline stages.
 * Extends RawTelemetry with all computed/converted fields.
 * Total: 10 raw fields + 37 decorated fields = 47 fields
 */
export interface DecoratedTelemetry extends
  RawTelemetry,
  UnitConvertedFields,
  MotionDetectionFields,
  GyroBiasFields,
  OrientationFields,
  MagCalibrationFields,
  MagResidualFields,
  MagnetDetectionFields,
  KalmanFilteredFields {
  // All fields inherited from component interfaces
  // Allows indexing for dynamic access
  [key: string]: number | string | boolean | undefined;
}

// ============================================================================
// SESSION STORAGE TYPES
// ============================================================================

/**
 * Telemetry sample stored in session.
 * Simplified structure for session JSON files.
 * @deprecated Prefer DecoratedTelemetry for full pipeline data
 */
export interface TelemetrySample extends RawTelemetry {
  orientation?: Quaternion;
  fingerMagnet?: {
    detected: boolean;
    confidence: number;
  };
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

/**
 * Extract orientation as Quaternion from decorated telemetry.
 */
export function extractQuaternion(t: DecoratedTelemetry): Quaternion | null {
  if (t.orientation_w === undefined) return null;
  return {
    w: t.orientation_w,
    x: t.orientation_x ?? 0,
    y: t.orientation_y ?? 0,
    z: t.orientation_z ?? 0,
  };
}

/**
 * Extract orientation as EulerAngles from decorated telemetry.
 */
export function extractEulerAngles(t: DecoratedTelemetry): EulerAngles | null {
  if (t.euler_roll === undefined) return null;
  return {
    roll: t.euler_roll,
    pitch: t.euler_pitch ?? 0,
    yaw: t.euler_yaw ?? 0,
  };
}

/**
 * Extract magnetic residual as Vector3 from decorated telemetry.
 */
export function extractMagResidual(t: DecoratedTelemetry): Vector3 | null {
  if (t.residual_mx === undefined) return null;
  return {
    x: t.residual_mx,
    y: t.residual_my ?? 0,
    z: t.residual_mz ?? 0,
  };
}

/**
 * Extract filtered magnetic field as Vector3 from decorated telemetry.
 */
export function extractFilteredMag(t: DecoratedTelemetry): Vector3 | null {
  if (t.filtered_mx === undefined) return null;
  return {
    x: t.filtered_mx,
    y: t.filtered_my ?? 0,
    z: t.filtered_mz ?? 0,
  };
}
