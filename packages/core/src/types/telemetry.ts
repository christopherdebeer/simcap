/**
 * Telemetry Types - Progressive Pipeline Stages
 *
 * Defines the structure of sensor data as it flows through the processing pipeline.
 * Each stage EXTENDS the previous with REQUIRED fields (not optional).
 *
 * Pipeline Stages:
 *   Stage 0: Raw Telemetry (LSB from device)
 *   Stage 1: Unit Conversion (g, deg/s, uT)
 *   Stage 2: Motion Detection (isMoving, std deviation)
 *   Stage 3: Gyroscope Bias Calibration
 *   Stage 4: Orientation Estimation (IMU sensor fusion)
 *   Stage 5: Magnetometer Calibration (hard/soft iron)
 *   Stage 6: Magnetic Residual (Earth field subtracted)
 *   Stage 7: Magnet Detection (finger magnet presence)
 *   Stage 8: Kalman Filtering (smoothed magnetic field)
 *
 * DESIGN PRINCIPLE: Each stage type has REQUIRED fields. Use type guards
 * to narrow from the union type returned by process().
 *
 * @module core/types/telemetry
 */

import type { Vector3, Quaternion, EulerAngles } from './geometry';

// Re-export geometry types for convenience
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

  // ===== Optional environmental sensors =====
  /** Light sensor (0-1 normalized, optional) */
  l?: number;
  /** Capacitive sensor (raw value, optional) */
  c?: number;
  /** Battery percentage (0-100, optional) */
  b?: number;
  /** Device state (0=idle, 1=streaming, optional) */
  s?: number;
  /** Button press count (optional) */
  n?: number;

  // ===== New v0.4.0 fields =====
  /** Sampling mode: L=LOW_POWER, N=NORMAL, H=HIGH_RES, B=BURST (optional) */
  mode?: string;
  /** Context: u=unknown, s=stored, h=held, a=active, t=table (optional) */
  ctx?: string;
  /** Grip detected: 0=no, 1=yes (optional) */
  grip?: number;
}

// ============================================================================
// STAGE 1: UNIT CONVERSION
// ============================================================================

/**
 * Telemetry with physical units applied.
 * Raw LSB values converted to g, deg/s, and µT.
 */
export interface UnitConvertedTelemetry extends RawTelemetry {
  /** Time delta since last sample (seconds) */
  dt: number;
  /** Acceleration X (g) */
  ax_g: number;
  /** Acceleration Y (g) */
  ay_g: number;
  /** Acceleration Z (g) */
  az_g: number;
  /** Angular velocity X (degrees/second) */
  gx_dps: number;
  /** Angular velocity Y (degrees/second) */
  gy_dps: number;
  /** Angular velocity Z (degrees/second) */
  gz_dps: number;
  /** Magnetic field X (µT, aligned to accel frame) */
  mx_ut: number;
  /** Magnetic field Y (µT, aligned to accel frame) */
  my_ut: number;
  /** Magnetic field Z (µT, aligned to accel frame) */
  mz_ut: number;
}

// ============================================================================
// STAGE 2: MOTION DETECTION
// ============================================================================

/**
 * Telemetry with motion detection applied.
 * Indicates whether device is stationary or moving.
 */
export interface MotionDetectedTelemetry extends UnitConvertedTelemetry {
  /** Whether device is currently moving */
  isMoving: boolean;
  /** Standard deviation of acceleration magnitude (LSB) */
  accelStd: number;
  /** Standard deviation of angular velocity magnitude (LSB) */
  gyroStd: number;
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
 * Telemetry with gyroscope bias calibration status.
 */
export interface GyroBiasCalibratedTelemetry extends MotionDetectedTelemetry {
  /** Whether gyroscope bias has been calibrated */
  gyroBiasCalibrated: boolean;
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
// STAGE 4: ORIENTATION ESTIMATION (IMU SENSOR FUSION)
// ============================================================================

/**
 * Telemetry with orientation from IMU sensor fusion (Madgwick AHRS).
 * Quaternion and Euler angles are REQUIRED at this stage.
 */
export interface OrientationEstimatedTelemetry extends GyroBiasCalibratedTelemetry {
  /** Quaternion W component */
  orientation_w: number;
  /** Quaternion X component */
  orientation_x: number;
  /** Quaternion Y component */
  orientation_y: number;
  /** Quaternion Z component */
  orientation_z: number;
  /** Euler roll angle (degrees) */
  euler_roll: number;
  /** Euler pitch angle (degrees) */
  euler_pitch: number;
  /** Euler yaw angle (degrees) */
  euler_yaw: number;
}

/**
 * Telemetry with 9-DOF magnetometer-assisted orientation.
 * Extends OrientationEstimatedTelemetry with AHRS magnetic residual.
 */
export interface OrientationWithMagTelemetry extends OrientationEstimatedTelemetry {
  /** AHRS magnetic residual X (µT) */
  ahrs_mag_residual_x: number;
  /** AHRS magnetic residual Y (µT) */
  ahrs_mag_residual_y: number;
  /** AHRS magnetic residual Z (µT) */
  ahrs_mag_residual_z: number;
  /** AHRS magnetic residual magnitude (µT) */
  ahrs_mag_residual_magnitude: number;
}

// ============================================================================
// STAGE 5: MAGNETOMETER CALIBRATION
// ============================================================================

/**
 * Telemetry with magnetometer calibration applied.
 * Includes hard iron correction and calibration state.
 */
export interface MagCalibratedTelemetry extends OrientationEstimatedTelemetry {
  /** Hard iron corrected magnetic field X (µT) */
  iron_mx: number;
  /** Hard iron corrected magnetic field Y (µT) */
  iron_my: number;
  /** Hard iron corrected magnetic field Z (µT) */
  iron_mz: number;
  /** Whether calibration is ready for use */
  mag_cal_ready: boolean;
  /** Calibration confidence (0-1) */
  mag_cal_confidence: number;
  /** Mean residual after calibration (µT) */
  mag_cal_mean_residual: number;
  /** Estimated Earth field magnitude (µT) */
  mag_cal_earth_magnitude: number;
  /** Whether hard iron calibration is complete */
  mag_cal_hard_iron: boolean;
  /** Whether soft iron calibration is complete */
  mag_cal_soft_iron: boolean;
}

// ============================================================================
// STAGE 6: MAGNETIC RESIDUAL
// ============================================================================

/**
 * Telemetry with magnetic residual calculated.
 * Residual = Measured field - Expected Earth field (in device frame).
 */
export interface MagResidualTelemetry extends MagCalibratedTelemetry {
  /** Magnetic residual X (µT) */
  residual_mx: number;
  /** Magnetic residual Y (µT) */
  residual_my: number;
  /** Magnetic residual Z (µT) */
  residual_mz: number;
  /** Magnetic residual magnitude (µT) */
  residual_magnitude: number;
}

// ============================================================================
// STAGE 7: MAGNET DETECTION
// ============================================================================

/** Magnet detection status levels */
export type MagnetStatus = 'none' | 'possible' | 'likely' | 'confirmed';

/**
 * Telemetry with magnet detection applied.
 * Detects presence of finger magnets based on magnetic residual.
 */
export interface MagnetDetectedTelemetry extends MagResidualTelemetry {
  /** Detection status (none/possible/likely/confirmed) */
  magnet_status: MagnetStatus;
  /** Detection confidence (0-1) */
  magnet_confidence: number;
  /** Whether magnet is detected (status != 'none') */
  magnet_detected: boolean;
  /** Whether baseline has been established */
  magnet_baseline_established: boolean;
  /** Baseline residual magnitude (µT) */
  magnet_baseline_residual: number;
  /** Current deviation from baseline (µT) */
  magnet_deviation: number;
}

// ============================================================================
// STAGE 8: KALMAN FILTERING
// ============================================================================

/**
 * Fully processed telemetry with Kalman filtering.
 * This is the final stage of the pipeline.
 */
export interface FilteredTelemetry extends MagnetDetectedTelemetry {
  /** Kalman-filtered magnetic field X (µT) */
  filtered_mx: number;
  /** Kalman-filtered magnetic field Y (µT) */
  filtered_my: number;
  /** Kalman-filtered magnetic field Z (µT) */
  filtered_mz: number;
}

// ============================================================================
// PIPELINE STAGES UNION & TYPE GUARDS
// ============================================================================

/**
 * All possible telemetry pipeline stages.
 * The processor returns this union - use type guards to narrow.
 */
export type TelemetryPipelineStage =
  | RawTelemetry
  | UnitConvertedTelemetry
  | MotionDetectedTelemetry
  | GyroBiasCalibratedTelemetry
  | OrientationEstimatedTelemetry
  | OrientationWithMagTelemetry
  | MagCalibratedTelemetry
  | MagResidualTelemetry
  | MagnetDetectedTelemetry
  | FilteredTelemetry;

/**
 * Type guard: Check if telemetry has unit conversion applied.
 */
export function hasUnitConversion(t: TelemetryPipelineStage): t is UnitConvertedTelemetry {
  return 'dt' in t && 'ax_g' in t && 'mx_ut' in t;
}

/**
 * Type guard: Check if telemetry has motion detection applied.
 */
export function hasMotionDetection(t: TelemetryPipelineStage): t is MotionDetectedTelemetry {
  return 'isMoving' in t && 'accelStd' in t && 'gyroStd' in t;
}

/**
 * Type guard: Check if telemetry has gyro bias calibration status.
 */
export function hasGyroBiasStatus(t: TelemetryPipelineStage): t is GyroBiasCalibratedTelemetry {
  return 'gyroBiasCalibrated' in t;
}

/**
 * Type guard: Check if telemetry has orientation estimation.
 */
export function hasOrientation(t: TelemetryPipelineStage): t is OrientationEstimatedTelemetry {
  return 'orientation_w' in t && 'euler_roll' in t;
}

/**
 * Type guard: Check if telemetry has 9-DOF magnetometer-assisted orientation.
 */
export function hasOrientationWithMag(t: TelemetryPipelineStage): t is OrientationWithMagTelemetry {
  return hasOrientation(t) && 'ahrs_mag_residual_magnitude' in t;
}

/**
 * Type guard: Check if telemetry has magnetometer calibration.
 */
export function hasMagCalibration(t: TelemetryPipelineStage): t is MagCalibratedTelemetry {
  return 'iron_mx' in t && 'mag_cal_ready' in t;
}

/**
 * Type guard: Check if telemetry has magnetic residual.
 */
export function hasMagResidual(t: TelemetryPipelineStage): t is MagResidualTelemetry {
  return 'residual_mx' in t && 'residual_magnitude' in t;
}

/**
 * Type guard: Check if telemetry has magnet detection.
 */
export function hasMagnetDetection(t: TelemetryPipelineStage): t is MagnetDetectedTelemetry {
  return 'magnet_status' in t && 'magnet_detected' in t;
}

/**
 * Type guard: Check if telemetry has Kalman filtering.
 */
export function hasKalmanFiltering(t: TelemetryPipelineStage): t is FilteredTelemetry {
  return 'filtered_mx' in t && 'filtered_my' in t && 'filtered_mz' in t;
}

/**
 * Type guard: Check if telemetry is fully processed through all stages.
 */
export function isFullyProcessed(t: TelemetryPipelineStage): t is FilteredTelemetry {
  return hasKalmanFiltering(t);
}

// ============================================================================
// DECORATED TELEMETRY (PROCESSOR OUTPUT)
// ============================================================================

/**
 * Telemetry with optional pipeline fields.
 *
 * This is the return type of TelemetryProcessor.process() because the processor
 * conditionally adds fields based on runtime state:
 * - Stages 0-3 (unit conversion, motion, gyro bias) are always applied
 * - Stage 4 (orientation) requires IMU initialization
 * - Stages 5-8 (mag calibration, residual, detection, filtering) require orientation
 *
 * For type-safe access to specific fields, use the stage types with type guards:
 *
 * @example
 * ```typescript
 * const result = processor.process(raw);
 * if (hasOrientation(result)) {
 *   // result is OrientationEstimatedTelemetry - euler_roll is guaranteed
 *   console.log(result.euler_roll);
 * }
 * if (hasMagResidual(result)) {
 *   // result is MagResidualTelemetry - residual_magnitude is guaranteed
 *   const residual = extractMagResidual(result);
 * }
 * ```
 */
export interface DecoratedTelemetry extends RawTelemetry {
  // Stage 1: Unit conversion
  dt?: number;
  ax_g?: number;
  ay_g?: number;
  az_g?: number;
  gx_dps?: number;
  gy_dps?: number;
  gz_dps?: number;
  mx_ut?: number;
  my_ut?: number;
  mz_ut?: number;

  // Stage 2: Motion detection
  isMoving?: boolean;
  accelStd?: number;
  gyroStd?: number;

  // Stage 3: Gyro bias
  gyroBiasCalibrated?: boolean;

  // Stage 4: Orientation
  orientation_w?: number;
  orientation_x?: number;
  orientation_y?: number;
  orientation_z?: number;
  euler_roll?: number;
  euler_pitch?: number;
  euler_yaw?: number;
  ahrs_mag_residual_x?: number;
  ahrs_mag_residual_y?: number;
  ahrs_mag_residual_z?: number;
  ahrs_mag_residual_magnitude?: number;

  // Stage 5: Mag calibration
  iron_mx?: number;
  iron_my?: number;
  iron_mz?: number;
  mag_cal_ready?: boolean;
  mag_cal_confidence?: number;
  mag_cal_mean_residual?: number;
  mag_cal_earth_magnitude?: number;
  mag_cal_hard_iron?: boolean;
  mag_cal_soft_iron?: boolean;

  // Stage 6: Mag residual
  residual_mx?: number;
  residual_my?: number;
  residual_mz?: number;
  residual_magnitude?: number;

  // Stage 7: Magnet detection
  magnet_status?: MagnetStatus;
  magnet_confidence?: number;
  magnet_detected?: boolean;
  magnet_baseline_established?: boolean;
  magnet_baseline_residual?: number;
  magnet_deviation?: number;

  // Stage 8: Kalman filtering
  filtered_mx?: number;
  filtered_my?: number;
  filtered_mz?: number;

  // Dynamic access for backward compatibility
  [key: string]: number | string | boolean | undefined;
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

/**
 * Alternative structure using Vector3 for physical values.
 */
export interface PhysicalTelemetry {
  /** Acceleration vector (g) */
  accel: Vector3;
  /** Angular velocity vector (deg/s) */
  gyro: Vector3;
  /** Magnetic field vector (µT) */
  mag: Vector3;
  /** Timestamp (ms) */
  timestamp: number;
}

/**
 * Session storage format for JSON files.
 * Extends RawTelemetry with optional processed fields that are
 * persisted in session recordings.
 */
export interface TelemetrySample extends RawTelemetry {
  /** Device orientation at sample time */
  orientation?: Quaternion;
  /** Finger magnet detection result */
  fingerMagnet?: {
    detected: boolean;
    confidence: number;
  };
}

// ============================================================================
// EXTRACTION UTILITIES
// ============================================================================

/**
 * Extract orientation as Quaternion from telemetry with orientation.
 */
export function extractQuaternion(t: OrientationEstimatedTelemetry): Quaternion {
  return {
    w: t.orientation_w,
    x: t.orientation_x,
    y: t.orientation_y,
    z: t.orientation_z,
  };
}

/**
 * Extract orientation as EulerAngles from telemetry with orientation.
 */
export function extractEulerAngles(t: OrientationEstimatedTelemetry): EulerAngles {
  return {
    roll: t.euler_roll,
    pitch: t.euler_pitch,
    yaw: t.euler_yaw,
  };
}

/**
 * Extract magnetic residual as Vector3 from telemetry with residual.
 */
export function extractMagResidual(t: MagResidualTelemetry): Vector3 {
  return {
    x: t.residual_mx,
    y: t.residual_my,
    z: t.residual_mz,
  };
}

/**
 * Extract filtered magnetic field as Vector3 from fully processed telemetry.
 */
export function extractFilteredMag(t: FilteredTelemetry): Vector3 {
  return {
    x: t.filtered_mx,
    y: t.filtered_my,
    z: t.filtered_mz,
  };
}

/**
 * Safe extraction of quaternion from any telemetry (returns null if not available).
 */
export function tryExtractQuaternion(t: TelemetryPipelineStage): Quaternion | null {
  if (hasOrientation(t)) {
    return extractQuaternion(t);
  }
  return null;
}

/**
 * Safe extraction of euler angles from any telemetry (returns null if not available).
 */
export function tryExtractEulerAngles(t: TelemetryPipelineStage): EulerAngles | null {
  if (hasOrientation(t)) {
    return extractEulerAngles(t);
  }
  return null;
}

/**
 * Safe extraction of magnetic residual from any telemetry (returns null if not available).
 */
export function tryExtractMagResidual(t: TelemetryPipelineStage): Vector3 | null {
  if (hasMagResidual(t)) {
    return extractMagResidual(t);
  }
  return null;
}

/**
 * Safe extraction of filtered mag from any telemetry (returns null if not available).
 */
export function tryExtractFilteredMag(t: TelemetryPipelineStage): Vector3 | null {
  if (hasKalmanFiltering(t)) {
    return extractFilteredMag(t);
  }
  return null;
}
