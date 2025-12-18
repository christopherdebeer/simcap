/**
 * Sensor Configuration and Factory Functions
 *
 * Provides standardized sensor parameters and factory functions for
 * creating consistently configured signal processing components.
 *
 * @module shared/sensor-config
 */

import type { Vector3, Quaternion } from '@core/types';

// Import filter classes from @filters package
import { MadgwickAHRS, MotionDetector, KalmanFilter, KalmanFilter3D } from '@filters';

// ===== Sensor Scale Constants =====

/** Accelerometer scale factor: LSB per g (LSM6DS3 at ±2g range) */
export const ACCEL_SCALE = 8192;

/** Gyroscope scale factor: LSB per deg/s (LSM6DS3 at 245dps range) */
export const GYRO_SCALE = 114.28;

/** Default sample frequency for GAMBIT firmware (Hz) */
export const DEFAULT_SAMPLE_FREQ = 26;

/**
 * Magnetometer scale factor: LSB to μT
 * @deprecated Use magLsbToMicroTesla() from sensor-units for correct conversion
 */
export const MAG_SCALE_LSB_TO_UT = 100 / 6842;

// ===== Type Definitions =====

export interface AccelRaw {
  ax: number;
  ay: number;
  az: number;
}

export interface GyroRaw {
  gx: number;
  gy: number;
  gz: number;
}

export interface MadgwickOptions {
  sampleFreq?: number;
  beta?: number;
}

export interface KalmanFilter3DOptions {
  /** Process noise (R) */
  R?: number;
  /** Measurement noise (Q) */
  Q?: number;
}

export interface MotionDetectorOptions {
  accelThreshold?: number;
  gyroThreshold?: number;
  windowSize?: number;
}

export interface KalmanFilter1DOptions {
  R?: number;
  Q?: number;
}

export interface GyroBiasState {
  calibrated: boolean;
  stationaryCount: number;
  bias: Vector3;
}

export interface CubeFilterOptions {
  accAlpha?: number;
  gyroAlpha?: number;
  magAlpha?: number;
}

// ===== Unit Conversion Functions =====

/** Convert accelerometer reading from LSB to g's */
export function accelLsbToG(lsb: number): number {
  return lsb / ACCEL_SCALE;
}

/** Convert gyroscope reading from LSB to deg/s */
export function gyroLsbToDps(lsb: number): number {
  return lsb / GYRO_SCALE;
}

/** Convert accelerometer vector from LSB to g's */
export function convertAccelToG(raw: AccelRaw): AccelRaw {
  return {
    ax: accelLsbToG(raw.ax || 0),
    ay: accelLsbToG(raw.ay || 0),
    az: accelLsbToG(raw.az || 0)
  };
}

/** Convert gyroscope vector from LSB to deg/s */
export function convertGyroToDps(raw: GyroRaw): GyroRaw {
  return {
    gx: gyroLsbToDps(raw.gx || 0),
    gy: gyroLsbToDps(raw.gy || 0),
    gz: gyroLsbToDps(raw.gz || 0)
  };
}

// ===== Factory Functions =====

/**
 * Create a MadgwickAHRS instance with standard GAMBIT configuration
 */
export function createMadgwickAHRS(options: MadgwickOptions = {}): MadgwickAHRS {
  const config = {
    sampleFreq: options.sampleFreq || DEFAULT_SAMPLE_FREQ,
    beta: options.beta || 0.05
  };

  return new MadgwickAHRS(config);
}

/**
 * Create a KalmanFilter3D instance with standard GAMBIT configuration
 */
export function createKalmanFilter3D(options: KalmanFilter3DOptions = {}): KalmanFilter3D {
  const config = {
    R: options.R ?? 0.1,
    Q: options.Q ?? 1.0
  };

  return new KalmanFilter3D(config);
}

/**
 * Create a MotionDetector instance with standard GAMBIT configuration
 */
export function createMotionDetector(options: MotionDetectorOptions = {}): MotionDetector {
  const config = {
    accelThreshold: options.accelThreshold || 200,
    gyroThreshold: options.gyroThreshold || 300,
    windowSize: options.windowSize || 10
  };

  return new MotionDetector(config);
}

/**
 * Create a 1D KalmanFilter instance for single-axis filtering
 */
export function createKalmanFilter1D(options: KalmanFilter1DOptions = {}): KalmanFilter {
  const config = {
    R: options.R || 0.01,
    Q: options.Q || 3
  };

  return new KalmanFilter(config);
}

// ===== LowPassFilter Class =====

/**
 * Simple low-pass filter for smoothing sensor data
 * Formula: output = alpha * newValue + (1 - alpha) * previousValue
 */
export class LowPassFilter {
  private alpha: number;
  private value: number | null;

  constructor(alpha = 0.3) {
    this.alpha = alpha;
    this.value = null;
  }

  filter(newValue: number): number {
    if (this.value === null) {
      this.value = newValue;
    } else {
      this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
    }
    return this.value;
  }

  reset(): void {
    this.value = null;
  }

  setValue(value: number): void {
    this.value = value;
  }

  getValue(): number | null {
    return this.value;
  }
}

/** Create a LowPassFilter instance */
export function createLowPassFilter(alpha = 0.3): LowPassFilter {
  return new LowPassFilter(alpha);
}

// ===== Calibration Configuration =====

/** Number of stationary samples required before gyro bias calibration */
export const STATIONARY_SAMPLES_FOR_CALIBRATION = 20;

/** Create gyroscope bias calibration state object */
export function createGyroBiasState(): GyroBiasState {
  return {
    calibrated: false,
    stationaryCount: 0,
    bias: { x: 0, y: 0, z: 0 }
  };
}

// ===== Cube Visualization Filter Configuration =====

export interface AxisFilters {
  x: LowPassFilter;
  y: LowPassFilter;
  z: LowPassFilter;
}

export interface CubeFilters {
  acc: AxisFilters;
  gyro: AxisFilters;
  mag: AxisFilters;
}

/** Create filter set for 3D cube visualization */
export function createCubeFilters(options: CubeFilterOptions = {}): CubeFilters {
  const accAlpha = options.accAlpha || 0.4;
  const gyroAlpha = options.gyroAlpha || 0.3;
  const magAlpha = options.magAlpha || 0.3;

  return {
    acc: {
      x: createLowPassFilter(accAlpha),
      y: createLowPassFilter(accAlpha),
      z: createLowPassFilter(accAlpha)
    },
    gyro: {
      x: createLowPassFilter(gyroAlpha),
      y: createLowPassFilter(gyroAlpha),
      z: createLowPassFilter(gyroAlpha)
    },
    mag: {
      x: createLowPassFilter(magAlpha),
      y: createLowPassFilter(magAlpha),
      z: createLowPassFilter(magAlpha)
    }
  };
}

// ===== Default Export =====

export default {
  // Constants
  ACCEL_SCALE,
  GYRO_SCALE,
  DEFAULT_SAMPLE_FREQ,
  MAG_SCALE_LSB_TO_UT,
  STATIONARY_SAMPLES_FOR_CALIBRATION,

  // Conversion functions
  accelLsbToG,
  gyroLsbToDps,
  convertAccelToG,
  convertGyroToDps,

  // Factory functions
  createMadgwickAHRS,
  createKalmanFilter3D,
  createMotionDetector,
  createKalmanFilter1D,
  createLowPassFilter,
  createGyroBiasState,
  createCubeFilters
};
