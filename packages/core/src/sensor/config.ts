/**
 * Sensor Configuration Constants
 *
 * Hardware-specific constants for GAMBIT sensor processing.
 * Based on LSM6DS3 IMU and MMC5603NJ magnetometer in Puck.js v2.1a.
 */

/**
 * Accelerometer scale factor: LSB per g
 * LSM6DS3 at ±2g range, 16-bit resolution
 */
export const ACCEL_SCALE = 8192;

/**
 * Gyroscope scale factor: LSB per deg/s
 * LSM6DS3 at 245dps range
 */
export const GYRO_SCALE = 114.28;

/**
 * Default sample frequency for GAMBIT firmware (Hz)
 * Matches firmware accelOn rate
 */
export const DEFAULT_SAMPLE_FREQ = 26;

/**
 * Magnetometer scale factors by hardware version
 */
export const MAG_SCALE = {
  /** MMC5603NJ (Puck.js v2.1a): μT per LSB */
  MMC5603NJ: 100 / 1024,  // 0.09765625 μT/LSB
  /** LIS3MDL (Puck.js v2): μT per LSB */
  LIS3MDL: 100 / 6842,    // 0.014616 μT/LSB
} as const;

/** Default magnetometer scale (current hardware) */
export const MAG_SCALE_DEFAULT = MAG_SCALE.MMC5603NJ;

// ===== Unit Conversion Functions =====

/**
 * Convert accelerometer reading from LSB to g's
 */
export function accelLsbToG(lsb: number): number {
  return lsb / ACCEL_SCALE;
}

/**
 * Convert gyroscope reading from LSB to deg/s
 */
export function gyroLsbToDps(lsb: number): number {
  return lsb / GYRO_SCALE;
}

/**
 * Convert gyroscope reading from LSB to rad/s
 */
export function gyroLsbToRads(lsb: number): number {
  return (lsb / GYRO_SCALE) * (Math.PI / 180);
}

/**
 * Convert magnetometer reading from LSB to μT
 */
export function magLsbToMicroTesla(lsb: number, scale = MAG_SCALE_DEFAULT): number {
  return lsb * scale;
}

/**
 * Convert accelerometer vector from LSB to g's
 */
export function convertAccelToG(raw: { ax: number; ay: number; az: number }) {
  return {
    x: accelLsbToG(raw.ax),
    y: accelLsbToG(raw.ay),
    z: accelLsbToG(raw.az),
  };
}

/**
 * Convert gyroscope vector from LSB to deg/s
 */
export function convertGyroToDps(raw: { gx: number; gy: number; gz: number }) {
  return {
    x: gyroLsbToDps(raw.gx),
    y: gyroLsbToDps(raw.gy),
    z: gyroLsbToDps(raw.gz),
  };
}

/**
 * Convert magnetometer vector from LSB to μT
 */
export function convertMagToMicroTesla(
  raw: { mx: number; my: number; mz: number },
  scale = MAG_SCALE_DEFAULT
) {
  return {
    x: magLsbToMicroTesla(raw.mx, scale),
    y: magLsbToMicroTesla(raw.my, scale),
    z: magLsbToMicroTesla(raw.mz, scale),
  };
}
