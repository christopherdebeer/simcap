/**
 * Sensor Units and Conversion Configuration
 *
 * CRITICAL: This module defines the authoritative unit conversions for all sensors.
 * All conversions must preserve raw data and only ADD decorated fields.
 *
 * UNITS POLICY:
 * - RAW values are ALWAYS preserved in their native sensor units (LSB)
 * - CONVERTED values are added as decorated fields with explicit unit suffixes
 * - Session metadata MUST track which conversions were applied
 *
 * @module shared/sensor-units
 */

// ===== Type Definitions =====

export interface SensorSpec {
  sensor: string;
  rawUnit: string;
  convertedUnit: string;
  range: string;
  resolution: number;
  sensitivity: number;
  conversionFactor: number;
  reference: string;
}

export interface MagSensorSpec extends SensorSpec {
  puckVersion: string;
  gaussToMicroTesla: number;
  notes?: string[];
}

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

export interface MagRaw {
  mx: number;
  my: number;
  mz: number;
}

export interface RawSensorData extends AccelRaw, GyroRaw, MagRaw {}

export interface SensorUnitMetadata {
  version: string;
  date: string;
  sensors: {
    accelerometer: {
      sensor: string;
      rawUnit: string;
      convertedUnit: string;
      conversionFactor: number;
      range: string;
    };
    gyroscope: {
      sensor: string;
      rawUnit: string;
      convertedUnit: string;
      conversionFactor: number;
      range: string;
    };
    magnetometer: {
      sensor: string;
      rawUnit: string;
      convertedUnit: string;
      conversionFactor: number;
      range: string;
    };
  };
  fieldNaming: {
    raw: string;
    converted: string;
    note: string;
  };
}

export interface SensorValidationResult {
  valid: boolean;
  warnings: string[];
  accelMaxLsb: number;
  gyroMaxLsb: number;
  magMaxLsb: number;
  magMagnitudeUt: number;
}

// ===== Sensor Hardware Specifications =====

/**
 * Accelerometer: LSM6DS3 (Puck.js)
 * Range: ±2g (default), Resolution: 16-bit, Sensitivity: 8192 LSB/g
 */
export const ACCEL_SPEC: SensorSpec = {
  sensor: 'LSM6DS3',
  rawUnit: 'LSB',
  convertedUnit: 'g',
  range: '±2g',
  resolution: 16,
  sensitivity: 8192,
  conversionFactor: 1 / 8192,
  reference: 'https://www.st.com/resource/en/datasheet/lsm6ds3.pdf'
};

/**
 * Gyroscope: LSM6DS3 (Puck.js)
 * Range: ±245 dps (default), Resolution: 16-bit, Sensitivity: 114.28 LSB/dps
 */
export const GYRO_SPEC: SensorSpec = {
  sensor: 'LSM6DS3',
  rawUnit: 'LSB',
  convertedUnit: 'deg/s',
  range: '±245dps',
  resolution: 16,
  sensitivity: 114.28,
  conversionFactor: 1 / 114.28,
  reference: 'https://www.st.com/resource/en/datasheet/lsm6ds3.pdf'
};

/**
 * Magnetometer: LIS3MDL (Puck.js v2) - LEGACY
 * Range: ±4 gauss, Resolution: 16-bit, Sensitivity: 6842 LSB/gauss
 */
export const MAG_SPEC_LIS3MDL: MagSensorSpec = {
  sensor: 'LIS3MDL',
  puckVersion: 'v2',
  rawUnit: 'LSB',
  convertedUnit: 'µT',
  range: '±4gauss',
  resolution: 16,
  sensitivity: 6842,
  gaussToMicroTesla: 100,
  conversionFactor: 100 / 6842,
  reference: 'https://www.st.com/resource/en/datasheet/lis3mdl.pdf'
};

/**
 * Magnetometer: MMC5603NJ (Puck.js v2.1a) - CURRENT
 * Range: ±30 gauss, Resolution: 16-bit, Sensitivity: 1024 LSB/gauss
 */
export const MAG_SPEC_MMC5603NJ: MagSensorSpec = {
  sensor: 'MMC5603NJ',
  puckVersion: 'v2.1a',
  rawUnit: 'LSB',
  convertedUnit: 'µT',
  range: '±30gauss',
  resolution: 16,
  sensitivity: 1024,
  gaussToMicroTesla: 100,
  conversionFactor: 100 / 1024,
  reference: 'src/device/GAMBIT/MMC5603NJ.pdf',
  notes: [
    'Puck.mag() returns RAW LSB values',
    'Must multiply by conversionFactor to get µT',
    "Earth's magnetic field: ~25-65 µT total",
    'Edinburgh, UK: ~50.5 µT total',
    '16-bit mode: 1024 counts/Gauss (from datasheet page 2)',
    'Expected ~461 LSB for 45 µT Earth field'
  ]
};

/** Active magnetometer specification (default: MMC5603NJ for Puck.js v2.1a) */
export const MAG_SPEC: MagSensorSpec = MAG_SPEC_MMC5603NJ;

// ===== Backward Compatibility Exports =====

export const ACCEL_SCALE = ACCEL_SPEC.sensitivity;
export const GYRO_SCALE = GYRO_SPEC.sensitivity;
export const MAG_SCALE_LSB_TO_UT = MAG_SPEC.conversionFactor;

// ===== Unit Conversion Functions =====

/** Convert accelerometer from LSB to g */
export function accelLsbToG(lsb: number): number {
  return lsb * ACCEL_SPEC.conversionFactor;
}

/** Convert gyroscope from LSB to deg/s */
export function gyroLsbToDps(lsb: number): number {
  return lsb * GYRO_SPEC.conversionFactor;
}

/** Convert magnetometer from LSB to µT */
export function magLsbToMicroTesla(lsb: number): number {
  return lsb * MAG_SPEC.conversionFactor;
}

/** Convert accelerometer vector from LSB to g */
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

/** Convert magnetometer vector from LSB to µT */
export function convertMagToMicroTesla(raw: MagRaw): MagRaw {
  return {
    mx: magLsbToMicroTesla(raw.mx || 0),
    my: magLsbToMicroTesla(raw.my || 0),
    mz: magLsbToMicroTesla(raw.mz || 0)
  };
}

// ===== Unit Metadata for Sessions =====

/** Get sensor unit configuration for session metadata */
export function getSensorUnitMetadata(): SensorUnitMetadata {
  return {
    version: '1.0.0',
    date: new Date().toISOString(),
    sensors: {
      accelerometer: {
        sensor: ACCEL_SPEC.sensor,
        rawUnit: ACCEL_SPEC.rawUnit,
        convertedUnit: ACCEL_SPEC.convertedUnit,
        conversionFactor: ACCEL_SPEC.conversionFactor,
        range: ACCEL_SPEC.range
      },
      gyroscope: {
        sensor: GYRO_SPEC.sensor,
        rawUnit: GYRO_SPEC.rawUnit,
        convertedUnit: GYRO_SPEC.convertedUnit,
        conversionFactor: GYRO_SPEC.conversionFactor,
        range: GYRO_SPEC.range
      },
      magnetometer: {
        sensor: MAG_SPEC.sensor,
        rawUnit: MAG_SPEC.rawUnit,
        convertedUnit: MAG_SPEC.convertedUnit,
        conversionFactor: MAG_SPEC.conversionFactor,
        range: MAG_SPEC.range
      }
    },
    fieldNaming: {
      raw: 'ax, ay, az, gx, gy, gz, mx, my, mz (LSB)',
      converted: 'ax_g, ay_g, az_g (g), gx_dps, gy_dps, gz_dps (deg/s), mx_ut, my_ut, mz_ut (µT)',
      note: 'Raw values ALWAYS preserved. Converted fields added as decorations.'
    }
  };
}

/** Validate that raw sensor values are in expected LSB range */
export function validateRawSensorUnits(raw: Partial<RawSensorData>): SensorValidationResult {
  const warnings: string[] = [];

  // Check accelerometer (should be -16384 to +16384 at ±2g)
  const accelMax = Math.max(
    Math.abs(raw.ax || 0),
    Math.abs(raw.ay || 0),
    Math.abs(raw.az || 0)
  );
  if (accelMax < 100) {
    warnings.push('Accelerometer values too small - may already be converted to g');
  } else if (accelMax > 32768) {
    warnings.push('Accelerometer values out of range for 16-bit sensor');
  }

  // Check gyroscope (should be -28000 to +28000 at ±245dps)
  const gyroMax = Math.max(
    Math.abs(raw.gx || 0),
    Math.abs(raw.gy || 0),
    Math.abs(raw.gz || 0)
  );
  if (gyroMax < 100) {
    warnings.push('Gyroscope values too small - may already be converted to deg/s');
  } else if (gyroMax > 32768) {
    warnings.push('Gyroscope values out of range for 16-bit sensor');
  }

  // Check magnetometer (should be -27368 to +27368 at ±4 gauss)
  const magMax = Math.max(
    Math.abs(raw.mx || 0),
    Math.abs(raw.my || 0),
    Math.abs(raw.mz || 0)
  );
  if (magMax < 100) {
    warnings.push('Magnetometer values too small - may already be converted to µT');
  } else if (magMax > 32768) {
    warnings.push('Magnetometer values out of range for 16-bit sensor');
  }

  // Check magnetometer magnitude
  const magMagnitudeLsb = Math.sqrt(
    (raw.mx || 0) ** 2 +
    (raw.my || 0) ** 2 +
    (raw.mz || 0) ** 2
  );
  const magMagnitudeUt = magMagnitudeLsb * MAG_SPEC.conversionFactor;
  if (magMagnitudeUt < 10) {
    warnings.push(`Magnetometer magnitude after conversion is ${magMagnitudeUt.toFixed(1)} µT - suspiciously low`);
  }

  return {
    valid: warnings.length === 0,
    warnings,
    accelMaxLsb: accelMax,
    gyroMaxLsb: gyroMax,
    magMaxLsb: magMax,
    magMagnitudeUt
  };
}

// ===== Magnetometer Axis Alignment =====

/**
 * Align magnetometer axes to accelerometer/gyroscope frame
 *
 * The LIS3MDL magnetometer has TRANSPOSED X/Y axes relative to LSM6DS3:
 *   Magnetometer native:  +X → fingers, +Y → wrist, +Z → palm
 *   Accel/Gyro native:    +X → wrist,   +Y → fingers, +Z → palm
 */
export function magAlignToAccelFrame(raw: MagRaw): MagRaw {
  return {
    mx: raw.my,  // Mag +Y (wrist) → Accel +X (wrist)
    my: raw.mx,  // Mag +X (fingers) → Accel +Y (fingers)
    mz: raw.mz   // Z unchanged (into palm)
  };
}

/** Convert magnetometer from LSB to µT AND align to accel frame */
export function convertMagToMicroTeslaAligned(raw: MagRaw): MagRaw {
  const converted = convertMagToMicroTesla(raw);
  return magAlignToAccelFrame(converted);
}

// ===== Default Export =====

export default {
  // Specifications
  ACCEL_SPEC,
  GYRO_SPEC,
  MAG_SPEC,
  MAG_SPEC_LIS3MDL,
  MAG_SPEC_MMC5603NJ,

  // Backward compatibility
  ACCEL_SCALE,
  GYRO_SCALE,
  MAG_SCALE_LSB_TO_UT,

  // Conversion functions
  accelLsbToG,
  gyroLsbToDps,
  magLsbToMicroTesla,
  convertAccelToG,
  convertGyroToDps,
  convertMagToMicroTesla,

  // Magnetometer axis alignment
  magAlignToAccelFrame,
  convertMagToMicroTeslaAligned,

  // Metadata and validation
  getSensorUnitMetadata,
  validateRawSensorUnits
};
