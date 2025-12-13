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

// ===== Sensor Hardware Specifications =====

/**
 * Accelerometer: LSM6DS3 (Puck.js)
 *
 * Range: ±2g (default)
 * Resolution: 16-bit
 * Sensitivity: 8192 LSB/g
 *
 * UNITS:
 * - Raw (firmware): LSB (int16)
 * - Converted: g (standard gravity)
 */
export const ACCEL_SPEC = {
    sensor: 'LSM6DS3',
    rawUnit: 'LSB',
    convertedUnit: 'g',
    range: '±2g',
    resolution: 16,
    sensitivity: 8192, // LSB per g
    conversionFactor: 1 / 8192, // g per LSB
    reference: 'https://www.st.com/resource/en/datasheet/lsm6ds3.pdf'
};

/**
 * Gyroscope: LSM6DS3 (Puck.js)
 *
 * Range: ±245 dps (default)
 * Resolution: 16-bit
 * Sensitivity: 114.28 LSB/dps
 *
 * UNITS:
 * - Raw (firmware): LSB (int16)
 * - Converted: deg/s (degrees per second)
 */
export const GYRO_SPEC = {
    sensor: 'LSM6DS3',
    rawUnit: 'LSB',
    convertedUnit: 'deg/s',
    range: '±245dps',
    resolution: 16,
    sensitivity: 114.28, // LSB per deg/s
    conversionFactor: 1 / 114.28, // deg/s per LSB
    reference: 'https://www.st.com/resource/en/datasheet/lsm6ds3.pdf'
};

/**
 * Magnetometer: LIS3MDL (Puck.js)
 *
 * Range: ±4 gauss (default)
 * Resolution: 16-bit
 * Sensitivity: 6842 LSB/gauss
 *
 * UNITS:
 * - Raw (firmware): LSB (int16)
 * - Converted: µT (microtesla)
 *
 * CONVERSION: 1 gauss = 100 µT
 * Therefore: 1 µT = 6842/100 = 68.42 LSB
 *           1 LSB = 100/6842 = 0.014616 µT
 *
 * CRITICAL: Puck.mag() returns RAW LSB values, NOT physical units!
 */
export const MAG_SPEC = {
    sensor: 'LIS3MDL',
    rawUnit: 'LSB',
    convertedUnit: 'µT',
    range: '±4gauss',
    resolution: 16,
    sensitivity: 6842, // LSB per gauss
    gaussToMicroTesla: 100,
    conversionFactor: 100 / 6842, // µT per LSB (0.014616)
    reference: 'https://www.st.com/resource/en/datasheet/lis3mdl.pdf',
    notes: [
        'Puck.mag() returns RAW LSB values',
        'Must multiply by conversionFactor to get µT',
        'Earth\'s magnetic field: ~25-65 µT total',
        'Edinburgh, UK: ~50.5 µT total'
    ]
};

// ===== Backward Compatibility Exports =====
// These maintain compatibility with existing code

export const ACCEL_SCALE = ACCEL_SPEC.sensitivity;
export const GYRO_SCALE = GYRO_SPEC.sensitivity;
export const MAG_SCALE_LSB_TO_UT = MAG_SPEC.conversionFactor;

// ===== Unit Conversion Functions =====

/**
 * Convert accelerometer from LSB to g
 * PRESERVES raw value, returns converted value
 *
 * @param {number} lsb - Raw value in LSB
 * @returns {number} Value in g
 */
export function accelLsbToG(lsb) {
    return lsb * ACCEL_SPEC.conversionFactor;
}

/**
 * Convert gyroscope from LSB to deg/s
 * PRESERVES raw value, returns converted value
 *
 * @param {number} lsb - Raw value in LSB
 * @returns {number} Value in deg/s
 */
export function gyroLsbToDps(lsb) {
    return lsb * GYRO_SPEC.conversionFactor;
}

/**
 * Convert magnetometer from LSB to µT
 * PRESERVES raw value, returns converted value
 *
 * @param {number} lsb - Raw value in LSB
 * @returns {number} Value in µT (microtesla)
 */
export function magLsbToMicroTesla(lsb) {
    return lsb * MAG_SPEC.conversionFactor;
}

/**
 * Convert accelerometer vector from LSB to g
 * PRESERVES raw values, returns new object with converted values
 *
 * @param {Object} raw - {ax, ay, az} in LSB
 * @returns {Object} {ax, ay, az} in g
 */
export function convertAccelToG(raw) {
    return {
        ax: accelLsbToG(raw.ax || 0),
        ay: accelLsbToG(raw.ay || 0),
        az: accelLsbToG(raw.az || 0)
    };
}

/**
 * Convert gyroscope vector from LSB to deg/s
 * PRESERVES raw values, returns new object with converted values
 *
 * @param {Object} raw - {gx, gy, gz} in LSB
 * @returns {Object} {gx, gy, gz} in deg/s
 */
export function convertGyroToDps(raw) {
    return {
        gx: gyroLsbToDps(raw.gx || 0),
        gy: gyroLsbToDps(raw.gy || 0),
        gz: gyroLsbToDps(raw.gz || 0)
    };
}

/**
 * Convert magnetometer vector from LSB to µT
 * PRESERVES raw values, returns new object with converted values
 *
 * @param {Object} raw - {mx, my, mz} in LSB
 * @returns {Object} {mx, my, mz} in µT
 */
export function convertMagToMicroTesla(raw) {
    return {
        mx: magLsbToMicroTesla(raw.mx || 0),
        my: magLsbToMicroTesla(raw.my || 0),
        mz: magLsbToMicroTesla(raw.mz || 0)
    };
}

// ===== Unit Metadata for Sessions =====

/**
 * Get sensor unit configuration for session metadata
 * This should be stored with each session to document conversions
 *
 * @returns {Object} Sensor specifications and conversion factors
 */
export function getSensorUnitMetadata() {
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

/**
 * Validate that raw sensor values are in expected LSB range
 * Useful for detecting if values have already been converted
 *
 * @param {Object} raw - Raw sensor data {ax, ay, az, gx, gy, gz, mx, my, mz}
 * @returns {Object} {valid: boolean, warnings: string[]}
 */
export function validateRawSensorUnits(raw) {
    const warnings = [];

    // Check accelerometer (should be -16384 to +16384 at ±2g)
    const accelMax = Math.max(Math.abs(raw.ax || 0), Math.abs(raw.ay || 0), Math.abs(raw.az || 0));
    if (accelMax < 100) {
        warnings.push('Accelerometer values too small - may already be converted to g');
    } else if (accelMax > 32768) {
        warnings.push('Accelerometer values out of range for 16-bit sensor');
    }

    // Check gyroscope (should be -28000 to +28000 at ±245dps)
    const gyroMax = Math.max(Math.abs(raw.gx || 0), Math.abs(raw.gy || 0), Math.abs(raw.gz || 0));
    if (gyroMax < 100) {
        warnings.push('Gyroscope values too small - may already be converted to deg/s');
    } else if (gyroMax > 32768) {
        warnings.push('Gyroscope values out of range for 16-bit sensor');
    }

    // Check magnetometer (should be -27368 to +27368 at ±4 gauss)
    const magMax = Math.max(Math.abs(raw.mx || 0), Math.abs(raw.my || 0), Math.abs(raw.mz || 0));
    if (magMax < 100) {
        warnings.push('Magnetometer values too small - may already be converted to µT');
    } else if (magMax > 32768) {
        warnings.push('Magnetometer values out of range for 16-bit sensor');
    }

    // Check magnetometer magnitude (after conversion, should be 20-2000 µT for Earth + environment)
    const magMagnitudeLsb = Math.sqrt(
        (raw.mx || 0) ** 2 +
        (raw.my || 0) ** 2 +
        (raw.mz || 0) ** 2
    );
    const magMagnitudeUt = magMagnitudeLsb * MAG_SPEC.conversionFactor;
    if (magMagnitudeUt < 10) {
        warnings.push(`Magnetometer magnitude after conversion is ${magMagnitudeUt.toFixed(1)} µT - suspiciously low`);
    } else if (magMagnitudeUt > 200 && magMagnitudeUt < 5000) {
        // This is actually fine - could be environmental contamination
        // Just note it
    }

    return {
        valid: warnings.length === 0,
        warnings: warnings,
        accelMaxLsb: accelMax,
        gyroMaxLsb: gyroMax,
        magMaxLsb: magMax,
        magMagnitudeUt: magMagnitudeUt
    };
}

// ===== Magnetometer Axis Alignment =====

/**
 * TODO: [SENSOR-002] Align magnetometer axes to accelerometer/gyroscope frame
 *
 * The LIS3MDL magnetometer has TRANSPOSED X/Y axes relative to LSM6DS3:
 *
 *   Magnetometer native:  +X → fingers, +Y → wrist, +Z → palm
 *   Accel/Gyro native:    +X → wrist,   +Y → fingers, +Z → palm
 *
 * This function swaps X and Y to align magnetometer data with the
 * accelerometer/gyroscope coordinate frame used by the IMU fusion.
 *
 * WHEN TO USE:
 *   - Before passing magnetometer data to calibration (if calibration
 *     will be used with orientation-dependent operations)
 *   - Before any operation that combines mag data with orientation
 *
 * WHEN NOT TO USE:
 *   - For standalone magnetometer operations (hard/soft iron cal)
 *   - When consistency with historical data is required
 *
 * @param {Object} raw - {mx, my, mz} in magnetometer native frame
 * @returns {Object} {mx, my, mz} in accelerometer/gyroscope frame
 */
export function magAlignToAccelFrame(raw) {
    // Swap X and Y to align with accelerometer frame
    return {
        mx: raw.my,  // Mag +Y (wrist) → Accel +X (wrist)
        my: raw.mx,  // Mag +X (fingers) → Accel +Y (fingers)
        mz: raw.mz   // Z unchanged (into palm)
    };
}

/**
 * Convert magnetometer from LSB to µT AND align to accel frame
 *
 * @param {Object} raw - {mx, my, mz} in LSB
 * @returns {Object} {mx, my, mz} in µT, aligned to accel/gyro frame
 */
export function convertMagToMicroTeslaAligned(raw) {
    // First convert to µT
    const converted = convertMagToMicroTesla(raw);
    // Then align axes
    return magAlignToAccelFrame(converted);
}

// ===== Default Export =====

export default {
    // Specifications
    ACCEL_SPEC,
    GYRO_SPEC,
    MAG_SPEC,

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
