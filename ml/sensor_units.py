"""
Sensor Units and Conversion Configuration

CRITICAL: This module defines the authoritative unit conversions for all sensors.
All conversions must preserve raw data and only ADD decorated fields.

UNITS POLICY:
- RAW values are ALWAYS preserved in their native sensor units (LSB)
- CONVERTED values are added as decorated fields with explicit unit suffixes
- Session metadata MUST track which conversions were applied

This is the Python equivalent of src/web/GAMBIT/shared/sensor-units.js
"""

from typing import Dict, Tuple, Any
import numpy as np

# ===== Sensor Hardware Specifications =====

ACCEL_SPEC = {
    'sensor': 'LSM6DS3',
    'raw_unit': 'LSB',
    'converted_unit': 'g',
    'range': '±2g',
    'resolution': 16,
    'sensitivity': 8192,  # LSB per g
    'conversion_factor': 1 / 8192,  # g per LSB
    'reference': 'https://www.st.com/resource/en/datasheet/lsm6ds3.pdf'
}

GYRO_SPEC = {
    'sensor': 'LSM6DS3',
    'raw_unit': 'LSB',
    'converted_unit': 'deg/s',
    'range': '±245dps',
    'resolution': 16,
    'sensitivity': 114.28,  # LSB per deg/s
    'conversion_factor': 1 / 114.28,  # deg/s per LSB
    'reference': 'https://www.st.com/resource/en/datasheet/lsm6ds3.pdf'
}

# Magnetometer: LIS3MDL (Puck.js v2) - LEGACY
MAG_SPEC_LIS3MDL = {
    'sensor': 'LIS3MDL',
    'puck_version': 'v2',
    'raw_unit': 'LSB',
    'converted_unit': 'µT',
    'range': '±4gauss',
    'resolution': 16,
    'sensitivity': 6842,  # LSB per gauss
    'gauss_to_microtesla': 100,
    'conversion_factor': 100 / 6842,  # µT per LSB (0.014616)
    'reference': 'https://www.st.com/resource/en/datasheet/lis3mdl.pdf'
}

# Magnetometer: MMC5603NJ (Puck.js v2.1a) - CURRENT
# From datasheet: 16-bit mode = 1024 counts/Gauss
MAG_SPEC_MMC5603NJ = {
    'sensor': 'MMC5603NJ',
    'puck_version': 'v2.1a',
    'raw_unit': 'LSB',
    'converted_unit': 'µT',
    'range': '±30gauss',
    'resolution': 16,  # Espruino uses 16-bit mode
    'sensitivity': 1024,  # LSB per gauss (16-bit mode from datasheet)
    'gauss_to_microtesla': 100,
    'conversion_factor': 100 / 1024,  # µT per LSB (0.09765625)
    'reference': 'src/device/GAMBIT/MMC5603NJ.pdf',
    'notes': [
        'Puck.mag() returns RAW LSB values',
        'Must multiply by conversion_factor to get µT',
        'Earth\'s magnetic field: ~25-65 µT total',
        'Edinburgh, UK: ~50.5 µT total',
        '16-bit mode: 1024 counts/Gauss (from datasheet page 2)',
        'Expected ~461 LSB for 45 µT Earth field'
    ]
}

# Active magnetometer specification
# IMPORTANT: Set this based on your Puck.js hardware version!
# - Puck.js v2: LIS3MDL (MAG_SPEC_LIS3MDL)
# - Puck.js v2.1a: MMC5603NJ (MAG_SPEC_MMC5603NJ)
# Default: MMC5603NJ (Puck.js v2.1a) - most common current hardware
MAG_SPEC = MAG_SPEC_MMC5603NJ

# ===== Backward Compatibility Constants =====

ACCEL_SCALE = ACCEL_SPEC['sensitivity']
GYRO_SCALE = GYRO_SPEC['sensitivity']
MAG_SCALE_LSB_TO_UT = MAG_SPEC['conversion_factor']

# ===== Unit Conversion Functions =====

def accel_lsb_to_g(lsb: float) -> float:
    """
    Convert accelerometer from LSB to g
    PRESERVES raw value, returns converted value

    Args:
        lsb: Raw value in LSB

    Returns:
        Value in g
    """
    return lsb * ACCEL_SPEC['conversion_factor']


def gyro_lsb_to_dps(lsb: float) -> float:
    """
    Convert gyroscope from LSB to deg/s
    PRESERVES raw value, returns converted value

    Args:
        lsb: Raw value in LSB

    Returns:
        Value in deg/s
    """
    return lsb * GYRO_SPEC['conversion_factor']


def mag_lsb_to_microtesla(lsb: float) -> float:
    """
    Convert magnetometer from LSB to µT
    PRESERVES raw value, returns converted value

    CRITICAL: This is the fix for the unit conversion bug.
    Firmware returns RAW LSB values that must be converted.

    Args:
        lsb: Raw value in LSB

    Returns:
        Value in µT (microtesla)
    """
    return lsb * MAG_SPEC['conversion_factor']


def convert_accel_to_g(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Convert accelerometer vector from LSB to g
    PRESERVES raw values, returns new dict with converted values

    Args:
        raw: {ax, ay, az} in LSB

    Returns:
        {ax, ay, az} in g
    """
    return {
        'ax': accel_lsb_to_g(raw.get('ax', 0)),
        'ay': accel_lsb_to_g(raw.get('ay', 0)),
        'az': accel_lsb_to_g(raw.get('az', 0))
    }


def convert_gyro_to_dps(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Convert gyroscope vector from LSB to deg/s
    PRESERVES raw values, returns new dict with converted values

    Args:
        raw: {gx, gy, gz} in LSB

    Returns:
        {gx, gy, gz} in deg/s
    """
    return {
        'gx': gyro_lsb_to_dps(raw.get('gx', 0)),
        'gy': gyro_lsb_to_dps(raw.get('gy', 0)),
        'gz': gyro_lsb_to_dps(raw.get('gz', 0))
    }


def convert_mag_to_microtesla(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Convert magnetometer vector from LSB to µT
    PRESERVES raw values, returns new dict with converted values

    Args:
        raw: {mx, my, mz} in LSB

    Returns:
        {mx, my, mz} in µT
    """
    return {
        'mx': mag_lsb_to_microtesla(raw.get('mx', 0)),
        'my': mag_lsb_to_microtesla(raw.get('my', 0)),
        'mz': mag_lsb_to_microtesla(raw.get('mz', 0))
    }


# ===== Session Metadata =====

def get_sensor_unit_metadata() -> Dict[str, Any]:
    """
    Get sensor unit configuration for session metadata
    This should be stored with each session to document conversions

    Returns:
        Sensor specifications and conversion factors
    """
    from datetime import datetime

    return {
        'version': '1.0.0',
        'date': datetime.utcnow().isoformat() + 'Z',
        'sensors': {
            'accelerometer': {
                'sensor': ACCEL_SPEC['sensor'],
                'raw_unit': ACCEL_SPEC['raw_unit'],
                'converted_unit': ACCEL_SPEC['converted_unit'],
                'conversion_factor': ACCEL_SPEC['conversion_factor'],
                'range': ACCEL_SPEC['range']
            },
            'gyroscope': {
                'sensor': GYRO_SPEC['sensor'],
                'raw_unit': GYRO_SPEC['raw_unit'],
                'converted_unit': GYRO_SPEC['converted_unit'],
                'conversion_factor': GYRO_SPEC['conversion_factor'],
                'range': GYRO_SPEC['range']
            },
            'magnetometer': {
                'sensor': MAG_SPEC['sensor'],
                'raw_unit': MAG_SPEC['raw_unit'],
                'converted_unit': MAG_SPEC['converted_unit'],
                'conversion_factor': MAG_SPEC['conversion_factor'],
                'range': MAG_SPEC['range']
            }
        },
        'field_naming': {
            'raw': 'ax, ay, az, gx, gy, gz, mx, my, mz (LSB)',
            'converted': 'ax_g, ay_g, az_g (g), gx_dps, gy_dps, gz_dps (deg/s), mx_ut, my_ut, mz_ut (µT)',
            'note': 'Raw values ALWAYS preserved. Converted fields added as decorations.'
        }
    }


def validate_raw_sensor_units(raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate that raw sensor values are in expected LSB range
    Useful for detecting if values have already been converted

    Args:
        raw: Raw sensor data {ax, ay, az, gx, gy, gz, mx, my, mz}

    Returns:
        {valid: bool, warnings: list, ...stats}
    """
    warnings = []

    # Check accelerometer (should be -16384 to +16384 at ±2g)
    accel_max = max(
        abs(raw.get('ax', 0)),
        abs(raw.get('ay', 0)),
        abs(raw.get('az', 0))
    )
    if accel_max < 100:
        warnings.append('Accelerometer values too small - may already be converted to g')
    elif accel_max > 32768:
        warnings.append('Accelerometer values out of range for 16-bit sensor')

    # Check gyroscope (should be -28000 to +28000 at ±245dps)
    gyro_max = max(
        abs(raw.get('gx', 0)),
        abs(raw.get('gy', 0)),
        abs(raw.get('gz', 0))
    )
    if gyro_max < 100:
        warnings.append('Gyroscope values too small - may already be converted to deg/s')
    elif gyro_max > 32768:
        warnings.append('Gyroscope values out of range for 16-bit sensor')

    # Check magnetometer (should be -27368 to +27368 at ±4 gauss)
    mag_max = max(
        abs(raw.get('mx', 0)),
        abs(raw.get('my', 0)),
        abs(raw.get('mz', 0))
    )
    if mag_max < 100:
        warnings.append('Magnetometer values too small - may already be converted to µT')
    elif mag_max > 32768:
        warnings.append('Magnetometer values out of range for 16-bit sensor')

    # Check magnetometer magnitude
    mag_magnitude_lsb = np.sqrt(
        raw.get('mx', 0) ** 2 +
        raw.get('my', 0) ** 2 +
        raw.get('mz', 0) ** 2
    )
    mag_magnitude_ut = mag_magnitude_lsb * MAG_SPEC['conversion_factor']

    if mag_magnitude_ut < 10:
        warnings.append(f"Magnetometer magnitude after conversion is {mag_magnitude_ut:.1f} µT - suspiciously low")

    return {
        'valid': len(warnings) == 0,
        'warnings': warnings,
        'accel_max_lsb': accel_max,
        'gyro_max_lsb': gyro_max,
        'mag_max_lsb': mag_max,
        'mag_magnitude_ut': mag_magnitude_ut
    }


def decorate_sample_with_units(sample: Dict[str, Any], in_place: bool = False) -> Dict[str, Any]:
    """
    Add unit-converted fields to a sample
    PRESERVES raw LSB values, adds decorated fields

    Args:
        sample: Raw sample dict with ax, ay, az, gx, gy, gz, mx, my, mz in LSB
        in_place: If True, modify sample dict in place. If False, return new dict.

    Returns:
        Sample dict with added fields: *_g, *_dps, *_ut
    """
    if in_place:
        decorated = sample
    else:
        decorated = sample.copy()

    # Convert accelerometer (if present)
    if 'ax' in sample:
        decorated['ax_g'] = accel_lsb_to_g(sample['ax'])
        decorated['ay_g'] = accel_lsb_to_g(sample['ay'])
        decorated['az_g'] = accel_lsb_to_g(sample['az'])

    # Convert gyroscope (if present)
    if 'gx' in sample:
        decorated['gx_dps'] = gyro_lsb_to_dps(sample['gx'])
        decorated['gy_dps'] = gyro_lsb_to_dps(sample['gy'])
        decorated['gz_dps'] = gyro_lsb_to_dps(sample['gz'])

    # Convert magnetometer (if present and not null)
    if 'mx' in sample and sample['mx'] is not None:
        decorated['mx_ut'] = mag_lsb_to_microtesla(sample['mx'])
        decorated['my_ut'] = mag_lsb_to_microtesla(sample['my'])
        decorated['mz_ut'] = mag_lsb_to_microtesla(sample['mz'])

    return decorated


def check_if_already_converted(sample: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check if a sample already has converted fields
    Helps with idempotency - don't double-convert!

    Args:
        sample: Sample dict

    Returns:
        {accel: bool, gyro: bool, mag: bool} - True if converted fields exist
    """
    return {
        'accel': 'ax_g' in sample or 'ay_g' in sample or 'az_g' in sample,
        'gyro': 'gx_dps' in sample or 'gy_dps' in sample or 'gz_dps' in sample,
        'mag': 'mx_ut' in sample or 'my_ut' in sample or 'mz_ut' in sample
    }


__all__ = [
    # Specs
    'ACCEL_SPEC',
    'GYRO_SPEC',
    'MAG_SPEC',

    # Constants (backward compat)
    'ACCEL_SCALE',
    'GYRO_SCALE',
    'MAG_SCALE_LSB_TO_UT',

    # Conversion functions
    'accel_lsb_to_g',
    'gyro_lsb_to_dps',
    'mag_lsb_to_microtesla',
    'convert_accel_to_g',
    'convert_gyro_to_dps',
    'convert_mag_to_microtesla',

    # Metadata and validation
    'get_sensor_unit_metadata',
    'validate_raw_sensor_units',
    'decorate_sample_with_units',
    'check_if_already_converted'
]
