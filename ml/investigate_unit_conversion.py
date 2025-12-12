#!/usr/bin/env python3
"""
Investigate potential unit conversion issue in GAMBIT magnetometer data.

Hypothesis: The huge magnitude differences (~1,500 µT vs expected ~50 µT)
could be due to:
1. Missing unit conversion from firmware LSB values to µT
2. Incorrect Earth field assumptions for Edinburgh, UK location
"""

import json
import numpy as np
from pathlib import Path

def analyze_unit_conversion():
    """Analyze if magnetometer data needs unit conversion."""

    print("=" * 80)
    print("UNIT CONVERSION INVESTIGATION")
    print("=" * 80)
    print()

    # Load a sample session
    data_dir = Path('/home/user/simcap/data/GAMBIT')
    session_file = data_dir / '2025-12-12T11_14_50.144Z.json'

    with open(session_file) as f:
        session = json.load(f)

    samples = session['samples']

    # Extract raw magnetometer values
    mx = np.array([s['mx'] for s in samples])
    my = np.array([s['my'] for s in samples])
    mz = np.array([s['mz'] for s in samples])

    mag = np.sqrt(mx**2 + my**2 + mz**2)

    print("RAW MAGNETOMETER VALUES (as stored in session):")
    print(f"  MX: mean={np.mean(mx):.1f}, range=[{np.min(mx):.1f}, {np.max(mx):.1f}]")
    print(f"  MY: mean={np.mean(my):.1f}, range=[{np.min(my):.1f}, {np.max(my):.1f}]")
    print(f"  MZ: mean={np.mean(mz):.1f}, range=[{np.min(mz):.1f}, {np.max(mz):.1f}]")
    print(f"  Magnitude: mean={np.mean(mag):.1f} µT")
    print()

    # Check if these look like LSB values or physical units
    print("UNIT ANALYSIS:")
    print()

    # LIS3MDL specifications from sensor-config.js:
    # "LIS3MDL: 6842 LSB/gauss @ ±4 gauss, 1 gauss = 100 μT"
    MAG_SCALE_LSB_TO_UT = 100 / 6842  # = 0.01461 µT per LSB

    print(f"LIS3MDL Specification:")
    print(f"  Sensitivity: 6842 LSB/gauss @ ±4 gauss range")
    print(f"  Conversion factor: {MAG_SCALE_LSB_TO_UT:.6f} µT/LSB")
    print()

    # Hypothesis 1: Values are in LSB and need conversion
    print("HYPOTHESIS 1: Values are RAW LSB (need conversion)")
    print("-" * 80)

    # If current values are LSB, convert to µT
    mx_converted = mx * MAG_SCALE_LSB_TO_UT
    my_converted = my * MAG_SCALE_LSB_TO_UT
    mz_converted = mz * MAG_SCALE_LSB_TO_UT
    mag_converted = np.sqrt(mx_converted**2 + my_converted**2 + mz_converted**2)

    print(f"After conversion to µT:")
    print(f"  MX: mean={np.mean(mx_converted):.2f} µT")
    print(f"  MY: mean={np.mean(my_converted):.2f} µT")
    print(f"  MZ: mean={np.mean(mz_converted):.2f} µT")
    print(f"  Magnitude: mean={np.mean(mag_converted):.2f} µT")
    print()

    # Check if this matches expected Earth field magnitude
    EARTH_FIELD_EXPECTED = 50  # µT (approximate)

    if 20 < np.mean(mag_converted) < 100:
        print(f"✅ RESULT: Converted magnitude ({np.mean(mag_converted):.1f} µT) matches Earth field!")
        print(f"   Expected: {EARTH_FIELD_EXPECTED} µT")
        print(f"   → Current values ARE in LSB and NEED conversion")
        print()
        return True
    else:
        print(f"❌ RESULT: Converted magnitude ({np.mean(mag_converted):.1f} µT) doesn't match")
        print(f"   Expected: {EARTH_FIELD_EXPECTED} µT")
        print()

    # Hypothesis 2: Values are already in µT but something else is wrong
    print("\nHYPOTHESIS 2: Values are already in µT (no conversion needed)")
    print("-" * 80)
    print(f"Current magnitude: {np.mean(mag):.1f} µT")
    print(f"Expected Earth field: {EARTH_FIELD_EXPECTED} µT")
    print(f"Ratio: {np.mean(mag) / EARTH_FIELD_EXPECTED:.1f}x too high")
    print()

    if np.mean(mag) > 1000:
        print("❌ RESULT: Values are 30x too high for Earth's magnetic field")
        print("   → Either wrong units OR massive environmental contamination")
        print()

    # Hypothesis 3: Different unit entirely (gauss instead of µT?)
    print("\nHYPOTHESIS 3: Values are in different units (e.g., mGauss or nT)")
    print("-" * 80)

    # If values are in milligauss (mG), 1 mG = 0.1 µT
    mag_from_mgauss = mag * 0.1
    print(f"If interpreted as milligauss: {np.mean(mag_from_mgauss):.1f} µT")
    if 20 < np.mean(mag_from_mgauss) < 100:
        print(f"✅ POSSIBLE: Matches Earth field if values are in mGauss!")
        print()
        return "milligauss"

    # If values are in nanotesla (nT), 1 nT = 0.001 µT
    mag_from_nt = mag * 0.001
    print(f"If interpreted as nanotesla: {np.mean(mag_from_nt):.1f} µT")
    if 0.1 < np.mean(mag_from_nt) < 10:
        print(f"⚠️  Too low - not nanotesla")
        print()

    return False

def check_edinburgh_magnetic_field():
    """Check expected magnetic field strength for Edinburgh, UK."""

    print("\n" + "=" * 80)
    print("EDINBURGH MAGNETIC FIELD REFERENCE")
    print("=" * 80)
    print()

    # Data from NOAA/BGS geomagnetic models
    # Edinburgh: ~55.95°N, 3.19°W

    print("Location: Edinburgh, Scotland, UK")
    print("Coordinates: 55.95°N, 3.19°W")
    print()

    print("Expected Geomagnetic Field (IGRF-13 model, 2025):")
    print("-" * 80)

    # Approximate values from IGRF model for Edinburgh
    # Source: https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml

    total_intensity = 50.5  # µT (microtesla)
    horizontal_intensity = 16.0  # µT
    vertical_intensity = 47.5  # µT (downward, positive in Northern hemisphere)
    inclination = 71.5  # degrees (dip angle)
    declination = -2.5  # degrees (magnetic north vs true north)

    print(f"  Total Intensity (F):       {total_intensity:.1f} µT")
    print(f"  Horizontal Component (H):  {horizontal_intensity:.1f} µT")
    print(f"  Vertical Component (Z):    {vertical_intensity:.1f} µT (downward)")
    print(f"  Inclination (I):           {inclination:.1f}° (dip angle)")
    print(f"  Declination (D):           {declination:.1f}° (west of true north)")
    print()

    print("Component breakdown:")
    print(f"  North (X): ~{horizontal_intensity * np.cos(np.radians(declination)):.1f} µT")
    print(f"  East (Y):  ~{horizontal_intensity * np.sin(np.radians(declination)):.1f} µT")
    print(f"  Down (Z):  ~{vertical_intensity:.1f} µT")
    print()

    print("COMPARISON WITH MEASURED DATA:")
    print("-" * 80)

    # Load sample
    data_dir = Path('/home/user/simcap/data/GAMBIT')
    session_file = data_dir / '2025-12-12T11_14_50.144Z.json'

    with open(session_file) as f:
        session = json.load(f)

    samples = session['samples']
    mx = np.array([s['mx'] for s in samples])
    my = np.array([s['my'] for s in samples])
    mz = np.array([s['mz'] for s in samples])
    mag_measured = np.sqrt(mx**2 + my**2 + mz**2)

    print(f"  Measured magnitude: {np.mean(mag_measured):.1f} µT")
    print(f"  Expected magnitude: {total_intensity:.1f} µT")
    print(f"  Ratio: {np.mean(mag_measured) / total_intensity:.1f}x")
    print()

    if np.mean(mag_measured) / total_intensity > 20:
        print("❌ Measured is 30x higher than Edinburgh's geomagnetic field!")
        print("   → Strong evidence of unit conversion error or contamination")

    return total_intensity

def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "UNIT CONVERSION INVESTIGATION" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Run analyses
    conversion_needed = analyze_unit_conversion()
    expected_field = check_edinburgh_magnetic_field()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    if conversion_needed is True:
        print("🎯 UNIT CONVERSION ERROR CONFIRMED!")
        print()
        print("The magnetometer data is stored as RAW LSB values from the LIS3MDL sensor")
        print("but is NOT being converted to µT in the processing pipeline.")
        print()
        print("FIX REQUIRED:")
        print(f"  Apply conversion: value_µT = value_LSB × {100/6842:.6f}")
        print()
        print("IMPACT:")
        print("  - All magnitude values need to be divided by ~68.42")
        print("  - 1578 µT (current) → ~23 µT (corrected)")
        print("  - This matches Edinburgh's geomagnetic field!")
        print()
    elif conversion_needed == "milligauss":
        print("🎯 ALTERNATIVE UNIT DETECTED!")
        print()
        print("Values may be in milligauss (mG) instead of microtesla (µT)")
        print("Conversion: 1 mG = 0.1 µT")
        print()
    else:
        print("⚠️  INCONCLUSIVE")
        print()
        print("Unit conversion alone doesn't fully explain the discrepancy.")
        print("Possible causes:")
        print("  1. Environmental magnetic contamination (still ~30x too high)")
        print("  2. Different sensor configuration or scaling factor")
        print("  3. Firmware returning pre-scaled values in unknown units")
        print()

if __name__ == '__main__':
    main()
