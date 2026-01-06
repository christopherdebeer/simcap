#!/usr/bin/env python3
"""
Convert Calibration File Units

Converts gambit_calibration.json from LSB to µT units.

CRITICAL: According to sensor-units-policy.md, calibration offsets are stored
in LSB and must be converted to µT to match the unit-converted sensor data.

Usage:
    python ml/convert_calibration_file.py --file data/GAMBIT/gambit_calibration.json --dry-run
    python ml/convert_calibration_file.py --file data/GAMBIT/gambit_calibration.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import shutil

from ml.sensor_units import MAG_SPEC, mag_lsb_to_microtesla


def convert_calibration_file(filepath: Path, dry_run: bool = False, backup: bool = True):
    """
    Convert calibration file from LSB to µT

    Args:
        filepath: Path to gambit_calibration.json
        dry_run: If True, don't write changes
        backup: If True, create .bak backup before modifying

    Returns:
        Converted calibration dict
    """
    print(f"\nProcessing: {filepath.name}")

    # Load calibration
    with open(filepath, 'r') as f:
        cal = json.load(f)

    # Check if already converted
    if 'units' in cal and cal['units'].get('hardIronOffset') == 'µT':
        print(f"  ✓ Already converted to µT, skipping")
        return cal

    # Display original values
    print(f"\n  Original values (LSB):")
    if 'hardIronOffset' in cal:
        print(f"    Hard Iron Offset: {cal['hardIronOffset']}")
    if 'earthField' in cal:
        print(f"    Earth Field: {cal['earthField']}")
    if 'earthFieldMagnitude' in cal:
        print(f"    Earth Field Magnitude: {cal['earthFieldMagnitude']:.2f}")

    # Convert hard iron offset
    if 'hardIronOffset' in cal:
        cal['hardIronOffset'] = {
            'x': mag_lsb_to_microtesla(cal['hardIronOffset']['x']),
            'y': mag_lsb_to_microtesla(cal['hardIronOffset']['y']),
            'z': mag_lsb_to_microtesla(cal['hardIronOffset']['z'])
        }

    # Convert Earth field
    if 'earthField' in cal:
        cal['earthField'] = {
            'x': mag_lsb_to_microtesla(cal['earthField']['x']),
            'y': mag_lsb_to_microtesla(cal['earthField']['y']),
            'z': mag_lsb_to_microtesla(cal['earthField']['z'])
        }

    # Convert Earth field magnitude
    if 'earthFieldMagnitude' in cal:
        cal['earthFieldMagnitude'] = mag_lsb_to_microtesla(cal['earthFieldMagnitude'])

    # Add units metadata
    cal['units'] = {
        'hardIronOffset': 'µT',
        'earthField': 'µT',
        'earthFieldMagnitude': 'µT',
        'converted_from': 'LSB',
        'conversion_factor': MAG_SPEC['conversion_factor'],
        'conversion_date': datetime.utcnow().isoformat() + 'Z',
        'script': 'convert_calibration_file.py'
    }

    # Display converted values
    print(f"\n  Converted values (µT):")
    if 'hardIronOffset' in cal:
        print(f"    Hard Iron Offset: {{x: {cal['hardIronOffset']['x']:.2f}, " +
              f"y: {cal['hardIronOffset']['y']:.2f}, z: {cal['hardIronOffset']['z']:.2f}}}")
    if 'earthField' in cal:
        print(f"    Earth Field: {{x: {cal['earthField']['x']:.2f}, " +
              f"y: {cal['earthField']['y']:.2f}, z: {cal['earthField']['z']:.2f}}}")
    if 'earthFieldMagnitude' in cal:
        print(f"    Earth Field Magnitude: {cal['earthFieldMagnitude']:.2f} µT")

    # Check reasonableness
    if 'earthFieldMagnitude' in cal:
        mag = cal['earthFieldMagnitude']
        if mag < 10 or mag > 200:
            print(f"\n  ⚠️  WARNING: Earth field magnitude is {mag:.2f} µT")
            print(f"      Expected: 25-65 µT for Earth's field")
            print(f"      This calibration may be invalid")

    # Write back
    if not dry_run:
        if backup:
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            shutil.copy2(filepath, backup_path)
            print(f"\n  ✓ Backup created: {backup_path.name}")

        with open(filepath, 'w') as f:
            json.dump(cal, f, indent=2)
        print(f"  ✓ File updated")
    else:
        print(f"\n  🔍 DRY RUN - no changes made")

    return cal


def main():
    parser = argparse.ArgumentParser(
        description='Convert GAMBIT calibration file from LSB to µT',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--file', '-f', type=Path, required=True,
                       help='Path to gambit_calibration.json')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating .bak backup (not recommended)')

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    print("=" * 80)
    print("GAMBIT Calibration File Unit Conversion")
    print("=" * 80)
    print(f"Dry run: {'YES' if args.dry_run else 'NO'}")
    print(f"Create backup: {'NO' if args.no_backup else 'YES'}")

    try:
        cal = convert_calibration_file(
            args.file,
            dry_run=args.dry_run,
            backup=not args.no_backup
        )

        print("\n" + "=" * 80)
        if args.dry_run:
            print("🔍 DRY RUN - No files were modified")
            print("   Run without --dry-run to apply changes")
        else:
            print("✅ Conversion complete!")
            if not args.no_backup:
                print("   Original file backed up with .bak extension")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
