#!/usr/bin/env python3
"""
Convert Legacy Session Data - Add Unit-Converted Fields

This script processes existing session files that contain RAW LSB values
and adds properly unit-converted fields while preserving the originals.

POLICY:
- NEVER modifies raw LSB values (mx, my, mz, ax, ay, az, gx, gy, gz)
- ADDS decorated fields (mx_ut, my_ut, mz_ut, ax_g, ay_g, az_g, etc.)
- Adds metadata tracking which conversions were applied
- Idempotent: safe to run multiple times

Usage:
    python ml/convert_legacy_units.py --input data/GAMBIT/
    python ml/convert_legacy_units.py --file data/GAMBIT/session.json
    python ml/convert_legacy_units.py --file session.json --dry-run
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import shutil
from datetime import datetime

from sensor_units import (
    decorate_sample_with_units,
    get_sensor_unit_metadata,
    check_if_already_converted,
    validate_raw_sensor_units
)


def process_session_file(filepath: Path, dry_run: bool = False, backup: bool = True) -> Dict[str, Any]:
    """
    Process a single session file to add unit conversions

    Args:
        filepath: Path to session JSON file
        dry_run: If True, don't write changes
        backup: If True, create .bak backup before modifying

    Returns:
        Stats dict with counts of converted fields
    """
    print(f"\nProcessing: {filepath.name}")

    # Load session
    with open(filepath, 'r') as f:
        session = json.load(f)

    # Handle both session formats
    if isinstance(session, dict) and 'samples' in session:
        samples = session['samples']
        metadata = session.get('metadata', {})
    elif isinstance(session, list):
        samples = session
        metadata = {}
    else:
        print(f"  ⚠️  Unknown format, skipping")
        return {'skipped': 1}

    if not samples:
        print(f"  ⚠️  No samples, skipping")
        return {'skipped': 1}

    # Check first sample to see if already converted
    first_sample = samples[0]
    already_converted = check_if_already_converted(first_sample)

    if all(already_converted.values()):
        print(f"  ✓ Already converted, skipping")
        return {'already_converted': 1}

    # Validate that we have raw LSB values
    validation = validate_raw_sensor_units(first_sample)
    if not validation['valid']:
        print(f"  ⚠️  Validation warnings:")
        for warning in validation['warnings']:
            print(f"      - {warning}")
        print(f"  Magnetometer magnitude (after conversion): {validation['mag_magnitude_ut']:.1f} µT")

    # Decorate all samples with unit conversions
    samples_converted = 0
    samples_with_mag = 0

    for sample in samples:
        # Check what's already present
        converted = check_if_already_converted(sample)

        # Add missing conversions in place
        decorate_sample_with_units(sample, in_place=True)

        samples_converted += 1
        if 'mx_ut' in sample:
            samples_with_mag += 1

    # Add conversion metadata
    if isinstance(session, dict):
        if 'metadata' not in session:
            session['metadata'] = {}

        session['metadata']['unit_conversion'] = {
            'applied': True,
            'version': '1.0.0',
            'date': datetime.utcnow().isoformat() + 'Z',
            'script': 'convert_legacy_units.py',
            'sensor_specs': get_sensor_unit_metadata()
        }

    print(f"  ✓ Converted {samples_converted} samples")
    print(f"  ✓ {samples_with_mag} samples have magnetometer data")

    # Write back (with backup)
    if not dry_run:
        if backup:
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            shutil.copy2(filepath, backup_path)
            print(f"  ✓ Backup created: {backup_path.name}")

        with open(filepath, 'w') as f:
            json.dump(session, f, indent=2)
        print(f"  ✓ File updated")
    else:
        print(f"  🔍 DRY RUN - no changes made")

    return {
        'converted': 1,
        'samples_converted': samples_converted,
        'samples_with_mag': samples_with_mag
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert legacy GAMBIT session data to add unit-converted fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all sessions in directory (with backups)
  python ml/convert_legacy_units.py --input data/GAMBIT/

  # Convert single file
  python ml/convert_legacy_units.py --file data/GAMBIT/session.json

  # Dry run to see what would happen
  python ml/convert_legacy_units.py --input data/GAMBIT/ --dry-run

  # Convert without backups (not recommended)
  python ml/convert_legacy_units.py --input data/GAMBIT/ --no-backup
        """
    )

    parser.add_argument('--input', '-i', type=Path,
                       help='Directory containing session JSON files')
    parser.add_argument('--file', '-f', type=Path,
                       help='Single session JSON file to convert')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating .bak backups (not recommended)')
    parser.add_argument('--pattern', default='*.json',
                       help='File pattern to match (default: *.json)')

    args = parser.parse_args()

    if not args.input and not args.file:
        parser.error('Must specify either --input or --file')

    # Collect files to process
    files = []
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            return 1
        files.append(args.file)
    else:
        if not args.input.exists():
            print(f"Error: Directory not found: {args.input}")
            return 1
        files = list(args.input.glob(args.pattern))

    # Filter out backup files and special files
    files = [
        f for f in files
        if not f.name.endswith('.bak')
        and f.name not in ['manifest.json', 'gambit_calibration.json']
    ]

    if not files:
        print("No files to process")
        return 0

    print("=" * 80)
    print(f"GAMBIT Legacy Unit Conversion")
    print("=" * 80)
    print(f"Files to process: {len(files)}")
    print(f"Dry run: {'YES' % if args.dry_run else 'NO'}")
    print(f"Create backups: {'NO' if args.no_backup else 'YES'}")
    print()

    # Process files
    stats = {
        'converted': 0,
        'already_converted': 0,
        'skipped': 0,
        'errors': 0,
        'total_samples': 0
    }

    for filepath in files:
        try:
            result = process_session_file(
                filepath,
                dry_run=args.dry_run,
                backup=not args.no_backup
            )

            for key, value in result.items():
                if key in stats:
                    stats[key] += value
                elif key == 'samples_converted':
                    stats['total_samples'] += value

        except Exception as e:
            print(f"  ❌ Error: {e}")
            stats['errors'] += 1

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files converted:        {stats['converted']}")
    print(f"Already converted:      {stats['already_converted']}")
    print(f"Skipped:                {stats['skipped']}")
    print(f"Errors:                 {stats['errors']}")
    print(f"Total samples processed: {stats['total_samples']}")
    print()

    if args.dry_run:
        print("🔍 DRY RUN - No files were modified")
        print("   Run without --dry-run to apply changes")
    else:
        print("✅ Conversion complete!")
        if not args.no_backup:
            print("   Original files backed up with .bak extension")

    return 0 if stats['errors'] == 0 else 1


if __name__ == '__main__':
    exit(main())
