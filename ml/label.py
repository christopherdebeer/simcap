#!/usr/bin/env python3
"""
SIMCAP Data Labeling Tool

Command-line tool for annotating GAMBIT sensor data with gesture labels.

Usage:
    python -m ml.label data/GAMBIT/2024-01-05T16:41:20.581Z.json

Interactive commands:
    l <start> <end> <gesture>  - Label a segment
    s                          - Show current labels
    p                          - Play/visualize data
    d <index>                  - Delete label at index
    m <key> <value>            - Set metadata
    w                          - Write and save
    q                          - Quit without saving
    h                          - Help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .schema import (
    Gesture, SessionMetadata, LabeledSegment,
    FEATURE_NAMES
)
from .data_loader import load_session_data


def print_help():
    """Print interactive commands help."""
    print("""
Commands:
    l <start> <end> <gesture>  - Label segment (e.g., 'l 0 100 fist')
    ls                         - List all labels
    p [start] [end]            - Print/visualize data segment
    d <index>                  - Delete label at index
    m <key> <value>            - Set metadata (subject_id, environment, hand, split)
    info                       - Show session info
    gestures                   - List available gestures
    w                          - Write/save labels
    q                          - Quit without saving
    h                          - Help

Gestures: """ + ', '.join(Gesture.names()))


def visualize_segment(data: np.ndarray, start: int = 0, end: int = None,
                      labels: Optional[List[LabeledSegment]] = None):
    """
    Print ASCII visualization of sensor data segment.
    """
    if end is None:
        end = min(start + 50, len(data))

    end = min(end, len(data))

    print(f"\nSamples {start}-{end} ({end-start} samples)")
    print("=" * 70)

    # Header
    print(f"{'Sample':>6} | {'Acc X':>8} {'Acc Y':>8} {'Acc Z':>8} | "
          f"{'Gyro X':>8} {'Gyro Y':>8} {'Gyro Z':>8}")
    print("-" * 70)

    # Find label for each sample
    sample_labels = {}
    if labels:
        for seg in labels:
            for i in range(seg.start_sample, seg.end_sample):
                sample_labels[i] = seg.gesture.name[:4]

    for i in range(start, end):
        sample = data[i]
        label_str = sample_labels.get(i, "")
        print(f"{i:>6} | {sample[0]:>8.0f} {sample[1]:>8.0f} {sample[2]:>8.0f} | "
              f"{sample[3]:>8.0f} {sample[4]:>8.0f} {sample[5]:>8.0f} "
              f"{label_str}")

    print("-" * 70)

    # Simple magnitude plot using ASCII
    print("\nAccelerometer magnitude (normalized):")
    mags = np.sqrt(np.sum(data[start:end, :3] ** 2, axis=1))
    min_mag, max_mag = mags.min(), mags.max()
    range_mag = max_mag - min_mag + 1e-8

    for i, mag in enumerate(mags):
        norm = (mag - min_mag) / range_mag
        bar = "#" * int(norm * 40)
        label_str = sample_labels.get(start + i, "")
        print(f"{start+i:>5} |{bar:<40}| {label_str}")


def interactive_label(json_path: Path):
    """
    Interactive labeling session for a single data file.
    """
    # Load data
    print(f"\nLoading: {json_path}")
    data = load_session_data(json_path)
    print(f"Loaded {len(data)} samples ({len(data)/50:.1f} seconds at 50Hz)")

    # Load or create metadata
    meta_path = json_path.with_suffix('.meta.json')
    timestamp = json_path.stem

    if meta_path.exists():
        meta = SessionMetadata.load(str(meta_path))
        print(f"Loaded existing metadata with {len(meta.labels)} labels")
    else:
        meta = SessionMetadata(timestamp=timestamp)
        print("Created new metadata")

    modified = False

    # Interactive loop
    print_help()

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted")
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        try:
            if action == 'h':
                print_help()

            elif action == 'q':
                if modified:
                    confirm = input("Unsaved changes. Quit anyway? [y/N] ")
                    if confirm.lower() != 'y':
                        continue
                print("Exiting without saving")
                break

            elif action == 'w':
                meta.save(str(meta_path))
                print(f"Saved: {meta_path}")
                modified = False

            elif action == 'gestures':
                print("Available gestures:")
                for g in Gesture:
                    print(f"  {g.value}: {g.name.lower()}")

            elif action == 'info':
                print(f"\nSession: {meta.timestamp}")
                print(f"  Subject: {meta.subject_id}")
                print(f"  Environment: {meta.environment}")
                print(f"  Hand: {meta.hand}")
                print(f"  Split: {meta.split}")
                print(f"  Device: {meta.device_id}")
                print(f"  Notes: {meta.session_notes}")
                print(f"  Samples: {len(data)}")
                print(f"  Labels: {len(meta.labels)}")

            elif action == 'ls':
                if not meta.labels:
                    print("No labels defined")
                else:
                    print("\nLabels:")
                    for i, seg in enumerate(meta.labels):
                        duration = (seg.end_sample - seg.start_sample) / 50
                        print(f"  [{i}] {seg.start_sample:>5}-{seg.end_sample:<5} "
                              f"({duration:.1f}s) {seg.gesture.name:<12} "
                              f"[{seg.confidence}] {seg.notes}")

            elif action == 'l':
                if len(parts) < 4:
                    print("Usage: l <start> <end> <gesture> [confidence] [notes]")
                    continue
                start = int(parts[1])
                end = int(parts[2])
                gesture_name = parts[3]
                confidence = parts[4] if len(parts) > 4 else "high"
                notes = " ".join(parts[5:]) if len(parts) > 5 else ""

                if start < 0 or end > len(data) or start >= end:
                    print(f"Invalid range. Data has {len(data)} samples.")
                    continue

                try:
                    gesture = Gesture.from_name(gesture_name)
                except KeyError:
                    print(f"Unknown gesture: {gesture_name}")
                    print("Available: " + ", ".join(Gesture.names()))
                    continue

                seg = LabeledSegment(
                    start_sample=start,
                    end_sample=end,
                    gesture=gesture,
                    confidence=confidence,
                    notes=notes
                )
                meta.labels.append(seg)
                modified = True
                print(f"Added: {start}-{end} as {gesture.name}")

            elif action == 'p':
                start = int(parts[1]) if len(parts) > 1 else 0
                end = int(parts[2]) if len(parts) > 2 else start + 50
                visualize_segment(data, start, end, meta.labels)

            elif action == 'd':
                if len(parts) < 2:
                    print("Usage: d <index>")
                    continue
                idx = int(parts[1])
                if 0 <= idx < len(meta.labels):
                    removed = meta.labels.pop(idx)
                    modified = True
                    print(f"Removed label: {removed.gesture.name}")
                else:
                    print(f"Invalid index. Have {len(meta.labels)} labels.")

            elif action == 'm':
                if len(parts) < 3:
                    print("Usage: m <key> <value>")
                    print("Keys: subject_id, environment, hand, split, device_id, session_notes")
                    continue
                key = parts[1]
                value = " ".join(parts[2:])
                if hasattr(meta, key):
                    setattr(meta, key, value)
                    modified = True
                    print(f"Set {key} = {value}")
                else:
                    print(f"Unknown key: {key}")

            else:
                print(f"Unknown command: {action}. Type 'h' for help.")

        except (ValueError, IndexError) as e:
            print(f"Error: {e}")


def batch_info(data_dir: Path):
    """Show info about all sessions in directory."""
    print(f"\nSessions in {data_dir}:\n")
    print(f"{'Filename':<40} {'Samples':>8} {'Labels':>8} {'Split':>10}")
    print("-" * 70)

    for json_path in sorted(data_dir.glob('*.json')):
        if json_path.name.endswith('.meta.json'):
            continue

        data = load_session_data(json_path)
        meta_path = json_path.with_suffix('.meta.json')

        num_labels = 0
        split = "-"
        if meta_path.exists():
            meta = SessionMetadata.load(str(meta_path))
            num_labels = len(meta.labels)
            split = meta.split

        print(f"{json_path.name:<40} {len(data):>8} {num_labels:>8} {split:>10}")


def main():
    parser = argparse.ArgumentParser(description='SIMCAP Data Labeling Tool')
    parser.add_argument(
        'path', type=str, nargs='?',
        help='Path to JSON data file or data directory'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Show info about all sessions in directory'
    )
    args = parser.parse_args()

    if args.path is None:
        args.path = 'data/GAMBIT'

    path = Path(args.path)

    if path.is_dir():
        if args.info:
            batch_info(path)
        else:
            print(f"Directory specified. Use --info to list sessions, or specify a .json file.")
            batch_info(path)
    elif path.is_file() and path.suffix == '.json':
        interactive_label(path)
    else:
        print(f"Error: {path} is not a valid JSON file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
