#!/usr/bin/env python3
"""
Analyze magnet orientations from vector signatures.

Given: Alternating orientation with thumb and index the same.
Pattern: Thumb=Index=A, Middle=B, Ring=A, Pinky=B (or inverse)

This explains non-additivity: opposite polarities cancel partially.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_session(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def finger_code(fingers: dict) -> str:
    code = ''
    for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        state = fingers.get(f, 'unknown')
        if state == 'extended':
            code += '0'
        elif state == 'partial':
            code += '1'
        elif state == 'flexed':
            code += '2'
        else:
            code += '?'
    return code


def main():
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    
    print("=" * 80)
    print("MAGNET ORIENTATION ANALYSIS")
    print("=" * 80)
    print("\nKnown: Alternating orientation, Thumb=Index same orientation")
    print("Pattern: T=I=A, M=B, R=A, P=B  (or inverse)")
    
    session = load_session(session_path)
    samples = session.get('samples', [])
    labels = session.get('labels', [])
    
    # Extract magnetometer vectors
    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])
    
    # Group by configuration
    config_vectors = defaultdict(lambda: {'mx': [], 'my': [], 'mz': []})
    
    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})
        
        if not fingers:
            continue
            
        code = finger_code(fingers)
        if '?' in code:
            continue
            
        for i in range(start, min(end, len(mx))):
            config_vectors[code]['mx'].append(mx[i])
            config_vectors[code]['my'].append(my[i])
            config_vectors[code]['mz'].append(mz[i])
    
    # Calculate mean vectors
    mean_vectors = {}
    for code, data in config_vectors.items():
        mean_vectors[code] = np.array([
            np.mean(data['mx']),
            np.mean(data['my']),
            np.mean(data['mz'])
        ])
    
    # Get baseline
    baseline = mean_vectors.get('00000', np.zeros(3))
    
    print("\n" + "=" * 80)
    print("1. SINGLE FINGER DELTA VECTORS (relative to baseline)")
    print("=" * 80)
    
    single_codes = {
        '20000': 'Thumb',
        '02000': 'Index', 
        '00200': 'Middle',
        '00020': 'Ring',
        '00002': 'Pinky'
    }
    
    finger_deltas = {}
    
    print(f"\nBaseline (00000): [{baseline[0]:>8.0f}, {baseline[1]:>8.0f}, {baseline[2]:>8.0f}] µT")
    print("-" * 70)
    
    for code, name in single_codes.items():
        if code in mean_vectors:
            delta = mean_vectors[code] - baseline
            mag = np.linalg.norm(delta)
            finger_deltas[name] = delta
            
            # Normalize to get direction
            direction = delta / mag if mag > 0 else delta
            
            print(f"{name:8s}: Δ=[{delta[0]:>+8.0f}, {delta[1]:>+8.0f}, {delta[2]:>+8.0f}] µT  |Δ|={mag:>8.0f} µT")
    
    # 2. Check orientation pattern
    print("\n" + "=" * 80)
    print("2. ORIENTATION PATTERN ANALYSIS")
    print("=" * 80)
    
    print("\nDot products between finger deltas (positive = same direction):")
    print("-" * 70)
    
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    # Compute normalized directions
    directions = {}
    for name, delta in finger_deltas.items():
        mag = np.linalg.norm(delta)
        directions[name] = delta / mag if mag > 0 else delta
    
    # Dot product matrix
    print("\n         ", end="")
    for f in fingers:
        print(f"{f:>8s}", end="")
    print()
    
    for f1 in fingers:
        print(f"{f1:8s}", end="")
        for f2 in fingers:
            if f1 in directions and f2 in directions:
                dot = np.dot(directions[f1], directions[f2])
                print(f"{dot:>8.2f}", end="")
            else:
                print(f"{'N/A':>8s}", end="")
        print()
    
    # Interpret pattern
    print("\n" + "-" * 70)
    print("INTERPRETATION (expected: T=I=R same, M=P opposite):")
    
    # Check expected pairs
    pairs_same = [('Thumb', 'Index'), ('Thumb', 'Ring'), ('Index', 'Ring')]
    pairs_opposite = [('Thumb', 'Middle'), ('Thumb', 'Pinky'), 
                      ('Index', 'Middle'), ('Index', 'Pinky'),
                      ('Ring', 'Middle'), ('Ring', 'Pinky')]
    
    print("\nExpected SAME orientation (dot ≈ +1):")
    for f1, f2 in pairs_same:
        if f1 in directions and f2 in directions:
            dot = np.dot(directions[f1], directions[f2])
            match = "✓" if dot > 0.5 else "✗"
            print(f"  {f1:8s} · {f2:8s} = {dot:>+6.2f}  {match}")
    
    print("\nExpected OPPOSITE orientation (dot ≈ -1):")
    for f1, f2 in pairs_opposite:
        if f1 in directions and f2 in directions:
            dot = np.dot(directions[f1], directions[f2])
            match = "✓" if dot < -0.5 else "✗"
            print(f"  {f1:8s} · {f2:8s} = {dot:>+6.2f}  {match}")
    
    # 3. Non-additivity explained
    print("\n" + "=" * 80)
    print("3. NON-ADDITIVITY EXPLAINED BY ORIENTATION")
    print("=" * 80)
    
    # Check some multi-finger combinations
    combos = [
        ('22000', 'Thumb+Index', ['Thumb', 'Index']),
        ('00022', 'Ring+Pinky', ['Ring', 'Pinky']),
        ('00222', 'Mid+Ring+Pinky', ['Middle', 'Ring', 'Pinky']),
        ('22222', 'All flexed', ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']),
    ]
    
    print("\nPredicted (sum of singles) vs Measured:")
    print("-" * 70)
    
    for code, name, fingers_list in combos:
        if code not in mean_vectors:
            continue
            
        # Sum of individual deltas
        predicted_delta = np.sum([finger_deltas[f] for f in fingers_list if f in finger_deltas], axis=0)
        predicted_mag = np.linalg.norm(baseline + predicted_delta)
        
        # Actual measured
        actual_mag = np.linalg.norm(mean_vectors[code])
        
        # Delta from measured
        measured_delta = mean_vectors[code] - baseline
        
        print(f"\n{name} ({code}):")
        print(f"  Predicted sum: [{predicted_delta[0]:>+8.0f}, {predicted_delta[1]:>+8.0f}, {predicted_delta[2]:>+8.0f}] µT")
        print(f"  Measured Δ:    [{measured_delta[0]:>+8.0f}, {measured_delta[1]:>+8.0f}, {measured_delta[2]:>+8.0f}] µT")
        print(f"  |Predicted|: {np.linalg.norm(predicted_delta):>8.0f} µT")
        print(f"  |Measured|:  {np.linalg.norm(measured_delta):>8.0f} µT")
        
        if np.linalg.norm(predicted_delta) > 0:
            ratio = np.linalg.norm(measured_delta) / np.linalg.norm(predicted_delta)
            if ratio < 1:
                print(f"  → {(1-ratio)*100:.0f}% CANCELLATION (opposite orientations)")
            else:
                print(f"  → {(ratio-1)*100:.0f}% AMPLIFICATION (same orientations)")
    
    # 4. Per-axis analysis
    print("\n" + "=" * 80)
    print("4. PER-AXIS SIGNATURE ANALYSIS")
    print("=" * 80)
    
    print("\nWhich axis shows strongest signal per finger:")
    print("-" * 70)
    
    for name, delta in finger_deltas.items():
        abs_delta = np.abs(delta)
        max_axis = ['X', 'Y', 'Z'][np.argmax(abs_delta)]
        sign = '+' if delta[np.argmax(abs_delta)] > 0 else '-'
        print(f"  {name:8s}: Primary axis = {sign}{max_axis} ({abs_delta[np.argmax(abs_delta)]:.0f} µT)")
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate average same/opposite dot products
    same_dots = []
    opposite_dots = []
    
    for f1, f2 in pairs_same:
        if f1 in directions and f2 in directions:
            same_dots.append(np.dot(directions[f1], directions[f2]))
    
    for f1, f2 in pairs_opposite:
        if f1 in directions and f2 in directions:
            opposite_dots.append(np.dot(directions[f1], directions[f2]))
    
    avg_same = np.mean(same_dots) if same_dots else 0
    avg_opposite = np.mean(opposite_dots) if opposite_dots else 0
    
    print(f"""
    ALTERNATING ORIENTATION PATTERN:
    - Expected: Thumb=Index=Ring (group A), Middle=Pinky (group B)
    - Group A should have positive dot products with each other
    - Group A·B should have negative dot products
    
    MEASURED:
    - Average dot product (same group): {avg_same:+.2f}
    - Average dot product (opposite group): {avg_opposite:+.2f}
    
    IMPLICATIONS:
    - Non-additivity is EXPECTED due to field cancellation
    - Multi-finger combinations need measured signatures (not summed singles)
    - Classifier should use full vector, not just magnitude
    """)


if __name__ == '__main__':
    main()
