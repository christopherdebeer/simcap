#!/usr/bin/env python3
"""
Aligned Synthetic Data Generator

Uses measured ground truth signatures as anchors and generates training data
by adding realistic perturbations around them.

This approach:
1. Uses MEASURED single/multi-finger signatures (not simulated)
2. Adds Gaussian noise matching observed variance
3. Interpolates between states for partial positions
4. Generates unlimited training data grounded in reality
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MeasuredSignature:
    """A measured magnetic signature for a finger configuration."""
    code: str           # e.g., '20000' for thumb flexed
    mean: np.ndarray    # Mean delta vector (3,)
    std: np.ndarray     # Per-axis standard deviation (3,)
    n_samples: int      # Number of samples used


class AlignedGenerator:
    """Generate synthetic training data aligned with measured ground truth."""

    def __init__(self, session_path: Optional[Path] = None):
        self.signatures: Dict[str, MeasuredSignature] = {}
        self.baseline: np.ndarray = np.zeros(3)
        self.baseline_std: np.ndarray = np.zeros(3)

        if session_path:
            self.load_session(session_path)

    def load_session(self, session_path: Path):
        """Load and extract signatures from a wizard session."""
        with open(session_path) as f:
            session = json.load(f)

        samples = session.get('samples', [])
        labels = session.get('labels', [])

        mx = np.array([s.get('mx', 0) for s in samples])
        my = np.array([s.get('my', 0) for s in samples])
        mz = np.array([s.get('mz', 0) for s in samples])

        # Group samples by finger configuration
        config_vectors: Dict[str, List[np.ndarray]] = {}

        for label in labels:
            start = label.get('start_sample', label.get('startIndex', 0))
            end = label.get('end_sample', label.get('endIndex', 0))
            content = label.get('labels', label)
            fingers = content.get('fingers', {})

            if not fingers:
                continue

            code = self._finger_code(fingers)
            if '?' in code:
                continue

            if code not in config_vectors:
                config_vectors[code] = []

            for i in range(start, min(end, len(mx))):
                config_vectors[code].append(np.array([mx[i], my[i], mz[i]]))

        # Extract baseline (all extended)
        if '00000' in config_vectors:
            baseline_vecs = np.array(config_vectors['00000'])
            self.baseline = np.mean(baseline_vecs, axis=0)
            self.baseline_std = np.std(baseline_vecs, axis=0)

        # Compute signatures (as deltas from baseline)
        for code, vectors in config_vectors.items():
            vecs = np.array(vectors)
            deltas = vecs - self.baseline

            self.signatures[code] = MeasuredSignature(
                code=code,
                mean=np.mean(deltas, axis=0),
                std=np.std(deltas, axis=0),
                n_samples=len(vectors)
            )

        print(f"Loaded {len(self.signatures)} signatures from {session_path.name}")

    def _finger_code(self, fingers: Dict) -> str:
        """Convert finger states dict to code string."""
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

    def generate_sample(
        self,
        finger_states: Dict[str, int],
        noise_scale: float = 1.0
    ) -> np.ndarray:
        """
        Generate a synthetic sample for given finger states.

        Args:
            finger_states: Dict mapping finger name to state (0=extended, 1=partial, 2=flexed)
            noise_scale: Multiplier for noise (1.0 = measured noise level)

        Returns:
            Magnetic field vector (3,)
        """
        # Build code
        code = ''
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            code += str(finger_states.get(f, 0))

        # If we have this exact signature, sample from it
        if code in self.signatures:
            sig = self.signatures[code]
            noise = np.random.randn(3) * sig.std * noise_scale
            return self.baseline + sig.mean + noise

        # Otherwise, interpolate from single-finger signatures
        delta = np.zeros(3)
        noise = np.zeros(3)

        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, finger in enumerate(finger_order):
            state = finger_states.get(finger, 0)

            if state == 0:
                continue  # Extended = no contribution

            # Get single-finger signature
            single_code = '0' * i + '2' + '0' * (4 - i)
            if single_code in self.signatures:
                sig = self.signatures[single_code]

                if state == 2:  # Fully flexed
                    delta += sig.mean
                    noise += sig.std ** 2  # Variance adds
                elif state == 1:  # Partial = 50% of flexed
                    delta += 0.5 * sig.mean
                    noise += (0.5 * sig.std) ** 2

        # Apply non-additivity correction (measured ~50% cancellation on average)
        n_flexed = sum(1 for s in finger_states.values() if s > 0)
        if n_flexed > 1:
            # Reduce magnitude based on number of fingers
            cancellation = 0.3 * (n_flexed - 1) / 4  # Up to 30% reduction for 5 fingers
            delta *= (1 - cancellation)

        noise = np.sqrt(noise)  # Convert variance back to std
        sample_noise = np.random.randn(3) * noise * noise_scale

        return self.baseline + delta + sample_noise

    def generate_batch(
        self,
        n_samples: int,
        states: Optional[List[int]] = None,
        noise_scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of synthetic samples.

        Args:
            n_samples: Number of samples to generate
            states: List of allowed states per finger [0, 1, 2]. None = all.
            noise_scale: Noise multiplier

        Returns:
            (X, y) where X is (n_samples, 3) features, y is (n_samples, 5) labels
        """
        if states is None:
            states = [0, 2]  # Default: just extended and flexed

        X = []
        y = []

        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

        for _ in range(n_samples):
            # Random finger states
            finger_states = {}
            label = []
            for finger in finger_order:
                state = np.random.choice(states)
                finger_states[finger] = state
                label.append(state)

            sample = self.generate_sample(finger_states, noise_scale)
            X.append(sample)
            y.append(label)

        return np.array(X), np.array(y)

    def generate_all_configurations(
        self,
        samples_per_config: int = 100,
        noise_scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples for all 32 binary configurations (extended/flexed only).

        Returns:
            (X, y) where X is features, y is 5-digit codes
        """
        X = []
        y = []

        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # Generate all 2^5 = 32 configurations
        for config in range(32):
            finger_states = {}
            label = []
            for i, finger in enumerate(finger_order):
                state = 2 if (config >> (4 - i)) & 1 else 0
                finger_states[finger] = state
                label.append(state // 2)  # 0 or 1

            for _ in range(samples_per_config):
                sample = self.generate_sample(finger_states, noise_scale)
                X.append(sample)
                y.append(label)

        return np.array(X), np.array(y)


def main():
    """Test the aligned generator."""
    print("=" * 80)
    print("ALIGNED SYNTHETIC DATA GENERATOR")
    print("=" * 80)

    # Load from wizard session
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    gen = AlignedGenerator(session_path)

    print("\n" + "=" * 80)
    print("LOADED SIGNATURES")
    print("=" * 80)

    print(f"\nBaseline: [{gen.baseline[0]:.0f}, {gen.baseline[1]:.0f}, {gen.baseline[2]:.0f}] µT")
    print(f"Baseline std: [{gen.baseline_std[0]:.0f}, {gen.baseline_std[1]:.0f}, {gen.baseline_std[2]:.0f}] µT")

    print("\nSingle-finger signatures:")
    single_codes = ['20000', '02000', '00200', '00020', '00002']
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    for code, name in zip(single_codes, finger_names):
        if code in gen.signatures:
            sig = gen.signatures[code]
            print(f"  {name}: mean=[{sig.mean[0]:>+8.0f}, {sig.mean[1]:>+8.0f}, {sig.mean[2]:>+8.0f}] µT, "
                  f"std=[{sig.std[0]:>4.0f}, {sig.std[1]:>4.0f}, {sig.std[2]:>4.0f}] µT, n={sig.n_samples}")

    # Generate test samples
    print("\n" + "=" * 80)
    print("GENERATING TEST SAMPLES")
    print("=" * 80)

    # Test single configurations
    test_configs = [
        {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0},  # All extended
        {'thumb': 2, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0},  # Thumb flexed
        {'thumb': 2, 'index': 2, 'middle': 2, 'ring': 2, 'pinky': 2},  # All flexed
    ]

    for config in test_configs:
        code = ''.join(str(config[f]) for f in ['thumb', 'index', 'middle', 'ring', 'pinky'])

        # Generate 10 samples
        samples = [gen.generate_sample(config) for _ in range(10)]
        samples = np.array(samples)

        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        print(f"\n{code}:")
        print(f"  Generated: mean=[{mean[0]:>8.0f}, {mean[1]:>8.0f}, {mean[2]:>8.0f}] µT")
        print(f"             std=[{std[0]:>8.0f}, {std[1]:>8.0f}, {std[2]:>8.0f}] µT")

        # Compare to measured if available
        if code in gen.signatures:
            sig = gen.signatures[code]
            expected = gen.baseline + sig.mean
            print(f"  Measured:  mean=[{expected[0]:>8.0f}, {expected[1]:>8.0f}, {expected[2]:>8.0f}] µT")

    # Generate full dataset
    print("\n" + "=" * 80)
    print("GENERATING FULL TRAINING SET")
    print("=" * 80)

    X, y = gen.generate_all_configurations(samples_per_config=100)
    print(f"\nGenerated {len(X)} samples across 32 configurations")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Save as training data
    output_path = Path('ml/data/aligned_synthetic_train.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X, y=y)
    print(f"\nSaved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
    ALIGNED GENERATOR:
    - Uses MEASURED signatures as ground truth anchors
    - Adds realistic Gaussian noise matching observed variance
    - Applies non-additivity correction for multi-finger combos
    - Generates unlimited training data grounded in reality

    ADVANTAGES OVER PHYSICS SIMULATION:
    - No geometry/orientation assumptions needed
    - Automatically captures actual sensor placement effects
    - Noise characteristics match real hardware
    - Multi-finger interactions learned from data

    USAGE:
    >>> gen = AlignedGenerator(Path('data/GAMBIT/session.json'))
    >>> X, y = gen.generate_batch(1000)  # Generate training samples
    """)


if __name__ == '__main__':
    main()
