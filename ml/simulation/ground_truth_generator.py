"""
Ground Truth Signature-Based Synthetic Data Generator

Uses measured magnetic signatures from wizard-labeled sessions to generate
realistic training data. Key advantages over physics-based simulation:

1. NON-ADDITIVITY: Multi-finger combinations are measured directly, not computed
2. REALISTIC NOISE: Noise models derived from actual sensor measurements
3. ANCHORED: All synthetic data is anchored to real measurements

The approach:
- Load ground truth signatures from labeled wizard sessions
- For each finger configuration, sample from measured distribution
- Add realistic noise matching observed variance
- Optionally interpolate between configurations for data augmentation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import random


@dataclass
class FingerSignature:
    """Magnetic signature for a finger configuration."""
    code: str  # e.g., "00000", "20000", "22222"
    mean: np.ndarray  # Mean field [mx, my, mz] in µT
    std: np.ndarray   # Standard deviation per axis
    n_samples: int    # Number of samples this is based on

    # Optional: full distribution for better sampling
    samples: Optional[np.ndarray] = None


@dataclass
class SignatureDatabase:
    """Database of ground truth finger signatures."""
    signatures: Dict[str, FingerSignature] = field(default_factory=dict)
    baseline_code: str = "00000"
    source_session: Optional[str] = None

    def add_signature(self, code: str, mean: np.ndarray, std: np.ndarray,
                      n_samples: int, samples: Optional[np.ndarray] = None):
        """Add a signature to the database."""
        self.signatures[code] = FingerSignature(
            code=code, mean=mean, std=std,
            n_samples=n_samples, samples=samples
        )

    def get_baseline(self) -> np.ndarray:
        """Get baseline field (all fingers extended)."""
        if self.baseline_code in self.signatures:
            return self.signatures[self.baseline_code].mean
        return np.zeros(3)

    def get_delta(self, code: str) -> np.ndarray:
        """Get field delta from baseline for a configuration."""
        if code not in self.signatures:
            return np.zeros(3)
        return self.signatures[code].mean - self.get_baseline()

    def sample(self, code: str, n: int = 1) -> np.ndarray:
        """Sample from signature distribution."""
        if code not in self.signatures:
            # Unknown configuration - return baseline with extra noise
            baseline = self.get_baseline()
            return baseline + np.random.randn(n, 3) * 100  # Higher uncertainty

        sig = self.signatures[code]

        if sig.samples is not None and len(sig.samples) > 10:
            # Sample from actual measurements with replacement
            indices = np.random.randint(0, len(sig.samples), size=n)
            return sig.samples[indices]
        else:
            # Sample from Gaussian with measured mean/std
            return np.random.randn(n, 3) * sig.std + sig.mean


def load_signatures_from_session(session_path: Path) -> SignatureDatabase:
    """
    Load ground truth signatures from a wizard-labeled session.

    Args:
        session_path: Path to session JSON file

    Returns:
        SignatureDatabase with measured signatures
    """
    with open(session_path) as f:
        session = json.load(f)

    samples = session.get('samples', [])
    labels = session.get('labels', [])

    # Extract magnetometer data
    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])

    # Build signature database
    db = SignatureDatabase(source_session=session_path.name)

    # Group samples by finger code
    from collections import defaultdict
    code_samples = defaultdict(list)

    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})

        # Convert to code
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

        if '?' in code or end <= start:
            continue

        # Collect samples for this code
        for i in range(start, min(end, len(mx))):
            code_samples[code].append([mx[i], my[i], mz[i]])

    # Compute statistics for each code
    for code, samps in code_samples.items():
        samps = np.array(samps)
        db.add_signature(
            code=code,
            mean=np.mean(samps, axis=0),
            std=np.std(samps, axis=0),
            n_samples=len(samps),
            samples=samps
        )

    print(f"Loaded {len(db.signatures)} signatures from {session_path.name}")
    for code, sig in sorted(db.signatures.items()):
        print(f"  {code}: n={sig.n_samples}, mean=[{sig.mean[0]:.0f}, {sig.mean[1]:.0f}, {sig.mean[2]:.0f}]")

    return db


def load_signatures_from_json(json_path: Path) -> SignatureDatabase:
    """
    Load signatures from pre-computed JSON file.
    """
    with open(json_path) as f:
        data = json.load(f)

    db = SignatureDatabase()

    # Load baseline
    if 'baseline' in data:
        baseline = np.array(data['baseline']['mean'])
        db.add_signature('00000', baseline, np.array([100, 100, 100]), 1)

    # Load all signatures
    if 'all_signatures' in data:
        for code, sig_vec in data['all_signatures'].items():
            # Signatures are stored as deltas, convert to absolute
            if 'baseline' in data:
                mean = np.array(sig_vec) + np.array(data['baseline']['mean'])
            else:
                mean = np.array(sig_vec)

            db.add_signature(
                code=code,
                mean=mean,
                std=np.array([1000, 1000, 2000]),  # Approximate from observations
                n_samples=100
            )

    return db


class GroundTruthGenerator:
    """
    Generate synthetic training data anchored to ground truth signatures.
    """

    def __init__(
        self,
        signature_db: SignatureDatabase,
        sample_rate: float = 26.0,
        noise_scale: float = 1.0
    ):
        """
        Initialize generator.

        Args:
            signature_db: Database of ground truth signatures
            sample_rate: Target sample rate in Hz
            noise_scale: Scale factor for noise (1.0 = match measured noise)
        """
        self.db = signature_db
        self.sample_rate = sample_rate
        self.noise_scale = noise_scale

        # Available configurations
        self.available_codes = list(signature_db.signatures.keys())
        print(f"Generator initialized with {len(self.available_codes)} configurations")

    def generate_segment(
        self,
        code: str,
        duration_sec: float = 2.0,
        add_drift: bool = True,
        add_motion_noise: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate a segment of samples for a specific finger configuration.

        Args:
            code: Finger configuration code (e.g., "22000")
            duration_sec: Duration in seconds
            add_drift: Add slow temporal drift
            add_motion_noise: Add noise correlated with simulated motion

        Returns:
            Dict with 'mx', 'my', 'mz', 'ax', 'ay', 'az' arrays
        """
        n_samples = int(duration_sec * self.sample_rate)

        # Sample from signature distribution
        mag_samples = self.db.sample(code, n_samples)

        # Add temporal correlation (low-pass filter the noise)
        if add_drift:
            drift = np.cumsum(np.random.randn(n_samples, 3) * 0.1, axis=0)
            drift = drift - drift.mean(axis=0)  # Zero-mean drift
            mag_samples = mag_samples + drift * self.noise_scale

        # Add motion-correlated noise
        if add_motion_noise:
            # Simulate IMU motion
            motion_scale = np.random.uniform(0.5, 2.0)
            motion = np.sin(np.linspace(0, 4 * np.pi, n_samples)[:, np.newaxis] +
                          np.random.randn(3))
            mag_samples = mag_samples + motion * 50 * motion_scale * self.noise_scale

        # Generate synthetic accelerometer (gravity + small motion)
        ax = np.random.randn(n_samples) * 50 + 0
        ay = np.random.randn(n_samples) * 50 + 0
        az = np.random.randn(n_samples) * 50 + 8192  # Gravity

        return {
            'mx': mag_samples[:, 0],
            'my': mag_samples[:, 1],
            'mz': mag_samples[:, 2],
            'ax': ax,
            'ay': ay,
            'az': az
        }

    def generate_transition(
        self,
        code_from: str,
        code_to: str,
        duration_sec: float = 0.5
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Generate a transition between two configurations.

        Returns:
            Tuple of (samples dict, interpolation weights)
        """
        n_samples = int(duration_sec * self.sample_rate)

        # Sigmoid interpolation for smooth transition
        t = np.linspace(-3, 3, n_samples)
        weights = 1 / (1 + np.exp(-t))

        # Sample from both distributions
        samples_from = self.db.sample(code_from, n_samples)
        samples_to = self.db.sample(code_to, n_samples)

        # Interpolate
        mag_samples = samples_from * (1 - weights[:, np.newaxis]) + \
                      samples_to * weights[:, np.newaxis]

        # Add transition noise
        transition_noise = np.random.randn(n_samples, 3) * 200 * np.sin(np.pi * weights)[:, np.newaxis]
        mag_samples = mag_samples + transition_noise

        ax = np.random.randn(n_samples) * 100  # More motion during transition
        ay = np.random.randn(n_samples) * 100
        az = np.random.randn(n_samples) * 100 + 8192

        return {
            'mx': mag_samples[:, 0],
            'my': mag_samples[:, 1],
            'mz': mag_samples[:, 2],
            'ax': ax,
            'ay': ay,
            'az': az
        }, weights

    def generate_session(
        self,
        n_segments: int = 20,
        segment_duration: float = 2.0,
        include_transitions: bool = True,
        codes: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate a complete synthetic session.

        Args:
            n_segments: Number of pose segments
            segment_duration: Duration per segment in seconds
            include_transitions: Add transitions between poses
            codes: List of codes to use (random if None)

        Returns:
            Session dict in SIMCAP format
        """
        if codes is None:
            codes = [random.choice(self.available_codes) for _ in range(n_segments)]

        samples = []
        labels = []
        current_sample = 0

        for i, code in enumerate(codes):
            # Generate segment
            segment = self.generate_segment(code, segment_duration)
            n = len(segment['mx'])

            # Create sample dicts
            for j in range(n):
                samples.append({
                    'mx': float(segment['mx'][j]),
                    'my': float(segment['my'][j]),
                    'mz': float(segment['mz'][j]),
                    'ax': float(segment['ax'][j]),
                    'ay': float(segment['ay'][j]),
                    'az': float(segment['az'][j]),
                    'timestamp': current_sample + j
                })

            # Create label
            fingers = {
                'thumb': 'extended' if code[0] == '0' else ('partial' if code[0] == '1' else 'flexed'),
                'index': 'extended' if code[1] == '0' else ('partial' if code[1] == '1' else 'flexed'),
                'middle': 'extended' if code[2] == '0' else ('partial' if code[2] == '1' else 'flexed'),
                'ring': 'extended' if code[3] == '0' else ('partial' if code[3] == '1' else 'flexed'),
                'pinky': 'extended' if code[4] == '0' else ('partial' if code[4] == '1' else 'flexed'),
            }

            labels.append({
                'start_sample': current_sample,
                'end_sample': current_sample + n,
                'labels': {
                    'fingers': fingers,
                    'calibration': 'none',
                    'motion': 'static'
                }
            })

            current_sample += n

            # Add transition to next pose
            if include_transitions and i < len(codes) - 1:
                next_code = codes[i + 1]
                trans, _ = self.generate_transition(code, next_code, 0.3)
                n_trans = len(trans['mx'])

                for j in range(n_trans):
                    samples.append({
                        'mx': float(trans['mx'][j]),
                        'my': float(trans['my'][j]),
                        'mz': float(trans['mz'][j]),
                        'ax': float(trans['ax'][j]),
                        'ay': float(trans['ay'][j]),
                        'az': float(trans['az'][j]),
                        'timestamp': current_sample + j
                    })

                # Transition label (mark as unlabeled or motion)
                labels.append({
                    'start_sample': current_sample,
                    'end_sample': current_sample + n_trans,
                    'labels': {
                        'fingers': {},
                        'calibration': 'none',
                        'motion': 'moving'
                    }
                })

                current_sample += n_trans

        return {
            'version': '2.1',
            'metadata': {
                'sample_rate': self.sample_rate,
                'device': 'SYNTHETIC',
                'session_type': 'synthetic',
                'generator': 'ground_truth_signature',
                'source_signatures': self.db.source_session,
                'generated_at': datetime.now().isoformat(),
                'n_configurations': len(self.available_codes)
            },
            'samples': samples,
            'labels': labels
        }

    def generate_balanced_dataset(
        self,
        samples_per_class: int = 500,
        segment_duration: float = 2.0
    ) -> List[Dict]:
        """
        Generate a balanced dataset with equal samples per configuration.

        Args:
            samples_per_class: Target samples per finger configuration
            segment_duration: Duration per segment

        Returns:
            List of session dicts
        """
        sessions = []

        samples_per_segment = int(segment_duration * self.sample_rate)
        segments_per_class = max(1, samples_per_class // samples_per_segment)

        for code in self.available_codes:
            session = self.generate_session(
                n_segments=segments_per_class,
                segment_duration=segment_duration,
                include_transitions=False,
                codes=[code] * segments_per_class
            )
            session['metadata']['target_class'] = code
            sessions.append(session)

        print(f"Generated {len(sessions)} sessions with ~{samples_per_class} samples per class")
        return sessions


def main():
    """Generate synthetic training data from ground truth signatures."""

    print("=" * 80)
    print("GROUND TRUTH SIGNATURE-BASED SYNTHETIC DATA GENERATOR")
    print("=" * 80)

    # Load signatures from wizard session
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')

    if session_path.exists():
        print(f"\nLoading signatures from: {session_path}")
        db = load_signatures_from_session(session_path)
    else:
        # Try loading from pre-computed JSON
        json_path = Path('ml/finger_magnet_signatures.json')
        if json_path.exists():
            print(f"\nLoading signatures from: {json_path}")
            db = load_signatures_from_json(json_path)
        else:
            print("No signature source found!")
            return

    # Create generator
    generator = GroundTruthGenerator(db, sample_rate=26.0)

    # Generate a sample session
    print("\n" + "=" * 80)
    print("GENERATING SAMPLE SESSION")
    print("=" * 80)

    session = generator.generate_session(n_segments=10, segment_duration=2.0)

    print(f"\nGenerated session:")
    print(f"  Samples: {len(session['samples'])}")
    print(f"  Labels: {len(session['labels'])}")
    print(f"  Duration: {len(session['samples']) / 26:.1f} seconds")

    # Save sample session
    output_path = Path('ml/synthetic_ground_truth_sample.json')
    with open(output_path, 'w') as f:
        json.dump(session, f, indent=2)
    print(f"\nSaved sample session to: {output_path}")

    # Generate balanced dataset
    print("\n" + "=" * 80)
    print("GENERATING BALANCED DATASET")
    print("=" * 80)

    sessions = generator.generate_balanced_dataset(samples_per_class=500)

    # Combine into single dataset
    all_samples = []
    all_labels = []
    sample_offset = 0

    for sess in sessions:
        for sample in sess['samples']:
            sample['timestamp'] += sample_offset
            all_samples.append(sample)

        for label in sess['labels']:
            label['start_sample'] += sample_offset
            label['end_sample'] += sample_offset
            all_labels.append(label)

        sample_offset = len(all_samples)

    combined = {
        'version': '2.1',
        'metadata': {
            'sample_rate': 26.0,
            'device': 'SYNTHETIC',
            'session_type': 'synthetic_balanced',
            'generator': 'ground_truth_signature',
            'n_classes': len(db.signatures),
            'samples_per_class': 500,
            'generated_at': datetime.now().isoformat()
        },
        'samples': all_samples,
        'labels': all_labels
    }

    output_path = Path('ml/synthetic_balanced_dataset.json')
    with open(output_path, 'w') as f:
        json.dump(combined, f)
    print(f"\nSaved balanced dataset to: {output_path}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Total labels: {len(all_labels)}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    from collections import Counter
    code_counts = Counter()

    for label in all_labels:
        fingers = label['labels'].get('fingers', {})
        if fingers:
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

            n_samples = label['end_sample'] - label['start_sample']
            code_counts[code] += n_samples

    print("\nSamples per configuration:")
    for code, count in sorted(code_counts.items()):
        print(f"  {code}: {count} samples")


if __name__ == '__main__':
    main()
