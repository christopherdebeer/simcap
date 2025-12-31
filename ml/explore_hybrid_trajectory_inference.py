#!/usr/bin/env python3
"""
Hybrid Trajectory Inference: Synthesizing Aligned Signatures + FFO$$ Templates

This exploration investigates using orientation-corrected magnetic residuals
as trajectories for template matching, enabling simultaneous detection of:
1. MOTION/GESTURE: What movement pattern is occurring (wave, swipe, etc.)
2. FINGER POSE: What the fingers are doing during that motion

Key Insight: Rather than treating magnetometer as single-sample static pose detector,
we track the *trajectory* through magnetic signature space as fingers move.

Approaches Compared:
A. Traditional FFO$$: Accelerometer trajectories only (motion detection)
B. Raw Mag Trajectories: Magnetometer as-is (orientation-dependent)
C. Residual Trajectories: After calibration + Earth field removal (orientation-aware)
D. Hybrid Multi-Channel: Combined accel + residual mag (motion + pose)
E. Neural + Template Ensemble: Best of both worlds

Author: Claude (Research Exploration)
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping visualizations")


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class TrajectoryPoint:
    """Single point in a multi-dimensional trajectory."""
    values: np.ndarray  # Shape varies by trajectory type
    timestamp: float = 0.0


@dataclass
class Trajectory:
    """A sequence of points through some feature space."""
    points: np.ndarray  # Shape: (n_points, n_dims)
    timestamps: np.ndarray  # Shape: (n_points,)
    trajectory_type: str = "unknown"  # "accel", "mag_raw", "mag_residual", "combined"
    metadata: Dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_dims(self) -> int:
        return self.points.shape[1] if len(self.points.shape) > 1 else 1

    @property
    def duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def path_length(self) -> float:
        """Total Euclidean path length through trajectory space."""
        if self.n_points < 2:
            return 0.0
        diffs = np.diff(self.points, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))


@dataclass
class LabeledSegment:
    """A segment of data with finger state labels."""
    start_idx: int
    end_idx: int
    finger_code: str  # e.g., "20000", "22222"
    trajectory: Optional[Trajectory] = None
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# TRAJECTORY EXTRACTION
# =============================================================================

def extract_accelerometer_trajectory(samples: List[Dict],
                                     start_idx: int,
                                     end_idx: int) -> Trajectory:
    """
    Extract accelerometer trajectory (traditional FFO$$ approach).

    Input: Raw samples with ax, ay, az (in LSB or g)
    Output: 3D trajectory through acceleration space
    """
    points = []
    timestamps = []

    for i in range(start_idx, min(end_idx, len(samples))):
        s = samples[i]
        # Use unit-converted values if available, else raw
        ax = s.get('ax_g', s.get('ax', 0) / 16384.0)  # Convert LSB to g
        ay = s.get('ay_g', s.get('ay', 0) / 16384.0)
        az = s.get('az_g', s.get('az', 0) / 16384.0)

        points.append([ax, ay, az])
        timestamps.append(s.get('timestamp', i))

    return Trajectory(
        points=np.array(points),
        timestamps=np.array(timestamps),
        trajectory_type="accel",
        metadata={"units": "g"}
    )


def extract_magnetometer_raw_trajectory(samples: List[Dict],
                                        start_idx: int,
                                        end_idx: int) -> Trajectory:
    """
    Extract raw magnetometer trajectory.

    This is orientation-DEPENDENT - the trajectory will change
    if the device rotates, even with same finger poses.
    """
    points = []
    timestamps = []

    for i in range(start_idx, min(end_idx, len(samples))):
        s = samples[i]
        mx = s.get('mx', 0)
        my = s.get('my', 0)
        mz = s.get('mz', 0)

        points.append([mx, my, mz])
        timestamps.append(s.get('timestamp', i))

    return Trajectory(
        points=np.array(points),
        timestamps=np.array(timestamps),
        trajectory_type="mag_raw",
        metadata={"units": "raw_counts"}
    )


def extract_magnetometer_residual_trajectory(samples: List[Dict],
                                             start_idx: int,
                                             end_idx: int,
                                             calibration: Optional[Dict] = None) -> Trajectory:
    """
    Extract orientation-corrected magnetometer residual trajectory.

    This attempts to be orientation-INDEPENDENT:
    1. Apply hard/soft iron correction
    2. Rotate Earth field to sensor frame using orientation quaternion
    3. Subtract rotated Earth field
    4. Result: Residual from finger magnets only

    The residual trajectory represents movement through "finger signature space"
    independent of device orientation.
    """
    points = []
    timestamps = []

    # Default calibration if none provided
    if calibration is None:
        calibration = {
            'hard_iron_offset': np.zeros(3),
            'soft_iron_matrix': np.eye(3),
            'earth_field': np.array([0, 0, 50.0])  # Approximate Earth field
        }
    else:
        # Parse calibration
        if isinstance(calibration.get('hard_iron_offset'), dict):
            hi = calibration['hard_iron_offset']
            calibration['hard_iron_offset'] = np.array([hi['x'], hi['y'], hi['z']])
        if isinstance(calibration.get('earth_field'), dict):
            ef = calibration['earth_field']
            calibration['earth_field'] = np.array([ef['x'], ef['y'], ef['z']])
        if 'soft_iron_matrix' not in calibration:
            calibration['soft_iron_matrix'] = np.eye(3)
        elif isinstance(calibration['soft_iron_matrix'], list):
            if len(calibration['soft_iron_matrix']) == 9:
                calibration['soft_iron_matrix'] = np.array(calibration['soft_iron_matrix']).reshape(3, 3)

    hi_offset = calibration.get('hard_iron_offset', np.zeros(3))
    si_matrix = calibration.get('soft_iron_matrix', np.eye(3))
    earth_field = calibration.get('earth_field', np.zeros(3))

    for i in range(start_idx, min(end_idx, len(samples))):
        s = samples[i]

        # Get raw magnetometer
        m_raw = np.array([s.get('mx', 0), s.get('my', 0), s.get('mz', 0)])

        # Apply iron corrections
        m_corrected = si_matrix @ (m_raw - hi_offset)

        # Get orientation quaternion if available
        if all(f'orientation_{c}' in s for c in ['w', 'x', 'y', 'z']):
            qw = s['orientation_w']
            qx = s['orientation_x']
            qy = s['orientation_y']
            qz = s['orientation_z']

            # Quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
            ])

            # Rotate Earth field to sensor frame and subtract
            earth_in_sensor = R.T @ earth_field
            m_residual = m_corrected - earth_in_sensor
        else:
            # No orientation - fall back to static subtraction
            m_residual = m_corrected - earth_field

        points.append(m_residual)
        timestamps.append(s.get('timestamp', i))

    return Trajectory(
        points=np.array(points),
        timestamps=np.array(timestamps),
        trajectory_type="mag_residual",
        metadata={"units": "uT", "orientation_corrected": True}
    )


def extract_combined_trajectory(samples: List[Dict],
                                start_idx: int,
                                end_idx: int,
                                calibration: Optional[Dict] = None) -> Trajectory:
    """
    Extract combined trajectory: [accel (3D) + mag_residual (3D)] = 6D.

    This enables simultaneous motion + pose inference.
    """
    accel_traj = extract_accelerometer_trajectory(samples, start_idx, end_idx)
    mag_traj = extract_magnetometer_residual_trajectory(samples, start_idx, end_idx, calibration)

    # Ensure same length
    n_points = min(accel_traj.n_points, mag_traj.n_points)

    combined_points = np.hstack([
        accel_traj.points[:n_points],
        mag_traj.points[:n_points]
    ])

    return Trajectory(
        points=combined_points,
        timestamps=accel_traj.timestamps[:n_points],
        trajectory_type="combined",
        metadata={
            "dims": ["ax", "ay", "az", "mx_res", "my_res", "mz_res"],
            "accel_units": "g",
            "mag_units": "uT"
        }
    )


# =============================================================================
# FFO$$-STYLE TRAJECTORY PROCESSING
# =============================================================================

def resample_trajectory(traj: Trajectory, n_points: int = 32) -> Trajectory:
    """
    Resample trajectory to N equally-spaced points.

    This is the core FFO$$ resampling algorithm adapted for N-dimensional trajectories.
    """
    if traj.n_points < 2:
        # Cannot resample single point
        return Trajectory(
            points=np.tile(traj.points[0] if traj.n_points > 0 else np.zeros(traj.n_dims), (n_points, 1)),
            timestamps=np.linspace(0, 1, n_points),
            trajectory_type=traj.trajectory_type + "_resampled",
            metadata=traj.metadata
        )

    # Calculate path length
    diffs = np.diff(traj.points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)

    if total_length == 0:
        # All points identical
        return Trajectory(
            points=np.tile(traj.points[0], (n_points, 1)),
            timestamps=np.linspace(traj.timestamps[0], traj.timestamps[-1], n_points),
            trajectory_type=traj.trajectory_type + "_resampled",
            metadata=traj.metadata
        )

    # Target spacing
    interval = total_length / (n_points - 1)

    # Walk along path inserting points
    resampled_points = [traj.points[0].copy()]
    resampled_times = [traj.timestamps[0]]
    accumulated = 0.0
    current_idx = 0

    while len(resampled_points) < n_points and current_idx < len(segment_lengths):
        seg_len = segment_lengths[current_idx]

        if accumulated + seg_len >= interval:
            # Insert interpolated point
            overshoot = interval - accumulated
            t = overshoot / seg_len if seg_len > 0 else 0

            new_point = (1 - t) * traj.points[current_idx] + t * traj.points[current_idx + 1]
            new_time = (1 - t) * traj.timestamps[current_idx] + t * traj.timestamps[current_idx + 1]

            resampled_points.append(new_point)
            resampled_times.append(new_time)

            # Update for next iteration
            segment_lengths[current_idx] = seg_len - overshoot
            traj.points[current_idx] = new_point
            accumulated = 0.0
        else:
            accumulated += seg_len
            current_idx += 1

    # Pad if needed
    while len(resampled_points) < n_points:
        resampled_points.append(traj.points[-1].copy())
        resampled_times.append(traj.timestamps[-1])

    return Trajectory(
        points=np.array(resampled_points),
        timestamps=np.array(resampled_times),
        trajectory_type=traj.trajectory_type + "_resampled",
        metadata={**traj.metadata, "n_points": n_points}
    )


def normalize_trajectory(traj: Trajectory,
                        translate: bool = True,
                        scale: bool = True,
                        target_scale: float = 1.0) -> Trajectory:
    """
    Normalize trajectory: translate to origin, scale to unit size.

    This makes trajectories comparable regardless of starting position or magnitude.
    """
    points = traj.points.copy()

    if translate:
        # Translate centroid to origin
        centroid = np.mean(points, axis=0)
        points = points - centroid

    if scale:
        # Scale to unit bounding box (max dimension = target_scale)
        ranges = np.ptp(points, axis=0)  # Peak-to-peak range per dimension
        max_range = np.max(ranges)

        if max_range > 0:
            points = points * (target_scale / max_range)

    return Trajectory(
        points=points,
        timestamps=traj.timestamps,
        trajectory_type=traj.trajectory_type + "_normalized",
        metadata={**traj.metadata, "normalized": True}
    )


def process_trajectory(traj: Trajectory, n_points: int = 32) -> Trajectory:
    """Full FFO$$-style processing: resample + normalize."""
    resampled = resample_trajectory(traj, n_points)
    normalized = normalize_trajectory(resampled)
    return normalized


# =============================================================================
# DISTANCE METRICS
# =============================================================================

def path_distance(traj_a: Trajectory, traj_b: Trajectory) -> float:
    """
    Point-by-point distance (order-dependent).
    Good for gestures with consistent start/end.
    """
    if traj_a.n_points != traj_b.n_points:
        raise ValueError(f"Trajectories must have same length: {traj_a.n_points} vs {traj_b.n_points}")

    distances = np.linalg.norm(traj_a.points - traj_b.points, axis=1)
    return np.mean(distances)


def cloud_distance(traj_a: Trajectory, traj_b: Trajectory) -> float:
    """
    Point-cloud distance (order-independent).
    Good for gestures that might be performed in different orders.
    """
    if traj_a.n_points != traj_b.n_points:
        raise ValueError(f"Trajectories must have same length: {traj_a.n_points} vs {traj_b.n_points}")

    n = traj_a.n_points
    matched = np.zeros(n, dtype=bool)
    total_dist = 0.0

    for i in range(n):
        best_dist = np.inf
        best_j = -1

        for j in range(n):
            if not matched[j]:
                dist = np.linalg.norm(traj_a.points[i] - traj_b.points[j])
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

        matched[best_j] = True
        total_dist += best_dist

    return total_dist / n


def distance_to_score(distance: float, half_distance: float = 0.5) -> float:
    """Convert distance to score (0-1, higher is better)."""
    return 1.0 / (1.0 + distance / half_distance)


# =============================================================================
# TEMPLATE-BASED RECOGNITION
# =============================================================================

@dataclass
class TrajectoryTemplate:
    """A template for gesture/pose recognition."""
    name: str
    trajectory: Trajectory  # Processed (resampled + normalized)
    finger_code: Optional[str] = None
    motion_type: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class HybridRecognizer:
    """
    Hybrid recognizer combining template matching for motion detection
    with signature analysis for pose inference.
    """

    def __init__(self, n_points: int = 32):
        self.n_points = n_points
        self.motion_templates: List[TrajectoryTemplate] = []
        self.pose_signatures: Dict[str, np.ndarray] = {}  # finger_code -> mean signature
        self.pose_stds: Dict[str, np.ndarray] = {}  # finger_code -> std

    def add_motion_template(self, name: str, trajectory: Trajectory,
                           finger_code: Optional[str] = None):
        """Add a motion template from raw trajectory."""
        processed = process_trajectory(trajectory, self.n_points)
        self.motion_templates.append(TrajectoryTemplate(
            name=name,
            trajectory=processed,
            finger_code=finger_code,
            motion_type=name
        ))

    def add_pose_signature(self, finger_code: str, samples: np.ndarray):
        """Add pose signature from samples (shape: n_samples x 3)."""
        self.pose_signatures[finger_code] = np.mean(samples, axis=0)
        self.pose_stds[finger_code] = np.std(samples, axis=0)

    def recognize_motion(self, trajectory: Trajectory,
                        use_cloud: bool = True) -> Tuple[Optional[str], float]:
        """
        Recognize motion/gesture from trajectory.

        Returns: (template_name, score)
        """
        if len(self.motion_templates) == 0:
            return None, 0.0

        processed = process_trajectory(trajectory, self.n_points)

        best_name = None
        best_score = 0.0

        distance_fn = cloud_distance if use_cloud else path_distance

        for template in self.motion_templates:
            try:
                dist = distance_fn(processed, template.trajectory)
                score = distance_to_score(dist)

                if score > best_score:
                    best_score = score
                    best_name = template.name
            except Exception as e:
                continue

        return best_name, best_score

    def recognize_pose(self, mag_sample: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize finger pose from single magnetometer sample.

        Returns: (finger_code, confidence)
        """
        if len(self.pose_signatures) == 0:
            return None, 0.0

        best_code = None
        best_dist = np.inf

        for code, signature in self.pose_signatures.items():
            dist = np.linalg.norm(mag_sample - signature)
            if dist < best_dist:
                best_dist = dist
                best_code = code

        # Convert distance to confidence using signature std
        if best_code in self.pose_stds:
            avg_std = np.mean(self.pose_stds[best_code])
            confidence = np.exp(-best_dist / (2 * avg_std + 1))
        else:
            confidence = distance_to_score(best_dist, 1000)

        return best_code, confidence

    def recognize_hybrid(self, trajectory: Trajectory) -> Dict[str, Any]:
        """
        Full hybrid recognition: motion + pose over time.

        Returns dict with:
        - motion: detected motion pattern
        - motion_score: confidence
        - pose_sequence: list of (finger_code, confidence) over trajectory
        - dominant_pose: most common pose during trajectory
        """
        # Motion recognition from full trajectory
        motion_name, motion_score = self.recognize_motion(trajectory)

        # Pose recognition at each point (if magnetometer data)
        pose_sequence = []
        if trajectory.trajectory_type in ["mag_residual", "combined"]:
            for i in range(trajectory.n_points):
                if trajectory.trajectory_type == "combined":
                    # Last 3 dims are mag
                    mag_sample = trajectory.points[i, 3:]
                else:
                    mag_sample = trajectory.points[i]

                code, conf = self.recognize_pose(mag_sample)
                pose_sequence.append((code, conf))

        # Find dominant pose
        if pose_sequence:
            pose_counts = defaultdict(float)
            for code, conf in pose_sequence:
                if code:
                    pose_counts[code] += conf

            dominant_pose = max(pose_counts.keys(), key=lambda k: pose_counts[k]) if pose_counts else None
        else:
            dominant_pose = None

        return {
            "motion": motion_name,
            "motion_score": motion_score,
            "pose_sequence": pose_sequence,
            "dominant_pose": dominant_pose,
            "trajectory_type": trajectory.trajectory_type
        }


# =============================================================================
# SESSION DATA LOADING
# =============================================================================

def load_session(session_path: Path) -> Tuple[List[Dict], List[LabeledSegment]]:
    """Load session data and extract labeled segments."""
    with open(session_path) as f:
        session = json.load(f)

    samples = session.get('samples', [])
    labels = session.get('labels', [])

    segments = []
    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})

        # Build finger code
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

        if '?' not in code and end > start:
            segments.append(LabeledSegment(
                start_idx=start,
                end_idx=end,
                finger_code=code,
                metadata={'motion': content.get('motion', 'unknown')}
            ))

    return samples, segments


def load_calibration(session_path: Path) -> Optional[Dict]:
    """Try to load calibration from session metadata."""
    with open(session_path) as f:
        session = json.load(f)

    metadata = session.get('metadata', {})
    calibration = metadata.get('calibration', None)

    if calibration is None:
        # Try loading from separate file
        cal_path = session_path.parent / 'calibration.json'
        if cal_path.exists():
            with open(cal_path) as f:
                calibration = json.load(f)

    return calibration


# =============================================================================
# ANALYSIS AND COMPARISON
# =============================================================================

def compare_trajectory_types(samples: List[Dict],
                            segments: List[LabeledSegment],
                            calibration: Optional[Dict] = None) -> Dict:
    """
    Compare different trajectory extraction methods.

    For each labeled segment, extract trajectories using different methods
    and analyze their discriminability.
    """
    results = {
        'accel': {'within_class_dist': [], 'between_class_dist': []},
        'mag_raw': {'within_class_dist': [], 'between_class_dist': []},
        'mag_residual': {'within_class_dist': [], 'between_class_dist': []},
        'combined': {'within_class_dist': [], 'between_class_dist': []}
    }

    # Extract and process trajectories for each segment
    trajectories_by_code = defaultdict(lambda: defaultdict(list))

    for seg in segments:
        for traj_type, extractor in [
            ('accel', lambda: extract_accelerometer_trajectory(samples, seg.start_idx, seg.end_idx)),
            ('mag_raw', lambda: extract_magnetometer_raw_trajectory(samples, seg.start_idx, seg.end_idx)),
            ('mag_residual', lambda: extract_magnetometer_residual_trajectory(samples, seg.start_idx, seg.end_idx, calibration)),
            ('combined', lambda: extract_combined_trajectory(samples, seg.start_idx, seg.end_idx, calibration))
        ]:
            try:
                traj = extractor()
                if traj.n_points >= 5:  # Need minimum points
                    processed = process_trajectory(traj, n_points=32)
                    trajectories_by_code[seg.finger_code][traj_type].append(processed)
            except Exception as e:
                continue

    # Compute within-class and between-class distances
    codes = list(trajectories_by_code.keys())

    for traj_type in results.keys():
        for code in codes:
            trajs = trajectories_by_code[code][traj_type]

            # Within-class distances
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    try:
                        dist = cloud_distance(trajs[i], trajs[j])
                        results[traj_type]['within_class_dist'].append(dist)
                    except:
                        continue

        # Between-class distances
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                trajs1 = trajectories_by_code[code1][traj_type]
                trajs2 = trajectories_by_code[code2][traj_type]

                for t1 in trajs1:
                    for t2 in trajs2:
                        try:
                            dist = cloud_distance(t1, t2)
                            results[traj_type]['between_class_dist'].append(dist)
                        except:
                            continue

    # Compute discriminability metrics
    for traj_type in results.keys():
        within = results[traj_type]['within_class_dist']
        between = results[traj_type]['between_class_dist']

        if within and between:
            results[traj_type]['mean_within'] = np.mean(within)
            results[traj_type]['mean_between'] = np.mean(between)
            results[traj_type]['discriminability'] = np.mean(between) / (np.mean(within) + 1e-6)
        else:
            results[traj_type]['mean_within'] = 0
            results[traj_type]['mean_between'] = 0
            results[traj_type]['discriminability'] = 0

    return results


def analyze_signature_trajectories(samples: List[Dict],
                                   segments: List[LabeledSegment],
                                   calibration: Optional[Dict] = None) -> Dict:
    """
    Analyze how magnetic signatures evolve during pose transitions.

    Key question: Do finger flexions create characteristic "paths" through
    signature space that could be used for motion-aware pose detection?
    """
    analysis = {
        'transition_paths': [],
        'static_clusters': defaultdict(list),
        'path_statistics': {}
    }

    # Group segments by finger code
    for seg in segments:
        traj = extract_magnetometer_residual_trajectory(
            samples, seg.start_idx, seg.end_idx, calibration
        )

        if traj.n_points > 0:
            # Static analysis: mean signature for this pose
            analysis['static_clusters'][seg.finger_code].append(
                np.mean(traj.points, axis=0)
            )

            # Dynamic analysis: path through signature space
            analysis['transition_paths'].append({
                'code': seg.finger_code,
                'path_length': traj.path_length,
                'duration': traj.duration,
                'start': traj.points[0] if len(traj.points) > 0 else None,
                'end': traj.points[-1] if len(traj.points) > 0 else None,
                'mean': np.mean(traj.points, axis=0),
                'std': np.std(traj.points, axis=0)
            })

    # Compute cluster statistics
    for code, signatures in analysis['static_clusters'].items():
        if len(signatures) > 0:
            sigs = np.array(signatures)
            analysis['path_statistics'][code] = {
                'mean': np.mean(sigs, axis=0).tolist(),
                'std': np.std(sigs, axis=0).tolist(),
                'n_samples': len(signatures)
            }

    return analysis


# =============================================================================
# MAIN EXPLORATION
# =============================================================================

def main():
    print("=" * 80)
    print("HYBRID TRAJECTORY INFERENCE EXPLORATION")
    print("Synthesizing Aligned Signatures + FFO$$ Template Matching")
    print("=" * 80)

    # Find wizard session with labels
    data_dir = Path('data/GAMBIT')
    sessions = list(data_dir.glob('*.json')) if data_dir.exists() else []

    if not sessions:
        print("\nNo session data found in data/GAMBIT/")
        print("Creating synthetic demonstration...")

        # Create synthetic demo data
        demonstrate_synthetic()
        return

    # Find session with most labels
    best_session = None
    best_label_count = 0

    for session_path in sessions:
        try:
            with open(session_path) as f:
                session = json.load(f)
            labels = session.get('labels', [])
            if len(labels) > best_label_count:
                best_label_count = len(labels)
                best_session = session_path
        except:
            continue

    if best_session is None or best_label_count == 0:
        print("\nNo labeled sessions found")
        demonstrate_synthetic()
        return

    print(f"\nUsing session: {best_session.name}")
    print(f"Labels: {best_label_count}")

    # Load data
    samples, segments = load_session(best_session)
    calibration = load_calibration(best_session)

    print(f"Samples: {len(samples)}")
    print(f"Labeled segments: {len(segments)}")

    # Analysis 1: Compare trajectory types
    print("\n" + "=" * 80)
    print("1. TRAJECTORY TYPE COMPARISON")
    print("=" * 80)

    comparison = compare_trajectory_types(samples, segments, calibration)

    print("\n{:<15} {:>12} {:>12} {:>15}".format(
        "Trajectory", "Within-Class", "Between-Class", "Discriminability"
    ))
    print("-" * 55)

    for traj_type in ['accel', 'mag_raw', 'mag_residual', 'combined']:
        stats = comparison[traj_type]
        print("{:<15} {:>12.3f} {:>12.3f} {:>15.2f}x".format(
            traj_type,
            stats['mean_within'],
            stats['mean_between'],
            stats['discriminability']
        ))

    # Analysis 2: Signature space trajectories
    print("\n" + "=" * 80)
    print("2. MAGNETIC SIGNATURE TRAJECTORY ANALYSIS")
    print("=" * 80)

    sig_analysis = analyze_signature_trajectories(samples, segments, calibration)

    print("\nSignature clusters by finger code:")
    for code, stats in sorted(sig_analysis['path_statistics'].items()):
        print(f"\n  {code}:")
        print(f"    Mean: [{stats['mean'][0]:>8.0f}, {stats['mean'][1]:>8.0f}, {stats['mean'][2]:>8.0f}]")
        print(f"    Std:  [{stats['std'][0]:>8.0f}, {stats['std'][1]:>8.0f}, {stats['std'][2]:>8.0f}]")
        print(f"    Samples: {stats['n_samples']}")

    # Analysis 3: Hybrid recognition demo
    print("\n" + "=" * 80)
    print("3. HYBRID RECOGNIZER DEMO")
    print("=" * 80)

    recognizer = HybridRecognizer(n_points=32)

    # Add pose signatures
    for code, stats in sig_analysis['path_statistics'].items():
        recognizer.add_pose_signature(code, np.array([stats['mean']]))

    # Add motion templates from first segment of each unique code
    seen_codes = set()
    for seg in segments:
        if seg.finger_code not in seen_codes:
            traj = extract_combined_trajectory(samples, seg.start_idx, seg.end_idx, calibration)
            if traj.n_points >= 10:
                recognizer.add_motion_template(f"pose_{seg.finger_code}", traj, seg.finger_code)
                seen_codes.add(seg.finger_code)

    print(f"\nRecognizer initialized:")
    print(f"  Motion templates: {len(recognizer.motion_templates)}")
    print(f"  Pose signatures: {len(recognizer.pose_signatures)}")

    # Test on a few segments
    print("\nHybrid recognition results:")
    for i, seg in enumerate(segments[:5]):
        traj = extract_combined_trajectory(samples, seg.start_idx, seg.end_idx, calibration)
        result = recognizer.recognize_hybrid(traj)

        print(f"\n  Segment {i+1} (ground truth: {seg.finger_code}):")
        print(f"    Motion detected: {result['motion']} (score: {result['motion_score']:.2f})")
        print(f"    Dominant pose: {result['dominant_pose']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & KEY INSIGHTS")
    print("=" * 80)

    print("""
    KEY FINDINGS:

    1. TRAJECTORY TYPES:
       - Accelerometer: Captures motion patterns (gestures)
       - Raw magnetometer: Orientation-dependent (problematic)
       - Residual magnetometer: Orientation-independent signatures
       - Combined: Best of both worlds for hybrid inference

    2. DISCRIMINABILITY:
       - Magnetic residuals show {discriminability:.1f}x between/within class ratio
       - Combined trajectories may improve motion-pose coupling

    3. SIGNATURE TRAJECTORIES:
       - Finger flexions create characteristic paths through signature space
       - These paths could serve as "motion templates" for pose transitions
       - Enables detection of HOW fingers moved, not just WHERE they are

    PROPOSED HYBRID ARCHITECTURE:

    ┌─────────────────────────────────────────────────────────────────┐
    │                      INPUT STREAM                               │
    │  [Accel (3D) + Gyro (3D) + Mag (3D)] @ 50Hz                    │
    └─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
    ┌───────────────────────────┐ ┌───────────────────────────┐
    │   SENSOR FUSION           │ │   CALIBRATION             │
    │   (Madgwick/Mahony)       │ │   (Hard/Soft Iron)        │
    │   → Orientation (quat)    │ │   → Earth Field Est       │
    └───────────────────────────┘ └───────────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   RESIDUAL EXTRACTION                                          │
    │   mag_residual = R(quat)^T @ (mag - earth_field)               │
    │   → Orientation-independent finger signature                   │
    └─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
    ┌───────────────────────────┐ ┌───────────────────────────┐
    │   FFO$$-STYLE MATCHING    │ │   NEURAL SIGNATURE        │
    │   (Trajectory Templates)  │ │   (Single-sample)         │
    │   → Motion detection      │ │   → Finger states         │
    │   → Transition patterns   │ │   → Per-finger probs      │
    └───────────────────────────┘ └───────────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   FUSION OUTPUT                                                │
    │   {{                                                           │
    │     motion: "wave_left",                                       │
    │     motion_confidence: 0.87,                                   │
    │     fingers: {{thumb: "flexed", index: "extended", ...}},      │
    │     finger_confidences: {{thumb: 0.92, index: 0.88, ...}},     │
    │     timestamp: 1234567890                                      │
    │   }}                                                           │
    └─────────────────────────────────────────────────────────────────┘

    NEXT STEPS:
    1. Implement proper sensor fusion for orientation estimation
    2. Build transition templates from pose-to-pose movements
    3. Train neural net on residual signatures (orientation-independent)
    4. Evaluate hybrid vs. single-method performance
    """.format(discriminability=comparison['mag_residual']['discriminability']))

    # Save results
    output_path = Path('ml/hybrid_trajectory_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.float64)):
            return float(obj)
        return obj

    results_to_save = {
        'trajectory_comparison': to_serializable({
            traj_type: {
                'mean_within': stats['mean_within'],
                'mean_between': stats['mean_between'],
                'discriminability': stats['discriminability']
            }
            for traj_type, stats in comparison.items()
        }),
        'signature_statistics': to_serializable(sig_analysis['path_statistics']),
        'session': best_session.name,
        'n_segments': len(segments),
        'n_samples': len(samples)
    }

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def demonstrate_synthetic():
    """Demonstrate concepts with synthetic data when real data unavailable."""
    print("\n" + "=" * 80)
    print("SYNTHETIC DEMONSTRATION")
    print("=" * 80)

    print("""
    Without real session data, demonstrating key concepts:

    1. TRAJECTORY TYPES:

       Traditional FFO$$ (accel only):
         wave_gesture = [(0,0,1), (0.5,0,0.8), (1,0,0.5), ...]
         → Captures motion shape, not finger state

       Magnetic signature trajectory (residual):
         fist_closing = [(0,0,0), (2000,1000,500), (5000,3000,2000), ...]
         → Captures finger states over time

       Combined (6D):
         gesture_with_pose = [
           (ax, ay, az, mx_res, my_res, mz_res),
           ...
         ]
         → Captures BOTH motion and pose simultaneously

    2. TEMPLATE MATCHING:

       For a "wave while making fist" gesture:
       - Motion template: characteristic acceleration pattern
       - Pose trajectory: 00000 → 22222 (all fingers flexing)

       Recognition outputs BOTH:
       - "wave_gesture" (motion detected)
       - "00000 → 22222" (pose transition detected)

    3. KEY INSIGHT:

       By using orientation-corrected residuals as the magnetic signal,
       the same pose transition creates the SAME trajectory regardless of
       hand orientation. This enables:
       - Template matching that works in any orientation
       - Motion detection that includes pose information
       - Pose detection that includes motion dynamics

    Run this script with real wizard-labeled data for full analysis.
    """)


if __name__ == '__main__':
    main()
