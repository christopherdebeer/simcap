"""
Hand Pose Generation for Magnetic Field Simulation

Models the geometry of a human hand with finger magnets attached to fingertips.
Generates kinematically valid poses for training data generation.

Coordinate system (right hand, palm down):
- X: Lateral (thumb side positive)
- Y: Forward (fingertip direction positive)
- Z: Vertical (palm side positive, back of hand negative)

Sensor is assumed to be on the back of the wrist at origin [0, 0, 0].
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class FingerState(Enum):
    """Finger flexion states matching SIMCAP label schema."""
    EXTENDED = 'extended'
    PARTIAL = 'partial'
    FLEXED = 'flexed'
    UNKNOWN = 'unknown'


@dataclass
class FingerGeometry:
    """Geometric parameters for a single finger."""
    name: str
    base_position: np.ndarray      # Position of MCP joint (mm)
    proximal_length: float         # Proximal phalanx length (mm)
    intermediate_length: float     # Intermediate phalanx length (mm)
    distal_length: float           # Distal phalanx length (mm)
    flexion_range: Tuple[float, float]  # Min/max flexion angle (degrees)

    @property
    def total_length(self) -> float:
        """Total finger length from MCP to tip."""
        return self.proximal_length + self.intermediate_length + self.distal_length


# Default hand geometry (adult male, approximate values)
DEFAULT_FINGER_GEOMETRY = {
    'thumb': FingerGeometry(
        name='thumb',
        base_position=np.array([25.0, 20.0, -5.0]),  # Offset from wrist sensor
        proximal_length=30.0,
        intermediate_length=0.0,   # Thumb has only 2 phalanges
        distal_length=25.0,
        flexion_range=(0, 80)
    ),
    'index': FingerGeometry(
        name='index',
        base_position=np.array([35.0, 55.0, 0.0]),
        proximal_length=40.0,
        intermediate_length=22.0,
        distal_length=18.0,
        flexion_range=(0, 100)
    ),
    'middle': FingerGeometry(
        name='middle',
        base_position=np.array([15.0, 60.0, 0.0]),
        proximal_length=44.0,
        intermediate_length=26.0,
        distal_length=20.0,
        flexion_range=(0, 100)
    ),
    'ring': FingerGeometry(
        name='ring',
        base_position=np.array([-5.0, 55.0, 0.0]),
        proximal_length=40.0,
        intermediate_length=24.0,
        distal_length=18.0,
        flexion_range=(0, 100)
    ),
    'pinky': FingerGeometry(
        name='pinky',
        base_position=np.array([-25.0, 45.0, 0.0]),
        proximal_length=30.0,
        intermediate_length=18.0,
        distal_length=15.0,
        flexion_range=(0, 100)
    )
}


@dataclass
class HandPose:
    """Complete hand pose with all finger positions."""
    finger_states: Dict[str, FingerState]
    fingertip_positions: Dict[str, np.ndarray]  # mm relative to wrist sensor
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            'finger_states': {k: v.value for k, v in self.finger_states.items()},
            'fingertip_positions': {
                k: v.tolist() for k, v in self.fingertip_positions.items()
            },
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'HandPose':
        """Create from dictionary."""
        return cls(
            finger_states={k: FingerState(v) for k, v in data['finger_states'].items()},
            fingertip_positions={
                k: np.array(v) for k, v in data['fingertip_positions'].items()
            },
            timestamp=data.get('timestamp', 0.0)
        )


class HandPoseGenerator:
    """
    Generate kinematically valid hand poses for simulation.

    The generator uses a simplified kinematic model where each finger's
    tip position is determined by its flexion state (extended/partial/flexed).
    """

    def __init__(
        self,
        geometry: Optional[Dict[str, FingerGeometry]] = None,
        sensor_position: Optional[np.ndarray] = None,
        randomize_geometry: bool = False
    ):
        """
        Initialize hand pose generator.

        Args:
            geometry: Finger geometry parameters. Defaults to adult male.
            sensor_position: Position of wrist sensor. Defaults to origin.
            randomize_geometry: Apply random variation to geometry (±10%)
        """
        self.geometry = geometry or DEFAULT_FINGER_GEOMETRY.copy()
        self.sensor_position = sensor_position if sensor_position is not None else np.zeros(3)

        if randomize_geometry:
            self._randomize_geometry()

    def _randomize_geometry(self, scale: float = 0.1):
        """Apply random variation to hand geometry for domain randomization."""
        for finger in self.geometry.values():
            # Vary lengths by ±scale
            finger.proximal_length *= np.random.uniform(1 - scale, 1 + scale)
            finger.intermediate_length *= np.random.uniform(1 - scale, 1 + scale)
            finger.distal_length *= np.random.uniform(1 - scale, 1 + scale)

            # Vary base positions by ±5mm
            finger.base_position += np.random.uniform(-5, 5, size=3)

    def compute_fingertip_position(
        self,
        finger: str,
        state: FingerState,
        noise_mm: float = 0.0
    ) -> np.ndarray:
        """
        Compute fingertip position for a given finger and state.

        The model simplifies finger kinematics:
        - Extended: Finger straight out, tip at maximum distance
        - Flexed: Finger curled in, tip close to palm
        - Partial: Intermediate position

        Args:
            finger: Finger name ('thumb', 'index', etc.)
            state: Flexion state
            noise_mm: Random position noise to add (mm)

        Returns:
            3D fingertip position in mm relative to wrist sensor
        """
        geom = self.geometry[finger]
        base = geom.base_position.copy()
        total_length = geom.total_length

        if state == FingerState.EXTENDED:
            # Finger extended straight out
            if finger == 'thumb':
                # Thumb extends laterally
                tip = base + np.array([total_length * 0.7, total_length * 0.5, 0.0])
            else:
                # Fingers extend forward
                tip = base + np.array([0.0, total_length, 0.0])

        elif state == FingerState.FLEXED:
            # Finger curled toward palm
            if finger == 'thumb':
                tip = base + np.array([15.0, 10.0, -20.0])  # Curled in
            else:
                # Fingertips curve back toward wrist and down
                tip = base + np.array([0.0, 20.0, -30.0])

        elif state == FingerState.PARTIAL:
            # Halfway between extended and flexed
            extended_tip = self.compute_fingertip_position(
                finger, FingerState.EXTENDED, noise_mm=0
            )
            flexed_tip = self.compute_fingertip_position(
                finger, FingerState.FLEXED, noise_mm=0
            )
            tip = (extended_tip + flexed_tip) / 2

        else:  # UNKNOWN - treat as extended
            tip = self.compute_fingertip_position(
                finger, FingerState.EXTENDED, noise_mm=0
            )

        # Add position noise
        if noise_mm > 0:
            tip += np.random.normal(0, noise_mm, size=3)

        return tip

    def generate_pose(
        self,
        finger_states: Dict[str, FingerState],
        noise_mm: float = 0.0,
        timestamp: float = 0.0
    ) -> HandPose:
        """
        Generate a complete hand pose from finger states.

        Args:
            finger_states: Dict mapping finger names to states
            noise_mm: Position noise to add (mm)
            timestamp: Timestamp for the pose

        Returns:
            HandPose with all fingertip positions
        """
        positions = {}
        for finger, state in finger_states.items():
            if finger in self.geometry:
                positions[finger] = self.compute_fingertip_position(
                    finger, state, noise_mm
                )

        return HandPose(
            finger_states=finger_states,
            fingertip_positions=positions,
            timestamp=timestamp
        )

    def generate_static_pose(
        self,
        pose_name: str,
        noise_mm: float = 0.0
    ) -> HandPose:
        """
        Generate a named static pose.

        Supports all poses defined in POSE_TEMPLATES plus additional poses.

        Args:
            pose_name: Name of the pose
            noise_mm: Position noise (mm)

        Returns:
            HandPose for the specified pose
        """
        all_extended = {f: FingerState.EXTENDED for f in self.geometry}
        all_flexed = {f: FingerState.FLEXED for f in self.geometry}
        all_partial = {f: FingerState.PARTIAL for f in self.geometry}

        # Check if pose is in POSE_TEMPLATES
        if pose_name in POSE_TEMPLATES:
            states = pose_template_to_states(POSE_TEMPLATES[pose_name])

        elif pose_name in ('open_palm', 'all_extended'):
            states = all_extended

        elif pose_name in ('fist', 'all_flexed'):
            states = all_flexed

        elif pose_name == 'pointing':
            states = all_flexed.copy()
            states['index'] = FingerState.EXTENDED

        elif pose_name == 'thumbs_up':
            states = all_flexed.copy()
            states['thumb'] = FingerState.EXTENDED

        elif pose_name == 'peace':
            states = all_flexed.copy()
            states['index'] = FingerState.EXTENDED
            states['middle'] = FingerState.EXTENDED

        elif pose_name == 'three_fingers':
            states = all_flexed.copy()
            states['index'] = FingerState.EXTENDED
            states['middle'] = FingerState.EXTENDED
            states['ring'] = FingerState.EXTENDED

        elif pose_name == 'pinch':
            states = all_extended.copy()
            states['thumb'] = FingerState.PARTIAL
            states['index'] = FingerState.PARTIAL

        elif pose_name == 'rest':
            states = all_partial

        else:
            # Generate a random pose if unknown
            states = {f: np.random.choice([FingerState.EXTENDED, FingerState.PARTIAL, FingerState.FLEXED])
                      for f in self.geometry}

        pose = self.generate_pose(states, noise_mm)

        # Apply close-range modification for high-magnitude poses
        if pose_name in CLOSE_RANGE_POSES:
            pose = self._apply_close_range_offset(pose, pose_name)

        return pose

    def _apply_close_range_offset(self, pose: HandPose, pose_name: str) -> HandPose:
        """
        Modify pose to bring fingers closer to sensor for higher field magnitudes.

        This addresses the sim-to-real gap where real data shows magnitudes up to
        200+ μT while standard simulation caps at ~85 μT.
        """
        close_positions = {}
        # Calibrated to produce ~120-180 μT max (matching real data P99)
        offset_scale = 0.75 if pose_name == 'fist_tight' else 0.80

        for finger, pos in pose.fingertip_positions.items():
            # Move position closer to sensor (origin)
            # Scale down the distance while maintaining direction
            new_pos = pos * offset_scale
            # Slight Z adjustment to bring closer to wrist level
            new_pos[2] = new_pos[2] * 0.9 - 3
            close_positions[finger] = new_pos

        return HandPose(
            finger_states=pose.finger_states,
            fingertip_positions=close_positions,
            timestamp=pose.timestamp
        )

    def generate_transition(
        self,
        start_states: Dict[str, FingerState],
        end_states: Dict[str, FingerState],
        num_frames: int = 10,
        noise_mm: float = 0.0
    ) -> List[HandPose]:
        """
        Generate smooth transition between two poses.

        Uses linear interpolation of fingertip positions.

        Args:
            start_states: Initial finger states
            end_states: Final finger states
            num_frames: Number of intermediate frames
            noise_mm: Position noise (mm)

        Returns:
            List of HandPose objects for the transition
        """
        start_pose = self.generate_pose(start_states, noise_mm=0)
        end_pose = self.generate_pose(end_states, noise_mm=0)

        poses = []
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0

            # Interpolate positions
            positions = {}
            for finger in self.geometry:
                start_pos = start_pose.fingertip_positions[finger]
                end_pos = end_pose.fingertip_positions[finger]
                positions[finger] = start_pos + t * (end_pos - start_pos)

                # Add noise
                if noise_mm > 0:
                    positions[finger] += np.random.normal(0, noise_mm, size=3)

            # Determine state at this frame
            states = {}
            for finger in self.geometry:
                if t < 0.33:
                    states[finger] = start_states[finger]
                elif t > 0.67:
                    states[finger] = end_states[finger]
                else:
                    states[finger] = FingerState.PARTIAL

            poses.append(HandPose(
                finger_states=states,
                fingertip_positions=positions,
                timestamp=t
            ))

        return poses

    def generate_random_pose(self, noise_mm: float = 1.0) -> HandPose:
        """Generate a random valid pose."""
        states = {
            finger: np.random.choice([
                FingerState.EXTENDED,
                FingerState.PARTIAL,
                FingerState.FLEXED
            ])
            for finger in self.geometry
        }
        return self.generate_pose(states, noise_mm)

    def distance_to_sensor(self, pose: HandPose) -> Dict[str, float]:
        """Calculate distance from each fingertip to the sensor."""
        distances = {}
        for finger, position in pose.fingertip_positions.items():
            distances[finger] = np.linalg.norm(position - self.sensor_position)
        return distances


# Predefined pose templates
POSE_TEMPLATES = {
    'open_palm': {'thumb': 'extended', 'index': 'extended', 'middle': 'extended',
                  'ring': 'extended', 'pinky': 'extended'},
    'fist': {'thumb': 'flexed', 'index': 'flexed', 'middle': 'flexed',
             'ring': 'flexed', 'pinky': 'flexed'},
    'pointing': {'thumb': 'flexed', 'index': 'extended', 'middle': 'flexed',
                 'ring': 'flexed', 'pinky': 'flexed'},
    'thumbs_up': {'thumb': 'extended', 'index': 'flexed', 'middle': 'flexed',
                  'ring': 'flexed', 'pinky': 'flexed'},
    'peace': {'thumb': 'flexed', 'index': 'extended', 'middle': 'extended',
              'ring': 'flexed', 'pinky': 'flexed'},
    'ok_sign': {'thumb': 'partial', 'index': 'partial', 'middle': 'extended',
                'ring': 'extended', 'pinky': 'extended'},
    'rock': {'thumb': 'extended', 'index': 'extended', 'middle': 'flexed',
             'ring': 'flexed', 'pinky': 'extended'},
    'call_me': {'thumb': 'extended', 'index': 'flexed', 'middle': 'flexed',
                'ring': 'flexed', 'pinky': 'extended'},
    # Close-range poses for higher field magnitudes (based on sim-to-real analysis)
    'fist_tight': {'thumb': 'flexed', 'index': 'flexed', 'middle': 'flexed',
                   'ring': 'flexed', 'pinky': 'flexed'},  # Uses special close position
    'pinch': {'thumb': 'partial', 'index': 'partial', 'middle': 'flexed',
              'ring': 'flexed', 'pinky': 'flexed'},
    'grab': {'thumb': 'partial', 'index': 'partial', 'middle': 'partial',
             'ring': 'partial', 'pinky': 'partial'},
}

# Poses that should position fingers closer to the sensor (for high-magnitude signals)
CLOSE_RANGE_POSES = {'fist_tight', 'pinch', 'grab'}


def pose_template_to_states(template: Dict[str, str]) -> Dict[str, FingerState]:
    """Convert string template to FingerState dict."""
    return {finger: FingerState(state) for finger, state in template.items()}


if __name__ == '__main__':
    # Test hand pose generation
    print("Hand Pose Generation Test")
    print("=" * 50)

    generator = HandPoseGenerator()

    # Generate some static poses
    for pose_name in ['open_palm', 'fist', 'pointing', 'peace']:
        pose = generator.generate_static_pose(pose_name)
        distances = generator.distance_to_sensor(pose)

        print(f"\n{pose_name}:")
        for finger, pos in pose.fingertip_positions.items():
            dist = distances[finger]
            print(f"  {finger:8s}: {pos} (distance: {dist:.1f} mm)")

    # Generate a transition
    print("\nTransition: fist → open_palm")
    start = pose_template_to_states(POSE_TEMPLATES['fist'])
    end = pose_template_to_states(POSE_TEMPLATES['open_palm'])
    transition = generator.generate_transition(start, end, num_frames=5)

    for i, pose in enumerate(transition):
        distances = generator.distance_to_sensor(pose)
        avg_dist = np.mean(list(distances.values()))
        print(f"  Frame {i}: avg distance = {avg_dist:.1f} mm")
