/**
 * Sensor-to-Hand Orientation Mathematical Model
 *
 * This module defines the mathematical relationship between IMU sensor data
 * and the 3D hand model orientation. Built from first principles based on
 * observed behavior.
 *
 * @module shared/orientation-model
 */

// ===== Type Definitions =====

export interface EulerAngles {
  roll: number;
  pitch: number;
  yaw: number;
}

export interface OrientationConfig {
  negateRoll: boolean;
  negatePitch: boolean;
  negateYaw: boolean;
  rollOffset: number;
  pitchOffset: number;
  yawOffset: number;
  eulerOrder: string;
}

export interface ThreeJSEuler {
  x: number;
  y: number;
  z: number;
  order: string;
}

export interface PoseDescription {
  sensor: {
    roll: string;
    pitch: string;
    yaw: string;
  };
  physical: {
    roll?: string;
    pitch?: string;
    yaw?: string;
  };
  expected: {
    roll?: string;
    pitch?: string;
    yaw?: string;
  };
}

export interface ValidationTestCase {
  name: string;
  description: string;
  sensorEuler: EulerAngles;
  physicalPose: string;
  validation: string[];
}

export interface ValidationResult {
  name: string;
  input: EulerAngles;
  output: EulerAngles;
  description: string;
  physicalPose: string;
  validation: string[];
}

// ===== Configuration =====

/**
 * Orientation mapping configuration
 * These offsets align the hand model with sensor orientation
 */
export const ORIENTATION_CONFIG: OrientationConfig = {
  // Whether to negate each axis before applying offset
  negateRoll: false,   // Changed: was true (inverted)
  negatePitch: true,   // Changed: was false (inverted)
  negateYaw: false,    // Keep: works correctly

  // Offset values in degrees to align neutral pose
  rollOffset: 180,
  pitchOffset: 180,
  yawOffset: -180,

  // Three.js Euler order
  eulerOrder: 'YXZ'
};

// ===== Mapping Functions =====

/**
 * Map sensor Euler angles to hand model orientation
 *
 * @param sensorEuler - {roll, pitch, yaw} from Madgwick AHRS (degrees)
 * @param config - Optional override for ORIENTATION_CONFIG
 * @returns {roll, pitch, yaw} for hand model (degrees)
 */
export function mapSensorToHand(
  sensorEuler: EulerAngles,
  config: OrientationConfig = ORIENTATION_CONFIG
): EulerAngles {
  const { roll: s_roll, pitch: s_pitch, yaw: s_yaw } = sensorEuler;

  return {
    roll: (config.negateRoll ? -s_roll : s_roll) + config.rollOffset,
    pitch: (config.negatePitch ? -s_pitch : s_pitch) + config.pitchOffset,
    yaw: (config.negateYaw ? -s_yaw : s_yaw) + config.yawOffset
  };
}

/**
 * Map sensor Euler angles to Three.js rotation values
 * Returns radians ready for rotation.set(x, y, z, order)
 *
 * @param sensorEuler - {roll, pitch, yaw} from Madgwick AHRS (degrees)
 * @param config - Optional override for ORIENTATION_CONFIG
 * @returns {x, y, z, order} for Three.js Euler
 */
export function mapSensorToThreeJS(
  sensorEuler: EulerAngles,
  config: OrientationConfig = ORIENTATION_CONFIG
): ThreeJSEuler {
  const handAngles = mapSensorToHand(sensorEuler, config);
  const deg2rad = Math.PI / 180;

  return {
    x: handAngles.pitch * deg2rad,   // Three.js X = pitch
    y: handAngles.yaw * deg2rad,     // Three.js Y = yaw
    z: handAngles.roll * deg2rad,    // Three.js Z = roll
    order: config.eulerOrder
  };
}

// ===== Debug Helpers =====

/**
 * Debug helper: Describe what SHOULD happen for a given sensor orientation
 * Use this to validate physical behavior matches expectations
 *
 * @param sensorEuler - {roll, pitch, yaw} from sensor (degrees)
 * @returns Human-readable description of expected hand pose
 */
export function describeExpectedPose(sensorEuler: EulerAngles): PoseDescription {
  const { roll, pitch, yaw } = sensorEuler;

  const descriptions: PoseDescription = {
    sensor: {
      roll: roll.toFixed(1),
      pitch: pitch.toFixed(1),
      yaw: yaw.toFixed(1)
    },
    physical: {},
    expected: {}
  };

  // Describe what the sensor reading means physically
  if (Math.abs(roll) < 10) {
    descriptions.physical.roll = "Level left-right";
  } else if (roll > 0) {
    descriptions.physical.roll = `Thumb-side up ${roll.toFixed(0)}°`;
  } else {
    descriptions.physical.roll = `Pinky-side up ${(-roll).toFixed(0)}°`;
  }

  if (Math.abs(pitch) < 10) {
    descriptions.physical.pitch = "Level front-back";
  } else if (pitch > 0) {
    descriptions.physical.pitch = `Fingers pointing up ${pitch.toFixed(0)}°`;
  } else {
    descriptions.physical.pitch = `Fingers pointing down ${(-pitch).toFixed(0)}°`;
  }

  descriptions.physical.yaw = `Heading ${((yaw + 360) % 360).toFixed(0)}°`;

  // Describe what the hand model SHOULD show
  descriptions.expected.roll = descriptions.physical.roll?.replace("Thumb", "Right").replace("Pinky", "Left");
  descriptions.expected.pitch = descriptions.physical.pitch;
  descriptions.expected.yaw = descriptions.physical.yaw;

  return descriptions;
}

// ===== Validation =====

/**
 * Generate validation test cases for the orientation model
 * Each test case describes a physical pose and expected behavior
 */
export function getValidationTestCases(): ValidationTestCase[] {
  return [
    {
      name: "FLAT_PALM_UP",
      description: "Device flat on desk, palm facing ceiling",
      sensorEuler: { roll: 0, pitch: 0, yaw: 0 },
      physicalPose: "Hand flat, palm UP, fingers pointing away from viewer",
      validation: [
        "Hand should show PALM facing UP (toward ceiling)",
        "Fingers should extend AWAY from camera (into screen)",
        "Thumb should be on LEFT side (right hand)"
      ]
    },
    {
      name: "TIP_FORWARD_30",
      description: "Tip device forward 30° (fingers pointing down)",
      sensorEuler: { roll: 0, pitch: -30, yaw: 0 },
      physicalPose: "Hand tipped forward, fingertips pointing toward floor",
      validation: [
        "Fingers should point DOWN and slightly toward camera",
        "Palm should still face mostly UP",
        "This tests PITCH mapping"
      ]
    },
    {
      name: "TIP_BACKWARD_30",
      description: "Tip device backward 30° (fingers pointing up)",
      sensorEuler: { roll: 0, pitch: 30, yaw: 0 },
      physicalPose: "Hand tipped backward, fingertips pointing toward ceiling",
      validation: [
        "Fingers should point UP toward ceiling",
        "Palm should face away from camera",
        "This tests PITCH mapping (opposite direction)"
      ]
    },
    {
      name: "TILT_LEFT_30",
      description: "Tilt device left 30° (pinky side down)",
      sensorEuler: { roll: -30, pitch: 0, yaw: 0 },
      physicalPose: "Hand tilted to the left, pinky side toward floor",
      validation: [
        "Pinky side should be DOWN (toward floor)",
        "Thumb side should be UP (toward ceiling)",
        "This tests ROLL mapping"
      ]
    },
    {
      name: "TILT_RIGHT_30",
      description: "Tilt device right 30° (thumb side down)",
      sensorEuler: { roll: 30, pitch: 0, yaw: 0 },
      physicalPose: "Hand tilted to the right, thumb side toward floor",
      validation: [
        "Thumb side should be DOWN (toward floor)",
        "Pinky side should be UP (toward ceiling)",
        "This tests ROLL mapping (opposite direction)"
      ]
    },
    {
      name: "ROTATE_CW_45",
      description: "Rotate device clockwise 45° while flat",
      sensorEuler: { roll: 0, pitch: 0, yaw: 45 },
      physicalPose: "Hand flat but rotated clockwise (viewed from above)",
      validation: [
        "Palm should still face UP",
        "Fingers should point to the LEFT side of screen",
        "This tests YAW mapping"
      ]
    },
    {
      name: "ROTATE_CCW_45",
      description: "Rotate device counter-clockwise 45° while flat",
      sensorEuler: { roll: 0, pitch: 0, yaw: -45 },
      physicalPose: "Hand flat but rotated counter-clockwise (viewed from above)",
      validation: [
        "Palm should still face UP",
        "Fingers should point to the RIGHT side of screen",
        "This tests YAW mapping (opposite direction)"
      ]
    },
    {
      name: "PALM_DOWN",
      description: "Device flipped 180° (palm facing floor)",
      sensorEuler: { roll: 180, pitch: 0, yaw: 0 },
      physicalPose: "Hand flipped over, palm facing floor",
      validation: [
        "Palm (sensor disc) should face DOWN/toward camera",
        "Back of hand should face UP/away from camera",
        "This tests extreme ROLL"
      ]
    },
    {
      name: "FINGERS_UP_90",
      description: "Hand vertical with fingers pointing up",
      sensorEuler: { roll: 0, pitch: 90, yaw: 0 },
      physicalPose: "Hand vertical, fingertips pointing at ceiling",
      validation: [
        "Fingers should point straight UP",
        "Palm should face toward camera",
        "This tests extreme PITCH"
      ]
    }
  ];
}

/**
 * Validate orientation mapping against expected behavior
 * Returns a report of what matches and what doesn't
 *
 * @param mappingFn - Function that takes sensorEuler and returns handAngles
 * @returns Validation report
 */
export function validateMapping(
  mappingFn: (euler: EulerAngles) => EulerAngles
): ValidationResult[] {
  const testCases = getValidationTestCases();
  const results: ValidationResult[] = [];

  for (const testCase of testCases) {
    const handAngles = mappingFn(testCase.sensorEuler);
    results.push({
      name: testCase.name,
      input: testCase.sensorEuler,
      output: handAngles,
      description: testCase.description,
      physicalPose: testCase.physicalPose,
      validation: testCase.validation
    });
  }

  return results;
}

// ===== Default Export =====

export default {
  ORIENTATION_CONFIG,
  mapSensorToHand,
  mapSensorToThreeJS,
  describeExpectedPose,
  getValidationTestCases,
  validateMapping
};
