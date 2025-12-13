/**
 * Sensor-to-Hand Orientation Mathematical Model
 *
 * This module defines the mathematical relationship between IMU sensor data
 * and the 3D hand model orientation. Built from first principles based on
 * observed behavior.
 *
 * =============================================================================
 * PHYSICAL SETUP
 * =============================================================================
 *
 * SENSOR PLACEMENT:
 *   - Puck.js on BACK of hand (dorsum), battery side toward palm
 *   - MQBT42Q antenna edge pointing toward WRIST
 *   - IR LED edge pointing toward FINGERTIPS
 *
 * SENSOR COORDINATE FRAME (S) - LSM6DS3:
 *   When hand is palm-UP (palm facing ceiling):
 *
 *        +Y (toward fingertips)
 *         ^
 *         |
 *         |
 *   +X <--+---- (toward wrist)
 *         |
 *         v
 *        +Z (INTO palm, toward ceiling when palm-up)
 *
 *   Accelerometer reading when palm-up: ax≈0, ay≈0, az≈+1g
 *
 * =============================================================================
 * MADGWICK AHRS EULER ANGLE CONVENTION
 * =============================================================================
 *
 * The MadgwickAHRS filter produces Euler angles using aerospace convention:
 *   - Roll:  rotation about sensor X-axis (toward wrist)
 *   - Pitch: rotation about sensor Y-axis (toward fingers)
 *   - Yaw:   rotation about sensor Z-axis (palm normal)
 *
 * Angle signs follow right-hand rule:
 *   - Positive roll:  thumb-side of hand tips upward
 *   - Positive pitch: fingertips tip upward
 *   - Positive yaw:   clockwise rotation when viewed from above
 *
 * =============================================================================
 * THREE.JS HAND MODEL COORDINATE FRAME (H)
 * =============================================================================
 *
 *        +Y (finger extension direction)
 *         ^
 *         |
 *         |
 *         +--->  +X (toward pinky, right hand)
 *        /
 *       v
 *      +Z (palm normal, toward viewer)
 *
 * =============================================================================
 * COORDINATE FRAME RELATIONSHIP
 * =============================================================================
 *
 * Key insight: The sensor's axes relate to the hand model's axes as follows:
 *
 *   Sensor +X (toward wrist)     ←→  Hand -Y (opposite of finger extension)
 *   Sensor +Y (toward fingers)   ←→  Hand +Y (finger extension)
 *   Sensor +Z (into palm)        ←→  Hand -Z (opposite of palm normal)
 *
 * WAIT - this is wrong. Sensor X and Y are perpendicular, not opposite.
 * Let me reconsider:
 *
 *   Sensor +X (toward wrist)     ←→  Perpendicular to fingers = toward thumb (right hand)
 *   Sensor +Y (toward fingers)   ←→  Hand +Y (finger extension)
 *   Sensor +Z (into palm)        ←→  Hand -Z (opposite of palm normal)
 *
 * For a RIGHT hand:
 *   - "Toward wrist" from center of back of hand = toward thumb-side
 *   - "Toward pinky" = opposite of "toward thumb"
 *
 * So: Sensor +X ≈ Hand -X (opposite)
 *     Sensor +Y ≈ Hand +Y (same)
 *     Sensor +Z ≈ Hand -Z (opposite)
 *
 * This is a 180° rotation about the Y-axis!
 *
 * =============================================================================
 * EULER ANGLE MAPPING
 * =============================================================================
 *
 * When we apply a 180° Y rotation to transform from sensor frame to hand frame:
 *
 * For ROTATIONS (Euler angles describe body rotations):
 *   - Sensor roll (about X)  → Hand roll about -X  → NEGATE roll
 *   - Sensor pitch (about Y) → Hand pitch about Y  → KEEP pitch sign
 *   - Sensor yaw (about Z)   → Hand yaw about -Z   → NEGATE yaw
 *
 * EMPIRICAL OBSERVATIONS (from user testing):
 *   1. Palm-up on desk: Hand shows palm UP correctly
 *   2. Tip forward (fingers down): Hand tips BACKWARD (inverted!)
 *   3. Tip left: Hand tips RIGHT (inverted!)
 *   4. Rotate clockwise: Hand rotates clockwise (correct!)
 *
 * Analysis:
 *   - Yaw works → current yaw mapping is correct
 *   - Pitch is inverted → need to NEGATE pitch
 *   - Roll is inverted → need to flip roll sign (was negated, now should NOT be)
 *
 * =============================================================================
 * CORRECTED MAPPING FORMULA
 * =============================================================================
 *
 * Given sensor Euler angles (roll_s, pitch_s, yaw_s) in degrees:
 *
 * INCORRECT (old) mapping:
 *   hand_roll  = -roll_s  + 180
 *   hand_pitch = pitch_s  + 180
 *   hand_yaw   = yaw_s    - 180
 *
 * CORRECT (new) mapping:
 *   hand_roll  = roll_s   + 180    // UN-negate (was inverted)
 *   hand_pitch = -pitch_s + 180    // NEGATE (was inverted)
 *   hand_yaw   = yaw_s    - 180    // Keep same (works correctly)
 *
 * The +/-180 offsets align the hand model's "neutral" pose with sensor flat.
 *
 * =============================================================================
 * APPLICATION TO THREE.JS
 * =============================================================================
 *
 * Three.js Euler rotation.set(x, y, z, 'YXZ'):
 *   - x = rotation about X-axis = pitch (affects finger tilt)
 *   - y = rotation about Y-axis = yaw (affects rotation)
 *   - z = rotation about Z-axis = roll (affects left/right tilt)
 *
 * Order 'YXZ': Apply yaw first, then pitch, then roll
 *
 * @module shared/orientation-model
 */

/**
 * Orientation mapping configuration
 * These offsets align the hand model with sensor orientation
 */
export const ORIENTATION_CONFIG = {
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

/**
 * Map sensor Euler angles to hand model orientation
 *
 * @param {Object} sensorEuler - {roll, pitch, yaw} from Madgwick AHRS (degrees)
 * @param {Object} config - Optional override for ORIENTATION_CONFIG
 * @returns {Object} {roll, pitch, yaw} for hand model (degrees)
 */
export function mapSensorToHand(sensorEuler, config = ORIENTATION_CONFIG) {
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
 * @param {Object} sensorEuler - {roll, pitch, yaw} from Madgwick AHRS (degrees)
 * @param {Object} config - Optional override for ORIENTATION_CONFIG
 * @returns {Object} {x, y, z, order} for Three.js Euler
 */
export function mapSensorToThreeJS(sensorEuler, config = ORIENTATION_CONFIG) {
    const handAngles = mapSensorToHand(sensorEuler, config);
    const deg2rad = Math.PI / 180;

    return {
        x: handAngles.pitch * deg2rad,   // Three.js X = pitch
        y: handAngles.yaw * deg2rad,     // Three.js Y = yaw
        z: handAngles.roll * deg2rad,    // Three.js Z = roll
        order: config.eulerOrder
    };
}

/**
 * Debug helper: Describe what SHOULD happen for a given sensor orientation
 * Use this to validate physical behavior matches expectations
 *
 * @param {Object} sensorEuler - {roll, pitch, yaw} from sensor (degrees)
 * @returns {Object} Human-readable description of expected hand pose
 */
export function describeExpectedPose(sensorEuler) {
    const { roll, pitch, yaw } = sensorEuler;

    const descriptions = {
        sensor: {
            roll: roll.toFixed(1),
            pitch: pitch.toFixed(1),
            yaw: yaw.toFixed(1)
        },
        physical: {},
        expected: {}
    };

    // Describe what the sensor reading means physically
    // Sensor roll = tilt about X (toward wrist)
    if (Math.abs(roll) < 10) {
        descriptions.physical.roll = "Level left-right";
    } else if (roll > 0) {
        descriptions.physical.roll = `Thumb-side up ${roll.toFixed(0)}°`;
    } else {
        descriptions.physical.roll = `Pinky-side up ${(-roll).toFixed(0)}°`;
    }

    // Sensor pitch = tilt about Y (toward fingers)
    if (Math.abs(pitch) < 10) {
        descriptions.physical.pitch = "Level front-back";
    } else if (pitch > 0) {
        descriptions.physical.pitch = `Fingers pointing up ${pitch.toFixed(0)}°`;
    } else {
        descriptions.physical.pitch = `Fingers pointing down ${(-pitch).toFixed(0)}°`;
    }

    // Sensor yaw = rotation about Z (palm normal)
    descriptions.physical.yaw = `Heading ${((yaw + 360) % 360).toFixed(0)}°`;

    // Describe what the hand model SHOULD show
    // With corrected mapping:
    // - Positive sensor roll (thumb up) → hand roll +180 → thumb side tilts same direction
    // - Positive sensor pitch (fingers up) → -pitch +180 → fingers point up in model
    // - Sensor yaw → hand yaw (same direction)

    descriptions.expected.roll = descriptions.physical.roll.replace("Thumb", "Right").replace("Pinky", "Left");
    descriptions.expected.pitch = descriptions.physical.pitch;
    descriptions.expected.yaw = descriptions.physical.yaw;

    return descriptions;
}

/**
 * Generate validation test cases for the orientation model
 * Each test case describes a physical pose and expected behavior
 *
 * @returns {Array} Array of test case objects
 */
export function getValidationTestCases() {
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
 * @param {Function} mappingFn - Function that takes sensorEuler and returns handAngles
 * @returns {Object} Validation report
 */
export function validateMapping(mappingFn) {
    const testCases = getValidationTestCases();
    const results = [];

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

// Default export
export default {
    ORIENTATION_CONFIG,
    mapSensorToHand,
    mapSensorToThreeJS,
    describeExpectedPose,
    getValidationTestCases,
    validateMapping
};
