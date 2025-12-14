/**
 * Orientation Calibration System
 *
 * A first-principles framework for determining the correct mapping between
 * sensor orientation data and 3D hand model orientation.
 *
 * =============================================================================
 * PROBLEM STATEMENT
 * =============================================================================
 *
 * We have a chain of coordinate frame transformations:
 *
 *   [Physical World] → [Sensor] → [AHRS] → [Mapping] → [Hand Model] → [Render View]
 *        (W)            (S)        (A)       (M)          (H)            (V)
 *
 * Each transition introduces potential for:
 *   - Axis permutations (which physical axis maps to which coordinate)
 *   - Sign inversions (positive vs negative direction)
 *   - Offset rotations (base orientation differences)
 *   - Euler order mismatches (ZYX vs YXZ vs XYZ etc.)
 *   - Gimbal lock artifacts near singularities
 *
 * With 3 axes, each having 2 possible signs, and 6 possible permutations,
 * there are 6 × 2³ = 48 possible axis mappings per transformation stage.
 *
 * =============================================================================
 * AXIS COUPLING PROBLEM
 * =============================================================================
 *
 * When we observe that rotating around ONE physical axis causes MULTIPLE
 * Euler angles to change, this indicates one of:
 *
 *   1. AXIS PERMUTATION: The Euler angle "pitch" might actually be measuring
 *      what we call "roll" physically. Need to test all 6 permutations.
 *
 *   2. EULER ORDER MISMATCH: Different extraction orders (ZYX, YXZ, XYZ, etc.)
 *      produce different Euler angles from the same quaternion. If the
 *      AHRS uses ZYX but we apply YXZ, the mapping will be wrong.
 *
 *   3. GIMBAL LOCK: Near ±90° pitch, roll and yaw become coupled and can
 *      exchange values wildly. This is a fundamental Euler angle limitation.
 *
 *   4. SENSOR AXIS MISALIGNMENT: The physical sensor axes may not align
 *      with our assumed orientation model.
 *
 * The solution requires:
 *   a) Capturing observations that can distinguish these cases
 *   b) Testing multiple hypothesis configurations
 *   c) Finding the configuration that eliminates coupling
 *
 * =============================================================================
 * COORDINATE FRAMES
 * =============================================================================
 *
 * 1. WORLD FRAME (W)
 *    - Absolute reference frame
 *    - Gravity vector: (0, 0, -g) pointing DOWN
 *    - Magnetic North: varies by location
 *    - This is the frame the user perceives
 *
 * 2. SENSOR FRAME (S) - LSM6DS3
 *    - Internal IMU coordinate system
 *    - Accelerometer measures: reaction to gravity (points UP when stationary)
 *    - Gyroscope measures: angular velocity
 *    - Defined by IC package orientation on PCB
 *
 * 3. DEVICE FRAME (D)
 *    - How the sensor is mounted on the Puck.js
 *    - Relationship S→D is fixed by hardware
 *    - Documented as: X→wrist, Y→fingers, Z→into palm
 *
 * 4. AHRS OUTPUT FRAME (A)
 *    - The coordinate frame of Madgwick filter output
 *    - Euler angles (roll, pitch, yaw) convention matters
 *    - Rotation order (intrinsic vs extrinsic) matters
 *
 * 5. HAND MODEL FRAME (H)
 *    - Three.js convention: right-handed, Y-up
 *    - Hand model specific: what direction is "forward"?
 *
 * 6. RENDER VIEW FRAME (V)
 *    - Camera position and orientation
 *    - What the user sees on screen
 *
 * =============================================================================
 * INFORMATION-THEORETIC APPROACH
 * =============================================================================
 *
 * We treat calibration as a constraint satisfaction problem.
 *
 * An OBSERVATION consists of:
 *   1. Physical State (P): Known real-world configuration
 *      - Device orientation (flat, tilted, etc.)
 *      - Which way is "up", "forward", etc.
 *
 *   2. Sensor Data (S): Raw IMU readings
 *      - Accelerometer: ax, ay, az
 *      - Gyroscope: gx, gy, gz
 *      - (Optional) Magnetometer: mx, my, mz
 *
 *   3. AHRS Output (A): Filter-derived orientation
 *      - Euler angles: roll, pitch, yaw
 *      - Quaternion: w, x, y, z
 *
 *   4. Render State (R): Applied hand model orientation
 *      - Three.js Euler: x, y, z, order
 *      - Effective visual orientation
 *
 *   5. User Observation (U): What the user sees
 *      - Palm facing direction
 *      - Finger pointing direction
 *      - Thumb position
 *      - Match/mismatch with physical state
 *
 * Each observation provides CONSTRAINTS on the unknown mapping M.
 *
 * Given enough observations, we can:
 *   a) Determine if current mapping is correct
 *   b) Identify which axes are inverted
 *   c) Solve for the correct transformation
 *
 * =============================================================================
 * CANONICAL REFERENCE POSES
 * =============================================================================
 *
 * We define a set of unambiguous physical configurations that provide
 * maximum information about the axis mappings.
 *
 * Key poses that constrain each axis:
 *
 * 1. FLAT_PALM_UP: Device flat on horizontal surface
 *    - Constrains: Gravity alignment, neutral orientation
 *    - Expected: Palm faces ceiling, fingers away, thumb left (RH)
 *
 * 2. PITCH_FORWARD_45: Tip forward ~45°
 *    - Constrains: Pitch axis sign
 *    - Expected: Fingers point toward floor
 *
 * 3. PITCH_BACKWARD_45: Tip backward ~45°
 *    - Constrains: Pitch axis sign (confirmation)
 *    - Expected: Fingers point toward ceiling
 *
 * 4. ROLL_LEFT_45: Tilt left ~45°
 *    - Constrains: Roll axis sign
 *    - Expected: Pinky side toward floor
 *
 * 5. ROLL_RIGHT_45: Tilt right ~45°
 *    - Constrains: Roll axis sign (confirmation)
 *    - Expected: Thumb side toward floor
 *
 * 6. YAW_CW_90: Rotate clockwise 90° while flat
 *    - Constrains: Yaw axis sign
 *    - Expected: Fingers point left
 *
 * 7. YAW_CCW_90: Rotate counter-clockwise 90° while flat
 *    - Constrains: Yaw axis sign (confirmation)
 *    - Expected: Fingers point right
 *
 * 8. VERTICAL_FINGERS_UP: Standing on edge, fingers pointing up
 *    - Constrains: Pitch at ±90°
 *    - Expected: Fingers point at ceiling
 *
 * =============================================================================
 * OBSERVATION DATA STRUCTURE
 * =============================================================================
 */

/**
 * Multi-choice answer options for validation questions
 * These replace binary yes/no to capture more nuanced observations
 */
export const ANSWER_OPTIONS = {
    // For direction/position questions
    DIRECTION: {
        CORRECT: { id: 'correct', label: 'Correct', value: 1 },
        MOSTLY_CORRECT: { id: 'mostly_correct', label: 'Mostly correct (small offset)', value: 0.7 },
        WRONG_AXIS: { id: 'wrong_axis', label: 'Wrong axis moved instead', value: -0.5 },
        OPPOSITE: { id: 'opposite', label: 'Opposite direction', value: -1 },
        NO_MOVEMENT: { id: 'no_movement', label: 'No movement', value: 0 },
        COUPLED: { id: 'coupled', label: 'Coupled - multiple axes moved', value: null },
        UNCLEAR: { id: 'unclear', label: 'Cannot determine', value: null }
    },
    // For static position questions
    POSITION: {
        YES: { id: 'yes', label: 'Yes', value: true },
        MOSTLY: { id: 'mostly', label: 'Mostly', value: true },
        NO: { id: 'no', label: 'No', value: false },
        UNCLEAR: { id: 'unclear', label: 'Cannot determine', value: null }
    }
};

/**
 * Axis coupling observation structure
 * Captures when expected single-axis movement affects multiple axes
 */
export const COUPLING_TYPES = {
    NONE: 'none',                    // Expected single axis moved correctly
    PARTIAL: 'partial',              // Expected axis moved, but others too
    WRONG_PRIMARY: 'wrong_primary',  // Different axis was primary mover
    SWAPPED: 'swapped',              // Two axes appear swapped
    ALL_COUPLED: 'all_coupled'       // All three axes moved together
};

/**
 * Reference pose definitions
 */
export const REFERENCE_POSES = {
    FLAT_PALM_UP: {
        id: 'FLAT_PALM_UP',
        name: 'Flat, Palm Up',
        description: 'Device flat on desk, battery side down, screen up',
        instructions: [
            'Place device on flat horizontal surface',
            'Battery/back should touch the surface',
            'Screen/top should face ceiling'
        ],
        physicalState: {
            deviceOrientation: 'horizontal',
            gravityDirection: '-Z in device frame',
            expectedAngles: { roll: 0, pitch: 0, yaw: 'any' }
        },
        expectedVisual: {
            palmFacing: 'up',         // toward ceiling
            fingerPointing: 'away',   // away from viewer (into screen)
            thumbPosition: 'left'     // left side of screen (for right hand)
        },
        primaryAxis: null,  // Reference pose, no primary movement
        validationQuestions: [
            { id: 'palm_up', question: 'Palm faces UP (toward ceiling)?', type: 'position', axis: null },
            { id: 'fingers_away', question: 'Fingers point AWAY from you?', type: 'position', axis: null },
            { id: 'thumb_left', question: 'Thumb is on LEFT side?', type: 'position', axis: null }
        ],
        // New: coupling observation for baseline
        couplingQuestion: null  // No movement expected
    },

    PITCH_FORWARD_45: {
        id: 'PITCH_FORWARD_45',
        name: 'Pitch Forward 45°',
        description: 'Tip device forward so fingers point toward floor',
        instructions: [
            'Start from flat position',
            'Tilt device forward ~45°',
            'Finger-edge should point toward floor'
        ],
        physicalState: {
            deviceOrientation: 'pitched forward',
            primaryAxis: 'pitch',
            expectedAngles: { roll: 0, pitch: -45, yaw: 'any' }
        },
        expectedVisual: {
            palmFacing: 'up-forward',
            fingerPointing: 'down',
            thumbPosition: 'left'
        },
        primaryAxis: 'pitch',
        validationQuestions: [
            {
                id: 'finger_direction',
                question: 'Which way do the fingers point?',
                type: 'direction',
                axis: 'pitch',
                options: [
                    { id: 'down_correct', label: 'DOWN toward floor (correct)', value: 'correct' },
                    { id: 'up_opposite', label: 'UP toward ceiling (opposite)', value: 'opposite' },
                    { id: 'left_wrong', label: 'LEFT (wrong axis)', value: 'wrong_axis' },
                    { id: 'right_wrong', label: 'RIGHT (wrong axis)', value: 'wrong_axis' },
                    { id: 'no_change', label: 'No change from flat', value: 'no_movement' },
                    { id: 'multiple', label: 'Multiple directions at once', value: 'coupled' }
                ]
            }
        ],
        couplingQuestion: {
            id: 'coupling_pitch_fwd',
            question: 'Did you observe movement on other axes?',
            options: [
                { id: 'only_pitch', label: 'Only pitch changed (fingers up/down)', value: 'none' },
                { id: 'pitch_plus_roll', label: 'Pitch + roll changed (left/right tilt too)', value: 'partial' },
                { id: 'pitch_plus_yaw', label: 'Pitch + yaw changed (rotation too)', value: 'partial' },
                { id: 'all_axes', label: 'All three axes changed', value: 'all_coupled' },
                { id: 'wrong_axis', label: 'Roll or yaw changed but NOT pitch', value: 'wrong_primary' }
            ]
        }
    },

    PITCH_BACKWARD_45: {
        id: 'PITCH_BACKWARD_45',
        name: 'Pitch Backward 45°',
        description: 'Tip device backward so fingers point toward ceiling',
        instructions: [
            'Start from flat position',
            'Tilt device backward ~45°',
            'Finger-edge should point toward ceiling'
        ],
        physicalState: {
            deviceOrientation: 'pitched backward',
            primaryAxis: 'pitch',
            expectedAngles: { roll: 0, pitch: 45, yaw: 'any' }
        },
        expectedVisual: {
            palmFacing: 'up-backward',
            fingerPointing: 'up',
            thumbPosition: 'left'
        },
        primaryAxis: 'pitch',
        validationQuestions: [
            {
                id: 'finger_direction',
                question: 'Which way do the fingers point?',
                type: 'direction',
                axis: 'pitch',
                options: [
                    { id: 'up_correct', label: 'UP toward ceiling (correct)', value: 'correct' },
                    { id: 'down_opposite', label: 'DOWN toward floor (opposite)', value: 'opposite' },
                    { id: 'left_wrong', label: 'LEFT (wrong axis)', value: 'wrong_axis' },
                    { id: 'right_wrong', label: 'RIGHT (wrong axis)', value: 'wrong_axis' },
                    { id: 'no_change', label: 'No change from flat', value: 'no_movement' },
                    { id: 'multiple', label: 'Multiple directions at once', value: 'coupled' }
                ]
            }
        ],
        couplingQuestion: {
            id: 'coupling_pitch_back',
            question: 'Did you observe movement on other axes?',
            options: [
                { id: 'only_pitch', label: 'Only pitch changed (fingers up/down)', value: 'none' },
                { id: 'pitch_plus_roll', label: 'Pitch + roll changed', value: 'partial' },
                { id: 'pitch_plus_yaw', label: 'Pitch + yaw changed', value: 'partial' },
                { id: 'all_axes', label: 'All three axes changed', value: 'all_coupled' },
                { id: 'wrong_axis', label: 'Roll or yaw changed but NOT pitch', value: 'wrong_primary' }
            ]
        }
    },

    ROLL_LEFT_45: {
        id: 'ROLL_LEFT_45',
        name: 'Roll Left 45°',
        description: 'Tilt device left so pinky side goes down',
        instructions: [
            'Start from flat position',
            'Tilt device to YOUR LEFT ~45°',
            'Pinky-edge should be lower than thumb-edge'
        ],
        physicalState: {
            deviceOrientation: 'rolled left',
            primaryAxis: 'roll',
            expectedAngles: { roll: -45, pitch: 0, yaw: 'any' }
        },
        expectedVisual: {
            palmFacing: 'up-left',
            fingerPointing: 'away',
            thumbPosition: 'up-left',
            pinkyPosition: 'down-right'
        },
        primaryAxis: 'roll',
        validationQuestions: [
            {
                id: 'tilt_direction',
                question: 'Which side of the hand tilts down?',
                type: 'direction',
                axis: 'roll',
                options: [
                    { id: 'pinky_down', label: 'PINKY side down (correct)', value: 'correct' },
                    { id: 'thumb_down', label: 'THUMB side down (opposite)', value: 'opposite' },
                    { id: 'fingers_tilt', label: 'Fingers tilt up/down instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'rotation', label: 'Hand rotates CW/CCW instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'no_change', label: 'No change from flat', value: 'no_movement' },
                    { id: 'multiple', label: 'Multiple movements at once', value: 'coupled' }
                ]
            }
        ],
        couplingQuestion: {
            id: 'coupling_roll_left',
            question: 'Did you observe movement on other axes?',
            options: [
                { id: 'only_roll', label: 'Only roll changed (left/right tilt)', value: 'none' },
                { id: 'roll_plus_pitch', label: 'Roll + pitch changed (fingers tilted too)', value: 'partial' },
                { id: 'roll_plus_yaw', label: 'Roll + yaw changed (rotation too)', value: 'partial' },
                { id: 'all_axes', label: 'All three axes changed', value: 'all_coupled' },
                { id: 'wrong_axis', label: 'Pitch or yaw changed but NOT roll', value: 'wrong_primary' }
            ]
        }
    },

    ROLL_RIGHT_45: {
        id: 'ROLL_RIGHT_45',
        name: 'Roll Right 45°',
        description: 'Tilt device right so thumb side goes down',
        instructions: [
            'Start from flat position',
            'Tilt device to YOUR RIGHT ~45°',
            'Thumb-edge should be lower than pinky-edge'
        ],
        physicalState: {
            deviceOrientation: 'rolled right',
            primaryAxis: 'roll',
            expectedAngles: { roll: 45, pitch: 0, yaw: 'any' }
        },
        expectedVisual: {
            palmFacing: 'up-right',
            fingerPointing: 'away',
            thumbPosition: 'down-right',
            pinkyPosition: 'up-left'
        },
        primaryAxis: 'roll',
        validationQuestions: [
            {
                id: 'tilt_direction',
                question: 'Which side of the hand tilts down?',
                type: 'direction',
                axis: 'roll',
                options: [
                    { id: 'thumb_down', label: 'THUMB side down (correct)', value: 'correct' },
                    { id: 'pinky_down', label: 'PINKY side down (opposite)', value: 'opposite' },
                    { id: 'fingers_tilt', label: 'Fingers tilt up/down instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'rotation', label: 'Hand rotates CW/CCW instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'no_change', label: 'No change from flat', value: 'no_movement' },
                    { id: 'multiple', label: 'Multiple movements at once', value: 'coupled' }
                ]
            }
        ],
        couplingQuestion: {
            id: 'coupling_roll_right',
            question: 'Did you observe movement on other axes?',
            options: [
                { id: 'only_roll', label: 'Only roll changed (left/right tilt)', value: 'none' },
                { id: 'roll_plus_pitch', label: 'Roll + pitch changed', value: 'partial' },
                { id: 'roll_plus_yaw', label: 'Roll + yaw changed', value: 'partial' },
                { id: 'all_axes', label: 'All three axes changed', value: 'all_coupled' },
                { id: 'wrong_axis', label: 'Pitch or yaw changed but NOT roll', value: 'wrong_primary' }
            ]
        }
    },

    YAW_CW_90: {
        id: 'YAW_CW_90',
        name: 'Yaw Clockwise 90°',
        description: 'Rotate device clockwise 90° while keeping flat',
        instructions: [
            'Start from flat position',
            'Keep device flat (parallel to floor)',
            'Rotate CLOCKWISE 90° (viewed from above)',
            'Fingers should now point to YOUR LEFT'
        ],
        physicalState: {
            deviceOrientation: 'horizontal, rotated CW',
            primaryAxis: 'yaw',
            expectedAngles: { roll: 0, pitch: 0, yaw: 90 }
        },
        expectedVisual: {
            palmFacing: 'up',
            fingerPointing: 'left',
            thumbPosition: 'away'
        },
        primaryAxis: 'yaw',
        validationQuestions: [
            {
                id: 'rotation_direction',
                question: 'Which way do the fingers point after rotation?',
                type: 'direction',
                axis: 'yaw',
                options: [
                    { id: 'left_correct', label: 'LEFT (correct)', value: 'correct' },
                    { id: 'right_opposite', label: 'RIGHT (opposite)', value: 'opposite' },
                    { id: 'up_wrong', label: 'UP/DOWN instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'tilt_wrong', label: 'Hand tilted left/right instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'no_change', label: 'No change from flat', value: 'no_movement' },
                    { id: 'multiple', label: 'Multiple movements at once', value: 'coupled' }
                ]
            }
        ],
        couplingQuestion: {
            id: 'coupling_yaw_cw',
            question: 'Did you observe movement on other axes?',
            options: [
                { id: 'only_yaw', label: 'Only yaw changed (rotation)', value: 'none' },
                { id: 'yaw_plus_pitch', label: 'Yaw + pitch changed', value: 'partial' },
                { id: 'yaw_plus_roll', label: 'Yaw + roll changed', value: 'partial' },
                { id: 'all_axes', label: 'All three axes changed', value: 'all_coupled' },
                { id: 'wrong_axis', label: 'Pitch or roll changed but NOT yaw', value: 'wrong_primary' }
            ]
        }
    },

    YAW_CCW_90: {
        id: 'YAW_CCW_90',
        name: 'Yaw Counter-Clockwise 90°',
        description: 'Rotate device counter-clockwise 90° while keeping flat',
        instructions: [
            'Start from flat position',
            'Keep device flat (parallel to floor)',
            'Rotate COUNTER-CLOCKWISE 90° (viewed from above)',
            'Fingers should now point to YOUR RIGHT'
        ],
        physicalState: {
            deviceOrientation: 'horizontal, rotated CCW',
            primaryAxis: 'yaw',
            expectedAngles: { roll: 0, pitch: 0, yaw: -90 }
        },
        expectedVisual: {
            palmFacing: 'up',
            fingerPointing: 'right',
            thumbPosition: 'toward'
        },
        primaryAxis: 'yaw',
        validationQuestions: [
            {
                id: 'rotation_direction',
                question: 'Which way do the fingers point after rotation?',
                type: 'direction',
                axis: 'yaw',
                options: [
                    { id: 'right_correct', label: 'RIGHT (correct)', value: 'correct' },
                    { id: 'left_opposite', label: 'LEFT (opposite)', value: 'opposite' },
                    { id: 'up_wrong', label: 'UP/DOWN instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'tilt_wrong', label: 'Hand tilted left/right instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'no_change', label: 'No change from flat', value: 'no_movement' },
                    { id: 'multiple', label: 'Multiple movements at once', value: 'coupled' }
                ]
            }
        ],
        couplingQuestion: {
            id: 'coupling_yaw_ccw',
            question: 'Did you observe movement on other axes?',
            options: [
                { id: 'only_yaw', label: 'Only yaw changed (rotation)', value: 'none' },
                { id: 'yaw_plus_pitch', label: 'Yaw + pitch changed', value: 'partial' },
                { id: 'yaw_plus_roll', label: 'Yaw + roll changed', value: 'partial' },
                { id: 'all_axes', label: 'All three axes changed', value: 'all_coupled' },
                { id: 'wrong_axis', label: 'Pitch or roll changed but NOT yaw', value: 'wrong_primary' }
            ]
        }
    },

    VERTICAL_FINGERS_UP: {
        id: 'VERTICAL_FINGERS_UP',
        name: 'Vertical, Fingers Up',
        description: 'Device standing on wrist edge, fingers pointing at ceiling',
        instructions: [
            'Stand device on its wrist edge',
            'Fingers should point straight UP at ceiling',
            'Palm should face toward you'
        ],
        physicalState: {
            deviceOrientation: 'vertical',
            primaryAxis: 'pitch',
            expectedAngles: { roll: 0, pitch: 90, yaw: 'any' }
        },
        expectedVisual: {
            palmFacing: 'toward',
            fingerPointing: 'up',
            thumbPosition: 'left'
        },
        primaryAxis: 'pitch',
        validationQuestions: [
            {
                id: 'vertical_direction',
                question: 'Which way do the fingers point?',
                type: 'direction',
                axis: 'pitch',
                options: [
                    { id: 'up_correct', label: 'UP at ceiling (correct)', value: 'correct' },
                    { id: 'down_opposite', label: 'DOWN at floor (opposite)', value: 'opposite' },
                    { id: 'sideways_wrong', label: 'Sideways instead (wrong axis)', value: 'wrong_axis' },
                    { id: 'no_change', label: 'Same as flat pose', value: 'no_movement' },
                    { id: 'erratic', label: 'Erratic/unstable (gimbal lock?)', value: 'coupled' }
                ]
            }
        ],
        couplingQuestion: {
            id: 'coupling_vertical',
            question: 'At 90° pitch, is behavior stable?',
            options: [
                { id: 'stable', label: 'Stable - fingers point up consistently', value: 'none' },
                { id: 'jittery', label: 'Jittery - small oscillations', value: 'partial' },
                { id: 'roll_yaw_swap', label: 'Roll and yaw seem swapped/coupled', value: 'swapped' },
                { id: 'unstable', label: 'Unstable - jumps between orientations', value: 'all_coupled' }
            ]
        }
    }
};

/**
 * Create an observation record (v2 with enhanced feedback)
 *
 * @param {string} poseId - Reference pose ID
 * @param {Object} sensorData - Raw sensor readings
 * @param {Object} ahrsOutput - AHRS filter output
 * @param {Object} renderState - Applied Three.js rotation
 * @param {Object} userAnswers - User's answers to validation questions (multi-choice)
 * @param {Object} mappingConfig - Current axis sign and offset configuration
 * @param {Object} baselineAhrs - Optional AHRS from FLAT_PALM_UP for delta analysis
 * @returns {Object} Complete observation record
 */
export function createObservation(poseId, sensorData, ahrsOutput, renderState, userAnswers, mappingConfig, baselineAhrs = null) {
    const pose = REFERENCE_POSES[poseId];
    if (!pose) {
        throw new Error(`Unknown pose: ${poseId}`);
    }

    const timestamp = Date.now();
    const isoTimestamp = new Date(timestamp).toISOString();

    // Calculate deltas from baseline if provided
    let deltaFromBaseline = null;
    if (baselineAhrs) {
        deltaFromBaseline = {
            roll: ahrsOutput.roll - baselineAhrs.roll,
            pitch: ahrsOutput.pitch - baselineAhrs.pitch,
            yaw: ahrsOutput.yaw - baselineAhrs.yaw
        };
    }

    // Analyze axis coupling from AHRS data
    const couplingAnalysis = analyzeAxisCoupling(pose, ahrsOutput, deltaFromBaseline);

    return {
        // Metadata
        meta: {
            version: '2.0.0',  // Updated version for new format
            timestamp,
            isoTimestamp,
            poseId,
            poseName: pose.name
        },

        // Reference pose definition
        referencePose: {
            id: pose.id,
            name: pose.name,
            description: pose.description,
            physicalState: pose.physicalState,
            expectedVisual: pose.expectedVisual,
            primaryAxis: pose.primaryAxis
        },

        // Raw sensor data
        sensor: {
            accelerometer: {
                x: sensorData.ax,
                y: sensorData.ay,
                z: sensorData.az,
                unit: sensorData.accelUnit || 'unknown'
            },
            gyroscope: {
                x: sensorData.gx,
                y: sensorData.gy,
                z: sensorData.gz,
                unit: sensorData.gyroUnit || 'unknown'
            },
            magnetometer: sensorData.mx !== undefined ? {
                x: sensorData.mx,
                y: sensorData.my,
                z: sensorData.mz,
                unit: sensorData.magUnit || 'unknown'
            } : null
        },

        // AHRS filter output
        ahrs: {
            euler: {
                roll: ahrsOutput.roll,
                pitch: ahrsOutput.pitch,
                yaw: ahrsOutput.yaw,
                unit: 'degrees'
            },
            quaternion: ahrsOutput.quaternion || null,
            deltaFromBaseline
        },

        // Current mapping configuration
        mapping: {
            axisSigns: {
                negateRoll: mappingConfig.negateRoll,
                negatePitch: mappingConfig.negatePitch,
                negateYaw: mappingConfig.negateYaw
            },
            offsets: {
                roll: mappingConfig.rollOffset,
                pitch: mappingConfig.pitchOffset,
                yaw: mappingConfig.yawOffset
            },
            eulerOrder: mappingConfig.eulerOrder || 'YXZ'
        },

        // Rendered state (what was actually applied to Three.js)
        render: {
            euler: {
                x: renderState.x,  // pitch in radians
                y: renderState.y,  // yaw in radians
                z: renderState.z,  // roll in radians
                order: renderState.order
            },
            eulerDegrees: {
                x: renderState.x * 180 / Math.PI,
                y: renderState.y * 180 / Math.PI,
                z: renderState.z * 180 / Math.PI
            }
        },

        // User observations (v2 multi-choice format)
        userObservations: {
            answers: userAnswers,
            couplingAnswer: userAnswers.coupling || null,
            derivedState: deriveStateFromAnswersV2(userAnswers, pose)
        },

        // Analysis
        analysis: {
            ...analyzeObservationV2(pose, ahrsOutput, userAnswers),
            coupling: couplingAnalysis
        }
    };
}

/**
 * Analyze axis coupling from AHRS data
 * Determines if rotating around one physical axis causes unexpected changes in other Euler angles
 */
function analyzeAxisCoupling(pose, ahrsOutput, deltaFromBaseline) {
    if (!pose.primaryAxis || !deltaFromBaseline) {
        return { type: COUPLING_TYPES.NONE, details: 'No baseline or not a movement pose' };
    }

    const SIGNIFICANT_THRESHOLD = 15;  // degrees
    const PRIMARY_THRESHOLD = 20;      // expected primary axis should move more than this

    const deltas = {
        roll: Math.abs(deltaFromBaseline.roll),
        pitch: Math.abs(deltaFromBaseline.pitch),
        yaw: Math.abs(deltaFromBaseline.yaw)
    };

    const primaryAxis = pose.primaryAxis;
    const otherAxes = ['roll', 'pitch', 'yaw'].filter(a => a !== primaryAxis);

    const primaryMoved = deltas[primaryAxis] > PRIMARY_THRESHOLD;
    const other0Moved = deltas[otherAxes[0]] > SIGNIFICANT_THRESHOLD;
    const other1Moved = deltas[otherAxes[1]] > SIGNIFICANT_THRESHOLD;

    // Determine coupling type
    let type;
    let details;

    if (!primaryMoved && !other0Moved && !other1Moved) {
        type = COUPLING_TYPES.NONE;
        details = 'No significant movement detected';
    } else if (primaryMoved && !other0Moved && !other1Moved) {
        type = COUPLING_TYPES.NONE;
        details = `Only ${primaryAxis} changed as expected`;
    } else if (!primaryMoved && (other0Moved || other1Moved)) {
        type = COUPLING_TYPES.WRONG_PRIMARY;
        const movedAxes = otherAxes.filter((a, i) => i === 0 ? other0Moved : other1Moved);
        details = `Expected ${primaryAxis} to move, but ${movedAxes.join(' and ')} moved instead`;
    } else if (primaryMoved && (other0Moved || other1Moved)) {
        if (other0Moved && other1Moved) {
            type = COUPLING_TYPES.ALL_COUPLED;
            details = `All three axes moved when only ${primaryAxis} should have`;
        } else {
            type = COUPLING_TYPES.PARTIAL;
            const coupledAxis = other0Moved ? otherAxes[0] : otherAxes[1];
            details = `${primaryAxis} moved as expected, but ${coupledAxis} also moved (coupling)`;
        }
    } else {
        type = COUPLING_TYPES.NONE;
        details = 'Unknown state';
    }

    return {
        type,
        details,
        deltas,
        expectedPrimaryAxis: primaryAxis,
        primaryAxisMoved: primaryMoved,
        otherAxesMoved: { [otherAxes[0]]: other0Moved, [otherAxes[1]]: other1Moved }
    };
}

/**
 * Derive axis inversion state from user answers (v2 multi-choice format)
 */
function deriveStateFromAnswersV2(answers, pose) {
    const state = {
        pitchInverted: null,
        rollInverted: null,
        yawInverted: null,
        pitchWrongAxis: false,
        rollWrongAxis: false,
        yawWrongAxis: false,
        coupling: null,
        matchesExpected: null
    };

    for (const q of pose.validationQuestions) {
        const answer = answers[q.id];
        if (answer === undefined || !q.axis) continue;

        // Multi-choice answers store the value directly
        if (answer === 'correct') {
            state[`${q.axis}Inverted`] = false;
        } else if (answer === 'opposite') {
            state[`${q.axis}Inverted`] = true;
        } else if (answer === 'wrong_axis') {
            state[`${q.axis}WrongAxis`] = true;
        } else if (answer === 'coupled') {
            state.coupling = 'detected';
        }
    }

    // Handle coupling question
    if (answers.coupling) {
        state.coupling = answers.coupling;
    }

    // Check if primary question answered correctly
    const primaryQuestion = pose.validationQuestions[0];
    if (primaryQuestion) {
        state.matchesExpected = answers[primaryQuestion.id] === 'correct';
    }

    return state;
}

/**
 * Analyze observation for calibration insights (v2 with coupling)
 */
function analyzeObservationV2(pose, ahrsOutput, userAnswers) {
    const analysis = {
        angleDeviations: {},
        axisIssues: [],
        suggestions: [],
        couplingDetected: false,
        primaryAxisStatus: null
    };

    // Check angle deviations from expected
    const expected = pose.physicalState.expectedAngles;
    if (expected) {
        if (typeof expected.roll === 'number') {
            analysis.angleDeviations.roll = ahrsOutput.roll - expected.roll;
        }
        if (typeof expected.pitch === 'number') {
            analysis.angleDeviations.pitch = ahrsOutput.pitch - expected.pitch;
        }
        if (typeof expected.yaw === 'number') {
            analysis.angleDeviations.yaw = ahrsOutput.yaw - expected.yaw;
        }
    }

    // Analyze multi-choice answers
    for (const q of pose.validationQuestions) {
        const answer = userAnswers[q.id];
        if (!answer || !q.axis) continue;

        if (answer === 'opposite') {
            analysis.axisIssues.push({
                axis: q.axis,
                issue: 'inverted',
                evidence: `User reported ${q.axis} shows opposite direction`
            });
            analysis.suggestions.push(`${q.axis.toUpperCase()} axis appears INVERTED - toggle negate${capitalize(q.axis)}`);
        } else if (answer === 'wrong_axis') {
            analysis.axisIssues.push({
                axis: q.axis,
                issue: 'wrong_axis',
                evidence: `User reported wrong axis responded to ${pose.primaryAxis} movement`
            });
            analysis.suggestions.push(`AXIS PERMUTATION suspected: ${pose.primaryAxis} movement affected ${q.axis} instead`);
            analysis.couplingDetected = true;
        } else if (answer === 'coupled') {
            analysis.couplingDetected = true;
            analysis.suggestions.push('COUPLING detected: Multiple axes respond to single-axis physical movement');
        } else if (answer === 'correct') {
            analysis.primaryAxisStatus = 'correct';
        }
    }

    // Analyze coupling question
    if (userAnswers.coupling) {
        const couplingValue = userAnswers.coupling;
        if (couplingValue !== 'none') {
            analysis.couplingDetected = true;
            if (couplingValue === 'wrong_primary') {
                analysis.suggestions.push('AXIS SWAP: The expected axis did NOT respond - consider Euler order or axis permutation');
            } else if (couplingValue === 'all_coupled') {
                analysis.suggestions.push('FULL COUPLING: All axes respond together - likely Euler order mismatch or sensor alignment issue');
            } else if (couplingValue === 'swapped') {
                analysis.suggestions.push('AXIS SWAP at gimbal lock: Roll and Yaw appear swapped near 90° pitch');
            }
        }
    }

    return analysis;
}

// Keep old functions for backwards compatibility
function deriveStateFromAnswers(answers, pose) {
    return deriveStateFromAnswersV2(answers, pose);
}

function analyzeObservation(pose, ahrsOutput, userAnswers) {
    return analyzeObservationV2(pose, ahrsOutput, userAnswers);
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

// =============================================================================
// QUATERNION DIAGNOSTICS
// =============================================================================
// These functions help diagnose axis mapping issues by working with quaternions
// directly, bypassing Euler angle representation problems.

/**
 * All 6 axis permutations for testing which mapping might be correct
 */
export const AXIS_PERMUTATIONS = [
    { name: 'XYZ (default)', map: { x: 'x', y: 'y', z: 'z' } },
    { name: 'XZY', map: { x: 'x', y: 'z', z: 'y' } },
    { name: 'YXZ', map: { x: 'y', y: 'x', z: 'z' } },
    { name: 'YZX', map: { x: 'y', y: 'z', z: 'x' } },
    { name: 'ZXY', map: { x: 'z', y: 'x', z: 'y' } },
    { name: 'ZYX', map: { x: 'z', y: 'y', z: 'x' } }
];

/**
 * All 12 Euler rotation orders
 */
export const EULER_ORDERS = [
    'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX',  // Proper Euler (intrinsic)
    'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ'   // Tait-Bryan
];

/**
 * Convert quaternion to Euler angles with specified order
 * This allows testing different extraction orders to find the correct one
 *
 * @param {Object} q - Quaternion {w, x, y, z}
 * @param {string} order - Euler order (e.g., 'YXZ', 'ZYX')
 * @returns {Object} {roll, pitch, yaw} in degrees
 */
export function quaternionToEuler(q, order = 'YXZ') {
    const { w, x, y, z } = q;

    // Normalize quaternion
    const n = Math.sqrt(w*w + x*x + y*y + z*z);
    const qn = { w: w/n, x: x/n, y: y/n, z: z/n };

    let roll, pitch, yaw;
    const rad2deg = 180 / Math.PI;

    // Different extraction formulas based on order
    // Using ZYX (aerospace) as reference implementation
    switch (order) {
        case 'ZYX': {
            // Roll (X), Pitch (Y), Yaw (Z)
            const sinr_cosp = 2 * (qn.w * qn.x + qn.y * qn.z);
            const cosr_cosp = 1 - 2 * (qn.x * qn.x + qn.y * qn.y);
            roll = Math.atan2(sinr_cosp, cosr_cosp);

            const sinp = 2 * (qn.w * qn.y - qn.z * qn.x);
            pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);

            const siny_cosp = 2 * (qn.w * qn.z + qn.x * qn.y);
            const cosy_cosp = 1 - 2 * (qn.y * qn.y + qn.z * qn.z);
            yaw = Math.atan2(siny_cosp, cosy_cosp);
            break;
        }
        case 'YXZ': {
            // Three.js default order
            const sinp = 2 * (qn.w * qn.x - qn.y * qn.z);
            pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);

            const siny = 2 * (qn.w * qn.y + qn.x * qn.z);
            const cosy = 1 - 2 * (qn.x * qn.x + qn.y * qn.y);
            yaw = Math.atan2(siny, cosy);

            const sinr = 2 * (qn.w * qn.z + qn.x * qn.y);
            const cosr = 1 - 2 * (qn.x * qn.x + qn.z * qn.z);
            roll = Math.atan2(sinr, cosr);
            break;
        }
        case 'XYZ': {
            const sinr = 2 * (qn.w * qn.x - qn.y * qn.z);
            roll = Math.abs(sinr) >= 1 ? Math.sign(sinr) * Math.PI / 2 : Math.asin(sinr);

            const sinp = 2 * (qn.w * qn.y + qn.x * qn.z);
            const cosp = 1 - 2 * (qn.x * qn.x + qn.y * qn.y);
            pitch = Math.atan2(sinp, cosp);

            const siny = 2 * (qn.w * qn.z - qn.x * qn.y);
            const cosy = 1 - 2 * (qn.y * qn.y + qn.z * qn.z);
            yaw = Math.atan2(siny, cosy);
            break;
        }
        default:
            // Fall back to ZYX
            return quaternionToEuler(q, 'ZYX');
    }

    return {
        roll: roll * rad2deg,
        pitch: pitch * rad2deg,
        yaw: yaw * rad2deg
    };
}

/**
 * Test all Euler extraction orders against observation data
 * Helps identify which Euler order the AHRS filter is using
 *
 * @param {Object} quaternion - Raw quaternion from AHRS
 * @param {Object} expectedAngles - Expected approximate angles {roll, pitch, yaw}
 * @returns {Array} Sorted array of {order, angles, error} by error
 */
export function testEulerOrders(quaternion, expectedAngles) {
    const results = [];

    for (const order of ['ZYX', 'YXZ', 'XYZ', 'XZY', 'YZX', 'ZXY']) {
        const angles = quaternionToEuler(quaternion, order);

        // Calculate error as sum of squared differences (ignoring yaw if 'any')
        let error = 0;
        if (typeof expectedAngles.roll === 'number') {
            error += Math.pow(angles.roll - expectedAngles.roll, 2);
        }
        if (typeof expectedAngles.pitch === 'number') {
            error += Math.pow(angles.pitch - expectedAngles.pitch, 2);
        }
        if (typeof expectedAngles.yaw === 'number') {
            error += Math.pow(angles.yaw - expectedAngles.yaw, 2);
        }

        results.push({
            order,
            angles,
            error: Math.sqrt(error)
        });
    }

    return results.sort((a, b) => a.error - b.error);
}

/**
 * Analyze quaternion for gimbal lock proximity
 * Gimbal lock occurs when pitch approaches ±90°
 *
 * @param {Object} q - Quaternion {w, x, y, z}
 * @returns {Object} Gimbal lock analysis
 */
export function analyzeGimbalLock(q) {
    // The gimbal lock singularity occurs when the pitch component
    // of the rotation approaches ±90°

    const { w, x, y, z } = q;

    // For ZYX order, gimbal lock occurs when sin(pitch) = ±1
    // sin(pitch) = 2(wy - zx)
    const sinPitch = 2 * (w * y - z * x);

    const isNearGimbalLock = Math.abs(sinPitch) > 0.99;
    const pitchAngle = Math.asin(Math.min(1, Math.max(-1, sinPitch))) * 180 / Math.PI;

    return {
        isNearGimbalLock,
        sinPitch,
        estimatedPitch: pitchAngle,
        warning: isNearGimbalLock ?
            'Near gimbal lock - roll and yaw may be unreliable' :
            'No gimbal lock concern'
    };
}

/**
 * Generate diagnostic report comparing AHRS output with physical expectations
 *
 * @param {Object} observation - Full observation record
 * @returns {Object} Diagnostic report
 */
export function generateDiagnosticReport(observation) {
    const report = {
        timestamp: new Date().toISOString(),
        pose: observation.referencePose.id,
        issues: [],
        recommendations: []
    };

    // Check for coupling
    if (observation.analysis?.coupling?.type !== COUPLING_TYPES.NONE) {
        report.issues.push({
            type: 'coupling',
            severity: 'high',
            details: observation.analysis.coupling.details
        });
    }

    // If quaternion available, test Euler orders
    if (observation.ahrs?.quaternion) {
        const expected = observation.referencePose.physicalState.expectedAngles;
        const eulerTests = testEulerOrders(observation.ahrs.quaternion, expected);

        if (eulerTests.length > 0 && eulerTests[0].error < eulerTests[1].error * 0.5) {
            report.recommendations.push({
                type: 'euler_order',
                recommendation: `Consider using Euler order: ${eulerTests[0].order}`,
                confidence: 'high',
                details: eulerTests.slice(0, 3)
            });
        }

        // Check gimbal lock
        const gimbalAnalysis = analyzeGimbalLock(observation.ahrs.quaternion);
        if (gimbalAnalysis.isNearGimbalLock) {
            report.issues.push({
                type: 'gimbal_lock',
                severity: 'medium',
                details: gimbalAnalysis.warning
            });
        }
    }

    // Check user-reported issues
    if (observation.userObservations?.derivedState) {
        const derived = observation.userObservations.derivedState;

        if (derived.coupling && derived.coupling !== 'none') {
            report.issues.push({
                type: 'user_reported_coupling',
                severity: 'high',
                details: `User observed ${derived.coupling} coupling behavior`
            });
        }

        for (const axis of ['pitch', 'roll', 'yaw']) {
            if (derived[`${axis}WrongAxis`]) {
                report.issues.push({
                    type: 'wrong_axis',
                    severity: 'high',
                    details: `${axis} movement triggered wrong axis response`
                });
                report.recommendations.push({
                    type: 'axis_permutation',
                    recommendation: `Test axis permutations - ${axis} may be mapped to different axis`,
                    confidence: 'medium'
                });
            }
        }
    }

    return report;
}

/**
 * Observation store - collects multiple observations
 */
export class ObservationStore {
    constructor() {
        this.observations = [];
        this.sessionId = `cal_${Date.now()}`;
    }

    add(observation) {
        this.observations.push(observation);
        return this.observations.length - 1;
    }

    getAll() {
        return this.observations;
    }

    clear() {
        this.observations = [];
    }

    /**
     * Analyze all observations to derive calibration
     */
    analyzeAll() {
        const axisVotes = {
            pitch: { inverted: 0, correct: 0 },
            roll: { inverted: 0, correct: 0 },
            yaw: { inverted: 0, correct: 0 }
        };

        for (const obs of this.observations) {
            const derived = obs.userObservations.derivedState;
            for (const axis of ['pitch', 'roll', 'yaw']) {
                if (derived[`${axis}Inverted`] === true) {
                    axisVotes[axis].inverted++;
                } else if (derived[`${axis}Inverted`] === false) {
                    axisVotes[axis].correct++;
                }
            }
        }

        const recommendation = {
            negatePitch: axisVotes.pitch.inverted > axisVotes.pitch.correct,
            negateRoll: axisVotes.roll.inverted > axisVotes.roll.correct,
            negateYaw: axisVotes.yaw.inverted > axisVotes.yaw.correct,
            confidence: {
                pitch: Math.abs(axisVotes.pitch.inverted - axisVotes.pitch.correct) / Math.max(1, axisVotes.pitch.inverted + axisVotes.pitch.correct),
                roll: Math.abs(axisVotes.roll.inverted - axisVotes.roll.correct) / Math.max(1, axisVotes.roll.inverted + axisVotes.roll.correct),
                yaw: Math.abs(axisVotes.yaw.inverted - axisVotes.yaw.correct) / Math.max(1, axisVotes.yaw.inverted + axisVotes.yaw.correct)
            },
            votes: axisVotes,
            totalObservations: this.observations.length
        };

        return recommendation;
    }

    /**
     * Export all observations as JSON for later analysis
     */
    export() {
        return {
            sessionId: this.sessionId,
            exportedAt: new Date().toISOString(),
            observationCount: this.observations.length,
            observations: this.observations,
            analysis: this.analyzeAll()
        };
    }

    /**
     * Import previously exported observations
     */
    import(data) {
        if (data.observations) {
            this.observations = data.observations;
            this.sessionId = data.sessionId || this.sessionId;
        }
    }
}

/**
 * Get ordered list of calibration poses for systematic calibration
 */
export function getCalibrationSequence() {
    return [
        'FLAT_PALM_UP',      // Baseline
        'PITCH_FORWARD_45',  // Test pitch sign
        'PITCH_BACKWARD_45', // Confirm pitch sign
        'ROLL_LEFT_45',      // Test roll sign
        'ROLL_RIGHT_45',     // Confirm roll sign
        'YAW_CW_90',         // Test yaw sign
        'YAW_CCW_90'         // Confirm yaw sign
    ];
}

export default {
    REFERENCE_POSES,
    ANSWER_OPTIONS,
    COUPLING_TYPES,
    AXIS_PERMUTATIONS,
    EULER_ORDERS,
    createObservation,
    ObservationStore,
    getCalibrationSequence,
    quaternionToEuler,
    testEulerOrders,
    analyzeGimbalLock,
    generateDiagnosticReport
};
