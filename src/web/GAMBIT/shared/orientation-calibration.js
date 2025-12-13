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
 *
 * With 3 axes, each having 2 possible signs, and 6 possible permutations,
 * there are 6 × 2³ = 48 possible axis mappings per transformation stage.
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
        validationQuestions: [
            { id: 'palm_up', question: 'Palm faces UP (toward ceiling)?', axis: null },
            { id: 'fingers_away', question: 'Fingers point AWAY from you?', axis: null },
            { id: 'thumb_left', question: 'Thumb is on LEFT side?', axis: null }
        ]
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
        validationQuestions: [
            { id: 'fingers_down', question: 'Fingers point DOWN toward floor?', axis: 'pitch', expectedSign: 'correct' },
            { id: 'fingers_up_wrong', question: 'Fingers point UP (wrong)?', axis: 'pitch', expectedSign: 'inverted' }
        ]
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
        validationQuestions: [
            { id: 'fingers_up', question: 'Fingers point UP toward ceiling?', axis: 'pitch', expectedSign: 'correct' },
            { id: 'fingers_down_wrong', question: 'Fingers point DOWN (wrong)?', axis: 'pitch', expectedSign: 'inverted' }
        ]
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
        validationQuestions: [
            { id: 'pinky_down', question: 'Pinky side is DOWN, thumb side UP?', axis: 'roll', expectedSign: 'correct' },
            { id: 'pinky_up_wrong', question: 'Pinky side is UP (opposite)?', axis: 'roll', expectedSign: 'inverted' }
        ]
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
        validationQuestions: [
            { id: 'thumb_down', question: 'Thumb side is DOWN, pinky side UP?', axis: 'roll', expectedSign: 'correct' },
            { id: 'thumb_up_wrong', question: 'Thumb side is UP (opposite)?', axis: 'roll', expectedSign: 'inverted' }
        ]
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
        validationQuestions: [
            { id: 'fingers_left', question: 'Fingers point to YOUR LEFT?', axis: 'yaw', expectedSign: 'correct' },
            { id: 'fingers_right_wrong', question: 'Fingers point to YOUR RIGHT (opposite)?', axis: 'yaw', expectedSign: 'inverted' }
        ]
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
        validationQuestions: [
            { id: 'fingers_right', question: 'Fingers point to YOUR RIGHT?', axis: 'yaw', expectedSign: 'correct' },
            { id: 'fingers_left_wrong', question: 'Fingers point to YOUR LEFT (opposite)?', axis: 'yaw', expectedSign: 'inverted' }
        ]
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
        validationQuestions: [
            { id: 'fingers_ceiling', question: 'Fingers point at CEILING?', axis: 'pitch', expectedSign: 'correct' },
            { id: 'fingers_floor_wrong', question: 'Fingers point at FLOOR (opposite)?', axis: 'pitch', expectedSign: 'inverted' }
        ]
    }
};

/**
 * Create an observation record
 *
 * @param {string} poseId - Reference pose ID
 * @param {Object} sensorData - Raw sensor readings
 * @param {Object} ahrsOutput - AHRS filter output
 * @param {Object} renderState - Applied Three.js rotation
 * @param {Object} userAnswers - User's answers to validation questions
 * @param {Object} mappingConfig - Current axis sign and offset configuration
 * @returns {Object} Complete observation record
 */
export function createObservation(poseId, sensorData, ahrsOutput, renderState, userAnswers, mappingConfig) {
    const pose = REFERENCE_POSES[poseId];
    if (!pose) {
        throw new Error(`Unknown pose: ${poseId}`);
    }

    const timestamp = Date.now();
    const isoTimestamp = new Date(timestamp).toISOString();

    return {
        // Metadata
        meta: {
            version: '1.0.0',
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
            expectedVisual: pose.expectedVisual
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
            quaternion: ahrsOutput.quaternion || null
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

        // User observations
        userObservations: {
            answers: userAnswers,
            derivedState: deriveStateFromAnswers(userAnswers, pose)
        },

        // Analysis
        analysis: analyzeObservation(pose, ahrsOutput, userAnswers)
    };
}

/**
 * Derive axis inversion state from user answers
 */
function deriveStateFromAnswers(answers, pose) {
    const state = {
        pitchInverted: null,
        rollInverted: null,
        yawInverted: null,
        matchesExpected: null
    };

    for (const q of pose.validationQuestions) {
        const answer = answers[q.id];
        if (answer === undefined) continue;

        if (q.axis && q.expectedSign) {
            if (answer === true) {
                // User said yes to this question
                if (q.expectedSign === 'correct') {
                    state[`${q.axis}Inverted`] = false;
                } else if (q.expectedSign === 'inverted') {
                    state[`${q.axis}Inverted`] = true;
                }
            }
        }
    }

    // Check if all expected answers were positive
    state.matchesExpected = pose.validationQuestions
        .filter(q => q.expectedSign === 'correct')
        .every(q => answers[q.id] === true);

    return state;
}

/**
 * Analyze observation for calibration insights
 */
function analyzeObservation(pose, ahrsOutput, userAnswers) {
    const analysis = {
        angleDeviations: {},
        axisIssues: [],
        suggestions: []
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

    // Identify axis issues from user answers
    for (const q of pose.validationQuestions) {
        const answer = userAnswers[q.id];
        if (answer === true && q.expectedSign === 'inverted') {
            analysis.axisIssues.push({
                axis: q.axis,
                issue: 'inverted',
                evidence: q.question
            });
            analysis.suggestions.push(`${q.axis.toUpperCase()} axis appears INVERTED - toggle negate${capitalize(q.axis)}`);
        }
    }

    return analysis;
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
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
    createObservation,
    ObservationStore,
    getCalibrationSequence
};
