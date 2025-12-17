/**
 * Orientation Model Tests
 *
 * Validates the sensor-to-hand orientation mapping.
 * Can be run in browser console or via Node.js.
 *
 * To run in browser:
 *   1. Open GAMBIT/index.html
 *   2. Open browser console
 *   3. Import and run: import('./shared/orientation-model.test.ts')
 *
 * @module shared/orientation-model.test
 */

import {
  ORIENTATION_CONFIG,
  mapSensorToHand,
  mapSensorToThreeJS,
  describeExpectedPose,
  validateMapping,
  type EulerAngles,
  type OrientationConfig,
  type ThreeJSEuler
} from './orientation-model';

// ===== Type Definitions =====

interface TestCase {
  name: string;
  sensor: EulerAngles;
}

interface ValidationResult {
  name: string;
  description: string;
  input: EulerAngles;
  output: EulerAngles;
  physicalPose: string;
  validation: string[];
}

// ===== Test Utilities =====

const assert = (condition: boolean, message: string): boolean => {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    return false;
  }
  console.log(`PASS: ${message}`);
  return true;
};

const assertApprox = (actual: number, expected: number, tolerance: number, message: string): boolean => {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    console.error(`FAIL: ${message} (got ${actual}, expected ${expected}, diff ${diff})`);
    return false;
  }
  console.log(`PASS: ${message} (${actual} ≈ ${expected})`);
  return true;
};

/**
 * Test 1: Neutral position mapping
 * When sensor is flat (roll=0, pitch=0, yaw=0), hand should be palm-up
 */
function testNeutralPosition(): void {
  console.log('\n=== Test: Neutral Position ===');
  const sensor: EulerAngles = { roll: 0, pitch: 0, yaw: 0 };
  const hand = mapSensorToHand(sensor);

  console.log('Sensor:', sensor);
  console.log('Hand:', hand);

  // With new mapping: roll = 0 + 180 = 180, pitch = -0 + 180 = 180, yaw = 0 - 180 = -180
  assertApprox(hand.roll, 180, 0.1, 'Neutral roll should be 180°');
  assertApprox(hand.pitch, 180, 0.1, 'Neutral pitch should be 180°');
  assertApprox(hand.yaw, -180, 0.1, 'Neutral yaw should be -180°');
}

/**
 * Test 2: Forward tilt mapping
 * When sensor tips forward (negative pitch), hand should show fingers pointing down
 */
function testForwardTilt(): void {
  console.log('\n=== Test: Forward Tilt (30°) ===');
  const sensor: EulerAngles = { roll: 0, pitch: -30, yaw: 0 };
  const hand = mapSensorToHand(sensor);

  console.log('Sensor:', sensor);
  console.log('Hand:', hand);

  // With pitch negation: hand_pitch = -(-30) + 180 = 210
  // This should make fingers point downward
  assertApprox(hand.pitch, 210, 0.1, 'Forward tilt should increase hand pitch to 210°');
  assertApprox(hand.roll, 180, 0.1, 'Roll should stay at 180°');
}

/**
 * Test 3: Backward tilt mapping
 * When sensor tips backward (positive pitch), hand should show fingers pointing up
 */
function testBackwardTilt(): void {
  console.log('\n=== Test: Backward Tilt (30°) ===');
  const sensor: EulerAngles = { roll: 0, pitch: 30, yaw: 0 };
  const hand = mapSensorToHand(sensor);

  console.log('Sensor:', sensor);
  console.log('Hand:', hand);

  // With pitch negation: hand_pitch = -(30) + 180 = 150
  // This should make fingers point upward
  assertApprox(hand.pitch, 150, 0.1, 'Backward tilt should decrease hand pitch to 150°');
}

/**
 * Test 4: Left tilt mapping
 * When sensor tilts left (negative roll), pinky side should go down
 */
function testLeftTilt(): void {
  console.log('\n=== Test: Left Tilt (30°) ===');
  const sensor: EulerAngles = { roll: -30, pitch: 0, yaw: 0 };
  const hand = mapSensorToHand(sensor);

  console.log('Sensor:', sensor);
  console.log('Hand:', hand);

  // With roll NOT negated: hand_roll = -30 + 180 = 150
  assertApprox(hand.roll, 150, 0.1, 'Left tilt should decrease hand roll to 150°');
}

/**
 * Test 5: Right tilt mapping
 * When sensor tilts right (positive roll), thumb side should go down
 */
function testRightTilt(): void {
  console.log('\n=== Test: Right Tilt (30°) ===');
  const sensor: EulerAngles = { roll: 30, pitch: 0, yaw: 0 };
  const hand = mapSensorToHand(sensor);

  console.log('Sensor:', sensor);
  console.log('Hand:', hand);

  // With roll NOT negated: hand_roll = 30 + 180 = 210
  assertApprox(hand.roll, 210, 0.1, 'Right tilt should increase hand roll to 210°');
}

/**
 * Test 6: Clockwise rotation mapping
 * When sensor rotates clockwise (positive yaw), hand should rotate clockwise
 */
function testClockwiseRotation(): void {
  console.log('\n=== Test: Clockwise Rotation (45°) ===');
  const sensor: EulerAngles = { roll: 0, pitch: 0, yaw: 45 };
  const hand = mapSensorToHand(sensor);

  console.log('Sensor:', sensor);
  console.log('Hand:', hand);

  // Yaw is not negated: hand_yaw = 45 - 180 = -135
  assertApprox(hand.yaw, -135, 0.1, 'Clockwise rotation should change yaw to -135°');
}

/**
 * Test 7: Three.js format output
 * Verify the output is in correct format for Three.js
 */
function testThreeJSFormat(): void {
  console.log('\n=== Test: Three.js Format ===');
  const sensor: EulerAngles = { roll: 30, pitch: -45, yaw: 60 };
  const result = mapSensorToThreeJS(sensor);

  console.log('Sensor:', sensor);
  console.log('Three.js:', result);

  assert(typeof result.x === 'number', 'x should be a number');
  assert(typeof result.y === 'number', 'y should be a number');
  assert(typeof result.z === 'number', 'z should be a number');
  assert(result.order === 'YXZ', 'Euler order should be YXZ');

  // Check values are in radians
  const deg2rad = Math.PI / 180;
  const expectedPitch = (-(-45) + 180) * deg2rad; // x = pitch
  const expectedYaw = (60 - 180) * deg2rad;       // y = yaw
  const expectedRoll = (30 + 180) * deg2rad;      // z = roll

  assertApprox(result.x, expectedPitch, 0.01, `x (pitch) should be ${expectedPitch.toFixed(3)}`);
  assertApprox(result.y, expectedYaw, 0.01, `y (yaw) should be ${expectedYaw.toFixed(3)}`);
  assertApprox(result.z, expectedRoll, 0.01, `z (roll) should be ${expectedRoll.toFixed(3)}`);
}

/**
 * Test 8: Describe expected pose
 * Verify human-readable descriptions are generated correctly
 */
function testDescribeExpectedPose(): void {
  console.log('\n=== Test: Describe Expected Pose ===');

  const poses: EulerAngles[] = [
    { roll: 0, pitch: 0, yaw: 0 },
    { roll: 30, pitch: 0, yaw: 0 },
    { roll: 0, pitch: -30, yaw: 0 },
    { roll: 0, pitch: 0, yaw: 90 }
  ];

  for (const pose of poses) {
    const desc = describeExpectedPose(pose);
    console.log(`\nSensor (roll=${pose.roll}, pitch=${pose.pitch}, yaw=${pose.yaw}):`);
    console.log('  Physical:', desc.physical);
    console.log('  Expected:', desc.expected);
  }
}

/**
 * Test 9: Validation test cases
 * Run all predefined test cases and display results
 */
function testValidationCases(): void {
  console.log('\n=== Test: Validation Cases ===');
  const results = validateMapping(mapSensorToHand) as ValidationResult[];

  for (const result of results) {
    console.log(`\n[${result.name}] ${result.description}`);
    console.log(`  Input:  roll=${result.input.roll}°, pitch=${result.input.pitch}°, yaw=${result.input.yaw}°`);
    console.log(`  Output: roll=${result.output.roll.toFixed(1)}°, pitch=${result.output.pitch.toFixed(1)}°, yaw=${result.output.yaw.toFixed(1)}°`);
    console.log(`  Physical: ${result.physicalPose}`);
    console.log('  Validate:');
    for (const v of result.validation) {
      console.log(`    - ${v}`);
    }
  }
}

/**
 * Test 10: Compare old vs new mapping
 * Show the difference between the old (inverted) and new (corrected) mapping
 */
function testOldVsNewMapping(): void {
  console.log('\n=== Test: Old vs New Mapping Comparison ===');

  // Old config (from threejs-hand-skeleton.js before fix)
  const oldConfig: OrientationConfig = {
    negateRoll: true,    // Was negated
    negatePitch: false,  // Was not negated
    negateYaw: false,
    rollOffset: 180,
    pitchOffset: 180,
    yawOffset: -180,
    eulerOrder: 'YXZ'
  };

  // New config (corrected)
  const newConfig = ORIENTATION_CONFIG;

  const testCases: TestCase[] = [
    { name: 'Neutral', sensor: { roll: 0, pitch: 0, yaw: 0 } },
    { name: 'Forward 30°', sensor: { roll: 0, pitch: -30, yaw: 0 } },
    { name: 'Left 30°', sensor: { roll: -30, pitch: 0, yaw: 0 } },
    { name: 'CW Rotate 45°', sensor: { roll: 0, pitch: 0, yaw: 45 } }
  ];

  console.log('\n  Sensor Input        |     OLD Mapping        |     NEW Mapping');
  console.log('  --------------------|------------------------|------------------------');

  for (const tc of testCases) {
    const old = mapSensorToHand(tc.sensor, oldConfig);
    const neu = mapSensorToHand(tc.sensor, newConfig);

    const sensorStr = `r=${tc.sensor.roll}, p=${tc.sensor.pitch}, y=${tc.sensor.yaw}`.padEnd(18);
    const oldStr = `r=${old.roll.toFixed(0)}, p=${old.pitch.toFixed(0)}, y=${old.yaw.toFixed(0)}`.padEnd(22);
    const newStr = `r=${neu.roll.toFixed(0)}, p=${neu.pitch.toFixed(0)}, y=${neu.yaw.toFixed(0)}`;

    console.log(`  ${sensorStr} | ${oldStr} | ${newStr}`);
  }

  console.log('\nNote: The key differences are:');
  console.log('  - Roll: old=NEGATE, new=KEEP (fixes left/right tilt inversion)');
  console.log('  - Pitch: old=KEEP, new=NEGATE (fixes forward/back tilt inversion)');
}

// Run all tests
export function runAllTests(): void {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║         ORIENTATION MODEL VALIDATION TESTS                     ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  testNeutralPosition();
  testForwardTilt();
  testBackwardTilt();
  testLeftTilt();
  testRightTilt();
  testClockwiseRotation();
  testThreeJSFormat();
  testDescribeExpectedPose();
  testValidationCases();
  testOldVsNewMapping();

  console.log('\n╔════════════════════════════════════════════════════════════════╗');
  console.log('║                    TESTS COMPLETE                              ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
}

// Auto-run if executed directly
declare global {
  interface Window {
    runOrientationTests: typeof runAllTests;
  }
}

if (typeof window !== 'undefined') {
  console.log('Run tests with: runOrientationTests()');
  window.runOrientationTests = runAllTests;
} else {
  runAllTests();
}

export default runAllTests;
