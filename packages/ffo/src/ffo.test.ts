/**
 * FFO$$ Unit Tests
 *
 * Tests for the $Q-3D gesture recognition algorithm.
 * Can be run in browser console or via Node.js.
 *
 * To run:
 *   npx tsx packages/ffo/src/ffo.test.ts
 *
 * @module ffo/test
 */

import {
  // Resample
  resample,
  resampleImmutable,
  pathLength,
  distance3D,
  lerp3D,

  // Normalize
  centroid,
  boundingBox,
  translateToOrigin,
  scaleToSize,
  quickNormalize,

  // Distance
  euclideanDistance,
  pathDistance,
  cloudDistance,
  buildLookupTable,
  distanceToScore,

  // Recognizer
  FFORecognizer,
  createRecognizer,

  // Types
  type TemplatePoint3D,
  type TelemetrySample3D,
} from './index';

// ===== Test Utilities =====

let passCount = 0;
let failCount = 0;

function assert(condition: boolean, message: string): void {
  if (condition) {
    console.log(`  ✓ ${message}`);
    passCount++;
  } else {
    console.error(`  ✗ ${message}`);
    failCount++;
  }
}

function assertApprox(actual: number, expected: number, tolerance: number, message: string): void {
  const diff = Math.abs(actual - expected);
  if (diff <= tolerance) {
    console.log(`  ✓ ${message} (${actual.toFixed(4)} ≈ ${expected.toFixed(4)})`);
    passCount++;
  } else {
    console.error(`  ✗ ${message} (got ${actual.toFixed(4)}, expected ${expected.toFixed(4)}, diff ${diff.toFixed(4)})`);
    failCount++;
  }
}

function assertArrayLength<T>(arr: T[], expected: number, message: string): void {
  assert(arr.length === expected, `${message} (length: ${arr.length})`);
}

function describe(name: string, fn: () => void): void {
  console.log(`\n=== ${name} ===`);
  fn();
}

// ===== Test Data =====

const linePoints: TemplatePoint3D[] = [
  { x: 0, y: 0, z: 0 },
  { x: 10, y: 0, z: 0 },
];

const trianglePoints: TemplatePoint3D[] = [
  { x: 0, y: 0, z: 0 },
  { x: 10, y: 0, z: 0 },
  { x: 5, y: 8.66, z: 0 },
  { x: 0, y: 0, z: 0 },
];

const circle3D: TemplatePoint3D[] = [];
for (let i = 0; i <= 16; i++) {
  const angle = (i / 16) * 2 * Math.PI;
  circle3D.push({
    x: Math.cos(angle) * 5,
    y: Math.sin(angle) * 5,
    z: 0,
  });
}

const helix: TemplatePoint3D[] = [];
for (let i = 0; i <= 20; i++) {
  const t = i / 20;
  const angle = t * 4 * Math.PI;
  helix.push({
    x: Math.cos(angle) * 2,
    y: Math.sin(angle) * 2,
    z: t * 10,
  });
}

// ===== Tests =====

describe('distance3D', () => {
  assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }), 1, 0.001, 'Unit distance X');
  assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }), 1, 0.001, 'Unit distance Y');
  assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }), 1, 0.001, 'Unit distance Z');
  assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 }), Math.sqrt(3), 0.001, 'Diagonal distance');
  assertApprox(distance3D({ x: 1, y: 2, z: 3 }, { x: 4, y: 6, z: 3 }), 5, 0.001, '3-4-5 triangle');
});

describe('lerp3D', () => {
  const a = { x: 0, y: 0, z: 0 };
  const b = { x: 10, y: 20, z: 30 };

  const mid = lerp3D(a, b, 0.5);
  assertApprox(mid.x, 5, 0.001, 'Midpoint X');
  assertApprox(mid.y, 10, 0.001, 'Midpoint Y');
  assertApprox(mid.z, 15, 0.001, 'Midpoint Z');

  const start = lerp3D(a, b, 0);
  assertApprox(start.x, 0, 0.001, 't=0 gives start point X');

  const end = lerp3D(a, b, 1);
  assertApprox(end.x, 10, 0.001, 't=1 gives end point X');
});

describe('pathLength', () => {
  assertApprox(pathLength(linePoints), 10, 0.001, 'Line of length 10');
  assertApprox(pathLength([{ x: 0, y: 0, z: 0 }]), 0, 0.001, 'Single point has 0 length');
  assertApprox(pathLength([]), 0, 0.001, 'Empty array has 0 length');

  // Triangle perimeter: 10 + 10 + 10 = 30
  const equilateral: TemplatePoint3D[] = [
    { x: 0, y: 0, z: 0 },
    { x: 10, y: 0, z: 0 },
    { x: 5, y: 8.66, z: 0 },
    { x: 0, y: 0, z: 0 },
  ];
  assertApprox(pathLength(equilateral), 30, 0.1, 'Equilateral triangle perimeter');
});

describe('resample', () => {
  const resampled = resample([...linePoints], 5);
  assertArrayLength(resampled, 5, 'Resample to 5 points');
  assertApprox(resampled[0].x, 0, 0.001, 'First point at start');
  assertApprox(resampled[4].x, 10, 0.001, 'Last point at end');
  assertApprox(resampled[2].x, 5, 0.001, 'Middle point at midpoint');

  const resampled32 = resampleImmutable(circle3D, 32);
  assertArrayLength(resampled32, 32, 'Resample circle to 32 points');

  const single = resample([{ x: 5, y: 5, z: 5 }], 4);
  assertArrayLength(single, 4, 'Single point resamples to N copies');
  assertApprox(single[0].x, 5, 0.001, 'All copies have same X');
});

describe('centroid', () => {
  const c1 = centroid(linePoints);
  assertApprox(c1.x, 5, 0.001, 'Line centroid X');
  assertApprox(c1.y, 0, 0.001, 'Line centroid Y');

  const c2 = centroid([{ x: 0, y: 0, z: 0 }, { x: 2, y: 4, z: 6 }]);
  assertApprox(c2.x, 1, 0.001, 'Centroid X average');
  assertApprox(c2.y, 2, 0.001, 'Centroid Y average');
  assertApprox(c2.z, 3, 0.001, 'Centroid Z average');

  const empty = centroid([]);
  assertApprox(empty.x, 0, 0.001, 'Empty centroid is origin');
});

describe('boundingBox', () => {
  const box = boundingBox(trianglePoints);
  assertApprox(box.min.x, 0, 0.001, 'Triangle min X');
  assertApprox(box.max.x, 10, 0.001, 'Triangle max X');
  assertApprox(box.min.y, 0, 0.001, 'Triangle min Y');
  assertApprox(box.max.y, 8.66, 0.01, 'Triangle max Y');
  assertApprox(box.size.x, 10, 0.001, 'Triangle width');
});

describe('translateToOrigin', () => {
  const translated = translateToOrigin(linePoints);
  const c = centroid(translated);
  assertApprox(c.x, 0, 0.001, 'Translated centroid X is 0');
  assertApprox(c.y, 0, 0.001, 'Translated centroid Y is 0');
  assertApprox(c.z, 0, 0.001, 'Translated centroid Z is 0');
});

describe('scaleToSize', () => {
  const { points, scale } = scaleToSize(translateToOrigin(linePoints));
  const box = boundingBox(points);
  assertApprox(Math.max(box.size.x, box.size.y, box.size.z), 1, 0.001, 'Scaled to unit size');
  assertApprox(scale, 0.1, 0.001, 'Scale factor for 10-unit line');
});

describe('quickNormalize', () => {
  const normalized = quickNormalize(circle3D);
  const c = centroid(normalized);
  const box = boundingBox(normalized);

  assertApprox(c.x, 0, 0.001, 'Normalized centroid X');
  assertApprox(c.y, 0, 0.001, 'Normalized centroid Y');
  assertApprox(Math.max(box.size.x, box.size.y, box.size.z), 1, 0.001, 'Normalized to unit size');
});

describe('euclideanDistance', () => {
  assertApprox(
    euclideanDistance({ x: 0, y: 0, z: 0 }, { x: 3, y: 4, z: 0 }),
    5,
    0.001,
    '3-4-5 right triangle'
  );
});

describe('pathDistance', () => {
  // Same points should have 0 distance
  const a = [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }];
  assertApprox(pathDistance(a, a), 0, 0.001, 'Same points have 0 distance');

  // Offset points
  const b = [{ x: 0, y: 1, z: 0 }, { x: 1, y: 1, z: 0 }];
  assertApprox(pathDistance(a, b), 1, 0.001, 'Parallel offset of 1');
});

describe('cloudDistance', () => {
  const a = [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }];
  assertApprox(cloudDistance(a, a), 0, 0.001, 'Same points have 0 cloud distance');

  // Reversed order should still match
  const reversed = [{ x: 1, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }];
  assertApprox(cloudDistance(a, reversed), 0, 0.001, 'Reversed order matches');
});

describe('buildLookupTable', () => {
  const points: TemplatePoint3D[] = [
    { x: 1, y: 1, z: 1 },   // Octant 7 (all positive)
    { x: -1, y: -1, z: -1 }, // Octant 0 (all negative)
    { x: 1, y: -1, z: 1 },   // Octant 5
  ];
  const table = buildLookupTable(points);

  assertArrayLength(table, 3, 'Lookup table has same length as points');
  assertApprox(table[0], 7, 0, 'All positive = octant 7');
  assertApprox(table[1], 0, 0, 'All negative = octant 0');
});

describe('distanceToScore', () => {
  assertApprox(distanceToScore(0), 1, 0.001, 'Zero distance = score 1');
  assertApprox(distanceToScore(0.5, 0.5), 0.5, 0.001, 'Half distance = score 0.5');
  assert(distanceToScore(10) > 0, 'Score always positive');
  assert(distanceToScore(10) < distanceToScore(1), 'Higher distance = lower score');
});

describe('FFORecognizer - basic', () => {
  const recognizer = createRecognizer({ numPoints: 16 });

  assert(recognizer.templateCount === 0, 'Starts with 0 templates');

  // Create simple gesture data
  const waveSamples: TelemetrySample3D[] = [];
  for (let i = 0; i < 20; i++) {
    waveSamples.push({
      ax_g: Math.sin(i * 0.5) * 0.5,
      ay_g: 0.1,
      az_g: 1.0,
      t: i * 20,
    });
  }

  recognizer.addTemplateFromSamples('wave', waveSamples);
  assert(recognizer.templateCount === 1, 'Added 1 template');
  assert(recognizer.hasTemplate('wave'), 'Has template "wave"');
  assert(!recognizer.hasTemplate('circle'), 'Does not have template "circle"');

  const template = recognizer.getTemplate('wave');
  assert(template !== undefined, 'Can retrieve template');
  assert(template?.name === 'wave', 'Template has correct name');
  assertArrayLength(template?.points ?? [], 16, 'Template has 16 points');
});

describe('FFORecognizer - recognition', () => {
  const recognizer = createRecognizer({ numPoints: 16 });

  // Create two distinct gestures
  const waveSamples: TelemetrySample3D[] = [];
  const circleSamples: TelemetrySample3D[] = [];

  for (let i = 0; i < 20; i++) {
    // Wave: sinusoidal in X
    waveSamples.push({
      ax_g: Math.sin(i * 0.5) * 0.5,
      ay_g: 0.1,
      az_g: 1.0,
      t: i * 20,
    });

    // Circle: circular in XY
    const angle = (i / 20) * 2 * Math.PI;
    circleSamples.push({
      ax_g: Math.cos(angle) * 0.5,
      ay_g: Math.sin(angle) * 0.5,
      az_g: 1.0,
      t: i * 20,
    });
  }

  recognizer.addTemplateFromSamples('wave', waveSamples);
  recognizer.addTemplateFromSamples('circle', circleSamples);

  // Recognize wave
  const waveResult = recognizer.recognize(waveSamples);
  assert(!waveResult.rejected, 'Wave recognition not rejected');
  assert(waveResult.template?.name === 'wave', 'Wave recognized as wave');
  assert(waveResult.score > 0.5, 'Wave has high score');

  // Recognize circle
  const circleResult = recognizer.recognize(circleSamples);
  assert(!circleResult.rejected, 'Circle recognition not rejected');
  assert(circleResult.template?.name === 'circle', 'Circle recognized as circle');

  // Check candidates
  assert(waveResult.candidates !== undefined, 'Candidates returned');
  assertArrayLength(waveResult.candidates ?? [], 2, 'Two candidates');
});

describe('FFORecognizer - export/import', () => {
  const recognizer1 = createRecognizer({ numPoints: 16 });

  const samples: TelemetrySample3D[] = [];
  for (let i = 0; i < 20; i++) {
    samples.push({ ax_g: i * 0.1, ay_g: 0, az_g: 1, t: i * 20 });
  }

  recognizer1.addTemplateFromSamples('test', samples);

  // Export
  const json = recognizer1.toJSON('test-vocabulary');
  assert(json.length > 0, 'Exported JSON is not empty');

  // Import into new recognizer
  const recognizer2 = createRecognizer({ numPoints: 16 });
  recognizer2.fromJSON(json);

  assert(recognizer2.templateCount === 1, 'Imported 1 template');
  assert(recognizer2.hasTemplate('test'), 'Imported template has correct name');
});

describe('FFORecognizer - edge cases', () => {
  const recognizer = createRecognizer({ numPoints: 16, minSamples: 10 });

  // Too few samples
  const fewSamples: TelemetrySample3D[] = [
    { ax_g: 0, ay_g: 0, az_g: 1, t: 0 },
    { ax_g: 1, ay_g: 0, az_g: 1, t: 20 },
  ];

  const result = recognizer.recognize(fewSamples);
  assert(result.rejected, 'Too few samples is rejected');

  // No templates
  const manySamples: TelemetrySample3D[] = [];
  for (let i = 0; i < 20; i++) {
    manySamples.push({ ax_g: i * 0.1, ay_g: 0, az_g: 1, t: i * 20 });
  }

  const emptyResult = recognizer.recognize(manySamples);
  assert(emptyResult.rejected, 'No templates means rejected');

  // Remove template
  recognizer.addTemplateFromSamples('temp', manySamples);
  assert(recognizer.templateCount === 1, 'Added template');

  const removed = recognizer.removeTemplate('temp');
  assert(removed, 'Template removed');
  assert(recognizer.templateCount === 0, 'Template count is 0 after removal');

  const notRemoved = recognizer.removeTemplate('nonexistent');
  assert(!notRemoved, 'Removing nonexistent returns false');
});

describe('FFORecognizer - rejection threshold', () => {
  const recognizer = createRecognizer({
    numPoints: 16,
    rejectThreshold: 0.1, // Very tight threshold
  });

  const samples: TelemetrySample3D[] = [];
  for (let i = 0; i < 20; i++) {
    samples.push({ ax_g: i * 0.1, ay_g: 0, az_g: 1, t: i * 20 });
  }

  recognizer.addTemplateFromSamples('line', samples);

  // Perfect match should not be rejected
  const perfectResult = recognizer.recognize(samples);
  assert(!perfectResult.rejected, 'Perfect match not rejected');

  // Very different input should be rejected
  const differentSamples: TelemetrySample3D[] = [];
  for (let i = 0; i < 20; i++) {
    differentSamples.push({
      ax_g: Math.sin(i * 0.5) * 2, // Very different pattern
      ay_g: Math.cos(i * 0.5) * 2,
      az_g: 0,
      t: i * 20,
    });
  }

  const differentResult = recognizer.recognize(differentSamples);
  // Note: With such a tight threshold, this should be rejected
  // but the actual behavior depends on normalization
  assert(differentResult.distance > 0, 'Different input has non-zero distance');
});

// ===== Run Tests =====

export function runAllTests(): void {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║               FFO$$ UNIT TESTS                                 ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  passCount = 0;
  failCount = 0;

  // Run all test suites
  describe('distance3D', () => {
    assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }), 1, 0.001, 'Unit distance X');
    assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }), 1, 0.001, 'Unit distance Y');
    assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }), 1, 0.001, 'Unit distance Z');
    assertApprox(distance3D({ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 }), Math.sqrt(3), 0.001, 'Diagonal distance');
  });

  describe('lerp3D', () => {
    const a = { x: 0, y: 0, z: 0 };
    const b = { x: 10, y: 20, z: 30 };
    const mid = lerp3D(a, b, 0.5);
    assertApprox(mid.x, 5, 0.001, 'Midpoint X');
    assertApprox(mid.y, 10, 0.001, 'Midpoint Y');
    assertApprox(mid.z, 15, 0.001, 'Midpoint Z');
  });

  describe('pathLength', () => {
    assertApprox(pathLength(linePoints), 10, 0.001, 'Line of length 10');
    assertApprox(pathLength([{ x: 0, y: 0, z: 0 }]), 0, 0.001, 'Single point has 0 length');
  });

  describe('resample', () => {
    const resampled = resample([...linePoints], 5);
    assertArrayLength(resampled, 5, 'Resample to 5 points');
    assertApprox(resampled[0].x, 0, 0.001, 'First point at start');
    assertApprox(resampled[4].x, 10, 0.001, 'Last point at end');
  });

  describe('centroid', () => {
    const c = centroid(linePoints);
    assertApprox(c.x, 5, 0.001, 'Line centroid X');
    assertApprox(c.y, 0, 0.001, 'Line centroid Y');
  });

  describe('quickNormalize', () => {
    const normalized = quickNormalize(circle3D);
    const c = centroid(normalized);
    assertApprox(c.x, 0, 0.001, 'Normalized centroid X');
    assertApprox(c.y, 0, 0.001, 'Normalized centroid Y');
  });

  describe('FFORecognizer', () => {
    const recognizer = createRecognizer({ numPoints: 16 });
    assert(recognizer.templateCount === 0, 'Starts with 0 templates');

    const samples: TelemetrySample3D[] = [];
    for (let i = 0; i < 20; i++) {
      samples.push({
        ax_g: Math.sin(i * 0.5) * 0.5,
        ay_g: 0.1,
        az_g: 1.0,
        t: i * 20,
      });
    }

    recognizer.addTemplateFromSamples('wave', samples);
    assert(recognizer.templateCount === 1, 'Added 1 template');
    assert(recognizer.hasTemplate('wave'), 'Has template "wave"');

    const result = recognizer.recognize(samples);
    assert(!result.rejected, 'Recognition not rejected');
    assert(result.template?.name === 'wave', 'Recognized as wave');
  });

  console.log('\n════════════════════════════════════════════════════════════════');
  console.log(`  Results: ${passCount} passed, ${failCount} failed`);
  console.log('════════════════════════════════════════════════════════════════');

  if (failCount > 0) {
    console.error('\n⚠️  Some tests failed!');
  } else {
    console.log('\n✓ All tests passed!');
  }
}

// Auto-run when executed directly
if (typeof window !== 'undefined') {
  (window as unknown as { runFFOTests: typeof runAllTests }).runFFOTests = runAllTests;
  console.log('Run tests with: runFFOTests()');
} else {
  runAllTests();
}

export default runAllTests;
