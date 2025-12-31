# FFO$$ Template Matching: Deep Technical Analysis

**Status:** Research Document
**Date:** December 2025
**Related Documents:**
- [Aligned Finger Model](./aligned-finger-model-analysis.md)
- [Symbiosis Analysis](./aligned-ffo-symbiosis.md)
- [FFO$$ Research Overview](./ffo-dollar-research-analysis.md)

---

## Executive Summary

**FFO$$ (Fist Full Of Dollars)** adapts the $-family of gesture recognizers to 3D IMU sensor data. The core innovation is **template-based matching with geometric normalization**, enabling gesture recognition from just 1-3 training examples per class.

**Key Characteristics:**
- **Minimal training data**: Works with 1-10 templates per gesture
- **Sub-millisecond inference**: $Q achieves O(n) complexity via lookup tables
- **Interpretable**: Templates are human-readable gesture traces
- **No training phase**: Templates are stored directly, not learned
- **Trailing window matching**: Constant window of recent samples matched against templates

---

## 1. The $-Family Algorithm Lineage

### 1.1 Evolution from 2D to 3D

| Algorithm | Year | Original Domain | Complexity | FFO$$ Adaptation |
|-----------|------|-----------------|------------|------------------|
| **$1** | 2007 | Pen/touch 2D | O(n) | rotateZ for 3D alignment |
| **$N** | 2010 | Multi-stroke 2D | O(n! × n²) | Not implemented |
| **$P** | 2012 | Point-cloud 2D | O(n²) | cloudDistance for 3D |
| **$Q** | 2018 | Point-cloud 2D | O(n) | **Primary: octant-based bins** |

### 1.2 Core Philosophy

The $-family shares fundamental principles that contrast with neural networks:

```
$-Family Philosophy:
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  1. TEMPLATES, NOT WEIGHTS                                     │
│     Store exemplar gestures directly                           │
│                                                                │
│  2. GEOMETRIC NORMALIZATION                                    │
│     Remove position, scale, (rotation) variance                │
│                                                                │
│  3. DISTANCE-BASED MATCHING                                    │
│     Find nearest template by Euclidean distance                │
│                                                                │
│  4. FEW-SHOT LEARNING                                          │
│     1-3 examples per class is often sufficient                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. FFO$$ Processing Pipeline

### 2.1 Complete Pipeline

```
Input: TelemetrySample3D[] (accelerometer ax, ay, az at ~50Hz)
       ↓
┌──────────────────────────────────────────────────────────────────┐
│ 1. TRAJECTORY EXTRACTION                                         │
│    extractTrajectory(samples) → TemplatePoint3D[]                │
│    - Maps ax_g, ay_g, az_g to 3D points                          │
└──────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. GRAVITY REMOVAL (optional)                                    │
│    removeGravityApprox(points) → TemplatePoint3D[]               │
│    - Estimates gravity from first 5 samples (assumes rest)       │
│    - Subtracts from all points                                   │
└──────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. RESAMPLING                                                    │
│    resampleImmutable(points, N=32) → TemplatePoint3D[]           │
│    - Compute total path length                                   │
│    - Insert N equally-spaced points along path                   │
│    - Interpolate between original points                         │
└──────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. NORMALIZATION                                                 │
│    quickNormalize(points) → TemplatePoint3D[]                    │
│    - Translate: centroid to origin                               │
│    - Scale: max dimension to 1.0 (preserve aspect ratio)         │
│    - Rotate (optional): align indicative angle                   │
└──────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────┐
│ 5. LOOKUP TABLE (for $Q speed)                                   │
│    buildLookupTable(points) → number[]                           │
│    - Assign each point to one of 8 octants                       │
│    - Enables O(n) approximate matching                           │
└──────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────┐
│ 6. DISTANCE COMPUTATION                                          │
│    lookupDistance(input, templates) → number                     │
│    - Match only within same/adjacent octants                     │
│    - Greedy nearest-neighbor assignment                          │
│    - Sum of matched point distances                              │
└──────────────────────────────────────────────────────────────────┘
       ↓
Output: Best matching template + distance + score
```

### 2.2 Key Implementation Files

| File | Purpose |
|------|---------|
| [`packages/ffo/src/recognizer.ts`](../../packages/ffo/src/recognizer.ts) | Main FFORecognizer class |
| [`packages/ffo/src/resample.ts`](../../packages/ffo/src/resample.ts) | Trajectory resampling |
| [`packages/ffo/src/normalize.ts`](../../packages/ffo/src/normalize.ts) | Geometric normalization |
| [`packages/ffo/src/distance.ts`](../../packages/ffo/src/distance.ts) | Distance metrics ($1, $P, $Q) |
| [`packages/ffo/src/types.ts`](../../packages/ffo/src/types.ts) | Type definitions |

---

## 3. Trajectory Extraction

### 3.1 From IMU Samples to 3D Points

```typescript
// From resample.ts:

export function extractTrajectory(samples: TelemetrySample3D[]): TemplatePoint3D[] {
  return samples.map((s) => ({
    x: s.ax_g,  // Accelerometer X in g
    y: s.ay_g,  // Accelerometer Y in g
    z: s.az_g,  // Accelerometer Z in g
  }));
}
```

**Key Decision:** Uses **accelerometer** (not gyroscope or magnetometer) because:
1. Accelerometer captures hand motion in 3D space
2. Gravity provides stable "down" reference
3. No integration drift (unlike velocity/position)

### 3.2 Gravity Removal

```typescript
export function removeGravityApprox(points: TemplatePoint3D[]): TemplatePoint3D[] {
  // Estimate gravity from first 5 samples (assuming device at rest)
  const windowSize = Math.min(5, points.length);
  const gravity = { x: 0, y: 0, z: 0 };

  for (let i = 0; i < windowSize; i++) {
    gravity.x += points[i].x / windowSize;
    gravity.y += points[i].y / windowSize;
    gravity.z += points[i].z / windowSize;
  }

  // Subtract estimated gravity from all points
  return points.map((p) => ({
    x: p.x - gravity.x,
    y: p.y - gravity.y,
    z: p.z - gravity.z,
  }));
}
```

**Assumption:** Device starts at rest. For continuous recognition, consider:
1. Madgwick/Mahony filter for orientation-based gravity subtraction
2. High-pass filtering to remove DC component

---

## 4. Resampling Algorithm

### 4.1 The Core Algorithm (from $1 paper)

The resampling step is **essential** for template matching because:
1. Gestures are performed at different speeds
2. Different devices have different sample rates
3. Templates must have identical point counts for comparison

```typescript
// From resample.ts:

export function resample(points: TemplatePoint3D[], n: number = 32): TemplatePoint3D[] {
  const totalLength = pathLength(points);
  const interval = totalLength / (n - 1);  // Desired spacing

  const resampled: TemplatePoint3D[] = [{ ...points[0] }];
  let accumulatedDistance = 0;
  let i = 1;

  while (resampled.length < n && i < points.length) {
    const segmentDist = distance3D(points[i - 1], points[i]);

    if (accumulatedDistance + segmentDist >= interval) {
      // Insert interpolated point
      const overshoot = interval - accumulatedDistance;
      const t = overshoot / segmentDist;
      const newPoint = lerp3D(points[i - 1], points[i], t);
      resampled.push(newPoint);

      // Continue from new point
      points.splice(i, 0, newPoint);
      accumulatedDistance = 0;
    } else {
      accumulatedDistance += segmentDist;
      i++;
    }
  }

  return resampled;
}
```

### 4.2 Visual Representation

```
Original trajectory (variable sampling):
  •   •     •           •   • •  •     •
  ├───┼─────┼───────────┼───┼─┼──┼─────┤
  0                                    L

Resampled (N=8, equal spacing):
  •     •     •     •     •     •     •     •
  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  0    L/7  2L/7  3L/7  4L/7  5L/7  6L/7   L
```

### 4.3 Configuration

```typescript
const config: RecognizerConfig = {
  numPoints: 32,        // Standard: 32 points
  minSamples: 10,       // Minimum input samples required
  // ...
};
```

**Why N=32?** The $Q paper suggests 32 as optimal trade-off between accuracy and speed. Lower values (16) work for simple gestures; higher values (64) for complex ones.

---

## 5. Normalization Transforms

### 5.1 Translation to Origin

```typescript
// From normalize.ts:

export function centroid(points: TemplatePoint3D[]): Vector3 {
  const sum = points.reduce((acc, p) => ({
    x: acc.x + p.x,
    y: acc.y + p.y,
    z: acc.z + p.z,
  }), { x: 0, y: 0, z: 0 });

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
    z: sum.z / points.length,
  };
}

export function translateToOrigin(points: TemplatePoint3D[]): TemplatePoint3D[] {
  const c = centroid(points);
  return points.map((p) => ({
    x: p.x - c.x,
    y: p.y - c.y,
    z: p.z - c.z,
  }));
}
```

**Purpose:** Makes gestures position-independent. A wave gesture at (0,0,0) matches a wave at (10,10,10).

### 5.2 Scaling to Unit Size

```typescript
export function scaleToSize(points: TemplatePoint3D[], targetScale = 1.0) {
  const bounds = boundingBox(points);
  const maxDimension = Math.max(bounds.size.x, bounds.size.y, bounds.size.z);

  if (maxDimension === 0) {
    return { points: [...points], scale: 1 };
  }

  const scale = targetScale / maxDimension;

  return {
    points: points.map((p) => ({
      x: p.x * scale,
      y: p.y * scale,
      z: p.z * scale,
    })),
    scale,
  };
}
```

**Purpose:** Makes gestures size-independent. A small wave matches a large wave.

### 5.3 Rotation Alignment (Optional)

```typescript
export function indicativeAngles(points: TemplatePoint3D[], center?: Vector3) {
  const c = center ?? centroid(points);
  const first = points[0];

  // Vector from centroid to first point
  const dx = first.x - c.x;
  const dy = first.y - c.y;
  const dz = first.z - c.z;
  const r = Math.sqrt(dx * dx + dy * dy + dz * dz);

  // Spherical coordinates
  const theta = Math.atan2(dy, dx);  // Azimuth
  const phi = Math.acos(dz / r);     // Elevation

  return { theta, phi };
}

export function rotateAroundZ(points: TemplatePoint3D[], angle: number) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  return points.map((p) => ({
    x: p.x * cos - p.y * sin,
    y: p.x * sin + p.y * cos,
    z: p.z,  // Z unchanged
  }));
}
```

**Purpose:** Rotation normalization for $1-style matching. Skipped in $P/$Q which are rotation-invariant.

### 5.4 Quick vs. Full Normalization

```typescript
// For $P/$Q (rotation-invariant)
export function quickNormalize(points: TemplatePoint3D[]): TemplatePoint3D[] {
  return normalize(points, { translate: true, scale: true, rotate: false }).points;
}

// For $1-style (sequential matching)
export function fullNormalize(points: TemplatePoint3D[]): TemplatePoint3D[] {
  return normalize(points, { translate: true, scale: true, rotate: true }).points;
}
```

---

## 6. Distance Metrics

### 6.1 $1-Style: Path Distance (Sequential)

```typescript
// From distance.ts:

export function pathDistance(a: TemplatePoint3D[], b: TemplatePoint3D[]): number {
  // Points matched by index (order matters!)
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += euclideanDistance(a[i], b[i]);
  }
  return sum / a.length;
}
```

**Characteristics:**
- Order-dependent: a[0]↔b[0], a[1]↔b[1], etc.
- Requires rotation normalization
- Best for unistroke gestures drawn in consistent direction

### 6.2 $P-Style: Cloud Distance (Permutation-Invariant)

```typescript
export function cloudDistance(a: TemplatePoint3D[], b: TemplatePoint3D[]): number {
  const n = a.length;
  const matched = new Array<boolean>(n).fill(false);
  let sum = 0;

  // Greedy nearest-neighbor matching
  for (let i = 0; i < n; i++) {
    let minDist = Infinity;
    let minIndex = -1;

    for (let j = 0; j < n; j++) {
      if (!matched[j]) {
        const d = squaredDistance(a[i], b[j]);
        if (d < minDist) {
          minDist = d;
          minIndex = j;
        }
      }
    }

    matched[minIndex] = true;
    sum += Math.sqrt(minDist);
  }

  return sum / n;
}
```

**Characteristics:**
- Order-independent: Points matched by proximity
- O(n²) complexity
- Better for multi-stroke or reversible gestures

### 6.3 $Q-Style: Lookup Distance (Fast Approximation)

```typescript
// Octant-based binning (8 bins for 3D)
export function buildLookupTable(points: TemplatePoint3D[]): number[] {
  return points.map((p) => {
    let bin = 0;
    if (p.x >= 0) bin |= 1;  // Bit 0: positive X
    if (p.y >= 0) bin |= 2;  // Bit 1: positive Y
    if (p.z >= 0) bin |= 4;  // Bit 2: positive Z
    return bin;  // 0-7 octant index
  });
}

export function lookupDistance(
  a: TemplatePoint3D[], aTable: number[],
  b: TemplatePoint3D[], bTable: number[]
): number {
  // Group b points by octant
  const binToPoints: Map<number, number[]> = new Map();
  for (let j = 0; j < b.length; j++) {
    const bin = bTable[j];
    if (!binToPoints.has(bin)) binToPoints.set(bin, []);
    binToPoints.get(bin)!.push(j);
  }

  const matched = new Array<boolean>(a.length).fill(false);
  let sum = 0;

  for (let i = 0; i < a.length; i++) {
    const aBin = aTable[i];
    let minDist = Infinity;
    let minIndex = -1;

    // Check same octant and adjacent octants
    const binsToCheck = [aBin];
    for (let flip = 0; flip < 3; flip++) {
      binsToCheck.push(aBin ^ (1 << flip));  // Flip one bit
    }

    for (const bin of binsToCheck) {
      const candidates = binToPoints.get(bin) ?? [];
      for (const j of candidates) {
        if (!matched[j]) {
          const d = squaredDistance(a[i], b[j]);
          if (d < minDist) {
            minDist = d;
            minIndex = j;
          }
        }
      }
    }

    // Fallback to full search if no match in nearby bins
    if (minIndex === -1) { /* ... full search ... */ }

    matched[minIndex] = true;
    sum += Math.sqrt(minDist);
  }

  return sum / a.length;
}
```

**Characteristics:**
- O(n) average case (vs. O(n²) for $P)
- 142× faster than $P (per original paper)
- Slight accuracy trade-off (uses octant approximation)
- Ideal for real-time recognition

### 6.4 Octant Visualization

```
        +Z
         │
    ┌────┼────┐
   3│    │    │7     Octant encoding (3 bits):
    │   1│5   │        bit 0 = X sign (+1 if positive)
────┼────┼────┼────Y   bit 1 = Y sign (+2 if positive)
   2│    │    │6       bit 2 = Z sign (+4 if positive)
    │   0│4   │
    └────┼────┘
         │
        -Z
        -X     +X
```

---

## 7. Score Conversion

### 7.1 Distance to Score

```typescript
export function distanceToScore(distance: number, halfDistance: number = 0.5): number {
  // Sigmoid-like: score = 1 / (1 + d/halfDist)
  return 1 / (1 + distance / halfDistance);
}
```

| Distance | Score |
|----------|-------|
| 0.0 | 1.00 |
| 0.25 | 0.67 |
| 0.50 | 0.50 |
| 1.00 | 0.33 |
| 2.00 | 0.20 |

### 7.2 Rejection Threshold

```typescript
const result = recognizer.recognize(samples);

if (result.rejected) {
  console.log("Unknown gesture");
} else {
  console.log(`Detected: ${result.template.name} (score: ${result.score})`);
}
```

Rejection occurs when `distance > rejectThreshold`, preventing false positives for unknown gestures.

---

## 8. Template Recording and Storage

### 8.1 Recording a Template

```typescript
// From FFORecognizer:

addTemplateFromSamples(
  name: string,
  samples: TelemetrySample3D[],
  source: string = 'recorded'
): GestureTemplate {
  // 1. Process input (extract, remove gravity, resample, normalize)
  const points = this.processInput(samples);

  // 2. Compute duration
  const duration = samples[samples.length - 1].t - samples[0].t;

  // 3. Create template with lookup table
  const template: GestureTemplate = {
    id: generateId(),
    name,
    points,
    meta: {
      n: this.config.numPoints,
      source,
      created: new Date().toISOString(),
      duration,
      lookupTable: buildLookupTable(points),
    },
  };

  this.templates.push(template);
  return template;
}
```

### 8.2 Vocabulary Export/Import

```typescript
// Export for persistence
const vocabulary = recognizer.export('my-gestures');
localStorage.setItem('ffo-vocabulary', JSON.stringify(vocabulary));

// Import from storage
const saved = JSON.parse(localStorage.getItem('ffo-vocabulary'));
recognizer.import(saved);
```

### 8.3 Template Format

```json
{
  "version": "1.0.0",
  "templates": [
    {
      "id": "tmpl_1735123456_abc123",
      "name": "wave",
      "points": [
        { "x": -0.42, "y": 0.31, "z": 0.08 },
        { "x": -0.38, "y": 0.35, "z": 0.12 }
        // ... 32 total points
      ],
      "meta": {
        "n": 32,
        "source": "recorded",
        "created": "2025-12-31T12:00:00.000Z",
        "duration": 850,
        "lookupTable": [0, 0, 1, 1, 3, 3, 7, 7, ...]
      }
    }
  ],
  "rejectThreshold": 0.5
}
```

---

## 9. Real-Time Recognition: Trailing Window

### 9.1 The Trailing Window Concept

FFO$$ operates on a **constant trailing window** of recent samples:

```
Time →
        ┌─────────────────────────────────────┐
Samples │ • • • • • • • • • • • • • • • • • • │ • • • (incoming)
        └─────────────────────────────────────┘
              ↑                               ↑
           Window start                   Window end
              (N samples ago)              (now)
```

### 9.2 Continuous Recognition Flow

```typescript
class ContinuousRecognizer {
  private buffer: TelemetrySample3D[] = [];
  private windowSize: number = 50;  // ~1 second at 50Hz

  addSample(sample: TelemetrySample3D): RecognitionResult | null {
    this.buffer.push(sample);

    // Keep only last windowSize samples
    if (this.buffer.length > this.windowSize) {
      this.buffer.shift();
    }

    // Run recognition when buffer is full
    if (this.buffer.length === this.windowSize) {
      return this.recognizer.recognize(this.buffer);
    }

    return null;
  }
}
```

### 9.3 Stride for Efficiency

Rather than recognizing on every sample, use a stride:

```typescript
// Recognize every 10 samples (~200ms at 50Hz)
if (this.sampleCount % 10 === 0) {
  const result = this.recognize();
  if (result.score > 0.7) {
    this.onGestureDetected(result);
  }
}
```

---

## 10. Performance Characteristics

### 10.1 Computational Complexity

| Stage | Complexity | Notes |
|-------|------------|-------|
| Trajectory extraction | O(n) | Simple array map |
| Gravity removal | O(n) | Subtract constant |
| Resampling | O(n) | Path walk with interpolation |
| Normalization | O(n) | Centroid + scale |
| Lookup table build | O(n) | One pass |
| $Q distance | O(n) | Octant-based matching |
| **Total per recognition** | **O(n × T)** | n=window, T=templates |

### 10.2 Memory Requirements

| Component | Size |
|-----------|------|
| Single template (32 points) | ~768 bytes (32 × 3 × 8) |
| Lookup table | 32 bytes (1 per point) |
| Recognition buffer | ~3.6 KB (50 × 9 × 8) |
| 10 template vocabulary | ~8 KB |

### 10.3 Latency (Measured)

| Device | Recognition Time |
|--------|------------------|
| Desktop Chrome | <0.5 ms |
| Mobile Safari | ~1-2 ms |
| Puck.js (Espruino) | ~5-10 ms (estimated) |

---

## 11. Comparison to Neural Networks

### 11.1 Head-to-Head

| Aspect | FFO$$ | CNN |
|--------|-------|-----|
| Training data | 1-10 examples | 100s-1000s |
| Training time | Instant | Minutes-hours |
| Model size | ~1-10 KB | ~75-150 KB |
| Inference time | <1 ms | 5-50 ms |
| Accuracy (clear gestures) | 80-95% | 85-95% |
| Accuracy (subtle variations) | 70-85% | 80-90% |
| Generalization | Low | High |
| Interpretability | High | Low |
| On-device Puck.js | Yes | Difficult |

### 11.2 When to Use FFO$$

**FFO$$ excels for:**
- Rapid prototyping without training cycles
- Constrained devices (Puck.js, microcontrollers)
- User-defined gestures (personalization)
- Well-defined, distinct gestures
- Debugging (can visualize templates)

**CNN excels for:**
- Subtle gesture variations
- Large gesture vocabularies
- Production deployment with robustness
- Transfer learning across users/devices

---

## 12. FFO$$ for Finger Tracking?

### 12.1 Current State

FFO$$ is designed for **motion gestures** (acceleration traces), not static poses. However, it could potentially be extended:

### 12.2 Approach A: Magnetic Trajectory Templates

Record the magnetometer trajectory during a finger flexion:

```typescript
// Template for "index finger flexing"
const indexFlexTemplate = {
  points: [
    // Magnetometer values during flexion motion
    { x: 0, y: 0, z: 0 },        // Start (extended)
    { x: 100, y: 200, z: 150 },  // Mid-flex
    { x: 1624, y: 5978, z: 10548 } // End (fully flexed)
    // ... resampled to 32 points
  ]
};
```

**Challenge:** Static poses have no trajectory (zero path length).

### 12.3 Approach B: Static Pose as Point Cloud

Treat a static magnetometer reading as a 1-point template:

```typescript
// Template for "index finger flexed" (static)
const indexFlexedTemplate = {
  point: { mx: 1624, my: 5978, mz: 10548 },  // Single 3D signature
  tolerance: 500  // µT match radius
};

// Match by simple distance threshold
function matchStaticPose(sample, templates) {
  for (const t of templates) {
    const dist = euclidean(sample, t.point);
    if (dist < t.tolerance) {
      return t.name;
    }
  }
  return null;
}
```

This is essentially **nearest-neighbor classification**—what the aligned model does!

### 12.4 Synthesis with Aligned Model

See [Symbiosis Analysis](./aligned-ffo-symbiosis.md) for how aligned signatures could serve as FFO$$ templates.

---

## 13. Future Directions

### 13.1 Proposed Enhancements

1. **Orientation-aware matching**: Transform templates to palm-centric frame
2. **Temporal smoothing**: Require N consecutive matches to trigger
3. **Multi-template voting**: Average distance across multiple examples
4. **Adaptive rejection**: Learn rejection threshold from negative examples

### 13.2 Integration with Aligned Model

1. Use aligned signatures as static pose templates
2. Use FFO$$ for transition detection (pose-to-pose motion)
3. Ensemble: FFO$$ for fast rejection, CNN for confirmation

### 13.3 Puck.js Deployment

The $Q algorithm was explicitly designed for "low-powered mobiles and wearables." A Puck.js deployment would:
- Run entirely on-device (no BLE streaming required)
- Store ~10-20 templates in flash (~10 KB)
- Recognize in <10ms per window

---

## 14. References

1. **$1 Recognizer**: Wobbrock, J.O., Wilson, A.D., Li, Y. (2007). "Gestures without libraries, toolkits or training." UIST '07.
2. **$P Recognizer**: Vatavu, R., Anthony, L., Wobbrock, J.O. (2012). "Gestures as point clouds." ICMI '12.
3. **$Q Recognizer**: Vatavu, R., Anthony, L., Wobbrock, J.O. (2018). "$Q: A super-quick, articulation-invariant stroke-gesture recognizer." MobileHCI '18.
4. **ACE Lab Resources**: https://depts.washington.edu/acelab/proj/dollar/

### SIMCAP Implementation

- [`packages/ffo/`](../../packages/ffo/) - FFO$$ library
- [`apps/ffo/`](../../apps/ffo/) - FFO$$ demonstration app
- [`docs/design/ffo-dollar-research-analysis.md`](./ffo-dollar-research-analysis.md) - Original research overview

---

<link rel="stylesheet" href="../../src/simcap.css">
