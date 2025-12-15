# GAMBIT Index vs Collector Implementation Comparison

**Date:** 2025-12-15
**Purpose:** Verify equivalence between index.html (gold standard) and collector-app.js implementations

---

## Executive Summary

### Key Findings

❌ **NOT EQUIVALENT** - Significant differences found:

1. **Hand Rendering:** Both use the LEGACY `Hand3DRenderer` (2D canvas), but index.html ALSO uses `ThreeJSHandSkeleton` (Three.js WebGL)
2. **Orientation Mapping:** Equivalent - both use `updateFromSensorFusion(euler)` via `Hand3DRenderer`
3. **Magnetic Residual Visualization:**
   - ✅ Collector: HAS 3D trajectory visualization via `MagneticTrajectory`
   - ❌ Index: MISSING - no magnetic trajectory visualization

### Critical Issue

**Index.html is missing the magnetic residual 3D trajectory visualization** that exists in collector-app.js. This needs to be added to maintain feature parity.

---

## Detailed Comparison

### 1. Hand Rendering Implementation

#### Index.html (Gold Standard)

**Uses TWO renderers:**

1. **Hand3DRenderer** (Legacy 2D Canvas)
   - File: `hand-3d-renderer.js`
   - Location: `index.html:1004-1042`
   - Canvas ID: `handCanvas3D`
   - Initialization:
     ```javascript
     hand3DRenderer = new Hand3DRenderer(canvas, {
         backgroundColor: '#efefef',
         orientationMode: 'sensor_fusion',
         orientationFiltering: true,
         orientationAlpha: 0.3,  // More responsive
         pitchOffset: 0,
         yawOffset: 0,
         rollOffset: 0
     });
     ```

2. **ThreeJSHandSkeleton** (Modern Three.js WebGL)
   - File: `shared/threejs-hand-skeleton.js`
   - Location: `index.html:1044-1143`
   - Container ID: `threeHandContainer`
   - Initialization:
     ```javascript
     threeHandSkeleton = new ThreeJSHandSkeleton(container, {
         width: 300,
         height: 300,
         backgroundColor: 0x1a1a2e,
         lerpFactor: 0.15,
         handedness: 'right'
     });

     // Set orientation offsets
     threeHandSkeleton.setOrientationOffsets({
         roll: 180,
         pitch: 180,
         yaw: -180
     });
     ```

**Orientation Updates:**
```javascript
// Line 1876: Hand3DRenderer (legacy canvas)
if (hand3DRenderer && handOrientationEnabled) {
    hand3DRenderer.updateFromSensorFusion(handEuler);
}

// Line 1881: ThreeJSHandSkeleton (modern)
if (threeHandSkeleton && threeHandEnabled) {
    threeHandSkeleton.updateOrientation(handEuler);
}
```

#### Collector-app.js

**Uses ONLY ONE renderer:**

1. **Hand3DRenderer** (Legacy 2D Canvas)
   - File: `hand-3d-renderer.js`
   - Location: `collector-app.js:1071-1095`
   - Canvas ID: `handCanvas3D`
   - Initialization:
     ```javascript
     hand3DRenderer = new Hand3DRenderer(canvas3D, {
         backgroundColor: '#ffffff',
         orientationMode: 'static',  // Switches to sensor_fusion when pose tracking enabled
         orientationFiltering: poseEstimationOptions.smoothHandOrientation,
         orientationAlpha: poseEstimationOptions.handOrientationAlpha,  // 0.1
         pitchOffset: 0,
         yawOffset: 0,
         rollOffset: 0
     });
     ```

**Orientation Updates:**
- Via telemetry processor callback: `telemetry-handler.js:52-54`
  ```javascript
  onOrientationUpdate: (euler, quaternion) => {
      if (deps.hand3DRenderer && euler) {
          deps.hand3DRenderer.updateFromSensorFusion(euler);
      }
  }
  ```
- Also in pose estimation: `collector-app.js:993-994`
  ```javascript
  if (poseEstimationOptions.enableHandOrientation && euler && hand3DRenderer) {
      hand3DRenderer.updateFromSensorFusion(euler);
  }
  ```

#### Comparison Summary

| Feature | Index.html | Collector-app.js | Status |
|---------|-----------|------------------|--------|
| **Hand3DRenderer (2D canvas)** | ✅ Yes | ✅ Yes | ✅ Equivalent |
| **ThreeJSHandSkeleton (WebGL)** | ✅ Yes | ❌ No | ❌ Missing in collector |
| **Orientation mapping** | `updateFromSensorFusion()` | `updateFromSensorFusion()` | ✅ Equivalent |
| **Axis mapping** | V2 corrected | V2 corrected | ✅ Equivalent |

**Recommendation:**
- Mark `Hand3DRenderer` as LEGACY (used in both, should be removed from both)
- Add `ThreeJSHandSkeleton` to collector-app.js to match gold standard

---

### 2. Orientation Mapping Verification

Both implementations use **identical orientation mapping** via `Hand3DRenderer.updateFromSensorFusion()`:

#### Mapping Logic (hand-3d-renderer.js:222-267)

```javascript
updateFromSensorFusion(euler) {
    // AXIS MAPPING (V2 corrected 2025-12-14):
    // Physical movements → AHRS Reports → Renderer:
    //   - Physical ROLL (tilt pinky/thumb)  → AHRS roll  → renderer.yaw   → RotY
    //   - Physical PITCH (tilt fingers)     → AHRS pitch → renderer.pitch → RotX
    //   - Physical YAW (spin while flat)    → AHRS yaw   → renderer.roll  → RotZ

    const mappedOrientation = {
        pitch: -euler.pitch + this.orientationOffset.pitch,  // AHRS pitch → RotX
        yaw:   -euler.roll  + this.orientationOffset.yaw,    // AHRS roll  → RotY (negated)
        roll:   euler.yaw   + this.orientationOffset.roll    // AHRS yaw   → RotZ
    };

    this.setOrientation(mappedOrientation);
}
```

**Source Data:**
- **Index:** Gets euler from local `telemetryProcessor` (line 1869-1873)
- **Collector:** Gets euler from shared `telemetryProcessor` (telemetry-handler.js:107)

Both use the **same TelemetryProcessor** from `shared/telemetry-processor.js`, which uses the **same Madgwick AHRS** implementation.

**✅ CONFIRMED EQUIVALENT**

---

### 3. Magnetic Residual Visualization

#### Index.html (Gold Standard)

**❌ MISSING - No magnetic trajectory visualization**

The index.html reads magnetic data but does NOT visualize the 3D residual trajectory:

```javascript
// Line 1793-1806: Reads fused_mx for plotting
mx = decoratedData.fused_mx ?? decoratedData.calibrated_mx ?? decoratedData.mx_ut ?? 0;
```

But there is **no MagneticTrajectory instance** or 3D visualization in index.html.

#### Collector-app.js

**✅ HAS magnetic trajectory visualization**

**Implementation:** `modules/magnetic-trajectory.js`

**Initialization:** `collector-app.js:1100-1140`
```javascript
magTrajectory = new MagneticTrajectory(canvas, {
    maxPoints: 200,
    trajectoryColor: '#4ecdc4',
    backgroundColor: '#ffffff',
    autoNormalize: true,
    showMarkers: true,
    showCube: false
});
```

**Data Source:** Residual magnetic field after environment correction

**Update Path:**
1. `telemetry-handler.js:187-190` → calls `updateMagTrajectory(decorated)`
2. `collector-app.js:1163-1173` → `magTrajectory.addPoint(fused_mx, fused_my, fused_mz)`

**What is Visualized:**
```javascript
// From telemetry-processor.js:356-358
const fused = this.calibration.correct(
    { x: mx_ut, y: my_ut, z: mz_ut },  // Magnetometer in µT
    quatOrientation                     // Device orientation
);
decorated.fused_mx = fused.x;  // Earth field + hard/soft iron REMOVED
decorated.fused_my = fused.y;  // = Residual field (finger magnet signal)
decorated.fused_mz = fused.z;
```

**Residual Calculation:**
```
Residual = Measured - (Earth Field + Hard Iron + Soft Iron correction)
```

The `MagneticTrajectory` displays this residual field in 3D isometric projection, showing the pure finger magnet signal after:
1. ✅ Hard iron offset subtracted
2. ✅ Soft iron matrix correction applied
3. ✅ Earth magnetic field subtracted (orientation-corrected)

---

## Critical Gaps

### Index.html Missing Features (vs Collector)

1. **❌ Magnetic Residual 3D Trajectory**
   - Missing: `MagneticTrajectory` module
   - Missing: Canvas element for trajectory visualization
   - Missing: Stats display for trajectory magnitude

2. **ThreeJSHandSkeleton Present (Good)**
   - Index has modern Three.js hand skeleton
   - Collector is missing this (should add)

### Collector-app.js Missing Features (vs Index)

1. **❌ ThreeJSHandSkeleton**
   - Missing: Modern Three.js WebGL hand renderer
   - Only has legacy 2D canvas renderer

---

## Recommendations

### 1. Remove Legacy Hand3DRenderer (Priority: HIGH)

**Action:** Remove `Hand3DRenderer` from BOTH implementations

**Reason:**
- It's 2D canvas-based (outdated)
- `ThreeJSHandSkeleton` is superior (WebGL, better performance, more features)
- Currently duplicated in both implementations
- Increases maintenance burden

**Steps:**
1. Remove `hand-3d-renderer.js` file
2. Remove `Hand3DRenderer` initialization from index.html
3. Remove `Hand3DRenderer` initialization from collector-app.js
4. Add `ThreeJSHandSkeleton` to collector-app.js
5. Update all orientation calls to use `threeHandSkeleton.updateOrientation()`

### 2. Add Magnetic Trajectory to Index.html (Priority: CRITICAL)

**Action:** Add `MagneticTrajectory` visualization to index.html

**Reason:**
- Essential for visualizing finger magnet signals
- Already working in collector
- User requirement: "ensure both index and collector display visually the residual magnetic signal"

**Steps:**
1. Import `MagneticTrajectory` module in index.html
2. Add canvas element: `<canvas id="magTrajectoryCanvas"></canvas>`
3. Initialize after telemetry processor setup:
   ```javascript
   const magTrajectory = new MagneticTrajectory($('magTrajectoryCanvas'), {
       maxPoints: 200,
       trajectoryColor: '#4ecdc4',
       autoNormalize: true,
       showMarkers: true,
       showCube: true
   });
   ```
4. Update telemetry callback to call:
   ```javascript
   if (decoratedData.fused_mx !== undefined) {
       magTrajectory.addPoint(
           decoratedData.fused_mx,
           decoratedData.fused_my,
           decoratedData.fused_mz
       );
   }
   ```

### 3. Standardize on ThreeJSHandSkeleton (Priority: HIGH)

**Action:** Use `ThreeJSHandSkeleton` in both implementations

**Current State:**
- Index: Has both (Hand3DRenderer + ThreeJSHandSkeleton)
- Collector: Only has Hand3DRenderer

**Target State:**
- Both: Only ThreeJSHandSkeleton

**Benefits:**
- Modern WebGL rendering
- Better performance
- Consistent across implementations
- Already has V2 corrected axis mapping

### 4. Verify Residual Field Calculation (Priority: MEDIUM)

**Action:** Confirm residual calculation is orientation-corrected

**Current Implementation:**
```javascript
// telemetry-processor.js:352-358
const fused = this.calibration.correct(
    { x: mx_ut, y: my_ut, z: mz_ut },
    quatOrientation  // ← Device orientation passed here
);
```

**Verify:**
1. Earth field is rotated to sensor frame using `quatOrientation`
2. Subtraction happens in sensor frame (not world frame)
3. Result is pure residual in sensor frame

**Expected Behavior:**
- When device rotates, Earth field contribution should be constant (removed)
- Only finger magnet field should vary with finger position
- Trajectory should show finger motion independent of device orientation

---

## Implementation Checklist

### Phase 1: Add Missing Features

- [ ] Add `MagneticTrajectory` to index.html
  - [ ] Import module
  - [ ] Add canvas element to HTML
  - [ ] Initialize visualization
  - [ ] Wire up telemetry updates
  - [ ] Add clear/pause controls
  - [ ] Add stats display

### Phase 2: Standardize Hand Rendering

- [ ] Add `ThreeJSHandSkeleton` to collector-app.js
  - [ ] Import module
  - [ ] Add container element
  - [ ] Initialize with correct offsets
  - [ ] Wire up orientation updates
  - [ ] Add toggle controls

- [ ] Remove `Hand3DRenderer` from index.html
  - [ ] Remove initialization code
  - [ ] Remove canvas element
  - [ ] Remove event handlers
  - [ ] Update orientation update paths

- [ ] Remove `Hand3DRenderer` from collector-app.js
  - [ ] Remove initialization code
  - [ ] Remove canvas element
  - [ ] Remove event handlers
  - [ ] Update orientation update paths

- [ ] Delete `hand-3d-renderer.js` file

### Phase 3: Verification

- [ ] Test index.html magnetic trajectory
  - [ ] Verify residual field displayed
  - [ ] Verify orientation correction works
  - [ ] Verify trajectory updates in real-time
  - [ ] Test clear/pause controls

- [ ] Test collector hand skeleton
  - [ ] Verify orientation mapping correct
  - [ ] Verify axis signs match index
  - [ ] Verify smooth animation
  - [ ] Test toggle controls

- [ ] Cross-verify both implementations
  - [ ] Side-by-side visual comparison
  - [ ] Same sensor data → same visualizations
  - [ ] Document any remaining differences

---

## Current Status

### Index.html (Gold Standard)

| Feature | Status | Notes |
|---------|--------|-------|
| Hand3DRenderer (2D canvas) | ✅ Present | LEGACY - should remove |
| ThreeJSHandSkeleton (WebGL) | ✅ Present | CORRECT - keep this |
| Orientation mapping | ✅ Correct | V2 axis mapping |
| Magnetic trajectory 3D viz | ❌ Missing | **CRITICAL GAP** |
| Residual field calculation | ✅ Present | Via TelemetryProcessor |

### Collector-app.js

| Feature | Status | Notes |
|---------|--------|-------|
| Hand3DRenderer (2D canvas) | ✅ Present | LEGACY - should remove |
| ThreeJSHandSkeleton (WebGL) | ❌ Missing | **Should add** |
| Orientation mapping | ✅ Correct | V2 axis mapping |
| Magnetic trajectory 3D viz | ✅ Present | CORRECT - port to index |
| Residual field calculation | ✅ Present | Via TelemetryProcessor |

---

## Conclusion

**Index.html and collector-app.js are NOT fully equivalent.**

**Critical Action Items:**

1. **Add magnetic trajectory to index.html** (user requirement)
2. **Remove legacy Hand3DRenderer from both** (code quality)
3. **Add ThreeJSHandSkeleton to collector** (consistency)

**After these changes, both implementations will be equivalent and use only modern, correct code.**
