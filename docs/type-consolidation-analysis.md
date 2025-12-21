# Type Consolidation Analysis

## Executive Summary

This analysis reviews all TypeScript types and interfaces across the SIMCAP codebase, identifying opportunities for consolidation (reducing duplication) and differentiation (clarifying semantic distinctions).

**Key Findings:**
- 6 major type duplications requiring consolidation
- 3 naming conflicts to resolve
- 4 areas where differentiation would improve clarity

---

## 1. Duplications Requiring Consolidation

### 1.1 Geometry Types: Vector3, Quaternion, EulerAngles

**Current State:** Defined in 4+ locations

| Location | Vector3 | Quaternion | EulerAngles |
|----------|---------|------------|-------------|
| `@core/types/telemetry.ts` | ✓ | ✓ | ✓ |
| `@filters/filters.ts` | ✓ | ✓ | ✓ |
| `src/types/globals.d.ts` | ✓ | ✓ | ✓ |
| `@orientation/orientation-model.ts` | - | - | ✓ |

**Impact:**
- Risk of drift between definitions
- No single source of truth
- Import confusion

**Recommendation:**
Create `packages/core/src/types/geometry.ts` as the canonical source:

```typescript
// packages/core/src/types/geometry.ts

/** 3D vector with x, y, z components */
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/** Quaternion for 3D rotation representation (w, x, y, z) */
export interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

/** Euler angles in degrees */
export interface EulerAngles {
  roll: number;   // Rotation around X axis
  pitch: number;  // Rotation around Y axis
  yaw: number;    // Rotation around Z axis
}

/** 3x3 matrix for rotation/transformation */
export type Matrix3x3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number]
];
```

**Migration Path:**
1. Create `geometry.ts` in core/types
2. Re-export from `@core/types` index
3. Update all imports to use `import { Vector3 } from '@core/types'`
4. Remove duplicate definitions from other files
5. Update `globals.d.ts` to reference core types

---

### 1.2 TelemetrySample Duplication

**Current State:** Defined in 2 locations with different structures

| Location | Structure |
|----------|-----------|
| `@core/types/telemetry.ts` | Extends `RawTelemetry`, adds `orientation?`, `fingerMagnet?` |
| `@api/types.ts` | Standalone with just 9 axes + timestamp |

**Analysis:**
```typescript
// Core version (full)
interface TelemetrySample extends RawTelemetry {
  orientation?: Quaternion;
  fingerMagnet?: { detected: boolean; confidence: number; };
}

// API version (minimal)
interface TelemetrySample {
  ax, ay, az, gx, gy, gz, mx, my, mz: number;
  t: number;
}
```

**Recommendation:**
Keep the API version as `RawTelemetry` (which it already matches), and have API import from core:

```typescript
// packages/api/src/types.ts
import type { RawTelemetry, TelemetrySample } from '@core/types';

// For uploads, use RawTelemetry (no computed fields)
export interface SessionData {
  samples: RawTelemetry[];  // Raw data for uploads
  // ...
}
```

---

### 1.3 SessionData Duplication

**Current State:** Defined in 2 locations with conflicting structures

| Location | Purpose | Fields |
|----------|---------|--------|
| `@core/types/session.ts` | Full session schema | version, device, calibration, geomagneticLocation, samples, labels, metadata |
| `@api/types.ts` | Upload/API schema | version, timestamp, samples, labels, metadata |

**Impact:**
- Confusion about which SessionData to use
- Type mismatches between storage and transmission

**Recommendation:**
Differentiate with clear names:

```typescript
// @core/types/session.ts - Full session with all data
export interface SessionData {
  version: string;
  device: DeviceInfo;
  calibration: CalibrationData;
  geomagneticLocation?: GeomagneticLocation;
  samples: TelemetrySample[];
  labels: LabelSegment[];
  metadata?: SessionMetadata;
}

// @api/types.ts - Payload for API transmission
export interface SessionPayload {
  version: string;
  timestamp: string;
  samples: RawTelemetry[];
  labels: LabelSegment[];
  metadata?: SessionMetadata;
}
```

---

### 1.4 SessionInfo Duplication

**Current State:** Defined in 2 locations (identical)

| Location | Fields |
|----------|--------|
| `@core/types/session.ts` | filename, pathname, url, downloadUrl, size, uploadedAt, timestamp |
| `@api/types.ts` | Same (with JSDoc comments) |

**Recommendation:**
Keep only in `@api/types.ts` (it's API response data), remove from core:

```typescript
// @api/types.ts
export interface SessionInfo {
  filename: string;
  pathname: string;
  url: string;
  downloadUrl: string;
  size: number;
  uploadedAt: string;
  timestamp: string;
}
```

---

### 1.5 FirmwareInfo Duplication

**Current State:** Defined in 3 locations

| Location | Name | Fields |
|----------|------|--------|
| `src/types/globals.d.ts` | `GambitFirmwareInfo` | name, version |
| `apps/gambit/gambit-client.ts` | `FirmwareInfo` | id, version, build?, uptime? |
| `apps/loader/loader-app.ts` | `FirmwareInfo` | name, id, version, author, uptime, features |

**Recommendation:**
Create canonical type in `@core/types/device.ts`:

```typescript
// @core/types/device.ts

/** Firmware identification from device */
export interface FirmwareInfo {
  /** Firmware identifier (e.g., "GAMBIT") */
  id: string;
  /** Semantic version (e.g., "1.2.0") */
  version: string;
  /** Build identifier */
  build?: string;
  /** Uptime in milliseconds */
  uptime?: number;
  /** Human-readable name */
  name?: string;
  /** Author information */
  author?: string;
  /** Feature flags */
  features?: string[];
}
```

Update `globals.d.ts` to use this type and apps to import from core.

---

### 1.6 Puck Types Duplication

**Current State:** Defined in 2 locations

| Location | Types |
|----------|-------|
| `packages/puck/src/types.ts` | Full typed definitions (PuckConnection, PuckStatic, etc.) |
| `apps/gambit/gambit-client.ts` | Inline minimal definitions |

**Recommendation:**
Import from `@puck/types` instead of redefining:

```typescript
// apps/gambit/gambit-client.ts
import type { PuckConnection, PuckStatic } from '@puck/types';

declare const Puck: PuckStatic;
```

---

## 2. Naming Conflicts to Resolve

### 2.1 FingerState vs FingerPosition

**Problem:** Two completely different concepts share similar names

| Location | Type | Meaning |
|----------|------|---------|
| `@core/types/session.ts` | `FingerState = 'extended' \| 'flexed' \| 'unknown'` | Labeling state |
| `@filters/filters.ts` | `FingerState = { x, y, z, vx, vy, vz }` | Position/velocity tracking |

**Recommendation:**
Rename to clarify semantics:

```typescript
// @core/types/hand.ts

/** Finger flexion label for data annotation */
export type FingerLabel = 'extended' | 'flexed' | 'unknown';

/** Per-finger labels */
export interface FingerLabels {
  thumb: FingerLabel;
  index: FingerLabel;
  middle: FingerLabel;
  ring: FingerLabel;
  pinky: FingerLabel;
}

// @filters/types.ts

/** Finger position and velocity state for tracking */
export interface FingerPosition {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}
```

---

### 2.2 MotionType vs MotionState

**Problem:** Same concept, different names

| Location | Name | Type |
|----------|------|------|
| `@core/types/session.ts` | `MotionType` | `'static' \| 'dynamic'` |
| `apps/gambit/modules/state.ts` | `MotionState` | `'static' \| 'dynamic'` |
| `@filters/filters.ts` | `MotionState` | `{ isMoving, accelStd, gyroStd }` |

**Recommendation:**
- Use `MotionLabel` for the labeling type (`'static' | 'dynamic'`)
- Keep `MotionState` for the detector state object

```typescript
// @core/types/session.ts
export type MotionLabel = 'static' | 'dynamic';

// @filters/filters.ts
export interface MotionDetectorState {
  isMoving: boolean;
  accelStd: number;
  gyroStd: number;
}
```

---

### 2.3 FingerStates Number vs String

**Problem:** Same interface name, different value types

| Location | Values |
|----------|--------|
| `@core/types/session.ts` | `FingerState` (string: extended/flexed/unknown) |
| `apps/gambit/hand-model.ts` | `number` (0-1 flexion amount) |

**Recommendation:**
```typescript
// @core/types/hand.ts
export interface FingerLabels {  // For annotation
  thumb: FingerLabel;  // 'extended' | 'flexed' | 'unknown'
  // ...
}

export interface FingerFlexion {  // For continuous tracking
  thumb: number;  // 0.0 (extended) to 1.0 (flexed)
  index: number;
  middle: number;
  ring: number;
  pinky: number;
}
```

---

## 3. Differentiation Opportunities

### 3.1 Raw vs Processed Telemetry Chain

**Current Pipeline:** Good structure, but could be clearer

```
RawTelemetry (LSB) → PhysicalTelemetry (units) → ProcessedTelemetry (orientation) → TelemetrySample (storage)
```

**Recommendation:** Document the pipeline and ensure imports flow correctly:

```typescript
// @core/types/telemetry.ts

/** Raw sensor reading in LSB from device (no processing) */
export interface RawTelemetry { ax, ay, az, gx, gy, gz, mx, my, mz, t: number }

/** Telemetry converted to physical units (g, deg/s, µT) */
export interface PhysicalTelemetry {
  accel: Vector3;      // g
  gyro: Vector3;       // deg/s
  mag: Vector3;        // µT
  timestamp: number;   // ms
}

/** With computed orientation */
export interface OrientedTelemetry extends PhysicalTelemetry {
  orientation: Quaternion;
  euler: EulerAngles;
}

/** Session sample with optional computed fields */
export interface TelemetrySample extends RawTelemetry {
  orientation?: Quaternion;
  fingerMagnet?: FingerMagnetDetection;
}
```

---

### 3.2 GeomagneticLocation vs GeomagneticReference

**Current State:** Good semantic difference, needs documentation

| Type | Purpose | Fields |
|------|---------|--------|
| `GeomagneticLocation` | Session metadata, includes location | city, country, lat, lon, declination, inclination, intensity, horizontal, vertical |
| `GeomagneticReference` | Filter configuration | horizontal, vertical, declination |

**Recommendation:** Document the relationship:

```typescript
// @core/types/session.ts

/** Full geomagnetic data with location context */
export interface GeomagneticLocation {
  city: string;
  country: string;
  lat: number;
  lon: number;
  declination: number;   // degrees
  inclination: number;   // degrees
  intensity: number;     // µT (total)
  horizontal: number;    // µT
  vertical: number;      // µT
}

// @filters/filters.ts

/**
 * Geomagnetic reference for orientation filter.
 * Subset of GeomagneticLocation - use toGeomagneticReference() to convert.
 */
export interface GeomagneticReference {
  horizontal: number;  // µT
  vertical: number;    // µT
  declination: number; // degrees
}

/** Convert full location to filter reference */
export function toGeomagneticReference(loc: GeomagneticLocation): GeomagneticReference {
  return { horizontal: loc.horizontal, vertical: loc.vertical, declination: loc.declination };
}
```

---

### 3.3 CalibrationState Extraction

**Current:** Inline union type in multiple places

```typescript
calibration?: 'none' | 'mag' | 'gyro'  // in LabelSegment
type CalibrationState = 'none' | 'mag' | 'gyro'  // in state.ts
```

**Recommendation:** Export from core:

```typescript
// @core/types/session.ts
export type CalibrationLabel = 'none' | 'mag' | 'gyro';

export interface LabelSegment {
  // ...
  calibration?: CalibrationLabel;
}
```

---

## 4. Proposed Type Organization

```
packages/core/src/types/
├── index.ts           # Re-exports all types
├── geometry.ts        # Vector3, Quaternion, EulerAngles, Matrix3x3
├── telemetry.ts       # RawTelemetry, PhysicalTelemetry, TelemetrySample
├── session.ts         # SessionData, LabelSegment, CalibrationData
├── hand.ts            # FingerLabel, FingerLabels, FingerFlexion
├── device.ts          # DeviceInfo, FirmwareInfo, CompatibilityResult
└── geomagnetic.ts     # GeomagneticLocation

packages/filters/src/
├── types.ts           # Filter-specific: GeomagneticReference, FingerPosition, MotionDetectorState
└── filters.ts         # Implementation (imports from types.ts)

packages/api/src/
├── types.ts           # API: SessionInfo, SessionPayload, UploadProgress, etc.
└── client.ts          # Implementation

packages/puck/src/
└── types.ts           # PuckConnection, PuckStatic, callbacks
```

---

## 5. Implementation Priority

### High Priority (Causes Active Issues)
1. **Consolidate Vector3/Quaternion/EulerAngles** - Multiple definitions risk drift
2. **Rename FingerState conflicts** - Same name, different meanings
3. **Import Puck types properly** - Eliminate inline redefinitions

### Medium Priority (Code Quality)
4. **Rename SessionData in API** - Clarify upload vs full session
5. **Consolidate FirmwareInfo** - Single source of truth
6. **Extract CalibrationLabel type** - DRY principle

### Low Priority (Nice to Have)
7. **Remove duplicate SessionInfo** - Already in API package
8. **Add GeomagneticReference converter** - Better documentation
9. **Rename MotionType → MotionLabel** - Consistency

---

## 6. Telemetry Processing Pipeline (Deep Dive)

The telemetry data flows through an 8-stage processing pipeline, with each stage adding decorated fields to the raw sample. Total: **47 fields** (10 raw + 37 decorated).

### Pipeline Stages

```
Device (Raw LSB)
     ↓
[Stage 1: Unit Conversion]
    → dt, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, mx_ut, my_ut, mz_ut
     ↓
[Stage 2: Motion Detection]
    → isMoving, accelStd, gyroStd
     ↓
[Stage 3: Gyro Bias Calibration]
    → gyroBiasCalibrated
     ↓
[Stage 4: IMU Fusion (Madgwick AHRS)]
    → orientation_w/x/y/z, euler_roll/pitch/yaw, ahrs_mag_residual_*
     ↓
[Stage 5: Magnetometer Calibration]
    → iron_mx/y/z, mag_cal_ready, mag_cal_confidence, mag_cal_*
     ↓
[Stage 6: Magnetic Residual]
    → residual_mx/y/z, residual_magnitude
     ↓
[Stage 7: Magnet Detection]
    → magnet_status, magnet_confidence, magnet_detected, magnet_deviation
     ↓
[Stage 8: Kalman Filtering]
    → filtered_mx/y/z
     ↓
DecoratedTelemetry (stored/exported)
```

### Type Hierarchy

```typescript
// Stage interfaces build on each other:
RawTelemetry                    // 10 fields: ax,ay,az,gx,gy,gz,mx,my,mz,t
  + UnitConvertedFields         // +10: dt, *_g, *_dps, *_ut
  + MotionDetectionFields       // +3: isMoving, accelStd, gyroStd
  + GyroBiasFields              // +1: gyroBiasCalibrated
  + OrientationFields           // +11: orientation_*, euler_*, ahrs_mag_residual_*
  + MagCalibrationFields        // +9: iron_*, mag_cal_*
  + MagResidualFields           // +4: residual_*
  + MagnetDetectionFields       // +6: magnet_*
  + KalmanFilteredFields        // +3: filtered_*
  = DecoratedTelemetry          // 47 total fields
```

### Key Implementation Files

| File | Purpose |
|------|---------|
| `apps/gambit/shared/telemetry-processor.ts` | Main pipeline (890 lines) |
| `packages/filters/src/filters.ts` | Madgwick AHRS, motion detection, Kalman |
| `apps/gambit/shared/unified-mag-calibration.ts` | Hard/soft iron, Earth field estimation |
| `apps/gambit/shared/magnet-detector.ts` | Finger magnet detection |
| `packages/core/src/types/telemetry.ts` | Type definitions for all stages |

---

## 7. Migration Checklist

### Completed ✅

- [x] Create `packages/core/src/types/geometry.ts` - Vector3, Quaternion, EulerAngles, Matrix types
- [x] Create `packages/core/src/types/hand.ts` - FingerLabel, FingerLabels, FingerFlexion, tracking types
- [x] Create `packages/core/src/types/device.ts` - DeviceInfo, FirmwareInfo, connection types
- [x] Update `packages/core/src/types/telemetry.ts` - Full 8-stage pipeline types
- [x] Update `packages/core/src/types/session.ts` - Use new types, add GeomagneticReference
- [x] Update `packages/core/src/types/index.ts` exports
- [x] TypeScript compilation passes

### Remaining

- [ ] Update `packages/filters/src/filters.ts` to import geometry types from core
- [ ] Update `src/types/globals.d.ts` to reference core types
- [ ] Update `apps/gambit/gambit-client.ts` to import Puck types from @puck
- [ ] Update `apps/gambit/shared/telemetry-processor.ts` to import DecoratedTelemetry from core
- [ ] Update `apps/gambit/modules/telemetry-handler.ts` to import types from core
- [ ] Rename API `SessionData` → `SessionPayload`
- [ ] Update all import statements across codebase
- [ ] Update tests
