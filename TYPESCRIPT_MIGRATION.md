# TypeScript Migration Plan for SIMCAP

## Current Status (Updated: 2025-12-17)

### ✅ TypeScript Migration Complete

All JavaScript application code has been converted to TypeScript. The migration is functionally complete.

#### API Routes (100% TypeScript)
- `api/sessions.ts` ✓
- `api/upload.ts` ✓
- `api/visualizations.ts` ✓
- `api/explorer.ts` ✓

#### Web Application Entry Points (100% TypeScript modules)
- `gambit-app.ts` - Main GAMBIT application (89KB)
- `collector-app.ts` - Data collection app (58KB)
- `synth-app.ts` - Audio synthesizer app (~460 lines extracted from inline)
- `loader-app.ts` - Firmware loader (~1318 lines extracted from inline)

#### Core Libraries (100% TypeScript)
- `gambit-client.ts` - BLE device communication with typed event overloads
- `filters.ts` - MadgwickAHRS, ComplementaryFilter, MotionDetector
- `kalman.ts` - KalmanFilter, KalmanFilter3D
- `gesture-inference.ts` - TensorFlow.js gesture recognition
- `hand-model.ts` - Hand geometry types
- `calibration-config.ts` - Calibration configuration

#### Modules Directory (100% TypeScript)
- All 10+ modules in `src/web/GAMBIT/modules/` converted

#### Cleanup Completed
- Removed all duplicate .js files where .ts versions exist
- Removed `window.GambitClient` patterns - using proper ES module imports
- Updated globals.d.ts for external libraries only

### 📁 Files Intentionally Kept as JavaScript

These files are NOT TypeScript because they serve specific purposes:

1. **Device Firmware** (runs on Espruino, not in browser):
   - `src/device/GAMBIT/app.js`
   - `src/device/BAE/app.js`
   - `src/device/KEYBOARD/app.js`
   - `src/device/MOUSE/app.js`
   - `src/device/P0/*.js`

2. **External Libraries** (vendored, not our code):
   - `src/web/GAMBIT/puck.js` - Espruino Puck.js BLE library
   - `src/web/P0/assets/*.js` - D3, jQuery, Rickshaw

### 🔮 Optional Future Phases (Structural Reorganization)

These phases are about project structure improvement, not TypeScript conversion:

- **Phase 3**: Create `packages/filters/`, `packages/orientation/`, `packages/puck/`
  - Would split existing TypeScript into reusable packages
  - Not required for TypeScript migration

- **Phase 4**: Restructure to `apps/` directory
  - Would move `src/web/*` to `apps/*`
  - Not required for TypeScript migration

---

## Strategy: Vite + Reorganized Project Structure

Vite provides HMR, excellent TypeScript support, and encourages better code organization. This plan restructures the project for long-term maintainability.

---

## Target Project Structure

```
simcap/
├── api/                          # Vercel serverless (unchanged location)
│   ├── sessions.ts
│   ├── upload.ts
│   └── visualizations.ts
│
├── apps/                         # Web applications (Vite entry points)
│   ├── gambit/                   # Main GAMBIT app
│   │   ├── index.html
│   │   ├── main.ts               # Entry point (extracted from inline)
│   │   ├── components/           # UI components
│   │   │   ├── cube-visualizer.ts
│   │   │   ├── telemetry-display.ts
│   │   │   └── calibration-panel.ts
│   │   └── features/             # Feature modules
│   │       ├── connection/
│   │       ├── recording/
│   │       └── inference/
│   │
│   ├── collector/                # Data collection app
│   │   ├── index.html
│   │   ├── main.ts
│   │   └── components/
│   │
│   ├── viz/                      # Visualization explorer
│   │   ├── index.html
│   │   └── main.ts
│   │
│   └── loader/                   # Firmware loader
│       ├── index.html
│       └── main.ts
│
├── packages/                     # Shared packages
│   ├── core/                     # Core types & utilities
│   │   ├── src/
│   │   │   ├── types/
│   │   │   │   ├── telemetry.ts
│   │   │   │   ├── session.ts
│   │   │   │   ├── calibration.ts
│   │   │   │   └── index.ts
│   │   │   ├── sensor/
│   │   │   │   ├── config.ts
│   │   │   │   ├── units.ts
│   │   │   │   └── index.ts
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── filters/                  # Signal processing (converted globals)
│   │   ├── src/
│   │   │   ├── madgwick.ts
│   │   │   ├── kalman.ts
│   │   │   ├── motion-detector.ts
│   │   │   └── index.ts
│   │   └── package.json
│   │
│   ├── orientation/              # Quaternion/Euler math
│   │   ├── src/
│   │   │   ├── model.ts
│   │   │   ├── calibration.ts
│   │   │   └── index.ts
│   │   └── package.json
│   │
│   └── puck/                     # BLE device communication
│       ├── src/
│       │   ├── connection.ts
│       │   ├── protocol.ts
│       │   └── index.ts
│       └── package.json
│
├── public/                       # Static assets (copied to dist)
│   ├── assets/
│   │   ├── fonts/
│   │   └── images/
│   └── models/                   # TF.js models
│       └── gesture_v1/
│
├── data/                         # Session data (Git LFS)
│   └── GAMBIT/
│
├── vite.config.ts
├── tsconfig.json
├── package.json
└── vercel.json
```

---

## Phase 0: Foundation Setup

### Install Dependencies

```bash
npm install -D vite typescript @types/node
npm install -D @vercel/node  # Types for API routes
```

### Root `tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "baseUrl": ".",
    "paths": {
      "@core/*": ["packages/core/src/*"],
      "@filters/*": ["packages/filters/src/*"],
      "@orientation/*": ["packages/orientation/src/*"],
      "@puck/*": ["packages/puck/src/*"]
    }
  },
  "include": ["apps/**/*", "packages/**/*", "api/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### `vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',

  // Path aliases matching tsconfig
  resolve: {
    alias: {
      '@core': resolve(__dirname, 'packages/core/src'),
      '@filters': resolve(__dirname, 'packages/filters/src'),
      '@orientation': resolve(__dirname, 'packages/orientation/src'),
      '@puck': resolve(__dirname, 'packages/puck/src'),
    }
  },

  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        gambit: resolve(__dirname, 'apps/gambit/index.html'),
        collector: resolve(__dirname, 'apps/collector/index.html'),
        viz: resolve(__dirname, 'apps/viz/index.html'),
        loader: resolve(__dirname, 'apps/loader/index.html'),
      }
    }
  },

  // Dev server
  server: {
    port: 3000,
    open: '/apps/gambit/'
  },

  // Optimize external dependencies
  optimizeDeps: {
    include: ['three', '@tensorflow/tfjs']
  }
});
```

### `package.json` Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "typecheck": "tsc --noEmit",
    "lint": "eslint apps packages api --ext .ts,.tsx"
  }
}
```

### `vercel.json`

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "installCommand": "npm install",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [
    { "source": "/gambit/(.*)", "destination": "/apps/gambit/$1" },
    { "source": "/collector/(.*)", "destination": "/apps/collector/$1" },
    { "source": "/viz/(.*)", "destination": "/apps/viz/$1" },
    { "source": "/loader/(.*)", "destination": "/apps/loader/$1" }
  ]
}
```

---

## Current Status (December 2024)

### Completed
- [x] Phase 0: Foundation Setup (Vite + TypeScript config)
- [x] Phase 1: API Routes converted to TypeScript
  - `api/sessions.ts` ✓
  - `api/upload.ts` ✓
  - `api/visualizations.ts` ✓
- [x] Core filter modules converted:
  - `kalman.ts` - KalmanFilter, KalmanFilter3D
  - `filters.ts` - MadgwickAHRS, ComplementaryFilter, MotionDetector, KalmanFilter6D, MultiFingerKalmanFilter, ParticleFilter, magneticLikelihood
- [x] App entry points:
  - `gambit-app.ts` - Main GAMBIT application entry
  - `modules/session-playback.ts` - Session playback functionality

### In Progress
- [ ] Convert remaining JS files:
  - `gambit-client.js` - BLE client for Puck.js
  - `gesture-inference.js` - TensorFlow.js gesture recognition
  - `hand-model.js` - Three.js hand visualization
  - `puck.js` - External Puck.js library (keep as-is)
  - `api/explorer.js` - API explorer utility

### Remaining Work
- [ ] Update HTML files to use TypeScript imports via Vite
- [ ] Create packages/core types package
- [ ] Migrate shared/ modules to packages/

---

## Phase 1: API Routes (1-2 hours)

Vercel natively compiles TypeScript. Just rename and add types.

### `api/sessions.ts`

```typescript
import { list } from '@vercel/blob';
import type { VercelRequest, VercelResponse } from '@vercel/node';

interface SessionInfo {
  filename: string;
  pathname: string;
  url: string;
  downloadUrl: string;
  size: number;
  uploadedAt: string;
  timestamp: string;
}

interface SessionsResponse {
  sessions: SessionInfo[];
  count: number;
  generatedAt: string;
}

export default async function handler(
  request: VercelRequest,
  response: VercelResponse<SessionsResponse | { error: string }>
) {
  if (request.method !== 'GET') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  const { blobs } = await list({ prefix: 'sessions/', limit: 1000 });

  const sessions: SessionInfo[] = blobs
    .filter(blob => blob.pathname.endsWith('.json'))
    .map(blob => ({
      filename: blob.pathname.replace('sessions/', ''),
      pathname: blob.pathname,
      url: blob.url,
      downloadUrl: blob.downloadUrl,
      size: blob.size,
      uploadedAt: blob.uploadedAt,
      timestamp: blob.pathname.replace('sessions/', '').replace('.json', '').replace(/_/g, ':'),
    }))
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  response.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=300');
  return response.status(200).json({
    sessions,
    count: sessions.length,
    generatedAt: new Date().toISOString(),
  });
}
```

### Checklist
- [x] Rename `api/sessions.js` → `api/sessions.ts`
- [x] Rename `api/upload.js` → `api/upload.ts`
- [x] Rename `api/visualizations.js` → `api/visualizations.ts`
- [x] Add type annotations
- [ ] Test with `vercel dev`

---

## Phase 2: Core Types Package (1 day)

Create the foundational types that all apps share.

### `packages/core/src/types/telemetry.ts`

```typescript
/** Raw sensor reading in LSB (Least Significant Bits) */
export interface RawTelemetry {
  ax: number;  // Accelerometer X (LSB)
  ay: number;  // Accelerometer Y (LSB)
  az: number;  // Accelerometer Z (LSB)
  gx: number;  // Gyroscope X (LSB)
  gy: number;  // Gyroscope Y (LSB)
  gz: number;  // Gyroscope Z (LSB)
  mx: number;  // Magnetometer X (LSB)
  my: number;  // Magnetometer Y (LSB)
  mz: number;  // Magnetometer Z (LSB)
  t: number;   // Timestamp (ms)
}

/** Telemetry converted to physical units */
export interface PhysicalTelemetry {
  accel: Vector3;      // g's
  gyro: Vector3;       // deg/s
  mag: Vector3;        // μT
  timestamp: number;   // ms
}

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

export interface EulerAngles {
  roll: number;   // degrees
  pitch: number;  // degrees
  yaw: number;    // degrees
}
```

### `packages/core/src/types/session.ts`

```typescript
import type { RawTelemetry, Quaternion } from './telemetry';

export interface SessionData {
  version: string;
  device: DeviceInfo;
  calibration: CalibrationData;
  geomagneticLocation?: GeomagneticLocation;
  samples: TelemetrySample[];
  labels: LabelSegment[];
}

export interface DeviceInfo {
  id: string;
  firmware: string;
  hardware?: string;
}

export interface TelemetrySample extends RawTelemetry {
  orientation?: Quaternion;
}

export interface LabelSegment {
  startIndex: number;
  endIndex: number;
  pose?: string;
  fingers?: FingerStates;
  motion?: MotionType;
  custom?: string[];
}

export interface FingerStates {
  thumb: FingerState;
  index: FingerState;
  middle: FingerState;
  ring: FingerState;
  pinky: FingerState;
}

export type FingerState = 'extended' | 'flexed' | 'unknown';
export type MotionType = 'static' | 'dynamic';
```

### `packages/core/src/sensor/config.ts`

```typescript
/** LSM6DS3 accelerometer scale: LSB per g at ±2g range */
export const ACCEL_SCALE = 8192;

/** LSM6DS3 gyroscope scale: LSB per deg/s at 245dps range */
export const GYRO_SCALE = 114.28;

/** Default sample frequency for GAMBIT firmware (Hz) */
export const DEFAULT_SAMPLE_FREQ = 26;

/** Convert accelerometer LSB to g's */
export function accelLsbToG(lsb: number): number {
  return lsb / ACCEL_SCALE;
}

/** Convert gyroscope LSB to deg/s */
export function gyroLsbToDps(lsb: number): number {
  return lsb / GYRO_SCALE;
}
```

---

## Phase 3: Migrate Shared Modules (2-3 days)

Move existing `/shared/` modules into packages.

### Migration Map

| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `shared/sensor-config.js` | `packages/core/src/sensor/config.ts` | Pure functions |
| `shared/sensor-units.js` | `packages/core/src/sensor/units.ts` | Pure functions |
| `shared/orientation-model.js` | `packages/orientation/src/model.ts` | Quaternion math |
| `shared/orientation-calibration.js` | `packages/orientation/src/calibration.ts` | Calibration |
| `shared/telemetry-processor.js` | `packages/core/src/telemetry/processor.ts` | Uses above |
| `shared/geomagnetic-field.js` | `packages/core/src/geo/field.ts` | Lookup tables |
| `shared/blob-upload.js` | `packages/core/src/api/upload.ts` | API client |
| `filters.js` | `packages/filters/src/` | Split into files |
| `kalman.js` | `packages/filters/src/kalman.ts` | Signal filter |
| `puck.js` | `packages/puck/src/` | BLE connection |

### Example: Converting filters.js

```typescript
// packages/filters/src/madgwick.ts

export interface MadgwickOptions {
  sampleFreq?: number;
  beta?: number;
}

export class MadgwickAHRS {
  private sampleFreq: number;
  private beta: number;
  private q: [number, number, number, number];

  constructor(options: MadgwickOptions = {}) {
    this.sampleFreq = options.sampleFreq ?? 26;
    this.beta = options.beta ?? 0.05;
    this.q = [1, 0, 0, 0];
  }

  update(gx: number, gy: number, gz: number, ax: number, ay: number, az: number): void {
    // ... algorithm implementation
  }

  getQuaternion() {
    return { w: this.q[0], x: this.q[1], y: this.q[2], z: this.q[3] };
  }
}
```

---

## Phase 4: App Modules (2-3 days)

Move `/modules/` into app-specific feature folders.

### Migration Map

| Current Location | New Location |
|-----------------|--------------|
| `modules/state.js` | `apps/collector/state.ts` |
| `modules/logger.js` | `packages/core/src/utils/logger.ts` |
| `modules/connection-manager.js` | `apps/collector/features/connection/manager.ts` |
| `modules/telemetry-handler.js` | `apps/collector/features/telemetry/handler.ts` |
| `modules/recording-controls.js` | `apps/collector/features/recording/controls.ts` |
| `modules/calibration-ui.js` | `apps/collector/components/calibration.ts` |

### State Management with Types

```typescript
// apps/collector/state.ts
import type { TelemetrySample, LabelSegment, FingerStates } from '@core/types';

export interface AppState {
  connected: boolean;
  recording: boolean;
  paused: boolean;
  sessionData: TelemetrySample[];
  labels: LabelSegment[];
  currentLabelStart: number | null;
  gambitClient: GambitClient | null;
  firmwareVersion: string | null;
  currentLabels: CurrentLabels;
}

export interface CurrentLabels {
  pose: string | null;
  fingers: FingerStates;
  motion: 'static' | 'dynamic';
  calibration: 'none' | 'mag' | 'gyro';
  custom: string[];
}

export const state: AppState = {
  connected: false,
  recording: false,
  paused: false,
  sessionData: [],
  labels: [],
  currentLabelStart: null,
  gambitClient: null,
  firmwareVersion: null,
  currentLabels: {
    pose: null,
    fingers: { thumb: 'unknown', index: 'unknown', middle: 'unknown', ring: 'unknown', pinky: 'unknown' },
    motion: 'static',
    calibration: 'none',
    custom: []
  }
};
```

---

## Phase 5: Extract Inline Scripts (1 week)

The largest refactor: move 3,600 lines from `index.html` into modules.

### Strategy

1. **Create entry point** `apps/gambit/main.ts`
2. **Extract by feature:**
   - Connection handling → `features/connection/`
   - Telemetry processing → `features/telemetry/`
   - 3D visualization → `components/cube-visualizer.ts`
   - Gesture inference → `features/inference/`
   - UI updates → `components/`
3. **Replace inline script** with single import

### Example Entry Point

```typescript
// apps/gambit/main.ts
import { initConnection } from './features/connection';
import { initTelemetry } from './features/telemetry';
import { CubeVisualizer } from './components/cube-visualizer';
import { GestureInference } from './features/inference';

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const cube = new CubeVisualizer('#cube-container');
  const inference = new GestureInference('/models/gesture_v1/model.json');

  initConnection({
    onConnect: (client) => {
      console.log('Connected to GAMBIT');
    },
    onTelemetry: (data) => {
      cube.updateOrientation(data.quaternion);
      inference.process(data);
    }
  });
});
```

### Updated HTML

```html
<!-- apps/gambit/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GAMBIT</title>
  <link rel="stylesheet" href="/assets/simcap.css">
</head>
<body>
  <div id="app">
    <!-- UI structure -->
  </div>
  <script type="module" src="./main.ts"></script>
</body>
</html>
```

Vite handles the TypeScript compilation automatically.

---

## Incremental Migration Path

You don't need to restructure everything at once. Here's how to migrate incrementally:

### Week 1: Foundation ✅ COMPLETE
- [x] Set up Vite + TypeScript config
- [x] Convert API routes to TypeScript
- [x] Create `packages/core/src/types/`

### Week 2: Core Packages (IN PROGRESS)
- [ ] Migrate `shared/sensor-*.js` → `packages/core/`
- [ ] Migrate `shared/orientation-*.js` → `packages/orientation/`
- [x] Create type declarations for globals (temporary) - `src/types/globals.d.ts`

### Week 3: Collector App (PARTIAL)
- [ ] Move `collector.html` → `apps/collector/index.html`
- [x] Convert `modules/*.js` → TypeScript (session-playback.ts done)
- [ ] Update imports to use packages

### Week 4+: Main App (IN PROGRESS)
- [x] Extract inline scripts from `index.html` - gambit-app.ts created
- [x] Create `apps/gambit/main.ts` - using gambit-app.ts as entry point
- [ ] Migrate feature by feature

### Additional Progress
- [x] `kalman.ts` - Kalman filter with ES module exports
- [x] `filters.ts` - MadgwickAHRS, ComplementaryFilter, MotionDetector
- [x] `gesture-inference.ts` - GestureInference, FingerTrackingInference
- [x] `gambit-client.ts` - BLE device communication
- [x] `hand-model.ts` - Hand geometry types
- [x] `session-playback.ts` - Session playback module
- [x] HTML files updated to use ES module entry points

---

## Temporary Compatibility Layer

During migration, create type declarations for unconverted globals:

```typescript
// src/types/globals.d.ts
declare class MadgwickAHRS {
  constructor(options?: { sampleFreq?: number; beta?: number });
  update(gx: number, gy: number, gz: number, ax: number, ay: number, az: number): void;
  updateIMU(gx: number, gy: number, gz: number, ax: number, ay: number, az: number, mx: number, my: number, mz: number): void;
  getQuaternion(): { w: number; x: number; y: number; z: number };
}

declare class KalmanFilter {
  constructor(options?: { R?: number; Q?: number });
  filter(value: number): number;
}

declare const Puck: {
  connect(callback: (connection: any) => void): Promise<void>;
  write(data: string, callback?: () => void): void;
};
```

This lets you use globals with type safety while migrating.

---

## Summary

| Phase | Scope | Effort | Vite Benefit |
|-------|-------|--------|--------------|
| 0 | Setup Vite + TS | 2-4 hours | Foundation |
| 1 | API routes | 1-2 hours | Native TS |
| 2 | Core types | 1 day | Path aliases |
| 3 | Shared modules | 2-3 days | HMR during dev |
| 4 | App modules | 2-3 days | Fast rebuilds |
| 5 | Inline extraction | 1 week | Module splitting |

**Key benefits of Vite approach:**
- Hot Module Replacement for fast development
- Native TypeScript support (no separate tsc step for dev)
- Path aliases (`@core/`, `@filters/`) for clean imports
- Automatic code splitting in production
- Better long-term organization with packages structure
