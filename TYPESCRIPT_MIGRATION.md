# TypeScript Migration Plan for SIMCAP

## Current State Analysis

| Layer | Pattern | Complexity | Priority |
|-------|---------|------------|----------|
| **Vercel API** (`/api/`) | Pure ES modules | Low | **Phase 1** |
| **Shared modules** (`/shared/`) | ES modules | Medium | **Phase 2** |
| **App modules** (`/modules/`) | ES modules | Medium | **Phase 2** |
| **Global scripts** (puck.js, filters.js) | Window globals | High | Phase 4 (defer) |
| **Inline HTML scripts** | `<script type="module">` | Very High | Phase 5 (defer) |

## Recommended Strategy: Incremental esbuild + Vercel Native TS

### Why This Approach

1. **Vercel supports TypeScript natively** for API routes - zero config
2. **esbuild** is fastest, simplest bundler - 100x faster than webpack
3. **No Vite** - preserves current HTML-centric static serving (no dev server required)
4. **Incremental** - existing JS continues to work during migration
5. **Type safety** without disrupting working code

---

## Phase 1: API Routes (Immediate Win)

Vercel auto-compiles TypeScript API routes. Just rename files:

```bash
api/sessions.js    → api/sessions.ts
api/upload.js      → api/upload.ts
api/visualizations.js → api/visualizations.ts
```

**Setup:**

```bash
npm install -D typescript @types/node @vercel/blob
```

Create `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "outDir": "dist",
    "rootDir": ".",
    "baseUrl": ".",
    "paths": {
      "@shared/*": ["src/web/GAMBIT/shared/*"]
    }
  },
  "include": ["api/**/*", "src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**Example conversion** (`api/sessions.ts`):

```typescript
import { list, ListBlobResult } from '@vercel/blob';
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

export default async function handler(
  request: VercelRequest,
  response: VercelResponse
) {
  if (request.method !== 'GET') {
    return response.status(405).json({ error: 'Method not allowed' });
  }
  // ... rest of implementation with types
}
```

**Effort:** ~1-2 hours for all 3 API files

---

## Phase 2: Shared Modules (Core Type Safety)

Convert ES modules in `/shared/` and `/modules/` to TypeScript. These are the highest-value targets because they define core interfaces.

**Setup build script** in `package.json`:

```json
{
  "scripts": {
    "build": "esbuild src/web/GAMBIT/shared/*.ts src/web/GAMBIT/modules/*.ts --outdir=src/web/GAMBIT/dist --format=esm --bundle=false",
    "build:watch": "npm run build -- --watch",
    "typecheck": "tsc --noEmit"
  },
  "devDependencies": {
    "esbuild": "^0.24.0",
    "typescript": "^5.7.0"
  }
}
```

**Key interfaces to define first** (`src/web/GAMBIT/shared/types.ts`):

```typescript
// Core telemetry types
export interface RawTelemetry {
  ax: number; ay: number; az: number;  // Accelerometer (LSB)
  gx: number; gy: number; gz: number;  // Gyroscope (LSB)
  mx: number; my: number; mz: number;  // Magnetometer (LSB)
  t: number;                            // Timestamp
}

export interface ProcessedTelemetry extends RawTelemetry {
  quaternion: Quaternion;
  euler: EulerAngles;
}

export interface Quaternion {
  w: number; x: number; y: number; z: number;
}

export interface EulerAngles {
  roll: number; pitch: number; yaw: number;
}

// Session format
export interface SessionData {
  version: string;
  device: DeviceInfo;
  samples: TelemetrySample[];
  labels: LabelSegment[];
  calibration: CalibrationData;
}

// Global dependencies (from legacy scripts)
declare global {
  interface Window {
    MadgwickAHRS: typeof MadgwickAHRS;
    KalmanFilter: typeof KalmanFilter;
    KalmanFilter3D: typeof KalmanFilter3D;
    MotionDetector: typeof MotionDetector;
    Puck: typeof Puck;
  }
}
```

**Migration order for shared modules:**

1. `types.ts` (new - core interfaces)
2. `sensor-config.ts` (pure functions, no deps)
3. `sensor-units.ts` (pure functions)
4. `orientation-model.ts` (math utilities)
5. `geomagnetic-field.ts` (lookup tables)
6. `telemetry-processor.ts` (uses above)
7. `blob-upload.ts` (API client)

**File structure after Phase 2:**

```
src/web/GAMBIT/
├── shared/
│   ├── types.ts              # Core type definitions
│   ├── sensor-config.ts      # Converted from .js
│   ├── sensor-units.ts
│   ├── orientation-model.ts
│   └── index.ts              # Re-exports all
├── dist/                     # esbuild output (ES modules)
│   ├── types.js
│   ├── sensor-config.js
│   └── ...
```

**HTML import change:**

```html
<!-- Before -->
<script type="module">
  import { ACCEL_SCALE } from './shared/sensor-config.js';
</script>

<!-- After -->
<script type="module">
  import { ACCEL_SCALE } from './dist/sensor-config.js';
</script>
```

**Effort:** ~1-2 days for shared modules

---

## Phase 3: App Modules

Convert `/modules/` (collector-app components) following same pattern:

1. `state.ts` - Add interfaces for state shape
2. `logger.ts` - Simple utility
3. `connection-manager.ts` - BLE connection logic
4. `telemetry-handler.ts` - Uses shared types
5. `recording-controls.ts`
6. `calibration-ui.ts`
7. `wizard.ts`

**These modules already use clean ES module patterns**, making conversion straightforward.

**Effort:** ~2-3 days

---

## Phase 4: Global Scripts (Optional/Deferred)

The global scripts (`puck.js`, `filters.js`, `kalman.js`) work well as-is. Options:

### Option A: Type Declarations Only (Recommended)

Create `.d.ts` files without modifying original JS:

```typescript
// src/types/filters.d.ts
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
```

### Option B: Full Conversion (Future)

Only if major changes needed. These are stable, tested algorithms.

---

## Phase 5: Inline HTML Scripts (Long-term)

The 3,600-line `<script type="module">` in `index.html` is the largest migration:

### Strategy: Extract to Entry Point

1. Create `src/web/GAMBIT/app.ts` as main entry
2. Move inline logic to importable modules
3. Replace inline script with single import:

```html
<!-- index.html - After extraction -->
<script type="module" src="./dist/app.js"></script>
```

**This is a significant refactor** - defer until Phases 1-3 complete.

---

## Build & Deploy Configuration

### Development Workflow

```bash
# Terminal 1: Watch TypeScript compilation
npm run build:watch

# Terminal 2: Local server (any static server works)
npx serve .
```

### Vercel Build

Update `vercel.json`:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "installCommand": "npm install",
  "buildCommand": "npm run build"
}
```

### CI Type Checking

Add to GitHub Actions:

```yaml
- name: Type Check
  run: npm run typecheck
```

---

## Migration Checklist

### Phase 1: API (1-2 hours)
- [ ] Add TypeScript + @types/node
- [ ] Create tsconfig.json
- [ ] Rename api/sessions.js → .ts + add types
- [ ] Rename api/upload.js → .ts + add types
- [ ] Rename api/visualizations.js → .ts + add types
- [ ] Test deployment

### Phase 2: Shared Modules (1-2 days)
- [ ] Add esbuild
- [ ] Create shared/types.ts
- [ ] Convert sensor-config.js → .ts
- [ ] Convert sensor-units.js → .ts
- [ ] Convert orientation-model.js → .ts
- [ ] Convert remaining shared modules
- [ ] Update HTML imports to use /dist/

### Phase 3: App Modules (2-3 days)
- [ ] Convert state.js → .ts
- [ ] Convert logger.js → .ts
- [ ] Convert remaining modules
- [ ] Update collector-app.js → .ts

### Phase 4: Type Declarations (Optional)
- [ ] Create filters.d.ts
- [ ] Create puck.d.ts
- [ ] Create kalman.d.ts

### Phase 5: Index.html Extraction (Future)
- [ ] Create app.ts entry point
- [ ] Extract inline modules
- [ ] Bundle with esbuild

---

## Alternative: Vite (If More Tooling Desired)

If you want HMR and more developer tooling, Vite is an option:

```bash
npm install -D vite
```

```typescript
// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  root: 'src/web/GAMBIT',
  build: {
    outDir: '../../../dist/GAMBIT',
    rollupOptions: {
      input: {
        main: 'src/web/GAMBIT/index.html',
        collector: 'src/web/GAMBIT/collector.html'
      }
    }
  }
});
```

**Tradeoff:** Vite requires restructuring HTML to work with its dev server. The esbuild approach preserves current static-file workflow.

---

## Summary

| Phase | Scope | Effort | Value |
|-------|-------|--------|-------|
| 1 | API routes | 1-2 hours | High (immediate safety) |
| 2 | Shared modules | 1-2 days | High (core type safety) |
| 3 | App modules | 2-3 days | Medium (collector app) |
| 4 | Type declarations | 2-4 hours | Low (optional) |
| 5 | HTML extraction | 1 week+ | Medium (major refactor) |

**Recommended start:** Phase 1 (API) this week, Phase 2 (shared) next sprint.
