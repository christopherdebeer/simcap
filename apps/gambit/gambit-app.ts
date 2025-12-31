/**
 * =====================================================================
 * GAMBIT Application Entry Point
 * =====================================================================
 *
 * Main application module for the GAMBIT sensor visualization interface.
 * Extracted from index.html inline script for TypeScript migration.
 */

import type { GeomagneticLocation } from './shared/geomagnetic-field.js';
import type { EulerAngles, TelemetrySample } from '@core/types';
import type { ReferencePose, ValidationQuestion, CouplingQuestion, Observation } from './shared/orientation-calibration.js';
import type { SessionMetadata, PlaybackState } from './modules/session-playback.js';
import type { GestureUIController } from './modules/gesture-inference-module.js';

// ===== Import Global Classes (previously loaded via <script> tags) =====
// These are now ES modules that also export to window for backward compatibility
import { GambitClient } from './gambit-client';
import { KalmanFilter } from '@filters';

// Import gesture-inference to set up window globals before gesture-inference-module uses them
import './gesture-inference.js';

// ===== Type Definitions =====

interface DataStageInfo {
    label: string;
    info: string;
    magUnit: string;
    accUnit: string;
    gyrUnit: string;
}

type DataStage = 'raw' | 'converted' | 'calibrated' | 'fused' | 'filtered';

/** Tracking state for min/max normalization */
interface MinMaxEntry {
    values: number[];
    min: number;
    max: number;
    kalman: KalmanFilter;
    kalmanValues: number[];
}

/** Playback UI element references */
interface PlaybackUIElements {
    select: HTMLSelectElement | null;
    playBtn: HTMLButtonElement | null;
    stopBtn: HTMLButtonElement | null;
    slider: HTMLInputElement | null;
    timeEl: HTMLElement | null;
    statusEl: HTMLElement | null;
    card: HTMLElement | null;
    speedSelect: HTMLSelectElement | null;
}

/** Extended telemetry with computed values for playback */
interface ExtendedTelemetry extends TelemetrySample {
    /** Pre-computed Euler roll from session playback */
    euler_roll?: number;
    /** Pre-computed Euler pitch from session playback */
    euler_pitch?: number;
    /** Pre-computed Euler yaw from session playback */
    euler_yaw?: number;
}

/** Minimal sensor data for calibration and internal use (subset of RawTelemetry) */
interface SensorSnapshot {
    ax: number;
    ay: number;
    az: number;
    gx: number;
    gy: number;
    gz: number;
    mx?: number;
    my?: number;
    mz?: number;
    t?: number;
}

// ===== Window Augmentation =====

declare global {
    interface Window {
        connect: HTMLButtonElement;
        getdata: HTMLButtonElement;
        saveSession: (showResult?: boolean) => Promise<void>;
        clearSession: () => void;
        resetCamera: () => void;
    }
}

// ===== DOM Helpers =====


/**
 * =====================================================================
 * SENSOR-TO-HAND ORIENTATION MAPPING DOCUMENTATION
 * =====================================================================
 *
 * This file uses sensor data from a Puck.js device to drive 3D hand
 * visualization. Understanding the coordinate systems is CRITICAL.
 *
 * SENSOR PLACEMENT:
 * - Puck.js positioned on PALM of hand (battery side against palm skin)
 * - MQBT42Q module/aerial facing toward WRIST
 * - LEDs facing toward FINGERS (and toward ceiling when palm-up)
 *
 * =====================================================================
 * COORDINATE FRAME DEFINITIONS
 * =====================================================================
 *
 * ACCELEROMETER & GYROSCOPE AXES (LSM6DS3):
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  +X axis → toward WRIST (aerial/MQBT42Q direction)              │
 * │  +Y axis → toward FINGERS (IR LED direction)                    │
 * │  +Z axis → INTO PALM (toward battery, perpendicular to PCB)     │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * MAGNETOMETER AXES (LIS3MDL) - **TRANSPOSED X/Y**:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  +X axis → toward FINGERS (same as Accel +Y)                    │
 * │  +Y axis → toward WRIST (same as Accel +X)                      │
 * │  +Z axis → INTO PALM (same as Accel +Z)                         │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * THREE.JS HAND MODEL COORDINATE FRAME:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  +X axis → toward PINKY (thumb on -X, right hand)               │
 * │  +Y axis → FINGER EXTENSION direction (fingers grow in +Y)      │
 * │  +Z axis → PALM NORMAL (toward viewer when palm faces camera)   │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * =====================================================================
 * CRITICAL: MAGNETOMETER AXIS TRANSPOSITION
 * =====================================================================
 *
 * The magnetometer (LIS3MDL) has X and Y axes SWAPPED relative to the
 * accelerometer/gyroscope (LSM6DS3). This is a HARDWARE characteristic.
 *
 * To align magnetometer data with accel/gyro coordinate frame:
 *   aligned_mx = raw_my  (mag Y becomes accel-frame X)
 *   aligned_my = raw_mx  (mag X becomes accel-frame Y)
 *   aligned_mz = raw_mz  (Z remains the same)
 *
 * TODO: [SENSOR-001] Magnetometer axis alignment not implemented
 *   - Current code uses raw magnetometer axes without swapping
 *   - Affects: calibration.js (hard/soft iron, earth field)
 *   - Affects: Any future magnetometer-based orientation fusion
 *   - See: docs/procedures/orientation-validation-protocol.md
 *   - Priority: HIGH if using mag for orientation, LOW if mag-only for finger tracking
 *
 * TODO: [SENSOR-002] Create magAlignToAccelFrame() utility function
 *   - Location: shared/sensor-units.js
 *   - Should swap X/Y and document the transformation
 *
 * =====================================================================
 * ORIENTATION MAPPING (Sensor → Hand Visualization)
 * =====================================================================
 *
 * Madgwick AHRS produces Euler angles in SENSOR coordinate frame:
 *   - Roll: rotation around sensor X (wrist-finger axis)
 *   - Pitch: rotation around sensor Y (left-right axis)
 *   - Yaw: rotation around sensor Z (palm normal)
 *
 * CURRENT MAPPING (hand-3d-renderer.js updateFromSensorFusion):
 *   hand_pitch = sensor_pitch + 90°   (palm up when sensor level)
 *   hand_yaw   = sensor_yaw + 180°    (fingers away from viewer)
 *   hand_roll  = -sensor_roll + 180°  (correct chirality)
 *
 * CURRENT MAPPING (threejs-hand-skeleton.js):
 *   Offsets: { roll: 180°, pitch: 90°, yaw: 180° }
 *   Applied as: rotation.set(pitch, yaw, roll, "YXZ")
 *
 * TODO: [ORIENT-001] Validate both renderers produce consistent orientation
 *   - hand-3d-renderer.js uses canvas 2D with custom matrix math
 *   - threejs-hand-skeleton.js uses Three.js Euler with "YXZ" order
 *   - Different Euler conventions may cause subtle differences
 *
 * TODO: [ORIENT-002] Document expected behavior for each test pose
 *   - See: docs/procedures/orientation-validation-protocol.md
 *   - Need to verify: forward tilt, backward tilt, left/right roll, yaw
 *
 * =====================================================================
 * REFERENCE LINKS
 * =====================================================================
 *
 * - Sensor specs: shared/sensor-units.js
 * - Orientation filter: filters.js (MadgwickAHRS)
 * - Telemetry pipeline: shared/telemetry-processor.js
 * - Calibration: calibration.js
 * - Hand renderers: hand-3d-renderer.js, shared/threejs-hand-skeleton.js
 * - Validation: docs/procedures/orientation-validation-protocol.md
 *
 * =====================================================================
 */

// ===== Import shared modules =====
import {
    LowPassFilter
} from './shared/sensor-config.js';
import { TelemetryProcessor } from './shared/telemetry-processor.js';
import {
    SessionPlayback,
    formatTime,
    formatSessionDisplay
} from './modules/session-playback.js';
import {
    createGesture,
    createGestureUI,
    isGestureInferenceAvailable
} from './modules/gesture-inference-module.js';
import {
    getBrowserLocation,
    getDefaultLocation,
    exportLocationMetadata,
    formatLocation
} from './shared/geomagnetic-field.js';
import { ThreeJSHandSkeleton, MagSample, AxisProgress } from './shared/threejs-hand-skeleton.js';
import { MagneticTrajectory } from './modules/magnetic-trajectory.js';
import {
    REFERENCE_POSES,
    createObservation,
    ObservationStore,
    generateDiagnosticReport
} from './shared/orientation-calibration.js';
import {
    uploadViaProxy,
    getUploadSecret,
    setUploadSecret,
    hasUploadSecret
} from './shared/github-upload.js';
import { initLogger, log, getLogBuffer, exportLog } from './modules/logger.js';

// ===== GambitClient for Frame-Based Protocol =====
const gambitClient = new GambitClient({ debug: true });
let isStreaming = false;

function updateConnectionStatus(connected: boolean): void {
    console.log(`SIMCAP Device connected: ${connected}`)
    const deviceStatus = document.getElementById('deviceStatus');
    if (connected) {
        window.connect.innerHTML = "Disconnect"
        window.getdata.disabled = false
        if (deviceStatus) {
            deviceStatus.classList.add('ready');
            deviceStatus.querySelector('span:last-child')!.textContent = 'Connected';
        }
    } else {
        window.connect.innerHTML = "Connect"
        window.getdata.disabled = true
        isStreaming = false;
        window.getdata.innerHTML = "Get data"
        // Clear device info on disconnect
        firmwareVersion = null;
        samplingMode = null;
        deviceContext = null;
        if (deviceStatus) {
            deviceStatus.classList.remove('ready');
            deviceStatus.querySelector('span:last-child')!.textContent = 'Disconnected';
        }
    }
    updateDeviceInfo();
}

/**
 * Update device info display (firmware version, sampling mode, context)
 */
function updateDeviceInfo(): void {
    const deviceInfoEl = document.getElementById('deviceInfo');
    if (!deviceInfoEl) return;

    if (!firmwareVersion && !samplingMode && !deviceContext) {
        deviceInfoEl.style.display = 'none';
        return;
    }

    const parts: string[] = [];
    if (firmwareVersion) parts.push(`v${firmwareVersion}`);
    if (samplingMode) parts.push(samplingMode);
    if (deviceContext && deviceContext !== 'unknown') parts.push(deviceContext);

    deviceInfoEl.textContent = parts.join(' | ');
    deviceInfoEl.style.display = parts.length > 0 ? 'inline' : 'none';
}

let ghToken: string | undefined;
let proxySecret: string | undefined;  // GitHub upload API proxy secret
let uploadMethod: 'proxy' | 'github' = 'proxy';
let firmwareVersion: string | null = null;  // Store firmware version for session metadata
let samplingMode: string | null = null;  // Store sampling mode (v0.4.0+)
let deviceContext: string | null = null;  // Store device context (v0.4.0+)
let geomagneticLocation: GeomagneticLocation | null = null;  // Store geomagnetic location for session metadata

let uploadTimeout: ReturnType<typeof setTimeout> | undefined;
let sessionData: TelemetrySample[] = []

// ===== Data Stage Selection =====
// Controls which processing stage is displayed in the sensor data UI
let currentDataStage: DataStage = 'converted';

const dataStageInfo: Record<DataStage, DataStageInfo> = {
    raw: { label: 'Raw (LSB)', info: 'Raw sensor values in LSB (least significant bits) - direct from hardware', magUnit: 'LSB', accUnit: 'LSB', gyrUnit: 'LSB' },
    converted: { label: 'Converted', info: 'Converted to physical units: µT (magnetometer), g (accelerometer), °/s (gyroscope)', magUnit: 'µT', accUnit: 'g', gyrUnit: '°/s' },
    calibrated: { label: 'Calibrated', info: 'Iron-corrected magnetometer (hard/soft iron compensation applied)', magUnit: 'µT', accUnit: 'g', gyrUnit: '°/s' },
    fused: { label: 'Fused', info: 'Earth field removed - shows residual magnetic field (finger magnet signal)', magUnit: 'µT', accUnit: 'g', gyrUnit: '°/s' },
    filtered: { label: 'Filtered', info: 'Kalman filtered for noise reduction (best for visualization)', magUnit: 'µT', accUnit: 'g', gyrUnit: '°/s' }
};

function updateDataStageUI() {
    const info = dataStageInfo[currentDataStage];
    const magLabel = document.getElementById('magLabel');
    const accLabel = document.getElementById('accLabel');
    const gyrLabel = document.getElementById('gyrLabel');
    const infoEl = document.getElementById('dataStageInfo');
    const gyroStatus = document.getElementById('gyroStatus');
    
    if (magLabel) magLabel.textContent = `Mag (${info.magUnit}):`;
    if (accLabel) accLabel.textContent = `Acc (${info.accUnit}):`;
    if (gyrLabel) {
        gyrLabel.innerHTML = `Gyro (${info.gyrUnit}): <span id="gyroStatus" style="font-size: 0.7em; color: var(--warning);">⏳</span>`;
    }
    if (infoEl) infoEl.textContent = info.info;
}

// Initialize data stage selector
document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('dataStageSelect');
    if (select) {
        select.addEventListener('change', (e) => {
            currentDataStage = (e.target as HTMLSelectElement).value as DataStage;
            updateDataStageUI();
            console.log('[GAMBIT] Data stage:', currentDataStage);
        });
    }
});

// ===== Calibration Confidence UI =====
let lastConfidenceUpdate = 0;
const CONFIDENCE_UPDATE_INTERVAL = 500;  // Update UI every 500ms
let lastCalibrationSave = 0;
const CALIBRATION_SAVE_INTERVAL = 10000;  // Save calibration every 10 seconds when ready
let wasAutoHardIronReady = false;  // Track transition to ready state

// ===== Initialize Logger =====
// Initialize with UI element for log display
initLogger(document.getElementById('log'));

// ===== Telemetry Processor (shared module) =====
// Handles: unit conversion, IMU fusion, gyro bias calibration, mag calibration, filtering
// Calibration (iron + Earth) is loaded from localStorage by TelemetryProcessor
const telemetryProcessor = new TelemetryProcessor({
    useMagnetometer: true, // Re-enabled for drift investigation
    onLog: log,  // Route diagnostic logs through shared logger
    onGyroBiasCalibrated: () => {
        log('[AHRS] Gyroscope bias calibration complete');
        // Update visual indicator
        const gyroStatus = document.getElementById('gyroStatus');
        if (gyroStatus) {
            gyroStatus.textContent = '✓';
            gyroStatus.style.color = 'var(--success)';
            gyroStatus.title = 'Gyroscope bias calibrated';
        }
    }
});

// ===== Geomagnetic Location Detection =====
/**
 * Initialize geomagnetic location
 * Try browser geolocation first, fall back to Edinburgh default
 */
async function initGeomagneticLocation() {
    console.log('[GAMBIT] Initializing geomagnetic location...');

    try {
        const location = await getBrowserLocation();
        geomagneticLocation = location.selected;

        const locationStr = formatLocation(location.selected);
        const fieldStr = `${location.selected.intensity.toFixed(1)} µT`;
        const declStr = `${location.selected.declination.toFixed(1)}°`;

        console.log(`[GAMBIT] ✓ Location detected: ${locationStr}`);
        console.log(`[GAMBIT] Magnetic field: ${fieldStr}, Declination: ${declStr}`);
    } catch (error) {
        // Fall back to default location (Edinburgh)
        geomagneticLocation = getDefaultLocation();

        const locationStr = formatLocation(geomagneticLocation);
        console.warn('[GAMBIT] Geolocation failed, using default:', (error as Error).message);
        console.log(`[GAMBIT] Location: ${locationStr} (default)`);
        console.log(`[GAMBIT] Magnetic field: ${geomagneticLocation!.intensity.toFixed(1)} µT, Declination: ${geomagneticLocation!.declination.toFixed(1)}°`);
    }
    
}

// Initialize geomagnetic location on page load
initGeomagneticLocation();

// ===== Three.js Hand Skeleton =====
let threeHandSkeleton: ThreeJSHandSkeleton | null = null;
let threeHandEnabled = true;

// ===== Magnetic Trajectory Visualization =====
let magTrajectory: MagneticTrajectory | null = null;
let magTrajectoryEnabled = true;

function initThreeHandSkeleton() {
    const container = document.getElementById('threeHandContainer');
    if (!container || typeof THREE === 'undefined') {
        console.warn('[ThreeHand] Three.js or container not available');
        return;
    }

    try {
        threeHandSkeleton = new ThreeJSHandSkeleton(container, {
            width: 500,
            height: 300,
            backgroundColor: 0xffffff,
            lerpFactor: 0.15,
            handedness: 'right'  // Default to right hand
        });

        // Set initial orientation offsets to match sensor-to-hand mapping
        // Correct offsets determined through testing: roll: 180, pitch: 180, yaw: -180
        threeHandSkeleton.setOrientationOffsets({
            roll: 180,
            pitch: 180,
            yaw: -180
        });

        console.log('[ThreeHand] Skeleton initialized');

        // Initialize magnetometer calibration visualization
        const location = geomagneticLocation || getDefaultLocation();
        const expectedMag = location
            ? Math.sqrt(location.horizontal ** 2 + location.vertical ** 2)
            : 50;

        threeHandSkeleton.initMagCalibrationVis({
            enabled: true,
            maxPoints: 300,
            pointSize: 2,           // Small pixel dots (sizeAttenuation: false)
            expectedMagnitude: expectedMag,
            showExpectedSphere: true,
            showHardIronMarker: true,
            showAxisCoverage: true,
            showPointCloud: true,
            scale: 0.02
        });
        console.log('[ThreeHand] Mag calibration visualization initialized');

        // Toggle handler
        const toggle = document.getElementById('threeHandToggle') as HTMLInputElement | null;
        if (toggle) {
            toggle.addEventListener('change', () => {
                threeHandEnabled = toggle.checked;
                console.log('[ThreeHand] IMU tracking:', threeHandEnabled ? 'ON' : 'OFF');
            });
        }

        // Reset orientation button
        const orientationResetBtn = document.getElementById('threeResetBtn');
        if (orientationResetBtn) {
            orientationResetBtn.addEventListener('click', () => {
                if (threeHandSkeleton) {
                    threeHandSkeleton.resetOrientation();
                    console.log('[ThreeHand] Orientation reset');
                }
            });
        }

        // Test curl button
        const curlBtn = document.getElementById('threeTestCurlBtn');
        let testCurlState = 0;
        if (curlBtn) {
            curlBtn.addEventListener('click', () => {
                if (threeHandSkeleton) {
                    testCurlState = (testCurlState + 1) % 4;
                    const curlAmounts = [0, 0.33, 0.66, 1.0];
                    const curl = curlAmounts[testCurlState];
                    threeHandSkeleton.setFingerCurls({
                        thumb: curl,
                        index: curl,
                        middle: curl,
                        ring: curl,
                        pinky: curl
                    });
                    console.log('[ThreeHand] Test curl:', curl);
                }
            });
        }

        // ===== DEBUG OFFSET CONTROLS =====
        // Real-time offset adjustment for debugging orientation mapping
        const rollSlider = document.getElementById('rollOffsetSlider') as HTMLInputElement | null;
        const pitchSlider = document.getElementById('pitchOffsetSlider') as HTMLInputElement | null;
        const yawSlider = document.getElementById('yawOffsetSlider') as HTMLInputElement | null;
        const rollValue = document.getElementById('rollOffsetValue');
        const pitchValue = document.getElementById('pitchOffsetValue');
        const yawValue = document.getElementById('yawOffsetValue');

        function updateOffsets() {
            if (threeHandSkeleton && rollSlider && pitchSlider && yawSlider) {
                const offsets = {
                    roll: parseFloat(rollSlider.value),
                    pitch: parseFloat(pitchSlider.value),
                    yaw: parseFloat(yawSlider.value)
                };
                threeHandSkeleton.setOrientationOffsets(offsets);

                // Update display values
                if (rollValue) rollValue.textContent = `${offsets.roll}°`;
                if (pitchValue) pitchValue.textContent = `${offsets.pitch}°`;
                if (yawValue) yawValue.textContent = `${offsets.yaw}°`;

                console.log('[ThreeHand] Offsets updated:', offsets);
            }
        }

        // Slider event listeners
        if (rollSlider) rollSlider.addEventListener('input', updateOffsets);
        if (pitchSlider) pitchSlider.addEventListener('input', updateOffsets);
        if (yawSlider) yawSlider.addEventListener('input', updateOffsets);

        // Reset to default button (180, 180, -180)
        const offsetResetBtn = document.getElementById('threeOffsetReset');
        if (offsetResetBtn) {
            offsetResetBtn.addEventListener('click', () => {
                if (rollSlider) rollSlider.value = '180';
                if (pitchSlider) pitchSlider.value = '180';
                if (yawSlider) yawSlider.value = '-180';
                updateOffsets();
                console.log('[ThreeHand] Offsets reset to default');
            });
        }

        // Zero all button
        const offsetZeroBtn = document.getElementById('threeOffsetZero');
        if (offsetZeroBtn) {
            offsetZeroBtn.addEventListener('click', () => {
                if (rollSlider) rollSlider.value = '0';
                if (pitchSlider) pitchSlider.value = '0';
                if (yawSlider) yawSlider.value = '0';
                updateOffsets();
                console.log('[ThreeHand] Offsets zeroed');
            });
        }

        // Hand chirality buttons
        const handLeftBtn = document.getElementById('handLeft');
        const handRightBtn = document.getElementById('handRight');

        function setHandChirality(hand: 'left' | 'right') {
            if (threeHandSkeleton) {
                threeHandSkeleton.setHandedness(hand);

                // Update button styles
                if (hand === 'left') {
                    if (handLeftBtn) handLeftBtn.style.background = 'var(--success)';
                    if (handRightBtn) handRightBtn.style.background = '';
                } else {
                    if (handLeftBtn) handLeftBtn.style.background = '';
                    if (handRightBtn) handRightBtn.style.background = 'var(--success)';
                }

                console.log('[ThreeHand] Hand set to:', hand);
            }
        }

        if (handLeftBtn) {
            handLeftBtn.addEventListener('click', () => setHandChirality('left'));
        }
        if (handRightBtn) {
            handRightBtn.addEventListener('click', () => setHandChirality('right'));
        }

        // Test pose buttons - simulate specific orientations
        document.getElementById('testPalmUp')?.addEventListener('click', () => {
            if (threeHandSkeleton) {
                threeHandSkeleton.updateOrientation({ roll: 0, pitch: 0, yaw: 0 });
                console.log('[ThreeHand] Test: Palm Up (roll=0, pitch=0, yaw=0)');
            }
        });

        document.getElementById('testPalmDown')?.addEventListener('click', () => {
            if (threeHandSkeleton) {
                threeHandSkeleton.updateOrientation({ roll: 180, pitch: 0, yaw: 0 });
                console.log('[ThreeHand] Test: Palm Down (roll=180, pitch=0, yaw=0)');
            }
        });

        document.getElementById('testFingersUp')?.addEventListener('click', () => {
            if (threeHandSkeleton) {
                threeHandSkeleton.updateOrientation({ roll: 0, pitch: -90, yaw: 0 });
                console.log('[ThreeHand] Test: Fingers Up (roll=0, pitch=-90, yaw=0)');
            }
        });

        document.getElementById('testFingersForward')?.addEventListener('click', () => {
            if (threeHandSkeleton) {
                threeHandSkeleton.updateOrientation({ roll: 0, pitch: 0, yaw: 90 });
                console.log('[ThreeHand] Test: Fingers Forward (roll=0, pitch=0, yaw=90)');
            }
        });

        // ===== AXIS SIGN TOGGLES =====
        // Allow user to toggle axis negation to fix inversions
        const negateRollToggle = document.getElementById('negateRollToggle') as HTMLInputElement | null;
        const negatePitchToggle = document.getElementById('negatePitchToggle') as HTMLInputElement | null;
        const negateYawToggle = document.getElementById('negateYawToggle') as HTMLInputElement | null;

        function updateAxisSigns() {
            if (threeHandSkeleton) {
                threeHandSkeleton.setAxisSigns({
                    negateRoll: negateRollToggle?.checked || false,
                    negatePitch: negatePitchToggle?.checked || false,
                    negateYaw: negateYawToggle?.checked || false
                });
                console.log('[ThreeHand] Axis signs:', {
                    negateRoll: negateRollToggle?.checked,
                    negatePitch: negatePitchToggle?.checked,
                    negateYaw: negateYawToggle?.checked
                });
            }
        }

        negateRollToggle?.addEventListener('change', updateAxisSigns);
        negatePitchToggle?.addEventListener('change', updateAxisSigns);
        negateYawToggle?.addEventListener('change', updateAxisSigns);

        // Initialize with default signs (pitch negated based on model)
        updateAxisSigns();

    } catch (err) {
        console.error('[ThreeHand] Failed to initialize:', err);
    }
}

// ===== Magnetic Trajectory Visualization =====
function initMagneticTrajectory() {
    const canvas = document.getElementById('magTrajectoryCanvas');
    if (!canvas) {
        console.warn('[MagTrajectory] Canvas not found');
        return;
    }

    try {
        magTrajectory = new MagneticTrajectory(canvas as HTMLCanvasElement, {
            maxPoints: 150,
            scale: 0.35,
            autoNormalize: true,
            trajectoryColor: '#4ecdc4',
            showMarkers: true,
            showCube: true,
            showScaleKey: true,
            fixedBounds: 1000,
            backgroundColor: null,
            scaleKeyPosition: 'top-right',
            minAlpha: 0.1, 
        });

        console.log('[MagTrajectory] Initialized');

        // Toggle handler
        const toggle = document.getElementById('magTrajectoryToggle');
        if (toggle) {
            toggle.addEventListener('change', (e) => {
                magTrajectoryEnabled = (e.target as HTMLInputElement).checked;
                console.log('[MagTrajectory]', magTrajectoryEnabled ? 'ON' : 'OFF');
            });
        }

        // Clear button
        const clearBtn = document.getElementById('clearMagTrajectoryBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                if (magTrajectory) {
                    magTrajectory.clear();
                    updateMagTrajectoryStats();
                    console.log('[MagTrajectory] Cleared');
                }
            });
        }
    } catch (err) {
        console.error('[MagTrajectory] Failed to initialize:', err);
    }
}

// Update magnetic trajectory stats display
function updateMagTrajectoryStats() {
    const statsDiv = document.getElementById('magTrajectoryStats');
    if (!statsDiv || !magTrajectory) return;

    const stats = magTrajectory.getStats();
    if (stats.count === 0) {
        statsDiv.textContent = 'Points: 0 | Magnitude: --µT (min: --, max: --, avg: --)';
    } else {
        statsDiv.textContent = `Points: ${stats.count} | Magnitude: ${stats.magnitude.avg.toFixed(2)}µT (min: ${stats.magnitude.min.toFixed(2)}, max: ${stats.magnitude.max.toFixed(2)}, avg: ${stats.magnitude.avg.toFixed(2)})`;
    }
}

// ===== CALIBRATION SYSTEM =====
const observationStore = new ObservationStore();
let currentCalibrationPose: ReferencePose | null = null;
let currentUserAnswers: Record<string, string> = {};
let latestSensorData: SensorSnapshot | null = null;
let latestAhrsOutput: EulerAngles | null = null;
let baselineAhrsOutput: EulerAngles | null = null;  // Stored from FLAT_PALM_UP pose

// Initialize calibration UI
function initCalibrationUI() {
    const poseSelect = document.getElementById('referencePoseSelect');
    const captureBtn = document.getElementById('captureObservation');
    const resetBtn = document.getElementById('resetCalibration');
    const exportBtn = document.getElementById('exportObservations');
    const applyBtn = document.getElementById('applyRecommendation');

    // Pose selection handler
    poseSelect?.addEventListener('change', (e) => {
        const poseId = (e.target as HTMLSelectElement).value as keyof typeof REFERENCE_POSES;
        if (poseId && REFERENCE_POSES[poseId]) {
            selectCalibrationPose(poseId);
        } else {
            clearCalibrationPose();
        }
    });

    // Capture button
    captureBtn?.addEventListener('click', captureCurrentObservation);

    // Reset button
    resetBtn?.addEventListener('click', () => {
        observationStore.clear();
        currentUserAnswers = {};
        baselineAhrsOutput = null;  // Clear baseline
        updateObservationLog();
        updateAnalysis();
        (poseSelect as HTMLSelectElement)!.value = '';
        clearCalibrationPose();
        console.log('[Calibration] Reset - baseline cleared');
    });

    // Export button
    exportBtn?.addEventListener('click', exportObservationsToFile);

    // Apply recommendation button
    applyBtn?.addEventListener('click', applyCalibrationRecommendation);
}

function selectCalibrationPose(poseId: keyof typeof REFERENCE_POSES) {
    currentCalibrationPose = REFERENCE_POSES[poseId];
    currentUserAnswers = {};

    // Show instructions
    const instructionsDiv = document.getElementById('poseInstructions');
    const instructionsList = document.getElementById('poseInstructionsList');
    if (instructionsDiv && instructionsList) {
        instructionsList.innerHTML = currentCalibrationPose.instructions
            .map((i: string) => `<li>${i}</li>`).join('');
        instructionsDiv.style.display = 'block';
    }

    // Show validation questions with multi-choice options
    const questionsDiv = document.getElementById('validationQuestions');
    const questionsList = document.getElementById('questionsList');
    if (questionsDiv && questionsList) {
        let questionsHtml = '';

        // Render each validation question
        currentCalibrationPose.validationQuestions.forEach((q: ValidationQuestion) => {
            if (q.type === 'position') {
                // Simple yes/no/mostly for static position questions
                questionsHtml += `
                    <div style="margin: 8px 0; padding: 6px; background: rgba(255,255,255,0.03); border-radius: 4px;">
                        <div style="font-weight: bold; margin-bottom: 4px;">${q.question}</div>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                            <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                                <input type="radio" name="q_${q.id}" class="calibration-radio" data-qid="${q.id}" value="yes">
                                <span>Yes</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                                <input type="radio" name="q_${q.id}" class="calibration-radio" data-qid="${q.id}" value="mostly">
                                <span>Mostly</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                                <input type="radio" name="q_${q.id}" class="calibration-radio" data-qid="${q.id}" value="no">
                                <span>No</span>
                            </label>
                        </div>
                    </div>
                `;
            } else if (q.type === 'direction' && q.options) {
                // Multi-choice for direction/movement questions
                questionsHtml += `
                    <div style="margin: 8px 0; padding: 6px; background: rgba(255,255,255,0.03); border-radius: 4px;">
                        <div style="font-weight: bold; margin-bottom: 4px;">${q.question}</div>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            ${q.options.map((opt) => `
                                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                                    <input type="radio" name="q_${q.id}" class="calibration-radio" data-qid="${q.id}" value="${opt.value}">
                                    <span>${opt.label}</span>
                                </label>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
        });

        // Add coupling question if present
        if (currentCalibrationPose.couplingQuestion) {
            const cq: CouplingQuestion = currentCalibrationPose.couplingQuestion;
            questionsHtml += `
                <div style="margin: 8px 0; padding: 6px; background: rgba(255,200,0,0.1); border-radius: 4px; border-left: 3px solid rgba(255,200,0,0.5);">
                    <div style="font-weight: bold; margin-bottom: 4px;">${cq.question}</div>
                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        ${cq.options.map((opt) => `
                            <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                                <input type="radio" name="q_coupling" class="calibration-radio" data-qid="coupling" value="${opt.value}">
                                <span>${opt.label}</span>
                            </label>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        questionsList.innerHTML = questionsHtml;
        questionsDiv.style.display = 'block';

        // Add change handlers for all radio buttons
        questionsList.querySelectorAll('.calibration-radio').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const target = e.target as HTMLInputElement;
                currentUserAnswers[target.dataset.qid!] = target.value;
                updateCaptureButton();
            });
        });
    }

    updateCaptureButton();
}

function clearCalibrationPose() {
    currentCalibrationPose = null;
    currentUserAnswers = {};
    document.getElementById('poseInstructions')!.style.display = 'none';
    document.getElementById('validationQuestions')!.style.display = 'none';
    updateCaptureButton();
}

function updateCaptureButton() {
    const btn = document.getElementById('captureObservation');
    if (btn) {
        // Enable if pose selected and primary question answered
        const hasAnswers = Object.keys(currentUserAnswers).length > 0;
        (btn as HTMLButtonElement).disabled = !currentCalibrationPose || !hasAnswers;
    }
}

function captureCurrentObservation() {
    if (!currentCalibrationPose || !latestAhrsOutput) {
        console.warn('[Calibration] Missing pose or sensor data');
        return;
    }

    // Get current mapping config from UI
    const mappingConfig = {
        negateRoll: (document.getElementById('negateRollToggle') as HTMLInputElement)?.checked || false,
        negatePitch: (document.getElementById('negatePitchToggle') as HTMLInputElement)?.checked || false,
        negateYaw: (document.getElementById('negateYawToggle') as HTMLInputElement)?.checked || false,
        rollOffset: parseInt((document.getElementById('rollOffsetSlider') as HTMLInputElement)?.value) || 180,
        pitchOffset: parseInt((document.getElementById('pitchOffsetSlider') as HTMLInputElement)?.value) || 180,
        yawOffset: parseInt((document.getElementById('yawOffsetSlider') as HTMLInputElement)?.value) || -180,
        eulerOrder: 'YXZ'
    };

    // Get render state from ThreeJS skeleton
    const renderState = threeHandSkeleton ? threeHandSkeleton.getRenderState() : { x: 0, y: 0, z: 0, order: 'YXZ' };

    // Store baseline if this is FLAT_PALM_UP pose
    if (currentCalibrationPose.id === 'FLAT_PALM_UP') {
        baselineAhrsOutput = { ...latestAhrsOutput };
        console.log('[Calibration] Baseline stored:', baselineAhrsOutput);
    }

    // Create observation (v2 with baseline support)
    const observation = createObservation(
        currentCalibrationPose.id,
        latestSensorData || { ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 },
        latestAhrsOutput!,
        renderState,
        currentUserAnswers,
        mappingConfig,
        baselineAhrsOutput as any  // Pass baseline for delta analysis
    );

    observationStore.add(observation);
    console.log('[Calibration] Observation captured:', observation);

    // Generate diagnostic report if coupling detected
    if (observation.analysis?.coupling?.type !== 'none') {
        const report = generateDiagnosticReport(observation);
        console.log('[Calibration] Diagnostic report:', report);
    }

    // Update UI
    updateObservationLog();
    updateAnalysis();

    // Reset for next observation
    document.querySelectorAll('.calibration-radio').forEach(r => (r as HTMLInputElement).checked = false);
    currentUserAnswers = {};
    updateCaptureButton();
}

function updateObservationLog() {
    const countEl = document.getElementById('observationCount');
    const logEl = document.getElementById('observationLog');
    const observations = observationStore.getAll();

    if (countEl) countEl.textContent = String(observations.length);

    if (logEl) {
        if (observations.length === 0) {
            logEl.innerHTML = 'No observations yet. Start with FLAT_PALM_UP to set baseline.';
        } else {
            logEl.innerHTML = observations.map((obs: Observation, i: number) => {
                const coupling = obs.analysis?.coupling;
                const couplingColor = !coupling || coupling.type === 'none' ? 'green' :
                    coupling.type === 'partial' ? 'orange' : 'red';
                const couplingText = !coupling || coupling.type === 'none' ? 'OK' :
                    coupling.type === 'partial' ? 'PARTIAL' :
                    coupling.type === 'wrong_primary' ? 'WRONG AXIS' :
                    coupling.type === 'all_coupled' ? 'ALL COUPLED' : coupling.type;

                // Show delta from baseline if available
                let deltaInfo = '';
                if (obs.ahrs?.deltaFromBaseline) {
                    const d = obs.ahrs.deltaFromBaseline;
                    deltaInfo = `<div style="font-size: 0.55rem; color: var(--fg-muted);">
                        Δ r=${d.roll.toFixed(1)}° p=${d.pitch.toFixed(1)}° y=${d.yaw.toFixed(1)}°
                    </div>`;
                }

                return `
                    <div style="border-bottom: 1px solid var(--border); padding: 4px 0;">
                        <strong>#${i + 1}</strong> ${obs.referencePose.name}
                        <span style="color: ${couplingColor}; float: right; font-size: 0.6rem;">${couplingText}</span><br>
                        <span style="font-size: 0.55rem;">AHRS: r=${obs.ahrs.euler.roll.toFixed(1)}° p=${obs.ahrs.euler.pitch.toFixed(1)}° y=${obs.ahrs.euler.yaw.toFixed(1)}°</span>
                        ${deltaInfo}
                        ${obs.userObservations.derivedState.matchesExpected ?
                            '<span style="color: green; font-size: 0.6rem;">✓ Correct</span>' :
                            '<span style="color: orange; font-size: 0.6rem;">⚠ Issue</span>'}
                    </div>
                `;
            }).join('');
        }
    }
}

function updateAnalysis() {
    const analysisDiv = document.getElementById('calibrationAnalysis');
    const resultsEl = document.getElementById('analysisResults');
    const applyBtn = document.getElementById('applyRecommendation');

    const observations = observationStore.getAll();
    if (observations.length < 1) {
        analysisDiv!.style.display = 'none';
        applyBtn!.style.display = 'none';
        return;
    }

    const analysis = observationStore.analyzeAll();
    analysisDiv!.style.display = 'block';

    // Count coupling issues
    let couplingCount = 0;
    let wrongAxisCount = 0;
    observations.forEach((obs: Observation) => {
        if (obs.analysis?.coupling?.type === 'all_coupled' || obs.analysis?.coupling?.type === 'partial') {
            couplingCount++;
        }
        if (obs.analysis?.coupling?.type === 'wrong_primary') {
            wrongAxisCount++;
        }
    });

    let html = `<div style="margin-top: 4px;">`;
    html += `<div>Observations: ${analysis.totalObservations}</div>`;

    // Show coupling warning if detected
    if (couplingCount > 0 || wrongAxisCount > 0) {
        html += `<div style="margin-top: 6px; padding: 6px; background: rgba(255,100,0,0.15); border-radius: 4px; border-left: 3px solid orange;">`;
        html += `<strong style="color: orange;">⚠ Axis Issues Detected:</strong>`;
        if (couplingCount > 0) {
            html += `<div>• Coupling detected in ${couplingCount} observation(s)</div>`;
        }
        if (wrongAxisCount > 0) {
            html += `<div>• Wrong axis response in ${wrongAxisCount} observation(s)</div>`;
        }
        html += `<div style="font-size: 0.6rem; margin-top: 4px; color: var(--fg-muted);">
            This suggests Euler order mismatch or axis permutation issues.
            Consider testing different Euler orders or sensor axis mapping.
        </div>`;
        html += `</div>`;
    }

    html += `<div style="margin-top: 6px;"><strong>Axis Sign Recommendations:</strong></div>`;
    html += `<div>• Negate Pitch: ${analysis.negatePitch ? 'YES' : 'NO'} (confidence: ${(analysis.confidence.pitch * 100).toFixed(0)}%)</div>`;
    html += `<div>• Negate Roll: ${analysis.negateRoll ? 'YES' : 'NO'} (confidence: ${(analysis.confidence.roll * 100).toFixed(0)}%)</div>`;
    html += `<div>• Negate Yaw: ${analysis.negateYaw ? 'YES' : 'NO'} (confidence: ${(analysis.confidence.yaw * 100).toFixed(0)}%)</div>`;
    html += `</div>`;

    resultsEl!.innerHTML = html;

    // Show apply button if we have recommendations with decent confidence
    const hasConfidence = analysis.confidence.pitch > 0.3 || analysis.confidence.roll > 0.3 || analysis.confidence.yaw > 0.3;
    applyBtn!.style.display = hasConfidence && couplingCount === 0 ? 'block' : 'none';
    (window as any)._calibrationRecommendation = analysis;
}

function applyCalibrationRecommendation() {
    const rec = (window as any)._calibrationRecommendation;
    if (!rec) return;

    (document.getElementById('negateRollToggle') as HTMLInputElement)!.checked = rec.negateRoll;
    (document.getElementById('negatePitchToggle') as HTMLInputElement)!.checked = rec.negatePitch;
    (document.getElementById('negateYawToggle') as HTMLInputElement)!.checked = rec.negateYaw;

    // Trigger update
    if (threeHandSkeleton) {
        threeHandSkeleton.setAxisSigns({
            negateRoll: rec.negateRoll,
            negatePitch: rec.negatePitch,
            negateYaw: rec.negateYaw
        });
    }

    console.log('[Calibration] Applied recommendation:', rec);
}

function exportObservationsToFile() {
    const data = observationStore.export();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `calibration-${data.sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log('[Calibration] Exported:', data);
}

// Update sensor state display
function updateCalibrationSensorDisplay(euler: EulerAngles | null, sensorData: SensorSnapshot | null) {
    latestAhrsOutput = euler;
    latestSensorData = sensorData;

    const sensorStateDesc = document.getElementById('sensorStateDesc');
    if (!sensorStateDesc || !euler) return;

    sensorStateDesc.innerHTML = `
        <div><strong>AHRS Euler:</strong> roll=${euler.roll.toFixed(1)}° pitch=${euler.pitch.toFixed(1)}° yaw=${euler.yaw.toFixed(1)}°</div>
        ${sensorData ? `<div><strong>Accel:</strong> x=${(sensorData.ax||0).toFixed(2)} y=${(sensorData.ay||0).toFixed(2)} z=${(sensorData.az||0).toFixed(2)}</div>` : ''}
    `;
}

// Initialize calibration UI on page load
setTimeout(initCalibrationUI, 100);

// Legacy validation panel update (kept for compatibility)
function updateValidationPanel(euler: EulerAngles | null) {
    updateCalibrationSensorDisplay(euler, latestSensorData);
}

// LowPassFilter class is already defined in hand-3d-renderer.js
// Reuse it for cube smoothing

// Filters for each cube axis (smooth display)
// Lower alpha = smoother but more lag, higher = more responsive
// NOTE: With stable sensor data (Puck.accelOn fix), we can use higher alpha
// Previous jitter-compensating values: acc=0.2, gyro=0.1, mag=0.2
const cubeFilters = {
    acc: { x: new LowPassFilter(0.4), y: new LowPassFilter(0.4), z: new LowPassFilter(0.4) },
    gyro: { x: new LowPassFilter(0.3), y: new LowPassFilter(0.3), z: new LowPassFilter(0.3) },  // Was 0.1 for jittery data
    mag: { x: new LowPassFilter(0.3), y: new LowPassFilter(0.3), z: new LowPassFilter(0.3) }
};

// Track last timestamp for dt calculation
let lastTimestamp = null;

// ===== GambitClient Event Handlers =====
// Handle telemetry data from frame-based protocol
gambitClient.on('data', function(t: ExtendedTelemetry) {
    wrappedUpdateData(t);
    
    // Auto-upload to GitHub after 1s of no data
    if (uploadTimeout) clearTimeout(uploadTimeout);
    uploadTimeout = setTimeout(function() {
        if (sessionData.length > 0) {
            console.log("Uploading session data via", uploadMethod);
            uploadSessionData(sessionData);  // Pass array directly, wrapper functions add v2.1 schema
            sessionData = [];
        }
    }, 1000);
});

// Handle connection events
gambitClient.on('connect', function() {
    console.log('[GambitClient] Connected');
    updateConnectionStatus(true);
});

gambitClient.on('disconnect', function() {
    console.log('[GambitClient] Disconnected');
    updateConnectionStatus(false);
});

gambitClient.on('firmware', function(fw) {
    console.log('[GambitClient] Firmware:', fw);
    firmwareVersion = fw.version || 'unknown';
    updateDeviceInfo();
});

// Handle mode changes (v0.4.0+)
gambitClient.on('mode', function(event) {
    console.log('[GambitClient] Mode:', event.mode);
    samplingMode = event.mode;
    updateDeviceInfo();
});

// Handle context changes (v0.4.0+)
gambitClient.on('context', function(event) {
    console.log('[GambitClient] Context:', event.context);
    deviceContext = event.context;
    updateDeviceInfo();
});

const minMaxs: Record<string, MinMaxEntry> = {}

function updateMinMaxsReturnNorm(key: string, val: number): number {
    if (typeof minMaxs[key] === 'undefined') {
        var k = new KalmanFilter({R: 0.01, Q: 3})
        minMaxs[key] = {
            values: [],
            min: val,
            max: val, 
            kalman: k,
            kalmanValues: []
        }
    } else if (minMaxs[key].min > val) {
        minMaxs[key].min = val
    } else if (minMaxs[key].max < val) {
        minMaxs[key].max = val
    }

    var kalmanValue = minMaxs[key].kalman.filter(val)
    minMaxs[key].values.push(val)
    minMaxs[key].kalmanValues.push(kalmanValue)

    if ((minMaxs[key].max - minMaxs[key].min) === 0) {
        return 0.5
    }

    return (val - minMaxs[key].min) / (minMaxs[key].max - minMaxs[key].min)
}

function normalise(t: ExtendedTelemetry) {
    return {
        l: t.l,
        t: t.t,
        s: t.s,
        c: t.c,
        n: t.n,
        b: t.b,
        mx: updateMinMaxsReturnNorm('mx', t.mx),
        my: updateMinMaxsReturnNorm('my', t.my),
        mz: updateMinMaxsReturnNorm('mz', t.mz),
        ax: updateMinMaxsReturnNorm('ax', t.ax ),
        ay: updateMinMaxsReturnNorm('ay', t.ay ),
        az: updateMinMaxsReturnNorm('az', t.az ),
        gx: updateMinMaxsReturnNorm('gx', t.gx ),
        gy: updateMinMaxsReturnNorm('gy', t.gy ),
        gz: updateMinMaxsReturnNorm('gz', t.gz ),
    }
}

/**
 * Update calibration confidence display
 * Shows incremental calibration status and confidence metrics
 */
/**
 * Update magnet detection UI display
 */
function updateMagnetDetectionUI(_decoratedData?: TelemetrySample) {
    const statusEl = document.getElementById('magnetStatusValue');
    const confEl = document.getElementById('magnetConfidenceValue');
    const barEl = document.getElementById('magnetConfidenceBar');
    const residualEl = document.getElementById('magnetAvgResidual');
    
    // Get magnet state from telemetry processor
    const magnetState = telemetryProcessor.getMagnetState();
    if (!magnetState) return;
    
    // Update status text with icon
    if (statusEl) {
        const icons = { none: '○', possible: '◐', likely: '◑', confirmed: '🧲' };
        const labels = { none: 'No Magnets', possible: 'Possible', likely: 'Likely', confirmed: 'Confirmed' };
        const colors = { none: '#888', possible: '#f0ad4e', likely: '#5bc0de', confirmed: '#5cb85c' };
        
        statusEl.textContent = `${icons[magnetState.status] || '?'} ${labels[magnetState.status] || '--'}`;
        statusEl.style.color = colors[magnetState.status] || '#888';
    }
    
    // Update confidence
    const confPct = Math.round(magnetState.confidence * 100);
    if (confEl) confEl.textContent = `${confPct}%`;
    
    // Update bar
    if (barEl) {
        barEl.style.width = `${confPct}%`;
        const colors = { none: '#888', possible: '#f0ad4e', likely: '#5bc0de', confirmed: '#5cb85c' };
        barEl.style.background = colors[magnetState.status] || '#888';
    }
    
    // Update residual
    if (residualEl) {
        residualEl.textContent = magnetState.avgResidual > 0 ? 
            `${magnetState.avgResidual.toFixed(1)} µT` : '-- µT';
    }
}

function updateCalibrationConfidenceUI() {
    // Get calibration state from TelemetryProcessor's UnifiedMagCalibration
    const magCal = telemetryProcessor.getMagCalibration();
    const state = magCal.getState();

    const overall = Math.round(state.confidence * 100);
    const meanResidual = state.meanResidual;
    const earthMag = state.earthMagnitude;
    const totalSamples = state.totalSamples;

    // Update text displays
    const overallEl = document.getElementById('overallConfidence');
    const hardIronEl = document.getElementById('hardIronConf');
    const softIronEl = document.getElementById('softIronConf');
    const earthFieldEl = document.getElementById('earthFieldConf');
    const baselineEl = document.getElementById('baselineConf');
    const samplesEl = document.getElementById('calibSamples');
    const fieldMagEl = document.getElementById('earthFieldMag');
    const statusEl = document.getElementById('calibStatus');
    const barEl = document.getElementById('confidenceBar');
    const meanResidualEl = document.getElementById('meanResidual');
    const residualQualityEl = document.getElementById('residualQuality');
    const calibQualityEl = document.getElementById('calibQuality');
    const calibDiversityEl = document.getElementById('calibDiversity');

    // Auto hard iron progress UI elements
    const autoProgressEl = document.getElementById('autoHardIronProgress');
    const autoBarEl = document.getElementById('autoHardIronBar');
    const autoStatusEl = document.getElementById('autoHardIronStatus');
    const softIronScaleEl = document.getElementById('softIronScale');
    const autoRangesEl = document.getElementById('autoHardIronRanges');

    // Update auto hard iron progress
    const autoProgress = Math.round(state.autoHardIronProgress * 100);
    if (autoProgressEl) {
        autoProgressEl.textContent = `${autoProgress}%`;
        if (state.autoHardIronReady) {
            autoProgressEl.style.color = 'var(--success)';
        } else if (autoProgress >= 50) {
            autoProgressEl.style.color = 'var(--warning)';
        } else {
            autoProgressEl.style.color = 'var(--fg-muted)';
        }
    }

    if (autoBarEl) {
        autoBarEl.style.width = `${autoProgress}%`;
        if (state.autoHardIronReady) {
            autoBarEl.style.background = 'var(--success)';
        } else if (autoProgress >= 50) {
            autoBarEl.style.background = 'var(--warning)';
        } else {
            autoBarEl.style.background = 'var(--accent)';
        }
    }

    if (autoStatusEl) {
        if (state.autoHardIronReady) {
            autoStatusEl.textContent = '✓ Auto calibration complete';
            autoStatusEl.style.color = 'var(--success)';
        } else if (autoProgress >= 50) {
            autoStatusEl.textContent = 'Keep rotating...';
            autoStatusEl.style.color = 'var(--warning)';
        } else {
            autoStatusEl.textContent = 'Rotate device to calibrate...';
            autoStatusEl.style.color = 'var(--fg-muted)';
        }
    }

    // Soft iron scale factors
    if (softIronScaleEl) {
        if (state.autoHardIronReady) {
            const scale = state.autoSoftIronScale;
            softIronScaleEl.textContent = `${scale.x.toFixed(2)}, ${scale.y.toFixed(2)}, ${scale.z.toFixed(2)}`;
        } else {
            softIronScaleEl.textContent = '--';
        }
    }

    // Auto hard iron ranges
    if (autoRangesEl) {
        const ranges = state.autoHardIronRanges;
        if (ranges.x > 0 || ranges.y > 0 || ranges.z > 0) {
            autoRangesEl.textContent = `${ranges.x.toFixed(0)}, ${ranges.y.toFixed(0)}, ${ranges.z.toFixed(0)}`;
        } else {
            autoRangesEl.textContent = '--';
        }
    }

    if (overallEl) overallEl.textContent = `${overall}%`;
    if (hardIronEl) hardIronEl.textContent = state.hardIronCalibrated ? '✓ Wizard' : (state.autoHardIronReady ? '✓ Auto' : '--');
    if (softIronEl) softIronEl.textContent = state.softIronCalibrated ? '✓' : '--';
    if (earthFieldEl) earthFieldEl.textContent = state.ready ? '✓ Auto' : 'Building...';

    // Extended Baseline status
    if (baselineEl) {
        if (state.extendedBaselineActive) {
            const mag = state.extendedBaselineMagnitude;
            const quality = mag < 60 ? '✓' : mag < 80 ? '⚠' : '⚠';
            baselineEl.textContent = `${quality} ${mag.toFixed(0)}µT`;
            baselineEl.style.color = mag < 80 ? 'var(--success)' : 'var(--warning)';
        } else if (state.capturingBaseline) {
            baselineEl.textContent = `⏳ ${state.baselineSampleCount}`;
            baselineEl.style.color = 'var(--fg-muted)';
        } else if (state.autoBaselineRetryCount >= state.autoBaselineMaxRetries) {
            baselineEl.textContent = '✗ Failed';
            baselineEl.style.color = 'var(--error)';
        } else {
            baselineEl.textContent = '...';
            baselineEl.style.color = 'var(--fg-muted)';
        }
    }

    // Calibration quality (orientation diversity)
    if (calibQualityEl) {
        const quality = Math.round((state.calibrationQuality || 0) * 100);
        calibQualityEl.textContent = `${quality}%`;
        calibQualityEl.style.color = quality >= 70 ? 'var(--success)' : quality >= 40 ? 'var(--warning)' : 'var(--fg-muted)';
    }
    if (calibDiversityEl) {
        const diversity = Math.round((state.diversityRatio || 0) * 100);
        calibDiversityEl.textContent = `${diversity}%`;
        calibDiversityEl.style.color = diversity >= 50 ? 'var(--success)' : diversity >= 25 ? 'var(--warning)' : 'var(--fg-muted)';
    }
    if (samplesEl) samplesEl.textContent = totalSamples.toString();

    if (fieldMagEl) {
        fieldMagEl.textContent = earthMag > 0 ? `${earthMag.toFixed(1)} µT` : '-- µT';
    }

    // Update mean residual display (the TRUE measure of calibration quality)
    if (meanResidualEl) {
        if (meanResidual !== undefined && meanResidual !== Infinity && !isNaN(meanResidual)) {
            meanResidualEl.textContent = `${meanResidual.toFixed(1)} µT`;
            // Color based on residual quality
            if (meanResidual < 5) {
                meanResidualEl.style.color = 'var(--success)';
            } else if (meanResidual < 10) {
                meanResidualEl.style.color = 'var(--warning)';
            } else {
                meanResidualEl.style.color = 'var(--error)';
            }
        } else {
            meanResidualEl.textContent = '-- µT';
            meanResidualEl.style.color = 'var(--fg-muted)';
        }
    }

    // Update residual quality indicator
    if (residualQualityEl) {
        let interpretation = '--';
        if (meanResidual !== undefined && meanResidual !== Infinity && !isNaN(meanResidual)) {
            interpretation = meanResidual < 5 ? 'excellent' :
                            meanResidual < 10 ? 'good' :
                            meanResidual < 15 ? 'moderate' : 'poor';
        }
        residualQualityEl.textContent = interpretation.toUpperCase();
        if (interpretation === 'excellent') {
            residualQualityEl.style.color = 'var(--success)';
        } else if (interpretation === 'good') {
            residualQualityEl.style.color = '#8bc34a';
        } else if (interpretation === 'moderate') {
            residualQualityEl.style.color = 'var(--warning)';
        } else {
            residualQualityEl.style.color = 'var(--error)';
        }
    }

    // Update progress bar
    if (barEl) {
        barEl.style.width = `${overall}%`;
        // Color based on confidence level
        if (overall >= 70) {
            barEl.style.background = 'var(--success)';
        } else if (overall >= 40) {
            barEl.style.background = 'var(--warning)';
        } else {
            barEl.style.background = 'var(--error)';
        }
    }

    // Update status text - based on calibration state
    if (statusEl) {
        let status = '';
        if (totalSamples < 50) {
            status = 'Collecting samples...';
        } else if (!state.ready) {
            status = 'Building Earth field estimate...';
        } else if (state.capturingBaseline) {
            status = `Capturing baseline (${state.baselineSampleCount}/${50})...`;
        } else if (state.extendedBaselineActive) {
            // Have baseline - show quality based on residual
            if (meanResidual !== undefined && meanResidual !== Infinity && !isNaN(meanResidual)) {
                if (meanResidual < 10) {
                    status = '✓ Calibrated (Earth + Baseline)';
                } else if (meanResidual < 30) {
                    status = 'Calibrated - rotate for diversity';
                } else {
                    status = 'High residual - extend fingers';
                }
            } else {
                status = '✓ Calibrated (Earth + Baseline)';
            }
        } else {
            // No baseline yet
            if (state.autoBaselineRetryCount >= state.autoBaselineMaxRetries) {
                status = '⚠ Baseline failed - extend fingers & recapture';
            } else {
                status = 'Earth auto ✓ | Baseline capturing...';
            }
        }
        statusEl.textContent = status;
    }

    // Update overall confidence color
    if (overallEl) {
        if (overall >= 70) {
            overallEl.style.color = 'var(--success)';
        } else if (overall >= 40) {
            overallEl.style.color = 'var(--warning)';
        } else {
            overallEl.style.color = 'var(--error)';
        }
    }

    // Update 3D magnetometer calibration visualization
    updateMagCalibrationVis({
        autoHardIronEstimate: state.autoHardIronEstimate,
        autoHardIronRanges: state.autoHardIronRanges
    });

    // Save calibration to localStorage for next session bootstrap
    const now = performance.now();
    if (state.autoHardIronReady) {
        // Save immediately when auto hard iron first becomes ready
        if (!wasAutoHardIronReady) {
            wasAutoHardIronReady = true;
            magCal.save('gambit_calibration');
            console.log('[Calibration] Saved auto hard iron calibration for next session bootstrap');
        } else if (now - lastCalibrationSave > CALIBRATION_SAVE_INTERVAL) {
            // Also save periodically to capture refined estimates
            lastCalibrationSave = now;
            magCal.save('gambit_calibration');
        }
    }
}

// Update 3D magnetometer calibration visualization
function updateMagCalibrationVis(calState: {
    autoHardIronEstimate?: { x: number; y: number; z: number };
    autoHardIronRanges?: { x: number; y: number; z: number };
}): void {
    if (!threeHandSkeleton || !threeHandSkeleton.isMagCalibrationEnabled()) return;

    // Get recent magnetometer samples from session data
    const recentSamples: MagSample[] = [];
    const startIdx = Math.max(0, sessionData.length - 300);
    for (let i = startIdx; i < sessionData.length; i++) {
        const sample = sessionData[i] as unknown as Record<string, number>;
        if (sample.mx_ut !== undefined) {
            recentSamples.push({
                x: sample.mx_ut,
                y: sample.my_ut,
                z: sample.mz_ut
            });
        }
    }

    // Get hard iron offset from calibration state
    const hardIronOffset: MagSample = calState.autoHardIronEstimate || { x: 0, y: 0, z: 0 };

    // Calculate expected magnitude for target range
    const location = geomagneticLocation || getDefaultLocation();
    const expectedMag = location
        ? Math.sqrt(location.horizontal ** 2 + location.vertical ** 2)
        : 50;
    const targetRange = expectedMag * 1.5;  // Match threshold from unified-mag-calibration

    // Calculate axis progress
    const ranges = calState.autoHardIronRanges || { x: 0, y: 0, z: 0 };
    const axisProgress: AxisProgress = {
        x: Math.min(1, ranges.x / targetRange),
        y: Math.min(1, ranges.y / targetRange),
        z: Math.min(1, ranges.z / targetRange)
    };

    // Update the 3D visualization
    threeHandSkeleton.updateMagCalibration({
        samples: recentSamples,
        hardIronOffset: hardIronOffset,
        axisProgress: axisProgress,
        expectedMagnitude: expectedMag
    });
}

function updateData(prenorm: ExtendedTelemetry) {
    // Process telemetry through shared TelemetryProcessor
    // This handles: unit conversion, IMU fusion, gyro bias, mag calibration (unified), filtering
    const decoratedData = telemetryProcessor.process(prenorm);

    // Update calibration confidence UI periodically
    const now = performance.now();
    if (now - lastConfidenceUpdate > CONFIDENCE_UPDATE_INTERVAL) {
        lastConfidenceUpdate = now;
        updateCalibrationConfidenceUI();
        updateMagnetDetectionUI(decoratedData);
    }

    // Store decorated data (raw + calibrated + filtered fields)
    sessionData.push(decoratedData);

    // ===== Update UI displays =====
    // Battery
    if (prenorm.b !== null && prenorm.b !== undefined) {
        (window as any).b.value = Math.floor(prenorm.b);
        (window as any).b.title = prenorm.b + '% battery';
    }

    // Button state
    (window as any).state.innerHTML = prenorm.s || 0;
    (window as any).count.innerHTML = prenorm.n || 0;

    // Sensor values display based on selected data stage
    let mx, my, mz, ax, ay, az, gx, gy, gz;
    let magDecimals = 1, accDecimals = 3, gyrDecimals = 1;
    
    switch (currentDataStage) {
        case 'raw':
            // Raw LSB values from sensor
            mx = prenorm.mx || 0;
            my = prenorm.my || 0;
            mz = prenorm.mz || 0;
            ax = prenorm.ax || 0;
            ay = prenorm.ay || 0;
            az = prenorm.az || 0;
            gx = prenorm.gx || 0;
            gy = prenorm.gy || 0;
            gz = prenorm.gz || 0;
            magDecimals = 0;
            accDecimals = 0;
            gyrDecimals = 0;
            break;
        case 'calibrated':
            // Iron-corrected magnetometer (hard/soft iron compensation)
            mx = (decoratedData as any).calibrated_mx ?? decoratedData.mx_ut ?? 0;
            my = (decoratedData as any).calibrated_my ?? decoratedData.my_ut ?? 0;
            mz = (decoratedData as any).calibrated_mz ?? decoratedData.mz_ut ?? 0;
            ax = decoratedData.ax_g || 0;
            ay = decoratedData.ay_g || 0;
            az = decoratedData.az_g || 0;
            gx = decoratedData.gx_dps || 0;
            gy = decoratedData.gy_dps || 0;
            gz = decoratedData.gz_dps || 0;
            break;
        case 'fused':
            // Earth field removed (residual = finger magnet signal)
            mx = decoratedData.residual_mx ?? decoratedData.mx_ut ?? 0;
            my = decoratedData.residual_my ?? decoratedData.my_ut ?? 0;
            mz = decoratedData.residual_mz ?? decoratedData.mz_ut ?? 0;
            ax = decoratedData.ax_g || 0;
            ay = decoratedData.ay_g || 0;
            az = decoratedData.az_g || 0;
            gx = decoratedData.gx_dps || 0;
            gy = decoratedData.gy_dps || 0;
            gz = decoratedData.gz_dps || 0;
            break;
        case 'filtered':
            // Kalman filtered (best for visualization)
            mx = decoratedData.filtered_mx ?? decoratedData.residual_mx ?? decoratedData.mx_ut ?? 0;
            my = decoratedData.filtered_my ?? decoratedData.residual_my ?? decoratedData.my_ut ?? 0;
            mz = decoratedData.filtered_mz ?? decoratedData.residual_mz ?? decoratedData.mz_ut ?? 0;
            ax = decoratedData.ax_g || 0;
            ay = decoratedData.ay_g || 0;
            az = decoratedData.az_g || 0;
            gx = decoratedData.gx_dps || 0;
            gy = decoratedData.gy_dps || 0;
            gz = decoratedData.gz_dps || 0;
            break;
        case 'converted':
        default:
            // Converted to physical units (default)
            mx = decoratedData.mx_ut || 0;
            my = decoratedData.my_ut || 0;
            mz = decoratedData.mz_ut || 0;
            ax = decoratedData.ax_g || 0;
            ay = decoratedData.ay_g || 0;
            az = decoratedData.az_g || 0;
            gx = decoratedData.gx_dps || 0;
            gy = decoratedData.gy_dps || 0;
            gz = decoratedData.gz_dps || 0;
            break;
    }
    
    // Update display
    (window as any).mX.innerHTML = mx.toFixed(magDecimals);
    (window as any).mY.innerHTML = my.toFixed(magDecimals);
    (window as any).mZ.innerHTML = mz.toFixed(magDecimals);
    (window as any).aX.innerHTML = ax.toFixed(accDecimals);
    (window as any).aY.innerHTML = ay.toFixed(accDecimals);
    (window as any).aZ.innerHTML = az.toFixed(accDecimals);
    (window as any).gX.innerHTML = gx.toFixed(gyrDecimals);
    (window as any).gY.innerHTML = gy.toFixed(gyrDecimals);
    (window as any).gZ.innerHTML = gz.toFixed(gyrDecimals);

    // Update calibration status indicator
    const dataStageInfoEl = document.getElementById('dataStageInfo');
    if (dataStageInfoEl && currentDataStage === 'fused') {
        if ((decoratedData as any).fused_incomplete) {
            if ((decoratedData as any).fused_uncalibrated) {
                dataStageInfoEl.innerHTML = '⚠️ <b>Best Effort:</b> Showing raw magnetic field (iron calibration missing)';
                dataStageInfoEl.style.color = '#ff9500';
            } else {
                dataStageInfoEl.innerHTML = '⚠️ <b>Best Effort:</b> Showing iron-corrected field (Earth field calibration missing)';
                dataStageInfoEl.style.color = '#ff9500';
            }
        } else {
            dataStageInfoEl.innerHTML = dataStageInfo['fused'].info;
            dataStageInfoEl.style.color = 'var(--fg-muted)';
        }
    }

    // ===== CUBE VISUALIZATION =====
    // Get orientation from telemetry processor
    const euler = telemetryProcessor.getEulerAngles() || { roll: 0, pitch: 0, yaw: 0 };

    // --- ACCELEROMETER CUBE: Tilt from gravity ---
    // Use raw values for cube visualization (always)
    const cubeAx = prenorm.ax || 0;
    const cubeAy = prenorm.ay || 0;
    const cubeAz = prenorm.az || 0;
    
    const accRoll = Math.atan2(cubeAy, cubeAz) * (180 / Math.PI);
    const accPitch = Math.atan2(-cubeAx, Math.sqrt(cubeAy * cubeAy + cubeAz * cubeAz)) * (180 / Math.PI);
    
    const filteredAccRoll = cubeFilters.acc.x.filter(accRoll);
    const filteredAccPitch = cubeFilters.acc.y.filter(accPitch);
    
    (window as any).cubeA.style = `transform: rotateX(${filteredAccPitch}deg) rotateY(${filteredAccRoll}deg) rotateZ(0deg);`;

    // --- GYROSCOPE CUBE: Fused orientation from TelemetryProcessor ---
    const filteredGyroRoll = cubeFilters.gyro.x.filter(euler.roll);
    const filteredGyroPitch = cubeFilters.gyro.y.filter(euler.pitch);
    const filteredGyroYaw = cubeFilters.gyro.z.filter(euler.yaw);

    (window as any).cubeG.style = `transform: rotateX(${filteredGyroPitch}deg) rotateY(${filteredGyroRoll}deg) rotateZ(${filteredGyroYaw}deg);`;

    // Update hand 3D renderer with orientation using quaternion (avoids gimbal lock)
    // Priority: stored quaternion from playback > live processing result > live AHRS
    const handQuaternion = (prenorm as any).orientation_w !== undefined ? {
        w: (prenorm as any).orientation_w,
        x: (prenorm as any).orientation_x ?? 0,
        y: (prenorm as any).orientation_y ?? 0,
        z: (prenorm as any).orientation_z ?? 0
    } : decoratedData.orientation_w !== undefined ? {
        w: decoratedData.orientation_w,
        x: decoratedData.orientation_x ?? 0,
        y: decoratedData.orientation_y ?? 0,
        z: decoratedData.orientation_z ?? 0
    } : telemetryProcessor.getQuaternion();

    // Update Three.js hand skeleton with quaternion (gimbal lock free)
    if (threeHandSkeleton && threeHandEnabled && handQuaternion) {
        threeHandSkeleton.updateQuaternion(handQuaternion);
    }

    // Keep Euler angles for other uses (cube display, validation panel)
    const handEuler = {
        roll: (prenorm as any).euler_roll !== undefined ? (prenorm as any).euler_roll : euler.roll,
        pitch: (prenorm as any).euler_pitch !== undefined ? (prenorm as any).euler_pitch : euler.pitch,
        yaw: (prenorm as any).euler_yaw !== undefined ? (prenorm as any).euler_yaw : euler.yaw
    };

    // Update magnetic trajectory with residual field
    if (magTrajectory && magTrajectoryEnabled && decoratedData.residual_mx !== undefined) {
        magTrajectory.addPoint(
            decoratedData.residual_mx!,
            decoratedData.residual_my!,
            decoratedData.residual_mz!
        );

        // Update stats every 50 points to reduce DOM updates
        if ((magTrajectory as any).points && (magTrajectory as any).points.length % 50 === 0) {
            updateMagTrajectoryStats();
        }
    }

    // Update calibration system with current sensor state
    latestSensorData = { ax, ay, az, gx, gy, gz, mx, my, mz, t: prenorm.t || 0 };
    updateValidationPanel(handEuler);

    // Update Euler debug display
    const threeEulerDebug = document.getElementById('threeEulerDebug');
    if (threeEulerDebug && euler) {
        threeEulerDebug.textContent = `Roll: ${euler.roll.toFixed(1)}° | Pitch: ${euler.pitch.toFixed(1)}° | Yaw: ${euler.yaw.toFixed(1)}°`;
    }

    // --- MAGNETOMETER CUBE: Shows magnetic field vector direction ---
    // Use raw values for cube visualization (always)
    const cubeMx = prenorm.mx || 0;
    const cubeMy = prenorm.my || 0;
    const cubeMz = prenorm.mz || 0;
    
    const magAzimuth = Math.atan2(cubeMy, cubeMx) * (180 / Math.PI);
    const magMagnitude = Math.sqrt(cubeMx * cubeMx + cubeMy * cubeMy + cubeMz * cubeMz);
    const magElevation = magMagnitude > 0.1 ? 
        Math.asin(cubeMz / magMagnitude) * (180 / Math.PI) : 0;
    const magRoll = Math.atan2(cubeMx, Math.sqrt(cubeMy * cubeMy + cubeMz * cubeMz)) * (180 / Math.PI);
    
    const filteredMagAzimuth = cubeFilters.mag.z.filter(magAzimuth);
    const filteredMagElevation = cubeFilters.mag.x.filter(magElevation);
    const filteredMagRoll = cubeFilters.mag.y.filter(magRoll);
    
    (window as any).cubeM.style = `transform: rotateZ(${-filteredMagAzimuth}deg) rotateX(${filteredMagElevation}deg) rotateY(${filteredMagRoll}deg);`;
}

var newlineRegex = /\n/g;

const btoa = (str: string): string => {
if (typeof window === 'undefined' || !window.btoa)  {
    // const Buffer = require('buffer')
    return Buffer.from(str, 'binary').toString('base64')
} else return window.btoa(str)
}

const atob = (str: string): string => {
if (typeof window === 'undefined' || !window.atob)  {
    // const Buffer = require('buffer')
    return Buffer.from(str, 'base64').toString('binary')
}
else return window.atob(str)
}

function b64EncodeUnicode(str: string): string {
return btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, function(match: string, p1: string) {
    return String.fromCharCode(parseInt('0x' + p1));
}));
}

function b64DecodeUnicode(str: string): string {
    // atob on Mobile Safari for iOS 9 will throw an exception if there's a newline.
    var b64Decoded = atob(str.replace(newlineRegex, ''));
    var decodedWithUnicodeHexesRestored = Array.prototype.map.call(
        b64Decoded,
        hexEncodeCharCode
    )
    .join('');

    return decodeURIComponent(decodedWithUnicodeHexesRestored);
}

function hexEncodeCharCode(c: string): string {
    return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
}

// GitHub LFS Upload - ensures large files are stored properly in Git LFS
// (Inline version of shared/github-lfs-upload.js for non-module scripts)

async function sha256ForLFS(content: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(content);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

function createLFSPointer(oid: string, size: number): string {
    return `version https://git-lfs.github.com/spec/v1\noid sha256:${oid}\nsize ${size}\n`;
}

async function getFileShaForLFS(token: string, owner: string, repo: string, path: string): Promise<string | null> {
    const endpoint = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;
    try {
        const response = await fetch(endpoint, {
            method: 'GET',
            headers: {
                'Authorization': `token ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            }
        });
        if (response.ok) {
            const data = await response.json();
            return data.sha;
        }
        return null;
    } catch (e) {
        return null;
    }
}

// ===== GitHub Upload via API Proxy =====
// Uses the shared github-upload module with /api/github-upload endpoint
async function proxyPutData(rawSessionData: TelemetrySample[]) {
    // Check for secret using the module's function (reads from localStorage)
    if (!hasUploadSecret()) {
        console.log("proxyPutData: No upload secret configured. Skipping upload.");
        updateUploadStatus('No upload secret configured', 'error');
        return;
    }

    const d = new Date();
    const filename = `${d.toISOString().replace(/:/g, '_')}.json`;

    // Build v2.1 schema export data
    const exportData = {
        version: '2.1',
        timestamp: d.toISOString(),
        samples: rawSessionData,
        labels: [],
        metadata: {
            sample_rate: 26,
            device: 'GAMBIT',
            firmware_version: firmwareVersion || 'unknown',
            calibration: telemetryProcessor.getMagCalibration()?.toJSON(),
            location: exportLocationMetadata(geomagneticLocation),
            subject_id: 'unknown',
            environment: 'unknown',
            hand: 'unknown',
            split: 'train',
            magnet_config: 'none',
            magnet_type: 'unknown',
            notes: '',
            session_type: 'streaming'
        }
    };

    const content = JSON.stringify(exportData, null, 2);

    console.log("proxyPutData: Starting GitHub upload for", filename);
    updateUploadStatus('Uploading...', 'progress');

    try {
        // Use the shared github-upload module (API proxy)
        const result = await uploadViaProxy({
            branch: 'data',
            path: `GAMBIT/${filename}`,
            content,
            message: 'GAMBIT session data',
            onProgress: (progress) => {
                console.log(`proxyPutData: ${progress.stage} - ${progress.message}`);
                if (progress.stage === 'uploading') {
                    updateUploadStatus('Uploading...', 'progress');
                }
            }
        });

        console.log("proxyPutData: ✓ Upload complete!", result.url);
        updateUploadStatus(`✓ Uploaded: ${filename}`, 'success');
        return result;
    } catch (err) {
        console.error("proxyPutData: Upload failed", err);
        updateUploadStatus(`✗ Upload failed: ${(err as Error).message}`, 'error');
        throw err;
    }
}

// Helper to update upload status in UI
function updateUploadStatus(message: string, type: string) {
    const statusEl = document.getElementById('uploadStatus');
    if (statusEl) {
        statusEl.textContent = message;
        statusEl.style.color = type === 'error' ? 'var(--error)' :
                               type === 'success' ? 'var(--success)' :
                               'var(--fg-muted)';
    }
}

// Dispatcher function - routes to proxy or direct github based on uploadMethod
async function uploadSessionData(rawSessionData: TelemetrySample[]) {
    if (uploadMethod === 'proxy') {
        return proxyPutData(rawSessionData);
    } else {
        return ghPutData(rawSessionData);
    }
}

async function ghPutData(rawSessionData: TelemetrySample[]) {

    if (!ghToken) {
        console.log("ghToken not found. skipping upload.");
        updateUploadStatus('No GitHub token configured', 'error');
        return;
    }

    // Build v2.1 schema export data (consistent with collector-app.js)
    const exportData = {
        version: '2.1',
        timestamp: new Date().toISOString(),
        samples: rawSessionData,
        labels: [],  // index.html doesn't support labeling
        metadata: {
            sample_rate: 26,  // Match firmware accelOn rate
            device: 'GAMBIT',
            firmware_version: firmwareVersion || 'unknown',
            calibration: telemetryProcessor.getMagCalibration()?.toJSON(),
            location: exportLocationMetadata(geomagneticLocation),
            subject_id: 'unknown',
            environment: 'unknown',
            hand: 'unknown',
            split: 'train',
            magnet_config: 'none',
            magnet_type: 'unknown',
            notes: '',
            session_type: 'streaming'  // Distinguishes from collector's 'recording'
        }
    };

    const content = JSON.stringify(exportData, null, 2);
    const d = new Date()
    const filename = `${d.toISOString()}.json`
    const filePath = `data/GAMBIT/${filename}`;
    const owner = 'christopherdebeer';
    const repo = 'simcap';

    console.log("ghPutData: Starting LFS upload for", filename);
    updateUploadStatus('Uploading via GitHub LFS...', 'progress');

    try {
        // Calculate content hash and size for LFS
        const encoder = new TextEncoder();
        const contentBytes = encoder.encode(content);
        const size = contentBytes.length;
        const oid = await sha256ForLFS(content);
        
        console.log("ghPutData: Content hash:", oid.substring(0, 16) + "...", "Size:", size);

        // Step 1: Request LFS upload URL via Batch API
        const lfsEndpoint = `https://github.com/${owner}/${repo}.git/info/lfs/objects/batch`;
        
        const lfsResponse = await fetch(lfsEndpoint, {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${btoa(`${owner}:${ghToken}`)}`,
                'Content-Type': 'application/vnd.git-lfs+json',
                'Accept': 'application/vnd.git-lfs+json'
            },
            body: JSON.stringify({
                operation: 'upload',
                transfers: ['basic'],
                objects: [{ oid: oid, size: size }]
            })
        });

        if (!lfsResponse.ok) {
            const errorText = await lfsResponse.text();
            throw new Error(`LFS Batch API error (${lfsResponse.status}): ${errorText}`);
        }

        const lfsData = await lfsResponse.json();
        const lfsObject = lfsData.objects?.[0];
        if (!lfsObject) {
            throw new Error('No LFS object in response');
        }

        // Step 2: Upload to LFS storage (if needed)
        const uploadAction = lfsObject.actions?.upload;
        if (uploadAction) {
            console.log("ghPutData: Uploading to LFS storage...");
            const uploadResponse = await fetch(uploadAction.href, {
                method: 'PUT',
                headers: {
                    ...(uploadAction.header || {}),
                    'Content-Type': 'application/octet-stream'
                },
                body: contentBytes
            });

            if (!uploadResponse.ok) {
                const errorText = await uploadResponse.text();
                throw new Error(`LFS upload failed (${uploadResponse.status}): ${errorText}`);
            }
            console.log("ghPutData: LFS upload complete");
        } else {
            console.log("ghPutData: File already exists in LFS storage");
        }

        // Step 3: Create commit with LFS pointer
        const pointerContent = createLFSPointer(oid, size);
        const pointerBase64 = btoa(pointerContent);
        
        const existingSha = await getFileShaForLFS(ghToken, owner, repo, filePath);
        
        const commitEndpoint = `https://api.github.com/repos/${owner}/${repo}/contents/${filePath}`;
        const commitBody = {
            message: `GAMBIT Data ingest ${filename}`,
            content: pointerBase64
        };
        
        if (existingSha) {
            (commitBody as any).sha = existingSha;
        }

        const commitResponse = await fetch(commitEndpoint, {
            method: 'PUT',
            headers: {
                'Authorization': `token ${ghToken}`,
                'Content-Type': 'application/json',
                'Accept': 'application/vnd.github.v3+json'
            },
            body: JSON.stringify(commitBody)
        });

        if (!commitResponse.ok) {
            const errorData = await commitResponse.json();
            throw new Error(`Commit failed (${commitResponse.status}): ${errorData.message || JSON.stringify(errorData)}`);
        }

        const commitResult = await commitResponse.json();
        console.log("ghPutData: ✓ Upload complete!", commitResult.content?.html_url);
        console.log("ghPutData: LFS OID:", oid.substring(0, 16) + "...");
        updateUploadStatus(`✓ Uploaded: ${filename}`, 'success');

    } catch(err) {
        console.error("ghPutData: Upload failed", err);
        updateUploadStatus(`✗ Upload failed: ${(err as Error).message}`, 'error');
    }

}

// ===== Cloud Upload Storage & Event Handlers =====
// Load upload settings from localStorage
try {
    // Load proxy secret - use the module's getter for consistency
    const s = getUploadSecret();
    if (s) {
        proxySecret = s;
        const proxySecretInput = document.getElementById('proxySecret') as HTMLInputElement;
        if (proxySecretInput) proxySecretInput.value = s;
    }

    // Load upload method preference
    const m = localStorage.getItem("simcap_upload_method");
    if (m && (m === 'proxy' || m === 'github')) {
        uploadMethod = m;
        const methodSelect = document.getElementById('uploadMethod') as HTMLSelectElement;
        if (methodSelect) methodSelect.value = m;
    }

    // Load GitHub token
    const t = localStorage.getItem("ghToken");
    if (t) {
        ghToken = t;
        const tokenInput = document.getElementById('token') as HTMLInputElement;
        if (tokenInput) tokenInput.value = t;
    }
    else console.log("No ghToken in localStorage");

    // Update UI visibility based on current method
    updateUploadMethodUI();
} catch(e) {
    console.log("Error loading upload settings from localStorage:", e);
}

// Helper to show/hide upload method rows
function updateUploadMethodUI() {
    const proxyRow = document.getElementById('proxySecretRow');
    const ghRow = document.getElementById('ghTokenRow');
    if (proxyRow) proxyRow.style.display = uploadMethod === 'proxy' ? 'flex' : 'none';
    if (ghRow) ghRow.style.display = uploadMethod === 'github' ? 'flex' : 'none';
}

// Upload method selector
const uploadMethodSelect = document.getElementById('uploadMethod');
if (uploadMethodSelect) {
    uploadMethodSelect.addEventListener('change', function(e) {
        uploadMethod = (e.target as HTMLSelectElement).value as 'proxy' | 'github';
        localStorage.setItem("simcap_upload_method", uploadMethod);
        updateUploadMethodUI();
        console.log("Upload method changed to:", uploadMethod);
    });
}

// Proxy secret input - use the module's setter for consistency
const proxySecretInput = document.getElementById('proxySecret');
if (proxySecretInput) {
    proxySecretInput.addEventListener('change', function(e) {
        proxySecret = (e.target as HTMLInputElement).value;
        setUploadSecret(proxySecret || null);  // Use module function
        console.log("Proxy secret updated");
    });
}

// GitHub token input
const tokenInput = document.getElementById('token');
if (tokenInput) {
    tokenInput.addEventListener('change', function(e) {
        ghToken = (e.target as HTMLInputElement).value;
        localStorage.setItem("ghToken", ghToken);
        console.log("GitHub token updated");
    });
}

// ===== Connect Button - Uses GambitClient =====
window.connect.onclick = async function() {
    if (gambitClient.isConnected()) {
        // Disconnect
        console.log('[GAMBIT] Disconnecting...');
        if (isStreaming) {
            await gambitClient.stopStreaming();
        }
        gambitClient.disconnect();
    } else {
        // Connect using GambitClient (handles frame-based protocol)
        console.log('[GAMBIT] Connecting via GambitClient...');
        try {
            const fw = await gambitClient.connect();
            console.log('[GAMBIT] Connected, firmware:', fw);
        } catch (err) {
            console.error('[GAMBIT] Connection failed:', err);
        }
    }
}

// ===== Get Data Button - Uses GambitClient Streaming =====
window.getdata.onclick = async function() {
    if (!gambitClient.isConnected()) {
        console.warn('[GAMBIT] Not connected');
        return;
    }

    if (isStreaming) {
        // Stop streaming
        console.log('[GAMBIT] Stopping stream...');
        await gambitClient.stopStreaming();
        isStreaming = false;
        window.getdata.innerHTML = "Get data";
    } else {
        // Start streaming (uses frame-based protocol)
        console.log('[GAMBIT] Starting stream via GambitClient...');
        try {
            await gambitClient.startStreaming();
            isStreaming = true;
            window.getdata.innerHTML = "Stop";
        } catch (err) {
            console.error('[GAMBIT] Failed to start streaming:', err);
        }
    }
}

// ===== Gesture Inference (using module) =====
// Note: GestureInference type comes from dynamically loaded script
 
let gestureInference: { load: () => Promise<void>; isReady: boolean; addSample: (sample: TelemetrySample) => void; labels: string[] } | null = null;
let gestureUI: GestureUIController | null = null;

// Initialize gesture inference using the module
async function initGestureInference() {
    console.log('=== GAMBIT Gesture Inference Initialization ===');
    console.log('[GAMBIT] Checking gesture inference globals:', {
        GestureInference: typeof (window as any).GestureInference,
        createGestureInference: typeof (window as any).createGestureInference,
        MagneticFingerInference: typeof (window as any).MagneticFingerInference,
        createMagneticFingerInference: typeof (window as any).createMagneticFingerInference,
        GESTURE_MODELS: typeof (window as any).GESTURE_MODELS,
        FINGER_MODELS: typeof (window as any).FINGER_MODELS
    });

    // Create UI controller using the module helper
    gestureUI = createGestureUI({
        statusEl: document.getElementById('modelStatus'),
        nameEl: document.getElementById('gestureName'),
        confidenceEl: document.getElementById('gestureConfidence'),
        timeEl: document.getElementById('inferenceTime'),
        displayEl: document.getElementById('gestureDisplay'),
        probabilitiesEl: document.getElementById('gestureProbabilities')
    });

    // Check if gesture inference is available (requires gesture-inference.js loaded)
    if (!isGestureInferenceAvailable()) {
        console.warn('[GAMBIT] Gesture inference not available - globals not found');
        gestureUI.setStatus('error', 'Gesture inference not loaded');
        return;
    }

    try {
        // Create gesture inference using the module wrapper
        gestureInference = createGesture('v1', {
            confidenceThreshold: 0.3,
            onPrediction: (result) => {
                console.log('[GAMBIT] Prediction:', result.gesture, result.confidence.toFixed(2));
                gestureUI?.updatePrediction(result);
            },
            onReady: () => {
                console.log('[GAMBIT] ✓ Model ready');
                gestureUI?.setStatus('ready', 'Model ready (v1)');
                if (gestureInference) {
                    gestureUI?.initProbabilityBars(gestureInference.labels);
                }
            },
            onError: (error: Error) => {
                console.error('[GAMBIT] Model error:', error.message);
                gestureUI?.setStatus('error', 'Model error: ' + error.message);
            }
        });

        await gestureInference!.load();
    } catch (error) {
        console.error('[GAMBIT] Failed to initialize gesture inference:', error);
        gestureUI?.setStatus('error', 'Model unavailable: ' + (error as Error).message);
    }
}

// Hook into data updates to feed inference
const originalUpdateData = updateData;
let sampleCount = 0;
const wrappedUpdateData = function(prenorm: ExtendedTelemetry) {
    // Call original function
    originalUpdateData(prenorm);

    sampleCount++;

    // Feed to gesture inference
    if (gestureInference && gestureInference.isReady) {
        gestureInference.addSample(prenorm);

        // Log first few samples for debugging
        if (sampleCount <= 3) {
            console.log(`[GAMBIT] Sample ${sampleCount} fed to inference:`, {
                ax: prenorm.ax,
                ay: prenorm.ay,
                az: prenorm.az,
                gx: prenorm.gx,
                gy: prenorm.gy,
                gz: prenorm.gz,
                mx: prenorm.mx,
                my: prenorm.my,
                mz: prenorm.mz
            });
        }
    } else if (gestureInference && !gestureInference.isReady) {
        if (sampleCount === 1) {
            console.warn('[GAMBIT] Data received but model not ready yet');
        }
    } else {
        if (sampleCount === 1) {
            console.warn('[GAMBIT] Data received but gesture inference not initialized');
        }
    }
};

// Initialize on page load
try {
    initGestureInference();
} catch (e) {
    console.error('[GAMBIT] Failed to init gesture inference:', e);
}

try {
    initThreeHandSkeleton();
} catch (e) {
    console.error('[GAMBIT] Failed to init Three.js hand:', e);
}

try {
    initMagneticTrajectory();
} catch (e) {
    console.error('[GAMBIT] Failed to init magnetic trajectory:', e);
}

// ===== Reset Orientation Button =====
// Allows user to re-initialize the AHRS from current accelerometer reading
document.getElementById('threeResetBtn')?.addEventListener('click', () => {
    console.log('[GAMBIT] Resetting orientation...');
    telemetryProcessor.reset();
    // Reset cube filters too
    Object.values(cubeFilters).forEach(filterSet => {
        Object.values(filterSet).forEach(filter => filter.reset());
    });
    console.log('[GAMBIT] Orientation reset complete');
});

// ===== Session Playback System (using module) =====
let sessionPlayback: SessionPlayback | null = null;

// UI elements for playback
const playbackElements: PlaybackUIElements = {
    select: null,
    playBtn: null,
    stopBtn: null,
    slider: null,
    timeEl: null,
    statusEl: null,
    card: null,
    speedSelect: null
};

// Update UI based on playback state
function updatePlaybackUI(state: PlaybackState) {
    if (!playbackElements.timeEl) return;

    // Update time display
    playbackElements.timeEl.textContent = 
        `${formatTime(state.currentTime)} / ${formatTime(state.duration)}`;
    
    // Update slider
    if (playbackElements.slider) {
        playbackElements.slider.value = String(state.currentIndex);
    }

    // Update button states
    if (playbackElements.playBtn) {
        playbackElements.playBtn.textContent = state.isPlaying ? 'Pause' : 'Play';
    }

    // Update card styling
    if (playbackElements.card) {
        if (state.isPlaying) {
            playbackElements.card.classList.add('playing');
        } else {
            playbackElements.card.classList.remove('playing');
        }
    }

    // Update status text
    if (playbackElements.statusEl) {
        if (state.isPlaying) {
            playbackElements.statusEl.textContent = 'Playing...';
        } else if (state.currentIndex === 0 && !state.isPlaying) {
            playbackElements.statusEl.textContent = state.session ? 
                `Loaded: ${state.totalSamples} samples` : 'Stopped';
        } else {
            playbackElements.statusEl.textContent = 'Paused';
        }
    }

    // Update stop button
    if (playbackElements.stopBtn) {
        playbackElements.stopBtn.disabled = state.currentIndex === 0 && !state.isPlaying;
    }
}

// Populate session selector
function populateSessionSelect(sessions: SessionMetadata[]) {
    if (!playbackElements.select) return;

    playbackElements.select.innerHTML = '<option value="">-- Select a session --</option>';

    sessions.forEach((session: SessionMetadata, index: number) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = formatSessionDisplay(session);
        playbackElements.select!.appendChild(option);
    });

    if (playbackElements.statusEl) {
        playbackElements.statusEl.textContent = `${sessions.length} sessions available`;
    }
}

// Initialize playback system using the module
async function initPlayback() {
    // Cache UI elements
    playbackElements.select = document.getElementById('sessionSelect') as HTMLSelectElement | null;
    playbackElements.playBtn = document.getElementById('playBtn') as HTMLButtonElement | null;
    playbackElements.stopBtn = document.getElementById('stopBtn') as HTMLButtonElement | null;
    playbackElements.slider = document.getElementById('playbackSlider') as HTMLInputElement | null;
    playbackElements.timeEl = document.getElementById('playbackTime');
    playbackElements.statusEl = document.getElementById('playbackStatus');
    playbackElements.card = document.getElementById('playbackCard');
    playbackElements.speedSelect = document.getElementById('playbackSpeed') as HTMLSelectElement | null;

    // Create SessionPlayback instance using the module
    // Create SessionPlayback instance using the module
    // Uses /api/sessions endpoint by default (configured in session-playback.js)
    sessionPlayback = new SessionPlayback({
        sampleRate: 20,
        
        // Feed samples through updateData (same as live data)
        onSample: (sample) => {
            updateData(sample);
        },
        
        // Update UI on state changes
        onStateChange: (state) => {
            updatePlaybackUI(state);
        },
        
        // Handle manifest loaded
        onManifestLoaded: (sessions) => {
            populateSessionSelect(sessions);
        },
        
        // Handle session loaded
        onSessionLoaded: ({ session, sampleCount, duration }) => {
            if (playbackElements.playBtn) {
                playbackElements.playBtn.disabled = false;
            }
            if (playbackElements.slider) {
                playbackElements.slider.disabled = false;
                playbackElements.slider.max = String(sampleCount - 1);
                playbackElements.slider.value = '0';
            }
        },
        
        // Handle errors
        onError: (error) => {
            console.error('[Playback] Error:', error);
            if (playbackElements.statusEl) {
                playbackElements.statusEl.textContent = 'Error: ' + error.message;
            }
        }
    });

    // Wire up UI controls
    if (playbackElements.select) {
        playbackElements.select.addEventListener('change', async (e: Event) => {
            sessionPlayback?.stop();
            if ((e.target as HTMLSelectElement).value !== '') {
                if (playbackElements.statusEl) {
                    playbackElements.statusEl.textContent = 'Loading session...';
                }
                if (playbackElements.playBtn) {
                    playbackElements.playBtn.disabled = true;
                }
                await sessionPlayback?.loadSession(parseInt((e.target as HTMLSelectElement).value));
            }
        });
    }

    if (playbackElements.playBtn) {
        playbackElements.playBtn.addEventListener('click', () => {
            sessionPlayback?.toggle();
        });
    }

    if (playbackElements.stopBtn) {
        playbackElements.stopBtn.addEventListener('click', () => {
            sessionPlayback?.stop();
        });
    }

    if (playbackElements.slider) {
        playbackElements.slider.addEventListener('input', (e: Event) => {
            sessionPlayback?.pause();
            sessionPlayback?.seekTo(parseInt((e.target as HTMLInputElement).value));
        });
    }

    if (playbackElements.speedSelect) {
        playbackElements.speedSelect.addEventListener('change', (e: Event) => {
            sessionPlayback?.setSpeed(parseFloat((e.target as HTMLSelectElement).value));
        });
    }

    // Load manifest
    try {
        if (playbackElements.statusEl) {
            playbackElements.statusEl.textContent = 'Loading sessions...';
        }
        await sessionPlayback?.loadManifest();
    } catch (error) {
        if (playbackElements.statusEl) {
            playbackElements.statusEl.textContent = 'Failed to load sessions';
        }
        if (playbackElements.select) {
            playbackElements.select.innerHTML = '<option value="">-- Error loading --</option>';
        }
    }
}

// Initialize playback system
try {
    initPlayback();
} catch (e) {
    console.error('[GAMBIT] Failed to init playback:', e);
}

// ===== Global Debug Functions =====
// Expose diagnostic functions for mobile debugging (call from console)

/**
 * Get log buffer
 * Usage: getLog() - returns array of log entries
 *        getLog(true) - also prints to console
 */
(window as any).getLog = function(print: boolean = false): string[] {
    const buffer = getLogBuffer();
    if (print) {
        console.log('=== LOG BUFFER ===');
        buffer.forEach(entry => console.log(entry));
        console.log('=== END LOG ===');
    }
    return buffer;
};

/**
 * Get magnetometer diagnostic summary
 * Usage: getMagDiagSummary()
 */
(window as any).getMagDiagSummary = function(): void {
    const summary = telemetryProcessor.getDiagnosticSummary();
    const calState = telemetryProcessor.getMagCalibration().getState();

    console.log('=== MAG DIAGNOSTIC SUMMARY ===');
    console.log(`useMag: ${summary.useMagnetometer}, trust: ${summary.magTrust}`);
    console.log(`hasIronCal: ${summary.hasIronCal}`);
    console.log(`geomagRef: ${summary.geomagRef ? `H=${summary.geomagRef.horizontal.toFixed(1)} V=${summary.geomagRef.vertical.toFixed(1)} D=${summary.geomagRef.declination.toFixed(1)}` : 'none'}`);
    console.log(`lastResidual: ${summary.lastResidual.toFixed(1)} µT (should be ~0 without magnets)`);
    console.log(`lastYaw: ${summary.lastYaw.toFixed(1)}°`);
    console.log(`samples: ${summary.sampleCount}`);
    console.log(`calReady: ${calState.ready}, confidence: ${(calState.confidence * 100).toFixed(0)}%`);
    console.log(`earthMag: ${calState.earthMagnitude.toFixed(1)} µT`);
    console.log('==============================');
};

/**
 * Export diagnostic log as downloadable file
 * Usage: exportLog()
 */
(window as any).exportLog = exportLog;
