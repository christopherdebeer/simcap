/**
 * GAMBIT Collector Application
 * Main entry point that coordinates all modules
 */

import { state, resetSession, GambitClient, getSessionSegmentJSON, getSessionJSON } from './modules/state.js';
import { initLogger, log, copyToClipboard, copyLogToClipboard } from './modules/logger.js';
import { KalmanFilter3D } from '@filters';
import { MadgwickAHRS } from '@filters';
import {
    initCalibration,
    updateCalibrationStatus,
    initCalibrationUI,
    calibrationInstance,
    setStoreSessionCallback as setCalibrationSessionCallback,
    CalibrationResult,
    CalibrationSample
} from './modules/calibration-ui.js';
import { onTelemetry, setDependencies as setTelemetryDeps, resetIMU, getProcessor } from './modules/telemetry-handler.js';
import { setCallbacks as setConnectionCallbacks, initConnectionUI } from './modules/connection-manager.js';
import { setCallbacks as setRecordingCallbacks, initRecordingUI, startRecording, pauseRecording, resumeRecording } from './modules/recording-controls.js';
import {
    initWizard,
    setDependencies as setWizardDeps,
    getWizardState,
    getCalibrationBuffers
} from './modules/wizard.js';
import { MagneticTrajectory } from './modules/magnetic-trajectory.js';
import { ThreeJSHandSkeleton, type MagSample, type AxisProgress } from './shared/threejs-hand-skeleton.js';
import { MagneticFingerInference, createMagneticFingerInference } from './gesture-inference.js';
import type { FingerPrediction, MagneticSample } from './gesture-inference.js';
import {
    getBrowserLocation,
    getDefaultLocation,
    exportLocationMetadata,
    formatLocation,
    formatFieldData,
    GeomagneticLocation
} from './shared/geomagnetic-field.js';
import {
    uploadSessionWithRetry,
    uploadToGitHub as uploadDirectToGitHub,
    getUploadSecret,
    setUploadSecret,
    hasUploadSecret
} from './shared/github-upload.js';

// ===== Type Definitions =====

interface PoseState {
    enabled: boolean;
    currentPose: FingerStates | null;
    confidence: number;
    updateCount: number;
    orientation?: EulerAngles | null;
}

interface FingerStates {
    thumb: number;
    index: number;
    middle: number;
    ring: number;
    pinky: number;
}

interface EulerAngles {
    roll: number;
    pitch: number;
    yaw: number;
}

interface PoseEstimationOptions {
    useCalibration: boolean;
    useFiltering: boolean;
    useOrientation: boolean;
    show3DOrientation: boolean;
    useParticleFilter: boolean;
    useMLInference: boolean;
    enableHandOrientation: boolean;
    smoothHandOrientation: boolean;
    handOrientationAlpha: number;
}

interface PoseUpdateData {
    magField: { x: number; y: number; z: number };
    orientation: { w: number; x: number; y: number; z: number } | null;
    euler: EulerAngles | null;
    sample: Record<string, any>;
}

interface ExportData {
    version: string;
    timestamp: string;
    samples: any[];
    labels: any[];
    metadata: Record<string, any>;
}

// ===== Window Augmentation =====

declare global {
    interface Window {
        appState: typeof state;
        updateUI: typeof updateUI;
        addCustomLabel: typeof addCustomLabel;
        addPresetLabels: typeof addPresetLabels;
        removeActiveLabel: typeof removeActiveLabel;
        clearFingerStates: typeof clearFingerStates;
        removeCustomLabel: typeof removeCustomLabel;
        toggleCustomLabel: typeof toggleCustomLabel;
        deleteCustomLabelDef: typeof deleteCustomLabelDef;
        deleteLabel: (index: number) => void;
        copyLabelSegment: (index: number) => Promise<void>;
        copyLog: () => Promise<boolean>;
        copySession: () => Promise<void>;
        setHandPreviewMode: typeof setHandPreviewMode;
        togglePoseOption: typeof togglePoseOption;
        updateHandOrientationAlpha: typeof updateHandOrientationAlpha;
        getCurrentOrientation: () => { w: number; x: number; y: number; z: number } | null;
    }
}

// Export state for global access (used by inline functions in HTML)
window.appState = state;

// Export function to get current orientation (for calibration)
window.getCurrentOrientation = () => {
    if (typeof imuFusion !== 'undefined' && imuFusion) {
        return imuFusion.getQuaternion();
    }
    return null;
};

// DOM helper
const $ = (id: string): HTMLElement | null => document.getElementById(id);

// Initialize filters and fusion
const magFilter = new KalmanFilter3D({
    R: 0.1,  // Process noise
    Q: 1.0   // Measurement noise
});

// NOTE: With stable sensor data (Puck.accelOn fix in firmware v0.3.6),
// we can use standard AHRS parameters instead of jitter-compensating values
const imuFusion = new MadgwickAHRS({
    sampleFreq: 26,  // Match firmware accelOn rate (26Hz, not 50Hz)
    beta: 0.05       // Standard Madgwick gain (was 0.02 for jittery data)
});

// Pose estimation state
const poseState: PoseState = {
    enabled: false,
    currentPose: null,
    confidence: 0,
    updateCount: 0
};

// Hand visualization
let threeHandSkeleton: ThreeJSHandSkeleton | null = null;
let handPreviewMode: 'labels' | 'predictions' = 'labels';

// ML-based finger inference
let magneticFingerInference: MagneticFingerInference | null = null;
let lastFingerPrediction: FingerPrediction | null = null;

// Magnetic trajectory visualization
let magTrajectory: MagneticTrajectory | null = null;
let magTrajectoryPaused = false;

// Pose estimation options
const poseEstimationOptions: PoseEstimationOptions = {
    useCalibration: true,
    useFiltering: true,
    useOrientation: true,
    show3DOrientation: false,
    useParticleFilter: false,
    useMLInference: false,
    enableHandOrientation: true,
    smoothHandOrientation: true,
    handOrientationAlpha: 0.1
};

// ML Finger tracking inference
let fingerTrackingInference: any = null;
let mlModelLoaded = false;

// GitHub token (for upload functionality)
let ghToken: string | null = null;

// Upload method: 'proxy' (via API proxy) or 'github' (direct with PAT)
let uploadMethod: 'proxy' | 'github' = 'proxy';

// Live calibration UI update interval
let calibrationUIInterval: ReturnType<typeof setInterval> | null = null;
const CALIBRATION_UI_UPDATE_INTERVAL = 500;
let lastCalibrationSave = 0;
const CALIBRATION_SAVE_INTERVAL = 10000;  // Save calibration every 10 seconds when ready
let wasAutoHardIronReady = false;  // Track transition to ready state

// ===== Performance Optimizations =====

// DOM element cache for frequently accessed elements (avoid repeated getElementById)
const domCache: Record<string, HTMLElement | null> = {};
function $cached(id: string): HTMLElement | null {
    if (!(id in domCache)) {
        domCache[id] = document.getElementById(id);
    }
    return domCache[id];
}

// Track labels list state to avoid unnecessary rebuilds
let lastLabelsCount = -1;
let lastLabelsHash = '';

// requestAnimationFrame throttling for live display
let liveDisplayRAFPending = false;
let pendingLiveDisplayData: { raw: any; decorated: any } | null = null;

// Throttle calibration UI updates (use RAF instead of blocking setInterval)
let calibrationUIRAFPending = false;
let lastCalibrationUIUpdate = 0;
const CALIBRATION_UI_MIN_INTERVAL = 200;  // Max 5 updates/second instead of constant 500ms

/**
 * Update calibration confidence UI
 */
function updateCalibrationConfidenceUI(): void {
    const processor = getProcessor();
    if (!processor) return;

    const magCal = processor.getMagCalibration();
    if (!magCal) return;

    const calState = magCal.getState();

    const overall = Math.round(calState.confidence * 100);
    const meanResidual = calState.meanResidual;
    const earthMag = calState.earthMagnitude;
    const totalSamples = calState.totalSamples;

    // Use cached DOM elements to avoid repeated getElementById calls
    const overallEl = $cached('overallConfidence');
    const hardIronEl = $cached('hardIronConf');
    const earthFieldEl = $cached('earthFieldConf');
    const samplesEl = $cached('calibSamples');
    const fieldMagEl = $cached('earthFieldMag');
    const statusEl = $cached('calibStatus');
    const barEl = $cached('confidenceBar');
    const meanResidualEl = $cached('meanResidual');
    const residualQualityEl = $cached('residualQuality');

    // Auto hard iron progress UI elements (cached)
    const autoProgressEl = $cached('autoHardIronProgress');
    const autoBarEl = $cached('autoHardIronBar');
    const autoStatusEl = $cached('autoHardIronStatus');
    const softIronScaleEl = $cached('softIronScale');
    const autoRangesEl = $cached('autoHardIronRanges');

    // Update auto hard iron progress
    const autoProgress = Math.round(calState.autoHardIronProgress * 100);
    if (autoProgressEl) {
        autoProgressEl.textContent = `${autoProgress}%`;
        if (calState.autoHardIronReady) {
            autoProgressEl.style.color = 'var(--success)';
        } else if (autoProgress >= 50) {
            autoProgressEl.style.color = 'var(--warning)';
        } else {
            autoProgressEl.style.color = 'var(--fg-muted)';
        }
    }

    if (autoBarEl) {
        (autoBarEl as HTMLElement).style.width = `${autoProgress}%`;
        if (calState.autoHardIronReady) {
            (autoBarEl as HTMLElement).style.background = 'var(--success)';
        } else if (autoProgress >= 50) {
            (autoBarEl as HTMLElement).style.background = 'var(--warning)';
        } else {
            (autoBarEl as HTMLElement).style.background = 'var(--accent)';
        }
    }

    if (autoStatusEl) {
        if (calState.autoHardIronReady) {
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
        if (calState.autoHardIronReady) {
            const scale = calState.autoSoftIronScale;
            softIronScaleEl.textContent = `${scale.x.toFixed(2)}, ${scale.y.toFixed(2)}, ${scale.z.toFixed(2)}`;
        } else {
            softIronScaleEl.textContent = '--';
        }
    }

    // Auto hard iron ranges
    if (autoRangesEl) {
        const ranges = calState.autoHardIronRanges;
        if (ranges.x > 0 || ranges.y > 0 || ranges.z > 0) {
            autoRangesEl.textContent = `${ranges.x.toFixed(0)}, ${ranges.y.toFixed(0)}, ${ranges.z.toFixed(0)}`;
        } else {
            autoRangesEl.textContent = '--';
        }
    }

    if (overallEl) overallEl.textContent = `${overall}%`;
    if (hardIronEl) hardIronEl.textContent = calState.hardIronCalibrated ? '✓ Wizard' : (calState.autoHardIronReady ? '✓ Auto' : '--');
    if (earthFieldEl) earthFieldEl.textContent = calState.ready ? '✓ Auto' : 'Building...';
    if (samplesEl) samplesEl.textContent = totalSamples.toString();

    if (fieldMagEl) {
        fieldMagEl.textContent = earthMag > 0 ? `${earthMag.toFixed(1)} µT` : '-- µT';
    }

    if (meanResidualEl) {
        if (meanResidual !== undefined && meanResidual !== Infinity && !isNaN(meanResidual)) {
            meanResidualEl.textContent = `${meanResidual.toFixed(1)} µT`;
            if (meanResidual < 5) {
                meanResidualEl.style.color = 'var(--success)';
            } else if (meanResidual < 10) {
                meanResidualEl.style.color = 'var(--warning)';
            } else {
                meanResidualEl.style.color = 'var(--danger)';
            }
        } else {
            meanResidualEl.textContent = '-- µT';
            meanResidualEl.style.color = 'var(--fg-muted)';
        }
    }

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
            residualQualityEl.style.color = 'var(--danger)';
        }
    }

    if (barEl) {
        (barEl as HTMLElement).style.width = `${overall}%`;
        if (overall >= 70) {
            (barEl as HTMLElement).style.background = 'var(--success)';
        } else if (overall >= 40) {
            (barEl as HTMLElement).style.background = 'var(--warning)';
        } else {
            (barEl as HTMLElement).style.background = 'var(--danger)';
        }
    }

    if (statusEl) {
        const status: string[] = [];
        if (meanResidual !== undefined && meanResidual !== Infinity && !isNaN(meanResidual)) {
            if (meanResidual < 5) {
                status.push('✓ Excellent calibration');
            } else if (meanResidual < 10) {
                status.push('Good calibration');
            } else if (meanResidual < 15) {
                status.push('Moderate - keep rotating');
            } else {
                status.push('Poor - rotate more');
            }
        } else if (totalSamples < 50) {
            status.push('Collecting samples...');
        } else if (!calState.ready) {
            status.push('Building Earth field estimate...');
        } else {
            status.push('Calibrated (auto)');
        }
        statusEl.textContent = status.join(' | ');
    }

    if (overallEl) {
        if (overall >= 70) {
            overallEl.style.color = 'var(--success)';
        } else if (overall >= 40) {
            overallEl.style.color = 'var(--warning)';
        } else {
            overallEl.style.color = 'var(--danger)';
        }
    }

    // Update magnetometer calibration 3D visualization
    updateMagCalibrationVis(calState);

    // Save calibration to localStorage for next session bootstrap
    const now = performance.now();
    if (calState.autoHardIronReady) {
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

/**
 * Update magnetometer calibration 3D visualization
 * Shows point cloud, expected sphere, and axis coverage
 */
function updateMagCalibrationVis(calState: {
    autoHardIronEstimate?: { x: number; y: number; z: number };
    autoHardIronRanges?: { x: number; y: number; z: number };
}): void {
    if (!threeHandSkeleton || !threeHandSkeleton.isMagCalibrationEnabled()) return;

    // Get recent magnetometer samples from session data (last 300)
    // Session data is decorated telemetry with mx_ut, my_ut, mz_ut
    const recentSamples: MagSample[] = [];
    const startIdx = Math.max(0, state.sessionData.length - 300);
    for (let i = startIdx; i < state.sessionData.length; i++) {
        const sample = state.sessionData[i] as unknown as Record<string, number>;
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

    // Calculate axis progress
    // Expected range is 2x magnitude, we want progress toward that
    const location = state.geomagneticLocation || getDefaultLocation();
    // Default to 50 µT if no location (typical mid-latitude value)
    const expectedMag = location
        ? Math.sqrt(location.horizontal ** 2 + location.vertical ** 2)
        : 50;
    const targetRange = expectedMag * 1.5;  // Match the 1.5x threshold from calibration

    const ranges = calState.autoHardIronRanges || { x: 0, y: 0, z: 0 };
    const axisProgress: AxisProgress = {
        x: Math.min(1, ranges.x / targetRange),
        y: Math.min(1, ranges.y / targetRange),
        z: Math.min(1, ranges.z / targetRange)
    };

    // Bulk update the visualization
    threeHandSkeleton.updateMagCalibration({
        samples: recentSamples,
        hardIronOffset: hardIronOffset,
        axisProgress: axisProgress,
        expectedMagnitude: expectedMag
    });
}

/**
 * Update magnet detection UI display
 */
function updateMagnetDetectionUI(): void {
    const processor = getProcessor();
    if (!processor) return;

    const magnetState = processor.getMagnetState();
    if (!magnetState) return;

    // Use cached DOM elements
    const statusEl = $cached('magnetStatusValue');
    const confEl = $cached('magnetConfidenceValue');
    const barEl = $cached('magnetConfidenceBar');
    const residualEl = $cached('magnetAvgResidual');

    type MagnetStatus = 'none' | 'possible' | 'likely' | 'confirmed';
    const status = magnetState.status as MagnetStatus;

    if (statusEl) {
        const icons: Record<MagnetStatus, string> = { none: '○', possible: '◐', likely: '◑', confirmed: '🧲' };
        const labels: Record<MagnetStatus, string> = { none: 'No Magnets', possible: 'Possible', likely: 'Likely', confirmed: 'Confirmed' };
        const colors: Record<MagnetStatus, string> = { none: 'var(--fg-muted)', possible: 'var(--warning)', likely: '#5bc0de', confirmed: 'var(--success)' };

        statusEl.textContent = `${icons[status] || '?'} ${labels[status] || '--'}`;
        statusEl.style.color = colors[status] || 'var(--fg-muted)';
    }

    const confPct = Math.round(magnetState.confidence * 100);
    if (confEl) confEl.textContent = `${confPct}%`;

    if (barEl) {
        const colors: Record<MagnetStatus, string> = { none: 'var(--fg-muted)', possible: 'var(--warning)', likely: '#5bc0de', confirmed: 'var(--success)' };
        (barEl as HTMLElement).style.width = `${confPct}%`;
        (barEl as HTMLElement).style.background = colors[status] || 'var(--fg-muted)';
    }

    if (residualEl) {
        residualEl.textContent = magnetState.avgResidual > 0 ?
            `${magnetState.avgResidual.toFixed(1)} µT` : '-- µT';
    }
}

/**
 * Start periodic calibration UI updates using requestAnimationFrame
 * More efficient than setInterval as it:
 * - Pauses when tab is inactive
 * - Syncs with screen refresh rate
 * - Doesn't block main thread
 */
function startCalibrationUIUpdates(): void {
    if (calibrationUIInterval) return;

    // Use a flag to track if we should continue the loop
    calibrationUIInterval = 1 as any;  // Non-null to indicate running

    function calibrationUILoop(): void {
        if (!calibrationUIInterval) return;  // Stop if cleared

        const now = performance.now();

        // Only update if enough time has passed (throttle to ~5 updates/second)
        if (now - lastCalibrationUIUpdate >= CALIBRATION_UI_MIN_INTERVAL) {
            lastCalibrationUIUpdate = now;
            updateCalibrationConfidenceUI();
            updateMagnetDetectionUI();
        }

        // Schedule next frame
        requestAnimationFrame(calibrationUILoop);
    }

    requestAnimationFrame(calibrationUILoop);
    console.log('[GAMBIT Collector] Started calibration UI updates (RAF-based)');
}

/**
 * Stop periodic calibration UI updates
 */
function stopCalibrationUIUpdates(): void {
    if (calibrationUIInterval) {
        calibrationUIInterval = null;  // Setting to null stops the RAF loop
        console.log('[GAMBIT Collector] Stopped calibration UI updates');
    }
}

/**
 * Initialize application
 */
async function init(): Promise<void> {
    console.log('[GAMBIT Collector] Initializing...');

    initLogger($('log') as HTMLElement);

    const calInstance = initCalibration();

    setWizardDeps({
        state: state,
        startRecording: startRecording,
        $: $,
        log: log
    });

    setTelemetryDeps({
        calibrationInstance: calInstance,
        wizard: getWizardState(),
        calibrationBuffers: getCalibrationBuffers(),
        poseState: poseState,
        threeHandSkeleton: () => threeHandSkeleton,
        updatePoseEstimation: updatePoseEstimationFromMag,
        updateUI: updateUI,
        updateMagTrajectory: updateMagneticTrajectoryFromTelemetry,
        $: $
    });

    setConnectionCallbacks(updateUI, updateCalibrationStatus, startCalibrationUIUpdates, stopCalibrationUIUpdates);
    setRecordingCallbacks(updateUI, closeCurrentLabel);

    initConnectionUI($('connectBtn') as HTMLButtonElement);
    initRecordingUI({
        start: $('startBtn') as HTMLButtonElement,
        pause: $('pauseBtn') as HTMLButtonElement,
        stop: $('stopBtn') as HTMLButtonElement,
        clear: $('clearBtn') as HTMLButtonElement
    });
    initCalibrationUI();

    setCalibrationSessionCallback(storeCalibrationSessionData);

    initLabelManagement();
    initExport();
    initPoseEstimation();
    initWizard();
    initHandVisualization();
    initMagneticTrajectory();
    initCollapsibleSections();
    loadCustomLabels();

    // Load GitHub token
    try {
        ghToken = localStorage.getItem('gh_token');
        const tokenInput = $('ghToken') as HTMLInputElement | null;
        if (ghToken) {
            log('GitHub token loaded');
            if (tokenInput) {
                tokenInput.value = ghToken;
            }
        }
        if (tokenInput) {
            tokenInput.addEventListener('change', (e) => {
                ghToken = (e.target as HTMLInputElement).value;
                if (ghToken) {
                    localStorage.setItem('gh_token', ghToken);
                    log('GitHub token saved');
                } else {
                    localStorage.removeItem('gh_token');
                    log('GitHub token cleared');
                }
                updateUI();
            });
        }
    } catch (e) {
        console.warn('Failed to load GitHub token:', e);
    }

    // Initialize proxy upload secret
    try {
        const proxySecretInput = $('proxySecret') as HTMLInputElement | null;
        if (hasUploadSecret()) {
            log('Upload secret loaded');
            if (proxySecretInput) {
                proxySecretInput.value = '••••••••';
            }
        }
        if (proxySecretInput) {
            proxySecretInput.addEventListener('change', (e) => {
                const secret = (e.target as HTMLInputElement).value;
                if (secret && secret !== '••••••••') {
                    setUploadSecret(secret);
                    (e.target as HTMLInputElement).value = '••••••••';
                    log('Upload secret saved');
                } else if (!secret) {
                    setUploadSecret(null);
                    log('Upload secret cleared');
                }
                updateUI();
            });
        }

        const uploadMethodSelect = $('uploadMethod') as HTMLSelectElement | null;
        if (uploadMethodSelect) {
            uploadMethodSelect.value = uploadMethod;
            uploadMethodSelect.addEventListener('change', (e) => {
                uploadMethod = (e.target as HTMLSelectElement).value as 'proxy' | 'github';
                log(`Upload method: ${uploadMethod === 'proxy' ? 'API Proxy' : 'Direct GitHub'}`);
                updateUI();
            });
        }
    } catch (e) {
        console.warn('Failed to initialize upload:', e);
    }

    await initGeomagneticLocation();

    updateUI();
    updateCalibrationStatus();

    log('GAMBIT Collector ready');
}

/**
 * Initialize geomagnetic location
 */
async function initGeomagneticLocation(): Promise<void> {
    console.log('[GAMBIT] Initializing geomagnetic location...');
    log('Detecting geomagnetic location...');

    try {
        const location = await getBrowserLocation();
        state.geomagneticLocation = location.selected;

        const locationStr = formatLocation(location.selected);
        const fieldStr = `${location.selected.intensity.toFixed(1)} µT`;
        const declStr = `${location.selected.declination.toFixed(1)}°`;

        console.log(`[GAMBIT] ✓ Location detected: ${locationStr}`);
        console.log(`[GAMBIT] Magnetic field: ${fieldStr}, Declination: ${declStr}`);

        log(`Location: ${locationStr} (auto-detected, ±${location.accuracy.toFixed(0)}m)`);
        log(`Magnetic field: ${fieldStr}, Declination: ${declStr}`);
    } catch (error) {
        state.geomagneticLocation = getDefaultLocation();

        const locationStr = formatLocation(state.geomagneticLocation!);
        console.warn('[GAMBIT] Geolocation failed, using default:', (error as Error).message);

        log(`Location: ${locationStr} (default - ${(error as Error).message})`);
        log(`Magnetic field: ${state.geomagneticLocation!.intensity.toFixed(1)} µT, Declination: ${state.geomagneticLocation!.declination.toFixed(1)}°`);
    }
}

/**
 * Update UI state
 * Performance: Uses cached DOM elements and only rebuilds labels when changed
 */
function updateUI(): void {
    // Use cached DOM elements for frequently accessed elements
    const statusIndicator = $cached('statusIndicator');
    const connectBtn = $cached('connectBtn');
    const startBtn = $cached('startBtn') as HTMLButtonElement | null;
    const stopBtn = $cached('stopBtn') as HTMLButtonElement | null;
    const clearBtn = $cached('clearBtn') as HTMLButtonElement | null;
    const exportBtn = $cached('exportBtn') as HTMLButtonElement | null;
    const sampleCount = $cached('sampleCount');
    const progressFill = $cached('progressFill');
    const labelCount = $cached('labelCount');
    const labelsList = $cached('labelsList');

    if (statusIndicator) {
        if (state.recording && state.paused) {
            statusIndicator.className = 'status connected';
            statusIndicator.textContent = 'Paused';
        } else if (state.recording) {
            statusIndicator.className = 'status recording';
            statusIndicator.textContent = 'Recording...';
        } else if (state.connected) {
            statusIndicator.className = 'status connected';
            statusIndicator.textContent = 'Connected';
        } else {
            statusIndicator.className = 'status disconnected';
            statusIndicator.textContent = 'Disconnected';
        }
    }

    if (connectBtn) connectBtn.textContent = state.connected ? 'Disconnect' : 'Connect Device';
    if (startBtn) startBtn.disabled = !state.connected || state.recording;

    const pauseBtn = $cached('pauseBtn') as HTMLButtonElement | null;
    if (pauseBtn) {
        pauseBtn.disabled = !state.recording;
        pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
        pauseBtn.className = state.paused ? 'btn-success' : 'btn-warning';
    }

    if (stopBtn) stopBtn.disabled = !state.recording;
    if (clearBtn) clearBtn.disabled = state.sessionData.length === 0 || state.recording;
    if (exportBtn) exportBtn.disabled = state.sessionData.length === 0 || state.recording;

    const uploadBtn = $cached('uploadBtn') as HTMLButtonElement | null;
    if (uploadBtn) {
        const hasAuth = uploadMethod === 'proxy' ? hasUploadSecret() : !!ghToken;
        uploadBtn.disabled = state.sessionData.length === 0 || state.recording || !hasAuth;
        uploadBtn.title = hasAuth ? 'Upload session data' : `Configure ${uploadMethod === 'proxy' ? 'upload secret' : 'GitHub token'} first`;
    }

    const wizardBtn = $cached('wizardBtn') as HTMLButtonElement | null;
    if (wizardBtn) {
        wizardBtn.disabled = !state.connected;
        wizardBtn.title = state.connected ? 'Start guided data collection' : 'Connect device to enable';
    }

    const poseEstimationBtn = $cached('poseEstimationBtn') as HTMLButtonElement | null;
    if (poseEstimationBtn) {
        poseEstimationBtn.disabled = !state.connected;
        poseEstimationBtn.textContent = poseState.enabled ? '🎯 Disable Pose Tracking' : '🎯 Enable Pose Tracking';
        poseEstimationBtn.title = state.connected
            ? (poseState.enabled ? 'Disable similarity-based pose tracking' : 'Enable similarity-based pose tracking')
            : 'Connect device to enable';
    }

    if (sampleCount) sampleCount.textContent = state.sessionData.length.toString();
    if (progressFill) {
        (progressFill as HTMLElement).style.width = Math.min(100, state.sessionData.length / 15) + '%';
    }

    if (labelCount) labelCount.textContent = state.labels.length.toString();

    // PERFORMANCE: Only rebuild labels list when labels actually change
    // Compute a simple hash based on labels count and last label's end_sample
    const lastLabel = state.labels[state.labels.length - 1] as any;
    const currentLabelsHash = state.labels.length === 0
        ? 'empty'
        : `${state.labels.length}-${lastLabel?.end_sample ?? lastLabel?.endIndex ?? 0}`;

    if (labelsList && currentLabelsHash !== lastLabelsHash) {
        lastLabelsHash = currentLabelsHash;

        if (state.labels.length === 0) {
            labelsList.innerHTML = '<div style="color: #666; text-align: center; padding: 10px;">No labels yet.</div>';
        } else {
            labelsList.innerHTML = state.labels.map((l: any, i: number) => {
                // Handle both flat format (collector) and nested format (wizard)
                const startSample = l.start_sample ?? l.startIndex ?? 0;
                const endSample = l.end_sample ?? l.endIndex ?? 0;
                const labels = l.labels ?? l; // Use nested labels or flat structure

                const duration = ((endSample - startSample) / 50).toFixed(1);
                const tags: string[] = [];

                if (labels.pose) {
                    tags.push(`<span class="label-tag pose">${labels.pose}</span>`);
                }
                if (labels.fingers) {
                    const fs = ['thumb', 'index', 'middle', 'ring', 'pinky']
                        .map(f => {
                            const s = labels.fingers[f];
                            if (s === 'extended') return '0';
                            if (s === 'partial') return '1';
                            if (s === 'flexed') return '2';
                            return '?';
                        }).join('');
                    if (fs !== '?????') {
                        tags.push(`<span class="label-tag finger">${fs}</span>`);
                    }
                }
                if (labels.calibration && labels.calibration !== 'none') {
                    tags.push(`<span class="label-tag calibration">${labels.calibration}</span>`);
                }
                (labels.custom || []).forEach((c: string) => {
                    tags.push(`<span class="label-tag custom">${c}</span>`);
                });

                return `
                    <div class="label-item">
                        <span class="time-range">${startSample}-${endSample} (${duration}s)</span>
                        <div class="label-tags">${tags.join('')}</div>
                        <button class="btn-secondary btn-tiny" onclick="window.copyLabelSegment(${i})" title="Copy segment data">📋</button>
                        <button class="btn-danger btn-tiny" onclick="window.deleteLabel(${i})">×</button>
                    </div>
                `;
            }).join('');
        }
    }

    updateActiveLabelsDisplay();
}

/**
 * Update active labels display
 */
function updateActiveLabelsDisplay(): void {
    const display = $cached('activeLabelsDisplay');
    if (!display) return;

    const tags: string[] = [];

    if (state.currentLabels.pose) {
        tags.push(`<span class="active-label-chip label-type-pose" onclick="removeActiveLabel('pose')" style="cursor: pointer;" title="Click to remove">${state.currentLabels.pose} ×</span>`);
    }

    const fingerStates = ['thumb', 'index', 'middle', 'ring', 'pinky']
        .map(f => {
            const s = (state.currentLabels.fingers as any)[f];
            if (s === 'extended') return '0';
            if (s === 'partial') return '1';
            if (s === 'flexed') return '2';
            return '?';
        }).join('');
    if (fingerStates !== '?????') {
        tags.push(`<span class="active-label-chip label-type-finger" onclick="clearFingerStates()" style="cursor: pointer;" title="Click to clear">fingers:${fingerStates} ×</span>`);
    }

    if (state.currentLabels.motion !== 'static') {
        tags.push(`<span class="active-label-chip label-type-motion" onclick="removeActiveLabel('motion')" style="cursor: pointer;" title="Click to remove">motion:${state.currentLabels.motion} ×</span>`);
    }

    if (state.currentLabels.calibration !== 'none') {
        tags.push(`<span class="active-label-chip label-type-calibration" onclick="removeActiveLabel('calibration')" style="cursor: pointer;" title="Click to remove">calibration:${state.currentLabels.calibration} ×</span>`);
    }

    (state.currentLabels.custom || []).forEach((c: string) => {
        tags.push(`<span class="active-label-chip label-type-custom" onclick="removeCustomLabel('${c}')" style="cursor: pointer;" title="Click to remove">custom:${c} ×</span>`);
    });

    display.innerHTML = tags.length > 0 ? tags.join('') : '<span style="color: #666;">No labels active</span>';
}

/**
 * Remove active label
 */
function removeActiveLabel(type: string): void {
    if (type === 'pose') {
        state.currentLabels.pose = null;
        document.querySelectorAll('[data-pose]').forEach(b => b.classList.remove('active'));
        log('Pose cleared');
    } else if (type === 'motion') {
        state.currentLabels.motion = 'static';
        document.querySelectorAll('[data-motion]').forEach(b => b.classList.remove('active'));
        document.querySelector('[data-motion="static"]')?.classList.add('active');
        log('Motion reset to static');
    } else if (type === 'calibration') {
        state.currentLabels.calibration = 'none';
        document.querySelectorAll('[data-calibration]').forEach(b => b.classList.remove('active'));
        log('Calibration cleared');
    }
    onLabelsChanged();
}

/**
 * Clear all finger states
 */
function clearFingerStates(): void {
    state.currentLabels.fingers = {
        thumb: 'unknown',
        index: 'unknown',
        middle: 'unknown',
        ring: 'unknown',
        pinky: 'unknown'
    };
    document.querySelectorAll('.finger-state-btn').forEach(b => b.classList.remove('active'));
    log('Finger states cleared');
    onLabelsChanged();
}

/**
 * Remove custom label
 */
function removeCustomLabel(label: string): void {
    const index = state.currentLabels.custom.indexOf(label);
    if (index !== -1) {
        state.currentLabels.custom.splice(index, 1);
        log(`Custom label deactivated: ${label}`);
        onLabelsChanged();
        renderCustomLabels();
    }
}

/**
 * Close current label segment
 */
function closeCurrentLabel(): void {
    if (state.currentLabelStart !== null && state.sessionData.length > state.currentLabelStart) {
        const segment = {
            startIndex: state.currentLabelStart,
            endIndex: state.sessionData.length - 1,
            pose: state.currentLabels.pose ?? undefined,
            fingers: state.currentLabels.fingers,
            motion: state.currentLabels.motion,
            calibration: state.currentLabels.calibration,
            custom: [...state.currentLabels.custom]
        };
        state.labels.push(segment);
        log(`Label segment: ${state.currentLabelStart} - ${segment.endIndex}`);
    }
    state.currentLabelStart = state.sessionData.length;
}

/**
 * Initialize label management
 */
function initLabelManagement(): void {
    document.querySelectorAll('[data-pose]').forEach(btn => {
        btn.addEventListener('click', () => {
            const pose = (btn as HTMLElement).dataset.pose!;
            if (state.currentLabels.pose === pose) {
                state.currentLabels.pose = null;
            } else {
                state.currentLabels.pose = pose;
            }
            document.querySelectorAll('[data-pose]').forEach(b => b.classList.remove('active'));
            if (state.currentLabels.pose) {
                btn.classList.add('active');
            }
            onLabelsChanged();
            log(`Pose: ${state.currentLabels.pose || 'none'}`);
        });
    });

    document.querySelectorAll('.finger-state-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const finger = (btn as HTMLElement).dataset.finger!;
            const newState = (btn as HTMLElement).dataset.state!;
            (state.currentLabels.fingers as any)[finger] = newState;
            updateFingerButtons(finger);
            onLabelsChanged();
        });
    });

    document.querySelectorAll('[data-motion]').forEach(btn => {
        btn.addEventListener('click', () => {
            const motion = (btn as HTMLElement).dataset.motion!;
            state.currentLabels.motion = motion as 'static' | 'dynamic';
            document.querySelectorAll('[data-motion]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            onLabelsChanged();
            log(`Motion: ${motion}`);
        });
    });

    document.querySelectorAll('[data-calibration]').forEach(btn => {
        btn.addEventListener('click', () => {
            const cal = (btn as HTMLElement).dataset.calibration!;
            state.currentLabels.calibration = cal as 'none' | 'mag' | 'gyro';
            document.querySelectorAll('[data-calibration]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            onLabelsChanged();
            log(`Calibration: ${cal}`);
        });
    });

    const customLabelInput = $('customLabelInput') as HTMLInputElement | null;
    if (customLabelInput) {
        customLabelInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                addCustomLabel();
            }
        });
    }

    window.deleteLabel = (index: number) => {
        if (confirm('Delete this label segment?')) {
            state.labels.splice(index, 1);
            log(`Label segment ${index} deleted`);
            updateUI();
        }
    };

    window.copyLabelSegment = async (index: number) => {
        const segmentJSON = getSessionSegmentJSON(index);
        if (segmentJSON) {
            await copyToClipboard(segmentJSON, `Label segment ${index}`);
        } else {
            log(`Failed to get segment ${index}`);
        }
    };
}

/**
 * Update finger state buttons
 */
function updateFingerButtons(finger: string): void {
    const currentState = (state.currentLabels.fingers as any)[finger];
    document.querySelectorAll(`[data-finger="${finger}"]`).forEach(btn => {
        btn.classList.toggle('active', (btn as HTMLElement).dataset.state === currentState);
    });
}

/**
 * Handle labels changed
 */
function onLabelsChanged(): void {
    if (state.recording) {
        closeCurrentLabel();
    }
    updateActiveLabelsDisplay();
    updateHandVisualization();
}

/**
 * Add custom label
 */
function addCustomLabel(): void {
    const input = $('customLabelInput') as HTMLInputElement | null;
    if (!input) return;

    const value = input.value.trim();
    if (!value) return;

    if (!state.customLabelDefinitions.includes(value)) {
        state.customLabelDefinitions.push(value);
        saveCustomLabels();
        renderCustomLabels();
        log(`Custom label defined: ${value}`);
    }
    input.value = '';
}

/**
 * Toggle a custom label's active state
 */
function toggleCustomLabel(label: string): void {
    const index = state.currentLabels.custom.indexOf(label);
    if (index === -1) {
        state.currentLabels.custom.push(label);
        log(`Custom label activated: ${label}`);
    } else {
        state.currentLabels.custom.splice(index, 1);
        log(`Custom label deactivated: ${label}`);
    }
    onLabelsChanged();
    renderCustomLabels();
}

/**
 * Delete a custom label definition
 */
function deleteCustomLabelDef(label: string): void {
    const defIndex = state.customLabelDefinitions.indexOf(label);
    if (defIndex !== -1) {
        state.customLabelDefinitions.splice(defIndex, 1);
    }
    const activeIndex = state.currentLabels.custom.indexOf(label);
    if (activeIndex !== -1) {
        state.currentLabels.custom.splice(activeIndex, 1);
        onLabelsChanged();
    }
    saveCustomLabels();
    renderCustomLabels();
    log(`Custom label deleted: ${label}`);
}

/**
 * Save custom labels to localStorage
 */
function saveCustomLabels(): void {
    try {
        localStorage.setItem('gambit_custom_labels', JSON.stringify(state.customLabelDefinitions));
    } catch (e) {
        console.error('Failed to save custom labels:', e);
    }
}

/**
 * Add preset labels
 */
function addPresetLabels(preset: string): void {
    const presets: Record<string, string[]> = {
        phase1: ['warmup', 'baseline', 'initial'],
        phase2: ['calibrated', 'training', 'validation'],
        quality: ['good', 'noisy', 'drift'],
        transitions: ['entering', 'holding', 'exiting']
    };

    const labels = presets[preset] || [];
    let addedCount = 0;
    labels.forEach(label => {
        if (!state.customLabelDefinitions.includes(label)) {
            state.customLabelDefinitions.push(label);
            addedCount++;
        }
    });

    if (addedCount > 0) {
        saveCustomLabels();
        renderCustomLabels();
        log(`Added ${addedCount} preset label definitions: ${labels.join(', ')}`);
    }
}

/**
 * Load custom labels from localStorage
 */
function loadCustomLabels(): void {
    try {
        const saved = localStorage.getItem('gambit_custom_labels');
        if (saved) {
            state.customLabelDefinitions = JSON.parse(saved);
        }
    } catch (e) {
        console.error('Failed to load custom labels:', e);
    }
    renderCustomLabels();
}

/**
 * Render custom labels
 */
function renderCustomLabels(): void {
    const container = $('customLabelsList');
    if (!container) return;

    if (state.customLabelDefinitions.length === 0) {
        container.innerHTML = '<span style="color: var(--fg-muted); font-size: 11px;">No custom labels defined</span>';
        return;
    }

    container.innerHTML = state.customLabelDefinitions.map(label => {
        const isActive = state.currentLabels.custom.includes(label);
        return `
            <div class="custom-label-tag ${isActive ? 'active' : ''}"
                 onclick="window.toggleCustomLabel('${label}')"
                 title="Click to ${isActive ? 'deactivate' : 'activate'}">
                <span>${label}</span>
                <span class="remove" onclick="event.stopPropagation(); window.deleteCustomLabelDef('${label}')" title="Delete">×</span>
            </div>
        `;
    }).join('');
}

/**
 * Initialize pose estimation
 */
function initPoseEstimation(): void {
    const poseEstimationBtn = $('poseEstimationBtn');
    if (poseEstimationBtn) {
        poseEstimationBtn.addEventListener('click', togglePoseEstimation);
    }
}

/**
 * Toggle pose estimation
 */
function togglePoseEstimation(): void {
    if (!state.connected) {
        log('Error: Connect device first');
        return;
    }

    poseState.enabled = !poseState.enabled;

    if (poseState.enabled) {
        log('Pose tracking enabled');
        const statusSection = $('poseEstimationStatus');
        if (statusSection) {
            statusSection.style.display = 'block';
        }
        renderPoseEstimationOptions();
        updatePoseEstimationDisplay();
    } else {
        log('Pose tracking disabled');
        const statusSection = $('poseEstimationStatus');
        if (statusSection) {
            statusSection.style.display = 'none';
        }
        poseState.currentPose = null;
        poseState.confidence = 0;
        poseState.updateCount = 0;
        poseState.orientation = null;

        const orientationInfo = $('handOrientationInfo');
        if (orientationInfo) {
            orientationInfo.style.display = 'none';
        }
    }

    updateUI();
}

/**
 * Render pose estimation options UI
 */
function renderPoseEstimationOptions(): void {
    const statusSection = $('poseEstimationStatus');
    if (!statusSection) return;

    if ($('poseOptionsPanel')) return;

    const optionsHTML = `
        <div id="poseOptionsPanel" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);">
            <div style="font-size: 11px; color: var(--fg-muted); margin-bottom: 8px;">Data Processing:</div>
            <div style="display: flex; gap: 6px; font-size: 11px; margin-bottom: 12px;">
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="useCalibrationToggle" ${poseEstimationOptions.useCalibration ? 'checked' : ''}
                           onchange="togglePoseOption('useCalibration')" />
                    <span>Use Calibration</span>
                </label>
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="useFilteringToggle" ${poseEstimationOptions.useFiltering ? 'checked' : ''}
                           onchange="togglePoseOption('useFiltering')" />
                    <span>Use Kalman Filtering</span>
                </label>
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="useOrientationToggle" ${poseEstimationOptions.useOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('useOrientation')" />
                    <span>Use IMU Orientation</span>
                </label>
            </div>
            <div style="font-size: 11px; color: var(--fg-muted); margin-bottom: 8px;">3D Hand Orientation:</div>
            <div style="display: inline-flex; gap: 6px; font-size: 11px; margin-bottom: 12px;">
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="enableHandOrientationToggle" ${poseEstimationOptions.enableHandOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('enableHandOrientation')" />
                    <span>Enable Sensor Fusion</span>
                </label>
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="smoothHandOrientationToggle" ${poseEstimationOptions.smoothHandOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('smoothHandOrientation')" />
                    <span>Smooth Orientation</span>
                </label>
                <div style="display: inline-flex; flex-direction: column; align-items: center; gap: 8px; padding-left: 22px;">
                    <span style="color: var(--fg-muted);">Smoothing:</span>
                    <input type="range" id="handOrientationAlphaSlider" min="5" max="50" value="${poseEstimationOptions.handOrientationAlpha * 100}"
                           style="flex: 1; max-width: 100px;"
                           onchange="updateHandOrientationAlpha(this.value)" />
                    <span id="handOrientationAlphaValue" style="width: 30px; text-align: right;">${(poseEstimationOptions.handOrientationAlpha * 100).toFixed(0)}%</span>
                </div>
            </div>
            <div style="font-size: 11px; color: var(--fg-muted); margin-bottom: 8px;">Finger Tracking:</div>
            <div style="display: inline-flex; gap: 6px; font-size: 11px; margin-bottom: 12px;">
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="useMLInferenceToggle" ${poseEstimationOptions.useMLInference ? 'checked' : ''}
                           ${magneticFingerInference ? '' : 'disabled'}
                           onchange="togglePoseOption('useMLInference')" />
                    <span>ML-Based Inference</span>
                    <span id="mlInferenceStatus" style="font-size: 10px; color: ${magneticFingerInference ? 'var(--success)' : 'var(--fg-muted)'};">
                        ${magneticFingerInference ? '✓ Model loaded' : '(loading...)'}
                    </span>
                </label>
            </div>
        </div>
    `;

    statusSection.insertAdjacentHTML('beforeend', optionsHTML);
}

/**
 * Toggle pose option
 */
function togglePoseOption(option: keyof PoseEstimationOptions): void {
    (poseEstimationOptions as any)[option] = !(poseEstimationOptions as any)[option];

    if (option === 'show3DOrientation') {
        const cube = $('orientation3DCube');
        if (cube) {
            cube.style.display = poseEstimationOptions.show3DOrientation ? 'block' : 'none';
        }
    }

    log(`Pose option ${option}: ${(poseEstimationOptions as any)[option]}`);
}

/**
 * Update hand orientation alpha
 */
function updateHandOrientationAlpha(value: string): void {
    const alpha = parseInt(value, 10) / 100;
    poseEstimationOptions.handOrientationAlpha = alpha;

    const valueDisplay = $('handOrientationAlphaValue');
    if (valueDisplay) {
        valueDisplay.textContent = `${value}%`;
    }
}

/**
 * Update pose options UI (called when model loads)
 */
function updatePoseOptionsUI(): void {
    const mlToggle = $('useMLInferenceToggle') as HTMLInputElement | null;
    const mlStatus = $('mlInferenceStatus');

    if (mlToggle) {
        mlToggle.disabled = !magneticFingerInference;
        mlToggle.checked = poseEstimationOptions.useMLInference;
    }

    if (mlStatus) {
        if (magneticFingerInference) {
            mlStatus.textContent = '✓ Model loaded';
            mlStatus.style.color = 'var(--success)';
        } else {
            mlStatus.textContent = '(loading...)';
            mlStatus.style.color = 'var(--fg-muted)';
        }
    }
}

window.togglePoseOption = togglePoseOption;
window.updateHandOrientationAlpha = updateHandOrientationAlpha;

/**
 * Update pose estimation from mag data
 */
function updatePoseEstimationFromMag(data: PoseUpdateData): void {
    if (!poseState.enabled) return;

    const { magField, orientation, euler, sample } = data;
    poseState.updateCount++;

    let mx = magField.x;
    let my = magField.y;
    let mz = magField.z;

    // Get accelerometer data for ML inference
    let ax = 0, ay = 0, az = 1;
    if (sample) {
        ax = sample.ax_g ?? 0;
        ay = sample.ay_g ?? 0;
        az = sample.az_g ?? 1;

        if (poseEstimationOptions.useFiltering && sample.filtered_mx !== undefined) {
            mx = sample.filtered_mx;
            my = sample.filtered_my;
            mz = sample.filtered_mz;
        } else if (poseEstimationOptions.useCalibration) {
            if (sample.fused_mx !== undefined) {
                mx = sample.fused_mx;
                my = sample.fused_my;
                mz = sample.fused_mz;
            } else if (sample.calibrated_mx !== undefined) {
                mx = sample.calibrated_mx;
                my = sample.calibrated_my;
                mz = sample.calibrated_mz;
            }
        }
    }

    const strength = Math.sqrt(mx * mx + my * my + mz * mz);

    // Use ML inference if enabled and model is loaded
    if (poseEstimationOptions.useMLInference && magneticFingerInference) {
        const magneticSample: MagneticSample = {
            mx_ut: mx,
            my_ut: my,
            mz_ut: mz,
            ax_g: ax,
            ay_g: ay,
            az_g: az
        };

        // Run ML inference (async, updates lastFingerPrediction via callback)
        magneticFingerInference.predict(magneticSample).then(prediction => {
            if (prediction) {
                // Update pose state from ML prediction
                poseState.currentPose = {
                    thumb: prediction.states['thumb'] || 0,
                    index: prediction.states['index'] || 0,
                    middle: prediction.states['middle'] || 0,
                    ring: prediction.states['ring'] || 0,
                    pinky: prediction.states['pinky'] || 0
                };
                poseState.confidence = prediction.overallConfidence;

                // Update 3D hand visualization with finger curls
                if (threeHandSkeleton && handPreviewMode === 'predictions' && magneticFingerInference) {
                    const curls = magneticFingerInference.toCurls(prediction);
                    threeHandSkeleton.setFingerCurls(curls);
                }
            }
        });
    } else {
        // Fallback to threshold-based estimation
        poseState.confidence = Math.min(1.0, strength / 100);

        let baseThreshold = 20;
        let highThreshold = 60;

        if (poseEstimationOptions.useOrientation && euler) {
            const tiltFactor = Math.abs(Math.cos(euler.pitch * Math.PI / 180) * Math.cos(euler.roll * Math.PI / 180));
            baseThreshold *= tiltFactor;
            highThreshold *= tiltFactor;
            poseState.orientation = { roll: euler.roll, pitch: euler.pitch, yaw: euler.yaw };
        }

        if (strength < baseThreshold) {
            poseState.currentPose = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
        } else if (strength > highThreshold) {
            poseState.currentPose = { thumb: 2, index: 2, middle: 2, ring: 2, pinky: 2 };
        } else {
            const flex = (strength - baseThreshold) / (highThreshold - baseThreshold);
            const fingerState = Math.round(flex * 2);
            poseState.currentPose = { thumb: fingerState, index: fingerState, middle: fingerState, ring: fingerState, pinky: fingerState };
        }
    }

    // Use quaternion for hand orientation to avoid gimbal lock at steep pitches
    if (poseEstimationOptions.enableHandOrientation && threeHandSkeleton) {
        const quaternion = imuFusion.getQuaternion();
        if (quaternion) {
            threeHandSkeleton.updateQuaternion(quaternion);
        }
    }

    if (poseState.updateCount % 10 === 0) {
        updatePoseEstimationDisplay();
        if (handPreviewMode === 'predictions') {
            updateHandVisualization();
        }
    }
}

/**
 * Update pose estimation display
 */
function updatePoseEstimationDisplay(): void {
    const statusText = $('poseStatusText');
    const confidenceText = $('poseConfidenceText');
    const updatesText = $('poseUpdatesText');

    if (statusText) {
        statusText.textContent = poseState.enabled ? 'Active' : 'Disabled';
        statusText.style.color = poseState.enabled ? 'var(--success)' : 'var(--fg-muted)';
    }

    if (confidenceText) {
        confidenceText.textContent = Math.round(poseState.confidence * 100) + '%';
    }

    if (updatesText) {
        updatesText.textContent = poseState.updateCount.toString();
    }
}

/**
 * Initialize hand visualization
 */
function initHandVisualization(): void {
    const container = $('threeHandContainer');
    if (container && typeof THREE !== 'undefined') {
        try {
            threeHandSkeleton = new ThreeJSHandSkeleton(container as HTMLElement, {
                width: 400,
                height: 400,
                backgroundColor: 0xffffff,
                lerpFactor: 0.15,
                handedness: 'right'
            });

            threeHandSkeleton.setOrientationOffsets({
                roll: 180,
                pitch: 180,
                yaw: -180
            });

            // Initialize magnetometer calibration visualization
            // Get expected magnitude from current location
            const location = state.geomagneticLocation || getDefaultLocation();
            // Default to 50 µT if no location (typical mid-latitude value)
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
                scale: 0.02  // 50 µT = 1 unit in scene
            });

            log('Three.js hand skeleton initialized with mag calibration vis');
        } catch (err) {
            console.error('Failed to initialize Three.js hand skeleton:', err);
        }
    }

    // Initialize ML-based finger inference
    initMagneticFingerInference();

    updateHandVisualization();
}

/**
 * Initialize ML-based magnetic finger inference
 */
async function initMagneticFingerInference(): Promise<void> {
    try {
        log('Loading ML finger inference model...');
        magneticFingerInference = createMagneticFingerInference({
            smoothingAlpha: 0.4,
            onPrediction: (prediction: FingerPrediction) => {
                lastFingerPrediction = prediction;
            },
            onReady: () => {
                log('✓ ML finger inference model loaded');
                poseEstimationOptions.useMLInference = true;
                updatePoseOptionsUI();
            },
            onError: (error: Error) => {
                console.warn('ML inference model failed to load:', error.message);
                log('ML inference model unavailable');
            }
        });

        await magneticFingerInference.load();
    } catch (err) {
        console.warn('Failed to initialize ML finger inference:', err);
    }
}

/**
 * Initialize magnetic trajectory
 */
function initMagneticTrajectory(): void {
    const canvas = $('magTrajectoryCanvas') as HTMLCanvasElement | null;
    if (canvas) {
        magTrajectory = new MagneticTrajectory(canvas, {
            maxPoints: 200,
            trajectoryColor: '#4ecdc4',
            backgroundColor: '#ffffff',
            autoNormalize: true,
            showMarkers: true,
            showCube: true
        });

        const clearBtn = $('clearTrajectoryBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                magTrajectory!.clear();
                updateMagTrajectoryStats();
                log('Magnetic trajectory cleared');
            });
        }

        magTrajectory.addPoint(0, 0, 0);
        magTrajectory.addPoint(1, 1, 1);
        updateMagTrajectoryStats();

        const pauseBtn = $('pauseTrajectoryBtn') as HTMLButtonElement | null;
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => {
                magTrajectoryPaused = !magTrajectoryPaused;
                pauseBtn.textContent = magTrajectoryPaused ? 'Resume' : 'Pause';
                pauseBtn.className = magTrajectoryPaused ? 'btn-primary' : 'btn-secondary';
                log(`Magnetic trajectory ${magTrajectoryPaused ? 'paused' : 'resumed'}`);
            });
        }

        log('Magnetic trajectory visualizer initialized');
    }
}

/**
 * Update magnetic trajectory stats
 */
function updateMagTrajectoryStats(): void {
    const statsEl = $('trajStats');
    if (statsEl && magTrajectory) {
        const stats = magTrajectory.getStats();
        if (stats.count === 0) {
            statsEl.textContent = 'No data';
        } else {
            statsEl.textContent = `${stats.count} points | Magnitude: ${stats.magnitude.min.toFixed(2)} - ${stats.magnitude.max.toFixed(2)} μT (avg: ${stats.magnitude.avg.toFixed(2)} μT)`;
        }
    }
}

/**
 * Update magnetic trajectory from telemetry
 */
function updateMagneticTrajectoryFromTelemetry(data: { fused_mx?: number; fused_my?: number; fused_mz?: number }): void {
    if (!magTrajectory || magTrajectoryPaused) return;
    if (!data || data.fused_mx === undefined) return;

    magTrajectory.addPoint(data.fused_mx, data.fused_my!, data.fused_mz!);

    if ((magTrajectory as any).points.length % 50 === 0) {
        updateMagTrajectoryStats();
    }
}

/**
 * Set hand preview mode
 */
function setHandPreviewMode(mode: 'labels' | 'predictions'): void {
    handPreviewMode = mode;

    const labelsBtn = $('handModeLabels');
    const predictionsBtn = $('handModePredictions');
    const indicator = $('handModeIndicator');
    const description = $('handPreviewDescription');
    const predictionInfo = $('handPredictionInfo');

    if (labelsBtn && predictionsBtn) {
        if (mode === 'labels') {
            labelsBtn.className = 'btn-primary btn-small';
            predictionsBtn.className = 'btn-secondary btn-small';
            if (indicator) indicator.innerHTML = '📋 Showing: Manual Labels';
            if (description) description.textContent = 'Visual representation of manually selected finger states';
            if (predictionInfo) predictionInfo.style.display = 'none';
        } else {
            labelsBtn.className = 'btn-secondary btn-small';
            predictionsBtn.className = 'btn-primary btn-small';
            if (indicator) indicator.innerHTML = '🎯 Showing: Pose Predictions';
            if (description) description.textContent = 'Real-time pose estimation from magnetic field data';
            if (predictionInfo) predictionInfo.style.display = 'block';
        }
    }

    updateHandVisualization();
    log(`Hand preview mode: ${mode}`);
}

/**
 * Update hand visualization
 */
function updateHandVisualization(): void {
    let fingerStates: FingerStates;
    if (handPreviewMode === 'labels') {
        fingerStates = convertFingerLabelsToStates(state.currentLabels.fingers);
    } else {
        if (poseState.enabled && poseState.currentPose) {
            fingerStates = poseState.currentPose;
        } else {
            fingerStates = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
        }
    }
}

/**
 * Convert finger labels to states
 */
function convertFingerLabelsToStates(fingerLabels: any): FingerStates {
    const states: FingerStates = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
    const fingers: (keyof FingerStates)[] = ['thumb', 'index', 'middle', 'ring', 'pinky'];

    for (const finger of fingers) {
        const label = fingerLabels[finger];
        if (label === 'extended') {
            states[finger] = 0;
        } else if (label === 'partial') {
            states[finger] = 1;
        } else if (label === 'flexed') {
            states[finger] = 2;
        } else {
            states[finger] = 0;
        }
    }

    return states;
}

window.setHandPreviewMode = setHandPreviewMode;

/**
 * Initialize export functionality
 */
function initExport(): void {
    const exportBtn = $('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportData);
    }

    const uploadBtn = $('uploadBtn');
    if (uploadBtn) {
        uploadBtn.addEventListener('click', uploadSession);
    }
}

/**
 * Upload session
 */
async function uploadSession(): Promise<void> {
    if (uploadMethod === 'proxy') {
        await uploadToProxy();
    } else {
        await uploadToGitHub();
    }
}

/**
 * Upload to GitHub via API proxy (recommended - keeps token server-side)
 */
async function uploadToProxy(): Promise<void> {
    if (state.sessionData.length === 0) {
        log('No data to upload');
        return;
    }

    if (!hasUploadSecret()) {
        log('Error: No upload secret configured');
        return;
    }

    const uploadBtn = $('uploadBtn') as HTMLButtonElement;
    const originalText = uploadBtn.textContent;

    try {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';
        log('Uploading to GitHub (data branch)...');

        const data = buildExportData();

        const timestamp = new Date().toISOString().replace(/:/g, '_');
        const filename = `${timestamp}.json`;
        const content = JSON.stringify(data, null, 2);

        const result = await uploadSessionWithRetry({
            branch: 'data',
            filename,
            content,
            onProgress: (progress) => {
                uploadBtn.textContent = progress.message;
                log(`Upload: ${progress.message}`);
            }
        });

        log(`Uploaded: ${result.filename} (${(result.size / 1024).toFixed(1)} KB)`);
        log(`URL: ${result.url}`);

    } catch (e) {
        console.error('[GAMBIT] GitHub upload failed:', e);
        log(`Upload failed: ${(e as Error).message}`);
    } finally {
        uploadBtn.disabled = state.sessionData.length === 0 || state.recording || !hasUploadSecret();
        uploadBtn.textContent = originalText || 'Upload';
    }
}

/**
 * Store calibration session data
 */
function storeCalibrationSessionData(samples: CalibrationSample[], stepName: string, result: CalibrationResult): void {
    const calibrationLabels: Record<string, string> = {
        'HARD_IRON': 'hard_iron',
        'SOFT_IRON': 'soft_iron'
    };

    const calibrationLabel = calibrationLabels[stepName] || stepName.toLowerCase();
    const startIndex = state.sessionData.length;

    const timestamp = Date.now();
    samples.forEach((sample, i) => {
        state.sessionData.push({
            ...sample,
            timestamp: timestamp + (i * (1000 / 26)),
            calibration_step: stepName
        } as any);
    });

    const segment = {
        startIndex: startIndex,
        endIndex: state.sessionData.length - 1,
        pose: undefined,
        fingers: {
            thumb: 'unknown' as const,
            index: 'unknown' as const,
            middle: 'unknown' as const,
            ring: 'unknown' as const,
            pinky: 'unknown' as const
        },
        motion: 'static' as const,
        calibration: calibrationLabel === 'hard_iron' || calibrationLabel === 'soft_iron' ? 'mag' as const : 'none' as const,
        custom: ['calibration_session', `cal_${calibrationLabel}`]
    };

    state.labels.push(segment);

    log(`Calibration session stored: ${samples.length} samples for ${stepName}`);
    updateUI();
}

/**
 * Upload to GitHub directly (requires personal GitHub token)
 */
async function uploadToGitHub(): Promise<void> {
    if (state.sessionData.length === 0) {
        log('No data to upload');
        return;
    }

    if (!ghToken) {
        log('Error: No GitHub token configured');
        return;
    }

    const uploadBtn = $('uploadBtn') as HTMLButtonElement;
    const originalText = uploadBtn.textContent;

    try {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';
        log('Uploading to GitHub (data branch)...');

        const exportData = buildExportData();

        const timestamp = new Date().toISOString().replace(/:/g, '_');
        const filename = `${timestamp}.json`;
        const content = JSON.stringify(exportData, null, 2);

        const result = await uploadDirectToGitHub({
            token: ghToken,
            branch: 'data',
            path: `GAMBIT/${filename}`,
            content: content,
            message: `GAMBIT session: ${filename}`,
            onProgress: (progress) => {
                uploadBtn.textContent = progress.message;
                log(`Upload: ${progress.message}`);
            }
        });

        log(`Uploaded: ${result.filename} (${(result.size / 1024).toFixed(1)} KB)`);
        log(`URL: ${result.url}`);

    } catch (e) {
        console.error('[GAMBIT] Upload failed:', e);
        log(`Upload failed: ${(e as Error).message}`);
    } finally {
        const hasAuth = uploadMethod === 'proxy' ? hasUploadSecret() : !!ghToken;
        uploadBtn.disabled = state.sessionData.length === 0 || state.recording || !hasAuth;
        uploadBtn.textContent = originalText || 'Upload';
    }
}

/**
 * Build export data
 */
function buildExportData(options: Record<string, any> = {}): ExportData {
    const subjectId = ($('subjectId') as HTMLInputElement)?.value || 'unknown';
    const environment = ($('environment') as HTMLInputElement)?.value || 'unknown';
    const hand = ($('hand') as HTMLInputElement)?.value || 'unknown';
    const split = ($('split') as HTMLInputElement)?.value || 'train';
    const magnetConfig = ($('magnetConfig') as HTMLInputElement)?.value || 'none';
    const magnetType = ($('magnetType') as HTMLInputElement)?.value || 'unknown';
    const sessionNotes = ($('sessionNotes') as HTMLTextAreaElement)?.value || '';

    return {
        version: '2.1',
        timestamp: new Date().toISOString(),
        samples: state.sessionData,
        labels: state.labels,
        metadata: {
            sample_rate: 26,
            device: 'GAMBIT',
            firmware_version: state.firmwareVersion || 'unknown',
            calibration: calibrationInstance ? (calibrationInstance as any).toJSON() : null,
            location: exportLocationMetadata(state.geomagneticLocation),
            subject_id: subjectId,
            environment: environment,
            hand: hand,
            split: split,
            magnet_config: magnetConfig,
            magnet_type: magnetType,
            notes: sessionNotes,
            session_type: options.sessionType || 'recording',
            ...options
        }
    };
}

/**
 * Export data
 */
function exportData(): void {
    if (state.sessionData.length === 0) {
        log('No data to export');
        return;
    }

    const data = buildExportData();

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const timestamp = new Date().toISOString().replace(/:/g, '_');
    a.download = `${timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);

    log('Data exported');
}

/**
 * Initialize collapsible sections
 */
function initCollapsibleSections(): void {
    document.querySelectorAll('.collapsible').forEach(header => {
        header.addEventListener('click', () => {
            header.classList.toggle('collapsed');

            const section = header.parentElement;
            const contents = section?.querySelectorAll('.collapse-content');
            contents?.forEach(content => {
                content.classList.toggle('hidden');
            });

            const sectionId = section?.id;
            if (sectionId) {
                const isCollapsed = header.classList.contains('collapsed');
                localStorage.setItem(`section_${sectionId}_collapsed`, String(isCollapsed));
            }
        });

        const section = header.parentElement;
        const sectionId = section?.id;
        if (sectionId) {
            const savedState = localStorage.getItem(`section_${sectionId}_collapsed`);
            if (savedState === 'true') {
                header.classList.add('collapsed');
                const contents = section?.querySelectorAll('.collapse-content');
                contents?.forEach(content => {
                    content.classList.add('hidden');
                });
            }
        }
    });
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for debugging and onclick handlers
window.appState = state;
window.updateUI = updateUI;
window.addCustomLabel = addCustomLabel;
window.addPresetLabels = addPresetLabels;
window.removeActiveLabel = removeActiveLabel;
window.clearFingerStates = clearFingerStates;
window.removeCustomLabel = removeCustomLabel;
window.toggleCustomLabel = toggleCustomLabel;
window.deleteCustomLabelDef = deleteCustomLabelDef;

// Copy functions
window.copyLog = copyLogToClipboard;
window.copySession = async () => {
    const sessionJSON = getSessionJSON();
    await copyToClipboard(sessionJSON, 'Session data');
};
