/**
 * GAMBIT Collector Application
 * Main entry point that coordinates all modules
 */

import { state, resetSession } from './modules/state.js';
import { initLogger, log } from './modules/logger.js';
import {
    initCalibration,
    updateCalibrationStatus,
    initCalibrationUI,
    calibrationInstance,
    setStoreSessionCallback as setCalibrationSessionCallback
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
import { ThreeJSHandSkeleton } from './shared/threejs-hand-skeleton.js';
import {
    getBrowserLocation,
    getDefaultLocation,
    exportLocationMetadata,
    formatLocation,
    formatFieldData
} from './shared/geomagnetic-field.js';

// Export state for global access (used by inline functions in HTML)
window.appState = state;
window.log = log;

// Export function to get current orientation (for calibration)
window.getCurrentOrientation = () => {
    if (typeof imuFusion !== 'undefined' && imuFusion) {
        return imuFusion.getQuaternion();
    }
    return null;
};

// DOM helper
const $ = (id) => document.getElementById(id);

// Initialize filters and fusion
const magFilter = new KalmanFilter3D({
    processNoise: 0.1,
    measurementNoise: 1.0
});

// NOTE: With stable sensor data (Puck.accelOn fix in firmware v0.3.6),
// we can use standard AHRS parameters instead of jitter-compensating values
const imuFusion = new MadgwickAHRS({
    sampleFreq: 26,  // Match firmware accelOn rate (26Hz, not 50Hz)
    beta: 0.05       // Standard Madgwick gain (was 0.02 for jittery data)
});

// Pose estimation state
const poseState = {
    enabled: false,
    currentPose: null,
    confidence: 0,
    updateCount: 0
};

// Hand visualization
let threeHandSkeleton = null;
let handPreviewMode = 'labels'; // 'labels' or 'predictions'

// Magnetic trajectory visualization
let magTrajectory = null;
let magTrajectoryPaused = false;

// Pose estimation options
const poseEstimationOptions = {
    useCalibration: true,      // Use calibrated magnetic field data
    useFiltering: true,        // Use Kalman filtering on magnetic data
    useOrientation: true,      // Use IMU orientation for pose context
    show3DOrientation: false,  // Show 3D orientation cube
    useParticleFilter: false,  // Use ParticleFilter vs threshold-based estimation
    useMLInference: false,     // Use ML model for finger tracking (requires trained model)
    // 3D Hand orientation options
    enableHandOrientation: true,  // Apply sensor fusion to 3D hand orientation
    smoothHandOrientation: true,  // Apply low-pass filtering to hand orientation
    handOrientationAlpha: 0.1     // Smoothing factor (0-1, lower = smoother) - reduced from 0.15 for stability
};

// ML Finger tracking inference
let fingerTrackingInference = null;
let mlModelLoaded = false;

// GitHub token (for upload functionality)
let ghToken = null;

// Live calibration UI update interval
let calibrationUIInterval = null;
const CALIBRATION_UI_UPDATE_INTERVAL = 500; // Update every 500ms

/**
 * Update calibration confidence UI
 * Shows incremental calibration status and confidence metrics
 * Mirrors the implementation in index.html
 */
function updateCalibrationConfidenceUI() {
    // Get telemetry processor from telemetry handler
    const processor = getProcessor();
    if (!processor) return;

    // Get calibration state from TelemetryProcessor's UnifiedMagCalibration
    const magCal = processor.getMagCalibration();
    if (!magCal) return;

    const calState = magCal.getState();

    const overall = Math.round(calState.confidence * 100);
    const meanResidual = calState.meanResidual;
    const earthMag = calState.earthMagnitude;
    const totalSamples = calState.totalSamples;

    // Update text displays
    const overallEl = $('overallConfidence');
    const hardIronEl = $('hardIronConf');
    const earthFieldEl = $('earthFieldConf');
    const samplesEl = $('calibSamples');
    const fieldMagEl = $('earthFieldMag');
    const statusEl = $('calibStatus');
    const barEl = $('confidenceBar');
    const meanResidualEl = $('meanResidual');
    const residualQualityEl = $('residualQuality');

    if (overallEl) overallEl.textContent = `${overall}%`;
    if (hardIronEl) hardIronEl.textContent = calState.hardIronEnabled ? '✓' : '--';
    if (earthFieldEl) earthFieldEl.textContent = calState.ready ? '✓ Auto' : 'Building...';
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
                meanResidualEl.style.color = 'var(--danger)';
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
            residualQualityEl.style.color = 'var(--danger)';
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
            barEl.style.background = 'var(--danger)';
        }
    }

    // Update status text - now based on residual
    if (statusEl) {
        const status = [];
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

    // Update overall confidence color
    if (overallEl) {
        if (overall >= 70) {
            overallEl.style.color = 'var(--success)';
        } else if (overall >= 40) {
            overallEl.style.color = 'var(--warning)';
        } else {
            overallEl.style.color = 'var(--danger)';
        }
    }
}

/**
 * Update magnet detection UI display
 * Shows finger magnet detection status and confidence
 * Mirrors the implementation in index.html
 */
function updateMagnetDetectionUI() {
    // Get telemetry processor from telemetry handler
    const processor = getProcessor();
    if (!processor) return;

    // Get magnet state from telemetry processor
    const magnetState = processor.getMagnetState();
    if (!magnetState) return;

    const statusEl = $('magnetStatusValue');
    const confEl = $('magnetConfidenceValue');
    const barEl = $('magnetConfidenceBar');
    const residualEl = $('magnetAvgResidual');

    // Update status text with icon
    if (statusEl) {
        const icons = { none: '○', possible: '◐', likely: '◑', confirmed: '🧲' };
        const labels = { none: 'No Magnets', possible: 'Possible', likely: 'Likely', confirmed: 'Confirmed' };
        const colors = { none: 'var(--fg-muted)', possible: 'var(--warning)', likely: '#5bc0de', confirmed: 'var(--success)' };

        statusEl.textContent = `${icons[magnetState.status] || '?'} ${labels[magnetState.status] || '--'}`;
        statusEl.style.color = colors[magnetState.status] || 'var(--fg-muted)';
    }

    // Update confidence
    const confPct = Math.round(magnetState.confidence * 100);
    if (confEl) confEl.textContent = `${confPct}%`;

    // Update bar
    if (barEl) {
        barEl.style.width = `${confPct}%`;
        const colors = { none: 'var(--fg-muted)', possible: 'var(--warning)', likely: '#5bc0de', confirmed: 'var(--success)' };
        barEl.style.background = colors[magnetState.status] || 'var(--fg-muted)';
    }

    // Update residual
    if (residualEl) {
        residualEl.textContent = magnetState.avgResidual > 0 ?
            `${magnetState.avgResidual.toFixed(1)} µT` : '-- µT';
    }
}

/**
 * Start periodic calibration UI updates
 */
function startCalibrationUIUpdates() {
    if (calibrationUIInterval) return; // Already running

    calibrationUIInterval = setInterval(() => {
        updateCalibrationConfidenceUI();
        updateMagnetDetectionUI();
    }, CALIBRATION_UI_UPDATE_INTERVAL);

    console.log('[GAMBIT Collector] Started calibration UI updates');
}

/**
 * Stop periodic calibration UI updates
 */
function stopCalibrationUIUpdates() {
    if (calibrationUIInterval) {
        clearInterval(calibrationUIInterval);
        calibrationUIInterval = null;
        console.log('[GAMBIT Collector] Stopped calibration UI updates');
    }
}

/**
 * Initialize application
 */
async function init() {
    console.log('[GAMBIT Collector] Initializing...');

    // Initialize logger
    initLogger($('log'));

    // Initialize calibration
    const calInstance = initCalibration();

    // Set wizard dependencies
    setWizardDeps({
        state: state,
        startRecording: startRecording,
        $: $,
        log: log
    });

    // Set telemetry dependencies
    setTelemetryDeps({
        calibrationInstance: calInstance,
        magFilter: magFilter,
        imuFusion: imuFusion,
        wizard: getWizardState(),
        calibrationBuffers: getCalibrationBuffers(),
        poseState: poseState,
        threeHandSkeleton: () => threeHandSkeleton,
        updatePoseEstimation: updatePoseEstimationFromMag,
        updateUI: updateUI,
        updateMagTrajectory: updateMagneticTrajectoryFromTelemetry,
        updateMagTrajectoryStats: updateMagTrajectoryStats,
        $: $
    });

    // Set connection callbacks (including calibration UI start/stop)
    setConnectionCallbacks(updateUI, updateCalibrationStatus, startCalibrationUIUpdates, stopCalibrationUIUpdates);

    // Set recording callbacks
    setRecordingCallbacks(updateUI, closeCurrentLabel);

    // Initialize UI controls
    initConnectionUI($('connectBtn'));
    initRecordingUI({
        start: $('startBtn'),
        pause: $('pauseBtn'),
        stop: $('stopBtn'),
        clear: $('clearBtn')
    });
    initCalibrationUI();

    // Set up calibration session storage callback
    setCalibrationSessionCallback(storeCalibrationSessionData);

    // Initialize label management
    initLabelManagement();

    // Initialize export functionality
    initExport();

    // Initialize pose estimation functionality
    initPoseEstimation();

    // Initialize wizard functionality
    initWizard();

    // Initialize hand visualization
    initHandVisualization();

    // Initialize magnetic trajectory visualization
    initMagneticTrajectory();

    // Initialize collapsible sections
    initCollapsibleSections();

    // Load custom labels from localStorage
    loadCustomLabels();

    // Try to load GitHub token
    try {
        ghToken = localStorage.getItem('gh_token');
        const tokenInput = $('ghToken');
        if (ghToken) {
            log('GitHub token loaded');
            // Populate the token input if it exists
            if (tokenInput) {
                tokenInput.value = ghToken;
            }
        }
        // Add change handler for token input
        if (tokenInput) {
            tokenInput.addEventListener('change', (e) => {
                ghToken = e.target.value;
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

    // Initialize geomagnetic location
    await initGeomagneticLocation();

    // Initial UI update
    updateUI();
    updateCalibrationStatus();

    log('GAMBIT Collector ready');
}

/**
 * Initialize geomagnetic location
 * Try browser geolocation first, fall back to Edinburgh default
 */
async function initGeomagneticLocation() {
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
        // Fall back to default location (Edinburgh)
        state.geomagneticLocation = getDefaultLocation();

        const locationStr = formatLocation(state.geomagneticLocation);
        console.warn('[GAMBIT] Geolocation failed, using default:', error.message);

        log(`Location: ${locationStr} (default - ${error.message})`);
        log(`Magnetic field: ${state.geomagneticLocation.intensity.toFixed(1)} µT, Declination: ${state.geomagneticLocation.declination.toFixed(1)}°`);
    }
}

/**
 * Update UI state
 */
function updateUI() {
    const statusIndicator = $('statusIndicator');
    const connectBtn = $('connectBtn');
    const startBtn = $('startBtn');
    const stopBtn = $('stopBtn');
    const clearBtn = $('clearBtn');
    const exportBtn = $('exportBtn');
    const sampleCount = $('sampleCount');
    const progressFill = $('progressFill');
    const labelCount = $('labelCount');
    const labelsList = $('labelsList');

    // Status indicator
    if (statusIndicator) {
        if (state.recording && state.paused) {
            statusIndicator.className = 'status connected';  // Yellow-ish for paused
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

    // Buttons
    if (connectBtn) connectBtn.textContent = state.connected ? 'Disconnect' : 'Connect Device';
    if (startBtn) startBtn.disabled = !state.connected || state.recording;

    // Pause button - only enabled during recording
    const pauseBtn = $('pauseBtn');
    if (pauseBtn) {
        pauseBtn.disabled = !state.recording;
        pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
        pauseBtn.className = state.paused ? 'btn-success' : 'btn-warning';
    }

    if (stopBtn) stopBtn.disabled = !state.recording;
    if (clearBtn) clearBtn.disabled = state.sessionData.length === 0 || state.recording;
    if (exportBtn) exportBtn.disabled = state.sessionData.length === 0 || state.recording;

    // Upload button
    const uploadBtn = $('uploadBtn');
    if (uploadBtn) {
        uploadBtn.disabled = state.sessionData.length === 0 || state.recording || !ghToken;
    }

    // Wizard button - enable when connected
    const wizardBtn = $('wizardBtn');
    if (wizardBtn) {
        wizardBtn.disabled = !state.connected;
        wizardBtn.title = state.connected ? 'Start guided data collection' : 'Connect device to enable';
    }

    // Pose estimation button - enable when connected
    const poseEstimationBtn = $('poseEstimationBtn');
    if (poseEstimationBtn) {
        poseEstimationBtn.disabled = !state.connected;
        // Update button text based on pose state
        poseEstimationBtn.textContent = poseState.enabled ? '🎯 Disable Pose Tracking' : '🎯 Enable Pose Tracking';
        poseEstimationBtn.title = state.connected
            ? (poseState.enabled ? 'Disable similarity-based pose tracking' : 'Enable similarity-based pose tracking')
            : 'Connect device to enable';
    }

    // Sample count
    if (sampleCount) sampleCount.textContent = state.sessionData.length;
    if (progressFill) {
        progressFill.style.width = Math.min(100, state.sessionData.length / 15) + '%';
    }

    // Labels
    if (labelCount) labelCount.textContent = state.labels.length;
    if (labelsList) {
        if (state.labels.length === 0) {
            labelsList.innerHTML = '<div style="color: #666; text-align: center; padding: 10px;">No labels yet.</div>';
        } else {
            labelsList.innerHTML = state.labels.map((l, i) => {
                const duration = ((l.end_sample - l.start_sample) / 50).toFixed(1);
                const tags = [];

                if (l.labels.pose) {
                    tags.push(`<span class="label-tag pose">${l.labels.pose}</span>`);
                }
                if (l.labels.fingers) {
                    const fs = ['thumb', 'index', 'middle', 'ring', 'pinky']
                        .map(f => {
                            const s = l.labels.fingers[f];
                            if (s === 'extended') return '0';
                            if (s === 'partial') return '1';
                            if (s === 'flexed') return '2';
                            return '?';
                        }).join('');
                    if (fs !== '?????') {
                        tags.push(`<span class="label-tag finger">${fs}</span>`);
                    }
                }
                if (l.labels.calibration && l.labels.calibration !== 'none') {
                    tags.push(`<span class="label-tag calibration">${l.labels.calibration}</span>`);
                }
                (l.labels.custom || []).forEach(c => {
                    tags.push(`<span class="label-tag custom">${c}</span>`);
                });

                return `
                    <div class="label-item">
                        <span class="time-range">${l.start_sample}-${l.end_sample} (${duration}s)</span>
                        <div class="label-tags">${tags.join('')}</div>
                        <button class="btn-danger btn-tiny" onclick="window.deleteLabel(${i})">×</button>
                    </div>
                `;
            }).join('');
        }
    }

    // Update active labels display
    updateActiveLabelsDisplay();
}

/**
 * Update active labels display
 */
function updateActiveLabelsDisplay() {
    const display = $('activeLabelsDisplay');
    if (!display) return;

    const tags = [];

    if (state.currentLabels.pose) {
        tags.push(`<span class="active-label-chip label-type-pose" onclick="removeActiveLabel('pose')" style="cursor: pointer;" title="Click to remove">${state.currentLabels.pose} ×</span>`);
    }

    const fingerStates = ['thumb', 'index', 'middle', 'ring', 'pinky']
        .map(f => {
            const s = state.currentLabels.fingers[f];
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

    (state.currentLabels.custom || []).forEach(c => {
        tags.push(`<span class="active-label-chip label-type-custom" onclick="removeCustomLabel('${c}')" style="cursor: pointer;" title="Click to remove">custom:${c} ×</span>`);
    });

    display.innerHTML = tags.length > 0 ? tags.join('') : '<span style="color: #666;">No labels active</span>';
}

/**
 * Remove active label
 */
function removeActiveLabel(type) {
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
function clearFingerStates() {
    state.currentLabels.fingers = {
        thumb: null,
        index: null,
        middle: null,
        ring: null,
        pinky: null
    };
    document.querySelectorAll('.finger-state-btn').forEach(b => b.classList.remove('active'));
    log('Finger states cleared');
    onLabelsChanged();
}

/**
 * Remove custom label
 */
function removeCustomLabel(label) {
    const index = state.currentLabels.custom.indexOf(label);
    if (index !== -1) {
        state.currentLabels.custom.splice(index, 1);
        log(`Custom label deactivated: ${label}`);
        onLabelsChanged();
        renderCustomLabels();  // Update the custom labels list to reflect active state
    }
}

/**
 * Close current label segment
 */
function closeCurrentLabel() {
    if (state.currentLabelStart !== null && state.sessionData.length > state.currentLabelStart) {
        const segment = {
            start_sample: state.currentLabelStart,
            end_sample: state.sessionData.length - 1,
            labels: JSON.parse(JSON.stringify(state.currentLabels))
        };
        state.labels.push(segment);
        log(`Label segment: ${state.currentLabelStart} - ${segment.end_sample}`);
    }
    state.currentLabelStart = state.sessionData.length;
}

/**
 * Initialize label management
 */
function initLabelManagement() {
    // Pose selection
    document.querySelectorAll('[data-pose]').forEach(btn => {
        btn.addEventListener('click', () => {
            const pose = btn.dataset.pose;
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

    // Finger state selection
    document.querySelectorAll('.finger-state-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const finger = btn.dataset.finger;
            const newState = btn.dataset.state;
            state.currentLabels.fingers[finger] = newState;
            updateFingerButtons(finger);
            onLabelsChanged();
        });
    });

    // Motion state
    document.querySelectorAll('[data-motion]').forEach(btn => {
        btn.addEventListener('click', () => {
            const motion = btn.dataset.motion;
            state.currentLabels.motion = motion;
            document.querySelectorAll('[data-motion]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            onLabelsChanged();
            log(`Motion: ${motion}`);
        });
    });

    // Calibration state
    document.querySelectorAll('[data-calibration]').forEach(btn => {
        btn.addEventListener('click', () => {
            const cal = btn.dataset.calibration;
            state.currentLabels.calibration = cal;
            document.querySelectorAll('[data-calibration]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            onLabelsChanged();
            log(`Calibration: ${cal}`);
        });
    });

    // Custom label input
    const customLabelInput = $('customLabelInput');
    if (customLabelInput) {
        customLabelInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                addCustomLabel();
            }
        });
    }

    // Delete label function (global for onclick)
    window.deleteLabel = (index) => {
        if (confirm('Delete this label segment?')) {
            state.labels.splice(index, 1);
            log(`Label segment ${index} deleted`);
            updateUI();
        }
    };
}

/**
 * Update finger state buttons
 */
function updateFingerButtons(finger) {
    const currentState = state.currentLabels.fingers[finger];
    document.querySelectorAll(`[data-finger="${finger}"]`).forEach(btn => {
        btn.classList.toggle('active', btn.dataset.state === currentState);
    });
}

/**
 * Handle labels changed
 */
function onLabelsChanged() {
    if (state.recording) {
        closeCurrentLabel();
    }
    updateActiveLabelsDisplay();
    updateHandVisualization();
}

/**
 * Add custom label to definitions (but don't activate it)
 */
function addCustomLabel() {
    const input = $('customLabelInput');
    if (!input) return;

    const value = input.value.trim();
    if (!value) return;

    // Add to definitions if not already present
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
 * @param {string} label - The label to toggle
 */
function toggleCustomLabel(label) {
    const index = state.currentLabels.custom.indexOf(label);
    if (index === -1) {
        // Activate the label
        state.currentLabels.custom.push(label);
        log(`Custom label activated: ${label}`);
    } else {
        // Deactivate the label
        state.currentLabels.custom.splice(index, 1);
        log(`Custom label deactivated: ${label}`);
    }
    onLabelsChanged();
    renderCustomLabels();
}

/**
 * Delete a custom label definition
 * @param {string} label - The label to delete
 */
function deleteCustomLabelDef(label) {
    // Remove from definitions
    const defIndex = state.customLabelDefinitions.indexOf(label);
    if (defIndex !== -1) {
        state.customLabelDefinitions.splice(defIndex, 1);
    }
    // Also remove from active if present
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
 * Save custom label definitions to localStorage
 */
function saveCustomLabels() {
    try {
        localStorage.setItem('gambit_custom_labels', JSON.stringify(state.customLabelDefinitions));
    } catch (e) {
        console.error('Failed to save custom labels:', e);
    }
}

/**
 * Add preset labels to definitions (doesn't activate them)
 */
function addPresetLabels(preset) {
    const presets = {
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
function loadCustomLabels() {
    try {
        const saved = localStorage.getItem('gambit_custom_labels');
        if (saved) {
            state.customLabelDefinitions = JSON.parse(saved);
        }
    } catch (e) {
        console.error('Failed to load custom labels:', e);
    }
    // Always render (shows empty state if no labels)
    renderCustomLabels();
}

/**
 * Render custom labels with toggle and delete controls
 */
function renderCustomLabels() {
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
 * Initialize pose estimation functionality
 */
function initPoseEstimation() {
    const poseEstimationBtn = $('poseEstimationBtn');
    if (poseEstimationBtn) {
        poseEstimationBtn.addEventListener('click', togglePoseEstimation);
    }
}

/**
 * Toggle pose estimation on/off
 */
function togglePoseEstimation() {
    if (!state.connected) {
        log('Error: Connect device first');
        return;
    }

    poseState.enabled = !poseState.enabled;

    if (poseState.enabled) {
        log('Pose tracking enabled');
        // Show pose estimation status section
        const statusSection = $('poseEstimationStatus');
        if (statusSection) {
            statusSection.style.display = 'block';
        }
        renderPoseEstimationOptions();
        updatePoseEstimationDisplay();

    } else {
        log('Pose tracking disabled');
        // Hide pose estimation status section
        const statusSection = $('poseEstimationStatus');
        if (statusSection) {
            statusSection.style.display = 'none';
        }
        // Reset pose state
        poseState.currentPose = null;
        poseState.confidence = 0;
        poseState.updateCount = 0;
        poseState.orientation = null;

        // Hide orientation display
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
function renderPoseEstimationOptions() {
    const statusSection = $('poseEstimationStatus');
    if (!statusSection) return;

    // Check if options already rendered
    if ($('poseOptionsPanel')) return;

    const optionsHTML = `
        <div id="poseOptionsPanel" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);">
            <div style="font-size: 11px; color: var(--fg-muted); margin-bottom: 8px;">Data Processing:</div>
            <div style="display: flex; gap: 6px; font-size: 11px; margin-bottom: 12px;">
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="useCalibrationToggle" ${poseEstimationOptions.useCalibration ? 'checked' : ''}
                           onchange="togglePoseOption('useCalibration')" />
                    <span>Use Calibration (iron + earth field correction)</span>
                </label>
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="useFilteringToggle" ${poseEstimationOptions.useFiltering ? 'checked' : ''}
                           onchange="togglePoseOption('useFiltering')" />
                    <span>Use Kalman Filtering (noise reduction)</span>
                </label>
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="useOrientationToggle" ${poseEstimationOptions.useOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('useOrientation')" />
                    <span>Use IMU Orientation (sensor fusion context)</span>
                </label>
            </div>

            <div style="font-size: 11px; color: var(--fg-muted); margin-bottom: 8px;">3D Hand Orientation:</div>
            <div style="display: inline-flex; gap: 6px; font-size: 11px; margin-bottom: 12px;">
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="enableHandOrientationToggle" ${poseEstimationOptions.enableHandOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('enableHandOrientation')" />
                    <span>Enable Sensor Fusion (orient hand from IMU)</span>
                </label>
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="smoothHandOrientationToggle" ${poseEstimationOptions.smoothHandOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('smoothHandOrientation')" />
                    <span>Smooth Orientation (low-pass filtering)</span>
                </label>
                <div style="display: inline-flex; flex-direction: column;  align-items: center; gap: 8px; padding-left: 22px;">
                    <span style="color: var(--fg-muted);">Smoothing:</span>
                    <input type="range" id="handOrientationAlphaSlider" min="5" max="50" value="${poseEstimationOptions.handOrientationAlpha * 100}"
                           style="flex: 1; max-width: 100px;"
                           onchange="updateHandOrientationAlpha(this.value)" />
                    <span id="handOrientationAlphaValue" style="width: 30px; text-align: right;">${(poseEstimationOptions.handOrientationAlpha * 100).toFixed(0)}%</span>
                </div>
            </div>

            <div style="font-size: 11px; color: var(--fg-muted); margin-bottom: 8px;">Debug:</div>
            <div style="display: inline-flex; gap: 6px; font-size: 11px;">
                <label style="display: inline-flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input style="flex-basis: 0;" type="checkbox" id="show3DOrientationToggle" ${poseEstimationOptions.show3DOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('show3DOrientation')" />
                    <span>Show 3D Orientation Cube</span>
                </label>
            </div>
            <div id="orientation3DCube" style="display: none; margin-top: 12px; perspective: 500px; height: 80px;">
                <div id="orientationCube" class="cube" style="margin: 0 auto; width: 60px; height: 60px; transform-style: preserve-3d; transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg);">
                    <div class="face front" style="background: rgba(90,90,90,.7); width: 100%; height: 100%; position: absolute; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; transform: translateZ(30px);">F</div>
                    <div class="face back" style="background: rgba(0,210,0,.7); width: 100%; height: 100%; position: absolute; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; transform: rotateY(180deg) translateZ(30px);">B</div>
                    <div class="face right" style="background: rgba(210,0,0,.7); width: 100%; height: 100%; position: absolute; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; transform: rotateY(90deg) translateZ(30px);">R</div>
                    <div class="face left" style="background: rgba(0,0,210,.7); width: 100%; height: 100%; position: absolute; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; transform: rotateY(-90deg) translateZ(30px);">L</div>
                    <div class="face top" style="background: rgba(210,210,0,.7); width: 100%; height: 100%; position: absolute; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; transform: rotateX(90deg) translateZ(30px);">T</div>
                    <div class="face bottom" style="background: rgba(210,0,210,.7); width: 100%; height: 100%; position: absolute; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; transform: rotateX(-90deg) translateZ(30px);">Bo</div>
                </div>
                <div id="orientationAngles" style="text-align: center; margin-top: 8px; font-size: 10px; color: var(--fg-muted); font-family: monospace;"></div>
            </div>
        </div>
    `;

    statusSection.insertAdjacentHTML('beforeend', optionsHTML);
}

/**
 * Toggle pose estimation option
 */
function togglePoseOption(option) {
    poseEstimationOptions[option] = !poseEstimationOptions[option];

    // Handle specific option changes
    switch (option) {
        case 'show3DOrientation':
            const cube = $('orientation3DCube');
            if (cube) {
                cube.style.display = poseEstimationOptions.show3DOrientation ? 'block' : 'none';
            }
            break;

    }

    log(`Pose option ${option}: ${poseEstimationOptions[option]}`);
}

/**
 * Update hand orientation smoothing alpha
 */
function updateHandOrientationAlpha(value) {
    const alpha = value / 100;
    poseEstimationOptions.handOrientationAlpha = alpha;

    // Update display
    const valueDisplay = $('handOrientationAlphaValue');
    if (valueDisplay) {
        valueDisplay.textContent = `${value}%`;
    }
}

// Export for HTML onclick
window.togglePoseOption = togglePoseOption;
window.updateHandOrientationAlpha = updateHandOrientationAlpha;

/**
 * Update pose estimation from magnetic field data
 * Called by telemetry handler when pose tracking is enabled
 * @param {Object} data - Sensor data {magField, orientation, euler, sample}
 */
function updatePoseEstimationFromMag(data) {
    if (!poseState.enabled) return;

    const { magField, orientation, euler, sample } = data;
    poseState.updateCount++;

    // Get magnetic field data - use calibrated/filtered if options enabled
    let mx = magField.x;
    let my = magField.y;
    let mz = magField.z;

    if (sample) {
        // Select best available data based on options
        if (poseEstimationOptions.useFiltering && sample.filtered_mx !== undefined) {
            // Use Kalman filtered data
            mx = sample.filtered_mx;
            my = sample.filtered_my;
            mz = sample.filtered_mz;
        } else if (poseEstimationOptions.useCalibration) {
            // Prefer fused (calibrated + earth field removed) > calibrated (iron corrected) > raw
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

    // Calculate field strength as a simple confidence metric
    const strength = Math.sqrt(mx * mx + my * my + mz * mz);

    // Simple confidence based on field strength (normalized)
    // Typical finger magnet field: 10-100 µT above background
    poseState.confidence = Math.min(1.0, strength / 100);

    // Enhanced pose estimation using orientation context
    let baseThreshold = 20;
    let highThreshold = 60;

    if (poseEstimationOptions.useOrientation && euler) {
        // Adjust thresholds based on device orientation
        // When tilted, gravity affects the perceived magnetic field strength
        const tiltFactor = Math.abs(Math.cos(euler.pitch * Math.PI / 180) * Math.cos(euler.roll * Math.PI / 180));
        baseThreshold *= tiltFactor;
        highThreshold *= tiltFactor;

        // Store orientation for 3D visualization
        poseState.orientation = { roll: euler.roll, pitch: euler.pitch, yaw: euler.yaw };
    }

    // Simple pose estimation based on field strength thresholds
    // TODO: Implement proper template matching using labeled training data
    if (strength < baseThreshold) {
        // Low field - likely open hand (fingers extended)
        poseState.currentPose = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
    } else if (strength > highThreshold) {
        // High field - likely fist (fingers flexed)
        poseState.currentPose = { thumb: 2, index: 2, middle: 2, ring: 2, pinky: 2 };
    } else {
        // Medium field - partial flexion
        const flex = (strength - baseThreshold) / (highThreshold - baseThreshold); // 0-1 range
        const fingerState = Math.round(flex * 2); // 0, 1, or 2
        poseState.currentPose = { thumb: fingerState, index: fingerState, middle: fingerState, ring: fingerState, pinky: fingerState };
    }

    // Update 3D hand orientation from sensor fusion (every sample for smooth motion)
    if (poseEstimationOptions.enableHandOrientation && euler && threeHandSkeleton) {
        threeHandSkeleton.updateOrientation(euler);
    }

    // Update display every 10 samples to avoid excessive DOM updates
    if (poseState.updateCount % 10 === 0) {
        updatePoseEstimationDisplay();
        if (handPreviewMode === 'predictions') {
            updateHandVisualization();
        }
        if (poseEstimationOptions.show3DOrientation && poseState.orientation) {
            update3DOrientationCube(poseState.orientation);
        }

        // Update hand orientation display
        updateHandOrientationDisplay(euler);
    }
}

/**
 * Update hand orientation display
 */
function updateHandOrientationDisplay(euler) {
    const orientationInfo = $('handOrientationInfo');
    const anglesDisplay = $('handOrientationAngles');

    if (!orientationInfo || !anglesDisplay) return;

    // Show/hide based on sensor fusion state
    if (poseState.enabled && poseEstimationOptions.enableHandOrientation && euler) {
        orientationInfo.style.display = 'block';
        anglesDisplay.textContent = `Roll: ${euler.roll.toFixed(1)}° | Pitch: ${euler.pitch.toFixed(1)}° | Yaw: ${euler.yaw.toFixed(1)}°`;
    } else {
        orientationInfo.style.display = 'none';
    }
}

/**
 * Update 3D orientation cube visualization
 */
function update3DOrientationCube(orientation) {
    const cube = $('orientationCube');
    const angles = $('orientationAngles');

    if (cube) {
        cube.style.transform = `rotateX(${orientation.pitch}deg) rotateY(${orientation.roll}deg) rotateZ(${orientation.yaw}deg)`;
    }

    if (angles) {
        angles.textContent = `R: ${orientation.roll.toFixed(1)}° P: ${orientation.pitch.toFixed(1)}° Y: ${orientation.yaw.toFixed(1)}°`;
    }
}

/**
 * Update pose estimation display
 */
function updatePoseEstimationDisplay() {
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
        updatesText.textContent = poseState.updateCount;
    }
}

/**
 * Initialize hand visualization
 */
function initHandVisualization() {
    // Initialize Three.js hand skeleton
    const container = $('threeHandContainer');
    if (container && typeof THREE !== 'undefined') {
        try {
            threeHandSkeleton = new ThreeJSHandSkeleton(container, {
                width: 400,
                height: 400,
                backgroundColor: 0x1a1a2e,
                lerpFactor: 0.15,
                handedness: 'right'
            });

            // Set orientation offsets to match sensor-to-hand mapping
            threeHandSkeleton.setOrientationOffsets({
                roll: 180,
                pitch: 180,
                yaw: -180
            });

            log('Three.js hand skeleton initialized');
        } catch (err) {
            console.error('Failed to initialize Three.js hand skeleton:', err);
        }
    } else {
        console.warn('Three.js hand skeleton not available (Three.js not loaded or container missing)');
    }

    // Initial update
    updateHandVisualization();
}

/**
 * Initialize magnetic trajectory visualization
 */
function initMagneticTrajectory() {
    const canvas = $('magTrajectoryCanvas');
    if (canvas) {
        magTrajectory = new MagneticTrajectory(canvas, {
            maxPoints: 200,
            trajectoryColor: '#4ecdc4',
            backgroundColor: '#ffffff',
            autoNormalize: true,
            showMarkers: true,
            showCube: true
        });

        // Clear button
        const clearBtn = $('clearTrajectoryBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                magTrajectory.clear();
                updateMagTrajectoryStats();
                log('Magnetic trajectory cleared');
            });
        }

        magTrajectory.addPoint(0, 0, 0);
        magTrajectory.addPoint(1, 1, 1);
        updateMagTrajectoryStats();

        // Pause/Resume button
        const pauseBtn = $('pauseTrajectoryBtn');
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => {
                magTrajectoryPaused = !magTrajectoryPaused;
                pauseBtn.textContent = magTrajectoryPaused ? 'Resume' : 'Pause';
                pauseBtn.className = magTrajectoryPaused ? 'btn-primary' : 'btn-secondary';
                log(`Magnetic trajectory ${magTrajectoryPaused ? 'paused' : 'resumed'}`);
            });
        }

        log('Magnetic trajectory visualizer initialized');
    } else {
        console.warn('Magnetic trajectory canvas not found');
    }
}

/**
 * Update magnetic trajectory stats display
 */
function updateMagTrajectoryStats() {
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
 * Update magnetic trajectory from telemetry data
 * Called by telemetry handler when new residual magnetic field data arrives
 * @param {Object} data - Contains fused_mx, fused_my, fused_mz
 */
function updateMagneticTrajectoryFromTelemetry(data) {
    if (!magTrajectory || magTrajectoryPaused) return;
    if (!data || data.fused_mx === undefined) return;

    magTrajectory.addPoint(data.fused_mx, data.fused_my, data.fused_mz);

    // Update stats every 50 points
    if (magTrajectory.points.length % 50 === 0) {
        updateMagTrajectoryStats();
    }
}

/**
 * Set hand preview mode
 */
function setHandPreviewMode(mode) {
    handPreviewMode = mode;

    // Update button states
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

            // Update description based on sensor fusion state
            if (poseEstimationOptions.enableHandOrientation && poseState.enabled) {
                if (description) description.textContent = 'Real-time pose estimation with 3D sensor fusion orientation';
            } else {
                if (description) description.textContent = 'Real-time pose estimation from magnetic field data';
            }

            if (predictionInfo) predictionInfo.style.display = 'block';
        }
    }

    updateHandVisualization();
    log(`Hand preview mode: ${mode}`);
}

/**
 * Update hand visualization based on current mode
 */
function updateHandVisualization() {
    // Determine finger states based on preview mode
    let fingerStates;
    if (handPreviewMode === 'labels') {
        // Show manual labels
        fingerStates = convertFingerLabelsToStates(state.currentLabels.fingers);
    } else {
        // Show predictions (if available)
        if (poseState.enabled && poseState.currentPose) {
            fingerStates = poseState.currentPose;
        } else {
            // No predictions available - show neutral
            fingerStates = {
                thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0
            };
        }
    }

}

/**
 * Convert finger label strings to numeric states for visualization
 */
function convertFingerLabelsToStates(fingerLabels) {
    const states = {};
    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];

    for (const finger of fingers) {
        const label = fingerLabels[finger];
        if (label === 'extended') {
            states[finger] = 0;
        } else if (label === 'partial') {
            states[finger] = 1;
        } else if (label === 'flexed') {
            states[finger] = 2;
        } else {
            states[finger] = 0; // Default to extended for unknown
        }
    }

    return states;
}

// Export for HTML onclick handlers
window.setHandPreviewMode = setHandPreviewMode;

/**
 * Initialize export functionality
 */
function initExport() {
    const exportBtn = $('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportData);
    }

    // Initialize upload functionality
    const uploadBtn = $('uploadBtn');
    if (uploadBtn) {
        uploadBtn.addEventListener('click', uploadToGitHub);
    }
}

/**
 * Store calibration session data for export/upload
 * @param {Array} samples - Raw calibration samples
 * @param {string} stepName - Calibration step name (HARD_IRON, SOFT_IRON)
 * @param {Object} result - Calibration result with quality metrics
 */
function storeCalibrationSessionData(samples, stepName, result) {
    // Map step name to calibration label format
    // NOTE: Earth field calibration removed - now auto-estimated in real-time
    const calibrationLabels = {
        'HARD_IRON': 'hard_iron',
        'SOFT_IRON': 'soft_iron'
    };

    const calibrationLabel = calibrationLabels[stepName] || stepName.toLowerCase();
    const startIndex = state.sessionData.length;

    // Add timestamp to each sample and store
    const timestamp = Date.now();
    samples.forEach((sample, i) => {
        state.sessionData.push({
            ...sample,
            timestamp: timestamp + (i * (1000 / 26)),  // Approximate timestamp at 26Hz
            calibration_step: stepName
        });
    });

    // Create a label segment for this calibration
    const segment = {
        start_sample: startIndex,
        end_sample: state.sessionData.length,
        labels: {
            pose: null,
            fingers: {
                thumb: 'unknown',
                index: 'unknown',
                middle: 'unknown',
                ring: 'unknown',
                pinky: 'unknown'
            },
            motion: 'static',
            calibration: calibrationLabel,
            custom: ['calibration_session', `cal_${calibrationLabel}`]
        },
        metadata: {
            session_type: 'calibration',
            calibration_step: stepName,
            quality: result.quality,
            sample_count: samples.length,
            result_summary: result.quality ? {
                quality: result.quality.toFixed(3),
                magnitude: result.magnitude?.toFixed(2),
                avgDeviation: result.avgDeviation?.toFixed(2)
            } : null
        }
    };

    state.labels.push(segment);

    log(`Calibration session stored: ${samples.length} samples for ${stepName}`);
    updateUI();
}

/**
 * Upload data to GitHub
 */
async function uploadToGitHub() {
    if (state.sessionData.length === 0) {
        log('No data to upload');
        return;
    }

    if (!ghToken) {
        log('Error: No GitHub token configured');
        return;
    }

    const uploadBtn = $('uploadBtn');
    const originalText = uploadBtn.textContent;

    try {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';
        log('Uploading to GitHub...');

        // Build export data with metadata
        const exportData = buildExportData();

        // Create filename with timestamp
        const timestamp = new Date().toISOString().replace(/:/g, '_');
        const filename = `${timestamp}.json`;
        const content = JSON.stringify(exportData, null, 2);

        // GitHub API endpoint
        const endpoint = `https://api.github.com/repos/christopherdebeer/simcap/contents/data/GAMBIT/${filename}`;

        // Base64 encode content
        const b64Content = btoa(unescape(encodeURIComponent(content)));

        const response = await fetch(endpoint, {
            method: 'PUT',
            headers: {
                'Authorization': `token ${ghToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: `GAMBIT Data ingest ${filename}`,
                content: b64Content
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || `HTTP ${response.status}`);
        }

        const result = await response.json();
        log(`Uploaded: ${result.content?.name || filename}`);

    } catch (e) {
        console.error('[GAMBIT] Upload failed:', e);
        log(`Upload failed: ${e.message}`);
    } finally {
        uploadBtn.disabled = state.sessionData.length === 0 || state.recording || !ghToken;
        uploadBtn.textContent = originalText;
    }
}

/**
 * Build export data object with metadata
 * @param {Object} options - Optional overrides for metadata
 * @returns {Object} Export data object
 */
function buildExportData(options = {}) {
    // Get session metadata from form fields
    const subjectId = $('subjectId')?.value || 'unknown';
    const environment = $('environment')?.value || 'unknown';
    const hand = $('hand')?.value || 'unknown';
    const split = $('split')?.value || 'train';
    const magnetConfig = $('magnetConfig')?.value || 'none';
    const magnetType = $('magnetType')?.value || 'unknown';
    const sessionNotes = $('sessionNotes')?.value || '';

    return {
        version: '2.1',
        timestamp: new Date().toISOString(),
        samples: state.sessionData,
        labels: state.labels,
        metadata: {
            sample_rate: 26,  // Match firmware accelOn rate
            device: 'GAMBIT',
            firmware_version: state.firmwareVersion || 'unknown',
            calibration: calibrationInstance ? calibrationInstance.toJSON() : null,
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
 * Export data to JSON file
 */
function exportData() {
    if (state.sessionData.length === 0) {
        log('No data to export');
        return;
    }

    const data = buildExportData();

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // Use ISO timestamp with colons replaced by underscores for ML pipeline compatibility
    const timestamp = new Date().toISOString().replace(/:/g, '_');
    a.download = `${timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);

    log('Data exported');
}

/**
 * Initialize collapsible sections
 */
function initCollapsibleSections() {
    document.querySelectorAll('.collapsible').forEach(header => {
        header.addEventListener('click', () => {
            header.classList.toggle('collapsed');
            
            const section = header.parentElement;
            const contents = section.querySelectorAll('.collapse-content');
            contents.forEach(content => {
                content.classList.toggle('hidden');
            });
            
            const sectionId = section.id;
            if (sectionId) {
                const isCollapsed = header.classList.contains('collapsed');
                localStorage.setItem(`section_${sectionId}_collapsed`, isCollapsed);
            }
        });

        const section = header.parentElement;
        const sectionId = section.id;
        if (sectionId) {
            const savedState = localStorage.getItem(`section_${sectionId}_collapsed`);
            if (savedState === 'true') {
                header.classList.add('collapsed');
                const contents = section.querySelectorAll('.collapse-content');
                contents.forEach(content => {
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
window.log = log;
window.updateUI = updateUI;
window.addCustomLabel = addCustomLabel;
window.addPresetLabels = addPresetLabels;
window.removeActiveLabel = removeActiveLabel;
window.clearFingerStates = clearFingerStates;
window.removeCustomLabel = removeCustomLabel;
window.toggleCustomLabel = toggleCustomLabel;
window.deleteCustomLabelDef = deleteCustomLabelDef;
