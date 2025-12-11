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
    calibrationInstance
} from './modules/calibration-ui.js';
import { onTelemetry, setDependencies as setTelemetryDeps, resetIMU } from './modules/telemetry-handler.js';
import { setCallbacks as setConnectionCallbacks, initConnectionUI } from './modules/connection-manager.js';
import { setCallbacks as setRecordingCallbacks, initRecordingUI, startRecording } from './modules/recording-controls.js';
import {
    initWizard,
    setDependencies as setWizardDeps,
    getWizardState,
    getCalibrationBuffers
} from './modules/wizard.js';

// Export state for global access (used by inline functions in HTML)
window.appState = state;
window.log = log;

// DOM helper
const $ = (id) => document.getElementById(id);

// Initialize filters and fusion
const magFilter = new KalmanFilter3D({
    processNoise: 0.1,
    measurementNoise: 1.0
});

const imuFusion = new MadgwickAHRS({
    sampleFreq: 50,
    beta: 0.1
});

// Pose estimation state
const poseState = {
    enabled: false,
    currentPose: null,
    confidence: 0,
    updateCount: 0
};

// Particle filter for pose estimation
let poseFilter = null;
let poseEstimationEnabled = false;

// Default reference pose (palm-down, fingers extended, typical geometry)
// Positions in mm relative to sensor (at origin)
const defaultReferencePose = {
    thumb:  {x: -30, y: 40, z: 10},   // Left side, forward
    index:  {x: -15, y: 60, z: 10},   // Slightly left, far forward
    middle: {x: 0,   y: 65, z: 10},   // Center, farthest forward
    ring:   {x: 15,  y: 60, z: 10},   // Slightly right, forward
    pinky:  {x: 30,  y: 50, z: 10}    // Right side, less forward
};

// Magnet configuration (default N52 3mm x 2mm cylindrical magnets)
const magnetConfig = {
    thumb:  {moment: {x: 0, y: 0, z: 0.01}},
    index:  {moment: {x: 0, y: 0, z: 0.01}},
    middle: {moment: {x: 0, y: 0, z: 0.01}},
    ring:   {moment: {x: 0, y: 0, z: 0.01}},
    pinky:  {moment: {x: 0, y: 0, z: 0.01}}
};

// Hand visualization
let handVisualizer = null;
let hand3DRenderer = null;
let handPreviewMode = 'labels'; // 'labels' or 'predictions'
let handViewMode = '2d'; // '2d' or '3d'

// Pose estimation options
const poseEstimationOptions = {
    useCalibration: true,  // Use calibrated magnetic field data
    useOrientation: true,  // Use IMU orientation for context
    show3DOrientation: false,  // Show 3D orientation cube
    useParticleFilter: false  // Use ParticleFilter vs threshold-based estimation
};

// GitHub token (for upload functionality)
let ghToken = null;

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
        updatePoseEstimation: updatePoseEstimationFromMag,
        updateUI: updateUI,
        $: $
    });

    // Set connection callbacks
    setConnectionCallbacks(updateUI, updateCalibrationStatus);

    // Set recording callbacks
    setRecordingCallbacks(updateUI, closeCurrentLabel);

    // Initialize UI controls
    initConnectionUI($('connectBtn'));
    initRecordingUI({
        start: $('startBtn'),
        stop: $('stopBtn'),
        clear: $('clearBtn')
    });
    initCalibrationUI();

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

    // Initialize collapsible sections
    initCollapsibleSections();

    // Load custom labels from localStorage
    loadCustomLabels();

    // Try to load GitHub token
    try {
        ghToken = localStorage.getItem('gh_token');
        if (ghToken) {
            log('GitHub token loaded');
        }
    } catch (e) {
        console.warn('Failed to load GitHub token:', e);
    }

    // Initial UI update
    updateUI();
    updateCalibrationStatus();

    log('GAMBIT Collector ready');
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
        if (state.recording) {
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
        tags.push(`<span class="active-label-chip" onclick="removeActiveLabel('pose')" style="cursor: pointer;" title="Click to remove">${state.currentLabels.pose} ×</span>`);
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
        tags.push(`<span class="active-label-chip" onclick="clearFingerStates()" style="cursor: pointer;" title="Click to clear">${fingerStates} ×</span>`);
    }

    if (state.currentLabels.motion !== 'static') {
        tags.push(`<span class="active-label-chip" onclick="removeActiveLabel('motion')" style="cursor: pointer;" title="Click to remove">${state.currentLabels.motion} ×</span>`);
    }

    if (state.currentLabels.calibration !== 'none') {
        tags.push(`<span class="active-label-chip" onclick="removeActiveLabel('calibration')" style="cursor: pointer;" title="Click to remove">${state.currentLabels.calibration} ×</span>`);
    }

    (state.currentLabels.custom || []).forEach(c => {
        tags.push(`<span class="active-label-chip" onclick="removeCustomLabel('${c}')" style="cursor: pointer;" title="Click to remove">${c} ×</span>`);
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
        log(`Custom label removed: ${label}`);
        onLabelsChanged();
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
 * Add custom label
 */
function addCustomLabel() {
    const input = $('customLabelInput');
    if (!input) return;

    const value = input.value.trim();
    if (!value) return;

    if (!state.currentLabels.custom.includes(value)) {
        state.currentLabels.custom.push(value);
        onLabelsChanged();
        log(`Custom label added: ${value}`);
    }
    input.value = '';
    updateActiveLabelsDisplay();
}

/**
 * Add preset labels
 */
function addPresetLabels(preset) {
    const presets = {
        phase1: ['warmup', 'baseline', 'initial'],
        phase2: ['calibrated', 'training', 'validation'],
        quality: ['good', 'noisy', 'drift'],
        transitions: ['entering', 'holding', 'exiting']
    };

    const labels = presets[preset] || [];
    labels.forEach(label => {
        if (!state.currentLabels.custom.includes(label)) {
            state.currentLabels.custom.push(label);
        }
    });
    
    if (labels.length > 0) {
        onLabelsChanged();
        log(`Added ${labels.length} preset labels: ${labels.join(', ')}`);
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
            renderCustomLabels();
        }
    } catch (e) {
        console.error('Failed to load custom labels:', e);
    }
}

/**
 * Render custom labels (stub - implement if needed)
 */
function renderCustomLabels() {
    // TODO: Implement custom label rendering if needed
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
            <div style="font-size: 11px; color: var(--fg-muted); margin-bottom: 8px;">Estimation Options:</div>
            <div style="display: flex; flex-direction: column; gap: 6px; font-size: 11px;">
                <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input type="checkbox" id="useCalibrationToggle" ${poseEstimationOptions.useCalibration ? 'checked' : ''}
                           onchange="togglePoseOption('useCalibration')" />
                    <span>Use Calibration (when available)</span>
                </label>
                <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input type="checkbox" id="useOrientationToggle" ${poseEstimationOptions.useOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('useOrientation')" />
                    <span>Use IMU Orientation (sensor fusion)</span>
                </label>
                <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                    <input type="checkbox" id="show3DOrientationToggle" ${poseEstimationOptions.show3DOrientation ? 'checked' : ''}
                           onchange="togglePoseOption('show3DOrientation')" />
                    <span>Show 3D Orientation Cube</span>
                </label>
            </div>
            <div id="orientation3DCube" style="display: none; margin-top: 12px; perspective: 500px; height: 120px;">
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

    // Show/hide 3D cube
    if (option === 'show3DOrientation') {
        const cube = $('orientation3DCube');
        if (cube) {
            cube.style.display = poseEstimationOptions.show3DOrientation ? 'block' : 'none';
        }
    }

    log(`Pose option ${option}: ${poseEstimationOptions[option]}`);
}

// Export for HTML onclick
window.togglePoseOption = togglePoseOption;

/**
 * Update pose estimation from magnetic field data
 * Called by telemetry handler when pose tracking is enabled
 * @param {Object} data - Sensor data {magField, orientation, euler, sample}
 */
function updatePoseEstimationFromMag(data) {
    if (!poseState.enabled) return;

    const { magField, orientation, euler, sample } = data;
    poseState.updateCount++;

    // Get magnetic field data - use calibrated if option enabled and available
    let mx = magField.x;
    let my = magField.y;
    let mz = magField.z;

    if (poseEstimationOptions.useCalibration && sample) {
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

    // Update display every 10 samples to avoid excessive DOM updates
    if (poseState.updateCount % 10 === 0) {
        updatePoseEstimationDisplay();
        if (handPreviewMode === 'predictions') {
            updateHandVisualization();
        }
        if (poseEstimationOptions.show3DOrientation && poseState.orientation) {
            update3DOrientationCube(poseState.orientation);
        }
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
    // Initialize 2D visualizer
    const canvas2D = $('handCanvas');
    if (canvas2D && typeof HandVisualizer2D !== 'undefined') {
        handVisualizer = new HandVisualizer2D(canvas2D, {
            showLabels: true,
            backgroundColor: 'var(--bg-surface)'
        });
        handVisualizer.startAnimation();
    } else {
        console.warn('2D hand visualization not available');
    }

    // Initialize 3D renderer
    const canvas3D = $('handCanvas3D');
    if (canvas3D && typeof Hand3DRenderer !== 'undefined') {
        hand3DRenderer = new Hand3DRenderer(canvas3D, {
            backgroundColor: '#ffffff'
        });
        hand3DRenderer.startAnimation();
    } else {
        console.warn('3D hand renderer not available');
    }

    // Initial update
    updateHandVisualization();
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

    if (labelsBtn && predictionsBtn) {
        if (mode === 'labels') {
            labelsBtn.className = 'btn-primary btn-small';
            predictionsBtn.className = 'btn-secondary btn-small';
            if (indicator) indicator.innerHTML = '📋 Showing: Manual Labels';
            if (description) description.textContent = 'Visual representation of manually selected finger states';
        } else {
            labelsBtn.className = 'btn-secondary btn-small';
            predictionsBtn.className = 'btn-primary btn-small';
            if (indicator) indicator.innerHTML = '🎯 Showing: Pose Predictions';
            if (description) description.textContent = 'Real-time pose estimation from magnetic field data';
        }
    }

    updateHandVisualization();
    log(`Hand preview mode: ${mode}`);
}

/**
 * Set hand view mode (2D or 3D)
 */
function setHandViewMode(mode) {
    handViewMode = mode;

    // Update button states
    const btn2D = $('handView2D');
    const btn3D = $('handView3D');
    const canvas2D = $('handCanvas');
    const canvas3D = $('handCanvas3D');

    if (btn2D && btn3D) {
        if (mode === '2d') {
            btn2D.className = 'btn-primary btn-small';
            btn3D.className = 'btn-secondary btn-small';
            if (canvas2D) canvas2D.style.display = 'block';
            if (canvas3D) canvas3D.style.display = 'none';
        } else {
            btn2D.className = 'btn-secondary btn-small';
            btn3D.className = 'btn-primary btn-small';
            if (canvas2D) canvas2D.style.display = 'none';
            if (canvas3D) canvas3D.style.display = 'block';
        }
    }

    updateHandVisualization();
    log(`Hand view mode: ${mode}`);
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

    // Update active renderer based on view mode
    if (handViewMode === '2d' && handVisualizer) {
        handVisualizer.setFingerStates(fingerStates);
    } else if (handViewMode === '3d' && hand3DRenderer) {
        hand3DRenderer.setFingerPoses(fingerStates);

        // Update orientation from IMU if available
        if (poseState.orientation) {
            hand3DRenderer.setOrientation({
                pitch: poseState.orientation.pitch,
                yaw: poseState.orientation.yaw,
                roll: poseState.orientation.roll
            });
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
window.setHandViewMode = setHandViewMode;

/**
 * Initialize export functionality
 */
function initExport() {
    const exportBtn = $('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportData);
    }
}

/**
 * Export data to JSON file
 */
function exportData() {
    if (state.sessionData.length === 0) {
        log('No data to export');
        return;
    }

    const exportData = {
        version: '2.0',
        timestamp: new Date().toISOString(),
        samples: state.sessionData,
        labels: state.labels,
        metadata: {
            sample_rate: 50,
            device: 'GAMBIT',
            calibration: calibrationInstance ? calibrationInstance.toJSON() : null
        }
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gambit_session_${Date.now()}.json`;
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

// Export for debugging
window.appState = state;
window.log = log;
window.updateUI = updateUI;
window.addCustomLabel = addCustomLabel;
window.addPresetLabels = addPresetLabels;
window.removeActiveLabel = removeActiveLabel;
window.clearFingerStates = clearFingerStates;
window.removeCustomLabel = removeCustomLabel;
