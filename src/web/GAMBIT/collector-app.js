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

// Hand visualization
let handVisualizer = null;
let handPreviewMode = 'labels'; // 'labels' or 'predictions'

// Wizard and calibration buffers (for wizard functionality)
const wizard = {
    active: false,
    mode: null,
    currentStep: 0,
    steps: [],
    phase: null,  // 'transition' | 'hold'
    phaseStart: null
};

const calibrationBuffers = {};

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

    // Set telemetry dependencies
    setTelemetryDeps({
        calibrationInstance: calInstance,
        magFilter: magFilter,
        imuFusion: imuFusion,
        wizard: wizard,
        calibrationBuffers: calibrationBuffers,
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

    // Initialize wizard functionality
    initWizard();

    // Initialize pose estimation functionality
    initPoseEstimation();

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
 * Initialize wizard functionality
 */
function initWizard() {
    const wizardBtn = $('wizardBtn');
    if (wizardBtn) {
        wizardBtn.addEventListener('click', startWizard);
    }
}

/**
 * Start data collection wizard
 */
function startWizard() {
    if (!state.connected) {
        log('Error: Connect device first');
        return;
    }

    // Initialize wizard state
    wizard.active = true;
    wizard.mode = 'data_collection';
    wizard.currentStep = 0;
    wizard.steps = [
        {
            id: 'intro',
            title: 'Data Collection Wizard',
            instruction: 'Welcome to guided data collection',
            description: 'This wizard will guide you through collecting labeled training data for hand pose recognition.'
        },
        {
            id: 'rest_pose',
            title: 'Collect: Rest Pose',
            instruction: 'Hold your hand in a relaxed rest position',
            description: 'Keep your hand still for 3 seconds while we collect baseline data.',
            duration: 3000,
            labels: { pose: 'rest', motion: 'static' }
        },
        {
            id: 'fist_pose',
            title: 'Collect: Fist',
            instruction: 'Make a fist',
            description: 'Close your hand into a fist and hold steady for 3 seconds.',
            duration: 3000,
            labels: { pose: 'fist', motion: 'static' }
        },
        {
            id: 'open_pose',
            title: 'Collect: Open Hand',
            instruction: 'Open your hand with fingers extended',
            description: 'Spread your fingers wide and hold for 3 seconds.',
            duration: 3000,
            labels: { pose: 'open_palm', motion: 'static' }
        },
        {
            id: 'complete',
            title: 'Collection Complete!',
            instruction: '✅ Data collection finished',
            description: 'You have successfully collected training data for 3 hand poses.'
        }
    ];

    log('Data collection wizard started');
    showWizardModal();
    renderWizardStep();
}

/**
 * Show wizard modal
 */
function showWizardModal() {
    const overlay = $('wizardOverlay');
    if (overlay) {
        overlay.classList.add('active');
    }
}

/**
 * Close wizard modal
 */
function closeWizard() {
    wizard.active = false;
    wizard.currentStep = 0;
    wizard.phase = null;

    const overlay = $('wizardOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }

    log('Wizard closed');
}

/**
 * Render current wizard step
 */
function renderWizardStep() {
    const step = wizard.steps[wizard.currentStep];
    if (!step) return;

    const title = $('wizardTitle');
    const phase = $('wizardPhase');
    const content = $('wizardContent');
    const progressFill = $('wizardProgressFill');
    const stepText = $('wizardStepText');
    const timeText = $('wizardTimeText');
    const samplesText = $('wizardSamples');
    const labelsText = $('wizardLabels');

    if (title) title.textContent = step.title;
    if (phase) phase.textContent = `Step ${wizard.currentStep + 1} of ${wizard.steps.length}`;

    const progress = ((wizard.currentStep) / wizard.steps.length) * 100;
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (stepText) stepText.textContent = `Step ${wizard.currentStep + 1} of ${wizard.steps.length}`;
    if (timeText) timeText.textContent = step.duration ? `~${step.duration / 1000}s` : '';

    if (samplesText) samplesText.textContent = state.sessionData.length;
    if (labelsText) labelsText.textContent = state.labels.length;

    // Render step content
    if (content) {
        let html = `
            <div class="wizard-instruction">${step.instruction}</div>
            <div class="wizard-description">${step.description}</div>
        `;

        if (step.id === 'intro') {
            html += `
                <div class="wizard-controls">
                    <button class="btn-primary" onclick="nextWizardStep()">Start Collection</button>
                    <button class="btn-secondary" onclick="closeWizard()">Cancel</button>
                </div>
            `;
        } else if (step.id === 'complete') {
            html += `
                <div class="wizard-complete">
                    <div class="stats">
                        ${state.sessionData.length} samples collected<br>
                        ${state.labels.length} labels created
                    </div>
                </div>
                <div class="wizard-controls">
                    <button class="btn-success" onclick="closeWizard()">Done</button>
                </div>
            `;
        } else {
            // Collection step
            html += `
                <div class="wizard-controls">
                    <button class="btn-success" onclick="startWizardCollection()">Ready - Start</button>
                    <button class="btn-secondary" onclick="nextWizardStep()">Skip</button>
                </div>
            `;
        }

        content.innerHTML = html;
    }
}

/**
 * Start collection for current wizard step
 */
async function startWizardCollection() {
    const step = wizard.steps[wizard.currentStep];
    if (!step || !step.duration) return;

    // Apply labels for this step
    if (step.labels) {
        if (step.labels.pose) state.currentLabels.pose = step.labels.pose;
        if (step.labels.motion) state.currentLabels.motion = step.labels.motion;
    }

    // Start recording if not already
    if (!state.recording) {
        await startRecording();
    }

    // Show countdown
    const content = $('wizardContent');
    let remainingMs = step.duration;

    const countdownInterval = setInterval(() => {
        remainingMs -= 100;
        const seconds = Math.ceil(remainingMs / 1000);

        if (content) {
            content.innerHTML = `
                <div class="wizard-instruction">Hold steady...</div>
                <div class="wizard-countdown">
                    <div class="countdown-circle ${seconds <= 1 ? 'warning' : ''}">
                        ${seconds}
                    </div>
                </div>
            `;
        }

        if (remainingMs <= 0) {
            clearInterval(countdownInterval);
            nextWizardStep();
        }
    }, 100);
}

/**
 * Move to next wizard step
 */
function nextWizardStep() {
    wizard.currentStep++;

    if (wizard.currentStep >= wizard.steps.length) {
        closeWizard();
    } else {
        renderWizardStep();
    }
}

// Export wizard functions to window for onclick handlers
window.closeWizard = closeWizard;
window.nextWizardStep = nextWizardStep;
window.startWizardCollection = startWizardCollection;

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
 * Update pose estimation from magnetic field data
 * Called by telemetry handler when pose tracking is enabled
 * @param {Object} magField - Filtered magnetic field {x, y, z}
 */
function updatePoseEstimationFromMag(magField) {
    if (!poseState.enabled) return;

    // Simple similarity-based pose estimation
    // TODO: Implement proper pose estimation algorithm with template matching
    // For now, just track that we're receiving data
    poseState.updateCount++;

    // Calculate field strength as a simple confidence metric
    const strength = Math.sqrt(magField.x * magField.x + magField.y * magField.y + magField.z * magField.z);

    // Simple confidence based on field strength (normalized)
    // Typical finger magnet field: 10-100 µT above background
    poseState.confidence = Math.min(1.0, strength / 100);

    // Simple pose estimation based on field strength thresholds
    // This is a placeholder - real implementation would use template matching
    if (strength < 20) {
        // Low field - likely open hand (fingers extended)
        poseState.currentPose = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
    } else if (strength > 60) {
        // High field - likely fist (fingers flexed)
        poseState.currentPose = { thumb: 2, index: 2, middle: 2, ring: 2, pinky: 2 };
    } else {
        // Medium field - partial flexion
        const flex = (strength - 20) / 40; // 0-1 range
        const state = Math.round(flex * 2); // 0, 1, or 2
        poseState.currentPose = { thumb: state, index: state, middle: state, ring: state, pinky: state };
    }

    // Update display every 10 samples to avoid excessive DOM updates
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
    const canvas = $('handCanvas');
    if (!canvas || typeof HandVisualizer2D === 'undefined') {
        console.warn('Hand visualization not available');
        return;
    }

    // Create visualizer
    handVisualizer = new HandVisualizer2D(canvas, {
        showLabels: true,
        backgroundColor: 'var(--bg)'
    });

    // Start animation loop
    handVisualizer.startAnimation();

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
 * Update hand visualization based on current mode
 */
function updateHandVisualization() {
    if (!handVisualizer) return;

    if (handPreviewMode === 'labels') {
        // Show manual labels
        const fingerStates = convertFingerLabelsToStates(state.currentLabels.fingers);
        handVisualizer.setFingerStates(fingerStates);
    } else {
        // Show predictions (if available)
        if (poseState.enabled && poseState.currentPose) {
            handVisualizer.setFingerStates(poseState.currentPose);
        } else {
            // No predictions available - show neutral
            handVisualizer.setFingerStates({
                thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0
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
