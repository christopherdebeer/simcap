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
import { setCallbacks as setRecordingCallbacks, initRecordingUI } from './modules/recording-controls.js';

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
        updatePoseEstimation: null,  // TODO: implement if needed
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
        tags.push(`<span class="label-tag pose">${state.currentLabels.pose}</span>`);
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
        tags.push(`<span class="label-tag finger">${fingerStates}</span>`);
    }

    if (state.currentLabels.motion !== 'static') {
        tags.push(`<span class="label-tag motion">${state.currentLabels.motion}</span>`);
    }

    if (state.currentLabels.calibration !== 'none') {
        tags.push(`<span class="label-tag calibration">${state.currentLabels.calibration}</span>`);
    }

    (state.currentLabels.custom || []).forEach(c => {
        tags.push(`<span class="label-tag custom">${c}</span>`);
    });

    display.innerHTML = tags.length > 0 ? tags.join('') : '<span style="color: #666;">No labels active</span>';
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
    document.querySelectorAll('.section-header').forEach(header => {
        header.addEventListener('click', () => {
            const section = header.parentElement;
            section.classList.toggle('collapsed');

            // Save collapsed state
            const sectionId = section.id;
            if (sectionId) {
                const isCollapsed = section.classList.contains('collapsed');
                localStorage.setItem(`section_${sectionId}_collapsed`, isCollapsed);
            }
        });

        // Restore collapsed state
        const section = header.parentElement;
        const sectionId = section.id;
        if (sectionId) {
            const savedState = localStorage.getItem(`section_${sectionId}_collapsed`);
            if (savedState === 'true') {
                section.classList.add('collapsed');
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
