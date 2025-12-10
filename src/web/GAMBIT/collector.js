/**
 * GAMBIT Collector Application
 * Data collection interface for magnetic finger tracking
 */

// =============================================================================
// State
// =============================================================================

const state = {
    connected: false,
    recording: false,
    sessionData: [],
    labels: [],  // V2 multi-label segments
    currentLabelStart: null,
    gambitClient: null,

    // Multi-label current state
    currentLabels: {
        pose: null,
        fingers: {
            thumb: 'unknown',
            index: 'unknown',
            middle: 'unknown',
            ring: 'unknown',
            pinky: 'unknown'
        },
        motion: 'static',
        calibration: 'none',
        custom: []
    },

    // Custom labels defined for this session
    customLabelDefinitions: []
};

// =============================================================================
// Calibration and Filtering
// =============================================================================

// Single calibration instance used for both wizard and real-time correction
let calibrationInstance = null;
let calibrationInterval = null;

function initCalibration() {
    if (typeof EnvironmentalCalibration === 'undefined') {
        console.error('Error: calibration.js not loaded');
        return null;
    }
    calibrationInstance = new EnvironmentalCalibration();

    // Try to load from localStorage using the class method
    try {
        const loaded = calibrationInstance.load('gambit_calibration');
        if (loaded) {
            console.log('[Calibration] Loaded from localStorage');
        }
    } catch (e) {
        console.log('[Calibration] No previous calibration found');
    }
    return calibrationInstance;
}

// Initialize calibration early so it's available for data processing
calibrationInstance = initCalibration();

// Initialize 3D Kalman filter for magnetometer smoothing
const magFilter = new KalmanFilter3D({
    processNoise: 0.1,
    measurementNoise: 1.0
});

// Initialize IMU sensor fusion for orientation estimation
// Uses Madgwick AHRS to fuse accelerometer + gyroscope into device orientation
// This orientation is used to subtract Earth's magnetic field from magnetometer readings
const imuFusion = new MadgwickAHRS({
    sampleFreq: 50,   // 50 Hz telemetry rate
    beta: 0.1         // Filter gain (lower = smoother, higher = faster response)
});
let lastTelemetryTime = null;
let imuInitialized = false;

// =============================================================================
// Real-time Pose Estimation
// =============================================================================

// Initialize particle filter for finger position tracking
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

// Pose estimation state
const poseState = {
    currentPose: null,         // Latest pose estimate
    confidence: 0,             // 0-1, based on particle spread
    updateCount: 0,            // Number of updates
    enabled: false             // Can be toggled by user
};

function initializePoseEstimation(referencePose = null) {
    try {
        const initialPose = referencePose || defaultReferencePose;

        poseFilter = new ParticleFilter({
            numParticles: 500,
            positionNoise: 5.0,    // mm
            velocityNoise: 2.0     // mm/s
        });

        poseFilter.initialize(initialPose);
        poseState.enabled = true;
        poseEstimationEnabled = true;

        console.log('[PoseEstimation] Initialized with', Object.keys(initialPose).length, 'fingers');
        log('Pose estimation enabled');
    } catch (e) {
        console.error('[PoseEstimation] Failed to initialize:', e);
        poseEstimationEnabled = false;
    }
}

function updatePoseEstimation(magneticField) {
    if (!poseEstimationEnabled || !poseFilter) return;

    try {
        // Prediction step (assume 50Hz = 0.02s)
        poseFilter.predict(0.02);

        // Update with measurement
        poseFilter.update(magneticField, (particle, measurement) => {
            return magneticLikelihood(particle, measurement, magnetConfig);
        });

        // Get pose estimate
        poseState.currentPose = poseFilter.estimate();
        poseState.updateCount++;

        // Calculate confidence based on particle diversity
        // Higher diversity = lower confidence
        const diversity = poseFilter.getDiversity();
        poseState.confidence = Math.max(0, 1.0 - diversity / 100);

        // Update visualization every 10 samples (5Hz) to reduce overhead
        if (poseState.updateCount % 10 === 0) {
            updatePoseVisualization();
        }
    } catch (e) {
        console.error('[PoseEstimation] Update failed:', e);
    }
}

function updatePoseVisualization() {
    // Update the UI status display with current pose estimation state

    if (!poseState.currentPose) return;

    try {
        // Update confidence display
        const confidencePercent = (poseState.confidence * 100).toFixed(1);
        const confidenceEl = $('poseConfidenceText');
        if (confidenceEl) {
            confidenceEl.textContent = confidencePercent + '%';

            // Color code by confidence level
            if (poseState.confidence > 0.7) {
                confidenceEl.style.color = '#00ff88'; // High confidence - green
            } else if (poseState.confidence > 0.4) {
                confidenceEl.style.color = '#ffa502'; // Medium confidence - orange
            } else {
                confidenceEl.style.color = '#e74c3c'; // Low confidence - red
            }
        }

        // Update updates counter
        const updatesEl = $('poseUpdatesText');
        if (updatesEl) {
            updatesEl.textContent = poseState.updateCount;
        }

        // Update hand visualizer if in predictions mode
        if (typeof handPreviewMode !== 'undefined' && handPreviewMode === 'predictions') {
            updateHandVisualizerFromPredictions();
        }

        // Log detailed pose data periodically (every second)
        if (poseState.updateCount % 50 === 0) {
            console.log('[PoseEstimation] Pose:', poseState.currentPose);
            console.log('[PoseEstimation] Confidence:', confidencePercent + '%');
        }
    } catch (e) {
        console.error('[PoseEstimation] Visualization update failed:', e);
    }
}

// Sample buffers for wizard data collection steps
const calibrationBuffers = {
    reference_pose: [],
    magnet_baseline: []
};

// GitHub token (will be loaded from localStorage)
let ghToken = null;

// Preset label sets
const PRESETS = {
    phase1: ['single:thumb', 'single:index', 'flex_test', 'extend_test', 'snr_test'],
    phase2: ['pose:00000', 'pose:22222', 'pose:02000', 'pose:00200', 'multi_finger'],
    quality: ['quality:good', 'quality:noisy', 'quality:artifact', 'quality:review'],
    transitions: ['trans:rest_fist', 'trans:fist_open', 'trans:flex_extend', 'trans:spread']
};

// =============================================================================
// DOM Elements
// =============================================================================

const $ = id => document.getElementById(id);
const connectBtn = $('connectBtn');
const startBtn = $('startBtn');
const stopBtn = $('stopBtn');
const clearBtn = $('clearBtn');
const exportBtn = $('exportBtn');
const statusIndicator = $('statusIndicator');
const sampleCount = $('sampleCount');
const progressFill = $('progressFill');
const labelsList = $('labelsList');
const labelCount = $('labelCount');
const logDiv = $('log');
const activeLabelsDisplay = $('activeLabelsDisplay');
const customLabelsList = $('customLabelsList');

// =============================================================================
// Logging
// =============================================================================

function log(msg) {
    const time = new Date().toLocaleTimeString();
    logDiv.innerHTML = `[${time}] ${msg}<br>` + logDiv.innerHTML;
    if (logDiv.children.length > 50) {
        logDiv.removeChild(logDiv.lastChild);
    }
}

// =============================================================================
// Custom Labels - Load/Save
// =============================================================================

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

function saveCustomLabels() {
    try {
        localStorage.setItem('gambit_custom_labels', JSON.stringify(state.customLabelDefinitions));
    } catch (e) {
        console.error('Failed to save custom labels:', e);
    }
}

// =============================================================================
// Custom Labels - Management
// =============================================================================

window.addCustomLabel = function() {
    const input = $('customLabelInput');
    const label = input.value.trim().toLowerCase().replace(/\s+/g, '_');
    if (label && !state.customLabelDefinitions.includes(label)) {
        state.customLabelDefinitions.push(label);
        saveCustomLabels();
        renderCustomLabels();
        input.value = '';
        log(`Added custom label: ${label}`);
    }
};

window.addPresetLabels = function(preset) {
    const labels = PRESETS[preset] || [];
    labels.forEach(label => {
        // Add to definitions if not already present
        if (!state.customLabelDefinitions.includes(label)) {
            state.customLabelDefinitions.push(label);
        }
        // Also activate the label if not already active
        if (!state.currentLabels.custom.includes(label)) {
            state.currentLabels.custom.push(label);
        }
    });
    saveCustomLabels();
    renderCustomLabels();
    updateActiveLabelsDisplay();
    onLabelsChanged();
    log(`Added and activated preset: ${preset}`);
};

window.removeCustomLabel = function(label) {
    state.customLabelDefinitions = state.customLabelDefinitions.filter(l => l !== label);
    state.currentLabels.custom = state.currentLabels.custom.filter(l => l !== label);
    saveCustomLabels();
    renderCustomLabels();
    updateActiveLabelsDisplay();
};

window.toggleCustomLabel = function(label) {
    const idx = state.currentLabels.custom.indexOf(label);
    if (idx >= 0) {
        state.currentLabels.custom.splice(idx, 1);
    } else {
        state.currentLabels.custom.push(label);
    }
    onLabelsChanged();
    renderCustomLabels();
    updateActiveLabelsDisplay();
};

function renderCustomLabels() {
    customLabelsList.innerHTML = state.customLabelDefinitions.map(label => {
        const isActive = state.currentLabels.custom.includes(label);
        return `
            <span class="custom-label-tag ${isActive ? 'active' : ''}" onclick="toggleCustomLabel('${label}')">
                ${label}
                <span class="remove" onclick="event.stopPropagation(); removeCustomLabel('${label}')">&times;</span>
            </span>
        `;
    }).join('');
}

// =============================================================================
// Labels - Get Current Labels Object
// =============================================================================

function getCurrentLabelsObject() {
    const labels = {
        motion: state.currentLabels.motion,
        calibration: state.currentLabels.calibration,
        custom: [...state.currentLabels.custom],
        confidence: 'high',
        quality_notes: ''
    };

    if (state.currentLabels.pose) {
        labels.pose = state.currentLabels.pose;
    }

    // Only include fingers if any are set
    const hasFingerState = Object.values(state.currentLabels.fingers).some(s => s !== 'unknown');
    if (hasFingerState) {
        labels.fingers = { ...state.currentLabels.fingers };
    }

    return labels;
}

// =============================================================================
// Labels - Changed Handler
// =============================================================================

function onLabelsChanged() {
    if (state.recording) {
        closeCurrentLabel();
        state.currentLabelStart = state.sessionData.length;
    }
    updateActiveLabelsDisplay();
}

// =============================================================================
// Labels - Update Active Labels Display
// =============================================================================

function updateActiveLabelsDisplay() {
    const chips = [];

    if (state.currentLabels.pose) {
        chips.push(`<span class="active-label-chip">pose:${state.currentLabels.pose}</span>`);
    }

    // Show finger states if any set
    const fingerStr = ['thumb', 'index', 'middle', 'ring', 'pinky']
        .map(f => {
            const s = state.currentLabels.fingers[f];
            if (s === 'extended') return '0';
            if (s === 'partial') return '1';
            if (s === 'flexed') return '2';
            return '?';
        }).join('');
    if (fingerStr !== '?????') {
        chips.push(`<span class="active-label-chip">fingers:${fingerStr}</span>`);
    }

    chips.push(`<span class="active-label-chip">motion:${state.currentLabels.motion}</span>`);

    if (state.currentLabels.calibration !== 'none') {
        chips.push(`<span class="active-label-chip" style="background:#e74c3c;color:#fff">cal:${state.currentLabels.calibration}</span>`);
    }

    state.currentLabels.custom.forEach(c => {
        chips.push(`<span class="active-label-chip" style="background:#e67e22">${c}</span>`);
    });

    activeLabelsDisplay.innerHTML = chips.length > 0 ? chips.join('') : '<span style="color: #666;">No labels selected</span>';
}

// =============================================================================
// UI Updates
// =============================================================================

function updateUI() {
    // Status indicator
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

    // Buttons
    connectBtn.textContent = state.connected ? 'Disconnect' : 'Connect Device';
    startBtn.disabled = !state.connected || state.recording;
    stopBtn.disabled = !state.recording;
    clearBtn.disabled = state.sessionData.length === 0 || state.recording;
    exportBtn.disabled = state.sessionData.length === 0 || state.recording;

    // Upload button (check if ghToken is available)
    const uploadBtnEl = $('uploadBtn');
    if (uploadBtnEl) {
        uploadBtnEl.disabled = state.sessionData.length === 0 || state.recording || !ghToken;
    }

    // Sample count
    sampleCount.textContent = state.sessionData.length;
    progressFill.style.width = Math.min(100, state.sessionData.length / 15) + '%';

    // Labels
    labelCount.textContent = state.labels.length;
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
                    <button class="btn-danger btn-tiny" onclick="deleteLabel(${i})">×</button>
                </div>
            `;
        }).join('');
    }
}

// =============================================================================
// Pose Selection Event Handlers
// =============================================================================

document.querySelectorAll('[data-pose]').forEach(btn => {
    btn.addEventListener('click', () => {
        const pose = btn.dataset.pose;

        // Toggle - if already selected, deselect
        if (state.currentLabels.pose === pose) {
            state.currentLabels.pose = null;
        } else {
            state.currentLabels.pose = pose;
        }

        // Update UI
        document.querySelectorAll('[data-pose]').forEach(b => b.classList.remove('active'));
        if (state.currentLabels.pose) {
            btn.classList.add('active');
        }

        onLabelsChanged();
        log(`Pose: ${state.currentLabels.pose || 'none'}`);
    });
});

// =============================================================================
// Finger State Selection Event Handlers
// =============================================================================

document.querySelectorAll('.finger-state-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const finger = btn.dataset.finger;
        const newState = btn.dataset.state;

        // Toggle - if already selected, set to unknown
        if (state.currentLabels.fingers[finger] === newState) {
            state.currentLabels.fingers[finger] = 'unknown';
        } else {
            state.currentLabels.fingers[finger] = newState;
        }

        // Update UI - clear all for this finger, then set active
        document.querySelectorAll(`.finger-state-btn[data-finger="${finger}"]`).forEach(b => {
            b.classList.remove('active');
        });
        if (state.currentLabels.fingers[finger] !== 'unknown') {
            btn.classList.add('active');
        }

        onLabelsChanged();
        updateHandVisualizer();
        log(`Finger ${finger}: ${state.currentLabels.fingers[finger]}`);
    });
});

// =============================================================================
// Motion State Selection Event Handlers
// =============================================================================

document.querySelectorAll('[data-motion]').forEach(btn => {
    btn.addEventListener('click', () => {
        state.currentLabels.motion = btn.dataset.motion;

        document.querySelectorAll('[data-motion]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        onLabelsChanged();
        log(`Motion: ${state.currentLabels.motion}`);
    });
});

// =============================================================================
// Calibration Selection Event Handlers
// =============================================================================

document.querySelectorAll('[data-calibration]').forEach(btn => {
    btn.addEventListener('click', () => {
        const cal = btn.dataset.calibration;

        // Toggle - if already selected, set to none
        if (state.currentLabels.calibration === cal) {
            state.currentLabels.calibration = 'none';
        } else {
            state.currentLabels.calibration = cal;
        }

        document.querySelectorAll('[data-calibration]').forEach(b => b.classList.remove('active'));
        if (state.currentLabels.calibration !== 'none') {
            btn.classList.add('active');
        }

        onLabelsChanged();
        log(`Calibration: ${state.currentLabels.calibration}`);
    });
});

// =============================================================================
// Label Management
// =============================================================================

function closeCurrentLabel() {
    if (state.currentLabelStart !== null && state.sessionData.length > state.currentLabelStart) {
        const labelsObj = getCurrentLabelsObject();
        state.labels.push({
            start_sample: state.currentLabelStart,
            end_sample: state.sessionData.length,
            labels: labelsObj,
            notes: '',
            _version: 2
        });
        log(`Label segment: ${state.currentLabelStart}-${state.sessionData.length}`);
    }
}

window.deleteLabel = function(index) {
    state.labels.splice(index, 1);
    updateUI();
};

// =============================================================================
// Data Handling - Telemetry
// =============================================================================

function onTelemetry(telemetry) {
    if (!state.recording) return;

    // IMPORTANT: Preserve raw data, only DECORATE with processed fields
    // Create a decorated copy of telemetry with additional processed fields
    const decoratedTelemetry = {...telemetry};

    // Calculate time step for IMU fusion
    const now = performance.now();
    const dt = lastTelemetryTime ? (now - lastTelemetryTime) / 1000 : 0.02; // Default 50Hz
    lastTelemetryTime = now;

    // Update IMU sensor fusion to estimate device orientation
    // Uses accelerometer + gyroscope (NOT magnetometer - it's our measurement target)
    if (telemetry.ax !== undefined && telemetry.gx !== undefined) {
        if (!imuInitialized && Math.abs(telemetry.ax) + Math.abs(telemetry.ay) + Math.abs(telemetry.az) > 0.5) {
            // Initialize orientation from accelerometer (assumes stationary)
            imuFusion.initFromAccelerometer(telemetry.ax, telemetry.ay, telemetry.az);
            imuInitialized = true;
        }
        // Update orientation estimate
        imuFusion.update(
            telemetry.ax, telemetry.ay, telemetry.az,  // Accelerometer
            telemetry.gx, telemetry.gy, telemetry.gz,  // Gyroscope
            dt,                                         // Time step
            true                                        // Gyro is in deg/s
        );
    }

    // Get current orientation for Earth field subtraction
    const orientation = imuInitialized ? imuFusion.getQuaternion() : null;
    const euler = imuInitialized ? imuFusion.getEulerAngles() : null;

    // Add orientation to telemetry
    if (orientation) {
        decoratedTelemetry.orientation_w = orientation.w;
        decoratedTelemetry.orientation_x = orientation.x;
        decoratedTelemetry.orientation_y = orientation.y;
        decoratedTelemetry.orientation_z = orientation.z;
        decoratedTelemetry.euler_roll = euler.roll;
        decoratedTelemetry.euler_pitch = euler.pitch;
        decoratedTelemetry.euler_yaw = euler.yaw;
    }

    // Apply calibration correction (adds calibrated_ fields - iron correction only)
    if (calibrationInstance &&
        calibrationInstance.hardIronCalibrated &&
        calibrationInstance.softIronCalibrated) {
        try {
            // Iron correction only (no Earth field subtraction yet)
            const ironCorrected = calibrationInstance.correctIronOnly({
                x: telemetry.mx,
                y: telemetry.my,
                z: telemetry.mz
            });
            decoratedTelemetry.calibrated_mx = ironCorrected.x;
            decoratedTelemetry.calibrated_my = ironCorrected.y;
            decoratedTelemetry.calibrated_mz = ironCorrected.z;

            // Full correction with Earth field subtraction (requires orientation)
            if (calibrationInstance.earthFieldCalibrated && orientation) {
                // Create Quaternion object for calibration.correct()
                const quatOrientation = new Quaternion(
                    orientation.w, orientation.x, orientation.y, orientation.z
                );
                const fused = calibrationInstance.correct(
                    { x: telemetry.mx, y: telemetry.my, z: telemetry.mz },
                    quatOrientation
                );
                decoratedTelemetry.fused_mx = fused.x;
                decoratedTelemetry.fused_my = fused.y;
                decoratedTelemetry.fused_mz = fused.z;
            }
        } catch (e) {
            // Calibration failed, skip decoration
            console.debug('[Calibration] Correction failed:', e.message);
        }
    }

    // Apply Kalman filtering (adds filtered_ fields)
    // Use best available source: fused > calibrated > raw
    try {
        const magInput = {
            x: decoratedTelemetry.fused_mx || decoratedTelemetry.calibrated_mx || telemetry.mx,
            y: decoratedTelemetry.fused_my || decoratedTelemetry.calibrated_my || telemetry.my,
            z: decoratedTelemetry.fused_mz || decoratedTelemetry.calibrated_mz || telemetry.mz
        };
        const filteredMag = magFilter.update(magInput);
        decoratedTelemetry.filtered_mx = filteredMag.x;
        decoratedTelemetry.filtered_my = filteredMag.y;
        decoratedTelemetry.filtered_mz = filteredMag.z;

        // Update pose estimation with filtered magnetic field
        if (poseEstimationEnabled) {
            updatePoseEstimation({
                x: filteredMag.x,
                y: filteredMag.y,
                z: filteredMag.z
            });
        }
    } catch (e) {
        // Filtering failed, skip decoration
    }

    // Store decorated telemetry (includes raw + processed fields)
    state.sessionData.push(decoratedTelemetry);

    // Collect samples for calibration buffers during wizard
    if (wizard.active && wizard.phase === 'hold') {
        const currentStep = wizard.steps[wizard.currentStep];
        if (currentStep && calibrationBuffers[currentStep.id]) {
            calibrationBuffers[currentStep.id].push({
                mx: telemetry.mx,
                my: telemetry.my,
                mz: telemetry.mz
            });
        }
    }

    // Update live display (show calibrated values if available, otherwise raw)
    $('ax').textContent = telemetry.ax;
    $('ay').textContent = telemetry.ay;
    $('az').textContent = telemetry.az;
    $('gx').textContent = telemetry.gx;
    $('gy').textContent = telemetry.gy;
    $('gz').textContent = telemetry.gz;
    $('mx').textContent = (decoratedTelemetry.calibrated_mx || telemetry.mx).toFixed(2);
    $('my').textContent = (decoratedTelemetry.calibrated_my || telemetry.my).toFixed(2);
    $('mz').textContent = (decoratedTelemetry.calibrated_mz || telemetry.mz).toFixed(2);

    // Update sample count (throttled)
    if (state.sessionData.length % 10 === 0) {
        updateUI();
    }
}

// =============================================================================
// Connection
// =============================================================================

connectBtn.addEventListener('click', async () => {
    if (state.connected) {
        console.log('[GAMBIT] Disconnecting...');
        if (state.gambitClient) {
            state.gambitClient.disconnect();
            state.gambitClient = null;
        }
        state.connected = false;
        log('Disconnected');
        updateUI();
        updateCalibrationStatus();
        return;
    }

    log('Connecting...');

    try {
        state.gambitClient = new GambitClient({
            debug: true,
            autoKeepalive: false
        });

        state.gambitClient.on('data', onTelemetry);

        state.gambitClient.on('firmware', (info) => {
            console.log('[GAMBIT] Firmware info:', info);
            // Check compatibility with minimum version 0.1.0
            const compat = state.gambitClient.checkCompatibility('0.1.0');
            if (!compat.compatible) {
                log(`Incompatible firmware: ${compat.reason}`);
                setTimeout(() => state.gambitClient.disconnect(), 3000);
                return;
            }
            log(`Firmware: ${info.name} v${info.version}`);
        });

        state.gambitClient.on('disconnect', () => {
            console.log('[GAMBIT] Device disconnected');
            state.connected = false;
            state.recording = false;
            log('Connection closed');
            updateUI();
            updateCalibrationStatus();
        });

        state.gambitClient.on('error', (err) => {
            console.error('[GAMBIT] Error:', err);
            log(`Error: ${err.message}`);
        });

        await state.gambitClient.connect();

        state.connected = true;
        log('Connected!');
        updateUI();
        updateCalibrationStatus();

    } catch (e) {
        console.error('[GAMBIT] Connection error:', e);
        log(`Connection failed: ${e.message}`);
        if (state.gambitClient) {
            state.gambitClient.disconnect();
            state.gambitClient = null;
        }
    }
});

// =============================================================================
// Calibration Wizard UI
// =============================================================================

function updateCalibrationStatus() {
    const statusText = $('calStatusText');
    const detailsText = $('calDetails');
    const saveBtn = $('saveCalibration');

    if (!calibrationInstance) {
        statusText.textContent = 'Not Initialized';
        detailsText.textContent = 'Calibration module not loaded';
        saveBtn.disabled = true;
        return;
    }

    // Use the class's calibration flags (set by the calibration methods)
    const hasEarth = calibrationInstance.earthFieldCalibrated;
    const hasHardIron = calibrationInstance.hardIronCalibrated;
    const hasSoftIron = calibrationInstance.softIronCalibrated;
    const complete = hasEarth && hasHardIron && hasSoftIron;

    statusText.textContent = complete ? 'Calibrated' : (hasEarth || hasHardIron || hasSoftIron) ? 'Partial' : 'Not Calibrated';
    statusText.style.color = complete ? 'var(--success)' : (hasEarth || hasHardIron || hasSoftIron) ? 'var(--warning)' : 'var(--fg-muted)';

    const steps = [];
    if (hasEarth) steps.push('Earth');
    if (hasHardIron) steps.push('Hard Iron');
    if (hasSoftIron) steps.push('Soft Iron');
    detailsText.textContent = steps.length ? `Complete: ${steps.join(', ')}` : 'No calibration data';

    saveBtn.disabled = !complete;

    $('startEarthCal').disabled = !state.connected;
    $('startHardIronCal').disabled = !state.connected || !hasEarth;
    $('startSoftIronCal').disabled = !state.connected || !hasHardIron;
}

function runCalibrationStep(stepName, durationMs, sampleHandler, completionHandler) {
    if (!state.gambitClient || !state.connected) {
        log('Error: Not connected');
        return;
    }

    const buffer = [];
    const startTime = Date.now();
    const progressDiv = $(`${stepName}Progress`);
    const qualityDiv = $(`${stepName}Quality`);

    log(`Starting ${stepName} calibration (${durationMs/1000}s)...`);

    const dataHandler = (sample) => {
        buffer.push(sample);
        const elapsed = Date.now() - startTime;
        const progress = Math.min(100, (elapsed / durationMs) * 100);
        progressDiv.textContent = `${progress.toFixed(0)}%`;
    };

    state.gambitClient.on('data', dataHandler);

    setTimeout(() => {
        state.gambitClient.off('data', dataHandler);
        progressDiv.textContent = 'Done';

        if (buffer.length < 10) {
            log(`Error: Insufficient samples (${buffer.length})`);
            qualityDiv.textContent = '❌ Failed: insufficient data';
            qualityDiv.style.color = 'var(--danger)';
            return;
        }

        const result = completionHandler(buffer);
        sampleHandler(result);

        log(`${stepName} complete: ${buffer.length} samples`);
    }, durationMs);
}

$('startEarthCal').addEventListener('click', () => {
    runCalibrationStep('earth', 5000,
        (result) => {
            const quality = result.quality > 0.9 ? 'Excellent' : result.quality > 0.7 ? 'Good' : 'Poor';
            const emoji = result.quality > 0.9 ? '✅' : result.quality > 0.7 ? '⚠️' : '❌';
            $('earthQuality').textContent = `${emoji} ${quality} (quality: ${result.quality.toFixed(2)})`;
            $('earthQuality').style.color = result.quality > 0.7 ? 'var(--success)' : 'var(--danger)';

            // Save after each step
            calibrationInstance.save('gambit_calibration');
            updateCalibrationStatus();
        },
        (buffer) => {
            // Convert buffer to samples format expected by the class
            const samples = buffer.map(s => ({x: s.mx, y: s.my, z: s.mz}));
            return calibrationInstance.runEarthFieldCalibration(samples);
        }
    );
});

$('startHardIronCal').addEventListener('click', () => {
    runCalibrationStep('hardIron', 10000,
        (result) => {
            const quality = result.quality > 0.9 ? 'Excellent' : result.quality > 0.7 ? 'Good' : 'Poor';
            const emoji = result.quality > 0.9 ? '✅' : result.quality > 0.7 ? '⚠️' : '❌';
            $('hardIronQuality').textContent = `${emoji} ${quality} (sphericity: ${result.quality.toFixed(2)})`;
            $('hardIronQuality').style.color = result.quality > 0.7 ? 'var(--success)' : 'var(--danger)';

            // Save after each step
            calibrationInstance.save('gambit_calibration');
            updateCalibrationStatus();
        },
        (buffer) => {
            // Convert buffer to samples format expected by the class
            const samples = buffer.map(s => ({x: s.mx, y: s.my, z: s.mz}));
            return calibrationInstance.runHardIronCalibration(samples);
        }
    );
});

$('startSoftIronCal').addEventListener('click', () => {
    runCalibrationStep('softIron', 10000,
        (result) => {
            const quality = result.quality > 0.9 ? 'Excellent' : result.quality > 0.7 ? 'Good' : 'Poor';
            const emoji = result.quality > 0.9 ? '✅' : result.quality > 0.7 ? '⚠️' : '❌';
            $('softIronQuality').textContent = `${emoji} ${quality} (quality: ${result.quality.toFixed(2)})`;
            $('softIronQuality').style.color = result.quality > 0.7 ? 'var(--success)' : 'var(--danger)';

            // Save complete calibration
            calibrationInstance.save('gambit_calibration');
            updateCalibrationStatus();
            log('Calibration saved to localStorage');
        },
        (buffer) => {
            // Convert buffer to samples format expected by the class
            const samples = buffer.map(s => ({x: s.mx, y: s.my, z: s.mz}));
            return calibrationInstance.runSoftIronCalibration(samples);
        }
    );
});

$('saveCalibration').addEventListener('click', () => {
    // Use the class's toJSON method for consistent format
    const calData = {
        ...calibrationInstance.toJSON(),
        timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(calData, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gambit_calibration.json';
    a.click();
    URL.revokeObjectURL(url);

    log('Calibration exported to file');
});

$('loadCalibration').addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (evt) => {
            try {
                const data = JSON.parse(evt.target.result);
                // Use the class's fromJSON method
                calibrationInstance = EnvironmentalCalibration.fromJSON(data);
                // Also save to localStorage
                calibrationInstance.save('gambit_calibration');
                updateCalibrationStatus();
                log(`Loaded calibration from ${file.name}`);
            } catch (err) {
                log(`Error loading calibration: ${err.message}`);
            }
        };
        reader.readAsText(file);
    };
    input.click();
});

$('resetCalibration').addEventListener('click', () => {
    if (confirm('Reset all calibration data?')) {
        // Create fresh instance with default values
        calibrationInstance = new EnvironmentalCalibration();
        localStorage.removeItem('gambit_calibration');
        updateCalibrationStatus();
        $('earthQuality').textContent = '';
        $('hardIronQuality').textContent = '';
        $('softIronQuality').textContent = '';
        log('Calibration reset');
    }
});

// Update calibration status UI (calibrationInstance already initialized earlier)
updateCalibrationStatus();

// =============================================================================
// Recording Controls
// =============================================================================

startBtn.addEventListener('click', async () => {
    if (!state.gambitClient || !state.connected) {
        log('Error: Not connected to device');
        return;
    }

    try {
        state.recording = true;
        state.currentLabelStart = state.sessionData.length;
        log('Recording started');

        await state.gambitClient.startStreaming();
        log('Data collection active');

        updateUI();
    } catch (e) {
        console.error('[GAMBIT] Failed to start recording:', e);
        log('Error: Failed to start data collection');
        state.recording = false;
        updateUI();
    }
});

stopBtn.addEventListener('click', async () => {
    closeCurrentLabel();
    state.recording = false;
    state.currentLabelStart = null;
    log('Recording stopped');

    if (state.gambitClient && state.connected) {
        try {
            await state.gambitClient.stopStreaming();
            log('Data streaming stopped');
        } catch (e) {
            console.error('[GAMBIT] Failed to stop streaming:', e);
        }
    }

    updateUI();
});

clearBtn.addEventListener('click', () => {
    if (confirm('Clear all session data?')) {
        state.sessionData = [];
        state.labels = [];
        state.currentLabelStart = null;
        log('Session cleared');
        updateUI();
    }
});

// =============================================================================
// Magnet Config
// =============================================================================

function getMagnetConfig() {
    const configType = $('magnetConfig').value;
    const magnetType = $('magnetType').value;

    if (configType === 'none') return null;

    const config = {
        magnet_type: magnetType
    };

    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    const alternating = ['north_palm', 'south_palm', 'north_palm', 'south_palm', 'north_palm'];

    fingers.forEach((f, i) => {
        config[f] = { present: false, polarity: 'unknown' };
    });

    if (configType === 'single_index') {
        config.index = { present: true, polarity: 'north_palm' };
    } else if (configType === 'single_thumb') {
        config.thumb = { present: true, polarity: 'north_palm' };
    } else if (configType === 'alternating') {
        fingers.forEach((f, i) => {
            config[f] = { present: true, polarity: alternating[i] };
        });
    }

    return config;
}

// =============================================================================
// Export
// =============================================================================

exportBtn.addEventListener('click', () => {
    const timestamp = new Date().toISOString();

    // Create data file
    const dataBlob = new Blob([JSON.stringify(state.sessionData)], { type: 'application/json' });
    const dataUrl = URL.createObjectURL(dataBlob);

    // Create metadata file (V2 format)
    const metadata = {
        timestamp: timestamp,
        subject_id: $('subjectId').value,
        environment: $('environment').value,
        hand: $('hand').value,
        split: $('split').value,
        device_id: 'puck_default',
        labels: [],  // V1 labels empty for backwards compat
        labels_v2: state.labels,
        session_notes: $('sessionNotes').value,
        sample_rate_hz: 50,
        magnet_config: getMagnetConfig(),
        custom_label_definitions: state.customLabelDefinitions
    };
    const metaBlob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' });
    const metaUrl = URL.createObjectURL(metaBlob);

    // Download data file
    const dataLink = document.createElement('a');
    dataLink.href = dataUrl;
    dataLink.download = `${timestamp}.json`;
    dataLink.click();

    // Download metadata file
    setTimeout(() => {
        const metaLink = document.createElement('a');
        metaLink.href = metaUrl;
        metaLink.download = `${timestamp}.meta.json`;
        metaLink.click();
        log(`Exported: ${timestamp}`);
    }, 100);
});

// =============================================================================
// GitHub Upload
// =============================================================================

// Load saved token
try {
    const savedToken = localStorage.getItem('gambit_gh_token');
    if (savedToken) {
        ghToken = savedToken;
        $('ghToken').value = savedToken;
    }
} catch (e) {
    console.error('Failed to load GitHub token:', e);
}

// Save token on change
$('ghToken').addEventListener('change', (e) => {
    ghToken = e.target.value;
    if (ghToken) {
        localStorage.setItem('gambit_gh_token', ghToken);
        log('GitHub token saved');
    } else {
        localStorage.removeItem('gambit_gh_token');
    }
    updateUI();
});

// Base64 encode for GitHub API
function b64EncodeUnicode(str) {
    return btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, function(match, p1) {
        return String.fromCharCode('0x' + p1);
    }));
}

// GitHub API endpoint
function getGHEndpoint(path) {
    return `https://api.github.com/repos/christopherdebeer/simcap/contents/${path}`;
}

// Upload file to GitHub
async function ghPutFile(path, content, message) {
    if (!ghToken) {
        throw new Error('GitHub token not set');
    }

    const endpoint = getGHEndpoint(path);

    // Check if file exists (to get SHA for update)
    let sha = null;
    try {
        const getResp = await fetch(endpoint, {
            method: 'GET',
            headers: {
                'Authorization': `token ${ghToken}`,
                'Content-Type': 'application/json'
            }
        });
        if (getResp.ok) {
            const existing = await getResp.json();
            sha = existing.sha;
        }
    } catch (e) {
        // File doesn't exist, that's fine
    }

    // Create/update file
    const body = {
        message: message,
        content: b64EncodeUnicode(content)
    };
    if (sha) {
        body.sha = sha;
    }

    const putResp = await fetch(endpoint, {
        method: 'PUT',
        headers: {
            'Authorization': `token ${ghToken}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    });

    if (!putResp.ok) {
        const error = await putResp.json();
        throw new Error(error.message || 'Upload failed');
    }

    return await putResp.json();
}

// Upload button handler
const uploadBtn = $('uploadBtn');

uploadBtn.addEventListener('click', async () => {
    if (!ghToken) {
        log('Error: GitHub token not set');
        return;
    }

    if (state.sessionData.length === 0) {
        log('Error: No data to upload');
        return;
    }

    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';

    try {
        const timestamp = new Date().toISOString();
        const basePath = `data/GAMBIT/labeled/${timestamp}`;

        // Upload data file
        log('Uploading data file...');
        await ghPutFile(
            `${basePath}.json`,
            JSON.stringify(state.sessionData),
            `GAMBIT labeled data: ${timestamp}`
        );

        // Upload metadata file
        log('Uploading metadata...');
        const metadata = {
            timestamp: timestamp,
            subject_id: $('subjectId').value,
            environment: $('environment').value,
            hand: $('hand').value,
            split: $('split').value,
            device_id: 'puck_default',
            labels: [],
            labels_v2: state.labels,
            session_notes: $('sessionNotes').value,
            sample_rate_hz: 50,
            magnet_config: getMagnetConfig(),
            custom_label_definitions: state.customLabelDefinitions
        };

        await ghPutFile(
            `${basePath}.meta.json`,
            JSON.stringify(metadata, null, 2),
            `GAMBIT labeled metadata: ${timestamp}`
        );

        log(`Uploaded to GitHub: ${timestamp}`);

    } catch (e) {
        console.error('Upload failed:', e);
        log(`Upload failed: ${e.message}`);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload to GitHub';
        updateUI();
    }
});

// =============================================================================
// Collapsible Sections
// =============================================================================

document.querySelectorAll('.collapsible').forEach(header => {
    header.addEventListener('click', () => {
        header.classList.toggle('collapsed');
        // Find the collapse-content within the same section (may not be immediate sibling)
        const section = header.closest('section');
        const content = section ? section.querySelector('.collapse-content') : header.nextElementSibling;
        if (content && content.classList.contains('collapse-content')) {
            content.classList.toggle('hidden');
        }
    });
});

// =============================================================================
// Custom label input enter key
// =============================================================================

$('customLabelInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        addCustomLabel();
    }
});

// =============================================================================
// Data Collection Wizard
// =============================================================================

const wizard = {
    active: false,
    mode: null, // 'quick' or 'full'
    currentStep: 0,
    totalSteps: 0,
    paused: false,
    countdown: 0,
    countdownInterval: null,
    startSamples: 0,
    startLabels: 0,
    steps: []
};

// Wizard step definitions
// Each step has: transition time (unlabeled) + hold time (labeled)
const TRANSITION_TIME = 5; // seconds of unlabeled transition
const HOLD_TIME = 3; // seconds of labeled hold
const HOLD_TIME_MED = 6; // seconds of labeled hold
const HOLD_TIME_LONG = 10; // seconds of labeled hold

const WIZARD_STEPS = {
    // Reference poses for data collection (calibration is handled separately by manual wizard)
    reference: [
        { id: 'reference_pose', label: 'Reference Pose', icon: '✋', transition: TRANSITION_TIME, hold: HOLD_TIME_LONG, desc: 'Hold hand flat, palm down, fingers together.' },
        { id: 'magnet_baseline', label: 'Magnet Baseline', icon: '📍', transition: TRANSITION_TIME, hold: HOLD_TIME_LONG, desc: 'Keep hand in reference pose with magnets attached.' }
    ],
    fingers: [
        { id: 'finger_isolation:thumb', label: 'Thumb Isolation', icon: '👍', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Move only your thumb through full range of motion.' },
        { id: 'finger_isolation:index', label: 'Index Isolation', icon: '☝️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Move only your index finger through full range.' },
        { id: 'finger_isolation:middle', label: 'Middle Isolation', icon: '🖕', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Move only your middle finger through full range.' },
        { id: 'finger_isolation:ring', label: 'Ring Isolation', icon: '💍', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Move only your ring finger through full range.' },
        { id: 'finger_isolation:pinky', label: 'Pinky Isolation', icon: '🤙', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Move only your pinky finger through full range.' }
    ],
    // Finger tracking: All 5 magnets attached (standard configuration)
    fingerTracking5Mag: [
        { id: 'ft5:reference', label: 'Reference (5 mag)', icon: '✋', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Palm down, all fingers extended. All 5 magnets attached.' },
        { id: 'ft5:all_extended', label: 'All Extended', icon: '🖐️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Spread all fingers wide, fully extended.' },
        { id: 'ft5:all_flexed', label: 'All Flexed (Fist)', icon: '✊', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Make a tight fist, all fingers flexed.' },
        { id: 'ft5:thumb_flex', label: 'Thumb Flex Only', icon: '👍', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex only thumb, others extended.' },
        { id: 'ft5:index_flex', label: 'Index Flex Only', icon: '☝️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex only index finger, others extended.' },
        { id: 'ft5:middle_flex', label: 'Middle Flex Only', icon: '🖕', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex only middle finger, others extended.' },
        { id: 'ft5:ring_flex', label: 'Ring Flex Only', icon: '💍', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex only ring finger, others extended.' },
        { id: 'ft5:pinky_flex', label: 'Pinky Flex Only', icon: '🤙', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex only pinky finger, others extended.' },
        { id: 'ft5:thumb_index', label: 'Thumb+Index Flex', icon: '🤏', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex thumb and index only.' },
        { id: 'ft5:ring_pinky', label: 'Ring+Pinky Flex', icon: '🤟', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex ring and pinky only.' },
        { id: 'ft5:middle_ring_pinky', label: 'Mid+Ring+Pinky Flex', icon: '✌️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex middle, ring, pinky (peace sign).' }
    ],
    // Finger tracking: No magnets (baseline noise floor)
    fingerTrackingNoMag: [
        { id: 'ft0:reference', label: 'Reference (no mag)', icon: '✋', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Palm down, all fingers extended. NO magnets attached.' },
        { id: 'ft0:all_extended', label: 'All Extended', icon: '🖐️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Spread all fingers wide.' },
        { id: 'ft0:all_flexed', label: 'All Flexed (Fist)', icon: '✊', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Make a tight fist.' },
        { id: 'ft0:index_flex', label: 'Index Flex Only', icon: '☝️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex only index finger.' },
        { id: 'ft0:thumb_flex', label: 'Thumb Flex Only', icon: '👍', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Flex only thumb.' },
        { id: 'ft0:random_motion', label: 'Random Motion', icon: '👋', transition: TRANSITION_TIME, hold: HOLD_TIME_LONG, desc: 'Move fingers randomly to capture baseline noise.' }
    ],
    // Finger tracking: Index magnet only (proof-of-concept)
    fingerTrackingIndexOnly: [
        { id: 'ft1:reference', label: 'Reference (index mag)', icon: '✋', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Palm down, all extended. Magnet on INDEX ONLY.' },
        { id: 'ft1:index_extended', label: 'Index Extended', icon: '☝️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Index finger fully extended (far from sensor).' },
        { id: 'ft1:index_partial', label: 'Index Partial Flex', icon: '👆', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Index finger half-flexed (mid-range).' },
        { id: 'ft1:index_flexed', label: 'Index Fully Flexed', icon: '👊', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Index finger fully flexed (close to palm).' },
        { id: 'ft1:index_sweep_slow', label: 'Index Slow Sweep', icon: '🔄', transition: TRANSITION_TIME, hold: HOLD_TIME_LONG, desc: 'Slowly flex and extend index finger repeatedly.' },
        { id: 'ft1:index_sweep_fast', label: 'Index Fast Sweep', icon: '⚡', transition: TRANSITION_TIME, hold: HOLD_TIME_LONG, desc: 'Quickly flex and extend index finger repeatedly.' },
        { id: 'ft1:index_abduction', label: 'Index Abduction', icon: '↔️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Move index finger side-to-side (abduction/adduction).' },
        { id: 'ft1:other_fingers', label: 'Other Fingers Motion', icon: '🖐️', transition: TRANSITION_TIME, hold: HOLD_TIME_MED, desc: 'Move other fingers while index stays still (interference test).' }
    ],
    asl: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map(letter => ({
        id: `asl:${letter.toLowerCase()}`,
        label: `Letter ${letter}`,
        letter: letter,
        transition: TRANSITION_TIME,
        hold: HOLD_TIME,
        desc: `Form ASL letter ${letter} and hold steady.`
    })),
    gestures: [
        { id: 'pose:fist', label: 'Fist', icon: '✊', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Make a tight fist.' },
        { id: 'pose:open_palm', label: 'Open Palm', icon: '🖐️', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Spread all fingers wide.' },
        { id: 'pose:pinch', label: 'Pinch', icon: '🤏', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Touch thumb and index fingertips.' },
        { id: 'pose:point', label: 'Point', icon: '👆', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Point with index finger, others closed.' },
        { id: 'pose:thumbs_up', label: 'Thumbs Up', icon: '👍', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Classic thumbs up gesture.' },
        { id: 'pose:ok', label: 'OK Sign', icon: '👌', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Form OK sign with thumb and index.' }
    ]
};

function openWizard() {
    $('wizardOverlay').classList.add('active');
    wizard.active = true;
    wizard.startSamples = state.sessionData.length;
    wizard.startLabels = state.labels.length;
    showWizardModeSelection();
}

window.closeWizard = function() {
    if (wizard.countdownInterval) {
        clearInterval(wizard.countdownInterval);
        wizard.countdownInterval = null;
    }
    if (wizard.active && state.recording) {
        closeCurrentLabel();
        state.recording = false;
        state.currentLabelStart = null;
        if (state.gambitClient && state.connected) {
            state.gambitClient.stopStreaming().catch(() => {});
        }
    }
    $('wizardOverlay').classList.remove('active');
    wizard.active = false;
    wizard.paused = false;
    updateUI();
};

function showWizardModeSelection() {
    $('wizardTitle').textContent = 'Data Collection Wizard';
    $('wizardPhase').textContent = 'Select collection mode';
    $('wizardProgressFill').style.width = '0%';
    $('wizardStepText').textContent = 'Step 0 of 0';
    $('wizardTimeText').textContent = '';
    $('wizardStats').style.display = 'none';

    $('wizardContent').innerHTML = `
        <div class="wizard-instruction">Choose a data collection mode</div>

        <div style="font-size: 11px; color: var(--accent); margin: 10px 0 5px; font-weight: 500;">General Collection</div>
        <div class="wizard-start-options">
            <div class="wizard-option" onclick="startWizard('quick')">
                <h4>⚡ Quick Collection</h4>
                <p>Reference poses + finger isolation</p>
                <div class="duration">~50 seconds • 7 steps</div>
            </div>
            <div class="wizard-option" onclick="startWizard('full')">
                <h4>📚 Full Collection</h4>
                <p>Includes ASL alphabet (A-Z) + gestures</p>
                <div class="duration">~210 seconds • 39 steps</div>
            </div>
        </div>

        <div style="font-size: 11px; color: var(--accent); margin: 15px 0 5px; font-weight: 500;">Magnetic Finger Tracking</div>
        <div class="wizard-start-options">
            <div class="wizard-option" onclick="startWizard('ft5mag')" style="border-color: var(--success);">
                <h4>🧲 5 Magnets (Standard)</h4>
                <p>All fingers with magnets - full tracking data</p>
                <div class="duration">~85 seconds • 11 steps</div>
            </div>
            <div class="wizard-option" onclick="startWizard('ft1mag')" style="border-color: var(--warning);">
                <h4>☝️ Index Only (PoC)</h4>
                <p>Single magnet on index - proof of concept</p>
                <div class="duration">~65 seconds • 8 steps</div>
            </div>
            <div class="wizard-option" onclick="startWizard('ft0mag')" style="border-color: var(--fg-muted);">
                <h4>📊 No Magnets (Baseline)</h4>
                <p>Captures noise floor without magnets</p>
                <div class="duration">~45 seconds • 6 steps</div>
            </div>
        </div>

        <div style="margin-top: 15px; padding: 10px; background: var(--bg); border-radius: 4px; font-size: 11px; color: var(--fg-muted);">
            <strong>Note:</strong> Complete magnetometer calibration before collecting finger tracking data. Calibration removes environmental interference to isolate magnet signals.
        </div>
    `;
}

window.startWizard = async function(mode) {
    wizard.mode = mode;
    wizard.currentStep = 0;
    wizard.paused = false;

    // Build step list based on mode
    switch (mode) {
        case 'quick':
            wizard.steps = [
                ...WIZARD_STEPS.reference,
                ...WIZARD_STEPS.fingers
            ];
            break;
        case 'full':
            wizard.steps = [
                ...WIZARD_STEPS.reference,
                ...WIZARD_STEPS.fingers,
                ...WIZARD_STEPS.asl,
                ...WIZARD_STEPS.gestures
            ];
            break;
        case 'ft5mag':
            // 5-magnet finger tracking (standard configuration)
            wizard.steps = [...WIZARD_STEPS.fingerTracking5Mag];
            break;
        case 'ft1mag':
            // Index-only magnet (proof of concept)
            wizard.steps = [...WIZARD_STEPS.fingerTrackingIndexOnly];
            break;
        case 'ft0mag':
            // No magnets (baseline noise floor)
            wizard.steps = [...WIZARD_STEPS.fingerTrackingNoMag];
            break;
        default:
            wizard.steps = [...WIZARD_STEPS.reference, ...WIZARD_STEPS.fingers];
    }

    wizard.totalSteps = wizard.steps.length;
    $('wizardStats').style.display = 'flex';

    // Start recording if not already
    if (!state.recording && state.connected && state.gambitClient) {
        try {
            state.recording = true;
            state.currentLabelStart = state.sessionData.length;
            await state.gambitClient.startStreaming();
            log('Wizard: Recording started');
        } catch (e) {
            log('Wizard: Failed to start recording');
            closeWizard();
            return;
        }
    }

    runWizardStep();
};

function runWizardStep() {
    if (wizard.currentStep >= wizard.totalSteps) {
        showWizardComplete();
        return;
    }

    const step = wizard.steps[wizard.currentStep];
    const progress = ((wizard.currentStep) / wizard.totalSteps) * 100;
    const remainingTime = wizard.steps.slice(wizard.currentStep).reduce((sum, s) => sum + s.transition + s.hold, 0);

    // Update progress
    $('wizardProgressFill').style.width = `${progress}%`;
    $('wizardStepText').textContent = `Step ${wizard.currentStep + 1} of ${wizard.totalSteps}`;
    $('wizardTimeText').textContent = `~${remainingTime}s remaining`;

    // Determine phase label
    let phaseLabel = 'Reference Poses';
    if (step.id.startsWith('finger_isolation')) phaseLabel = 'Finger Isolation';
    else if (step.id.startsWith('asl:')) phaseLabel = 'ASL Alphabet';
    else if (step.id.startsWith('pose:')) phaseLabel = 'Common Gestures';
    else if (step.id.startsWith('ft5:')) phaseLabel = 'Finger Tracking (5 Magnets)';
    else if (step.id.startsWith('ft1:')) phaseLabel = 'Finger Tracking (Index Only)';
    else if (step.id.startsWith('ft0:')) phaseLabel = 'Baseline (No Magnets)';
    $('wizardPhase').textContent = phaseLabel;

    // Start with TRANSITION phase (unlabeled)
    // Close any previous label and ensure no labels are active during transition
    closeCurrentLabel();
    state.currentLabels.calibration = 'none';
    state.currentLabels.custom = state.currentLabels.custom.filter(c =>
        !c.startsWith('asl:') && !c.startsWith('pose:') && !c.startsWith('finger_isolation') &&
        !c.startsWith('ft5:') && !c.startsWith('ft1:') && !c.startsWith('ft0:')
    );
    state.currentLabelStart = null; // No label during transition
    updateActiveLabelsDisplay();

    // Build visual content
    let visualHTML = '';
    if (step.letter) {
        visualHTML = `<div class="asl-letter">${step.letter}</div>`;
    } else if (step.icon) {
        visualHTML = `<div class="calibration-icon">${step.icon}</div>`;
    }

    // Show transition UI
    $('wizardContent').innerHTML = `
        <div class="wizard-instruction" style="color: #ffa502;">GET READY: ${step.label}</div>
        <div class="wizard-description">${step.desc}</div>
        <div class="wizard-visual">${visualHTML}</div>
        <div class="wizard-countdown">
            <div class="countdown-circle" id="countdownCircle" style="border-color: #ffa502;">
                <svg><circle cx="44" cy="44" r="40" id="countdownRing" style="stroke: #ffa502;"></circle></svg>
                <span id="countdownNumber" style="color: #ffa502;">${step.transition}</span>
            </div>
        </div>
        <div style="color: #888; font-size: 11px; margin-top: 10px;">⏳ Transition - get into position (not recording label)</div>
        <div class="wizard-controls">
            <button class="btn-secondary" onclick="wizardPause()" id="wizardPauseBtn">⏸️ Pause</button>
            <button class="btn-warning" onclick="wizardSkip()">⏭️ Skip</button>
        </div>
        ${step.id.startsWith('asl:') ? '<button class="phase-skip-btn" onclick="wizardSkipPhase(\'asl\')">Skip entire ASL phase →</button>' : ''}
    `;

    // Start transition countdown
    wizard.countdown = step.transition;
    wizard.phase = 'transition';
    const circumference = 2 * Math.PI * 40;
    $('countdownRing').style.strokeDasharray = circumference;

    if (wizard.countdownInterval) clearInterval(wizard.countdownInterval);

    wizard.countdownInterval = setInterval(() => {
        if (wizard.paused) return;

        wizard.countdown -= 0.1;
        const countdownEl = $('countdownNumber');
        const ringEl = $('countdownRing');
        const circleEl = $('countdownCircle');

        if (countdownEl && wizard.phase === 'transition') {
            countdownEl.textContent = Math.ceil(wizard.countdown);
            const offset = circumference * (1 - wizard.countdown / step.transition);
            ringEl.style.strokeDashoffset = offset;
        }

        // Update stats
        $('wizardSamples').textContent = state.sessionData.length - wizard.startSamples;
        $('wizardLabels').textContent = state.labels.length - wizard.startLabels;

        // Transition complete - start HOLD phase (labeled)
        if (wizard.countdown <= 0 && wizard.phase === 'transition') {
            wizard.phase = 'hold';
            wizard.countdown = step.hold;

            // NOW apply the label
            if (step.id.startsWith('asl:') || step.id.startsWith('pose:') || step.id.startsWith('finger_isolation')) {
                // Add to custom labels
                if (!state.customLabelDefinitions.includes(step.id)) {
                    state.customLabelDefinitions.push(step.id);
                }
                state.currentLabels.custom.push(step.id);
                state.currentLabels.calibration = 'none';
            } else {
                // Use calibration marker
                state.currentLabels.calibration = step.id;
            }
            state.currentLabelStart = state.sessionData.length;
            updateActiveLabelsDisplay();

            // Update UI for hold phase
            $('wizardContent').innerHTML = `
                <div class="wizard-instruction" style="color: #00ff88;">HOLD: ${step.label}</div>
                <div class="wizard-description">${step.desc}</div>
                <div class="wizard-visual">${visualHTML}</div>
                <div class="wizard-countdown">
                    <div class="countdown-circle" id="countdownCircle" style="border-color: #00ff88;">
                        <svg><circle cx="44" cy="44" r="40" id="countdownRing" style="stroke: #00ff88;"></circle></svg>
                        <span id="countdownNumber" style="color: #00ff88;">${step.hold}</span>
                    </div>
                </div>
                <div style="color: #00ff88; font-size: 11px; margin-top: 10px;">🔴 RECORDING: ${step.id}</div>
                <div class="wizard-controls">
                    <button class="btn-secondary" onclick="wizardPause()" id="wizardPauseBtn">⏸️ Pause</button>
                    <button class="btn-warning" onclick="wizardSkip()">⏭️ Skip</button>
                </div>
                ${step.id.startsWith('asl:') ? '<button class="phase-skip-btn" onclick="wizardSkipPhase(\'asl\')">Skip entire ASL phase →</button>' : ''}
            `;

            $('countdownRing').style.strokeDasharray = circumference;
            $('countdownRing').style.strokeDashoffset = 0;
            return;
        }

        // Hold phase countdown
        if (wizard.phase === 'hold' && countdownEl) {
            countdownEl.textContent = Math.ceil(wizard.countdown);
            const offset = circumference * (1 - wizard.countdown / step.hold);
            if (ringEl) ringEl.style.strokeDashoffset = offset;

            // Warning color when low
            if (wizard.countdown <= 1 && circleEl) {
                circleEl.style.borderColor = '#ffa502';
                if (ringEl) ringEl.style.stroke = '#ffa502';
                countdownEl.style.color = '#ffa502';
            }
        }

        // Hold complete - move to next step
        if (wizard.countdown <= 0 && wizard.phase === 'hold') {
            clearInterval(wizard.countdownInterval);
            wizard.countdownInterval = null;

            // Close the label for this step
            closeCurrentLabel();
            state.currentLabels.custom = state.currentLabels.custom.filter(c => c !== step.id);
            state.currentLabels.calibration = 'none';
            updateActiveLabelsDisplay();

            // Log reference pose data collection (if applicable)
            if (calibrationBuffers[step.id] && calibrationBuffers[step.id].length > 0) {
                log(`${step.label}: collected ${calibrationBuffers[step.id].length} samples`);
                // Clear buffer for next run
                calibrationBuffers[step.id] = [];
            }

            wizard.currentStep++;
            runWizardStep();
        }
    }, 100);
}

window.wizardPause = function() {
    wizard.paused = !wizard.paused;
    const btn = $('wizardPauseBtn');
    if (btn) {
        btn.textContent = wizard.paused ? '▶️ Resume' : '⏸️ Pause';
    }
};

window.wizardSkip = function() {
    if (wizard.countdownInterval) {
        clearInterval(wizard.countdownInterval);
        wizard.countdownInterval = null;
    }

    const step = wizard.steps[wizard.currentStep];
    state.currentLabels.custom = state.currentLabels.custom.filter(c => c !== step.id);
    state.currentLabels.calibration = 'none';

    wizard.currentStep++;
    runWizardStep();
};

window.wizardSkipPhase = function(phase) {
    if (wizard.countdownInterval) {
        clearInterval(wizard.countdownInterval);
        wizard.countdownInterval = null;
    }

    // Skip all remaining steps of this phase
    while (wizard.currentStep < wizard.totalSteps) {
        const step = wizard.steps[wizard.currentStep];
        if (phase === 'asl' && step.id.startsWith('asl:')) {
            wizard.currentStep++;
        } else {
            break;
        }
    }

    state.currentLabels.custom = state.currentLabels.custom.filter(c => !c.startsWith('asl:'));
    state.currentLabels.calibration = 'none';

    runWizardStep();
};

function showWizardComplete() {
    closeCurrentLabel();

    const samplesCollected = state.sessionData.length - wizard.startSamples;
    const labelsCreated = state.labels.length - wizard.startLabels;

    $('wizardPhase').textContent = 'Complete!';
    $('wizardProgressFill').style.width = '100%';
    $('wizardStepText').textContent = `${wizard.totalSteps} steps completed`;
    $('wizardTimeText').textContent = '';

    $('wizardContent').innerHTML = `
        <div class="wizard-complete">
            <h3>✅ Calibration Complete!</h3>
            <div class="stats">
                <p>📊 ${samplesCollected} samples collected</p>
                <p>🏷️ ${labelsCreated} labels created</p>
            </div>
            <div class="wizard-controls">
                <button class="btn-success" onclick="closeWizard()">Done</button>
            </div>
        </div>
    `;

    log(`Wizard complete: ${samplesCollected} samples, ${labelsCreated} labels`);
}

// Wizard button handler
const wizardBtn = $('wizardBtn');
wizardBtn.addEventListener('click', () => {
    if (state.connected && !state.recording) {
        openWizard();
    } else if (state.recording) {
        log('Stop recording first to use wizard');
    }
});

// Pose estimation button handler
const poseEstimationBtn = $('poseEstimationBtn');
poseEstimationBtn.addEventListener('click', () => {
    if (!poseEstimationEnabled) {
        // Enable pose estimation
        initializePoseEstimation();
        poseEstimationBtn.textContent = '⏸️ Disable Pose Tracking';
        poseEstimationBtn.classList.remove('btn-secondary');
        poseEstimationBtn.classList.add('btn-primary');
        $('poseEstimationStatus').style.display = 'block';
        $('poseStatusText').textContent = 'Active';
        $('poseStatusText').style.color = '#00ff88';
    } else {
        // Disable pose estimation
        poseEstimationEnabled = false;
        poseState.enabled = false;
        poseEstimationBtn.textContent = '🎯 Enable Pose Tracking';
        poseEstimationBtn.classList.remove('btn-primary');
        poseEstimationBtn.classList.add('btn-secondary');
        $('poseEstimationStatus').style.display = 'none';
        $('poseStatusText').textContent = 'Disabled';
        $('poseStatusText').style.color = '#888';
        log('Pose estimation disabled');
    }
});

// Update wizard button and pose estimation button state in updateUI
const originalUpdateUI = updateUI;
updateUI = function() {
    originalUpdateUI();
    wizardBtn.disabled = !state.connected || state.recording;
    poseEstimationBtn.disabled = !state.connected || state.recording;
};

// =============================================================================
// Hand Preview Mode
// =============================================================================

// Mode: 'labels' (manual) or 'predictions' (from pose estimation)
let handPreviewMode = 'labels';

// Set hand preview mode
window.setHandPreviewMode = function(mode) {
    handPreviewMode = mode;

    const labelsBtn = $('handModeLabels');
    const predictionsBtn = $('handModePredictions');
    const indicator = $('handModeIndicator');
    const description = $('handPreviewDescription');
    const predictionInfo = $('handPredictionInfo');

    if (mode === 'labels') {
        // Labels mode
        labelsBtn.classList.remove('btn-secondary');
        labelsBtn.classList.add('btn-primary');
        predictionsBtn.classList.remove('btn-primary');
        predictionsBtn.classList.add('btn-secondary');

        indicator.textContent = '📋 Showing: Manual Labels';
        indicator.style.background = 'var(--accent)';
        indicator.style.color = 'var(--bg)';

        description.textContent = 'Visual representation of manually selected finger states';
        predictionInfo.style.display = 'none';

        // Update visualizer with current labels
        updateHandVisualizer();
    } else {
        // Predictions mode
        labelsBtn.classList.remove('btn-primary');
        labelsBtn.classList.add('btn-secondary');
        predictionsBtn.classList.remove('btn-secondary');
        predictionsBtn.classList.add('btn-primary');

        indicator.textContent = '🎯 Showing: Real-time Predictions';
        indicator.style.background = '#9b59b6';
        indicator.style.color = '#fff';

        description.textContent = 'Finger states estimated from magnetic field data (experimental)';
        predictionInfo.style.display = 'block';

        // Check if pose estimation is enabled
        if (!poseEstimationEnabled) {
            indicator.textContent = '⚠️ Predictions: Pose Tracking Disabled';
            indicator.style.background = 'var(--warning)';
            indicator.style.color = 'var(--bg)';
            $('handPredictionConfidence').textContent = 'Enable Pose Tracking first';
            $('handPredictionConfidence').style.color = 'var(--fg-muted)';
        } else {
            updateHandVisualizerFromPredictions();
        }
    }

    log(`Hand preview mode: ${mode}`);
};

// Convert 3D pose positions to discrete finger states
// Based on Z-position relative to reference pose
// Higher Z = more extended, Lower Z = more flexed
function poseToFingerStates(pose) {
    if (!pose) return null;

    const fingerStates = {};
    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];

    // Reference Z positions for each finger (from defaultReferencePose)
    // These represent "extended" position
    const referenceZ = {
        thumb: 10,
        index: 10,
        middle: 10,
        ring: 10,
        pinky: 10
    };

    // Thresholds for state classification (in mm deviation from reference)
    const EXTENDED_THRESHOLD = 5;   // Within 5mm of reference = extended
    const PARTIAL_THRESHOLD = 15;   // 5-15mm below reference = partial
    // Below 15mm = flexed

    for (const finger of fingers) {
        if (!pose[finger]) {
            fingerStates[finger] = 0; // Default to extended
            continue;
        }

        const currentZ = pose[finger].z;
        const refZ = referenceZ[finger];
        const deviation = refZ - currentZ; // Positive = finger moved down (flexed)

        if (deviation < EXTENDED_THRESHOLD) {
            fingerStates[finger] = 0; // Extended
        } else if (deviation < PARTIAL_THRESHOLD) {
            fingerStates[finger] = 1; // Partial
        } else {
            fingerStates[finger] = 2; // Flexed
        }
    }

    return fingerStates;
}

// Update hand visualizer from pose predictions
function updateHandVisualizerFromPredictions() {
    if (!handVisualizer || handPreviewMode !== 'predictions') return;
    if (!poseState.currentPose) return;

    try {
        const fingerStates = poseToFingerStates(poseState.currentPose);
        if (fingerStates) {
            handVisualizer.setFingerStates(fingerStates);
        }

        // Update confidence display in hand preview section
        const confidenceEl = $('handPredictionConfidence');
        if (confidenceEl) {
            const confidencePercent = (poseState.confidence * 100).toFixed(0);
            confidenceEl.textContent = confidencePercent + '%';

            // Color code by confidence level
            if (poseState.confidence > 0.7) {
                confidenceEl.style.color = '#00ff88'; // High confidence - green
            } else if (poseState.confidence > 0.4) {
                confidenceEl.style.color = '#ffa502'; // Medium confidence - orange
            } else {
                confidenceEl.style.color = '#e74c3c'; // Low confidence - red
            }
        }
    } catch (e) {
        console.error('Failed to update hand visualizer from predictions:', e);
    }
}

// =============================================================================
// Hand Visualizer
// =============================================================================

// Initialize hand visualizer
let handVisualizer = null;
try {
    const canvas = $('handCanvas');
    if (canvas) {
        handVisualizer = new HandVisualizer2D(canvas);
        handVisualizer.startAnimation();
        log('Hand visualizer initialized');
    }
} catch (e) {
    console.error('Failed to initialize hand visualizer:', e);
}

// Function to update hand visualizer from current finger states
function updateHandVisualizer() {
    if (!handVisualizer) return;

    const stateToNumber = (state) => {
        if (state === 'extended') return 0;
        if (state === 'partial') return 1;
        if (state === 'flexed') return 2;
        return 0; // unknown defaults to extended
    };

    try {
        handVisualizer.setFingerStates({
            thumb: stateToNumber(state.currentLabels.fingers.thumb),
            index: stateToNumber(state.currentLabels.fingers.index),
            middle: stateToNumber(state.currentLabels.fingers.middle),
            ring: stateToNumber(state.currentLabels.fingers.ring),
            pinky: stateToNumber(state.currentLabels.fingers.pinky)
        });
    } catch (e) {
        console.error('Failed to update hand visualizer:', e);
    }
}

// =============================================================================
// Initialize
// =============================================================================

loadCustomLabels();
updateActiveLabelsDisplay();
updateUI();

// Log calibration status
if (calibrationInstance &&
    calibrationInstance.earthFieldCalibrated &&
    calibrationInstance.hardIronCalibrated &&
    calibrationInstance.softIronCalibrated) {
    log('Ready. Calibration loaded. Connect device to start.');
} else {
    log('Ready. No calibration found - use Calibration Wizard after connecting.');
}
