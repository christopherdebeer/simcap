/**
 * Data Collection Wizard Module
 * Guided multi-step data collection with automatic label application
 */

// Wizard state
const wizard = {
    active: false,
    mode: null,
    currentStep: 0,
    steps: [],
    phase: null,  // 'transition' | 'hold'
    phaseStart: null
};

// Calibration buffers (for wizard functionality)
const calibrationBuffers = {};

// Module dependencies (set via setDependencies)
let deps = {
    state: null,
    startRecording: null,
    $: null,
    log: null
};

// Wizard step timing constants
const TRANSITION_TIME = 5; // seconds of unlabeled transition
const HOLD_TIME = 3; // seconds of labeled hold
const HOLD_TIME_MED = 6;
const HOLD_TIME_LONG = 10;

/**
 * Wizard step definitions
 */
const WIZARD_STEPS = {
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
    gestures: [
        { id: 'pose:fist', label: 'Fist', icon: '✊', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Make a tight fist.' },
        { id: 'pose:open_palm', label: 'Open Palm', icon: '🖐️', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Spread all fingers wide.' },
        { id: 'pose:pinch', label: 'Pinch', icon: '🤏', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Touch thumb and index fingertips.' },
        { id: 'pose:point', label: 'Point', icon: '👆', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Point with index finger, others closed.' },
        { id: 'pose:thumbs_up', label: 'Thumbs Up', icon: '👍', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Classic thumbs up gesture.' },
        { id: 'pose:ok', label: 'OK Sign', icon: '👌', transition: TRANSITION_TIME, hold: HOLD_TIME, desc: 'Form OK sign with thumb and index.' }
    ]
};

/**
 * Set module dependencies
 * @param {Object} dependencies - Required dependencies
 */
export function setDependencies(dependencies) {
    deps = { ...deps, ...dependencies };
}

/**
 * Get wizard state (for telemetry handler)
 * @returns {Object} Wizard state object
 */
export function getWizardState() {
    return wizard;
}

/**
 * Get calibration buffers (for telemetry handler)
 * @returns {Object} Calibration buffers object
 */
export function getCalibrationBuffers() {
    return calibrationBuffers;
}

/**
 * Initialize wizard functionality
 */
export function initWizard() {
    const wizardBtn = deps.$('wizardBtn');
    if (wizardBtn) {
        wizardBtn.addEventListener('click', startWizard);
    }

    // Export functions to window for onclick handlers in HTML
    window.closeWizard = closeWizard;
    window.nextWizardStep = nextWizardStep;
    window.startWizardCollection = startWizardCollection;
    window.startWizardMode = startWizardMode;
}

/**
 * Start data collection wizard - shows mode selection
 */
function startWizard() {
    if (!deps.state.connected) {
        deps.log('Error: Connect device first');
        return;
    }

    wizard.active = true;
    deps.log('Data collection wizard started');
    showWizardModal();
    showWizardModeSelection();
}

/**
 * Show wizard mode selection screen
 */
function showWizardModeSelection() {
    const title = deps.$('wizardTitle');
    const phase = deps.$('wizardPhase');
    const content = deps.$('wizardContent');
    const progressFill = deps.$('wizardProgressFill');
    const stepText = deps.$('wizardStepText');
    const timeText = deps.$('wizardTimeText');
    const stats = deps.$('wizardStats');

    if (title) title.textContent = 'Data Collection Wizard';
    if (phase) phase.textContent = 'Select collection mode';
    if (progressFill) progressFill.style.width = '0%';
    if (stepText) stepText.textContent = 'Step 0 of 0';
    if (timeText) timeText.textContent = '';
    if (stats) stats.style.display = 'none';

    if (content) {
        content.innerHTML = `
            <div class="wizard-instruction">Choose a data collection mode</div>

            <div style="font-size: 11px; color: var(--accent); margin: 10px 0 5px; font-weight: 500;">General Collection</div>
            <div class="wizard-start-options">
                <div class="wizard-option" onclick="window.startWizardMode('quick')">
                    <h4>⚡ Quick Collection</h4>
                    <p>Reference poses + finger isolation</p>
                    <div class="duration">~50 seconds • 7 steps</div>
                </div>
                <div class="wizard-option" onclick="window.startWizardMode('full')">
                    <h4>📚 Full Collection</h4>
                    <p>Quick + common gestures</p>
                    <div class="duration">~80 seconds • 13 steps</div>
                </div>
            </div>

            <div style="font-size: 11px; color: var(--accent); margin: 15px 0 5px; font-weight: 500;">Magnetic Finger Tracking</div>
            <div class="wizard-start-options">
                <div class="wizard-option" onclick="window.startWizardMode('ft5mag')" style="border-color: var(--success);">
                    <h4>🧲 5 Magnets (Standard)</h4>
                    <p>All fingers with magnets - full tracking data</p>
                    <div class="duration">~85 seconds • 11 steps</div>
                </div>
            </div>

            <div style="margin-top: 15px; padding: 10px; background: var(--bg); border-radius: 4px; font-size: 11px; color: var(--fg-muted);">
                <strong>Note:</strong> Complete magnetometer calibration before collecting finger tracking data. Calibration removes environmental interference to isolate magnet signals.
            </div>
        `;
    }
}

/**
 * Show wizard modal
 */
function showWizardModal() {
    const overlay = deps.$('wizardOverlay');
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

    const overlay = deps.$('wizardOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }

    deps.log('Wizard closed');
}

/**
 * Render current wizard step
 */
function renderWizardStep() {
    const step = wizard.steps[wizard.currentStep];
    if (!step) return;

    const title = deps.$('wizardTitle');
    const phase = deps.$('wizardPhase');
    const content = deps.$('wizardContent');
    const progressFill = deps.$('wizardProgressFill');
    const stepText = deps.$('wizardStepText');
    const timeText = deps.$('wizardTimeText');
    const samplesText = deps.$('wizardSamples');
    const labelsText = deps.$('wizardLabels');

    if (title) title.textContent = step.title;
    if (phase) phase.textContent = `Step ${wizard.currentStep + 1} of ${wizard.steps.length}`;

    const progress = ((wizard.currentStep) / wizard.steps.length) * 100;
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (stepText) stepText.textContent = `Step ${wizard.currentStep + 1} of ${wizard.steps.length}`;
    if (timeText) timeText.textContent = step.duration ? `~${step.duration / 1000}s` : '';

    if (samplesText) samplesText.textContent = deps.state.sessionData.length;
    if (labelsText) labelsText.textContent = deps.state.labels.length;

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
                        ${deps.state.sessionData.length} samples collected<br>
                        ${deps.state.labels.length} labels created
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
        if (step.labels.pose) deps.state.currentLabels.pose = step.labels.pose;
        if (step.labels.motion) deps.state.currentLabels.motion = step.labels.motion;
    }

    // Start recording if not already
    if (!deps.state.recording) {
        await deps.startRecording();
    }

    // Show countdown
    const content = deps.$('wizardContent');
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

/**
 * Start wizard with selected mode
 */
async function startWizardMode(mode) {
    wizard.mode = mode;
    wizard.currentStep = 0;

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
                ...WIZARD_STEPS.gestures
            ];
            break;
        case 'ft5mag':
            wizard.steps = [...WIZARD_STEPS.fingerTracking5Mag];
            break;
        default:
            wizard.steps = [...WIZARD_STEPS.reference, ...WIZARD_STEPS.fingers];
    }

    const stats = deps.$('wizardStats');
    if (stats) stats.style.display = 'flex';

    // Start recording if not already
    if (!deps.state.recording && deps.state.connected && deps.state.gambitClient) {
        await deps.startRecording();
    }

    deps.log(`Wizard mode: ${mode} (${wizard.steps.length} steps)`);
    renderWizardStep();
}
