/**
 * Data Collection Wizard Module
 * Guided multi-step data collection with automatic label application
 */

import type { FingerStates, MotionType } from '@core/types';

// ===== Type Definitions =====

export interface WizardStep {
  id: string;
  label: string;
  icon: string;
  transition: number;
  hold: number;
  desc: string;
  title?: string;
  description?: string;
}

export interface WizardState {
  active: boolean;
  mode: string | null;
  currentStep: number;
  steps: WizardStep[];
  phase: 'transition' | 'hold' | null;
  phaseStart: number | null;
}

export interface CalibrationBuffers {
  [key: string]: Array<{ mx: number; my: number; mz: number }>;
}

export interface WizardLabels {
  pose: string | null;
  fingers: Partial<FingerStates>;
  motion: MotionType;
  custom: string[];
}

interface AppState {
  connected: boolean;
  recording: boolean;
  sessionData: any[];
  labels: any[];
  currentLabelStart: number | null;
  currentLabels: {
    pose: string | null;
    fingers: Partial<FingerStates>;
    motion: MotionType;
    custom: string[];
  };
  gambitClient: any;
}

interface WizardDependencies {
  state: AppState | null;
  startRecording: (() => Promise<void>) | null;
  $: ((id: string) => HTMLElement | null) | null;
  log: ((msg: string) => void) | null;
}

interface CountdownOptions {
  title: string;
  subtitle: string;
  phaseLabel: string;
  phaseColor: string;
}

// ===== Module State =====

const wizard: WizardState = {
    active: false,
    mode: null,
    currentStep: 0,
    steps: [],
    phase: null,
    phaseStart: null
};

const calibrationBuffers: CalibrationBuffers = {};

let deps: WizardDependencies = {
    state: null,
    startRecording: null,
    $: null,
    log: null
};

// ===== Constants =====

const TRANSITION_TIME = 5;
const HOLD_TIME = 3;
const HOLD_TIME_MED = 6;
const HOLD_TIME_LONG = 10;

const WIZARD_STEPS: Record<string, WizardStep[]> = {
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

// ===== Dependency Management =====

export function setDependencies(dependencies: Partial<WizardDependencies>): void {
    deps = { ...deps, ...dependencies };
}

export function getWizardState(): WizardState {
    return wizard;
}

export function getCalibrationBuffers(): CalibrationBuffers {
    return calibrationBuffers;
}

// ===== Initialization =====

export function initWizard(): void {
    const wizardBtn = deps.$?.('wizardBtn');
    if (wizardBtn) {
        wizardBtn.addEventListener('click', startWizard);
    }

    (window as any).closeWizard = closeWizard;
    (window as any).nextWizardStep = nextWizardStep;
    (window as any).startWizardCollection = startWizardCollection;
    (window as any).startWizardMode = startWizardMode;
}

// ===== Wizard Functions =====

function startWizard(): void {
    if (!deps.state?.connected) {
        deps.log?.('Error: Connect device first');
        return;
    }

    wizard.active = true;
    deps.log?.('Data collection wizard started');
    showWizardModal();
    showWizardModeSelection();
}

function showWizardModeSelection(): void {
    const title = deps.$?.('wizardTitle');
    const phase = deps.$?.('wizardPhase');
    const content = deps.$?.('wizardContent');
    const progressFill = deps.$?.('wizardProgressFill');
    const stepText = deps.$?.('wizardStepText');
    const timeText = deps.$?.('wizardTimeText');
    const stats = deps.$?.('wizardStats');

    if (title) title.textContent = 'Data Collection Wizard';
    if (phase) phase.textContent = 'Select collection mode';
    if (progressFill) (progressFill as HTMLElement).style.width = '0%';
    if (stepText) stepText.textContent = 'Step 0 of 0';
    if (timeText) timeText.textContent = '';
    if (stats) (stats as HTMLElement).style.display = 'none';

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

function showWizardModal(): void {
    const overlay = deps.$?.('wizardOverlay');
    if (overlay) {
        overlay.classList.add('active');
    }
}

export function closeWizard(): void {
    wizard.active = false;
    wizard.currentStep = 0;
    wizard.phase = null;

    const overlay = deps.$?.('wizardOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }

    deps.log?.('Wizard closed');
}

function parseStepLabels(id: string): WizardLabels {
    const labels: WizardLabels = {
        pose: null,
        fingers: { thumb: undefined, index: undefined, middle: undefined, ring: undefined, pinky: undefined },
        motion: 'static',
        custom: []
    };

    if (!id) return labels;

    if (id.startsWith('pose:')) {
        labels.pose = id.replace('pose:', '');
        return labels;
    }

    if (id.startsWith('finger_isolation:')) {
        const finger = id.replace('finger_isolation:', '');
        labels.custom.push(`isolation_${finger}`);
        labels.motion = 'dynamic';
        return labels;
    }

    if (id.startsWith('ft5:')) {
        const pose = id.replace('ft5:', '');
        labels.custom.push('ft5mag');

        switch (pose) {
            case 'reference':
            case 'all_extended':
                labels.fingers = { thumb: 'extended', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended' };
                break;
            case 'all_flexed':
                labels.fingers = { thumb: 'flexed', index: 'flexed', middle: 'flexed', ring: 'flexed', pinky: 'flexed' };
                labels.pose = 'fist';
                break;
            case 'thumb_flex':
                labels.fingers = { thumb: 'flexed', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended' };
                break;
            case 'index_flex':
                labels.fingers = { thumb: 'extended', index: 'flexed', middle: 'extended', ring: 'extended', pinky: 'extended' };
                break;
            case 'middle_flex':
                labels.fingers = { thumb: 'extended', index: 'extended', middle: 'flexed', ring: 'extended', pinky: 'extended' };
                break;
            case 'ring_flex':
                labels.fingers = { thumb: 'extended', index: 'extended', middle: 'extended', ring: 'flexed', pinky: 'extended' };
                break;
            case 'pinky_flex':
                labels.fingers = { thumb: 'extended', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'flexed' };
                break;
            case 'thumb_index':
                labels.fingers = { thumb: 'flexed', index: 'flexed', middle: 'extended', ring: 'extended', pinky: 'extended' };
                labels.pose = 'pinch';
                break;
            case 'ring_pinky':
                labels.fingers = { thumb: 'extended', index: 'extended', middle: 'extended', ring: 'flexed', pinky: 'flexed' };
                break;
            case 'middle_ring_pinky':
                labels.fingers = { thumb: 'extended', index: 'extended', middle: 'flexed', ring: 'flexed', pinky: 'flexed' };
                labels.pose = 'peace';
                break;
        }
        labels.custom.push(pose);
        return labels;
    }

    if (id === 'reference_pose' || id === 'magnet_baseline') {
        labels.custom.push(id);
        labels.fingers = { thumb: 'extended', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended' };
        return labels;
    }

    return labels;
}

function renderWizardStep(): void {
    const step = wizard.steps[wizard.currentStep];
    if (!step) return;

    const title = deps.$?.('wizardTitle');
    const phase = deps.$?.('wizardPhase');
    const content = deps.$?.('wizardContent');
    const progressFill = deps.$?.('wizardProgressFill');
    const stepText = deps.$?.('wizardStepText');
    const timeText = deps.$?.('wizardTimeText');
    const samplesText = deps.$?.('wizardSamples');
    const labelsText = deps.$?.('wizardLabels');

    const stepTitle = step.label || step.title || 'Step';
    const stepInstruction = `${step.icon || '📍'} ${stepTitle}`;
    const stepDescription = step.desc || step.description || '';
    const totalDuration = (step.transition || 0) + (step.hold || 0);

    if (title) title.textContent = stepTitle;
    if (phase) phase.textContent = `Step ${wizard.currentStep + 1} of ${wizard.steps.length}`;

    const progress = ((wizard.currentStep) / wizard.steps.length) * 100;
    if (progressFill) (progressFill as HTMLElement).style.width = `${progress}%`;
    if (stepText) stepText.textContent = `Step ${wizard.currentStep + 1} of ${wizard.steps.length}`;
    if (timeText) timeText.textContent = totalDuration ? `~${totalDuration}s` : '';

    if (samplesText && deps.state) samplesText.textContent = String(deps.state.sessionData.length);
    if (labelsText && deps.state) labelsText.textContent = String(deps.state.labels.length);

    if (content) {
        let html = `
            <div class="wizard-instruction">${stepInstruction}</div>
            <div class="wizard-description">${stepDescription}</div>
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
                        ${deps.state?.sessionData.length || 0} samples collected<br>
                        ${deps.state?.labels.length || 0} labels created
                    </div>
                </div>
                <div class="wizard-controls">
                    <button class="btn-success" onclick="closeWizard()">Done</button>
                </div>
            `;
        } else {
            html += `
                <div style="font-size: 11px; color: var(--fg-muted); margin: 10px 0;">
                    ${step.transition}s transition → ${step.hold}s labeled hold
                </div>
                <div class="wizard-controls">
                    <button class="btn-success" onclick="startWizardCollection()">Ready - Start</button>
                    <button class="btn-secondary" onclick="nextWizardStep()">Skip</button>
                </div>
            `;
        }

        content.innerHTML = html;
    }
}

function applyLabels(labels: WizardLabels): void {
    if (!deps.state) return;

    if (labels.pose) {
        deps.state.currentLabels.pose = labels.pose;
    }
    if (labels.fingers) {
        deps.state.currentLabels.fingers = { ...labels.fingers } as any;
    }
    if (labels.motion) {
        deps.state.currentLabels.motion = labels.motion;
    }
    if (labels.custom && labels.custom.length > 0) {
        labels.custom.forEach(c => {
            if (!deps.state!.currentLabels.custom.includes(c)) {
                deps.state!.currentLabels.custom.push(c);
            }
        });
    }
}

function clearLabels(): void {
    if (!deps.state) return;

    deps.state.currentLabels.pose = null;
    deps.state.currentLabels.fingers = { thumb: undefined, index: undefined, middle: undefined, ring: undefined, pinky: undefined } as any;
    deps.state.currentLabels.motion = 'static';
    deps.state.currentLabels.custom = [];
}

function closeAndStartNewLabel(): void {
    if (!deps.state) return;

    if (deps.state.recording && deps.state.currentLabelStart !== null && deps.state.sessionData.length > deps.state.currentLabelStart) {
        const segment = {
            start_sample: deps.state.currentLabelStart,
            end_sample: deps.state.sessionData.length - 1,
            labels: JSON.parse(JSON.stringify(deps.state.currentLabels))
        };
        deps.state.labels.push(segment);
        deps.log?.(`Label: ${segment.start_sample}-${segment.end_sample}`);
    }
    deps.state.currentLabelStart = deps.state.sessionData.length;
}

async function startWizardCollection(): Promise<void> {
    const step = wizard.steps[wizard.currentStep];
    if (!step) return;

    const transitionTime = (step.transition || TRANSITION_TIME) * 1000;
    const holdTime = (step.hold || HOLD_TIME) * 1000;
    const stepLabels = parseStepLabels(step.id);

    if (!deps.state?.recording && deps.startRecording) {
        await deps.startRecording();
    }

    const content = deps.$?.('wizardContent');
    const stepTitle = step.label || 'Step';
    const stepIcon = step.icon || '📍';

    wizard.phase = 'transition';
    wizard.phaseStart = Date.now();

    clearLabels();
    closeAndStartNewLabel();

    await runCountdown(content, transitionTime, {
        title: `${stepIcon} Get Ready`,
        subtitle: `Move to: ${stepTitle}`,
        phaseLabel: 'TRANSITION',
        phaseColor: 'var(--warning)'
    });

    wizard.phase = 'hold';
    wizard.phaseStart = Date.now();

    applyLabels(stepLabels);
    closeAndStartNewLabel();

    await runCountdown(content, holdTime, {
        title: `${stepIcon} HOLD: ${stepTitle}`,
        subtitle: step.desc || 'Hold this position steady',
        phaseLabel: 'RECORDING',
        phaseColor: 'var(--success)'
    });

    closeAndStartNewLabel();
    clearLabels();

    wizard.phase = null;
    nextWizardStep();
}

function runCountdown(content: HTMLElement | null | undefined, durationMs: number, options: CountdownOptions): Promise<void> {
    return new Promise((resolve) => {
        let remainingMs = durationMs;
        const startTime = Date.now();

        const updateDisplay = () => {
            const seconds = Math.ceil(remainingMs / 1000);
            const progress = ((durationMs - remainingMs) / durationMs) * 100;

            if (content) {
                content.innerHTML = `
                    <div class="wizard-phase-indicator" style="color: ${options.phaseColor}; font-weight: bold; font-size: 12px; margin-bottom: 5px;">
                        ${options.phaseLabel}
                    </div>
                    <div class="wizard-instruction">${options.title}</div>
                    <div class="wizard-description">${options.subtitle}</div>
                    <div class="wizard-countdown">
                        <div class="countdown-circle ${seconds <= 2 ? 'warning' : ''}" style="border-color: ${options.phaseColor};">
                            ${seconds}
                        </div>
                    </div>
                    <div class="wizard-progress-bar" style="margin-top: 15px; height: 4px; background: var(--border); border-radius: 2px;">
                        <div style="width: ${progress}%; height: 100%; background: ${options.phaseColor}; border-radius: 2px; transition: width 0.1s;"></div>
                    </div>
                `;
            }
        };

        updateDisplay();

        const countdownInterval = setInterval(() => {
            remainingMs = durationMs - (Date.now() - startTime);

            if (remainingMs <= 0) {
                clearInterval(countdownInterval);
                resolve();
            } else {
                updateDisplay();
            }
        }, 100);
    });
}

export function nextWizardStep(): void {
    wizard.currentStep++;

    if (wizard.currentStep >= wizard.steps.length) {
        closeWizard();
    } else {
        renderWizardStep();
    }
}

export async function startWizardMode(mode: string): Promise<void> {
    wizard.mode = mode;
    wizard.currentStep = 0;

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

    const stats = deps.$?.('wizardStats');
    if (stats) (stats as HTMLElement).style.display = 'flex';

    if (!deps.state?.recording && deps.state?.connected && deps.state?.gambitClient && deps.startRecording) {
        await deps.startRecording();
    }

    deps.log?.(`Wizard mode: ${mode} (${wizard.steps.length} steps)`);
    renderWizardStep();
}

// ===== Default Export =====

export default {
    setDependencies,
    getWizardState,
    getCalibrationBuffers,
    initWizard,
    closeWizard,
    nextWizardStep,
    startWizardMode
};
