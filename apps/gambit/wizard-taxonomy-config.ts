/**
 * Comprehensive Wizard Configuration for Finger State Taxonomy
 *
 * This configuration extends the basic wizard with:
 * 1. Coarse-grained finger state collection (32 configurations)
 * 2. Fine-grained semantic pose capture
 * 3. Multi-orientation capture for robustness
 * 4. Non-semantic phases (erratic, full range, orientation sweeps)
 *
 * Reference: docs/design/finger-state-taxonomy.md
 */

import type { FingerStates, MotionType } from '@core/types';

// ===== Type Definitions =====

export interface ExtendedWizardStep {
  id: string;
  label: string;
  icon: string;
  transition: number;
  hold: number;
  desc: string;
  fingerCode?: string;          // 5-digit finger state code (e.g., "00000")
  orientation?: string;         // palm_down, palm_up, etc.
  phase?: 'static' | 'dynamic' | 'erratic' | 'sweep';
  motion?: MotionType;
  category?: string;            // For grouping in UI
}

export interface WizardMode {
  id: string;
  name: string;
  description: string;
  duration: string;
  steps: ExtendedWizardStep[];
  priority: 'required' | 'recommended' | 'optional';
}

// ===== Constants =====

const T_SHORT = 3;    // Short transition
const T_MED = 5;      // Medium transition
const T_LONG = 8;     // Long transition (for orientation changes)

const H_SHORT = 3;    // Short hold
const H_MED = 5;      // Medium hold
const H_LONG = 8;     // Long hold (for robust capture)
const H_ERRATIC = 10; // Erratic phase
const H_SWEEP = 12;   // Orientation sweep

// ===== Finger Code Definitions =====

/**
 * All 32 binary finger configurations
 * Code: T I M R P (Thumb, Index, Middle, Ring, Pinky)
 * 0 = Extended, 2 = Flexed
 */
export const FINGER_CODES: Record<string, { name: string; semantic?: string }> = {
  '00000': { name: 'all_extended', semantic: 'open_palm' },
  '00002': { name: 'four_fingers', semantic: 'four' },
  '00020': { name: 'ring_flex' },
  '00022': { name: 'three_fingers', semantic: 'three' },
  '00200': { name: 'middle_flex' },
  '00202': { name: 'middle_pinky_flex' },
  '00220': { name: 'middle_ring_flex' },
  '00222': { name: 'peace', semantic: 'peace' },
  '02000': { name: 'index_flex' },
  '02002': { name: 'index_pinky_flex' },
  '02020': { name: 'index_ring_flex' },
  '02022': { name: 'index_ring_pinky_flex' },
  '02200': { name: 'index_middle_flex' },
  '02202': { name: 'index_middle_pinky_flex' },
  '02220': { name: 'index_middle_ring_flex' },
  '02222': { name: 'point', semantic: 'point' },
  '20000': { name: 'thumb_flex', semantic: 'thumbs_down' },
  '20002': { name: 'shaka', semantic: 'shaka' },
  '20020': { name: 'thumb_ring_flex' },
  '20022': { name: 'thumb_ring_pinky_flex' },
  '20200': { name: 'thumb_middle_flex' },
  '20202': { name: 'thumb_middle_pinky_flex' },
  '20220': { name: 'thumb_middle_ring_flex' },
  '20222': { name: 'thumb_mid_ring_pinky_flex' },
  '22000': { name: 'thumb_index_flex' },
  '22002': { name: 'thumb_index_pinky_flex' },
  '22020': { name: 'thumb_index_ring_flex' },
  '22022': { name: 'thumb_index_ring_pinky_flex' },
  '22200': { name: 'thumb_index_middle_flex' },
  '22202': { name: 'thumb_index_middle_pinky_flex' },
  '22220': { name: 'thumb_index_middle_ring_flex' },
  '22222': { name: 'all_flexed', semantic: 'fist' },
};

// ===== Orientation Definitions =====

export const ORIENTATIONS = {
  palm_down: { icon: '⬇️', desc: 'Palm facing down (reference)' },
  palm_up: { icon: '⬆️', desc: 'Palm facing up' },
  palm_left: { icon: '⬅️', desc: 'Palm facing left' },
  palm_right: { icon: '➡️', desc: 'Palm facing right' },
  palm_forward: { icon: '👋', desc: 'Palm facing away from you' },
  palm_back: { icon: '🤚', desc: 'Palm facing toward you' },
};

// ===== Helper Functions =====

function fingerCodeToStates(code: string): Partial<FingerStates> {
  const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const;
  const states: Partial<FingerStates> = {};

  for (let i = 0; i < 5; i++) {
    const digit = code[i];
    states[fingers[i]] = digit === '0' ? 'extended' : 'flexed';
  }

  return states;
}

function generatePoseStep(
  code: string,
  orientation: string = 'palm_down',
  hold: number = H_MED
): ExtendedWizardStep {
  const info = FINGER_CODES[code];
  const orientInfo = ORIENTATIONS[orientation as keyof typeof ORIENTATIONS];

  return {
    id: `pose:${code}:${orientation}`,
    label: `${info.semantic || info.name} (${orientInfo.icon})`,
    icon: orientInfo.icon,
    transition: T_MED,
    hold: hold,
    desc: `Hold ${info.name} pose with ${orientInfo.desc.toLowerCase()}`,
    fingerCode: code,
    orientation: orientation,
    phase: 'static',
    category: 'coarse_state',
  };
}

// ===== Step Generation Functions =====

/**
 * Generate steps for all 32 finger configurations
 */
function generateCoarseStateSteps(orientations: string[] = ['palm_down']): ExtendedWizardStep[] {
  const steps: ExtendedWizardStep[] = [];

  for (const code of Object.keys(FINGER_CODES)) {
    for (const orientation of orientations) {
      steps.push(generatePoseStep(code, orientation, H_MED));
    }
  }

  return steps;
}

/**
 * Generate core semantic pose steps with multiple orientations
 */
function generateCorePoseSteps(): ExtendedWizardStep[] {
  const corePoses = ['00000', '22222', '02222', '20000', '00222', '22000'];
  const orientations = ['palm_down', 'palm_up', 'palm_left', 'palm_right'];
  const steps: ExtendedWizardStep[] = [];

  for (const code of corePoses) {
    for (const orientation of orientations) {
      steps.push(generatePoseStep(code, orientation, H_LONG));
    }
  }

  return steps;
}

/**
 * Generate erratic/noise capture steps
 */
function generateErraticSteps(): ExtendedWizardStep[] {
  return [
    {
      id: 'erratic:fingers',
      label: 'Erratic Finger Movement',
      icon: '🌀',
      transition: T_SHORT,
      hold: H_ERRATIC,
      desc: 'Wiggle all fingers randomly and continuously',
      phase: 'erratic',
      motion: 'dynamic',
      category: 'robustness',
    },
    {
      id: 'erratic:orientation',
      label: 'Erratic Orientation',
      icon: '🔄',
      transition: T_SHORT,
      hold: H_ERRATIC,
      desc: 'Keep fingers in fist, rotate hand randomly in all directions',
      fingerCode: '22222',
      phase: 'erratic',
      motion: 'dynamic',
      category: 'robustness',
    },
    {
      id: 'erratic:combined',
      label: 'Erratic Combined',
      icon: '🎲',
      transition: T_SHORT,
      hold: H_ERRATIC + 5,
      desc: 'Random finger movements while rotating hand randomly',
      phase: 'erratic',
      motion: 'dynamic',
      category: 'robustness',
    },
  ];
}

/**
 * Generate full range of motion steps for each finger
 */
function generateFullRangeSteps(): ExtendedWizardStep[] {
  const fingers = [
    { name: 'thumb', icon: '👍' },
    { name: 'index', icon: '☝️' },
    { name: 'middle', icon: '🖕' },
    { name: 'ring', icon: '💍' },
    { name: 'pinky', icon: '🤙' },
  ];

  const steps: ExtendedWizardStep[] = [];

  for (const finger of fingers) {
    steps.push({
      id: `range:${finger.name}`,
      label: `${finger.name.charAt(0).toUpperCase() + finger.name.slice(1)} Full Range`,
      icon: finger.icon,
      transition: T_SHORT,
      hold: H_MED,
      desc: `Slowly move ${finger.name} through complete flex/extend range (3 cycles)`,
      phase: 'dynamic',
      motion: 'dynamic',
      category: 'full_range',
    });
  }

  // Add combined range
  steps.push({
    id: 'range:all',
    label: 'All Fingers Range',
    icon: '🖐️',
    transition: T_SHORT,
    hold: H_LONG,
    desc: 'Move all fingers simultaneously through full range (3 cycles)',
    phase: 'dynamic',
    motion: 'dynamic',
    category: 'full_range',
  });

  steps.push({
    id: 'range:sequential',
    label: 'Sequential Wave',
    icon: '👋',
    transition: T_SHORT,
    hold: H_LONG,
    desc: 'Wave pattern: each finger in sequence, thumb to pinky (3 cycles)',
    phase: 'dynamic',
    motion: 'dynamic',
    category: 'full_range',
  });

  return steps;
}

/**
 * Generate orientation sweep steps for key poses
 */
function generateOrientationSweepSteps(): ExtendedWizardStep[] {
  const poses = [
    { code: '00000', name: 'open_palm' },
    { code: '22222', name: 'fist' },
    { code: '02222', name: 'point' },
  ];

  const steps: ExtendedWizardStep[] = [];

  for (const pose of poses) {
    steps.push({
      id: `sweep:slow:${pose.code}`,
      label: `${pose.name} Slow Rotation`,
      icon: '🔄',
      transition: T_SHORT,
      hold: H_SWEEP,
      desc: `Hold ${pose.name}, slowly rotate hand 360° (10 seconds)`,
      fingerCode: pose.code,
      phase: 'sweep',
      motion: 'dynamic',
      category: 'orientation_sweep',
    });

    steps.push({
      id: `sweep:fast:${pose.code}`,
      label: `${pose.name} Fast Rotation`,
      icon: '💫',
      transition: T_SHORT,
      hold: H_MED,
      desc: `Hold ${pose.name}, quickly rotate hand 360° (3 seconds)`,
      fingerCode: pose.code,
      phase: 'sweep',
      motion: 'dynamic',
      category: 'orientation_sweep',
    });
  }

  return steps;
}

/**
 * Generate calibration steps
 */
function generateCalibrationSteps(): ExtendedWizardStep[] {
  return [
    {
      id: 'cal:earth_field',
      label: 'Earth Field Baseline',
      icon: '🌍',
      transition: T_MED,
      hold: H_LONG,
      desc: 'Place device on stable surface, no magnets nearby. Stay still.',
      category: 'calibration',
    },
    {
      id: 'cal:hard_iron',
      label: 'Hard Iron Calibration',
      icon: '🧲',
      transition: T_MED,
      hold: 20,
      desc: 'Slowly rotate device in figure-8 pattern. Cover all orientations.',
      phase: 'dynamic',
      motion: 'dynamic',
      category: 'calibration',
    },
    {
      id: 'cal:reference',
      label: 'Reference Pose',
      icon: '✋',
      transition: T_MED,
      hold: H_LONG,
      desc: 'Hand flat, palm down, fingers extended and together.',
      fingerCode: '00000',
      orientation: 'palm_down',
      category: 'calibration',
    },
    {
      id: 'cal:per_finger:thumb',
      label: 'Thumb Isolation',
      icon: '👍',
      transition: T_SHORT,
      hold: H_MED,
      desc: 'Flex only thumb, other fingers extended',
      fingerCode: '20000',
      category: 'calibration',
    },
    {
      id: 'cal:per_finger:index',
      label: 'Index Isolation',
      icon: '☝️',
      transition: T_SHORT,
      hold: H_MED,
      desc: 'Flex only index finger, others extended',
      fingerCode: '02000',
      category: 'calibration',
    },
    {
      id: 'cal:per_finger:middle',
      label: 'Middle Isolation',
      icon: '🖕',
      transition: T_SHORT,
      hold: H_MED,
      desc: 'Flex only middle finger, others extended',
      fingerCode: '00200',
      category: 'calibration',
    },
    {
      id: 'cal:per_finger:ring',
      label: 'Ring Isolation',
      icon: '💍',
      transition: T_SHORT,
      hold: H_MED,
      desc: 'Flex only ring finger, others extended',
      fingerCode: '00020',
      category: 'calibration',
    },
    {
      id: 'cal:per_finger:pinky',
      label: 'Pinky Isolation',
      icon: '🤙',
      transition: T_SHORT,
      hold: H_MED,
      desc: 'Flex only pinky finger, others extended',
      fingerCode: '00002',
      category: 'calibration',
    },
  ];
}

// ===== Wizard Mode Definitions =====

export const WIZARD_MODES: WizardMode[] = [
  // Quick Calibration (2 min)
  {
    id: 'quick_cal',
    name: 'Quick Calibration',
    description: 'Essential calibration and baseline capture',
    duration: '~2 min',
    priority: 'required',
    steps: [
      ...generateCalibrationSteps().slice(0, 3), // Earth, hard iron, reference
      generatePoseStep('22222', 'palm_down', H_MED), // Fist baseline
      generatePoseStep('00000', 'palm_down', H_MED), // Open baseline
    ],
  },

  // Core Poses (5 min)
  {
    id: 'core_poses',
    name: 'Core Poses',
    description: '8 essential poses with multiple orientations',
    duration: '~5 min',
    priority: 'required',
    steps: generateCorePoseSteps(),
  },

  // Full Taxonomy - All 32 Configurations (15 min)
  {
    id: 'full_taxonomy',
    name: 'Full Finger Taxonomy',
    description: 'All 32 binary finger configurations (single orientation)',
    duration: '~15 min',
    priority: 'recommended',
    steps: generateCoarseStateSteps(['palm_down']),
  },

  // Full Taxonomy with Orientations (45 min)
  {
    id: 'full_taxonomy_oriented',
    name: 'Full Taxonomy + Orientations',
    description: 'All 32 configurations × 4 orientations',
    duration: '~45 min',
    priority: 'optional',
    steps: generateCoarseStateSteps(['palm_down', 'palm_up', 'palm_left', 'palm_right']),
  },

  // Robustness Session (10 min)
  {
    id: 'robustness',
    name: 'Robustness Training',
    description: 'Erratic movement, full range, orientation sweeps',
    duration: '~10 min',
    priority: 'recommended',
    steps: [
      ...generateErraticSteps(),
      ...generateFullRangeSteps(),
      ...generateOrientationSweepSteps(),
    ],
  },

  // Semantic Poses (Extended vocabulary)
  {
    id: 'semantic_poses',
    name: 'Semantic Pose Vocabulary',
    description: 'Common gestures and counting poses',
    duration: '~8 min',
    priority: 'optional',
    steps: [
      // Counting 1-5
      generatePoseStep('02222', 'palm_forward', H_MED), // One/Point
      generatePoseStep('00222', 'palm_forward', H_MED), // Two/Peace
      generatePoseStep('00022', 'palm_forward', H_MED), // Three
      generatePoseStep('00002', 'palm_forward', H_MED), // Four
      generatePoseStep('00000', 'palm_forward', H_MED), // Five

      // Common gestures
      generatePoseStep('20002', 'palm_forward', H_MED), // Shaka
      {
        id: 'pose:ok_sign',
        label: 'OK Sign',
        icon: '👌',
        transition: T_MED,
        hold: H_MED,
        desc: 'Thumb and index form circle, other fingers extended',
        phase: 'static',
        category: 'semantic',
      },
      {
        id: 'pose:pinch',
        label: 'Pinch',
        icon: '🤏',
        transition: T_MED,
        hold: H_MED,
        desc: 'Thumb and index tips touching, others relaxed',
        phase: 'static',
        category: 'semantic',
      },
      {
        id: 'pose:grab',
        label: 'Grab Pose',
        icon: '🤜',
        transition: T_MED,
        hold: H_MED,
        desc: 'Fingers curved as if gripping a ball',
        phase: 'static',
        category: 'semantic',
      },
    ],
  },

  // Transitions (dynamic poses)
  {
    id: 'transitions',
    name: 'Pose Transitions',
    description: 'Capture transitions between key poses',
    duration: '~5 min',
    priority: 'optional',
    steps: [
      {
        id: 'trans:rest_to_fist',
        label: 'Rest → Fist',
        icon: '✊',
        transition: T_SHORT,
        hold: H_MED,
        desc: 'Start relaxed, slowly close to fist',
        phase: 'dynamic',
        motion: 'dynamic',
        category: 'transition',
      },
      {
        id: 'trans:fist_to_open',
        label: 'Fist → Open',
        icon: '🖐️',
        transition: T_SHORT,
        hold: H_MED,
        desc: 'Start with fist, slowly spread all fingers',
        phase: 'dynamic',
        motion: 'dynamic',
        category: 'transition',
      },
      {
        id: 'trans:open_to_point',
        label: 'Open → Point',
        icon: '👆',
        transition: T_SHORT,
        hold: H_MED,
        desc: 'Start open, curl all but index',
        phase: 'dynamic',
        motion: 'dynamic',
        category: 'transition',
      },
      {
        id: 'trans:point_to_peace',
        label: 'Point → Peace',
        icon: '✌️',
        transition: T_SHORT,
        hold: H_MED,
        desc: 'From pointing, extend middle finger',
        phase: 'dynamic',
        motion: 'dynamic',
        category: 'transition',
      },
      {
        id: 'trans:open_to_pinch',
        label: 'Open → Pinch',
        icon: '🤏',
        transition: T_SHORT,
        hold: H_MED,
        desc: 'From open, bring thumb and index together',
        phase: 'dynamic',
        motion: 'dynamic',
        category: 'transition',
      },
    ],
  },

  // Comprehensive Session (everything)
  {
    id: 'comprehensive',
    name: 'Comprehensive Collection',
    description: 'Complete data collection: calibration + all poses + robustness',
    duration: '~45 min',
    priority: 'optional',
    steps: [
      ...generateCalibrationSteps(),
      ...generateCoarseStateSteps(['palm_down']),
      ...generateErraticSteps(),
      ...generateFullRangeSteps(),
      ...generateOrientationSweepSteps().slice(0, 6), // Top 3 poses only
    ],
  },
];

// ===== Export Helpers =====

export function getWizardMode(modeId: string): WizardMode | undefined {
  return WIZARD_MODES.find((m) => m.id === modeId);
}

export function parseExtendedStepLabels(step: ExtendedWizardStep): {
  pose: string | null;
  fingers: Partial<FingerStates>;
  motion: MotionType;
  custom: string[];
} {
  const labels = {
    pose: null as string | null,
    fingers: {} as Partial<FingerStates>,
    motion: (step.motion || 'static') as MotionType,
    custom: [] as string[],
  };

  // Parse finger code
  if (step.fingerCode) {
    labels.fingers = fingerCodeToStates(step.fingerCode);
    labels.custom.push(`code:${step.fingerCode}`);

    // Check for semantic pose name
    const codeInfo = FINGER_CODES[step.fingerCode];
    if (codeInfo?.semantic) {
      labels.pose = codeInfo.semantic;
    }
  }

  // Add orientation
  if (step.orientation) {
    labels.custom.push(`orientation:${step.orientation}`);
  }

  // Add phase
  if (step.phase) {
    labels.custom.push(`phase:${step.phase}`);
  }

  // Add category
  if (step.category) {
    labels.custom.push(`category:${step.category}`);
  }

  return labels;
}

export function getTotalDuration(mode: WizardMode): number {
  return mode.steps.reduce((total, step) => total + step.transition + step.hold, 0);
}

export function getStepCount(mode: WizardMode): number {
  return mode.steps.length;
}

// ===== Summary Statistics =====

export function getCollectionSummary(): {
  totalConfigurations: number;
  totalModes: number;
  requiredModes: string[];
  recommendedModes: string[];
  optionalModes: string[];
} {
  return {
    totalConfigurations: Object.keys(FINGER_CODES).length,
    totalModes: WIZARD_MODES.length,
    requiredModes: WIZARD_MODES.filter((m) => m.priority === 'required').map((m) => m.id),
    recommendedModes: WIZARD_MODES.filter((m) => m.priority === 'recommended').map((m) => m.id),
    optionalModes: WIZARD_MODES.filter((m) => m.priority === 'optional').map((m) => m.id),
  };
}

export default {
  FINGER_CODES,
  ORIENTATIONS,
  WIZARD_MODES,
  getWizardMode,
  parseExtendedStepLabels,
  getTotalDuration,
  getStepCount,
  getCollectionSummary,
};
