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
  aslLetter?: string;           // ASL letter for Gallaudet font display
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

// ===== ASL Letter Mapping =====

/**
 * Maps finger configurations to ASL letters (A-Z) for Gallaudet font display.
 * The Gallaudet font renders letters as ASL hand signs.
 *
 * Key ASL hand shapes:
 * - A: Fist with thumb alongside (not over)
 * - B: Flat hand, fingers together, thumb across palm
 * - C: Curved hand (cupped shape)
 * - D: Index up, others curved to touch thumb
 * - E: Fingers curled, thumb across
 * - F: Index-thumb circle, others extended
 * - I: Pinky up only
 * - L: Thumb and index extended (L shape)
 * - O: Fingertips touch thumb (O shape)
 * - S: Fist with thumb over fingers
 * - U: Index and middle up together
 * - V: Index and middle spread (peace/victory)
 * - W: Index, middle, ring up
 * - Y: Thumb and pinky extended (shaka/hang loose)
 */
export const ASL_LETTER_MAP: Record<string, string> = {
  // Exact matches (binary states)
  '00000': 'B', // All extended = B (flat hand)
  '22222': 'S', // All flexed = S (fist)
  '02222': 'D', // Point = D (index up)
  '00222': 'V', // Peace = V (index + middle up spread)
  '00022': 'W', // Three fingers = W
  '20002': 'Y', // Shaka = Y (thumb + pinky)
  '20000': 'A', // Thumb flex = A (fist-like with thumb)
  '22002': 'I', // Index + middle + ring flex = I (pinky up)

  // Common curled/intermediate states
  '11111': 'C', // All curled = C (cupped hand)
  '01111': 'O', // Thumb extended, others curled = O shape
  '21111': 'E', // All curled, thumb flex = E

  // Compound poses with curled states
  '00002': '4', // Four fingers (use number 4)
  '02000': 'G', // Index only flex (pointing sideways)
  '22000': 'L', // Thumb + index flex (inverted L shape - use for pinch)
};

/**
 * Get ASL letter for a finger code, if available.
 * Returns undefined if no mapping exists.
 */
export function getASLLetter(fingerCode: string): string | undefined {
  return ASL_LETTER_MAP[fingerCode];
}

// ===== Finger Code Definitions =====

/**
 * Finger configurations with optional curled (intermediate) states.
 * Code: T I M R P (Thumb, Index, Middle, Ring, Pinky)
 * 0 = Extended, 1 = Curled (partial), 2 = Flexed (full)
 *
 * Binary states (0/2 only) give 32 configurations.
 * Ternary states (0/1/2) give 243 configurations for fine-grained capture.
 */
export const FINGER_CODES: Record<string, { name: string; semantic?: string; aslLetter?: string }> = {
  // === Binary states (0 = extended, 2 = flexed) ===
  '00000': { name: 'all_extended', semantic: 'open_palm', aslLetter: 'B' },
  '00002': { name: 'four_fingers', semantic: 'four', aslLetter: '4' },
  '00020': { name: 'ring_flex' },
  '00022': { name: 'three_fingers', semantic: 'three', aslLetter: 'W' },
  '00200': { name: 'middle_flex' },
  '00202': { name: 'middle_pinky_flex' },
  '00220': { name: 'middle_ring_flex' },
  '00222': { name: 'peace', semantic: 'peace', aslLetter: 'V' },
  '02000': { name: 'index_flex', aslLetter: 'G' },
  '02002': { name: 'index_pinky_flex' },
  '02020': { name: 'index_ring_flex' },
  '02022': { name: 'index_ring_pinky_flex' },
  '02200': { name: 'index_middle_flex', aslLetter: 'U' },
  '02202': { name: 'index_middle_pinky_flex' },
  '02220': { name: 'index_middle_ring_flex' },
  '02222': { name: 'point', semantic: 'point', aslLetter: 'D' },
  '20000': { name: 'thumb_flex', semantic: 'thumbs_down', aslLetter: 'A' },
  '20002': { name: 'shaka', semantic: 'shaka', aslLetter: 'Y' },
  '20020': { name: 'thumb_ring_flex' },
  '20022': { name: 'thumb_ring_pinky_flex' },
  '20200': { name: 'thumb_middle_flex' },
  '20202': { name: 'thumb_middle_pinky_flex' },
  '20220': { name: 'thumb_middle_ring_flex' },
  '20222': { name: 'thumb_mid_ring_pinky_flex' },
  '22000': { name: 'thumb_index_flex', aslLetter: 'L' },
  '22002': { name: 'thumb_index_pinky_flex', aslLetter: 'I' },
  '22020': { name: 'thumb_index_ring_flex' },
  '22022': { name: 'thumb_index_ring_pinky_flex' },
  '22200': { name: 'thumb_index_middle_flex' },
  '22202': { name: 'thumb_index_middle_pinky_flex' },
  '22220': { name: 'thumb_index_middle_ring_flex' },
  '22222': { name: 'all_flexed', semantic: 'fist', aslLetter: 'S' },

  // === Curled/intermediate states (1 = curled) ===
  '11111': { name: 'all_curled', semantic: 'claw', aslLetter: 'C' },
  '01111': { name: 'thumb_ext_curled', semantic: 'grasp', aslLetter: 'O' },
  '21111': { name: 'thumb_flex_curled', semantic: 'claw_closed', aslLetter: 'E' },
  '10000': { name: 'thumb_curled' },
  '01000': { name: 'index_curled', aslLetter: 'X' },
  '00100': { name: 'middle_curled' },
  '00010': { name: 'ring_curled' },
  '00001': { name: 'pinky_curled' },
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
    // 0 = extended, 1 = curled, 2 = flexed
    if (digit === '0') {
      states[fingers[i]] = 'extended';
    } else if (digit === '1') {
      states[fingers[i]] = 'curled' as any; // curled is now a valid FingerLabel
    } else {
      states[fingers[i]] = 'flexed';
    }
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
    icon: info.aslLetter ? info.aslLetter : orientInfo.icon, // Use ASL letter as icon if available
    transition: T_MED,
    hold: hold,
    desc: `Hold ${info.name} pose with ${orientInfo.desc.toLowerCase()}`,
    fingerCode: code,
    orientation: orientation,
    phase: 'static',
    category: 'coarse_state',
    aslLetter: info.aslLetter, // Include ASL letter for font display
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

// ===== Pairwise Calibration Steps =====

/**
 * Generate pairwise calibration steps for physics model fitting.
 * These combinations are critical for understanding finger interaction effects.
 *
 * Priority pairings:
 * - Adjacent pairs: T-I, I-M, M-R, R-P (strongest magnetic coupling)
 * - Non-adjacent pairs: T-M, T-R, T-P, I-R, I-P, M-P
 */
function generatePairwiseCalibrationSteps(): ExtendedWizardStep[] {
  const steps: ExtendedWizardStep[] = [];

  // Reference baseline (all extended)
  steps.push({
    ...generatePoseStep('00000', 'palm_down', H_LONG),
    id: 'pair_cal:baseline',
    label: 'Baseline (All Extended)',
    desc: 'Reference position: palm down, all fingers fully extended',
  });

  // All single-finger flexes (ground truth for individual magnets)
  const singleFlexCodes = ['20000', '02000', '00200', '00020', '00002'];
  const fingerNames = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];

  for (let i = 0; i < 5; i++) {
    steps.push({
      ...generatePoseStep(singleFlexCodes[i], 'palm_down', H_MED),
      id: `pair_cal:single:${fingerNames[i].toLowerCase()}`,
      label: `${fingerNames[i]} Only`,
      desc: `Flex only ${fingerNames[i].toLowerCase()}, all others extended`,
    });
  }

  // Adjacent pairs (critical for understanding coupling)
  const adjacentPairs = [
    { code: '22000', name: 'Thumb+Index', desc: 'Adjacent pair: strongest coupling expected' },
    { code: '02200', name: 'Index+Middle', desc: 'Adjacent pair: strong coupling' },
    { code: '00220', name: 'Middle+Ring', desc: 'Adjacent pair with opposite polarity' },
    { code: '00022', name: 'Ring+Pinky', desc: 'Adjacent pair with opposite polarity' },
  ];

  for (const pair of adjacentPairs) {
    steps.push({
      ...generatePoseStep(pair.code, 'palm_down', H_MED),
      id: `pair_cal:adjacent:${pair.name.toLowerCase().replace('+', '_')}`,
      label: pair.name,
      desc: pair.desc,
    });
  }

  // Non-adjacent pairs (for understanding distant interactions)
  const nonAdjacentPairs = [
    { code: '20200', name: 'Thumb+Middle', desc: 'Skip-1 pair' },
    { code: '20020', name: 'Thumb+Ring', desc: 'Skip-2 pair (opposite polarity)' },
    { code: '20002', name: 'Thumb+Pinky', desc: 'Skip-3 pair (shaka)' },
    { code: '02020', name: 'Index+Ring', desc: 'Skip-1 pair (opposite polarity)' },
    { code: '02002', name: 'Index+Pinky', desc: 'Skip-2 pair' },
    { code: '00202', name: 'Middle+Pinky', desc: 'Skip-1 pair' },
  ];

  for (const pair of nonAdjacentPairs) {
    steps.push({
      ...generatePoseStep(pair.code, 'palm_down', H_MED),
      id: `pair_cal:nonadj:${pair.name.toLowerCase().replace('+', '_')}`,
      label: pair.name,
      desc: pair.desc,
    });
  }

  // All flexed (for full interaction measurement)
  steps.push({
    ...generatePoseStep('22222', 'palm_down', H_LONG),
    id: 'pair_cal:all_flexed',
    label: 'All Flexed (Fist)',
    desc: 'All fingers flexed for full interaction measurement',
  });

  return steps;
}

// ===== Curled/Intermediate State Steps =====

/**
 * Generate steps for curled (intermediate) finger states.
 * These capture partial flexion for finer-grained training data.
 */
function generateCurledStateSteps(): ExtendedWizardStep[] {
  const steps: ExtendedWizardStep[] = [];

  // Reference poses
  steps.push({
    ...generatePoseStep('00000', 'palm_down', H_MED),
    id: 'curled:baseline',
    label: 'Baseline (All Extended)',
    desc: 'Start with all fingers fully extended',
  });

  // All curled (claw pose)
  steps.push({
    id: 'curled:all',
    label: 'Claw (All Curled)',
    icon: 'C',
    transition: T_MED,
    hold: H_MED,
    desc: 'Curl all fingers like a claw - partial flexion, not fully closed',
    fingerCode: '11111',
    phase: 'static',
    category: 'curled',
    aslLetter: 'C',
  });

  // Individual finger curls
  const fingers = [
    { code: '10000', name: 'Thumb', desc: 'Curl only thumb halfway' },
    { code: '01000', name: 'Index', desc: 'Curl only index finger halfway', asl: 'X' },
    { code: '00100', name: 'Middle', desc: 'Curl only middle finger halfway' },
    { code: '00010', name: 'Ring', desc: 'Curl only ring finger halfway' },
    { code: '00001', name: 'Pinky', desc: 'Curl only pinky finger halfway' },
  ];

  for (const finger of fingers) {
    steps.push({
      id: `curled:single:${finger.name.toLowerCase()}`,
      label: `${finger.name} Curled`,
      icon: finger.asl || '🦎',
      transition: T_SHORT,
      hold: H_MED,
      desc: finger.desc,
      fingerCode: finger.code,
      phase: 'static',
      category: 'curled',
      aslLetter: finger.asl,
    });
  }

  // Grasp pose (thumb extended, others curled)
  steps.push({
    id: 'curled:grasp',
    label: 'Grasp (O-shape)',
    icon: 'O',
    transition: T_MED,
    hold: H_MED,
    desc: 'Fingertips curled toward thumb - like holding a ball',
    fingerCode: '01111',
    phase: 'static',
    category: 'curled',
    aslLetter: 'O',
  });

  // E pose (all curled, thumb flexed across)
  steps.push({
    id: 'curled:e_pose',
    label: 'E Pose',
    icon: 'E',
    transition: T_MED,
    hold: H_MED,
    desc: 'All fingers curled, thumb tucked - ASL letter E',
    fingerCode: '21111',
    phase: 'static',
    category: 'curled',
    aslLetter: 'E',
  });

  // Transition from extended to curled to flexed
  steps.push({
    id: 'curled:range:all',
    label: 'Full Range (Ext→Curl→Flex)',
    icon: '🔄',
    transition: T_SHORT,
    hold: H_LONG,
    desc: 'Slowly transition: fully extended → curled → fully flexed → back (3 cycles)',
    phase: 'dynamic',
    motion: 'dynamic',
    category: 'curled',
  });

  // All flexed as endpoint
  steps.push({
    ...generatePoseStep('22222', 'palm_down', H_MED),
    id: 'curled:endpoint',
    label: 'Fist (All Flexed)',
    desc: 'End with all fingers fully flexed',
  });

  return steps;
}

// ===== Wizard Mode Definitions =====

export const WIZARD_MODES: WizardMode[] = [
  // Pairwise Calibration for Physics Model (5 min)
  {
    id: 'pairwise_cal',
    name: 'Pairwise Calibration',
    description: 'All single + pairwise combos for physics model fitting',
    duration: '~5 min',
    priority: 'required',
    steps: generatePairwiseCalibrationSteps(),
  },

  // Curled/Intermediate States (3 min)
  {
    id: 'curled_states',
    name: 'Curled States',
    description: 'Intermediate finger positions (partial flexion)',
    duration: '~3 min',
    priority: 'recommended',
    steps: generateCurledStateSteps(),
  },

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
  ASL_LETTER_MAP,
  getASLLetter,
  getWizardMode,
  parseExtendedStepLabels,
  getTotalDuration,
  getStepCount,
  getCollectionSummary,
};
