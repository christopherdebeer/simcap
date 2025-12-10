/**
 * Application State Management
 * Central state for GAMBIT Collector
 */

export const state = {
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

/**
 * Reset session data while preserving connection and calibration
 */
export function resetSession() {
    state.sessionData = [];
    state.labels = [];
    state.currentLabelStart = null;
    state.currentLabels = {
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
    };
}
