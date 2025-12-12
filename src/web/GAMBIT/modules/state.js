/**
 * Application State Management
 * Central state for GAMBIT Collector
 */

export const state = {
    connected: false,
    recording: false,
    paused: false,  // Recording pause state
    sessionData: [],
    labels: [],  // V2 multi-label segments
    currentLabelStart: null,
    gambitClient: null,
    firmwareVersion: null,  // Firmware version from connected device

    // Geomagnetic field location
    geomagneticLocation: null,  // Current selected location from lookup table

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

    // Custom labels defined for this session (separate from active)
    customLabelDefinitions: [],

    // Active custom labels (subset of definitions that are currently applied)
    activeCustomLabels: []
};

/**
 * Reset session data while preserving connection and calibration
 */
export function resetSession() {
    state.sessionData = [];
    state.labels = [];
    state.currentLabelStart = null;
    state.paused = false;
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
    // Note: customLabelDefinitions and activeCustomLabels are preserved
}
