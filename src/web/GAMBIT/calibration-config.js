/**
 * Calibration Configuration
 * Single source of truth for calibration sample counts, rates, and quality thresholds
 *
 * NOTE: Earth field calibration has been removed from the wizard.
 * Earth field is now auto-estimated in real-time using UnifiedMagCalibration
 * (200-sample sliding window, orientation-compensated world-frame averaging).
 */

export const CALIBRATION_CONFIG = {
    SAMPLE_RATE: 50,

    HARD_IRON: {
        sampleCount: 1000,
        sampleRate: 50,
        minSamples: 100,
        qualityThresholds: {
            excellent: 0.9,
            good: 0.7
        }
    },

    SOFT_IRON: {
        sampleCount: 1000,
        sampleRate: 50,
        minSamples: 200,
        qualityThresholds: {
            excellent: 0.9,
            good: 0.7
        }
    }
};

export function getDuration(stepName) {
    const config = CALIBRATION_CONFIG[stepName];
    if (!config) {
        throw new Error(`Unknown calibration step: ${stepName}`);
    }
    return Math.ceil((config.sampleCount / config.sampleRate) * 1000);
}

export function formatDuration(stepName) {
    const durationMs = getDuration(stepName);
    const seconds = durationMs / 1000;
    return `${seconds} second${seconds !== 1 ? 's' : ''}`;
}

export function validateSampleCount(stepName, actualCount) {
    const config = CALIBRATION_CONFIG[stepName];
    if (!config) {
        throw new Error(`Unknown calibration step: ${stepName}`);
    }

    return {
        valid: actualCount >= config.minSamples,
        actualCount,
        minSamples: config.minSamples,
        expectedSamples: config.sampleCount,
        percentage: Math.round((actualCount / config.sampleCount) * 100)
    };
}
