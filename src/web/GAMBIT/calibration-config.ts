/**
 * Calibration Configuration
 * Single source of truth for calibration sample counts, rates, and quality thresholds
 *
 * NOTE: Earth field calibration has been removed from the wizard.
 * Earth field is now auto-estimated in real-time using UnifiedMagCalibration
 * (200-sample sliding window, orientation-compensated world-frame averaging).
 */

export interface CalibrationStepConfig {
    sampleCount: number;
    sampleRate: number;
    minSamples: number;
    qualityThresholds: {
        excellent: number;
        good: number;
    };
}

export interface CalibrationConfigType {
    SAMPLE_RATE: number;
    HARD_IRON: CalibrationStepConfig;
    SOFT_IRON: CalibrationStepConfig;
    [key: string]: number | CalibrationStepConfig;
}

export interface ValidationResult {
    valid: boolean;
    actualCount: number;
    minSamples: number;
    expectedSamples: number;
    percentage: number;
}

export const CALIBRATION_CONFIG: CalibrationConfigType = {
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

export function getDuration(stepName: string): number {
    const config = CALIBRATION_CONFIG[stepName] as CalibrationStepConfig | undefined;
    if (!config || typeof config === 'number') {
        throw new Error(`Unknown calibration step: ${stepName}`);
    }
    return Math.ceil((config.sampleCount / config.sampleRate) * 1000);
}

export function formatDuration(stepName: string): string {
    const durationMs = getDuration(stepName);
    const seconds = durationMs / 1000;
    return `${seconds} second${seconds !== 1 ? 's' : ''}`;
}

export function validateSampleCount(stepName: string, actualCount: number): ValidationResult {
    const config = CALIBRATION_CONFIG[stepName] as CalibrationStepConfig | undefined;
    if (!config || typeof config === 'number') {
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
