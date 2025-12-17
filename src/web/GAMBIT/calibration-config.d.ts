/**
 * Type declarations for calibration-config.js
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
}

export interface ValidationResult {
  valid: boolean;
  actualCount: number;
  minSamples: number;
  expectedSamples: number;
  percentage: number;
}

export const CALIBRATION_CONFIG: CalibrationConfigType;

export function getDuration(stepName: string): number;
export function formatDuration(stepName: string): string;
export function validateSampleCount(stepName: string, actualCount: number): ValidationResult;
