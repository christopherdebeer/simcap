/**
 * Calibration UI Controller
 * Manages calibration steps and UI updates with proper streaming control
 */

import { state } from './state.js';
import { log } from './logger.js';
import { CALIBRATION_CONFIG, validateSampleCount } from '../calibration-config.js';
import { formatFieldData } from '../shared/geomagnetic-field.js';
import { UnifiedMagCalibration } from '../shared/unified-mag-calibration.js';
import { magLsbToMicroTesla } from '../shared/sensor-units.js';
import type { TelemetrySample } from '@core/types';

// ===== Type Definitions =====

export interface CalibrationResult {
  quality?: number;
  offset?: { x: number; y: number; z: number };
  [key: string]: any;
}

export interface CalibrationSample {
  mx: number;
  my: number;
  mz: number;
  [key: string]: any;
}

export interface CollectSamplesResult {
  collectedCount: number;
  durationMs: number;
}

export interface ValidationResult {
  valid: boolean;
  actualCount: number;
  minSamples: number;
  expectedSamples: number;
  percentage: number;
}

export interface BaselineCaptureResult {
  success: boolean;
  magnitude?: number;
  quality?: string;
  reason?: string;
  suggestion?: string;
}

type StoreSessionCallback = (samples: CalibrationSample[], stepName: string, result: CalibrationResult) => void;
type SampleHandler = (result: CalibrationResult) => void;
type CompletionHandler = (buffer: CalibrationSample[]) => CalibrationResult;

// ===== Module State =====

let storeCalibrationSession: StoreSessionCallback | null = null;

export let calibrationInstance: UnifiedMagCalibration | null = null;
export let calibrationInterval: ReturnType<typeof setInterval> | null = null;

// ===== Setup Functions =====

/**
 * Set callback for storing calibration sessions
 * @param callback - Function(samples, stepName, result) to store calibration data
 */
export function setStoreSessionCallback(callback: StoreSessionCallback | null): void {
    storeCalibrationSession = callback;
}

/**
 * Initialize calibration instance
 * @returns Calibration instance
 */
export function initCalibration(): UnifiedMagCalibration {
    calibrationInstance = new UnifiedMagCalibration();

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

// ===== UI Functions =====

/**
 * Update calibration status UI
 */
export function updateCalibrationStatus(): void {
    const $ = (id: string): HTMLElement | null => document.getElementById(id);

    const statusText = $('calStatusText');
    const detailsText = $('calDetails');
    const saveBtn = $('saveCalibration') as HTMLButtonElement | null;

    const geoStr = formatFieldData(state.geomagneticLocation);
    if (detailsText && geoStr) {
        detailsText.textContent = `LOC: ${geoStr.location} ${geoStr.declination}  ${geoStr.intensity}. CAL: `;
    }

    if (!calibrationInstance) {
        if (statusText) statusText.textContent = 'Not Initialized';
        if (detailsText) detailsText.textContent += 'Calibration module not loaded';
        if (saveBtn) saveBtn.disabled = true;
        return;
    }

    const hasHardIron = calibrationInstance.hardIronCalibrated;
    const hasSoftIron = calibrationInstance.softIronCalibrated;
    const complete = hasHardIron && hasSoftIron;

    if (statusText) {
        statusText.textContent = complete ? 'Calibrated' : (hasHardIron || hasSoftIron) ? 'Partial' : 'Not Calibrated';
        statusText.style.color = complete ? 'var(--success)' : (hasHardIron || hasSoftIron) ? 'var(--warning)' : 'var(--fg-muted)';
    }

    const steps: string[] = [];
    if (hasHardIron) steps.push('Hard Iron');
    if (hasSoftIron) steps.push('Soft Iron');
    if (detailsText) {
        detailsText.textContent += steps.length ? `Complete: ${steps.join(', ')} (Earth: auto)` : 'No iron calibration (Earth: auto)';
    }

    if (saveBtn) saveBtn.disabled = !complete;

    const startHardIronBtn = $('startHardIronCal') as HTMLButtonElement | null;
    const startSoftIronBtn = $('startSoftIronCal') as HTMLButtonElement | null;
    const startBaselineBtn = $('startBaselineCal') as HTMLButtonElement | null;
    if (startHardIronBtn) startHardIronBtn.disabled = !state.connected;
    if (startSoftIronBtn) startSoftIronBtn.disabled = !state.connected || !hasHardIron;
    if (startBaselineBtn) startBaselineBtn.disabled = !state.connected;

    const baselineStatus = $('baselineStatus');
    const baselineQuality = $('baselineQuality');
    if (calibrationInstance) {
        const calState = calibrationInstance.getState();

        if (baselineStatus) {
            if (calState.extendedBaselineActive) {
                const mag = calState.extendedBaselineMagnitude;
                const quality = mag < 60 ? 'Excellent' : mag < 80 ? 'Good' : 'Acceptable';
                baselineStatus.textContent = `✓ Auto-captured (${quality}, ${mag.toFixed(1)} µT)`;
                baselineStatus.style.color = 'var(--success)';
            } else if (calState.capturingBaseline) {
                baselineStatus.textContent = `⏳ Capturing... (${calState.baselineSampleCount} samples)`;
                baselineStatus.style.color = 'var(--fg-muted)';
            } else if (calState.autoBaselineRetryCount >= calState.autoBaselineMaxRetries) {
                baselineStatus.textContent = `⚠️ Auto-capture failed (magnets detected?) - use manual recapture`;
                baselineStatus.style.color = 'var(--warning)';
            } else {
                baselineStatus.textContent = `○ Waiting for samples...`;
                baselineStatus.style.color = 'var(--fg-muted)';
            }
        }

        if (baselineQuality && calState.extendedBaselineActive) {
            const mag = calState.extendedBaselineMagnitude;
            const quality = mag < 60 ? 'Excellent' : mag < 80 ? 'Good' : 'Acceptable';
            const emoji = mag < 60 ? '✅' : mag < 80 ? '⚠️' : '⚠️';
            baselineQuality.textContent = `${emoji} ${quality} (magnitude: ${mag.toFixed(1)} µT)`;
            baselineQuality.style.color = mag < 80 ? 'var(--success)' : 'var(--warning)';
        }
    }
}

// ===== Calibration Step Runner =====

/**
 * Run a calibration step using guaranteed sample collection
 */
export async function runCalibrationStep(
    stepName: string,
    sampleHandler: SampleHandler,
    completionHandler: CompletionHandler
): Promise<void> {
    if (!state.gambitClient || !state.connected) {
        log('Error: Not connected');
        return;
    }

    const $ = (id: string): HTMLElement | null => document.getElementById(id);
    const config = (CALIBRATION_CONFIG as any)[stepName];

    const elementName = stepName.toLowerCase().replace('_field', '').replace('_iron', 'Iron');
    const progressDiv = $(`${elementName}Progress`);
    const qualityDiv = $(`${elementName}Quality`);
    const buffer: CalibrationSample[] = [];

    if (!config) {
        log(`Error: Unknown calibration step: ${stepName}`);
        return;
    }

    const { sampleCount, sampleRate, minSamples } = config;
    const expectedDuration = Math.ceil((sampleCount / sampleRate) * 1000);

    log(`Starting ${stepName} calibration: ${sampleCount} samples @ ${sampleRate}Hz (~${(expectedDuration/1000).toFixed(1)}s)...`);
    if (progressDiv) progressDiv.textContent = 'Starting...';

    try {
        const dataHandler = (sample: TelemetrySample) => {
            buffer.push(sample as CalibrationSample);
            const progress = Math.min(100, (buffer.length / sampleCount) * 100);
            if (progressDiv) progressDiv.textContent = `${buffer.length}/${sampleCount} (${progress.toFixed(0)}%)`;
        };

        state.gambitClient.on('data', dataHandler);

        const result: CollectSamplesResult = await state.gambitClient.collectSamples(sampleCount, sampleRate);

        state.gambitClient.off('data', dataHandler);
        if (progressDiv) progressDiv.textContent = `✓ ${result.collectedCount} samples in ${(result.durationMs/1000).toFixed(1)}s`;

        const validation: ValidationResult = validateSampleCount(stepName, buffer.length);
        if (!validation.valid) {
            log(`Error: Insufficient samples (${validation.actualCount}/${validation.minSamples} minimum)`);
            if (qualityDiv) {
                qualityDiv.textContent = `❌ Failed: only ${validation.actualCount} samples (need ${validation.minSamples}+)`;
                qualityDiv.style.color = 'var(--danger)';
            }
            return;
        }

        if (validation.percentage < 95) {
            log(`Warning: Lower sample count than expected (${validation.actualCount}/${validation.expectedSamples}, ${validation.percentage}%)`);
        }

        try {
            const calibResult = completionHandler(buffer);
            sampleHandler(calibResult);
            log(`${stepName} complete: ${buffer.length} samples, quality=${calibResult.quality?.toFixed(2) || 'N/A'}`);

            if (storeCalibrationSession && buffer.length > 0) {
                storeCalibrationSession(buffer, stepName, calibResult);
            }
        } catch (error) {
            const err = error as Error;
            log(`Error: ${err.message}`);
            if (qualityDiv) {
                qualityDiv.textContent = `❌ Failed: ${err.message}`;
                qualityDiv.style.color = 'var(--danger)';
            }
        }

    } catch (error) {
        const err = error as Error;
        log(`Error during calibration: ${err.message}`);
        if (qualityDiv) {
            qualityDiv.textContent = `❌ Failed: ${err.message}`;
            qualityDiv.style.color = 'var(--danger)';
        }
        if (progressDiv) progressDiv.textContent = 'Failed';
    }
}

// ===== UI Initialization =====

/**
 * Initialize calibration UI controls
 */
export function initCalibrationUI(): void {
    const $ = (id: string): HTMLElement | null => document.getElementById(id);

    // Hard Iron Calibration
    const startHardIronCal = $('startHardIronCal');
    if (startHardIronCal) {
        startHardIronCal.addEventListener('click', () => {
            runCalibrationStep('HARD_IRON',
                (result) => {
                    const thresholds = (CALIBRATION_CONFIG as any).HARD_IRON.qualityThresholds;
                    const quality = (result.quality || 0) > thresholds.excellent ? 'Excellent' : (result.quality || 0) > thresholds.good ? 'Good' : 'Poor';
                    const emoji = (result.quality || 0) > thresholds.excellent ? '✅' : (result.quality || 0) > thresholds.good ? '⚠️' : '❌';
                    const hardIronQuality = $('hardIronQuality');
                    if (hardIronQuality) {
                        hardIronQuality.textContent = `${emoji} ${quality} (sphericity: ${(result.quality || 0).toFixed(2)})`;
                        hardIronQuality.style.color = (result.quality || 0) > thresholds.good ? 'var(--success)' : 'var(--danger)';
                    }

                    calibrationInstance?.save('gambit_calibration');
                    updateCalibrationStatus();
                },
                (buffer) => {
                    const samples = buffer.map(s => ({
                        x: magLsbToMicroTesla(s.mx),
                        y: magLsbToMicroTesla(s.my),
                        z: magLsbToMicroTesla(s.mz)
                    }));
                    return calibrationInstance!.runHardIronCalibration(samples);
                }
            );
        });
    }

    // Soft Iron Calibration
    const startSoftIronCal = $('startSoftIronCal');
    if (startSoftIronCal) {
        startSoftIronCal.addEventListener('click', () => {
            runCalibrationStep('SOFT_IRON',
                (result) => {
                    const thresholds = (CALIBRATION_CONFIG as any).SOFT_IRON.qualityThresholds;
                    const quality = (result.quality || 0) > thresholds.excellent ? 'Excellent' : (result.quality || 0) > thresholds.good ? 'Good' : 'Poor';
                    const emoji = (result.quality || 0) > thresholds.excellent ? '✅' : (result.quality || 0) > thresholds.good ? '⚠️' : '❌';
                    const softIronQuality = $('softIronQuality');
                    if (softIronQuality) {
                        softIronQuality.textContent = `${emoji} ${quality} (quality: ${(result.quality || 0).toFixed(2)})`;
                        softIronQuality.style.color = (result.quality || 0) > thresholds.good ? 'var(--success)' : 'var(--danger)';
                    }

                    calibrationInstance?.save('gambit_calibration');
                    updateCalibrationStatus();
                    log('Calibration saved to localStorage');
                },
                (buffer) => {
                    const samples = buffer.map(s => ({
                        x: magLsbToMicroTesla(s.mx),
                        y: magLsbToMicroTesla(s.my),
                        z: magLsbToMicroTesla(s.mz)
                    }));
                    return calibrationInstance!.runSoftIronCalibration(samples);
                }
            );
        });
    }

    // Extended Baseline Manual Recapture
    const startBaselineCal = $('startBaselineCal');
    if (startBaselineCal) {
        startBaselineCal.addEventListener('click', async () => {
            if (!state.gambitClient || !state.connected) {
                log('Error: Not connected');
                return;
            }

            const progressDiv = $('baselineProgress');
            const qualityDiv = $('baselineQuality');
            const statusDiv = $('baselineStatus');

            const sampleCount = 78;
            const sampleRate = 26;

            log('Starting Extended Baseline recapture: hold hand still with fingers extended...');
            if (progressDiv) progressDiv.textContent = 'Starting...';
            if (qualityDiv) qualityDiv.textContent = '';
            if (statusDiv) statusDiv.textContent = '⏳ Manual capture in progress...';

            calibrationInstance?.clearExtendedBaseline();
            calibrationInstance?.startBaselineCapture();

            try {
                const dataHandler = (sample: TelemetrySample) => {
                    const magUT = {
                        x: magLsbToMicroTesla(sample.mx),
                        y: magLsbToMicroTesla(sample.my),
                        z: magLsbToMicroTesla(sample.mz)
                    };

                    const orientation = (state as any).ahrs?.getQuaternion() || { w: 1, x: 0, y: 0, z: 0 };
                    calibrationInstance?.update(magUT.x, magUT.y, magUT.z, orientation);
                };

                state.gambitClient.on('data', dataHandler);

                const result: CollectSamplesResult = await state.gambitClient.collectSamples(sampleCount, sampleRate);

                state.gambitClient.off('data', dataHandler);
                if (progressDiv) progressDiv.textContent = `✓ ${result.collectedCount} samples in ${(result.durationMs/1000).toFixed(1)}s`;

                const baselineResult: BaselineCaptureResult = calibrationInstance!.endBaselineCapture();

                if (baselineResult.success && baselineResult.magnitude !== undefined && baselineResult.quality) {
                    const mag = baselineResult.magnitude;
                    const quality = baselineResult.quality.charAt(0).toUpperCase() + baselineResult.quality.slice(1);
                    const emoji = mag < 60 ? '✅' : mag < 80 ? '⚠️' : '⚠️';
                    if (qualityDiv) {
                        qualityDiv.textContent = `${emoji} ${quality} (magnitude: ${mag.toFixed(1)} µT)`;
                        qualityDiv.style.color = mag < 80 ? 'var(--success)' : 'var(--warning)';
                    }

                    calibrationInstance?.save('gambit_calibration');
                    log(`Extended Baseline captured: ${mag.toFixed(1)} µT (${quality})`);
                } else {
                    if (qualityDiv) {
                        qualityDiv.textContent = `❌ Failed: ${baselineResult.reason}`;
                        qualityDiv.style.color = 'var(--danger)';
                        if (baselineResult.suggestion) {
                            qualityDiv.textContent += ` - ${baselineResult.suggestion}`;
                        }
                    }
                    log(`Baseline capture failed: ${baselineResult.reason}`);
                }

                updateCalibrationStatus();

            } catch (error) {
                const err = error as Error;
                log(`Error during baseline capture: ${err.message}`);
                if (qualityDiv) {
                    qualityDiv.textContent = `❌ Failed: ${err.message}`;
                    qualityDiv.style.color = 'var(--danger)';
                }
                if (progressDiv) progressDiv.textContent = 'Failed';
            }
        });
    }

    // Save Calibration to File
    const saveCalibration = $('saveCalibration');
    if (saveCalibration) {
        saveCalibration.addEventListener('click', () => {
            const calData = {
                ...calibrationInstance?.toJSON(),
                timestamp: new Date().toISOString()
            };

            const blob = new Blob([JSON.stringify(calData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'gambit_calibration.json';
            a.click();
            URL.revokeObjectURL(url);

            log('Calibration file downloaded');
        });
    }

    // Load Calibration from File
    const loadCalibration = $('loadCalibration');
    if (loadCalibration) {
        loadCalibration.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'application/json';
            input.onchange = async (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (!file) return;

                try {
                    const text = await file.text();
                    const data = JSON.parse(text);

                    calibrationInstance?.fromJSON(data);
                    calibrationInstance?.save('gambit_calibration');

                    log('Calibration loaded successfully');
                    updateCalibrationStatus();
                } catch (error) {
                    const err = error as Error;
                    log(`Error loading calibration: ${err.message}`);
                }
            };
            input.click();
        });
    }

    // Reset Calibration
    const resetCalibration = $('resetCalibration');
    if (resetCalibration) {
        resetCalibration.addEventListener('click', () => {
            if (confirm('Clear all calibration data? This cannot be undone.')) {
                calibrationInstance = initCalibration();
                localStorage.removeItem('gambit_calibration');
                log('Calibration cleared');
                updateCalibrationStatus();
            }
        });
    }
}

// ===== Default Export =====

export default {
    setStoreSessionCallback,
    initCalibration,
    updateCalibrationStatus,
    runCalibrationStep,
    initCalibrationUI,
    calibrationInstance,
    calibrationInterval
};
