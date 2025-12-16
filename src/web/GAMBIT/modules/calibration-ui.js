/**
 * Calibration UI Controller
 * Manages calibration steps and UI updates with proper streaming control
 */

import { state } from './state.js';
import { log } from './logger.js';
import { CALIBRATION_CONFIG, validateSampleCount } from '../calibration-config.js';
import { formatFieldData } from '../shared/geomagnetic-field.js';
// Callback for storing calibration session data
let storeCalibrationSession = null;

/**
 * Set callback for storing calibration sessions
 * @param {Function} callback - Function(samples, stepName, result) to store calibration data
 */
export function setStoreSessionCallback(callback) {
    storeCalibrationSession = callback;
}

// Single calibration instance used for both wizard and real-time correction
export let calibrationInstance = null;
export let calibrationInterval = null;

/**
 * Initialize calibration instance
 * @returns {EnvironmentalCalibration} Calibration instance
 */
export function initCalibration() {
    if (typeof EnvironmentalCalibration === 'undefined') {
        console.error('Error: calibration.js not loaded');
        return null;
    }
    calibrationInstance = new EnvironmentalCalibration();

    // Try to load from localStorage using the class method
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

/**
 * Update calibration status UI
 *
 * NOTE: Earth field calibration has been removed from the wizard.
 * Earth field is now estimated automatically in real-time using
 * UnifiedMagCalibration (200-sample sliding window).
 */
export function updateCalibrationStatus() {
    const $ = (id) => document.getElementById(id);

    const statusText = $('calStatusText');
    const detailsText = $('calDetails');
    const saveBtn = $('saveCalibration');

    const geoStr = formatFieldData(state.geomagneticLocation);
    detailsText.textContent = `LOC: ${geoStr.location} ${geoStr.declination}  ${geoStr.intensity}. CAL: `;

    if (!calibrationInstance) {
        statusText.textContent = 'Not Initialized';
        detailsText.textContent += 'Calibration module not loaded';
        saveBtn.disabled = true;
        return;
    }

    // Iron calibration from wizard (Earth field is auto-estimated in real-time)
    const hasHardIron = calibrationInstance.hardIronCalibrated;
    const hasSoftIron = calibrationInstance.softIronCalibrated;
    const complete = hasHardIron && hasSoftIron;

    statusText.textContent = complete ? 'Calibrated' : (hasHardIron || hasSoftIron) ? 'Partial' : 'Not Calibrated';
    statusText.style.color = complete ? 'var(--success)' : (hasHardIron || hasSoftIron) ? 'var(--warning)' : 'var(--fg-muted)';

    const steps = [];
    if (hasHardIron) steps.push('Hard Iron');
    if (hasSoftIron) steps.push('Soft Iron');
    detailsText.textContent += steps.length ? `Complete: ${steps.join(', ')} (Earth: auto)` : 'No iron calibration (Earth: auto)';

    saveBtn.disabled = !complete;

    // Hard Iron enabled immediately, Soft Iron requires Hard Iron first
    const startHardIronBtn = $('startHardIronCal');
    const startSoftIronBtn = $('startSoftIronCal');
    if (startHardIronBtn) startHardIronBtn.disabled = !state.connected;
    if (startSoftIronBtn) startSoftIronBtn.disabled = !state.connected || !hasHardIron;
}

/**
 * Run a calibration step using guaranteed sample collection
 * Uses collectSamples() for exact sample counts
 *
 * @param {string} stepName - Name of calibration step (e.g., 'HARD_IRON', 'SOFT_IRON')
 * @param {Function} sampleHandler - Handler called with calibration result
 * @param {Function} completionHandler - Handler to process collected samples
 */
export async function runCalibrationStep(stepName, sampleHandler, completionHandler) {
    if (!state.gambitClient || !state.connected) {
        log('Error: Not connected');
        return;
    }

    const $ = (id) => document.getElementById(id);
    const config = CALIBRATION_CONFIG[stepName];
    
    const elementName = stepName.toLowerCase().replace('_field', '').replace('_iron', 'Iron');
    const progressDiv = $(`${elementName}Progress`);
    const qualityDiv = $(`${elementName}Quality`);
    const buffer = [];

    if (!config) {
        log(`Error: Unknown calibration step: ${stepName}`);
        return;
    }

    const { sampleCount, sampleRate, minSamples } = config;
    const expectedDuration = Math.ceil((sampleCount / sampleRate) * 1000);

    log(`Starting ${stepName} calibration: ${sampleCount} samples @ ${sampleRate}Hz (~${(expectedDuration/1000).toFixed(1)}s)...`);
    progressDiv.textContent = 'Starting...';

    try {
        const dataHandler = (sample) => {
            buffer.push(sample);
            const progress = Math.min(100, (buffer.length / sampleCount) * 100);
            progressDiv.textContent = `${buffer.length}/${sampleCount} (${progress.toFixed(0)}%)`;
        };

        state.gambitClient.on('data', dataHandler);

        const result = await state.gambitClient.collectSamples(sampleCount, sampleRate);
        
        state.gambitClient.off('data', dataHandler);
        progressDiv.textContent = `✓ ${result.collectedCount} samples in ${(result.durationMs/1000).toFixed(1)}s`;

        const validation = validateSampleCount(stepName, buffer.length);
        if (!validation.valid) {
            log(`Error: Insufficient samples (${validation.actualCount}/${validation.minSamples} minimum)`);
            qualityDiv.textContent = `❌ Failed: only ${validation.actualCount} samples (need ${validation.minSamples}+)`;
            qualityDiv.style.color = 'var(--danger)';
            return;
        }

        if (validation.percentage < 95) {
            log(`Warning: Lower sample count than expected (${validation.actualCount}/${validation.expectedSamples}, ${validation.percentage}%)`);
        }

        try {
            const calibResult = completionHandler(buffer);
            sampleHandler(calibResult);
            log(`${stepName} complete: ${buffer.length} samples, quality=${calibResult.quality?.toFixed(2) || 'N/A'}`);

            // Store calibration session data for export/upload
            if (storeCalibrationSession && buffer.length > 0) {
                storeCalibrationSession(buffer, stepName, calibResult);
            }
        } catch (error) {
            log(`Error: ${error.message}`);
            qualityDiv.textContent = `❌ Failed: ${error.message}`;
            qualityDiv.style.color = 'var(--danger)';
        }

    } catch (error) {
        log(`Error during calibration: ${error.message}`);
        qualityDiv.textContent = `❌ Failed: ${error.message}`;
        qualityDiv.style.color = 'var(--danger)';
        progressDiv.textContent = 'Failed';
    }
}

/**
 * Initialize calibration UI controls
 */
export function initCalibrationUI() {
    const $ = (id) => document.getElementById(id);

    // NOTE: Earth Field Calibration has been removed from the wizard.
    // Earth field is now auto-estimated in real-time by UnifiedMagCalibration.

    // Hard Iron Calibration
    const startHardIronCal = $('startHardIronCal');
    if (startHardIronCal) {
        startHardIronCal.addEventListener('click', () => {
        runCalibrationStep('HARD_IRON',
            (result) => {
                const thresholds = CALIBRATION_CONFIG.HARD_IRON.qualityThresholds;
                const quality = result.quality > thresholds.excellent ? 'Excellent' : result.quality > thresholds.good ? 'Good' : 'Poor';
                const emoji = result.quality > thresholds.excellent ? '✅' : result.quality > thresholds.good ? '⚠️' : '❌';
                $('hardIronQuality').textContent = `${emoji} ${quality} (sphericity: ${result.quality.toFixed(2)})`;
                $('hardIronQuality').style.color = result.quality > thresholds.good ? 'var(--success)' : 'var(--danger)';

                calibrationInstance.save('gambit_calibration');
                updateCalibrationStatus();
            },
            (buffer) => {
                const samples = buffer.map(s => ({x: s.mx, y: s.my, z: s.mz}));
                return calibrationInstance.runHardIronCalibration(samples);
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
                const thresholds = CALIBRATION_CONFIG.SOFT_IRON.qualityThresholds;
                const quality = result.quality > thresholds.excellent ? 'Excellent' : result.quality > thresholds.good ? 'Good' : 'Poor';
                const emoji = result.quality > thresholds.excellent ? '✅' : result.quality > thresholds.good ? '⚠️' : '❌';
                $('softIronQuality').textContent = `${emoji} ${quality} (quality: ${result.quality.toFixed(2)})`;
                $('softIronQuality').style.color = result.quality > thresholds.good ? 'var(--success)' : 'var(--danger)';

                calibrationInstance.save('gambit_calibration');
                updateCalibrationStatus();
                log('Calibration saved to localStorage');
            },
            (buffer) => {
                const samples = buffer.map(s => ({x: s.mx, y: s.my, z: s.mz}));
                return calibrationInstance.runSoftIronCalibration(samples);
            }
        );
        });
    }

    // Save Calibration to File
    const saveCalibration = $('saveCalibration');
    if (saveCalibration) {
        saveCalibration.addEventListener('click', () => {
        // Use the class's toJSON method for consistent format
        const calData = {
            ...calibrationInstance.toJSON(),
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
            const file = e.target.files[0];
            if (!file) return;

            try {
                const text = await file.text();
                const data = JSON.parse(text);

                // Use the class's fromJSON method
                calibrationInstance.fromJSON(data);
                calibrationInstance.save('gambit_calibration');

                log('Calibration loaded successfully');
                updateCalibrationStatus();
            } catch (error) {
                log(`Error loading calibration: ${error.message}`);
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
                calibrationInstance = initCalibration(); // Reset to fresh instance
                localStorage.removeItem('gambit_calibration');
                log('Calibration cleared');
                updateCalibrationStatus();
            }
        });
    }
}
