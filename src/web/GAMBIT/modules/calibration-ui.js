/**
 * Calibration UI Controller
 * Manages calibration steps and UI updates with proper streaming control
 */

import { state } from './state.js';
import { log } from './logger.js';

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
 */
export function updateCalibrationStatus() {
    const $ = (id) => document.getElementById(id);

    const statusText = $('calStatusText');
    const detailsText = $('calDetails');
    const saveBtn = $('saveCalibration');

    if (!calibrationInstance) {
        statusText.textContent = 'Not Initialized';
        detailsText.textContent = 'Calibration module not loaded';
        saveBtn.disabled = true;
        return;
    }

    // Use the class's calibration flags (set by the calibration methods)
    const hasEarth = calibrationInstance.earthFieldCalibrated;
    const hasHardIron = calibrationInstance.hardIronCalibrated;
    const hasSoftIron = calibrationInstance.softIronCalibrated;
    const complete = hasEarth && hasHardIron && hasSoftIron;

    statusText.textContent = complete ? 'Calibrated' : (hasEarth || hasHardIron || hasSoftIron) ? 'Partial' : 'Not Calibrated';
    statusText.style.color = complete ? 'var(--success)' : (hasEarth || hasHardIron || hasSoftIron) ? 'var(--warning)' : 'var(--fg-muted)';

    const steps = [];
    if (hasEarth) steps.push('Earth');
    if (hasHardIron) steps.push('Hard Iron');
    if (hasSoftIron) steps.push('Soft Iron');
    detailsText.textContent = steps.length ? `Complete: ${steps.join(', ')}` : 'No calibration data';

    saveBtn.disabled = !complete;

    $('startEarthCal').disabled = !state.connected;
    $('startHardIronCal').disabled = !state.connected || !hasEarth;
    $('startSoftIronCal').disabled = !state.connected || !hasHardIron;
}

/**
 * Run a calibration step with automatic streaming management
 * FIXED: Now ensures data streaming is active before collecting samples
 *
 * @param {string} stepName - Name of calibration step (e.g., 'earth', 'hardIron')
 * @param {number} durationMs - Duration to collect data in milliseconds
 * @param {Function} sampleHandler - Handler called with calibration result
 * @param {Function} completionHandler - Handler to process collected samples
 */
export async function runCalibrationStep(stepName, durationMs, sampleHandler, completionHandler) {
    if (!state.gambitClient || !state.connected) {
        log('Error: Not connected');
        return;
    }

    const $ = (id) => document.getElementById(id);
    const buffer = [];
    const startTime = Date.now();
    const progressDiv = $(`${stepName}Progress`);
    const qualityDiv = $(`${stepName}Quality`);

    // FIX: Track whether we started streaming (so we can stop it after)
    let streamingStartedByUs = false;

    log(`Starting ${stepName} calibration (${durationMs/1000}s)...`);

    try {
        // FIX: Ensure streaming is active before collecting data
        if (!state.recording) {
            log(`Starting data streaming for calibration...`);
            await state.gambitClient.startStreaming();
            streamingStartedByUs = true;
        }

        const dataHandler = (sample) => {
            buffer.push(sample);
            const elapsed = Date.now() - startTime;
            const progress = Math.min(100, (elapsed / durationMs) * 100);
            progressDiv.textContent = `${progress.toFixed(0)}%`;
        };

        state.gambitClient.on('data', dataHandler);

        // Wait for data collection to complete
        await new Promise((resolve) => {
            setTimeout(() => {
                state.gambitClient.off('data', dataHandler);
                progressDiv.textContent = 'Done';

                // FIX: Stop streaming if we started it (and not recording)
                if (streamingStartedByUs && !state.recording) {
                    state.gambitClient.stopStreaming()
                        .then(() => log('Data streaming stopped'))
                        .catch((e) => console.warn('Failed to stop streaming:', e));
                }

                if (buffer.length < 10) {
                    log(`Error: Insufficient samples (${buffer.length})`);
                    qualityDiv.textContent = '❌ Failed: insufficient data';
                    qualityDiv.style.color = 'var(--danger)';
                    resolve();
                    return;
                }

                try {
                    const result = completionHandler(buffer);
                    sampleHandler(result);
                    log(`${stepName} complete: ${buffer.length} samples`);
                } catch (error) {
                    log(`Error: ${error.message}`);
                    qualityDiv.textContent = `❌ Failed: ${error.message}`;
                    qualityDiv.style.color = 'var(--danger)';
                }

                resolve();
            }, durationMs);
        });

    } catch (error) {
        log(`Error starting calibration: ${error.message}`);
        qualityDiv.textContent = '❌ Failed: could not start streaming';
        qualityDiv.style.color = 'var(--danger)';

        // Clean up streaming if we started it
        if (streamingStartedByUs && !state.recording) {
            state.gambitClient.stopStreaming().catch(() => {});
        }
    }
}

/**
 * Initialize calibration UI controls
 */
export function initCalibrationUI() {
    const $ = (id) => document.getElementById(id);

    // Earth Field Calibration
    const startEarthCal = $('startEarthCal');
    if (startEarthCal) {
        startEarthCal.addEventListener('click', () => {
        runCalibrationStep('earth', 10000,
            (result) => {
                const quality = result.quality > 0.9 ? 'Excellent' : result.quality > 0.7 ? 'Good' : 'Poor';
                const emoji = result.quality > 0.9 ? '✅' : result.quality > 0.7 ? '⚠️' : '❌';
                $('earthQuality').textContent = `${emoji} ${quality} (quality: ${result.quality.toFixed(2)})`;
                $('earthQuality').style.color = result.quality > 0.7 ? 'var(--success)' : 'var(--danger)';

                // Save after each step
                calibrationInstance.save('gambit_calibration');
                updateCalibrationStatus();
            },
            (buffer) => {
                // Convert buffer to samples format expected by the class
                const samples = buffer.map(s => ({x: s.mx, y: s.my, z: s.mz}));
                return calibrationInstance.runEarthFieldCalibration(samples);
            }
        );
        });
    }

    // Hard Iron Calibration
    const startHardIronCal = $('startHardIronCal');
    if (startHardIronCal) {
        startHardIronCal.addEventListener('click', () => {
        runCalibrationStep('hardIron', 20000,
            (result) => {
                const quality = result.quality > 0.9 ? 'Excellent' : result.quality > 0.7 ? 'Good' : 'Poor';
                const emoji = result.quality > 0.9 ? '✅' : result.quality > 0.7 ? '⚠️' : '❌';
                $('hardIronQuality').textContent = `${emoji} ${quality} (sphericity: ${result.quality.toFixed(2)})`;
                $('hardIronQuality').style.color = result.quality > 0.7 ? 'var(--success)' : 'var(--danger)';

                // Save after each step
                calibrationInstance.save('gambit_calibration');
                updateCalibrationStatus();
            },
            (buffer) => {
                // Convert buffer to samples format expected by the class
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
        runCalibrationStep('softIron', 20000,
            (result) => {
                const quality = result.quality > 0.9 ? 'Excellent' : result.quality > 0.7 ? 'Good' : 'Poor';
                const emoji = result.quality > 0.9 ? '✅' : result.quality > 0.7 ? '⚠️' : '❌';
                $('softIronQuality').textContent = `${emoji} ${quality} (quality: ${result.quality.toFixed(2)})`;
                $('softIronQuality').style.color = result.quality > 0.7 ? 'var(--success)' : 'var(--danger)';

                // Save complete calibration
                calibrationInstance.save('gambit_calibration');
                updateCalibrationStatus();
                log('Calibration saved to localStorage');
            },
            (buffer) => {
                // Convert buffer to samples format expected by the class
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
