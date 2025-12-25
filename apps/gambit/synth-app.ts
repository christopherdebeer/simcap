/**
 * GAMBIT Synth Application
 * Real-time sensor-reactive audio synthesis
 *
 * SYNCHRONIZED with collector via shared sensor configuration.
 * Uses physical units (g, °/s, µT) for consistent parameter mapping.
 */

import { GambitClient } from './gambit-client';
import {
    ACCEL_SPEC,
    GYRO_SPEC,
    MAG_SPEC,
    accelLsbToG,
    gyroLsbToDps,
    magLsbToMicroTesla
} from './shared/sensor-units';

// ===== Type Definitions =====

// Extended telemetry including environmental sensors from firmware
interface SynthTelemetry {
    ax: number; ay: number; az: number;  // Accelerometer (LSB)
    gx: number; gy: number; gz: number;  // Gyroscope (LSB)
    mx: number; my: number; mz: number;  // Magnetometer (LSB)
    t: number;   // Timestamp
    l: number;   // Light sensor (0-1)
    c: number;   // Capacitive sensor
    b: number;   // Battery percentage
    s: number;   // Status
    n: number;   // Sample number
}

interface SensorRanges {
    [key: string]: { min: number; max: number };
}

// Sensor data in PHYSICAL UNITS (converted from LSB)
interface SensorData {
    // Physical units (for audio parameter mapping)
    ax_g: number; ay_g: number; az_g: number;     // Acceleration in g
    gx_dps: number; gy_dps: number; gz_dps: number; // Angular velocity in °/s
    mx_ut: number; my_ut: number; mz_ut: number;   // Magnetic field in µT (aligned to accel frame)
    // Environmental (already in usable units)
    l: number; t: number; c: number;
    b: number; s: number; n: number;
}

// ===== Audio State =====

let audioContext: AudioContext | null = null;
let oscillator: OscillatorNode | null = null;
let subOscillator: OscillatorNode | null = null;  // Sub-oscillator for bass
let subGain: GainNode | null = null;              // Sub-oscillator level
let gainNode: GainNode | null = null;
let filterNode: BiquadFilterNode | null = null;
let lfoOscillator: OscillatorNode | null = null;
let lfoGain: GainNode | null = null;
let analyser: AnalyserNode | null = null;
let reverbNode: ConvolverNode | null = null;
let dryGain: GainNode | null = null;
let wetGain: GainNode | null = null;
let compressor: DynamicsCompressorNode | null = null;  // Dynamics processing
let isPlaying = false;

// Smoothing time constant (seconds) - adjustable for responsiveness vs smoothness
let smoothingTime = 0.02;  // 20ms default, good balance

// ===== GAMBIT Client State =====

let gambitClient: GambitClient | null = null;
let isConnected = false;

// ===== Sensor Configuration =====
// Ranges in PHYSICAL UNITS (synchronized with collector)
// These define the expected operating range for audio parameter mapping

const ranges: SensorRanges = {
    // Accelerometer in g (±2g range, typical motion ±1g)
    ax_g: { min: -2, max: 2 },
    ay_g: { min: -2, max: 2 },
    az_g: { min: -2, max: 2 },
    // Gyroscope in °/s (±245°/s range, typical motion ±100°/s)
    gx_dps: { min: -200, max: 200 },
    gy_dps: { min: -200, max: 200 },
    gz_dps: { min: -200, max: 200 },
    // Magnetometer in µT (Earth field ~25-65 µT, with finger magnets up to ±100 µT)
    mx_ut: { min: -100, max: 100 },
    my_ut: { min: -100, max: 100 },
    mz_ut: { min: -100, max: 100 },
    // Environmental sensors (unchanged)
    l: { min: 0, max: 1 },
    t: { min: 15, max: 35 },
    c: { min: 0, max: 100 }
};

let sensorData: SensorData = {
    ax_g: 0, ay_g: 0, az_g: 0,
    gx_dps: 0, gy_dps: 0, gz_dps: 0,
    mx_ut: 0, my_ut: 0, mz_ut: 0,
    l: 0, t: 0, c: 0,
    b: 0, s: 0, n: 0
};

// ===== UI Elements =====

let connectBtn: HTMLButtonElement;
let toggleSynthBtn: HTMLButtonElement;
let statusDiv: HTMLElement;
let batteryDiv: HTMLElement;
let batteryFill: HTMLElement;
let batteryText: HTMLElement;
let oscTypeSelect: HTMLSelectElement;
let baseFreqSlider: HTMLInputElement;
let baseFreqValue: HTMLElement;
let filterQSlider: HTMLInputElement;
let filterQValue: HTMLElement;
let reverbMixSlider: HTMLInputElement;
let reverbMixValue: HTMLElement;
let canvas: HTMLCanvasElement;
let canvasCtx: CanvasRenderingContext2D;

// ===== Helper Functions =====

function resizeCanvas(): void {
    if (canvas) {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
}

function normalize(value: number, key: string): number {
    const range = ranges[key];
    if (!range) return 0.5;
    return Math.max(0, Math.min(1, (value - range.min) / (range.max - range.min)));
}

// ===== Audio Functions =====

/**
 * Create a higher quality reverb impulse response
 * Uses multiple decay stages for more natural room simulation
 */
function createReverbImpulse(): AudioBuffer {
    const sampleRate = audioContext!.sampleRate;
    const length = sampleRate * 2.5; // 2.5 second reverb
    const impulse = audioContext!.createBuffer(2, length, sampleRate);
    const leftChannel = impulse.getChannelData(0);
    const rightChannel = impulse.getChannelData(1);

    // Multi-stage decay for more natural reverb
    // Early reflections (first 50ms) - higher density, faster decay
    // Late reverb (50ms+) - diffuse, slower decay
    const earlyReflectionEnd = sampleRate * 0.05;
    const predelay = sampleRate * 0.01; // 10ms predelay

    for (let i = 0; i < length; i++) {
        let sample: number;

        if (i < predelay) {
            // Predelay - silence
            sample = 0;
        } else if (i < earlyReflectionEnd) {
            // Early reflections - sharper attack, faster decay
            const t = (i - predelay) / sampleRate;
            const earlyDecay = Math.exp(-t / 0.02); // 20ms decay
            sample = (Math.random() * 2 - 1) * earlyDecay * 0.8;
        } else {
            // Late reverb - diffuse, slower decay with modulation
            const t = i / sampleRate;
            const lateDecay = Math.exp(-t / 0.8); // 800ms decay
            // Add subtle modulation for richness
            const mod = 1 + 0.1 * Math.sin(i * 0.0001);
            sample = (Math.random() * 2 - 1) * lateDecay * mod * 0.5;
        }

        // Slight stereo decorrelation for width
        leftChannel[i] = sample;
        rightChannel[i] = sample * (0.9 + 0.1 * Math.random());
    }

    // Apply a gentle high-frequency rolloff (simple 1-pole lowpass)
    let prevL = 0, prevR = 0;
    const lpfCoeff = 0.7;
    for (let i = 0; i < length; i++) {
        leftChannel[i] = prevL = lpfCoeff * prevL + (1 - lpfCoeff) * leftChannel[i];
        rightChannel[i] = prevR = lpfCoeff * prevR + (1 - lpfCoeff) * rightChannel[i];
    }

    return impulse;
}

function updateReverbMix(): void {
    if (!dryGain || !wetGain) return;
    const mix = parseFloat(reverbMixSlider.value);
    dryGain.gain.value = 1 - mix;
    wetGain.gain.value = mix;
}

function initAudio(): void {
    audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

    // Main oscillator
    oscillator = audioContext.createOscillator();
    oscillator.type = oscTypeSelect.value as OscillatorType;
    oscillator.frequency.value = parseFloat(baseFreqSlider.value);

    // Sub-oscillator (one octave below, sine wave for clean bass)
    subOscillator = audioContext.createOscillator();
    subOscillator.type = 'sine';
    subOscillator.frequency.value = parseFloat(baseFreqSlider.value) / 2;
    subGain = audioContext.createGain();
    subGain.gain.value = 0.3;  // Sub level relative to main

    // Filter
    filterNode = audioContext.createBiquadFilter();
    filterNode.type = 'lowpass';
    filterNode.frequency.value = 2000;
    filterNode.Q.value = parseFloat(filterQSlider.value);

    // LFO for vibrato/tremolo
    lfoOscillator = audioContext.createOscillator();
    lfoOscillator.frequency.value = 5;
    lfoGain = audioContext.createGain();
    lfoGain.gain.value = 50;

    // Gain node for volume control
    gainNode = audioContext.createGain();
    gainNode.gain.value = 0.3;

    // Dynamics compressor for cleaner output
    compressor = audioContext.createDynamicsCompressor();
    compressor.threshold.value = -24;  // Start compressing at -24dB
    compressor.knee.value = 12;        // Soft knee for natural sound
    compressor.ratio.value = 4;        // 4:1 compression ratio
    compressor.attack.value = 0.003;   // 3ms attack
    compressor.release.value = 0.1;    // 100ms release

    // Analyser for visualization
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;

    // High quality reverb
    reverbNode = audioContext.createConvolver();
    reverbNode.buffer = createReverbImpulse();

    // Dry/Wet gain nodes for reverb mix
    dryGain = audioContext.createGain();
    wetGain = audioContext.createGain();

    // Connect LFO to main oscillator frequency
    lfoOscillator.connect(lfoGain);
    lfoGain.connect(oscillator.frequency);

    // Audio graph:
    // Main oscillator ─┐
    //                  ├──> filter -> dry/wet split -> gain -> compressor -> analyser -> output
    // Sub oscillator ──┘
    oscillator.connect(filterNode);
    subOscillator.connect(subGain);
    subGain.connect(filterNode);

    filterNode.connect(dryGain);
    filterNode.connect(reverbNode);
    reverbNode.connect(wetGain);

    dryGain.connect(gainNode);
    wetGain.connect(gainNode);
    gainNode.connect(compressor);
    compressor.connect(analyser);
    analyser.connect(audioContext.destination);

    // Start oscillators
    oscillator.start();
    subOscillator.start();
    lfoOscillator.start();

    // Update dry/wet mix
    updateReverbMix();

    isPlaying = true;
    toggleSynthBtn.textContent = 'Stop Synth';
    toggleSynthBtn.classList.add('active');
    statusDiv.textContent = 'Synthesizer Active - Move device to control sound';
    statusDiv.classList.add('playing');

    // Start visualization
    visualize();
}

function stopAudio(): void {
    if (oscillator) {
        oscillator.stop();
        oscillator = null;
    }
    if (subOscillator) {
        subOscillator.stop();
        subOscillator = null;
    }
    if (lfoOscillator) {
        lfoOscillator.stop();
        lfoOscillator = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    // Reset other audio nodes
    subGain = null;
    compressor = null;

    isPlaying = false;
    toggleSynthBtn.textContent = 'Start Synth';
    toggleSynthBtn.classList.remove('active');
    statusDiv.classList.remove('playing');
    if (isConnected) {
        statusDiv.textContent = 'Connected - Press device button to start data capture';
        statusDiv.classList.add('connected');
    }
}

function updateSynthesis(): void {
    if (!isPlaying || !oscillator || !audioContext) return;

    const baseFreq = parseFloat(baseFreqSlider.value);
    const time = audioContext.currentTime;

    // Frequency control from accelerometer Y (in g, ±2g range maps to ±2 octaves)
    const ayNorm = normalize(sensorData.ay_g, 'ay_g');
    const freqMultiplier = Math.pow(2, (ayNorm - 0.5) * 4);
    const targetFreq = baseFreq * freqMultiplier;
    oscillator.frequency.setTargetAtTime(targetFreq, time, smoothingTime);
    // Update sub-oscillator (one octave below)
    if (subOscillator) {
        subOscillator.frequency.setTargetAtTime(targetFreq / 2, time, smoothingTime);
    }
    document.getElementById('freqParam')!.textContent = Math.round(targetFreq) + ' Hz';

    // Filter cutoff from gyroscope Z (in °/s)
    const gzNorm = normalize(sensorData.gz_dps, 'gz_dps');
    const filterFreq = 200 + gzNorm * 4800;
    filterNode!.frequency.setTargetAtTime(filterFreq, time, smoothingTime);
    document.getElementById('filterParam')!.textContent = Math.round(filterFreq) + ' Hz';

    // Volume from light sensor (already 0-1)
    const lightNorm = sensorData.l;
    const volume = Math.max(0.1, lightNorm) * 0.5;
    gainNode!.gain.setTargetAtTime(volume, time, smoothingTime);
    document.getElementById('volumeParam')!.textContent = Math.round(volume * 100) + '%';

    // LFO rate from magnetometer X (in µT, aligned to accel frame)
    const mxNorm = normalize(sensorData.mx_ut, 'mx_ut');
    const lfoRate = 0.5 + mxNorm * 15;
    lfoOscillator!.frequency.setTargetAtTime(lfoRate, time, smoothingTime);
    document.getElementById('lfoParam')!.textContent = lfoRate.toFixed(2) + ' Hz';

    // Detune from gyroscope X (in °/s)
    const gxNorm = normalize(sensorData.gx_dps, 'gx_dps');
    const detune = (gxNorm - 0.5) * 100;
    oscillator.detune.setTargetAtTime(detune, time, smoothingTime);
    document.getElementById('detuneParam')!.textContent = Math.round(detune) + ' cents';

    // Reverb mix from accelerometer Z (in g)
    const azNorm = normalize(sensorData.az_g, 'az_g');
    const reverbAmount = azNorm;
    dryGain!.gain.setTargetAtTime(1 - reverbAmount, time, smoothingTime);
    wetGain!.gain.setTargetAtTime(reverbAmount, time, smoothingTime);
    document.getElementById('reverbParam')!.textContent = Math.round(reverbAmount * 100) + '%';
}

function visualize(): void {
    if (!isPlaying || !analyser) return;

    requestAnimationFrame(visualize);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#0a0a0a';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--success').trim() || '#00ff88';
    canvasCtx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

// ===== Display Functions =====

function updateSensorDisplay(): void {
    // Display in physical units (matching collector)
    document.getElementById('ax')!.textContent = sensorData.ax_g.toFixed(3) + ' g';
    document.getElementById('ay')!.textContent = sensorData.ay_g.toFixed(3) + ' g';
    document.getElementById('az')!.textContent = sensorData.az_g.toFixed(3) + ' g';
    document.getElementById('gx')!.textContent = sensorData.gx_dps.toFixed(1) + ' °/s';
    document.getElementById('gy')!.textContent = sensorData.gy_dps.toFixed(1) + ' °/s';
    document.getElementById('gz')!.textContent = sensorData.gz_dps.toFixed(1) + ' °/s';
    document.getElementById('mx')!.textContent = sensorData.mx_ut.toFixed(1) + ' µT';
    document.getElementById('my')!.textContent = sensorData.my_ut.toFixed(1) + ' µT';
    document.getElementById('mz')!.textContent = sensorData.mz_ut.toFixed(1) + ' µT';
    document.getElementById('light')!.textContent = sensorData.l.toFixed(3);
    document.getElementById('temp')!.textContent = sensorData.t.toFixed(1) + ' °C';
    document.getElementById('cap')!.textContent = sensorData.c.toFixed(0);
}

function updateBattery(percent: number): void {
    batteryDiv.style.display = 'flex';
    batteryFill.style.width = percent + '%';
    batteryText.textContent = percent + '%';

    if (percent < 20) {
        batteryFill.style.background = 'var(--danger)';
    } else if (percent < 50) {
        batteryFill.style.background = 'var(--warning)';
    } else {
        batteryFill.style.background = 'var(--success)';
    }
}

// ===== Telemetry Handler =====

/**
 * Process incoming telemetry from GAMBIT device.
 * Converts raw LSB values to physical units and applies axis alignment.
 * This matches the collector's telemetry-processor.ts for consistency.
 */
function onTelemetry(telemetry: SynthTelemetry): void {
    // Convert accelerometer: LSB → g
    const ax_g = accelLsbToG(telemetry.ax || 0);
    const ay_g = accelLsbToG(telemetry.ay || 0);
    const az_g = accelLsbToG(telemetry.az || 0);

    // Convert gyroscope: LSB → °/s
    const gx_dps = gyroLsbToDps(telemetry.gx || 0);
    const gy_dps = gyroLsbToDps(telemetry.gy || 0);
    const gz_dps = gyroLsbToDps(telemetry.gz || 0);

    // Convert magnetometer: LSB → µT
    const mx_ut_raw = magLsbToMicroTesla(telemetry.mx || 0);
    const my_ut_raw = magLsbToMicroTesla(telemetry.my || 0);
    const mz_ut_raw = magLsbToMicroTesla(telemetry.mz || 0);

    // ===== Magnetometer Axis Alignment =====
    // Puck.js has different axis orientation for magnetometer vs accel/gyro:
    //   Accel/Gyro: X→aerial, Y→IR LEDs, Z→into PCB
    //   Magnetometer: X→IR LEDs, Y→aerial, Z→into PCB
    // Swap X and Y to align magnetometer to accel/gyro frame.
    // Additionally, negate Y to match accelerometer sign convention.
    // (This matches telemetry-processor.ts lines 391-393)
    const mx_ut = my_ut_raw;    // Mag Y (aerial) → aligned X (aerial)
    const my_ut = -mx_ut_raw;   // Mag X (IR LEDs) → aligned Y (IR LEDs), NEGATED
    const mz_ut = mz_ut_raw;    // Z unchanged

    sensorData = {
        ax_g, ay_g, az_g,
        gx_dps, gy_dps, gz_dps,
        mx_ut, my_ut, mz_ut,
        l: telemetry.l || 0,
        t: telemetry.t || 0,
        c: telemetry.c || 0,
        b: telemetry.b || 0,
        s: telemetry.s || 0,
        n: telemetry.n || 0
    };

    updateSensorDisplay();
    updateBattery(sensorData.b);

    if (isPlaying) {
        updateSynthesis();
    }
}

// ===== Connection Handler =====

async function handleConnect(): Promise<void> {
    if (isConnected) {
        // Disconnect
        console.log('[GAMBIT] Disconnecting from device...');
        if (gambitClient) {
            gambitClient.disconnect();
            gambitClient = null;
        }
        isConnected = false;
        connectBtn.textContent = 'Connect Device';
        connectBtn.classList.remove('connected');
        statusDiv.textContent = 'Disconnected - Click "Connect Device" to begin';
        statusDiv.classList.remove('connected', 'playing');
        toggleSynthBtn.disabled = true;
        batteryDiv.style.display = 'none';

        if (isPlaying) {
            stopAudio();
        }
    } else {
        // Connect
        try {
            statusDiv.textContent = 'Connecting...';

            gambitClient = new GambitClient({
                debug: true,
                autoKeepalive: true,
                keepaliveInterval: 25000
            });

            gambitClient.on('data', onTelemetry as (data: unknown) => void);

            gambitClient.on('firmware', (info) => {
                console.log('[GAMBIT] Firmware info:', info);

                const compat = gambitClient!.checkCompatibility('0.1.0');
                if (!compat.compatible) {
                    statusDiv.textContent = `Incompatible firmware: ${compat.reason}`;
                    statusDiv.classList.add('error');
                    setTimeout(() => gambitClient!.disconnect(), 3000);
                    return;
                }

                statusDiv.textContent = `Connected - ${info.id} v${info.version}`;
            });

            gambitClient.on('disconnect', () => {
                console.log('[GAMBIT] Device disconnected');
                isConnected = false;
                connectBtn.textContent = 'Connect Device';
                connectBtn.classList.remove('connected');
                statusDiv.classList.remove('connected', 'playing');
                toggleSynthBtn.disabled = true;
                if (isPlaying) {
                    stopAudio();
                }
            });

            gambitClient.on('error', (err) => {
                console.error('[GAMBIT] Error:', err);
                statusDiv.textContent = 'Error: ' + err.message;
            });

            await gambitClient.connect();

            isConnected = true;
            connectBtn.textContent = 'Disconnect';
            connectBtn.classList.add('connected');
            statusDiv.classList.add('connected');
            toggleSynthBtn.disabled = false;

            await gambitClient.startStreaming();
            statusDiv.textContent = 'Connected - Receiving sensor data';

        } catch (e) {
            console.error('[GAMBIT] Connection error:', e);
            statusDiv.textContent = 'Connection failed: ' + (e as Error).message;
            if (gambitClient) {
                gambitClient.disconnect();
                gambitClient = null;
            }
        }
    }
}

// ===== Initialization =====

function init(): void {
    // Get UI elements
    connectBtn = document.getElementById('connectBtn') as HTMLButtonElement;
    toggleSynthBtn = document.getElementById('toggleSynthBtn') as HTMLButtonElement;
    statusDiv = document.getElementById('status')!;
    batteryDiv = document.getElementById('battery')!;
    batteryFill = document.getElementById('batteryFill')!;
    batteryText = document.getElementById('batteryText')!;
    oscTypeSelect = document.getElementById('oscType') as HTMLSelectElement;
    baseFreqSlider = document.getElementById('baseFreq') as HTMLInputElement;
    baseFreqValue = document.getElementById('baseFreqValue')!;
    filterQSlider = document.getElementById('filterQ') as HTMLInputElement;
    filterQValue = document.getElementById('filterQValue')!;
    reverbMixSlider = document.getElementById('reverbMix') as HTMLInputElement;
    reverbMixValue = document.getElementById('reverbMixValue')!;
    canvas = document.getElementById('waveformCanvas') as HTMLCanvasElement;
    canvasCtx = canvas.getContext('2d')!;

    // Initialize canvas
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Connect button
    connectBtn.addEventListener('click', handleConnect);

    // Toggle synthesizer
    toggleSynthBtn.addEventListener('click', () => {
        if (isPlaying) {
            stopAudio();
        } else {
            initAudio();
        }
    });

    // Configuration controls
    oscTypeSelect.addEventListener('change', () => {
        if (oscillator && isPlaying) {
            oscillator.type = oscTypeSelect.value as OscillatorType;
        }
    });

    baseFreqSlider.addEventListener('input', () => {
        baseFreqValue.textContent = baseFreqSlider.value;
    });

    filterQSlider.addEventListener('input', () => {
        filterQValue.textContent = filterQSlider.value;
        if (filterNode && isPlaying) {
            filterNode.Q.value = parseFloat(filterQSlider.value);
        }
    });

    reverbMixSlider.addEventListener('input', () => {
        reverbMixValue.textContent = reverbMixSlider.value;
        if (isPlaying) {
            updateReverbMix();
        }
    });

    console.log('[synth-app] GAMBIT Synth initialized');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
