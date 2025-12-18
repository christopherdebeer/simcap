/**
 * GAMBIT Synth Application
 * Real-time sensor-reactive audio synthesis
 */

import { GambitClient } from './gambit-client';

// ===== Type Definitions =====

// Extended telemetry including environmental sensors from firmware
interface SynthTelemetry {
    ax: number; ay: number; az: number;  // Accelerometer
    gx: number; gy: number; gz: number;  // Gyroscope
    mx: number; my: number; mz: number;  // Magnetometer
    t: number;   // Timestamp
    l: number;   // Light sensor
    c: number;   // Capacitive sensor
    b: number;   // Battery percentage
    s: number;   // Status
    n: number;   // Sample number
}

interface SensorRanges {
    [key: string]: { min: number; max: number };
}

interface SensorData {
    ax: number; ay: number; az: number;
    gx: number; gy: number; gz: number;
    mx: number; my: number; mz: number;
    l: number; t: number; c: number;
    b: number; s: number; n: number;
}

// ===== Audio State =====

let audioContext: AudioContext | null = null;
let oscillator: OscillatorNode | null = null;
let gainNode: GainNode | null = null;
let filterNode: BiquadFilterNode | null = null;
let lfoOscillator: OscillatorNode | null = null;
let lfoGain: GainNode | null = null;
let analyser: AnalyserNode | null = null;
let reverbNode: ConvolverNode | null = null;
let dryGain: GainNode | null = null;
let wetGain: GainNode | null = null;
let isPlaying = false;

// ===== GAMBIT Client State =====

let gambitClient: GambitClient | null = null;
let isConnected = false;

// ===== Sensor Configuration =====

const ranges: SensorRanges = {
    ax: { min: -8000, max: 8000 },
    ay: { min: -8000, max: 8000 },
    az: { min: -8000, max: 8000 },
    gx: { min: -2000, max: 2000 },
    gy: { min: -2000, max: 2000 },
    gz: { min: -2000, max: 2000 },
    mx: { min: -50, max: 50 },
    my: { min: -50, max: 50 },
    mz: { min: -50, max: 50 },
    l: { min: 0, max: 1 },
    t: { min: 15, max: 35 },
    c: { min: 0, max: 100 }
};

let sensorData: SensorData = {
    ax: 0, ay: 0, az: 0,
    gx: 0, gy: 0, gz: 0,
    mx: 0, my: 0, mz: 0,
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

function createReverbImpulse(): AudioBuffer {
    const sampleRate = audioContext!.sampleRate;
    const length = sampleRate * 2; // 2 second reverb
    const impulse = audioContext!.createBuffer(2, length, sampleRate);
    const leftChannel = impulse.getChannelData(0);
    const rightChannel = impulse.getChannelData(1);

    for (let i = 0; i < length; i++) {
        const decay = Math.exp(-i / (sampleRate * 0.5));
        leftChannel[i] = (Math.random() * 2 - 1) * decay;
        rightChannel[i] = (Math.random() * 2 - 1) * decay;
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

    // Analyser for visualization
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;

    // Simple reverb (convolver with generated impulse response)
    reverbNode = audioContext.createConvolver();
    reverbNode.buffer = createReverbImpulse();

    // Dry/Wet gain nodes for reverb mix
    dryGain = audioContext.createGain();
    wetGain = audioContext.createGain();

    // Connect LFO
    lfoOscillator.connect(lfoGain);
    lfoGain.connect(oscillator.frequency);

    // Connect audio graph: oscillator -> filter -> dry/wet split -> analyser -> output
    oscillator.connect(filterNode);
    filterNode.connect(dryGain);
    filterNode.connect(reverbNode);
    reverbNode.connect(wetGain);

    dryGain.connect(gainNode);
    wetGain.connect(gainNode);
    gainNode.connect(analyser);
    analyser.connect(audioContext.destination);

    // Start oscillators
    oscillator.start();
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
    if (lfoOscillator) {
        lfoOscillator.stop();
        lfoOscillator = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
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

    // Frequency control from accelerometer Y (-1 to 1 range, +/-2 octaves)
    const ayNorm = normalize(sensorData.ay, 'ay');
    const freqMultiplier = Math.pow(2, (ayNorm - 0.5) * 4);
    const targetFreq = baseFreq * freqMultiplier;
    oscillator.frequency.setTargetAtTime(targetFreq, audioContext.currentTime, 0.01);
    document.getElementById('freqParam')!.textContent = Math.round(targetFreq) + ' Hz';

    // Filter cutoff from gyroscope Z
    const gzNorm = normalize(sensorData.gz, 'gz');
    const filterFreq = 200 + gzNorm * 4800;
    filterNode!.frequency.setTargetAtTime(filterFreq, audioContext.currentTime, 0.01);
    document.getElementById('filterParam')!.textContent = Math.round(filterFreq) + ' Hz';

    // Volume from light sensor
    const lightNorm = sensorData.l;
    const volume = Math.max(0.1, lightNorm) * 0.5;
    gainNode!.gain.setTargetAtTime(volume, audioContext.currentTime, 0.01);
    document.getElementById('volumeParam')!.textContent = Math.round(volume * 100) + '%';

    // LFO rate from magnetometer X
    const mxNorm = normalize(sensorData.mx, 'mx');
    const lfoRate = 0.5 + mxNorm * 15;
    lfoOscillator!.frequency.setTargetAtTime(lfoRate, audioContext.currentTime, 0.01);
    document.getElementById('lfoParam')!.textContent = lfoRate.toFixed(2) + ' Hz';

    // Detune from gyroscope X
    const gxNorm = normalize(sensorData.gx, 'gx');
    const detune = (gxNorm - 0.5) * 100;
    oscillator.detune.setTargetAtTime(detune, audioContext.currentTime, 0.01);
    document.getElementById('detuneParam')!.textContent = Math.round(detune) + ' cents';

    // Reverb mix from accelerometer Z
    const azNorm = normalize(sensorData.az, 'az');
    const reverbAmount = azNorm;
    dryGain!.gain.setTargetAtTime(1 - reverbAmount, audioContext.currentTime, 0.01);
    wetGain!.gain.setTargetAtTime(reverbAmount, audioContext.currentTime, 0.01);
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
    document.getElementById('ax')!.textContent = sensorData.ax.toFixed(0);
    document.getElementById('ay')!.textContent = sensorData.ay.toFixed(0);
    document.getElementById('az')!.textContent = sensorData.az.toFixed(0);
    document.getElementById('gx')!.textContent = sensorData.gx.toFixed(0);
    document.getElementById('gy')!.textContent = sensorData.gy.toFixed(0);
    document.getElementById('gz')!.textContent = sensorData.gz.toFixed(0);
    document.getElementById('mx')!.textContent = sensorData.mx.toFixed(2);
    document.getElementById('my')!.textContent = sensorData.my.toFixed(2);
    document.getElementById('mz')!.textContent = sensorData.mz.toFixed(2);
    document.getElementById('light')!.textContent = sensorData.l.toFixed(3);
    document.getElementById('temp')!.textContent = sensorData.t.toFixed(1);
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

function onTelemetry(telemetry: SynthTelemetry): void {
    sensorData = {
        ax: telemetry.ax || 0,
        ay: telemetry.ay || 0,
        az: telemetry.az || 0,
        gx: telemetry.gx || 0,
        gy: telemetry.gy || 0,
        gz: telemetry.gz || 0,
        mx: telemetry.mx || 0,
        my: telemetry.my || 0,
        mz: telemetry.mz || 0,
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
