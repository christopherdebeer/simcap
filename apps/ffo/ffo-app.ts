/**
 * FFO$$ Application
 *
 * Web interface for template-based gesture recognition using the $Q-3D algorithm.
 * Allows recording, managing, and testing gesture templates from SIMCAP devices.
 *
 * @module ffo-app
 */

import { GambitClient } from '../gambit/gambit-client';
import {
  FFORecognizer,
  createRecognizer,
  type TelemetrySample3D,
  type RecognitionResult,
  type GestureTemplate,
  type RecognizerConfig,
} from '../../packages/ffo/src/index';

// Declare THREE.js from CDN (loaded via script tag)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const THREE: any;

// ============================================================================
// TYPES
// ============================================================================

interface AppState {
  connected: boolean;
  recording: boolean;
  recognizing: boolean;
  client: GambitClient | null;
  recognizer: FFORecognizer;
  recordingBuffer: TelemetrySample3D[];
  recordingStartTime: number;
  lastRecognitionResult: RecognitionResult | null;
  trajectoryPoints: Array<{ x: number; y: number; z: number }>;
  sampleTimes: number[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface ThreeContext {
  scene: any;
  camera: any;
  renderer: any;
  controls: any;
  trajectoryLine: any;
  templateLines: any[];
}

// ============================================================================
// STATE
// ============================================================================

const state: AppState = {
  connected: false,
  recording: false,
  recognizing: false,
  client: null,
  recognizer: createRecognizer({ numPoints: 32 }),
  recordingBuffer: [],
  recordingStartTime: 0,
  lastRecognitionResult: null,
  trajectoryPoints: [],
  sampleTimes: [],
};

let threeContext: ThreeContext | null = null;
let recognitionInterval: number | null = null;
let recordingTimer: number | null = null;

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const elements = {
  // Connection
  connectBtn: document.getElementById('connectBtn') as HTMLButtonElement,
  disconnectBtn: document.getElementById('disconnectBtn') as HTMLButtonElement,
  connectionStatus: document.getElementById('connectionStatus') as HTMLElement,
  deviceInfo: document.getElementById('deviceInfo') as HTMLElement,

  // Recording
  recordBtn: document.getElementById('recordBtn') as HTMLButtonElement,
  stopRecordBtn: document.getElementById('stopRecordBtn') as HTMLButtonElement,
  saveTemplateBtn: document.getElementById('saveTemplateBtn') as HTMLButtonElement,
  templateName: document.getElementById('templateName') as HTMLInputElement,
  recordingStatus: document.getElementById('recordingStatus') as HTMLElement,
  recordingIndicator: document.getElementById('recordingIndicator') as HTMLElement,
  recordingTimer: document.getElementById('recordingTimer') as HTMLElement,
  sampleCount: document.getElementById('sampleCount') as HTMLElement,
  recordingFeedback: document.getElementById('recordingFeedback') as HTMLElement,

  // Trajectory
  trajectoryContainer: document.getElementById('trajectoryContainer') as HTMLElement,
  clearTrajectoryBtn: document.getElementById('clearTrajectoryBtn') as HTMLButtonElement,
  resetViewBtn: document.getElementById('resetViewBtn') as HTMLButtonElement,

  // Recognition
  startRecognitionBtn: document.getElementById('startRecognitionBtn') as HTMLButtonElement,
  stopRecognitionBtn: document.getElementById('stopRecognitionBtn') as HTMLButtonElement,
  recognitionStatus: document.getElementById('recognitionStatus') as HTMLElement,
  recognitionResult: document.getElementById('recognitionResult') as HTMLElement,
  recognizedGesture: document.getElementById('recognizedGesture') as HTMLElement,
  recognitionScore: document.getElementById('recognitionScore') as HTMLElement,
  recognitionDistance: document.getElementById('recognitionDistance') as HTMLElement,
  candidatesList: document.getElementById('candidatesList') as HTMLElement,

  // Templates
  templateCount: document.getElementById('templateCount') as HTMLElement,
  templateList: document.getElementById('templateList') as HTMLElement,
  vocabInfo: document.getElementById('vocabInfo') as HTMLElement,
  exportVocabBtn: document.getElementById('exportVocabBtn') as HTMLButtonElement,
  importVocabBtn: document.getElementById('importVocabBtn') as HTMLButtonElement,
  clearVocabBtn: document.getElementById('clearVocabBtn') as HTMLButtonElement,
  importFileInput: document.getElementById('importFileInput') as HTMLInputElement,

  // Config
  configNumPoints: document.getElementById('configNumPoints') as HTMLInputElement,
  configRejectThreshold: document.getElementById('configRejectThreshold') as HTMLInputElement,
  configMinSamples: document.getElementById('configMinSamples') as HTMLInputElement,
  configRemoveGravity: document.getElementById('configRemoveGravity') as HTMLInputElement,
  configUseLookupTable: document.getElementById('configUseLookupTable') as HTMLInputElement,
  applyConfigBtn: document.getElementById('applyConfigBtn') as HTMLButtonElement,

  // Sensor Data
  accelData: document.getElementById('accelData') as HTMLElement,
  gyroData: document.getElementById('gyroData') as HTMLElement,
  sampleRate: document.getElementById('sampleRate') as HTMLElement,

  // Log
  log: document.getElementById('log') as HTMLElement,
};

// ============================================================================
// LOGGING
// ============================================================================

function log(message: string): void {
  const timestamp = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.textContent = `[${timestamp}] ${message}`;
  elements.log.appendChild(line);
  elements.log.scrollTop = elements.log.scrollHeight;
  console.log(`[FFO$$] ${message}`);
}

// ============================================================================
// THREE.JS VISUALIZATION
// ============================================================================

function initThreeJS(): void {
  const container = elements.trajectoryContainer;
  const width = container.clientWidth;
  const height = container.clientHeight || 300;

  // Scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  // Camera
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
  camera.position.set(2, 2, 2);
  camera.lookAt(0, 0, 0);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // Controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Grid
  const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
  scene.add(gridHelper);

  // Axes
  const axesHelper = new THREE.AxesHelper(1);
  scene.add(axesHelper);

  // Store context
  threeContext = {
    scene,
    camera,
    renderer,
    controls,
    trajectoryLine: null,
    templateLines: [],
  };

  // Animation loop
  function animate(): void {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Handle resize
  window.addEventListener('resize', () => {
    const newWidth = container.clientWidth;
    const newHeight = container.clientHeight || 300;
    camera.aspect = newWidth / newHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, newHeight);
  });
}

function updateTrajectory(points: Array<{ x: number; y: number; z: number }>): void {
  if (!threeContext) return;

  // Remove old line
  if (threeContext.trajectoryLine) {
    threeContext.scene.remove(threeContext.trajectoryLine);
    threeContext.trajectoryLine.geometry.dispose();
    (threeContext.trajectoryLine.material as THREE.Material).dispose();
  }

  if (points.length < 2) return;

  // Create new line
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);

  for (let i = 0; i < points.length; i++) {
    positions[i * 3] = points[i].x;
    positions[i * 3 + 1] = points[i].y;
    positions[i * 3 + 2] = points[i].z;

    // Color gradient: green -> yellow -> red
    const t = i / (points.length - 1);
    colors[i * 3] = t; // R
    colors[i * 3 + 1] = 1 - t * 0.5; // G
    colors[i * 3 + 2] = 0; // B
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    linewidth: 2,
  });

  threeContext.trajectoryLine = new THREE.Line(geometry, material);
  threeContext.scene.add(threeContext.trajectoryLine);
}

function clearTrajectory(): void {
  state.trajectoryPoints = [];
  updateTrajectory([]);
  log('Trajectory cleared');
}

function resetView(): void {
  if (!threeContext) return;
  threeContext.camera.position.set(2, 2, 2);
  threeContext.camera.lookAt(0, 0, 0);
  threeContext.controls.reset();
}

// ============================================================================
// CONNECTION
// ============================================================================

async function connect(): Promise<void> {
  if (state.connected) return;

  log('Connecting to device...');
  elements.connectBtn.disabled = true;

  try {
    state.client = new GambitClient({ debug: false, autoKeepalive: true });

    state.client.on('data', onTelemetry);
    state.client.on('disconnect', onDisconnect);
    state.client.on('firmware', (info: { name: string; version: string }) => {
      log(`Firmware: ${info.name} v${info.version}`);
      elements.deviceInfo.textContent = `${info.name} v${info.version}`;
    });

    await state.client.connect();

    state.connected = true;
    updateConnectionUI();
    log('Connected!');
  } catch (e) {
    const error = e as Error;
    log(`Connection failed: ${error.message}`);
    elements.connectBtn.disabled = false;
    state.client = null;
  }
}

function disconnect(): void {
  if (!state.connected || !state.client) return;

  state.client.disconnect();
  state.client = null;
  state.connected = false;

  if (state.recording) stopRecording();
  if (state.recognizing) stopRecognition();

  updateConnectionUI();
  log('Disconnected');
}

function onDisconnect(): void {
  state.connected = false;
  state.client = null;

  if (state.recording) stopRecording();
  if (state.recognizing) stopRecognition();

  updateConnectionUI();
  log('Device disconnected');
}

function updateConnectionUI(): void {
  elements.connectBtn.disabled = state.connected;
  elements.disconnectBtn.disabled = !state.connected;
  elements.recordBtn.disabled = !state.connected;
  elements.startRecognitionBtn.disabled = !state.connected || state.recognizer.templateCount === 0;

  if (state.connected) {
    elements.connectionStatus.textContent = 'Connected';
    elements.connectionStatus.classList.add('connected');
  } else {
    elements.connectionStatus.textContent = 'Disconnected';
    elements.connectionStatus.classList.remove('connected');
    elements.deviceInfo.textContent = '';
  }
}

// ============================================================================
// TELEMETRY
// ============================================================================

function onTelemetry(data: {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  t: number;
}): void {
  // Convert raw LSB to physical units
  const ax_g = data.ax / 16384;
  const ay_g = data.ay / 16384;
  const az_g = data.az / 16384;
  const gx_dps = data.gx / 131;
  const gy_dps = data.gy / 131;
  const gz_dps = data.gz / 131;

  // Update sensor display
  elements.accelData.textContent = `${ax_g.toFixed(2)} / ${ay_g.toFixed(2)} / ${az_g.toFixed(2)}`;
  elements.gyroData.textContent = `${gx_dps.toFixed(1)} / ${gy_dps.toFixed(1)} / ${gz_dps.toFixed(1)}`;

  // Calculate sample rate
  state.sampleTimes.push(Date.now());
  if (state.sampleTimes.length > 50) state.sampleTimes.shift();
  if (state.sampleTimes.length >= 2) {
    const elapsed = state.sampleTimes[state.sampleTimes.length - 1] - state.sampleTimes[0];
    const rate = ((state.sampleTimes.length - 1) / elapsed) * 1000;
    elements.sampleRate.textContent = `${rate.toFixed(1)} Hz`;
  }

  // Create sample for FFO$$
  const sample: TelemetrySample3D = {
    ax_g,
    ay_g,
    az_g,
    t: data.t,
  };

  // Add to trajectory visualization
  state.trajectoryPoints.push({ x: ax_g, y: ay_g, z: az_g });
  if (state.trajectoryPoints.length > 200) {
    state.trajectoryPoints.shift();
  }
  updateTrajectory(state.trajectoryPoints);

  // Recording
  if (state.recording) {
    state.recordingBuffer.push(sample);
    elements.sampleCount.textContent = `${state.recordingBuffer.length} samples`;
  }

  // Recognition (sliding window)
  if (state.recognizing && state.recordingBuffer.length >= state.recognizer.getConfig().minSamples) {
    // Keep a sliding window of samples
    if (state.recordingBuffer.length > 100) {
      state.recordingBuffer.shift();
    }
  }
}

// ============================================================================
// RECORDING
// ============================================================================

function startRecording(): void {
  if (state.recording || !state.connected) return;

  state.recording = true;
  state.recordingBuffer = [];
  state.recordingStartTime = Date.now();

  elements.recordBtn.disabled = true;
  elements.stopRecordBtn.disabled = false;
  elements.saveTemplateBtn.disabled = true;
  elements.recordingStatus.textContent = 'Recording';
  elements.recordingStatus.classList.add('recording');
  elements.recordingIndicator.classList.add('active');

  // Update timer
  recordingTimer = window.setInterval(() => {
    const elapsed = (Date.now() - state.recordingStartTime) / 1000;
    elements.recordingTimer.textContent = `${elapsed.toFixed(1)}s`;
  }, 100);

  log('Recording started - perform your gesture');
}

function stopRecording(): void {
  if (!state.recording) return;

  state.recording = false;

  if (recordingTimer) {
    clearInterval(recordingTimer);
    recordingTimer = null;
  }

  elements.recordBtn.disabled = !state.connected;
  elements.stopRecordBtn.disabled = true;
  elements.saveTemplateBtn.disabled = state.recordingBuffer.length < state.recognizer.getConfig().minSamples;
  elements.recordingStatus.textContent = 'Idle';
  elements.recordingStatus.classList.remove('recording');
  elements.recordingIndicator.classList.remove('active');

  const duration = (Date.now() - state.recordingStartTime) / 1000;
  elements.recordingFeedback.textContent = `Captured ${state.recordingBuffer.length} samples in ${duration.toFixed(1)}s`;
  log(`Recording stopped: ${state.recordingBuffer.length} samples`);
}

function saveTemplate(): void {
  const name = elements.templateName.value.trim();
  if (!name) {
    elements.recordingFeedback.textContent = 'Please enter a gesture name';
    return;
  }

  if (state.recordingBuffer.length < state.recognizer.getConfig().minSamples) {
    elements.recordingFeedback.textContent = `Need at least ${state.recognizer.getConfig().minSamples} samples`;
    return;
  }

  try {
    state.recognizer.addTemplateFromSamples(name, state.recordingBuffer, 'recorded');
    elements.recordingFeedback.textContent = `Template "${name}" saved!`;
    elements.templateName.value = '';
    state.recordingBuffer = [];
    elements.saveTemplateBtn.disabled = true;
    updateTemplateUI();
    log(`Template "${name}" saved`);
  } catch (e) {
    const error = e as Error;
    elements.recordingFeedback.textContent = `Error: ${error.message}`;
    log(`Error saving template: ${error.message}`);
  }
}

// ============================================================================
// RECOGNITION
// ============================================================================

function startRecognition(): void {
  if (state.recognizing || !state.connected) return;
  if (state.recognizer.templateCount === 0) {
    log('No templates - add templates first');
    return;
  }

  state.recognizing = true;
  state.recordingBuffer = [];

  elements.startRecognitionBtn.disabled = true;
  elements.stopRecognitionBtn.disabled = false;
  elements.recognitionStatus.textContent = 'Active';
  elements.recognitionStatus.classList.add('connected');

  // Recognize every 500ms
  recognitionInterval = window.setInterval(() => {
    if (state.recordingBuffer.length >= state.recognizer.getConfig().minSamples) {
      const result = state.recognizer.recognize(state.recordingBuffer);
      state.lastRecognitionResult = result;
      updateRecognitionUI(result);
    }
  }, 500);

  log('Recognition started');
}

function stopRecognition(): void {
  if (!state.recognizing) return;

  state.recognizing = false;
  state.recordingBuffer = [];

  if (recognitionInterval) {
    clearInterval(recognitionInterval);
    recognitionInterval = null;
  }

  elements.startRecognitionBtn.disabled = !state.connected || state.recognizer.templateCount === 0;
  elements.stopRecognitionBtn.disabled = true;
  elements.recognitionStatus.textContent = 'Idle';
  elements.recognitionStatus.classList.remove('connected');

  log('Recognition stopped');
}

function updateRecognitionUI(result: RecognitionResult): void {
  if (result.rejected || !result.template) {
    elements.recognizedGesture.textContent = '?';
    elements.recognitionScore.textContent = '--';
    elements.recognitionDistance.textContent = 'No match';
    elements.recognitionResult.classList.remove('matched');
    elements.recognitionResult.classList.add('rejected');
  } else {
    elements.recognizedGesture.textContent = result.template.name;
    elements.recognitionScore.textContent = `${(result.score * 100).toFixed(0)}%`;
    elements.recognitionDistance.textContent = `Distance: ${result.distance.toFixed(4)}`;
    elements.recognitionResult.classList.add('matched');
    elements.recognitionResult.classList.remove('rejected');
  }

  // Update candidates
  elements.candidatesList.innerHTML = '';
  if (result.candidates) {
    for (const candidate of result.candidates.slice(0, 5)) {
      const div = document.createElement('div');
      div.className = 'candidate-item';
      div.innerHTML = `
        <div class="name">${candidate.template.name}</div>
        <div class="score-bar">
          <div class="score-fill" style="width: ${candidate.score * 100}%"></div>
        </div>
      `;
      elements.candidatesList.appendChild(div);
    }
  }
}

// ============================================================================
// TEMPLATE MANAGEMENT
// ============================================================================

function updateTemplateUI(): void {
  const count = state.recognizer.templateCount;
  elements.templateCount.textContent = count.toString();
  elements.exportVocabBtn.disabled = count === 0;
  elements.clearVocabBtn.disabled = count === 0;
  elements.startRecognitionBtn.disabled = !state.connected || count === 0;

  // Update vocab info
  const vocab = state.recognizer.export();
  elements.vocabInfo.innerHTML = `
    <span class="name">${vocab.meta?.name || 'Untitled Vocabulary'}</span>
    <span class="stats">${count} template${count !== 1 ? 's' : ''}</span>
  `;

  // Update template list
  if (count === 0) {
    elements.templateList.innerHTML = `
      <div style="text-align: center; color: var(--fg-muted); padding: var(--space-md);">
        No templates yet. Record a gesture to create one.
      </div>
    `;
  } else {
    elements.templateList.innerHTML = '';
    for (const name of state.recognizer.templateNames) {
      const template = state.recognizer.getTemplate(name);
      if (!template) continue;

      const div = document.createElement('div');
      div.className = 'template-item';
      div.innerHTML = `
        <span class="name">${template.name}</span>
        <span class="meta">${template.meta.n} pts</span>
        <div class="actions">
          <button class="btn" data-action="delete" data-name="${template.name}">Delete</button>
        </div>
      `;
      elements.templateList.appendChild(div);
    }
  }
}

function deleteTemplate(name: string): void {
  if (confirm(`Delete template "${name}"?`)) {
    state.recognizer.removeTemplate(name);
    updateTemplateUI();
    log(`Template "${name}" deleted`);
  }
}

function exportVocabulary(): void {
  const json = state.recognizer.toJSON('FFO$$ Vocabulary');
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ffo-vocabulary-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  log('Vocabulary exported');
}

function importVocabulary(): void {
  elements.importFileInput.click();
}

function handleImportFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const json = e.target?.result as string;
      state.recognizer.fromJSON(json, true);
      updateTemplateUI();
      log(`Imported vocabulary: ${state.recognizer.templateCount} templates`);
    } catch (err) {
      const error = err as Error;
      log(`Import failed: ${error.message}`);
    }
  };
  reader.readAsText(file);
  input.value = '';
}

function clearVocabulary(): void {
  if (confirm('Clear all templates?')) {
    state.recognizer.clearTemplates();
    updateTemplateUI();
    log('All templates cleared');
  }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

function applyConfiguration(): void {
  const config: Partial<RecognizerConfig> = {
    numPoints: parseInt(elements.configNumPoints.value) || 32,
    rejectThreshold: parseFloat(elements.configRejectThreshold.value) || null,
    minSamples: parseInt(elements.configMinSamples.value) || 15,
    removeGravity: elements.configRemoveGravity.checked,
    useLookupTable: elements.configUseLookupTable.checked,
  };

  // Warn if changing numPoints with existing templates
  if (state.recognizer.templateCount > 0 && config.numPoints !== state.recognizer.getConfig().numPoints) {
    if (!confirm('Changing numPoints will make existing templates incompatible. Clear templates and continue?')) {
      return;
    }
    state.recognizer.clearTemplates();
    updateTemplateUI();
  }

  state.recognizer.setConfig(config);
  log(`Configuration applied: N=${config.numPoints}, threshold=${config.rejectThreshold}`);
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

function initEventHandlers(): void {
  // Connection
  elements.connectBtn.addEventListener('click', connect);
  elements.disconnectBtn.addEventListener('click', disconnect);

  // Recording
  elements.recordBtn.addEventListener('click', startRecording);
  elements.stopRecordBtn.addEventListener('click', stopRecording);
  elements.saveTemplateBtn.addEventListener('click', saveTemplate);

  // Trajectory
  elements.clearTrajectoryBtn.addEventListener('click', clearTrajectory);
  elements.resetViewBtn.addEventListener('click', resetView);

  // Recognition
  elements.startRecognitionBtn.addEventListener('click', startRecognition);
  elements.stopRecognitionBtn.addEventListener('click', stopRecognition);

  // Templates
  elements.exportVocabBtn.addEventListener('click', exportVocabulary);
  elements.importVocabBtn.addEventListener('click', importVocabulary);
  elements.clearVocabBtn.addEventListener('click', clearVocabulary);
  elements.importFileInput.addEventListener('change', handleImportFile);
  elements.templateList.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    if (target.dataset.action === 'delete' && target.dataset.name) {
      deleteTemplate(target.dataset.name);
    }
  });

  // Config
  elements.applyConfigBtn.addEventListener('click', applyConfiguration);

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;

    switch (e.key) {
      case 'r':
        if (state.connected && !state.recording) startRecording();
        break;
      case 's':
        if (state.recording) stopRecording();
        break;
      case 'c':
        clearTrajectory();
        break;
    }
  });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

function init(): void {
  log('FFO$$ initialized');
  log('Keyboard shortcuts: R=record, S=stop, C=clear');

  initThreeJS();
  initEventHandlers();
  updateConnectionUI();
  updateTemplateUI();
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
