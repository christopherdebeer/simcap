/**
 * Hand Model Visualization for Magnetic Finger Tracking
 *
 * Provides 3D visualization of hand pose estimates
 * for real-time feedback during data collection.
 */

// ===== Type Definitions =====

export interface HandVisualizerOptions {
  autoRotate?: boolean;
}

export interface FingerStates {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
}

interface FingerData {
  name: string;
  x: number;
  z: number;
  angle: number;
  length: number;
  color: string;
}

interface FingerElement {
  element: HTMLDivElement;
  baseData: FingerData;
}

// ===== Hand Visualizer 3D =====

/**
 * 3D Hand Visualizer using CSS 3D transforms
 *
 * Simpler than WebGL, works without additional libraries.
 */
export class HandVisualizer3D {
  private container: HTMLElement;
  private options: HandVisualizerOptions;
  private scene: HTMLDivElement;
  private hand: HTMLDivElement;
  private fingerElements: Record<string, FingerElement> = {};
  private rotationX: number;
  private rotationY: number;
  private autoRotate: boolean;
  private fingerStates: FingerStates;

  constructor(container: HTMLElement, options: HandVisualizerOptions = {}) {
    this.container = container;
    this.options = options;

    // Create 3D scene container
    this.scene = document.createElement('div');
    this.scene.style.cssText = `
      width: 100%;
      height: 100%;
      perspective: 800px;
      perspective-origin: 50% 50%;
    `;
    container.appendChild(this.scene);

    // Create hand container
    this.hand = document.createElement('div');
    this.hand.style.cssText = `
      width: 100%;
      height: 100%;
      position: relative;
      transform-style: preserve-3d;
      transform: rotateX(-20deg) rotateY(0deg);
    `;
    this.scene.appendChild(this.hand);

    // Rotation state
    this.rotationX = -20;
    this.rotationY = 0;
    this.autoRotate = options.autoRotate || false;

    // Finger states
    this.fingerStates = {
      thumb: 0,
      index: 0,
      middle: 0,
      ring: 0,
      pinky: 0
    };

    // Create finger elements
    this._createHand();
  }

  /**
   * Create hand elements
   */
  private _createHand(): void {
    const fingerData: FingerData[] = [
      { name: 'thumb', x: -40, z: 20, angle: -30, length: 60, color: '#e74c3c' },
      { name: 'index', x: -20, z: -30, angle: -10, length: 80, color: '#e67e22' },
      { name: 'middle', x: 0, z: -35, angle: 0, length: 90, color: '#f1c40f' },
      { name: 'ring', x: 20, z: -30, angle: 10, length: 80, color: '#2ecc71' },
      { name: 'pinky', x: 35, z: -20, angle: 20, length: 65, color: '#3498db' }
    ];

    // Create palm
    const palm = document.createElement('div');
    palm.style.cssText = `
      position: absolute;
      left: 50%;
      top: 50%;
      width: 80px;
      height: 100px;
      background: #2c3e50;
      border-radius: 40px 40px 20px 20px;
      transform: translate(-50%, -50%);
      transform-style: preserve-3d;
    `;
    this.hand.appendChild(palm);

    // Create fingers
    for (const finger of fingerData) {
      const el = document.createElement('div');
      el.className = `finger finger-${finger.name}`;
      el.style.cssText = `
        position: absolute;
        left: 50%;
        top: 50%;
        width: 18px;
        height: ${finger.length}px;
        background: ${finger.color};
        border-radius: 9px;
        transform-origin: center bottom;
        transform: translate(-50%, -100%)
                   translateX(${finger.x}px)
                   translateZ(${finger.z}px)
                   rotateY(${finger.angle}deg);
        transform-style: preserve-3d;
      `;

      // Add segments
      for (let i = 0; i < 3; i++) {
        const segment = document.createElement('div');
        segment.className = `segment segment-${i}`;
        segment.style.cssText = `
          position: absolute;
          width: 100%;
          height: ${finger.length / 3}px;
          background: ${finger.color};
          border-radius: 9px;
          top: ${i * (finger.length / 3)}px;
          transform-origin: center top;
          transform-style: preserve-3d;
        `;
        el.appendChild(segment);
      }

      this.hand.appendChild(el);
      this.fingerElements[finger.name] = {
        element: el,
        baseData: finger
      };
    }
  }

  /**
   * Set finger states
   */
  setFingerStates(states: Partial<FingerStates>): void {
    for (const [finger, state] of Object.entries(states)) {
      if (finger in this.fingerStates) {
        this.fingerStates[finger as keyof FingerStates] = state;
        this._updateFingerPose(finger);
      }
    }
  }

  /**
   * Update individual finger pose
   */
  private _updateFingerPose(finger: string): void {
    const fingerEl = this.fingerElements[finger];
    if (!fingerEl) return;

    const { element } = fingerEl;
    const state = this.fingerStates[finger as keyof FingerStates];
    const flexAngle = state * 30; // 0-60 degrees per segment

    const segments = element.querySelectorAll('.segment');
    segments.forEach((seg) => {
      (seg as HTMLElement).style.transform = `rotateX(${flexAngle}deg)`;
    });
  }

  /**
   * Set hand rotation
   */
  setRotation(x: number, y: number): void {
    this.rotationX = x;
    this.rotationY = y;
    this.hand.style.transform = `rotateX(${x}deg) rotateY(${y}deg)`;
  }

  /**
   * Start auto-rotation
   */
  startAutoRotate(): void {
    this.autoRotate = true;
    const animate = () => {
      if (!this.autoRotate) return;
      this.rotationY += 0.5;
      this.setRotation(this.rotationX, this.rotationY);
      requestAnimationFrame(animate);
    };
    animate();
  }

  /**
   * Stop auto-rotation
   */
  stopAutoRotate(): void {
    this.autoRotate = false;
  }
}

// ===== Finger State Display =====

/**
 * Finger State Display
 *
 * Simple text-based display of finger states,
 * useful for debugging and compact display.
 */
export class FingerStateDisplay {
  private container: HTMLElement;
  private rows: Record<string, HTMLElement> = {};

  constructor(container: HTMLElement) {
    this.container = container;
    this.container.innerHTML = `
      <div style="font-family: monospace; font-size: 14px; color: #fff;">
        <div class="finger-row" data-finger="thumb">
          <span style="color: #e74c3c;">T:</span>
          <span class="state">---</span>
          <span class="bar"></span>
        </div>
        <div class="finger-row" data-finger="index">
          <span style="color: #e67e22;">I:</span>
          <span class="state">---</span>
          <span class="bar"></span>
        </div>
        <div class="finger-row" data-finger="middle">
          <span style="color: #f1c40f;">M:</span>
          <span class="state">---</span>
          <span class="bar"></span>
        </div>
        <div class="finger-row" data-finger="ring">
          <span style="color: #2ecc71;">R:</span>
          <span class="state">---</span>
          <span class="bar"></span>
        </div>
        <div class="finger-row" data-finger="pinky">
          <span style="color: #3498db;">P:</span>
          <span class="state">---</span>
          <span class="bar"></span>
        </div>
      </div>
    `;

    this.container.querySelectorAll('.finger-row').forEach(row => {
      const el = row as HTMLElement;
      const finger = el.dataset.finger;
      if (finger) {
        this.rows[finger] = el;
      }
    });
  }

  /**
   * Update display
   */
  setFingerStates(states: Partial<FingerStates>): void {
    const stateLabels = ['EXT', 'PRT', 'FLX'];

    for (const [finger, state] of Object.entries(states)) {
      const row = this.rows[finger];
      if (!row) continue;

      const stateEl = row.querySelector('.state');
      const barEl = row.querySelector('.bar');

      if (stateEl) {
        const stateIdx = Math.round(Math.max(0, Math.min(2, state)));
        stateEl.textContent = stateLabels[stateIdx];
      }

      if (barEl) {
        // Progress bar
        const pct = (state / 2) * 100;
        const color = state < 1 ? '#2ecc71' : state < 1.5 ? '#f39c12' : '#e74c3c';
        barEl.innerHTML = `
          <span style="display: inline-block; width: 50px; height: 8px; background: #333; border-radius: 4px; overflow: hidden;">
            <span style="display: block; width: ${pct}%; height: 100%; background: ${color};"></span>
          </span>
        `;
      }
    }
  }
}

// Export as globals for backward compatibility
declare global {
  interface Window {
    HandVisualizer3D: typeof HandVisualizer3D;
    FingerStateDisplay: typeof FingerStateDisplay;
  }
}

if (typeof window !== 'undefined') {
  window.HandVisualizer3D = HandVisualizer3D;
  window.FingerStateDisplay = FingerStateDisplay;
}
