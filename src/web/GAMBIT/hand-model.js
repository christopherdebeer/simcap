/**
 * Hand Model Visualization for Magnetic Finger Tracking
 *
 * Provides 2D and 3D visualization of hand pose estimates
 * for real-time feedback during data collection.
 */

/**
 * 2D Hand Visualizer
 *
 * Renders a palm-down view of the hand with finger positions.
 * Finger flexion is shown as arc length along each finger.
 */
class HandVisualizer2D {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Options
        this.palmColor = options.palmColor || '#2c3e50';
        this.fingerColors = options.fingerColors || {
            thumb: '#e74c3c',
            index: '#e67e22',
            middle: '#f1c40f',
            ring: '#2ecc71',
            pinky: '#3498db'
        };
        this.backgroundColor = options.backgroundColor || '#1a1a2e';
        this.showLabels = options.showLabels !== false;

        // Hand geometry (normalized, palm-centered)
        this.fingerLengths = {
            thumb: 0.6,
            index: 0.8,
            middle: 0.85,
            ring: 0.8,
            pinky: 0.65
        };

        this.fingerWidths = {
            thumb: 0.15,
            index: 0.12,
            middle: 0.12,
            ring: 0.11,
            pinky: 0.10
        };

        // Finger base positions (angles from palm center)
        this.fingerAngles = {
            thumb: -70 * Math.PI / 180,
            index: -30 * Math.PI / 180,
            middle: 0,
            ring: 20 * Math.PI / 180,
            pinky: 45 * Math.PI / 180
        };

        this.fingerBaseDistances = {
            thumb: 0.45,
            index: 0.5,
            middle: 0.52,
            ring: 0.48,
            pinky: 0.42
        };

        // Current finger states (0 = extended, 1 = partial, 2 = flexed)
        this.fingerStates = {
            thumb: 0,
            index: 0,
            middle: 0,
            ring: 0,
            pinky: 0
        };

        // Animation
        this.targetStates = { ...this.fingerStates };
        this.animationSpeed = 0.15;

        this._resize();
    }

    /**
     * Resize handler
     */
    _resize() {
        // Get CSS size
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;

        // Scale context
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        // Calculate drawing scale
        this.width = rect.width;
        this.height = rect.height;
        this.scale = Math.min(this.width, this.height) * 0.4;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2 + this.scale * 0.1;
    }

    /**
     * Set finger states
     * @param {Object} states - {thumb: 0-2, index: 0-2, ...}
     */
    setFingerStates(states) {
        for (const [finger, state] of Object.entries(states)) {
            if (this.targetStates[finger] !== undefined) {
                this.targetStates[finger] = Math.max(0, Math.min(2, state));
            }
        }
    }

    /**
     * Set from binary string (e.g., "00000" for all extended)
     */
    setFromBinaryString(str) {
        const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
        for (let i = 0; i < 5 && i < str.length; i++) {
            const char = str[i];
            if (char === '0') this.targetStates[fingers[i]] = 0;
            else if (char === '1') this.targetStates[fingers[i]] = 1;
            else if (char === '2') this.targetStates[fingers[i]] = 2;
        }
    }

    /**
     * Update animation
     */
    _updateAnimation() {
        for (const finger of Object.keys(this.fingerStates)) {
            const diff = this.targetStates[finger] - this.fingerStates[finger];
            this.fingerStates[finger] += diff * this.animationSpeed;
        }
    }

    /**
     * Draw the hand
     */
    render() {
        this._updateAnimation();

        const ctx = this.ctx;

        // Clear
        ctx.fillStyle = this.backgroundColor;
        ctx.fillRect(0, 0, this.width, this.height);

        // Draw palm
        this._drawPalm();

        // Draw fingers
        for (const finger of ['pinky', 'ring', 'middle', 'index', 'thumb']) {
            this._drawFinger(finger);
        }

        // Draw labels if enabled
        if (this.showLabels) {
            this._drawLabels();
        }
    }

    /**
     * Draw palm
     */
    _drawPalm() {
        const ctx = this.ctx;
        const s = this.scale;

        ctx.fillStyle = this.palmColor;
        ctx.beginPath();

        // Draw palm as rounded rectangle
        const palmWidth = s * 0.7;
        const palmHeight = s * 0.6;

        ctx.ellipse(
            this.centerX,
            this.centerY,
            palmWidth / 2,
            palmHeight / 2,
            0, 0, Math.PI * 2
        );
        ctx.fill();

        // Wrist
        ctx.fillRect(
            this.centerX - palmWidth * 0.35,
            this.centerY + palmHeight * 0.3,
            palmWidth * 0.7,
            s * 0.3
        );
    }

    /**
     * Draw a finger
     */
    _drawFinger(finger) {
        const ctx = this.ctx;
        const s = this.scale;

        const angle = this.fingerAngles[finger];
        const baseDist = this.fingerBaseDistances[finger] * s;
        const length = this.fingerLengths[finger] * s;
        const width = this.fingerWidths[finger] * s;

        // Base position
        const baseX = this.centerX + Math.sin(angle) * baseDist;
        const baseY = this.centerY - Math.cos(angle) * baseDist * 0.8;

        // Flexion affects how much of the finger is visible
        const state = this.fingerStates[finger];
        const visibleLength = length * (1 - state * 0.35);

        // Tip position
        const tipX = baseX + Math.sin(angle) * visibleLength;
        const tipY = baseY - Math.cos(angle) * visibleLength * (1 - state * 0.2);

        // Draw finger
        ctx.strokeStyle = this.fingerColors[finger];
        ctx.lineWidth = width;
        ctx.lineCap = 'round';

        ctx.beginPath();
        ctx.moveTo(baseX, baseY);

        // Curved path for flexed fingers
        if (state > 0.5) {
            const midX = baseX + Math.sin(angle) * visibleLength * 0.5;
            const midY = baseY - Math.cos(angle) * visibleLength * 0.3;
            ctx.quadraticCurveTo(midX, midY, tipX, tipY);
        } else {
            ctx.lineTo(tipX, tipY);
        }

        ctx.stroke();

        // Draw fingertip
        ctx.fillStyle = this.fingerColors[finger];
        ctx.beginPath();
        ctx.arc(tipX, tipY, width * 0.6, 0, Math.PI * 2);
        ctx.fill();
    }

    /**
     * Draw state labels
     */
    _drawLabels() {
        const ctx = this.ctx;

        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#888';

        // Draw state indicators
        const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
        const labels = ['T', 'I', 'M', 'R', 'P'];

        for (let i = 0; i < 5; i++) {
            const x = this.width * 0.2 + i * (this.width * 0.6 / 4);
            const y = this.height - 20;

            const state = Math.round(this.fingerStates[fingers[i]]);
            const stateText = state === 0 ? 'E' : state === 1 ? 'P' : 'F';

            ctx.fillStyle = this.fingerColors[fingers[i]];
            ctx.fillText(`${labels[i]}:${stateText}`, x, y);
        }
    }

    /**
     * Start animation loop
     */
    startAnimation() {
        const animate = () => {
            this.render();
            this._animationFrame = requestAnimationFrame(animate);
        };
        animate();
    }

    /**
     * Stop animation
     */
    stopAnimation() {
        if (this._animationFrame) {
            cancelAnimationFrame(this._animationFrame);
        }
    }
}

/**
 * 3D Hand Visualizer using CSS 3D transforms
 *
 * Simpler than WebGL, works without additional libraries.
 */
class HandVisualizer3D {
    constructor(container, options = {}) {
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

        // Create finger elements
        this.fingerElements = {};
        this._createHand();

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
    }

    /**
     * Create hand elements
     */
    _createHand() {
        const fingerData = [
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
    setFingerStates(states) {
        for (const [finger, state] of Object.entries(states)) {
            if (this.fingerStates[finger] !== undefined) {
                this.fingerStates[finger] = state;
                this._updateFingerPose(finger);
            }
        }
    }

    /**
     * Update individual finger pose
     */
    _updateFingerPose(finger) {
        const { element, baseData } = this.fingerElements[finger];
        if (!element) return;

        const state = this.fingerStates[finger];
        const flexAngle = state * 30; // 0-60 degrees per segment

        const segments = element.querySelectorAll('.segment');
        segments.forEach((seg, i) => {
            seg.style.transform = `rotateX(${flexAngle}deg)`;
        });
    }

    /**
     * Set hand rotation
     */
    setRotation(x, y) {
        this.rotationX = x;
        this.rotationY = y;
        this.hand.style.transform = `rotateX(${x}deg) rotateY(${y}deg)`;
    }

    /**
     * Start auto-rotation
     */
    startAutoRotate() {
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
    stopAutoRotate() {
        this.autoRotate = false;
    }
}

/**
 * Finger State Display
 *
 * Simple text-based display of finger states,
 * useful for debugging and compact display.
 */
class FingerStateDisplay {
    constructor(container) {
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

        this.rows = {};
        this.container.querySelectorAll('.finger-row').forEach(row => {
            this.rows[row.dataset.finger] = row;
        });
    }

    /**
     * Update display
     */
    setFingerStates(states) {
        const stateLabels = ['EXT', 'PRT', 'FLX'];

        for (const [finger, state] of Object.entries(states)) {
            const row = this.rows[finger];
            if (!row) continue;

            const stateEl = row.querySelector('.state');
            const barEl = row.querySelector('.bar');

            const stateIdx = Math.round(Math.max(0, Math.min(2, state)));
            stateEl.textContent = stateLabels[stateIdx];

            // Progress bar
            const pct = (state / 2) * 100;
            barEl.innerHTML = `
                <span style="display: inline-block; width: 50px; height: 8px; background: #333; border-radius: 4px; overflow: hidden;">
                    <span style="display: block; width: ${pct}%; height: 100%; background: ${state < 1 ? '#2ecc71' : state < 1.5 ? '#f39c12' : '#e74c3c'};"></span>
                </span>
            `;
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        HandVisualizer2D,
        HandVisualizer3D,
        FingerStateDisplay
    };
}
