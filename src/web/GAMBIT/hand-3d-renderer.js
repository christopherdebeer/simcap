/**
 * 3D Hand Renderer
 * Real-time 3D hand visualization with joint-level control
 * Supports pose labels and orientation from IMU sensor fusion
 *
 * The sensor is positioned in the palm (face up), so orientation from
 * the IMU is used to rotate the hand model in 3D space.
 */

/**
 * Simple low-pass filter for smooth value transitions
 */
class LowPassFilter {
    constructor(alpha = 0.3) {
        this.alpha = alpha;  // 0-1, lower = smoother but more lag
        this.value = null;
    }

    filter(newValue) {
        if (this.value === null) {
            this.value = newValue;
        } else {
            this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
        }
        return this.value;
    }

    reset() {
        this.value = null;
    }

    setValue(value) {
        this.value = value;
    }
}

class Hand3DRenderer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.options = options;

        // Hand structure: palm center at origin, fingers extend in +Y
        // Finger data: [name, baseX, baseZ, spreadAngle, lengths[3], constraints MCP/PIP/DIP]
        this.fingers = [
            ['Thumb',  -0.8, 0.3, -60, [0.4, 0.35, 0.3], [-20,60], [-10,90], [0,80]],
            ['Index',  -0.4, 0,    -8, [0.5, 0.35, 0.25], [-20,20], [0,100], [0,90]],
            ['Middle', 0,    0,     0, [0.55, 0.4, 0.28], [-15,15], [0,100], [0,90]],
            ['Ring',   0.4,  0,     8, [0.5, 0.35, 0.25], [-15,15], [0,100], [0,90]],
            ['Pinky',  0.75, 0.1,  18, [0.4, 0.28, 0.2], [-20,20], [0,100], [0,90]]
        ];

        // Joint angles per finger: [spread, MCP, PIP, DIP]
        // 0 = extended, higher values = more flexed
        this.joints = this.fingers.map(() => [0, 0, 0, 0]);

        // Hand orientation (degrees) - face-on static view
        // When sensor fusion is active, this is updated from IMU Euler angles
        this.orientation = {
            pitch: 0,
            yaw: 0,
            roll: 0
        };

        // Target orientation for smooth interpolation
        this._targetOrientation = { pitch: 0, yaw: 0, roll: 0 };

        // Low-pass filter state for smooth orientation transitions
        this._orientationFilters = {
            pitch: new LowPassFilter(options.orientationAlpha || 0.15),
            yaw: new LowPassFilter(options.orientationAlpha || 0.15),
            roll: new LowPassFilter(options.orientationAlpha || 0.15)
        };

        // Orientation mode: 'static' or 'sensor_fusion'
        this.orientationMode = options.orientationMode || 'static';

        // Orientation filtering enabled
        this.orientationFilteringEnabled = options.orientationFiltering !== false;

        // Base orientation offset (to align hand model with sensor orientation)
        // Sensor is in palm facing up, so we need to offset to match
        this.orientationOffset = {
            pitch: options.pitchOffset || 0,    // Tilt forward/back
            yaw: options.yawOffset || 0,        // Rotation around vertical
            roll: options.rollOffset || 0       // Tilt left/right
        };

        // Colors
        this.fingerColors = options.fingerColors || [
            '#e74c3c', // Thumb - red
            '#e67e22', // Index - orange
            '#f1c40f', // Middle - yellow
            '#2ecc71', // Ring - green
            '#3498db'  // Pinky - blue
        ];
        this.palmColor = options.palmColor || '#446';

        // Use fixed canvas dimensions from HTML attributes
        this.W = canvas.width;
        this.H = canvas.height;
    }

    /**
     * Set finger poses from state values (0=extended, 1=partial, 2=flexed)
     * @param {Object} poses - {thumb: 0-2, index: 0-2, middle: 0-2, ring: 0-2, pinky: 0-2}
     */
    setFingerPoses(poses) {
        const fingerNames = ['thumb', 'index', 'middle', 'ring', 'pinky'];

        fingerNames.forEach((name, i) => {
            const state = poses[name] || 0;

            // Map state (0-2) to joint angles
            // Extended (0): all joints 0
            // Partial (1): MCP=40, PIP=30, DIP=20
            // Flexed (2): MCP=80, PIP=70, DIP=60
            const intensity = state / 2; // 0 to 1

            this.joints[i][0] = 0; // Spread stays at 0
            this.joints[i][1] = intensity * 80; // MCP
            this.joints[i][2] = intensity * 70; // PIP
            this.joints[i][3] = intensity * 60; // DIP
        });
    }

    /**
     * Set hand orientation from IMU sensor fusion (Euler angles)
     * Applies low-pass filtering when enabled for smooth transitions
     * @param {Object} orientation - {pitch, yaw, roll} in degrees (from Madgwick AHRS)
     * @param {boolean} immediate - If true, skip filtering and apply immediately
     */
    setOrientation(orientation, immediate = false) {
        // Store target orientation
        if (orientation.pitch !== undefined) this._targetOrientation.pitch = orientation.pitch;
        if (orientation.yaw !== undefined) this._targetOrientation.yaw = orientation.yaw;
        if (orientation.roll !== undefined) this._targetOrientation.roll = orientation.roll;

        // Apply filtering if enabled and not immediate
        if (this.orientationFilteringEnabled && !immediate) {
            this.orientation.pitch = this._orientationFilters.pitch.filter(this._targetOrientation.pitch);
            this.orientation.yaw = this._orientationFilters.yaw.filter(this._targetOrientation.yaw);
            this.orientation.roll = this._orientationFilters.roll.filter(this._targetOrientation.roll);
        } else {
            // Direct assignment without filtering
            this.orientation.pitch = this._targetOrientation.pitch;
            this.orientation.yaw = this._targetOrientation.yaw;
            this.orientation.roll = this._targetOrientation.roll;
        }
    }

    /**
     * Set orientation mode
     * @param {string} mode - 'static' or 'sensor_fusion'
     */
    setOrientationMode(mode) {
        this.orientationMode = mode;
        if (mode === 'static') {
            // Reset to face-on view
            this.resetOrientation();
        }
    }

    /**
     * Reset orientation to default face-on view
     */
    resetOrientation() {
        this.orientation = { pitch: 0, yaw: 0, roll: 0 };
        this._targetOrientation = { pitch: 0, yaw: 0, roll: 0 };
        this._orientationFilters.pitch.reset();
        this._orientationFilters.yaw.reset();
        this._orientationFilters.roll.reset();
    }

    /**
     * Update orientation from sensor fusion (IMU Euler angles)
     * Maps IMU orientation to hand model orientation
     * Sensor is in palm facing up, so we apply appropriate transformations
     * @param {Object} euler - {roll, pitch, yaw} in degrees from Madgwick AHRS
     */
    updateFromSensorFusion(euler) {
        if (this.orientationMode !== 'sensor_fusion') return;
        if (!euler) return;

        // Map IMU Euler angles to hand model orientation
        // The sensor is in the palm, facing up:
        // - IMU roll -> hand pitch (tilting forward/back)
        // - IMU pitch -> hand roll (tilting left/right)
        // - IMU yaw -> hand yaw (rotating around vertical axis)
        const mappedOrientation = {
            pitch: -euler.pitch + this.orientationOffset.pitch,  // Invert pitch for natural feel
            yaw: euler.yaw + this.orientationOffset.yaw,
            roll: euler.roll + this.orientationOffset.roll
        };

        this.setOrientation(mappedOrientation);
    }

    /**
     * Enable/disable orientation filtering
     * @param {boolean} enabled
     */
    setOrientationFiltering(enabled) {
        this.orientationFilteringEnabled = enabled;
        if (!enabled) {
            // Reset filters when disabled
            this._orientationFilters.pitch.reset();
            this._orientationFilters.yaw.reset();
            this._orientationFilters.roll.reset();
        }
    }

    /**
     * Set orientation filter alpha (smoothing factor)
     * @param {number} alpha - 0-1, lower = smoother but more lag
     */
    setOrientationFilterAlpha(alpha) {
        this._orientationFilters.pitch.alpha = alpha;
        this._orientationFilters.yaw.alpha = alpha;
        this._orientationFilters.roll.alpha = alpha;
    }

    /**
     * Set orientation offset (to align hand model with sensor)
     * @param {Object} offset - {pitch, yaw, roll} offset in degrees
     */
    setOrientationOffset(offset) {
        if (offset.pitch !== undefined) this.orientationOffset.pitch = offset.pitch;
        if (offset.yaw !== undefined) this.orientationOffset.yaw = offset.yaw;
        if (offset.roll !== undefined) this.orientationOffset.roll = offset.roll;
    }

    /**
     * Get current orientation state
     * @returns {Object} Current orientation, mode, and filter settings
     */
    getOrientationState() {
        return {
            orientation: { ...this.orientation },
            targetOrientation: { ...this._targetOrientation },
            mode: this.orientationMode,
            filteringEnabled: this.orientationFilteringEnabled,
            offset: { ...this.orientationOffset }
        };
    }

    /**
     * Render the hand
     */
    render() {
        const ctx = this.ctx;

        // Clear
        ctx.fillStyle = this.options.backgroundColor || '#1a1a2e';
        ctx.fillRect(0, 0, this.W, this.H);

        const pitch = this._rad(this.orientation.pitch);
        const yaw = this._rad(this.orientation.yaw);
        const roll = this._rad(this.orientation.roll);

        // Hand transform
        let handM = this._matMul(this._matRotY(yaw), this._matRotX(pitch));
        handM = this._matMul(handM, this._matRotZ(roll));

        const lines = [];
        const pts = [];

        // Palm - simple box
        const palm = [
            [-0.9,-0.3,0.1], [0.8,-0.3,0.1], [0.9,0.6,-0.05], [-0.7,0.6,0.15],
            [-0.9,-0.3,-0.1], [0.8,-0.3,-0.1], [0.9,0.6,-0.15], [-0.7,0.6,-0.05]
        ];
        const palmT = palm.map(p => this._matApply(handM, p));

        // Palm edges
        [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a,b]) => {
            lines.push([palmT[a], palmT[b], this.palmColor]);
        });

        // Fingers
        this.fingers.forEach((f, fi) => {
            const [name, bx, bz, spread, lens] = f;
            const [spreadAdj, mcp, pip, dip] = this.joints[fi];

            // Base position on palm
            let m = this._matMul(handM, this._matTrans(bx, 0.6, bz));

            // Spread angle (base splay)
            m = this._matMul(m, this._matRotZ(this._rad(spread + spreadAdj)));

            // For thumb, rotate base differently
            if (fi === 0) {
                m = this._matMul(m, this._matRotY(this._rad(-45)));
                m = this._matMul(m, this._matRotZ(this._rad(-30)));
            }

            const color = this.fingerColors[fi];
            let pos = this._matApply(m, [0,0,0]);
            pts.push([pos, color]);

            // Each segment (MCP, PIP, DIP)
            [mcp, pip, dip].forEach((angle, si) => {
                // Flex around X axis (curl toward palm)
                m = this._matMul(m, this._matRotX(this._rad(angle)));
                m = this._matMul(m, this._matTrans(0, lens[si], 0));

                const newPos = this._matApply(m, [0,0,0]);
                lines.push([pos, newPos, color]);
                pts.push([newPos, color]);
                pos = newPos;
            });
        });

        // Sort by depth and draw
        lines.sort((a,b) => (a[0][2]+a[1][2])/2 - (b[0][2]+b[1][2])/2);
        lines.forEach(([p1, p2, col]) => {
            const a = this._project(p1), b = this._project(p2);
            ctx.strokeStyle = col;
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(a[0], a[1]);
            ctx.lineTo(b[0], b[1]);
            ctx.stroke();
        });

        pts.sort((a,b) => a[0][2] - b[0][2]);
        pts.forEach(([p, col]) => {
            const pr = this._project(p);
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(pr[0], pr[1], 6, 0, Math.PI*2);
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
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

    // Math helpers
    _rad(d) { return d * Math.PI / 180; }

    _matId() { return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }

    _matRotX(a) {
        const c=Math.cos(a), s=Math.sin(a);
        return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1];
    }

    _matRotY(a) {
        const c=Math.cos(a), s=Math.sin(a);
        return [c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1];
    }

    _matRotZ(a) {
        const c=Math.cos(a), s=Math.sin(a);
        return [c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1];
    }

    _matTrans(x,y,z) {
        return [1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1];
    }

    _matMul(a,b) {
        const r = [];
        for(let i=0; i<4; i++) {
            for(let j=0; j<4; j++) {
                r[i*4+j] = a[i*4]*b[j] + a[i*4+1]*b[4+j] + a[i*4+2]*b[8+j] + a[i*4+3]*b[12+j];
            }
        }
        return r;
    }

    _matApply(m, p) {
        return [
            m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3],
            m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7],
            m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11]
        ];
    }

    _project(p) {
        const scale = 400, z = p[2] + 4;
        return [
            this.W/2 + p[0] * scale / z,
            this.H/2 - p[1] * scale / z,
            z
        ];
    }
}

// Export for use in collector
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Hand3DRenderer;
}
