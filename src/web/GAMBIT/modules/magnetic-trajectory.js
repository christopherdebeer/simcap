/**
 * Magnetic Trajectory Visualizer v1.1.0
 * Real-time 3D visualization of residual magnetic field (finger magnet signals)
 *
 * Usage:
 *   const viz = new MagneticTrajectory(canvasElement, options);
 *   viz.addPoint(fused_mx, fused_my, fused_mz);
 *   viz.clear();
 * 
 * Normalization:
 *   Uses fixed bounds based on information-theoretic SNR analysis:
 *   - Sensor noise floor: ~1 μT (LIS3MDL)
 *   - Earth field variation: ~10 μT
 *   - Finger magnet signal: 14-141 μT (depending on distance/magnet size)
 *   - Default fixedBounds: 200 μT (conservative upper bound)
 *   
 *   This provides consistent scaling across sessions, preserving SNR relationships.
 *   See docs/design/magnetic-finger-tracking-analysis.md for physics details.
 */

const LINE_WIDTH = 5

/**
 * Default fixed bounds for magnetometer normalization (in μT)
 * Based on information-theoretic SNR analysis:
 * - Max expected signal: ~150 μT (6x3mm magnet at 50mm)
 * - Safety margin: 1.33x
 * - Result: 200 μT provides consistent scaling while accommodating outliers
 */
const DEFAULT_FIXED_BOUNDS_UT = 200

export class MagneticTrajectory {
    /**
     * @param {HTMLCanvasElement} canvas - Canvas element to draw on
     * @param {Object} options - Configuration options
     * @param {number} options.maxPoints - Maximum number of points to keep (default: 200)
     * @param {number} options.scale - Visual scale factor (default: 0.35)
     * @param {boolean} options.autoNormalize - Auto-normalize to fit bounds (default: false, uses fixedBounds)
     * @param {number} options.fixedBounds - Fixed bounds in µT (default: 200 based on SNR analysis)
     * @param {string} options.trajectoryColor - Line color (default: '#4ecdc4')
     * @param {boolean} options.showMarkers - Show start/end markers (default: true)
     * @param {boolean} options.showCube - Show bounding cube (default: true)
     * @param {string} options.backgroundColor - Canvas background color (default: null for transparent)
     * @param {number} options.minAlpha - Minimum alpha for oldest points (default: 0, fully transparent)
     * @param {number} options.maxAlpha - Maximum alpha for newest points (default: 1, fully opaque)
     * @param {boolean} options.showScaleKey - Show scale bar and units legend (default: true)
     * @param {string} options.scaleKeyPosition - Position: 'bottom-left', 'bottom-right', 'top-left', 'top-right' (default: 'bottom-left')
     */
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Configuration
        this.maxPoints = options.maxPoints || 200;
        this.scale = options.scale || 0.35;
        // Default to fixed bounds (SNR-based normalization) unless explicitly set to auto
        this.autoNormalize = options.autoNormalize === true;
        this.fixedBounds = options.fixedBounds !== undefined ? options.fixedBounds : DEFAULT_FIXED_BOUNDS_UT;
        this.trajectoryColor = options.trajectoryColor || '#4ecdc4';
        this.showMarkers = options.showMarkers !== false;
        this.showCube = options.showCube !== false;
        this.backgroundColor = options.backgroundColor || null;
        // Gradient fade settings: oldest points fade to minAlpha, newest to maxAlpha
        this.minAlpha = options.minAlpha !== undefined ? options.minAlpha : 0;
        this.maxAlpha = options.maxAlpha !== undefined ? options.maxAlpha : 1;
        // Scale key settings
        this.showScaleKey = options.showScaleKey !== false;
        this.scaleKeyPosition = options.scaleKeyPosition || 'bottom-left';

        // Data buffer
        this.points = [];

        // Animation frame tracking
        this.animationFrame = null;
        this.lastRenderTime = 0;
        this.renderThrottle = 50; // ms between renders
        this.clear();
    }

    /**
     * Add a new magnetic field measurement
     * @param {number} mx - X component (µT)
     * @param {number} my - Y component (µT)
     * @param {number} mz - Z component (µT)
     */
    addPoint(mx, my, mz) {
        // Skip invalid points
        if (mx === undefined || my === undefined || mz === undefined) return;
        if (!isFinite(mx) || !isFinite(my) || !isFinite(mz)) return;

        this.points.push([mx, my, mz]);

        // Keep buffer size limited
        if (this.points.length > this.maxPoints) {
            this.points.shift();
        }

        // Throttled render
        this.scheduleRender();
    }

    /**
     * Schedule a render (throttled to avoid excessive redraws)
     */
    scheduleRender() {
        const now = performance.now();
        if (now - this.lastRenderTime >= this.renderThrottle) {
            this.render();
        } else if (!this.animationFrame) {
            this.animationFrame = requestAnimationFrame(() => {
                this.animationFrame = null;
                this.render();
            });
        }
    }

    /**
     * Clear all points
     */
    clear() {
        this.points = [];
        this.render();
    }

    /**
     * Project 3D point to 2D isometric view
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} z - Z coordinate
     * @returns {Array} [screenX, screenY]
     */
    project(x, y, z) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const cx = w / 2;
        const cy = h / 2;
        const scale = Math.min(w, h) * this.scale;

        // Isometric projection
        return [
            cx + (x - z * 0.5) * scale,
            cy + (-y + z * 0.3) * scale
        ];
    }

    /**
     * Normalize points to [-1, 1] cube
     * @returns {Array} Normalized points
     */
    normalizePoints() {
        if (this.points.length === 0) return [];

        // Use fixed bounds if specified
        if (this.fixedBounds) {
            const b = this.fixedBounds;
            return this.points.map(p => [
                Math.max(-1, Math.min(1, p[0] / b)),
                Math.max(-1, Math.min(1, p[1] / b)),
                Math.max(-1, Math.min(1, p[2] / b))
            ]);
        }

        // Auto-normalize to fit data
        if (!this.autoNormalize) {
            return this.points;
        }

        // Find bounding box
        let min = [Infinity, Infinity, Infinity];
        let max = [-Infinity, -Infinity, -Infinity];

        for (const p of this.points) {
            for (let i = 0; i < 3; i++) {
                min[i] = Math.min(min[i], p[i]);
                max[i] = Math.max(max[i], p[i]);
            }
        }

        const range = max.map((v, i) => v - min[i] || 1);
        return this.points.map(p => p.map((v, i) => ((v - min[i]) / range[i]) * 2 - 1));
    }

    /**
     * Draw bounding cube wireframe
     */
    drawCube() {
        const corners = [];
        for (let i = 0; i < 8; i++) {
            corners.push([
                (i & 1) * 2 - 1,
                ((i >> 1) & 1) * 2 - 1,
                ((i >> 2) & 1) * 2 - 1
            ]);
        }

        const edges = [
            [0,1],[2,3],[4,5],[6,7],  // X edges
            [0,2],[1,3],[4,6],[5,7],  // Y edges
            [0,4],[1,5],[2,6],[3,7]   // Z edges
        ];

        this.ctx.strokeStyle = '#33333317';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();

        for (const [a, b] of edges) {
            const [x1, y1] = this.project(...corners[a]);
            const [x2, y2] = this.project(...corners[b]);
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
        }

        this.ctx.stroke();
    }

    /**
     * Draw trajectory path with gradient fade
     * Oldest points fade to minAlpha, newest points are at maxAlpha
     * @param {Array} normalizedPoints - Points normalized to [-1, 1]
     */
    drawTrajectory(normalizedPoints) {
        if (normalizedPoints.length < 2) return;

        this.ctx.strokeStyle = this.trajectoryColor;
        this.ctx.lineWidth = LINE_WIDTH;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Draw each segment individually with varying alpha
        // This is necessary because Canvas 2D globalAlpha applies to entire stroke,
        // not per-segment within a single path
        const n = normalizedPoints.length;
        const alphaRange = this.maxAlpha - this.minAlpha;

        for (let i = 1; i < n; i++) {
            // Calculate alpha based on position in trajectory
            // i=1 (oldest visible segment) -> minAlpha
            // i=n-1 (newest segment) -> maxAlpha
            const t = (i - 1) / (n - 2 || 1);  // Normalize to [0, 1]
            this.ctx.globalAlpha = this.minAlpha + t * alphaRange;

            // Draw this segment
            const [x1, y1] = this.project(...normalizedPoints[i - 1]);
            const [x2, y2] = this.project(...normalizedPoints[i]);

            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
            this.ctx.stroke();
        }

        this.ctx.globalAlpha = 1.0;
    }

    

    /**
     * Draw scale key showing units and bounds
     * Displays a scale bar and the current normalization bounds in μT
     */
    drawScaleKey() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const padding = 10;
        const barHeight = 4;
        
        // Calculate scale bar length (represents half the bounds, i.e., one side of the cube)
        // The cube spans [-1, 1] which is 2 * fixedBounds in μT
        // Scale bar represents fixedBounds/2 (quarter of full range) for readability
        const cubeSize = Math.min(w, h) * this.scale * 2;  // Full cube width in pixels
        const scaleBarPixels = cubeSize * 0.25;  // 25% of cube = fixedBounds/2 μT
        const scaleBarValue = this.fixedBounds ? this.fixedBounds / 2 : 50;  // μT represented by bar
        
        // Determine position based on scaleKeyPosition
        let x, y, textAlign;
        switch (this.scaleKeyPosition) {
            case 'top-left':
                x = padding;
                y = padding + 12;
                textAlign = 'left';
                break;
            case 'top-right':
                x = w - padding - scaleBarPixels;
                y = padding + 12;
                textAlign = 'right';
                break;
            case 'bottom-right':
                x = w - padding - scaleBarPixels;
                y = h - padding - 20;
                textAlign = 'right';
                break;
            case 'bottom-left':
            default:
                x = padding;
                y = h - padding - 20;
                textAlign = 'left';
                break;
        }

        // Draw scale bar
        this.ctx.fillStyle = '#666';
        this.ctx.fillRect(x, y, scaleBarPixels, barHeight);
        
        // Draw end caps
        const capHeight = 8;
        this.ctx.fillRect(x, y - (capHeight - barHeight) / 2, 2, capHeight);
        this.ctx.fillRect(x + scaleBarPixels - 2, y - (capHeight - barHeight) / 2, 2, capHeight);

        // Draw scale value label
        this.ctx.fillStyle = '#888';
        this.ctx.font = '10px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        this.ctx.fillText(`${scaleBarValue} μT`, x + scaleBarPixels / 2, y + barHeight + 2);

        // Draw bounds label below
        if (this.fixedBounds) {
            this.ctx.fillStyle = '#666';
            this.ctx.font = '9px sans-serif';
            this.ctx.textAlign = textAlign;
            const boundsX = textAlign === 'left' ? x : x + scaleBarPixels;
            this.ctx.fillText(`Scale: ±${this.fixedBounds} μT`, boundsX, y + barHeight + 14);
        } else if (this.autoNormalize) {
            this.ctx.fillStyle = '#666';
            this.ctx.font = '9px sans-serif';
            this.ctx.textAlign = textAlign;
            const boundsX = textAlign === 'left' ? x : x + scaleBarPixels;
            this.ctx.fillText('Scale: auto', boundsX, y + barHeight + 14);
        }
    }

    /**
     * Main render function
     */
    render() {
        this.lastRenderTime = performance.now();

        const w = this.canvas.width;
        const h = this.canvas.height;

        // Clear canvas
        if (this.backgroundColor) {
            this.ctx.fillStyle = this.backgroundColor;
            this.ctx.fillRect(0, 0, w, h);
        } else {
            this.ctx.clearRect(0, 0, w, h);
        }

        // Draw cube wireframe
        if (this.showCube) {
            this.drawCube();
        }

        // Normalize and draw trajectory
        const normalizedPoints = this.normalizePoints();
        this.drawTrajectory(normalizedPoints);

        // // Draw start/end markers
        // if (this.showMarkers) {
        //     this.drawMarkers(normalizedPoints);
        // }

        // Draw scale key with units
        if (this.showScaleKey) {
            this.drawScaleKey();
        }
    }

    /**
     * Get current statistics
     * @returns {Object} Stats about current trajectory
     */
    getStats() {
        if (this.points.length === 0) {
            return { count: 0, magnitude: { min: 0, max: 0, avg: 0 } };
        }

        const magnitudes = this.points.map(p => Math.sqrt(p[0]**2 + p[1]**2 + p[2]**2));

        return {
            count: this.points.length,
            magnitude: {
                min: Math.min(...magnitudes),
                max: Math.max(...magnitudes),
                avg: magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length
            }
        };
    }
}
