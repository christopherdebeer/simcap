/**
 * Magnetic Trajectory Visualizer
 * Real-time 3D visualization of residual magnetic field (finger magnet signals)
 *
 * Usage:
 *   const viz = new MagneticTrajectory(canvasElement, options);
 *   viz.addPoint(fused_mx, fused_my, fused_mz);
 *   viz.clear();
 */

const LINE_WIDTH = 5

export class MagneticTrajectory {
    /**
     * @param {HTMLCanvasElement} canvas - Canvas element to draw on
     * @param {Object} options - Configuration options
     * @param {number} options.maxPoints - Maximum number of points to keep (default: 200)
     * @param {number} options.scale - Visual scale factor (default: 0.35)
     * @param {boolean} options.autoNormalize - Auto-normalize to fit bounds (default: true)
     * @param {number} options.fixedBounds - Fixed bounds in µT (default: null for auto)
     * @param {string} options.trajectoryColor - Line color (default: '#4ecdc4')
     * @param {boolean} options.showMarkers - Show start/end markers (default: true)
     * @param {boolean} options.showCube - Show bounding cube (default: true)
     * @param {string} options.backgroundColor - Canvas background color (default: null for transparent)
     */
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Configuration
        this.maxPoints = options.maxPoints || 200;
        this.scale = options.scale || 0.35;
        this.autoNormalize = options.autoNormalize !== false;
        this.fixedBounds = options.fixedBounds || null;
        this.trajectoryColor = options.trajectoryColor || '#4ecdc4';
        this.showMarkers = options.showMarkers !== false;
        this.showCube = options.showCube !== false;
        this.backgroundColor = options.backgroundColor || null;

        // Data buffer
        this.points = [];

        // Animation frame tracking
        this.animationFrame = null;
        this.lastRenderTime = 0;
        this.renderThrottle = 50; // ms between renders
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

        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = LINE_WIDTH;
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
     * Draw trajectory path
     * @param {Array} normalizedPoints - Points normalized to [-1, 1]
     */
    drawTrajectory(normalizedPoints) {
        if (normalizedPoints.length < 2) return;

        this.ctx.strokeStyle = this.trajectoryColor;
        this.ctx.lineWidth = LINE_WIDTH;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Draw with gradient (fade old points)
        const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, 0);

        this.ctx.beginPath();
        const [sx, sy] = this.project(...normalizedPoints[0]);
        this.ctx.moveTo(sx, sy);

        for (let i = 1; i < normalizedPoints.length; i++) {
            const [px, py] = this.project(...normalizedPoints[i]);

            // Vary opacity based on age
            const age = i / normalizedPoints.length;
            this.ctx.globalAlpha = 0.3 + age * 0.7;

            this.ctx.lineTo(px, py);
        }

        this.ctx.stroke();
        this.ctx.globalAlpha = 1.0;
    }

    /**
     * Draw start/end markers
     * @param {Array} normalizedPoints - Points normalized to [-1, 1]
     */
    drawMarkers(normalizedPoints) {
        if (normalizedPoints.length === 0) return;

        // Start marker (green)
        const [sx, sy] = this.project(...normalizedPoints[0]);
        this.ctx.fillStyle = '#2ecc71';
        this.ctx.beginPath();
        this.ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.fillStyle = '#fff';
        this.ctx.font = '9px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('S', sx, sy);

        // End marker (red) - only if we have multiple points
        if (normalizedPoints.length > 1) {
            const [ex, ey] = this.project(...normalizedPoints[normalizedPoints.length - 1]);
            this.ctx.fillStyle = '#e74c3c';
            this.ctx.beginPath();
            this.ctx.arc(ex, ey, 6, 0, Math.PI * 2);
            this.ctx.fill();

            this.ctx.fillStyle = '#fff';
            this.ctx.fillText('E', ex, ey);
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

        // Draw start/end markers
        if (this.showMarkers) {
            this.drawMarkers(normalizedPoints);
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
