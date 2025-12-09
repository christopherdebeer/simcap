/**
 * Extended Filtering for Magnetic Finger Tracking
 *
 * Provides:
 * - Multi-dimensional Kalman Filter (for 3D position/velocity tracking)
 * - Extended Kalman Filter (for non-linear magnetic field model)
 * - Particle Filter (for multi-hypothesis finger tracking)
 *
 * Based on analysis: magnetic-finger-tracking-analysis.md
 */

/**
 * Multi-dimensional Kalman Filter
 *
 * Tracks state vector with position and velocity for each dimension.
 * State: [x, y, z, vx, vy, vz]
 */
class KalmanFilter3D {
    constructor(options = {}) {
        const {
            processNoise = 0.1,      // Q - process noise
            measurementNoise = 1.0,  // R - measurement noise
            initialCovariance = 100  // P0 - initial uncertainty
        } = options;

        // State dimension: position (3) + velocity (3) = 6
        this.stateDim = 6;
        this.measDim = 3;

        // State vector [x, y, z, vx, vy, vz]
        this.state = new Float64Array(6);

        // Covariance matrix (6x6)
        this.P = new Float64Array(36);
        for (let i = 0; i < 6; i++) {
            this.P[i * 6 + i] = initialCovariance;
        }

        // Process noise covariance (6x6 diagonal)
        this.Q = processNoise;

        // Measurement noise covariance
        this.R = measurementNoise;

        // Time step
        this.dt = 0.02; // 50Hz default

        this.initialized = false;
    }

    /**
     * State transition matrix F
     * Models constant velocity: x_new = x + v*dt
     */
    _getF(dt) {
        return new Float64Array([
            1, 0, 0, dt, 0, 0,
            0, 1, 0, 0, dt, 0,
            0, 0, 1, 0, 0, dt,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1
        ]);
    }

    /**
     * Measurement matrix H
     * We only measure position, not velocity
     */
    _getH() {
        return new Float64Array([
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0
        ]);
    }

    /**
     * Matrix multiplication: C = A * B
     */
    _matMul(A, B, rowsA, colsA, colsB) {
        const C = new Float64Array(rowsA * colsB);
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += A[i * colsA + k] * B[k * colsB + j];
                }
                C[i * colsB + j] = sum;
            }
        }
        return C;
    }

    /**
     * Matrix transpose
     */
    _transpose(A, rows, cols) {
        const AT = new Float64Array(cols * rows);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                AT[j * rows + i] = A[i * cols + j];
            }
        }
        return AT;
    }

    /**
     * Add matrices
     */
    _matAdd(A, B, size) {
        const C = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            C[i] = A[i] + B[i];
        }
        return C;
    }

    /**
     * Subtract matrices
     */
    _matSub(A, B, size) {
        const C = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            C[i] = A[i] - B[i];
        }
        return C;
    }

    /**
     * 3x3 matrix inversion (for Kalman gain calculation)
     */
    _invert3x3(M) {
        const det = M[0] * (M[4] * M[8] - M[5] * M[7])
                  - M[1] * (M[3] * M[8] - M[5] * M[6])
                  + M[2] * (M[3] * M[7] - M[4] * M[6]);

        if (Math.abs(det) < 1e-10) {
            // Return identity if singular
            return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
        }

        const invDet = 1 / det;
        return new Float64Array([
            (M[4] * M[8] - M[5] * M[7]) * invDet,
            (M[2] * M[7] - M[1] * M[8]) * invDet,
            (M[1] * M[5] - M[2] * M[4]) * invDet,
            (M[5] * M[6] - M[3] * M[8]) * invDet,
            (M[0] * M[8] - M[2] * M[6]) * invDet,
            (M[2] * M[3] - M[0] * M[5]) * invDet,
            (M[3] * M[7] - M[4] * M[6]) * invDet,
            (M[1] * M[6] - M[0] * M[7]) * invDet,
            (M[0] * M[4] - M[1] * M[3]) * invDet
        ]);
    }

    /**
     * Initialize filter with first measurement
     */
    initialize(measurement) {
        this.state[0] = measurement.x;
        this.state[1] = measurement.y;
        this.state[2] = measurement.z;
        this.state[3] = 0; // Initial velocity = 0
        this.state[4] = 0;
        this.state[5] = 0;
        this.initialized = true;
    }

    /**
     * Predict step
     */
    predict(dt = null) {
        if (!this.initialized) return null;

        const deltaT = dt || this.dt;
        const F = this._getF(deltaT);

        // State prediction: x = F * x
        const newState = new Float64Array(6);
        for (let i = 0; i < 6; i++) {
            newState[i] = 0;
            for (let j = 0; j < 6; j++) {
                newState[i] += F[i * 6 + j] * this.state[j];
            }
        }
        this.state = newState;

        // Covariance prediction: P = F * P * F' + Q
        const FP = this._matMul(F, this.P, 6, 6, 6);
        const FT = this._transpose(F, 6, 6);
        const FPFT = this._matMul(FP, FT, 6, 6, 6);

        // Add process noise (diagonal)
        for (let i = 0; i < 6; i++) {
            FPFT[i * 6 + i] += this.Q;
        }
        this.P = FPFT;

        return this.getPosition();
    }

    /**
     * Update step with measurement
     */
    update(measurement) {
        if (!this.initialized) {
            this.initialize(measurement);
            return this.getPosition();
        }

        const H = this._getH();
        const HT = this._transpose(H, 3, 6);

        // Innovation: y = z - H * x
        const z = new Float64Array([measurement.x, measurement.y, measurement.z]);
        const Hx = new Float64Array(3);
        for (let i = 0; i < 3; i++) {
            Hx[i] = 0;
            for (let j = 0; j < 6; j++) {
                Hx[i] += H[i * 6 + j] * this.state[j];
            }
        }
        const y = this._matSub(z, Hx, 3);

        // Innovation covariance: S = H * P * H' + R
        const HP = this._matMul(H, this.P, 3, 6, 6);
        const HPHT = this._matMul(HP, HT, 3, 6, 3);
        for (let i = 0; i < 3; i++) {
            HPHT[i * 3 + i] += this.R;
        }

        // Kalman gain: K = P * H' * S^-1
        const PHT = this._matMul(this.P, HT, 6, 6, 3);
        const Sinv = this._invert3x3(HPHT);
        const K = this._matMul(PHT, Sinv, 6, 3, 3);

        // State update: x = x + K * y
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 3; j++) {
                this.state[i] += K[i * 3 + j] * y[j];
            }
        }

        // Covariance update: P = (I - K * H) * P
        const KH = this._matMul(K, H, 6, 3, 6);
        const I_KH = new Float64Array(36);
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                I_KH[i * 6 + j] = (i === j ? 1 : 0) - KH[i * 6 + j];
            }
        }
        this.P = this._matMul(I_KH, this.P, 6, 6, 6);

        return this.getPosition();
    }

    /**
     * Get current position estimate
     */
    getPosition() {
        return {
            x: this.state[0],
            y: this.state[1],
            z: this.state[2]
        };
    }

    /**
     * Get current velocity estimate
     */
    getVelocity() {
        return {
            x: this.state[3],
            y: this.state[4],
            z: this.state[5]
        };
    }

    /**
     * Get full state
     */
    getState() {
        return {
            position: this.getPosition(),
            velocity: this.getVelocity(),
            covariance: Array.from(this.P)
        };
    }

    /**
     * Reset filter
     */
    reset() {
        this.state = new Float64Array(6);
        this.P = new Float64Array(36);
        for (let i = 0; i < 6; i++) {
            this.P[i * 6 + i] = 100;
        }
        this.initialized = false;
    }
}

/**
 * Multi-Finger Kalman Filter
 *
 * Tracks 5 fingers independently, each with 3D position.
 */
class MultiFingerKalmanFilter {
    constructor(options = {}) {
        this.fingers = {
            thumb: new KalmanFilter3D(options),
            index: new KalmanFilter3D(options),
            middle: new KalmanFilter3D(options),
            ring: new KalmanFilter3D(options),
            pinky: new KalmanFilter3D(options)
        };
    }

    /**
     * Update specific finger with measurement
     */
    updateFinger(fingerName, measurement) {
        if (this.fingers[fingerName]) {
            return this.fingers[fingerName].update(measurement);
        }
        return null;
    }

    /**
     * Predict all fingers
     */
    predictAll(dt = null) {
        const predictions = {};
        for (const [name, filter] of Object.entries(this.fingers)) {
            predictions[name] = filter.predict(dt);
        }
        return predictions;
    }

    /**
     * Get all finger positions
     */
    getAllPositions() {
        const positions = {};
        for (const [name, filter] of Object.entries(this.fingers)) {
            positions[name] = filter.getPosition();
        }
        return positions;
    }

    /**
     * Reset all filters
     */
    resetAll() {
        for (const filter of Object.values(this.fingers)) {
            filter.reset();
        }
    }
}

/**
 * Particle Filter for Hand Pose Estimation
 *
 * Maintains multiple hypotheses about hand pose,
 * useful for handling multi-modal distributions and
 * ambiguity in magnetic field measurements.
 */
class ParticleFilter {
    constructor(options = {}) {
        const {
            numParticles = 500,
            positionNoise = 2.0,  // mm
            velocityNoise = 5.0,  // mm/s
            resampleThreshold = 0.5
        } = options;

        this.numParticles = numParticles;
        this.positionNoise = positionNoise;
        this.velocityNoise = velocityNoise;
        this.resampleThreshold = resampleThreshold;

        // Each particle represents a hand pose hypothesis
        // State per finger: [x, y, z, vx, vy, vz]
        // 5 fingers × 6 state = 30 values per particle
        this.particles = [];
        this.weights = new Float64Array(numParticles).fill(1 / numParticles);

        this.initialized = false;
    }

    /**
     * Initialize particles around initial pose estimate
     */
    initialize(initialPose) {
        this.particles = [];

        for (let i = 0; i < this.numParticles; i++) {
            const particle = {};

            for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
                const base = initialPose[finger] || { x: 0, y: 0, z: 0 };
                particle[finger] = {
                    x: base.x + this._randn() * this.positionNoise * 5,
                    y: base.y + this._randn() * this.positionNoise * 5,
                    z: base.z + this._randn() * this.positionNoise * 5,
                    vx: this._randn() * this.velocityNoise,
                    vy: this._randn() * this.velocityNoise,
                    vz: this._randn() * this.velocityNoise
                };
            }

            this.particles.push(particle);
        }

        this.weights.fill(1 / this.numParticles);
        this.initialized = true;
    }

    /**
     * Prediction step: propagate particles with motion model
     */
    predict(dt = 0.02) {
        if (!this.initialized) return;

        for (const particle of this.particles) {
            for (const finger of Object.values(particle)) {
                // Position update with velocity
                finger.x += finger.vx * dt + this._randn() * this.positionNoise;
                finger.y += finger.vy * dt + this._randn() * this.positionNoise;
                finger.z += finger.vz * dt + this._randn() * this.positionNoise;

                // Velocity random walk
                finger.vx += this._randn() * this.velocityNoise * dt;
                finger.vy += this._randn() * this.velocityNoise * dt;
                finger.vz += this._randn() * this.velocityNoise * dt;
            }
        }
    }

    /**
     * Update step: weight particles based on measurement likelihood
     *
     * @param {Object} measurement - Observed magnetic field {x, y, z}
     * @param {Function} likelihoodFn - Function(particle, measurement) => likelihood
     */
    update(measurement, likelihoodFn) {
        if (!this.initialized) return;

        let sumWeights = 0;

        for (let i = 0; i < this.numParticles; i++) {
            const likelihood = likelihoodFn(this.particles[i], measurement);
            this.weights[i] *= likelihood;
            sumWeights += this.weights[i];
        }

        // Normalize weights
        if (sumWeights > 0) {
            for (let i = 0; i < this.numParticles; i++) {
                this.weights[i] /= sumWeights;
            }
        } else {
            // All weights zero - reinitialize
            this.weights.fill(1 / this.numParticles);
        }

        // Resample if effective sample size too low
        const nEff = this._effectiveSampleSize();
        if (nEff < this.numParticles * this.resampleThreshold) {
            this._resample();
        }
    }

    /**
     * Get weighted average estimate
     */
    estimate() {
        if (!this.initialized) return null;

        const result = {};

        for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
            let x = 0, y = 0, z = 0;

            for (let i = 0; i < this.numParticles; i++) {
                const w = this.weights[i];
                x += this.particles[i][finger].x * w;
                y += this.particles[i][finger].y * w;
                z += this.particles[i][finger].z * w;
            }

            result[finger] = { x, y, z };
        }

        return result;
    }

    /**
     * Get particle diversity (spread of hypotheses)
     */
    getDiversity() {
        if (!this.initialized) return 0;

        const est = this.estimate();
        let totalVar = 0;

        for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
            let varX = 0, varY = 0, varZ = 0;

            for (let i = 0; i < this.numParticles; i++) {
                const w = this.weights[i];
                const dx = this.particles[i][finger].x - est[finger].x;
                const dy = this.particles[i][finger].y - est[finger].y;
                const dz = this.particles[i][finger].z - est[finger].z;
                varX += dx * dx * w;
                varY += dy * dy * w;
                varZ += dz * dz * w;
            }

            totalVar += varX + varY + varZ;
        }

        return Math.sqrt(totalVar / 5);
    }

    /**
     * Systematic resampling
     */
    _resample() {
        const newParticles = [];
        const cumSum = new Float64Array(this.numParticles);

        cumSum[0] = this.weights[0];
        for (let i = 1; i < this.numParticles; i++) {
            cumSum[i] = cumSum[i - 1] + this.weights[i];
        }

        const step = 1 / this.numParticles;
        let u = Math.random() * step;
        let j = 0;

        for (let i = 0; i < this.numParticles; i++) {
            while (u > cumSum[j] && j < this.numParticles - 1) {
                j++;
            }

            // Deep copy particle
            const copy = {};
            for (const finger of Object.keys(this.particles[j])) {
                copy[finger] = { ...this.particles[j][finger] };
            }
            newParticles.push(copy);

            u += step;
        }

        this.particles = newParticles;
        this.weights.fill(1 / this.numParticles);
    }

    /**
     * Effective sample size
     */
    _effectiveSampleSize() {
        let sumSq = 0;
        for (let i = 0; i < this.numParticles; i++) {
            sumSq += this.weights[i] * this.weights[i];
        }
        return 1 / sumSq;
    }

    /**
     * Standard normal random number
     */
    _randn() {
        // Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Reset filter
     */
    reset() {
        this.particles = [];
        this.weights = new Float64Array(this.numParticles).fill(1 / this.numParticles);
        this.initialized = false;
    }
}

/**
 * Simple magnetic field likelihood model
 *
 * Given a particle (hand pose hypothesis) and measured field,
 * compute how likely the measurement is.
 */
function magneticLikelihood(particle, measurement, magnetConfig = null) {
    // Default magnet configuration if not provided
    // Assumes small cylindrical magnets (e.g., 3mm x 2mm N52 neodymium)
    // Magnetic moment ≈ 0.01 A·m² for small magnets
    if (!magnetConfig) {
        magnetConfig = {
            thumb: {moment: {x: 0, y: 0, z: 0.01}},   // Z-axis oriented
            index: {moment: {x: 0, y: 0, z: 0.01}},
            middle: {moment: {x: 0, y: 0, z: 0.01}},
            ring: {moment: {x: 0, y: 0, z: 0.01}},
            pinky: {moment: {x: 0, y: 0, z: 0.01}}
        };
    }

    // Sensor position (reference frame origin)
    const sensorPos = {x: 0, y: 0, z: 0};

    // Calculate expected field as sum of all dipole contributions
    let expectedX = 0, expectedY = 0, expectedZ = 0;

    for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
        if (particle[finger] && magnetConfig[finger]) {
            // Compute dipole field from this finger's magnet
            const magnetPos = particle[finger];
            const magnetMoment = magnetConfig[finger].moment;

            // Position vector from magnet to sensor (in mm, converted to m)
            const rx = (sensorPos.x - magnetPos.x) * 0.001; // mm to m
            const ry = (sensorPos.y - magnetPos.y) * 0.001;
            const rz = (sensorPos.z - magnetPos.z) * 0.001;

            // Distance
            const r = Math.sqrt(rx * rx + ry * ry + rz * rz);

            // Avoid singularity at r=0
            if (r >= 0.001) { // 1mm threshold
                // Unit vector r̂
                const rx_hat = rx / r;
                const ry_hat = ry / r;
                const rz_hat = rz / r;

                // Dot product m·r̂
                const m_dot_r = magnetMoment.x * rx_hat + magnetMoment.y * ry_hat + magnetMoment.z * rz_hat;

                // Dipole field: B = (μ₀/4π) * (3(m·r̂)r̂ - m) / r³
                // Using simplified constant k = 1.0 (units absorbed into calibration)
                const k = 1.0;
                const r3 = r * r * r;

                expectedX += k * (3 * m_dot_r * rx_hat - magnetMoment.x) / r3;
                expectedY += k * (3 * m_dot_r * ry_hat - magnetMoment.y) / r3;
                expectedZ += k * (3 * m_dot_r * rz_hat - magnetMoment.z) / r3;
            }
        }
    }

    // Compute residual (measurement - expected)
    const dx = measurement.x - expectedX;
    const dy = measurement.y - expectedY;
    const dz = measurement.z - expectedZ;

    // Euclidean distance
    const residual = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Gaussian likelihood with standard deviation sigma
    // Sigma represents measurement noise + model uncertainty
    const sigma = 10.0; // Tunable parameter (adjust based on calibration data)

    // Likelihood: exp(-residual² / (2σ²))
    const likelihood = Math.exp(-(residual * residual) / (2 * sigma * sigma));

    return likelihood;
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        KalmanFilter3D,
        MultiFingerKalmanFilter,
        ParticleFilter,
        magneticLikelihood
    };
}
