/**
 * Incremental Magnetometer Calibration
 *
 * @deprecated Use UnifiedMagCalibration instead. This class is kept for backwards
 * compatibility but is no longer used in the telemetry pipeline.
 *
 * See: shared/unified-mag-calibration.js
 * See: docs/technical/earth-field-subtraction-investigation.md
 *
 * Builds calibration from streaming data rather than relying on stored values.
 * This approach was validated by analyze_raw_magnetic.py which achieved < 5 µT
 * residual on 12/14 sessions by computing calibration fresh per-session.
 *
 * Key insight: The stored calibration had incorrect Earth field magnitude (6.8 µT
 * vs expected 25-65 µT), causing ~50 µT residuals. Computing from streaming data
 * with orientation compensation yields correct results.
 *
 * @module shared/incremental-calibration
 */

/**
 * IncrementalCalibration class
 *
 * Accumulates magnetometer samples and orientation data to build calibration
 * incrementally. Tracks confidence metrics based on:
 * - Sample count
 * - Orientation coverage (octants visited)
 * - Sphericity (how spherical the data distribution is)
 * - Earth field stability (consistency of estimates)
 */
export class IncrementalCalibration {
    constructor(options = {}) {
        // Configuration
        // NOTE: Hard iron min/max method needs FULL orientation coverage to work
        // Sample count alone is not enough - need good octant coverage
        this.minSamplesHardIron = options.minSamplesHardIron || 200;
        this.minSamplesEarthField = options.minSamplesEarthField || 100;
        this.minOctantCoverage = options.minOctantCoverage || 6; // At least 6 of 8 octants
        this.windowSize = options.windowSize || 500; // Rolling window for estimates
        this.debug = options.debug || false;
        this._debugLogCount = 0;
        
        // Known geomagnetic reference (if provided, use this instead of estimating direction)
        // Format: { horizontal: µT, vertical: µT, declination: degrees }
        this._geomagneticRef = options.geomagneticRef || null;
        
        // Flag to use known reference for Earth field direction
        this._useKnownEarthFieldDirection = options.useKnownEarthFieldDirection !== false; // Default: true

        // Hard iron estimation (min/max method)
        this.hardIron = {
            minX: Infinity, maxX: -Infinity,
            minY: Infinity, maxY: -Infinity,
            minZ: Infinity, maxZ: -Infinity,
            sampleCount: 0
        };

        // Earth field estimation (world frame average)
        this.earthField = {
            sumX: 0, sumY: 0, sumZ: 0,
            sumSqX: 0, sumSqY: 0, sumSqZ: 0,
            sampleCount: 0,
            recentMagnitudes: [] // Track recent magnitude estimates for stability
        };

        // Orientation coverage tracking (8 octants)
        this.octantCounts = new Array(8).fill(0);

        // Rolling buffer for recent samples (for recalibration)
        this.recentSamples = [];
        
        // ALL accumulated samples for Earth field estimation (like Python script)
        // The Python script achieves < 5 µT by using ALL session data, not a rolling window
        this.allSamples = [];
        this.maxAllSamples = 5000; // Cap to prevent memory issues

        // Computed calibration values
        this._hardIronOffset = { x: 0, y: 0, z: 0 };
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;
        
        // Track when hard iron stabilizes (for Earth field computation)
        this._hardIronStabilizedAt = 0; // Sample count when hard iron stopped changing significantly
        this._lastHardIronChange = 0; // Sample count of last significant hard iron change

        // Confidence metrics
        this._hardIronConfidence = 0;
        this._earthFieldConfidence = 0;
        
        // Residual-based confidence (the true measure of calibration quality)
        this._meanResidual = Infinity;
        this._residualConfidence = 0;
        this._recentResiduals = []; // Track recent residual magnitudes
    }

    /**
     * Add a sample to the incremental calibration
     * @param {Object} mag - Magnetometer reading in µT {x, y, z}
     * @param {Object} orientation - Quaternion {w, x, y, z} from IMU fusion
     */
    addSample(mag, orientation) {
        if (!mag || mag.x === undefined) return;

        // Log first sample to verify unit conversion
        if (this.debug && this.hardIron.sampleCount === 0) {
            const rawMag = Math.sqrt(mag.x**2 + mag.y**2 + mag.z**2);
            console.log(`[IncrementalCal] First sample received:
  Mag (µT): [${mag.x.toFixed(1)}, ${mag.y.toFixed(1)}, ${mag.z.toFixed(1)}] |${rawMag.toFixed(1)}| µT
  Orientation: ${orientation ? `w=${orientation.w?.toFixed(3)} x=${orientation.x?.toFixed(3)} y=${orientation.y?.toFixed(3)} z=${orientation.z?.toFixed(3)}` : 'null'}
  Expected magnitude for Edinburgh: ~50 µT`);
        }

        // Update hard iron bounds (min/max)
        this._updateHardIronBounds(mag);

        // If we have orientation, update Earth field estimate
        if (orientation && orientation.w !== undefined) {
            this._updateEarthFieldEstimate(mag, orientation);
        }

        // Store in rolling buffer (for recent residuals)
        this.recentSamples.push({ mag: { ...mag }, orientation: orientation ? { ...orientation } : null });
        if (this.recentSamples.length > this.windowSize) {
            this.recentSamples.shift();
        }
        
        // Store in ALL samples buffer (for Earth field estimation like Python script)
        // The Python script achieves < 5 µT by using ALL session data
        if (orientation && orientation.w !== undefined) {
            this.allSamples.push({ mag: { ...mag }, orientation: { ...orientation } });
            if (this.allSamples.length > this.maxAllSamples) {
                // Remove oldest samples when buffer is full
                this.allSamples.shift();
            }
        }

        // Recompute calibration values
        this._computeCalibration();
        
        // Compute residual for this sample (if we have orientation)
        if (orientation && orientation.w !== undefined && this._earthFieldMagnitude > 0) {
            const residual = this.computeResidual(mag, orientation);
            if (residual) {
                this._recentResiduals.push(residual.magnitude);
                // Keep last 100 residuals
                if (this._recentResiduals.length > 100) {
                    this._recentResiduals.shift();
                }
                this._updateResidualConfidence();
            }
        }
    }

    /**
     * Update hard iron min/max bounds
     * @private
     */
    _updateHardIronBounds(mag) {
        const hi = this.hardIron;

        hi.minX = Math.min(hi.minX, mag.x);
        hi.maxX = Math.max(hi.maxX, mag.x);
        hi.minY = Math.min(hi.minY, mag.y);
        hi.maxY = Math.max(hi.maxY, mag.y);
        hi.minZ = Math.min(hi.minZ, mag.z);
        hi.maxZ = Math.max(hi.maxZ, mag.z);
        hi.sampleCount++;

        // Update octant coverage (for confidence)
        // CRITICAL: Use min/max center for octant calculation, NOT the current hard iron offset
        // The hard iron offset is [0,0,0] until computed, which would bias octant detection
        // Using min/max center gives us a running estimate of the true center
        const centerX = (hi.maxX + hi.minX) / 2;
        const centerY = (hi.maxY + hi.minY) / 2;
        const centerZ = (hi.maxZ + hi.minZ) / 2;
        const cx = mag.x - centerX;
        const cy = mag.y - centerY;
        const cz = mag.z - centerZ;
        const octant = (cx >= 0 ? 4 : 0) + (cy >= 0 ? 2 : 0) + (cz >= 0 ? 1 : 0);
        this.octantCounts[octant]++;
    }

    /**
     * Update Earth field estimate using orientation to transform to world frame
     * @private
     */
    _updateEarthFieldEstimate(mag, q) {
        // NOTE: We no longer accumulate sums here because hard iron offset changes
        // over time. Instead, we store raw samples and recompute in _computeCalibration
        // using the current hard iron offset.

        // Just track sample count for minimum threshold check
        this.earthField.sampleCount++;
    }

    /**
     * Convert quaternion to 3x3 rotation matrix
     * @private
     */
    _quaternionToRotationMatrix(q) {
        const { w, x, y, z } = q;
        return [
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ];
    }

    /**
     * Compute calibration values from accumulated data
     * @private
     */
    _computeCalibration() {
        const hi = this.hardIron;

        // Hard iron offset = center of min/max bounds
        // CRITICAL: Require both sample count AND octant coverage
        // The min/max method only works with complete orientation coverage
        const occupiedOctants = this.octantCounts.filter(c => c > 0).length;
        const hasEnoughSamples = hi.sampleCount >= this.minSamplesHardIron;
        const hasEnoughCoverage = occupiedOctants >= this.minOctantCoverage;
        
        // Check sphericity - ranges should be roughly equal for good calibration
        const rangeX = hi.maxX - hi.minX;
        const rangeY = hi.maxY - hi.minY;
        const rangeZ = hi.maxZ - hi.minZ;
        const minRange = Math.min(rangeX, rangeY, rangeZ);
        const maxRange = Math.max(rangeX, rangeY, rangeZ);
        const sphericity = maxRange > 0 ? minRange / maxRange : 0;
        
        // Require minimum sphericity for initial calibration (50%)
        // The Python script achieves < 5 µT residual without strict sphericity requirements
        // because it uses ALL session data. We should do the same.
        const minSphericity = 0.50;
        const hasGoodSphericity = sphericity >= minSphericity;
        
        // Require minimum range on each axis (at least 60 µT = 60% of expected 100 µT)
        // Lowered from 80 µT because some axes may have physical limitations
        const minAxisRange = 60; // µT
        const hasGoodRanges = rangeX >= minAxisRange && rangeY >= minAxisRange && rangeZ >= minAxisRange;
        const isFirstCalibration = this._hardIronOffset.x === 0 && this._hardIronOffset.y === 0 && this._hardIronOffset.z === 0;
        
        // For first calibration, require BOTH good sphericity AND good ranges
        // After initial calibration, continue updating to improve
        if (hasEnoughSamples && hasEnoughCoverage && ((hasGoodSphericity && hasGoodRanges) || !isFirstCalibration)) {
            const newOffset = {
                x: (hi.maxX + hi.minX) / 2,
                y: (hi.maxY + hi.minY) / 2,
                z: (hi.maxZ + hi.minZ) / 2
            };
            
            // Log when hard iron is first computed or significantly changes
            const offsetChanged = isFirstCalibration || 
                Math.abs(newOffset.x - this._hardIronOffset.x) > 2 ||
                Math.abs(newOffset.y - this._hardIronOffset.y) > 2 ||
                Math.abs(newOffset.z - this._hardIronOffset.z) > 2;
                
            if (offsetChanged && this.debug) {
                console.log(`[IncrementalCal] Hard iron ${isFirstCalibration ? 'computed' : 'updated'} at sample ${hi.sampleCount}:
  Offset: [${newOffset.x.toFixed(1)}, ${newOffset.y.toFixed(1)}, ${newOffset.z.toFixed(1)}] µT
  Ranges: X=${rangeX.toFixed(1)}, Y=${rangeY.toFixed(1)}, Z=${rangeZ.toFixed(1)} µT
  Sphericity: ${(sphericity * 100).toFixed(0)}% (${sphericity > 0.7 ? 'good' : sphericity > 0.5 ? 'moderate' : 'poor - rotate more'})
  Octant coverage: ${occupiedOctants}/8
  Min: [${hi.minX.toFixed(1)}, ${hi.minY.toFixed(1)}, ${hi.minZ.toFixed(1)}] µT
  Max: [${hi.maxX.toFixed(1)}, ${hi.maxY.toFixed(1)}, ${hi.maxZ.toFixed(1)}] µT`);
            }
            
            this._hardIronOffset = newOffset;
        } else if (this.debug && hi.sampleCount % 100 === 0 && hi.sampleCount > 0) {
            // Log progress toward hard iron calibration
            if (!hasEnoughCoverage) {
                console.log(`[IncrementalCal] Hard iron: waiting for coverage (${occupiedOctants}/${this.minOctantCoverage} octants, ${hi.sampleCount} samples)`);
            } else if (!hasGoodSphericity && isFirstCalibration) {
                console.log(`[IncrementalCal] Hard iron: waiting for sphericity (${(sphericity * 100).toFixed(0)}%/${(minSphericity * 100).toFixed(0)}%, ${hi.sampleCount} samples)
  Ranges: X=${rangeX.toFixed(1)}, Y=${rangeY.toFixed(1)}, Z=${rangeZ.toFixed(1)} µT`);
            } else if (!hasGoodRanges && isFirstCalibration) {
                console.log(`[IncrementalCal] Hard iron: waiting for axis ranges (need ${minAxisRange}µT each)
  Ranges: X=${rangeX.toFixed(1)}, Y=${rangeY.toFixed(1)}, Z=${rangeZ.toFixed(1)} µT
  Sphericity: ${(sphericity * 100).toFixed(0)}%`);
            }
        }

        // Recompute Earth field from rolling window using CURRENT hard iron offset
        // This fixes the issue where early samples had wrong hard iron correction
        this._recomputeEarthFieldFromWindow();

        // Compute confidence metrics
        this._computeConfidence();
    }

    /**
     * Recompute Earth field from ALL accumulated samples with current hard iron
     * Using ALL samples (like Python script) instead of rolling window for stable estimates
     * @private
     */
    _recomputeEarthFieldFromWindow() {
        // Use ALL accumulated samples (like Python script) for stable Earth field estimate
        // The Python script achieves < 5 µT by using ALL session data
        const samples = this.allSamples;
        if (samples.length < this.minSamplesEarthField) {
            if (this.debug && samples.length % 50 === 0 && samples.length > 0) {
                console.log(`[IncrementalCal] Earth field: waiting for samples (${samples.length}/${this.minSamplesEarthField})`);
            }
            return;
        }

        // CRITICAL: Don't compute Earth field until hard iron is stable
        // Hard iron needs enough samples AND good coverage to be reliable
        const occupiedOctants = this.octantCounts.filter(c => c > 0).length;
        const hardIronReady = this._hardIronOffset.x !== 0 || this._hardIronOffset.y !== 0 || this._hardIronOffset.z !== 0;
        
        if (!hardIronReady) {
            if (this.debug && this.hardIron.sampleCount % 100 === 0 && this.hardIron.sampleCount > 0) {
                console.log(`[IncrementalCal] Earth field: waiting for hard iron (${occupiedOctants}/${this.minOctantCoverage} octants, ${this.hardIron.sampleCount} samples)`);
            }
            return;
        }

        // All samples in allSamples already have orientation (filtered on insert)
        const validSamples = samples;
        if (validSamples.length < this.minSamplesEarthField) {
            return;
        }

        // Transform each sample to world frame using current hard iron
        const earthVectors = [];
        const magnitudes = [];

        for (const s of validSamples) {
            const mag = s.mag;
            const q = s.orientation;

            // Apply current hard iron correction (in magnetometer frame)
            const corrected = {
                x: mag.x - this._hardIronOffset.x,
                y: mag.y - this._hardIronOffset.y,
                z: mag.z - this._hardIronOffset.z
            };

            // NOTE: Do NOT swap axes here. The Python analysis script (analyze_raw_magnetic.py)
            // achieves < 5 µT residual WITHOUT any axis swapping. The quaternion from IMU fusion
            // is already in a consistent frame with the magnetometer data.

            // Convert quaternion to rotation matrix
            const R = this._quaternionToRotationMatrix(q);

            // Transform to world frame: R.T @ corrected (sensor->world)
            // R transforms world->sensor, so R.T transforms sensor->world
            // R.T means we use columns of R as rows: R[j][i] instead of R[i][j]
            const magWorld = {
                x: R[0][0] * corrected.x + R[1][0] * corrected.y + R[2][0] * corrected.z,
                y: R[0][1] * corrected.x + R[1][1] * corrected.y + R[2][1] * corrected.z,
                z: R[0][2] * corrected.x + R[1][2] * corrected.y + R[2][2] * corrected.z
            };

            earthVectors.push(magWorld);
            magnitudes.push(Math.sqrt(magWorld.x ** 2 + magWorld.y ** 2 + magWorld.z ** 2));
        }

        // Average to get Earth field estimate
        const n = earthVectors.length;
        const prevMagnitude = this._earthFieldMagnitude;
        this._earthFieldWorld = {
            x: earthVectors.reduce((sum, v) => sum + v.x, 0) / n,
            y: earthVectors.reduce((sum, v) => sum + v.y, 0) / n,
            z: earthVectors.reduce((sum, v) => sum + v.z, 0) / n
        };
        
        // CRITICAL FIX: Use average of individual sample magnitudes, NOT magnitude of averaged vector
        // When orientation coverage is incomplete, averaged vectors partially cancel out,
        // giving a smaller magnitude than the actual field. The Python script achieves
        // < 5 µT residual by using full session data with complete coverage.
        // For live streaming with limited coverage, we use the average magnitude instead.
        const avgMagnitude = magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length;
        
        // The direction comes from the averaged vector, but magnitude from average of magnitudes
        const vectorMagnitude = Math.sqrt(
            this._earthFieldWorld.x ** 2 +
            this._earthFieldWorld.y ** 2 +
            this._earthFieldWorld.z ** 2
        );
        
        // CRITICAL: Always estimate Earth field from data, NOT from known geomagnetic reference
        // The known reference assumes NED world frame, but the IMU's world frame may be different.
        // The Python script achieves < 5 µT residual by estimating from data, which automatically
        // accounts for whatever world frame the IMU is using.
        //
        // The averaged vector gives us the Earth field direction in the IMU's world frame.
        // We use the average magnitude to scale it (since averaging can reduce magnitude).
        
        if (vectorMagnitude > 0.1) {
            // Scale the averaged direction vector to have the correct magnitude
            const scale = avgMagnitude / vectorMagnitude;
            this._earthFieldWorld.x *= scale;
            this._earthFieldWorld.y *= scale;
            this._earthFieldWorld.z *= scale;
        }
        this._earthFieldMagnitude = avgMagnitude;
        
        // Log comparison with known reference if available (for debugging only)
        if (this.debug && prevMagnitude === 0 && this._geomagneticRef) {
            const ref = this._geomagneticRef;
            const expectedMag = Math.sqrt((ref.horizontal || 16)**2 + (ref.vertical || 47.8)**2);
            console.log(`[IncrementalCal] Earth field ESTIMATED from data (not using known reference):
  Estimated (IMU world): [${this._earthFieldWorld.x.toFixed(1)}, ${this._earthFieldWorld.y.toFixed(1)}, ${this._earthFieldWorld.z.toFixed(1)}] µT
  Estimated magnitude: ${this._earthFieldMagnitude.toFixed(1)} µT
  Expected magnitude (Edinburgh): ${expectedMag.toFixed(1)} µT
  Note: Direction depends on IMU world frame orientation`);
        }

        // Log when Earth field is first computed
        if (prevMagnitude === 0 && this._earthFieldMagnitude > 0 && this.debug) {
            const avgMag = magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length;
            const minMag = Math.min(...magnitudes);
            const maxMag = Math.max(...magnitudes);
            console.log(`[IncrementalCal] Earth field computed at sample ${this.hardIron.sampleCount}:
  Earth field (world): [${this._earthFieldWorld.x.toFixed(1)}, ${this._earthFieldWorld.y.toFixed(1)}, ${this._earthFieldWorld.z.toFixed(1)}] µT
  Earth field magnitude: ${this._earthFieldMagnitude.toFixed(1)} µT
  Sample magnitudes: avg=${avgMag.toFixed(1)}, min=${minMag.toFixed(1)}, max=${maxMag.toFixed(1)} µT
  Valid samples used: ${n}
  Hard iron offset: [${this._hardIronOffset.x.toFixed(1)}, ${this._hardIronOffset.y.toFixed(1)}, ${this._hardIronOffset.z.toFixed(1)}] µT`);
        }

        // Store recent magnitudes for stability metric
        this.earthField.recentMagnitudes = magnitudes.slice(-100);
    }

    /**
     * Compute confidence metrics
     * @private
     */
    _computeConfidence() {
        // Hard iron confidence based on:
        // 1. Sample count (more = better)
        // 2. Sphericity (should be roughly spherical)
        // 3. Coverage (should cover all octants)

        const hi = this.hardIron;
        if (hi.sampleCount < this.minSamplesHardIron) {
            this._hardIronConfidence = 0;
        } else {
            const rangeX = hi.maxX - hi.minX;
            const rangeY = hi.maxY - hi.minY;
            const rangeZ = hi.maxZ - hi.minZ;

            // Sphericity: min range / max range (1.0 = perfect sphere)
            const minRange = Math.min(rangeX, rangeY, rangeZ);
            const maxRange = Math.max(rangeX, rangeY, rangeZ);
            const sphericity = maxRange > 0 ? minRange / maxRange : 0;

            // Coverage: fraction of octants visited
            const occupiedOctants = this.octantCounts.filter(c => c > 0).length;
            const coverage = occupiedOctants / 8;

            // Sample factor: ramps up to 1.0 at 500 samples
            const sampleFactor = Math.min(1, hi.sampleCount / 500);

            this._hardIronConfidence = sphericity * coverage * sampleFactor;
        }

        // Earth field confidence based on:
        // 1. Sample count in rolling window
        // 2. Stability (low variance in recent magnitude estimates)
        // NOTE: We do NOT validate magnitude against expected Earth field values.
        // The calibration should subtract whatever field we observe, not validate it.

        const ef = this.earthField;
        const validSamples = this.recentSamples.filter(s => s.orientation && s.orientation.w !== undefined);

        if (validSamples.length < this.minSamplesEarthField) {
            this._earthFieldConfidence = 0;
        } else {
            // Sample factor: ramps to 1.0 at 200 valid samples in window
            const sampleFactor = Math.min(1, validSamples.length / 200);

            // Stability: based on variance of recent magnitudes
            // This is the key metric - if estimates are consistent, we're confident
            let stability = 0.5; // Default moderate stability
            if (ef.recentMagnitudes.length >= 10) {
                const mean = ef.recentMagnitudes.reduce((a, b) => a + b, 0) / ef.recentMagnitudes.length;
                if (mean > 0) {
                    const variance = ef.recentMagnitudes.reduce((sum, m) => sum + (m - mean) ** 2, 0) / ef.recentMagnitudes.length;
                    const stdDev = Math.sqrt(variance);
                    // Good stability if coefficient of variation < 10%
                    // CV = stdDev/mean, so stability = 1 - CV*5 (caps at 50% CV)
                    const cv = stdDev / mean;
                    stability = Math.max(0, Math.min(1, 1 - cv * 5));
                }
            }

            // Confidence = sample coverage × stability
            // No magnitude sanity check - we trust whatever field we observe
            this._earthFieldConfidence = sampleFactor * Math.max(0.5, stability);
        }
    }

    /**
     * Update residual-based confidence
     * This is the TRUE measure of calibration quality - how well does our
     * calibration explain the observed data?
     * @private
     */
    _updateResidualConfidence() {
        if (this._recentResiduals.length < 10) {
            this._residualConfidence = 0;
            this._meanResidual = Infinity;
            return;
        }

        // Calculate mean residual magnitude
        const sum = this._recentResiduals.reduce((a, b) => a + b, 0);
        this._meanResidual = sum / this._recentResiduals.length;

        // Confidence based on how small the residual is:
        // - Residual < 2 µT → 100% confidence (excellent calibration)
        // - Residual = 5 µT → 75% confidence (good calibration)
        // - Residual = 10 µT → 50% confidence (moderate calibration)
        // - Residual > 20 µT → 0% confidence (poor calibration)
        //
        // Using exponential decay: confidence = exp(-residual / 10)
        // This gives: 2µT→82%, 5µT→61%, 10µT→37%, 20µT→14%
        //
        // Or linear: confidence = max(0, 1 - residual/20)
        // This gives: 2µT→90%, 5µT→75%, 10µT→50%, 20µT→0%
        
        // Using linear for intuitive interpretation
        this._residualConfidence = Math.max(0, Math.min(1, 1 - this._meanResidual / 20));
    }

    /**
     * Set geomagnetic reference for known Earth field direction
     * @param {Object} ref - { horizontal: µT, vertical: µT, declination: degrees }
     */
    setGeomagneticReference(ref) {
        this._geomagneticRef = ref;
        // Always log this - it's critical for debugging
        if (ref) {
            console.log(`[IncrementalCal] Geomagnetic reference set:
  Horizontal: ${ref.horizontal?.toFixed(1)} µT
  Vertical: ${ref.vertical?.toFixed(1)} µT
  Declination: ${ref.declination?.toFixed(1)}°`);
        } else {
            console.warn('[IncrementalCal] setGeomagneticReference called with null/undefined ref');
        }
    }

    /**
     * Get hard iron offset
     * @returns {Object} {x, y, z} in µT
     */
    getHardIronOffset() {
        return { ...this._hardIronOffset };
    }

    /**
     * Get Earth field in world frame
     * @returns {Object} {x, y, z} in µT
     */
    getEarthFieldWorld() {
        return { ...this._earthFieldWorld };
    }

    /**
     * Get Earth field magnitude
     * @returns {number} Magnitude in µT
     */
    getEarthFieldMagnitude() {
        return this._earthFieldMagnitude;
    }

    /**
     * Get overall confidence (0.0 - 1.0)
     * Now based primarily on RESIDUAL magnitude - the true measure of calibration quality
     * @returns {number}
     */
    getConfidence() {
        // If we have residual data, use it as the primary confidence metric
        // Residual-based confidence is the most meaningful measure
        if (this._recentResiduals.length >= 10) {
            // Weight residual confidence heavily, but require minimum samples
            const sampleFactor = Math.min(1, this.hardIron.sampleCount / 200);
            return this._residualConfidence * sampleFactor;
        }
        
        // Fall back to old method if not enough residual data yet
        return Math.min(this._hardIronConfidence, this._earthFieldConfidence);
    }

    /**
     * Get mean residual magnitude
     * @returns {number} Mean residual in µT (lower is better)
     */
    getMeanResidual() {
        return this._meanResidual;
    }

    /**
     * Get residual-based confidence
     * @returns {number} 0.0 - 1.0 based on how small residuals are
     */
    getResidualConfidence() {
        return this._residualConfidence;
    }

    /**
     * Get detailed confidence breakdown
     * @returns {Object}
     */
    getConfidenceDetails() {
        const hi = this.hardIron;
        const ef = this.earthField;

        return {
            overall: this.getConfidence(),
            residual: {
                confidence: this._residualConfidence,
                meanMagnitude: this._meanResidual,
                sampleCount: this._recentResiduals.length,
                interpretation: this._meanResidual < 5 ? 'excellent' :
                               this._meanResidual < 10 ? 'good' :
                               this._meanResidual < 15 ? 'moderate' : 'poor'
            },
            hardIron: {
                confidence: this._hardIronConfidence,
                sampleCount: hi.sampleCount,
                offset: this._hardIronOffset,
                ranges: {
                    x: hi.maxX - hi.minX,
                    y: hi.maxY - hi.minY,
                    z: hi.maxZ - hi.minZ
                },
                sphericity: this._computeSphericity(),
                coverage: this.octantCounts.filter(c => c > 0).length / 8
            },
            earthField: {
                confidence: this._earthFieldConfidence,
                sampleCount: ef.sampleCount,
                vector: this._earthFieldWorld,
                magnitude: this._earthFieldMagnitude,
                stability: this._computeEarthFieldStability()
            }
        };
    }

    /**
     * Compute sphericity metric
     * @private
     */
    _computeSphericity() {
        const hi = this.hardIron;
        if (hi.sampleCount < 10) return 0;

        const rangeX = hi.maxX - hi.minX;
        const rangeY = hi.maxY - hi.minY;
        const rangeZ = hi.maxZ - hi.minZ;
        const minRange = Math.min(rangeX, rangeY, rangeZ);
        const maxRange = Math.max(rangeX, rangeY, rangeZ);

        return maxRange > 0 ? minRange / maxRange : 0;
    }

    /**
     * Compute Earth field stability metric
     * @private
     */
    _computeEarthFieldStability() {
        const mags = this.earthField.recentMagnitudes;
        if (mags.length < 10) return 0;

        const mean = mags.reduce((a, b) => a + b, 0) / mags.length;
        const variance = mags.reduce((sum, m) => sum + (m - mean) ** 2, 0) / mags.length;
        const stdDev = Math.sqrt(variance);

        return {
            mean,
            stdDev,
            coefficientOfVariation: mean > 0 ? (stdDev / mean) * 100 : 0 // as percentage
        };
    }

    /**
     * Compute residual for a sample using current calibration
     * @param {Object} mag - Magnetometer reading in µT {x, y, z}
     * @param {Object} orientation - Quaternion {w, x, y, z}
     * @returns {Object} {residual: {x, y, z}, magnitude: number}
     */
    computeResidual(mag, orientation) {
        if (!mag || !orientation) return null;

        // Apply hard iron correction (in magnetometer frame)
        const corrected = {
            x: mag.x - this._hardIronOffset.x,
            y: mag.y - this._hardIronOffset.y,
            z: mag.z - this._hardIronOffset.z
        };

        // NOTE: Do NOT swap axes. Matches Python analyze_raw_magnetic.py which achieves < 5 µT residual.

        // Rotate Earth field from world to sensor frame
        // R transforms world->sensor directly
        const R = this._quaternionToRotationMatrix(orientation);
        const earthSensor = {
            x: R[0][0] * this._earthFieldWorld.x + R[0][1] * this._earthFieldWorld.y + R[0][2] * this._earthFieldWorld.z,
            y: R[1][0] * this._earthFieldWorld.x + R[1][1] * this._earthFieldWorld.y + R[1][2] * this._earthFieldWorld.z,
            z: R[2][0] * this._earthFieldWorld.x + R[2][1] * this._earthFieldWorld.y + R[2][2] * this._earthFieldWorld.z
        };

        // Residual = measured - expected (both in sensor frame)
        const residual = {
            x: corrected.x - earthSensor.x,
            y: corrected.y - earthSensor.y,
            z: corrected.z - earthSensor.z
        };

        const magnitude = Math.sqrt(residual.x ** 2 + residual.y ** 2 + residual.z ** 2);

        // Debug logging (every 100 samples)
        if (this.debug && this._debugLogCount % 100 === 0) {
            const correctedMag = Math.sqrt(corrected.x**2 + corrected.y**2 + corrected.z**2);
            const earthSensorMag = Math.sqrt(earthSensor.x**2 + earthSensor.y**2 + earthSensor.z**2);
            console.log(`[IncrementalCal] Sample ${this._debugLogCount}:
  Quaternion: w=${orientation.w.toFixed(3)} x=${orientation.x.toFixed(3)} y=${orientation.y.toFixed(3)} z=${orientation.z.toFixed(3)}
  Corrected mag (sensor): [${corrected.x.toFixed(1)}, ${corrected.y.toFixed(1)}, ${corrected.z.toFixed(1)}] |${correctedMag.toFixed(1)}| µT
  Earth field (world): [${this._earthFieldWorld.x.toFixed(1)}, ${this._earthFieldWorld.y.toFixed(1)}, ${this._earthFieldWorld.z.toFixed(1)}] |${this._earthFieldMagnitude.toFixed(1)}| µT
  Earth field (sensor): [${earthSensor.x.toFixed(1)}, ${earthSensor.y.toFixed(1)}, ${earthSensor.z.toFixed(1)}] |${earthSensorMag.toFixed(1)}| µT
  Residual: [${residual.x.toFixed(1)}, ${residual.y.toFixed(1)}, ${residual.z.toFixed(1)}] |${magnitude.toFixed(1)}| µT`);
        }
        this._debugLogCount++;

        return { residual, magnitude };
    }

    /**
     * Reset calibration state
     */
    reset() {
        this.hardIron = {
            minX: Infinity, maxX: -Infinity,
            minY: Infinity, maxY: -Infinity,
            minZ: Infinity, maxZ: -Infinity,
            sampleCount: 0
        };

        this.earthField = {
            sumX: 0, sumY: 0, sumZ: 0,
            sumSqX: 0, sumSqY: 0, sumSqZ: 0,
            sampleCount: 0,
            recentMagnitudes: []
        };

        this.octantCounts = new Array(8).fill(0);
        this.recentSamples = [];
        this.allSamples = [];

        this._hardIronOffset = { x: 0, y: 0, z: 0 };
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;
        this._hardIronConfidence = 0;
        this._earthFieldConfidence = 0;
        this._meanResidual = Infinity;
        this._residualConfidence = 0;
        this._recentResiduals = [];
    }

    /**
     * Export calibration for saving
     * @returns {Object}
     */
    toJSON() {
        return {
            hardIronOffset: this._hardIronOffset,
            earthField: this._earthFieldWorld,
            earthFieldMagnitude: this._earthFieldMagnitude,
            confidence: {
                overall: this.getConfidence(),
                hardIron: this._hardIronConfidence,
                earthField: this._earthFieldConfidence
            },
            stats: {
                hardIronSamples: this.hardIron.sampleCount,
                earthFieldSamples: this.earthField.sampleCount,
                sphericity: this._computeSphericity(),
                coverage: this.octantCounts.filter(c => c > 0).length / 8,
                meanResidual: this._meanResidual,
                residualConfidence: this._residualConfidence
            },
            timestamp: new Date().toISOString(),
            units: {
                hardIronOffset: 'µT',
                earthField: 'µT',
                earthFieldMagnitude: 'µT'
            }
        };
    }
}

/**
 * Create an IncrementalCalibration instance
 * @param {Object} options - Configuration options
 * @returns {IncrementalCalibration}
 */
export function createIncrementalCalibration(options = {}) {
    return new IncrementalCalibration(options);
}

export default {
    IncrementalCalibration,
    createIncrementalCalibration
};
