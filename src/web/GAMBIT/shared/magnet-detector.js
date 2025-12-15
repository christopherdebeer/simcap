/**
 * Magnet Detector
 * 
 * Detects the presence of finger magnets based on magnetometer residual analysis.
 * Uses empirically-derived thresholds from analysis of sessions with/without magnets.
 * 
 * Reference: docs/finger-magnet-detection-analysis.md
 * 
 * @module shared/magnet-detector
 */

/**
 * Detection thresholds in µT (microtesla)
 * 
 * IMPORTANT: These thresholds are for CHANGE detection, not absolute values.
 * The detector establishes a baseline during the first N samples, then
 * detects significant deviations from that baseline.
 * 
 * Based on empirical analysis:
 * - Normal variation: ±10 µT from baseline
 * - Magnet presence: +30-100 µT above baseline
 */
export const MAGNET_THRESHOLDS = {
    NONE: 10,       // < 10 µT above baseline: No magnets detected
    POSSIBLE: 20,   // 10-20 µT above baseline: Possible magnet presence
    LIKELY: 35,     // 20-35 µT above baseline: Magnets likely present
    CONFIRMED: 50   // > 50 µT above baseline: Strong magnet signal confirmed
};

/**
 * Baseline configuration
 */
export const BASELINE_CONFIG = {
    SAMPLES: 100,           // Samples to establish baseline
    EXPECTED_MAG: 50,       // Expected Earth field magnitude (µT)
    MAX_DEVIATION: 30       // Max acceptable deviation from expected (µT)
};

/**
 * Detection status enum
 */
export const MagnetStatus = {
    NONE: 'none',
    POSSIBLE: 'possible',
    LIKELY: 'likely',
    CONFIRMED: 'confirmed'
};

/**
 * MagnetDetector class
 * 
 * Analyzes magnetometer residual magnitude to detect finger magnet presence.
 * Uses a sliding window for stable detection and provides confidence scores.
 */
export class MagnetDetector {
    /**
     * Create a MagnetDetector instance
     * @param {Object} options - Configuration options
     * @param {number} [options.windowSize=50] - Sliding window size (samples)
     * @param {number} [options.baselineSamples=100] - Samples to establish baseline
     * @param {Object} [options.thresholds] - Custom detection thresholds
     * @param {Function} [options.onStatusChange] - Callback when status changes
     */
    constructor(options = {}) {
        this.windowSize = options.windowSize || 50; // 1 second at 50Hz
        this.baselineSamples = options.baselineSamples || BASELINE_CONFIG.SAMPLES;
        this.thresholds = { ...MAGNET_THRESHOLDS, ...options.thresholds };
        this.onStatusChange = options.onStatusChange || null;
        
        // Sliding window of residual magnitudes
        this.residualHistory = [];
        
        // Baseline tracking
        this.baselineEstablished = false;
        this.baselineResidual = 0;
        this.baselineSum = 0;
        this.baselineCount = 0;
        this.baselineMin = Infinity;
        this.baselineMax = -Infinity;
        
        // Current detection state
        this.currentStatus = MagnetStatus.NONE;
        this.currentConfidence = 0;
        this.avgResidual = 0;
        this.maxResidual = 0;
        this.deviationFromBaseline = 0;
        
        // Statistics for session
        this.sampleCount = 0;
        this.sumResidual = 0;
        this.peakResidual = 0;
        
        // Hysteresis to prevent rapid status changes
        this.statusHoldCount = 0;
        this.statusHoldThreshold = 10; // Hold status for at least 10 samples
    }
    
    /**
     * Process a new residual magnitude sample
     * @param {number} residualMagnitude - Residual magnitude in µT
     * @returns {Object} Detection result
     */
    update(residualMagnitude) {
        // Handle null/undefined
        if (residualMagnitude == null || isNaN(residualMagnitude)) {
            return this.getState();
        }
        
        // Update sliding window
        this.residualHistory.push(residualMagnitude);
        if (this.residualHistory.length > this.windowSize) {
            this.residualHistory.shift();
        }
        
        // Update session statistics
        this.sampleCount++;
        this.sumResidual += residualMagnitude;
        this.peakResidual = Math.max(this.peakResidual, residualMagnitude);
        
        // Calculate window statistics
        if (this.residualHistory.length > 0) {
            this.avgResidual = this.residualHistory.reduce((a, b) => a + b, 0) 
                             / this.residualHistory.length;
            this.maxResidual = Math.max(...this.residualHistory);
        }
        
        // Baseline establishment phase
        if (!this.baselineEstablished) {
            this.baselineSum += residualMagnitude;
            this.baselineCount++;
            this.baselineMin = Math.min(this.baselineMin, residualMagnitude);
            this.baselineMax = Math.max(this.baselineMax, residualMagnitude);
            
            if (this.baselineCount >= this.baselineSamples) {
                this.baselineResidual = this.baselineSum / this.baselineCount;
                this.baselineEstablished = true;
                console.log(`[MagnetDetector] Baseline established: ${this.baselineResidual.toFixed(1)} µT (range: ${this.baselineMin.toFixed(1)}-${this.baselineMax.toFixed(1)} µT)`);
            }
            
            // During baseline phase, always report "none"
            return this.getState();
        }
        
        // Calculate deviation from baseline
        this.deviationFromBaseline = this.avgResidual - this.baselineResidual;
        
        // Determine status based on deviation from baseline (not absolute value)
        const newStatus = this._classifyDeviation(this.deviationFromBaseline);
        const newConfidence = this._calculateConfidenceFromDeviation(this.deviationFromBaseline);
        
        // Apply hysteresis
        if (newStatus !== this.currentStatus) {
            this.statusHoldCount++;
            if (this.statusHoldCount >= this.statusHoldThreshold) {
                const oldStatus = this.currentStatus;
                this.currentStatus = newStatus;
                this.statusHoldCount = 0;
                
                // Notify status change
                if (this.onStatusChange) {
                    this.onStatusChange(newStatus, oldStatus, this.getState());
                }
            }
        } else {
            this.statusHoldCount = 0;
        }
        
        this.currentConfidence = newConfidence;
        
        return this.getState();
    }
    
    /**
     * Classify deviation from baseline into status
     * @param {number} deviation - Deviation from baseline in µT
     * @returns {string} Status string
     */
    _classifyDeviation(deviation) {
        // Only positive deviations indicate magnets (magnets add to field, not subtract)
        if (deviation < this.thresholds.NONE) {
            return MagnetStatus.NONE;
        } else if (deviation < this.thresholds.POSSIBLE) {
            return MagnetStatus.POSSIBLE;
        } else if (deviation < this.thresholds.LIKELY) {
            return MagnetStatus.LIKELY;
        } else {
            return MagnetStatus.CONFIRMED;
        }
    }
    
    /**
     * Calculate confidence score from deviation
     * @param {number} deviation - Deviation from baseline in µT
     * @returns {number} Confidence score (0-1)
     */
    _calculateConfidenceFromDeviation(deviation) {
        if (deviation < this.thresholds.NONE) {
            return 0;
        } else if (deviation < this.thresholds.POSSIBLE) {
            return 0.3 * (deviation - this.thresholds.NONE) / 
                   (this.thresholds.POSSIBLE - this.thresholds.NONE);
        } else if (deviation < this.thresholds.LIKELY) {
            return 0.3 + 0.4 * (deviation - this.thresholds.POSSIBLE) / 
                   (this.thresholds.LIKELY - this.thresholds.POSSIBLE);
        } else if (deviation < this.thresholds.CONFIRMED) {
            return 0.7 + 0.2 * (deviation - this.thresholds.LIKELY) / 
                   (this.thresholds.CONFIRMED - this.thresholds.LIKELY);
        } else {
            return Math.min(1.0, 0.9 + 0.1 * (deviation - this.thresholds.CONFIRMED) / 50);
        }
    }
    
    /**
     * Classify residual magnitude into status
     * @param {number} residual - Average residual in µT
     * @returns {string} Status string
     */
    _classifyResidual(residual) {
        if (residual < this.thresholds.NONE) {
            return MagnetStatus.NONE;
        } else if (residual < this.thresholds.POSSIBLE) {
            return MagnetStatus.POSSIBLE;
        } else if (residual < this.thresholds.LIKELY) {
            return MagnetStatus.LIKELY;
        } else {
            return MagnetStatus.CONFIRMED;
        }
    }
    
    /**
     * Calculate confidence score (0-1)
     * @param {number} residual - Average residual in µT
     * @returns {number} Confidence score
     */
    _calculateConfidence(residual) {
        if (residual < this.thresholds.NONE) {
            return 0;
        } else if (residual < this.thresholds.POSSIBLE) {
            // Linear interpolation from 0 to 0.3
            return 0.3 * (residual - this.thresholds.NONE) / 
                   (this.thresholds.POSSIBLE - this.thresholds.NONE);
        } else if (residual < this.thresholds.LIKELY) {
            // Linear interpolation from 0.3 to 0.7
            return 0.3 + 0.4 * (residual - this.thresholds.POSSIBLE) / 
                   (this.thresholds.LIKELY - this.thresholds.POSSIBLE);
        } else if (residual < this.thresholds.CONFIRMED) {
            // Linear interpolation from 0.7 to 0.9
            return 0.7 + 0.2 * (residual - this.thresholds.LIKELY) / 
                   (this.thresholds.CONFIRMED - this.thresholds.LIKELY);
        } else {
            // Asymptotic approach to 1.0
            return Math.min(1.0, 0.9 + 0.1 * (residual - this.thresholds.CONFIRMED) / 50);
        }
    }
    
    /**
     * Get current detection state
     * @returns {Object} Detection state
     */
    getState() {
        return {
            status: this.currentStatus,
            confidence: this.currentConfidence,
            avgResidual: this.avgResidual,
            maxResidual: this.maxResidual,
            detected: this.currentStatus !== MagnetStatus.NONE,
            sampleCount: this.sampleCount,
            // Baseline info
            baselineEstablished: this.baselineEstablished,
            baselineResidual: this.baselineResidual,
            deviationFromBaseline: this.deviationFromBaseline
        };
    }
    
    /**
     * Get session statistics
     * @returns {Object} Session statistics
     */
    getSessionStats() {
        return {
            sampleCount: this.sampleCount,
            meanResidual: this.sampleCount > 0 ? this.sumResidual / this.sampleCount : 0,
            peakResidual: this.peakResidual,
            finalStatus: this.currentStatus,
            finalConfidence: this.currentConfidence
        };
    }
    
    /**
     * Check if magnets are detected
     * @returns {boolean}
     */
    isDetected() {
        return this.currentStatus !== MagnetStatus.NONE;
    }
    
    /**
     * Check if magnets are confirmed (high confidence)
     * @returns {boolean}
     */
    isConfirmed() {
        return this.currentStatus === MagnetStatus.CONFIRMED;
    }
    
    /**
     * Reset detector state
     */
    reset() {
        this.residualHistory = [];
        this.currentStatus = MagnetStatus.NONE;
        this.currentConfidence = 0;
        this.avgResidual = 0;
        this.maxResidual = 0;
        this.deviationFromBaseline = 0;
        this.sampleCount = 0;
        this.sumResidual = 0;
        this.peakResidual = 0;
        this.statusHoldCount = 0;
        
        // Reset baseline
        this.baselineEstablished = false;
        this.baselineResidual = 0;
        this.baselineSum = 0;
        this.baselineCount = 0;
        this.baselineMin = Infinity;
        this.baselineMax = -Infinity;
    }
    
    /**
     * Get human-readable status label
     * @returns {string}
     */
    getStatusLabel() {
        switch (this.currentStatus) {
            case MagnetStatus.NONE:
                return 'No Magnets';
            case MagnetStatus.POSSIBLE:
                return 'Possible';
            case MagnetStatus.LIKELY:
                return 'Likely';
            case MagnetStatus.CONFIRMED:
                return 'Confirmed';
            default:
                return 'Unknown';
        }
    }
    
    /**
     * Get status color for UI
     * @returns {string} CSS color
     */
    getStatusColor() {
        switch (this.currentStatus) {
            case MagnetStatus.NONE:
                return '#888888'; // Gray
            case MagnetStatus.POSSIBLE:
                return '#f0ad4e'; // Orange/warning
            case MagnetStatus.LIKELY:
                return '#5bc0de'; // Blue/info
            case MagnetStatus.CONFIRMED:
                return '#5cb85c'; // Green/success
            default:
                return '#888888';
        }
    }
    
    /**
     * Get status icon for UI
     * @returns {string} Emoji icon
     */
    getStatusIcon() {
        switch (this.currentStatus) {
            case MagnetStatus.NONE:
                return '○';
            case MagnetStatus.POSSIBLE:
                return '◐';
            case MagnetStatus.LIKELY:
                return '◑';
            case MagnetStatus.CONFIRMED:
                return '🧲';
            default:
                return '?';
        }
    }
}

/**
 * Create a MagnetDetector instance with standard configuration
 * @param {Object} options - Configuration options
 * @returns {MagnetDetector}
 */
export function createMagnetDetector(options = {}) {
    return new MagnetDetector(options);
}

// Default export
export default {
    MagnetDetector,
    createMagnetDetector,
    MagnetStatus,
    MAGNET_THRESHOLDS
};
