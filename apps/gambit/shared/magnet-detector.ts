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

// ===== Type Definitions =====

export type MagnetStatusType = 'none' | 'possible' | 'likely' | 'confirmed';

export interface MagnetThresholds {
  NONE: number;
  POSSIBLE: number;
  LIKELY: number;
  CONFIRMED: number;
}

export interface BaselineConfig {
  SAMPLES: number;
  EXPECTED_MAG: number;
  MAX_DEVIATION: number;
}

export interface MagnetDetectorOptions {
  windowSize?: number;
  baselineSamples?: number;
  thresholds?: Partial<MagnetThresholds>;
  onStatusChange?: ((newStatus: MagnetStatusType, oldStatus: MagnetStatusType, state: MagnetDetectorState) => void) | null;
}

export interface MagnetDetectorState {
  status: MagnetStatusType;
  confidence: number;
  avgResidual: number;
  maxResidual: number;
  detected: boolean;
  sampleCount: number;
  baselineEstablished: boolean;
  baselineResidual: number;
  deviationFromBaseline: number;
}

export interface SessionStats {
  sampleCount: number;
  meanResidual: number;
  peakResidual: number;
  finalStatus: MagnetStatusType;
  finalConfidence: number;
}

// ===== Constants =====

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
export const MAGNET_THRESHOLDS: MagnetThresholds = {
    NONE: 10,       // < 10 µT above baseline: No magnets detected
    POSSIBLE: 20,   // 10-20 µT above baseline: Possible magnet presence
    LIKELY: 35,     // 20-35 µT above baseline: Magnets likely present
    CONFIRMED: 50   // > 50 µT above baseline: Strong magnet signal confirmed
};

/**
 * Baseline configuration
 */
export const BASELINE_CONFIG: BaselineConfig = {
    SAMPLES: 100,           // Samples to establish baseline
    EXPECTED_MAG: 50,       // Expected Earth field magnitude (µT)
    MAX_DEVIATION: 30       // Max acceptable deviation from expected (µT)
};

/**
 * Detection status enum
 */
export const MagnetStatus: Record<string, MagnetStatusType> = {
    NONE: 'none',
    POSSIBLE: 'possible',
    LIKELY: 'likely',
    CONFIRMED: 'confirmed'
} as const;

// ===== MagnetDetector Class =====

/**
 * MagnetDetector class
 *
 * Analyzes magnetometer residual magnitude to detect finger magnet presence.
 * Uses a sliding window for stable detection and provides confidence scores.
 */
export class MagnetDetector {
    private windowSize: number;
    private baselineSamples: number;
    private thresholds: MagnetThresholds;
    private onStatusChange: ((newStatus: MagnetStatusType, oldStatus: MagnetStatusType, state: MagnetDetectorState) => void) | null;

    // Sliding window of residual magnitudes
    private residualHistory: number[] = [];

    // Baseline tracking
    private baselineEstablished: boolean = false;
    private baselineResidual: number = 0;
    private baselineSum: number = 0;
    private baselineCount: number = 0;
    private baselineMin: number = Infinity;
    private baselineMax: number = -Infinity;

    // Current detection state
    private currentStatus: MagnetStatusType = 'none';
    private currentConfidence: number = 0;
    private avgResidual: number = 0;
    private maxResidual: number = 0;
    private deviationFromBaseline: number = 0;

    // Statistics for session
    private sampleCount: number = 0;
    private sumResidual: number = 0;
    private peakResidual: number = 0;

    // Hysteresis to prevent rapid status changes
    private statusHoldCount: number = 0;
    private statusHoldThreshold: number = 10;

    /**
     * Create a MagnetDetector instance
     * @param options - Configuration options
     */
    constructor(options: MagnetDetectorOptions = {}) {
        this.windowSize = options.windowSize || 50; // 1 second at 50Hz
        this.baselineSamples = options.baselineSamples || BASELINE_CONFIG.SAMPLES;
        this.thresholds = { ...MAGNET_THRESHOLDS, ...options.thresholds };
        this.onStatusChange = options.onStatusChange || null;
    }

    /**
     * Process a new residual magnitude sample
     * @param residualMagnitude - Residual magnitude in µT
     * @returns Detection result
     */
    update(residualMagnitude: number): MagnetDetectorState {
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
     */
    private _classifyDeviation(deviation: number): MagnetStatusType {
        // Only positive deviations indicate magnets (magnets add to field, not subtract)
        if (deviation < this.thresholds.NONE) {
            return 'none';
        } else if (deviation < this.thresholds.POSSIBLE) {
            return 'possible';
        } else if (deviation < this.thresholds.LIKELY) {
            return 'likely';
        } else {
            return 'confirmed';
        }
    }

    /**
     * Calculate confidence score from deviation
     */
    private _calculateConfidenceFromDeviation(deviation: number): number {
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
     */
    private _classifyResidual(residual: number): MagnetStatusType {
        if (residual < this.thresholds.NONE) {
            return 'none';
        } else if (residual < this.thresholds.POSSIBLE) {
            return 'possible';
        } else if (residual < this.thresholds.LIKELY) {
            return 'likely';
        } else {
            return 'confirmed';
        }
    }

    /**
     * Calculate confidence score (0-1)
     */
    private _calculateConfidence(residual: number): number {
        if (residual < this.thresholds.NONE) {
            return 0;
        } else if (residual < this.thresholds.POSSIBLE) {
            return 0.3 * (residual - this.thresholds.NONE) /
                   (this.thresholds.POSSIBLE - this.thresholds.NONE);
        } else if (residual < this.thresholds.LIKELY) {
            return 0.3 + 0.4 * (residual - this.thresholds.POSSIBLE) /
                   (this.thresholds.LIKELY - this.thresholds.POSSIBLE);
        } else if (residual < this.thresholds.CONFIRMED) {
            return 0.7 + 0.2 * (residual - this.thresholds.LIKELY) /
                   (this.thresholds.CONFIRMED - this.thresholds.LIKELY);
        } else {
            return Math.min(1.0, 0.9 + 0.1 * (residual - this.thresholds.CONFIRMED) / 50);
        }
    }

    /**
     * Get current detection state
     */
    getState(): MagnetDetectorState {
        return {
            status: this.currentStatus,
            confidence: this.currentConfidence,
            avgResidual: this.avgResidual,
            maxResidual: this.maxResidual,
            detected: this.currentStatus !== 'none',
            sampleCount: this.sampleCount,
            baselineEstablished: this.baselineEstablished,
            baselineResidual: this.baselineResidual,
            deviationFromBaseline: this.deviationFromBaseline
        };
    }

    /**
     * Get session statistics
     */
    getSessionStats(): SessionStats {
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
     */
    isDetected(): boolean {
        return this.currentStatus !== 'none';
    }

    /**
     * Check if magnets are confirmed (high confidence)
     */
    isConfirmed(): boolean {
        return this.currentStatus === 'confirmed';
    }

    /**
     * Reset detector state
     */
    reset(): void {
        this.residualHistory = [];
        this.currentStatus = 'none';
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
     */
    getStatusLabel(): string {
        switch (this.currentStatus) {
            case 'none':
                return 'No Magnets';
            case 'possible':
                return 'Possible';
            case 'likely':
                return 'Likely';
            case 'confirmed':
                return 'Confirmed';
            default:
                return 'Unknown';
        }
    }

    /**
     * Get status color for UI
     */
    getStatusColor(): string {
        switch (this.currentStatus) {
            case 'none':
                return '#888888'; // Gray
            case 'possible':
                return '#f0ad4e'; // Orange/warning
            case 'likely':
                return '#5bc0de'; // Blue/info
            case 'confirmed':
                return '#5cb85c'; // Green/success
            default:
                return '#888888';
        }
    }

    /**
     * Get status icon for UI
     */
    getStatusIcon(): string {
        switch (this.currentStatus) {
            case 'none':
                return '○';
            case 'possible':
                return '◐';
            case 'likely':
                return '◑';
            case 'confirmed':
                return '🧲';
            default:
                return '?';
        }
    }
}

// ===== Factory Function =====

/**
 * Create a MagnetDetector instance with standard configuration
 * @param options - Configuration options
 * @returns MagnetDetector instance
 */
export function createMagnetDetector(options: MagnetDetectorOptions = {}): MagnetDetector {
    return new MagnetDetector(options);
}

// ===== Default Export =====

export default {
    MagnetDetector,
    createMagnetDetector,
    MagnetStatus,
    MAGNET_THRESHOLDS
};
