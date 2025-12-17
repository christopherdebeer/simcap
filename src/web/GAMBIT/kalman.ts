/**
 * KalmanFilter
 * @class
 * @author Wouter Bulten
 * @see {@link http://github.com/wouterbulten/kalmanjs}
 * @version Version: 1.0.0-beta
 * @copyright Copyright 2015-2018 Wouter Bulten
 * @license MIT License
 * @preserve
 */

export interface KalmanFilterOptions {
  /** Process noise (default: 1) */
  R?: number;
  /** Measurement noise (default: 1) */
  Q?: number;
  /** State vector (default: 1) */
  A?: number;
  /** Control vector (default: 0) */
  B?: number;
  /** Measurement vector (default: 1) */
  C?: number;
}

export class KalmanFilter {
  private R: number;  // Process noise
  private Q: number;  // Measurement noise
  private A: number;  // State vector
  private B: number;  // Control vector
  private C: number;  // Measurement vector
  private cov: number;
  private x: number;  // Estimated signal without noise

  /**
   * Create 1-dimensional kalman filter
   */
  constructor(options: KalmanFilterOptions = {}) {
    this.R = options.R ?? 1;
    this.Q = options.Q ?? 1;
    this.A = options.A ?? 1;
    this.B = options.B ?? 0;
    this.C = options.C ?? 1;
    this.cov = NaN;
    this.x = NaN;
  }

  /**
   * Filter a new value
   * @param z Measurement
   * @param u Control (default: 0)
   * @returns Filtered value
   */
  filter(z: number, u: number = 0): number {
    if (isNaN(this.x)) {
      this.x = (1 / this.C) * z;
      this.cov = (1 / this.C) * this.Q * (1 / this.C);
    } else {
      // Compute prediction
      const predX = this.predict(u);
      const predCov = this.uncertainty();
      
      // Kalman gain
      const K = predCov * this.C * (1 / (this.C * predCov * this.C + this.Q));
      
      // Correction
      this.x = predX + K * (z - this.C * predX);
      this.cov = predCov - K * this.C * predCov;
    }

    return this.x;
  }

  /**
   * Predict next value
   * @param u Control (default: 0)
   * @returns Predicted value
   */
  predict(u: number = 0): number {
    return this.A * this.x + this.B * u;
  }

  /**
   * Return uncertainty of filter
   * @returns Uncertainty value
   */
  uncertainty(): number {
    return this.A * this.cov * this.A + this.R;
  }

  /**
   * Return the last filtered measurement
   * @returns Last measurement
   */
  lastMeasurement(): number {
    return this.x;
  }

  /**
   * Set measurement noise Q
   * @param noise New measurement noise value
   */
  setMeasurementNoise(noise: number): void {
    this.Q = noise;
  }

  /**
   * Set the process noise R
   * @param noise New process noise value
   */
  setProcessNoise(noise: number): void {
    this.R = noise;
  }

  /**
   * Reset filter state
   */
  reset(): void {
    this.x = NaN;
    this.cov = NaN;
  }
}

/**
 * 3D Kalman Filter for vector data (e.g., magnetometer)
 */
export class KalmanFilter3D {
  private filterX: KalmanFilter;
  private filterY: KalmanFilter;
  private filterZ: KalmanFilter;

  constructor(options: KalmanFilterOptions = {}) {
    this.filterX = new KalmanFilter(options);
    this.filterY = new KalmanFilter(options);
    this.filterZ = new KalmanFilter(options);
  }

  /**
   * Filter a 3D vector
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @returns Filtered vector
   */
  filter(x: number, y: number, z: number): { x: number; y: number; z: number } {
    return {
      x: this.filterX.filter(x),
      y: this.filterY.filter(y),
      z: this.filterZ.filter(z)
    };
  }

  /**
   * Set measurement noise for all axes
   */
  setMeasurementNoise(noise: number): void {
    this.filterX.setMeasurementNoise(noise);
    this.filterY.setMeasurementNoise(noise);
    this.filterZ.setMeasurementNoise(noise);
  }

  /**
   * Set process noise for all axes
   */
  setProcessNoise(noise: number): void {
    this.filterX.setProcessNoise(noise);
    this.filterY.setProcessNoise(noise);
    this.filterZ.setProcessNoise(noise);
  }

  /**
   * Update filter with a 3D vector object (alternative to filter())
   * @param input Object with x, y, z components
   * @returns Filtered vector
   */
  update(input: { x: number; y: number; z: number }): { x: number; y: number; z: number } {
    return this.filter(input.x, input.y, input.z);
  }

  /**
   * Reset all filter axes
   */
  reset(): void {
    this.filterX.reset();
    this.filterY.reset();
    this.filterZ.reset();
  }
}

// Export as global for backward compatibility with script tags
declare global {
  interface Window {
    KalmanFilter: typeof KalmanFilter;
    KalmanFilter3D: typeof KalmanFilter3D;
  }
}

if (typeof window !== 'undefined') {
  window.KalmanFilter = KalmanFilter;
  window.KalmanFilter3D = KalmanFilter3D;
}

export default KalmanFilter;
