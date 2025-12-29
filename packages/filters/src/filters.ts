/**
 * Extended Filtering for Magnetic Finger Tracking
 *
 * Provides:
 * - IMU Sensor Fusion (Madgwick AHRS for orientation estimation)
 * - Multi-dimensional Kalman Filter (for 3D position/velocity tracking)
 * - Extended Kalman Filter (for non-linear magnetic field model)
 * - Particle Filter (for multi-hypothesis finger tracking)
 *
 * Based on analysis: magnetic-finger-tracking-analysis.md
 */

// Import canonical types from core
import type {
  Vector3,
  Quaternion,
  EulerAngles,
  FingerName,
  HandPose,
  FingerTrackingState,
  GeomagneticReference,
  MotionDetectorState,
} from '@core/types';

// Re-export core types for consumers of this module
export type {
  Vector3,
  Quaternion,
  EulerAngles,
  FingerName,
  HandPose,
  FingerTrackingState,
  GeomagneticReference,
  MotionDetectorState,
} from '@core/types';

// ===== Filter-Specific Type Definitions =====

export interface MadgwickOptions {
  sampleFreq?: number;
  beta?: number;
  geomagneticRef?: GeomagneticReference | null;
}

export interface MotionDetectorOptions {
  accelThreshold?: number;
  gyroThreshold?: number;
  windowSize?: number;
}

/** @deprecated Use MotionDetectorState from @core/types */
export interface MotionState {
  isMoving: boolean;
  accelStd: number;
  gyroStd: number;
}

export interface KalmanFilter3DOptions {
  processNoise?: number;
  measurementNoise?: number;
  initialCovariance?: number;
}

export interface ParticleFilterOptions {
  numParticles?: number;
  positionNoise?: number;
  velocityNoise?: number;
  resampleThreshold?: number;
}

/** Particle state for multi-hypothesis hand tracking */
export interface Particle {
  thumb: FingerTrackingState;
  index: FingerTrackingState;
  middle: FingerTrackingState;
  ring: FingerTrackingState;
  pinky: FingerTrackingState;
}

/** Magnet configuration for magnetic field model */
export interface MagnetConfig {
  thumb?: { moment: Vector3 };
  index?: { moment: Vector3 };
  middle?: { moment: Vector3 };
  ring?: { moment: Vector3 };
  pinky?: { moment: Vector3 };
}

const FINGER_NAMES: FingerName[] = ['thumb', 'index', 'middle', 'ring', 'pinky'];

/**
 * Madgwick AHRS (Attitude and Heading Reference System)
 *
 * Estimates device orientation from accelerometer and gyroscope data.
 * Uses gradient descent optimization to fuse sensor readings into a quaternion.
 */
export class MadgwickAHRS {
  private sampleFreq: number;
  private beta: number;
  private q: Quaternion;
  private gyroBias: Vector3;
  private biasAlpha: number;
  private geomagneticRef: GeomagneticReference | null;
  private magRefNormalized: Vector3 | null;
  private magTrust: number;
  private hardIron: Vector3;
  private _lastMagExpected: Vector3 | null;
  private _lastMagResidual: Vector3 | null;

  constructor(options: MadgwickOptions = {}) {
    this.sampleFreq = options.sampleFreq ?? 50;
    this.beta = options.beta ?? 0.1;
    this.q = { w: 1, x: 0, y: 0, z: 0 };
    this.gyroBias = { x: 0, y: 0, z: 0 };
    this.biasAlpha = 0.2;  // Fast convergence: 50 samples → >99% converged
    this.geomagneticRef = options.geomagneticRef ?? null;
    this.magRefNormalized = null;
    this.magTrust = 1.0;
    this.hardIron = { x: 0, y: 0, z: 0 };
    this._lastMagExpected = null;
    this._lastMagResidual = null;
  }

  update(
    ax: number, ay: number, az: number,
    gx: number, gy: number, gz: number,
    dt: number | null = null,
    gyroInDegrees: boolean = true
  ): Quaternion {
    const deltaT = dt || (1.0 / this.sampleFreq);

    if (gyroInDegrees) {
      gx = gx * Math.PI / 180;
      gy = gy * Math.PI / 180;
      gz = gz * Math.PI / 180;
    }

    gx -= this.gyroBias.x;
    gy -= this.gyroBias.y;
    gz -= this.gyroBias.z;

    let { w: q0, x: q1, y: q2, z: q3 } = this.q;

    const qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
    const qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
    const qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
    const qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

    const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);
    if (accelNorm > 0.01) {
      const recipNorm = 1.0 / accelNorm;
      ax *= recipNorm;
      ay *= recipNorm;
      az *= recipNorm;

      const _2q0 = 2 * q0;
      const _2q1 = 2 * q1;
      const _2q2 = 2 * q2;
      const _2q3 = 2 * q3;
      const _4q0 = 4 * q0;
      const _4q1 = 4 * q1;
      const _4q2 = 4 * q2;
      const _8q1 = 8 * q1;
      const _8q2 = 8 * q2;
      const q0q0 = q0 * q0;
      const q1q1 = q1 * q1;
      const q2q2 = q2 * q2;
      const q3q3 = q3 * q3;

      let s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
      let s1 = _4q1 * q3q3 - _2q3 * ax + 4 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
      let s2 = 4 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
      let s3 = 4 * q1q1 * q3 - _2q1 * ax + 4 * q2q2 * q3 - _2q2 * ay;

      const sNorm = 1.0 / Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
      s0 *= sNorm;
      s1 *= sNorm;
      s2 *= sNorm;
      s3 *= sNorm;

      q0 += (qDot1 - this.beta * s0) * deltaT;
      q1 += (qDot2 - this.beta * s1) * deltaT;
      q2 += (qDot3 - this.beta * s2) * deltaT;
      q3 += (qDot4 - this.beta * s3) * deltaT;
    } else {
      q0 += qDot1 * deltaT;
      q1 += qDot2 * deltaT;
      q2 += qDot3 * deltaT;
      q3 += qDot4 * deltaT;
    }

    const qNorm = 1.0 / Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    this.q = {
      w: q0 * qNorm,
      x: q1 * qNorm,
      y: q2 * qNorm,
      z: q3 * qNorm
    };

    return this.q;
  }

  updateGyroBias(gx: number, gy: number, gz: number, gyroInDegrees: boolean = true): void {
    if (gyroInDegrees) {
      gx = gx * Math.PI / 180;
      gy = gy * Math.PI / 180;
      gz = gz * Math.PI / 180;
    }

    this.gyroBias.x += this.biasAlpha * (gx - this.gyroBias.x);
    this.gyroBias.y += this.biasAlpha * (gy - this.gyroBias.y);
    this.gyroBias.z += this.biasAlpha * (gz - this.gyroBias.z);
  }

  getQuaternion(): Quaternion {
    return { ...this.q };
  }

  getEulerAngles(): EulerAngles {
    const { w, x, y, z } = this.q;

    const sinr_cosp = 2 * (w * x + y * z);
    const cosr_cosp = 1 - 2 * (x * x + y * y);
    const roll = Math.atan2(sinr_cosp, cosr_cosp);

    const sinp = 2 * (w * y - z * x);
    let pitch: number;
    if (Math.abs(sinp) >= 1) {
      pitch = Math.sign(sinp) * Math.PI / 2;
    } else {
      pitch = Math.asin(sinp);
    }

    const siny_cosp = 2 * (w * z + x * y);
    const cosy_cosp = 1 - 2 * (y * y + z * z);
    const yaw = Math.atan2(siny_cosp, cosy_cosp);

    return {
      roll: roll * 180 / Math.PI,
      pitch: pitch * 180 / Math.PI,
      yaw: yaw * 180 / Math.PI
    };
  }

  getRotationMatrix(): number[][] {
    const { w, x, y, z } = this.q;

    return [
      [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
      [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
      [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ];
  }

  transformToDeviceFrame(v: Vector3): Vector3 {
    const R = this.getRotationMatrix();
    return {
      x: R[0][0] * v.x + R[0][1] * v.y + R[0][2] * v.z,
      y: R[1][0] * v.x + R[1][1] * v.y + R[1][2] * v.z,
      z: R[2][0] * v.x + R[2][1] * v.y + R[2][2] * v.z
    };
  }

  reset(): void {
    this.q = { w: 1, x: 0, y: 0, z: 0 };
    this.gyroBias = { x: 0, y: 0, z: 0 };
  }

  initFromAccelerometer(ax: number, ay: number, az: number): void {
    const norm = Math.sqrt(ax * ax + ay * ay + az * az);
    if (norm < 0.01) return;

    ax /= norm;
    ay /= norm;
    az /= norm;

    const roll = Math.atan2(ay, az);
    const pitch = Math.atan2(-ax, Math.sqrt(ay * ay + az * az));

    const cr = Math.cos(roll / 2);
    const sr = Math.sin(roll / 2);
    const cp = Math.cos(pitch / 2);
    const sp = Math.sin(pitch / 2);

    this.q = {
      w: cr * cp,
      x: sr * cp,
      y: cr * sp,
      z: -sr * sp
    };
  }

  setGeomagneticReference(geoRef: GeomagneticReference): void {
    this.geomagneticRef = geoRef;

    if (geoRef) {
      const h = geoRef.horizontal;
      const v = geoRef.vertical;
      const mag = Math.sqrt(h * h + v * v);
      this.magRefNormalized = {
        x: h / mag,
        y: 0,
        z: v / mag
      };
    }
  }

  setHardIronOffset(offset: Vector3): void {
    this.hardIron = { ...offset };
  }

  setMagTrust(trust: number): void {
    this.magTrust = Math.max(0, Math.min(1, trust));
  }

  updateWithMag(
    ax: number, ay: number, az: number,
    gx: number, gy: number, gz: number,
    mx: number, my: number, mz: number,
    dt: number | null = null,
    gyroInDegrees: boolean = true,
    applyHardIron: boolean = true
  ): Quaternion {
    const magNorm = Math.sqrt(mx * mx + my * my + mz * mz);
    if (this.magTrust < 0.01 || magNorm < 0.01) {
      return this.update(ax, ay, az, gx, gy, gz, dt, gyroInDegrees);
    }

    const deltaT = dt || (1.0 / this.sampleFreq);

    if (gyroInDegrees) {
      gx = gx * Math.PI / 180;
      gy = gy * Math.PI / 180;
      gz = gz * Math.PI / 180;
    }

    gx -= this.gyroBias.x;
    gy -= this.gyroBias.y;
    gz -= this.gyroBias.z;

    if (applyHardIron) {
      mx -= this.hardIron.x;
      my -= this.hardIron.y;
      mz -= this.hardIron.z;
    }

    let { w: q0, x: q1, y: q2, z: q3 } = this.q;

    const qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
    const qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
    const qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
    const qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

    const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);
    if (accelNorm < 0.01) {
      q0 += qDot1 * deltaT;
      q1 += qDot2 * deltaT;
      q2 += qDot3 * deltaT;
      q3 += qDot4 * deltaT;
    } else {
      const recipAccelNorm = 1.0 / accelNorm;
      ax *= recipAccelNorm;
      ay *= recipAccelNorm;
      az *= recipAccelNorm;

      const recipMagNorm = 1.0 / magNorm;
      mx *= recipMagNorm;
      my *= recipMagNorm;
      mz *= recipMagNorm;

      const _2q0mx = 2 * q0 * mx;
      const _2q0my = 2 * q0 * my;
      const _2q0mz = 2 * q0 * mz;
      const _2q1mx = 2 * q1 * mx;
      const _2q0 = 2 * q0;
      const _2q1 = 2 * q1;
      const _2q2 = 2 * q2;
      const _2q3 = 2 * q3;
      const _2q0q2 = 2 * q0 * q2;
      const _2q2q3 = 2 * q2 * q3;
      const q0q0 = q0 * q0;
      const q0q1 = q0 * q1;
      const q0q2 = q0 * q2;
      const q0q3 = q0 * q3;
      const q1q1 = q1 * q1;
      const q1q2 = q1 * q2;
      const q1q3 = q1 * q3;
      const q2q2 = q2 * q2;
      const q2q3 = q2 * q3;
      const q3q3 = q3 * q3;

      const hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3;
      const hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3;
      const _2bx = Math.sqrt(hx * hx + hy * hy);
      const _2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3;
      const _4bx = 2 * _2bx;
      const _4bz = 2 * _2bz;

      let s0 = -_2q2 * (2 * q1q3 - _2q0q2 - ax) + _2q1 * (2 * q0q1 + _2q2q3 - ay) - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
      let s1 = _2q3 * (2 * q1q3 - _2q0q2 - ax) + _2q0 * (2 * q0q1 + _2q2q3 - ay) - 4 * q1 * (1 - 2 * q1q1 - 2 * q2q2 - az) + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
      let s2 = -_2q0 * (2 * q1q3 - _2q0q2 - ax) + _2q3 * (2 * q0q1 + _2q2q3 - ay) - 4 * q2 * (1 - 2 * q1q1 - 2 * q2q2 - az) + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
      let s3 = _2q1 * (2 * q1q3 - _2q0q2 - ax) + _2q2 * (2 * q0q1 + _2q2q3 - ay) + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);

      const sNorm = 1.0 / Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
      s0 *= sNorm;
      s1 *= sNorm;
      s2 *= sNorm;
      s3 *= sNorm;

      const effectiveBeta = this.beta * (1.0 + this.magTrust);

      q0 += (qDot1 - effectiveBeta * s0) * deltaT;
      q1 += (qDot2 - effectiveBeta * s1) * deltaT;
      q2 += (qDot3 - effectiveBeta * s2) * deltaT;
      q3 += (qDot4 - effectiveBeta * s3) * deltaT;
    }

    const qNorm = 1.0 / Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    this.q = {
      w: q0 * qNorm,
      x: q1 * qNorm,
      y: q2 * qNorm,
      z: q3 * qNorm
    };

    this._computeMagResidual(mx * magNorm, my * magNorm, mz * magNorm);

    return this.q;
  }

  private _computeMagResidual(mx: number, my: number, mz: number): void {
    if (!this.geomagneticRef) {
      this._lastMagExpected = null;
      this._lastMagResidual = null;
      return;
    }

    const expected = this.transformToDeviceFrame({
      x: this.geomagneticRef.horizontal,
      y: 0,
      z: this.geomagneticRef.vertical
    });

    expected.x += this.hardIron.x;
    expected.y += this.hardIron.y;
    expected.z += this.hardIron.z;

    this._lastMagExpected = expected;

    this._lastMagResidual = {
      x: mx - expected.x,
      y: my - expected.y,
      z: mz - expected.z
    };
  }

  getExpectedMagField(): Vector3 | null {
    return this._lastMagExpected;
  }

  getMagResidual(): Vector3 | null {
    return this._lastMagResidual;
  }

  getMagResidualMagnitude(): number {
    if (!this._lastMagResidual) return 0;
    const r = this._lastMagResidual;
    return Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
  }

  /**
   * Get current gyroscope bias estimate (in radians/sec)
   */
  getGyroBias(): Vector3 {
    return { ...this.gyroBias };
  }

  /**
   * Get current gyroscope bias estimate (in degrees/sec)
   */
  getGyroBiasDegrees(): Vector3 {
    return {
      x: this.gyroBias.x * 180 / Math.PI,
      y: this.gyroBias.y * 180 / Math.PI,
      z: this.gyroBias.z * 180 / Math.PI
    };
  }

  /**
   * Set gyroscope bias directly (in radians/sec)
   */
  setGyroBias(bias: Vector3): void {
    this.gyroBias = { ...bias };
  }

  /**
   * Set gyroscope bias directly (in degrees/sec)
   */
  setGyroBiasDegrees(bias: Vector3): void {
    this.gyroBias = {
      x: bias.x * Math.PI / 180,
      y: bias.y * Math.PI / 180,
      z: bias.z * Math.PI / 180
    };
  }
}

/**
 * Complementary Filter (Simple Alternative)
 */
export class ComplementaryFilter {
  private alpha: number;
  private roll: number;
  private pitch: number;
  private yaw: number;

  constructor(alpha: number = 0.98) {
    this.alpha = alpha;
    this.roll = 0;
    this.pitch = 0;
    this.yaw = 0;
  }

  update(
    ax: number, ay: number, az: number,
    gx: number, gy: number, gz: number,
    dt: number,
    gyroInDegrees: boolean = true
  ): EulerAngles {
    if (!gyroInDegrees) {
      gx = gx * 180 / Math.PI;
      gy = gy * 180 / Math.PI;
      gz = gz * 180 / Math.PI;
    }

    const accelRoll = Math.atan2(ay, az) * 180 / Math.PI;
    const accelPitch = Math.atan2(-ax, Math.sqrt(ay * ay + az * az)) * 180 / Math.PI;

    this.roll = this.alpha * (this.roll + gx * dt) + (1 - this.alpha) * accelRoll;
    this.pitch = this.alpha * (this.pitch + gy * dt) + (1 - this.alpha) * accelPitch;
    this.yaw += gz * dt;

    return this.getEulerAngles();
  }

  getEulerAngles(): EulerAngles {
    return {
      roll: this.roll,
      pitch: this.pitch,
      yaw: this.yaw
    };
  }

  getQuaternion(): Quaternion {
    const cr = Math.cos(this.roll * Math.PI / 360);
    const sr = Math.sin(this.roll * Math.PI / 360);
    const cp = Math.cos(this.pitch * Math.PI / 360);
    const sp = Math.sin(this.pitch * Math.PI / 360);
    const cy = Math.cos(this.yaw * Math.PI / 360);
    const sy = Math.sin(this.yaw * Math.PI / 360);

    return {
      w: cr * cp * cy + sr * sp * sy,
      x: sr * cp * cy - cr * sp * sy,
      y: cr * sp * cy + sr * cp * sy,
      z: cr * cp * sy - sr * sp * cy
    };
  }

  reset(): void {
    this.roll = 0;
    this.pitch = 0;
    this.yaw = 0;
  }
}

/**
 * Motion Detector
 */
export class MotionDetector {
  private accelThreshold: number;
  private gyroThreshold: number;
  private windowSize: number;
  private recentAccel: number[];
  private recentGyro: number[];
  private isMoving: boolean;
  private accelStd: number;
  private gyroStd: number;

  constructor(options: MotionDetectorOptions = {}) {
    this.accelThreshold = options.accelThreshold ?? 2000;
    this.gyroThreshold = options.gyroThreshold ?? 500;
    this.windowSize = options.windowSize ?? 10;
    this.recentAccel = [];
    this.recentGyro = [];
    this.isMoving = false;
    this.accelStd = 0;
    this.gyroStd = 0;
  }

  update(ax: number, ay: number, az: number, gx: number, gy: number, gz: number): MotionState {
    const accelMag = Math.sqrt(ax*ax + ay*ay + az*az);
    const gyroMag = Math.sqrt(gx*gx + gy*gy + gz*gz);

    this.recentAccel.push(accelMag);
    this.recentGyro.push(gyroMag);

    if (this.recentAccel.length > this.windowSize) {
      this.recentAccel.shift();
      this.recentGyro.shift();
    }

    if (this.recentAccel.length < this.windowSize / 2) {
      return this.getState();
    }

    this.accelStd = this._std(this.recentAccel);
    this.gyroStd = this._std(this.recentGyro);

    this.isMoving = (this.accelStd > this.accelThreshold) ||
                   (this.gyroStd > this.gyroThreshold);

    return this.getState();
  }

  getState(): MotionState {
    return {
      isMoving: this.isMoving,
      accelStd: this.accelStd,
      gyroStd: this.gyroStd
    };
  }

  reset(): void {
    this.recentAccel = [];
    this.recentGyro = [];
    this.isMoving = false;
    this.accelStd = 0;
    this.gyroStd = 0;
  }

  private _std(arr: number[]): number {
    if (arr.length === 0) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((a, b) => a + (b - mean) * (b - mean), 0) / arr.length;
    return Math.sqrt(variance);
  }
}

/**
 * Multi-dimensional Kalman Filter with Velocity
 * Tracks state vector with position and velocity for each dimension.
 * State: [x, y, z, vx, vy, vz]
 */
export class KalmanFilter6D {
  private stateDim: number;
  private measDim: number;
  private state: Float64Array;
  private P: Float64Array;
  private Q: number;
  private R: number;
  private dt: number;
  private initialized: boolean;

  constructor(options: KalmanFilter3DOptions = {}) {
    this.stateDim = 6;
    this.measDim = 3;
    this.state = new Float64Array(6);
    this.P = new Float64Array(36);
    for (let i = 0; i < 6; i++) {
      this.P[i * 6 + i] = options.initialCovariance ?? 100;
    }
    this.Q = options.processNoise ?? 1.0;
    this.R = options.measurementNoise ?? 1.0;
    this.dt = 0.02;
    this.initialized = false;
  }

  private _getF(dt: number): Float64Array {
    return new Float64Array([
      1, 0, 0, dt, 0, 0,
      0, 1, 0, 0, dt, 0,
      0, 0, 1, 0, 0, dt,
      0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 1
    ]);
  }

  private _getH(): Float64Array {
    return new Float64Array([
      1, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0
    ]);
  }

  private _matMul(A: Float64Array, B: Float64Array, rowsA: number, colsA: number, colsB: number): Float64Array {
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

  private _transpose(A: Float64Array, rows: number, cols: number): Float64Array {
    const AT = new Float64Array(cols * rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        AT[j * rows + i] = A[i * cols + j];
      }
    }
    return AT;
  }

  private _matSub(A: Float64Array, B: Float64Array, size: number): Float64Array {
    const C = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      C[i] = A[i] - B[i];
    }
    return C;
  }

  private _invert3x3(M: Float64Array): Float64Array {
    const det = M[0] * (M[4] * M[8] - M[5] * M[7])
              - M[1] * (M[3] * M[8] - M[5] * M[6])
              + M[2] * (M[3] * M[7] - M[4] * M[6]);

    if (Math.abs(det) < 1e-10) {
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

  initialize(measurement: Vector3): void {
    this.state[0] = measurement.x;
    this.state[1] = measurement.y;
    this.state[2] = measurement.z;
    this.state[3] = 0;
    this.state[4] = 0;
    this.state[5] = 0;
    this.initialized = true;
  }

  predict(dt: number | null = null): Vector3 | null {
    if (!this.initialized) return null;

    const deltaT = dt || this.dt;
    const F = this._getF(deltaT);

    const newState = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      newState[i] = 0;
      for (let j = 0; j < 6; j++) {
        newState[i] += F[i * 6 + j] * this.state[j];
      }
    }
    this.state = newState;

    const FP = this._matMul(F, this.P, 6, 6, 6);
    const FT = this._transpose(F, 6, 6);
    const FPFT = this._matMul(FP, FT, 6, 6, 6);

    for (let i = 0; i < 6; i++) {
      FPFT[i * 6 + i] += this.Q;
    }
    this.P = FPFT;

    return this.getPosition();
  }

  update(measurement: Vector3): Vector3 {
    if (!this.initialized) {
      this.initialize(measurement);
      return this.getPosition();
    }

    const H = this._getH();
    const HT = this._transpose(H, 3, 6);

    const z = new Float64Array([measurement.x, measurement.y, measurement.z]);
    const Hx = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
      Hx[i] = 0;
      for (let j = 0; j < 6; j++) {
        Hx[i] += H[i * 6 + j] * this.state[j];
      }
    }
    const y = this._matSub(z, Hx, 3);

    const HP = this._matMul(H, this.P, 3, 6, 6);
    const HPHT = this._matMul(HP, HT, 3, 6, 3);
    for (let i = 0; i < 3; i++) {
      HPHT[i * 3 + i] += this.R;
    }

    const PHT = this._matMul(this.P, HT, 6, 6, 3);
    const Sinv = this._invert3x3(HPHT);
    const K = this._matMul(PHT, Sinv, 6, 3, 3);

    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 3; j++) {
        this.state[i] += K[i * 3 + j] * y[j];
      }
    }

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

  getPosition(): Vector3 {
    return {
      x: this.state[0],
      y: this.state[1],
      z: this.state[2]
    };
  }

  getVelocity(): Vector3 {
    return {
      x: this.state[3],
      y: this.state[4],
      z: this.state[5]
    };
  }

  reset(): void {
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
 * Tracks 5 fingers independently, each with 3D position.
 */
export class MultiFingerKalmanFilter {
  private fingers: Record<FingerName, KalmanFilter6D>;

  constructor(options: KalmanFilter3DOptions = {}) {
    this.fingers = {
      thumb: new KalmanFilter6D(options),
      index: new KalmanFilter6D(options),
      middle: new KalmanFilter6D(options),
      ring: new KalmanFilter6D(options),
      pinky: new KalmanFilter6D(options)
    };
  }

  updateFinger(fingerName: FingerName, measurement: Vector3): Vector3 | null {
    if (this.fingers[fingerName]) {
      return this.fingers[fingerName].update(measurement);
    }
    return null;
  }

  predictAll(dt: number | null = null): Record<FingerName, Vector3 | null> {
    const predictions: Record<string, Vector3 | null> = {};
    for (const name of FINGER_NAMES) {
      predictions[name] = this.fingers[name].predict(dt);
    }
    return predictions as Record<FingerName, Vector3 | null>;
  }

  getAllPositions(): Record<FingerName, Vector3> {
    const positions: Record<string, Vector3> = {};
    for (const name of FINGER_NAMES) {
      positions[name] = this.fingers[name].getPosition();
    }
    return positions as Record<FingerName, Vector3>;
  }

  resetAll(): void {
    for (const name of FINGER_NAMES) {
      this.fingers[name].reset();
    }
  }
}

/**
 * Particle Filter for Hand Pose Estimation
 */
export class ParticleFilter {
  private numParticles: number;
  private positionNoise: number;
  private velocityNoise: number;
  private resampleThreshold: number;
  private particles: Particle[];
  private weights: Float64Array;
  private initialized: boolean;

  constructor(options: ParticleFilterOptions = {}) {
    this.numParticles = options.numParticles ?? 500;
    this.positionNoise = options.positionNoise ?? 2.0;
    this.velocityNoise = options.velocityNoise ?? 5.0;
    this.resampleThreshold = options.resampleThreshold ?? 0.5;
    this.particles = [];
    this.weights = new Float64Array(this.numParticles).fill(1 / this.numParticles);
    this.initialized = false;
  }

  initialize(initialPose: Partial<HandPose>): void {
    this.particles = [];

    for (let i = 0; i < this.numParticles; i++) {
      const particle: Particle = {} as Particle;

      for (const finger of FINGER_NAMES) {
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

  predict(dt: number = 0.02): void {
    if (!this.initialized) return;

    for (const particle of this.particles) {
      for (const finger of FINGER_NAMES) {
        const f = particle[finger];
        f.x += f.vx * dt + this._randn() * this.positionNoise;
        f.y += f.vy * dt + this._randn() * this.positionNoise;
        f.z += f.vz * dt + this._randn() * this.positionNoise;
        f.vx += this._randn() * this.velocityNoise * dt;
        f.vy += this._randn() * this.velocityNoise * dt;
        f.vz += this._randn() * this.velocityNoise * dt;
      }
    }
  }

  update(measurement: Vector3, likelihoodFn: (particle: Particle, measurement: Vector3) => number): void {
    if (!this.initialized) return;

    let sumWeights = 0;

    for (let i = 0; i < this.numParticles; i++) {
      const likelihood = likelihoodFn(this.particles[i], measurement);
      this.weights[i] *= likelihood;
      sumWeights += this.weights[i];
    }

    if (sumWeights > 0) {
      for (let i = 0; i < this.numParticles; i++) {
        this.weights[i] /= sumWeights;
      }
    } else {
      this.weights.fill(1 / this.numParticles);
    }

    const nEff = this._effectiveSampleSize();
    if (nEff < this.numParticles * this.resampleThreshold) {
      this._resample();
    }
  }

  estimate(): HandPose | null {
    if (!this.initialized) return null;

    const result: HandPose = {} as HandPose;

    for (const finger of FINGER_NAMES) {
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

  getDiversity(): number {
    if (!this.initialized) return 0;

    const est = this.estimate();
    if (!est) return 0;

    let totalVar = 0;

    for (const finger of FINGER_NAMES) {
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

  private _resample(): void {
    const newParticles: Particle[] = [];
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

      const copy: Particle = {} as Particle;
      for (const finger of FINGER_NAMES) {
        copy[finger] = { ...this.particles[j][finger] };
      }
      newParticles.push(copy);

      u += step;
    }

    this.particles = newParticles;
    this.weights.fill(1 / this.numParticles);
  }

  private _effectiveSampleSize(): number {
    let sumSq = 0;
    for (let i = 0; i < this.numParticles; i++) {
      sumSq += this.weights[i] * this.weights[i];
    }
    return 1 / sumSq;
  }

  private _randn(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  reset(): void {
    this.particles = [];
    this.weights = new Float64Array(this.numParticles).fill(1 / this.numParticles);
    this.initialized = false;
  }
}

/**
 * Simple magnetic field likelihood model
 */
export function magneticLikelihood(
  particle: Particle,
  measurement: Vector3,
  magnetConfig: MagnetConfig | null = null
): number {
  if (!magnetConfig) {
    magnetConfig = {
      thumb: { moment: { x: 0, y: 0, z: 0.01 } },
      index: { moment: { x: 0, y: 0, z: 0.01 } },
      middle: { moment: { x: 0, y: 0, z: 0.01 } },
      ring: { moment: { x: 0, y: 0, z: 0.01 } },
      pinky: { moment: { x: 0, y: 0, z: 0.01 } }
    };
  }

  const sensorPos = { x: 0, y: 0, z: 0 };
  let expectedX = 0, expectedY = 0, expectedZ = 0;

  for (const finger of FINGER_NAMES) {
    if (particle[finger] && magnetConfig[finger]) {
      const magnetPos = particle[finger];
      const magnetMoment = magnetConfig[finger]!.moment;

      const rx = (sensorPos.x - magnetPos.x) * 0.001;
      const ry = (sensorPos.y - magnetPos.y) * 0.001;
      const rz = (sensorPos.z - magnetPos.z) * 0.001;

      const r = Math.sqrt(rx * rx + ry * ry + rz * rz);

      if (r >= 0.001) {
        const rx_hat = rx / r;
        const ry_hat = ry / r;
        const rz_hat = rz / r;

        const m_dot_r = magnetMoment.x * rx_hat + magnetMoment.y * ry_hat + magnetMoment.z * rz_hat;

        const k = 1.0;
        const r3 = r * r * r;

        expectedX += k * (3 * m_dot_r * rx_hat - magnetMoment.x) / r3;
        expectedY += k * (3 * m_dot_r * ry_hat - magnetMoment.y) / r3;
        expectedZ += k * (3 * m_dot_r * rz_hat - magnetMoment.z) / r3;
      }
    }
  }

  const dx = measurement.x - expectedX;
  const dy = measurement.y - expectedY;
  const dz = measurement.z - expectedZ;

  const residual = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const sigma = 10.0;

  return Math.exp(-(residual * residual) / (2 * sigma * sigma));
}
