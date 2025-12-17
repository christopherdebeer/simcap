/**
 * Type declarations for global scripts
 *
 * These declarations provide TypeScript support for legacy global scripts
 * (filters.js, kalman.js, puck.js) without converting them.
 */

// ===== filters.js =====

interface MadgwickOptions {
  sampleFreq?: number;
  beta?: number;
  geomagneticRef?: {
    horizontal: number;
    vertical: number;
    declination: number;
  } | null;
}

interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

interface Vector3 {
  x: number;
  y: number;
  z: number;
}

interface EulerAngles {
  roll: number;
  pitch: number;
  yaw: number;
}

declare class MadgwickAHRS {
  sampleFreq: number;
  beta: number;
  q: Quaternion;
  gyroBias: Vector3;
  magTrust: number;
  hardIron: Vector3;
  _lastMagExpected: Vector3 | null;
  _lastMagResidual: Vector3 | null;

  constructor(options?: MadgwickOptions);

  update(
    ax: number, ay: number, az: number,
    gx: number, gy: number, gz: number,
    dt?: number | null,
    gyroInDegrees?: boolean
  ): void;

  updateWithMag(
    ax: number, ay: number, az: number,
    gx: number, gy: number, gz: number,
    mx: number, my: number, mz: number,
    dt?: number | null,
    gyroInDegrees?: boolean
  ): { expected: Vector3; residual: Vector3 } | null;

  getQuaternion(): Quaternion;
  getEulerAngles(): EulerAngles;
  reset(): void;
}

interface KalmanFilter3DOptions {
  processNoise?: number;
  measurementNoise?: number;
}

declare class KalmanFilter3D {
  constructor(options?: KalmanFilter3DOptions);
  filter(x: number, y: number, z: number): Vector3;
  reset(): void;
}

interface MotionDetectorOptions {
  accelThreshold?: number;
  gyroThreshold?: number;
  windowSize?: number;
}

interface MotionDetectorResult {
  isStationary: boolean;
  accelStd: number;
  gyroStd: number;
}

declare class MotionDetector {
  constructor(options?: MotionDetectorOptions);
  update(ax: number, ay: number, az: number, gx: number, gy: number, gz: number): MotionDetectorResult;
  reset(): void;
}

// ===== kalman.js =====

interface KalmanFilterOptions {
  R?: number;  // Process noise
  Q?: number;  // Measurement noise
}

declare class KalmanFilter {
  constructor(options?: KalmanFilterOptions);
  filter(value: number): number;
  reset(): void;
}

// ===== puck.js =====

interface PuckConnectionCallback {
  (connection: any): void;
}

interface PuckWriteCallback {
  (): void;
}

declare const Puck: {
  debug: number;
  flowControl: boolean;
  chunkSize: number;

  connect(callback: PuckConnectionCallback): Promise<void>;
  write(data: string, callback?: PuckWriteCallback): void;
  eval(expression: string, callback: (result: any) => void): void;
  close(): void;
  isConnected(): boolean;

  setTime(): void;
  getBattery(): Promise<number>;
  LED1: { write: (value: boolean) => void };
  LED2: { write: (value: boolean) => void };
  LED3: { write: (value: boolean) => void };
};

