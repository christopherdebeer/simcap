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
    gyroInDegrees?: boolean,
    applyHardIron?: boolean
  ): { expected: Vector3; residual: Vector3 } | null;

  initFromAccelerometer(ax: number, ay: number, az: number): void;
  updateGyroBias(gx: number, gy: number, gz: number, isStationary: boolean): void;
  setMagTrust(trust: number): void;
  setGeomagneticReference(ref: { horizontal: number; vertical: number; declination: number }): void;
  getMagResidual(): Vector3 | null;
  getMagResidualMagnitude(): number;
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
  update(input: { x: number; y: number; z: number }): { x: number; y: number; z: number };
  reset(): void;
}

interface MotionDetectorOptions {
  accelThreshold?: number;
  gyroThreshold?: number;
  windowSize?: number;
}

interface MotionDetectorResult {
  isStationary: boolean;
  isMoving: boolean;
  accelStd: number;
  gyroStd: number;
}

declare class MotionDetector {
  constructor(options?: MotionDetectorOptions);
  update(ax: number, ay: number, az: number, gx: number, gy: number, gz: number): MotionDetectorResult;
  getState(): MotionDetectorResult;
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

// ===== gambit-client.js =====

interface GambitClientOptions {
  debug?: boolean;
  autoKeepalive?: boolean;
}

interface GambitFirmwareInfo {
  name: string;
  version: string;
}

interface GambitCompatibilityResult {
  compatible: boolean;
  reason?: string;
}

interface GambitClient {
  connected: boolean;

  connect(): Promise<void>;
  disconnect(): void;

  on(event: 'data', callback: (data: any) => void): void;
  on(event: 'firmware', callback: (info: GambitFirmwareInfo) => void): void;
  on(event: 'disconnect', callback: () => void): void;
  on(event: 'error', callback: (error: Error) => void): void;
  off(event: string, callback: (...args: any[]) => void): void;

  startStreaming(): void;
  stopStreaming(): void;

  collectSamples(count: number): Promise<any[]>;
  checkCompatibility(minVersion: string): GambitCompatibilityResult;
}

declare class GambitClient implements GambitClient {
  constructor(options?: GambitClientOptions);
}

// ===== Three.js =====

// THREE is loaded as a script tag globally
declare const THREE: typeof import('three');

