/**
 * Type declarations for orientation-calibration.js
 */

export interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

export interface EulerAngles {
  roll: number;
  pitch: number;
  yaw: number;
}

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface SensorData {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  mx: number;
  my: number;
  mz: number;
}

export interface AHRSOutput {
  quaternion: Quaternion;
  euler: EulerAngles;
}

export interface RenderState {
  rotation: Vector3;
  position: Vector3;
}

export interface UserAnswers {
  [key: string]: string;
}

export interface MappingConfig {
  [key: string]: any;
}

export interface Observation {
  poseId: string;
  timestamp: number;
  sensorData: SensorData;
  ahrsOutput: AHRSOutput;
  renderState: RenderState;
  userAnswers: UserAnswers;
  mappingConfig: MappingConfig;
}

export interface DiagnosticReport {
  summary: string;
  details: any[];
  recommendations: string[];
}

export interface CalibrationStep {
  id: string;
  name: string;
  description: string;
  pose: string;
  questions: string[];
}

export const ANSWER_OPTIONS: {
  YES: string;
  NO: string;
  PARTIAL: string;
  SKIP: string;
};

export const COUPLING_TYPES: {
  CORRECT: string;
  INVERTED: string;
  SWAPPED: string;
  UNKNOWN: string;
};

export const REFERENCE_POSES: {
  [key: string]: {
    name: string;
    description: string;
    expectedOrientation: EulerAngles;
  };
};

export function createObservation(
  poseId: string,
  sensorData: SensorData,
  ahrsOutput: AHRSOutput,
  renderState: RenderState,
  userAnswers: UserAnswers,
  mappingConfig: MappingConfig,
  baselineAhrs?: AHRSOutput | null
): Observation;

export function quaternionToEuler(q: Quaternion, order?: string): EulerAngles;

export function testEulerOrders(
  quaternion: Quaternion,
  expectedAngles: EulerAngles
): { order: string; angles: EulerAngles; error: number }[];

export function generateDiagnosticReport(observation: Observation): DiagnosticReport;

export function getCalibrationSequence(): CalibrationStep[];

export class ObservationStore {
  constructor();
  add(observation: Observation): void;
  getAll(): Observation[];
  clear(): void;
  exportJSON(): string;
  importJSON(json: string): void;
}
