/**
 * Session data types for SIMCAP
 *
 * These types define the structure of recorded sessions.
 */

import type { TelemetrySample, Quaternion, Vector3 } from './telemetry';

/** Device information */
export interface DeviceInfo {
  id: string;
  firmware: string;
  hardware?: string;
}

/** Finger state for labeling */
export type FingerState = 'extended' | 'flexed' | 'unknown';

/** All finger states */
export interface FingerStates {
  thumb: FingerState;
  index: FingerState;
  middle: FingerState;
  ring: FingerState;
  pinky: FingerState;
}

/** Motion type for labeling */
export type MotionType = 'static' | 'dynamic';

/** Label segment within a session */
export interface LabelSegment {
  startIndex: number;
  endIndex: number;
  pose?: string;
  fingers?: FingerStates;
  motion?: MotionType;
  calibration?: 'none' | 'mag' | 'gyro';
  custom?: string[];
}

/** Magnetometer calibration data (hard/soft iron) */
export interface MagCalibration {
  hardIron: Vector3;
  softIron: [
    [number, number, number],
    [number, number, number],
    [number, number, number]
  ];
  timestamp: string;
}

/** Gyroscope bias calibration */
export interface GyroBias {
  x: number;
  y: number;
  z: number;
  sampleCount: number;
}

/** Combined calibration data */
export interface CalibrationData {
  mag?: MagCalibration;
  gyroBias?: GyroBias;
}

/** Geomagnetic field reference for location */
export interface GeomagneticLocation {
  name: string;
  latitude: number;
  longitude: number;
  declination: number;  // degrees
  inclination: number;  // degrees
  intensity: number;    // μT
}

/** Complete session data structure */
export interface SessionData {
  version: string;
  device: DeviceInfo;
  calibration: CalibrationData;
  geomagneticLocation?: GeomagneticLocation;
  samples: TelemetrySample[];
  labels: LabelSegment[];
  metadata?: {
    recordedAt: string;
    duration: number;
    sampleCount: number;
  };
}

/** Session file info from API */
export interface SessionInfo {
  filename: string;
  pathname: string;
  url: string;
  downloadUrl: string;
  size: number;
  uploadedAt: string;
  timestamp: string;
}
