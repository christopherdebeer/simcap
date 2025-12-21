/**
 * Core Types for SIMCAP
 *
 * Re-exports all types from submodules for convenient importing:
 *
 *   import { Vector3, RawTelemetry, SessionData } from '@core/types';
 *
 * Type Modules:
 * - geometry: Vector3, Quaternion, EulerAngles, Matrix types
 * - telemetry: RawTelemetry, DecoratedTelemetry, pipeline stage types
 * - session: SessionData, LabelSegment, calibration types
 * - hand: FingerLabel, FingerLabels, tracking types
 * - device: DeviceInfo, FirmwareInfo, connection types
 *
 * @module core/types
 */

// Geometry primitives (canonical source)
export * from './geometry';

// Telemetry pipeline (8-stage processing)
export * from './telemetry';

// Session data and labels
export * from './session';

// Hand and finger types
export * from './hand';

// Device and firmware types
export * from './device';
