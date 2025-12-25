/**
 * FFO$$ Type Definitions
 *
 * Types for the $-family gesture recognition system adapted to 3D IMU data.
 *
 * @module ffo/types
 */

import type { Vector3, Quaternion, EulerAngles } from '@simcap/core/types';

// ============================================================================
// CORE TYPES
// ============================================================================

/**
 * A point in 3D space for $-family template matching.
 * Uses palm-centric coordinate system:
 * - x: across palm (positive = toward thumb)
 * - y: along fingers (positive = toward fingertips)
 * - z: out of palm (positive = away from palm)
 */
export interface TemplatePoint3D {
  x: number;
  y: number;
  z: number;
}

/**
 * A gesture template for $Q-3D matching.
 * Contains normalized 3D points resampled to a fixed count.
 */
export interface GestureTemplate {
  /** Unique identifier for this template */
  id: string;

  /** Human-readable gesture name */
  name: string;

  /** Normalized 3D points (resampled to N) */
  points: TemplatePoint3D[];

  /** Template metadata */
  meta: TemplateMeta;
}

/**
 * Metadata associated with a gesture template.
 */
export interface TemplateMeta {
  /** Number of points (typically 32 or 64) */
  n: number;

  /** Source of template (user ID, auto-generated, etc.) */
  source: string;

  /** Timestamp of creation (ISO 8601) */
  created: string;

  /** Optional: original duration in milliseconds */
  duration?: number;

  /** Optional: pre-computed lookup table for $Q speedup */
  lookupTable?: number[];
}

/**
 * Collection of templates forming a gesture vocabulary.
 */
export interface GestureVocabulary {
  /** Vocabulary version string */
  version: string;

  /** Collection of gesture templates */
  templates: GestureTemplate[];

  /** Optional: distance threshold for "no match" (rejection) */
  rejectThreshold?: number;

  /** Metadata about the vocabulary */
  meta?: VocabularyMeta;
}

/**
 * Metadata for a gesture vocabulary.
 */
export interface VocabularyMeta {
  /** Human-readable name */
  name?: string;

  /** Description of the vocabulary */
  description?: string;

  /** Creation timestamp (ISO 8601) */
  created?: string;

  /** Last modified timestamp (ISO 8601) */
  modified?: string;

  /** Author or source */
  author?: string;
}

// ============================================================================
// RECOGNITION RESULT TYPES
// ============================================================================

/**
 * Result of gesture recognition.
 */
export interface RecognitionResult {
  /** Best matching template (null if rejected) */
  template: GestureTemplate | null;

  /** Distance to best match (lower is better) */
  distance: number;

  /** Normalized score (0-1, higher is better) */
  score: number;

  /** Whether the result was rejected (above threshold) */
  rejected: boolean;

  /** Optional: all candidates ranked by distance */
  candidates?: RecognitionCandidate[];
}

/**
 * A candidate template with its match score.
 */
export interface RecognitionCandidate {
  /** Template that was matched */
  template: GestureTemplate;

  /** Distance to this template */
  distance: number;

  /** Normalized score (0-1) */
  score: number;
}

// ============================================================================
// INPUT TYPES (FROM TELEMETRY)
// ============================================================================

/**
 * Minimal telemetry sample for FFO$$ recognition.
 * Can be extracted from any stage that has unit conversion.
 */
export interface TelemetrySample3D {
  /** Acceleration X (g) */
  ax_g: number;
  /** Acceleration Y (g) */
  ay_g: number;
  /** Acceleration Z (g) */
  az_g: number;
  /** Timestamp (ms) */
  t: number;
}

/**
 * Telemetry sample with orientation for palm-centric transformation.
 */
export interface OrientedTelemetrySample extends TelemetrySample3D {
  /** Quaternion W component */
  orientation_w: number;
  /** Quaternion X component */
  orientation_x: number;
  /** Quaternion Y component */
  orientation_y: number;
  /** Quaternion Z component */
  orientation_z: number;
}

/**
 * A window of telemetry samples for gesture recognition.
 */
export type TelemetryWindow = TelemetrySample3D[];

/**
 * A window of oriented telemetry samples.
 */
export type OrientedTelemetryWindow = OrientedTelemetrySample[];

// ============================================================================
// CONFIGURATION TYPES
// ============================================================================

/**
 * Configuration for the FFO$$ recognizer.
 */
export interface RecognizerConfig {
  /** Number of points to resample trajectories to (default: 32) */
  numPoints: number;

  /** Whether to use orientation for palm-centric transformation */
  useOrientation: boolean;

  /** Distance threshold for rejection (null = no rejection) */
  rejectThreshold: number | null;

  /** Whether to compute lookup table for $Q optimization */
  useLookupTable: boolean;

  /** Minimum samples required for recognition */
  minSamples: number;

  /** Whether to remove gravity from accelerometer */
  removeGravity: boolean;
}

/**
 * Default recognizer configuration.
 */
export const DEFAULT_CONFIG: RecognizerConfig = {
  numPoints: 32,
  useOrientation: true,
  rejectThreshold: null,
  useLookupTable: true,
  minSamples: 10,
  removeGravity: true,
};

// ============================================================================
// UTILITY TYPES
// ============================================================================

/**
 * Bounding box for a set of 3D points.
 */
export interface BoundingBox3D {
  min: Vector3;
  max: Vector3;
  center: Vector3;
  size: Vector3;
}

/**
 * Result of trajectory normalization.
 */
export interface NormalizationResult {
  /** Normalized points */
  points: TemplatePoint3D[];

  /** Original centroid before translation */
  centroid: Vector3;

  /** Scale factor applied */
  scale: number;

  /** Bounding box before normalization */
  originalBounds: BoundingBox3D;
}

/**
 * Options for trajectory resampling.
 */
export interface ResampleOptions {
  /** Target number of points */
  n: number;

  /** Whether to preserve temporal ordering */
  preserveOrder?: boolean;
}

/**
 * Options for trajectory normalization.
 */
export interface NormalizeOptions {
  /** Whether to translate to origin */
  translate?: boolean;

  /** Whether to scale to unit size */
  scale?: boolean;

  /** Target scale (default: 1.0) */
  targetScale?: number;

  /** Whether to apply rotation normalization (for 2D compatibility) */
  rotate?: boolean;
}
