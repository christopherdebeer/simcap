/**
 * FFO$$ - Fist Full Of Dollars
 *
 * Template-based gesture recognition using the $-family algorithms
 * adapted for 3D IMU sensor data from SIMCAP devices.
 *
 * @module ffo
 *
 * @example
 * ```typescript
 * import { FFORecognizer, createRecognizer } from '@simcap/ffo';
 *
 * // Create recognizer
 * const recognizer = createRecognizer({ numPoints: 32 });
 *
 * // Add gesture templates from recorded samples
 * recognizer.addTemplateFromSamples('wave', waveSamples);
 * recognizer.addTemplateFromSamples('circle', circleSamples);
 *
 * // Recognize new gestures
 * const result = recognizer.recognize(inputSamples);
 * console.log(result.template?.name, result.score);
 * ```
 */

// ===== Main Recognizer =====
export { FFORecognizer, createRecognizer } from './recognizer';

// ===== Resampling =====
export {
  resample,
  resampleImmutable,
  resampleTelemetry,
  extractTrajectory,
  removeGravityApprox,
  pathLength,
  distance3D,
  lerp3D,
  suggestResampleCount,
} from './resample';

// ===== Normalization =====
export {
  normalize,
  quickNormalize,
  fullNormalize,
  centroid,
  boundingBox,
  translateToOrigin,
  scaleToSize,
  indicativeAngles,
  rotateAroundZ,
} from './normalize';

// ===== Distance Metrics =====
export {
  euclideanDistance,
  squaredDistance,
  pathDistance,
  distanceAtAngle,
  distanceWithRotation,
  cloudDistance,
  bidirectionalCloudDistance,
  buildLookupTable,
  buildDetailedLookupTable,
  lookupDistance,
  distanceToScore,
  dollarOneScore,
  isMatch,
} from './distance';

// ===== Types =====
export type {
  TemplatePoint3D,
  GestureTemplate,
  GestureVocabulary,
  TemplateMeta,
  VocabularyMeta,
  RecognitionResult,
  RecognitionCandidate,
  TelemetrySample3D,
  OrientedTelemetrySample,
  TelemetryWindow,
  OrientedTelemetryWindow,
  RecognizerConfig,
  BoundingBox3D,
  NormalizationResult,
  ResampleOptions,
  NormalizeOptions,
} from './types';

export { DEFAULT_CONFIG } from './types';
