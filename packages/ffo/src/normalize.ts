/**
 * FFO$$ Trajectory Normalization
 *
 * Implements normalization steps from the $-family algorithms adapted to 3D:
 * - Translation: Move centroid to origin
 * - Scaling: Scale to unit bounding box/sphere
 * - Rotation: (Optional) Align principal axis with coordinate axis
 *
 * Normalization ensures gestures can be compared regardless of position,
 * size, or orientation in space.
 *
 * @module ffo/normalize
 */

import type {
  TemplatePoint3D,
  BoundingBox3D,
  NormalizationResult,
  NormalizeOptions,
} from './types';
import type { Vector3 } from '@core/types';

/**
 * Default normalization options.
 */
const DEFAULT_OPTIONS: Required<NormalizeOptions> = {
  translate: true,
  scale: true,
  targetScale: 1.0,
  rotate: false,
};

/**
 * Calculate the centroid (center of mass) of a set of 3D points.
 *
 * @param points - Array of 3D points
 * @returns Centroid as Vector3
 */
export function centroid(points: TemplatePoint3D[]): Vector3 {
  if (points.length === 0) {
    return { x: 0, y: 0, z: 0 };
  }

  const sum = points.reduce(
    (acc, p) => ({
      x: acc.x + p.x,
      y: acc.y + p.y,
      z: acc.z + p.z,
    }),
    { x: 0, y: 0, z: 0 }
  );

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
    z: sum.z / points.length,
  };
}

/**
 * Calculate the bounding box of a set of 3D points.
 *
 * @param points - Array of 3D points
 * @returns Bounding box with min, max, center, and size
 */
export function boundingBox(points: TemplatePoint3D[]): BoundingBox3D {
  if (points.length === 0) {
    return {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 0, z: 0 },
      center: { x: 0, y: 0, z: 0 },
      size: { x: 0, y: 0, z: 0 },
    };
  }

  const min = { x: Infinity, y: Infinity, z: Infinity };
  const max = { x: -Infinity, y: -Infinity, z: -Infinity };

  for (const p of points) {
    min.x = Math.min(min.x, p.x);
    min.y = Math.min(min.y, p.y);
    min.z = Math.min(min.z, p.z);
    max.x = Math.max(max.x, p.x);
    max.y = Math.max(max.y, p.y);
    max.z = Math.max(max.z, p.z);
  }

  return {
    min,
    max,
    center: {
      x: (min.x + max.x) / 2,
      y: (min.y + max.y) / 2,
      z: (min.z + max.z) / 2,
    },
    size: {
      x: max.x - min.x,
      y: max.y - min.y,
      z: max.z - min.z,
    },
  };
}

/**
 * Translate points so the centroid is at the origin.
 *
 * @param points - Array of 3D points
 * @param center - Optional pre-computed centroid (computed if not provided)
 * @returns Translated points
 */
export function translateToOrigin(
  points: TemplatePoint3D[],
  center?: Vector3
): TemplatePoint3D[] {
  const c = center ?? centroid(points);

  return points.map((p) => ({
    x: p.x - c.x,
    y: p.y - c.y,
    z: p.z - c.z,
  }));
}

/**
 * Scale points to fit within a unit cube (or target scale).
 *
 * Uses the maximum dimension of the bounding box as the scale reference.
 * This preserves the aspect ratio of the gesture.
 *
 * @param points - Array of 3D points (should be centered at origin)
 * @param targetScale - Target size (default: 1.0)
 * @returns Object containing scaled points and scale factor
 */
export function scaleToSize(
  points: TemplatePoint3D[],
  targetScale: number = 1.0
): { points: TemplatePoint3D[]; scale: number } {
  if (points.length === 0) {
    return { points: [], scale: 1 };
  }

  const bounds = boundingBox(points);
  const maxDimension = Math.max(bounds.size.x, bounds.size.y, bounds.size.z);

  if (maxDimension === 0) {
    // All points are identical
    return { points: points.map((p) => ({ ...p })), scale: 1 };
  }

  const scale = targetScale / maxDimension;

  return {
    points: points.map((p) => ({
      x: p.x * scale,
      y: p.y * scale,
      z: p.z * scale,
    })),
    scale,
  };
}

/**
 * Calculate the indicative angle (direction from centroid to first point).
 *
 * This is the 3D equivalent of the $1 indicative angle, used for
 * rotation normalization. Returns angles in spherical coordinates.
 *
 * @param points - Array of 3D points
 * @param center - Optional pre-computed centroid
 * @returns Spherical angles { theta, phi } in radians
 */
export function indicativeAngles(
  points: TemplatePoint3D[],
  center?: Vector3
): { theta: number; phi: number } {
  if (points.length === 0) {
    return { theta: 0, phi: 0 };
  }

  const c = center ?? centroid(points);
  const first = points[0];

  // Vector from centroid to first point
  const dx = first.x - c.x;
  const dy = first.y - c.y;
  const dz = first.z - c.z;

  const r = Math.sqrt(dx * dx + dy * dy + dz * dz);

  if (r === 0) {
    return { theta: 0, phi: 0 };
  }

  // Spherical coordinates: theta = azimuth (XY plane), phi = elevation
  const theta = Math.atan2(dy, dx);
  const phi = Math.acos(dz / r);

  return { theta, phi };
}

/**
 * Rotate points to align the indicative angle with a reference direction.
 *
 * This is a simplified 3D rotation that aligns the XY projection
 * with the positive X axis (like $1 in 2D).
 *
 * @param points - Array of 3D points (should be centered at origin)
 * @param angle - Rotation angle in radians (around Z axis)
 * @returns Rotated points
 */
export function rotateAroundZ(points: TemplatePoint3D[], angle: number): TemplatePoint3D[] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  return points.map((p) => ({
    x: p.x * cos - p.y * sin,
    y: p.x * sin + p.y * cos,
    z: p.z,
  }));
}

/**
 * Full normalization pipeline for 3D trajectories.
 *
 * Applies translation, scaling, and optional rotation to create
 * a canonical representation suitable for template matching.
 *
 * @param points - Input trajectory
 * @param options - Normalization options
 * @returns Normalization result with normalized points and metadata
 *
 * @example
 * ```typescript
 * const raw = [{ x: 10, y: 20, z: 5 }, { x: 15, y: 25, z: 10 }];
 * const result = normalize(raw);
 * // result.points are centered at origin and scaled to unit size
 * ```
 */
export function normalize(
  points: TemplatePoint3D[],
  options: NormalizeOptions = {}
): NormalizationResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  if (points.length === 0) {
    return {
      points: [],
      centroid: { x: 0, y: 0, z: 0 },
      scale: 1,
      originalBounds: boundingBox([]),
    };
  }

  const originalCentroid = centroid(points);
  const originalBounds = boundingBox(points);

  let normalized = points.map((p) => ({ ...p }));

  // Step 1: Translate to origin
  if (opts.translate) {
    normalized = translateToOrigin(normalized, originalCentroid);
  }

  // Step 2: Scale to target size
  let scaleFactor = 1;
  if (opts.scale) {
    const scaled = scaleToSize(normalized, opts.targetScale);
    normalized = scaled.points;
    scaleFactor = scaled.scale;
  }

  // Step 3: Optional rotation normalization
  if (opts.rotate) {
    const angles = indicativeAngles(normalized);
    // Rotate to align with positive X axis in XY plane
    normalized = rotateAroundZ(normalized, -angles.theta);
  }

  return {
    points: normalized,
    centroid: originalCentroid,
    scale: scaleFactor,
    originalBounds,
  };
}

/**
 * Quick normalization for template matching.
 *
 * Applies translate and scale only (skips rotation).
 * This is sufficient for $P and $Q which are rotation-invariant.
 *
 * @param points - Input trajectory
 * @returns Normalized points
 */
export function quickNormalize(points: TemplatePoint3D[]): TemplatePoint3D[] {
  return normalize(points, { translate: true, scale: true, rotate: false }).points;
}

/**
 * Full normalization including rotation (for $1-style matching).
 *
 * @param points - Input trajectory
 * @returns Normalized points with rotation alignment
 */
export function fullNormalize(points: TemplatePoint3D[]): TemplatePoint3D[] {
  return normalize(points, { translate: true, scale: true, rotate: true }).points;
}
