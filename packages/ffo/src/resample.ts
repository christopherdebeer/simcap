/**
 * FFO$$ Trajectory Resampling
 *
 * Implements the resampling step of the $-family algorithms.
 * Resamples a variable-length trajectory to a fixed number of equally-spaced points.
 *
 * This is essential for template matching since it allows comparison of
 * gestures performed at different speeds.
 *
 * Based on: Wobbrock, J.O., Wilson, A.D., Li, Y. (2007). "Gestures without libraries,
 * toolkits or training: A $1 recognizer for user interface prototypes." UIST '07.
 *
 * @module ffo/resample
 */

import type { TemplatePoint3D, TelemetrySample3D, ResampleOptions } from './types';

/**
 * Calculate Euclidean distance between two 3D points.
 */
export function distance3D(a: TemplatePoint3D, b: TemplatePoint3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calculate total path length of a trajectory.
 */
export function pathLength(points: TemplatePoint3D[]): number {
  let length = 0;
  for (let i = 1; i < points.length; i++) {
    length += distance3D(points[i - 1], points[i]);
  }
  return length;
}

/**
 * Linear interpolation between two 3D points.
 *
 * @param a - Start point
 * @param b - End point
 * @param t - Interpolation factor (0 = a, 1 = b)
 * @returns Interpolated point
 */
export function lerp3D(a: TemplatePoint3D, b: TemplatePoint3D, t: number): TemplatePoint3D {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    z: a.z + (b.z - a.z) * t,
  };
}

/**
 * Resample a trajectory to N equally-spaced points.
 *
 * This is the core resampling algorithm from the $1 recognizer:
 * 1. Calculate total path length
 * 2. Determine ideal spacing (path length / (N-1))
 * 3. Walk along path, inserting points at equal intervals
 *
 * @param points - Input trajectory (variable length)
 * @param n - Target number of points (default: 32)
 * @returns Resampled trajectory with exactly N points
 *
 * @example
 * ```typescript
 * const raw = [{ x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, { x: 10, y: 10, z: 0 }];
 * const resampled = resample(raw, 5);
 * // Returns 5 equally-spaced points along the L-shaped path
 * ```
 */
export function resample(points: TemplatePoint3D[], n: number = 32): TemplatePoint3D[] {
  if (points.length === 0) {
    return [];
  }

  if (points.length === 1) {
    // Single point: replicate it N times
    return Array(n).fill({ ...points[0] });
  }

  if (n <= 1) {
    return [{ ...points[0] }];
  }

  const totalLength = pathLength(points);

  if (totalLength === 0) {
    // All points are the same: replicate the first point
    return Array(n).fill({ ...points[0] });
  }

  const interval = totalLength / (n - 1);
  const resampled: TemplatePoint3D[] = [{ ...points[0] }];

  let accumulatedDistance = 0;
  let i = 1;

  while (resampled.length < n && i < points.length) {
    const segmentDist = distance3D(points[i - 1], points[i]);

    if (accumulatedDistance + segmentDist >= interval) {
      // We need to insert a point within this segment
      const overshoot = interval - accumulatedDistance;
      const t = overshoot / segmentDist;
      const newPoint = lerp3D(points[i - 1], points[i], t);
      resampled.push(newPoint);

      // Insert the new point into the path for continued processing
      points.splice(i, 0, newPoint);
      accumulatedDistance = 0;
    } else {
      accumulatedDistance += segmentDist;
      i++;
    }
  }

  // Ensure we have exactly N points (handle floating point edge cases)
  while (resampled.length < n) {
    resampled.push({ ...points[points.length - 1] });
  }

  return resampled;
}

/**
 * Resample without modifying the input array.
 * Creates a copy of the input before resampling.
 */
export function resampleImmutable(points: TemplatePoint3D[], n: number = 32): TemplatePoint3D[] {
  const copy = points.map((p) => ({ ...p }));
  return resample(copy, n);
}

/**
 * Convert telemetry samples to 3D trajectory points.
 *
 * Extracts accelerometer data as the motion trajectory.
 * For orientation-aware conversion, use extractOrientedTrajectory instead.
 *
 * @param samples - Array of telemetry samples with unit-converted values
 * @returns Array of 3D points from accelerometer data
 */
export function extractTrajectory(samples: TelemetrySample3D[]): TemplatePoint3D[] {
  return samples.map((s) => ({
    x: s.ax_g,
    y: s.ay_g,
    z: s.az_g,
  }));
}

/**
 * Remove gravity component from accelerometer trajectory.
 *
 * Assumes gravity is approximately [0, 0, 1] in sensor frame during rest.
 * For accurate gravity removal, use orientation-based transformation.
 *
 * @param points - Accelerometer trajectory in g
 * @returns Trajectory with gravity approximation removed
 */
export function removeGravityApprox(points: TemplatePoint3D[]): TemplatePoint3D[] {
  if (points.length === 0) return [];

  // Estimate gravity from first few samples (assuming device starts at rest)
  const windowSize = Math.min(5, points.length);
  const gravity = { x: 0, y: 0, z: 0 };

  for (let i = 0; i < windowSize; i++) {
    gravity.x += points[i].x / windowSize;
    gravity.y += points[i].y / windowSize;
    gravity.z += points[i].z / windowSize;
  }

  return points.map((p) => ({
    x: p.x - gravity.x,
    y: p.y - gravity.y,
    z: p.z - gravity.z,
  }));
}

/**
 * Resample telemetry samples directly to template points.
 *
 * Convenience function that combines extraction, optional gravity removal,
 * and resampling in one step.
 *
 * @param samples - Telemetry samples
 * @param options - Resampling options
 * @returns Resampled trajectory as TemplatePoint3D[]
 */
export function resampleTelemetry(
  samples: TelemetrySample3D[],
  options: ResampleOptions & { removeGravity?: boolean } = { n: 32 }
): TemplatePoint3D[] {
  let trajectory = extractTrajectory(samples);

  if (options.removeGravity) {
    trajectory = removeGravityApprox(trajectory);
  }

  return resampleImmutable(trajectory, options.n);
}

/**
 * Calculate optimal resample count based on input length.
 *
 * The $Q paper suggests N=32 for most cases, but this can be adjusted
 * based on gesture complexity and available memory.
 *
 * @param inputLength - Number of input points
 * @param minN - Minimum resample count (default: 16)
 * @param maxN - Maximum resample count (default: 64)
 * @returns Recommended resample count
 */
export function suggestResampleCount(
  inputLength: number,
  minN: number = 16,
  maxN: number = 64
): number {
  // Rule of thumb: aim for N ≈ sqrt(inputLength) * 4, clamped to [minN, maxN]
  const suggested = Math.round(Math.sqrt(inputLength) * 4);
  return Math.max(minN, Math.min(maxN, suggested));
}
