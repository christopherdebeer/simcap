/**
 * FFO$$ Distance Metrics
 *
 * Implements distance metrics from the $-family algorithms adapted to 3D:
 * - $1: Average distance with optimal rotation alignment
 * - $P: Point-cloud distance (permutation-invariant)
 * - $Q: Fast lookup-table-based distance
 *
 * Lower distances indicate better matches.
 *
 * @module ffo/distance
 */

import type { TemplatePoint3D } from './types';

/**
 * Euclidean distance between two 3D points.
 */
export function euclideanDistance(a: TemplatePoint3D, b: TemplatePoint3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Squared Euclidean distance (faster, useful when comparing distances).
 */
export function squaredDistance(a: TemplatePoint3D, b: TemplatePoint3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

// ============================================================================
// $1-STYLE DISTANCE (ALIGNED, SEQUENTIAL)
// ============================================================================

/**
 * Path distance: average Euclidean distance between corresponding points.
 *
 * This is the core $1 distance metric. Points are matched by index,
 * so both arrays must have the same length (after resampling).
 *
 * @param a - First trajectory (N points)
 * @param b - Second trajectory (N points)
 * @returns Average distance between corresponding points
 */
export function pathDistance(a: TemplatePoint3D[], b: TemplatePoint3D[]): number {
  if (a.length !== b.length) {
    throw new Error(`Path distance requires equal-length arrays: ${a.length} vs ${b.length}`);
  }

  if (a.length === 0) return 0;

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += euclideanDistance(a[i], b[i]);
  }

  return sum / a.length;
}

/**
 * Rotate points around Z axis by given angle.
 */
function rotateZ(points: TemplatePoint3D[], angle: number): TemplatePoint3D[] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  return points.map((p) => ({
    x: p.x * cos - p.y * sin,
    y: p.x * sin + p.y * cos,
    z: p.z,
  }));
}

/**
 * Distance at a specific rotation angle.
 *
 * @param a - First trajectory
 * @param b - Second trajectory
 * @param angle - Rotation angle in radians
 * @returns Path distance with b rotated by angle
 */
export function distanceAtAngle(
  a: TemplatePoint3D[],
  b: TemplatePoint3D[],
  angle: number
): number {
  const rotated = rotateZ(b, angle);
  return pathDistance(a, rotated);
}

/**
 * Golden section search for optimal rotation angle.
 *
 * This is the $1 algorithm's method for finding the rotation
 * that minimizes distance between trajectories.
 *
 * @param a - First trajectory
 * @param b - Second trajectory
 * @param angleRange - Search range in radians (default: ±45°)
 * @param threshold - Angular precision threshold (default: 2°)
 * @returns Minimum distance found
 */
export function distanceWithRotation(
  a: TemplatePoint3D[],
  b: TemplatePoint3D[],
  angleRange: number = Math.PI / 4,
  threshold: number = Math.PI / 90
): number {
  const phi = 0.5 * (-1 + Math.sqrt(5)); // Golden ratio

  let x1 = phi * -angleRange + (1 - phi) * angleRange;
  let x2 = (1 - phi) * -angleRange + phi * angleRange;

  let f1 = distanceAtAngle(a, b, x1);
  let f2 = distanceAtAngle(a, b, x2);

  let left = -angleRange;
  let right = angleRange;

  while (Math.abs(right - left) > threshold) {
    if (f1 < f2) {
      right = x2;
      x2 = x1;
      f2 = f1;
      x1 = phi * left + (1 - phi) * right;
      f1 = distanceAtAngle(a, b, x1);
    } else {
      left = x1;
      x1 = x2;
      f1 = f2;
      x2 = (1 - phi) * left + phi * right;
      f2 = distanceAtAngle(a, b, x2);
    }
  }

  return Math.min(f1, f2);
}

// ============================================================================
// $P-STYLE DISTANCE (POINT CLOUD)
// ============================================================================

/**
 * Greedy point cloud distance.
 *
 * This is the core $P distance metric. It matches points greedily
 * by finding the closest unmatched point for each source point.
 *
 * Unlike path distance, this is order-independent (good for gestures
 * that might be drawn in different orders or directions).
 *
 * @param a - First point cloud (N points)
 * @param b - Second point cloud (N points)
 * @returns Sum of minimum distances for each point
 */
export function cloudDistance(a: TemplatePoint3D[], b: TemplatePoint3D[]): number {
  if (a.length !== b.length) {
    throw new Error(`Cloud distance requires equal-length arrays: ${a.length} vs ${b.length}`);
  }

  if (a.length === 0) return 0;

  const n = a.length;
  const matched = new Array<boolean>(n).fill(false);
  let sum = 0;

  for (let i = 0; i < n; i++) {
    let minDist = Infinity;
    let minIndex = -1;

    for (let j = 0; j < n; j++) {
      if (!matched[j]) {
        const d = squaredDistance(a[i], b[j]);
        if (d < minDist) {
          minDist = d;
          minIndex = j;
        }
      }
    }

    matched[minIndex] = true;
    sum += Math.sqrt(minDist);
  }

  return sum / n;
}

/**
 * Bidirectional cloud distance (more accurate but slower).
 *
 * Computes cloud distance in both directions and takes the minimum.
 *
 * @param a - First point cloud
 * @param b - Second point cloud
 * @returns Minimum of both directional distances
 */
export function bidirectionalCloudDistance(
  a: TemplatePoint3D[],
  b: TemplatePoint3D[]
): number {
  const d1 = cloudDistance(a, b);
  const d2 = cloudDistance(b, a);
  return Math.min(d1, d2);
}

// ============================================================================
// $Q-STYLE DISTANCE (FAST LOOKUP)
// ============================================================================

/**
 * Angular bins for $Q lookup table (8 bins for 3D).
 */
const Q_ANGULAR_BINS = 8;

/**
 * Build lookup table for $Q fast distance computation.
 *
 * The lookup table maps each point to an angular bin based on
 * its direction from the centroid. This allows O(1) nearest-neighbor
 * lookup during matching.
 *
 * @param points - Normalized points (centered at origin)
 * @returns Array of bin indices for each point
 */
export function buildLookupTable(points: TemplatePoint3D[]): number[] {
  return points.map((p) => {
    // Use octant-based binning for 3D (8 bins)
    let bin = 0;
    if (p.x >= 0) bin |= 1;
    if (p.y >= 0) bin |= 2;
    if (p.z >= 0) bin |= 4;
    return bin;
  });
}

/**
 * Build detailed lookup table for $Q (angular sectors).
 *
 * This version creates more granular bins using spherical coordinates.
 *
 * @param points - Normalized points (centered at origin)
 * @param thetaBins - Number of azimuthal bins (default: 8)
 * @param phiBins - Number of elevation bins (default: 4)
 * @returns Array of bin indices for each point
 */
export function buildDetailedLookupTable(
  points: TemplatePoint3D[],
  thetaBins: number = 8,
  phiBins: number = 4
): number[] {
  return points.map((p) => {
    const r = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

    if (r === 0) return 0;

    // Spherical coordinates
    const theta = Math.atan2(p.y, p.x); // [-π, π]
    const phi = Math.acos(p.z / r); // [0, π]

    // Convert to bin indices
    const thetaIdx = Math.floor(((theta + Math.PI) / (2 * Math.PI)) * thetaBins) % thetaBins;
    const phiIdx = Math.floor((phi / Math.PI) * phiBins) % phiBins;

    return thetaIdx + phiIdx * thetaBins;
  });
}

/**
 * Fast $Q-style distance using lookup tables.
 *
 * Uses the lookup tables to quickly find candidate matches,
 * then computes exact distances only for candidates in nearby bins.
 *
 * Time complexity: O(n) average case vs O(n²) for cloud distance.
 *
 * @param a - First trajectory (with lookup table)
 * @param aTable - Lookup table for a
 * @param b - Second trajectory (with lookup table)
 * @param bTable - Lookup table for b
 * @returns Approximate cloud distance
 */
export function lookupDistance(
  a: TemplatePoint3D[],
  aTable: number[],
  b: TemplatePoint3D[],
  bTable: number[]
): number {
  if (a.length !== b.length) {
    throw new Error(`Lookup distance requires equal-length arrays: ${a.length} vs ${b.length}`);
  }

  if (a.length === 0) return 0;

  const n = a.length;

  // Group b points by their bin
  const binToPoints: Map<number, number[]> = new Map();
  for (let j = 0; j < n; j++) {
    const bin = bTable[j];
    if (!binToPoints.has(bin)) {
      binToPoints.set(bin, []);
    }
    binToPoints.get(bin)!.push(j);
  }

  // For each point in a, find nearest in same or adjacent bins
  const matched = new Array<boolean>(n).fill(false);
  let sum = 0;

  for (let i = 0; i < n; i++) {
    const aBin = aTable[i];
    let minDist = Infinity;
    let minIndex = -1;

    // Check same bin and adjacent bins (for 3D octants, adjacent means differ by 1 bit)
    const binsToCheck = [aBin];
    for (let flip = 0; flip < 3; flip++) {
      binsToCheck.push(aBin ^ (1 << flip));
    }

    for (const bin of binsToCheck) {
      const candidates = binToPoints.get(bin) ?? [];
      for (const j of candidates) {
        if (!matched[j]) {
          const d = squaredDistance(a[i], b[j]);
          if (d < minDist) {
            minDist = d;
            minIndex = j;
          }
        }
      }
    }

    // Fallback: if no match in nearby bins, check all unmatched
    if (minIndex === -1) {
      for (let j = 0; j < n; j++) {
        if (!matched[j]) {
          const d = squaredDistance(a[i], b[j]);
          if (d < minDist) {
            minDist = d;
            minIndex = j;
          }
        }
      }
    }

    if (minIndex !== -1) {
      matched[minIndex] = true;
      sum += Math.sqrt(minDist);
    }
  }

  return sum / n;
}

// ============================================================================
// SCORE CONVERSION
// ============================================================================

/**
 * Convert distance to a normalized score (0-1, higher is better).
 *
 * Uses a sigmoid-like transformation based on the original $1 formula.
 *
 * @param distance - Raw distance value
 * @param halfDistance - Distance at which score = 0.5 (default: 0.5)
 * @returns Normalized score between 0 and 1
 */
export function distanceToScore(distance: number, halfDistance: number = 0.5): number {
  // Use exponential decay: score = 1 / (1 + d/halfDist)
  return 1 / (1 + distance / halfDistance);
}

/**
 * Alternative score using the original $1 formula.
 *
 * score = 1 - d / (0.5 * sqrt(2))
 *
 * @param distance - Raw distance value (should be normalized)
 * @returns Score between 0 and 1 (can be negative for large distances)
 */
export function dollarOneScore(distance: number): number {
  const diagonal = 0.5 * Math.sqrt(2);
  return Math.max(0, 1 - distance / diagonal);
}

/**
 * Check if a distance indicates a confident match.
 *
 * @param distance - Distance to check
 * @param threshold - Maximum distance for a match
 * @returns True if distance is below threshold
 */
export function isMatch(distance: number, threshold: number): boolean {
  return distance < threshold;
}
