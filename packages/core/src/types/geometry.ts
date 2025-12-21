/**
 * Core Geometry Types
 *
 * Canonical definitions for 3D geometry primitives used throughout SIMCAP.
 * All other modules should import these types from @core/types.
 *
 * @module core/types/geometry
 */

// ===== Vector Types =====

/** 3D vector with x, y, z components */
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/** 2D vector with x, y components */
export interface Vector2 {
  x: number;
  y: number;
}

// ===== Rotation Types =====

/**
 * Quaternion for 3D rotation representation.
 * Uses Hamilton convention: q = w + xi + yj + zk
 * Unit quaternions (|q| = 1) represent rotations.
 */
export interface Quaternion {
  /** Scalar (real) component */
  w: number;
  /** X component of imaginary part */
  x: number;
  /** Y component of imaginary part */
  y: number;
  /** Z component of imaginary part */
  z: number;
}

/**
 * Euler angles in degrees.
 * Uses aerospace convention: roll (X), pitch (Y), yaw (Z).
 */
export interface EulerAngles {
  /** Rotation around X axis (degrees) */
  roll: number;
  /** Rotation around Y axis (degrees) */
  pitch: number;
  /** Rotation around Z axis (degrees) */
  yaw: number;
}

/**
 * Euler angles in radians.
 * Alternative representation for math operations.
 */
export interface EulerAnglesRad {
  /** Rotation around X axis (radians) */
  roll: number;
  /** Rotation around Y axis (radians) */
  pitch: number;
  /** Rotation around Z axis (radians) */
  yaw: number;
}

// ===== Matrix Types =====

/** 3x3 matrix for rotation/transformation */
export type Matrix3x3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number]
];

/** 4x4 matrix for affine transformation */
export type Matrix4x4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number]
];

// ===== Utility Types =====

/** Point in 3D space (alias for Vector3) */
export type Point3D = Vector3;

/** Point in 2D space (alias for Vector2) */
export type Point2D = Vector2;

/** Tuple representation of Vector3 for array operations */
export type Vec3Tuple = [number, number, number];

/** Tuple representation of Quaternion for array operations */
export type QuatTuple = [number, number, number, number]; // [w, x, y, z]
