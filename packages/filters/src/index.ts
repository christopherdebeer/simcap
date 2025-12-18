/**
 * @filters - Signal processing filters for SIMCAP
 *
 * Provides:
 * - IMU Sensor Fusion (Madgwick AHRS for orientation estimation)
 * - Kalman Filters (1D, 3D, 6D variants)
 * - Motion Detection
 * - Particle Filter for multi-hypothesis tracking
 */

// Re-export everything from filters
export {
  // Types
  type Vector3,
  type Quaternion,
  type EulerAngles,
  type GeomagneticReference,
  type MadgwickOptions,
  type MotionDetectorOptions,
  type MotionState,
  type KalmanFilter3DOptions,
  type ParticleFilterOptions,
  type Particle,
  // Classes
  MadgwickAHRS,
  ComplementaryFilter,
  MotionDetector,
  KalmanFilter6D,
  MultiFingerKalmanFilter,
  ParticleFilter,
  // Functions
  magneticLikelihood,
} from './filters';

// Re-export from kalman
export {
  KalmanFilter,
  KalmanFilter3D,
  type KalmanFilterOptions,
} from './kalman';
