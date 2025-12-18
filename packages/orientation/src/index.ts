/**
 * @orientation - Orientation and sensor processing for SIMCAP
 *
 * Provides:
 * - Coordinate frame transformations (orientation-model)
 * - Orientation calibration utilities (orientation-calibration)
 * - Sensor unit conversions (sensor-units)
 */

// Re-export from orientation-model
export * from './orientation-model';

// Re-export from orientation-calibration
export * from './orientation-calibration';

// Re-export from sensor-units (authoritative source for sensor constants)
export * from './sensor-units';
