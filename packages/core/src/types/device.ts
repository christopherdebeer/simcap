/**
 * Device Types
 *
 * Types for device identification, firmware info, and compatibility.
 *
 * @module core/types/device
 */

// ===== Device Identification =====

/**
 * Device hardware and firmware identification.
 */
export interface DeviceInfo {
  /** Unique device identifier (BLE address or serial) */
  id: string;
  /** Firmware version string */
  firmware: string;
  /** Hardware revision (optional) */
  hardware?: string;
}

// ===== Firmware Information =====

/**
 * Firmware identification from device.
 * Comprehensive structure supporting various firmware implementations.
 */
export interface FirmwareInfo {
  /** Firmware identifier (e.g., "GAMBIT", "LOADER") */
  id: string;
  /** Semantic version (e.g., "1.2.0") */
  version: string;
  /** Build identifier or commit hash */
  build?: string;
  /** Uptime in milliseconds since boot */
  uptime?: number;
  /** Human-readable firmware name */
  name?: string;
  /** Author or maintainer */
  author?: string;
  /** Feature flags enabled in this build */
  features?: string[];
}

// ===== Compatibility =====

/**
 * Result of firmware compatibility check.
 */
export interface CompatibilityResult {
  /** Whether the firmware is compatible */
  compatible: boolean;
  /** Reason for incompatibility (if applicable) */
  reason?: string;
}

// ===== Connection State =====

/** Connection states for BLE devices */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * BLE connection status information.
 */
export interface ConnectionStatus {
  /** Current connection state */
  state: ConnectionState;
  /** Device being connected to (if any) */
  device?: DeviceInfo;
  /** Error message (if state is 'error') */
  error?: string;
  /** Connection timestamp */
  connectedAt?: string;
}

// ===== Battery and Diagnostics =====

/**
 * Device battery status.
 */
export interface BatteryStatus {
  /** Battery level (0-100) */
  level: number;
  /** Whether device is charging */
  charging?: boolean;
  /** Estimated time remaining (minutes) */
  timeRemaining?: number;
}

/**
 * Device log entry.
 */
export interface DeviceLogEntry {
  /** Log level character (E, W, I, D) */
  level: string;
  /** Timestamp (ms since boot) */
  time: number;
  /** Log message */
  msg: string;
}

/**
 * Device log statistics.
 */
export interface DeviceLogStats {
  total: number;
  errors: number;
  warnings: number;
  info: number;
  debug: number;
}

// ===== Sample Collection =====

/**
 * Result of collecting samples from device.
 */
export interface SampleCollectionResult {
  /** Number of samples collected */
  collectedCount: number;
  /** Number of samples requested */
  requestedCount: number;
  /** Actual collection duration (ms) */
  durationMs: number;
  /** Requested sample rate (Hz) */
  requestedHz: number;
  /** Actual achieved sample rate (Hz) */
  actualHz: number;
}
