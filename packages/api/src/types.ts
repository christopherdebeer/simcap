/**
 * Shared API Types
 *
 * Types shared between API handlers and client-side code to ensure
 * type safety across the API boundary.
 */

// ===== Common Types =====

/** Standard error response */
export interface ApiError {
  error: string;
  message?: string;
}

// ===== Sessions API Types =====

/** Session file information */
export interface SessionInfo {
  /** Filename without path (e.g., "2025-12-15T22_35_15.567Z.json") */
  filename: string;
  /** Full blob pathname (e.g., "sessions/2025-12-15T22_35_15.567Z.json") */
  pathname: string;
  /** Public URL to access the session */
  url: string;
  /** Download URL for the session */
  downloadUrl: string;
  /** File size in bytes */
  size: number;
  /** ISO timestamp when uploaded */
  uploadedAt: string;
  /** Session timestamp (with colons, e.g., "2025-12-15T22:35:15.567Z") */
  timestamp: string;
}

/** GET /api/sessions response */
export interface SessionsResponse {
  sessions: SessionInfo[];
  count: number;
  generatedAt: string;
}

// ===== Visualizations API Types =====

/** Window visualization entry */
export interface WindowEntry {
  window_num: number;
  filepath: string;
  images: Record<string, string>;
  trajectory_images: Record<string, string>;
}

/** Session visualization data */
export interface SessionVisualization {
  /** Session timestamp (normalized with colons) */
  timestamp: string;
  /** Session filename */
  filename: string;
  /** Main composite image URL */
  composite_image: string | null;
  /** Calibration stages plot URL */
  calibration_stages_image: string | null;
  /** 3D orientation plot URL */
  orientation_3d_image: string | null;
  /** Orientation tracking plot URL */
  orientation_track_image: string | null;
  /** Raw sensor axes plot URL */
  raw_axes_image: string | null;
  /** Trajectory comparison images by type */
  trajectory_comparison_images: Record<string, string>;
  /** Per-window visualizations */
  windows: WindowEntry[];
}

/** GET /api/visualizations response (list all) */
export interface VisualizationsListResponse {
  sessions: SessionVisualization[];
  count: number;
  totalFiles: number;
  generatedAt: string;
}

/** GET /api/visualizations?session=... response (single session) */
export interface VisualizationSessionResponse {
  session: SessionVisualization | null;
  found: boolean;
  generatedAt: string;
}

// ===== Upload API Types =====

/** Upload progress stages */
export type UploadStage = 'preparing' | 'uploading' | 'complete' | 'error' | 'attempt' | 'retry';

/** Upload progress callback data */
export interface UploadProgress {
  stage: UploadStage;
  message: string;
  url?: string;
}

/** Options for upload function */
export interface UploadOptions {
  /** Filename for the session (e.g., "2025-12-15T22_35_15.567Z.json") */
  filename: string;
  /** JSON content to upload */
  content: string;
  /** Progress callback */
  onProgress?: (progress: UploadProgress) => void;
}

/** Options for upload with retry */
export interface UploadWithRetryOptions extends UploadOptions {
  /** Maximum retry attempts (default: 3) */
  maxRetries?: number;
}

/** Upload result */
export interface UploadResult {
  success: boolean;
  /** Public URL of uploaded file */
  url: string;
  /** Blob pathname */
  pathname: string;
  /** File size in bytes */
  size: number;
  /** Original filename */
  filename: string;
}

/** Client payload sent with upload token request */
export interface UploadClientPayload {
  secret: string;
}

// ===== Combined Session Entry (for VIZ/Explorer) =====

/** Combined session with optional visualization data */
export interface SessionEntry extends SessionInfo {
  /** URL to fetch full session JSON (for lazy loading) */
  sessionUrl: string | null;
  /** Composite visualization image URL */
  composite_image: string | null;
  /** Calibration stages image URL */
  calibration_stages_image: string | null;
  /** 3D orientation image URL */
  orientation_3d_image: string | null;
  /** Orientation tracking image URL */
  orientation_track_image: string | null;
  /** Raw axes image URL */
  raw_axes_image: string | null;
  /** Trajectory comparison images */
  trajectory_comparison_images: Record<string, string>;
  /** Window visualizations */
  windows: WindowEntry[];
}

// ===== Session Data Types (for uploaded JSON content) =====

/** Raw telemetry sample */
export interface TelemetrySample {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  mx: number;
  my: number;
  mz: number;
  t: number;
}

/** Session metadata in uploaded file */
export interface SessionMetadata {
  sample_rate?: number;
  device?: string;
  firmware_version?: string;
  calibration?: unknown;
  location?: unknown;
  subject_id?: string;
  environment?: string;
  hand?: string;
  split?: string;
  magnet_config?: string;
  magnet_type?: string;
  notes?: string;
  session_type?: string;
}

/** Full session data structure (v2.1 schema) */
export interface SessionData {
  version: string;
  timestamp: string;
  samples: TelemetrySample[];
  labels: unknown[];
  metadata?: SessionMetadata;
}
