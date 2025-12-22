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
  filepath?: string;
  time_start?: number;
  time_end?: number;
  sample_count?: number;
  composite?: string;
  images: Record<string, string>;
  trajectory_images: Record<string, string>;
}

/** Visualization manifest (stored in blob storage) */
export interface VisualizationManifest {
  /** Manifest schema version */
  version: '1.0';
  /** Session timestamp (with colons) */
  sessionTimestamp: string;
  /** When this manifest was generated (ISO string) */
  generatedAt: string;
  /** Unique manifest ID: {session_ts}_{generated_ts} */
  manifestId: string;
  /** Session metadata */
  session: {
    filename: string;
    duration: number;
    sample_count: number;
    sample_rate: number;
    device?: string;
    firmware_version?: string | null;
    session_type?: string;
    hand?: string;
    magnet_type?: string;
    notes?: string | null;
    custom_labels?: string[];
  };
  /** Session-level images */
  images: {
    composite?: string;
    calibration_stages?: string;
    orientation_3d?: string;
    orientation_track?: string;
    raw_axes?: string;
  };
  /** Session-level trajectory comparison images */
  trajectory_comparison: Record<string, string>;
  /** Per-window visualizations */
  windows: WindowEntry[];
}

/** Session summary for listing (derived from manifest) */
export interface VisualizationSessionSummary {
  /** Session timestamp */
  sessionTimestamp: string;
  /** Latest manifest ID */
  latestManifestId: string;
  /** When latest manifest was generated */
  generatedAt: string;
  /** Number of previous manifest versions */
  previousVersions: number;
  /** Session filename */
  filename: string;
  /** Session duration */
  duration?: number;
  /** Number of windows */
  windowCount: number;
  /** Whether session has visualizations */
  hasVisualizations: boolean;
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

// ===== GitHub Storage Types =====

/** GitHub repository configuration */
export interface GitHubRepoConfig {
  owner: string;
  repo: string;
  dataBranch: string;
  imagesBranch: string;
}

/** Default GitHub configuration */
export const DEFAULT_GITHUB_CONFIG: GitHubRepoConfig = {
  owner: 'christopherdebeer',
  repo: 'simcap',
  dataBranch: 'data',
  imagesBranch: 'images',
};

/** GitHub upload request (for API proxy) */
export interface GitHubUploadRequest {
  secret: string;
  branch: string;
  path: string;
  content: string;
  message: string;
  validate?: boolean;
}

/** GitHub upload response */
export interface GitHubUploadResponse {
  success: boolean;
  url: string;
  pathname: string;
  branch: string;
  commitSha?: string;
  htmlUrl?: string;
}

/** GitHub-based session info (extends SessionInfo with GitHub URLs) */
export interface GitHubSessionInfo extends SessionInfo {
  /** Branch where session is stored */
  branch: string;
  /** Raw GitHub URL */
  rawUrl: string;
}

/** Session manifest stored in main branch */
export interface SessionManifest {
  generated: string;
  directory: string;
  branch: string;
  baseUrl: string;
  sessionCount: number;
  sessions: Array<{
    filename: string;
    timestamp: string;
    size: number;
    version?: string;
    sampleCount?: number;
    durationSec?: number;
    url: string;
  }>;
}

/** Visualization manifest index (stored in main branch) */
export interface VisualizationManifestIndex {
  generated: string;
  imageBranch: string;
  baseImageUrl: string;
  sessions: Array<{
    sessionTimestamp: string;
    generatedAt: string;
    hasComposite: boolean;
    windowCount: number;
    manifestPath: string;
  }>;
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

// ===== Session Payload Types (for uploaded JSON content) =====
// Note: These are API-specific types for the upload payload format.
// For core domain types, use @core/types (SessionData, TelemetrySample, etc.)

/** Raw telemetry sample in upload payload */
export interface UploadTelemetrySample {
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

/** Session metadata in upload payload */
export interface UploadSessionMetadata {
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

/**
 * Session upload payload (v2.1 schema)
 * This represents the JSON structure sent to the upload API.
 * Distinct from core SessionData which is the full domain model.
 */
export interface SessionPayload {
  version: string;
  timestamp: string;
  samples: UploadTelemetrySample[];
  labels: unknown[];
  metadata?: UploadSessionMetadata;
}
