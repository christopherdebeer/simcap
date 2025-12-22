/**
 * Typed API Client
 *
 * Provides type-safe access to SIMCAP APIs from client-side code.
 * Handles request/response serialization and error handling.
 */

import type {
  SessionsResponse,
  SessionInfo,
  VisualizationsListResponse,
  VisualizationSessionResponse,
  SessionVisualization,
  SessionEntry,
  SessionPayload,
  SessionData, // Deprecated alias, use SessionPayload
  ApiError,
  UploadOptions,
  UploadWithRetryOptions,
  UploadResult,
  UploadProgress,
  UploadClientPayload,
  VisualizationManifest,
  VisualizationSessionSummary,
} from './types';

// Re-export types for convenience
export type {
  SessionsResponse,
  SessionInfo,
  VisualizationsListResponse,
  VisualizationSessionResponse,
  SessionVisualization,
  SessionEntry,
  SessionPayload,
  SessionData, // Deprecated alias, use SessionPayload
  ApiError,
  UploadOptions,
  UploadWithRetryOptions,
  UploadResult,
  UploadProgress,
  UploadClientPayload,
  VisualizationManifest,
  VisualizationSessionSummary,
};

// ===== Configuration =====

export interface ApiClientConfig {
  /** Base URL for API endpoints (default: '') */
  baseUrl?: string;
  /** Default fetch options */
  fetchOptions?: RequestInit;
}

const defaultConfig: Required<ApiClientConfig> = {
  baseUrl: '',
  fetchOptions: {},
};

// ===== Error Handling =====

export class ApiClientError extends Error {
  constructor(
    message: string,
    public status: number,
    public response?: ApiError
  ) {
    super(message);
    this.name = 'ApiClientError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorData: ApiError | undefined;
    try {
      errorData = await response.json();
    } catch {
      // Response body not JSON
    }
    throw new ApiClientError(
      errorData?.error || `HTTP ${response.status}`,
      response.status,
      errorData
    );
  }
  return response.json();
}

// ===== API Client Class =====

export class ApiClient {
  private config: Required<ApiClientConfig>;

  constructor(config: ApiClientConfig = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  private url(path: string): string {
    return `${this.config.baseUrl}${path}`;
  }

  private async fetch<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(this.url(path), {
      ...this.config.fetchOptions,
      ...options,
    });
    return handleResponse<T>(response);
  }

  // ===== Sessions API =====

  /**
   * List all sessions
   */
  async listSessions(): Promise<SessionsResponse> {
    return this.fetch<SessionsResponse>('/api/sessions');
  }

  /**
   * Fetch session data by URL
   */
  async fetchSessionData(url: string): Promise<SessionPayload> {
    const response = await fetch(url);
    return handleResponse<SessionPayload>(response);
  }

  // ===== Visualizations API =====

  /**
   * List all visualization sessions (manifest-based)
   * Returns session summaries with manifest metadata
   */
  async listVisualizations(): Promise<{ sessions: VisualizationSessionSummary[]; count: number; generatedAt: string }> {
    return this.fetch<{ sessions: VisualizationSessionSummary[]; count: number; generatedAt: string }>('/api/visualizations');
  }

  /**
   * Get visualizations for a specific session
   * Returns both backward-compatible SessionVisualization and full manifest
   */
  async getSessionVisualization(sessionTimestamp: string): Promise<VisualizationSessionResponse & { manifest?: VisualizationManifest }> {
    const encoded = encodeURIComponent(sessionTimestamp);
    return this.fetch<VisualizationSessionResponse & { manifest?: VisualizationManifest }>(`/api/visualizations?session=${encoded}`);
  }

  /**
   * Get manifest history for a session
   */
  async getManifestHistory(sessionTimestamp: string): Promise<{
    sessionTimestamp: string;
    versions: Array<{ manifestId: string; generatedAt: string; url: string }>;
    count: number;
    generatedAt: string;
  }> {
    const encoded = encodeURIComponent(sessionTimestamp);
    return this.fetch(`/api/visualizations?session=${encoded}&history=true`);
  }

  // ===== Combined Session + Visualization Data =====

  /**
   * Fetch sessions with their visualization data combined
   * This merges data from /api/sessions and /api/visualizations
   * 
   * Note: With manifest-based system, this fetches session summaries first,
   * then lazily loads full manifests when needed.
   */
  async listSessionsWithVisualizations(): Promise<SessionEntry[]> {
    // Fetch both in parallel
    const [sessionsRes, vizRes] = await Promise.all([
      this.listSessions(),
      this.listVisualizations(),
    ]);

    // Create visualization lookup by timestamp
    const vizByTimestamp = new Map<string, VisualizationSessionSummary>();
    for (const viz of vizRes.sessions) {
      vizByTimestamp.set(viz.sessionTimestamp, viz);
    }

    // Merge sessions with visualization summaries
    // Note: Full visualization data requires fetching individual manifests
    return sessionsRes.sessions.map((session): SessionEntry => {
      const viz = vizByTimestamp.get(session.timestamp);
      return {
        ...session,
        sessionUrl: session.url,
        // These will be null until manifest is fetched
        composite_image: null,
        calibration_stages_image: null,
        orientation_3d_image: null,
        orientation_track_image: null,
        raw_axes_image: null,
        trajectory_comparison_images: {},
        windows: [],
        // Add manifest metadata for lazy loading
        hasVisualizations: viz?.hasVisualizations ?? false,
        manifestId: viz?.latestManifestId,
      } as SessionEntry & { hasVisualizations?: boolean; manifestId?: string };
    });
  }

  /**
   * Fetch full session entry with visualization data from manifest
   */
  async getSessionEntryWithVisualizations(sessionTimestamp: string): Promise<SessionEntry | null> {
    // Fetch session info and visualization manifest in parallel
    const [sessionsRes, vizRes] = await Promise.all([
      this.listSessions(),
      this.getSessionVisualization(sessionTimestamp),
    ]);

    // Find the session
    const session = sessionsRes.sessions.find(s => s.timestamp === sessionTimestamp);
    if (!session) return null;

    const viz = vizRes.session;
    
    return {
      ...session,
      sessionUrl: session.url,
      composite_image: viz?.composite_image ?? null,
      calibration_stages_image: viz?.calibration_stages_image ?? null,
      orientation_3d_image: viz?.orientation_3d_image ?? null,
      orientation_track_image: viz?.orientation_track_image ?? null,
      raw_axes_image: viz?.raw_axes_image ?? null,
      trajectory_comparison_images: viz?.trajectory_comparison_images ?? {},
      windows: viz?.windows ?? [],
    };
  }
}

// ===== Upload Secret Storage =====
// Note: Upload functionality is now in apps/gambit/shared/github-upload.ts

const STORAGE_KEY = 'simcap_upload_secret';

/**
 * Get the stored upload secret
 */
export function getUploadSecret(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * Set the upload secret
 */
export function setUploadSecret(secret: string | null): void {
  try {
    if (secret) {
      localStorage.setItem(STORAGE_KEY, secret);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    console.warn('localStorage not available');
  }
}

/**
 * Check if upload secret is configured
 */
export function hasUploadSecret(): boolean {
  const secret = getUploadSecret();
  return secret !== null && secret.length > 0;
}

// ===== Default Export =====

/** Default API client instance */
export const apiClient = new ApiClient();

export default apiClient;
