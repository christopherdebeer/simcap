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
  SessionData,
  ApiError,
  UploadOptions,
  UploadWithRetryOptions,
  UploadResult,
  UploadProgress,
  UploadClientPayload,
} from './types';

// Re-export types for convenience
export type {
  SessionsResponse,
  SessionInfo,
  VisualizationsListResponse,
  VisualizationSessionResponse,
  SessionVisualization,
  SessionEntry,
  SessionData,
  ApiError,
  UploadOptions,
  UploadWithRetryOptions,
  UploadResult,
  UploadProgress,
  UploadClientPayload,
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
  async fetchSessionData(url: string): Promise<SessionData> {
    const response = await fetch(url);
    return handleResponse<SessionData>(response);
  }

  // ===== Visualizations API =====

  /**
   * List all visualizations
   */
  async listVisualizations(): Promise<VisualizationsListResponse> {
    return this.fetch<VisualizationsListResponse>('/api/visualizations');
  }

  /**
   * Get visualizations for a specific session
   */
  async getSessionVisualization(sessionTimestamp: string): Promise<VisualizationSessionResponse> {
    const encoded = encodeURIComponent(sessionTimestamp);
    return this.fetch<VisualizationSessionResponse>(`/api/visualizations?session=${encoded}`);
  }

  // ===== Combined Session + Visualization Data =====

  /**
   * Fetch sessions with their visualization data combined
   * This merges data from /api/sessions and /api/visualizations
   */
  async listSessionsWithVisualizations(): Promise<SessionEntry[]> {
    // Fetch both in parallel
    const [sessionsRes, vizRes] = await Promise.all([
      this.listSessions(),
      this.listVisualizations(),
    ]);

    // Create visualization lookup by timestamp
    const vizByTimestamp = new Map<string, SessionVisualization>();
    for (const viz of vizRes.sessions) {
      vizByTimestamp.set(viz.timestamp, viz);
    }

    // Merge sessions with visualizations
    return sessionsRes.sessions.map((session): SessionEntry => {
      const viz = vizByTimestamp.get(session.timestamp);
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
    });
  }
}

// ===== Upload Functions =====

const STORAGE_KEY = 'simcap_upload_secret';
const UPLOAD_API_ENDPOINT = '/api/upload';

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

/**
 * Upload file to Vercel Blob using two-phase client upload
 *
 * Note: This requires @vercel/blob/client to be imported in the calling module
 * because it needs browser-specific functionality.
 */
export async function uploadToBlob(
  options: UploadOptions,
  uploadFn: (pathname: string, blob: Blob, config: { access: string; handleUploadUrl: string; clientPayload: string }) => Promise<{ url: string; pathname: string }>
): Promise<UploadResult> {
  const { filename, content, onProgress } = options;

  const secret = getUploadSecret();
  if (!secret) {
    throw new Error('Upload secret not configured. Call setUploadSecret() first.');
  }

  onProgress?.({ stage: 'preparing', message: 'Preparing upload...' });

  const blob = new Blob([content], { type: 'application/json' });
  const pathname = `sessions/${filename}`;

  onProgress?.({ stage: 'uploading', message: 'Uploading to Vercel Blob...' });

  try {
    const clientPayload: UploadClientPayload = { secret };
    const result = await uploadFn(pathname, blob, {
      access: 'public',
      handleUploadUrl: UPLOAD_API_ENDPOINT,
      clientPayload: JSON.stringify(clientPayload),
    });

    onProgress?.({
      stage: 'complete',
      message: 'Upload complete!',
      url: result.url,
    });

    return {
      success: true,
      url: result.url,
      pathname: result.pathname,
      size: blob.size,
      filename,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    onProgress?.({
      stage: 'error',
      message: `Upload failed: ${errorMessage}`,
    });
    throw error;
  }
}

/**
 * Upload with retry logic
 */
export async function uploadWithRetry(
  options: UploadWithRetryOptions,
  uploadFn: (pathname: string, blob: Blob, config: { access: string; handleUploadUrl: string; clientPayload: string }) => Promise<{ url: string; pathname: string }>
): Promise<UploadResult> {
  const { filename, content, maxRetries = 3, onProgress } = options;

  let lastError: Error | undefined;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      onProgress?.({
        stage: 'attempt',
        message: `Upload attempt ${attempt}/${maxRetries}...`,
      });

      return await uploadToBlob({ filename, content, onProgress }, uploadFn);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry auth errors
      if (lastError.message.includes('Unauthorized') || lastError.message.includes('secret')) {
        throw lastError;
      }

      if (attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000;
        onProgress?.({
          stage: 'retry',
          message: `Retrying in ${delay / 1000}s...`,
        });
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }

  throw lastError;
}

// ===== Default Export =====

/** Default API client instance */
export const apiClient = new ApiClient();

export default apiClient;
