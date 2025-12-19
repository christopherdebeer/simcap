/**
 * GitHub Upload Utility
 *
 * Uploads session data files to GitHub using the Contents API.
 * Files are committed directly to a specified branch (e.g., 'data').
 *
 * This module supports two modes:
 * 1. Direct upload with GitHub PAT (for local development)
 * 2. Proxied upload via API endpoint (for production - keeps PAT server-side)
 *
 * Usage:
 *   import { uploadToGitHub, uploadViaProxy } from './shared/github-upload.js';
 *
 *   // Direct upload (requires PAT)
 *   const result = await uploadToGitHub({
 *     token: 'ghp_xxx',
 *     branch: 'data',
 *     path: 'GAMBIT/2025-01-01T00_00_00.000Z.json',
 *     content: '{"version": "2.1", ...}',
 *     message: 'GAMBIT Data ingest'
 *   });
 *
 *   // Proxied upload (uses server-side PAT)
 *   const result = await uploadViaProxy({
 *     branch: 'data',
 *     path: 'GAMBIT/2025-01-01T00_00_00.000Z.json',
 *     content: '{"version": "2.1", ...}',
 *     message: 'GAMBIT Data ingest'
 *   });
 */

import type { UploadProgress, UploadResult } from '@api/types';

// Re-export types for convenience
export type { UploadProgress, UploadResult };

// ===== Constants =====

const GITHUB_API_URL = 'https://api.github.com';
const DEFAULT_OWNER = 'christopherdebeer';
const DEFAULT_REPO = 'simcap';
const STORAGE_KEY = 'simcap_upload_secret';
const UPLOAD_API_ENDPOINT = '/api/github-upload';

// ===== Types =====

export interface GitHubUploadOptions {
  /** GitHub personal access token */
  token: string;
  /** Repository owner (default: christopherdebeer) */
  owner?: string;
  /** Repository name (default: simcap) */
  repo?: string;
  /** Target branch (e.g., 'data', 'images') */
  branch: string;
  /** File path within repository */
  path: string;
  /** File content (string or base64 for binary) */
  content: string;
  /** Whether content is already base64 encoded */
  isBase64?: boolean;
  /** Commit message */
  message: string;
  /** Progress callback */
  onProgress?: (progress: UploadProgress) => void;
}

export interface ProxyUploadOptions {
  /** Target branch (e.g., 'data', 'images') */
  branch: string;
  /** File path within repository */
  path: string;
  /** File content */
  content: string;
  /** Commit message */
  message: string;
  /** Progress callback */
  onProgress?: (progress: UploadProgress) => void;
}

export interface GitHubUploadResult extends UploadResult {
  /** Git commit SHA */
  commitSha?: string;
  /** GitHub HTML URL to view the file */
  htmlUrl?: string;
  /** Raw content URL */
  rawUrl?: string;
}

interface GitHubContentResponse {
  content?: {
    name: string;
    path: string;
    sha: string;
    html_url: string;
    download_url: string;
  };
  commit: {
    sha: string;
    html_url: string;
  };
}

// ===== Secret Management =====

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
    // localStorage not available
  }
}

/**
 * Check if upload secret is configured
 */
export function hasUploadSecret(): boolean {
  const secret = getUploadSecret();
  return secret !== null && secret.length > 0;
}

// ===== Helper Functions =====

/**
 * Base64 encode a string (handles Unicode correctly)
 */
function base64Encode(str: string): string {
  // Convert to UTF-8 bytes then to base64
  const bytes = new TextEncoder().encode(str);
  const binString = Array.from(bytes, (byte) => String.fromCodePoint(byte)).join('');
  return btoa(binString);
}

/**
 * Get the SHA of an existing file (needed for updates)
 */
async function getFileSha(
  owner: string,
  repo: string,
  path: string,
  branch: string,
  token: string
): Promise<string | null> {
  const url = `${GITHUB_API_URL}/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `token ${token}`,
        Accept: 'application/vnd.github.v3+json',
      },
    });

    if (response.ok) {
      const data = await response.json();
      return data.sha;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Construct raw.githubusercontent.com URL for a file
 */
export function getRawUrl(
  owner: string,
  repo: string,
  branch: string,
  path: string
): string {
  return `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${path}`;
}

// ===== Upload Functions =====

/**
 * Upload file directly to GitHub using Contents API
 *
 * Requires a GitHub PAT with repo write access.
 */
export async function uploadToGitHub(
  options: GitHubUploadOptions
): Promise<GitHubUploadResult> {
  const {
    token,
    owner = DEFAULT_OWNER,
    repo = DEFAULT_REPO,
    branch,
    path,
    content,
    isBase64 = false,
    message,
    onProgress,
  } = options;

  onProgress?.({ stage: 'preparing', message: 'Preparing upload...' });

  // Encode content if not already base64
  const encodedContent = isBase64 ? content : base64Encode(content);
  const contentSize = new TextEncoder().encode(content).length;

  // Check if file already exists (for update)
  onProgress?.({ stage: 'preparing', message: 'Checking for existing file...' });
  const existingSha = await getFileSha(owner, repo, path, branch, token);

  // Prepare request body
  const body: Record<string, string> = {
    message,
    content: encodedContent,
    branch,
  };

  if (existingSha) {
    body.sha = existingSha;
  }

  onProgress?.({ stage: 'uploading', message: 'Committing to GitHub...' });

  // Create or update file
  const url = `${GITHUB_API_URL}/repos/${owner}/${repo}/contents/${path}`;
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      Authorization: `token ${token}`,
      'Content-Type': 'application/json',
      Accept: 'application/vnd.github.v3+json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json();
    const errorMessage = error.message || `HTTP ${response.status}`;
    onProgress?.({ stage: 'error', message: `Upload failed: ${errorMessage}` });
    throw new Error(errorMessage);
  }

  const result: GitHubContentResponse = await response.json();
  const rawUrl = getRawUrl(owner, repo, branch, path);

  onProgress?.({
    stage: 'complete',
    message: 'Upload complete!',
    url: rawUrl,
  });

  return {
    success: true,
    url: rawUrl,
    pathname: path,
    size: contentSize,
    filename: path.split('/').pop() || path,
    commitSha: result.commit?.sha,
    htmlUrl: result.content?.html_url,
    rawUrl,
  };
}

/**
 * Upload file via API proxy (keeps GitHub PAT server-side)
 *
 * Requires upload secret for authentication.
 */
export async function uploadViaProxy(
  options: ProxyUploadOptions
): Promise<GitHubUploadResult> {
  const { branch, path, content, message, onProgress } = options;

  const secret = getUploadSecret();
  if (!secret) {
    throw new Error('Upload secret not configured. Call setUploadSecret() first.');
  }

  onProgress?.({ stage: 'preparing', message: 'Preparing upload...' });

  const contentSize = new TextEncoder().encode(content).length;

  onProgress?.({ stage: 'uploading', message: 'Uploading via API...' });

  const response = await fetch(UPLOAD_API_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      secret,
      branch,
      path,
      content,
      message,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    const errorMessage = error.error || error.message || `HTTP ${response.status}`;
    onProgress?.({ stage: 'error', message: `Upload failed: ${errorMessage}` });
    throw new Error(errorMessage);
  }

  const result = await response.json();

  onProgress?.({
    stage: 'complete',
    message: 'Upload complete!',
    url: result.url,
  });

  return {
    success: true,
    url: result.url,
    pathname: path,
    size: contentSize,
    filename: path.split('/').pop() || path,
    commitSha: result.commitSha,
    htmlUrl: result.htmlUrl,
    rawUrl: result.url,
  };
}

/**
 * Upload session data with retry logic
 */
export async function uploadSessionWithRetry(options: {
  branch?: string;
  filename: string;
  content: string;
  maxRetries?: number;
  onProgress?: (progress: UploadProgress) => void;
}): Promise<GitHubUploadResult> {
  const {
    branch = 'data',
    filename,
    content,
    maxRetries = 3,
    onProgress,
  } = options;

  const path = `GAMBIT/${filename}`;
  const message = `GAMBIT session: ${filename}`;

  let lastError: Error | undefined;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      onProgress?.({
        stage: 'attempt',
        message: `Upload attempt ${attempt}/${maxRetries}...`,
      });

      return await uploadViaProxy({
        branch,
        path,
        content,
        message,
        onProgress,
      });
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry auth errors
      if (
        lastError.message.includes('Unauthorized') ||
        lastError.message.includes('secret') ||
        lastError.message.includes('401')
      ) {
        throw lastError;
      }

      if (attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000;
        onProgress?.({
          stage: 'retry',
          message: `Retrying in ${delay / 1000}s...`,
        });
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }

  throw lastError;
}

// ===== User Interaction =====

/**
 * Prompt user for upload secret via browser prompt
 */
export function promptForUploadSecret(): boolean {
  const current = getUploadSecret();
  const message = current
    ? 'Update upload secret (leave empty to keep current):'
    : 'Enter upload secret for session uploads:';

  const secret = prompt(message);

  if (secret === null) {
    return false; // User cancelled
  }

  if (secret.length > 0) {
    setUploadSecret(secret);
    return true;
  }

  return current !== null;
}

/**
 * Validate upload secret with server
 */
export async function validateUploadSecret(): Promise<boolean> {
  const secret = getUploadSecret();
  if (!secret) {
    return false;
  }

  try {
    const response = await fetch(UPLOAD_API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        secret,
        validate: true, // Special flag for validation-only request
      }),
    });

    return response.ok;
  } catch {
    return false;
  }
}

// ===== Default Export =====

export default {
  getUploadSecret,
  setUploadSecret,
  hasUploadSecret,
  uploadToGitHub,
  uploadViaProxy,
  uploadSessionWithRetry,
  promptForUploadSecret,
  validateUploadSecret,
  getRawUrl,
};
