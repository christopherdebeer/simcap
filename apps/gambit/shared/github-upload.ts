/**
 * GitHub Upload Utility
 *
 * Uploads session data files to GitHub using the Contents API.
 * Files are committed directly to a specified branch (e.g., 'data').
 *
 * This module supports three modes:
 * 1. Local filesystem (for local development - auto-detected)
 * 2. Proxied upload via API endpoint (for production - keeps PAT server-side)
 * 3. Direct upload with GitHub PAT (legacy, requires client-side token)
 *
 * In local mode, uploads are redirected to /api/local-storage for direct
 * filesystem writes, bypassing GitHub entirely.
 *
 * Usage:
 *   import { uploadToGitHub, uploadViaProxy } from './shared/github-upload.js';
 *
 *   // Smart upload (auto-detects local vs remote)
 *   const result = await uploadSessionSmart({
 *     filename: '2025-01-01T00_00_00.000Z.json',
 *     content: '{"version": "2.1", ...}'
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
const LOCAL_STORAGE_ENDPOINT = '/api/local-storage';

// ===== Local Mode Detection =====

let localModeCache: boolean | null = null;
let localModeCheckPromise: Promise<boolean> | null = null;

/**
 * Check if we're running in local development mode
 * Returns true if running on localhost and local storage API is available
 */
export async function isLocalMode(forceCheck = false): Promise<boolean> {
  // Return cached value if available and not forcing a check
  if (localModeCache !== null && !forceCheck) {
    return localModeCache;
  }

  // Check if we're on localhost
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (!['localhost', '127.0.0.1', '::1'].includes(hostname)) {
      localModeCache = false;
      return false;
    }
  }

  // If already checking, wait for that check
  if (localModeCheckPromise) {
    return localModeCheckPromise;
  }

  // Check server endpoint
  localModeCheckPromise = (async () => {
    try {
      const response = await fetch(LOCAL_STORAGE_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'status' })
      });

      if (response.ok) {
        const data = await response.json();
        localModeCache = data.mode === 'local';
        return localModeCache;
      }

      localModeCache = false;
      return false;
    } catch {
      localModeCache = false;
      return false;
    } finally {
      localModeCheckPromise = null;
    }
  })();

  return localModeCheckPromise;
}

/**
 * Clear the local mode cache (for testing)
 */
export function clearLocalModeCache(): void {
  localModeCache = null;
  localModeCheckPromise = null;
}

// Compression threshold: compress if content exceeds 100KB
const COMPRESSION_THRESHOLD = 100 * 1024;

// Chunk size targets (in bytes)
// Target ~2MB per chunk to stay safely under Vercel's 4.5MB limit
// After base64 encoding (+33%) and JSON wrapper, this keeps us safe
const DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024; // 2MB
const MIN_CHUNK_SIZE = 256 * 1024; // 256KB minimum
const CHUNK_REDUCTION_FACTOR = 0.5; // Halve chunk size on 413 error

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
  /** For chunked uploads: manifest with all chunk URLs */
  chunks?: ChunkManifest;
}

/** Manifest for chunked uploads */
export interface ChunkManifest {
  /** Total number of chunks */
  totalChunks: number;
  /** Original filename (without chunk suffix) */
  originalFilename: string;
  /** Timestamp of the session */
  timestamp: string;
  /** URLs of all chunk files */
  chunkUrls: string[];
  /** Sample counts per chunk */
  sampleCounts: number[];
  /** Total sample count across all chunks */
  totalSamples: number;
}

/** Options for chunked session upload */
export interface ChunkedUploadOptions {
  branch?: string;
  filename: string;
  content: string;
  maxRetries?: number;
  initialChunkSize?: number;
  onProgress?: (progress: UploadProgress) => void;
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
 * Compress a string using gzip and return base64-encoded result
 * Uses native CompressionStream API (available in modern browsers)
 */
async function compressToBase64(str: string): Promise<string> {
  const bytes = new TextEncoder().encode(str);
  const cs = new CompressionStream('gzip');
  const writer = cs.writable.getWriter();
  writer.write(bytes);
  writer.close();

  const compressedChunks: Uint8Array[] = [];
  const reader = cs.readable.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    compressedChunks.push(value);
  }

  // Concatenate chunks
  const totalLength = compressedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const compressed = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of compressedChunks) {
    compressed.set(chunk, offset);
    offset += chunk.length;
  }

  // Convert to base64
  const binString = Array.from(compressed, (byte) => String.fromCodePoint(byte)).join('');
  return btoa(binString);
}

/**
 * Check if compression is available (CompressionStream API)
 */
function isCompressionAvailable(): boolean {
  return typeof CompressionStream !== 'undefined';
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

// ===== Chunking Helpers =====

/**
 * Estimate the JSON size of an object in bytes
 */
function estimateJsonSize(obj: unknown): number {
  return new TextEncoder().encode(JSON.stringify(obj)).length;
}

/**
 * Parse session data from JSON string
 */
function parseSessionData(content: string): {
  version: string;
  timestamp: string;
  samples: unknown[];
  labels: unknown[];
  metadata?: unknown;
} | null {
  try {
    const data = JSON.parse(content);
    if (data && Array.isArray(data.samples)) {
      return data;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Split session samples into chunks that fit within target size
 */
function splitIntoChunks(
  samples: unknown[],
  targetChunkSize: number,
  baseOverhead: number
): unknown[][] {
  if (samples.length === 0) {
    return [[]];
  }

  // Estimate average sample size from first few samples
  const sampleSlice = samples.slice(0, Math.min(10, samples.length));
  const avgSampleSize = estimateJsonSize(sampleSlice) / sampleSlice.length;

  // Calculate samples per chunk (leaving room for JSON structure overhead)
  const availableSize = targetChunkSize - baseOverhead - 1000; // 1KB safety margin
  const samplesPerChunk = Math.max(1, Math.floor(availableSize / avgSampleSize));

  const chunks: unknown[][] = [];
  for (let i = 0; i < samples.length; i += samplesPerChunk) {
    chunks.push(samples.slice(i, i + samplesPerChunk));
  }

  return chunks;
}

/**
 * Create a chunk payload from session data
 */
function createChunkPayload(
  sessionData: {
    version: string;
    timestamp: string;
    samples: unknown[];
    labels: unknown[];
    metadata?: unknown;
  },
  chunkSamples: unknown[],
  chunkIndex: number,
  totalChunks: number,
  startIndex: number
): string {
  const chunkData = {
    version: sessionData.version,
    timestamp: sessionData.timestamp,
    samples: chunkSamples,
    labels: sessionData.labels, // Include all labels in each chunk for simplicity
    metadata: {
      ...(sessionData.metadata as object || {}),
      chunk_info: {
        chunk_index: chunkIndex,
        total_chunks: totalChunks,
        start_sample_index: startIndex,
        sample_count: chunkSamples.length,
      },
    },
  };
  return JSON.stringify(chunkData, null, 2);
}

/**
 * Check if an error is a 413 Payload Too Large error
 */
function is413Error(error: Error): boolean {
  const msg = error.message.toLowerCase();
  return msg.includes('413') ||
         msg.includes('payload too large') ||
         msg.includes('request entity too large') ||
         msg.includes('body exceeded');
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
 * Automatically compresses large payloads to avoid Vercel's 4.5MB limit.
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

  // Use compression for large payloads (>100KB) to avoid Vercel's size limit
  const shouldCompress = contentSize > COMPRESSION_THRESHOLD && isCompressionAvailable();
  let uploadContent: string;
  let compressed = false;

  if (shouldCompress) {
    onProgress?.({ stage: 'preparing', message: `Compressing ${(contentSize / 1024).toFixed(0)}KB...` });
    uploadContent = await compressToBase64(content);
    compressed = true;
    const compressedSize = uploadContent.length;
    const ratio = ((1 - compressedSize / contentSize) * 100).toFixed(0);
    onProgress?.({ stage: 'preparing', message: `Compressed to ${(compressedSize / 1024).toFixed(0)}KB (${ratio}% reduction)` });
  } else {
    uploadContent = content;
  }

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
      content: uploadContent,
      message,
      compressed, // Signal to server that content is gzip+base64 encoded
    }),
  });

  if (!response.ok) {
    // Handle non-JSON error responses (e.g., HTML error pages)
    let errorMessage: string;
    const responseText = await response.text().catch(() => '');
    try {
      const error = JSON.parse(responseText);
      errorMessage = error.error || error.message || `HTTP ${response.status}`;
    } catch {
      // Response wasn't JSON - likely an HTML error page or network issue
      errorMessage = `HTTP ${response.status}: ${response.statusText || 'Upload failed'}${responseText ? ` - ${responseText.slice(0, 100)}` : ''}`;
    }
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
 * Upload file to local filesystem via API
 *
 * Only works in local development mode. Uses /api/local-storage endpoint.
 */
export async function uploadToLocal(
  options: ProxyUploadOptions
): Promise<GitHubUploadResult> {
  const { path, content, onProgress } = options;

  onProgress?.({ stage: 'preparing', message: 'Preparing local write...' });

  const contentSize = new TextEncoder().encode(content).length;
  const filename = path.split('/').pop() || path;

  onProgress?.({ stage: 'uploading', message: 'Writing to local filesystem...' });

  const response = await fetch(LOCAL_STORAGE_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      action: 'write',
      filename,
      content,
    }),
  });

  if (!response.ok) {
    let errorMessage: string;
    try {
      const error = await response.json();
      errorMessage = error.error || `HTTP ${response.status}`;
    } catch {
      errorMessage = `HTTP ${response.status}: Local write failed`;
    }
    onProgress?.({ stage: 'error', message: `Local write failed: ${errorMessage}` });
    throw new Error(errorMessage);
  }

  const result = await response.json();

  onProgress?.({
    stage: 'complete',
    message: 'Local write complete!',
    url: result.path,
  });

  return {
    success: true,
    url: `file://${result.path}`,
    pathname: path,
    size: contentSize,
    filename,
    rawUrl: `file://${result.path}`,
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

/**
 * Upload a single chunk with retry logic
 * Used internally by uploadSessionChunked
 */
async function uploadChunkWithRetry(
  branch: string,
  path: string,
  content: string,
  message: string,
  maxRetries: number,
  onProgress?: (progress: UploadProgress) => void
): Promise<GitHubUploadResult> {
  let lastError: Error | undefined;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
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

      // Don't retry 413 errors at this level - let caller handle chunking
      if (is413Error(lastError)) {
        throw lastError;
      }

      if (attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000;
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }

  throw lastError;
}

/**
 * Upload session data in chunks
 * Splits large sessions into multiple files to avoid payload limits
 */
export async function uploadSessionChunked(
  options: ChunkedUploadOptions
): Promise<GitHubUploadResult> {
  const {
    branch = 'data',
    filename,
    content,
    maxRetries = 3,
    initialChunkSize = DEFAULT_CHUNK_SIZE,
    onProgress,
  } = options;

  // Parse the session data
  const sessionData = parseSessionData(content);
  if (!sessionData) {
    throw new Error('Invalid session data format - cannot parse for chunking');
  }

  const totalSamples = sessionData.samples.length;
  let currentChunkSize = initialChunkSize;

  // Calculate base overhead (everything except samples)
  const baseSession = {
    version: sessionData.version,
    timestamp: sessionData.timestamp,
    samples: [],
    labels: sessionData.labels,
    metadata: sessionData.metadata,
  };
  const baseOverhead = estimateJsonSize(baseSession);

  onProgress?.({
    stage: 'preparing',
    message: `Preparing upload: ${totalSamples} samples...`,
  });

  // Try progressively smaller chunk sizes until upload succeeds
  while (currentChunkSize >= MIN_CHUNK_SIZE) {
    try {
      const sampleChunks = splitIntoChunks(
        sessionData.samples,
        currentChunkSize,
        baseOverhead
      );
      const totalChunks = sampleChunks.length;

      onProgress?.({
        stage: 'preparing',
        message: `Splitting into ${totalChunks} chunks (~${(currentChunkSize / 1024 / 1024).toFixed(1)}MB each)...`,
      });

      // For single chunk, just upload directly
      if (totalChunks === 1) {
        onProgress?.({
          stage: 'uploading',
          message: 'Uploading (single file)...',
        });

        return await uploadChunkWithRetry(
          branch,
          `GAMBIT/${filename}`,
          content,
          `GAMBIT session: ${filename}`,
          maxRetries,
          onProgress
        );
      }

      // Upload chunks
      const chunkUrls: string[] = [];
      const sampleCounts: number[] = [];
      const baseFilename = filename.replace(/\.json$/, '');
      let sampleIndex = 0;

      for (let i = 0; i < totalChunks; i++) {
        const chunkFilename = `${baseFilename}_part_${String(i + 1).padStart(3, '0')}.json`;
        const chunkContent = createChunkPayload(
          sessionData,
          sampleChunks[i],
          i,
          totalChunks,
          sampleIndex
        );

        onProgress?.({
          stage: 'uploading',
          message: `Uploading chunk ${i + 1}/${totalChunks}...`,
        });

        const result = await uploadChunkWithRetry(
          branch,
          `GAMBIT/${chunkFilename}`,
          chunkContent,
          `GAMBIT session: ${filename} (part ${i + 1}/${totalChunks})`,
          maxRetries,
          onProgress
        );

        chunkUrls.push(result.url || result.rawUrl || '');
        sampleCounts.push(sampleChunks[i].length);
        sampleIndex += sampleChunks[i].length;
      }

      // Upload manifest file
      const manifest: ChunkManifest = {
        totalChunks,
        originalFilename: filename,
        timestamp: sessionData.timestamp,
        chunkUrls,
        sampleCounts,
        totalSamples,
      };

      const manifestFilename = `${baseFilename}_manifest.json`;
      const manifestContent = JSON.stringify(
        {
          version: sessionData.version,
          timestamp: sessionData.timestamp,
          type: 'chunked_session_manifest',
          manifest,
          metadata: sessionData.metadata,
        },
        null,
        2
      );

      onProgress?.({
        stage: 'uploading',
        message: 'Uploading manifest...',
      });

      const manifestResult = await uploadChunkWithRetry(
        branch,
        `GAMBIT/${manifestFilename}`,
        manifestContent,
        `GAMBIT session manifest: ${filename}`,
        maxRetries,
        onProgress
      );

      onProgress?.({
        stage: 'complete',
        message: `Upload complete! ${totalChunks} chunks uploaded.`,
        url: manifestResult.url,
      });

      return {
        success: true,
        url: manifestResult.url || '',
        pathname: `GAMBIT/${manifestFilename}`,
        size: new TextEncoder().encode(content).length,
        filename: manifestFilename,
        commitSha: manifestResult.commitSha,
        htmlUrl: manifestResult.htmlUrl,
        rawUrl: manifestResult.rawUrl,
        chunks: manifest,
      };

    } catch (error) {
      const uploadError = error instanceof Error ? error : new Error(String(error));

      // If we get a 413 error, reduce chunk size and retry
      if (is413Error(uploadError)) {
        const previousSize = currentChunkSize;
        currentChunkSize = Math.floor(currentChunkSize * CHUNK_REDUCTION_FACTOR);

        if (currentChunkSize >= MIN_CHUNK_SIZE) {
          onProgress?.({
            stage: 'retry',
            message: `Payload too large. Reducing chunk size from ${(previousSize / 1024 / 1024).toFixed(1)}MB to ${(currentChunkSize / 1024 / 1024).toFixed(1)}MB...`,
          });
          continue; // Retry with smaller chunks
        }
      }

      // Other errors or chunk size too small - propagate
      throw uploadError;
    }
  }

  throw new Error('Unable to upload: chunks too small. Session data may be too complex.');
}

/**
 * Smart upload that automatically chooses the best upload method
 *
 * Priority:
 * 1. Local mode: Write directly to filesystem (no auth required)
 * 2. Proxy mode: Upload via API proxy (server-side auth)
 * 3. Chunked mode: Split large files and upload in parts
 */
export async function uploadSessionSmart(options: {
  branch?: string;
  filename: string;
  content: string;
  maxRetries?: number;
  forceRemote?: boolean;
  onProgress?: (progress: UploadProgress) => void;
}): Promise<GitHubUploadResult> {
  const {
    branch = 'data',
    filename,
    content,
    maxRetries = 3,
    forceRemote = false,
    onProgress,
  } = options;

  const contentSize = new TextEncoder().encode(content).length;

  // Check for local mode first (unless explicitly forced to remote)
  if (!forceRemote) {
    const isLocal = await isLocalMode();
    if (isLocal) {
      onProgress?.({
        stage: 'preparing',
        message: `Local mode detected, writing to filesystem (${(contentSize / 1024).toFixed(1)}KB)...`,
      });

      try {
        return await uploadToLocal({
          branch,
          path: `GAMBIT/${filename}`,
          content,
          message: `GAMBIT session: ${filename}`,
          onProgress,
        });
      } catch (error) {
        // Log but don't fail - fall back to remote upload
        console.warn('[uploadSessionSmart] Local write failed, falling back to remote:', error);
        onProgress?.({
          stage: 'retry',
          message: 'Local write failed, trying remote upload...',
        });
      }
    }
  }

  // If content is small enough, try direct upload first
  if (contentSize < DEFAULT_CHUNK_SIZE) {
    try {
      return await uploadSessionWithRetry({
        branch,
        filename,
        content,
        maxRetries,
        onProgress,
      });
    } catch (error) {
      const uploadError = error instanceof Error ? error : new Error(String(error));

      // Fall back to chunked upload on 413 error
      if (is413Error(uploadError)) {
        onProgress?.({
          stage: 'retry',
          message: 'Payload too large, switching to chunked upload...',
        });
        return await uploadSessionChunked({
          branch,
          filename,
          content,
          maxRetries,
          onProgress,
        });
      }

      throw uploadError;
    }
  }

  // Large content - go directly to chunked upload
  onProgress?.({
    stage: 'preparing',
    message: `Large session (${(contentSize / 1024 / 1024).toFixed(1)}MB), using chunked upload...`,
  });

  return await uploadSessionChunked({
    branch,
    filename,
    content,
    maxRetries,
    onProgress,
  });
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
  // Local mode
  isLocalMode,
  clearLocalModeCache,
  uploadToLocal,
  // Secret management
  getUploadSecret,
  setUploadSecret,
  hasUploadSecret,
  // Upload functions
  uploadToGitHub,
  uploadViaProxy,
  uploadSessionWithRetry,
  uploadSessionChunked,
  uploadSessionSmart,
  // User interaction
  promptForUploadSecret,
  validateUploadSecret,
  getRawUrl,
};
