/**
 * Vercel Blob Upload Utility
 *
 * Uploads session data files to Vercel Blob storage using client-side upload.
 * Requires an upload secret stored in localStorage for authorization.
 *
 * Usage:
 *   import { uploadToBlob, getUploadSecret, setUploadSecret } from './shared/blob-upload.js';
 *
 *   // Set secret (one-time setup)
 *   setUploadSecret('your-secret');
 *
 *   // Upload session
 *   const result = await uploadToBlob({
 *     filename: '2025-01-01T00_00_00.000Z.json',
 *     content: '{"version": "2.1", ...}',
 *     onProgress: (progress) => console.log(progress)
 *   });
 */

// ===== Type Definitions =====

export type UploadStage = 'preparing' | 'uploading' | 'complete' | 'error' | 'attempt' | 'retry';

export interface UploadProgress {
  stage: UploadStage;
  message: string;
  url?: string;
}

export interface UploadOptions {
  filename: string;
  content: string;
  onProgress?: (progress: UploadProgress) => void;
}

export interface UploadWithRetryOptions extends UploadOptions {
  maxRetries?: number;
}

export interface UploadResult {
  success: boolean;
  url: string;
  pathname: string;
  size: number;
  filename: string;
}

// ===== Constants =====

const STORAGE_KEY = 'simcap_upload_secret';
const UPLOAD_API_ENDPOINT = '/api/upload';

// ===== Secret Management =====

/**
 * Get the stored upload secret
 * @returns The stored secret or null
 */
export function getUploadSecret(): string | null {
    try {
        return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
        console.warn('localStorage not available:', e);
        return null;
    }
}

/**
 * Set the upload secret
 * @param secret - The secret to store
 */
export function setUploadSecret(secret: string | null): void {
    try {
        if (secret) {
            localStorage.setItem(STORAGE_KEY, secret);
        } else {
            localStorage.removeItem(STORAGE_KEY);
        }
    } catch (e) {
        console.warn('localStorage not available:', e);
    }
}

/**
 * Check if upload secret is configured
 * @returns True if secret is set
 */
export function hasUploadSecret(): boolean {
    const secret = getUploadSecret();
    return secret !== null && secret.length > 0;
}

// ===== Upload Functions =====

/**
 * Upload file to Vercel Blob
 * @param options - Upload options
 * @returns Upload result with url, pathname, size
 */
export async function uploadToBlob(options: UploadOptions): Promise<UploadResult> {
    const { filename, content, onProgress } = options;

    const secret = getUploadSecret();
    if (!secret) {
        throw new Error('Upload secret not configured. Call setUploadSecret() first.');
    }

    onProgress?.({ stage: 'preparing', message: 'Preparing upload...' });

    // Create blob from content
    const blob = new Blob([content], { type: 'application/json' });
    const pathname = `sessions/${filename}`;

    onProgress?.({ stage: 'uploading', message: 'Uploading to Vercel Blob...' });

    try {
        // Use fetch to call our upload API with the secret header
        // This will get a signed URL and handle the upload
        const response = await fetch(UPLOAD_API_ENDPOINT, {
            method: 'POST',
            headers: {
                'x-upload-secret': secret,
                'x-vercel-blob-pathname': pathname,
                'content-type': 'application/json',
            },
            body: blob,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: response.statusText }));
            throw new Error((error as { error?: string }).error || `Upload failed: ${response.status}`);
        }

        const result = await response.json() as { url: string; pathname: string };

        onProgress?.({
            stage: 'complete',
            message: 'Upload complete!',
            url: result.url
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
            message: `Upload failed: ${errorMessage}`
        });
        throw error;
    }
}

/**
 * Upload session data with retry logic
 * @param options - Upload options
 * @returns Upload result
 */
export async function uploadSessionWithRetry(options: UploadWithRetryOptions): Promise<UploadResult> {
    const { filename, content, maxRetries = 3, onProgress } = options;

    let lastError: Error | undefined;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            onProgress?.({
                stage: 'attempt',
                message: `Upload attempt ${attempt}/${maxRetries}...`
            });

            return await uploadToBlob({ filename, content, onProgress });
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));

            // Don't retry auth errors
            if (lastError.message.includes('Unauthorized') || lastError.message.includes('secret')) {
                throw lastError;
            }

            if (attempt < maxRetries) {
                const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
                onProgress?.({
                    stage: 'retry',
                    message: `Retrying in ${delay/1000}s...`
                });
                await new Promise(r => setTimeout(r, delay));
            }
        }
    }

    throw lastError;
}

// ===== User Interaction =====

/**
 * Prompt user for upload secret via browser prompt
 * @returns True if secret was set
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

    return current !== null; // Keep existing if empty input
}

/**
 * Validate upload secret with server
 * @returns True if secret is valid
 */
export async function validateUploadSecret(): Promise<boolean> {
    const secret = getUploadSecret();
    if (!secret) {
        return false;
    }

    try {
        // Send minimal request to validate
        const response = await fetch(UPLOAD_API_ENDPOINT, {
            method: 'POST',
            headers: {
                'x-upload-secret': secret,
                'x-vercel-blob-pathname': 'sessions/.validate',
                'content-type': 'application/json',
            },
            body: '{}',
        });

        // 400 means auth passed but validation of content failed (expected)
        // 401 means auth failed
        return response.status !== 401;
    } catch (e) {
        return false;
    }
}

// ===== Default Export =====

export default {
    getUploadSecret,
    setUploadSecret,
    hasUploadSecret,
    uploadToBlob,
    uploadSessionWithRetry,
    promptForUploadSecret,
    validateUploadSecret
};
