/**
 * Vercel Blob Upload Utility
 *
 * Uploads session data files to Vercel Blob storage using client-side upload.
 * Uses the two-phase upload protocol from @vercel/blob/client.
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

import { upload } from 'https://esm.sh/@vercel/blob@0.27.1/client';

const STORAGE_KEY = 'simcap_upload_secret';
const UPLOAD_API_ENDPOINT = '/api/upload';

/**
 * Get the stored upload secret
 * @returns {string|null} The stored secret or null
 */
export function getUploadSecret() {
    try {
        return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
        console.warn('localStorage not available:', e);
        return null;
    }
}

/**
 * Set the upload secret
 * @param {string} secret - The secret to store
 */
export function setUploadSecret(secret) {
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
 * @returns {boolean} True if secret is set
 */
export function hasUploadSecret() {
    const secret = getUploadSecret();
    return secret !== null && secret.length > 0;
}

/**
 * Upload file to Vercel Blob using client-side upload
 * @param {Object} options - Upload options
 * @param {string} options.filename - Filename for the blob (e.g., '2025-01-01T00_00_00.000Z.json')
 * @param {string} options.content - File content to upload
 * @param {Function} [options.onProgress] - Progress callback
 * @returns {Promise<Object>} Upload result with url, pathname, size
 */
export async function uploadToBlob(options) {
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
        // Use the @vercel/blob/client upload function for two-phase upload
        const result = await upload(pathname, blob, {
            access: 'public',
            handleUploadUrl: UPLOAD_API_ENDPOINT,
            clientPayload: JSON.stringify({ filename }),
            multipart: false,
            // Pass the secret header to the handleUpload endpoint
            fetch: (url, init) => {
                return fetch(url, {
                    ...init,
                    headers: {
                        ...init?.headers,
                        'x-upload-secret': secret,
                    },
                });
            },
        });

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
        onProgress?.({
            stage: 'error',
            message: `Upload failed: ${error.message}`
        });
        throw error;
    }
}

/**
 * Upload session data with retry logic
 * @param {Object} options - Upload options
 * @param {string} options.filename - Filename for the blob
 * @param {string} options.content - File content to upload
 * @param {number} [options.maxRetries=3] - Maximum retry attempts
 * @param {Function} [options.onProgress] - Progress callback
 * @returns {Promise<Object>} Upload result
 */
export async function uploadSessionWithRetry(options) {
    const { filename, content, maxRetries = 3, onProgress } = options;

    let lastError;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            onProgress?.({
                stage: 'attempt',
                message: `Upload attempt ${attempt}/${maxRetries}...`
            });

            return await uploadToBlob({ filename, content, onProgress });
        } catch (error) {
            lastError = error;

            // Don't retry auth errors
            if (error.message.includes('Unauthorized') || error.message.includes('secret')) {
                throw error;
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

/**
 * Prompt user for upload secret via browser prompt
 * @returns {boolean} True if secret was set
 */
export function promptForUploadSecret() {
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
 * @returns {Promise<boolean>} True if secret is valid
 */
export async function validateUploadSecret() {
    const secret = getUploadSecret();
    if (!secret) {
        return false;
    }

    try {
        // Make a minimal request to check auth
        // The two-phase protocol requires specific body format,
        // so we just check if auth passes (non-401 response)
        const response = await fetch(UPLOAD_API_ENDPOINT, {
            method: 'POST',
            headers: {
                'x-upload-secret': secret,
                'content-type': 'application/json',
            },
            body: JSON.stringify({ type: 'blob.generate-client-token', payload: { pathname: 'sessions/.validate', callbackUrl: '' } }),
        });

        // 401 means auth failed, anything else means auth passed
        return response.status !== 401;
    } catch (e) {
        return false;
    }
}
