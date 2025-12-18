/**
 * GitHub LFS Upload Utility
 *
 * @deprecated This module is deprecated. Use Vercel Blob storage instead.
 * Session data is now uploaded to Vercel Blob via the /api/upload endpoint.
 * See blob-upload.ts for the current upload implementation.
 *
 * This file is kept for backward compatibility only and may be removed
 * in a future release.
 *
 * Uploads files to GitHub using the LFS API to properly store large files.
 * This bypasses the GitHub Contents API which doesn't support LFS.
 *
 * Usage (DEPRECATED):
 *   import { uploadToGitHubLFS } from './shared/github-lfs-upload.js';
 *
 *   await uploadToGitHubLFS({
 *     token: 'ghp_xxx',
 *     owner: 'christopherdebeer',
 *     repo: 'simcap',
 *     path: 'data/GAMBIT/2025-01-01T00_00_00.000Z.json',
 *     content: '{"version": "2.1", ...}',
 *     message: 'GAMBIT Data ingest'
 *   });
 */

console.warn('[DEPRECATED] github-lfs-upload.js is deprecated. Use Vercel Blob storage instead.');

// ===== Type Definitions =====

export interface LFSUploadOptions {
    /** GitHub personal access token */
    token: string;
    /** Repository owner */
    owner: string;
    /** Repository name */
    repo: string;
    /** File path in repository */
    path: string;
    /** File content to upload */
    content: string;
    /** Commit message */
    message: string;
    /** Progress callback */
    onProgress?: (progress: LFSProgress) => void;
}

export interface LFSProgress {
    stage: 'hashing' | 'lfs-batch' | 'uploading' | 'uploaded' | 'exists' | 'committing' | 'complete';
    message: string;
}

export interface LFSUploadResult {
    success: boolean;
    filename: string;
    sha?: string;
    lfsOid: string;
    size: number;
    url?: string;
}

export interface ContentsUploadResult {
    success: boolean;
    filename?: string;
    sha?: string;
    url?: string;
}

interface LFSBatchResponse {
    objects?: Array<{
        oid: string;
        size: number;
        actions?: {
            upload?: {
                href: string;
                header?: Record<string, string>;
            };
            verify?: {
                href: string;
                header?: Record<string, string>;
            };
        };
    }>;
}

interface CommitBody {
    message: string;
    content: string;
    sha?: string;
}

// ===== Helper Functions =====

/**
 * Calculate SHA256 hash of content
 * @param content - The content to hash
 * @returns Hex-encoded SHA256 hash
 */
async function sha256(content: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(content);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Create LFS pointer content
 * @param oid - SHA256 hash of the file
 * @param size - Size of the file in bytes
 * @returns LFS pointer file content
 */
function createLFSPointer(oid: string, size: number): string {
    return `version https://git-lfs.github.com/spec/v1
oid sha256:${oid}
size ${size}
`;
}

/**
 * Get the current SHA of a file (needed for updates)
 * @param options - Upload options
 * @returns SHA of existing file or null
 */
async function getFileSha(options: LFSUploadOptions): Promise<string | null> {
    const { token, owner, repo, path } = options;
    const endpoint = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;

    try {
        const response = await fetch(endpoint, {
            method: 'GET',
            headers: {
                'Authorization': `token ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            }
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

// ===== Main Functions =====

/**
 * Upload file to GitHub LFS
 * @deprecated Use Vercel Blob storage instead
 * @param options - Upload options
 * @returns Upload result
 */
export async function uploadToGitHubLFS(options: LFSUploadOptions): Promise<LFSUploadResult> {
    const { token, owner, repo, path, content, message, onProgress } = options;

    // Calculate content hash and size
    const encoder = new TextEncoder();
    const contentBytes = encoder.encode(content);
    const size = contentBytes.length;
    const oid = await sha256(content);

    onProgress?.({ stage: 'hashing', message: 'Calculated content hash' });

    // Step 1: Request LFS upload URL via Batch API
    const lfsEndpoint = `https://github.com/${owner}/${repo}.git/info/lfs/objects/batch`;

    let lfsResponse: Response;
    try {
        lfsResponse = await fetch(lfsEndpoint, {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${btoa(`${owner}:${token}`)}`,
                'Content-Type': 'application/vnd.git-lfs+json',
                'Accept': 'application/vnd.git-lfs+json'
            },
            body: JSON.stringify({
                operation: 'upload',
                transfers: ['basic'],
                objects: [{
                    oid: oid,
                    size: size
                }]
            })
        });
    } catch (e) {
        throw new Error(`LFS Batch API request failed: ${(e as Error).message}`);
    }

    if (!lfsResponse.ok) {
        const errorText = await lfsResponse.text();
        throw new Error(`LFS Batch API error (${lfsResponse.status}): ${errorText}`);
    }

    const lfsData: LFSBatchResponse = await lfsResponse.json();
    onProgress?.({ stage: 'lfs-batch', message: 'Got LFS upload URL' });

    // Check if object already exists (no upload action needed)
    const lfsObject = lfsData.objects?.[0];
    if (!lfsObject) {
        throw new Error('No LFS object in response');
    }

    // Step 2: Upload to LFS storage (if needed)
    const uploadAction = lfsObject.actions?.upload;
    if (uploadAction) {
        // Object doesn't exist yet, need to upload
        const uploadUrl = uploadAction.href;
        const uploadHeaders = uploadAction.header || {};

        onProgress?.({ stage: 'uploading', message: 'Uploading to LFS storage...' });

        const uploadResponse = await fetch(uploadUrl, {
            method: 'PUT',
            headers: {
                ...uploadHeaders,
                'Content-Type': 'application/octet-stream'
            },
            body: contentBytes
        });

        if (!uploadResponse.ok) {
            const errorText = await uploadResponse.text();
            throw new Error(`LFS upload failed (${uploadResponse.status}): ${errorText}`);
        }

        onProgress?.({ stage: 'uploaded', message: 'File uploaded to LFS storage' });

        // Verify upload if verify action exists
        const verifyAction = lfsObject.actions?.verify;
        if (verifyAction) {
            const verifyResponse = await fetch(verifyAction.href, {
                method: 'POST',
                headers: {
                    'Authorization': `Basic ${btoa(`${owner}:${token}`)}`,
                    'Content-Type': 'application/vnd.git-lfs+json',
                    ...(verifyAction.header || {})
                },
                body: JSON.stringify({
                    oid: oid,
                    size: size
                })
            });

            if (!verifyResponse.ok) {
                console.warn('LFS verify failed, but upload may still be successful');
            }
        }
    } else {
        onProgress?.({ stage: 'exists', message: 'File already exists in LFS storage' });
    }

    // Step 3: Create commit with LFS pointer
    const pointerContent = createLFSPointer(oid, size);
    const pointerBase64 = btoa(pointerContent);

    // Check if file exists (for update vs create)
    const existingSha = await getFileSha(options);

    const commitEndpoint = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;
    const commitBody: CommitBody = {
        message: message,
        content: pointerBase64
    };

    if (existingSha) {
        commitBody.sha = existingSha;
    }

    onProgress?.({ stage: 'committing', message: 'Creating commit with LFS pointer...' });

    const commitResponse = await fetch(commitEndpoint, {
        method: 'PUT',
        headers: {
            'Authorization': `token ${token}`,
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.github.v3+json'
        },
        body: JSON.stringify(commitBody)
    });

    if (!commitResponse.ok) {
        const errorData = await commitResponse.json();
        throw new Error(`Commit failed (${commitResponse.status}): ${errorData.message || JSON.stringify(errorData)}`);
    }

    const commitResult = await commitResponse.json();
    onProgress?.({ stage: 'complete', message: 'Upload complete!' });

    return {
        success: true,
        filename: path.split('/').pop() || '',
        sha: commitResult.content?.sha,
        lfsOid: oid,
        size: size,
        url: commitResult.content?.html_url
    };
}

/**
 * Simple upload using GitHub Contents API (non-LFS)
 * Use this for small files that don't need LFS
 * @deprecated Use Vercel Blob storage instead
 * @param options - Same as uploadToGitHubLFS
 * @returns Upload result
 */
export async function uploadToGitHubContents(options: LFSUploadOptions): Promise<ContentsUploadResult> {
    const { token, owner, repo, path, content, message } = options;

    const endpoint = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;

    // Check if file exists
    const existingSha = await getFileSha(options);

    // Base64 encode content
    const b64Content = btoa(unescape(encodeURIComponent(content)));

    const body: CommitBody = {
        message: message,
        content: b64Content
    };

    if (existingSha) {
        body.sha = existingSha;
    }

    const response = await fetch(endpoint, {
        method: 'PUT',
        headers: {
            'Authorization': `token ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return {
        success: true,
        filename: result.content?.name,
        sha: result.content?.sha,
        url: result.content?.html_url
    };
}

// ===== Default Export =====

export default uploadToGitHubLFS;
