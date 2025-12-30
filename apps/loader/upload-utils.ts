/**
 * Fault-Tolerant Firmware Upload Utilities
 *
 * Provides reliable firmware uploads with:
 * - CRC-8 checksum validation per chunk
 * - Retry logic with exponential backoff
 * - Resumable upload state
 * - Progress tracking with detailed diagnostics
 *
 * @module upload-utils
 */

// ===== CRC-8 Implementation =====
// Standard CRC-8 polynomial (x^8 + x^2 + x + 1) = 0x07
// Used for chunk-level validation during upload

const CRC8_TABLE = new Uint8Array(256);

// Pre-compute CRC-8 lookup table
(function initCrcTable() {
    const polynomial = 0x07;
    for (let i = 0; i < 256; i++) {
        let crc = i;
        for (let j = 0; j < 8; j++) {
            crc = (crc & 0x80) ? ((crc << 1) ^ polynomial) : (crc << 1);
        }
        CRC8_TABLE[i] = crc & 0xFF;
    }
})();

/**
 * Calculate CRC-8 checksum for a byte array
 */
export function crc8(data: Uint8Array | number[]): number {
    let crc = 0;
    for (let i = 0; i < data.length; i++) {
        crc = CRC8_TABLE[(crc ^ data[i]) & 0xFF];
    }
    return crc;
}

/**
 * Calculate CRC-8 checksum for a string (UTF-8 encoded)
 */
export function crc8String(str: string): number {
    const encoder = new TextEncoder();
    return crc8(encoder.encode(str));
}

/**
 * Validate data against expected CRC
 */
export function validateCrc8(data: Uint8Array | number[], expectedCrc: number): boolean {
    return crc8(data) === expectedCrc;
}

// ===== Upload State Management =====

export interface UploadChunk {
    index: number;
    offset: number;
    length: number;
    crc: number;
    verified: boolean;
    retries: number;
}

export interface UploadState {
    // Firmware info
    firmwareName: string;
    totalBytes: number;
    chunkSize: number;

    // Progress
    chunks: UploadChunk[];
    currentChunkIndex: number;
    bytesWritten: number;
    bytesVerified: number;

    // Timing
    startTime: number;
    lastActivityTime: number;

    // Status
    phase: 'preparing' | 'uploading' | 'verifying' | 'executing' | 'complete' | 'failed';
    error?: string;

    // Retry tracking
    totalRetries: number;
    maxRetriesPerChunk: number;
    consecutiveFailures: number;
}

/**
 * Create initial upload state
 */
export function createUploadState(
    firmwareName: string,
    code: string,
    chunkSize: number = 512,
    maxRetriesPerChunk: number = 3
): UploadState {
    const encoder = new TextEncoder();
    const encoded = encoder.encode(code);
    const totalBytes = encoded.length;
    const numChunks = Math.ceil(totalBytes / chunkSize);

    // Pre-calculate chunks with CRCs
    const chunks: UploadChunk[] = [];
    for (let i = 0; i < numChunks; i++) {
        const offset = i * chunkSize;
        const length = Math.min(chunkSize, totalBytes - offset);
        const chunkData = encoded.slice(offset, offset + length);

        chunks.push({
            index: i,
            offset,
            length,
            crc: crc8(chunkData),
            verified: false,
            retries: 0,
        });
    }

    return {
        firmwareName,
        totalBytes,
        chunkSize,
        chunks,
        currentChunkIndex: 0,
        bytesWritten: 0,
        bytesVerified: 0,
        startTime: Date.now(),
        lastActivityTime: Date.now(),
        phase: 'preparing',
        totalRetries: 0,
        maxRetriesPerChunk,
        consecutiveFailures: 0,
    };
}

/**
 * Get the next unverified chunk to upload
 */
export function getNextChunk(state: UploadState): UploadChunk | null {
    // First, try to continue from current position
    for (let i = state.currentChunkIndex; i < state.chunks.length; i++) {
        if (!state.chunks[i].verified) {
            return state.chunks[i];
        }
    }

    // Check for any unverified chunks (may have failed earlier)
    for (const chunk of state.chunks) {
        if (!chunk.verified && chunk.retries < state.maxRetriesPerChunk) {
            return chunk;
        }
    }

    return null;
}

/**
 * Mark a chunk as successfully verified
 */
export function markChunkVerified(state: UploadState, chunkIndex: number): void {
    const chunk = state.chunks[chunkIndex];
    if (chunk) {
        chunk.verified = true;
        state.bytesVerified += chunk.length;
        state.currentChunkIndex = Math.max(state.currentChunkIndex, chunkIndex + 1);
        state.consecutiveFailures = 0;
        state.lastActivityTime = Date.now();
    }
}

/**
 * Mark a chunk upload as failed (will retry)
 */
export function markChunkFailed(state: UploadState, chunkIndex: number): boolean {
    const chunk = state.chunks[chunkIndex];
    if (chunk) {
        chunk.retries++;
        state.totalRetries++;
        state.consecutiveFailures++;
        state.lastActivityTime = Date.now();

        // Check if we should give up on this chunk
        return chunk.retries < state.maxRetriesPerChunk;
    }
    return false;
}

/**
 * Calculate upload progress (0-100)
 */
export function getUploadProgress(state: UploadState): number {
    if (state.totalBytes === 0) return 0;
    return Math.round((state.bytesVerified / state.totalBytes) * 100);
}

/**
 * Get estimated time remaining in ms
 */
export function getEstimatedTimeRemaining(state: UploadState): number {
    const elapsed = Date.now() - state.startTime;
    const progress = state.bytesVerified / state.totalBytes;

    if (progress === 0) return -1;

    const totalEstimated = elapsed / progress;
    return Math.max(0, totalEstimated - elapsed);
}

/**
 * Check if upload has stalled (no progress for threshold ms)
 */
export function isUploadStalled(state: UploadState, thresholdMs: number = 30000): boolean {
    return Date.now() - state.lastActivityTime > thresholdMs;
}

/**
 * Serialize upload state for persistence/recovery
 */
export function serializeUploadState(state: UploadState): string {
    return JSON.stringify({
        firmwareName: state.firmwareName,
        totalBytes: state.totalBytes,
        chunkSize: state.chunkSize,
        chunks: state.chunks.map(c => ({
            index: c.index,
            offset: c.offset,
            length: c.length,
            crc: c.crc,
            verified: c.verified,
            retries: c.retries,
        })),
        currentChunkIndex: state.currentChunkIndex,
        bytesVerified: state.bytesVerified,
        phase: state.phase,
    });
}

/**
 * Deserialize upload state for resume
 */
export function deserializeUploadState(json: string): Partial<UploadState> | null {
    try {
        return JSON.parse(json);
    } catch {
        return null;
    }
}

// ===== Retry Logic =====

export interface RetryOptions {
    maxRetries: number;
    initialDelayMs: number;
    maxDelayMs: number;
    backoffMultiplier: number;
    onRetry?: (attempt: number, delay: number, error: Error) => void;
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
    maxRetries: 3,
    initialDelayMs: 500,
    maxDelayMs: 8000,
    backoffMultiplier: 2,
};

/**
 * Execute a function with retry logic and exponential backoff
 */
export async function withRetry<T>(
    fn: () => Promise<T>,
    options: Partial<RetryOptions> = {}
): Promise<T> {
    const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
    let lastError: Error = new Error('Unknown error');
    let delay = opts.initialDelayMs;

    for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
        try {
            return await fn();
        } catch (e) {
            lastError = e as Error;

            if (attempt < opts.maxRetries) {
                opts.onRetry?.(attempt + 1, delay, lastError);
                await sleep(delay);
                delay = Math.min(delay * opts.backoffMultiplier, opts.maxDelayMs);
            }
        }
    }

    throw lastError;
}

/**
 * Sleep for a given number of milliseconds
 */
export function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ===== Chunk Encoding =====

/**
 * Escape a string for inclusion in a JavaScript string literal
 * Handles special characters that could break the command
 */
export function escapeForJS(str: string): string {
    let result = '';
    for (let i = 0; i < str.length; i++) {
        const char = str[i];
        const code = char.charCodeAt(0);

        if (code < 32 || code === 92 || code === 39 || code === 34 || code > 126) {
            // Escape control chars, backslash, quotes, and non-ASCII
            result += '\\x' + code.toString(16).padStart(2, '0');
        } else {
            result += char;
        }
    }
    return result;
}

/**
 * Build a Storage.write command for a chunk
 */
export function buildWriteCommand(
    chunk: string,
    offset: number,
    totalBytes: number,
    isFirstChunk: boolean
): string {
    const escapedChunk = escapeForJS(chunk);

    if (isFirstChunk) {
        // First chunk includes total file size to pre-allocate
        return `require("Storage").write(".bootcde",'${escapedChunk}',${offset},${totalBytes});\n`;
    } else {
        return `require("Storage").write(".bootcde",'${escapedChunk}',${offset});\n`;
    }
}

/**
 * Build a command to read and verify a chunk's CRC on the device
 * Returns the CRC value for comparison
 */
export function buildVerifyCommand(offset: number, length: number): string {
    // This command reads the chunk from Storage and computes a simple checksum
    // We use XOR-based checksum for simplicity (Espruino doesn't have built-in CRC)
    return `(function(){var d=require("Storage").read(".bootcde",${offset},${length});var c=0;for(var i=0;i<d.length;i++)c^=d.charCodeAt(i);return c;})();\n`;
}

/**
 * Calculate XOR checksum (matching the on-device verification)
 */
export function xorChecksum(str: string): number {
    let checksum = 0;
    for (let i = 0; i < str.length; i++) {
        checksum ^= str.charCodeAt(i);
    }
    return checksum;
}

// ===== Upload Statistics =====

export interface UploadStats {
    totalBytes: number;
    bytesVerified: number;
    chunksTotal: number;
    chunksVerified: number;
    chunksFailed: number;
    totalRetries: number;
    elapsedMs: number;
    bytesPerSecond: number;
    estimatedRemainingMs: number;
    phase: string;
}

/**
 * Get current upload statistics
 */
export function getUploadStats(state: UploadState): UploadStats {
    const elapsed = Date.now() - state.startTime;
    const bytesPerSecond = elapsed > 0 ? (state.bytesVerified / elapsed) * 1000 : 0;
    const remaining = state.totalBytes - state.bytesVerified;
    const estimatedRemainingMs = bytesPerSecond > 0 ? (remaining / bytesPerSecond) * 1000 : -1;

    return {
        totalBytes: state.totalBytes,
        bytesVerified: state.bytesVerified,
        chunksTotal: state.chunks.length,
        chunksVerified: state.chunks.filter(c => c.verified).length,
        chunksFailed: state.chunks.filter(c => c.retries >= state.maxRetriesPerChunk).length,
        totalRetries: state.totalRetries,
        elapsedMs: elapsed,
        bytesPerSecond: Math.round(bytesPerSecond),
        estimatedRemainingMs: Math.round(estimatedRemainingMs),
        phase: state.phase,
    };
}

/**
 * Format upload stats for display
 */
export function formatUploadStats(stats: UploadStats): string {
    const progress = Math.round((stats.bytesVerified / stats.totalBytes) * 100);
    const speed = (stats.bytesPerSecond / 1024).toFixed(1);
    const remaining = stats.estimatedRemainingMs > 0
        ? `~${Math.ceil(stats.estimatedRemainingMs / 1000)}s remaining`
        : '';

    return `${progress}% (${stats.chunksVerified}/${stats.chunksTotal} chunks) @ ${speed} KB/s ${remaining}`;
}

// ===== Firmware Metadata =====

export interface FirmwareMetadata {
    name: string;
    path: string;
    size: number;
    minifiedSize?: number;
    checksum: number;
    version?: string;
}

/**
 * Extract firmware metadata from source code
 */
export function extractFirmwareMetadata(code: string, path: string): FirmwareMetadata {
    // Try to extract FIRMWARE_INFO from code
    const versionMatch = code.match(/version:\s*["']([^"']+)["']/);
    const nameMatch = code.match(/name:\s*["']([^"']+)["']/);
    const idMatch = code.match(/id:\s*["']([^"']+)["']/);

    return {
        name: idMatch?.[1] || nameMatch?.[1] || path.split('/').slice(-2, -1)[0] || 'unknown',
        path,
        size: code.length,
        checksum: xorChecksum(code),
        version: versionMatch?.[1],
    };
}
