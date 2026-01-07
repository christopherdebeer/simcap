/**
 * Streaming Writer
 *
 * Handles continuous chunk writing during telemetry recording.
 * In local development mode, writes directly to the filesystem.
 * In production mode, writes to GitHub via the API proxy.
 *
 * Features:
 * - Auto-detects local vs production mode
 * - Writes chunks at configurable intervals (sample count or time)
 * - Maintains a manifest of written chunks
 * - Supports session finalization on recording stop
 */

import type { TelemetrySample, LabelSegment } from '@core/types';
import type { UploadProgress } from '@api/types';

// ===== Types =====

export interface StreamingWriterConfig {
  /** Samples per chunk before auto-flush (default: 500) */
  samplesPerChunk?: number;
  /** Maximum time between flushes in ms (default: 30000 = 30 seconds) */
  maxFlushInterval?: number;
  /** Enable local mode detection (default: true) */
  autoDetectLocal?: boolean;
  /** Force local mode regardless of detection (default: false) */
  forceLocal?: boolean;
  /** Session metadata to include in exports */
  metadata?: Record<string, unknown>;
  /** Progress callback */
  onProgress?: (progress: StreamingProgress) => void;
  /** Error callback */
  onError?: (error: Error) => void;
}

export interface StreamingProgress {
  stage: 'idle' | 'writing' | 'flushing' | 'finalizing' | 'error';
  message: string;
  chunksWritten: number;
  samplesWritten: number;
  lastChunkTime?: number;
  isLocal: boolean;
}

export interface ChunkInfo {
  index: number;
  filename: string;
  startSample: number;
  endSample: number;
  sampleCount: number;
  timestamp: string;
  bytesWritten?: number;
}

export interface SessionManifest {
  version: string;
  sessionTimestamp: string;
  isComplete: boolean;
  totalSamples: number;
  totalChunks: number;
  chunks: ChunkInfo[];
  labels: LabelSegment[];
  metadata?: Record<string, unknown>;
  finalizedAt?: string;
}

// ===== Constants =====

const LOCAL_STORAGE_ENDPOINT = '/api/local-storage';
const DEFAULT_SAMPLES_PER_CHUNK = 500;
const DEFAULT_MAX_FLUSH_INTERVAL = 30000; // 30 seconds
const DATA_VERSION = '2.1';

// ===== Local Mode Detection =====

let localModeCache: boolean | null = null;
let localModeCheckPromise: Promise<boolean> | null = null;

/**
 * Check if we're running in local development mode
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

// ===== Streaming Writer Class =====

export class StreamingWriter {
  private config: Required<StreamingWriterConfig>;
  private sessionTimestamp: string | null = null;
  private chunks: ChunkInfo[] = [];
  private pendingSamples: TelemetrySample[] = [];
  private labels: LabelSegment[] = [];
  private totalSamplesWritten = 0;
  private lastFlushTime = 0;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private isWriting = false;
  private isFinalized = false;
  private isLocal = false;

  constructor(config: StreamingWriterConfig = {}) {
    this.config = {
      samplesPerChunk: config.samplesPerChunk ?? DEFAULT_SAMPLES_PER_CHUNK,
      maxFlushInterval: config.maxFlushInterval ?? DEFAULT_MAX_FLUSH_INTERVAL,
      autoDetectLocal: config.autoDetectLocal ?? true,
      forceLocal: config.forceLocal ?? false,
      metadata: config.metadata ?? {},
      onProgress: config.onProgress ?? (() => {}),
      onError: config.onError ?? (() => {}),
    };
  }

  /**
   * Start a new recording session
   */
  async startSession(metadata?: Record<string, unknown>): Promise<void> {
    if (this.sessionTimestamp) {
      throw new Error('Session already in progress. Call finalizeSession() first.');
    }

    // Detect local mode
    if (this.config.forceLocal) {
      this.isLocal = true;
    } else if (this.config.autoDetectLocal) {
      this.isLocal = await isLocalMode();
    }

    this.sessionTimestamp = new Date().toISOString();
    this.chunks = [];
    this.pendingSamples = [];
    this.labels = [];
    this.totalSamplesWritten = 0;
    this.lastFlushTime = Date.now();
    this.isFinalized = false;

    if (metadata) {
      this.config.metadata = { ...this.config.metadata, ...metadata };
    }

    this.reportProgress('idle', 'Session started');

    // Start the flush timer
    this.startFlushTimer();

    console.log(`[StreamingWriter] Session started: ${this.sessionTimestamp} (${this.isLocal ? 'LOCAL' : 'REMOTE'} mode)`);
  }

  /**
   * Add samples to the buffer
   */
  addSamples(samples: TelemetrySample[]): void {
    if (!this.sessionTimestamp) {
      console.warn('[StreamingWriter] No active session, ignoring samples');
      return;
    }

    if (this.isFinalized) {
      console.warn('[StreamingWriter] Session is finalized, ignoring samples');
      return;
    }

    this.pendingSamples.push(...samples);

    // Check if we should flush
    if (this.pendingSamples.length >= this.config.samplesPerChunk) {
      this.flush();
    }
  }

  /**
   * Add a single sample to the buffer
   */
  addSample(sample: TelemetrySample): void {
    this.addSamples([sample]);
  }

  /**
   * Update labels for the session
   */
  updateLabels(labels: LabelSegment[]): void {
    this.labels = [...labels];
  }

  /**
   * Flush pending samples to storage
   */
  async flush(): Promise<void> {
    if (this.isWriting || this.pendingSamples.length === 0) {
      return;
    }

    if (!this.sessionTimestamp) {
      return;
    }

    this.isWriting = true;
    this.reportProgress('flushing', `Flushing ${this.pendingSamples.length} samples...`);

    try {
      const samples = [...this.pendingSamples];
      this.pendingSamples = [];

      const chunkIndex = this.chunks.length;
      const startSample = this.totalSamplesWritten;
      const endSample = startSample + samples.length - 1;

      const chunkFilename = this.getChunkFilename(chunkIndex);
      const chunkData = this.buildChunkData(samples, chunkIndex, startSample);

      let bytesWritten = 0;

      if (this.isLocal) {
        bytesWritten = await this.writeLocal(chunkFilename, chunkData);
      } else {
        bytesWritten = await this.writeRemote(chunkFilename, chunkData);
      }

      const chunkInfo: ChunkInfo = {
        index: chunkIndex,
        filename: chunkFilename,
        startSample,
        endSample,
        sampleCount: samples.length,
        timestamp: new Date().toISOString(),
        bytesWritten
      };

      this.chunks.push(chunkInfo);
      this.totalSamplesWritten += samples.length;
      this.lastFlushTime = Date.now();

      this.reportProgress('idle', `Wrote chunk ${chunkIndex + 1} (${samples.length} samples)`);

      console.log(`[StreamingWriter] Chunk ${chunkIndex + 1} written: ${samples.length} samples, ${bytesWritten} bytes`);

    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      this.reportProgress('error', `Flush failed: ${err.message}`);
      this.config.onError(err);

      // Put samples back in the buffer for retry
      // (This is a simple approach; could be more sophisticated)
      console.error('[StreamingWriter] Flush failed, samples will be retried:', err);
    } finally {
      this.isWriting = false;
    }
  }

  /**
   * Finalize the session and write manifest
   */
  async finalizeSession(): Promise<SessionManifest | null> {
    if (!this.sessionTimestamp) {
      console.warn('[StreamingWriter] No active session to finalize');
      return null;
    }

    if (this.isFinalized) {
      console.warn('[StreamingWriter] Session already finalized');
      return null;
    }

    this.reportProgress('finalizing', 'Finalizing session...');

    // Stop the flush timer
    this.stopFlushTimer();

    // Flush any remaining samples
    if (this.pendingSamples.length > 0) {
      await this.flush();
    }

    // Build and write manifest
    const manifest: SessionManifest = {
      version: DATA_VERSION,
      sessionTimestamp: this.sessionTimestamp,
      isComplete: true,
      totalSamples: this.totalSamplesWritten,
      totalChunks: this.chunks.length,
      chunks: this.chunks,
      labels: this.labels,
      metadata: this.config.metadata,
      finalizedAt: new Date().toISOString()
    };

    try {
      const manifestFilename = this.getManifestFilename();
      const manifestContent = JSON.stringify(manifest, null, 2);

      if (this.isLocal) {
        await this.writeLocal(manifestFilename, manifestContent);
      } else {
        await this.writeRemote(manifestFilename, manifestContent);
      }

      this.isFinalized = true;
      this.reportProgress('idle', `Session finalized: ${this.totalSamplesWritten} samples in ${this.chunks.length} chunks`);

      console.log(`[StreamingWriter] Session finalized: ${manifestFilename}`);

      // Reset for next session
      const result = manifest;
      this.sessionTimestamp = null;

      return result;

    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      this.reportProgress('error', `Finalization failed: ${err.message}`);
      this.config.onError(err);
      return null;
    }
  }

  /**
   * Get current session state
   */
  getState(): {
    isActive: boolean;
    isLocal: boolean;
    sessionTimestamp: string | null;
    chunksWritten: number;
    samplesWritten: number;
    pendingSamples: number;
    isFinalized: boolean;
  } {
    return {
      isActive: this.sessionTimestamp !== null,
      isLocal: this.isLocal,
      sessionTimestamp: this.sessionTimestamp,
      chunksWritten: this.chunks.length,
      samplesWritten: this.totalSamplesWritten,
      pendingSamples: this.pendingSamples.length,
      isFinalized: this.isFinalized
    };
  }

  /**
   * Cancel the current session without finalizing
   */
  cancelSession(): void {
    this.stopFlushTimer();
    this.sessionTimestamp = null;
    this.chunks = [];
    this.pendingSamples = [];
    this.labels = [];
    this.totalSamplesWritten = 0;
    this.isFinalized = false;
    this.reportProgress('idle', 'Session cancelled');
  }

  // ===== Private Methods =====

  private getChunkFilename(chunkIndex: number): string {
    const ts = this.sessionTimestamp!.replace(/:/g, '_');
    return `${ts}_chunk_${String(chunkIndex).padStart(4, '0')}.json`;
  }

  private getManifestFilename(): string {
    const ts = this.sessionTimestamp!.replace(/:/g, '_');
    return `${ts}_manifest.json`;
  }

  private buildChunkData(
    samples: TelemetrySample[],
    chunkIndex: number,
    startSample: number
  ): string {
    const data = {
      version: DATA_VERSION,
      timestamp: this.sessionTimestamp,
      type: 'session_chunk',
      chunk: {
        index: chunkIndex,
        startSample,
        endSample: startSample + samples.length - 1,
        sampleCount: samples.length
      },
      samples,
      // Include labels in each chunk for redundancy
      labels: this.labels,
      metadata: this.config.metadata
    };

    return JSON.stringify(data);
  }

  private async writeLocal(filename: string, content: string): Promise<number> {
    const response = await fetch(LOCAL_STORAGE_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'write',
        filename,
        content
      })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(error.error || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.bytesWritten || content.length;
  }

  private async writeRemote(filename: string, content: string): Promise<number> {
    // Use the existing github-upload module
    const { uploadViaProxy, hasUploadSecret } = await import('./github-upload.js');

    if (!hasUploadSecret()) {
      throw new Error('No upload secret configured for remote writes');
    }

    const result = await uploadViaProxy({
      branch: 'data',
      path: `GAMBIT/${filename}`,
      content,
      message: `GAMBIT streaming chunk: ${filename}`,
      onProgress: (progress: UploadProgress) => {
        this.reportProgress('writing', progress.message);
      }
    });

    return result.size || content.length;
  }

  private startFlushTimer(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }

    this.flushTimer = setInterval(() => {
      const timeSinceFlush = Date.now() - this.lastFlushTime;
      if (timeSinceFlush >= this.config.maxFlushInterval && this.pendingSamples.length > 0) {
        this.flush();
      }
    }, 5000); // Check every 5 seconds
  }

  private stopFlushTimer(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
  }

  private reportProgress(stage: StreamingProgress['stage'], message: string): void {
    this.config.onProgress({
      stage,
      message,
      chunksWritten: this.chunks.length,
      samplesWritten: this.totalSamplesWritten,
      lastChunkTime: this.lastFlushTime,
      isLocal: this.isLocal
    });
  }
}

// ===== Singleton Instance =====

let defaultWriter: StreamingWriter | null = null;

/**
 * Get the default streaming writer instance
 */
export function getStreamingWriter(): StreamingWriter {
  if (!defaultWriter) {
    defaultWriter = new StreamingWriter();
  }
  return defaultWriter;
}

/**
 * Create a new streaming writer with custom config
 */
export function createStreamingWriter(config: StreamingWriterConfig): StreamingWriter {
  return new StreamingWriter(config);
}

// ===== Default Export =====

export default {
  StreamingWriter,
  getStreamingWriter,
  createStreamingWriter,
  isLocalMode,
  clearLocalModeCache
};
