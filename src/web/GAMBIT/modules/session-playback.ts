/**
 * GAMBIT Session Playback Module
 *
 * Handles loading and playing back recorded sensor sessions
 * for visualization and testing purposes.
 *
 * @module session-playback
 */

import type { TelemetrySample } from '@core/types';

// ===== Type Definitions =====

export interface SessionMetadata {
    filename: string;
    timestamp: string;
    durationSec?: number;
    sampleCount?: number;
    [key: string]: any;
}

export interface SessionManifest {
    sessions: SessionMetadata[];
}

export interface SessionData {
    samples?: TelemetrySample[];
    [key: string]: any;
}

export interface PlaybackState {
    isPlaying: boolean;
    currentIndex: number;
    totalSamples: number;
    currentTime: number;
    duration: number;
    progress: number;
    speed: number;
    session: SessionMetadata | null;
}

export interface SessionPlaybackConfig {
    manifestUrl: string;
    dataBaseUrl: string;
    sampleRate: number;
    defaultSpeed: number;
}

export interface SessionPlaybackOptions {
    /** URL to session manifest JSON */
    manifestUrl?: string;
    /** Base URL for session data files */
    dataBaseUrl?: string;
    /** Assumed sample rate in Hz */
    sampleRate?: number;
    /** Default playback speed */
    defaultSpeed?: number;
    /** Callback for each sample during playback */
    onSample?: (sample: TelemetrySample, index: number, total: number) => void;
    /** Callback when playback state changes */
    onStateChange?: (state: PlaybackState) => void;
    /** Callback when session is loaded */
    onSessionLoaded?: (info: { session: SessionMetadata; sampleCount: number; duration: number }) => void;
    /** Callback when manifest is loaded */
    onManifestLoaded?: (sessions: SessionMetadata[]) => void;
    /** Callback for errors */
    onError?: (error: Error) => void;
}

export interface LoadedSession {
    session: SessionMetadata;
    samples: TelemetrySample[];
}

// ===== Constants =====

/**
 * Default configuration
 */
const DEFAULT_CONFIG: SessionPlaybackConfig = {
    manifestUrl: '/api/sessions',  // Use API endpoint for Vercel Blob sessions
    dataBaseUrl: '',               // Sessions have full URLs from API
    sampleRate: 20,  // Hz - assumed sample rate for time calculations
    defaultSpeed: 1
};

// ===== Class =====

/**
 * Session Playback Controller
 *
 * Manages loading, playing, pausing, and seeking through recorded sessions.
 */
export class SessionPlayback {
    private config: SessionPlaybackConfig;

    // State
    private sessions: SessionMetadata[];
    private currentSession: SessionMetadata | null;
    private samples: TelemetrySample[];
    private currentIndex: number;
    private isPlaying: boolean;
    private playbackInterval: ReturnType<typeof setInterval> | null;
    private speed: number;

    // Callbacks
    private onSample: ((sample: TelemetrySample, index: number, total: number) => void) | null;
    private onStateChange: ((state: PlaybackState) => void) | null;
    private onSessionLoaded: ((info: { session: SessionMetadata; sampleCount: number; duration: number }) => void) | null;
    private onManifestLoaded: ((sessions: SessionMetadata[]) => void) | null;
    private onError: ((error: Error) => void) | null;

    constructor(options: SessionPlaybackOptions = {}) {
        this.config = { ...DEFAULT_CONFIG, ...options };

        // State
        this.sessions = [];
        this.currentSession = null;
        this.samples = [];
        this.currentIndex = 0;
        this.isPlaying = false;
        this.playbackInterval = null;
        this.speed = this.config.defaultSpeed;

        // Callbacks
        this.onSample = options.onSample || null;
        this.onStateChange = options.onStateChange || null;
        this.onSessionLoaded = options.onSessionLoaded || null;
        this.onManifestLoaded = options.onManifestLoaded || null;
        this.onError = options.onError || null;

        console.log('[SessionPlayback] Initialized with config:', {
            manifestUrl: this.config.manifestUrl,
            dataBaseUrl: this.config.dataBaseUrl,
            sampleRate: this.config.sampleRate
        });
    }

    /**
     * Load the session manifest
     * @returns Array of session metadata
     */
    async loadManifest(): Promise<SessionMetadata[]> {
        try {
            console.log('[SessionPlayback] Loading manifest from:', this.config.manifestUrl);

            const response = await fetch(this.config.manifestUrl);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const manifest: SessionManifest = await response.json();
            this.sessions = manifest.sessions || [];

            console.log('[SessionPlayback] Loaded manifest with', this.sessions.length, 'sessions');

            if (this.onManifestLoaded) {
                this.onManifestLoaded(this.sessions);
            }

            return this.sessions;
        } catch (error) {
            console.error('[SessionPlayback] Failed to load manifest:', error);
            if (this.onError) {
                this.onError(error as Error);
            }
            throw error;
        }
    }

    /**
     * Get list of available sessions
     * @returns Session metadata array
     */
    getSessions(): SessionMetadata[] {
        return this.sessions;
    }

    /**
     * Get session by index
     * @param index - Session index
     * @returns Session metadata
     */
    getSession(index: number): SessionMetadata | null {
        return this.sessions[index] || null;
    }

    /**
     * Load a specific session's data
     * @param indexOrFilename - Session index or filename
     * @returns Loaded session data
     */
    async loadSession(indexOrFilename: number | string): Promise<LoadedSession> {
        let session: SessionMetadata | undefined;

        if (typeof indexOrFilename === 'number') {
            session = this.sessions[indexOrFilename];
        } else {
            session = this.sessions.find(s => s.filename === indexOrFilename);
        }

        if (!session) {
            const error = new Error(`Session not found: ${indexOrFilename}`);
            if (this.onError) this.onError(error);
            throw error;
        }

        try {
            console.log('[SessionPlayback] Loading session:', session.filename);

            // Use session's URL directly if available (from API), otherwise construct from base URL
            const url = (session as any).url || (session as any).downloadUrl || (this.config.dataBaseUrl + session.filename);
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data: TelemetrySample[] | SessionData = await response.json();

            // Handle both v1.0 (array) and v2.0 (object with samples) formats
            if (Array.isArray(data)) {
                this.samples = data;
            } else if (data.samples && Array.isArray(data.samples)) {
                this.samples = data.samples;
            } else {
                throw new Error('Unknown data format');
            }

            this.currentSession = session;
            this.currentIndex = 0;

            console.log('[SessionPlayback] Loaded session with', this.samples.length, 'samples');

            if (this.onSessionLoaded) {
                this.onSessionLoaded({
                    session: this.currentSession,
                    sampleCount: this.samples.length,
                    duration: this.getDuration()
                });
            }

            this._emitStateChange();

            return {
                session: this.currentSession,
                samples: this.samples
            };
        } catch (error) {
            console.error('[SessionPlayback] Failed to load session:', error);
            if (this.onError) this.onError(error as Error);
            throw error;
        }
    }

    /**
     * Start playback
     */
    play(): void {
        if (this.samples.length === 0) {
            console.warn('[SessionPlayback] No samples to play');
            return;
        }

        if (this.isPlaying) {
            console.warn('[SessionPlayback] Already playing');
            return;
        }

        this.isPlaying = true;

        // Calculate interval based on speed (base is 1000/sampleRate ms)
        const baseInterval = 1000 / this.config.sampleRate;
        const interval = baseInterval / this.speed;

        this.playbackInterval = setInterval(() => {
            if (this.currentIndex >= this.samples.length) {
                this.stop();
                return;
            }

            const sample = this.samples[this.currentIndex];

            if (this.onSample) {
                this.onSample(sample, this.currentIndex, this.samples.length);
            }

            this.currentIndex++;
            this._emitStateChange();

        }, interval);

        console.log('[SessionPlayback] Started at', this.speed, 'x speed');
        this._emitStateChange();
    }

    /**
     * Pause playback
     */
    pause(): void {
        if (!this.isPlaying) return;

        this.isPlaying = false;

        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }

        console.log('[SessionPlayback] Paused at index', this.currentIndex);
        this._emitStateChange();
    }

    /**
     * Stop playback and reset to beginning
     */
    stop(): void {
        this.pause();
        this.currentIndex = 0;

        console.log('[SessionPlayback] Stopped');
        this._emitStateChange();
    }

    /**
     * Toggle play/pause
     */
    toggle(): void {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    /**
     * Seek to a specific position
     * @param index - Sample index to seek to
     */
    seekTo(index: number): void {
        this.currentIndex = Math.max(0, Math.min(index, this.samples.length - 1));

        // Emit the sample at this position
        if (this.samples[this.currentIndex] && this.onSample) {
            this.onSample(this.samples[this.currentIndex], this.currentIndex, this.samples.length);
        }

        this._emitStateChange();
    }

    /**
     * Seek to a specific time
     * @param seconds - Time in seconds
     */
    seekToTime(seconds: number): void {
        const index = Math.floor(seconds * this.config.sampleRate);
        this.seekTo(index);
    }

    /**
     * Seek by percentage
     * @param percent - Percentage (0-100)
     */
    seekToPercent(percent: number): void {
        const index = Math.floor((percent / 100) * this.samples.length);
        this.seekTo(index);
    }

    /**
     * Set playback speed
     * @param speed - Speed multiplier (e.g., 0.5, 1, 2, 4)
     */
    setSpeed(speed: number): void {
        this.speed = speed;

        // If playing, restart with new speed
        if (this.isPlaying) {
            this.pause();
            this.play();
        }

        console.log('[SessionPlayback] Speed set to', this.speed, 'x');
    }

    /**
     * Get current playback state
     * @returns Current state
     */
    getState(): PlaybackState {
        return {
            isPlaying: this.isPlaying,
            currentIndex: this.currentIndex,
            totalSamples: this.samples.length,
            currentTime: this.getCurrentTime(),
            duration: this.getDuration(),
            progress: this.getProgress(),
            speed: this.speed,
            session: this.currentSession
        };
    }

    /**
     * Get current time in seconds
     * @returns Current time
     */
    getCurrentTime(): number {
        return this.currentIndex / this.config.sampleRate;
    }

    /**
     * Get total duration in seconds
     * @returns Duration
     */
    getDuration(): number {
        return this.samples.length / this.config.sampleRate;
    }

    /**
     * Get progress as percentage
     * @returns Progress (0-100)
     */
    getProgress(): number {
        if (this.samples.length === 0) return 0;
        return (this.currentIndex / this.samples.length) * 100;
    }

    /**
     * Get current sample
     * @returns Current sample
     */
    getCurrentSample(): TelemetrySample | null {
        return this.samples[this.currentIndex] || null;
    }

    /**
     * Check if session is loaded
     * @returns true if session is loaded
     */
    hasSession(): boolean {
        return this.samples.length > 0;
    }

    /**
     * Emit state change event
     */
    private _emitStateChange(): void {
        if (this.onStateChange) {
            this.onStateChange(this.getState());
        }
    }

    /**
     * Dispose and clean up
     */
    dispose(): void {
        this.stop();
        this.sessions = [];
        this.samples = [];
        this.currentSession = null;
    }
}

// ===== Utility Functions =====

/**
 * Format seconds as M:SS
 * @param seconds - Time in seconds
 * @returns Formatted time string
 */
export function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format session for display
 * @param session - Session metadata
 * @returns Formatted display string
 */
export function formatSessionDisplay(session: SessionMetadata): string {
    const date = new Date(session.timestamp);
    const dateStr = date.toLocaleDateString();
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const duration = session.durationSec ? formatTime(session.durationSec) : '?';
    const samples = session.sampleCount || '?';

    return `${dateStr} ${timeStr} - ${duration} (${samples} samples)`;
}

/**
 * Create a session playback instance
 * @param options - Configuration options
 * @returns Playback instance
 */
export function createSessionPlayback(options: SessionPlaybackOptions = {}): SessionPlayback {
    return new SessionPlayback(options);
}

// ===== Default Export =====

export default SessionPlayback;
