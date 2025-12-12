/**
 * GAMBIT Session Playback Module
 * 
 * Handles loading and playing back recorded sensor sessions
 * for visualization and testing purposes.
 * 
 * @module session-playback
 */

/**
 * Default configuration
 */
const DEFAULT_CONFIG = {
    manifestUrl: '../../../data/GAMBIT/manifest.json',
    dataBaseUrl: '../../../data/GAMBIT/',
    sampleRate: 20,  // Hz - assumed sample rate for time calculations
    defaultSpeed: 1
};

/**
 * Session Playback Controller
 * 
 * Manages loading, playing, pausing, and seeking through recorded sessions.
 */
export class SessionPlayback {
    /**
     * @param {Object} options - Configuration options
     * @param {string} [options.manifestUrl] - URL to session manifest JSON
     * @param {string} [options.dataBaseUrl] - Base URL for session data files
     * @param {number} [options.sampleRate=20] - Assumed sample rate in Hz
     * @param {Function} [options.onSample] - Callback for each sample during playback
     * @param {Function} [options.onStateChange] - Callback when playback state changes
     * @param {Function} [options.onSessionLoaded] - Callback when session is loaded
     * @param {Function} [options.onManifestLoaded] - Callback when manifest is loaded
     * @param {Function} [options.onError] - Callback for errors
     */
    constructor(options = {}) {
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
     * @returns {Promise<Array>} Array of session metadata
     */
    async loadManifest() {
        try {
            console.log('[SessionPlayback] Loading manifest from:', this.config.manifestUrl);
            
            const response = await fetch(this.config.manifestUrl);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const manifest = await response.json();
            this.sessions = manifest.sessions || [];
            
            console.log('[SessionPlayback] Loaded manifest with', this.sessions.length, 'sessions');
            
            if (this.onManifestLoaded) {
                this.onManifestLoaded(this.sessions);
            }
            
            return this.sessions;
        } catch (error) {
            console.error('[SessionPlayback] Failed to load manifest:', error);
            if (this.onError) {
                this.onError(error);
            }
            throw error;
        }
    }
    
    /**
     * Get list of available sessions
     * @returns {Array} Session metadata array
     */
    getSessions() {
        return this.sessions;
    }
    
    /**
     * Get session by index
     * @param {number} index - Session index
     * @returns {Object|null} Session metadata
     */
    getSession(index) {
        return this.sessions[index] || null;
    }
    
    /**
     * Load a specific session's data
     * @param {number|string} indexOrFilename - Session index or filename
     * @returns {Promise<Object>} Loaded session data
     */
    async loadSession(indexOrFilename) {
        let session;
        
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
            
            const url = this.config.dataBaseUrl + session.filename;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
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
            if (this.onError) this.onError(error);
            throw error;
        }
    }
    
    /**
     * Start playback
     */
    play() {
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
    pause() {
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
    stop() {
        this.pause();
        this.currentIndex = 0;
        
        console.log('[SessionPlayback] Stopped');
        this._emitStateChange();
    }
    
    /**
     * Toggle play/pause
     */
    toggle() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }
    
    /**
     * Seek to a specific position
     * @param {number} index - Sample index to seek to
     */
    seekTo(index) {
        this.currentIndex = Math.max(0, Math.min(index, this.samples.length - 1));
        
        // Emit the sample at this position
        if (this.samples[this.currentIndex] && this.onSample) {
            this.onSample(this.samples[this.currentIndex], this.currentIndex, this.samples.length);
        }
        
        this._emitStateChange();
    }
    
    /**
     * Seek to a specific time
     * @param {number} seconds - Time in seconds
     */
    seekToTime(seconds) {
        const index = Math.floor(seconds * this.config.sampleRate);
        this.seekTo(index);
    }
    
    /**
     * Seek by percentage
     * @param {number} percent - Percentage (0-100)
     */
    seekToPercent(percent) {
        const index = Math.floor((percent / 100) * this.samples.length);
        this.seekTo(index);
    }
    
    /**
     * Set playback speed
     * @param {number} speed - Speed multiplier (e.g., 0.5, 1, 2, 4)
     */
    setSpeed(speed) {
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
     * @returns {Object} Current state
     */
    getState() {
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
     * @returns {number} Current time
     */
    getCurrentTime() {
        return this.currentIndex / this.config.sampleRate;
    }
    
    /**
     * Get total duration in seconds
     * @returns {number} Duration
     */
    getDuration() {
        return this.samples.length / this.config.sampleRate;
    }
    
    /**
     * Get progress as percentage
     * @returns {number} Progress (0-100)
     */
    getProgress() {
        if (this.samples.length === 0) return 0;
        return (this.currentIndex / this.samples.length) * 100;
    }
    
    /**
     * Get current sample
     * @returns {Object|null} Current sample
     */
    getCurrentSample() {
        return this.samples[this.currentIndex] || null;
    }
    
    /**
     * Check if session is loaded
     * @returns {boolean}
     */
    hasSession() {
        return this.samples.length > 0;
    }
    
    /**
     * Emit state change event
     * @private
     */
    _emitStateChange() {
        if (this.onStateChange) {
            this.onStateChange(this.getState());
        }
    }
    
    /**
     * Dispose and clean up
     */
    dispose() {
        this.stop();
        this.sessions = [];
        this.samples = [];
        this.currentSession = null;
    }
}

/**
 * Format seconds as M:SS
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
export function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format session for display
 * @param {Object} session - Session metadata
 * @returns {string} Formatted display string
 */
export function formatSessionDisplay(session) {
    const date = new Date(session.timestamp);
    const dateStr = date.toLocaleDateString();
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const duration = session.durationSec ? formatTime(session.durationSec) : '?';
    const samples = session.sampleCount || '?';
    
    return `${dateStr} ${timeStr} - ${duration} (${samples} samples)`;
}

/**
 * Create a session playback instance
 * @param {Object} options - Configuration options
 * @returns {SessionPlayback} Playback instance
 */
export function createSessionPlayback(options = {}) {
    return new SessionPlayback(options);
}

// Default export
export default SessionPlayback;
