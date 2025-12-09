/**
 * GAMBIT Client Library
 * 
 * Provides a consistent API for web UIs to interact with GAMBIT firmware devices.
 * Handles BLE connection, framing protocol parsing, and device commands.
 * 
 * Usage:
 *   const client = new GambitClient();
 *   await client.connect();
 *   
 *   // Get firmware info
 *   const fw = await client.getFirmware();
 *   
 *   // Get device logs
 *   const logs = await client.getLogs();
 *   
 *   // Stream telemetry
 *   client.on('data', (telemetry) => console.log(telemetry));
 *   await client.startStreaming();
 * 
 * @version 1.1.0
 * @requires puck.js
 */

(function(root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['./puck'], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory(require('./puck'));
    } else {
        // Export both names for compatibility
        const exports = factory(root.Puck);
        root.GambitClient = exports;
        root.GAMBITClient = exports;
    }
}(typeof self !== 'undefined' ? self : this, function(Puck) {

    // ===== Frame Parser =====
    // Protocol: \x02TYPE:LENGTH\nPAYLOAD\x03
    class FrameParser {
        constructor() {
            this.buffer = '';
            this.handlers = {};
            this.debug = false;
        }

        onData(data) {
            this.buffer += data;
            this.processBuffer();
        }

        processBuffer() {
            while (true) {
                const start = this.buffer.indexOf('\x02');
                if (start === -1) {
                    if (this.buffer.length > 10000) {
                        this.buffer = '';
                    }
                    return;
                }

                if (start > 0) {
                    this.buffer = this.buffer.slice(start);
                }

                const headerEnd = this.buffer.indexOf('\n');
                if (headerEnd === -1) return;

                const header = this.buffer.slice(1, headerEnd);
                const colonIdx = header.indexOf(':');
                if (colonIdx === -1) {
                    this.buffer = this.buffer.slice(1);
                    continue;
                }

                const type = header.slice(0, colonIdx);
                const length = parseInt(header.slice(colonIdx + 1), 10);

                if (isNaN(length) || length < 0) {
                    this.buffer = this.buffer.slice(1);
                    continue;
                }

                const payloadStart = headerEnd + 1;
                const payloadEnd = payloadStart + length;
                const frameEnd = payloadEnd + 1;

                if (this.buffer.length < frameEnd) {
                    return;
                }

                const payload = this.buffer.slice(payloadStart, payloadEnd);
                const etx = this.buffer[payloadEnd];

                if (etx !== '\x03') {
                    this.buffer = this.buffer.slice(1);
                    continue;
                }

                if (this.debug) {
                    console.log(`[FRAME] ${type}: ${length} bytes`);
                }

                try {
                    const data = JSON.parse(payload);
                    this.emit(type, data);
                } catch (e) {
                    console.error(`[FRAME] JSON parse error for ${type}:`, e);
                }

                this.buffer = this.buffer.slice(frameEnd);
            }
        }

        on(type, handler) {
            if (!this.handlers[type]) {
                this.handlers[type] = [];
            }
            this.handlers[type].push(handler);
        }

        off(type, handler) {
            if (!this.handlers[type]) return;
            if (handler) {
                this.handlers[type] = this.handlers[type].filter(h => h !== handler);
            } else {
                delete this.handlers[type];
            }
        }

        emit(type, data) {
            if (this.handlers[type]) {
                this.handlers[type].forEach(h => h(data));
            }
            if (this.handlers['*']) {
                this.handlers['*'].forEach(h => h(type, data));
            }
        }

        clear() {
            this.buffer = '';
        }
    }

    // ===== Legacy Message Parser =====
    // For backwards compatibility with older firmware using marker-based messages
    class LegacyParser {
        constructor() {
            this.buffer = '';
            this.handlers = {};
        }

        onData(data) {
            this.buffer += data;
            this.processBuffer();
        }

        processBuffer() {
            // Look for newline-delimited messages with markers
            const lines = this.buffer.split('\n');
            
            // Keep the last incomplete line in buffer
            this.buffer = lines.pop() || '';

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                // Check for known markers
                if (trimmed.startsWith('GAMBIT{')) {
                    this.parseAndEmit('telemetry', trimmed.slice(6));
                } else if (trimmed.startsWith('FIRMWARE{')) {
                    this.parseAndEmit('firmware', trimmed.slice(8));
                } else if (trimmed.startsWith('DEVICE_LOGS{')) {
                    this.parseAndEmit('logs', trimmed.slice(11));
                } else if (trimmed.startsWith('LOG_STATS{')) {
                    this.parseAndEmit('logStats', trimmed.slice(9));
                }
            }
        }

        parseAndEmit(type, jsonStr) {
            try {
                const data = JSON.parse(jsonStr);
                this.emit(type, data);
            } catch (e) {
                console.error(`[LEGACY] JSON parse error for ${type}:`, e);
            }
        }

        on(type, handler) {
            if (!this.handlers[type]) {
                this.handlers[type] = [];
            }
            this.handlers[type].push(handler);
        }

        off(type, handler) {
            if (!this.handlers[type]) return;
            if (handler) {
                this.handlers[type] = this.handlers[type].filter(h => h !== handler);
            } else {
                delete this.handlers[type];
            }
        }

        emit(type, data) {
            if (this.handlers[type]) {
                this.handlers[type].forEach(h => h(data));
            }
        }

        clear() {
            this.buffer = '';
        }
    }

    // ===== GAMBIT Client =====
    class GambitClient {
        constructor(options = {}) {
            this.connection = null;
            this.frameParser = new FrameParser();
            this.legacyParser = new LegacyParser();
            this.eventHandlers = {};
            this.debug = options.debug || false;
            this.firmwareInfo = null;
            this.keepaliveInterval = null;
            this.autoKeepalive = options.autoKeepalive !== false;
            this.keepaliveIntervalMs = options.keepaliveInterval || 25000; // 25s keepalive

            // Wire up frame parser events to client events
            this.frameParser.on('FW', (data) => this._handleFirmware(data));
            this.frameParser.on('LOGS', (data) => this._handleLogs(data));
            this.frameParser.on('LOGS_CLEARED', (data) => this._handleLogsCleared(data));
            this.frameParser.on('LOG_STATS', (data) => this._handleLogStats(data));

            // Wire up legacy parser for backwards compatibility
            this.legacyParser.on('telemetry', (data) => this.emit('data', data));
            this.legacyParser.on('firmware', (data) => this._handleFirmware(data));
            this.legacyParser.on('logs', (data) => this._handleLogs(data));
            this.legacyParser.on('logStats', (data) => this._handleLogStats(data));
        }

        // ===== Event Handling =====
        on(event, handler) {
            if (!this.eventHandlers[event]) {
                this.eventHandlers[event] = [];
            }
            this.eventHandlers[event].push(handler);
            return this;
        }

        off(event, handler) {
            if (!this.eventHandlers[event]) return this;
            if (handler) {
                this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
            } else {
                delete this.eventHandlers[event];
            }
            return this;
        }

        emit(event, data) {
            if (this.eventHandlers[event]) {
                this.eventHandlers[event].forEach(h => h(data));
            }
            return this;
        }

        // ===== Connection Management =====
        connect() {
            return new Promise((resolve, reject) => {
                if (this.connection && this.connection.isOpen) {
                    resolve(this.firmwareInfo);
                    return;
                }

                this._log('Connecting to GAMBIT device...');

                // Track if callback has been handled to prevent duplicate calls
                // Puck.js can call the callback multiple times (on connect and on disconnect)
                let callbackHandled = false;

                Puck.connect((conn) => {
                    // Ignore subsequent callback calls (e.g., on disconnect)
                    if (callbackHandled) {
                        this._log('Ignoring duplicate Puck.connect callback');
                        return;
                    }
                    callbackHandled = true;

                    if (!conn) {
                        reject(new Error('Connection failed - user cancelled or device unavailable'));
                        return;
                    }

                    this.connection = conn;
                    this._log('Connected!');

                    // Clear parser buffers
                    this.frameParser.clear();
                    this.legacyParser.clear();

                    // Set up data listener
                    conn.on('data', (data) => {
                        this._log(`Received ${data.length} bytes`);
                        // Feed to both parsers - frame parser handles framed messages,
                        // legacy parser handles marker-based messages
                        this.frameParser.onData(data);
                        this.legacyParser.onData(data);
                    });

                    conn.on('close', () => {
                        this._log('Connection closed');
                        this.connection = null;
                        this._stopKeepalive();
                        this.emit('disconnect');
                        this.emit('close');
                    });

                    this.emit('connect', conn);

                    // Query firmware info after connection
                    setTimeout(() => {
                        this._queryFirmware().then(resolve).catch(() => {
                            // Firmware query failed, but connection is still valid
                            resolve(null);
                        });
                    }, 500);
                });
            });
        }

        disconnect() {
            this._stopKeepalive();
            if (this.connection) {
                this.connection.close();
                this.connection = null;
            }
            return this;
        }

        isConnected() {
            return this.connection && this.connection.isOpen;
        }

        // ===== Device Commands =====
        write(cmd) {
            return new Promise((resolve, reject) => {
                if (!this.isConnected()) {
                    reject(new Error('Not connected'));
                    return;
                }

                this.connection.write(cmd, (err) => {
                    if (err) {
                        reject(err);
                    } else {
                        resolve();
                    }
                });
            });
        }

        /**
         * Query firmware information (internal)
         * @private
         */
        _queryFirmware() {
            return new Promise((resolve, reject) => {
                if (!this.isConnected()) {
                    reject(new Error('Not connected'));
                    return;
                }

                const timeout = setTimeout(() => {
                    this.frameParser.off('FW', handler);
                    // Try legacy format
                    reject(new Error('Firmware query timeout'));
                }, 3000);

                const handler = (data) => {
                    clearTimeout(timeout);
                    this.frameParser.off('FW', handler);
                    this.firmwareInfo = data;
                    resolve(data);
                };

                this.frameParser.on('FW', handler);
                this.write('\x10if(typeof getFirmware==="function")getFirmware();\n');
            });
        }

        /**
         * Get firmware information
         * @returns {Promise<Object>} Firmware info object
         */
        getFirmware() {
            if (this.firmwareInfo) {
                return Promise.resolve(this.firmwareInfo);
            }
            return this._queryFirmware();
        }

        /**
         * Check firmware compatibility
         * @param {string} minVersion - Minimum required version (e.g., "1.0.0")
         * @returns {Object} { compatible: boolean, reason?: string }
         */
        checkCompatibility(minVersion) {
            if (!this.firmwareInfo) {
                return { compatible: false, reason: 'No firmware info available' };
            }

            if (!this.firmwareInfo.id || this.firmwareInfo.id !== 'GAMBIT') {
                return { compatible: false, reason: 'Not GAMBIT firmware' };
            }

            if (!this.firmwareInfo.version) {
                return { compatible: true, reason: 'Version unknown, assuming compatible' };
            }

            // Simple version comparison
            const current = this.firmwareInfo.version.split('.').map(Number);
            const required = minVersion.split('.').map(Number);

            for (let i = 0; i < 3; i++) {
                const c = current[i] || 0;
                const r = required[i] || 0;
                if (c > r) return { compatible: true };
                if (c < r) return { compatible: false, reason: `Firmware ${this.firmwareInfo.version} < ${minVersion}` };
            }

            return { compatible: true };
        }

        /**
         * Get device logs
         * @param {number} [since] - Only get logs since this index
         * @returns {Promise<Object>} Logs response object
         */
        getLogs(since) {
            return new Promise((resolve, reject) => {
                if (!this.isConnected()) {
                    reject(new Error('Not connected'));
                    return;
                }

                const timeout = setTimeout(() => {
                    this.frameParser.off('LOGS', handler);
                    reject(new Error('Logs query timeout'));
                }, 5000);

                const handler = (data) => {
                    clearTimeout(timeout);
                    this.frameParser.off('LOGS', handler);
                    resolve(data);
                };

                this.frameParser.on('LOGS', handler);
                const cmd = since !== undefined 
                    ? `\x10if(typeof getLogs==="function")getLogs(${since});\n`
                    : '\x10if(typeof getLogs==="function")getLogs();\n';
                this.write(cmd);
            });
        }

        /**
         * Clear device logs
         * @returns {Promise<Object>} Clear confirmation
         */
        clearLogs() {
            return new Promise((resolve, reject) => {
                if (!this.isConnected()) {
                    reject(new Error('Not connected'));
                    return;
                }

                const timeout = setTimeout(() => {
                    this.frameParser.off('LOGS_CLEARED', handler);
                    reject(new Error('Clear logs timeout'));
                }, 3000);

                const handler = (data) => {
                    clearTimeout(timeout);
                    this.frameParser.off('LOGS_CLEARED', handler);
                    resolve(data);
                };

                this.frameParser.on('LOGS_CLEARED', handler);
                this.write('\x10if(typeof clearLogs==="function")clearLogs();\n');
            });
        }

        /**
         * Get log statistics
         * @returns {Promise<Object>} Log stats object
         */
        getLogStats() {
            return new Promise((resolve, reject) => {
                if (!this.isConnected()) {
                    reject(new Error('Not connected'));
                    return;
                }

                const timeout = setTimeout(() => {
                    this.frameParser.off('LOG_STATS', handler);
                    reject(new Error('Log stats timeout'));
                }, 3000);

                const handler = (data) => {
                    clearTimeout(timeout);
                    this.frameParser.off('LOG_STATS', handler);
                    resolve(data);
                };

                this.frameParser.on('LOG_STATS', handler);
                this.write('\x10if(typeof getLogStats==="function")getLogStats();\n');
            });
        }

        /**
         * Start telemetry streaming with automatic keepalive
         * Alias: startStream()
         * @returns {Promise<void>}
         */
        startStreaming() {
            return new Promise((resolve, reject) => {
                if (!this.isConnected()) {
                    reject(new Error('Not connected'));
                    return;
                }

                this._log('Starting telemetry stream...');
                
                // Send initial getData() to start streaming
                this.write('\x10if(typeof getData==="function")getData();\n')
                    .then(() => {
                        // Set up keepalive to prevent 30s timeout
                        if (this.autoKeepalive) {
                            this._startKeepalive();
                        }
                        this.emit('streamStart');
                        resolve();
                    })
                    .catch(reject);
            });
        }

        // Alias for backwards compatibility
        startStream() {
            return this.startStreaming();
        }

        /**
         * Stop telemetry streaming
         * Alias: stopStream()
         * @returns {Promise<void>}
         */
        stopStreaming() {
            return new Promise((resolve, reject) => {
                if (!this.isConnected()) {
                    reject(new Error('Not connected'));
                    return;
                }

                this._stopKeepalive();
                this._log('Stopping telemetry stream...');
                
                this.write('\x10if(typeof stopData==="function")stopData();\n')
                    .then(() => {
                        this.emit('streamStop');
                        resolve();
                    })
                    .catch(reject);
            });
        }

        // Alias for backwards compatibility
        stopStream() {
            return this.stopStreaming();
        }

        /**
         * Get battery percentage
         * @returns {Promise<number>}
         */
        getBattery() {
            return Puck.eval('Puck.getBatteryPercentage()');
        }

        /**
         * Get temperature
         * @returns {Promise<number>}
         */
        getTemperature() {
            return Puck.eval('E.getTemperature()');
        }

        /**
         * Reset the device
         * @returns {Promise<void>}
         */
        reset() {
            return this.write('reset();\n');
        }

        /**
         * Save current code to flash
         * @returns {Promise<void>}
         */
        save() {
            return this.write('save();\n');
        }

        // ===== Internal Methods =====
        _handleFirmware(data) {
            this.firmwareInfo = data;
            this.emit('firmware', data);
        }

        _handleLogs(data) {
            this.emit('logs', data);
        }

        _handleLogsCleared(data) {
            this.emit('logsCleared', data);
        }

        _handleLogStats(data) {
            this.emit('logStats', data);
        }

        _startKeepalive() {
            this._stopKeepalive();
            this.keepaliveInterval = setInterval(() => {
                if (this.isConnected()) {
                    this._log('Sending keepalive...');
                    this.write('\x10if(typeof getData==="function")getData();\n');
                }
            }, this.keepaliveIntervalMs);
        }

        _stopKeepalive() {
            if (this.keepaliveInterval) {
                clearInterval(this.keepaliveInterval);
                this.keepaliveInterval = null;
            }
        }

        _log(msg) {
            if (this.debug) {
                console.log(`[GambitClient] ${msg}`);
            }
        }
    }

    // ===== Utility Functions =====
    GambitClient.formatUptime = function(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
        if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
        if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
        return `${seconds}s`;
    };

    GambitClient.formatLogTime = function(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}:${String(minutes % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
        }
        return `${minutes}:${String(seconds % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0').substring(0, 1)}`;
    };

    GambitClient.LOG_LEVELS = {
        'E': { name: 'ERROR', color: '#ff4757' },
        'W': { name: 'WARN', color: '#ffa502' },
        'I': { name: 'INFO', color: '#00ff88' },
        'D': { name: 'DEBUG', color: '#888888' }
    };

    // Export classes for advanced usage
    GambitClient.FrameParser = FrameParser;
    GambitClient.LegacyParser = LegacyParser;

    return GambitClient;
}));
