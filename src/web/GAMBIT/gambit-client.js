/**
 * GAMBIT Client - Centralized BLE client for GAMBIT firmware
 *
 * Handles:
 * - BLE connection with proper packet buffering
 * - Firmware version detection and compatibility checking
 * - Data streaming with automatic keepalive
 * - Event-based architecture for easy integration
 *
 * Usage:
 *   const client = new GambitClient();
 *   client.on('data', (telemetry) => { ... });
 *   client.on('firmware', (info) => { ... });
 *   client.on('error', (err) => { ... });
 *   await client.connect();
 *   client.startStreaming();
 */

class GambitClient {
    constructor(options = {}) {
        this.options = {
            debug: options.debug !== false, // Debug enabled by default
            autoKeepalive: options.autoKeepalive !== false, // Auto-refresh stream
            keepaliveInterval: options.keepaliveInterval || 25000, // 25 seconds
            ...options
        };

        // State
        this.connected = false;
        this.streaming = false;
        this.connection = null;
        this.firmwareInfo = null;

        // Buffering for BLE packet fragmentation
        this.dataBuffer = '';
        this.dataPacketCount = 0;
        this.lastDataTime = null;

        // Event handlers
        this.eventHandlers = {
            connect: [],
            disconnect: [],
            data: [],
            firmware: [],
            error: [],
            debug: []
        };

        // Keepalive timer
        this.keepaliveTimer = null;
    }

    /**
     * Register event handler
     * Events: 'connect', 'disconnect', 'data', 'firmware', 'error', 'debug'
     */
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        } else {
            this.emit('error', new Error(`Unknown event: ${event}`));
        }
        return this;
    }

    /**
     * Remove event handler
     */
    off(event, handler) {
        if (this.eventHandlers[event]) {
            const idx = this.eventHandlers[event].indexOf(handler);
            if (idx >= 0) {
                this.eventHandlers[event].splice(idx, 1);
            }
        }
        return this;
    }

    /**
     * Emit event to all registered handlers
     */
    emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (e) {
                    console.error(`[GAMBIT] Error in ${event} handler:`, e);
                }
            });
        }
    }

    /**
     * Debug logging
     */
    debug(message, ...args) {
        if (this.options.debug) {
            console.log(`[GAMBIT] ${message}`, ...args);
            this.emit('debug', { message, args });
        }
    }

    /**
     * Connect to GAMBIT device via BLE
     */
    connect() {
        return new Promise((resolve, reject) => {
            if (this.connected) {
                reject(new Error('Already connected'));
                return;
            }

            this.debug('========== CONNECTION STARTED ==========');
            this.debug('Browser:', navigator.userAgent);
            this.debug('BLE support:', navigator.bluetooth ? 'YES' : 'NO');

            if (!navigator.bluetooth) {
                const err = new Error('Web Bluetooth not supported');
                this.emit('error', err);
                reject(err);
                return;
            }

            const connectStartTime = Date.now();

            Puck.connect(conn => {
                const connectDuration = Date.now() - connectStartTime;
                this.debug(`Connection callback fired (${connectDuration}ms)`);

                if (!conn) {
                    const err = new Error('Connection failed - user cancelled or device unavailable');
                    this.debug('ERROR:', err.message);
                    this.emit('error', err);
                    reject(err);
                    return;
                }

                this.debug('Connection successful!');
                this.debug('Device:', conn.device ? conn.device.name : 'N/A');

                this.connection = conn;
                this.connected = true;

                // Set up data handler with buffering
                conn.on('data', (data) => this.handleData(data));

                // Set up close handler
                conn.on('close', () => this.handleDisconnect());

                this.emit('connect', {
                    deviceName: conn.device ? conn.device.name : null,
                    deviceId: conn.device ? conn.device.id : null
                });

                // Query firmware version
                this.queryFirmwareInfo()
                    .then(() => resolve(this.firmwareInfo))
                    .catch(err => {
                        this.debug('Warning: Could not query firmware info:', err.message);
                        resolve(null); // Don't fail connection if firmware query fails
                    });
            });
        });
    }

    /**
     * Disconnect from device
     */
    disconnect() {
        this.debug('Disconnecting...');

        if (this.keepaliveTimer) {
            clearInterval(this.keepaliveTimer);
            this.keepaliveTimer = null;
        }

        if (this.connection) {
            this.connection.close();
            this.connection = null;
        }

        this.connected = false;
        this.streaming = false;
        this.dataBuffer = '';
        this.dataPacketCount = 0;
        this.lastDataTime = null;

        this.emit('disconnect', {});
    }

    /**
     * Handle incoming BLE data with proper buffering
     */
    handleData(data) {
        // CRITICAL: BLE sends data in small chunks (20-byte MTU)
        // Must buffer data until we have complete newline-terminated lines
        this.dataBuffer += data;

        let lineEnd;
        while ((lineEnd = this.dataBuffer.indexOf('\n')) >= 0) {
            const line = this.dataBuffer.substring(0, lineEnd).trim();
            this.dataBuffer = this.dataBuffer.substring(lineEnd + 1);

            this.processLine(line);
        }
    }

    /**
     * Process a complete line of data
     */
    processLine(line) {
        if (line.startsWith('GAMBIT')) {
            this.dataPacketCount++;
            const now = Date.now();
            const timeSinceLast = this.lastDataTime ? (now - this.lastDataTime) : 0;
            this.lastDataTime = now;

            // Log every 50th packet (once per second at 20Hz)
            if (this.dataPacketCount % 50 === 1) {
                this.debug(`Data packets received: ${this.dataPacketCount} (interval: ${timeSinceLast}ms)`);
            }

            try {
                const jsonStr = line.substring(6); // Remove "GAMBIT" prefix
                const telemetry = JSON.parse(jsonStr);

                this.emit('data', telemetry);
            } catch (e) {
                this.debug('ERROR parsing telemetry:', e);
                this.debug('Raw line:', line);
                this.emit('error', new Error(`JSON parse error: ${e.message}`));
            }
        } else if (line.startsWith('FIRMWARE')) {
            // Firmware info response
            try {
                const jsonStr = line.substring(8); // Remove "FIRMWARE" prefix
                const firmwareInfo = JSON.parse(jsonStr);
                this.firmwareInfo = firmwareInfo;
                this.emit('firmware', firmwareInfo);
            } catch (e) {
                this.debug('ERROR parsing firmware info:', e);
            }
        } else if (line.length > 0) {
            // Log other device output (ignore empty lines and prompts)
            if (!line.match(/^[>\s]*$/)) {
                this.debug('Device output:', line);
            }
        }
    }

    /**
     * Handle disconnection
     */
    handleDisconnect() {
        this.debug('Device connection closed');

        if (this.keepaliveTimer) {
            clearInterval(this.keepaliveTimer);
            this.keepaliveTimer = null;
        }

        this.connected = false;
        this.streaming = false;
        this.dataBuffer = '';
        this.dataPacketCount = 0;
        this.lastDataTime = null;

        this.emit('disconnect', {});
    }

    /**
     * Query firmware version and info
     */
    queryFirmwareInfo() {
        return new Promise((resolve, reject) => {
            if (!this.connected) {
                reject(new Error('Not connected'));
                return;
            }

            this.debug('Querying firmware info...');

            // Set up one-time handler for firmware response
            const firmwareHandler = (info) => {
                this.off('firmware', firmwareHandler);
                clearTimeout(timeout);
                this.debug('Firmware info received:', info);
                resolve(info);
            };

            const timeout = setTimeout(() => {
                this.off('firmware', firmwareHandler);
                reject(new Error('Firmware query timeout (5s) - device may not support getFirmware()'));
            }, 5000);

            this.on('firmware', firmwareHandler);

            // Send getFirmware() command
            this.connection.write('getFirmware()\n', (err) => {
                if (err) {
                    this.off('firmware', firmwareHandler);
                    clearTimeout(timeout);
                    reject(new Error(`getFirmware() command failed: ${err}`));
                }
            });
        });
    }

    /**
     * Start streaming sensor data
     */
    startStreaming() {
        return new Promise((resolve, reject) => {
            if (!this.connected) {
                reject(new Error('Not connected'));
                return;
            }

            if (this.streaming) {
                resolve(); // Already streaming
                return;
            }

            this.debug('Starting data stream...');
            this.debug('Sending: getData()');

            this.connection.write('getData()\n', (err) => {
                if (err) {
                    this.debug('ERROR: getData() command failed:', err);
                    reject(new Error(`Failed to start streaming: ${err}`));
                    return;
                }

                this.debug('getData() command sent successfully');
                this.streaming = true;

                // Set up auto-keepalive if enabled
                if (this.options.autoKeepalive) {
                    this.setupKeepalive();
                }

                resolve();
            });
        });
    }

    /**
     * Stop streaming sensor data
     */
    stopStreaming() {
        return new Promise((resolve, reject) => {
            if (!this.connected) {
                reject(new Error('Not connected'));
                return;
            }

            if (!this.streaming) {
                resolve(); // Already stopped
                return;
            }

            this.debug('Stopping data stream...');

            if (this.keepaliveTimer) {
                clearInterval(this.keepaliveTimer);
                this.keepaliveTimer = null;
            }

            this.connection.write('stopData()\n', (err) => {
                if (err) {
                    this.debug('ERROR: stopData() command failed:', err);
                    // Don't reject - streaming will stop after 30s timeout anyway
                }

                this.streaming = false;
                this.debug('Data stream stopped');
                resolve();
            });
        });
    }

    /**
     * Set up automatic keepalive to refresh stream before 30s timeout
     */
    setupKeepalive() {
        if (this.keepaliveTimer) {
            clearInterval(this.keepaliveTimer);
        }

        this.debug(`Setting up keepalive (${this.options.keepaliveInterval}ms interval)`);

        this.keepaliveTimer = setInterval(() => {
            if (!this.connected || !this.streaming) {
                clearInterval(this.keepaliveTimer);
                this.keepaliveTimer = null;
                return;
            }

            this.debug('Refreshing data stream (keepalive)...');
            this.connection.write('getData()\n', (err) => {
                if (err) {
                    this.debug('ERROR: getData() refresh failed:', err);
                    this.emit('error', new Error(`Keepalive failed: ${err}`));
                } else {
                    this.debug('Data stream refreshed');
                }
            });
        }, this.options.keepaliveInterval);
    }

    /**
     * Send custom command to device
     */
    sendCommand(command) {
        return new Promise((resolve, reject) => {
            if (!this.connected) {
                reject(new Error('Not connected'));
                return;
            }

            this.debug('Sending command:', command);

            this.connection.write(command + '\n', (err) => {
                if (err) {
                    this.debug('ERROR: Command failed:', err);
                    reject(err);
                } else {
                    this.debug('Command sent successfully');
                    resolve();
                }
            });
        });
    }

    /**
     * Check if firmware is compatible with this client
     */
    checkCompatibility(minVersion = '1.0.0') {
        if (!this.firmwareInfo) {
            return {
                compatible: false,
                reason: 'Firmware info not available - device may not be running GAMBIT firmware'
            };
        }

        if (this.firmwareInfo.id !== 'GAMBIT') {
            return {
                compatible: false,
                reason: `Wrong firmware - expected GAMBIT, got ${this.firmwareInfo.id}`
            };
        }

        // Semantic version comparison
        const parseVersion = (v) => v.split('.').map(Number);
        const fwVersion = parseVersion(this.firmwareInfo.version);
        const minVer = parseVersion(minVersion);

        for (let i = 0; i < 3; i++) {
            if (fwVersion[i] > minVer[i]) {
                return { compatible: true };
            }
            if (fwVersion[i] < minVer[i]) {
                return {
                    compatible: false,
                    reason: `Firmware version too old - need ${minVersion}, got ${this.firmwareInfo.version}`
                };
            }
        }

        return { compatible: true };
    }
}

// Export for use in web apps
if (typeof window !== 'undefined') {
    window.GambitClient = GambitClient;
}
