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
 *   // Collect specific number of samples
 *   const result = await client.collectSamples(500, 50); // 500 samples at 50Hz
 *
 * @version 2.0.0
 * @requires puck.js
 */

// ===== Type Definitions =====

export interface FirmwareInfo {
  id: string;
  version: string;
  build?: string;
  uptime?: number;
}

export interface LogEntry {
  level: string;
  time: number;
  msg: string;
}

export interface LogsResponse {
  logs: LogEntry[];
  total: number;
  since?: number;
}

export interface LogStats {
  total: number;
  errors: number;
  warnings: number;
  info: number;
  debug: number;
}

export interface CollectionResult {
  collectedCount: number;
  requestedCount: number;
  durationMs: number;
  requestedHz: number;
  actualHz: number;
}

export interface GambitClientOptions {
  debug?: boolean;
  autoKeepalive?: boolean;
  keepaliveInterval?: number;
}

export interface CollectOptions {
  progressTimeoutMs?: number;
}

export interface CompatibilityResult {
  compatible: boolean;
  reason?: string;
}

export interface TelemetryData {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  mx: number;
  my: number;
  mz: number;
  t: number;
}

export interface LogLevel {
  name: string;
  color: string;
}

type EventHandler<T = unknown> = (data: T) => void;
type FrameHandler<T = unknown> = (data: T) => void;
type WildcardHandler = (type: string, data: unknown) => void;

// Puck.js interface (external library)
interface PuckConnection {
  isOpen: boolean;
  on(event: 'data', handler: (data: string) => void): void;
  on(event: 'close', handler: () => void): void;
  write(data: string, callback?: (err?: Error) => void): void;
  close(): void;
}

interface PuckStatic {
  connect(callback: (conn: PuckConnection | null) => void): void;
  eval(code: string): Promise<unknown>;
}

declare const Puck: PuckStatic;

// ===== Frame Parser =====
// Protocol: \x02TYPE:LENGTH\nPAYLOAD\x03

export class FrameParser {
  private buffer: string = '';
  private handlers: Record<string, FrameHandler[]> = {};
  public debug: boolean = false;

  onData(data: string): void {
    this.buffer += data;
    this.processBuffer();
  }

  private processBuffer(): void {
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

  on(type: string, handler: FrameHandler | WildcardHandler): void {
    if (!this.handlers[type]) {
      this.handlers[type] = [];
    }
    this.handlers[type].push(handler as FrameHandler);
  }

  off(type: string, handler?: FrameHandler): void {
    if (!this.handlers[type]) return;
    if (handler) {
      this.handlers[type] = this.handlers[type].filter(h => h !== handler);
    } else {
      delete this.handlers[type];
    }
  }

  emit(type: string, data: unknown): void {
    if (this.handlers[type]) {
      this.handlers[type].forEach(h => h(data));
    }
    if (this.handlers['*']) {
      this.handlers['*'].forEach(h => (h as WildcardHandler)(type, data));
    }
  }

  clear(): void {
    this.buffer = '';
  }
}

// ===== GAMBIT Client =====

export class GambitClient {
  private connection: PuckConnection | null = null;
  private frameParser: FrameParser;
  private eventHandlers: Record<string, EventHandler[]> = {};
  private debug: boolean;
  private firmwareInfo: FirmwareInfo | null = null;
  private keepaliveInterval: ReturnType<typeof setInterval> | null = null;
  private autoKeepalive: boolean;
  private keepaliveIntervalMs: number;

  // Static properties
  static LOG_LEVELS: Record<string, LogLevel> = {
    'E': { name: 'ERROR', color: '#ff4757' },
    'W': { name: 'WARN', color: '#ffa502' },
    'I': { name: 'INFO', color: '#00ff88' },
    'D': { name: 'DEBUG', color: '#888888' }
  };

  static FrameParser = FrameParser;

  constructor(options: GambitClientOptions = {}) {
    this.frameParser = new FrameParser();
    this.debug = options.debug || false;
    this.autoKeepalive = options.autoKeepalive !== false;
    this.keepaliveIntervalMs = options.keepaliveInterval || 20000;

    // Wire up frame parser events to client events
    this.frameParser.on('FW', (data: unknown) => this._handleFirmware(data as FirmwareInfo));
    this.frameParser.on('T', (data: unknown) => this.emit('data', data));
    this.frameParser.on('LOGS', (data: unknown) => this._handleLogs(data as LogsResponse));
    this.frameParser.on('LOGS_CLEARED', (data: unknown) => this._handleLogsCleared(data));
    this.frameParser.on('LOG_STATS', (data: unknown) => this._handleLogStats(data as LogStats));
  }

  // ===== Event Handling =====

  on<T = unknown>(event: string, handler: EventHandler<T>): this {
    if (!this.eventHandlers[event]) {
      this.eventHandlers[event] = [];
    }
    this.eventHandlers[event].push(handler as EventHandler);
    return this;
  }

  off(event: string, handler?: EventHandler): this {
    if (!this.eventHandlers[event]) return this;
    if (handler) {
      this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
    } else {
      delete this.eventHandlers[event];
    }
    return this;
  }

  emit(event: string, data?: unknown): this {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event].forEach(h => h(data));
    }
    return this;
  }

  // ===== Connection Management =====

  connect(): Promise<FirmwareInfo | null> {
    return new Promise((resolve, reject) => {
      if (this.connection && this.connection.isOpen) {
        resolve(this.firmwareInfo);
        return;
      }

      this._log('Connecting to GAMBIT device...');

      let callbackHandled = false;

      Puck.connect((conn) => {
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

        this.frameParser.clear();

        conn.on('data', (data) => {
          this.frameParser.onData(data);
        });

        conn.on('close', () => {
          this._log('Connection closed');
          this.connection = null;
          this._stopKeepalive();
          this.emit('disconnect');
          this.emit('close');
        });

        this.emit('connect', conn);

        setTimeout(() => {
          this._queryFirmware().then(resolve).catch(() => {
            resolve(null);
          });
        }, 500);
      });
    });
  }

  disconnect(): this {
    this._stopKeepalive();
    if (this.connection) {
      this.connection.close();
      this.connection = null;
    }
    return this;
  }

  isConnected(): boolean {
    return !!(this.connection && this.connection.isOpen);
  }

  // ===== Device Commands =====

  write(cmd: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      this.connection!.write(cmd, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  }

  private _queryFirmware(): Promise<FirmwareInfo> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      const timeout = setTimeout(() => {
        this.frameParser.off('FW', handler);
        reject(new Error('Firmware query timeout'));
      }, 3000);

      const handler = (data: unknown) => {
        clearTimeout(timeout);
        this.frameParser.off('FW', handler);
        this.firmwareInfo = data as FirmwareInfo;
        resolve(data as FirmwareInfo);
      };

      this.frameParser.on('FW', handler);
      this.write('\x10if(typeof getFirmware==="function")getFirmware();\n');
    });
  }

  getFirmware(): Promise<FirmwareInfo | null> {
    if (this.firmwareInfo) {
      return Promise.resolve(this.firmwareInfo);
    }
    return this._queryFirmware();
  }

  checkCompatibility(minVersion: string): CompatibilityResult {
    if (!this.firmwareInfo) {
      return { compatible: false, reason: 'No firmware info available' };
    }

    if (!this.firmwareInfo.id || this.firmwareInfo.id !== 'GAMBIT') {
      return { compatible: false, reason: 'Not GAMBIT firmware' };
    }

    if (!this.firmwareInfo.version) {
      return { compatible: true, reason: 'Version unknown, assuming compatible' };
    }

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

  getLogs(since?: number): Promise<LogsResponse> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      const timeout = setTimeout(() => {
        this.frameParser.off('LOGS', handler);
        reject(new Error('Logs query timeout'));
      }, 5000);

      const handler = (data: unknown) => {
        clearTimeout(timeout);
        this.frameParser.off('LOGS', handler);
        resolve(data as LogsResponse);
      };

      this.frameParser.on('LOGS', handler);
      const cmd = since !== undefined
        ? `\x10if(typeof getLogs==="function")getLogs(${since});\n`
        : '\x10if(typeof getLogs==="function")getLogs();\n';
      this.write(cmd);
    });
  }

  clearLogs(): Promise<unknown> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      const timeout = setTimeout(() => {
        this.frameParser.off('LOGS_CLEARED', handler);
        reject(new Error('Clear logs timeout'));
      }, 3000);

      const handler = (data: unknown) => {
        clearTimeout(timeout);
        this.frameParser.off('LOGS_CLEARED', handler);
        resolve(data);
      };

      this.frameParser.on('LOGS_CLEARED', handler);
      this.write('\x10if(typeof clearLogs==="function")clearLogs();\n');
    });
  }

  getLogStats(): Promise<LogStats> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      const timeout = setTimeout(() => {
        this.frameParser.off('LOG_STATS', handler);
        reject(new Error('Log stats timeout'));
      }, 3000);

      const handler = (data: unknown) => {
        clearTimeout(timeout);
        this.frameParser.off('LOG_STATS', handler);
        resolve(data as LogStats);
      };

      this.frameParser.on('LOG_STATS', handler);
      this.write('\x10if(typeof getLogStats==="function")getLogStats();\n');
    });
  }

  startStreaming(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      this._log('Starting telemetry stream...');

      this.write('\x10if(typeof getData==="function")getData();\n')
        .then(() => {
          if (this.autoKeepalive) {
            this._startKeepalive();
          }
          this.emit('streamStart');
          resolve();
        })
        .catch(reject);
    });
  }

  startStream(): Promise<void> {
    return this.startStreaming();
  }

  stopStreaming(): Promise<void> {
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

  stopStream(): Promise<void> {
    return this.stopStreaming();
  }

  collectSamples(count: number, hz: number = 50, options: CollectOptions = {}): Promise<CollectionResult> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      if (!count || count < 1) {
        reject(new Error('Invalid sample count'));
        return;
      }

      if (!hz || hz < 1 || hz > 100) {
        reject(new Error('Invalid sample rate (1-100 Hz)'));
        return;
      }

      const intervalMs = Math.floor(1000 / hz);
      const expectedDurationMs = Math.ceil((count / hz) * 1000);
      const progressTimeoutMs = options.progressTimeoutMs || 5000;

      this._log(`Collecting ${count} samples @ ${hz}Hz (~${(expectedDurationMs / 1000).toFixed(1)}s expected, ${(progressTimeoutMs / 1000).toFixed(1)}s progress timeout)...`);

      let sampleCount = 0;
      let startTime: number | null = null;
      let progressTimeout: ReturnType<typeof setTimeout> | null = null;

      const resetProgressTimeout = () => {
        if (progressTimeout) clearTimeout(progressTimeout);
        progressTimeout = setTimeout(() => {
          cleanup();
          const elapsed = startTime ? Date.now() - startTime : 0;
          reject(new Error(`Collection stalled - no samples received for ${progressTimeoutMs}ms (${sampleCount}/${count} samples in ${elapsed}ms)`));
        }, progressTimeoutMs);
      };

      const cleanup = () => {
        if (progressTimeout) clearTimeout(progressTimeout);
        this.off('data', dataHandler);
      };

      const dataHandler = () => {
        if (!startTime) startTime = Date.now();
        sampleCount++;
        resetProgressTimeout();

        if (sampleCount >= count) {
          const duration = Date.now() - startTime;
          const actualHz = Math.round((sampleCount / duration) * 1000);
          cleanup();
          this._log(`Collection complete: ${sampleCount} samples in ${duration}ms (${actualHz}Hz)`);
          resolve({
            collectedCount: sampleCount,
            requestedCount: count,
            durationMs: duration,
            requestedHz: hz,
            actualHz: actualHz
          });
        }
      };

      this.on('data', dataHandler);
      resetProgressTimeout();

      this.write(`\x10if(typeof getData==="function")getData(${count}, ${intervalMs});\n`)
        .catch((err) => {
          cleanup();
          reject(err);
        });
    });
  }

  getBattery(): Promise<number> {
    return Puck.eval('Puck.getBatteryPercentage()') as Promise<number>;
  }

  getTemperature(): Promise<number> {
    return Puck.eval('E.getTemperature()') as Promise<number>;
  }

  reset(): Promise<void> {
    return this.write('reset();\n');
  }

  save(): Promise<void> {
    return this.write('save();\n');
  }

  // ===== Internal Methods =====

  private _handleFirmware(data: FirmwareInfo): void {
    this.firmwareInfo = data;
    this.emit('firmware', data);
  }

  private _handleLogs(data: LogsResponse): void {
    this.emit('logs', data);
  }

  private _handleLogsCleared(data: unknown): void {
    this.emit('logsCleared', data);
  }

  private _handleLogStats(data: LogStats): void {
    this.emit('logStats', data);
  }

  private _startKeepalive(): void {
    this._stopKeepalive();
    this.keepaliveInterval = setInterval(() => {
      if (this.isConnected()) {
        this._log('Sending keepalive...');
        this.write('\x10if(typeof getData==="function")getData();\n');
      }
    }, this.keepaliveIntervalMs);
  }

  private _stopKeepalive(): void {
    if (this.keepaliveInterval) {
      clearInterval(this.keepaliveInterval);
      this.keepaliveInterval = null;
    }
  }

  private _log(msg: string): void {
    if (this.debug) {
      console.log(`[GambitClient] ${msg}`);
    }
  }

  // ===== Static Utility Functions =====

  static formatUptime(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
    if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  }

  static formatLogTime(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}:${String(minutes % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
    }
    return `${minutes}:${String(seconds % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0').substring(0, 1)}`;
  }
}

// Export as globals for backward compatibility
declare global {
  interface Window {
    GambitClient: typeof GambitClient;
    GAMBITClient: typeof GambitClient;
  }
}

if (typeof window !== 'undefined') {
  window.GambitClient = GambitClient;
  window.GAMBITClient = GambitClient;
}
