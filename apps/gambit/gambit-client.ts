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

import type {
  RawTelemetry,
  TelemetrySample,
  FirmwareInfo as CoreFirmwareInfo,
  CompatibilityResult as CoreCompatibilityResult,
  DeviceLogEntry,
  DeviceLogStats,
  SampleCollectionResult,
} from '@core/types';
import type { PuckConnection, PuckStatic } from '@puck/types';

// Declare global Puck from external library
declare const Puck: PuckStatic;

// ===== Type Definitions =====

// Re-export core types with client-specific extensions
export interface FirmwareInfo extends CoreFirmwareInfo {}
export interface CompatibilityResult extends CoreCompatibilityResult {}

export interface LogEntry extends DeviceLogEntry {}

export interface LogsResponse {
  logs: LogEntry[];
  total: number;
  since?: number;
}

export interface LogStats extends DeviceLogStats {}

export interface CollectionResult extends SampleCollectionResult {}

export interface GambitClientOptions {
  debug?: boolean;
  autoKeepalive?: boolean;
  keepaliveInterval?: number;
}

// ===== New Frame Types for v0.4.0 =====

export interface ButtonEvent {
  gesture: 'SINGLE_TAP' | 'DOUBLE_TAP' | 'TRIPLE_TAP' | 'LONG_PRESS' | 'VERY_LONG_PRESS';
  time: number;
  pressCount: number;
}

export interface ModeChangeEvent {
  mode: 'LOW_POWER' | 'NORMAL' | 'HIGH_RES' | 'BURST';
  config: {
    name: string;
    accelHz: number;
    magEvery: number;
    lightEvery: number;
    battEvery: number;
  };
}

export interface ContextChangeEvent {
  context: 'unknown' | 'stored' | 'held' | 'active' | 'table';
  from: string;
}

export interface StreamEvent {
  mode?: string;
  hz?: number;
  count?: number | null;
  samples?: number;
  duration?: number;
  time: number;
}

export interface MarkEvent {
  time: number;
  sampleCount: number;
}

export interface CalibrationEvent {
  type: string;
  light?: number;
  cap?: number;
}

export interface ConnectionEvent {
  connected: boolean;
  addr?: string;
}

/** Connection quality statistics */
export interface ConnectionStats {
  connected: boolean;
  rssi: number | null;
  rssiQuality: 'excellent' | 'good' | 'fair' | 'weak' | 'poor' | 'unknown';
  duration: number;
  packetsSent: number;
  bytesSent: number;
  reconnects: number;
  avgPacketRate: number;
}

/** Beacon status from device */
export interface BeaconStatus {
  enabled: boolean;
  interval: number;
}

/** Auto mode status from device */
export interface AutoModeEvent {
  enabled: boolean;
}

/** Adaptive streaming event when quality changes mode */
export interface AdaptiveEvent {
  quality: 'excellent' | 'good' | 'fair' | 'weak' | 'poor' | 'unknown';
  rssi: number | null;
  deliveryRate: number;
  newMode: 'LOW_POWER' | 'NORMAL' | 'HIGH_RES' | 'BURST';
}

/** FIFO batch sample (raw sensor data) */
export interface FifoBatchSample {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  mx?: number;
  my?: number;
  mz?: number;
  t: number;
  n: number;
}

/** FIFO batch event from high-resolution streaming */
export interface FifoBatchEvent {
  samples: FifoBatchSample[];
  count: number;
  total: number;
}

export interface CollectOptions {
  progressTimeoutMs?: number;
}

/** @deprecated Use RawTelemetry from @core/types */
export interface TelemetryData extends RawTelemetry {}

export interface LogLevel {
  name: string;
  color: string;
}

type EventHandler<T = unknown> = (data: T) => void;
type FrameHandler<T = unknown> = (data: T) => void;
type WildcardHandler = (type: string, data: unknown) => void;

// ===== Binary Telemetry Parser =====
// Binary format (28 bytes):
// Header: [0xAB, 0xCD] (magic bytes)
// IMU:    [ax:2][ay:2][az:2][gx:2][gy:2][gz:2][mx:2][my:2][mz:2]
// Time:   [t:4]
// Flags:  [flags:1] - [mode:2][ctx:3][grip:1][hasLight:1][hasBatt:1]
// Aux:    [light:1][battery:1][temp:1]

const BINARY_MAGIC = 0xABCD;
const BINARY_PACKET_SIZE = 28;

const MODE_CODES = ['LOW_POWER', 'NORMAL', 'HIGH_RES', 'BURST'] as const;
const CTX_CODES = ['unknown', 'stored', 'held', 'active', 'table'] as const;

export interface BinaryTelemetry {
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
  mode: string;
  ctx: string;
  grip: number | null;
  l: number | null;
  b: number | null;
  temp: number | null;
  s: number;
  n: number;
}

export class BinaryTelemetryParser {
  private buffer: Uint8Array = new Uint8Array(0);
  private handlers: ((data: BinaryTelemetry) => void)[] = [];

  onBinaryData(data: Uint8Array): void {
    // Append to buffer
    const newBuffer = new Uint8Array(this.buffer.length + data.length);
    newBuffer.set(this.buffer);
    newBuffer.set(data, this.buffer.length);
    this.buffer = newBuffer;

    this.processBuffer();
  }

  private processBuffer(): void {
    while (this.buffer.length >= BINARY_PACKET_SIZE) {
      // Look for magic header
      let found = false;
      for (let i = 0; i <= this.buffer.length - BINARY_PACKET_SIZE; i++) {
        const magic = (this.buffer[i] << 8) | this.buffer[i + 1];
        if (magic === BINARY_MAGIC) {
          // Found a packet
          const packet = this.buffer.slice(i, i + BINARY_PACKET_SIZE);
          this.parsePacket(packet);
          this.buffer = this.buffer.slice(i + BINARY_PACKET_SIZE);
          found = true;
          break;
        }
      }
      if (!found) {
        // No valid packet found, trim buffer
        if (this.buffer.length > BINARY_PACKET_SIZE * 2) {
          this.buffer = this.buffer.slice(this.buffer.length - BINARY_PACKET_SIZE);
        }
        break;
      }
    }
  }

  private parsePacket(packet: Uint8Array): void {
    const view = new DataView(packet.buffer, packet.byteOffset, packet.length);

    // Parse IMU data (16-bit signed, little-endian)
    const ax = view.getInt16(2, true);
    const ay = view.getInt16(4, true);
    const az = view.getInt16(6, true);
    const gx = view.getInt16(8, true);
    const gy = view.getInt16(10, true);
    const gz = view.getInt16(12, true);
    const mx = view.getInt16(14, true);
    const my = view.getInt16(16, true);
    const mz = view.getInt16(18, true);

    // Parse timestamp (32-bit unsigned, little-endian)
    const t = view.getUint32(20, true);

    // Parse flags
    const flags = packet[24];
    const modeCode = (flags >> 6) & 0x03;
    const ctxCode = (flags >> 3) & 0x07;
    const gripBit = (flags >> 2) & 0x01;
    const hasLight = (flags >> 1) & 0x01;
    const hasBatt = flags & 0x01;

    // Parse auxiliary data
    const light = hasLight ? packet[25] / 255 : null;
    const batt = hasBatt ? packet[26] : null;
    const temp = packet[27] - 40; // Reverse offset

    const telemetry: BinaryTelemetry = {
      ax, ay, az,
      gx, gy, gz,
      mx, my, mz,
      t,
      mode: MODE_CODES[modeCode]?.charAt(0) || 'N',
      ctx: CTX_CODES[ctxCode]?.charAt(0) || 'u',
      grip: gripBit,
      l: light,
      b: batt,
      temp: temp,
      s: 1, // Streaming
      n: 0  // Not tracked in binary
    };

    for (const handler of this.handlers) {
      handler(telemetry);
    }
  }

  on(handler: (data: BinaryTelemetry) => void): void {
    this.handlers.push(handler);
  }

  off(handler: (data: BinaryTelemetry) => void): void {
    const idx = this.handlers.indexOf(handler);
    if (idx !== -1) {
      this.handlers.splice(idx, 1);
    }
  }

  reset(): void {
    this.buffer = new Uint8Array(0);
  }
}

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
  private binaryParser: BinaryTelemetryParser;
  private eventHandlers: Record<string, EventHandler[]> = {};
  private debug: boolean;
  private firmwareInfo: FirmwareInfo | null = null;
  private keepaliveInterval: ReturnType<typeof setInterval> | null = null;
  private autoKeepalive: boolean;
  private keepaliveIntervalMs: number;
  private useBinaryProtocol: boolean = false;

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
    this.binaryParser = new BinaryTelemetryParser();
    this.debug = options.debug || false;
    this.autoKeepalive = options.autoKeepalive !== false;
    this.keepaliveIntervalMs = options.keepaliveInterval || 20000;

    // Wire up frame parser events to client events
    this.frameParser.on('FW', (data: unknown) => this._handleFirmware(data as FirmwareInfo));
    this.frameParser.on('T', (data: unknown) => this.emit('data', data));
    this.frameParser.on('LOGS', (data: unknown) => this._handleLogs(data as LogsResponse));
    this.frameParser.on('LOGS_CLEARED', (data: unknown) => this._handleLogsCleared(data));
    this.frameParser.on('LOG_STATS', (data: unknown) => this._handleLogStats(data as LogStats));

    // Wire up binary parser to emit data events
    this.binaryParser.on((data: BinaryTelemetry) => {
      // Convert BinaryTelemetry to RawTelemetry format for consistency
      this.emit('data', data as unknown as TelemetrySample);
    });

    // New v0.4.0 frame handlers
    this.frameParser.on('BTN', (data: unknown) => this.emit('button', data));
    this.frameParser.on('MODE', (data: unknown) => this._handleModeChange(data as ModeChangeEvent));
    this.frameParser.on('CTX', (data: unknown) => this.emit('context', data));
    this.frameParser.on('STREAM_START', (data: unknown) => this.emit('streamStart', data));
    this.frameParser.on('STREAM_STOP', (data: unknown) => this.emit('streamStop', data));
    this.frameParser.on('MARK', (data: unknown) => this.emit('mark', data));
    this.frameParser.on('CAL', (data: unknown) => this.emit('calibration', data));
    this.frameParser.on('SLEEP', (data: unknown) => this.emit('sleep', data));
    this.frameParser.on('CONN', (data: unknown) => this.emit('connection', data));
    this.frameParser.on('CONN_STATS', (data: unknown) => this.emit('connectionStats', data));
    this.frameParser.on('BEACON_STATUS', (data: unknown) => this.emit('beaconStatus', data));
    this.frameParser.on('AUTO_MODE', (data: unknown) => this.emit('autoMode', data));
    this.frameParser.on('ADAPTIVE', (data: unknown) => this.emit('adaptive', data));
    this.frameParser.on('FIFO', (data: unknown) => this._handleFifoBatch(data as FifoBatchEvent));
  }

  // ===== Event Handling =====

  // Method overloads for type-safe event handling
  on(event: 'data', handler: (data: TelemetrySample) => void): this;
  on(event: 'firmware', handler: (info: FirmwareInfo) => void): this;
  on(event: 'disconnect', handler: () => void): this;
  on(event: 'close', handler: () => void): this;
  on(event: 'error', handler: (error: Error) => void): this;
  on(event: 'connect', handler: (conn: PuckConnection) => void): this;
  on(event: 'streamStart', handler: (data?: StreamEvent) => void): this;
  on(event: 'streamStop', handler: (data?: StreamEvent) => void): this;
  // New v0.4.0 events
  on(event: 'button', handler: (data: ButtonEvent) => void): this;
  on(event: 'mode', handler: (data: ModeChangeEvent) => void): this;
  on(event: 'context', handler: (data: ContextChangeEvent) => void): this;
  on(event: 'mark', handler: (data: MarkEvent) => void): this;
  on(event: 'calibration', handler: (data: CalibrationEvent) => void): this;
  on(event: 'sleep', handler: (data: { time: number }) => void): this;
  on(event: 'fifoBatch', handler: (data: FifoBatchEvent) => void): this;
  on(event: 'connection', handler: (data: ConnectionEvent) => void): this;
  on(event: 'connectionStats', handler: (data: ConnectionStats) => void): this;
  on(event: 'beaconStatus', handler: (data: BeaconStatus) => void): this;
  on(event: 'autoMode', handler: (data: AutoModeEvent) => void): this;
  on(event: 'adaptive', handler: (data: AdaptiveEvent) => void): this;
  on<T = unknown>(event: string, handler: EventHandler<T>): this;
  on<T = unknown>(event: string, handler: EventHandler<T>): this {
    if (!this.eventHandlers[event]) {
      this.eventHandlers[event] = [];
    }
    this.eventHandlers[event].push(handler as EventHandler);
    return this;
  }

  // Method overloads for type-safe event unsubscription
  off(event: 'data', handler?: (data: TelemetrySample) => void): this;
  off(event: 'firmware', handler?: (info: FirmwareInfo) => void): this;
  off(event: 'disconnect', handler?: () => void): this;
  off(event: 'close', handler?: () => void): this;
  off(event: 'error', handler?: (error: Error) => void): this;
  off(event: string, handler?: EventHandler): this;
  off(event: string, handler?: ((...args: any[]) => void)): this {
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
        this.binaryParser.reset();

        conn.on('data', (data: string | ArrayBuffer) => {
          if (this.useBinaryProtocol) {
            // Convert string to Uint8Array for binary parsing
            let bytes: Uint8Array;
            if (typeof data === 'string') {
              bytes = new Uint8Array(data.split('').map(c => c.charCodeAt(0)));
            } else {
              bytes = new Uint8Array(data);
            }
            this.binaryParser.onBinaryData(bytes);
          } else {
            // Text-based frame protocol
            this.frameParser.onData(data as string);
          }
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

  // ===== Mode Control (v0.4.0+) =====

  /**
   * Set the sampling mode on the device.
   * @param mode - LOW_POWER, NORMAL, HIGH_RES, or BURST
   */
  setMode(mode: 'LOW_POWER' | 'NORMAL' | 'HIGH_RES' | 'BURST'): Promise<void> {
    return this.write(`\x10if(typeof setMode==="function")setMode("${mode}");\n`);
  }

  /**
   * Get the current sampling mode.
   */
  getMode(): Promise<ModeChangeEvent> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      const timeout = setTimeout(() => {
        this.frameParser.off('MODE', handler);
        reject(new Error('Get mode timeout'));
      }, 3000);

      const handler = (data: unknown) => {
        clearTimeout(timeout);
        this.frameParser.off('MODE', handler);
        resolve(data as ModeChangeEvent);
      };

      this.frameParser.on('MODE', handler);
      this.write('\x10if(typeof getMode==="function"){var m=getMode();sendFrame("MODE",m);}\n');
    });
  }

  /**
   * Cycle to the next sampling mode.
   */
  cycleMode(): Promise<void> {
    return this.write('\x10if(typeof cycleMode==="function")cycleMode();\n');
  }

  /**
   * Calibrate context sensors (light and capacitive).
   * Should be called when device is not being held.
   */
  calibrateContext(): Promise<void> {
    return this.write('\x10if(typeof calibrateContext==="function")calibrateContext();\n');
  }

  /**
   * Show battery level via LED pattern.
   */
  showBattery(): Promise<void> {
    return this.write('\x10if(typeof showBatteryLevel==="function")showBatteryLevel();\n');
  }

  // ===== Binary Protocol (v0.4.0+) =====

  /**
   * Enable or disable binary telemetry protocol.
   * Binary protocol provides ~4x smaller packets for higher throughput.
   * @param enabled - Whether to use binary protocol
   */
  setBinaryProtocol(enabled: boolean): Promise<void> {
    this.useBinaryProtocol = enabled;
    this._log(`Binary protocol ${enabled ? 'enabled' : 'disabled'}`);
    if (enabled) {
      this.binaryParser.reset();
    } else {
      this.frameParser.clear();
    }
    return this.write(`\x10if(typeof setBinaryProtocol==="function")setBinaryProtocol(${enabled});\n`);
  }

  /**
   * Check if binary protocol is currently enabled.
   */
  isBinaryProtocol(): boolean {
    return this.useBinaryProtocol;
  }

  // ===== Wake-on-Touch (v0.4.0+) =====

  /**
   * Enable wake-on-touch feature.
   * Device will auto-wake from sleep when touched.
   * Requires context calibration first.
   */
  enableWakeOnTouch(): Promise<void> {
    this._log('Enabling wake-on-touch');
    return this.write('\x10if(typeof enableWakeOnTouch==="function")enableWakeOnTouch();\n');
  }

  /**
   * Disable wake-on-touch feature.
   */
  disableWakeOnTouch(): Promise<void> {
    this._log('Disabling wake-on-touch');
    return this.write('\x10if(typeof disableWakeOnTouch==="function")disableWakeOnTouch();\n');
  }

  /**
   * Check if wake-on-touch is available (requires calibration).
   */
  isWakeOnTouchAvailable(): Promise<boolean> {
    return Puck.eval('typeof wakeOnTouchEnabled !== "undefined" && capBaseline !== null') as Promise<boolean>;
  }

  // ===== FIFO High-Resolution Streaming (v0.4.0+) =====

  /**
   * Start high-resolution FIFO streaming.
   * Samples at 416Hz (16x normal) using hardware FIFO buffering.
   * Data is delivered in batches via 'fifoBatch' events.
   * @param count - Optional number of samples to collect
   * @param hz - Optional sample rate (104, 208, 416, 833, 1660)
   */
  startFifoStream(count?: number, hz?: number): Promise<void> {
    this._log(`Starting FIFO stream${count ? ` (${count} samples)` : ''}${hz ? ` @ ${hz}Hz` : ''}`);
    const countStr = count ? count.toString() : 'null';
    const hzStr = hz ? `0x0${hz >= 1660 ? 8 : hz >= 833 ? 7 : hz >= 416 ? 6 : hz >= 208 ? 5 : 4}` : 'null';
    return this.write(`\x10if(typeof startFifoStream==="function")startFifoStream(${countStr},${hzStr});\n`);
  }

  /**
   * Stop high-resolution FIFO streaming.
   */
  stopFifoStream(): Promise<void> {
    this._log('Stopping FIFO stream');
    return this.write('\x10if(typeof stopFifoStream==="function")stopFifoStream();\n');
  }

  /**
   * Check if FIFO streaming is available (firmware v0.4.0+).
   */
  isFifoAvailable(): boolean {
    if (!this.firmwareInfo?.version) return false;
    const [major, minor] = this.firmwareInfo.version.split('.').map(Number);
    return major > 0 || (major === 0 && minor >= 4);
  }

  // ===== Connection Quality (v0.4.0+) =====

  /**
   * Get connection quality statistics.
   * Returns RSSI, packet stats, and connection duration.
   */
  getConnectionStats(): Promise<ConnectionStats> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      const timeout = setTimeout(() => {
        this.frameParser.off('CONN_STATS', handler);
        reject(new Error('Connection stats timeout'));
      }, 3000);

      const handler = (data: unknown) => {
        clearTimeout(timeout);
        this.frameParser.off('CONN_STATS', handler);
        resolve(data as ConnectionStats);
      };

      this.frameParser.on('CONN_STATS', handler);
      this.write('\x10if(typeof getConnStats==="function")getConnStats();\n');
    });
  }

  // ===== Beaconing (v0.4.0+) =====

  /**
   * Enable background beaconing.
   * Device will advertise its status (battery, mode, context) even when not connected.
   * @param intervalMs - Update interval in milliseconds (default 5000)
   */
  enableBeaconing(intervalMs?: number): Promise<void> {
    this._log(`Enabling beaconing${intervalMs ? ` @ ${intervalMs}ms` : ''}`);
    const cmd = intervalMs
      ? `\x10if(typeof enableBeaconing==="function")enableBeaconing(${intervalMs});\n`
      : '\x10if(typeof enableBeaconing==="function")enableBeaconing();\n';
    return this.write(cmd);
  }

  /**
   * Disable background beaconing.
   */
  disableBeaconing(): Promise<void> {
    this._log('Disabling beaconing');
    return this.write('\x10if(typeof disableBeaconing==="function")disableBeaconing();\n');
  }

  /**
   * Get current beaconing status.
   */
  getBeaconStatus(): Promise<BeaconStatus> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected()) {
        reject(new Error('Not connected'));
        return;
      }

      const timeout = setTimeout(() => {
        this.frameParser.off('BEACON_STATUS', handler);
        reject(new Error('Beacon status timeout'));
      }, 3000);

      const handler = (data: unknown) => {
        clearTimeout(timeout);
        this.frameParser.off('BEACON_STATUS', handler);
        resolve(data as BeaconStatus);
      };

      this.frameParser.on('BEACON_STATUS', handler);
      this.write('\x10if(typeof getBeaconStatus==="function")getBeaconStatus();\n');
    });
  }

  // ===== Auto Mode (v0.4.0+) =====

  /**
   * Enable auto mode switching based on device context.
   * Device will automatically switch modes based on grip, motion, and light.
   */
  enableAutoMode(): Promise<void> {
    this._log('Enabling auto mode');
    return this.write('\x10if(typeof setAutoMode==="function")setAutoMode(true);\n');
  }

  /**
   * Disable auto mode switching for manual control.
   */
  disableAutoMode(): Promise<void> {
    this._log('Disabling auto mode');
    return this.write('\x10if(typeof setAutoMode==="function")setAutoMode(false);\n');
  }

  /**
   * Check if auto mode is currently enabled.
   */
  isAutoModeEnabled(): Promise<boolean> {
    return Puck.eval('typeof getAutoMode==="function"?getAutoMode():false') as Promise<boolean>;
  }

  // ===== Adaptive Streaming (v0.4.0+) =====

  /**
   * Enable adaptive streaming.
   * Device will automatically reduce streaming rate on poor connections.
   */
  enableAdaptiveStreaming(): Promise<void> {
    this._log('Enabling adaptive streaming');
    return this.write('\x10if(typeof enableAdaptiveStreaming==="function")enableAdaptiveStreaming();\n');
  }

  /**
   * Disable adaptive streaming.
   */
  disableAdaptiveStreaming(): Promise<void> {
    this._log('Disabling adaptive streaming');
    return this.write('\x10if(typeof disableAdaptiveStreaming==="function")disableAdaptiveStreaming();\n');
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

  private _handleModeChange(data: ModeChangeEvent): void {
    this._log(`Mode changed to: ${data.mode}`);
    this.emit('mode', data);
  }

  private _handleFifoBatch(data: FifoBatchEvent): void {
    this._log(`FIFO batch: ${data.count} samples (total: ${data.total})`);
    // Emit the batch event for high-resolution processing
    this.emit('fifoBatch', data);
    // Also emit individual samples as 'data' events for compatibility
    for (const sample of data.samples) {
      this.emit('data', sample as unknown as TelemetrySample);
    }
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

