/**
 * Type declarations for external scripts loaded via script tags
 *
 * Note: Most filter/client code is now proper TypeScript modules.
 * These declarations are only for external libraries that remain as global scripts.
 */

// ===== Common Types (used by inline scripts) =====

interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

interface Vector3 {
  x: number;
  y: number;
  z: number;
}

interface EulerAngles {
  roll: number;
  pitch: number;
  yaw: number;
}

// ===== puck.js (external BLE library) =====

interface PuckConnectionCallback {
  (connection: any): void;
}

interface PuckWriteCallback {
  (): void;
}

declare const Puck: {
  debug: number;
  flowControl: boolean;
  chunkSize: number;

  connect(callback: PuckConnectionCallback): Promise<void>;
  write(data: string, callback?: PuckWriteCallback): void;
  eval(expression: string, callback: (result: any) => void): void;
  close(): void;
  isConnected(): boolean;

  setTime(): void;
  getBattery(): Promise<number>;
  LED1: { write: (value: boolean) => void };
  LED2: { write: (value: boolean) => void };
  LED3: { write: (value: boolean) => void };

  // Optional logging function (set dynamically)
  log?: (level: number, message: string) => void;
};

// ===== Three.js (loaded via CDN) =====

declare const THREE: typeof import('three');

// ===== GambitClient (exposed by entry point modules) =====
// Note: The actual class is in gambit-client.ts, but synth-app.ts
// and loader-app.ts expose it globally for inline scripts.

interface GambitClientOptions {
  debug?: boolean;
  autoKeepalive?: boolean;
}

interface GambitFirmwareInfo {
  name: string;
  version: string;
}

interface GambitCompatibilityResult {
  compatible: boolean;
  reason?: string;
}

// Interface for inline script usage (matches the exported class)
interface GambitClientInterface {
  connected: boolean;

  connect(): Promise<void>;
  disconnect(): void;

  on(event: 'data', callback: (data: any) => void): void;
  on(event: 'firmware', callback: (info: GambitFirmwareInfo) => void): void;
  on(event: 'disconnect', callback: () => void): void;
  on(event: 'error', callback: (error: Error) => void): void;
  off(event: string, callback: (...args: any[]) => void): void;

  startStreaming(): void;
  stopStreaming(): void;

  collectSamples(count: number): Promise<any[]>;
  checkCompatibility(minVersion: string): GambitCompatibilityResult;
}

// Declare GambitClient as available on window (set by synth-app.ts/loader-app.ts)
declare global {
  interface Window {
    GambitClient: new (options?: GambitClientOptions) => GambitClientInterface;
  }
}
