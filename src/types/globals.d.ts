/**
 * Type declarations for external scripts loaded via script tags
 *
 * Note: Most filter/client code is now proper TypeScript modules.
 * These declarations are only for external libraries that remain as global scripts.
 *
 * Geometry types (Vector3, Quaternion, EulerAngles) are now in @core/types.
 * Import from there instead of using these global interfaces.
 */

// Import canonical types from core (for type checking global interfaces)
import type {
  Vector3 as CoreVector3,
  Quaternion as CoreQuaternion,
  EulerAngles as CoreEulerAngles,
  FirmwareInfo,
  CompatibilityResult,
} from '@core/types';

// Make this file a module to allow `declare global`
export {};

// ===== Common Types (for inline scripts that can't use imports) =====
// These mirror the core types for use in global scope

interface Quaternion extends CoreQuaternion {}
interface Vector3 extends CoreVector3 {}
interface EulerAngles extends CoreEulerAngles {}

// ===== puck.js (external BLE library) =====

interface PuckConnectionCallback {
  (connection: any): void;
}

interface PuckWriteCallback {
  (): void;
}

// ===== GambitClient (exposed by entry point modules) =====
// Note: The actual class is in gambit-client.ts, but synth-app.ts
// and loader-app.ts expose it globally for inline scripts.

interface GambitClientOptions {
  debug?: boolean;
  autoKeepalive?: boolean;
}

/** @deprecated Use FirmwareInfo from @core/types */
interface GambitFirmwareInfo {
  name: string;
  version: string;
}

/** @deprecated Use CompatibilityResult from @core/types */
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

// Declare globals available on window (set by synth-app.ts/loader-app.ts)
declare global {
  // Three.js is loaded via CDN
  const THREE: any;

  // Puck.js BLE library
  const Puck: {
    version: string;
    debug: number;
    flowControl: boolean;
    increaseMTU: boolean;
    timeoutNormal: number;
    timeoutNewline: number;
    timeoutMax: number;
    connect(callback: PuckConnectionCallback): Promise<void>;
    write(data: string, callback?: PuckWriteCallback): void;
    eval(expression: string, callback: (result: any) => void): void;
    close(): void;
    isConnected(): boolean;
    getConnection(): any;
    setTime(): void;
    log?: (level: number, message: string) => void;
    /** Called with upload progress - can be overridden for custom progress tracking */
    writeProgress: (charsSent?: number, charsTotal?: number) => void;
  };

  interface Window {
    GambitClient: new (options?: GambitClientOptions) => GambitClientInterface;
  }
}
