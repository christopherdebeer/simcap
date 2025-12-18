/**
 * Type definitions for Puck.js BLE library
 * @see https://www.puck-js.com/puck.js
 */

export interface PuckConnection {
  isOpen: boolean;
  isOpening: boolean;
  device?: {
    name: string;
    id: string;
  };
  on(event: 'data', handler: (data: string) => void): void;
  on(event: 'close', handler: () => void): void;
  on(event: string, handler: (...args: any[]) => void): void;
  write(data: string, callback?: (err?: Error) => void): void;
  close(): void;
}

export interface PuckConnectionCallback {
  (connection: PuckConnection | null): void;
}

export interface PuckWriteCallback {
  (): void;
}

export interface PuckEvalCallback<T = any> {
  (result: T): void;
}

export interface PuckLogCallback {
  (level: number, message: string): void;
}

export interface PuckLED {
  write: (value: boolean) => void;
}

export interface PuckStatic {
  /** Debug level (0=off, 1=errors, 2=warnings, 3=info) */
  debug: number;

  /** Enable flow control */
  flowControl: boolean;

  /** Chunk size for writes */
  chunkSize: number;

  /**
   * Connect to a Puck.js device
   * Opens the Web Bluetooth device chooser
   */
  connect(callback: PuckConnectionCallback): Promise<void>;

  /**
   * Write data to the connected device
   */
  write(data: string, callback?: PuckWriteCallback): void;

  /**
   * Evaluate JavaScript on the connected device
   */
  eval<T = any>(expression: string, callback: PuckEvalCallback<T>): void;

  /**
   * Close the current connection
   */
  close(): void;

  /**
   * Check if currently connected
   */
  isConnected(): boolean;

  /**
   * Sync time with the device
   */
  setTime(): void;

  /**
   * Get battery level (0-1)
   */
  getBattery(): Promise<number>;

  /** Red LED control */
  LED1: PuckLED;

  /** Green LED control */
  LED2: PuckLED;

  /** Blue LED control */
  LED3: PuckLED;

  /**
   * Optional logging function (can be overridden)
   */
  log?: PuckLogCallback;
}

// Re-export for convenience
export type { PuckStatic as Puck };
