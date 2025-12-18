/**
 * @puck - Puck.js BLE library wrapper for SIMCAP
 *
 * This package provides TypeScript types and utilities for the Puck.js
 * BLE library. The actual library is loaded via script tag from:
 * https://www.puck-js.com/puck.js
 *
 * Usage:
 * 1. Include <script src="/puck.js"></script> in your HTML
 * 2. Import types from this package: import { getPuck } from '@puck'
 * 3. Use getPuck() to access the global Puck object with proper types
 */

export * from './types';

import type { PuckStatic, PuckConnection } from './types';

// Re-export types
export type { PuckStatic, PuckConnection };

/**
 * Get the global Puck object with proper TypeScript types
 * @throws Error if Puck.js library is not loaded
 */
export function getPuck(): PuckStatic {
  if (typeof window === 'undefined') {
    throw new Error('Puck.js is only available in browser environments');
  }

  const puck = (window as any).Puck as PuckStatic | undefined;

  if (!puck) {
    throw new Error(
      'Puck.js library not loaded. Add <script src="/puck.js"></script> to your HTML.'
    );
  }

  return puck;
}

/**
 * Check if Puck.js library is available
 */
export function isPuckAvailable(): boolean {
  return typeof window !== 'undefined' && typeof (window as any).Puck !== 'undefined';
}

/**
 * Connect to a Puck.js device with Promise API
 */
export function connectPuck(): Promise<PuckConnection> {
  return new Promise((resolve, reject) => {
    const puck = getPuck();

    puck.connect((connection) => {
      if (connection) {
        resolve(connection);
      } else {
        reject(new Error('Connection cancelled or failed'));
      }
    });
  });
}

/**
 * Evaluate JavaScript on connected device with Promise API
 */
export function evalPuck<T = any>(expression: string): Promise<T> {
  return new Promise((resolve) => {
    const puck = getPuck();
    puck.eval(expression, resolve);
  });
}
