/**
 * Firmware Loader Application Entry Point
 *
 * This module imports GambitClient and exposes it globally
 * for the inline loader script in index.html.
 *
 * Future: Extract inline script into proper TypeScript modules.
 */

import { GambitClient } from '../GAMBIT/gambit-client';

// Expose GambitClient globally for inline script
declare global {
    interface Window {
        GambitClient: typeof GambitClient;
    }
}

window.GambitClient = GambitClient;

console.log('[loader-app] GambitClient loaded and available globally');
