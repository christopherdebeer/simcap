/**
 * GAMBIT Synth Application Entry Point
 *
 * This module imports GambitClient and exposes it globally
 * for the inline synth script in synth.html.
 *
 * Future: Extract inline script into proper TypeScript modules.
 */

import { GambitClient } from './gambit-client';

// Expose GambitClient globally for inline script
declare global {
    interface Window {
        GambitClient: typeof GambitClient;
    }
}

window.GambitClient = GambitClient;

console.log('[synth-app] GambitClient loaded and available globally');
