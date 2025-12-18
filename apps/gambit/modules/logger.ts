/**
 * Logging Utility
 * Provides timestamped logging to UI
 */

let logElement: HTMLElement | null = null;

/**
 * Initialize logger with DOM element
 * @param element - The log display element
 */
export function initLogger(element: HTMLElement): void {
    logElement = element;
}

/**
 * Log a message with timestamp
 * @param msg - Message to log
 */
export function log(msg: string): void {
    if (!logElement) {
        console.log(`[LOG] ${msg}`);
        return;
    }

    const time = new Date().toLocaleTimeString();
    logElement.innerHTML = `[${time}] ${msg}<br>` + logElement.innerHTML;

    // Keep only last 50 messages
    if (logElement.children.length > 50) {
        logElement.removeChild(logElement.lastChild!);
    }
}

// ===== Default Export =====

export default {
    initLogger,
    log
};
