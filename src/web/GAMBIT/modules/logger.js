/**
 * Logging Utility
 * Provides timestamped logging to UI
 */

let logElement = null;

/**
 * Initialize logger with DOM element
 * @param {HTMLElement} element - The log display element
 */
export function initLogger(element) {
    logElement = element;
}

/**
 * Log a message with timestamp
 * @param {string} msg - Message to log
 */
export function log(msg) {
    if (!logElement) {
        console.warn('Logger not initialized');
        return;
    }

    const time = new Date().toLocaleTimeString();
    logElement.innerHTML = `[${time}] ${msg}<br>` + logElement.innerHTML;

    // Keep only last 50 messages
    if (logElement.children.length > 50) {
        logElement.removeChild(logElement.lastChild);
    }
}
