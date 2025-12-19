/**
 * Logging Utility
 * Provides timestamped logging to UI with buffer for export
 */

let logElement: HTMLElement | null = null;
const logBuffer: string[] = [];
const MAX_BUFFER_SIZE = 500;
const MAX_UI_MESSAGES = 250;

/**
 * Initialize logger with DOM element
 * @param element - The log display element (optional)
 */
export function initLogger(element?: HTMLElement | null): void {
    logElement = element ?? null;
}

/**
 * Log a message with timestamp
 * @param msg - Message to log
 * @param toConsole - Also log to console (default: true)
 */
export function log(msg: string, toConsole: boolean = true): void {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const entry = `[${time}] ${msg}`;

    // Always add to buffer
    logBuffer.push(entry);
    if (logBuffer.length > MAX_BUFFER_SIZE) {
        logBuffer.shift();
    }

    // Log to console
    if (toConsole) {
        console.log(msg);
    }

    // Update UI if element exists
    if (logElement) {
        logElement.innerHTML = logElement.innerHTML + `<br>${entry}`;

        // Keep UI manageable
        const lines = logElement.innerHTML.split('<br>');
        if (lines.length > MAX_UI_MESSAGES) {
            logElement.innerHTML = lines.slice(0, MAX_UI_MESSAGES).join('<br>');
        }
    }
}

/**
 * Get the log buffer
 */
export function getLogBuffer(): string[] {
    return [...logBuffer];
}

/**
 * Clear the log buffer and UI
 */
export function clearLog(): void {
    logBuffer.length = 0;
    if (logElement) {
        logElement.innerHTML = '';
    }
}

/**
 * Export log buffer as downloadable file
 */
export function exportLog(filename?: string): void {
    const content = logBuffer.join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename ?? `log-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    log('Log exported');
}

/**
 * Copy log buffer to clipboard
 * @returns Promise that resolves when copy is complete
 */
export async function copyLogToClipboard(): Promise<boolean> {
    const content = logBuffer.join('\n');
    try {
        await navigator.clipboard.writeText(content);
        log('Log copied to clipboard');
        return true;
    } catch (err) {
        console.error('Failed to copy log:', err);
        log('Failed to copy log to clipboard');
        return false;
    }
}

/**
 * Copy arbitrary text to clipboard with feedback
 * @param text - Text to copy
 * @param description - Description for log message
 * @returns Promise that resolves when copy is complete
 */
export async function copyToClipboard(text: string, description: string = 'Data'): Promise<boolean> {
    try {
        await navigator.clipboard.writeText(text);
        log(`${description} copied to clipboard`);
        return true;
    } catch (err) {
        console.error('Failed to copy:', err);
        log(`Failed to copy ${description} to clipboard`);
        return false;
    }
}

// ===== Default Export =====

export default {
    initLogger,
    log,
    getLogBuffer,
    clearLog,
    exportLog,
    copyLogToClipboard,
    copyToClipboard
};
