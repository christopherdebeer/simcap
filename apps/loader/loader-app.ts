/**
 * SIMCAP Firmware Loader Application
 * Upload firmware to Espruino Puck.js devices via WebBLE
 */

// ===== Type Definitions =====

interface FirmwareDefinition {
    name: string;
    description: string;
    path: string;
    size: string;
}

interface FirmwareInfo {
    name?: string;
    id?: string;
    version?: string;
    author?: string;
    uptime?: number;
    features?: string[];
}

interface LogEntry {
    t: number;
    l: string;
    m: string;
}

interface LogsData {
    count: number;
    total?: number;
    entries: LogEntry[];
}

interface AppState {
    connected: boolean;
    connecting: boolean;
    connectingStatus: string;
    selectedFirmware: string | null;
    connection: PuckConnection | null;
    uploading: boolean;
}

interface PuckConnection {
    isOpen: boolean;
    isOpening: boolean;
    device?: { name: string; id: string };
    on(event: string, handler: (data: any) => void): void;
    write(data: string, callback?: (err?: Error) => void): void;
    ondata?: (data: string) => void;
}

// Puck is declared globally in src/types/globals.d.ts

// ===== Firmware Definitions =====

const FIRMWARE: Record<string, FirmwareDefinition> = {
    gambit: {
        name: "GAMBIT",
        description: "9-DoF IMU telemetry for ML data collection",
        path: "/src/device/GAMBIT/app.js",
        size: "~4 KB"
    },
    mouse: {
        name: "MOUSE",
        description: "BLE HID Mouse - tilt to move cursor",
        path: "/src/device/MOUSE/app.js",
        size: "~3 KB"
    },
    keyboard: {
        name: "KEYBOARD",
        description: "BLE HID Keyboard - macros & gestures",
        path: "/src/device/KEYBOARD/app.js",
        size: "~5 KB"
    },
    bae: {
        name: "BAE",
        description: "Bluetooth Advertise Everything (reference)",
        path: "/src/device/BAE/app.js",
        size: "~2 KB"
    }
};

// ===== State =====

const state: AppState = {
    connected: false,
    connecting: false,
    connectingStatus: '',
    selectedFirmware: null,
    connection: null,
    uploading: false
};

// ===== Console Output Buffering =====
// Prevents UI freeze by batching console updates during upload

let pauseConsoleOutput = false;
let consoleBuffer: string[] = [];
let consoleFlushRAF: number | null = null;
const MAX_CONSOLE_BUFFER = 100; // Max buffered messages before forced flush
const MAX_CONSOLE_LINES = 500; // Max lines in console output element

function flushConsoleBuffer(): void {
    if (consoleBuffer.length === 0) return;

    // Batch all buffered data into a single DOM update
    const fragment = document.createDocumentFragment();
    for (const data of consoleBuffer) {
        const span = document.createElement('span');
        span.textContent = data;
        fragment.appendChild(span);
    }
    consoleBuffer = [];

    consoleOutput.appendChild(fragment);

    // Limit total console lines to prevent memory bloat
    while (consoleOutput.childNodes.length > MAX_CONSOLE_LINES) {
        consoleOutput.removeChild(consoleOutput.firstChild!);
    }

    consoleOutput.scrollTop = consoleOutput.scrollHeight;
    consoleFlushRAF = null;
}

function appendConsoleData(data: string): void {
    if (pauseConsoleOutput) {
        // During upload, buffer but don't exceed limit
        if (consoleBuffer.length < MAX_CONSOLE_BUFFER) {
            consoleBuffer.push(data);
        }
        return;
    }

    // Normal operation: batch updates using requestAnimationFrame
    consoleBuffer.push(data);

    if (!consoleFlushRAF) {
        consoleFlushRAF = requestAnimationFrame(flushConsoleBuffer);
    }
}

// ===== DOM Elements =====

const $ = (id: string) => document.getElementById(id);

let connectBtn: HTMLButtonElement;
let statusBar: HTMLElement;
let statusText: HTMLElement;
let firmwareGrid: HTMLElement;
let uploadBtn: HTMLButtonElement;
let saveBtn: HTMLButtonElement;
let uploadCustomBtn: HTMLButtonElement;
let resetBtn: HTMLButtonElement;
let customCode: HTMLTextAreaElement;
let consoleOutput: HTMLElement;
let consoleInput: HTMLInputElement;
let sendCmd: HTMLButtonElement;
let progressContainer: HTMLElement;
let progressFill: HTMLElement;
let progressText: HTMLElement;
let deviceInfo: HTMLElement;
let firmwareInfo: HTMLElement;
let firmwareDetails: HTMLElement;
let deviceLogsOutput: HTMLElement;
let logStats: HTMLElement;
let logFilterRow: HTMLElement;
let fetchLogsBtn: HTMLButtonElement;
let clearLogsBtn: HTMLButtonElement;
let exportLogsBtn: HTMLButtonElement;

// ===== Console Logging =====

// Log buffer for batched updates during upload
let logBuffer: Array<{msg: string, type: string}> = [];
let logFlushRAF: number | null = null;

function flushLogBuffer(): void {
    if (logBuffer.length === 0) return;

    const fragment = document.createDocumentFragment();
    for (const {msg, type} of logBuffer) {
        const line = document.createElement('div');
        line.className = type;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        fragment.appendChild(line);
    }
    logBuffer = [];

    consoleOutput.appendChild(fragment);

    // Limit total lines
    while (consoleOutput.childNodes.length > MAX_CONSOLE_LINES) {
        consoleOutput.removeChild(consoleOutput.firstChild!);
    }

    consoleOutput.scrollTop = consoleOutput.scrollHeight;
    logFlushRAF = null;
}

function log(msg: string, type: string = 'info'): void {
    // During upload, only log important messages to console (skip BLE noise)
    if (pauseConsoleOutput) {
        // Only log errors, successes, and [UPLOAD] step messages
        if (type === 'error' || type === 'success' || msg.startsWith('[UPLOAD]')) {
            console.log(`[loader] ${msg}`);
            logBuffer.push({msg, type});
        }
        // Skip all BLE spam during upload
        return;
    }

    // Normal operation: log everything
    console.log(`[loader] ${msg}`);

    // Batch UI updates using requestAnimationFrame
    logBuffer.push({msg, type});

    if (!logFlushRAF) {
        logFlushRAF = requestAnimationFrame(flushLogBuffer);
    }
}

// ===== Frame Parser =====

class FrameParser {
    private buffer: string = '';
    private handlers: Record<string, (data: any, type?: string) => void> = {};
    private debug: boolean = true;

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

            if (this.buffer.length < frameEnd) return;

            const payload = this.buffer.slice(payloadStart, payloadEnd);
            const etx = this.buffer[payloadEnd];

            if (etx !== '\x03') {
                this.buffer = this.buffer.slice(1);
                continue;
            }

            if (this.debug) log(`[FRAME] Received: ${type} (${length} bytes)`, 'success');

            try {
                const data = JSON.parse(payload);
                if (this.handlers[type]) {
                    this.handlers[type](data);
                } else if (this.handlers['*']) {
                    this.handlers['*'](data, type);
                }
            } catch (e) {
                log(`[FRAME] JSON parse error: ${(e as Error).message}`, 'error');
            }

            this.buffer = this.buffer.slice(frameEnd);
        }
    }

    on(type: string, handler: (data: any, type?: string) => void): void {
        this.handlers[type] = handler;
    }

    clear(): void {
        this.buffer = '';
    }
}

const frameParser = new FrameParser();
let firmwareQueryTimeout: ReturnType<typeof setTimeout> | null = null;
let deviceLogsQueryTimeout: ReturnType<typeof setTimeout> | null = null;
let deviceLogs: LogEntry[] = [];

// ===== Module Bundler =====

const ESPRUINO_MODULE_URL = 'https://www.espruino.com/modules/';
const moduleCache: Record<string, string> = {};

async function fetchModule(moduleName: string): Promise<string> {
    if (moduleCache[moduleName]) return moduleCache[moduleName];

    const urls = [
        ESPRUINO_MODULE_URL + moduleName + '.min.js',
        ESPRUINO_MODULE_URL + moduleName + '.js'
    ];

    for (const url of urls) {
        try {
            const response = await fetch(url);
            if (response.ok) {
                const code = await response.text();
                moduleCache[moduleName] = code;
                return code;
            }
        } catch (e) {
            // Continue to next URL
        }
    }

    throw new Error(`Module ${moduleName} not found`);
}

async function bundleModules(code: string): Promise<string> {
    const requireRegex = /require\s*\(\s*["']([^"']+)["']\s*\)/g;
    const modules = new Set<string>();
    let match;

    while ((match = requireRegex.exec(code)) !== null) {
        modules.add(match[1]);
    }

    if (modules.size === 0) return code;

    const moduleCode: string[] = [];
    for (const moduleName of modules) {
        const modCode = await fetchModule(moduleName);
        moduleCode.push(`Modules.addCached("${moduleName}", function() {\n${modCode}\n});`);
    }

    return moduleCode.join('\n\n') + '\n\n' + code;
}

// ===== Helper Functions =====

function formatUptime(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
    if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
}

function formatLogTime(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
        return `${hours}:${String(minutes % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
    }
    return `${minutes}:${String(seconds % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0').substring(0, 1)}`;
}

function getLogLevelInfo(level: string): { name: string; class: string; color: string } {
    switch (level) {
        case 'E': return { name: 'ERROR', class: 'error', color: 'var(--danger)' };
        case 'W': return { name: 'WARN', class: 'warn', color: 'var(--warning)' };
        case 'I': return { name: 'INFO', class: 'success', color: 'var(--success)' };
        case 'D': return { name: 'DEBUG', class: 'info', color: 'var(--fg-muted)' };
        default: return { name: level, class: 'info', color: 'var(--text)' };
    }
}

// ===== UI Updates =====

function updateUI(): void {
    let statusClass = 'disconnected';
    let statusLabel = 'Not connected';

    if (state.uploading) {
        statusClass = 'uploading';
        statusLabel = 'Uploading...';
    } else if (state.connected) {
        statusClass = 'connected';
        statusLabel = 'Connected';
    } else if (state.connecting) {
        statusClass = 'connecting';
        statusLabel = state.connectingStatus || 'Connecting...';
    }

    statusBar.className = 'status-bar ' + statusClass;
    statusText.textContent = statusLabel;
    connectBtn.textContent = state.connected ? 'Disconnect' : (state.connecting ? 'Cancel' : 'Connect');
    connectBtn.disabled = state.uploading;

    const canUpload = state.connected && !state.uploading;
    uploadBtn.disabled = !canUpload || !state.selectedFirmware;
    saveBtn.disabled = !canUpload;
    uploadCustomBtn.disabled = !canUpload;
    resetBtn.disabled = !canUpload;
    sendCmd.disabled = !canUpload;

    deviceInfo.style.display = state.connected ? 'block' : 'none';

    const canQuery = state.connected && !state.uploading;
    fetchLogsBtn.disabled = !canQuery;
    clearLogsBtn.disabled = !canQuery;
    exportLogsBtn.disabled = deviceLogs.length === 0;
}

// ===== Firmware Grid =====

function renderFirmwareGrid(): void {
    firmwareGrid.innerHTML = Object.entries(FIRMWARE).map(([id, fw]) => `
        <div class="firmware-card ${state.selectedFirmware === id ? 'selected' : ''}" data-firmware="${id}">
            <h3>${fw.name}</h3>
            <p>${fw.description}</p>
            <div class="size">${fw.size}</div>
            <button class="copy-btn" data-firmware="${id}" onclick="event.stopPropagation()">Copy to Custom</button>
        </div>
    `).join('');

    firmwareGrid.querySelectorAll('.firmware-card').forEach(card => {
        card.addEventListener('click', (e) => {
            if ((e.target as HTMLElement).classList.contains('copy-btn')) return;
            state.selectedFirmware = (card as HTMLElement).dataset.firmware || null;
            renderFirmwareGrid();
            updateUI();
            if (state.selectedFirmware) log(`Selected: ${FIRMWARE[state.selectedFirmware].name}`);
        });
    });

    firmwareGrid.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const fwId = (btn as HTMLElement).dataset.firmware;
            if (!fwId) return;
            const fw = FIRMWARE[fwId];

            try {
                const response = await fetch(fw.path);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                customCode.value = await response.text();
                log(`${fw.name} copied to custom tab`, 'success');
                document.querySelector('.tab[data-tab="custom"]')?.dispatchEvent(new Event('click'));
            } catch (err) {
                log(`Failed to copy firmware: ${(err as Error).message}`, 'error');
            }
        });
    });
}

// ===== Device Logs =====

function renderDeviceLogs(): void {
    if (deviceLogs.length === 0) {
        deviceLogsOutput.innerHTML = '<span class="info">No logs retrieved yet.</span>';
        return;
    }

    const showError = ($('filterError') as HTMLInputElement)?.checked ?? true;
    const showWarn = ($('filterWarn') as HTMLInputElement)?.checked ?? true;
    const showInfo = ($('filterInfo') as HTMLInputElement)?.checked ?? true;
    const showDebug = ($('filterDebug') as HTMLInputElement)?.checked ?? true;

    const filteredLogs = deviceLogs.filter(entry => {
        if (entry.l === 'E' && !showError) return false;
        if (entry.l === 'W' && !showWarn) return false;
        if (entry.l === 'I' && !showInfo) return false;
        if (entry.l === 'D' && !showDebug) return false;
        return true;
    });

    if (filteredLogs.length === 0) {
        deviceLogsOutput.innerHTML = '<span class="info">No logs match current filters.</span>';
        return;
    }

    deviceLogsOutput.innerHTML = filteredLogs.map(entry => {
        const levelInfo = getLogLevelInfo(entry.l);
        return `<div class="${levelInfo.class}" style="margin-bottom: 2px;">` +
            `<span style="color: var(--fg-muted);">[${formatLogTime(entry.t)}]</span> ` +
            `<span style="color: ${levelInfo.color}; font-weight: 600;">[${levelInfo.name}]</span> ` +
            `${entry.m}</div>`;
    }).join('');

    deviceLogsOutput.scrollTop = deviceLogsOutput.scrollHeight;
}

// ===== Firmware Display =====

function updateFirmwareDisplay(info: FirmwareInfo | null): void {
    if (!info) {
        firmwareDetails.innerHTML = `
            <div class="firmware-status not-detected">
                <span class="firmware-icon">⚠️</span>
                <div>
                    <div>No SIMCAP firmware detected</div>
                    <div class="firmware-version">Device may have custom code</div>
                </div>
            </div>
            <button class="refresh-btn" id="refreshFirmwareBtn">🔄 Refresh</button>
        `;
        $('refreshFirmwareBtn')?.addEventListener('click', queryFirmwareInfo);
        return;
    }

    const uptimeStr = info.uptime !== undefined ? formatUptime(info.uptime) : '--';
    const featuresHtml = info.features?.length ?
        `<div class="firmware-features">${info.features.map(f => `<span class="firmware-feature-tag">${f}</span>`).join('')}</div>` : '';

    firmwareDetails.innerHTML = `
        <div class="firmware-status detected">
            <span class="firmware-icon">✅</span>
            <div>
                <div class="firmware-name">${info.name || info.id || 'Unknown'}</div>
                <div class="firmware-version">v${info.version || '?'} ${info.author ? `by ${info.author}` : ''}</div>
            </div>
        </div>
        <div class="firmware-meta">
            <div class="firmware-meta-item"><span class="label">ID:</span><span class="value">${info.id || '--'}</span></div>
            <div class="firmware-meta-item"><span class="label">Uptime:</span><span class="value">${uptimeStr}</span></div>
        </div>
        ${featuresHtml}
        <button class="refresh-btn" id="refreshFirmwareBtn">🔄 Refresh</button>
    `;
    $('refreshFirmwareBtn')?.addEventListener('click', queryFirmwareInfo);
    log(`Firmware detected: ${info.name || info.id} v${info.version || '?'}`, 'success');
}

function queryFirmwareInfo(): void {
    if (!state.connected || !state.connection) return;

    firmwareInfo.style.display = 'block';
    firmwareDetails.innerHTML = '<div class="firmware-status unknown"><span class="firmware-icon">⏳</span><span>Querying...</span></div>';

    if (firmwareQueryTimeout) clearTimeout(firmwareQueryTimeout);

    firmwareQueryTimeout = setTimeout(() => {
        firmwareQueryTimeout = null;
        updateFirmwareDisplay(null);
    }, 3000);

    state.connection.write('\x10if(typeof getFirmware==="function")getFirmware();\n', (err) => {
        if (err && firmwareQueryTimeout) {
            clearTimeout(firmwareQueryTimeout);
            firmwareQueryTimeout = null;
            updateFirmwareDisplay(null);
        }
    });
}

// ===== Upload Verification =====

/**
 * Verify firmware upload by querying getFirmware() and waiting for FW frame response.
 * Returns true if firmware responds correctly, false otherwise.
 */
async function verifyFirmwareUpload(): Promise<boolean> {
    if (!state.connection) return false;

    return new Promise((resolve) => {
        let resolved = false;
        let timeoutId: ReturnType<typeof setTimeout>;

        // Use the wildcard handler to catch the FW frame
        const verifyHandler = (data: any, type?: string) => {
            if (!resolved && type === 'FW' && data && (data.id || data.name)) {
                resolved = true;
                clearTimeout(timeoutId);
                log(`[UPLOAD] Verified: ${data.name || data.id} v${data.version || '?'}`, 'success');
                resolve(true);
            }
        };

        // Temporarily register wildcard handler
        frameParser.on('*', verifyHandler);

        // Timeout after 5 seconds
        timeoutId = setTimeout(() => {
            if (!resolved) {
                resolved = true;
                log('[UPLOAD] Verification timeout - no firmware response', 'warn');
                resolve(false);
            }
        }, 5000);

        // Query firmware
        state.connection!.write('\x10if(typeof getFirmware==="function")getFirmware();\n', (err) => {
            if (err && !resolved) {
                resolved = true;
                clearTimeout(timeoutId);
                resolve(false);
            }
        });
    });
}

// ===== Upload Code =====

/**
 * Upload code to Espruino device with proper BLE flow control.
 *
 * Key improvements over naive approach:
 * - Uses echo(0) to suppress echo (halves data transfer)
 * - Uses \x03 (Ctrl-C) to clear any pending input
 * - Uses \x10 prefix for echo-off-per-line during upload
 * - Lets puck.js library handle BLE chunking (20 bytes default)
 * - Proper progress tracking via puck.writeProgress
 * - Throttled UI updates to prevent browser freeze
 *
 * See: https://www.espruino.com/Interfacing
 */
async function uploadCode(code: string, name: string = 'code'): Promise<void> {
    if (!state.connected || state.uploading || !state.connection) return;

    const uploadStart = Date.now();
    const logStep = (step: string) => log(`[UPLOAD] ${step} (+${Date.now() - uploadStart}ms)`);

    state.uploading = true;
    updateUI();
    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Bundling modules...';
    logStep(`Starting upload of ${name}`);

    // Store original writeProgress handler
    const originalWriteProgress = Puck.writeProgress;

    // Throttle progress updates to prevent UI freeze
    let lastProgressUpdate = 0;
    let lastProgressLog = 0;
    let pendingProgress: { sent: number; total: number } | null = null;
    let progressRAF: number | null = null;

    const updateProgressUI = () => {
        if (pendingProgress && pendingProgress.total > 0) {
            const { sent, total } = pendingProgress;
            const uploadProgress = 15 + (sent / total) * 75;
            progressFill.style.width = uploadProgress + '%';
            progressText.textContent = `Uploading... ${(sent / 1024).toFixed(1)}/${(total / 1024).toFixed(1)} KB`;
            pendingProgress = null;
        }
        progressRAF = null;
    };

    try {
        // Step 1: Bundle any required modules
        progressFill.style.width = '5%';
        code = await bundleModules(code);
        logStep(`Bundled: ${code.length} bytes (${code.split('\n').length} lines)`);

        // Step 2: Prepare the code
        // - Prefix with \x03 to clear input line (Ctrl-C)
        // - Use \x10 at start of each line to suppress echo per-line
        progressText.textContent = 'Preparing code...';
        progressFill.style.width = '8%';

        // Split code into lines and prefix each with \x10 (echo off for line)
        const lines = code.split('\n');
        const preparedCode = lines.map(line => '\x10' + line).join('\n');
        logStep(`Prepared: ${preparedCode.length} bytes with \\x10 prefixes`);

        // Store for XOFF diagnostics
        uploadCodeLines = lines;
        uploadTotalBytes = preparedCode.length;
        uploadProgressBytes = 0;

        // Step 3: Clear saved code and reset to get clean memory state
        // IMPORTANT: reset() runs saved code from flash, which uses memory!
        // We must clear saved code first, otherwise device runs out of memory.
        progressText.textContent = 'Clearing saved code...';
        progressFill.style.width = '10%';

        // Pause console output during upload to prevent DOM flooding
        pauseConsoleOutput = true;

        logStep('Sending Ctrl-C...');
        await writeWithTimeout('\x03', 500); // Stop any running code

        logStep('Clearing saved code from flash...');
        await writeWithTimeout('E.setBootCode("");save();\n', 5000); // Clear boot code and save to flash
        await delay(500); // Wait for flash write

        logStep('Sending reset()...');
        progressText.textContent = 'Resetting device...';
        await writeWithTimeout('reset();\n', 3000); // Reset now starts with empty interpreter
        await delay(1000); // Wait for reset to complete

        logStep('Sending echo(0)...');
        await writeWithTimeout('echo(0);\n', 1000); // Disable echo
        await delay(200);
        logStep('Device ready (memory cleared), starting code upload');

        // Step 4: Upload the code using puck.js internal chunking
        progressText.textContent = 'Uploading code...';
        progressFill.style.width = '15%';

        // Dynamic timeout: reset when progress is made (handles XOFF/XON flow control)
        const STALL_TIMEOUT_MS = 30000; // 30s with no progress = stalled
        let lastProgressTime = Date.now();
        let stallTimeoutId: ReturnType<typeof setTimeout> | null = null;
        let rejectFn: ((err: Error) => void) | null = null;

        const resetStallTimeout = () => {
            lastProgressTime = Date.now();
            if (stallTimeoutId) clearTimeout(stallTimeoutId);
            stallTimeoutId = setTimeout(() => {
                const elapsed = Date.now() - uploadStart;
                const lastKB = lastProgressLog / 1024;
                rejectFn?.(new Error(`Upload stalled for ${STALL_TIMEOUT_MS/1000}s at ~${lastKB.toFixed(1)}KB (total: ${(elapsed/1000).toFixed(0)}s)`));
            }, STALL_TIMEOUT_MS);
        };

        // Set up throttled progress tracking via puck.js
        // Only update UI at most every 100ms to prevent freeze
        Puck.writeProgress = (sent?: number, total?: number) => {
            if (sent !== undefined && total !== undefined && total > 0) {
                const now = Date.now();
                pendingProgress = { sent, total };

                // Update global progress for XOFF diagnostics
                uploadProgressBytes = sent;

                // Reset stall timeout on any progress
                resetStallTimeout();

                // Throttle UI: update at most every 100ms
                if (now - lastProgressUpdate >= 100) {
                    lastProgressUpdate = now;
                    if (progressRAF) cancelAnimationFrame(progressRAF);
                    progressRAF = requestAnimationFrame(updateProgressUI);
                }

                // Log progress every 5KB
                if (sent - lastProgressLog >= 5000 || sent === total) {
                    lastProgressLog = sent;
                    logStep(`Progress: ${(sent / 1024).toFixed(1)}/${(total / 1024).toFixed(1)} KB (${Math.round(sent / total * 100)}%)`);
                }
            }
        };

        // Send the prepared code - let puck.js handle chunking
        // The \x10 prefix suppresses echo, \x03 at start clears any leftover
        const uploadData = '\x03' + preparedCode + '\n';
        logStep(`Calling puck.write() with ${uploadData.length} bytes`);

        // Track if upload is still in progress for connection close detection
        let uploadComplete = false;

        await new Promise<void>((resolve, reject) => {
            rejectFn = reject;
            resetStallTimeout(); // Start initial timeout

            // Poll for connection state during upload (puck.js doesn't support multiple handlers)
            const connectionCheck = setInterval(() => {
                if (!state.connected && !uploadComplete) {
                    clearInterval(connectionCheck);
                    if (stallTimeoutId) clearTimeout(stallTimeoutId);
                    reject(new Error('Connection lost during upload'));
                }
            }, 500);

            state.connection!.write(uploadData, (err?: Error) => {
                uploadComplete = true;
                clearInterval(connectionCheck);
                if (stallTimeoutId) clearTimeout(stallTimeoutId);
                if (err) {
                    logStep(`Write callback error: ${err.message}`);
                    reject(err);
                } else {
                    logStep('Write callback success');
                    resolve();
                }
            });
        });

        // Final progress update
        if (progressRAF) cancelAnimationFrame(progressRAF);
        updateProgressUI();

        // Step 5: Wait for code to finish executing
        // The write callback fires when data is sent, but device still needs time to parse/execute
        logStep('Code sent, waiting for execution...');
        progressText.textContent = 'Executing code...';
        progressFill.style.width = '90%';
        await delay(3000); // Give device 3s to parse and execute the code

        // Step 6: Re-enable echo
        logStep('Re-enabling echo...');
        progressText.textContent = 'Finalizing...';
        progressFill.style.width = '95%';
        await writeWithTimeout('echo(1);\n', 1000);
        await delay(500);

        // Step 7: Verify upload by querying firmware
        logStep('Verifying firmware...');
        progressText.textContent = 'Verifying...';

        const verified = await verifyFirmwareUpload();

        if (verified) {
            progressFill.style.width = '100%';
            progressText.textContent = 'Upload complete!';
            const totalTime = ((Date.now() - uploadStart) / 1000).toFixed(1);
            log(`[UPLOAD] ${name} uploaded and verified in ${totalTime}s`, 'success');
        } else {
            progressFill.style.width = '100%';
            progressText.textContent = 'Upload complete (unverified)';
            const totalTime = ((Date.now() - uploadStart) / 1000).toFixed(1);
            log(`[UPLOAD] ${name} uploaded in ${totalTime}s but verification failed - code may be corrupted`, 'warn');
        }

    } catch (err) {
        const totalTime = ((Date.now() - uploadStart) / 1000).toFixed(1);
        log(`[UPLOAD] FAILED after ${totalTime}s: ${(err as Error).message}`, 'error');
        progressText.textContent = `Error: ${(err as Error).message}`;

        // Try to re-enable echo on error
        try {
            await writeWithTimeout('echo(1);\n', 1000);
        } catch {
            // Ignore - connection may be broken
        }
    } finally {
        // Restore original progress handler
        Puck.writeProgress = originalWriteProgress;
        if (progressRAF) cancelAnimationFrame(progressRAF);

        // Clear XOFF diagnostic state
        uploadCodeLines = [];
        uploadTotalBytes = 0;
        uploadProgressBytes = 0;

        // Resume console output and flush buffers
        pauseConsoleOutput = false;
        flushConsoleBuffer();
        flushLogBuffer();

        state.uploading = false;
        updateUI();
        setTimeout(() => { progressContainer.style.display = 'none'; }, 2000);
    }
}

/**
 * Helper to write with timeout
 */
function writeWithTimeout(data: string, timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error(`Write timeout: ${data.substring(0, 20)}...`)), timeoutMs);
        state.connection!.write(data, (err?: Error) => {
            clearTimeout(timeout);
            if (err) reject(err);
            else resolve();
        });
    });
}

/**
 * Simple delay helper
 */
function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ===== Connection =====

async function handleConnect(): Promise<void> {
    if (state.connected) {
        Puck.close();
        state.connected = false;
        state.connection = null;
        log('Disconnected', 'warn');
        updateUI();
        return;
    }

    if (state.connecting) {
        Puck.close();
        state.connecting = false;
        state.connectingStatus = '';
        log('Connection cancelled', 'warn');
        updateUI();
        return;
    }

    state.connecting = true;
    state.connectingStatus = 'Scanning...';
    updateUI();
    log('Connecting...');

    const connectionTimeout = setTimeout(() => {
        if (state.connecting) {
            state.connecting = false;
            log('Connection timeout', 'error');
            Puck.close();
            updateUI();
        }
    }, 60000);

    Puck.connect(conn => {
        clearTimeout(connectionTimeout);
        state.connecting = false;
        state.connectingStatus = '';

        if (!conn) {
            log('Connection failed', 'error');
            updateUI();
            return;
        }

        state.connection = conn;
        state.connected = true;
        log('Connected!', 'success');

        conn.on('data', (data: string) => {
            frameParser.onData(data);
            // Use buffered console output to prevent UI freeze
            appendConsoleData(data);
        });

        frameParser.clear();

        conn.on('close', () => {
            state.connected = false;
            state.connection = null;
            log('Connection closed', 'warn');
            updateUI();
        });

        setTimeout(() => queryDeviceInfo(conn), 500);
        setTimeout(() => queryFirmwareInfo(), 1500);

        updateUI();
    });
}

function queryDeviceInfo(conn: PuckConnection): void {
    let batteryData = '';
    const originalHandler = conn.ondata;

    conn.ondata = (data: string) => {
        originalHandler?.(data);
        batteryData += data;
        const match = batteryData.match(/(\d+)/);
        if (match) {
            $('batteryLevel')!.textContent = match[1];
            conn.ondata = originalHandler;
            queryTemp();
        }
    };

    conn.write('\x10Bluetooth.println(JSON.stringify(Puck.getBatteryPercentage()))\n');

    function queryTemp(): void {
        let tempData = '';
        const tempHandler = conn.ondata;

        conn.ondata = (data: string) => {
            tempHandler?.(data);
            tempData += data;
            const match = tempData.match(/(\d+\.?\d*)/);
            if (match) {
                $('tempLevel')!.textContent = parseFloat(match[1]).toFixed(1);
                conn.ondata = tempHandler;
            }
        };

        conn.write('\x10Bluetooth.println(JSON.stringify(E.getTemperature()))\n');
    }
}

// ===== Frame Handlers =====

function setupFrameHandlers(): void {
    frameParser.on('FW', (data: FirmwareInfo) => {
        if (firmwareQueryTimeout) {
            clearTimeout(firmwareQueryTimeout);
            firmwareQueryTimeout = null;
        }
        updateFirmwareDisplay(data);
    });

    frameParser.on('LOGS', (data: LogsData) => {
        if (deviceLogsQueryTimeout) {
            clearTimeout(deviceLogsQueryTimeout);
            deviceLogsQueryTimeout = null;
        }
        deviceLogs = data.entries || [];
        logStats.style.display = 'block';
        logFilterRow.style.display = 'flex';
        $('logCount')!.textContent = String(data.total || deviceLogs.length);
        renderDeviceLogs();
        updateUI();
        log(`Fetched ${deviceLogs.length} log entries`, 'success');
    });

    frameParser.on('LOGS_CLEARED', () => {
        deviceLogs = [];
        renderDeviceLogs();
        updateUI();
        logStats.style.display = 'none';
        log('Device logs cleared', 'success');
    });
}

// ===== Event Listeners =====

function setupEventListeners(): void {
    connectBtn.addEventListener('click', handleConnect);

    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => (c as HTMLElement).style.display = 'none');
            tab.classList.add('active');
            const tabId = (tab as HTMLElement).dataset.tab;
            if (tabId) $('tab-' + tabId)!.style.display = 'block';
        });
    });

    uploadBtn.addEventListener('click', async () => {
        if (!state.selectedFirmware) return;
        const fw = FIRMWARE[state.selectedFirmware];
        try {
            const response = await fetch(fw.path);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            await uploadCode(await response.text(), fw.name);
        } catch (err) {
            log(`Failed to fetch firmware: ${(err as Error).message}`, 'error');
        }
    });

    saveBtn.addEventListener('click', () => {
        if (!state.connected) return;
        log('Saving to flash...');
        state.connection!.write('save();\n', (err) => {
            log(err ? 'Failed to save!' : 'Saved to flash!', err ? 'error' : 'success');
        });
    });

    uploadCustomBtn.addEventListener('click', async () => {
        const code = customCode.value.trim();
        if (code) await uploadCode(code, 'Custom Code');
    });

    resetBtn.addEventListener('click', () => {
        if (!state.connected) return;
        log('Resetting device...');
        state.connection!.write('reset();\n', (err) => {
            log(err ? 'Reset failed!' : 'Device reset!', err ? 'error' : 'success');
        });
    });

    sendCmd.addEventListener('click', () => {
        const cmd = consoleInput.value.trim();
        if (!cmd || !state.connected) return;
        log(`> ${cmd}`, 'info');
        state.connection!.write(cmd + '\n');
        consoleInput.value = '';
    });

    consoleInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendCmd.click();
    });

    fetchLogsBtn.addEventListener('click', () => {
        if (!state.connected) return;
        deviceLogsOutput.innerHTML = '<span class="info">Fetching logs...</span>';

        if (deviceLogsQueryTimeout) clearTimeout(deviceLogsQueryTimeout);
        deviceLogsQueryTimeout = setTimeout(() => {
            deviceLogsQueryTimeout = null;
            deviceLogsOutput.innerHTML = '<span class="warn">Timeout: Device may not support logging</span>';
        }, 5000);

        state.connection!.write('\x10if(typeof getLogs==="function")getLogs();\n');
    });

    clearLogsBtn.addEventListener('click', () => {
        if (!state.connected) return;
        state.connection!.write('\x10if(typeof clearLogs==="function")clearLogs();\n');
    });

    exportLogsBtn.addEventListener('click', () => {
        if (deviceLogs.length === 0) return;
        const exportData = {
            exportedAt: new Date().toISOString(),
            deviceName: state.connection?.device?.name || 'Unknown',
            logCount: deviceLogs.length,
            logs: deviceLogs
        };
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `device-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        log(`Exported ${deviceLogs.length} log entries`, 'success');
    });

    ['filterError', 'filterWarn', 'filterInfo', 'filterDebug'].forEach(id => {
        $(id)?.addEventListener('change', renderDeviceLogs);
    });
}

// ===== Puck.js Logging =====

// Track upload progress for XOFF diagnostics
let uploadProgressBytes = 0;
let uploadTotalBytes = 0;
let uploadCodeLines: string[] = [];

function setupPuckLogging(): void {
    if (typeof Puck === 'undefined') {
        log('WARNING: Puck.js library not loaded!', 'error');
        return;
    }

    // Level 1 = errors only, 2 = +warnings, 3 = +info (verbose)
    Puck.debug = 1; // Only errors - we handle important events ourselves
    const originalLog = Puck.log;

    // Patterns to always filter out (raw data transfer noise)
    const noisyPatterns = [
        /^BT> /,           // Raw BLE data
        /^Sending /,       // "Sending X bytes" or chunk data
        /^Got /,           // "Got X bytes"
        /^GATT /,          // GATT operations
        /^Write /,         // Write confirmations
        /^Received /,      // "Received X"
        /^Sent$/,          // "Sent"
    ];

    Puck.log = function(level: number, message: string) {
        // Skip calling originalLog to reduce noise in browser console
        // originalLog?.call(Puck, level, message);

        const isNoisy = noisyPatterns.some(p => p.test(message));

        // During upload, track XOFF events for diagnostics
        if (state.uploading) {
            // XOFF/XON detection - log with progress context
            if (message.includes('XOFF')) {
                const pct = uploadTotalBytes > 0 ? Math.round(uploadProgressBytes / uploadTotalBytes * 100) : 0;
                const approxLine = uploadCodeLines.length > 0
                    ? Math.round(pct / 100 * uploadCodeLines.length)
                    : 0;
                const linePreview = uploadCodeLines[approxLine]
                    ? uploadCodeLines[approxLine].substring(0, 50)
                    : '';
                log(`[UPLOAD] XOFF at ${pct}% (~line ${approxLine}): ${linePreview}...`, 'warn');
            } else if (message.includes('XON')) {
                const pct = uploadTotalBytes > 0 ? Math.round(uploadProgressBytes / uploadTotalBytes * 100) : 0;
                log(`[UPLOAD] XON resume at ${pct}%`, 'info');
            }

            if (level === 1 && !isNoisy) {
                log(`<BLE> ${message}`, 'error');
            }
            // Skip everything else during upload
            return;
        }

        // Normal operation: filter noise
        if (isNoisy) {
            return;
        }

        const logType = level === 1 ? 'error' : (level === 2 ? 'warn' : 'info');
        log(`<BLE> ${message}`, logType);

        if (state.connecting) {
            if (message.includes('Device Name:')) {
                state.connectingStatus = `Found: ${message.split('Device Name:')[1].trim()}`;
            } else if (message.includes('Getting Services')) {
                state.connectingStatus = 'Getting services...';
            } else if (message === 'Connected') {
                state.connectingStatus = 'Establishing connection...';
            } else if (message.includes('Disconnected') || message.includes('ERROR')) {
                state.connecting = false;
                state.connectingStatus = '';
            }
            updateUI();
        }
    };
}

// ===== Initialization =====

function init(): void {
    connectBtn = $('connectBtn') as HTMLButtonElement;
    statusBar = $('statusBar')!;
    statusText = $('statusText')!;
    firmwareGrid = $('firmwareGrid')!;
    uploadBtn = $('uploadBtn') as HTMLButtonElement;
    saveBtn = $('saveBtn') as HTMLButtonElement;
    uploadCustomBtn = $('uploadCustomBtn') as HTMLButtonElement;
    resetBtn = $('resetBtn') as HTMLButtonElement;
    customCode = $('customCode') as HTMLTextAreaElement;
    consoleOutput = $('consoleOutput')!;
    consoleInput = $('consoleInput') as HTMLInputElement;
    sendCmd = $('sendCmd') as HTMLButtonElement;
    progressContainer = $('progressContainer')!;
    progressFill = $('progressFill')!;
    progressText = $('progressText')!;
    deviceInfo = $('deviceInfo')!;
    firmwareInfo = $('firmwareInfo')!;
    firmwareDetails = $('firmwareDetails')!;
    deviceLogsOutput = $('deviceLogsOutput')!;
    logStats = $('logStats')!;
    logFilterRow = $('logFilterRow')!;
    fetchLogsBtn = $('fetchLogsBtn') as HTMLButtonElement;
    clearLogsBtn = $('clearLogsBtn') as HTMLButtonElement;
    exportLogsBtn = $('exportLogsBtn') as HTMLButtonElement;

    setupFrameHandlers();
    setupEventListeners();
    setupPuckLogging();
    renderFirmwareGrid();
    updateUI();

    log('Ready. Connect to a device to begin.');
    console.log('[loader-app] Firmware loader initialized');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
