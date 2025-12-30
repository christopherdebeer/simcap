#!/usr/bin/env npx tsx
/**
 * Firmware Build Script
 *
 * Processes Espruino firmware files at build time:
 * - Bundles required modules from espruino.com or local cache
 * - Minifies code using EspruinoTools
 * - Outputs production-ready firmware to dist/firmware/
 *
 * Usage:
 *   npx tsx scripts/build-firmware.ts [--watch] [--verbose]
 */

import { execSync, spawn } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, statSync, watchFile, unwatchFile } from 'fs';
import { join, dirname, basename } from 'path';

// Configuration
const FIRMWARE_SRC_DIR = 'src/device';
const FIRMWARE_OUT_DIR = 'dist/firmware';
const MODULE_CACHE_DIR = 'node_modules/.cache/espruino-modules';
const ESPRUINO_MODULE_URL = 'https://www.espruino.com/modules/';

// Board configurations for different devices
const BOARD_CONFIG: Record<string, string> = {
    'GAMBIT': 'PUCKJS',
    'MOUSE': 'PUCKJS',
    'KEYBOARD': 'PUCKJS',
    'BAE': 'PUCKJS',
};

interface BuildResult {
    device: string;
    inputPath: string;
    outputPath: string;
    originalSize: number;
    minifiedSize: number;
    reduction: number;
    modules: string[];
    success: boolean;
    error?: string;
}

interface BuildOptions {
    watch?: boolean;
    verbose?: boolean;
    devices?: string[];
}

// ===== Module Resolution =====

/**
 * Extract module names from require() statements
 */
function extractModules(code: string): string[] {
    const requireRegex = /require\s*\(\s*["']([^"']+)["']\s*\)/g;
    const modules = new Set<string>();
    let match;

    while ((match = requireRegex.exec(code)) !== null) {
        // Skip built-in modules (Storage, etc.)
        const moduleName = match[1];
        if (!isBuiltinModule(moduleName)) {
            modules.add(moduleName);
        }
    }

    return Array.from(modules);
}

/**
 * Check if a module is built into Espruino
 */
function isBuiltinModule(name: string): boolean {
    const builtins = [
        'Storage', 'Flash', 'fs', 'http', 'net', 'dgram', 'tls',
        'crypto', 'neopixel', 'Wifi', 'ESP8266', 'ESP32', 'CC3000',
        'WIZnet', 'AT', 'MQTT', 'tensorflow', 'heatshrink'
    ];
    return builtins.includes(name);
}

/**
 * Fetch a module from Espruino CDN or local cache
 */
async function fetchModule(moduleName: string, verbose: boolean): Promise<string> {
    // Check cache first
    const cachePath = join(MODULE_CACHE_DIR, `${moduleName}.min.js`);
    if (existsSync(cachePath)) {
        if (verbose) console.log(`  [cache] ${moduleName}`);
        return readFileSync(cachePath, 'utf-8');
    }

    // Fetch from CDN
    const urls = [
        `${ESPRUINO_MODULE_URL}${moduleName}.min.js`,
        `${ESPRUINO_MODULE_URL}${moduleName}.js`
    ];

    for (const url of urls) {
        try {
            if (verbose) console.log(`  [fetch] ${url}`);
            const response = await fetch(url);
            if (response.ok) {
                const code = await response.text();

                // Cache the module
                mkdirSync(dirname(cachePath), { recursive: true });
                writeFileSync(cachePath, code);

                return code;
            }
        } catch (e) {
            // Continue to next URL
        }
    }

    throw new Error(`Module ${moduleName} not found`);
}

/**
 * Bundle all required modules into the code
 */
async function bundleModules(code: string, verbose: boolean): Promise<{ code: string; modules: string[] }> {
    const modules = extractModules(code);

    if (modules.length === 0) {
        return { code, modules: [] };
    }

    if (verbose) console.log(`  Bundling ${modules.length} modules...`);

    const moduleCode: string[] = [];
    for (const moduleName of modules) {
        try {
            const modCode = await fetchModule(moduleName, verbose);
            // Wrap in Modules.addCached for Espruino's module system
            moduleCode.push(`Modules.addCached("${moduleName}", function() {\n${modCode}\n});`);
        } catch (e) {
            console.warn(`  [warn] Could not fetch module: ${moduleName}`);
        }
    }

    const bundledCode = moduleCode.join('\n\n') + '\n\n' + code;
    return { code: bundledCode, modules };
}

// ===== Minification =====

/**
 * Minify code using EspruinoTools CLI
 */
async function minifyWithEspruino(inputPath: string, outputPath: string, board: string, verbose: boolean): Promise<boolean> {
    return new Promise((resolve) => {
        const args = [
            'espruino',
            '--board', board,
            '--minify',
            '-o', outputPath,
            inputPath
        ];

        if (verbose) {
            console.log(`  Running: npx ${args.join(' ')}`);
        }

        try {
            execSync(`npx ${args.join(' ')}`, {
                stdio: verbose ? 'inherit' : 'pipe',
                encoding: 'utf-8'
            });
            resolve(true);
        } catch (e) {
            if (verbose) console.error(`  [error] Espruino minification failed`);
            resolve(false);
        }
    });
}

/**
 * Fallback minification using our own tokenizer
 * (Same logic as in loader-app.ts but at build time)
 */
function minifyFallback(code: string): string {
    const tokens: string[] = [];
    let i = 0;

    while (i < code.length) {
        // String literals - preserve exactly
        if (code[i] === '"' || code[i] === "'") {
            const quote = code[i];
            let str = quote;
            i++;
            while (i < code.length && code[i] !== quote) {
                if (code[i] === '\\' && i + 1 < code.length) {
                    str += code[i] + code[i + 1];
                    i += 2;
                } else {
                    str += code[i];
                    i++;
                }
            }
            if (i < code.length) {
                str += code[i];
                i++;
            }
            tokens.push(str);
        }
        // Template literals - preserve exactly
        else if (code[i] === '`') {
            let str = '`';
            i++;
            while (i < code.length && code[i] !== '`') {
                if (code[i] === '\\' && i + 1 < code.length) {
                    str += code[i] + code[i + 1];
                    i += 2;
                } else {
                    str += code[i];
                    i++;
                }
            }
            if (i < code.length) {
                str += code[i];
                i++;
            }
            tokens.push(str);
        }
        // Single-line comment - skip
        else if (code[i] === '/' && code[i + 1] === '/') {
            while (i < code.length && code[i] !== '\n') i++;
        }
        // Multi-line comment - skip
        else if (code[i] === '/' && code[i + 1] === '*') {
            i += 2;
            while (i < code.length - 1 && !(code[i] === '*' && code[i + 1] === '/')) i++;
            i += 2;
        }
        // Whitespace - collapse to single space/newline
        else if (/\s/.test(code[i])) {
            let hasNewline = false;
            while (i < code.length && /\s/.test(code[i])) {
                if (code[i] === '\n') hasNewline = true;
                i++;
            }
            tokens.push(hasNewline ? '\n' : ' ');
        }
        // Regular code
        else {
            tokens.push(code[i]);
            i++;
        }
    }

    let result = tokens.join('');

    // Remove unnecessary spaces around operators/punctuation
    result = result.replace(/ ?([{}[\](),;:=<>+\-*/%&|!?]) ?/g, '$1');

    // Restore necessary spaces (keywords)
    result = result.replace(/\b(var|let|const|function|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|new|typeof|instanceof|in|of)\b/g, ' $1 ');

    // Clean up
    result = result.replace(/  +/g, ' ');
    result = result.replace(/\n\n+/g, '\n');
    result = result.replace(/^\s+|\s+$/gm, '');

    return result;
}

// ===== Build Process =====

/**
 * Build a single firmware file
 */
async function buildFirmware(device: string, options: BuildOptions): Promise<BuildResult> {
    const inputPath = join(FIRMWARE_SRC_DIR, device, 'app.js');
    const outputPath = join(FIRMWARE_OUT_DIR, device, 'app.min.js');
    const board = BOARD_CONFIG[device] || 'PUCKJS';

    const result: BuildResult = {
        device,
        inputPath,
        outputPath,
        originalSize: 0,
        minifiedSize: 0,
        reduction: 0,
        modules: [],
        success: false,
    };

    try {
        if (!existsSync(inputPath)) {
            result.error = `Source file not found: ${inputPath}`;
            return result;
        }

        // Read source
        let code = readFileSync(inputPath, 'utf-8');
        result.originalSize = code.length;

        if (options.verbose) {
            console.log(`\nBuilding ${device}...`);
            console.log(`  Source: ${inputPath} (${(code.length / 1024).toFixed(1)} KB)`);
        }

        // Bundle modules
        const bundled = await bundleModules(code, options.verbose || false);
        code = bundled.code;
        result.modules = bundled.modules;

        // Create output directory
        mkdirSync(dirname(outputPath), { recursive: true });

        // Write bundled (but not minified) version for espruino CLI
        const tempPath = outputPath.replace('.min.js', '.bundled.js');
        writeFileSync(tempPath, code);

        // Try espruino CLI minification first
        const espruinoSuccess = await minifyWithEspruino(tempPath, outputPath, board, options.verbose || false);

        if (!espruinoSuccess) {
            // Fallback to our own minification
            if (options.verbose) console.log('  Using fallback minification...');
            const minified = minifyFallback(code);
            writeFileSync(outputPath, minified);
        }

        // Also write unminified version for debugging
        const debugPath = outputPath.replace('.min.js', '.debug.js');
        writeFileSync(debugPath, code);

        // Calculate results
        const minifiedCode = readFileSync(outputPath, 'utf-8');
        result.minifiedSize = minifiedCode.length;
        result.reduction = Math.round((1 - result.minifiedSize / result.originalSize) * 100);
        result.success = true;

        if (options.verbose) {
            console.log(`  Output: ${outputPath} (${(result.minifiedSize / 1024).toFixed(1)} KB)`);
            console.log(`  Reduction: ${result.reduction}%`);
            if (result.modules.length > 0) {
                console.log(`  Bundled modules: ${result.modules.join(', ')}`);
            }
        }

    } catch (e) {
        result.error = (e as Error).message;
        if (options.verbose) {
            console.error(`  [error] ${result.error}`);
        }
    }

    return result;
}

/**
 * Discover all firmware directories
 */
function discoverFirmwareDevices(): string[] {
    const devices: string[] = [];

    if (!existsSync(FIRMWARE_SRC_DIR)) {
        return devices;
    }

    for (const entry of readdirSync(FIRMWARE_SRC_DIR)) {
        const entryPath = join(FIRMWARE_SRC_DIR, entry);
        const appPath = join(entryPath, 'app.js');

        if (statSync(entryPath).isDirectory() && existsSync(appPath)) {
            devices.push(entry);
        }
    }

    return devices;
}

/**
 * Build all firmware files
 */
async function buildAll(options: BuildOptions): Promise<BuildResult[]> {
    const devices = options.devices || discoverFirmwareDevices();
    const results: BuildResult[] = [];

    console.log(`\n🔧 Building ${devices.length} firmware file(s)...\n`);

    for (const device of devices) {
        const result = await buildFirmware(device, options);
        results.push(result);

        // Summary line
        if (result.success) {
            const sizeInfo = `${(result.originalSize / 1024).toFixed(1)}KB → ${(result.minifiedSize / 1024).toFixed(1)}KB (-${result.reduction}%)`;
            console.log(`  ✓ ${device}: ${sizeInfo}`);
        } else {
            console.log(`  ✗ ${device}: ${result.error}`);
        }
    }

    // Summary
    const successful = results.filter(r => r.success);
    const totalOriginal = successful.reduce((sum, r) => sum + r.originalSize, 0);
    const totalMinified = successful.reduce((sum, r) => sum + r.minifiedSize, 0);
    const avgReduction = successful.length > 0
        ? Math.round((1 - totalMinified / totalOriginal) * 100)
        : 0;

    console.log(`\n📦 Build complete: ${successful.length}/${results.length} successful`);
    if (successful.length > 0) {
        console.log(`   Total: ${(totalOriginal / 1024).toFixed(1)}KB → ${(totalMinified / 1024).toFixed(1)}KB (-${avgReduction}%)`);
    }

    return results;
}

/**
 * Watch mode - rebuild on changes
 */
async function watchMode(options: BuildOptions): Promise<void> {
    console.log('\n👀 Watching for firmware changes...\n');

    const devices = options.devices || discoverFirmwareDevices();

    for (const device of devices) {
        const inputPath = join(FIRMWARE_SRC_DIR, device, 'app.js');

        if (existsSync(inputPath)) {
            watchFile(inputPath, { interval: 1000 }, async () => {
                console.log(`\n📝 Change detected in ${device}/app.js`);
                await buildFirmware(device, options);
            });
            console.log(`  Watching: ${inputPath}`);
        }
    }

    // Keep process running
    process.on('SIGINT', () => {
        console.log('\n\nStopping watch mode...');
        for (const device of devices) {
            const inputPath = join(FIRMWARE_SRC_DIR, device, 'app.js');
            if (existsSync(inputPath)) {
                unwatchFile(inputPath);
            }
        }
        process.exit(0);
    });

    // Initial build
    await buildAll(options);
}

// ===== CLI =====

async function main() {
    const args = process.argv.slice(2);

    const options: BuildOptions = {
        watch: args.includes('--watch') || args.includes('-w'),
        verbose: args.includes('--verbose') || args.includes('-v'),
        devices: args.filter(a => !a.startsWith('-')),
    };

    if (options.devices?.length === 0) {
        options.devices = undefined; // Build all
    }

    if (options.watch) {
        await watchMode(options);
    } else {
        const results = await buildAll(options);

        // Exit with error if any builds failed
        const failed = results.filter(r => !r.success);
        if (failed.length > 0) {
            process.exit(1);
        }
    }
}

main().catch(e => {
    console.error('Build failed:', e);
    process.exit(1);
});
