#!/usr/bin/env npx tsx
/**
 * Firmware Build Script
 *
 * Uses EspruinoTools CLI to process firmware files at build time:
 * - Bundles required modules (fetched from GitHub)
 * - Minifies code
 * - Outputs production-ready firmware to dist/firmware/
 *
 * The espruino CLI handles module resolution, bundling, and minification natively.
 * We configure it to use GitHub raw URLs since espruino.com may be blocked.
 *
 * Usage:
 *   npx tsx scripts/build-firmware.ts [--watch] [--verbose]
 */

import { execSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, statSync, watchFile, unwatchFile } from 'fs';
import { join, dirname } from 'path';

// Configuration
const FIRMWARE_SRC_DIR = 'src/device';
const FIRMWARE_OUT_DIR = '.build/firmware';  // Use .build/ to avoid Vite clearing dist/

// Board configurations for different devices
const BOARD_CONFIG: Record<string, string> = {
    'GAMBIT': 'PUCKJS',
    'MOUSE': 'PUCKJS',
    'KEYBOARD': 'PUCKJS',
    'BAE': 'PUCKJS',
};

// Module URLs - GitHub raw as primary, espruino.com as fallback
// EspruinoDocs repo contains the module source files
const MODULE_URLS = [
    'https://raw.githubusercontent.com/espruino/EspruinoDocs/master/modules',
    'https://www.espruino.com/modules',
].join('|');

interface BuildResult {
    device: string;
    inputPath: string;
    outputPath: string;
    originalSize: number;
    minifiedSize: number;
    reduction: number;
    success: boolean;
    error?: string;
}

interface BuildOptions {
    watch?: boolean;
    verbose?: boolean;
    devices?: string[];
}

// ===== Fallback Minification =====
// Used when espruino CLI fails (e.g., network issues)

function minifyFallback(code: string): string {
    const tokens: string[] = [];
    let i = 0;

    while (i < code.length) {
        // String literals - preserve exactly
        if (code[i] === '"' || code[i] === "'" || code[i] === '`') {
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
 * Build a single firmware file using espruino CLI
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
        success: false,
    };

    try {
        if (!existsSync(inputPath)) {
            result.error = `Source file not found: ${inputPath}`;
            return result;
        }

        // Read source
        const sourceCode = readFileSync(inputPath, 'utf-8');
        result.originalSize = sourceCode.length;

        if (options.verbose) {
            console.log(`\nBuilding ${device}...`);
            console.log(`  Source: ${inputPath} (${(sourceCode.length / 1024).toFixed(1)} KB)`);
        }

        // Create output directory
        mkdirSync(dirname(outputPath), { recursive: true });

        // Try espruino CLI with GitHub module URLs
        // Note: MODULE_URL needs proper quoting since it contains pipe characters
        const moduleUrlConfig = `MODULE_URL=${MODULE_URLS}`;
        const espruinoCmd = [
            'npx', 'espruino',
            '--board', board,
            '--minify',
            '--config', `"${moduleUrlConfig}"`,
            '-o', outputPath,
            inputPath
        ].join(' ');

        if (options.verbose) {
            console.log(`  Running: ${espruinoCmd}`);
        }

        let espruinoSuccess = false;
        try {
            execSync(espruinoCmd, {
                stdio: options.verbose ? 'inherit' : 'pipe',
                encoding: 'utf-8',
                timeout: 60000, // 60 second timeout
                shell: true, // Use shell for proper quoting
            });
            espruinoSuccess = existsSync(outputPath) &&
                              readFileSync(outputPath, 'utf-8').length > 0;
        } catch (e) {
            if (options.verbose) {
                console.log('  Espruino CLI failed, using fallback minification...');
            }
        }

        // Fallback to our own minification if espruino CLI fails
        if (!espruinoSuccess) {
            if (options.verbose) {
                console.log('  Using fallback minification...');
            }
            const minified = minifyFallback(sourceCode);
            writeFileSync(outputPath, minified);
        }

        // Also write unminified version for debugging
        const debugPath = outputPath.replace('.min.js', '.debug.js');
        writeFileSync(debugPath, sourceCode);

        // Calculate results
        const minifiedCode = readFileSync(outputPath, 'utf-8');
        result.minifiedSize = minifiedCode.length;
        result.reduction = Math.round((1 - result.minifiedSize / result.originalSize) * 100);
        result.success = true;

        if (options.verbose) {
            console.log(`  Output: ${outputPath} (${(result.minifiedSize / 1024).toFixed(1)} KB)`);
            console.log(`  Reduction: ${result.reduction}%`);
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
    const devices = options.devices?.length ? options.devices : discoverFirmwareDevices();
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

    const devices = options.devices?.length ? options.devices : discoverFirmwareDevices();

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
