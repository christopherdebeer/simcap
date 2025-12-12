/**
 * Playwright script to capture console errors from GAMBIT/index.html
 * Run: node debug-console.js
 */

const { chromium } = require('playwright');

const PORT = 8080;
const URL = `http://localhost:${PORT}/src/web/GAMBIT/index.html`;

async function main() {
    console.log('='.repeat(60));
    console.log('GAMBIT Console Error Debugger');
    console.log('='.repeat(60));
    console.log(`Loading: ${URL}\n`);

    const browser = await chromium.launch({
        headless: true
    });

    const context = await browser.newContext();
    const page = await context.newPage();

    // Collect all console messages
    const consoleMessages = [];
    const errors = [];

    page.on('console', msg => {
        const type = msg.type();
        const text = msg.text();
        const location = msg.location();

        const entry = {
            type,
            text,
            url: location.url,
            line: location.lineNumber,
            column: location.columnNumber
        };

        consoleMessages.push(entry);

        // Print immediately for errors
        if (type === 'error') {
            console.log('\n[CONSOLE ERROR]');
            console.log(`  Message: ${text}`);
            console.log(`  Location: ${location.url}:${location.lineNumber}:${location.columnNumber}`);
        }
    });

    // Capture page errors (uncaught exceptions)
    page.on('pageerror', error => {
        const entry = {
            name: error.name,
            message: error.message,
            stack: error.stack
        };
        errors.push(entry);

        console.log('\n' + '!'.repeat(60));
        console.log('[PAGE ERROR - UNCAUGHT EXCEPTION]');
        console.log('!'.repeat(60));
        console.log(`Name: ${error.name}`);
        console.log(`Message: ${error.message}`);
        console.log(`\nStack trace:`);
        console.log(error.stack);
        console.log('!'.repeat(60) + '\n');
    });

    // Capture request failures
    page.on('requestfailed', request => {
        console.log(`\n[REQUEST FAILED] ${request.url()}`);
        console.log(`  Failure: ${request.failure()?.errorText || 'unknown'}`);
    });

    try {
        // Navigate to page with extended timeout
        console.log('Navigating to page...');
        await page.goto(URL, {
            waitUntil: 'networkidle',
            timeout: 30000
        });

        console.log('Page loaded. Waiting 3 seconds for async initialization...\n');
        await page.waitForTimeout(3000);

        // Check if page became interactive
        const isInteractive = await page.evaluate(() => {
            return {
                documentReady: document.readyState,
                hasConnect: !!document.getElementById('connect'),
                hasPlaybackCard: !!document.getElementById('playbackCard'),
                hasHandCanvas: !!document.getElementById('handCanvas3D'),
                gambitClientDefined: typeof GambitClient !== 'undefined',
                hand3DRendererDefined: typeof Hand3DRenderer !== 'undefined',
                gestureInferenceDefined: typeof createGestureInference !== 'undefined'
            };
        });

        console.log('\n' + '='.repeat(60));
        console.log('PAGE STATE CHECK');
        console.log('='.repeat(60));
        console.log(`Document ready state: ${isInteractive.documentReady}`);
        console.log(`Connect button exists: ${isInteractive.hasConnect}`);
        console.log(`Playback card exists: ${isInteractive.hasPlaybackCard}`);
        console.log(`Hand canvas exists: ${isInteractive.hasHandCanvas}`);
        console.log(`GambitClient defined: ${isInteractive.gambitClientDefined}`);
        console.log(`Hand3DRenderer defined: ${isInteractive.hand3DRendererDefined}`);
        console.log(`createGestureInference defined: ${isInteractive.gestureInferenceDefined}`);

    } catch (e) {
        console.log('\n[NAVIGATION ERROR]');
        console.log(e.message);
    }

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total console messages: ${consoleMessages.length}`);
    console.log(`Total page errors: ${errors.length}`);

    const errorMessages = consoleMessages.filter(m => m.type === 'error');
    const warningMessages = consoleMessages.filter(m => m.type === 'warning');

    console.log(`Console errors: ${errorMessages.length}`);
    console.log(`Console warnings: ${warningMessages.length}`);

    if (errorMessages.length > 0) {
        console.log('\n--- All Console Errors ---');
        errorMessages.forEach((e, i) => {
            console.log(`\n[${i + 1}] ${e.text}`);
            console.log(`    at ${e.url}:${e.line}:${e.column}`);
        });
    }

    if (errors.length > 0) {
        console.log('\n--- All Page Errors (with full stacks) ---');
        errors.forEach((e, i) => {
            console.log(`\n[${i + 1}] ${e.name}: ${e.message}`);
            console.log(e.stack);
        });
    }

    await browser.close();
    console.log('\nDone.');
}

main().catch(console.error);
