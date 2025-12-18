import { test, expect, ConsoleMessage } from 'playwright/test';

/**
 * Page validation tests for SIMCAP
 *
 * Validates that all pages:
 * 1. Load successfully (HTTP 200)
 * 2. Don't have JavaScript errors in console
 * 3. Have expected title/content
 */

// Pages to test with their expected characteristics
const pages = [
  { path: '/', name: 'Landing Page', titleContains: 'SIMCAP' },
  { path: '/apps/gambit/', name: 'GAMBIT Interface', titleContains: 'GAMBIT' },
  { path: '/apps/gambit/collector.html', name: 'GAMBIT Collector', titleContains: 'Collector' },
  { path: '/apps/gambit/synth.html', name: 'GAMBIT Synth', titleContains: 'Synth' },
  { path: '/apps/loader/', name: 'Firmware Loader', titleContains: 'Loader' },
  { path: '/apps/viz/', name: 'Session Explorer', titleContains: 'VIZ' },
];

// Known console messages that are not errors (e.g., from external libraries)
const allowedConsolePatterns = [
  /eruda/i,  // Eruda dev console
  /DevTools/i,
  /Download the React DevTools/i,
  /third-party cookie/i,
  /TensorFlow/i,
  /WebGL/i,
  /tf\.js/i,
];

function isAllowedConsoleMessage(msg: string): boolean {
  return allowedConsolePatterns.some(pattern => pattern.test(msg));
}

test.describe('Page Load Validation', () => {
  for (const page of pages) {
    test(`${page.name} (${page.path}) loads without errors`, async ({ page: browserPage }) => {
      const consoleErrors: string[] = [];
      const consoleWarnings: string[] = [];

      // Collect console messages
      browserPage.on('console', (msg: ConsoleMessage) => {
        const text = msg.text();
        if (msg.type() === 'error' && !isAllowedConsoleMessage(text)) {
          consoleErrors.push(text);
        } else if (msg.type() === 'warning' && !isAllowedConsoleMessage(text)) {
          consoleWarnings.push(text);
        }
      });

      // Collect page errors (uncaught exceptions)
      browserPage.on('pageerror', (error) => {
        consoleErrors.push(`Page Error: ${error.message}`);
      });

      // Navigate to page
      const response = await browserPage.goto(page.path, {
        waitUntil: 'networkidle',
        timeout: 30000,
      });

      // Check HTTP status
      expect(response?.status(), `${page.name} should return HTTP 200`).toBe(200);

      // Check title
      const title = await browserPage.title();
      expect(title.toLowerCase(), `${page.name} title should contain "${page.titleContains}"`).toContain(page.titleContains.toLowerCase());

      // Wait a bit for any async errors
      await browserPage.waitForTimeout(1000);

      // Report console errors
      if (consoleErrors.length > 0) {
        console.log(`\n[${page.name}] Console Errors:`);
        consoleErrors.forEach(err => console.log(`  - ${err}`));
      }

      // Report warnings (informational, not failing)
      if (consoleWarnings.length > 0) {
        console.log(`\n[${page.name}] Console Warnings:`);
        consoleWarnings.forEach(warn => console.log(`  - ${warn}`));
      }

      // Fail if there are errors
      expect(consoleErrors, `${page.name} should have no console errors`).toHaveLength(0);
    });
  }
});

test.describe('Navigation Links', () => {
  test('Landing page links are valid', async ({ page }) => {
    await page.goto('/');

    // Get all internal links
    const links = await page.locator('a[href^="/"]').all();

    for (const link of links) {
      const href = await link.getAttribute('href');
      if (!href) continue;

      // Check each link returns 200
      const response = await page.request.get(href);
      expect(response.status(), `Link ${href} should be valid`).toBe(200);
    }
  });
});

test.describe('Console Logging', () => {
  test('GAMBIT page console output for debugging', async ({ page }) => {
    const consoleMessages: Array<{ type: string; text: string }> = [];

    page.on('console', (msg) => {
      consoleMessages.push({
        type: msg.type(),
        text: msg.text(),
      });
    });

    await page.goto('/apps/gambit/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    console.log('\n=== GAMBIT Console Output ===');
    consoleMessages.forEach(({ type, text }) => {
      const prefix = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
      console.log(`${prefix} [${type}] ${text}`);
    });
    console.log('=== End Console Output ===\n');
  });
});
