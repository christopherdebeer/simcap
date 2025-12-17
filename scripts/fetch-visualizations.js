#!/usr/bin/env node
/**
 * Fetch Visualizations from Vercel Blob
 *
 * Downloads visualization assets from Vercel Blob storage to local directory.
 *
 * Usage:
 *   npm run fetch:visualizations                    # Fetch all visualizations
 *   npm run fetch:visualizations -- --list          # List without downloading
 *   npm run fetch:visualizations -- --session <ts>  # Fetch specific session
 *
 * Environment:
 *   SIMCAP_API_URL - API base URL (default: https://simcap.vercel.app)
 */

import { mkdir, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');
const VIZ_DIR = join(PROJECT_ROOT, 'visualizations');

// API URL - can be overridden via environment
const API_BASE = process.env.SIMCAP_API_URL || 'https://simcap.vercel.app';

/**
 * Fetch visualization list from API
 */
async function fetchVisualizationList() {
  const response = await fetch(`${API_BASE}/api/visualizations`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data.sessions || [];
}

/**
 * Download a file
 */
async function downloadFile(url, outputPath) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();

  // Ensure directory exists
  const dir = dirname(outputPath);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }

  await writeFile(outputPath, Buffer.from(buffer));
  return buffer.byteLength;
}

/**
 * Format file size
 */
function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Extract relative path from blob URL
 */
function getRelativePath(url, prefix = 'visualizations/') {
  // URL format: https://...blob.vercel-storage.com/visualizations/path/file.png
  const match = url.match(/visualizations\/(.+)$/);
  if (match) return match[1];

  // Fallback: extract filename from URL
  const urlParts = new URL(url);
  return urlParts.pathname.split('/').pop();
}

/**
 * Collect all URLs from session visualization data
 */
function collectUrls(session) {
  const urls = [];

  // Composite and calibration images
  if (session.composite_image) {
    urls.push({ url: session.composite_image, type: 'composite' });
  }
  if (session.calibration_stages_image) {
    urls.push({ url: session.calibration_stages_image, type: 'calibration' });
  }
  if (session.orientation_3d_image) {
    urls.push({ url: session.orientation_3d_image, type: 'orientation_3d' });
  }
  if (session.orientation_track_image) {
    urls.push({ url: session.orientation_track_image, type: 'orientation_track' });
  }
  if (session.raw_axes_image) {
    urls.push({ url: session.raw_axes_image, type: 'raw_axes' });
  }

  // Trajectory comparison images
  if (session.trajectory_comparison_images) {
    for (const [key, url] of Object.entries(session.trajectory_comparison_images)) {
      urls.push({ url, type: `trajectory_${key}` });
    }
  }

  // Window images
  if (session.windows) {
    for (const window of session.windows) {
      if (window.filepath) {
        urls.push({ url: window.filepath, type: 'window_composite' });
      }
      if (window.images) {
        for (const [key, url] of Object.entries(window.images)) {
          urls.push({ url, type: `window_${key}` });
        }
      }
      if (window.trajectory_images) {
        for (const [key, url] of Object.entries(window.trajectory_images)) {
          urls.push({ url, type: `window_traj_${key}` });
        }
      }
    }
  }

  return urls;
}

/**
 * Main function
 */
async function main() {
  const args = process.argv.slice(2);
  const listOnly = args.includes('--list');
  const sessionIndex = args.indexOf('--session');
  const sessionFilter = sessionIndex >= 0 ? args[sessionIndex + 1] : null;

  console.log('Fetching visualization list from Vercel Blob...\n');

  try {
    const sessions = await fetchVisualizationList();

    if (sessions.length === 0) {
      console.log('No visualizations found in blob storage.');
      return;
    }

    // Apply filter if specified
    let filteredSessions = sessions;
    if (sessionFilter) {
      filteredSessions = sessions.filter(s =>
        s.timestamp.includes(sessionFilter) || s.filename?.includes(sessionFilter)
      );
      if (filteredSessions.length === 0) {
        console.log(`No sessions matching "${sessionFilter}" found.`);
        return;
      }
    }

    console.log(`Found ${filteredSessions.length} session(s) with visualizations:\n`);

    // Collect all file URLs
    const allUrls = [];
    for (const session of filteredSessions) {
      const urls = collectUrls(session);
      for (const item of urls) {
        item.session = session.timestamp;
      }
      allUrls.push(...urls);
    }

    // List mode
    if (listOnly) {
      for (const session of filteredSessions) {
        const urls = collectUrls(session);
        console.log(`  ${session.timestamp}`);
        console.log(`    Files: ${urls.length}`);
        console.log(`    Windows: ${session.windows?.length || 0}\n`);
      }
      console.log(`Total: ${filteredSessions.length} sessions, ${allUrls.length} files`);
      return;
    }

    // Download mode
    if (!existsSync(VIZ_DIR)) {
      await mkdir(VIZ_DIR, { recursive: true });
      console.log(`Created directory: ${VIZ_DIR}\n`);
    }

    let downloaded = 0;
    let skipped = 0;
    let failed = 0;
    let totalSize = 0;

    for (let i = 0; i < allUrls.length; i++) {
      const { url, type, session } = allUrls[i];

      // Skip non-URL values (might be relative paths from older data)
      if (!url || !url.startsWith('http')) {
        skipped++;
        continue;
      }

      const relativePath = getRelativePath(url);
      const outputPath = join(VIZ_DIR, relativePath);

      // Check if file already exists
      if (existsSync(outputPath)) {
        skipped++;
        continue;
      }

      try {
        const size = await downloadFile(url, outputPath);
        totalSize += size;
        downloaded++;

        // Progress indicator
        if (downloaded % 10 === 0 || downloaded === 1) {
          console.log(`  [${downloaded}/${allUrls.length - skipped}] Downloaded ${relativePath}`);
        }
      } catch (error) {
        console.error(`  FAIL: ${relativePath} - ${error.message}`);
        failed++;
      }
    }

    console.log(`\nSummary:`);
    console.log(`  Downloaded: ${downloaded} files (${formatSize(totalSize)})`);
    console.log(`  Skipped: ${skipped} (already exist or invalid)`);
    console.log(`  Failed: ${failed}`);
    console.log(`  Output directory: ${VIZ_DIR}`);

  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

main();
