#!/usr/bin/env node
/**
 * Fetch Sessions from Vercel Blob
 *
 * Downloads session JSON files from Vercel Blob storage to local data directory.
 *
 * Usage:
 *   npm run fetch:sessions                    # Fetch all sessions
 *   npm run fetch:sessions -- --list          # List sessions without downloading
 *   npm run fetch:sessions -- --session <ts>  # Fetch specific session by timestamp
 *
 * Environment:
 *   BLOB_READ_WRITE_TOKEN - Required for listing blobs (or uses public API)
 *   SIMCAP_API_URL - API base URL (default: https://simcap.vercel.app)
 */

import { mkdir, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');
const DATA_DIR = join(PROJECT_ROOT, 'data', 'GAMBIT');

// API URL - can be overridden via environment
const API_BASE = process.env.SIMCAP_API_URL || 'https://simcap.vercel.app';

/**
 * Fetch session list from API
 */
async function fetchSessionList() {
  const response = await fetch(`${API_BASE}/api/sessions`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data.sessions || [];
}

/**
 * Download a session file
 */
async function downloadSession(session, outputDir) {
  const response = await fetch(session.url);
  if (!response.ok) {
    throw new Error(`Failed to download ${session.filename}: ${response.status}`);
  }

  const content = await response.text();
  const outputPath = join(outputDir, session.filename);

  await writeFile(outputPath, content, 'utf-8');
  return outputPath;
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
 * Main function
 */
async function main() {
  const args = process.argv.slice(2);
  const listOnly = args.includes('--list');
  const sessionIndex = args.indexOf('--session');
  const sessionFilter = sessionIndex >= 0 ? args[sessionIndex + 1] : null;

  console.log('Fetching session list from Vercel Blob...\n');

  try {
    const sessions = await fetchSessionList();

    if (sessions.length === 0) {
      console.log('No sessions found in blob storage.');
      return;
    }

    // Apply filter if specified
    let filteredSessions = sessions;
    if (sessionFilter) {
      filteredSessions = sessions.filter(s =>
        s.timestamp.includes(sessionFilter) || s.filename.includes(sessionFilter)
      );
      if (filteredSessions.length === 0) {
        console.log(`No sessions matching "${sessionFilter}" found.`);
        return;
      }
    }

    console.log(`Found ${filteredSessions.length} session(s):\n`);

    // List mode
    if (listOnly) {
      for (const session of filteredSessions) {
        console.log(`  ${session.filename}`);
        console.log(`    Size: ${formatSize(session.size)}`);
        console.log(`    Uploaded: ${new Date(session.uploadedAt).toLocaleString()}`);
        console.log(`    URL: ${session.url}\n`);
      }
      console.log(`Total: ${filteredSessions.length} sessions`);
      return;
    }

    // Download mode
    // Ensure output directory exists
    if (!existsSync(DATA_DIR)) {
      await mkdir(DATA_DIR, { recursive: true });
      console.log(`Created directory: ${DATA_DIR}\n`);
    }

    let downloaded = 0;
    let skipped = 0;
    let failed = 0;

    for (const session of filteredSessions) {
      const outputPath = join(DATA_DIR, session.filename);

      // Check if file already exists
      if (existsSync(outputPath)) {
        console.log(`  SKIP: ${session.filename} (already exists)`);
        skipped++;
        continue;
      }

      try {
        await downloadSession(session, DATA_DIR);
        console.log(`  OK: ${session.filename} (${formatSize(session.size)})`);
        downloaded++;
      } catch (error) {
        console.error(`  FAIL: ${session.filename} - ${error.message}`);
        failed++;
      }
    }

    console.log(`\nSummary: ${downloaded} downloaded, ${skipped} skipped, ${failed} failed`);
    console.log(`Output directory: ${DATA_DIR}`);

  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

main();
