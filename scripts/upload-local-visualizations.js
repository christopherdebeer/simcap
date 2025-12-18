#!/usr/bin/env node

/**
 * Upload Local Visualizations to Vercel Blob
 *
 * Uploads visualization files from visualizations/ to Vercel Blob storage,
 * skipping any that already exist in the blob store.
 * 
 * Also generates and uploads manifests for each session to enable
 * the manifest-based visualization system.
 *
 * Usage:
 *   node scripts/upload-local-visualizations.js
 *   npm run upload:visualizations
 *
 * Environment:
 *   BLOB_READ_WRITE_TOKEN - Required for blob operations
 */

import { put, list } from '@vercel/blob';
import { readdir, readFile, stat, writeFile, mkdir } from 'fs/promises';
import { join, relative, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PROJECT_ROOT = join(__dirname, '..');
const LOCAL_VIZ_DIR = join(PROJECT_ROOT, 'visualizations');
const BLOB_PREFIX = 'visualizations/';
const MANIFEST_VERSION = '1.0';

/**
 * Recursively get all files in a directory
 */
async function getAllFiles(dir, baseDir = dir) {
  const files = [];
  const entries = await readdir(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      const subFiles = await getAllFiles(fullPath, baseDir);
      files.push(...subFiles);
    } else if (entry.isFile()) {
      const ext = entry.name.toLowerCase().split('.').pop();
      // Include images and session-data.js
      if (['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp'].includes(ext) || entry.name === 'session-data.js') {
        const relativePath = relative(baseDir, fullPath);
        files.push({ fullPath, relativePath });
      }
    }
  }
  
  return files;
}

/**
 * Get existing visualizations from blob storage
 */
async function getExistingBlobs() {
  console.log('📋 Checking existing visualizations in blob storage...');
  
  try {
    const existingFiles = new Set();
    let cursor = undefined;
    
    do {
      const { blobs, cursor: nextCursor } = await list({
        prefix: BLOB_PREFIX,
        limit: 1000,
        cursor,
      });
      
      for (const blob of blobs) {
        const relativePath = blob.pathname.replace(BLOB_PREFIX, '');
        existingFiles.add(relativePath);
      }
      
      cursor = nextCursor;
    } while (cursor);
    
    console.log(`   Found ${existingFiles.size} existing visualizations in blob storage`);
    return existingFiles;
  } catch (error) {
    console.error('❌ Failed to list blob visualizations:', error.message);
    throw error;
  }
}

/**
 * Upload a single file to blob storage
 */
async function uploadFile(file) {
  const blobPath = BLOB_PREFIX + file.relativePath;
  
  try {
    const content = await readFile(file.fullPath);
    
    // Determine content type
    const ext = file.relativePath.toLowerCase().split('.').pop();
    const contentTypes = {
      'png': 'image/png',
      'jpg': 'image/jpeg',
      'jpeg': 'image/jpeg',
      'gif': 'image/gif',
      'svg': 'image/svg+xml',
      'webp': 'image/webp',
      'js': 'application/javascript',
      'json': 'application/json',
    };
    const contentType = contentTypes[ext] || 'application/octet-stream';
    
    const blob = await put(blobPath, content, {
      access: 'public',
      contentType,
    });
    
    console.log(`   ✅ Uploaded: ${file.relativePath}`);
    return { success: true, relativePath: file.relativePath, url: blob.url, pathname: blobPath };
  } catch (error) {
    console.error(`   ❌ Failed to upload ${file.relativePath}:`, error.message);
    return { success: false, relativePath: file.relativePath, error: error.message };
  }
}

/**
 * Extract session timestamp from a visualization filepath
 */
function extractSessionTimestamp(filepath) {
  // Match patterns like:
  // composite_2025-12-15T22_35_15.567Z.png
  // windows_2025-12-15T22_35_15.567Z/window_001/...
  // trajectory_comparison_2025-12-15T22_35_15.567Z/...
  const patterns = [
    /(?:composite|calibration_stages|orientation_3d|orientation_track|raw_axes)_(.+?)\.png$/,
    /windows_(.+?)\/window_\d+/,
    /trajectory_comparison_(.+?)\//,
  ];
  
  for (const pattern of patterns) {
    const match = filepath.match(pattern);
    if (match) {
      return match[1];
    }
  }
  return null;
}

/**
 * Normalize timestamp (underscore to colon format)
 */
function normalizeTimestamp(timestamp) {
  // T22_35_15 -> T22:35:15
  return timestamp.replace(/T(\d{2})_(\d{2})_(\d{2})/, 'T$1:$2:$3');
}

/**
 * Denormalize timestamp (colon to underscore format)
 */
function denormalizeTimestamp(timestamp) {
  // T22:35:15 -> T22_35_15
  return timestamp.replace(/T(\d{2}):(\d{2}):(\d{2})/, 'T$1_$2_$3');
}

/**
 * Group uploaded files by session timestamp
 */
function groupFilesBySession(files) {
  const sessions = {};
  
  for (const file of files) {
    if (!file.success) continue;
    
    const ts = extractSessionTimestamp(file.relativePath);
    if (ts) {
      if (!sessions[ts]) {
        sessions[ts] = [];
      }
      sessions[ts].push(file);
    }
  }
  
  return sessions;
}

/**
 * Build a manifest from uploaded files for a session
 */
function buildManifest(sessionTimestamp, files) {
  const generatedAt = new Date().toISOString();
  const normalizedTs = normalizeTimestamp(sessionTimestamp);
  const generatedTsSafe = denormalizeTimestamp(generatedAt);
  const manifestId = `${sessionTimestamp}_${generatedTsSafe}`;
  
  // Build URL lookup
  const urlMap = {};
  for (const file of files) {
    if (file.relativePath && file.url) {
      urlMap[file.relativePath] = file.url;
    }
  }
  
  // Session-level images
  const images = {};
  for (const imgType of ['composite', 'calibration_stages', 'orientation_3d', 'orientation_track', 'raw_axes']) {
    const key = `${imgType}_${sessionTimestamp}.png`;
    if (urlMap[key]) {
      images[imgType] = urlMap[key];
    }
  }
  
  // Trajectory comparison images
  const trajectoryComparison = {};
  const trajPrefix = `trajectory_comparison_${sessionTimestamp}/`;
  for (const [relPath, url] of Object.entries(urlMap)) {
    if (relPath.startsWith(trajPrefix)) {
      // Extract type from filename (e.g., raw_3d.png -> raw)
      const filename = relPath.replace(trajPrefix, '').replace('.png', '');
      const trajType = filename.replace('_3d', '').replace('_overlay', '');
      trajectoryComparison[trajType] = url;
    }
  }
  
  // Window images
  const windowsMap = {};
  const windowPrefix = `windows_${sessionTimestamp}/`;
  
  for (const [relPath, url] of Object.entries(urlMap)) {
    if (!relPath.startsWith(windowPrefix)) continue;
    
    // Parse window path: windows_TS/window_001/image.png or windows_TS/window_001.png
    const windowMatch = relPath.match(
      new RegExp(`${windowPrefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}window_(\\d+)(?:/(.+)|\\.png)$`)
    );
    if (!windowMatch) continue;
    
    const windowNum = parseInt(windowMatch[1], 10);
    const subPath = windowMatch[2];
    
    if (!windowsMap[windowNum]) {
      windowsMap[windowNum] = {
        window_num: windowNum,
        images: {},
        trajectory_images: {},
      };
    }
    
    const window = windowsMap[windowNum];
    
    if (!subPath) {
      // This is the composite window image (window_001.png)
      window.composite = url;
    } else {
      // This is a sub-image
      const imageName = subPath.replace('.png', '');
      
      // Categorize trajectory vs regular images
      if (imageName.startsWith('trajectory_') && !['accel', 'gyro', 'mag', 'combined'].some(x => imageName.includes(x))) {
        // trajectory_raw, trajectory_iron, trajectory_fused, trajectory_filtered
        const trajKey = imageName.replace('trajectory_', '');
        window.trajectory_images[trajKey] = url;
      } else {
        window.images[imageName] = url;
      }
    }
  }
  
  // Convert windows map to sorted list
  const windows = Object.keys(windowsMap)
    .sort((a, b) => a - b)
    .map(k => windowsMap[k]);
  
  // Build session metadata (basic info)
  const session = {
    filename: `${sessionTimestamp}.json`,
    duration: 0,
    sample_count: 0,
    sample_rate: 50,
  };
  
  return {
    version: MANIFEST_VERSION,
    sessionTimestamp: normalizedTs,
    generatedAt,
    manifestId,
    session,
    images,
    trajectory_comparison: trajectoryComparison,
    windows,
  };
}

/**
 * Upload a manifest to blob storage
 */
async function uploadManifest(manifest) {
  const manifestId = manifest.manifestId;
  const blobPath = `visualizations/manifests/${manifestId}.json`;
  
  try {
    const content = JSON.stringify(manifest, null, 2);
    
    const blob = await put(blobPath, content, {
      access: 'public',
      contentType: 'application/json',
    });
    
    console.log(`   ✅ Manifest: ${blobPath}`);
    return { success: true, pathname: blobPath, url: blob.url };
  } catch (error) {
    console.error(`   ❌ Failed to upload manifest ${blobPath}:`, error.message);
    return { success: false, pathname: blobPath, error: error.message };
  }
}

/**
 * Main upload function
 */
async function main() {
  console.log('🚀 Upload Local Visualizations to Vercel Blob\n');
  
  // Check for token
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error('❌ BLOB_READ_WRITE_TOKEN environment variable is required');
    console.error('   Set it in .env.local or export it before running');
    process.exit(1);
  }
  
  try {
    // Get local files
    console.log('📁 Scanning local visualizations directory...');
    const localFiles = await getAllFiles(LOCAL_VIZ_DIR);
    console.log(`   Found ${localFiles.length} local visualization files\n`);
    
    if (localFiles.length === 0) {
      console.log('   No visualization files found in visualizations/ directory');
      return;
    }
    
    // Get existing blobs
    const existingBlobs = await getExistingBlobs();
    
    // Filter to only new files
    const filesToUpload = localFiles.filter(f => !existingBlobs.has(f.relativePath));
    
    console.log(`\n📤 Files to upload: ${filesToUpload.length}`);
    
    if (filesToUpload.length === 0) {
      console.log('   All local visualizations are already in blob storage!');
    } else {
      // Upload files
      console.log('\n📤 Uploading visualizations...\n');
      
      const results = [];
      for (const file of filesToUpload) {
        const result = await uploadFile(file);
        results.push(result);
      }
      
      const successCount = results.filter(r => r.success).length;
      const failCount = results.filter(r => !r.success).length;
      
      console.log(`\n✨ Upload complete!`);
      console.log(`   ✅ Successful: ${successCount}`);
      if (failCount > 0) {
        console.log(`   ❌ Failed: ${failCount}`);
      }
    }
    
    // Generate and upload manifests for all sessions
    console.log('\n📋 Generating manifests...\n');
    
    // Get all files (including existing ones) to build complete manifests
    const allLocalFiles = await getAllFiles(LOCAL_VIZ_DIR);
    
    // Build file info with URLs for existing files
    const allFilesWithUrls = [];
    for (const file of allLocalFiles) {
      // Check if we just uploaded it
      const uploadedFile = filesToUpload.find(f => f.relativePath === file.relativePath);
      if (uploadedFile) {
        // Use the result from upload
        const result = await uploadFile(file);
        if (result.success) {
          allFilesWithUrls.push(result);
        }
      } else {
        // File already exists in blob, construct URL
        const blobPath = BLOB_PREFIX + file.relativePath;
        // We need to get the actual URL from blob storage
        // For now, construct a placeholder that will be replaced
        allFilesWithUrls.push({
          success: true,
          relativePath: file.relativePath,
          url: `https://blob.vercel-storage.com/${blobPath}`,
          pathname: blobPath,
        });
      }
    }
    
    // Group by session
    const sessions = groupFilesBySession(allFilesWithUrls);
    console.log(`   Found ${Object.keys(sessions).length} session(s) with visualizations`);
    
    // Generate and upload manifests
    let manifestSuccess = 0;
    let manifestFail = 0;
    
    for (const [sessionTs, sessionFiles] of Object.entries(sessions)) {
      const manifest = buildManifest(sessionTs, sessionFiles);
      const result = await uploadManifest(manifest);
      
      if (result.success) {
        manifestSuccess++;
      } else {
        manifestFail++;
      }
    }
    
    console.log(`\n📋 Manifests: ${manifestSuccess} generated, ${manifestFail} failed`);
    
  } catch (error) {
    console.error('\n❌ Upload failed:', error.message);
    process.exit(1);
  }
}

main();
