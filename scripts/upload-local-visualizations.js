#!/usr/bin/env node

/**
 * Upload Local Visualizations to Vercel Blob
 *
 * Uploads visualization files from visualizations/ to Vercel Blob storage,
 * skipping any that already exist in the blob store.
 *
 * Usage:
 *   node scripts/upload-local-visualizations.js
 *   npm run upload:visualizations
 *
 * Environment:
 *   BLOB_READ_WRITE_TOKEN - Required for blob operations
 */

import { put, list } from '@vercel/blob';
import { readdir, readFile, stat } from 'fs/promises';
import { join, relative } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PROJECT_ROOT = join(__dirname, '..');
const LOCAL_VIZ_DIR = join(PROJECT_ROOT, 'visualizations');
const BLOB_PREFIX = 'visualizations/';

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
    } else if (entry.isFile() && (entry.name.endsWith('.png') || entry.name.endsWith('.jpg') || entry.name.endsWith('.jpeg'))) {
      const relativePath = relative(baseDir, fullPath);
      files.push({ fullPath, relativePath });
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
    let contentType = 'image/png';
    if (file.relativePath.endsWith('.jpg') || file.relativePath.endsWith('.jpeg')) {
      contentType = 'image/jpeg';
    }
    
    const blob = await put(blobPath, content, {
      access: 'public',
      contentType,
    });
    
    console.log(`   ✅ Uploaded: ${file.relativePath}`);
    return { success: true, relativePath: file.relativePath, url: blob.url };
  } catch (error) {
    console.error(`   ❌ Failed to upload ${file.relativePath}:`, error.message);
    return { success: false, relativePath: file.relativePath, error: error.message };
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
      return;
    }
    
    // Upload files
    console.log('\n📤 Uploading visualizations...\n');
    
    let successCount = 0;
    let failCount = 0;
    
    for (const file of filesToUpload) {
      const result = await uploadFile(file);
      if (result.success) {
        successCount++;
      } else {
        failCount++;
      }
    }
    
    console.log(`\n✨ Upload complete!`);
    console.log(`   ✅ Successful: ${successCount}`);
    if (failCount > 0) {
      console.log(`   ❌ Failed: ${failCount}`);
    }
    
  } catch (error) {
    console.error('\n❌ Upload failed:', error.message);
    process.exit(1);
  }
}

main();
