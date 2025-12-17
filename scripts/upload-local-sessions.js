#!/usr/bin/env node
/**
 * Upload Local Sessions to Vercel Blob
 * 
 * Uploads session files from data/GAMBIT/ to Vercel Blob storage,
 * skipping any that already exist in the blob store.
 * 
 * Usage:
 *   BLOB_READ_WRITE_TOKEN=xxx node scripts/upload-local-sessions.js
 * 
 * Or with npm script:
 *   npm run upload-sessions
 */

import { put, list } from '@vercel/blob';
import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

const LOCAL_SESSIONS_DIR = 'data/GAMBIT';
const BLOB_PREFIX = 'sessions/';

async function getExistingBlobSessions() {
  console.log('📋 Fetching existing sessions from Vercel Blob...');
  
  try {
    const { blobs } = await list({
      prefix: BLOB_PREFIX,
      limit: 1000,
    });
    
    // Extract filenames from blob paths
    const existingFiles = new Set(
      blobs
        .filter(blob => blob.pathname.endsWith('.json'))
        .map(blob => blob.pathname.replace(BLOB_PREFIX, ''))
    );
    
    console.log(`   Found ${existingFiles.size} existing sessions in blob storage`);
    return existingFiles;
  } catch (error) {
    console.error('❌ Failed to list blob sessions:', error.message);
    throw error;
  }
}

async function getLocalSessions() {
  console.log(`📂 Scanning local sessions in ${LOCAL_SESSIONS_DIR}...`);
  
  try {
    const files = await readdir(LOCAL_SESSIONS_DIR);
    const jsonFiles = files.filter(f => f.endsWith('.json') && f !== 'manifest.json');
    
    console.log(`   Found ${jsonFiles.length} local session files`);
    return jsonFiles;
  } catch (error) {
    console.error('❌ Failed to read local sessions:', error.message);
    throw error;
  }
}

async function uploadSession(filename) {
  const localPath = join(LOCAL_SESSIONS_DIR, filename);
  const blobPath = BLOB_PREFIX + filename;
  
  try {
    const content = await readFile(localPath, 'utf-8');
    
    // Validate JSON
    JSON.parse(content);
    
    const blob = await put(blobPath, content, {
      access: 'public',
      contentType: 'application/json',
    });
    
    console.log(`   ✅ Uploaded: ${filename}`);
    console.log(`      URL: ${blob.url}`);
    return { success: true, filename, url: blob.url };
  } catch (error) {
    console.error(`   ❌ Failed to upload ${filename}:`, error.message);
    return { success: false, filename, error: error.message };
  }
}

async function main() {
  console.log('🚀 Upload Local Sessions to Vercel Blob\n');
  
  // Check for token
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error('❌ BLOB_READ_WRITE_TOKEN environment variable is required');
    console.error('   Usage: BLOB_READ_WRITE_TOKEN=xxx node scripts/upload-local-sessions.js');
    process.exit(1);
  }
  
  try {
    // Get existing and local sessions
    const existingFiles = await getExistingBlobSessions();
    const localFiles = await getLocalSessions();
    
    // Find files that need uploading
    const filesToUpload = localFiles.filter(f => !existingFiles.has(f));
    
    console.log(`\n📤 Sessions to upload: ${filesToUpload.length}`);
    
    if (filesToUpload.length === 0) {
      console.log('   All local sessions are already in blob storage!');
      return;
    }
    
    // Upload each file
    console.log('\n📡 Uploading sessions...\n');
    const results = [];
    
    for (const filename of filesToUpload) {
      const result = await uploadSession(filename);
      results.push(result);
    }
    
    // Summary
    const successful = results.filter(r => r.success).length;
    const failed = results.filter(r => !r.success).length;
    
    console.log('\n📊 Summary:');
    console.log(`   ✅ Uploaded: ${successful}`);
    console.log(`   ❌ Failed: ${failed}`);
    console.log(`   ⏭️  Skipped (already exists): ${localFiles.length - filesToUpload.length}`);
    
    if (failed > 0) {
      process.exit(1);
    }
  } catch (error) {
    console.error('\n❌ Upload failed:', error.message);
    process.exit(1);
  }
}

main();
