/**
 * Local Storage API
 *
 * Writes telemetry data directly to the local filesystem during development.
 * This endpoint is only available when running locally (not in production).
 *
 * Uses the worktree setup: data/GAMBIT/ directory (symlinked to .worktrees/data)
 *
 * Endpoints:
 *   POST /api/local-storage - Write data to local filesystem
 *   GET /api/local-storage - Check local mode status and list files
 *
 * Request Body (POST):
 *   {
 *     action: 'write' | 'append' | 'finalize',
 *     filename: string,       // e.g., '2025-01-07T12_00_00.000Z.json'
 *     content: string,        // JSON content to write
 *     chunkIndex?: number,    // For chunked writes
 *     isManifest?: boolean,   // Whether this is a manifest file
 *   }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import * as fs from 'fs';
import * as path from 'path';

// Only allow local storage on localhost
const ALLOWED_HOSTS = ['localhost', '127.0.0.1', '::1'];

// Data directory path (relative to project root)
const DATA_DIR = 'data/GAMBIT';

interface LocalStorageRequest {
  action: 'write' | 'append' | 'finalize' | 'status' | 'list';
  filename?: string;
  content?: string;
  chunkIndex?: number;
  isManifest?: boolean;
}

interface LocalStorageResponse {
  success: boolean;
  mode?: 'local' | 'production';
  error?: string;
  path?: string;
  files?: string[];
  bytesWritten?: number;
}

/**
 * Check if we're running in local development mode
 */
function isLocalMode(req: VercelRequest): boolean {
  const host = req.headers.host || '';
  const hostname = host.split(':')[0];
  return ALLOWED_HOSTS.includes(hostname);
}

/**
 * Get the data directory path
 */
function getDataDir(): string {
  // Try worktree path first (project root)
  const worktreePath = path.resolve(process.cwd(), DATA_DIR);
  if (fs.existsSync(worktreePath)) {
    return worktreePath;
  }

  // Fall back to .worktrees path
  const directPath = path.resolve(process.cwd(), '.worktrees/data/GAMBIT');
  if (fs.existsSync(directPath)) {
    return directPath;
  }

  // Create the directory if it doesn't exist
  fs.mkdirSync(worktreePath, { recursive: true });
  return worktreePath;
}

/**
 * List files in the data directory
 */
function listFiles(): string[] {
  try {
    const dataDir = getDataDir();
    const files = fs.readdirSync(dataDir);
    return files
      .filter(f => f.endsWith('.json'))
      .sort()
      .reverse(); // Most recent first
  } catch {
    return [];
  }
}

/**
 * Write content to a file
 */
function writeFile(filename: string, content: string): { success: boolean; path?: string; bytesWritten?: number; error?: string } {
  try {
    const dataDir = getDataDir();
    const filePath = path.join(dataDir, filename);

    // Security: Ensure filename doesn't escape the data directory
    const resolvedPath = path.resolve(dataDir, filename);
    if (!resolvedPath.startsWith(dataDir)) {
      return { success: false, error: 'Invalid filename: path traversal detected' };
    }

    fs.writeFileSync(filePath, content, 'utf-8');

    return {
      success: true,
      path: filePath,
      bytesWritten: Buffer.byteLength(content, 'utf-8')
    };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : 'Write failed'
    };
  }
}

/**
 * Append content to a file
 */
function appendFile(filename: string, content: string): { success: boolean; path?: string; bytesWritten?: number; error?: string } {
  try {
    const dataDir = getDataDir();
    const filePath = path.join(dataDir, filename);

    // Security check
    const resolvedPath = path.resolve(dataDir, filename);
    if (!resolvedPath.startsWith(dataDir)) {
      return { success: false, error: 'Invalid filename: path traversal detected' };
    }

    fs.appendFileSync(filePath, content, 'utf-8');

    return {
      success: true,
      path: filePath,
      bytesWritten: Buffer.byteLength(content, 'utf-8')
    };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : 'Append failed'
    };
  }
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

  // Check if we're in local mode
  if (!isLocalMode(req)) {
    res.status(403).json({
      success: false,
      mode: 'production',
      error: 'Local storage is only available in development mode'
    } as LocalStorageResponse);
    return;
  }

  // GET: Return status and list files
  if (req.method === 'GET') {
    const files = listFiles();
    res.status(200).json({
      success: true,
      mode: 'local',
      files
    } as LocalStorageResponse);
    return;
  }

  // POST: Write data
  if (req.method === 'POST') {
    const body = req.body as LocalStorageRequest;

    if (!body.action) {
      res.status(400).json({
        success: false,
        error: 'Missing action'
      } as LocalStorageResponse);
      return;
    }

    // Status check action
    if (body.action === 'status') {
      res.status(200).json({
        success: true,
        mode: 'local'
      } as LocalStorageResponse);
      return;
    }

    // List files action
    if (body.action === 'list') {
      const files = listFiles();
      res.status(200).json({
        success: true,
        mode: 'local',
        files
      } as LocalStorageResponse);
      return;
    }

    // Write/append actions require filename and content
    if (!body.filename) {
      res.status(400).json({
        success: false,
        error: 'Missing filename'
      } as LocalStorageResponse);
      return;
    }

    if (!body.content && body.action !== 'finalize') {
      res.status(400).json({
        success: false,
        error: 'Missing content'
      } as LocalStorageResponse);
      return;
    }

    let result: { success: boolean; path?: string; bytesWritten?: number; error?: string };

    switch (body.action) {
      case 'write':
        result = writeFile(body.filename, body.content!);
        break;

      case 'append':
        result = appendFile(body.filename, body.content!);
        break;

      case 'finalize':
        // Finalize is just a marker that the session is complete
        // For now, it doesn't do anything special but could trigger
        // post-processing in the future
        result = { success: true };
        break;

      default:
        res.status(400).json({
          success: false,
          error: `Unknown action: ${body.action}`
        } as LocalStorageResponse);
        return;
    }

    if (result.success) {
      res.status(200).json({
        success: true,
        mode: 'local',
        path: result.path,
        bytesWritten: result.bytesWritten
      } as LocalStorageResponse);
    } else {
      res.status(500).json({
        success: false,
        error: result.error
      } as LocalStorageResponse);
    }
    return;
  }

  // Method not allowed
  res.status(405).json({
    success: false,
    error: 'Method not allowed'
  } as LocalStorageResponse);
}
