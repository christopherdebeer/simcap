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
 *     action: 'write' | 'append' | 'finalize' | 'status' | 'list',
 *     filename: string,       // e.g., '2025-01-07T12_00_00.000Z.json'
 *     content: string,        // JSON content to write
 *   }
 */

// CORS headers for all responses
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

// Only allow local storage on localhost
const ALLOWED_HOSTS = ['localhost', '127.0.0.1', '::1'];

// Data directory path (relative to project root)
const DATA_DIR = 'data/GAMBIT';

interface LocalStorageRequest {
  action: 'write' | 'append' | 'finalize' | 'status' | 'list';
  filename?: string;
  content?: string;
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
function isLocalMode(request: Request): boolean {
  const url = new URL(request.url);
  const hostname = url.hostname;
  return ALLOWED_HOSTS.includes(hostname);
}

/**
 * Get the data directory path and fs module (dynamic import for SSR)
 */
async function getFileSystem(): Promise<{
  fs: typeof import('fs');
  path: typeof import('path');
  dataDir: string;
} | null> {
  try {
    // Dynamic import for Node.js modules (works in Vite SSR and Vercel Node.js runtime)
    const fs = await import('fs');
    const path = await import('path');

    // Try worktree path first (project root)
    const worktreePath = path.resolve(process.cwd(), DATA_DIR);
    if (fs.existsSync(worktreePath)) {
      return { fs, path, dataDir: worktreePath };
    }

    // Fall back to .worktrees path
    const directPath = path.resolve(process.cwd(), '.worktrees/data/GAMBIT');
    if (fs.existsSync(directPath)) {
      return { fs, path, dataDir: directPath };
    }

    // Create the directory if it doesn't exist
    fs.mkdirSync(worktreePath, { recursive: true });
    return { fs, path, dataDir: worktreePath };
  } catch {
    // fs module not available (Edge runtime)
    return null;
  }
}

/**
 * List files in the data directory
 */
async function listFiles(): Promise<string[]> {
  const fsInfo = await getFileSystem();
  if (!fsInfo) return [];

  try {
    const files = fsInfo.fs.readdirSync(fsInfo.dataDir);
    return files
      .filter((f: string) => f.endsWith('.json'))
      .sort()
      .reverse(); // Most recent first
  } catch {
    return [];
  }
}

/**
 * Write content to a file
 */
async function writeFile(
  filename: string,
  content: string
): Promise<{ success: boolean; path?: string; bytesWritten?: number; error?: string }> {
  const fsInfo = await getFileSystem();
  if (!fsInfo) {
    return { success: false, error: 'Filesystem not available (Edge runtime)' };
  }

  try {
    const { fs, path, dataDir } = fsInfo;
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
async function appendFile(
  filename: string,
  content: string
): Promise<{ success: boolean; path?: string; bytesWritten?: number; error?: string }> {
  const fsInfo = await getFileSystem();
  if (!fsInfo) {
    return { success: false, error: 'Filesystem not available (Edge runtime)' };
  }

  try {
    const { fs, path, dataDir } = fsInfo;
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

function jsonResponse(data: LocalStorageResponse, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...CORS_HEADERS,
    },
  });
}

export default async function handler(request: Request): Promise<Response> {
  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: CORS_HEADERS,
    });
  }

  // Check if we're in local mode
  if (!isLocalMode(request)) {
    return jsonResponse({
      success: false,
      mode: 'production',
      error: 'Local storage is only available in development mode'
    }, 403);
  }

  // Check if filesystem is available
  const fsInfo = await getFileSystem();
  if (!fsInfo) {
    return jsonResponse({
      success: false,
      error: 'Filesystem not available in this runtime'
    }, 500);
  }

  // GET: Return status and list files
  if (request.method === 'GET') {
    const files = await listFiles();
    return jsonResponse({
      success: true,
      mode: 'local',
      files
    });
  }

  // POST: Write data
  if (request.method === 'POST') {
    let body: LocalStorageRequest;
    try {
      body = await request.json();
    } catch {
      return jsonResponse({
        success: false,
        error: 'Invalid JSON body'
      }, 400);
    }

    if (!body.action) {
      return jsonResponse({
        success: false,
        error: 'Missing action'
      }, 400);
    }

    // Status check action
    if (body.action === 'status') {
      return jsonResponse({
        success: true,
        mode: 'local'
      });
    }

    // List files action
    if (body.action === 'list') {
      const files = await listFiles();
      return jsonResponse({
        success: true,
        mode: 'local',
        files
      });
    }

    // Write/append actions require filename and content
    if (!body.filename) {
      return jsonResponse({
        success: false,
        error: 'Missing filename'
      }, 400);
    }

    if (!body.content && body.action !== 'finalize') {
      return jsonResponse({
        success: false,
        error: 'Missing content'
      }, 400);
    }

    let result: { success: boolean; path?: string; bytesWritten?: number; error?: string };

    switch (body.action) {
      case 'write':
        result = await writeFile(body.filename, body.content!);
        break;

      case 'append':
        result = await appendFile(body.filename, body.content!);
        break;

      case 'finalize':
        // Finalize is just a marker that the session is complete
        result = { success: true };
        break;

      default:
        return jsonResponse({
          success: false,
          error: `Unknown action: ${body.action}`
        }, 400);
    }

    if (result.success) {
      return jsonResponse({
        success: true,
        mode: 'local',
        path: result.path,
        bytesWritten: result.bytesWritten
      });
    } else {
      return jsonResponse({
        success: false,
        error: result.error
      }, 500);
    }
  }

  // Method not allowed
  return jsonResponse({
    success: false,
    error: 'Method not allowed'
  }, 405);
}
