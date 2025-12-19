/**
 * Sessions Listing API Handler
 *
 * Lists all session files stored in the GitHub 'data' branch.
 * Returns JSON manifest compatible with VIZ explorer and session playback.
 *
 * Data is read from:
 * 1. Primary: Session manifest in main branch (data/GAMBIT/manifest.json)
 * 2. Fallback: GitHub Contents API listing of data branch
 *
 * No authentication required - sessions list is public.
 */

// GitHub configuration
const GITHUB_OWNER = 'christopherdebeer';
const GITHUB_REPO = 'simcap';
const DATA_BRANCH = 'data';
const MAIN_BRANCH = 'main';
const MANIFEST_PATH = 'data/GAMBIT/manifest.json';
const DATA_PATH = 'GAMBIT';

// Types
interface SessionInfo {
  filename: string;
  pathname: string;
  url: string;
  downloadUrl: string;
  size: number;
  uploadedAt: string;
  timestamp: string;
}

interface SessionsResponse {
  sessions: SessionInfo[];
  count: number;
  generatedAt: string;
  source: 'manifest' | 'api';
}

interface ApiError {
  error: string;
  message?: string;
}

interface ManifestSession {
  filename: string;
  timestamp: string;
  size: number;
  version?: string;
  sampleCount?: number;
  durationSec?: number;
  url?: string;
}

interface SessionManifest {
  generated: string;
  directory: string;
  branch?: string;
  baseUrl?: string;
  sessionCount: number;
  sessions: ManifestSession[];
}

interface GitHubContentItem {
  name: string;
  path: string;
  sha: string;
  size: number;
  url: string;
  html_url: string;
  git_url: string;
  download_url: string;
  type: 'file' | 'dir';
}

/**
 * Construct raw.githubusercontent.com URL
 */
function getRawUrl(branch: string, path: string): string {
  return `https://raw.githubusercontent.com/${GITHUB_OWNER}/${GITHUB_REPO}/${branch}/${path}`;
}

/**
 * Try to fetch session manifest from main branch
 */
async function fetchManifest(): Promise<SessionManifest | null> {
  const url = getRawUrl(MAIN_BRANCH, MANIFEST_PATH);

  try {
    const response = await fetch(url, {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!response.ok) {
      console.log(`[sessions] Manifest not found at ${url}: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error('[sessions] Error fetching manifest:', error);
    return null;
  }
}

/**
 * List sessions from GitHub Contents API
 */
async function listFromGitHubAPI(): Promise<SessionInfo[]> {
  const url = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${DATA_PATH}?ref=${DATA_BRANCH}`;

  const response = await fetch(url, {
    headers: {
      Accept: 'application/vnd.github.v3+json',
      // Note: For higher rate limits, could add Authorization header with GITHUB_TOKEN
    },
  });

  if (!response.ok) {
    throw new Error(`GitHub API error: ${response.status}`);
  }

  const contents: GitHubContentItem[] = await response.json();

  return contents
    .filter((item) => item.type === 'file' && item.name.endsWith('.json'))
    .map((item) => {
      // Extract timestamp from filename (e.g., "2025-12-15T22_35_15.567Z.json")
      const timestamp = item.name.replace('.json', '').replace(/_/g, ':');
      const rawUrl = getRawUrl(DATA_BRANCH, `${DATA_PATH}/${item.name}`);

      return {
        filename: item.name,
        pathname: `${DATA_PATH}/${item.name}`,
        url: rawUrl,
        downloadUrl: rawUrl,
        size: item.size,
        uploadedAt: new Date().toISOString(), // GitHub API doesn't provide upload time
        timestamp,
      };
    })
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}

/**
 * Convert manifest sessions to SessionInfo format
 */
function manifestToSessionInfo(manifest: SessionManifest): SessionInfo[] {
  const baseUrl = manifest.baseUrl || getRawUrl(DATA_BRANCH, DATA_PATH);

  return manifest.sessions.map((session) => {
    const url = session.url || `${baseUrl}/${session.filename}`;

    return {
      filename: session.filename,
      pathname: `${DATA_PATH}/${session.filename}`,
      url,
      downloadUrl: url,
      size: session.size,
      uploadedAt: manifest.generated,
      timestamp: session.timestamp,
    };
  });
}

export default async function handler(request: Request): Promise<Response> {
  // Only allow GET requests
  if (request.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    let sessions: SessionInfo[];
    let source: 'manifest' | 'api';

    // Try manifest first (faster, cached)
    const manifest = await fetchManifest();

    if (manifest && manifest.sessions && manifest.sessions.length > 0) {
      console.log(`[sessions] Using manifest with ${manifest.sessions.length} sessions`);
      sessions = manifestToSessionInfo(manifest);
      source = 'manifest';
    } else {
      // Fall back to GitHub API
      console.log('[sessions] Falling back to GitHub API');
      sessions = await listFromGitHubAPI();
      source = 'api';
    }

    const responseData: SessionsResponse = {
      sessions,
      count: sessions.length,
      generatedAt: new Date().toISOString(),
      source,
    };

    return new Response(JSON.stringify(responseData), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
      },
    });
  } catch (error) {
    console.error('Sessions list error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';

    const errorResponse: ApiError = {
      error: 'Failed to list sessions',
      message,
    };

    return new Response(JSON.stringify(errorResponse), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
