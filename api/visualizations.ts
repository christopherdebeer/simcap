/**
 * Visualizations Listing API Handler
 *
 * Lists visualization manifests stored in GitHub main branch.
 * Images are stored in the 'images' branch, manifests in 'main'.
 *
 * Manifest structure: visualizations/manifests/{session_ts}_{generated_ts}.json
 *
 * Endpoints:
 * - GET /api/visualizations - List all sessions with latest manifests
 * - GET /api/visualizations?session={timestamp} - Get latest manifest for session
 * - GET /api/visualizations?session={timestamp}&history=true - Get all manifest versions
 *
 * No authentication required - visualizations are public.
 */

// Use Edge Runtime for Web API Response support
export const config = {
  runtime: 'edge',
};

// GitHub configuration
const GITHUB_OWNER = 'christopherdebeer';
const GITHUB_REPO = 'simcap';
const MAIN_BRANCH = 'main';
const IMAGES_BRANCH = 'images';
const MANIFESTS_PATH = 'visualizations/manifests';
const INDEX_PATH = 'visualizations/manifests/index.json';

// CORS headers for all responses
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

// Types
interface WindowEntry {
  window_num: number;
  filepath?: string;
  time_start?: number;
  time_end?: number;
  sample_count?: number;
  composite?: string;
  images: Record<string, string>;
  trajectory_images: Record<string, string>;
}

interface VisualizationManifest {
  version: '1.0';
  sessionTimestamp: string;
  generatedAt: string;
  manifestId: string;
  session: {
    filename: string;
    duration: number;
    sample_count: number;
    sample_rate: number;
    device?: string;
    firmware_version?: string | null;
    session_type?: string;
    hand?: string;
    magnet_type?: string;
    notes?: string | null;
    custom_labels?: string[];
  };
  images: {
    composite?: string;
    calibration_stages?: string;
    orientation_3d?: string;
    orientation_track?: string;
    raw_axes?: string;
  };
  trajectory_comparison: Record<string, string>;
  windows: Array<{
    window_num: number;
    composite?: string;
    time_start?: number;
    time_end?: number;
    sample_count?: number;
    images: Record<string, string>;
    trajectory_images: Record<string, string>;
  }>;
}

interface SessionVisualization {
  timestamp: string;
  filename: string;
  composite_image: string | null;
  calibration_stages_image: string | null;
  orientation_3d_image: string | null;
  orientation_track_image: string | null;
  raw_axes_image: string | null;
  trajectory_comparison_images: Record<string, string>;
  windows: WindowEntry[];
}

interface VisualizationSessionSummary {
  sessionTimestamp: string;
  latestManifestId: string;
  generatedAt: string;
  previousVersions: number;
  filename: string;
  duration?: number;
  windowCount: number;
  hasVisualizations: boolean;
}

interface ManifestIndexEntry {
  sessionTimestamp: string;
  generatedAt: string;
  hasComposite: boolean;
  windowCount: number;
  manifestPath: string;
}

interface ManifestIndex {
  generated: string;
  imageBranch: string;
  baseImageUrl: string;
  sessions: ManifestIndexEntry[];
}

interface GitHubContentItem {
  name: string;
  path: string;
  sha: string;
  size: number;
  url: string;
  html_url: string;
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
 * Normalize timestamp by converting underscores to colons
 */
function normalizeTimestamp(timestamp: string): string {
  return timestamp.replace(/T(\d{2})_(\d{2})_(\d{2})/, 'T$1:$2:$3');
}

/**
 * Denormalize timestamp by converting colons to underscores
 */
function denormalizeTimestamp(timestamp: string): string {
  return timestamp.replace(/T(\d{2}):(\d{2}):(\d{2})/, 'T$1_$2_$3');
}

/**
 * Parse manifest filename to extract session and generated timestamps
 */
function parseManifestFilename(filename: string): { sessionTs: string; generatedTs: string } | null {
  const base = filename.replace('.json', '');
  const match = base.match(/^(.+?)_(\d{4}-\d{2}-\d{2}T[\d_:]+\.\d{3,6}Z)$/);

  if (!match) {
    return null;
  }

  return {
    sessionTs: match[1],
    generatedTs: match[2],
  };
}

/**
 * Try to fetch manifest index from main branch
 */
async function fetchManifestIndex(): Promise<ManifestIndex | null> {
  const url = getRawUrl(MAIN_BRANCH, INDEX_PATH);

  try {
    const response = await fetch(url);
    if (!response.ok) {
      console.log(`[visualizations] Index not found at ${url}: ${response.status}`);
      return null;
    }

    const text = await response.text();

    // Check for Git LFS pointer content
    if (text.startsWith('version https://git-lfs.github.com')) {
      console.log('[visualizations] Index contains Git LFS pointer, skipping');
      return null;
    }

    return JSON.parse(text);
  } catch (error) {
    console.error('[visualizations] Error fetching index:', error);
    return null;
  }
}

/**
 * List manifest files from GitHub Contents API
 */
async function listManifestsFromGitHub(): Promise<GitHubContentItem[]> {
  const url = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${MANIFESTS_PATH}?ref=${MAIN_BRANCH}`;

  const response = await fetch(url, {
    headers: {
      Accept: 'application/vnd.github.v3+json',
    },
  });

  if (!response.ok) {
    if (response.status === 404) {
      return []; // Directory doesn't exist yet
    }
    throw new Error(`GitHub API error: ${response.status}`);
  }

  const contents: GitHubContentItem[] = await response.json();
  return contents.filter((item) => item.type === 'file' && item.name.endsWith('.json') && item.name !== 'index.json');
}

/**
 * Fetch a specific manifest
 */
async function fetchManifest(path: string): Promise<VisualizationManifest | null> {
  const url = getRawUrl(MAIN_BRANCH, path);

  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (error) {
    console.error('[visualizations] Error fetching manifest:', error);
    return null;
  }
}

/**
 * Group manifests by session timestamp
 */
interface ManifestGroup {
  sessionTs: string;
  latestPath: string;
  latestGeneratedTs: string;
  allVersions: Array<{ path: string; generatedTs: string }>;
}

function groupManifestsBySession(files: GitHubContentItem[]): Map<string, ManifestGroup> {
  const groups = new Map<string, ManifestGroup>();

  for (const file of files) {
    const parsed = parseManifestFilename(file.name);
    if (!parsed) continue;

    const { sessionTs, generatedTs } = parsed;
    const path = `${MANIFESTS_PATH}/${file.name}`;

    if (!groups.has(sessionTs)) {
      groups.set(sessionTs, {
        sessionTs,
        latestPath: path,
        latestGeneratedTs: generatedTs,
        allVersions: [],
      });
    }

    const group = groups.get(sessionTs)!;
    group.allVersions.push({ path, generatedTs });

    if (generatedTs > group.latestGeneratedTs) {
      group.latestPath = path;
      group.latestGeneratedTs = generatedTs;
    }
  }

  // Sort versions within each group
  for (const group of groups.values()) {
    group.allVersions.sort((a, b) => b.generatedTs.localeCompare(a.generatedTs));
  }

  return groups;
}

/**
 * Convert manifest to SessionVisualization format
 */
function manifestToSessionVisualization(manifest: VisualizationManifest): SessionVisualization {
  return {
    timestamp: manifest.sessionTimestamp,
    filename: manifest.session.filename,
    composite_image: manifest.images.composite || null,
    calibration_stages_image: manifest.images.calibration_stages || null,
    orientation_3d_image: manifest.images.orientation_3d || null,
    orientation_track_image: manifest.images.orientation_track || null,
    raw_axes_image: manifest.images.raw_axes || null,
    trajectory_comparison_images: manifest.trajectory_comparison,
    windows: manifest.windows.map((w) => ({
      window_num: w.window_num,
      filepath: w.composite,
      time_start: w.time_start,
      time_end: w.time_end,
      sample_count: w.sample_count,
      composite: w.composite,
      images: w.images,
      trajectory_images: w.trajectory_images,
    })),
  };
}

/**
 * Create session summary from manifest group
 */
function createSessionSummary(group: ManifestGroup): VisualizationSessionSummary {
  return {
    sessionTimestamp: normalizeTimestamp(group.sessionTs),
    latestManifestId: `${group.sessionTs}_${group.latestGeneratedTs}`,
    generatedAt: normalizeTimestamp(group.latestGeneratedTs),
    previousVersions: group.allVersions.length - 1,
    filename: `${group.sessionTs}.json`,
    windowCount: 0,
    hasVisualizations: true,
  };
}

export default async function handler(request: Request): Promise<Response> {
  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: CORS_HEADERS,
    });
  }

  if (request.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
    });
  }

  try {
    const url = new URL(request.url);
    const sessionParam = url.searchParams.get('session');
    const historyParam = url.searchParams.get('history') === 'true';

    const sessionFilter = sessionParam ? denormalizeTimestamp(sessionParam) : null;

    console.log('[visualizations] Request:', { sessionParam, sessionFilter, historyParam });

    // Try index first, fall back to API listing
    let groups: Map<string, ManifestGroup>;

    const index = await fetchManifestIndex();
    if (index && index.sessions && index.sessions.length > 0) {
      console.log(`[visualizations] Using index with ${index.sessions.length} sessions`);
      groups = new Map();

      for (const entry of index.sessions) {
        const sessionTs = denormalizeTimestamp(entry.sessionTimestamp);
        const generatedTs = denormalizeTimestamp(entry.generatedAt);

        groups.set(sessionTs, {
          sessionTs,
          latestPath: entry.manifestPath,
          latestGeneratedTs: generatedTs,
          allVersions: [{ path: entry.manifestPath, generatedTs }],
        });
      }
    } else {
      console.log('[visualizations] Falling back to GitHub API listing');
      const files = await listManifestsFromGitHub();
      groups = groupManifestsBySession(files);
    }

    console.log(`[visualizations] Found ${groups.size} sessions with manifests`);

    // Handle single session request
    if (sessionFilter) {
      const group = groups.get(sessionFilter);

      if (!group) {
        return new Response(
          JSON.stringify({
            session: null,
            found: false,
            generatedAt: new Date().toISOString(),
          }),
          {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
              'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
              ...CORS_HEADERS,
            },
          }
        );
      }

      // History mode
      if (historyParam) {
        const versions = group.allVersions.map((v) => ({
          manifestId: `${group.sessionTs}_${v.generatedTs}`,
          generatedAt: normalizeTimestamp(v.generatedTs),
          url: getRawUrl(MAIN_BRANCH, v.path),
        }));

        return new Response(
          JSON.stringify({
            sessionTimestamp: normalizeTimestamp(group.sessionTs),
            versions,
            count: versions.length,
            generatedAt: new Date().toISOString(),
          }),
          {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
              'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
              ...CORS_HEADERS,
            },
          }
        );
      }

      // Fetch latest manifest
      const manifest = await fetchManifest(group.latestPath);

      if (!manifest) {
        return new Response(
          JSON.stringify({
            session: null,
            found: false,
            error: 'Failed to fetch manifest',
            generatedAt: new Date().toISOString(),
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
          }
        );
      }

      const session = manifestToSessionVisualization(manifest);

      return new Response(
        JSON.stringify({
          session,
          manifest,
          previousVersions: group.allVersions.length - 1,
          found: true,
          generatedAt: new Date().toISOString(),
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
            ...CORS_HEADERS,
          },
        }
      );
    }

    // List all sessions
    const summaries: VisualizationSessionSummary[] = [];

    for (const group of groups.values()) {
      summaries.push(createSessionSummary(group));
    }

    summaries.sort((a, b) => b.sessionTimestamp.localeCompare(a.sessionTimestamp));

    return new Response(
      JSON.stringify({
        sessions: summaries,
        count: summaries.length,
        generatedAt: new Date().toISOString(),
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 's-maxage=300, stale-while-revalidate=600',
          ...CORS_HEADERS,
        },
      }
    );
  } catch (error) {
    console.error('Visualizations list error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';

    return new Response(
      JSON.stringify({
        error: 'Failed to list visualizations',
        message,
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
      }
    );
  }
}
