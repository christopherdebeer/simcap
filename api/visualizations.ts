/**
 * Visualizations Listing API Handler
 *
 * Lists visualization manifests stored in Vercel Blob.
 * Uses manifest-based system for efficient listing and versioning.
 *
 * Manifest naming: visualizations/manifests/{session_ts}_{generated_ts}.json
 * - session_ts: Session timestamp (underscore format)
 * - generated_ts: When manifest was generated (underscore format)
 *
 * Endpoints:
 * - GET /api/visualizations - List all sessions with latest manifests
 * - GET /api/visualizations?session={timestamp} - Get latest manifest for session
 * - GET /api/visualizations?session={timestamp}&history=true - Get all manifest versions
 *
 * No authentication required - visualizations are public.
 */

import { list, head } from '@vercel/blob';
import type { ListBlobResultBlob } from '@vercel/blob';

// Types defined locally (path aliases don't work in Vercel serverless)
interface WindowEntry {
  window_num: number;
  filepath: string;
  time_start?: number;
  time_end?: number;
  sample_count?: number;
  images: Record<string, string>;
  trajectory_images: Record<string, string>;
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

// Additional types needed by this handler (also defined locally for serverless compatibility)
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
    composite: string;
    time_start?: number;
    time_end?: number;
    sample_count?: number;
    images: Record<string, string>;
    trajectory_images: Record<string, string>;
  }>;
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

/**
 * Normalize timestamp by converting underscores to colons
 * Filenames use underscores (2025-12-15T22_35_15.567Z) because colons are invalid
 * But session timestamps use colons (2025-12-15T22:35:15.567Z)
 */
function normalizeTimestamp(timestamp: string): string {
  return timestamp.replace(/T(\d{2})_(\d{2})_(\d{2})/, 'T$1:$2:$3');
}

/**
 * Denormalize timestamp by converting colons to underscores (filename safe)
 */
function denormalizeTimestamp(timestamp: string): string {
  return timestamp.replace(/T(\d{2}):(\d{2}):(\d{2})/, 'T$1_$2_$3');
}

/**
 * Parse manifest filename to extract session and generated timestamps
 * Format: {session_ts}_{generated_ts}.json
 * 
 * Handles variations:
 * - Session timestamps may have colons (2025-12-15T22:40:44.984Z) or underscores (2025-12-15T22_35_15.567Z)
 * - Generated timestamps may have 3-6 decimal places (.567Z or .652014Z)
 */
function parseManifestFilename(filename: string): { sessionTs: string; generatedTs: string } | null {
  // Remove .json extension
  const base = filename.replace('.json', '');
  
  // Split on the second timestamp (look for pattern like _2025-12-...)
  // The manifest ID format is: 2025-12-15T22_35_15.567Z_2025-12-18T14_00_00.000000Z
  // Handle both colon and underscore formats, and 3-6 decimal places
  const match = base.match(/^(.+?)_(\d{4}-\d{2}-\d{2}T[\d_:]+\.\d{3,6}Z)$/);
  
  if (!match) {
    console.log('[visualizations] Failed to parse manifest filename:', filename);
    return null;
  }
  
  return {
    sessionTs: match[1],
    generatedTs: match[2],
  };
}

/**
 * Group manifests by session timestamp, keeping track of versions
 */
interface ManifestGroup {
  sessionTs: string;
  latestManifest: ListBlobResultBlob;
  latestGeneratedTs: string;
  allVersions: Array<{ blob: ListBlobResultBlob; generatedTs: string }>;
}

function groupManifestsBySession(blobs: ListBlobResultBlob[]): Map<string, ManifestGroup> {
  const groups = new Map<string, ManifestGroup>();
  
  for (const blob of blobs) {
    // Extract filename from pathname
    const filename = blob.pathname.split('/').pop() || '';
    const parsed = parseManifestFilename(filename);
    
    if (!parsed) continue;
    
    const { sessionTs, generatedTs } = parsed;
    
    if (!groups.has(sessionTs)) {
      groups.set(sessionTs, {
        sessionTs,
        latestManifest: blob,
        latestGeneratedTs: generatedTs,
        allVersions: [],
      });
    }
    
    const group = groups.get(sessionTs)!;
    group.allVersions.push({ blob, generatedTs });
    
    // Update latest if this is newer
    if (generatedTs > group.latestGeneratedTs) {
      group.latestManifest = blob;
      group.latestGeneratedTs = generatedTs;
    }
  }
  
  // Sort versions within each group (newest first)
  for (const group of groups.values()) {
    group.allVersions.sort((a, b) => b.generatedTs.localeCompare(a.generatedTs));
  }
  
  return groups;
}

/**
 * Fetch and parse a manifest from blob storage
 */
async function fetchManifest(url: string): Promise<VisualizationManifest | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.json() as VisualizationManifest;
  } catch (error) {
    console.error('[visualizations] Failed to fetch manifest:', error);
    return null;
  }
}

/**
 * Convert manifest to SessionVisualization format (for backward compatibility)
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
    windows: manifest.windows.map(w => ({
      window_num: w.window_num,
      filepath: w.composite,
      time_start: w.time_start,
      time_end: w.time_end,
      sample_count: w.sample_count,
      images: w.images,
      trajectory_images: w.trajectory_images,
    })),
  };
}

/**
 * Create session summary from manifest group
 */
function createSessionSummary(group: ManifestGroup): VisualizationSessionSummary {
  const normalizedTs = normalizeTimestamp(group.sessionTs);
  const normalizedGenTs = normalizeTimestamp(group.latestGeneratedTs);
  
  return {
    sessionTimestamp: normalizedTs,
    latestManifestId: `${group.sessionTs}_${group.latestGeneratedTs}`,
    generatedAt: normalizedGenTs,
    previousVersions: group.allVersions.length - 1,
    filename: `${group.sessionTs}.json`,
    windowCount: 0, // Will be populated when manifest is fetched
    hasVisualizations: true,
  };
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
    // Parse query parameters from URL
    const url = new URL(request.url);
    const sessionParam = url.searchParams.get('session');
    const historyParam = url.searchParams.get('history') === 'true';
    
    // Normalize session param if provided
    const sessionFilter = sessionParam ? denormalizeTimestamp(sessionParam) : null;

    console.log('[visualizations] Request URL:', request.url);
    console.log('[visualizations] Session param:', sessionParam);
    console.log('[visualizations] Session filter (denormalized):', sessionFilter);
    console.log('[visualizations] History mode:', historyParam);

    // List manifests from blob storage
    const manifestPrefix = 'visualizations/manifests/';
    let allManifestBlobs: ListBlobResultBlob[] = [];
    let cursor: string | undefined = undefined;

    console.log('[visualizations] Listing manifests with prefix:', manifestPrefix);

    do {
      const result: { blobs: ListBlobResultBlob[]; cursor?: string } = await list({
        prefix: manifestPrefix,
        limit: 1000,
        cursor,
      });
      console.log('[visualizations] Manifest list result - count:', result.blobs.length, 'cursor:', result.cursor);
      allManifestBlobs = allManifestBlobs.concat(result.blobs);
      cursor = result.cursor;
    } while (cursor);

    console.log('[visualizations] Total manifests found:', allManifestBlobs.length);

    // Group manifests by session
    const groups = groupManifestsBySession(allManifestBlobs);
    console.log('[visualizations] Sessions with manifests:', groups.size);

    // Handle single session request
    if (sessionFilter) {
      const group = groups.get(sessionFilter);
      
      if (!group) {
        console.log('[visualizations] No manifest found for session:', sessionFilter);
        return new Response(JSON.stringify({
          session: null,
          found: false,
          generatedAt: new Date().toISOString(),
        }), {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
          },
        });
      }

      // History mode: return all versions
      if (historyParam) {
        const versions = group.allVersions.map(v => ({
          manifestId: `${group.sessionTs}_${v.generatedTs}`,
          generatedAt: normalizeTimestamp(v.generatedTs),
          url: v.blob.url,
        }));
        
        return new Response(JSON.stringify({
          sessionTimestamp: normalizeTimestamp(group.sessionTs),
          versions,
          count: versions.length,
          generatedAt: new Date().toISOString(),
        }), {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
          },
        });
      }

      // Fetch latest manifest
      const manifest = await fetchManifest(group.latestManifest.url);
      
      if (!manifest) {
        return new Response(JSON.stringify({
          session: null,
          found: false,
          error: 'Failed to fetch manifest',
          generatedAt: new Date().toISOString(),
        }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }

      // Return in backward-compatible format
      const session = manifestToSessionVisualization(manifest);
      
      return new Response(JSON.stringify({
        session,
        manifest, // Also include full manifest for new clients
        previousVersions: group.allVersions.length - 1,
        found: true,
        generatedAt: new Date().toISOString(),
      }), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
        },
      });
    }

    // List all sessions
    const summaries: VisualizationSessionSummary[] = [];
    
    for (const group of groups.values()) {
      summaries.push(createSessionSummary(group));
    }

    // Sort by session timestamp (newest first)
    summaries.sort((a, b) => b.sessionTimestamp.localeCompare(a.sessionTimestamp));

    return new Response(JSON.stringify({
      sessions: summaries,
      count: summaries.length,
      generatedAt: new Date().toISOString(),
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 's-maxage=300, stale-while-revalidate=600',
      },
    });
  } catch (error) {
    console.error('Visualizations list error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';
    
    return new Response(JSON.stringify({
      error: 'Failed to list visualizations',
      message,
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
