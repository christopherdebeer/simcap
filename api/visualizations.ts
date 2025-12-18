/**
 * Visualizations Listing API Handler
 *
 * Lists all visualization assets stored in Vercel Blob.
 * Returns structured data for VIZ explorer compatible with session-data.js format.
 *
 * No authentication required - visualizations are public.
 */

import { list } from '@vercel/blob';
import type { ListBlobResultBlob } from '@vercel/blob';
import type { WindowEntry, SessionVisualization } from '@api/types';

/**
 * Normalize timestamp by converting underscores to colons
 * Filenames use underscores (2025-12-15T22_35_15.567Z) because colons are invalid
 * But session timestamps use colons (2025-12-15T22:35:15.567Z)
 */
function normalizeTimestamp(timestamp: string): string {
  // Convert underscores in time portion to colons: T22_35_15 -> T22:35:15
  return timestamp.replace(/T(\d{2})_(\d{2})_(\d{2})/, 'T$1:$2:$3');
}

type ImageType = 'composite' | 'calibration_stages' | 'orientation_3d' | 'orientation_track' | 'raw_axes' | 'window' | 'trajectory_comparison';

/**
 * Parse visualization files into session-grouped structure
 */
function groupVisualizationsBySession(blobs: ListBlobResultBlob[]): SessionVisualization[] {
  const sessions = new Map<string, SessionVisualization>();

  for (const blob of blobs) {
    const path = blob.pathname.replace('visualizations/', '');

    // Extract timestamp from filename patterns:
    // - composite_2025-12-15T22:40:44.984Z.png
    // - windows_2025-12-15T22:40:44.984Z/window_001/timeseries_accel.png
    // - trajectory_comparison_2025-12-15T22:40:44.984Z/raw_3d.png

    let timestamp: string | null = null;
    let imageType: ImageType | null = null;
    let windowNum: number | null = null;
    let subPath: string | null = null;

    // Match composite/calibration/orientation/raw_axes images
    const compositeMatch = path.match(/^(composite|calibration_stages|orientation_3d|orientation_track|raw_axes)_(.+?)\.png$/);
    if (compositeMatch) {
      imageType = compositeMatch[1] as ImageType;
      timestamp = compositeMatch[2];
    }

    // Match window images
    const windowMatch = path.match(/^windows_(.+?)\/window_(\d+)\/(.+)$/);
    if (windowMatch) {
      timestamp = windowMatch[1];
      windowNum = parseInt(windowMatch[2], 10);
      subPath = windowMatch[3];
      imageType = 'window';
    }

    // Match trajectory comparison images
    const trajMatch = path.match(/^trajectory_comparison_(.+?)\/(.+)$/);
    if (trajMatch) {
      timestamp = trajMatch[1];
      subPath = trajMatch[2];
      imageType = 'trajectory_comparison';
    }

    if (!timestamp) continue;

    // Normalize timestamp (convert underscores to colons for consistency with session timestamps)
    const normalizedTimestamp = normalizeTimestamp(timestamp);

    // Initialize session entry
    if (!sessions.has(normalizedTimestamp)) {
      sessions.set(normalizedTimestamp, {
        timestamp: normalizedTimestamp,
        filename: `${timestamp.replace(/:/g, '_')}.json`, // Keep underscores in filename
        composite_image: null,
        calibration_stages_image: null,
        orientation_3d_image: null,
        orientation_track_image: null,
        raw_axes_image: null,
        trajectory_comparison_images: {},
        windows: [],
      });
    }

    const session = sessions.get(normalizedTimestamp)!;

    // Assign to appropriate field
    switch (imageType) {
      case 'composite':
        session.composite_image = blob.url;
        break;
      case 'calibration_stages':
        session.calibration_stages_image = blob.url;
        break;
      case 'orientation_3d':
        session.orientation_3d_image = blob.url;
        break;
      case 'orientation_track':
        session.orientation_track_image = blob.url;
        break;
      case 'raw_axes':
        session.raw_axes_image = blob.url;
        break;
      case 'trajectory_comparison':
        if (subPath) {
          // e.g., raw_3d.png, filtered_3d.png, combined_overlay.png
          const trajType = subPath.replace('.png', '').replace('_3d', '');
          session.trajectory_comparison_images[trajType] = blob.url;
        }
        break;
      case 'window':
        if (windowNum !== null && subPath) {
          // Find or create window entry
          let window = session.windows.find(w => w.window_num === windowNum);
          if (!window) {
            window = {
              window_num: windowNum,
              filepath: `windows_${timestamp}/window_${String(windowNum).padStart(3, '0')}.png`,
              images: {},
              trajectory_images: {},
            };
            session.windows.push(window);
          }
          // Parse subPath to determine image type
          // e.g., timeseries_accel.png, trajectory_raw.png, signature.png
          const imageName = subPath.replace('.png', '');
          if (imageName.startsWith('trajectory_') && !imageName.includes('accel') && !imageName.includes('gyro') && !imageName.includes('mag') && !imageName.includes('combined')) {
            // trajectory_raw, trajectory_iron, trajectory_fused, trajectory_filtered, etc.
            const trajKey = imageName.replace('trajectory_', '');
            window.trajectory_images[trajKey] = blob.url;
          } else {
            window.images[imageName] = blob.url;
          }
        }
        break;
    }
  }

  // Sort windows within each session
  for (const session of sessions.values()) {
    session.windows.sort((a, b) => a.window_num - b.window_num);
  }

  // Convert to array sorted by timestamp (newest first)
  return Array.from(sessions.values())
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
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
    const sessionFilter = sessionParam ? normalizeTimestamp(sessionParam) : null;

    console.log('[visualizations] Request URL:', request.url);
    console.log('[visualizations] Session param:', sessionParam);
    console.log('[visualizations] Session filter (normalized):', sessionFilter);

    // Build prefix for blob listing
    const prefix = 'visualizations/';
    
    // List visualizations from blob storage
    // Note: list() has a default limit, so we may need to paginate for large stores
    let allBlobs: ListBlobResultBlob[] = [];
    let cursor: string | undefined = undefined;

    console.log('[visualizations] Listing blobs with prefix:', prefix);

    do {
      const result: { blobs: ListBlobResultBlob[]; cursor?: string } = await list({
        prefix,
        limit: 1000,
        cursor,
      });
      console.log('[visualizations] Blob list result - count:', result.blobs.length, 'cursor:', result.cursor);
      allBlobs = allBlobs.concat(result.blobs);
      cursor = result.cursor;
    } while (cursor);

    console.log('[visualizations] Total blobs found:', allBlobs.length);
    if (allBlobs.length > 0) {
      console.log('[visualizations] First few blob paths:', allBlobs.slice(0, 5).map(b => b.pathname));
    }

    // Group into session structure
    let sessions = groupVisualizationsBySession(allBlobs);

    console.log('[visualizations] Sessions found:', sessions.length);
    if (sessions.length > 0) {
      console.log('[visualizations] Session timestamps:', sessions.map(s => s.timestamp));
    }

    // Filter to specific session if requested
    if (sessionFilter) {
      console.log('[visualizations] Filtering for session:', sessionFilter);
      sessions = sessions.filter(s => s.timestamp === sessionFilter);
      console.log('[visualizations] After filter - sessions found:', sessions.length);
    }

    // Return single session object if filtering, array otherwise
    if (sessionFilter) {
      const session = sessions[0] || null;
      return new Response(JSON.stringify({
        session,
        found: session !== null,
        generatedAt: new Date().toISOString(),
      }), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 's-maxage=300, stale-while-revalidate=600',
        },
      });
    }

    return new Response(JSON.stringify({
      sessions,
      count: sessions.length,
      totalFiles: allBlobs.length,
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
