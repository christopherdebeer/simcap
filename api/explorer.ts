/**
 * Explorer Data API Handler
 *
 * Provides lightweight session + visualization metadata for the VIZ explorer.
 * Returns URLs and identifiers only - clients fetch individual session data directly.
 *
 * No authentication required - explorer data is public.
 */

import { list } from '@vercel/blob';

// ===== Type Definitions =====

interface WindowEntry {
  window_num: number;
  images: Record<string, string>;
  trajectory_images: Record<string, string>;
}

interface SessionEntry {
  timestamp: string;
  sessionUrl: string | null;        // URL to fetch full session JSON
  filename: string;
  size: number | null;
  uploadedAt: string | null;
  // Visualization URLs (not data)
  composite_image: string | null;
  calibration_stages_image: string | null;
  orientation_3d_image: string | null;
  orientation_track_image: string | null;
  raw_axes_image: string | null;
  trajectory_comparison_images: Record<string, string>;
  windows: WindowEntry[];
}

interface BlobItem {
  pathname: string;
  url: string;
  size?: number;
  uploadedAt?: Date;
}

interface ExplorerResponse {
  sessions: SessionEntry[];
  count: number;
  generatedAt: string;
}

interface ErrorResponse {
  error: string;
  message?: string;
}

// ===== Helper Functions =====

/**
 * Parse visualization files into session-grouped structure (URLs only, no data fetching)
 */
function groupVisualizationsBySession(blobs: BlobItem[]): Map<string, SessionEntry> {
  const sessions = new Map<string, SessionEntry>();

  for (const blob of blobs) {
    const path = blob.pathname.replace('visualizations/', '');

    let timestamp: string | null = null;
    let imageType: string | null = null;
    let windowNum: number | null = null;
    let subPath: string | null = null;

    // Match composite/calibration/orientation/raw_axes images
    const compositeMatch = path.match(/^(composite|calibration_stages|orientation_3d|orientation_track|raw_axes)_(.+?)\.png$/);
    if (compositeMatch) {
      imageType = compositeMatch[1];
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

    // Initialize session entry
    if (!sessions.has(timestamp)) {
      sessions.set(timestamp, {
        timestamp,
        sessionUrl: null,  // Will be populated from sessions list
        filename: `${timestamp.replace(/:/g, '_')}.json`,
        size: null,
        uploadedAt: null,
        composite_image: null,
        calibration_stages_image: null,
        orientation_3d_image: null,
        orientation_track_image: null,
        raw_axes_image: null,
        trajectory_comparison_images: {},
        windows: [],
      });
    }

    const session = sessions.get(timestamp)!;

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
          const trajType = subPath.replace('.png', '').replace('_3d', '');
          session.trajectory_comparison_images[trajType] = blob.url;
        }
        break;
      case 'window': {
        if (windowNum !== null && subPath) {
          let window = session.windows.find(w => w.window_num === windowNum);
          if (!window) {
            window = {
              window_num: windowNum,
              images: {},
              trajectory_images: {},
            };
            session.windows.push(window);
          }
          const imageName = subPath.replace('.png', '');
          if (imageName.startsWith('trajectory_') && !imageName.includes('accel') && !imageName.includes('gyro') && !imageName.includes('mag') && !imageName.includes('combined')) {
            const trajKey = imageName.replace('trajectory_', '');
            window.trajectory_images[trajKey] = blob.url;
          } else {
            window.images[imageName] = blob.url;
          }
        }
        break;
      }
    }
  }

  // Sort windows within each session
  for (const session of sessions.values()) {
    session.windows.sort((a, b) => a.window_num - b.window_num);
  }

  return sessions;
}

// ===== Main Handler =====

export default async function handler(request: Request): Promise<Response> {
  if (request.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' } as ErrorResponse), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    // Fetch visualizations and sessions in parallel
    const [vizResult, sessionsResult] = await Promise.all([
      (async () => {
        let allVizBlobs: BlobItem[] = [];
        let cursor: string | undefined = undefined;
        do {
          const result: { blobs: BlobItem[]; cursor?: string } = await list({
            prefix: 'visualizations/',
            limit: 1000,
            cursor,
          });
          allVizBlobs = allVizBlobs.concat(result.blobs);
          cursor = result.cursor;
        } while (cursor);
        return allVizBlobs;
      })(),
      list({ prefix: 'sessions/', limit: 1000 }),
    ]);

    // Group visualizations by session (URLs only)
    const sessionMap = groupVisualizationsBySession(vizResult);

    // Create lookup for session metadata (URL, size, uploadedAt)
    const sessionInfo = new Map<string, { url: string; size: number; uploadedAt: Date }>();
    for (const blob of sessionsResult.blobs) {
      if (blob.pathname.endsWith('.json')) {
        const filename = blob.pathname.replace('sessions/', '');
        const timestamp = filename.replace('.json', '').replace(/_/g, ':');
        sessionInfo.set(timestamp, {
          url: blob.url,
          size: blob.size,
          uploadedAt: blob.uploadedAt,
        });
      }
    }

    // Link session URLs to visualization entries (no data fetching)
    for (const [timestamp, session] of sessionMap) {
      const info = sessionInfo.get(timestamp);
      if (info) {
        session.sessionUrl = info.url;
        session.size = info.size;
        session.uploadedAt = info.uploadedAt.toISOString();
      }
    }

    // Also add sessions that have no visualizations yet
    for (const [timestamp, info] of sessionInfo) {
      if (!sessionMap.has(timestamp)) {
        sessionMap.set(timestamp, {
          timestamp,
          sessionUrl: info.url,
          filename: `${timestamp.replace(/:/g, '_')}.json`,
          size: info.size,
          uploadedAt: info.uploadedAt.toISOString(),
          composite_image: null,
          calibration_stages_image: null,
          orientation_3d_image: null,
          orientation_track_image: null,
          raw_axes_image: null,
          trajectory_comparison_images: {},
          windows: [],
        });
      }
    }

    // Convert to array sorted by timestamp (newest first)
    const sessions = Array.from(sessionMap.values())
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    const responseData: ExplorerResponse = {
      sessions,
      count: sessions.length,
      generatedAt: new Date().toISOString(),
    };

    return new Response(JSON.stringify(responseData), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
      },
    });
  } catch (error) {
    console.error('Explorer data error:', error);
    return new Response(JSON.stringify({
      error: 'Failed to load explorer data',
      message: (error as Error).message,
    } as ErrorResponse), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
