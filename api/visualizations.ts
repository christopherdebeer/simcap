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
import type { VercelRequest, VercelResponse } from '@vercel/node';

// Types for visualization structure
interface WindowEntry {
  window_num: number;
  filepath: string;
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

interface VisualizationsResponse {
  sessions: SessionVisualization[];
  count: number;
  totalFiles: number;
  generatedAt: string;
}

interface ErrorResponse {
  error: string;
  message?: string;
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

    // Initialize session entry
    if (!sessions.has(timestamp)) {
      sessions.set(timestamp, {
        timestamp,
        filename: `${timestamp}.json`,
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

export default async function handler(
  request: VercelRequest,
  response: VercelResponse
) {
  // Only allow GET requests
  if (request.method !== 'GET') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // List all visualizations from blob storage
    // Note: list() has a default limit, so we may need to paginate for large stores
    let allBlobs: ListBlobResultBlob[] = [];
    let cursor: string | undefined = undefined;

    do {
      const result: { blobs: ListBlobResultBlob[]; cursor?: string } = await list({
        prefix: 'visualizations/',
        limit: 1000,
        cursor,
      });
      allBlobs = allBlobs.concat(result.blobs);
      cursor = result.cursor;
    } while (cursor);

    // Group into session structure
    const sessions = groupVisualizationsBySession(allBlobs);

    // Set cache headers
    response.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=600');

    return response.status(200).json({
      sessions,
      count: sessions.length,
      totalFiles: allBlobs.length,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Visualizations list error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';
    return response.status(500).json({
      error: 'Failed to list visualizations',
      message,
    });
  }
}
