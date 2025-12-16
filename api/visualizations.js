/**
 * Visualizations Listing API Handler
 *
 * Lists all visualization assets stored in Vercel Blob.
 * Returns structured data for VIZ explorer compatible with session-data.js format.
 *
 * No authentication required - visualizations are public.
 */

import { list } from '@vercel/blob';

/**
 * Parse visualization files into session-grouped structure
 */
function groupVisualizationsBySession(blobs) {
  const sessions = new Map();

  for (const blob of blobs) {
    const path = blob.pathname.replace('visualizations/', '');

    // Extract timestamp from filename patterns:
    // - composite_2025-12-15T22:40:44.984Z.png
    // - windows_2025-12-15T22:40:44.984Z/window_001/timeseries_accel.png
    // - trajectory_comparison_2025-12-15T22:40:44.984Z/raw_3d.png

    let timestamp = null;
    let imageType = null;
    let windowNum = null;
    let subPath = null;

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

    const session = sessions.get(timestamp);

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
        // e.g., raw_3d.png, filtered_3d.png, combined_overlay.png
        const trajType = subPath.replace('.png', '').replace('_3d', '');
        session.trajectory_comparison_images[trajType] = blob.url;
        break;
      case 'window':
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
        break;
    }
  }

  // Sort windows within each session
  for (const session of sessions.values()) {
    session.windows.sort((a, b) => a.window_num - b.window_num);
  }

  // Convert to array sorted by timestamp (newest first)
  return Array.from(sessions.values())
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
}

export default async function handler(request, response) {
  // Only allow GET requests
  if (request.method !== 'GET') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // List all visualizations from blob storage
    // Note: list() has a default limit, so we may need to paginate for large stores
    let allBlobs = [];
    let cursor = undefined;

    do {
      const result = await list({
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
    return response.status(500).json({
      error: 'Failed to list visualizations',
      message: error.message,
    });
  }
}
