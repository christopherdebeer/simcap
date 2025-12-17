/**
 * Explorer Data API Handler
 *
 * Provides combined session + visualization data for the VIZ explorer.
 * Fetches visualizations list and enriches with session metadata.
 *
 * No authentication required - explorer data is public.
 */

import { list, head } from '@vercel/blob';

/**
 * Fetch and parse a session JSON file from blob storage
 */
async function fetchSessionMetadata(sessionUrl) {
  try {
    const response = await fetch(sessionUrl);
    if (!response.ok) return null;

    const data = await response.json();

    // Extract metadata from v2.x format
    if (data.samples && Array.isArray(data.samples)) {
      const samples = data.samples;
      const metadata = data.metadata || {};
      const labels = data.labels || [];

      // Calculate duration
      const sampleCount = samples.length;
      const sampleRate = metadata.sample_rate || 50;
      const duration = sampleCount / sampleRate;

      return {
        sample_rate: sampleRate,
        duration,
        sample_count: sampleCount,
        device: metadata.device || 'GAMBIT',
        firmware_version: metadata.firmware_version || null,
        session_type: metadata.session_type || 'recording',
        hand: metadata.hand || 'unknown',
        magnet_type: metadata.magnet_type || 'unknown',
        notes: metadata.notes || null,
        custom_labels: extractCustomLabels(labels),
        labels,
      };
    }

    // v1.x format - direct array
    if (Array.isArray(data)) {
      return {
        sample_rate: 50,
        duration: data.length / 50,
        sample_count: data.length,
        device: 'GAMBIT',
        firmware_version: null,
        session_type: 'recording',
        hand: 'unknown',
        magnet_type: 'unknown',
        notes: null,
        custom_labels: [],
        labels: [],
      };
    }

    return null;
  } catch (error) {
    console.error(`Failed to fetch session metadata: ${error.message}`);
    return null;
  }
}

/**
 * Extract custom labels from label segments
 */
function extractCustomLabels(labels) {
  const customLabels = new Set();
  for (const segment of labels) {
    if (segment.labels?.custom) {
      segment.labels.custom.forEach(l => customLabels.add(l));
    }
  }
  return Array.from(customLabels);
}

/**
 * Parse visualization files into session-grouped structure
 */
function groupVisualizationsBySession(blobs) {
  const sessions = new Map();

  for (const blob of blobs) {
    const path = blob.pathname.replace('visualizations/', '');

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
        filename: `${timestamp.replace(/:/g, '_')}.json`,
        composite_image: null,
        calibration_stages_image: null,
        raw_images: [],
        trajectory_comparison_images: {},
        windows: [],
        // Metadata (will be populated from session JSON)
        duration: 0,
        sample_rate: 50,
        device: null,
        firmware_version: null,
        session_type: 'recording',
        hand: 'unknown',
        magnet_type: 'unknown',
        notes: null,
        custom_labels: [],
        labels: [],
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
      case 'orientation_track':
      case 'raw_axes':
        session.raw_images.push(blob.url);
        break;
      case 'trajectory_comparison':
        const trajType = subPath.replace('.png', '').replace('_3d', '');
        session.trajectory_comparison_images[trajType] = blob.url;
        break;
      case 'window': {
        let window = session.windows.find(w => w.window_num === windowNum);
        if (!window) {
          window = {
            window_num: windowNum,
            filepath: blob.url,
            time_start: windowNum,  // Approximate - actual times from session data
            time_end: windowNum + 1,
            accel_mag_mean: 0,
            gyro_mag_mean: 0,
            images: {},
            trajectory_images: {},
          };
          session.windows.push(window);
        }
        const imageName = subPath.replace('.png', '');
        if (imageName.startsWith('trajectory_') && !imageName.includes('accel') && !imageName.includes('gyro') && !imageName.includes('mag') && !imageName.includes('combined')) {
          const trajKey = imageName.replace('trajectory_', '');
          window.trajectory_images[trajKey] = blob.url;
        } else if (imageName === 'composite' || imageName === 'window_composite') {
          window.filepath = blob.url;
        } else {
          window.images[imageName] = blob.url;
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

export default async function handler(request, response) {
  if (request.method !== 'GET') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Fetch all visualizations
    let allVizBlobs = [];
    let cursor = undefined;

    do {
      const result = await list({
        prefix: 'visualizations/',
        limit: 1000,
        cursor,
      });
      allVizBlobs = allVizBlobs.concat(result.blobs);
      cursor = result.cursor;
    } while (cursor);

    // Group visualizations by session
    const sessionMap = groupVisualizationsBySession(allVizBlobs);

    // Fetch session list to get metadata URLs
    const { blobs: sessionBlobs } = await list({
      prefix: 'sessions/',
      limit: 1000,
    });

    // Create lookup map for session URLs
    const sessionUrls = new Map();
    for (const blob of sessionBlobs) {
      if (blob.pathname.endsWith('.json')) {
        // Extract timestamp from filename
        const filename = blob.pathname.replace('sessions/', '');
        const timestamp = filename.replace('.json', '').replace(/_/g, ':');
        sessionUrls.set(timestamp, blob.url);
      }
    }

    // Enrich sessions with metadata (fetch in parallel, limit concurrency)
    const sessionEntries = Array.from(sessionMap.entries());
    const batchSize = 10;

    for (let i = 0; i < sessionEntries.length; i += batchSize) {
      const batch = sessionEntries.slice(i, i + batchSize);
      await Promise.all(
        batch.map(async ([timestamp, session]) => {
          const sessionUrl = sessionUrls.get(timestamp);
          if (sessionUrl) {
            const metadata = await fetchSessionMetadata(sessionUrl);
            if (metadata) {
              Object.assign(session, metadata);
            }
          }
        })
      );
    }

    // Convert to array sorted by timestamp (newest first)
    const sessions = Array.from(sessionMap.values())
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    // Set cache headers
    response.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=600');

    return response.status(200).json({
      sessions,
      count: sessions.length,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Explorer data error:', error);
    return response.status(500).json({
      error: 'Failed to load explorer data',
      message: error.message,
    });
  }
}
