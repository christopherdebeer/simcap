/**
 * Explorer Data API Handler
 *
 * Provides combined session + visualization data for the VIZ explorer.
 * Fetches visualizations list and enriches with session metadata.
 *
 * No authentication required - explorer data is public.
 */

import { list } from '@vercel/blob';

// ===== Type Definitions =====

interface LabelSegment {
  labels?: {
    custom?: string[];
  };
}

interface SessionMetadata {
  sample_rate: number;
  duration: number;
  sample_count: number;
  device: string;
  firmware_version: string | null;
  session_type: string;
  hand: string;
  magnet_type: string;
  notes: string | null;
  custom_labels: string[];
  labels: LabelSegment[];
}

interface WindowData {
  window_num: number;
  filepath: string;
  time_start: number;
  time_end: number;
  accel_mag_mean: number;
  gyro_mag_mean: number;
  images: Record<string, string>;
  trajectory_images: Record<string, string>;
}

interface SessionData extends SessionMetadata {
  timestamp: string;
  filename: string;
  composite_image: string | null;
  calibration_stages_image: string | null;
  raw_images: string[];
  trajectory_comparison_images: Record<string, string>;
  windows: WindowData[];
}

interface BlobItem {
  pathname: string;
  url: string;
}

interface ExplorerResponse {
  sessions: SessionData[];
  count: number;
  generatedAt: string;
}

interface ErrorResponse {
  error: string;
  message?: string;
}

// ===== Helper Functions =====

/**
 * Fetch and parse a session JSON file from blob storage
 */
async function fetchSessionMetadata(sessionUrl: string): Promise<SessionMetadata | null> {
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
    console.error(`Failed to fetch session metadata: ${(error as Error).message}`);
    return null;
  }
}

/**
 * Extract custom labels from label segments
 */
function extractCustomLabels(labels: LabelSegment[]): string[] {
  const customLabels = new Set<string>();
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
function groupVisualizationsBySession(blobs: BlobItem[]): Map<string, SessionData> {
  const sessions = new Map<string, SessionData>();

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
        filename: `${timestamp.replace(/:/g, '_')}.json`,
        composite_image: null,
        calibration_stages_image: null,
        raw_images: [],
        trajectory_comparison_images: {},
        windows: [],
        // Metadata (will be populated from session JSON)
        duration: 0,
        sample_rate: 50,
        sample_count: 0,
        device: 'GAMBIT',
        firmware_version: null,
        session_type: 'recording',
        hand: 'unknown',
        magnet_type: 'unknown',
        notes: null,
        custom_labels: [],
        labels: [],
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
      case 'orientation_track':
      case 'raw_axes':
        session.raw_images.push(blob.url);
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
    // Fetch all visualizations
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

    // Group visualizations by session
    const sessionMap = groupVisualizationsBySession(allVizBlobs);

    // Fetch session list to get metadata URLs
    const { blobs: sessionBlobs } = await list({
      prefix: 'sessions/',
      limit: 1000,
    });

    // Create lookup map for session URLs
    const sessionUrls = new Map<string, string>();
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
        'Cache-Control': 's-maxage=300, stale-while-revalidate=600',
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
