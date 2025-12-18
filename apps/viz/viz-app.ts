/**
 * SIMCAP Data Visualization Explorer Application
 * Interactive viewer for 9-axis IMU sensor data and gesture recognition
 */

// ===== Type Definitions =====

interface LabelSegment {
    start_sample: number;
    end_sample: number;
    labels?: {
        pose?: string;
        motion?: string;
        calibration?: string;
        custom?: string[];
    };
}

interface WindowEntry {
    window_num: number;
    filepath?: string;
    time_start?: number;
    time_end?: number;
    accel_mag_mean?: number;
    gyro_mag_mean?: number;
    sample_count?: number;
    has_visualizations?: boolean;
    images?: Record<string, string>;
    trajectory_images?: Record<string, string>;
}

interface SensorSample {
    ax: number;
    ay: number;
    az: number;
    gx: number;
    gy: number;
    gz: number;
    mx: number;
    my: number;
    mz: number;
    // Converted values (may or may not be present)
    ax_g?: number;
    ay_g?: number;
    az_g?: number;
    gx_dps?: number;
    gy_dps?: number;
    gz_dps?: number;
    mx_ut?: number;
    my_ut?: number;
    mz_ut?: number;
    // Other fields
    l?: number;
    t?: number;
    c?: number;
    s?: number;
    b?: number;
    n?: number;
    dt?: number;
    [key: string]: unknown;
}

interface SessionEntry {
    timestamp: string;
    sessionUrl: string | null;
    filename: string;
    size: number | null;
    uploadedAt: string | null;
    composite_image: string | null;
    calibration_stages_image: string | null;
    orientation_3d_image?: string | null;
    orientation_track_image?: string | null;
    raw_axes_image?: string | null;
    trajectory_comparison_images: Record<string, string>;
    windows: WindowEntry[];
    // Metadata (loaded on demand)
    _metadataLoaded?: boolean;
    duration?: number;
    sample_rate?: number;
    sample_count?: number;
    device?: string;
    firmware_version?: string | null;
    session_type?: string;
    hand?: string;
    magnet_type?: string;
    notes?: string | null;
    custom_labels?: string[];
    labels?: LabelSegment[];
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
    samples?: SensorSample[]; // Include samples for window calculation
}

interface WindowLabels {
    pose: string | null;
    motion: string | null;
    calibration: string | null;
    custom: string[];
}

interface ViewSettings {
    showComposite: boolean;
    showCalibration: boolean;
    showWindows: boolean;
    showRaw: boolean;
    showWinComposite: boolean;
    showWinAccelTime: boolean;
    showWinGyroTime: boolean;
    showWinMagTime: boolean;
    showWinAccel3d: boolean;
    showWinGyro3d: boolean;
    showWinMag3d: boolean;
    showWinCombined3d: boolean;
    showWinSignature: boolean;
    showWinStats: boolean;
    showWinTrajRaw: boolean;
    showWinTrajIron: boolean;
    showWinTrajFused: boolean;
    showWinTrajFiltered: boolean;
    showWinTrajCombined: boolean;
    showWinTrajStats: boolean;
}

// ===== State =====

let sessionsData: SessionEntry[] = [];
let filteredSessions: SessionEntry[] = [];
let dataLoaded = false;
let loadError: Error | null = null;
let defaultExpanded = false;
let selectedLabel = 'all';

// Cache for loaded session metadata
const sessionMetadataCache = new Map<string, SessionMetadata>();

// API base URL - use relative path for same-origin
const API_BASE = '';

// Sample rate for label calculations
const SAMPLE_RATE = 50;

// View settings
const viewSettings: ViewSettings = {
    showComposite: true,
    showCalibration: true,
    showWindows: true,
    showRaw: true,
    showWinComposite: true,
    showWinAccelTime: false,
    showWinGyroTime: false,
    showWinMagTime: false,
    showWinAccel3d: false,
    showWinGyro3d: false,
    showWinMag3d: false,
    showWinCombined3d: false,
    showWinSignature: false,
    showWinStats: false,
    showWinTrajRaw: false,
    showWinTrajIron: false,
    showWinTrajFused: false,
    showWinTrajFiltered: false,
    showWinTrajCombined: false,
    showWinTrajStats: false
};

// ===== DOM Helpers =====

function getEl(id: string): HTMLElement | null {
    return document.getElementById(id);
}

// ===== API Functions =====

interface SessionsApiResponse {
    sessions: Array<{
        filename: string;
        pathname: string;
        url: string;
        downloadUrl: string;
        size: number;
        uploadedAt: string;
        timestamp: string;
    }>;
    count: number;
    generatedAt: string;
}

interface VisualizationsApiResponse {
    session: {
        timestamp: string;
        filename: string;
        composite_image: string | null;
        calibration_stages_image: string | null;
        orientation_3d_image: string | null;
        orientation_track_image: string | null;
        raw_axes_image: string | null;
        trajectory_comparison_images: Record<string, string>;
        windows: WindowEntry[];
    } | null;
    found: boolean;
    generatedAt: string;
}

/**
 * Fetch session list from /api/sessions
 */
async function fetchSessionsList(): Promise<SessionEntry[]> {
    try {
        const response = await fetch(`${API_BASE}/api/sessions`);
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        const data: SessionsApiResponse = await response.json();

        // Transform sessions API response to SessionEntry format
        return (data.sessions || []).map(s => ({
            timestamp: s.timestamp,
            sessionUrl: s.url,
            filename: s.filename,
            size: s.size,
            uploadedAt: s.uploadedAt,
            // Visualization fields - will be populated lazily
            composite_image: null,
            calibration_stages_image: null,
            orientation_3d_image: null,
            orientation_track_image: null,
            raw_axes_image: null,
            trajectory_comparison_images: {},
            windows: [],
        }));
    } catch (error) {
        console.error('Failed to fetch sessions list:', error);
        throw error;
    }
}

/**
 * Fetch visualizations for a specific session from /api/visualizations?session=TIMESTAMP
 */
async function fetchSessionVisualizations(timestamp: string): Promise<VisualizationsApiResponse['session']> {
    try {
        const response = await fetch(`${API_BASE}/api/visualizations?session=${encodeURIComponent(timestamp)}`);
        if (!response.ok) {
            console.warn(`Failed to fetch visualizations for ${timestamp}: ${response.status}`);
            return null;
        }
        const data: VisualizationsApiResponse = await response.json();
        return data.session;
    } catch (error) {
        console.error(`Failed to fetch visualizations for ${timestamp}:`, error);
        return null;
    }
}

/**
 * Fetch session list (replaces fetchExplorerData)
 * Now uses /api/sessions instead of /api/explorer
 */
async function fetchExplorerData(): Promise<SessionEntry[]> {
    return fetchSessionsList();
}

async function fetchSessionMetadata(sessionUrl: string): Promise<SessionMetadata | null> {
    if (!sessionUrl) return null;

    // Check cache first
    if (sessionMetadataCache.has(sessionUrl)) {
        return sessionMetadataCache.get(sessionUrl)!;
    }

    try {
        const response = await fetch(sessionUrl);
        if (!response.ok) return null;

        const data = await response.json();
        let metadata: SessionMetadata | null = null;

        // Extract metadata from v2.x format
        if (data.samples && Array.isArray(data.samples)) {
            const samples = data.samples;
            const meta = data.metadata || {};
            const labels = data.labels || [];

            const sampleCount = samples.length;
            const sampleRate = meta.sample_rate || 50;
            const duration = sampleCount / sampleRate;

            metadata = {
                sample_rate: sampleRate,
                duration,
                sample_count: sampleCount,
                device: meta.device || 'GAMBIT',
                firmware_version: meta.firmware_version || null,
                session_type: meta.session_type || 'recording',
                hand: meta.hand || 'unknown',
                magnet_type: meta.magnet_type || 'unknown',
                notes: meta.notes || null,
                custom_labels: extractCustomLabels(labels),
                labels,
                samples, // Include samples for window calculation
            };
        } else if (Array.isArray(data)) {
            // v1.x format - direct array
            metadata = {
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
                samples: data as SensorSample[], // Include samples for window calculation
            };
        }

        // Cache the result
        if (metadata) {
            sessionMetadataCache.set(sessionUrl, metadata);
        }
        return metadata;
    } catch (error) {
        console.error(`Failed to fetch session metadata: ${(error as Error).message}`);
        return null;
    }
}

function extractCustomLabels(labels: LabelSegment[]): string[] {
    const customLabels = new Set<string>();
    for (const segment of labels) {
        if (segment.labels?.custom) {
            segment.labels.custom.forEach(l => customLabels.add(l));
        }
    }
    return Array.from(customLabels);
}

// ===== Window Calculation Functions =====

const WINDOW_DURATION = 1.0; // seconds per window

/**
 * Calculate windows from sample data
 * Each window is 1 second of data based on sample rate
 */
function calculateWindowsFromSamples(
    samples: SensorSample[],
    sampleRate: number,
    existingWindows: WindowEntry[] = []
): WindowEntry[] {
    if (!samples || samples.length === 0) {
        return existingWindows;
    }

    const samplesPerWindow = Math.floor(sampleRate * WINDOW_DURATION);
    const numWindows = Math.ceil(samples.length / samplesPerWindow);

    // Create a map of existing windows by window_num for merging
    const existingWindowMap = new Map<number, WindowEntry>();
    for (const w of existingWindows) {
        existingWindowMap.set(w.window_num, w);
    }

    const windows: WindowEntry[] = [];

    for (let i = 0; i < numWindows; i++) {
        const windowNum = i + 1;
        const startSample = i * samplesPerWindow;
        const endSample = Math.min((i + 1) * samplesPerWindow, samples.length);
        const windowSamples = samples.slice(startSample, endSample);

        const timeStart = startSample / sampleRate;
        const timeEnd = endSample / sampleRate;

        // Calculate statistics
        const stats = calculateWindowStats(windowSamples);

        // Check if we have an existing window with visualizations
        const existingWindow = existingWindowMap.get(windowNum);
        const hasVisualizations = !!(
            existingWindow?.filepath ||
            (existingWindow?.images && Object.keys(existingWindow.images).length > 0) ||
            (existingWindow?.trajectory_images && Object.keys(existingWindow.trajectory_images).length > 0)
        );

        // Merge with existing window data (preserving visualization URLs)
        const window: WindowEntry = {
            window_num: windowNum,
            time_start: timeStart,
            time_end: timeEnd,
            sample_count: windowSamples.length,
            accel_mag_mean: stats.accelMagMean,
            gyro_mag_mean: stats.gyroMagMean,
            has_visualizations: hasVisualizations,
            // Preserve existing visualization data
            filepath: existingWindow?.filepath,
            images: existingWindow?.images || {},
            trajectory_images: existingWindow?.trajectory_images || {},
        };

        windows.push(window);
    }

    return windows;
}

/**
 * Calculate statistics for a window of samples
 */
function calculateWindowStats(samples: SensorSample[]): {
    accelMagMean: number;
    gyroMagMean: number;
} {
    if (samples.length === 0) {
        return { accelMagMean: 0, gyroMagMean: 0 };
    }

    let accelMagSum = 0;
    let gyroMagSum = 0;

    for (const s of samples) {
        // Use converted values if available, otherwise use raw
        const ax = s.ax_g !== undefined ? s.ax_g : s.ax / 16384; // Approximate conversion
        const ay = s.ay_g !== undefined ? s.ay_g : s.ay / 16384;
        const az = s.az_g !== undefined ? s.az_g : s.az / 16384;

        const gx = s.gx_dps !== undefined ? s.gx_dps : s.gx / 131; // Approximate conversion
        const gy = s.gy_dps !== undefined ? s.gy_dps : s.gy / 131;
        const gz = s.gz_dps !== undefined ? s.gz_dps : s.gz / 131;

        accelMagSum += Math.sqrt(ax * ax + ay * ay + az * az);
        gyroMagSum += Math.sqrt(gx * gx + gy * gy + gz * gz);
    }

    return {
        accelMagMean: accelMagSum / samples.length,
        gyroMagMean: gyroMagSum / samples.length,
    };
}

async function enrichSessionMetadata(session: SessionEntry): Promise<SessionEntry> {
    if (session._metadataLoaded) return session;

    // Fetch session metadata and visualizations in parallel
    const [metadata, visualizations] = await Promise.all([
        fetchSessionMetadata(session.sessionUrl!),
        fetchSessionVisualizations(session.timestamp),
    ]);

    // Apply visualization data first (so windows can be merged with sample-calculated windows)
    if (visualizations) {
        session.composite_image = visualizations.composite_image;
        session.calibration_stages_image = visualizations.calibration_stages_image;
        session.orientation_3d_image = visualizations.orientation_3d_image;
        session.orientation_track_image = visualizations.orientation_track_image;
        session.raw_axes_image = visualizations.raw_axes_image;
        session.trajectory_comparison_images = visualizations.trajectory_comparison_images || {};
        // Store visualization windows for merging
        session.windows = visualizations.windows || [];
        console.log(`[viz-app] Loaded ${session.windows.length} visualization windows for ${session.timestamp}`);
    }

    if (metadata) {
        // Copy metadata fields (except samples which we'll use for window calculation)
        const { samples, ...metadataWithoutSamples } = metadata;
        Object.assign(session, metadataWithoutSamples);

        // Calculate windows from samples, merging with any existing visualization-based windows
        if (samples && samples.length > 0) {
            const sampleRate = metadata.sample_rate || 50;
            session.windows = calculateWindowsFromSamples(samples, sampleRate, session.windows);
            console.log(`[viz-app] Calculated ${session.windows.length} windows from ${samples.length} samples @ ${sampleRate}Hz`);
        }
    } else {
        // Provide defaults if metadata can't be loaded
        session.duration = session.duration || 0;
        session.sample_rate = session.sample_rate || 50;
        session.labels = session.labels || [];
        session.custom_labels = session.custom_labels || [];
    }
    session._metadataLoaded = true;
    return session;
}

// ===== URL Helpers =====

function getImageUrl(imagePath: string | null | undefined): string {
    if (!imagePath) return '';
    // If it's already a full URL, return as-is
    if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
        return imagePath;
    }
    // Otherwise, treat as relative path to visualizations directory
    return `../../../visualizations/${imagePath}`;
}

// ===== Label Functions =====

function getAllLabels(): string[] {
    const labels = new Set<string>();
    sessionsData.forEach(session => {
        // Add custom_labels from session
        if (session.custom_labels) {
            session.custom_labels.forEach(l => labels.add(l));
        }
        // Add labels from label segments
        if (session.labels) {
            session.labels.forEach(segment => {
                if (segment.labels) {
                    if (segment.labels.pose) labels.add(`pose:${segment.labels.pose}`);
                    if (segment.labels.motion) labels.add(`motion:${segment.labels.motion}`);
                    if (segment.labels.calibration && segment.labels.calibration !== 'none') {
                        labels.add(`calibration:${segment.labels.calibration}`);
                    }
                    if (segment.labels.custom) {
                        segment.labels.custom.forEach(l => labels.add(l));
                    }
                }
            });
        }
    });
    return Array.from(labels).sort();
}

function getLabelsForWindow(session: SessionEntry, window: WindowEntry): WindowLabels {
    const windowLabels: WindowLabels = {
        pose: null,
        motion: null,
        calibration: null,
        custom: []
    };

    if (!session.labels || window.time_start === undefined || window.time_end === undefined) {
        return windowLabels;
    }

    const windowStartSample = Math.floor(window.time_start * SAMPLE_RATE);
    const windowEndSample = Math.floor(window.time_end * SAMPLE_RATE);

    session.labels.forEach(segment => {
        const segmentStart = segment.start_sample;
        const segmentEnd = segment.end_sample;

        // Overlap check
        const overlaps = windowStartSample < segmentEnd && windowEndSample > segmentStart;

        if (overlaps && segment.labels) {
            if (segment.labels.pose && !windowLabels.pose) {
                windowLabels.pose = segment.labels.pose;
            }
            if (segment.labels.motion && !windowLabels.motion) {
                windowLabels.motion = segment.labels.motion;
            }
            if (segment.labels.calibration && segment.labels.calibration !== 'none' && !windowLabels.calibration) {
                windowLabels.calibration = segment.labels.calibration;
            }
            if (segment.labels.custom) {
                segment.labels.custom.forEach(l => {
                    if (!windowLabels.custom.includes(l)) {
                        windowLabels.custom.push(l);
                    }
                });
            }
        }
    });

    return windowLabels;
}

function formatWindowLabels(windowLabels: WindowLabels): string {
    const chips: string[] = [];
    if (windowLabels.pose) {
        chips.push(`<span class="chip" style="background: var(--accent);">${windowLabels.pose}</span>`);
    }
    if (windowLabels.motion && windowLabels.motion !== 'static') {
        chips.push(`<span class="chip" style="background: var(--warning);">${windowLabels.motion}</span>`);
    }
    if (windowLabels.calibration) {
        chips.push(`<span class="chip" style="background: var(--success);">${windowLabels.calibration}</span>`);
    }
    windowLabels.custom.forEach(l => {
        chips.push(`<span class="chip">${l}</span>`);
    });
    return chips.join('');
}

function sessionHasLabel(session: SessionEntry, label: string): boolean {
    if (label === 'all') return true;

    if (session.custom_labels && session.custom_labels.includes(label)) {
        return true;
    }

    if (session.labels) {
        for (const segment of session.labels) {
            if (segment.labels) {
                if (label.startsWith('pose:') && segment.labels.pose === label.replace('pose:', '')) {
                    return true;
                }
                if (label.startsWith('motion:') && segment.labels.motion === label.replace('motion:', '')) {
                    return true;
                }
                if (label.startsWith('calibration:') && segment.labels.calibration === label.replace('calibration:', '')) {
                    return true;
                }
                if (segment.labels.custom && segment.labels.custom.includes(label)) {
                    return true;
                }
            }
        }
    }

    return false;
}

// ===== Rendering Functions =====

function renderWindows(session: SessionEntry): string {
    return (session.windows || []).map(w => {
        const windowLabels = getLabelsForWindow(session, w);
        const timeRange = (w.time_start !== undefined && w.time_end !== undefined)
            ? `${w.time_start.toFixed(2)}s - ${w.time_end.toFixed(2)}s`
            : '';
        const statsText = (w.accel_mag_mean !== undefined && w.gyro_mag_mean !== undefined)
            ? `Accel: ${w.accel_mag_mean.toFixed(2)}g | Gyro: ${w.gyro_mag_mean.toFixed(1)}°/s`
            : '';
        const sampleCountText = w.sample_count ? `${w.sample_count} samples` : '';

        let html = `
        <div style="margin-bottom: var(--space-xl); padding: var(--space-lg); background: var(--bg-elevated); border: 1px solid var(--border);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-md);">
                <h4 style="margin: 0; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Window ${w.window_num}${timeRange ? ` | ${timeRange}` : ''}${sampleCountText ? ` | ${sampleCountText}` : ''}
                    <span class="chips-container">
                        ${formatWindowLabels(windowLabels)}
                    </span>
                </h4>
                ${statsText ? `<div style="font-size: 0.625rem; color: var(--fg-muted); text-transform: uppercase;">${statsText}</div>` : ''}
            </div>
            <div class="trajectory-grid">`;

        if (viewSettings.showWinComposite && w.filepath) {
            html += `<div class="trajectory-card" data-type="composite" onclick="window.vizApp.openModal('${getImageUrl(w.filepath)}')">
                <img src="${getImageUrl(w.filepath)}" class="trajectory-image" alt="Composite">
                <div class="trajectory-label">Composite</div>
            </div>`;
        }
        if (w.images?.timeseries_accel && viewSettings.showWinAccelTime) {
            html += `<div class="trajectory-card" data-type="accel-time" onclick="window.vizApp.openModal('${getImageUrl(w.images.timeseries_accel)}')">
                <img src="${getImageUrl(w.images.timeseries_accel)}" class="trajectory-image" alt="Accel Time">
                <div class="trajectory-label">Accel Time</div>
            </div>`;
        }
        if (w.images?.timeseries_gyro && viewSettings.showWinGyroTime) {
            html += `<div class="trajectory-card" data-type="gyro-time" onclick="window.vizApp.openModal('${getImageUrl(w.images.timeseries_gyro)}')">
                <img src="${getImageUrl(w.images.timeseries_gyro)}" class="trajectory-image" alt="Gyro Time">
                <div class="trajectory-label">Gyro Time</div>
            </div>`;
        }
        if (w.images?.timeseries_mag && viewSettings.showWinMagTime) {
            html += `<div class="trajectory-card" data-type="mag-time" onclick="window.vizApp.openModal('${getImageUrl(w.images.timeseries_mag)}')">
                <img src="${getImageUrl(w.images.timeseries_mag)}" class="trajectory-image" alt="Mag Time">
                <div class="trajectory-label">Mag Time</div>
            </div>`;
        }
        if (w.images?.trajectory_accel_3d && viewSettings.showWinAccel3d) {
            html += `<div class="trajectory-card" data-type="accel-3d" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_accel_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_accel_3d)}" class="trajectory-image" alt="Accel 3D">
                <div class="trajectory-label">Accel 3D</div>
            </div>`;
        }
        if (w.images?.trajectory_gyro_3d && viewSettings.showWinGyro3d) {
            html += `<div class="trajectory-card" data-type="gyro-3d" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_gyro_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_gyro_3d)}" class="trajectory-image" alt="Gyro 3D">
                <div class="trajectory-label">Gyro 3D</div>
            </div>`;
        }
        if (w.images?.trajectory_mag_3d && viewSettings.showWinMag3d) {
            html += `<div class="trajectory-card" data-type="mag-3d" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_mag_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_mag_3d)}" class="trajectory-image" alt="Mag 3D">
                <div class="trajectory-label">Mag 3D</div>
            </div>`;
        }
        if (w.images?.trajectory_combined_3d && viewSettings.showWinCombined3d) {
            html += `<div class="trajectory-card" data-type="combined-3d" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_combined_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_combined_3d)}" class="trajectory-image" alt="Combined 3D">
                <div class="trajectory-label">Combined 3D</div>
            </div>`;
        }
        if (w.images?.signature && viewSettings.showWinSignature) {
            html += `<div class="trajectory-card" data-type="signature" onclick="window.vizApp.openModal('${getImageUrl(w.images.signature)}')">
                <img src="${getImageUrl(w.images.signature)}" class="trajectory-image" alt="Signature">
                <div class="trajectory-label">Signature</div>
            </div>`;
        }
        if (w.images?.stats && viewSettings.showWinStats) {
            html += `<div class="trajectory-card" data-type="stats" onclick="window.vizApp.openModal('${getImageUrl(w.images.stats)}')">
                <img src="${getImageUrl(w.images.stats)}" class="trajectory-image" alt="Stats">
                <div class="trajectory-label">Stats</div>
            </div>`;
        }
        if (w.trajectory_images?.raw && viewSettings.showWinTrajRaw) {
            html += `<div class="trajectory-card" data-type="traj-raw" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.raw)}')">
                <img src="${getImageUrl(w.trajectory_images.raw)}" class="trajectory-image" alt="Traj Raw">
                <div class="trajectory-label">Traj Raw</div>
            </div>`;
        }
        if (w.trajectory_images?.iron && viewSettings.showWinTrajIron) {
            html += `<div class="trajectory-card" data-type="traj-iron" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.iron)}')">
                <img src="${getImageUrl(w.trajectory_images.iron)}" class="trajectory-image" alt="Traj Iron">
                <div class="trajectory-label">Traj Iron</div>
            </div>`;
        }
        if (w.trajectory_images?.fused && viewSettings.showWinTrajFused) {
            html += `<div class="trajectory-card" data-type="traj-fused" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.fused)}')">
                <img src="${getImageUrl(w.trajectory_images.fused)}" class="trajectory-image" alt="Traj Residual">
                <div class="trajectory-label">Traj Residual</div>
            </div>`;
        }
        if (w.trajectory_images?.filtered && viewSettings.showWinTrajFiltered) {
            html += `<div class="trajectory-card" data-type="traj-filtered" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.filtered)}')">
                <img src="${getImageUrl(w.trajectory_images.filtered)}" class="trajectory-image" alt="Traj Filtered">
                <div class="trajectory-label">Traj Filtered</div>
            </div>`;
        }
        if (w.trajectory_images?.combined && viewSettings.showWinTrajCombined) {
            html += `<div class="trajectory-card" data-type="traj-combined" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.combined)}')">
                <img src="${getImageUrl(w.trajectory_images.combined)}" class="trajectory-image" alt="Traj Combined">
                <div class="trajectory-label">Traj Combined</div>
            </div>`;
        }
        if (w.trajectory_images?.statistics && viewSettings.showWinTrajStats) {
            html += `<div class="trajectory-card" data-type="traj-stats" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.statistics)}')">
                <img src="${getImageUrl(w.trajectory_images.statistics)}" class="trajectory-image" alt="Traj Stats">
                <div class="trajectory-label">Traj Stats</div>
            </div>`;
        }

        // Show placeholder if no visualizations exist for this window
        if (!w.has_visualizations && !w.filepath) {
            html += `<div style="grid-column: 1 / -1; padding: var(--space-lg); text-align: center; color: var(--fg-muted); font-size: 0.75rem; border: 1px dashed var(--border); border-radius: 4px;">
                <div style="margin-bottom: var(--space-xs);">📊 No visualizations generated</div>
                <div style="font-size: 0.625rem;">Run <code>python -m ml.visualize --session ${w.window_num > 0 ? 'TIMESTAMP' : ''}</code> to generate</div>
            </div>`;
        }

        html += `</div></div>`;
        return html;
    }).join('');
}

function renderSessionContent(session: SessionEntry): string {
    let html = '';
    console.log(`Session content`, session);

    if (session.composite_image) {
        html += `
        <div class="image-section composite-section ${viewSettings.showComposite ? '' : 'hidden'}">
            <h3 class="section-title">Composite Session View</h3>
            <img src="${getImageUrl(session.composite_image)}" class="composite-image" onclick="window.vizApp.openModal(this.src)" alt="Composite view">
        </div>`;
    }

    if (session.calibration_stages_image) {
        html += `
        <div class="image-section calibration-section ${viewSettings.showCalibration ? '' : 'hidden'}">
            <h3 class="section-title">Magnetometer Calibration Stages</h3>
            <img src="${getImageUrl(session.calibration_stages_image)}" class="composite-image" onclick="window.vizApp.openModal(this.src)" alt="Calibration stages comparison">
        </div>`;
    }

    if (session.trajectory_comparison_images && Object.keys(session.trajectory_comparison_images).length > 0) {
        html += `
        <div class="image-section trajectory-section">
            <h3 class="section-title">Session-Level 3D Trajectory Comparison</h3>
            <div class="trajectory-grid">
                ${session.trajectory_comparison_images.raw && viewSettings.showWinTrajRaw ? `
                <div class="trajectory-card" data-type="raw" onclick="window.vizApp.openModal('${getImageUrl(session.trajectory_comparison_images.raw)}')">
                    <img src="${getImageUrl(session.trajectory_comparison_images.raw)}" class="trajectory-image" alt="Raw trajectory">
                    <div class="trajectory-label">Raw</div>
                </div>` : ''}
                ${session.trajectory_comparison_images.iron && viewSettings.showWinTrajIron ? `
                <div class="trajectory-card" data-type="iron" onclick="window.vizApp.openModal('${getImageUrl(session.trajectory_comparison_images.iron)}')">
                    <img src="${getImageUrl(session.trajectory_comparison_images.iron)}" class="trajectory-image" alt="Iron corrected trajectory">
                    <div class="trajectory-label">Iron Corrected</div>
                </div>` : ''}
                ${session.trajectory_comparison_images.fused && viewSettings.showWinTrajFused ? `
                <div class="trajectory-card" data-type="fused" onclick="window.vizApp.openModal('${getImageUrl(session.trajectory_comparison_images.fused)}')">
                    <img src="${getImageUrl(session.trajectory_comparison_images.fused)}" class="trajectory-image" alt="Residual trajectory">
                    <div class="trajectory-label">Residual</div>
                </div>` : ''}
                ${session.trajectory_comparison_images.filtered && viewSettings.showWinTrajFiltered ? `
                <div class="trajectory-card" data-type="filtered" onclick="window.vizApp.openModal('${getImageUrl(session.trajectory_comparison_images.filtered)}')">
                    <img src="${getImageUrl(session.trajectory_comparison_images.filtered)}" class="trajectory-image" alt="Filtered trajectory">
                    <div class="trajectory-label">Filtered</div>
                </div>` : ''}
                ${session.trajectory_comparison_images.combined && viewSettings.showWinTrajCombined ? `
                <div class="trajectory-card" data-type="combined" onclick="window.vizApp.openModal('${getImageUrl(session.trajectory_comparison_images.combined)}')">
                    <img src="${getImageUrl(session.trajectory_comparison_images.combined)}" class="trajectory-image" alt="Combined overlay">
                    <div class="trajectory-label">Combined Overlay</div>
                </div>` : ''}
                ${session.trajectory_comparison_images.statistics && viewSettings.showWinTrajStats ? `
                <div class="trajectory-card" data-type="statistics" onclick="window.vizApp.openModal('${getImageUrl(session.trajectory_comparison_images.statistics)}')">
                    <img src="${getImageUrl(session.trajectory_comparison_images.statistics)}" class="trajectory-image" alt="Statistics">
                    <div class="trajectory-label">Statistics</div>
                </div>` : ''}
            </div>
        </div>`;
    }

    if (session.windows && session.windows.length > 0) {
        html += `
        <div class="image-section windows-section ${viewSettings.showWindows ? '' : 'hidden'}">
            <h3 class="section-title">Per-Window Analysis (${session.windows.length} windows)</h3>
            ${renderWindows(session)}
        </div>`;
    }

    const rawImages = [
        session.orientation_3d_image,
        session.orientation_track_image,
        session.raw_axes_image
    ].filter(Boolean);

    if (rawImages.length > 0) {
        html += `
        <div class="image-section raw-section ${viewSettings.showRaw ? '' : 'hidden'}">
            <h3 class="section-title">Raw Axis & Orientation Views</h3>
            <div class="raw-images">
                ${rawImages.map(img => `
                    <img src="${getImageUrl(img)}" class="raw-image" onclick="window.vizApp.openModal(this.src)" alt="Raw axis view">
                `).join('')}
            </div>
        </div>`;
    }

    return html;
}

function renderSessionCard(session: SessionEntry, idx: number, card: HTMLElement): void {
    const header = card.querySelector('.session-header') as HTMLElement;
    const content = card.querySelector('.session-content') as HTMLElement;

    header.innerHTML = `
        <div>
            <div class="session-title">${session.filename}</div>
            <div class="session-info">
                ${formatTimestamp(session.timestamp.replaceAll('_', ':'))} |
                ${(session.duration || 0).toFixed(1)}s @ ${session.sample_rate || 50}Hz |
                ${session.windows.length} windows
                ${session.device ? ` | ${session.device}` : ''}
                ${session.firmware_version ? ` (fw:${session.firmware_version})` : ''}
                <span class="chips-container">
                    ${session.session_type && session.session_type !== 'recording' ? `<span class="chip" style="background: var(--warning);">${session.session_type}</span>` : ''}
                    ${session.hand && session.hand !== 'unknown' ? `<span class="chip">${session.hand} hand</span>` : ''}
                    ${session.magnet_type && session.magnet_type !== 'unknown' && session.magnet_type !== 'none' ? `<span class="chip" style="background: var(--success);">${session.magnet_type}</span>` : ''}
                    ${(session.custom_labels || []).map(l => `<span class="chip">${l}</span>`).join('')}
                </span>
            </div>
            ${session.notes ? `<div class="session-notes" style="font-size: 0.75rem; color: var(--fg-muted); margin-top: var(--space-xs); font-style: italic;">📝 ${session.notes}</div>` : ''}
        </div>
        <div class="expand-icon">▼</div>
    `;
    header.onclick = () => toggleSession(idx);

    content.innerHTML = renderSessionContent(session);
}

function renderSessions(): void {
    const container = getEl('session-list');
    if (!container) return;

    if (filteredSessions.length === 0) {
        container.innerHTML = '<div class="no-sessions">No sessions found matching your criteria</div>';
        return;
    }

    container.innerHTML = filteredSessions.map((session, idx) => {
        const hasMetadata = session._metadataLoaded;
        const durationText = hasMetadata ? `${(session.duration || 0).toFixed(1)}s @ ${session.sample_rate || 50}Hz | ` : '';
        const sizeText = !hasMetadata && session.size ? `${(session.size / 1024).toFixed(0)}KB | ` : '';

        return `
        <div class="session-card ${defaultExpanded ? 'expanded' : ''}" id="session-${idx}">
            <div class="session-header" onclick="window.vizApp.toggleSession(${idx})">
                <div>
                    <div class="session-title">${session.filename}</div>
                    <div class="session-info">
                        ${formatTimestamp(session.timestamp.replaceAll('_', ':'))} |
                        ${durationText}${sizeText}${session.windows.length} windows
                        ${hasMetadata && session.device ? ` | ${session.device}` : ''}
                        ${hasMetadata && session.firmware_version ? ` (fw:${session.firmware_version})` : ''}
                        <span class="chips-container">
                            ${hasMetadata && session.session_type && session.session_type !== 'recording' ? `<span class="chip" style="background: var(--warning);">${session.session_type}</span>` : ''}
                            ${hasMetadata && session.hand && session.hand !== 'unknown' ? `<span class="chip">${session.hand} hand</span>` : ''}
                            ${hasMetadata && session.magnet_type && session.magnet_type !== 'unknown' && session.magnet_type !== 'none' ? `<span class="chip" style="background: var(--success);">${session.magnet_type}</span>` : ''}
                            ${hasMetadata ? (session.custom_labels?.map(l => `<span class="chip">${l}</span>`).join('') || '') : ''}
                        </span>
                    </div>
                    ${hasMetadata && session.notes ? `<div class="session-notes" style="font-size: 0.75rem; color: var(--fg-muted); margin-top: var(--space-xs); font-style: italic;">📝 ${session.notes}</div>` : ''}
                </div>
                <div class="expand-icon">▼</div>
            </div>
            <div class="session-content">
                ${hasMetadata ? renderSessionContent(session) : '<div style="padding: var(--space-lg); text-align: center; color: var(--fg-muted);">Click to load session details...</div>'}
            </div>
        </div>
    `}).join('');
}

// ===== UI Functions =====

function populateLabelFilter(): void {
    const select = getEl('label-filter') as HTMLSelectElement;
    if (!select) return;
    const labels = getAllLabels();
    labels.forEach(label => {
        const option = document.createElement('option');
        option.value = label;
        option.textContent = label;
        select.appendChild(option);
    });
}

function applyFilters(): void {
    const searchBox = getEl('search-box') as HTMLInputElement;
    const query = searchBox?.value.toLowerCase() || '';
    filteredSessions = sessionsData.filter(s => {
        const matchesSearch = s.filename.toLowerCase().includes(query) ||
            s.timestamp.toLowerCase().includes(query);
        const matchesLabel = sessionHasLabel(s, selectedLabel);
        return matchesSearch && matchesLabel;
    });
    updateStats();
    renderSessions();
}

function updateStats(): void {
    const totalSessionsEl = getEl('total-sessions');
    const totalWindowsEl = getEl('total-windows');
    const totalDurationEl = getEl('total-duration');

    if (totalSessionsEl) totalSessionsEl.textContent = String(filteredSessions.length);

    const totalWindows = filteredSessions.reduce((sum, s) => sum + (s.windows?.length || 0), 0);
    if (totalWindowsEl) totalWindowsEl.textContent = String(totalWindows);

    const totalDuration = filteredSessions.reduce((sum, s) => sum + (s.duration || 0), 0);
    if (totalDurationEl) totalDurationEl.textContent = totalDuration.toFixed(1);
}

function updateViewSettings(): void {
    viewSettings.showComposite = (getEl('show-composite') as HTMLInputElement)?.checked ?? true;
    viewSettings.showCalibration = (getEl('show-calibration') as HTMLInputElement)?.checked ?? true;
    viewSettings.showWindows = (getEl('show-windows') as HTMLInputElement)?.checked ?? true;
    viewSettings.showRaw = (getEl('show-raw') as HTMLInputElement)?.checked ?? true;

    viewSettings.showWinComposite = (getEl('show-win-composite') as HTMLInputElement)?.checked ?? true;
    viewSettings.showWinAccelTime = (getEl('show-win-accel-time') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinGyroTime = (getEl('show-win-gyro-time') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinMagTime = (getEl('show-win-mag-time') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinAccel3d = (getEl('show-win-accel-3d') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinGyro3d = (getEl('show-win-gyro-3d') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinMag3d = (getEl('show-win-mag-3d') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinCombined3d = (getEl('show-win-combined-3d') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinSignature = (getEl('show-win-signature') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinStats = (getEl('show-win-stats') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinTrajRaw = (getEl('show-win-traj-raw') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinTrajIron = (getEl('show-win-traj-iron') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinTrajFused = (getEl('show-win-traj-fused') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinTrajFiltered = (getEl('show-win-traj-filtered') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinTrajCombined = (getEl('show-win-traj-combined') as HTMLInputElement)?.checked ?? false;
    viewSettings.showWinTrajStats = (getEl('show-win-traj-stats') as HTMLInputElement)?.checked ?? false;

    renderSessions();
}

function formatTimestamp(timestamp: string): string {
    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch {
        return timestamp;
    }
}

// ===== Public Functions (exposed to window) =====

async function toggleSession(idx: number): Promise<void> {
    const card = getEl(`session-${idx}`);
    if (!card) return;

    const session = filteredSessions[idx];

    if (!card.classList.contains('expanded') && !session._metadataLoaded && session.sessionUrl) {
        const content = card.querySelector('.session-content') as HTMLElement;
        content.innerHTML = '<div style="padding: var(--space-lg); text-align: center; color: var(--fg-muted);">Loading session data...</div>';
        card.classList.add('expanded');

        await enrichSessionMetadata(session);
        renderSessionCard(session, idx, card);
    } else {
        card.classList.toggle('expanded');
    }
}

function expandAll(): void {
    document.querySelectorAll('.session-card').forEach(card => {
        card.classList.add('expanded');
    });
    defaultExpanded = true;
}

function collapseAll(): void {
    document.querySelectorAll('.session-card').forEach(card => {
        card.classList.remove('expanded');
    });
    defaultExpanded = false;
}

function openModal(imageSrc: string): void {
    const modal = getEl('image-modal');
    const modalImg = getEl('modal-image') as HTMLImageElement;
    if (modal && modalImg) {
        modal.classList.add('active');
        modalImg.src = getImageUrl(imageSrc);
    }
}

function closeModal(): void {
    const modal = getEl('image-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// ===== Event Listeners =====

function setupEventListeners(): void {
    getEl('search-box')?.addEventListener('input', applyFilters);

    getEl('label-filter')?.addEventListener('change', (e) => {
        selectedLabel = (e.target as HTMLSelectElement).value;
        applyFilters();
    });

    getEl('sort-select')?.addEventListener('change', (e) => {
        const sortBy = (e.target as HTMLSelectElement).value;

        filteredSessions.sort((a, b) => {
            switch(sortBy) {
                case 'timestamp-desc':
                    return b.timestamp.localeCompare(a.timestamp);
                case 'timestamp-asc':
                    return a.timestamp.localeCompare(b.timestamp);
                case 'duration-desc':
                    return (b.duration || 0) - (a.duration || 0);
                case 'duration-asc':
                    return (a.duration || 0) - (b.duration || 0);
                default:
                    return 0;
            }
        });

        renderSessions();
    });

    // Section visibility checkboxes
    const checkboxIds = [
        'show-composite', 'show-calibration', 'show-windows', 'show-raw',
        'show-win-composite', 'show-win-accel-time', 'show-win-gyro-time', 'show-win-mag-time',
        'show-win-accel-3d', 'show-win-gyro-3d', 'show-win-mag-3d', 'show-win-combined-3d',
        'show-win-signature', 'show-win-stats',
        'show-win-traj-raw', 'show-win-traj-iron', 'show-win-traj-fused',
        'show-win-traj-filtered', 'show-win-traj-combined', 'show-win-traj-stats'
    ];

    checkboxIds.forEach(id => {
        getEl(id)?.addEventListener('change', updateViewSettings);
    });

    // Modal close on click outside
    getEl('image-modal')?.addEventListener('click', (e) => {
        if ((e.target as HTMLElement).id === 'image-modal') {
            closeModal();
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });

    // Button event listeners
    getEl('expand-all-btn')?.addEventListener('click', expandAll);
    getEl('collapse-all-btn')?.addEventListener('click', collapseAll);

    // Modal close button
    const closeBtn = document.querySelector('.modal-close');
    closeBtn?.addEventListener('click', closeModal);
}

// ===== Initialization =====

async function initializePage(): Promise<void> {
    const container = getEl('session-list');
    if (!container) return;

    container.innerHTML = '<div class="no-sessions">Loading session data...</div>';

    try {
        sessionsData = await fetchExplorerData();
        filteredSessions = [...sessionsData];
        dataLoaded = true;

        populateLabelFilter();
        updateStats();
        renderSessions();
        setupEventListeners();
    } catch (error) {
        loadError = error as Error;
        container.innerHTML = `<div class="no-sessions" style="color: var(--error);">
            Failed to load session data: ${(error as Error).message}<br><br>
            <button class="btn" onclick="window.vizApp.initializePage()">Retry</button>
        </div>`;
    }
}

// ===== Export to Window =====

// Expose functions for inline onclick handlers
interface VizApp {
    toggleSession: typeof toggleSession;
    expandAll: typeof expandAll;
    collapseAll: typeof collapseAll;
    openModal: typeof openModal;
    closeModal: typeof closeModal;
    initializePage: typeof initializePage;
}

// Extend window interface using a more flexible type assertion
const windowWithVizApp = window as unknown as Window & { vizApp: VizApp };
windowWithVizApp.vizApp = {
    toggleSession,
    expandAll,
    collapseAll,
    openModal,
    closeModal,
    initializePage
};

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePage);
} else {
    initializePage();
}

console.log('[viz-app] Visualization explorer initialized');

// Export for module context
export {};
