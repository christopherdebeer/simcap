/**
 * SIMCAP Data Visualization Explorer Application
 * Interactive viewer for visualization manifests - NO raw session data
 * 
 * This app ONLY displays sessions that have visualization manifests.
 * It does NOT fetch or process raw session JSON files.
 */

import { ApiClient, type VisualizationManifest, type VisualizationSessionSummary } from '@api/client';

// Create API client instance
const apiClient = new ApiClient();

// Debug flag for verbose logging
const DEBUG = true;
function debug(...args: unknown[]): void {
    if (DEBUG) console.log('[viz-app]', ...args);
}

// ===== Type Definitions =====

interface WindowEntry {
    window_num: number;
    filepath?: string;
    time_start?: number;
    time_end?: number;
    sample_count?: number;
    images?: Record<string, string>;
    trajectory_images?: Record<string, string>;
}

interface SessionEntry {
    timestamp: string;
    filename: string;
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
    composite_image: string | null;
    calibration_stages_image: string | null;
    orientation_3d_image?: string | null;
    orientation_track_image?: string | null;
    raw_axes_image?: string | null;
    trajectory_comparison_images: Record<string, string>;
    windows: WindowEntry[];
    _manifestLoaded?: boolean;
    _manifestId?: string;
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

// ===== Utility Functions =====

function formatTimestamp(timestamp: string): string {
    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch {
        return timestamp;
    }
}

function getImageUrl(imagePath: string | null | undefined): string {
    if (!imagePath) return '';
    if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
        return imagePath;
    }
    return `../../../visualizations/${imagePath}`;
}

// ===== API Functions =====

async function fetchVisualizationsList(): Promise<SessionEntry[]> {
    try {
        debug('Fetching visualizations list...');
        const response = await apiClient.listVisualizations();
        debug(`Found ${response.count} sessions with visualizations`);

        return response.sessions.map((summary: VisualizationSessionSummary) => ({
            timestamp: summary.sessionTimestamp,
            filename: summary.filename,
            composite_image: null,
            calibration_stages_image: null,
            orientation_3d_image: null,
            orientation_track_image: null,
            raw_axes_image: null,
            trajectory_comparison_images: {},
            windows: [],
            _manifestLoaded: false,
            _manifestId: summary.latestManifestId,
        }));
    } catch (error) {
        console.error('Failed to fetch visualizations list:', error);
        throw error;
    }
}

async function loadSessionManifest(session: SessionEntry): Promise<SessionEntry> {
    if (session._manifestLoaded) return session;

    try {
        debug(`Loading manifest for session: ${session.timestamp}`);
        const response = await apiClient.getSessionVisualization(session.timestamp);
        
        if (!response.found || !response.session) {
            debug(`No manifest found for ${session.timestamp}`);
            session._manifestLoaded = true;
            return session;
        }

        const viz = response.session;
        const manifest = response.manifest;

        session.composite_image = viz.composite_image;
        session.calibration_stages_image = viz.calibration_stages_image;
        session.orientation_3d_image = viz.orientation_3d_image;
        session.orientation_track_image = viz.orientation_track_image;
        session.raw_axes_image = viz.raw_axes_image;
        session.trajectory_comparison_images = viz.trajectory_comparison_images || {};
        session.windows = (viz.windows || []).map(w => ({
            window_num: w.window_num,
            filepath: w.filepath,
            time_start: w.time_start,
            time_end: w.time_end,
            sample_count: w.sample_count,
            images: w.images || {},
            trajectory_images: w.trajectory_images || {},
        }));

        if (manifest) {
            session.duration = manifest.session.duration;
            session.sample_rate = manifest.session.sample_rate;
            session.sample_count = manifest.session.sample_count;
            session.device = manifest.session.device;
            session.firmware_version = manifest.session.firmware_version ?? null;
            session.session_type = manifest.session.session_type;
            session.hand = manifest.session.hand;
            session.magnet_type = manifest.session.magnet_type;
            session.notes = manifest.session.notes ?? null;
            session.custom_labels = manifest.session.custom_labels || [];
        }

        session._manifestLoaded = true;
        debug(`Loaded manifest for ${session.timestamp}:`, {
            duration: session.duration,
            windowCount: session.windows.length,
            hasComposite: !!session.composite_image,
        });

        return session;
    } catch (error) {
        console.error(`Failed to load manifest for ${session.timestamp}:`, error);
        session._manifestLoaded = true;
        return session;
    }
}

// ===== Rendering Functions =====

function renderWindows(session: SessionEntry): string {
    return (session.windows || []).map(w => {
        const timeRange = (w.time_start !== undefined && w.time_end !== undefined)
            ? `${w.time_start.toFixed(2)}s - ${w.time_end.toFixed(2)}s`
            : '';
        const sampleCountText = w.sample_count ? `${w.sample_count} samples` : '';

        let html = `
        <div style="margin-bottom: var(--space-xl); padding: var(--space-lg); background: var(--bg-elevated); border: 1px solid var(--border);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-md);">
                <h4 style="margin: 0; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Window ${w.window_num}${timeRange ? ` | ${timeRange}` : ''}${sampleCountText ? ` | ${sampleCountText}` : ''}
                </h4>
            </div>
            <div class="trajectory-grid">`;

        if (viewSettings.showWinComposite && w.filepath) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.filepath)}')">
                <img src="${getImageUrl(w.filepath)}" class="trajectory-image" alt="Composite">
                <div class="trajectory-label">Composite</div>
            </div>`;
        }
        if (w.images?.timeseries_accel && viewSettings.showWinAccelTime) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.timeseries_accel)}')">
                <img src="${getImageUrl(w.images.timeseries_accel)}" class="trajectory-image" alt="Accel Time">
                <div class="trajectory-label">Accel Time</div>
            </div>`;
        }
        if (w.images?.timeseries_gyro && viewSettings.showWinGyroTime) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.timeseries_gyro)}')">
                <img src="${getImageUrl(w.images.timeseries_gyro)}" class="trajectory-image" alt="Gyro Time">
                <div class="trajectory-label">Gyro Time</div>
            </div>`;
        }
        if (w.images?.timeseries_mag && viewSettings.showWinMagTime) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.timeseries_mag)}')">
                <img src="${getImageUrl(w.images.timeseries_mag)}" class="trajectory-image" alt="Mag Time">
                <div class="trajectory-label">Mag Time</div>
            </div>`;
        }
        if (w.images?.trajectory_accel_3d && viewSettings.showWinAccel3d) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_accel_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_accel_3d)}" class="trajectory-image" alt="Accel 3D">
                <div class="trajectory-label">Accel 3D</div>
            </div>`;
        }
        if (w.images?.trajectory_gyro_3d && viewSettings.showWinGyro3d) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_gyro_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_gyro_3d)}" class="trajectory-image" alt="Gyro 3D">
                <div class="trajectory-label">Gyro 3D</div>
            </div>`;
        }
        if (w.images?.trajectory_mag_3d && viewSettings.showWinMag3d) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_mag_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_mag_3d)}" class="trajectory-image" alt="Mag 3D">
                <div class="trajectory-label">Mag 3D</div>
            </div>`;
        }
        if (w.images?.trajectory_combined_3d && viewSettings.showWinCombined3d) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.trajectory_combined_3d)}')">
                <img src="${getImageUrl(w.images.trajectory_combined_3d)}" class="trajectory-image" alt="Combined 3D">
                <div class="trajectory-label">Combined 3D</div>
            </div>`;
        }
        if (w.images?.signature && viewSettings.showWinSignature) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.signature)}')">
                <img src="${getImageUrl(w.images.signature)}" class="trajectory-image" alt="Signature">
                <div class="trajectory-label">Signature</div>
            </div>`;
        }
        if (w.images?.stats && viewSettings.showWinStats) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.images.stats)}')">
                <img src="${getImageUrl(w.images.stats)}" class="trajectory-image" alt="Stats">
                <div class="trajectory-label">Stats</div>
            </div>`;
        }
        if (w.trajectory_images?.raw && viewSettings.showWinTrajRaw) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.raw)}')">
                <img src="${getImageUrl(w.trajectory_images.raw)}" class="trajectory-image" alt="Traj Raw">
                <div class="trajectory-label">Traj Raw</div>
            </div>`;
        }
        if (w.trajectory_images?.iron && viewSettings.showWinTrajIron) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.iron)}')">
                <img src="${getImageUrl(w.trajectory_images.iron)}" class="trajectory-image" alt="Traj Iron">
                <div class="trajectory-label">Traj Iron</div>
            </div>`;
        }
        if (w.trajectory_images?.fused && viewSettings.showWinTrajFused) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.fused)}')">
                <img src="${getImageUrl(w.trajectory_images.fused)}" class="trajectory-image" alt="Traj Residual">
                <div class="trajectory-label">Traj Residual</div>
            </div>`;
        }
        if (w.trajectory_images?.filtered && viewSettings.showWinTrajFiltered) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.filtered)}')">
                <img src="${getImageUrl(w.trajectory_images.filtered)}" class="trajectory-image" alt="Traj Filtered">
                <div class="trajectory-label">Traj Filtered</div>
            </div>`;
        }
        if (w.trajectory_images?.combined && viewSettings.showWinTrajCombined) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.combined)}')">
                <img src="${getImageUrl(w.trajectory_images.combined)}" class="trajectory-image" alt="Traj Combined">
                <div class="trajectory-label">Traj Combined</div>
            </div>`;
        }
        if (w.trajectory_images?.statistics && viewSettings.showWinTrajStats) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(w.trajectory_images.statistics)}')">
                <img src="${getImageUrl(w.trajectory_images.statistics)}" class="trajectory-image" alt="Traj Stats">
                <div class="trajectory-label">Traj Stats</div>
            </div>`;
        }

        html += `</div></div>`;
        return html;
    }).join('');
}

function renderSessionContent(session: SessionEntry): string {
    let html = '';

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
            <img src="${getImageUrl(session.calibration_stages_image)}" class="composite-image" onclick="window.vizApp.openModal(this.src)" alt="Calibration stages">
        </div>`;
    }

    if (session.trajectory_comparison_images && Object.keys(session.trajectory_comparison_images).length > 0) {
        html += `
        <div class="image-section trajectory-section">
            <h3 class="section-title">Session-Level 3D Trajectory Comparison</h3>
            <div class="trajectory-grid">`;
        
        const trajImages = session.trajectory_comparison_images;
        if (trajImages.raw && viewSettings.showWinTrajRaw) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(trajImages.raw)}')">
                <img src="${getImageUrl(trajImages.raw)}" class="trajectory-image" alt="Raw">
                <div class="trajectory-label">Raw</div>
            </div>`;
        }
        if (trajImages.iron && viewSettings.showWinTrajIron) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(trajImages.iron)}')">
                <img src="${getImageUrl(trajImages.iron)}" class="trajectory-image" alt="Iron Corrected">
                <div class="trajectory-label">Iron Corrected</div>
            </div>`;
        }
        if (trajImages.fused && viewSettings.showWinTrajFused) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(trajImages.fused)}')">
                <img src="${getImageUrl(trajImages.fused)}" class="trajectory-image" alt="Residual">
                <div class="trajectory-label">Residual</div>
            </div>`;
        }
        if (trajImages.filtered && viewSettings.showWinTrajFiltered) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(trajImages.filtered)}')">
                <img src="${getImageUrl(trajImages.filtered)}" class="trajectory-image" alt="Filtered">
                <div class="trajectory-label">Filtered</div>
            </div>`;
        }
        if (trajImages.combined && viewSettings.showWinTrajCombined) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(trajImages.combined)}')">
                <img src="${getImageUrl(trajImages.combined)}" class="trajectory-image" alt="Combined">
                <div class="trajectory-label">Combined Overlay</div>
            </div>`;
        }
        if (trajImages.statistics && viewSettings.showWinTrajStats) {
            html += `<div class="trajectory-card" onclick="window.vizApp.openModal('${getImageUrl(trajImages.statistics)}')">
                <img src="${getImageUrl(trajImages.statistics)}" class="trajectory-image" alt="Statistics">
                <div class="trajectory-label">Statistics</div>
            </div>`;
        }
        html += `</div></div>`;
    }

    if (session.windows && session.windows.length > 0) {
        html += `
        <div class="image-section windows-section ${viewSettings.showWindows ? '' : 'hidden'}">
            <h3 class="section-title">Per-Window Analysis (${session.windows.length} windows)</h3>
            ${renderWindows(session)}
        </div>`;
    }

    const rawImages = [session.orientation_3d_image, session.orientation_track_image, session.raw_axes_image].filter(Boolean);
    if (rawImages.length > 0) {
        html += `
        <div class="image-section raw-section ${viewSettings.showRaw ? '' : 'hidden'}">
            <h3 class="section-title">Raw Axis & Orientation Views</h3>
            <div class="raw-images">
                ${rawImages.map(img => `<img src="${getImageUrl(img)}" class="raw-image" onclick="window.vizApp.openModal(this.src)" alt="Raw view">`).join('')}
            </div>
        </div>`;
    }

    if (!session.composite_image && session.windows.length === 0) {
        html += `<div style="padding: var(--space-xl); text-align: center; color: var(--fg-muted);">
            <div style="font-size: 1.5rem; margin-bottom: var(--space-md);">📊</div>
            <div>No visualizations available for this session</div>
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
                ${formatTimestamp(session.timestamp)} |
                ${(session.duration || 0).toFixed(1)}s @ ${session.sample_rate || 50}Hz |
                ${session.windows.length} windows
                ${session.device ? ` | ${session.device}` : ''}
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
        container.innerHTML = '<div class="no-sessions">No sessions with visualizations found</div>';
        return;
    }

    container.innerHTML = filteredSessions.map((session, idx) => {
        const hasManifest = session._manifestLoaded;
        const durationText = hasManifest ? `${(session.duration || 0).toFixed(1)}s @ ${session.sample_rate || 50}Hz | ` : '';

        return `
        <div class="session-card ${defaultExpanded ? 'expanded' : ''}" id="session-${idx}">
            <div class="session-header" onclick="window.vizApp.toggleSession(${idx})">
                <div>
                    <div class="session-title">${session.filename}</div>
                    <div class="session-info">
                        ${formatTimestamp(session.timestamp)} |
                        ${durationText}${hasManifest ? session.windows.length : '?'} windows
                        ${hasManifest && session.device ? ` | ${session.device}` : ''}
                        <span class="chips-container">
                            ${hasManifest && session.session_type && session.session_type !== 'recording' ? `<span class="chip" style="background: var(--warning);">${session.session_type}</span>` : ''}
                            ${hasManifest && session.hand && session.hand !== 'unknown' ? `<span class="chip">${session.hand} hand</span>` : ''}
                            ${hasManifest && session.magnet_type && session.magnet_type !== 'unknown' && session.magnet_type !== 'none' ? `<span class="chip" style="background: var(--success);">${session.magnet_type}</span>` : ''}
                            ${hasManifest ? (session.custom_labels?.map(l => `<span class="chip">${l}</span>`).join('') || '') : ''}
                        </span>
                    </div>
                    ${hasManifest && session.notes ? `<div class="session-notes" style="font-size: 0.75rem; color: var(--fg-muted); margin-top: var(--space-xs); font-style: italic;">📝 ${session.notes}</div>` : ''}
                </div>
                <div class="expand-icon">▼</div>
            </div>
            <div class="session-content">
                ${hasManifest ? renderSessionContent(session) : '<div style="padding: var(--space-lg); text-align: center; color: var(--fg-muted);">Click to load visualizations...</div>'}
            </div>
        </div>
    `}).join('');
}

// ===== UI Functions =====

function applyFilters(): void {
    const searchBox = getEl('search-box') as HTMLInputElement;
    const query = searchBox?.value.toLowerCase() || '';
    filteredSessions = sessionsData.filter(s => {
        return s.filename.toLowerCase().includes(query) || s.timestamp.toLowerCase().includes(query);
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

// ===== Public Functions =====

async function toggleSession(idx: number): Promise<void> {
    const card = getEl(`session-${idx}`);
    if (!card) return;

    const session = filteredSessions[idx];

    if (!card.classList.contains('expanded') && !session._manifestLoaded) {
        const content = card.querySelector('.session-content') as HTMLElement;
        content.innerHTML = '<div style="padding: var(--space-lg); text-align: center; color: var(--fg-muted);">Loading visualizations...</div>';
        card.classList.add('expanded');

        await loadSessionManifest(session);
        renderSessionCard(session, idx, card);
    } else {
        card.classList.toggle('expanded');
    }
}

function expandAll(): void {
    document.querySelectorAll('.session-card').forEach(card => card.classList.add('expanded'));
    defaultExpanded = true;
}

function collapseAll(): void {
    document.querySelectorAll('.session-card').forEach(card => card.classList.remove('expanded'));
    defaultExpanded = false;
}

function openModal(imageSrc: string): void {
    const modal = getEl('image-modal');
    const modalImg = getEl('modal-image') as HTMLImageElement;
    if (modal && modalImg) {
        modal.classList.add('active');
        modalImg.src = imageSrc;
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

    getEl('sort-select')?.addEventListener('change', (e) => {
        const sortBy = (e.target as HTMLSelectElement).value;
        filteredSessions.sort((a, b) => {
            switch(sortBy) {
                case 'timestamp-desc': return b.timestamp.localeCompare(a.timestamp);
                case 'timestamp-asc': return a.timestamp.localeCompare(b.timestamp);
                case 'duration-desc': return (b.duration || 0) - (a.duration || 0);
                case 'duration-asc': return (a.duration || 0) - (b.duration || 0);
                default: return 0;
            }
        });
        renderSessions();
    });

    const checkboxIds = [
        'show-composite', 'show-calibration', 'show-windows', 'show-raw',
        'show-win-composite', 'show-win-accel-time', 'show-win-gyro-time', 'show-win-mag-time',
        'show-win-accel-3d', 'show-win-gyro-3d', 'show-win-mag-3d', 'show-win-combined-3d',
        'show-win-signature', 'show-win-stats',
        'show-win-traj-raw', 'show-win-traj-iron', 'show-win-traj-fused',
        'show-win-traj-filtered', 'show-win-traj-combined', 'show-win-traj-stats'
    ];
    checkboxIds.forEach(id => getEl(id)?.addEventListener('change', updateViewSettings));

    getEl('image-modal')?.addEventListener('click', (e) => {
        if ((e.target as HTMLElement).id === 'image-modal') closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    getEl('expand-all-btn')?.addEventListener('click', expandAll);
    getEl('collapse-all-btn')?.addEventListener('click', collapseAll);
    document.querySelector('.modal-close')?.addEventListener('click', closeModal);
}

// ===== Initialization =====

async function initializePage(): Promise<void> {
    const container = getEl('session-list');
    if (!container) return;

    container.innerHTML = '<div class="no-sessions">Loading visualization data...</div>';

    try {
        sessionsData = await fetchVisualizationsList();
        filteredSessions = [...sessionsData];
        dataLoaded = true;

        updateStats();
        renderSessions();
        setupEventListeners();
    } catch (error) {
        loadError = error as Error;
        container.innerHTML = `<div class="no-sessions" style="color: var(--error);">
            Failed to load visualizations: ${(error as Error).message}<br><br>
            <button class="btn" onclick="window.vizApp.initializePage()">Retry</button>
        </div>`;
    }
}

// ===== Export to Window =====

interface VizApp {
    toggleSession: typeof toggleSession;
    expandAll: typeof expandAll;
    collapseAll: typeof collapseAll;
    openModal: typeof openModal;
    closeModal: typeof closeModal;
    initializePage: typeof initializePage;
}

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

console.log('[viz-app] Visualization explorer initialized (manifest-only mode)');

export {};
