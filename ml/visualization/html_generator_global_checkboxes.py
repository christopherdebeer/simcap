"""
HTML Generator with Global Checkboxes

Generates HTML explorer with global checkboxes controlling all window views.
"""

import json
from pathlib import Path
from typing import List, Dict


def generate_html_with_global_checkboxes(sessions_data: List[Dict], output_dir: Path) -> Path:
    """Generate HTML explorer with global view checkboxes."""
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIMCAP Data Explorer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .header {
            background: #1a1a1a;
            border-bottom: 2px solid #333;
            padding: 1.5rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #fff;
            margin-bottom: 0.5rem;
        }
        
        .filter-bar {
            background: #151515;
            border: 1px solid #333;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 4px;
        }
        
        .filter-section {
            margin-bottom: 0.75rem;
        }
        
        .filter-section:last-child {
            margin-bottom: 0;
        }
        
        .filter-label {
            display: block;
            font-size: 0.85rem;
            color: #999;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        
        .checkbox-item input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }
        
        .checkbox-item label {
            font-size: 0.9rem;
            cursor: pointer;
            user-select: none;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .session {
            background: #1a1a1a;
            border: 1px solid #333;
            margin-bottom: 2rem;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .session-header {
            background: #222;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #333;
        }
        
        .session-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 0.3rem;
        }
        
        .session-meta {
            font-size: 0.85rem;
            color: #888;
        }
        
        .session-images {
            padding: 1.5rem;
        }
        
        .composite-image {
            margin-bottom: 2rem;
        }
        
        .composite-image img {
            width: 100%;
            height: auto;
            border: 1px solid #333;
            border-radius: 2px;
        }
        
        .windows-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        
        .window-card {
            background: #151515;
            border: 1px solid #333;
            border-radius: 4px;
            overflow: hidden;
            transition: border-color 0.2s;
        }
        
        .window-card:hover {
            border-color: #555;
        }
        
        .window-header {
            padding: 0.75rem 1rem;
            background: #1a1a1a;
            border-bottom: 1px solid #333;
        }
        
        .window-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #fff;
        }
        
        .window-time {
            font-size: 0.8rem;
            color: #888;
            margin-top: 0.2rem;
        }
        
        .window-images {
            padding: 1rem;
        }
        
        .window-image {
            margin-bottom: 1rem;
        }
        
        .window-image:last-child {
            margin-bottom: 0;
        }
        
        .window-image img {
            width: 100%;
            height: auto;
            border: 1px solid #2a2a2a;
            border-radius: 2px;
            cursor: pointer;
        }
        
        .image-label {
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
        }
        
        .modal.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }
        
        .modal-close {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>SIMCAP DATA EXPLORER</h1>
        <div class="filter-bar">
            <div class="filter-section">
                <span class="filter-label">Window Views</span>
                <div class="checkbox-group" id="viewCheckboxes">
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_composite" value="composite" checked>
                        <label for="view_composite">Composite</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_timeseries_accel" value="timeseries_accel">
                        <label for="view_timeseries_accel">Accel TS</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_timeseries_gyro" value="timeseries_gyro">
                        <label for="view_timeseries_gyro">Gyro TS</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_timeseries_mag" value="timeseries_mag">
                        <label for="view_timeseries_mag">Mag TS</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_trajectory_accel_3d" value="trajectory_accel_3d">
                        <label for="view_trajectory_accel_3d">Accel 3D</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_trajectory_gyro_3d" value="trajectory_gyro_3d">
                        <label for="view_trajectory_gyro_3d">Gyro 3D</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_trajectory_mag_3d" value="trajectory_mag_3d">
                        <label for="view_trajectory_mag_3d">Mag 3D</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_trajectory_combined_3d" value="trajectory_combined_3d">
                        <label for="view_trajectory_combined_3d">Combined 3D</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_traj_raw" value="traj_raw">
                        <label for="view_traj_raw">Traj Raw</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_traj_iron" value="traj_iron">
                        <label for="view_traj_iron">Traj Iron</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_traj_fused" value="traj_fused">
                        <label for="view_traj_fused">Traj Fused</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_traj_filtered" value="traj_filtered">
                        <label for="view_traj_filtered">Traj Filtered</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_traj_combined" value="traj_combined">
                        <label for="view_traj_combined">Traj Combined</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_traj_stats" value="traj_stats">
                        <label for="view_traj_stats">Traj Stats</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_signature" value="signature">
                        <label for="view_signature">Signature</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="view_stats" value="stats">
                        <label for="view_stats">Stats</label>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container">
"""

    # Add sessions
    for session_data in sessions_data:
        html_content += f"""
        <div class="session">
            <div class="session-header">
                <div class="session-title">{session_data['filename']}</div>
                <div class="session-meta">
                    Duration: {session_data['duration']:.1f}s | 
                    Windows: {len(session_data['windows'])}
                </div>
            </div>
            <div class="session-images">
                <div class="composite-image">
                    <img src="{session_data['composite_image']}" alt="Composite" onclick="openModal(this.src)">
                </div>
                
                <div class="windows-grid">
"""

        # Add window cards
        for window in session_data['windows']:
            html_content += f"""
                    <div class="window-card">
                        <div class="window-header">
                            <div class="window-title">Window {window['window_num']}</div>
                            <div class="window-time">{window['time_start']:.1f}s - {window['time_end']:.1f}s</div>
                        </div>
                        <div class="window-images">
"""
            
            # Add composite window image (always visible when checked)
            html_content += f"""
                            <div class="window-image" data-view="composite">
                                <div class="image-label">Composite</div>
                                <img src="{window['filepath']}" alt="Window {window['window_num']}" onclick="openModal(this.src)">
                            </div>
"""
            
            # Add individual images based on availability
            view_labels = {
                'timeseries_accel': 'Accel TS',
                'timeseries_gyro': 'Gyro TS',
                'timeseries_mag': 'Mag TS',
                'trajectory_accel_3d': 'Accel 3D',
                'trajectory_gyro_3d': 'Gyro 3D',
                'trajectory_mag_3d': 'Mag 3D',
                'trajectory_combined_3d': 'Combined 3D',
                'traj_raw': 'Traj Raw',
                'traj_iron': 'Traj Iron',
                'traj_fused': 'Traj Fused',
                'traj_filtered': 'Traj Filtered',
                'traj_combined': 'Traj Combined',
                'traj_stats': 'Traj Stats',
                'signature': 'Signature',
                'stats': 'Stats',
            }
            
            for view_key, view_label in view_labels.items():
                if view_key in window.get('images', {}):
                    html_content += f"""
                            <div class="window-image" data-view="{view_key}" style="display: none;">
                                <div class="image-label">{view_label}</div>
                                <img src="{window['images'][view_key]}" alt="{view_label}" onclick="openModal(this.src)">
                            </div>
"""
            
            html_content += """
                        </div>
                    </div>
"""

        html_content += """
                </div>
            </div>
        </div>
"""

    # Close HTML and add JavaScript
    html_content += """
    </div>
    
    <div class="modal" id="imageModal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        const checkboxes = document.querySelectorAll('#viewCheckboxes input[type="checkbox"]');
        
        function updateVisibleViews() {
            const checkedViews = Array.from(checkboxes)
                .filter(cb => cb.checked)
                .map(cb => cb.value);
            
            document.querySelectorAll('.window-image').forEach(img => {
                const view = img.getAttribute('data-view');
                img.style.display = checkedViews.includes(view) ? 'block' : 'none';
            });
        }
        
        checkboxes.forEach(cb => {
            cb.addEventListener('change', updateVisibleViews);
        });
        
        function openModal(src) {
            document.getElementById('imageModal').classList.add('active');
            document.getElementById('modalImage').src = src;
        }
        
        function closeModal() {
            document.getElementById('imageModal').classList.remove('active');
        }
        
        document.getElementById('imageModal').addEventListener('click', function(e) {
            if (e.target.id === 'imageModal') closeModal();
        });
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeModal();
        });
        
        updateVisibleViews();
    </script>
</body>
</html>
"""

    output_file = output_dir / 'index.html'
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Generated HTML explorer with global checkboxes: {output_file}")
    return output_file
