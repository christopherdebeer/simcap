#!/usr/bin/env python3
"""
Z-Axis Anomaly Deep Dive

The session shows Z-axis range of 2814µT (expected ~100µT).
This script investigates when and why this occurs.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load session
session_path = Path('data/GAMBIT/2025-12-30T22_46_28.771Z.json')
with open(session_path, 'r') as f:
    data = json.load(f)

samples = data['samples']
n = len(samples)

# Extract magnetometer data
mx = np.array([s.get('mx_ut', s.get('mx', 0)) for s in samples])
my = np.array([s.get('my_ut', s.get('my', 0)) for s in samples])
mz = np.array([s.get('mz_ut', s.get('mz', 0)) for s in samples])
ts = np.array([s.get('timestamp', i*20) for i, s in enumerate(samples)])
ts_sec = (ts - ts[0]) / 1000.0

print("=" * 80)
print("Z-AXIS ANOMALY INVESTIGATION")
print("=" * 80)

print(f"\nTotal samples: {n}")
print(f"Duration: {ts_sec[-1]:.1f} seconds")

# Find anomalous Z values
NORMAL_THRESHOLD = 150  # µT - beyond this is clearly anomalous
anomalous_mask = np.abs(mz) > NORMAL_THRESHOLD
anomalous_indices = np.where(anomalous_mask)[0]

print(f"\n--- Anomalous Z Samples ---")
print(f"Threshold: |mz| > {NORMAL_THRESHOLD} µT")
print(f"Anomalous samples: {len(anomalous_indices)} ({len(anomalous_indices)/n*100:.1f}%)")

if len(anomalous_indices) > 0:
    print(f"\nAnomaly time range:")
    print(f"  First anomaly: sample {anomalous_indices[0]} (t={ts_sec[anomalous_indices[0]]:.1f}s)")
    print(f"  Last anomaly: sample {anomalous_indices[-1]} (t={ts_sec[anomalous_indices[-1]]:.1f}s)")

    # Find contiguous anomaly periods
    gaps = np.diff(anomalous_indices)
    break_points = np.where(gaps > 10)[0]  # More than 10 samples gap = new period

    print(f"\n--- Anomaly Periods ---")
    period_starts = [anomalous_indices[0]] + [anomalous_indices[bp+1] for bp in break_points]
    period_ends = [anomalous_indices[bp] for bp in break_points] + [anomalous_indices[-1]]

    for i, (start, end) in enumerate(zip(period_starts, period_ends)):
        duration = (end - start + 1) / 50  # samples to seconds
        max_z = np.max(np.abs(mz[start:end+1]))
        print(f"  Period {i+1}: samples {start}-{end} (t={ts_sec[start]:.1f}s-{ts_sec[end]:.1f}s), "
              f"duration={duration:.1f}s, max|Z|={max_z:.0f}µT")

# Clean data analysis (excluding anomalies)
clean_mask = ~anomalous_mask
clean_mz = mz[clean_mask]
clean_mx = mx[clean_mask]
clean_my = my[clean_mask]

print(f"\n--- Clean Data Statistics (|mz| <= {NORMAL_THRESHOLD}µT) ---")
print(f"Clean samples: {len(clean_mz)} ({len(clean_mz)/n*100:.1f}%)")
print(f"X range: [{clean_mx.min():.1f}, {clean_mx.max():.1f}] = {clean_mx.max()-clean_mx.min():.1f} µT")
print(f"Y range: [{clean_my.min():.1f}, {clean_my.max():.1f}] = {clean_my.max()-clean_my.min():.1f} µT")
print(f"Z range: [{clean_mz.min():.1f}, {clean_mz.max():.1f}] = {clean_mz.max()-clean_mz.min():.1f} µT")

clean_mag = np.sqrt(clean_mx**2 + clean_my**2 + clean_mz**2)
print(f"\nClean magnitude: mean={clean_mag.mean():.1f} ± {clean_mag.std():.1f} µT")

# Recalculate calibration with clean data
print("\n" + "=" * 80)
print("CALIBRATION WITH CLEAN DATA")
print("=" * 80)

# Min-max on clean data
hard_iron_clean = np.array([
    (clean_mx.max() + clean_mx.min()) / 2,
    (clean_my.max() + clean_my.min()) / 2,
    (clean_mz.max() + clean_mz.min()) / 2
])

ranges_clean = np.array([
    clean_mx.max() - clean_mx.min(),
    clean_my.max() - clean_my.min(),
    clean_mz.max() - clean_mz.min()
])

EARTH_MAG = 50.4
expected_range = 2 * EARTH_MAG
soft_iron_clean = expected_range / ranges_clean

print(f"\nClean data calibration:")
print(f"Hard iron: [{hard_iron_clean[0]:.1f}, {hard_iron_clean[1]:.1f}, {hard_iron_clean[2]:.1f}] µT")
print(f"Ranges: [{ranges_clean[0]:.1f}, {ranges_clean[1]:.1f}, {ranges_clean[2]:.1f}] µT")
print(f"Soft iron scale: [{soft_iron_clean[0]:.3f}, {soft_iron_clean[1]:.3f}, {soft_iron_clean[2]:.3f}]")

sphericity = np.min(ranges_clean) / np.max(ranges_clean)
print(f"Sphericity: {sphericity:.2f}")

# Apply clean calibration
corrected_clean = np.array([
    (clean_mx - hard_iron_clean[0]) * soft_iron_clean[0],
    (clean_my - hard_iron_clean[1]) * soft_iron_clean[1],
    (clean_mz - hard_iron_clean[2]) * soft_iron_clean[2]
])
corr_mag_clean = np.sqrt(corrected_clean[0]**2 + corrected_clean[1]**2 + corrected_clean[2]**2)
print(f"\nCorrected magnitude: {corr_mag_clean.mean():.1f} ± {corr_mag_clean.std():.1f} µT (expected 50.4)")

# Generate visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Z-Axis Anomaly Analysis - Session 2025-12-30', fontsize=14)

# Raw Z over time
ax1 = axes[0, 0]
ax1.plot(ts_sec, mz, 'b-', alpha=0.7, linewidth=0.5)
ax1.axhline(NORMAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
ax1.axhline(-NORMAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
ax1.fill_between(ts_sec, -NORMAL_THRESHOLD, NORMAL_THRESHOLD, alpha=0.1, color='green')
ax1.set_ylabel('Z (µT)')
ax1.set_title('Raw Z-axis over time')
ax1.set_xlabel('Time (s)')
ax1.grid(True, alpha=0.3)

# Magnitude over time
ax2 = axes[0, 1]
mag = np.sqrt(mx**2 + my**2 + mz**2)
ax2.plot(ts_sec, mag, 'b-', alpha=0.7, linewidth=0.5)
ax2.axhline(EARTH_MAG, color='g', linestyle='--', label='Expected 50µT')
ax2.set_ylabel('Magnitude (µT)')
ax2.set_title('Raw magnitude over time')
ax2.set_xlabel('Time (s)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# X,Y,Z comparison
ax3 = axes[1, 0]
ax3.plot(ts_sec, mx, label='X', alpha=0.7, linewidth=0.5)
ax3.plot(ts_sec, my, label='Y', alpha=0.7, linewidth=0.5)
ax3.plot(ts_sec, mz, label='Z', alpha=0.7, linewidth=0.5)
ax3.set_ylabel('µT')
ax3.set_title('All axes comparison')
ax3.set_xlabel('Time (s)')
ax3.legend()
ax3.set_ylim(-300, 300)
ax3.grid(True, alpha=0.3)

# Clean data histogram
ax4 = axes[1, 1]
ax4.hist(clean_mz, bins=50, alpha=0.7, label='Z (clean)')
ax4.hist(clean_mx, bins=50, alpha=0.7, label='X')
ax4.hist(clean_my, bins=50, alpha=0.7, label='Y')
ax4.set_xlabel('µT')
ax4.set_ylabel('Count')
ax4.set_title('Clean data distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Clean data 3D scatter (XY colored by Z)
ax5 = axes[2, 0]
scatter = ax5.scatter(clean_mx[::5], clean_my[::5], c=clean_mz[::5], cmap='coolwarm',
                      alpha=0.5, s=10, vmin=-100, vmax=100)
ax5.set_xlabel('X (µT)')
ax5.set_ylabel('Y (µT)')
ax5.set_title('Clean data XY (color=Z)')
plt.colorbar(scatter, ax=ax5, label='Z (µT)')
ax5.axis('equal')
ax5.grid(True, alpha=0.3)

# Corrected magnitude (clean data)
ax6 = axes[2, 1]
ax6.hist(corr_mag_clean, bins=50, alpha=0.7, color='green')
ax6.axvline(EARTH_MAG, color='r', linestyle='--', label=f'Expected {EARTH_MAG}µT')
ax6.set_xlabel('Magnitude (µT)')
ax6.set_ylabel('Count')
ax6.set_title('Corrected magnitude (clean data)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path('ml/z_axis_anomaly_analysis.png')
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")
plt.close()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
CRITICAL FINDING: Z-axis shows massive interference spikes

The session contains {len(anomalous_indices)} samples ({len(anomalous_indices)/n*100:.1f}%) with |Z| > {NORMAL_THRESHOLD}µT
Maximum Z value: {mz.max():.0f}µT (normal range is ±100µT)

This appears to be:
1. Magnetic interference from external source (phone, cable, etc.)
2. Or sensor malfunction during part of the session

Impact on calibration:
- Min-max calibration includes these outliers
- Z-axis range inflated from ~{ranges_clean[2]:.0f}µT to 2814µT
- Soft iron Z scale compressed to 0.036 (destroying Z information)
- H/V ratio becomes inverted

RECOMMENDATION:
Filter out samples with |mz| > {NORMAL_THRESHOLD}µT before calibration
Or use robust statistics (median, IQR) instead of min-max
""")
