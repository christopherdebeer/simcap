#!/usr/bin/env python3
"""
Analyze residual on clean data (excluding Z-axis spikes)
Investigate why residual is high even without finger magnets.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Edinburgh reference
EARTH_H = 16.0
EARTH_V = 47.8
EARTH_MAG = np.sqrt(EARTH_H**2 + EARTH_V**2)
EARTH_WORLD = np.array([EARTH_H, 0, EARTH_V])  # NED

# Load session
session_path = Path('data/GAMBIT/2025-12-30T22_46_28.771Z.json')
with open(session_path, 'r') as f:
    data = json.load(f)

samples = data['samples']
n = len(samples)

# Extract all data
mx = np.array([s.get('mx_ut', s.get('mx', 0)) for s in samples])
my = np.array([s.get('my_ut', s.get('my', 0)) for s in samples])
mz = np.array([s.get('mz_ut', s.get('mz', 0)) for s in samples])
ax = np.array([s.get('ax_g', s.get('ax', 0)) for s in samples])
ay = np.array([s.get('ay_g', s.get('ay', 0)) for s in samples])
az = np.array([s.get('az_g', s.get('az', 0)) for s in samples])
qw = np.array([s.get('orientation_w', 1) for s in samples])
qx = np.array([s.get('orientation_x', 0) for s in samples])
qy = np.array([s.get('orientation_y', 0) for s in samples])
qz = np.array([s.get('orientation_z', 0) for s in samples])
ts = np.array([s.get('timestamp', i*20) for i, s in enumerate(samples)])

# Filter out anomalous samples
ANOMALY_THRESHOLD = 150
clean_mask = np.abs(mz) <= ANOMALY_THRESHOLD
print(f"Clean samples: {clean_mask.sum()} / {n} ({clean_mask.sum()/n*100:.1f}%)")

# Apply clean mask
mx_c = mx[clean_mask]
my_c = my[clean_mask]
mz_c = mz[clean_mask]
ax_c = ax[clean_mask]
ay_c = ay[clean_mask]
az_c = az[clean_mask]
qw_c = qw[clean_mask]
qx_c = qx[clean_mask]
qy_c = qy[clean_mask]
qz_c = qz[clean_mask]
n_c = len(mx_c)

print("\n" + "=" * 80)
print("CLEAN DATA ANALYSIS")
print("=" * 80)

# Helper functions
def accel_to_roll_pitch(ax, ay, az):
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    if a_norm < 0.1:
        return 0, 0
    ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return roll, pitch

def tilt_compensate(mx, my, mz, roll, pitch):
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    mx_h = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p
    my_h = my * cos_r - mz * sin_r
    mz_h = -mx * sin_p + my * cos_r * sin_p + mz * cos_r * cos_p
    return mx_h, my_h, mz_h

def euler_to_rotation(roll, pitch, yaw):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

def quat_to_rotation(qw, qx, qy, qz):
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

# 1. Min-max calibration on clean data
print("\n--- Min-Max Calibration ---")
hard_iron = np.array([
    (mx_c.max() + mx_c.min()) / 2,
    (my_c.max() + my_c.min()) / 2,
    (mz_c.max() + mz_c.min()) / 2
])
ranges = np.array([
    mx_c.max() - mx_c.min(),
    my_c.max() - my_c.min(),
    mz_c.max() - mz_c.min()
])
soft_iron = 2 * EARTH_MAG / ranges

print(f"Hard iron: [{hard_iron[0]:.1f}, {hard_iron[1]:.1f}, {hard_iron[2]:.1f}] µT")
print(f"Ranges: [{ranges[0]:.1f}, {ranges[1]:.1f}, {ranges[2]:.1f}] µT")
print(f"Soft iron: [{soft_iron[0]:.3f}, {soft_iron[1]:.3f}, {soft_iron[2]:.3f}]")

# Apply min-max calibration
corrected_mm = np.array([
    (mx_c - hard_iron[0]) * soft_iron[0],
    (my_c - hard_iron[1]) * soft_iron[1],
    (mz_c - hard_iron[2]) * soft_iron[2]
])
corr_mag_mm = np.sqrt(corrected_mm[0]**2 + corrected_mm[1]**2 + corrected_mm[2]**2)
print(f"\nCorrected magnitude: {corr_mag_mm.mean():.1f} ± {corr_mag_mm.std():.1f} µT")

# 2. Compute residual using AHRS orientation
print("\n--- Residual with AHRS Orientation ---")
residuals_ahrs = []
for i in range(n_c):
    R = quat_to_rotation(qw_c[i], qx_c[i], qy_c[i], qz_c[i])
    earth_device = R @ EARTH_WORLD
    mag_vec = np.array([corrected_mm[0][i], corrected_mm[1][i], corrected_mm[2][i]])
    residuals_ahrs.append(np.linalg.norm(mag_vec - earth_device))

residuals_ahrs = np.array(residuals_ahrs)
print(f"AHRS residual: {residuals_ahrs.mean():.1f} ± {residuals_ahrs.std():.1f} µT")

# 3. Compute residual using accelerometer-derived orientation (independent yaw)
print("\n--- Residual with Accel-Derived Orientation ---")
residuals_accel = []
h_components = []
v_components = []

for i in range(n_c):
    roll, pitch = accel_to_roll_pitch(ax_c[i], ay_c[i], az_c[i])
    mx_h, my_h, mz_h = tilt_compensate(corrected_mm[0][i], corrected_mm[1][i], corrected_mm[2][i], roll, pitch)

    h_components.append(np.sqrt(mx_h**2 + my_h**2))
    v_components.append(mz_h)

    # Compute yaw from magnetometer
    yaw = np.arctan2(-my_h, mx_h)

    R = euler_to_rotation(roll, pitch, yaw)
    earth_device = R.T @ EARTH_WORLD
    mag_vec = np.array([corrected_mm[0][i], corrected_mm[1][i], corrected_mm[2][i]])
    residuals_accel.append(np.linalg.norm(mag_vec - earth_device))

residuals_accel = np.array(residuals_accel)
h_components = np.array(h_components)
v_components = np.array(v_components)

print(f"Accel-derived residual: {residuals_accel.mean():.1f} ± {residuals_accel.std():.1f} µT")
print(f"\nH component: {h_components.mean():.1f} µT (expected {EARTH_H})")
print(f"V component: {v_components.mean():.1f} µT (expected {EARTH_V})")
print(f"H/V ratio: {np.abs(h_components/v_components).mean():.2f} (expected {EARTH_H/EARTH_V:.2f})")

# 4. Orientation-aware calibration
print("\n" + "=" * 80)
print("ORIENTATION-AWARE CALIBRATION ON CLEAN DATA")
print("=" * 80)

# Use subset for optimization
step = max(1, n_c // 300)
indices = list(range(0, n_c, step))[:300]

mx_opt = mx_c[indices]
my_opt = my_c[indices]
mz_opt = mz_c[indices]
ax_opt = ax_c[indices]
ay_opt = ay_c[indices]
az_opt = az_c[indices]

print(f"Using {len(indices)} samples for optimization")

def residual_func(params):
    offset = params[:3]
    S = params[3:12].reshape(3, 3)

    raw = np.array([mx_opt, my_opt, mz_opt])
    centered = raw - offset.reshape(3, 1)
    corrected = S @ centered

    residuals = []
    for i in range(len(indices)):
        roll, pitch = accel_to_roll_pitch(ax_opt[i], ay_opt[i], az_opt[i])
        cmx, cmy, cmz = corrected[0][i], corrected[1][i], corrected[2][i]

        mx_h, my_h, _ = tilt_compensate(cmx, cmy, cmz, roll, pitch)
        yaw = np.arctan2(-my_h, mx_h)

        R = euler_to_rotation(roll, pitch, yaw)
        earth_device = R.T @ EARTH_WORLD
        mag_corr = np.array([cmx, cmy, cmz])
        diff = mag_corr - earth_device
        residuals.extend(diff.tolist())

    return np.array(residuals)

# Initial guess
offset_init = hard_iron.copy()
S_init = np.diag(soft_iron).flatten()
x0 = np.concatenate([offset_init, S_init])

print("Running optimization...")
result = least_squares(residual_func, x0, method='lm', max_nfev=10000, verbose=0)

offset_opt = result.x[:3]
S_opt = result.x[3:12].reshape(3, 3)

print(f"\n--- Optimized Calibration ---")
print(f"Hard iron: [{offset_opt[0]:.2f}, {offset_opt[1]:.2f}, {offset_opt[2]:.2f}] µT")
print(f"\nSoft iron matrix:")
for row in S_opt:
    print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")

# Apply optimized calibration to all clean data
raw_c = np.array([mx_c, my_c, mz_c])
corrected_opt = S_opt @ (raw_c - offset_opt.reshape(3, 1))
corr_mag_opt = np.sqrt(corrected_opt[0]**2 + corrected_opt[1]**2 + corrected_opt[2]**2)

print(f"\nOptimized corrected magnitude: {corr_mag_opt.mean():.1f} ± {corr_mag_opt.std():.1f} µT")

# Compute residual with optimized calibration
residuals_opt = []
h_opt = []
v_opt = []

for i in range(n_c):
    roll, pitch = accel_to_roll_pitch(ax_c[i], ay_c[i], az_c[i])
    cmx, cmy, cmz = corrected_opt[0][i], corrected_opt[1][i], corrected_opt[2][i]

    mx_h, my_h, mz_h = tilt_compensate(cmx, cmy, cmz, roll, pitch)
    h_opt.append(np.sqrt(mx_h**2 + my_h**2))
    v_opt.append(mz_h)

    yaw = np.arctan2(-my_h, mx_h)
    R = euler_to_rotation(roll, pitch, yaw)
    earth_device = R.T @ EARTH_WORLD
    mag_corr = np.array([cmx, cmy, cmz])
    residuals_opt.append(np.linalg.norm(mag_corr - earth_device))

residuals_opt = np.array(residuals_opt)
h_opt = np.array(h_opt)
v_opt = np.array(v_opt)

print(f"\n--- Optimized Results ---")
print(f"Residual: {residuals_opt.mean():.1f} ± {residuals_opt.std():.1f} µT")
print(f"H component: {h_opt.mean():.1f} µT (expected {EARTH_H})")
print(f"V component: {v_opt.mean():.1f} µT (expected {EARTH_V})")
valid_mask = np.abs(v_opt) > 1
if valid_mask.sum() > 0:
    hv_ratio = np.abs(h_opt[valid_mask] / v_opt[valid_mask]).mean()
    print(f"H/V ratio: {hv_ratio:.2f} (expected {EARTH_H/EARTH_V:.2f})")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Comparison of calibration methods on clean data:

Method              | Magnitude      | Residual       | H/V Ratio
--------------------|----------------|----------------|-----------
Min-Max (diagonal)  | {corr_mag_mm.mean():.1f} ± {corr_mag_mm.std():.1f} µT | {residuals_accel.mean():.1f} ± {residuals_accel.std():.1f} µT | {np.abs(h_components/v_components).mean():.2f}
Orientation-Aware   | {corr_mag_opt.mean():.1f} ± {corr_mag_opt.std():.1f} µT | {residuals_opt.mean():.1f} ± {residuals_opt.std():.1f} µT | {hv_ratio:.2f}
Expected            | 50.4 µT        | ~0 µT          | 0.33

Key observations:
1. Raw data ranges are {ranges.mean():.0f}µT (expected ~100µT) - sensor sees 2x Earth field
2. This suggests strong environmental magnetic interference or incorrect sensor gain
3. Even with optimization, residual remains high at {residuals_opt.mean():.1f}µT

Possible root causes:
1. Nearby ferromagnetic materials causing field distortion
2. Non-uniform soft iron distortion (spatially varying)
3. Sensor mounting near PCB components with local fields
4. Temperature-dependent calibration drift
""")

# Generate comparison plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Clean Data Calibration Comparison', fontsize=14)

ts_c = ts[clean_mask]
ts_c = (ts_c - ts_c[0]) / 1000.0

# Min-max corrected magnitude
ax = axes[0, 0]
ax.plot(ts_c, corr_mag_mm, 'b-', alpha=0.5, linewidth=0.5)
ax.axhline(EARTH_MAG, color='g', linestyle='--', label=f'Expected {EARTH_MAG}µT')
ax.set_ylabel('Magnitude (µT)')
ax.set_title('Min-Max Corrected Magnitude')
ax.legend()
ax.grid(True, alpha=0.3)

# Min-max residual
ax = axes[0, 1]
ax.plot(ts_c, residuals_accel, 'r-', alpha=0.5, linewidth=0.5)
ax.axhline(10, color='g', linestyle='--', label='Target <10µT')
ax.set_ylabel('Residual (µT)')
ax.set_title('Min-Max Earth Residual')
ax.legend()
ax.grid(True, alpha=0.3)

# Min-max H/V
ax = axes[0, 2]
ax.scatter(h_components[::10], v_components[::10], alpha=0.3, s=5)
ax.axhline(EARTH_V, color='g', linestyle='--', alpha=0.5)
ax.axvline(EARTH_H, color='g', linestyle='--', alpha=0.5)
ax.plot(EARTH_H, EARTH_V, 'go', markersize=10, label='Expected')
ax.set_xlabel('H (µT)')
ax.set_ylabel('V (µT)')
ax.set_title('Min-Max H/V Components')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 80)
ax.set_ylim(-60, 80)

# Optimized magnitude
ax = axes[1, 0]
ax.plot(ts_c, corr_mag_opt, 'b-', alpha=0.5, linewidth=0.5)
ax.axhline(EARTH_MAG, color='g', linestyle='--', label=f'Expected {EARTH_MAG}µT')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Magnitude (µT)')
ax.set_title('Orientation-Aware Corrected Magnitude')
ax.legend()
ax.grid(True, alpha=0.3)

# Optimized residual
ax = axes[1, 1]
ax.plot(ts_c, residuals_opt, 'r-', alpha=0.5, linewidth=0.5)
ax.axhline(10, color='g', linestyle='--', label='Target <10µT')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Residual (µT)')
ax.set_title('Orientation-Aware Earth Residual')
ax.legend()
ax.grid(True, alpha=0.3)

# Optimized H/V
ax = axes[1, 2]
ax.scatter(h_opt[::10], v_opt[::10], alpha=0.3, s=5)
ax.axhline(EARTH_V, color='g', linestyle='--', alpha=0.5)
ax.axvline(EARTH_H, color='g', linestyle='--', alpha=0.5)
ax.plot(EARTH_H, EARTH_V, 'go', markersize=10, label='Expected')
ax.set_xlabel('H (µT)')
ax.set_ylabel('V (µT)')
ax.set_title('Orientation-Aware H/V Components')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 80)
ax.set_ylim(-60, 80)

plt.tight_layout()
output_path = Path('ml/clean_data_calibration_comparison.png')
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")
plt.close()
