# SIMCAP Images Branch

This branch contains visualization images for the SIMCAP project.

## Structure

```
composite_{timestamp}.png           # Session composite image
calibration_stages_{timestamp}.png  # Calibration visualization
orientation_3d_{timestamp}.png      # 3D orientation plot
orientation_track_{timestamp}.png   # Orientation tracking plot
raw_axes_{timestamp}.png            # Raw sensor axes plot
windows_{timestamp}/                # Per-second window images
  └── window_001.png
  └── window_001/
      ├── accel.png
      ├── gyro.png
      ├── mag.png
      └── trajectory_*.png
trajectory_comparison_{timestamp}/  # Trajectory comparison images
  └── raw_3d.png
  └── raw_overlay.png
```

## Usage

Images are uploaded via the Python upload script:
```bash
python -m ml.github_upload --input-dir visualizations --manifest
```

Files are served via `raw.githubusercontent.com`:
```
https://raw.githubusercontent.com/christopherdebeer/simcap/images/{path}
```

## Note

This branch only contains visualization images. Source code is on the `main` branch.
