# SIMCAP Visualization Package

This package contains modular components for visualizing SIMCAP sensor data.

## Current Structure

```
ml/
├── visualize.py                    # Main entry point (COMPLETE - restored from git)
└── visualization/
    ├── __init__.py                 # Package exports
    ├── data_processor.py           # ✅ EXTRACTED - SensorDataProcessor class
    ├── visual_distinction.py       # ✅ EXTRACTED - VisualDistinctionEngine class
    └── README.md                   # This file
```

## Completed Modules

### 1. `data_processor.py` (~200 lines)
**Status:** ✅ Complete and extracted

**Contains:**
- `SensorDataProcessor` class
- Session loading from JSON files
- Sensor array extraction with calibration detection
- Magnitude calculations for all sensor types

**Usage:**
```python
from visualization import SensorDataProcessor

processor = SensorDataProcessor(data_dir)
sensors = processor.extract_sensor_arrays(session['data'])
```

### 2. `visual_distinction.py` (~80 lines)
**Status:** ✅ Complete and extracted

**Contains:**
- `VisualDistinctionEngine` class
- Color generation from sensor patterns
- Signature pattern creation for visual fingerprinting

**Usage:**
```python
from visualization import VisualDistinctionEngine

engine = VisualDistinctionEngine()
color = engine.sensor_to_color(sensors, start_idx, end_idx)
pattern = engine.create_signature_pattern(sensors, start_idx, end_idx)
```

## Remaining Work (Future Refactoring)

The following components remain in `visualize.py` and can be extracted as needed:

### 3. `calibration_viz.py` (Future - ~400 lines)
**Would contain:**
- `create_calibration_stages_image()` - 4-stage magnetometer comparison
- Calibration-specific plotting utilities
- SNR and quality metrics visualization
- Stage comparison logic with statistics

### 4. `trajectory_viz.py` (Future - ~400 lines)
**Would contain:**
- `create_trajectory_comparison_image()` - Full session 3D trajectories
- `_create_window_trajectory_comparison()` - Per-window trajectories
- Trajectory statistics computation
- 3D plotting utilities for magnetometer data

### 5. `window_viz.py` (Future - ~600 lines)
**Would contain:**
- `create_window_images()` - Per-second window generation
- Individual image generation (timeseries, 3D, signatures, stats)
- Window composite image creation
- Window metadata collection

### 6. `session_visualizer.py` (Future - ~300 lines)
**Would contain:**
- `SessionVisualizer` class (main coordinator)
- `create_composite_session_image()` - Full session overview
- `create_raw_axis_images()` - Detailed axis visualizations
- Orchestration of other visualization modules

### 7. `html_generator.py` (Future - ~800 lines)
**Would contain:**
- `HTMLGenerator` class
- Interactive HTML explorer generation
- JavaScript for filtering and navigation
- Responsive UI with modal image viewer

## Current Status

✅ **Original functionality fully restored** - All features from git HEAD are working
✅ **Modular foundation started** - Data processor and visual distinction extracted
📋 **Refactoring path defined** - Clear plan for future modularization

## Benefits of Current Approach

1. **Full functionality preserved** - All original features work immediately
2. **Modular foundation** - Two key components extracted and reusable
3. **Incremental refactoring** - Can extract more modules as needed
4. **No breaking changes** - Original `visualize.py` still works standalone

## Next Steps (Optional)

To continue modularization:

1. Extract `SessionVisualizer` class to `session_visualizer.py`
2. Move calibration visualization methods to `calibration_viz.py`
3. Move trajectory visualization methods to `trajectory_viz.py`
4. Move window generation to `window_viz.py`
5. Extract HTML generator to `html_generator.py`
6. Update `visualize.py` to import from modules
7. Update `__init__.py` to export all classes

## Testing

Test the current setup:
```bash
# Test with original visualize.py (full functionality)
python ml/visualize.py --data-dir data/GAMBIT --output-dir visualizations

# Test extracted modules
python -c "from visualization import SensorDataProcessor, VisualDistinctionEngine; print('✅ Modules imported successfully')"
```

## Design Philosophy

- **Pragmatic over perfect** - Restore full functionality first
- **Incremental refactoring** - Extract modules as needed
- **Backward compatible** - Keep original entry point working
- **Reusable components** - Extracted modules can be used independently
