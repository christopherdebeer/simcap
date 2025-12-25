# FFO$$ - Fist Full Of Dollars

Template-based gesture recognition using the $Q-3D algorithm adapted for IMU sensor data.

## Features

- **Record Gesture Templates**: Capture accelerometer traces from SIMCAP devices
- **Live Recognition**: Real-time gesture matching against stored templates
- **3D Trajectory Visualization**: See gesture paths in 3D space with Three.js
- **Vocabulary Management**: Export/import gesture vocabularies as JSON
- **Configurable Algorithm**: Adjust resample points, rejection threshold, etc.

## Quick Start

1. Open `/apps/ffo/` in your browser
2. Click **Connect** to pair with your SIMCAP device
3. Enter a gesture name (e.g., "wave") and click **Start Recording**
4. Perform the gesture while holding the device
5. Click **Stop** then **Save Template**
6. Add more templates, then click **Start Recognition**
7. Perform gestures to see real-time recognition results

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `R` | Start recording |
| `S` | Stop recording |
| `C` | Clear trajectory |

## How It Works

FFO$$ uses the **$Q-3D algorithm**, an adaptation of the [$Q Super-Quick Recognizer](https://depts.washington.edu/acelab/proj/dollar/qdollar.html) for 3D accelerometer data:

1. **Resample**: Variable-length gesture traces are resampled to N equally-spaced points
2. **Normalize**: Trajectories are translated to origin and scaled to unit size
3. **Match**: Point-cloud distance is computed using O(n) lookup table optimization
4. **Score**: Distance is converted to 0-1 confidence score

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Resample Points (N) | 32 | Number of points per template |
| Reject Threshold | 0.5 | Maximum distance for valid match |
| Min Samples | 15 | Minimum samples for recording |
| Remove Gravity | Yes | Subtract gravity estimate |
| Use Lookup Table | Yes | Enable $Q O(n) optimization |

## Vocabulary Format

Vocabularies are exported as JSON:

```json
{
  "version": "1.0.0",
  "templates": [
    {
      "id": "tmpl_12345_abc",
      "name": "wave",
      "points": [{ "x": 0.1, "y": 0.2, "z": 0.9 }, ...],
      "meta": {
        "n": 32,
        "source": "recorded",
        "created": "2025-12-25T..."
      }
    }
  ],
  "meta": {
    "name": "My Gestures"
  }
}
```

## Related

- **[GAMBIT](/apps/gambit/)** - Full sensor visualization and data collection
- **[@simcap/ffo](/packages/ffo/)** - Core FFO$$ algorithm library
- **[FFO$$ Research Analysis](/docs/design/ffo-dollar-research-analysis.md)** - Theoretical background
