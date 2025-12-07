# SIMCAP

*Sensor Inferred MOtion CAPture*

---

<link rel="stylesheet" href="src/simcap.css">

## Architecture

```mermaid
graph LR
    subgraph Device
        GAMBIT[GAMBIT Firmware]
        IMU[9-DoF IMU]
    end

    subgraph Web
        UI[Web UI]
        COLLECTOR[Collector]
    end

    subgraph ML
        LOADER[Data Loader]
        MODEL[CNN Model]
    end

    subgraph Storage
        DATA[(Labeled Data)]
    end

    IMU --> GAMBIT
    GAMBIT --> |BLE| UI
    GAMBIT --> |BLE| COLLECTOR
    COLLECTOR --> |Export| DATA
    UI --> |Upload| DATA
    DATA --> LOADER
    LOADER --> MODEL
```

## Components

### ML Pipeline

#### [ML Training Pipeline](ml/)
**Gesture Classification from IMU Data**

Python pipeline for training gesture classifiers: data loading, preprocessing, 1D CNN models, and TFLite deployment.

`Status: Active`

---

### Device Firmware

#### [GAMBIT Device](src/device/GAMBIT/)
**Gyroscope Accelerometer Magnetometer Baseline Inference Telemetry**

Espruino Puck.js firmware for 9-DoF IMU data collection at 50Hz with BLE streaming.

`Status: Active`

#### [BAE](src/device/BAE/)
**Bluetooth Advertise Everything**

Reference implementation for BLE advertising with EspruinoHub/MQTT integration.

`Status: Reference`

---

### Web Interfaces

#### [GAMBIT Web](src/web/GAMBIT/)
**Baseline Data Collection UI**

Web UI for real-time sensor visualization and GitHub data upload via WebBLE.

`Status: Active`

#### [GAMBIT Collector](src/web/GAMBIT/collector.html)
**Labeled Data Collection for ML Training**

Enhanced UI with gesture labeling, session metadata, and export for training pipeline.

`Status: Active`

#### [P0](src/web/P0/)
**Initial Prototype**

WebSocket-based data visualization and capture interface with D3.js charts.

`Status: Prototype`

---

### Documentation

#### [Documentation Index](docs/)
**Implementation & Design Docs**

Component documentation and system architecture overview.

`Status: Documentation`

#### [Design Documents](docs/design/)
**Conceptual Analysis & Research**

Vision vs. reality analysis, roadmap, and ML pipeline proposals.

`Status: Design`

---

### Concepts & Research

#### [JOYPAD](src/web/JOYPAD/)
**Dual-Hand Game Controller Concept**

Emulating controller HID from sensor data using two SIMCAP devices.

`Status: Concept`

#### [FFO$$](src/web/FFO$$/)
**Fist Full Of Dollars**

$1 family algorithms for gesture inference from low-dimensional observation.

`Status: Research`

---

## Quick Start

### Data Collection
1. Flash [GAMBIT firmware](src/device/GAMBIT/) to Puck.js
2. Open [Collector UI](src/web/GAMBIT/collector.html) locally or [Web UI](https://christopherdebeer.github.io/simcap/src/web/GAMBIT/)
3. Connect device and configure session metadata
4. Select gesture, record, and export labeled data

### ML Training
```bash
# Install dependencies
pip install -r ml/requirements.txt

# Label existing data (CLI)
python -m ml.label data/GAMBIT/

# Train model
python -m ml.train --data-dir data/GAMBIT --epochs 50
```

## Data

Baseline sensor data is stored in [`data/GAMBIT/`](data/GAMBIT/) with optional `.meta.json` label files.

## Links

- [GitHub Repository](https://github.com/christopherdebeer/simcap)
- [Hosted Web UI](https://christopherdebeer.github.io/simcap/src/web/GAMBIT/)
- [Espruino Puck.js](https://www.puck-js.com/)
