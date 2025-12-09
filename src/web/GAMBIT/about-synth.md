# GAMBIT Sensor Synthesizer

A real-time sensor-reactive audio synthesizer that transforms 9-axis IMU sensor data from the GAMBIT device into expressive musical sound. Move your Puck.js device to control pitch, timbre, volume, and effects in real-time.

## Features

### 🎵 Real-Time Audio Synthesis
- **Web Audio API** powered synthesis running entirely in the browser
- Multiple waveform types: sine, square, sawtooth, and triangle waves
- Low-pass filter with adjustable resonance
- Built-in reverb with dry/wet mix control
- LFO (Low-Frequency Oscillator) for vibrato and modulation effects

### 📡 Sensor-to-Sound Mapping

The synthesizer maps sensor data to audio parameters as follows:

| Sensor Axis | Audio Parameter | Range | Description |
|-------------|----------------|-------|-------------|
| **Accelerometer Y** | Frequency | ±2 octaves | Tilt forward/backward to change pitch |
| **Gyroscope Z** | Filter Cutoff | 200-5000 Hz | Rotate to sweep filter frequency |
| **Light Sensor** | Volume | 10-50% | Cover device to reduce volume |
| **Magnetometer X** | LFO Rate | 0.5-15.5 Hz | Vibrato/tremolo speed |
| **Gyroscope X** | Detune | ±50 cents | Fine pitch adjustment |
| **Accelerometer Z** | Reverb Mix | 0-100% | Tilt up/down to add reverb |

### 🎛️ Configurable Parameters

- **Oscillator Type**: Choose between sine, square, sawtooth, or triangle waveforms
- **Base Frequency**: Set the center pitch (100-800 Hz)
- **Filter Resonance**: Adjust filter Q factor (0.1-20)
- **Reverb Mix**: Set the base reverb amount (0-100%)

### 📊 Visual Feedback

- **Real-time waveform visualization** showing the generated audio signal
- **Live sensor value display** for all 9 IMU axes plus environmental sensors
- **Synthesis parameter monitoring** showing current values for all audio parameters
- **Battery level indicator** with color-coded status
- **Connection status** with clear visual indicators

## How to Use

### 1. Prerequisites

- **Hardware**: Espruino Puck.js v2 with GAMBIT firmware installed
- **Browser**: Chrome, Edge, Opera, or WebBLE-enabled browser
- **Permissions**: Bluetooth access enabled

### 2. Getting Started

1. **Load the Synthesizer**
   - Open `synth.html` in a WebBLE-compatible browser
   - Or navigate to the hosted version at: `https://christopherdebeer.github.io/simcap/src/web/GAMBIT/synth.html`

2. **Connect Your Device**
   - Click the **"Connect Device"** button
   - Select your Puck.js from the Bluetooth device picker
   - Wait for the "Connected" status message

3. **Start Data Capture**
   - **Press the button** on your Puck.js device
   - The device will begin streaming sensor data at 50 Hz for 30 seconds
   - LED will flash blue to confirm capture started

4. **Start the Synthesizer**
   - Click the **"Start Synth"** button
   - You should immediately hear a tone
   - Move the device to control the sound!

### 3. Playing the Synthesizer

Try these movements to explore the sound:

- **Tilt forward/backward** (Y-axis): Changes pitch up and down
- **Rotate left/right** (Z-axis): Sweeps the filter for a "wah" effect
- **Cover the light sensor**: Reduces volume (creates dynamics)
- **Wave near magnets**: Adds vibrato and modulation
- **Tilt up/down** (Z-axis): Mixes in reverb for spacious sound
- **Quick rotations** (X-axis): Fine-tunes the pitch

### 4. Tips for Musical Expression

- **Gesture smoothly** for continuous pitch changes
- **Use the light sensor** like a volume pedal
- **Combine movements** for complex timbral changes
- **Experiment with different waveforms** for varied textures:
  - **Sine**: Pure, smooth tones (flute-like)
  - **Square**: Hollow, woody sounds (clarinet-like)
  - **Sawtooth**: Bright, buzzy tones (brass-like)
  - **Triangle**: Mellow, soft sounds (gentle synth)

## Technical Details

### Audio Architecture

```
Oscillator (variable frequency/detune)
    ↓
LFO Modulation (variable rate)
    ↓
Low-Pass Filter (variable cutoff/resonance)
    ↓
Dry/Wet Split
    ↓
Reverb (convolution)
    ↓
Gain Control (variable volume)
    ↓
Analyzer (visualization)
    ↓
Audio Output
```

### Data Flow

1. **Device**: GAMBIT firmware samples IMU at 50 Hz
2. **Transmission**: BLE Nordic UART Service streams JSON telemetry
3. **Parsing**: Web app extracts sensor values from GAMBIT-prefixed JSON
4. **Normalization**: Sensor values scaled to 0-1 range using expected min/max
5. **Mapping**: Normalized values transformed to audio parameter ranges
6. **Synthesis**: Web Audio API nodes updated with smooth parameter changes
7. **Visualization**: Analyzer provides real-time waveform display

### Performance Optimization

- **Smooth parameter changes**: Uses `setTargetAtTime()` for glitch-free transitions
- **Efficient parsing**: Single-pass JSON parsing with error handling
- **Canvas optimization**: RequestAnimationFrame for 60 fps visualization
- **Audio node reuse**: Persistent nodes avoid creation/destruction overhead

## Sensor Data Format

The synthesizer expects GAMBIT telemetry in the following format:

```json
{
  "ax": -464,    // Accelerometer X (mg)
  "ay": -7949,   // Accelerometer Y (mg)
  "az": -2282,   // Accelerometer Z (mg)
  "gx": 12,      // Gyroscope X (°/s)
  "gy": -15,     // Gyroscope Y (°/s)
  "gz": 8,       // Gyroscope Z (°/s)
  "mx": 1.2,     // Magnetometer X (μT)
  "my": -0.8,    // Magnetometer Y (μT)
  "mz": 15.3,    // Magnetometer Z (μT)
  "l": 0.42,     // Light sensor (0-1)
  "t": 24.5,     // Temperature (°C)
  "c": 12,       // Capacitive sense
  "b": 87,       // Battery (%)
  "s": 1,        // State (0=idle, 1=capturing)
  "n": 5         // Button press count
}
```

## Browser Compatibility

### Supported Browsers
- ✅ Chrome 56+ (desktop and Android)
- ✅ Edge 79+
- ✅ Opera 43+
- ✅ Safari (iOS) via [WebBLE app](https://apps.apple.com/app/webble/id1193700808)

### Required Features
- Web Bluetooth API
- Web Audio API
- Canvas API
- ES6 JavaScript

## Customization

### Modify Sensor Mappings

Edit the `updateSynthesis()` function to change which sensors control which parameters:

```javascript
// Example: Use gyroscope Y for frequency instead of accelerometer Y
const gyNorm = normalize(sensorData.gy, 'gy');
const freqMultiplier = Math.pow(2, (gyNorm - 0.5) * 4);
const targetFreq = baseFreq * freqMultiplier;
oscillator.frequency.setTargetAtTime(targetFreq, audioContext.currentTime, 0.01);
```

### Adjust Parameter Ranges

Modify the mapping calculations to change sensitivity:

```javascript
// Example: Increase frequency range to ±3 octaves (instead of ±2)
const freqMultiplier = Math.pow(2, (ayNorm - 0.5) * 6); // ±3 octaves
```

### Add New Audio Effects

Insert additional Web Audio nodes into the signal chain:

```javascript
// Example: Add a delay effect
const delayNode = audioContext.createDelay();
delayNode.delayTime.value = 0.3;
filterNode.connect(delayNode);
delayNode.connect(dryGain);
```

## Troubleshooting

### No Sound
- Ensure synthesizer is started (button shows "Stop Synth")
- Check browser audio permissions
- Verify device is sending data (sensor values updating)
- Try increasing base frequency or adjusting light sensor value

### Connection Issues
- Ensure Bluetooth is enabled on your computer
- Make sure GAMBIT firmware is loaded on Puck.js
- Try refreshing the page and reconnecting
- Check that no other app is connected to the device

### Choppy Audio
- Close unnecessary browser tabs to free resources
- Reduce reverb mix for better performance
- Ensure stable Bluetooth connection
- Try a different USB Bluetooth adapter if on desktop

### Device Not Sending Data
- Press the button on the Puck.js to start capture
- Wait for blue LED flash confirmation
- Check battery level (needs >20% for reliable operation)
- Reflash GAMBIT firmware if persistent issues

## Future Enhancements

Potential improvements for the synthesizer:

- [ ] Multi-voice polyphony with gesture-based note triggering
- [ ] Recording and playback of sensor performance
- [ ] MIDI output for controlling external synthesizers
- [ ] Machine learning gesture recognition to trigger presets
- [ ] Additional effects: distortion, chorus, phaser
- [ ] Visual 3D representation synchronized with sound
- [ ] Preset system for saving/loading configurations
- [ ] Collaborative mode with multiple devices

## Credits

Part of the **SIMCAP** (Sensor Inferred Motion CAPture) platform.

**Technologies Used:**
- Web Bluetooth API
- Web Audio API
- HTML5 Canvas
- Espruino JavaScript (device firmware)
- Puck.js library (WebBLE interface)

## License

Inherits license from SIMCAP project.

## See Also

- [GAMBIT Firmware Documentation](../../../src/device/GAMBIT/README.md)
- [GAMBIT Data Collector](./collector.html)
- [Firmware Loader](../loader/index.html)
- [SIMCAP Project Overview](../../../README.md)

---

<link rel="stylesheet" href="../../simcap.css">
