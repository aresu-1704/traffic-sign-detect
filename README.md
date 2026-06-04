# Traffic Sign Detection Application

Ứng dụng phát hiện biển báo giao thông trong thời gian thực sử dụng YOLO và ONNX Runtime.

## 🎯 Features

- ✅ Real-time traffic sign detection
- ✅ GPU acceleration support (CUDA, TensorRT)
- ✅ Multi-threaded inference for smooth UI
- ✅ Performance monitoring (FPS, latency, CPU, RAM, temperature)
- ✅ Optimized ONNX model inference
- ✅ Professional GUI with status indicators
- ✅ Frame buffering with adaptive dropping
- ✅ Comprehensive logging

## 📋 Requirements

- Python 3.8+
- Webcam or camera device
- Windows/Linux/macOS

### Optional
- NVIDIA GPU with CUDA support (for GPU acceleration)

## 🚀 Installation

### 1. Setup Virtual Environment

```bash
cd traffic-sign-detect

# Create virtual environment (Already done)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Model File

Place your ONNX model in `models/` directory:
- `models/best.pt` (recommended)
- `models/vsyolo.pt`
- `models/yolo12n.pt`
- `models/yolo26n.pt`

## 🎮 Usage

### Basic Run

```bash
python app.py
```

### GUI Controls

- **▶ Play**: Start camera and detection
- **■ Stop**: Stop detection and release camera
- **Status Indicator**: Shows application status
  - Green: Ready/Running
  - Orange: Initializing
  - Red: Error/Stopped

### Keyboard Shortcuts

- Press `Window Close (X)`: Exit application (with confirmation)

## 📊 Performance Metrics

The overlay displays in real-time:

| Metric | Description | Unit |
|--------|-------------|------|
| CPU | CPU usage percentage | % |
| RAM | Memory usage percentage | % |
| TEMP | CPU temperature | °C/N/A |
| FPS | Frames per second (smoothed) | fps |
| LAT | Inference latency (smoothed) | ms |

## ⚙️ Configuration

All configurable parameters are in `config.py`:

```python
# Camera settings
CAMERA = {
    'id': 0,
    'width': 640,
    'height': 480,
    'fps': 30,
}

# Model settings
MODEL = {
    'imgsz': 320,
    'conf_threshold': 0.4,
    'use_gpu': True,
}

# Performance tuning
PERFORMANCE = {
    'ui_update_fps': 30,
    'metrics_window': 30,
}
```

See `OPTIMIZATION.md` for detailed tuning guide.

## 📁 Project Structure

```
traffic-sign-detect/
├── app.py                      # Main entry point
├── config.py                   # Configuration parameters
├── requirements.txt            # Python dependencies
├── OPTIMIZATION.md             # Optimization guide
├── README.md                   # This file
├── app.log                     # Application logs
├── .venv/                      # Virtual environment
│
├── models/
│   ├── best.pt                # ONNX model
│   ├── vsyolo.pt
│   ├── yolo12n.pt
│   └── yolo26n.pt
│
├── core/
│   ├── camera.py              # Camera handling
│   ├── detector.py            # ONNX inference
│   ├── monitor.py             # Performance monitoring
│   └── overlay.py             # Visualization overlay
│
└── ui/
    └── main_window.py         # Tkinter GUI
```

## 🔧 Troubleshooting

### Application Won't Start
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify virtual environment is activated
# (venv) should appear in terminal

# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Camera Not Found
- Ensure camera device is connected
- Check if another app is using the camera
- Try different camera ID in `config.py`:
  ```python
  CAMERA = {'id': 1}  # Try different ID
  ```

### Slow Inference (High Latency)
1. Enable GPU: `use_gpu: True` in config
2. Reduce input size: `imgsz: 256`
3. Increase confidence threshold: `conf_threshold: 0.5`
4. Close background applications

### GPU Not Detected
1. Install NVIDIA GPU support:
   ```bash
   pip install onnxruntime-gpu
   ```
2. Ensure NVIDIA drivers are installed
3. Check GPU availability:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Memory Issues
- Reduce frame queue size in `config.py`
- Close other applications
- Reduce camera resolution

## 📝 Logging

Application logs are saved in `app.log` with:
- Application lifecycle events
- Model loading details
- Inference timing information
- Error messages with stack traces

View logs:
```bash
# Real-time
tail -f app.log

# Windows
type app.log

# Search for errors
grep ERROR app.log
```

## 🎓 Tips for Best Performance

### For Real-time Performance (30+ FPS)
1. Use GPU if available
2. Set `imgsz: 320` (or lower for more speed)
3. Set `conf_threshold: 0.5` (higher = faster)
4. Reduce camera resolution to 640x480

### For High Accuracy
1. Set `imgsz: 640` (larger model input)
2. Set `conf_threshold: 0.3` (lower = more detections)
3. Use GPU for faster inference
4. Increase camera resolution

### For CPU-only Systems
1. Set `use_gpu: False`
2. Reduce `imgsz` to 256
3. Increase `conf_threshold` to 0.6
4. Close other applications
5. Lower camera FPS to 15

## 🤝 Support

For issues or questions:
1. Check `OPTIMIZATION.md` for detailed technical information
2. Review `app.log` for error messages
3. Verify `config.py` settings match your hardware

## 📜 License

This project uses:
- OpenCV (BSD)
- ONNX Runtime (MIT)
- Pillow (PIL License)
- psutil (BSD)

## 🎯 Next Steps

1. ✅ Run `python app.py` to start the application
2. ✅ Click "▶ Play" to begin detection
3. ✅ Monitor real-time metrics on screen
4. ✅ Adjust config.py for your hardware if needed
5. ✅ Check app.log for detailed information

Happy detecting! 🚗🛑
