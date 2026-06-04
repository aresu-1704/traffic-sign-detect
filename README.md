# Traffic Sign Detection Application

Ứng dụng phát hiện biển báo giao thông trong thời gian thực sử dụng YOLO và ONNX Runtime.

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

## Logging

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

## Tips for Best Performance

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

## License

This project uses:
- OpenCV (BSD)
- ONNX Runtime (MIT)
- Pillow (PIL License)
- psutil (BSD)

## Next Steps

1. Run `python app.py` to start the application
2. Click "▶ Play" to begin detection
3. Monitor real-time metrics on screen
4. Adjust config.py for your hardware if needed
5. Check app.log for detailed information

Happy detecting!
