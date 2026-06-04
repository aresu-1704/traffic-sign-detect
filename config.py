# config.py
"""Configuration parameters for Traffic Sign Detection Application."""

# Camera Configuration
CAMERA = {
    'id': 0,                    # Camera device ID
    'width': 640,               # Frame width
    'height': 480,              # Frame height
    'fps': 30,                  # Target FPS
}

# Model Configuration
MODEL = {
    'model_path': 'models/best.pt',     # Path to ONNX model
    'imgsz': 320,                       # Input image size
    'conf_threshold': 0.4,              # Confidence threshold
    'use_gpu': True,                    # Use GPU if available
}

# Performance Tuning
PERFORMANCE = {
    'inference_threads': 1,             # Number of inference threads
    'ui_update_fps': 30,                # UI update frequency
    'metrics_window': 30,               # Window size for averaging metrics
    'frame_queue_size': 2,              # Max frames in queue
}

# Inference Optimization
INFERENCE = {
    'graph_optimization': 'all',        # ONNX graph optimization level
    'inter_op_threads': 4,              # Inter-op parallelization threads
    'intra_op_threads': 4,              # Intra-op parallelization threads
    'execution_providers': [
        'CUDAExecutionProvider',         # NVIDIA GPU
        'TensorrtExecutionProvider',     # NVIDIA TensorRT
        'CPUExecutionProvider'           # CPU fallback
    ]
}

# Display Configuration
DISPLAY = {
    'window_title': 'Traffic Sign Detection - YOLO Monitor',
    'window_width': 1024,
    'window_height': 640,
    'overlay_bg_alpha': 0.3,            # Overlay background transparency
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',
    'log_file': 'app.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}
