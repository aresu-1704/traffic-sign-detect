# core/detector.py

import cv2
import numpy as np
import onnxruntime as ort
import logging
import time
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_labels(label_path=None):
    """Load class labels from label.yaml.
    
    Args:
        label_path: Path to label.yaml. If None, searches common locations.
        
    Returns:
        Dict mapping class_id (int) -> class_name (str)
    """
    search_paths = []
    if label_path:
        search_paths.append(Path(label_path))
    search_paths.extend([
        Path("label.yaml"),
        Path("labels.yaml"),
        Path("models/label.yaml"),
        Path("data/label.yaml"),
    ])
    
    for path in search_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if 'names' in data:
                    names = data['names']
                    if isinstance(names, list):
                        label_map = {i: name for i, name in enumerate(names)}
                    elif isinstance(names, dict):
                        label_map = {int(k): v for k, v in names.items()}
                    else:
                        continue
                    logger.info(f"Loaded {len(label_map)} labels from {path}")
                    return label_map
            except Exception as e:
                logger.warning(f"Failed to load labels from {path}: {e}")
    
    logger.warning("No label file found, using class IDs as labels")
    return {}


class Detector:
    """Optimized ONNX detector with GPU support and caching."""

    def __init__(
        self,
        model_path,
        imgsz=640,
        conf=0.4,
        use_gpu=True,
        label_path=None
    ):
        """Initialize detector.
        
        Args:
            model_path: Path to ONNX model
            imgsz: Input image size
            conf: Confidence threshold
            use_gpu: Use GPU if available
            label_path: Path to label.yaml (auto-detected if None)
        """
        self.imgsz = imgsz
        self.conf = conf
        self.inference_times = []
        
        # Load class labels
        self.labels = load_labels(label_path)
        
        # Set execution providers with GPU priority
        providers = []
        if use_gpu:
            providers = [
                ('CUDAExecutionProvider', {'device_id': 0}),
                ('TensorrtExecutionProvider', {'device_id': 0}),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=providers,
                sess_options=self._get_session_options()
            )
            logger.info(f"Model loaded: {model_path}")
            logger.info(f"Execution providers: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Cache input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Get model input shape
        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Input: {self.input_name}, Shape: {input_shape}")
        logger.info(f"Outputs: {self.output_names}")
        
        # Verify image size matches model input (usually shape is [1, 3, height, width])
        if len(input_shape) >= 3:
            expected_size = input_shape[2] if isinstance(input_shape[2], int) else input_shape[3]
            if expected_size and self.imgsz != expected_size:
                logger.warning(f"Image size mismatch - Using {self.imgsz}, Model expects {expected_size}")
                self.imgsz = expected_size
        
        # Determine output format from model output shape
        output_shape = self.session.get_outputs()[0].shape
        logger.info(f"Output shape: {output_shape}")
        
        if len(output_shape) >= 2:
            last_dim = output_shape[-1] if isinstance(output_shape[-1], int) else 0
            second_dim = output_shape[1] if isinstance(output_shape[1], int) else 0
            
            if last_dim in (5, 6, 7) and second_dim > last_dim:
                # End-to-end NMS format: [1, num_detections, 6]
                # Each row: [x1, y1, x2, y2, score, class_id]
                self._output_format = "e2e"
                self.num_classes = len(self.labels) if self.labels else 0
            elif last_dim > second_dim:
                # YOLOv5 format: [1, num_detections, 5+num_classes]
                self._output_format = "v5"
                self.num_classes = last_dim - 5
            else:
                # YOLOv8 raw format: [1, 4+num_classes, num_detections]
                self._output_format = "v8"
                self.num_classes = second_dim - 4
        else:
            self._output_format = "e2e"
            self.num_classes = len(self.labels) if self.labels else 0
        
        logger.info(f"Detected output format: {self._output_format}, num_classes: {self.num_classes}")
    
    @staticmethod
    def _get_session_options():
        """Configure session options for optimal performance."""
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.inter_op_num_threads = 4
        sess_opts.intra_op_num_threads = 4
        return sess_opts

    def preprocess(self, frame):
        """Optimized preprocessing with minimal operations."""
        # Resize once
        img = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        
        # Convert color and normalize in one pass
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Transpose and expand dims
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)  # Ensure float32 for ONNX

    def _get_label(self, class_id):
        """Get human-readable label for a class ID.
        
        Args:
            class_id: Integer class ID
            
        Returns:
            Class name string
        """
        return self.labels.get(class_id, f"class_{class_id}")

    def predict(self, frame):
        """Run inference and draw labeled bounding boxes.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            (frame, detections) tuple
        """
        h, w = frame.shape[:2]
        scale_x = w / self.imgsz
        scale_y = h / self.imgsz
        
        # Preprocess
        start_time = time.time()
        input_tensor = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        infer_time = time.time() - start_time
        self.inference_times.append(infer_time)
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
        
        # Parse detections based on output format
        raw_output = outputs[0]
        detections = []
        
        if self._output_format == "e2e":
            # End-to-end NMS: [1, num_detections, 6] -> [x1, y1, x2, y2, score, class_id]
            preds = raw_output[0]
            for det in preds:
                score = float(det[4])
                if score < self.conf:
                    continue
                
                x1 = int(det[0] * scale_x)
                y1 = int(det[1] * scale_y)
                x2 = int(det[2] * scale_x)
                y2 = int(det[3] * scale_y)
                class_id = int(det[5])
                
                # Clamp to frame boundaries
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_name = self._get_label(class_id)
                color = self._get_color(class_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_text = f"{class_name} {score:.0%}"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1 - 6, th + 6)
                cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 4, label_y + baseline - 4), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': score,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        elif self._output_format == "v8":
            # YOLOv8 raw: [1, 4+num_classes, num_detections] -> transpose to [num_detections, 4+num_classes]
            preds = raw_output[0].T
            for det in preds:
                x, y, bw, bh = det[:4]
                class_scores = det[4:]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
                
                if score < self.conf:
                    continue
                
                x1 = int((x - bw/2) * scale_x)
                y1 = int((y - bh/2) * scale_y)
                x2 = int((x + bw/2) * scale_x)
                y2 = int((y + bh/2) * scale_y)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_name = self._get_label(class_id)
                color = self._get_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{class_name} {score:.0%}"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1 - 6, th + 6)
                cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 4, label_y + baseline - 4), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': score,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        else:
            # YOLOv5: [1, num_detections, 5+num_classes]
            preds = raw_output[0]
            for det in preds:
                x, y, bw, bh = det[:4]
                obj_conf = det[4]
                class_scores = det[5:]
                class_id = int(np.argmax(class_scores))
                score = float(obj_conf * class_scores[class_id])
                
                if score < self.conf:
                    continue
                
                x1 = int((x - bw/2) * scale_x)
                y1 = int((y - bh/2) * scale_y)
                x2 = int((x + bw/2) * scale_x)
                y2 = int((y + bh/2) * scale_y)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_name = self._get_label(class_id)
                color = self._get_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{class_name} {score:.0%}"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1 - 6, th + 6)
                cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 4, label_y + baseline - 4), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': score,
                    'class_id': class_id,
                    'class_name': class_name
                })

        if detections:
            for d in detections:
                logger.info(f"[DETECT] {d['class_name']} ({d['confidence']:.0%}) at {d['bbox']}")

        return frame, detections
    
    @staticmethod
    def _get_color(class_id):
        """Generate a consistent color for each class ID."""
        # Use golden ratio to spread colors evenly across hue spectrum
        hue = int((class_id * 37) % 180)
        color_hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in color_bgr[0][0])
    
    def get_avg_inference_time(self):
        """Get average inference time over last 30 frames."""
        return sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0


class PtDetector:
    """Ultralytics YOLO .pt model detector with same interface as ONNX Detector."""

    def __init__(
        self,
        model_path,
        imgsz=640,
        conf=0.4,
        use_gpu=True,
        label_path=None
    ):
        """Initialize ultralytics YOLO detector.
        
        Args:
            model_path: Path to .pt model
            imgsz: Input image size (default 640 for .pt)
            conf: Confidence threshold
            use_gpu: Use GPU if available
            label_path: Path to label.yaml (auto-detected if None)
        """
        from ultralytics import YOLO
        
        self.imgsz = imgsz
        self.conf = conf
        self.inference_times = []
        
        # Load class labels from label.yaml
        self.labels = load_labels(label_path)
        
        # Load YOLO model
        logger.info(f"Loading ultralytics model: {model_path}")
        self.model = YOLO(model_path)
        
        # Auto-detect device
        import torch
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"Model loaded: {model_path} (device: {self.device}, imgsz: {imgsz})")

    def _get_label(self, class_id):
        """Get human-readable label for a class ID."""
        if self.labels:
            return self.labels.get(class_id, f"class_{class_id}")
        # Fallback to model's built-in names
        if hasattr(self.model, 'names') and class_id in self.model.names:
            return self.model.names[class_id]
        return f"class_{class_id}"

    def predict(self, frame):
        """Run inference and draw labeled bounding boxes.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            (frame, detections) tuple
        """
        start_time = time.time()
        
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )
        
        infer_time = time.time() - start_time
        self.inference_times.append(infer_time)
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Get bbox coordinates (xyxy format, already in pixel coords)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    score = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    class_name = self._get_label(class_id)
                    color = self._get_color(class_id)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label_text = f"{class_name} {score:.0%}"
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = max(y1 - 6, th + 6)
                    cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 4, label_y + baseline - 4), color, -1)
                    cv2.putText(frame, label_text, (x1 + 2, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': score,
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        if detections:
            for d in detections:
                logger.info(f"[DETECT] {d['class_name']} ({d['confidence']:.0%}) at {d['bbox']}")

        return frame, detections
    
    @staticmethod
    def _get_color(class_id):
        """Generate a consistent color for each class ID."""
        hue = int((class_id * 37) % 180)
        color_hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in color_bgr[0][0])
    
    def get_avg_inference_time(self):
        """Get average inference time over last 30 frames."""
        return sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0