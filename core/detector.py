# core/detector.py

import cv2
import numpy as np
import onnxruntime as ort
import logging
import time

logger = logging.getLogger(__name__)

class Detector:
    """Optimized ONNX detector with GPU support and caching."""

    def __init__(
        self,
        model_path,
        imgsz=640,
        conf=0.4,
        use_gpu=True
    ):
        """Initialize detector.
        
        Args:
            model_path: Path to ONNX model
            imgsz: Input image size
            conf: Confidence threshold
            use_gpu: Use GPU if available
        """
        self.imgsz = imgsz
        self.conf = conf
        self.inference_times = []
        
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

    def predict(self, frame):
        """Optimized inference with minimal operations."""
        h, w = frame.shape[:2]
        
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
        if len(self.inference_times) > 30:  # Keep last 30 measurements
            self.inference_times.pop(0)
        
        # Parse detections
        preds = outputs[0][0] if len(outputs[0].shape) > 2 else outputs[0]
        detections = []
        
        for det in preds:
            score = det[4]
            
            if score < self.conf:
                continue
            
            x, y, bw, bh = det[:4]
            
            # Denormalize coordinates
            x1 = int((x - bw/2) * w/self.imgsz)
            y1 = int((y - bh/2) * h/self.imgsz)
            x2 = int((x + bw/2) * w/self.imgsz)
            y2 = int((y + bh/2) * h/self.imgsz)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{score:.2f}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store detection info
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(score)
            })

        return frame, detections
    
    def get_avg_inference_time(self):
        """Get average inference time over last 30 frames."""
        return sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0