# ui/main_window.py

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
import logging
from queue import Queue
from pathlib import Path

from core.camera import Camera, VideoFile, get_available_cameras
from core.detector import Detector
from core.monitor import Monitor
from core.overlay import draw_overlay

logger = logging.getLogger(__name__)

class MainWindow:
    """Main application window with optimized threading and rendering."""

    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection - YOLO Monitor")
        self.root.geometry("1024x640")
        
        # Thread-safe queue for frame communication
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.camera = None
        self.detector = None
        self.monitor = None
        
        # Input source selection
        self.input_type = tk.StringVar(value="camera")  # "camera" or "video"
        self.video_file_path = tk.StringVar(value="")
        
        self._setup_ui()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_ui(self):
        """Setup UI components with proper layout."""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video display label
        self.label = tk.Label(main_frame, bg='black')
        self.label.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Input type selection
        tk.Label(settings_frame, text="Input:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(settings_frame, text="Camera", variable=self.input_type, value="camera", font=('Arial', 9), command=self._on_input_type_changed).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(settings_frame, text="Video File", variable=self.input_type, value="video", font=('Arial', 9), command=self._on_input_type_changed).pack(side=tk.LEFT, padx=2)
        
        # Camera selection
        tk.Label(settings_frame, text="Camera:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(settings_frame, textvariable=self.camera_var, width=15, state='readonly', font=('Arial', 9))
        self.camera_combo.pack(side=tk.LEFT, padx=2)
        self._update_camera_list()
        
        # Video file selection
        tk.Label(settings_frame, text="Video:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.video_label = tk.Label(settings_frame, text="(No file selected)", fg='gray', font=('Arial', 9), width=20)
        self.video_label.pack(side=tk.LEFT, padx=2)
        tk.Button(settings_frame, text="Browse...", command=self._select_video_file, font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # Model selection
        tk.Label(settings_frame, text="Model:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(settings_frame, textvariable=self.model_var, width=15, state='readonly', font=('Arial', 9))
        self.model_combo.pack(side=tk.LEFT, padx=2)
        self._update_model_list()
        
        # Refresh button for camera detection
        tk.Button(settings_frame, text="🔄 Refresh", command=self._update_camera_list, font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons
        self.btn_play = tk.Button(
            control_frame,
            text="▶ Play",
            command=self.start,
            width=15,
            height=2,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(
            control_frame,
            text="■ Stop",
            command=self.stop,
            width=15,
            height=2,
            bg='#f44336',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Status: Ready",
            fg='green',
            font=('Arial', 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _on_input_type_changed(self):
        """Handle input type change (camera vs video)."""
        if self.running:
            messagebox.showwarning("Warning", "Cannot change input while detecting. Stop first.")
            return
        
        input_type = self.input_type.get()
        if input_type == "camera":
            self.camera_combo.config(state='readonly')
            logger.info("Input source: Camera")
        else:
            if not self.video_file_path.get():
                messagebox.showwarning("Warning", "Please select a video file first")
                self.input_type.set("camera")
                return
            logger.info(f"Input source: Video file - {self.video_file_path.get()}")
    
    def _select_video_file(self):
        """Open file dialog to select video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_file_path.set(file_path)
            file_name = Path(file_path).name
            self.video_label.config(text=file_name, fg='black')
            self.input_type.set("video")
            self._on_input_type_changed()
            logger.info(f"Video file selected: {file_path}")
    
    def _update_camera_list(self):
        """Update available cameras list (runs scan in background thread)."""
        self.camera_combo['values'] = ["Scanning cameras..."]
        self.camera_combo.current(0)
        self.camera_combo.config(state='disabled')
        self.available_cameras = []
        
        def _scan():
            cameras = get_available_cameras()
            # Schedule UI update on main thread
            self.root.after(0, lambda: self._on_cameras_found(cameras))
        
        scan_thread = threading.Thread(target=_scan, daemon=True)
        scan_thread.start()
    
    def _on_cameras_found(self, cameras):
        """Callback when camera scan completes (runs on main thread)."""
        if cameras:
            camera_options = [name for _, name in cameras]
            self.camera_combo['values'] = camera_options
            self.camera_combo.current(0)
            self.available_cameras = cameras
            self.camera_combo.config(state='readonly')
            logger.info(f"Camera list updated: {len(cameras)} cameras found")
        else:
            self.camera_combo['values'] = ["No cameras found"]
            self.camera_combo.current(0)
            self.available_cameras = []
            self.camera_combo.config(state='readonly')
            logger.warning("No cameras found during scan")
    
    def _update_model_list(self):
        """Update available models list."""
        models_dir = Path("models")
        onnx_files = []
        if models_dir.exists():
            onnx_files = sorted([f.name for f in models_dir.glob("*.onnx")])
        
        if onnx_files:
            self.model_combo['values'] = onnx_files
            self.model_combo.current(0)
        else:
            self.model_combo['values'] = ["No models found"]
            self.model_combo.current(0)
        
        self.available_models = onnx_files
    
    def _init_components(self):
        """Initialize camera/video, detector, and monitor."""
        try:
            self.status_label.config(text="Status: Initializing...", fg='orange')
            self.root.update()
            
            # Initialize input source (camera or video)
            input_type = self.input_type.get()
            if input_type == "camera":
                if not self.available_cameras:
                    raise RuntimeError("No cameras available")
                selected_camera_idx = self.camera_combo.current()
                camera_id = self.available_cameras[selected_camera_idx][0]
                logger.info(f"Initializing camera {camera_id}...")
                self.camera = Camera(cam_id=camera_id, width=640, height=480, fps=30)
            else:
                video_path = self.video_file_path.get()
                if not video_path:
                    raise RuntimeError("No video file selected")
                logger.info(f"Loading video file: {video_path}...")
                self.camera = VideoFile(file_path=video_path)
            
            # Get selected model
            if not self.available_models:
                raise FileNotFoundError("No ONNX models found in models/ directory")
            selected_model = self.model_combo.get()
            model_path = Path("models") / selected_model
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            logger.info(f"Loading model: {selected_model}...")
            self.detector = Detector(
                str(model_path),
                imgsz=640,
                conf=0.4,
                use_gpu=True
            )
            
            logger.info("Initializing monitor...")
            self.monitor = Monitor(window_size=30)
            
            self.status_label.config(text="Status: Ready", fg='green')
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.status_label.config(text=f"Status: Error - {str(e)[:50]}", fg='red')
            messagebox.showerror("Initialization Error", f"Failed to initialize:\n{e}")
            raise
    
    def start(self):
        """Start the application."""
        if self.running:
            messagebox.showwarning("Warning", "Already running")
            return
        
        try:
            self._init_components()
            self.running = True
            self.btn_play.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running", fg='green')
            
            # Start inference thread
            inference_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True
            )
            inference_thread.start()
            
            # Start UI update loop
            self._ui_update_loop()
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            self.running = False
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
    
    def stop(self):
        """Stop the application."""
        self.running = False
        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", fg='red')
        
        if self.camera:
            self.camera.release()
        logger.info("Application stopped")
    
    def _inference_loop(self):
        """Continuous inference loop in separate thread."""
        failed_reads = 0
        max_failed_reads = 30
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    failed_reads += 1
                    if failed_reads > max_failed_reads:
                        logger.error(f"Camera disconnected or not responding (failed {failed_reads} reads)")
                        self.running = False
                        self.root.after(0, self._update_status)
                        break
                    continue
                
                failed_reads = 0  # Reset on successful read
                
                # Run detection
                start_time = time.time()
                frame, detections = self.detector.predict(frame)
                latency = time.time() - start_time
                
                # Get metrics
                metrics = self.monitor.get_metrics(latency)
                
                # Draw overlay
                frame = draw_overlay(frame, metrics)
                
                # Put frame in queue (drop old frames if queue is full)
                try:
                    self.frame_queue.put_nowait((frame, metrics))
                except:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame, metrics))
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                time.sleep(0.1)
    
    def _ui_update_loop(self):
        """Update UI with latest frame from queue."""
        if not self.running:
            return
        
        try:
            frame, metrics = self.frame_queue.get_nowait()
            
            # Convert BGR to RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display if needed
            display_size = (1024, 640)
            rgb = cv2.resize(rgb, display_size)
            
            # Convert to PIL and display
            image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image)
            
            self.label.config(image=photo)
            self.label.image = photo
            
        except:
            pass
        
        # Schedule next update
        if self.running:
            self.root.after(30, self._ui_update_loop)  # ~30fps UI update
    
    def on_closing(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to exit?"):
            self.running = False
            if self.camera:
                self.camera.release()
            self.root.destroy()
            logger.info("Application closed")