# core/camera.py

import cv2
import logging
import time
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def get_available_cameras(max_cameras=10):
    """Detect available cameras on the system using multiple backends.
    
    Tries DirectShow, MSMF, and default backends for maximum compatibility.
    Stops early after consecutive failures to avoid long scan times.
    
    Args:
        max_cameras: Maximum camera indices to check
        
    Returns:
        List of tuples (camera_id, name)
    """
    available_cameras = []
    consecutive_failures = 0
    max_consecutive_failures = 3  # Stop after 3 consecutive misses
    
    # Backends to try, in order of preference on Windows
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    logger.info(f"Scanning for available cameras (checking IDs 0-{max_cameras-1})...")
    
    for i in range(max_cameras):
        found = False
        
        for backend, backend_name in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Try to actually read a frame to confirm camera works
                    ret, _ = cap.read()
                    if ret:
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        camera_name = f"Camera {i} ({int(width)}x{int(height)}@{int(fps)}fps)"
                        available_cameras.append((i, camera_name))
                        logger.info(f"Found: {camera_name} (backend: {backend_name})")
                        found = True
                        break  # Found with this backend, no need to try others
                    else:
                        logger.debug(f"Camera {i} opened with {backend_name} but read failed")
            except Exception as e:
                logger.debug(f"Camera {i} failed with {backend_name}: {e}")
            finally:
                # Always release the capture to prevent handle leaks
                if cap is not None:
                    cap.release()
        
        if found:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures and i >= 1:
                logger.info(f"Stopping scan after {max_consecutive_failures} consecutive failures at index {i}")
                break
        
        time.sleep(0.05)  # Small delay between camera checks
    
    if not available_cameras:
        logger.warning("No cameras detected!")
    else:
        logger.info(f"Total cameras found: {len(available_cameras)}")
    
    return available_cameras


def get_windows_cameras_wmi():
    """Detect cameras using Windows WMI (more reliable for USB cameras).
    
    Returns:
        List of camera names
    """
    if platform.system() != "Windows":
        return []
    
    try:
        output = subprocess.check_output(
            'wmic path win32_pnpdevice where "Description like \'%camera%\'" get Name',
            shell=True,
            stderr=subprocess.DEVNULL,
            text=True
        )
        cameras = [line.strip() for line in output.split('\n') if line.strip() and line.strip() != 'Name']
        logger.info(f"WMI detected cameras: {cameras}")
        return cameras
    except Exception as e:
        logger.debug(f"WMI camera detection failed: {e}")
        return []

class Camera:
    """Optimized camera wrapper with configurable parameters."""

    def __init__(self, cam_id=0, width=640, height=480, fps=30):
        """Initialize camera with optimal settings.
        
        Args:
            cam_id: Camera device ID
            width: Frame width
            height: Frame height  
            fps: Target FPS
        """
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps
        self.read_failures = 0
        self.max_retries = 3
        
        # Try DirectShow backend first (better USB camera support)
        logger.info(f"Opening camera {cam_id} with DirectShow backend...")
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        
        # If DirectShow fails, fallback to auto-detect backend
        if not self.cap.isOpened():
            logger.warning(f"DirectShow failed for camera {cam_id}, trying auto-detect...")
            self.cap = cv2.VideoCapture(cam_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {cam_id}")
            raise RuntimeError(f"Cannot open camera {cam_id}")
        
        logger.info(f"Camera {cam_id} opened successfully")
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Add delay before accessing properties (fixes some USB camera issues)
        time.sleep(0.2)
        
        # Windows MSMF compatibility: try different buffer sizes
        # Don't set buffer size on all cameras - can cause issues
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        
        # Warm up camera - read a few frames to initialize properly
        logger.info("Warming up camera...")
        for _ in range(5):
            ret, _ = self.cap.read()
            if ret:
                time.sleep(0.05)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera initialized: {self.width}x{self.height}@{fps}fps")

    def read(self, max_retries=2):
        """Read frame from camera with retry logic.
        
        Args:
            max_retries: Number of retries on read failure
            
        Returns:
            (ret, frame) tuple
        """
        for attempt in range(max_retries):
            ret, frame = self.cap.read()
            
            if ret:
                self.read_failures = 0
                return ret, frame
            
            # Retry with small delay
            if attempt < max_retries - 1:
                time.sleep(0.01)
                logger.debug(f"Camera read retry {attempt + 1}/{max_retries}")
        
        # Track consecutive failures
        self.read_failures += 1
        if self.read_failures > 30:
            logger.error(f"Camera read failed {self.read_failures} times - possible disconnect")
        
        return False, None

    def release(self):
        """Release camera resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released")


class VideoFile:
    """Video file reader with same interface as Camera for compatibility."""
    
    def __init__(self, file_path, width=640, height=480, fps=30):
        """Initialize video file reader.
        
        Args:
            file_path: Path to video file
            width: Frame width for display (ignored, uses video resolution)
            height: Frame height for display (ignored, uses video resolution)
            fps: Target FPS (ignored, uses video FPS)
        """
        self.file_path = file_path
        self.read_failures = 0
        self.total_frames = 0
        self.current_frame = 0
        
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {file_path}")
            raise RuntimeError(f"Cannot open video file: {file_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video file opened: {self.file_path}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}")
        logger.info(f"  Total frames: {self.total_frames}")
    
    def read(self, max_retries=1):
        """Read frame from video file.
        
        Args:
            max_retries: Number of retries on read failure
            
        Returns:
            (ret, frame) tuple
        """
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame += 1
            self.read_failures = 0
            return ret, frame
        
        # Video ended
        self.read_failures += 1
        if self.read_failures == 1:
            logger.info(f"Video playback finished: {self.current_frame}/{self.total_frames} frames")
        
        return False, None
    
    def release(self):
        """Release video file resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Video file released")