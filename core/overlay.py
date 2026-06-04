# core/overlay.py

import cv2
import numpy as np

def draw_overlay(frame, metrics):
    """Draw performance metrics overlay on frame.
    
    Args:
        frame: Input frame
        metrics: Dict with system metrics
        
    Returns:
        Frame with overlay drawn
    """
    # Create background for text (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (480, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Format metrics text
    text = (
        f"CPU:{metrics['cpu']}% | "
        f"RAM:{metrics['ram']}% | "
        f"FPS:{metrics['fps']} | "
        f"LAT:{metrics['latency']}ms"
    )
    
    # Draw text with better visibility
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    
    return frame