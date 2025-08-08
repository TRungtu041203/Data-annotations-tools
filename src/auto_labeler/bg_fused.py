"""
bg_fuse.py
----------
Take a frame, its SAM masks, and a chosen background image → composite frame.

Usage (inside Phase-2 loop):
    fused = fuse_on_bg(frame_rgb, bg_rgb, mask_stack, alpha=0.7)
"""

from pathlib import Path
import cv2
import numpy as np
import json

def load_masks(json_path: Path) -> np.ndarray:
    """Return boolean mask stack with shape (N, H, W)."""
    import pycocotools.mask as mask_util
    
    data = json.loads(Path(json_path).read_text())
    annotations = data["annotations"]
    img_height = data["img_height"]
    img_width = data["img_width"]
    
    masks = []
    for obj in annotations:
        # Decode RLE mask
        rle = obj["segmentation"]
        mask = mask_util.decode(rle).astype(bool)
        masks.append(mask)
    
    if len(masks) == 0:
        # Return empty mask array if no detections
        return np.array([]).reshape(0, img_height, img_width)
    
    masks = np.stack(masks)  # (N, H, W)
    return masks

def fuse_on_bg(frame_rgb: np.ndarray,
               bg_rgb: np.ndarray,
               masks: np.ndarray,
               alpha: float = 0.7) -> np.ndarray:
    """
    Copy every object pixel from frame → BG.
    alpha < 1 ⇒ slight blend to soften hard edges.
    """
    # Get frame dimensions
    frame_h, frame_w = frame_rgb.shape[:2]
    
    # Resize background to match frame dimensions if needed
    if bg_rgb.shape[:2] != frame_rgb.shape[:2]:
        bg_resized = cv2.resize(bg_rgb, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    else:
        bg_resized = bg_rgb.copy()
    
    fused = bg_resized.copy().astype(np.float32)
    frame_float = frame_rgb.astype(np.float32)

    # For each instance mask
    for i in range(masks.shape[0]):
        mask = masks[i].astype(bool) #ensure boolean type
        #Blend: fused = alpha * frame + (1-alpha) * fused
        # 3-channel boolean indexing
        fused[mask] = (alpha * frame_float[mask] + (1 - alpha) * fused[mask])

    return fused.astype(np.uint8)
