import sys
from pathlib import Path
import numpy as np
import torch
import supervision as sv
import cv2
import json

# Add grounding_dino to path temporarily
sys.path.append(str(Path(__file__).parents[2] / "external" / "grounded-sam-2"))
from grounding_dino.groundingdino.util.inference import predict

__all__ = ["run_dino_on_rgb"]

def run_dino_on_rgb(det_model, rgb, prompt, conf_th, text_th, device):
    """
    One-shot Grounding-DINO → Supervision.Detections list
    (keeps detector glue in one place).
    """
    H, W = rgb.shape[:2]
    tensor = (
        torch.from_numpy(rgb)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    ).to(device)

    boxes, scores, _ = predict(
        model=det_model,
        image=tensor[0],
        caption=prompt,
        box_threshold=conf_th,
        text_threshold=text_th,
        device=device,
    )

    # cxcywh → xyxy (pixels)
    cx, cy, w, h = (boxes * torch.tensor([W, H, W, H], device=boxes.device)).unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

    return sv.Detections(
        xyxy=xyxy.cpu().numpy(),
        confidence=scores.cpu().numpy(),
        class_id=np.zeros(len(scores), dtype=int),
    )

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
    
    # Resize background to match frame dimensions
    bg_resized = cv2.resize(bg_rgb, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    
    fused = bg_resized.copy()
    obj = frame_rgb.copy()

    # For each instance mask
    for m in masks:
        # 3-channel boolean indexing
        fused[m] = (alpha * obj[m] + (1 - alpha) * fused[m]).astype(np.uint8)

    return fused

def visualize_detections(img_rgb, boxes_xyxy, masks, logits, phrases):
    """Create visualization with bounding boxes, masks, and confidence scores."""
    # First, blend the mask overlay with the original image
    overlay = img_rgb.copy()
    
    for i, (box, score, phrase) in enumerate(zip(boxes_xyxy, logits, phrases)):
        # Generate a unique color for each detection
        color = ((i * 50 + 100) % 255, (i * 100 + 100) % 255, (i * 150 + 100) % 255)
        
        # Masks are guaranteed to be 3D (N, H, W) after normalization
        mask = masks[i].astype(bool)
        
        # Draw mask with color overlay
        overlay[mask] = np.array(color)
    
    # Blend the mask overlay with the original image (with opacity)
    alpha = 0.3
    vis_img = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
    
    # Now draw clear bounding boxes and text on top (no opacity)
    for i, (box, score, phrase) in enumerate(zip(boxes_xyxy, logits, phrases)):
        # Generate the same unique color for each detection
        color = ((i * 50 + 100) % 255, (i * 100 + 100) % 255, (i * 150 + 100) % 255)
        
        # Draw bounding box
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_rgb.shape[1]))
        y1 = max(0, min(y1, img_rgb.shape[0]))
        x2 = max(0, min(x2, img_rgb.shape[1]))
        y2 = max(0, min(y2, img_rgb.shape[0]))
        
        # Draw clear bounding box with thinner lines
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 1)
        
        font_scale = 0.4
        thickness = 1
        # Add confidence and phrase text
        text = f"{phrase}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Background rectangle for text
        cv2.rectangle(vis_img, (x1, y1 - text_size[1] - 4), 
                     (x1 + text_size[0] - 2, y1), color, -1)
        
        # Text
        cv2.putText(vis_img, text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return vis_img