# scripts/test_fuse_one.py  (run inside conda env)
import cv2, json, numpy as np
from pathlib import Path
from auto_labeler.bg_fused import load_masks, fuse_on_bg

frame = cv2.cvtColor(cv2.imread("originals/afternoon/frame_000028.jpg"), cv2.COLOR_BGR2RGB)
bg    = cv2.cvtColor(cv2.imread("background/bg_afternoon.jpg"), cv2.COLOR_BGR2RGB)

masks = load_masks(Path("output/det_mid/frame_000028.json"))
fused = fuse_on_bg(frame, bg, masks, alpha=0.7)

# Create output directory if it doesn't exist
Path("bgfused").mkdir(exist_ok=True)
cv2.imwrite("bgfused/frame_000028.jpg", cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))
