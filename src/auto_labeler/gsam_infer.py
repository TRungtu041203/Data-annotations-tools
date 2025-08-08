"""
gsam_infer.py  —  Phase-2 demo
Run Grounded-SAM 2 on a folder of JPEGs and dump raw detections to JSON.

Example:
    python -m auto_labeler.gsam_infer ^
        --images images ^
        --out detections ^
        --conf 0.35 ^
        --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

from auto_labeler.utils import run_dino_on_rgb, load_masks, visualize_detections
from auto_labeler.bg_fused import load_masks, fuse_on_bg
import supervision as sv

# ---------- helpers ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # .../auto_labeler
EXT_DIR = PROJECT_ROOT / "external" / "grounded-sam-2"
CFG_PATH = (
    EXT_DIR
    / "grounding_dino"
    / "groundingdino"
    / "config"
    / "GroundingDINO_SwinB_cfg.py"
)

# --- Grounded-SAM 2 imports ---
import sys
sys.path.append(str(PROJECT_ROOT / "external" / "grounded-sam-2"))
sys.path.append(str(PROJECT_ROOT / "external" / "grounded-sam-2" / "grounding_dino"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict


def load_models(dino_ckpt: Path, sam_ckpt: Path, device: str = "cuda"):
    """Load Grounding DINO + SAM-2 once."""
    # Load Grounding DINO
    det = load_model(str(CFG_PATH), str(dino_ckpt))
    det.to(device).eval()

    # Load SAM-2
    sam_config = EXT_DIR / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(str(sam_config), str(sam_ckpt), device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    return det, predictor

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# ---------- core loop -------------------------------------------------------
def infer_folder(
    img_dir: Path,
    out_dir: Path,
    det,
    predictor,
    conf_thresh: float,
    text_thresh: float,
    use_slice: bool,
    slice_size: int,
    slice_overlap: float,
    max_detections: int = float('inf'), # Default to infinity
    nms_threshold: float = 0.5,  # NMS IoU threshold
    enable_viz: bool = True,
    bg_dir: Path = None,
    bg_name: str = None,
    device: str = "cuda",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(
        [*img_dir.glob("*.jpg"), *img_dir.glob("*.png"), *img_dir.glob("*.jpeg")]
    )

    print(f"Found {len(frames)} images in {img_dir}")
    print(f"Out directory: {out_dir}")

    # Track processed images for periodic cleanup
    processed_count = 0

    for img_path in tqdm(frames, desc="Grounded-SAM 2"):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        print(f"Processing {img_path.name} (size: {W}x{H})")

        bg_rgb = None
        if bg_dir is not None and bg_name is not None:
            bg_path = Path(bg_dir) / bg_name
            if bg_path.exists():
                bg_bgr = cv2.imread(str(bg_path))
                bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            else:
                print(f"Warning: Background image {bg_path} not found. Skipping fusion.")


        # Clear GPU cache before each image
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Remove unnecessary tensor creation - we don't use img_tensor
        # img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        # img_tensor = img_tensor.unsqueeze(0).to(device)

        # Grounding DINO → open-vocabulary boxes
        prompt = "person"

        if use_slice:
            # For batch processing, recreate the slicer for each image to avoid state issues
            def process_slice(img_slice):
                try:
                    # Convert BGR to RGB using OpenCV to avoid negative stride issues
                    img_rgb_slice = cv2.cvtColor(img_slice, cv2.COLOR_BGR2RGB)
                    
                    # Check if slice is too small (can cause issues)
                    if img_rgb_slice.shape[0] < 50 or img_rgb_slice.shape[1] < 50:
                        return sv.Detections.empty()
                    
                    # Clear GPU cache before processing each slice
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    
                    detections = run_dino_on_rgb(det, img_rgb_slice, prompt, conf_thresh, text_thresh, device)
                    
                    # Additional validation
                    if detections is None or len(detections.xyxy) == 0:
                        return sv.Detections.empty()
                    
                    return detections
                    
                except Exception as e:
                    # Return empty detections if processing fails
                    return sv.Detections.empty()
            
            # Force garbage collection before slice processing
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            
            try:
                # Create a fresh slicer for each image to avoid state accumulation
                slicer = sv.InferenceSlicer(
                    callback=process_slice,
                    slice_wh=(slice_size, slice_size),
                    overlap_wh=(int(slice_size * slice_overlap), int(slice_size * slice_overlap)),
                    overlap_ratio_wh=None,
                    iou_threshold=0.5,
                )
                
                print(f"  Running slice inference with {slice_size}x{slice_size} slices...")
                dets = slicer(img_bgr)
                
                # Force cleanup of the slicer
                del slicer
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                boxes_xyxy = torch.tensor(dets.xyxy, device=device)
                logits      = torch.tensor(dets.confidence, device=device)
                phrases     = ["person"] * len(boxes_xyxy)
                
                print(f"  Slice inference found {len(boxes_xyxy)} detections")
                
                # If slice inference produces no results, fall back to regular inference
                if len(boxes_xyxy) == 0:
                    print(f"  Slice inference found no detections, falling back to regular inference...")
                    dets = run_dino_on_rgb(det, img_rgb, prompt, conf_thresh, text_thresh, device)
                    boxes_xyxy = torch.tensor(dets.xyxy, device=device)
                    logits      = torch.tensor(dets.confidence, device=device)
                    phrases     = ["person"] * len(boxes_xyxy)
                    
            except Exception as e:
                print(f"  Slice inference failed completely, falling back to regular inference: {str(e)}")
                # Force cleanup on failure
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                dets = run_dino_on_rgb(det, img_rgb, prompt, conf_thresh, text_thresh, device)
                boxes_xyxy = torch.tensor(dets.xyxy, device=device)
                logits      = torch.tensor(dets.confidence, device=device)
                phrases     = ["person"] * len(boxes_xyxy)
        else:
            dets = run_dino_on_rgb(det, img_rgb, prompt,
                                  conf_thresh, text_thresh, device)
            boxes_xyxy = torch.tensor(dets.xyxy, device=device)
            logits      = torch.tensor(dets.confidence, device=device)
            phrases     = ["person"] * len(boxes_xyxy)
            
            # Debug: Print raw detection info
            print(f" Raw detections before filtering: {len(dets.xyxy)}")
            if len(dets.confidence) > 0:
                print(f" Confidence range: {dets.confidence.min():.3f} - {dets.confidence.max():.3f}")
            else:
                print(f" No detections found by Grounding DINO")


        print(f" Found {len(boxes_xyxy)} detections with confidence > {conf_thresh}")

        if len(boxes_xyxy) == 0:
            print(f" Skipping {img_path.name} - no detections")
            continue

        # Apply Non-Maximum Suppression to remove overlapping detections
        if len(boxes_xyxy) > 1:
            print(f" Applying NMS to remove overlapping detections...")
            
            # Convert to supervision Detections for NMS
            detections_for_nms = sv.Detections(
                xyxy=boxes_xyxy.cpu().numpy(),
                confidence=logits.cpu().numpy(),
                class_id=np.zeros(len(boxes_xyxy), dtype=int)
            )
            
            # Apply NMS with IoU threshold of 0.5
            nms_detections = detections_for_nms.with_nms(threshold=nms_threshold)
            
            # Update our detection arrays
            boxes_xyxy = torch.tensor(nms_detections.xyxy, device=device)
            logits = torch.tensor(nms_detections.confidence, device=device)
            phrases = ["person"] * len(boxes_xyxy)
            
            print(f" After NMS: {len(boxes_xyxy)} detections remaining")

        # Limit the number of detections to process (top N by confidence)
        if len(boxes_xyxy) > max_detections:
            print(f" Limiting to top {max_detections} detections (sorted by confidence)")
            # Sort by confidence (logits) and keep top N
            top_indices = torch.argsort(logits, descending=True)[:max_detections]
            boxes_xyxy = boxes_xyxy[top_indices]
            logits = logits[top_indices]
            phrases = [phrases[i] for i in top_indices]

        print(f" Processing {len(boxes_xyxy)} detections")

        # Ensure boxes are on the same device
        boxes_xyxy = boxes_xyxy.to(device)
        
        # Note: Both slice and non-slice inference return boxes in xyxy format
        # - Slice inference: supervision InferenceSlicer returns xyxy
        # - Non-slice inference: run_dino_on_rgb already converts cxcywh to xyxy
        # So no additional conversion is needed

        # SAM-2 masks - Process in batches to avoid memory issues
        print(f" Generating masks for {len(boxes_xyxy)} detections...")
        predictor.set_image(img_rgb)
        
        # Process masks in smaller batches to avoid memory issues
        batch_size = 10  # Process 10 boxes at a time
        all_masks = []
        all_scores = []
        
        for i in range(0, len(boxes_xyxy), batch_size):
            batch_boxes = boxes_xyxy[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(boxes_xyxy) + batch_size - 1)//batch_size}")
            
            masks, scores, _ = predictor.predict(
                box=batch_boxes.cpu().numpy(),
                multimask_output=False,
            )
            
            all_masks.append(masks)
            all_scores.append(scores)
            
            # Clear GPU cache after each batch
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate all results with proper dimension handling
        if len(all_masks) > 1:
            normalized_masks = []
            normalized_scores = []
            
            for mask_batch, score_batch in zip(all_masks, all_scores):
                # Normalize mask dimensions
                if mask_batch.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
                    mask_batch = mask_batch.squeeze(1)
                normalized_masks.append(mask_batch)
                
                # Normalize score dimensions
                if score_batch.ndim == 2:  # (N, 1) -> (N,)
                    score_batch = score_batch.squeeze(1)
                elif score_batch.ndim == 0:  # scalar -> (1,)
                    score_batch = np.array([score_batch])
                normalized_scores.append(score_batch)

            masks = np.concatenate(normalized_masks, axis=0)
            scores = np.concatenate(normalized_scores, axis=0)
        else:
            masks = all_masks[0]
            scores = all_scores[0]
            
            # Ensure consistent dimensions for single batch
            if masks.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
                masks = masks.squeeze(1)
            if scores.ndim == 2:  # (N, 1) -> (N,)
                scores = scores.squeeze(1)
            elif scores.ndim == 0:  # scalar -> (1,)
                scores = np.array([scores])
        
        if bg_rgb is not None and masks is not None and masks.shape[0] > 0:
            fused_img = fuse_on_bg(img_rgb, bg_rgb, masks, alpha = 0.8)
            fused_bgr = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_fused.jpg"), fused_bgr)
            print(f" Saved fused image to {img_path.stem}_fused.jpg")

        # save raw detections
        out = []
        for i, (b, s) in enumerate(zip(boxes_xyxy, logits)):
            # Masks are now guaranteed to be 3D (N, H, W) after normalization
            mask = masks[i]
            mask_rle = single_mask_to_rle(mask.astype(bool))
        
            out.append(
                {
                    "class_name": phrases[i] if i < len(phrases) else "person",
                    "box": [float(x) for x in b.tolist()],
                    "segmentation": mask_rle,
                    "score": float(s),
                }
            )
        
        results = {
            "image_path": str(img_path),
            "annotations": out,
            "box_format": "xyxy",
            "img_width": W,
            "img_height": H,
        }
        
        print(f" Saved {len(out)} detections to {img_path.stem}.json")

        # Save JSON file
        with open(out_dir / f"{img_path.stem}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Create and save visualization only if enabled
        if enable_viz:
            vis_img = visualize_detections(img_rgb, boxes_xyxy, masks, logits, phrases)
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_visualized.jpg"), vis_img_bgr)
            print(f" Saved visualization to {img_path.stem}_visualized.jpg")
        else:
            print(f" Skipped visualization (--no-viz enabled)")

        # Aggressive cleanup at the end of each image
        del masks, all_masks, all_scores, boxes_xyxy, logits, img_rgb, img_bgr
        if 'dets' in locals():
            del dets
        if 'phrases' in locals():
            del phrases
        
        # Force garbage collection
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        processed_count += 1
        
        # Periodic deep cleanup every 50 images
        if processed_count % 50 == 0:
            print(f"\n  === Periodic cleanup after {processed_count} images ===")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                # Force CUDA memory cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            print(f"  === Cleanup complete, continuing... ===\n")


# ---------- CLI -------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("detections"))
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--text_th", type=float, default=0.25, help="Text threshold for Grounding DINO")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-detections", type=int, default=float('inf'), help="Maximum number of detections to process per image. Use 'inf' for all detections.")
    p.add_argument("--nms-threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression (0.0-1.0)")
    p.add_argument("--dino-ckpt", type=Path, default=EXT_DIR / "gdino_checkpoints" / "groundingdino_swinb_cogcoor.pth")
    p.add_argument("--sam-ckpt", type=Path, default=EXT_DIR / "checkpoints" / "sam2.1_hiera_large.pt")
    p.add_argument("--with-slice", action="store_true", help="Enable slice-inference (Supervision Inference Slicer)")
    p.add_argument("--slice-size", type=int, default=0, help="0 = disable SAHI")
    p.add_argument("--slice-overlap", type=float, default=0.2, help="Fractional overlap between adjacent tiles (0-1)")
    p.add_argument("--no-viz", action="store_true", help="Skip visualization generation (faster processing)")
    p.add_argument("--bg-dir", type=Path, default=None, help = "Folder of empty background images (skip if None)")
    p.add_argument("--bg-name", type=str, default=None, help="Filename of BG to use (if you have >1 bg_imgs)")
    return p.parse_args()


def main():
    args = parse_cli()
    
    # Check GPU availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[INFO] Using CPU")
    
    det, predictor = load_models(args.dino_ckpt, args.sam_ckpt, args.device)
    t0 = time.time()
    infer_folder(
        args.images,
        args.out,
        det,
        predictor,
        args.conf,
        args.text_th,
        args.with_slice,       # maps to use_slice
        args.slice_size,       # maps to slice_size
        args.slice_overlap,    # maps to slice_overlap
        args.max_detections,
        args.nms_threshold,    # maps to nms_threshold
        not args.no_viz,       # enable_viz = not no_viz
        args.bg_dir,
        args.bg_name,       
        args.device,
    )
    print(f"[DONE] Inference in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

