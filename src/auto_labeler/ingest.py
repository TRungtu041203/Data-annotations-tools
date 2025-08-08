"""
auto_labeler.ingest
-------------------
Load a local video, sample frames at a user-defined FPS, optionally resize,
and save frames to disk.

Run from the project root:
    python -m auto_labeler.ingest --video sample.mp4 --out images --fps 5
"""
import numpy as np
import argparse
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

def combine_video_grid(video_paths, out_path, grid_shape=(2,2), resize_width=None):
    caps = cv2.VideoCapture(str(path) for path in video_paths)
    if not all([cap.isOpened() for cap in caps]):
        raise RuntimeError("One or more videos could not be opened.")
    
    frame_counts = min([cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps])
    min_frames = min(frame_counts)
    fps = min([cap.get(cv2.CAP_PROP_FPS) for cap in caps])

    frames = []
    for cap in caps: 
        ret, frame =  cap.read()
        if not ret:
            raise RuntimeError("Could not read frame from video")
        if resize_width:
            height = int(cap.shape[0]*resize_width/cap.shape[1])
            frame=cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)
        frames.append(frames)
    h, w, c = frames[0].shape

    #prepare grid
    grid_h = h * grid_shape[0]
    grid_w = w * grid_shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (grid_w, grid_h))

    for i in range(min_frames):
        grid = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret: 
                frame = np.zeros((h, w, c), dtype=np.uint8)
            if resize_width: 
                height = int(cap.shape[0]*resize_width/cap.shape[1])
                frame = cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)
            row = idx // grid_shape[1]
            col = idx % grid_shape[1]
            y1, y2 = row*h, (row+1)*h
            x1, x2 = col*w, (col+1)*w
            grid[y1:y2, x1:x2] = frame
        out.write(grid)
    for cap in caps: 
        cap.release()
    out.release()
    print(f"[DONE] Saved grid video to {out_path}")

def concatenate_video(video_paths, out_path, resize_width=None):
    caps = [cv2.VideoCapture(str(path)) for path in video_paths]
    if not all([cap.isOpened() for cap in caps]):
        raise RuntimeError("One of the videos can't be opened")
    
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    ret, frame = caps[0].read()
    if not ret: 
        raise RuntimeError("Can't get the 1st frame")
    if resize_width:
        height = int(frame.shape[0] * resize_width / frame.shape[1])
        frame = cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)
    h, w, c = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(str(out_path), fourcc, fps, (w,h))

    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_width:
                height = int(frame.shape[0] * resize_width / frame.shape[1])
                frame = cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)
            out.write(frame)
        cap.release()
    out.release()
    print(f"[DONE] Saved concanated videos to {out_path}")

def extract_frames(
    video_paths: list,
    out_dir: Path,
    target_fps: Optional[float] = None,
    resize_width: Optional[int] = None,
):
    """Extract frames from multiple videos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_saved = 0
    
    for video_idx, video_path in enumerate(video_paths):
        print(f"\n[INFO] Processing video {video_idx + 1}/{len(video_paths)}: {video_path.name}")
        
        # 1. Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_path}")
            continue

        # 2. Gather metadata
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute sampling step
        if target_fps is None or target_fps >= src_fps:
            step = 1
            effective_fps = src_fps  # for progress bar label
        else:
            step = max(1, round(src_fps / target_fps))  # Ensure step >= 1
            effective_fps = target_fps

        print(
            f"[INFO] Source FPS={src_fps:.2f}  Target FPS={effective_fps:.2f}  "
            f"Sampling every {step} frame(s)"
        )

        saved = 0
        with tqdm(total=total_frames // step, unit="frame", desc=f"Video {video_idx + 1}") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Keep only every `step`-th frame
                if frame_idx % step == 0:
                    # Optional resize
                    if resize_width is not None:
                        height = int(frame.shape[0] * resize_width / frame.shape[1])
                        frame = cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)

                    # Save as JPEG with video prefix
                    out_path = out_dir / f"{video_path.stem}_frame_{saved:06d}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                    total_saved += 1
                    pbar.update(1)

                frame_idx += 1

        cap.release()
        print(f"[DONE] Saved {saved} frames from {video_path.name}")
    
    print(f"\n[DONE] Total frames saved: {total_saved} to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Video processing utility.")
    parser.add_argument("--mode", choices=["extract", "grid", "concat"], default="extract",
                        help="Operation mode: extract frames, combine grid, or concatenate.")
    parser.add_argument("--video", type=Path, help="Path to input video (for extract - single video).")
    parser.add_argument("--videos", type=Path, nargs="+", help="Paths to input videos (for all modes).")
    parser.add_argument("--out", type=Path, default=Path("images"), help="Output folder or file.")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS after down-sampling (extract only).")
    parser.add_argument("--resize-width", type=int, default=None, help="Resize width in pixels (keeps aspect ratio).")
    parser.add_argument("--grid-shape", type=int, nargs=2, default=[2,2], help="Grid shape for grid mode (rows cols).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "extract":
        # Support both --video (single) and --videos (multiple)
        if args.videos:
            video_paths = args.videos
        elif args.video:
            video_paths = [args.video]
        else:
            raise ValueError("Must specify either --video or --videos for extract mode")
        
        extract_frames(
            video_paths=video_paths,
            out_dir=args.out,
            target_fps=args.fps,
            resize_width=args.resize_width,
        )
    elif args.mode == "grid":
        combine_video_grid(
            video_paths=args.videos,
            out_path=args.out,
            grid_shape=tuple(args.grid_shape),
            resize_width=args.resize_width,
        )
    elif args.mode == "concat":
        concatenate_video(
            video_paths=args.videos,
            out_path=args.out,
            resize_width=args.resize_width,
        )
