"""
auto_labeler.ingest
-------------------
Load a local video, sample frames at a user-defined FPS, optionally resize,
and save frames to disk.

Run from the project root:
    python -m auto_labeler.ingest --video sample.mp4 --out images --fps 5
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: Optional[float] = None,
    resize_width: Optional[int] = None,
):
    """
    Parameters
    ----------
    video_path : Path
        Path to the input video file.
    out_dir : Path
        Directory where JPEG frames will be saved.
    target_fps : float or None
        Desired output FPS. If None, keep every frame.
    resize_width : int or None
        If set, resize the **width** to this many pixels while
        keeping aspect ratio (height auto-scales).
    """
    # 1. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    # 2. Gather metadata
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute sampling step
    if target_fps is None or target_fps >= src_fps:
        step = 1
        target_fps = src_fps  # for progress bar label
    else:
        step = round(src_fps / target_fps)

    print(
        f"[INFO] Source FPS={src_fps:.2f}  Target FPS={target_fps:.2f}  "
        f"Sampling every {step} frame(s)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with tqdm(total=total_frames // step, unit="frame") as pbar:
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

                # Save as JPEG
                out_path = out_dir / f"frame_{saved:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                pbar.update(1)

            frame_idx += 1

    cap.release()
    print(f"[DONE] Saved {saved} frames to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a local video file.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video.")
    parser.add_argument("--out", type=Path, default=Path("images"), help="Output folder.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS after down-sampling (default: keep original).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Resize width in pixels (keeps aspect ratio).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(
        video_path=args.video,
        out_dir=args.out,
        target_fps=args.fps,
        resize_width=args.resize_width,
    )
