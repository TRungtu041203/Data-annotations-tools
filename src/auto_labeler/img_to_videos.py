import argparse
from pathlib import Path
import re #regular expression library
import cv2

def natural_keys(s):
    """Sort key (frame) numerically"""
    # Normally, if a computer sort things alphabetically, it look at "each character"
    # in a string left to right. Therefore, frame_10 would go after frame_1 but before frame 2
    # cause the 1 < 2
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else float('inf')
def make_video(frames_dir: Path, output_path: Path, fps: int):
    images = sorted([p for p in frames_dir.glob("*_fused.jpg")], key=lambda p:natural_keys(p.name))
    if  not images: 
        raise ValueError(f"No *_fused.jpg files found in {frames_dir}")
    
    #Read the 1st frame to get the frame size
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w,h))
    
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (h, w), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    print(f"[DONE] Wrote {len(images)} frames -> video {output_path}")

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, help="Folder path to fused frames")
    p.add_argument("--out", type=Path, default=Path("fused_video.mp4"), help="Folder for output video")
    p.add_argument("--fps", type=int, default = 15, help="Frame rates for frame parsing")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    make_video(args.frames, args.out, args.fps)




