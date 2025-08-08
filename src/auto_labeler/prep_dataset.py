import shutil
import random
from pathlib import Path
import argparse
from typing import List, Tuple

def find_fused_images(mask_img_dir: Path) -> List[Path]:
    """Find all *_fused.jpg images in the mask_img directory."""
    fused_images = list(mask_img_dir.glob("*_fused.jpg"))
    print(f"Found {len(fused_images)} fused images in {mask_img_dir}")
    return fused_images

def split_train_val(image_paths: List[Path], train_ratio: float = 0.8) -> Tuple[List[Path], List[Path]]:
    """Split image paths into train and validation sets."""
    # Shuffle for random split
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)
    
    # Calculate split point
    train_count = int(len(shuffled_paths) * train_ratio)
    
    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count:]
    
    print(f"Split: {len(train_paths)} train, {len(val_paths)} validation")
    return train_paths, val_paths

def setup_yolo_directories(yolo_dir: Path):
    """Create YOLO dataset directory structure."""
    dirs_to_create = [
        yolo_dir / "images" / "train",
        yolo_dir / "images" / "val", 
        yolo_dir / "labels" / "train",
        yolo_dir / "labels" / "val"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def copy_files(image_paths: List[Path], mask_img_dir: Path, yolo_dir: Path, split_name: str):
    """Copy images and corresponding JSON files to YOLO dataset structure."""
    images_dir = yolo_dir / "images" / split_name
    labels_dir = yolo_dir / "labels" / split_name
    
    copied_images = 0
    copied_labels = 0
    
    for image_path in image_paths:
        # Copy fused image
        dest_image = images_dir / image_path.name
        shutil.copy2(image_path, dest_image)
        copied_images += 1
        
        # Find corresponding JSON file (remove _fused suffix)
        json_name = image_path.stem.replace("_fused", "") + ".json"
        json_path = mask_img_dir / json_name
        
        if json_path.exists():
            dest_json = labels_dir / json_name
            shutil.copy2(json_path, dest_json)
            copied_labels += 1
        else:
            print(f"Warning: JSON file not found for {image_path.name}: {json_name}")
    
    print(f"Copied {copied_images} images and {copied_labels} labels to {split_name} set")

def prepare_yolo_dataset(mask_img_dir: Path, yolo_dir: Path, train_ratio: float = 0.8, seed: int = 42):
    """
    Prepare YOLO dataset from mask_img directory.
    
    Args:
        mask_img_dir: Directory containing *_fused.jpg and corresponding .json files
        yolo_dir: Output directory for YOLO dataset
        train_ratio: Fraction of data for training (default 0.8)
        seed: Random seed for reproducible splits
    """
    # Set random seed for reproducible splits
    random.seed(seed)
    
    print(f"Preparing YOLO dataset...")
    print(f"Source: {mask_img_dir}")
    print(f"Target: {yolo_dir}")
    print(f"Train/Val split: {train_ratio:.1%}/{1-train_ratio:.1%}")
    
    # Find all fused images
    fused_images = find_fused_images(mask_img_dir)
    if not fused_images:
        print("No fused images found! Make sure you have *_fused.jpg files.")
        return
    
    # Split into train/val
    train_images, val_images = split_train_val(fused_images, train_ratio)
    
    # Create directory structure
    setup_yolo_directories(yolo_dir)
    
    # Copy files
    copy_files(train_images, mask_img_dir, yolo_dir, "train")
    copy_files(val_images, mask_img_dir, yolo_dir, "val")
    
    print(f"\n[DONE] YOLO dataset prepared in {yolo_dir}")
    print(f"Next steps:")
    print(f"1. Convert JSON labels to YOLO format:")
    print(f"   python -m auto_labeler.json_to_yolo --input {yolo_dir}/labels/train --output {yolo_dir}/labels/train")
    print(f"   python -m auto_labeler.json_to_yolo --input {yolo_dir}/labels/val --output {yolo_dir}/labels/val")
    print(f"2. Update your YAML file path to: {yolo_dir.name}")

def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from mask_img directory")
    parser.add_argument("--mask-img-dir", type=Path, default=Path("output/mask_img"), 
                       help="Directory containing *_fused.jpg and .json files")
    parser.add_argument("--yolo-dir", type=Path, default=Path("yolo_train"), 
                       help="Output directory for YOLO dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                       help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    if not args.mask_img_dir.exists():
        print(f"Error: Source directory {args.mask_img_dir} does not exist")
        return
    
    prepare_yolo_dataset(args.mask_img_dir, args.yolo_dir, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()