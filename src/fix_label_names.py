#!/usr/bin/env python3
"""
Script to rename YOLO label files to match the "_fused" suffix in image filenames.
This fixes the mismatch where images are named "frame_XXXXXX_fused.jpg" 
but labels are named "frame_XXXXXX.txt" instead of "frame_XXXXXX_fused.txt".
"""
import os
import shutil
from pathlib import Path

def fix_label_names(dataset_path):
    """
    Rename label files to match image filenames with _fused suffix for both train and val splits.
    """
    dataset_path = Path(dataset_path)
    splits = ['train', 'val']
    for split in splits:
        labels_dir = dataset_path / 'labels' / split
        images_dir = dataset_path / 'images' / split

        if not labels_dir.exists():
            print(f"Labels directory not found: {labels_dir}")
            continue
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            continue

        label_files = list(labels_dir.glob('*.txt'))
        print(f"Found {len(label_files)} label files in {labels_dir}")

        renamed_count = 0

        for label_file in label_files:
            base_name = label_file.stem
            new_name = f"{base_name}_fused.txt"
            new_path = labels_dir / new_name

            expected_image = images_dir / f"{base_name}_fused.jpg"

            if expected_image.exists():
                try:
                    shutil.move(str(label_file), str(new_path))
                    renamed_count += 1
                    if renamed_count <= 5:
                        print(f"  Renamed: {label_file.name} → {new_name}")
                    elif renamed_count == 6:
                        print("  ... (continuing to rename remaining files)")
                except Exception as e:
                    print(f"  Error renaming {label_file.name}: {e}")
            else:
                print(f"  Warning: No matching image for {label_file.name}")

        print(f"Successfully renamed {renamed_count} label files in {split} split")

def verify_matching(dataset_path):
    """Verify that image and label files now match correctly."""
    dataset_path = Path(dataset_path)
    
    for split in ['train', 'val']:
        labels_dir = dataset_path / split / 'labels'
        images_dir = dataset_path / split / 'images'
        
        if not labels_dir.exists() or not images_dir.exists():
            continue
            
        print(f"\nVerifying {split} split...")
        
        # Get all image files
        image_files = {f.stem for f in images_dir.glob('*.jpg')}
        label_files = {f.stem for f in labels_dir.glob('*.txt')}
        
        matched = len(image_files & label_files)
        total_images = len(image_files)
        
        print(f"  Images: {total_images}")
        print(f"  Labels: {len(label_files)}")
        print(f"  Matched: {matched}")
        
        if matched < total_images:
            unmatched = image_files - label_files
            print(f"  Unmatched images: {len(unmatched)}")
            if len(unmatched) <= 5:
                for img in sorted(unmatched):
                    print(f"    - {img}.jpg")
            else:
                for img in sorted(list(unmatched)[:3]):
                    print(f"    - {img}.jpg")
                print(f"    ... and {len(unmatched) - 3} more")

if __name__ == "__main__":
    dataset_path = "yolo_train"
    
    print("Fixing YOLO label filenames to match image filenames...")
    fix_label_names(dataset_path)
    
    print("\nVerifying filename matching...")
    verify_matching(dataset_path)
    
    print("\nDone! YOLO should now be able to find matching labels for images.")
