#!/usr/bin/env python3
"""
Check YOLO label files for common issues that could cause DataLoader errors.
"""
import os
import glob
from pathlib import Path

def check_label_validity(dataset_path):
    """Check all label files for common issues."""
    issues = []
    
    for split in ['train', 'val']:
        labels_dir = Path(dataset_path) / split / 'labels'
        if not labels_dir.exists():
            continue
            
        print(f"\nChecking {split} labels...")
        label_files = list(labels_dir.glob('*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"{label_file.name}:{line_num} - Wrong number of values: {len(parts)}")
                        continue
                    
                    try:
                        class_id, x_center, y_center, width, height = parts
                        class_id = int(class_id)
                        x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
                        
                        # Check if coordinates are in valid range [0, 1]
                        if not (0 <= x_center <= 1):
                            issues.append(f"{label_file.name}:{line_num} - x_center out of range: {x_center}")
                        if not (0 <= y_center <= 1):
                            issues.append(f"{label_file.name}:{line_num} - y_center out of range: {y_center}")
                        if not (0 < width <= 1):
                            issues.append(f"{label_file.name}:{line_num} - width out of range: {width}")
                        if not (0 < height <= 1):
                            issues.append(f"{label_file.name}:{line_num} - height out of range: {height}")
                        
                        # Check for negative dimensions
                        if width <= 0:
                            issues.append(f"{label_file.name}:{line_num} - width <= 0: {width}")
                        if height <= 0:
                            issues.append(f"{label_file.name}:{line_num} - height <= 0: {height}")
                            
                    except ValueError as e:
                        issues.append(f"{label_file.name}:{line_num} - Invalid number format: {line}")
                        
            except Exception as e:
                issues.append(f"{label_file.name} - Error reading file: {e}")
    
    return issues

def check_image_label_correspondence(dataset_path):
    """Check if all images have corresponding labels and vice versa."""
    dataset_path = Path(dataset_path)
    
    for split in ['train', 'val']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
            
        print(f"\nChecking {split} image-label correspondence...")
        
        # Get base names (without extensions)
        image_bases = {f.stem for f in images_dir.glob('*.jpg')}
        label_bases = {f.stem for f in labels_dir.glob('*.txt')}
        
        missing_labels = image_bases - label_bases
        missing_images = label_bases - image_bases
        
        print(f"  Images: {len(image_bases)}")
        print(f"  Labels: {len(label_bases)}")
        print(f"  Images without labels: {len(missing_labels)}")
        print(f"  Labels without images: {len(missing_images)}")
        
        if missing_labels:
            print(f"  First few missing labels: {list(missing_labels)[:3]}")
        if missing_images:
            print(f"  First few missing images: {list(missing_images)[:3]}")

if __name__ == "__main__":
    dataset_path = "yolo_train"
    
    print("Checking YOLO dataset for issues...")
    
    # Check label validity
    issues = check_label_validity(dataset_path)
    
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("\nNo label validity issues found!")
    
    # Check image-label correspondence
    check_image_label_correspondence(dataset_path)
    
    print("\nTrying to load a sample with PyTorch to test DataLoader...")
    
    try:
        from PIL import Image
        import torch
        from pathlib import Path
        
        # Try to load a sample image and label
        sample_image = Path("yolo_train/train/images/frame_000000_fused.jpg")
        sample_label = Path("yolo_train/train/labels/frame_000000_fused.txt")
        
        if sample_image.exists() and sample_label.exists():
            # Load image
            img = Image.open(sample_image)
            print(f"Sample image loaded: {img.size} pixels, mode: {img.mode}")
            
            # Load label
            with open(sample_label, 'r') as f:
                labels = f.read().strip()
            label_count = len(labels.split('\n'))
            print(f"Sample label loaded: {label_count} annotations")
            
            print("Sample loading successful - DataLoader issue might be elsewhere")
        else:
            print("Sample files not found for testing")
            
    except ImportError:
        print("PIL/torch not available for testing")
    except Exception as e:
        print(f"Error loading sample: {e}")
