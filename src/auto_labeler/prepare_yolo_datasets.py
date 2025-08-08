#!/usr/bin/env python3
"""
Comprehensive YOLO dataset preparation script.
Combines functionality from prep_dataset.py, json_to_yolo.py, and clean_json.py
to prepare both original and background-fused datasets.
"""

import json
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

class YoloDatasetPreparator:
    def __init__(self, seed: int = 42):
        """Initialize the dataset preparator with random seed."""
        random.seed(seed)
        self.seed = seed
    
    def find_images_and_labels(self, source_dir: Path, use_fused: bool = False) -> List[Tuple[Path, Optional[Path]]]:
        """
        Find image-label pairs in the source directory.
        
        Args:
            source_dir: Directory containing images and JSON files
            use_fused: If True, look for *_fused.jpg images, otherwise *.jpg
            
        Returns:
            List of (image_path, json_path) tuples, where json_path can be None for images without detections
        """
        pairs = []
        
        if use_fused:
            # Look for fused images and corresponding JSON files
            fused_images = list(source_dir.glob("*_fused.jpg"))
            print(f"Found {len(fused_images)} fused images in {source_dir}")
            
            for image_path in fused_images:
                # Remove _fused suffix to find corresponding JSON
                json_name = image_path.stem.replace("_fused", "") + ".json"
                json_path = source_dir / json_name
                
                if json_path.exists():
                    pairs.append((image_path, json_path))
                else:
                    # Include images without JSON (no detections = negative examples)
                    pairs.append((image_path, None))
            
            # Add original images that don't have JSON files as negative examples
            # This ensures fused dataset has same negative examples as original dataset
            original_images = [img for img in source_dir.glob("*.jpg") if not img.name.endswith("_fused.jpg")]
            negative_originals = []
            
            for image_path in original_images:
                json_path = source_dir / f"{image_path.stem}.json"
                if not json_path.exists():
                    # This original image has no detections, add it as negative example
                    pairs.append((image_path, None))
                    negative_originals.append(image_path)
            
            if negative_originals:
                print(f"Added {len(negative_originals)} original images without detections as negative examples")
                    
        else:
            # Look for original images and corresponding JSON files
            jpg_images = [img for img in source_dir.glob("*.jpg") if not img.name.endswith("_fused.jpg")]
            print(f"Found {len(jpg_images)} original images in {source_dir}")
            
            for image_path in jpg_images:
                json_path = source_dir / f"{image_path.stem}.json"
                
                if json_path.exists():
                    pairs.append((image_path, json_path))
                else:
                    # Include images without JSON (no detections = negative examples)
                    pairs.append((image_path, None))
        
        # Count positive vs negative examples
        positive_pairs = sum(1 for _, json_path in pairs if json_path is not None)
        negative_pairs = len(pairs) - positive_pairs
        
        print(f"Found {len(pairs)} total images:")
        print(f"  • {positive_pairs} images with detections (positive examples)")
        print(f"  • {negative_pairs} images without detections (negative examples)")
        return pairs
    
    def split_train_val(self, pairs: List[Tuple[Path, Optional[Path]]], train_ratio: float = 0.8) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Tuple[Path, Optional[Path]]]]:
        """Split image-label pairs into train and validation sets."""
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)
        
        train_count = int(len(shuffled_pairs) * train_ratio)
        train_pairs = shuffled_pairs[:train_count]
        val_pairs = shuffled_pairs[train_count:]
        
        print(f"Split: {len(train_pairs)} train, {len(val_pairs)} validation")
        return train_pairs, val_pairs
    
    def setup_yolo_directories(self, yolo_dir: Path):
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
    
    def convert_json_to_yolo(self, json_path: Optional[Path], output_file: Path) -> int:
        """
        Convert single JSON file to YOLO format, or create empty file if no JSON.
        
        Args:
            json_path: Path to JSON file (can be None for images without detections)
            output_file: Path to output YOLO label file
            
        Returns:
            Number of annotations converted
        """
        if json_path is None or not json_path.exists():
            # Create empty label file for images with no detections
            with open(output_file, 'w') as f:
                f.write("")  # Empty file
            return 0
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_width = data['img_width']
        img_height = data['img_height']
        
        yolo_lines = []
        for annotation in data['annotations']:
            x1, y1, x2, y2 = annotation['box']
            
            # Convert to YOLO format (normalized xywh)
            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Class ID (0 for person)
            class_id = 0
            
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write YOLO format file
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        return len(yolo_lines)
    
    def copy_and_convert_pairs(self, pairs: List[Tuple[Path, Optional[Path]]], yolo_dir: Path, split_name: str):
        """Copy images and convert JSON labels to YOLO format."""
        images_dir = yolo_dir / "images" / split_name
        labels_dir = yolo_dir / "labels" / split_name
        
        copied_images = 0
        total_annotations = 0
        empty_labels = 0
        
        for image_path, json_path in pairs:
            # Copy image
            dest_image = images_dir / image_path.name
            shutil.copy2(image_path, dest_image)
            copied_images += 1
            
            # Convert JSON to YOLO format (or create empty label file)
            label_name = image_path.stem + ".txt"
            dest_label = labels_dir / label_name
            annotations_count = self.convert_json_to_yolo(json_path, dest_label)
            
            if annotations_count == 0:
                empty_labels += 1
            total_annotations += annotations_count
        
        print(f"✅ Copied {copied_images} images and converted {total_annotations} annotations to {split_name} set")
        if empty_labels > 0:
            print(f"   📝 Created {empty_labels} empty label files for images without detections (negative examples)")
        return copied_images, total_annotations
    
    def create_yaml_config(self, yolo_dir: Path, dataset_name: str):
        """Create YOLO dataset configuration YAML file."""
        yaml_content = f"""# YOLO dataset configuration for {dataset_name}
path: {yolo_dir.name}
train: images/train
val: images/val

names:
  0: person

# Dataset info
nc: 1  # number of classes
"""
        
        yaml_file = yolo_dir / f"{yolo_dir.name}.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created YAML config: {yaml_file}")
        return yaml_file
    
    def prepare_dataset(self, source_dir: Path, output_dir: Path, dataset_name: str, 
                       use_fused: bool = False, train_ratio: float = 0.8):
        """
        Prepare complete YOLO dataset.
        
        Args:
            source_dir: Source directory with images and JSON files
            output_dir: Output directory for YOLO dataset
            dataset_name: Name for the dataset
            use_fused: Whether to use fused images or original images
            train_ratio: Fraction of data for training
        """
        print(f"\n{'='*60}")
        print(f"Preparing {dataset_name} dataset")
        print(f"Source: {source_dir}")
        print(f"Output: {output_dir}")
        print(f"Using {'fused' if use_fused else 'original'} images")
        print(f"Train/Val split: {train_ratio:.1%}/{1-train_ratio:.1%}")
        print(f"{'='*60}")
        
        # Find image-label pairs
        pairs = self.find_images_and_labels(source_dir, use_fused)
        if not pairs:
            print(f"No valid image-label pairs found in {source_dir}")
            return
        
        # Split into train/val
        train_pairs, val_pairs = self.split_train_val(pairs, train_ratio)
        
        # Create directory structure
        self.setup_yolo_directories(output_dir)
        
        # Copy and convert files
        train_images, train_annotations = self.copy_and_convert_pairs(train_pairs, output_dir, "train")
        val_images, val_annotations = self.copy_and_convert_pairs(val_pairs, output_dir, "val")
        
        # Create YAML configuration
        yaml_file = self.create_yaml_config(output_dir, dataset_name)
        
        print(f"\n[DONE] {dataset_name} dataset prepared!")
        print(f"  - Train: {train_images} images, {train_annotations} annotations")
        print(f"  - Val: {val_images} images, {val_annotations} annotations")
        print(f"  - Config: {yaml_file}")
        
        return {
            'train_images': train_images,
            'train_annotations': train_annotations,
            'val_images': val_images,
            'val_annotations': val_annotations,
            'yaml_file': yaml_file
        }
    
    def clean_json_files(self, directory: Path):
        """Remove all JSON files from directory recursively."""
        count = 0
        for json_file in directory.rglob("*.json"):
            json_file.unlink()
            count += 1
        
        if count > 0:
            print(f"Cleaned up {count} JSON files from {directory}")

def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO datasets (original and fused)")
    parser.add_argument("--source", type=Path, required=True,
                       help="Source directory containing images and JSON files")
    parser.add_argument("--output-base", type=Path, default=Path("datasets"),
                       help="Base output directory (will create subdirectories)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    parser.add_argument("--prepare-original", action="store_true", default=True,
                       help="Prepare dataset with original images")
    parser.add_argument("--prepare-fused", action="store_true", default=True,
                       help="Prepare dataset with background-fused images")
    parser.add_argument("--skip-original", action="store_true",
                       help="Skip preparing original dataset")
    parser.add_argument("--skip-fused", action="store_true",
                       help="Skip preparing fused dataset")
    parser.add_argument("--cleanup-json", action="store_true",
                       help="Remove JSON files after conversion")
    
    args = parser.parse_args()
    
    if not args.source.exists():
        print(f"Error: Source directory {args.source} does not exist")
        return
    
    # Adjust flags based on skip options
    prepare_original = args.prepare_original and not args.skip_original
    prepare_fused = args.prepare_fused and not args.skip_fused
    
    if not prepare_original and not prepare_fused:
        print("Error: At least one dataset type must be prepared")
        return
    
    # Initialize preparator
    preparator = YoloDatasetPreparator(seed=args.seed)
    
    # Prepare datasets
    results = {}
    
    if prepare_original:
        original_dir = args.output_base / "yolo_original"
        results['original'] = preparator.prepare_dataset(
            source_dir=args.source,
            output_dir=original_dir,
            dataset_name="Original Images",
            use_fused=False,
            train_ratio=args.train_ratio
        )
    
    if prepare_fused:
        fused_dir = args.output_base / "yolo_fused"
        results['fused'] = preparator.prepare_dataset(
            source_dir=args.source,
            output_dir=fused_dir,
            dataset_name="Background Fused",
            use_fused=True,
            train_ratio=args.train_ratio
        )
    
    # Cleanup JSON files if requested
    if args.cleanup_json:
        for dataset_name, result in results.items():
            if result:
                dataset_dir = args.output_base / f"yolo_{dataset_name}"
                preparator.clean_json_files(dataset_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("DATASET PREPARATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, result in results.items():
        if result:
            print(f"\n{dataset_name.upper()} Dataset:")
            print(f"  - Train: {result['train_images']} images, {result['train_annotations']} annotations")
            print(f"  - Val: {result['val_images']} images, {result['val_annotations']} annotations")
            print(f"  - Config: {result['yaml_file']}")
            print(f"  - Training command:")
            print(f"    yolo train data={result['yaml_file']} model=yolov8s.pt epochs=100 batch=8 imgsz=1024")
    
    print(f"\nAll datasets ready for YOLO training!")

if __name__ == "__main__":
    main()
