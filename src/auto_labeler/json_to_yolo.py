import json
import argparse
from pathlib import Path

def convert_json_to_yolo(json_path: Path, output_dir: Path):
    """
    Convert Phase 2 JSON detection format to YOLO label format.
    
    Args:
        json_path: Path to JSON file
        output_dir: Directory to save YOLO label files
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    img_width = data['img_width']
    img_height = data['img_height']
    
    # Create output filename (same name as JSON but with .txt extension)
    output_file = output_dir / f"{json_path.stem}.txt"
    
    # Convert annotations
    yolo_lines = []
    for annotation in data['annotations']:
        # Extract bounding box (xyxy format)
        x1, y1, x2, y2 = annotation['box']
        
        # Convert to YOLO format (normalized xywh)
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Class ID (0 for person)
        class_id = 0
        
        # Format: class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    # Write YOLO format file
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    print(f"Converted {len(yolo_lines)} detections: {json_path.name} -> {output_file.name}")

def convert_folder(input_dir: Path, output_dir: Path):
    """Convert all JSON files in a folder to YOLO format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    for json_file in json_files:
        convert_json_to_yolo(json_file, output_dir)
    
    print(f"\n[DONE] Converted {len(json_files)} files to YOLO format in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert Phase 2 JSON to YOLO format")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file or folder")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for YOLO labels")
    
    args = parser.parse_args()
    
    if args.input.is_file():
        # Convert single file
        args.output.mkdir(parents=True, exist_ok=True)
        convert_json_to_yolo(args.input, args.output)
    elif args.input.is_dir():
        # Convert folder
        convert_folder(args.input, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()