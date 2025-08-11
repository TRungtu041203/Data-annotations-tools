from ultralytics import YOLO
import os
import cv2
import argparse
import numpy as np
import torch

def load_model_safely(model_path, device='auto'):
    """Load YOLO model with error handling for compatibility issues."""
    try:
        print(f"🔄 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Test model with dummy prediction
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(dummy_img, verbose=False)
        
        print(f"✅ Model loaded successfully: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {str(e)}")
        print("🔧 Trying alternative loading methods...")
        
        # Try loading with CPU first
        try:
            model = YOLO(model_path)
            model.to('cpu')
            print(f"✅ Model loaded on CPU: {model_path}")
            return model
        except Exception as e2:
            print(f"❌ CPU loading also failed: {str(e2)}")
            
        # Try loading checkpoint manually
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
                
                # Try to load with different approach
                model = YOLO()
                model.load(model_path)
                print(f"✅ Model loaded with alternative method: {model_path}")
                return model
        except Exception as e3:
            print(f"❌ Alternative loading failed: {str(e3)}")
            
        raise Exception(f"Failed to load model {model_path} with all methods tried")

def run_finetuned_only(model_ft, video_src, output_dir):
    """Run inference with only the fine-tuned model."""
    print("🚀 Running fine-tuned model only...")
    
    results = model_ft.predict(
        source=video_src,
        imgsz=960,
        conf=0.25,
        save=False,
        stream=True
    )

    output_path = os.path.join(output_dir, 'finetune_output.mp4')
    cap = cv2.VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    for result in results:
        frame = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0,255,0), -1)
            cv2.putText(frame, label, (x1, y1 - 4), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

        # Add title
        cv2.putText(frame, "Fine-tuned Model", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Fine-tuned model output saved to {output_path}")

def run_side_by_side_comparison(model_base, model_ft, video_src, output_dir):
    """Run side-by-side comparison of base and fine-tuned models."""
    print("🚀 Running side-by-side comparison...")
    
    # Get video properties
    cap = cv2.VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Setup output video (double width for side-by-side)
    output_path = os.path.join(output_dir, 'comparison_output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Get predictions from both models
    results_base = model_base.predict(source=video_src, imgsz=960, conf=0.25, save=False, stream=True)
    results_ft = model_ft.predict(source=video_src, imgsz=960, conf=0.25, save=False, stream=True)
    
    frame_count = 0
    for result_base, result_ft in zip(results_base, results_ft):
        # Process base model frame (left side)
        frame_base = result_base.orig_img.copy()
        boxes_base = result_base.boxes.xyxy.cpu().numpy()
        confs_base = result_base.boxes.conf.cpu().numpy()
        clss_base = result_base.boxes.cls.cpu().numpy()
        names_base = result_base.names
        
        person_count_base = 0
        for box, conf, cls in zip(boxes_base, confs_base, clss_base):
            if int(cls) != 0:  # Only process class 0 (person)
                continue
            person_count_base += 1
            x1, y1, x2, y2 = map(int, box)
            label = f"{names_base[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame_base, (x1, y1), (x2, y2), (255,0,0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame_base, (x1, y1 - th - 6), (x1 + tw, y1), (255,0,0), -1)
            cv2.putText(frame_base, label, (x1, y1 - 4), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        
        # Process fine-tuned model frame (right side)
        frame_ft = result_ft.orig_img.copy()
        boxes_ft = result_ft.boxes.xyxy.cpu().numpy()
        confs_ft = result_ft.boxes.conf.cpu().numpy()
        clss_ft = result_ft.boxes.cls.cpu().numpy()
        names_ft = result_ft.names
        
        person_count_ft = 0
        for box, conf, cls in zip(boxes_ft, confs_ft, clss_ft):
            person_count_ft += 1
            x1, y1, x2, y2 = map(int, box)
            label = f"{names_ft[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame_ft, (x1, y1), (x2, y2), (0,255,0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame_ft, (x1, y1 - th - 6), (x1 + tw, y1), (0,255,0), -1)
            cv2.putText(frame_ft, label, (x1, y1 - 4), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
        
        # Add titles and detection counts
        cv2.putText(frame_base, f"Base Model (Persons: {person_count_base})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame_ft, f"Fine-tuned Model (Persons: {person_count_ft})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Combine frames side by side
        combined_frame = np.hstack((frame_base, frame_ft))
        
        # Add separator line
        cv2.line(combined_frame, (width, 0), (width, height), (255,255,255), 2)
        
        out.write(combined_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    out.release()
    print(f"✅ Side-by-side comparison saved to {output_path}")

def run_both_outputs(model_base, model_ft, video_src, output_dir, conf=0.25, imgsz=960):
    """Run both fine-tuned only and side-by-side comparison outputs."""
    print("🚀 Running both outputs (fine-tuned only + side-by-side comparison)...")
    
    # First run fine-tuned only
    print("\n📊 Step 1/2: Generating fine-tuned model output...")
    run_finetuned_only(model_ft, video_src, output_dir)
    
    # Then run side-by-side comparison
    print("\n📊 Step 2/2: Generating side-by-side comparison...")
    run_side_by_side_comparison(model_base, model_ft, video_src, output_dir)
    
    print("\n✅ Both outputs completed!")
    print(f"   - Fine-tuned only: {os.path.join(output_dir, 'finetune_output.mp4')}")
    print(f"   - Side-by-side: {os.path.join(output_dir, 'comparison_output.mp4')}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Model Inference Comparison')
    parser.add_argument('--mode', choices=['finetune', 'compare', 'both'], default='compare',
                       help='Run mode: "finetune" for fine-tuned model only, "compare" for side-by-side comparison, "both" for both outputs')
    parser.add_argument('--video', type=str, default=r"C:\auto-labeling\data\videos\2min_outdoor.mp4",
                       help='Path to input video file')
    parser.add_argument('--base-model', type=str, default="../yolo_model/yolov8n.pt",
                       help='Path to base model')
    parser.add_argument('--finetune-model', type=str, default="../runs/original_2000/train/weights/best.pt",
                       help='Path to fine-tuned model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--imgsz', type=int, default=960,
                       help='Image size for inference')
    
    args = parser.parse_args()
    
    # Setup
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(WORK_DIR, 'pred')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("🔄 Loading models...")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except:
        print("❌ Cannot detect ultralytics version")
    
    try:
        model_ft = load_model_safely(args.finetune_model)
        print(f"✅ Fine-tuned model loaded: {args.finetune_model}")
    except Exception as e:
        print(f"❌ Failed to load fine-tuned model: {str(e)}")
        print("🔧 Suggestions:")
        print("   1. Update ultralytics: pip install ultralytics --upgrade")
        print("   2. Try using a fresh base model instead")
        print("   3. Retrain your model with current ultralytics version")
        return
    
    if args.mode in ['compare', 'both']:
        try:
            model_base = load_model_safely(args.base_model)
            print(f"✅ Base model loaded: {args.base_model}")
        except Exception as e:
            print(f"❌ Failed to load base model: {str(e)}")
            print("🔧 Trying to download fresh base model...")
            try:
                model_base = YOLO('yolov8n.pt')  # Download fresh model
                print("✅ Downloaded fresh YOLOv8n base model")
            except Exception as e2:
                print(f"❌ Failed to download base model: {str(e2)}")
                return
    
    # Check video file
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    print(f"📹 Input video: {args.video}")
    print(f"🎯 Mode: {args.mode}")
    print(f"📊 Confidence threshold: {args.conf}")
    print(f"🖼️ Image size: {args.imgsz}")
    print("-" * 50)
    
    # Run inference based on mode
    if args.mode == 'finetune':
        run_finetuned_only(model_ft, args.video, output_dir)
    elif args.mode == 'compare':
        run_side_by_side_comparison(model_base, model_ft, args.video, output_dir)
    elif args.mode == 'both':
        run_both_outputs(model_base, model_ft, args.video, output_dir, args.conf, args.imgsz)
    
    print("🎉 Processing completed!")

if __name__ == "__main__":
    main()
