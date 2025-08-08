from ultralytics import YOLO
import os
import cv2

def main():
    model_base = YOLO("../yolo_model/yolov8n.pt")
    model_ft = YOLO("../runs/original_2000/train/weights/best.pt")
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    video_src = r"C:\auto-labeling\data\videos\2min_outdoor.mp4"
    results = model_ft.predict(
        source=video_src,
        imgsz=960,
        conf=0.25,
        save=False,
        stream=True
    )

    output_dir = os.path.join(WORK_DIR, 'pred')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'finetune_output.mp4')

    cap = cv2.VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for result in results:
        frame = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0,255,0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Custom output saved to {output_path}")

    # Base model, only class 0 (person)
    results_base = model_base.predict(
        source=video_src,
        imgsz=960,
        conf=0.25,
        save=False,
        stream=True
    )

    output_path_base = os.path.join(output_dir, 'base_output.mp4')
    cap_base = cv2.VideoCapture(video_src)
    fps_base = cap_base.get(cv2.CAP_PROP_FPS)
    width_base = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_base = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_base = cv2.VideoWriter(output_path_base, fourcc, fps_base, (width_base, height_base))

    for result in results_base:
        frame = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, conf, cls in zip(boxes, confs, clss):
            if int(cls) != 0:
                continue  # Only process class 0 (person)
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (255,0,0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

        out_base.write(frame)

    cap_base.release()
    out_base.release()
    print(f"Custom output (base, person only) saved to {output_path_base}")

if __name__ == "__main__":
    main()
