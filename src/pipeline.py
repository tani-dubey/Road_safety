import time
import signal
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

from src.writers import init_csv, write_csv_row,write_json_event,already_exist
import src.util as util
from src.third_party.sort import Sort
from src.routers.route_metrices import get_metrics

import openvino as ov
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # src/


# ----------------- CONFIG -----------------
MODEL_COCO =  BASE_DIR/"model/yolov8n.pt"                 # vehicle detector
MODEL_LP =  BASE_DIR/"model/license_plate_detector_3.pt"   # license plate detector


OUT_CSV = BASE_DIR /"data/test.csv"                   # CSV to write
JSON_OUT = BASE_DIR /"data/events.jsonl"

CONF_THRESH = 0.25                       # detection confidence threshold
VEHICLE_CLASSES = [2, 3, 5, 7]           # COCO class ids considered "vehicles" (car, motorcycle, bus, truck etc.)
PRINT_EVERY_N_FRAMES = 100               # show progress every N frames
# ------------------------------------------


def run_pipeline(
    video_path: str,
    Use_OpenVINO: bool,
    OUT_CSV: str = OUT_CSV,
    JSON_OUT: str = JSON_OUT,
) -> dict:
    
    USE_OPENVINO = Use_OpenVINO

    print("Loading models...")
    if not USE_OPENVINO:
        vehicle_model = YOLO(MODEL_COCO)
        infer_request= None
        print("Vechicle detection using : PyTorch")
    else:
        core=ov.Core()
        ov_model= core.read_model(
            BASE_DIR/ "model/yolov8n_openvino_model/yolov8n.xml"
        )
        vehicle_model=core.compile_model(
            ov_model, "AUTO:GPU,CPU"
        )
        infer_request=vehicle_model.create_infer_request()
        print("Vehicle detction using : OpenVINO")
        
    # --- License plate model stays PyTorch ---
    lp_model = YOLO(MODEL_LP)
    tracker = Sort()

    print("Models loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    
    print(already_exist(OUT_CSV,JSON_OUT))
        
    init_csv(OUT_CSV)

    written_events = set()
    frame_nmr = -1
    total_frames= 0
    unique_vehicle=set()

    print(f"Processing video: {video_path}")

    while True:
        frame_nmr += 1
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break
        
        total_frames+=1
        
        # timestamps
        time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_utc = datetime.now(timezone.utc).isoformat()

        # -------- 1. Vehicle detection --------
        if not USE_OPENVINO:
            vehicle_dets= detect_vehiches_pytorch(frame,vehicle_model)
        else:
            vehicle_dets= detect_vehicles_openvino(frame, infer_request)
            
        # -------- 2. Tracking --------
        track_ids = tracker.update(vehicle_dets)

        # -------- 3. License plate detection --------
        lp_dets = lp_model(frame, conf=CONF_THRESH, verbose=False)[0]
        lp_boxes = lp_dets.boxes.data.tolist() if lp_dets and hasattr(lp_dets, "boxes") else []

        h, w = frame.shape[:2]

        # -------- 4. Assign plates to cars + OCR --------
        for lp in lp_boxes:
            x1, y1, x2, y2, lp_score, _ = lp

            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(lp, track_ids)
            if car_id == -1:
                continue
            
            unique_vehicle.add(car_id)
            
            key = (frame_nmr, car_id)
            if key in written_events:
                continue

            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w - 1, int(x2)), min(h - 1, int(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            crop = frame[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue

            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
                lp_text, lp_text_score = util.read_license_plate(thresh)
            except Exception as e:
                print(f"[frame {frame_nmr}] OCR error: {e}")
                continue

            if lp_text is None:
                continue

            # -------- 5. Write CSV --------
            write_csv_row(OUT_CSV, [
                frame_nmr,
                car_id,
                [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                [int(x1), int(y1), int(x2), int(y2)],
                float(lp_score),
                lp_text,
                float(lp_text_score),
                float(time_ms),
                time_utc
            ])

            # -------- 6. Write JSON --------
            write_json_event(JSON_OUT, {
                "frame_nmr": frame_nmr,
                "car_id": car_id,
                "car_bbox": [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                "license_plate": {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_score": float(lp_score),
                    "text": lp_text,
                    "text_score": float(lp_text_score)
                },
                "timestamp": {
                    "ms": float(time_ms),
                    "utc": time_utc
                }
            })

            written_events.add(key)

            print(
                f"[frame {frame_nmr}] car_id={car_id} "
                f"plate='{lp_text}' conf={lp_text_score:.2f}"
            )
            
    cap.release()
    print("Pipeline finished cleanly.")
    
    return {
        "total_frames": total_frames,
        "total_vehicles" :len(unique_vehicle),
        "output_files" :[str(OUT_CSV), str(JSON_OUT)]
    }
    
def detect_vehiches_pytorch(frame,vehicle_model):
    vehicle_dets = []
    dets = vehicle_model(frame, conf=CONF_THRESH, verbose=False)[0] 
    
    if dets and hasattr(dets, "boxes"):
        for d in dets.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = d
            if int(class_id) in VEHICLE_CLASSES:
                vehicle_dets.append([x1, y1, x2, y2, score])

    vehicle_dets = (
        np.asarray(vehicle_dets) if vehicle_dets else np.empty((0, 5))
    )
    return vehicle_dets

def detect_vehicles_openvino(frame, infer_request):
    vehicle_dets = []

    # preprocess
    img = cv2.resize(frame, (640, 640))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0

    # inference
    outputs = infer_request.infer({0: img})
    pred = list(outputs.values())[0][0]   # (84, 8400)

    h, w = frame.shape[:2]

    for i in range(pred.shape[1]):
        scores = pred[4:, i]
        class_id = int(np.argmax(scores))
        score = scores[class_id]

        if score < CONF_THRESH:
            continue
        if class_id not in VEHICLE_CLASSES:
            continue

        cx, cy, bw, bh = pred[:4, i]
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)

        vehicle_dets.append([x1, y1, x2, y2, score])

    return np.asarray(vehicle_dets) if vehicle_dets else np.empty((0, 5))


# -------- CLI entry (runs on terminal) --------
def main():
    video_path= BASE_DIR/"sample_video/sample.mp4"
    run_pipeline(video_path)
    
if __name__ == "__main__":
    main()