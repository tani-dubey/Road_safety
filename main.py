import time
import signal
import csv
import os
import json
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

#util module (must define get_car, read_license_plate, write_csv)
import src.util as util
from src.tracking.sort import Sort

# ----------------- CONFIG -----------------
MODEL_COCO = "model/yolov8n.pt"                 # vehicle detector
MODEL_LP = "model/license_plate_detector_3.pt"   # license plate detector provided by repo
VIDEO_PATH = "sample_video/sample.mp4"              # input video
OUT_CSV = "data/test.csv"                   # CSV to write (full dump)
JSON_OUT = "data/events.jsonl"

CONF_THRESH = 0.25                       # detection confidence threshold
VEHICLE_CLASSES = [2, 3, 5, 7]           # COCO class ids considered "vehicles" (car, motorcycle, bus, truck etc.)
FLUSH_INTERVAL_SECS = 5                  # flush CSV at least every N seconds
FLUSH_EVERY_N_FRAMES = 100               # also flush every N frames
PRINT_EVERY_N_FRAMES = 100               # show progress every N frames
# ------------------------------------------

# tracker
mot_tracker = Sort()

# load models
print("Loading YOLO models...")
coco_model = YOLO(MODEL_COCO)
license_plate_detector = YOLO(MODEL_LP)
print("Models loaded.")

# open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

## Create headers of csv --
CSV_HEADER = [
    "frame_nmr",
    "car_id",
    "car_bbox",
    "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number",
    "license_number_score",
    "timestamp_ms",
    "timestamp_utc"
]

if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

# results dict
results = {}

written_events = set()

# graceful stop variables
last_write = time.time()
stop_requested = False

def _sigint_handler(sig, frame):
    global stop_requested
    print("\nSIGINT received — will stop after finishing current frame and flush CSV.")
    stop_requested = True

signal.signal(signal.SIGINT, _sigint_handler)

print(f"Starting processing video: {VIDEO_PATH}")
frame_nmr = -1

try:
    while True:
        frame_nmr += 1
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            print(f"Skipping empty frame {frame_nmr}")
            continue

        results.setdefault(frame_nmr, {})

        # --- 1) detect vehicles (COCO) ---
        try:
            dets = coco_model(frame, conf=CONF_THRESH, verbose=False)[0]
        except Exception as e:
            print(f"[frame {frame_nmr}] coco_model inference error: {e}")
            dets = None

        detections_ = []
        if dets and hasattr(dets, "boxes") and len(dets.boxes):
            for detection in dets.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in VEHICLE_CLASSES:
                    detections_.append([x1, y1, x2, y2, score])

        dets_np = np.asarray(detections_) if len(detections_) else np.empty((0,5))

        # --- 2) update tracker (SORT) ---
        try:
            track_ids = mot_tracker.update(dets_np)
        except Exception as e:
            print(f"[frame {frame_nmr}] SORT update error: {e}")
            track_ids = np.empty((0,5))

        # --- timestamp for this frame ---
        time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_utc = datetime.now(timezone.utc).isoformat()

        # --- 3) detect license plates ---
        try:
            lps = license_plate_detector(frame, conf=CONF_THRESH, verbose=False)[0]
            lp_boxes = lps.boxes.data.tolist() if (hasattr(lps, "boxes") and len(lps.boxes)) else []
        except Exception as e:
            print(f"[frame {frame_nmr}] license_plate_detector error: {e}")
            lp_boxes = []
        
        # --- 4) assign plates to cars, crop, OCR, store results ---
        for lp in lp_boxes:
            x1, y1, x2, y2, score, class_id = lp

            try:
                xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(lp, track_ids)
            except Exception as e:
                print(f"[frame {frame_nmr}] get_car error: {e}")
                car_id = -1
                xcar1 = ycar1 = xcar2 = ycar2 = 0

            if car_id == -1:
                continue

            ix1 = max(0, int(x1)); iy1 = max(0, int(y1))
            ix2 = min(w - 1, int(x2)); iy2 = min(h - 1, int(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            licence_crop = frame[iy1:iy2, ix1:ix2]
            if licence_crop.size == 0:
                continue

            try:
                crop_gray = cv2.cvtColor(licence_crop, cv2.COLOR_BGR2GRAY)
                _, crop_thresh = cv2.threshold(crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                lp_text, lp_text_score = util.read_license_plate(crop_thresh)
            except Exception as e:
                print(f"[frame {frame_nmr}] OCR error: {e}")
                lp_text, lp_text_score = None, 0.0

            if lp_text is not None:
                key = (frame_nmr, car_id)
                if key not in written_events:
                    try:
                        with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                frame_nmr,
                                car_id,
                                [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                                [int(x1), int(y1), int(x2), int(y2)],
                                float(score),
                                lp_text,
                                float(lp_text_score),
                                float(time_ms),
                                time_utc
                            ])
                        written_events.add(key)

                    except Exception as e:
                        print(f"[frame {frame_nmr}] CSV append error:", e)
                    
                    json_event = {
                        "frame_nmr": frame_nmr,
                        "car_id": car_id,
                        "car_bbox": [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                        "license_plate": {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "bbox_score": float(score),
                            "text": lp_text,
                            "text_score": float(lp_text_score)
                        },
                        "timestamp": {
                            "ms": float(time_ms),
                            "utc": time_utc
                        }
                    }

                    with open(JSON_OUT, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(json_event) + "\n")

                print(f"[frame {frame_nmr}] car_id={car_id} plate='{lp_text}' conf={lp_text_score:.2f} time_ms={time_ms:.1f}")

   
        if frame_nmr % PRINT_EVERY_N_FRAMES == 0 and frame_nmr != 0:
            print(f"Processed frames: {frame_nmr}")

        if stop_requested:
            print("Stop requested; breaking main loop.")
            break

finally:
    cap.release()
    print("Processing finished.")