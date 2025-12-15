import cv2
from datetime import datetime, timezone
from ultralytics import YOLO
import numpy as np
import os
import util
from sort.sort import Sort  # tracking
import json
from pathlib import Path

# ---------------- CONFIG -----------------
VIDEO_DIR = r"sample_vids"
OUT_CSV = "test.csv"
CONF_THRESHOLD = 0.5
FLUSH_EVERY_N_FRAMES = 5       # flush results to CSV every N frames
MAX_COPIES_PER_PLATE = 2       # store up to 2 entries per plate dynamically
MAX_DETECTIONS_PER_PLATE = 2   # stop video if any plate detected more than this
RESULTS_FILE = Path("results.json")  # File to store all results
# -----------------------------------------

# Load YOLO models
MODEL_CAR = YOLO("yolov8n.pt")  # Identify type of vehicle
MODEL_LP = YOLO("license_plate_detector_3.pt")  # crop and send licence plate

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


def add_or_replace_plate(written_plates, lp_text, entry, bbox_score):
    if lp_text not in written_plates:
        written_plates[lp_text] = []

    plate_list = written_plates[lp_text]

    if len(plate_list) < MAX_COPIES_PER_PLATE:
        plate_list.append({"bbox_score": bbox_score, "entry": entry})
        return True

    # Replace lowest scored if new detection is better
    min_idx, min_score = None, float("inf")
    for i, it in enumerate(plate_list):
        if it["bbox_score"] < min_score:
            min_score = it["bbox_score"]
            min_idx = i

    if bbox_score > min_score and min_idx is not None:
        plate_list[min_idx] = {"bbox_score": bbox_score, "entry": entry}
        return True

    return False


def build_results_from_written(written_plates):
    results = {}
    for entries in written_plates.values():
        for item in entries:
            e = item["entry"]
            frame = e["frame"]
            car_id = e["car_id"]
            det_dict = e["data"]

            if frame not in results:
                results[frame] = {}
            results[frame][car_id] = det_dict
    return results


def process_video(video_path):
    print(f"▶ Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    written_plates = {}          # store top-2 per plate
    plate_detection_count = {}   # counts number of detections per plate
    csv_created = False          # ensure CSV is created at least once

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"🧩 DEBUG: Processing frame {frame_count}")

        # # Detect cars
        # car_detections = MODEL_CAR.predict(frame, conf=CONF_THRESHOLD, classes=[2, 3, 5, 7], verbose=False)[0]
        # car_boxes = []
        # for box in car_detections.boxes:
        #     x1, y1, x2, y2 = box.xyxy[0]
        #     car_boxes.append([x1, y1, x2, y2, box.conf[0]])
        # car_boxes = np.array(car_boxes)
        
        CLASS_NAMES = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        
        car_detections = MODEL_CAR.predict(
            frame,
            conf=CONF_THRESHOLD,
            classes=[2, 3, 5, 7],
            verbose=False
        )[0]

        vehicle_info = []  # store one or many vehicles per frame
        car_boxes = []
        
        # for box in car_detections.boxes:
        #     x1, y1, x2, y2 = box.xyxy[0]
        #     conf = float(box.conf[0])
        #     car_boxes.append([x1, y1, x2, y2, box.conf[0]])
        #     print("class id- ",cls_id)
        #     cls_id = int(box.cls[0])
        #     category = CLASS_NAMES.get(cls_id, "unknown")

        #     vehicle_info.append({
        #         "category": category,
        #         "bbox": [float(x1), float(y1), float(x2), float(y2)],
        #         "score": conf
        #     })
        for box in car_detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            car_boxes.append([x1, y1, x2, y2, box.conf[0]])

            cls_id = int(box.cls[0])        # FIX: assign first
            print("class id-", cls_id)      

            category = CLASS_NAMES.get(cls_id, "unknown")

            vehicle_info.append({
                "category": category,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": conf
            })

        car_boxes = np.array(car_boxes)
        
        print(f"This is vehicle info- {vehicle_info}")
        # Track cars
        vehicle_track_ids = tracker.update(car_boxes) if len(car_boxes) > 0 else []

        # Detect license plates
        lp_detections = MODEL_LP.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        print(f"🧩 DEBUG: Frame {frame_count}: {len(lp_detections.boxes)} license plates detected")

        for lp_box in lp_detections.boxes:
            # convert to ints for cropping and logs
            x1, y1, x2, y2 = map(int, lp_box.xyxy[0])
            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                print(f"⚠ Invalid plate bbox after clamp: {(x1, y1, x2, y2)}")
                continue

            lp_crop = frame[y1c:y2c, x1c:x2c]
            lp_text, lp_score = util.read_license_plate(lp_crop)
            print(f"🧩 DEBUG OCR OUTPUT: {lp_text}, score={lp_score}")
            
            if not lp_text:
                print(f"⚠ OCR failed or invalid text for bbox {(x1, y1, x2, y2)}")
                continue

            # Count detections per plate
            plate_detection_count[lp_text] = plate_detection_count.get(lp_text, 0) + 1

            if plate_detection_count[lp_text] > MAX_DETECTIONS_PER_PLATE:
                print(f"⚠ Plate '{lp_text}' detected more than {MAX_DETECTIONS_PER_PLATE} times. Moving to next video...")
                cap.release()
                return

            # Match plate with car
            # We call util.get_car exactly like before, but we'll log and still record if it returns -1
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(
                (x1, y1, x2, y2, float(lp_box.conf[0]), int(lp_box.cls[0]) if hasattr(lp_box, "cls") else 0),
                vehicle_track_ids
            )

            if car_id == -1:
                print(f"⚠ No matching car found for plate {lp_text} — WILL still log OCR result into CSV (car_id = -1).")

                # create detection dict with empty/placeholder car bbox so CSV writer sees something
                # detection_dict = {
                #     "car": {},  # no bbox available
                #     "license_plate": {
                #         "bbox": [x1, y1, x2, y2],
                #         "bbox_score": float(lp_box.conf[0]),
                #         "text": lp_text,
                #         "text_score": float(lp_score),
                #     },
                #     "detection_time": {
                #         "time_ms": int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                #         "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                #     },
                # }
                detection_dict = {
                    "vehicle": vehicle_info,   # <---- NEW SECTION HERE

                    # "car": {},  # (your placeholder, unchanged)

                    "license_plate": {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_score": float(lp_box.conf[0]),
                        "text": lp_text,
                        "text_score": float(lp_score),
                    },

                    "detection_time": {
                        "time_ms": int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }

                
                # Create list if file doesn't exist
                if not RESULTS_FILE.exists():
                    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                        json.dump([], f)

                # Append current detection_dict
                with open(RESULTS_FILE, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data.append(detection_dict)
                    f.seek(0)
                    json.dump(data, f, indent=4)

                entry = {"frame": frame_count, "car_id": -1, "data": detection_dict}
                bbox_score_val = float(lp_box.conf[0])

                # still add to written_plates (so CSV will contain this plate)
                added = add_or_replace_plate(written_plates, lp_text, entry, bbox_score_val)
                if added:
                    print(f"✅ (No car) Added plate {lp_text} (score {bbox_score_val}) to written_plates")
                continue
            
            entry = {"frame": frame_count, "car_id": car_id, "data": detection_dict}
            bbox_score_val = float(lp_box.conf[0])

            added = add_or_replace_plate(written_plates, lp_text, entry, bbox_score_val)
            if added:
                print(f"✅ Added plate {lp_text} (score {bbox_score_val}) to written_plates")

        # Flush CSV dynamically every N frames or on first detection
        if written_plates and (frame_count % FLUSH_EVERY_N_FRAMES == 0 or not csv_created):
            print(f"📦 Writing CSV with {len(written_plates)} unique plates...")
            util.write_csv(build_results_from_written(written_plates), OUT_CSV)
            csv_created = True
            print(f"📤 CSV updated at frame {frame_count}")
            try:
                csv_to_slides(OUT_CSV)
                print(f"📤 Google Sheet updated at frame {frame_count}")
            except Exception as e:
                print(f"⚠ Google Sheet update failed: {e}")

    # Final flush at video end
    if written_plates:
        util.write_csv(build_results_from_written(written_plates), OUT_CSV)
        print("📤 Final CSV write completed")
        try:
            csv_to_slides(OUT_CSV)
            print("📤 Final Google Sheet update completed!")
        except Exception as e:
            print(f"⚠ Google Sheet update failed: {e}")

    cap.release()
    print(f"✅ Finished: {video_path}\n")


def main():
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        print("🗑 Old CSV removed")

    for filename in os.listdir(VIDEO_DIR):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(VIDEO_DIR, filename)
            process_video(video_path)

    print("🏁 All videos processed successfully!")


if __name__ == "__main__":
    main()