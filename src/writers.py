import csv
import json
import os

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

def init_csv(OUT_CSV):
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

def write_csv_row(OUT_CSV, row):
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def write_json_event(JSON_OUT, event):
    with open(JSON_OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
