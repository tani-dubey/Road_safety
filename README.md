**🚗 Automatic Number Plate Recognition (ANPR) Pipeline**

A modular video-based ANPR system that detects vehicles, tracks them across frames, extracts license plates, performs OCR, and stores results in CSV and JSON formats.

Designed with clean separation of concerns and pipeline-based architecture, making it suitable for research, deployment, and future optimization (e.g., OpenVINO).

This project converts raw traffic video into structured, analytics-ready events
rather than only performing object detection.

**✨ Features**

* Vehicle detection using YOLO
* Vehicle tracking using SORT
* License plate detection using YOLO
* License plate OCR using EasyOCR
* **Car ↔ license plate association logic**
* Timestamped outputs in CSV and JSON
* **Event-based JSONL output for analytics pipelines**
* Clean, modular, mentor-ready project structure
* **FastAPI route for running the pipeline as a service**

**📁 Project Structure**
```text
project-root/
├── src/
│   ├── routers/             # FastAPI route definitions
│   │   ├── route_health.py  # Health check endpoint
│   │   ├── route_metrics.py # Metrics / stats endpoints
│   │   └── route_pipeline.py# Inference pipeline endpoint
│   │
│   ├── schema/              # Request/response data models
│   │   └── schema.py        # Pydantic schemas
|   |  
│   ├── pipeline.py            # Orchestrates full pipeline
│   ├── writers.py             # CSV & JSON writing logic
│   ├── util.py                # OCR & car–plate association
│   └── third_party/
│       └── sort.py             # SORT tracker (external, unmodified)
│
├── model/
│   ├── yolov8n.pt              # Vehicle detection model
│   └── license_plate_detector_3.pt  # Number plate detection model
│
├── data/
│   ├── test.csv                # CSV output (generated)
│   └── events.jsonl            # JSONL output (generated)
│
├── sample_video/
│   └── sample.mp4              # Input video
│
├── main.py                # Entry point
├── requirements.txt
├── README.md
└── .gitignore
```
**🔐 Credentials & Sensitive Files**

1. Model weights are excluded from the repository for size and licensing reasons
   and must be downloaded separately using the links below.
  
  * license_plate_detector_3.pt : https://drive.google.com/file/d/1x8JX678ifE6xXMzTUSN509oEudFXQ0Ac/view?usp=drive_link
  * yolov8n.pt : https://drive.google.com/file/d/1v9J71JRUy7azjnhlpB3MvEMO0rpXfUhD/view?usp=drive_link
    
2. generated CSV / JSON outputs
3. Sample data to test working : https://drive.google.com/file/d/1D_OLOY5hkPcQfv93_3PZYAzyuvJsbLke/view?usp=drive_link

All such files are excluded via .gitignore.

**📦 Directory Setup (Required)**
The pipeline assumes this directory structure and will fail if required
models or input videos are missing.

```text
model/
  ├── yolov8n.pt
  └── license_plate_detector_3.pt

sample_video/
  └── sample.mp4

data/
```
**▶️ How to Run**
1. Install dependencies
   ```text
   pip install -r requirements.txt
    ```
2. To run as an API service:
   ```bash
   uvicorn main:app --reload
   ```
3. Run the pipeline
   ```text
   python main.py
    ```
**✉️ Input Sample**

![Video Project](https://github.com/user-attachments/assets/d68ea5db-03b0-4690-ad7a-1b9a47828fc4)

Download this video : https://drive.google.com/file/d/1D_OLOY5hkPcQfv93_3PZYAzyuvJsbLke/view?usp=drive_link

**📤 Output Format**
1. CSV (` data/test.csv `) :- 
   Contains per-frame detections with:
   * Vehicle ID
   * Vehicle bounding box
   * License plate bounding box
   * OCR text + confidence
   * Timestamp (ms + UTC)
```text
frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score,timestamp_ms,timestamp_utc
0,3.0,"[752, 1370, 1430, 1982]","[986, 1788, 1181, 1840]",0.5806435942649841,IN41JMR,0.24334674029814932,0.0,2025-12-15T16:09:23.417376+00:00
0,4.0,"[2198, 1157, 2783, 1735]","[2411, 1565, 2581, 1624]",0.48309609293937683,MU51KSU,0.14795953771009374,0.0,2025-12-15T16:09:23.417376+00:00
1,3.0,"[752, 1370, 1430, 1982]","[987, 1788, 1181, 1840]",0.5870230197906494,NA13MPU,0.17951069297607156,16.666666666666668,2025-12-15T16:09:24.207745+00:00
```

2. JSONL (` data/events.jsonl `) :- 
   Structured event-based output for:
   * Dashboards
   * Databases
   * Analytics pipelines
```text
{"frame_nmr": 0, "car_id": 3.0, "car_bbox": [752, 1370, 1430, 1982], "license_plate": {"bbox": [986, 1788, 1181, 1840], "bbox_score": 0.5806435942649841, "text": "IN41JMR", "text_score": 0.24334674029814932}, "timestamp": {"ms": 0.0, "utc": "2025-12-15T16:09:23.417376+00:00"}}
{"frame_nmr": 0, "car_id": 4.0, "car_bbox": [2198, 1157, 2783, 1735], "license_plate": {"bbox": [2411, 1565, 2581, 1624], "bbox_score": 0.48309609293937683, "text": "MU51KSU", "text_score": 0.14795953771009374}, "timestamp": {"ms": 0.0, "utc": "2025-12-15T16:09:23.417376+00:00"}}
{"frame_nmr": 1, "car_id": 3.0, "car_bbox": [752, 1370, 1430, 1982], "license_plate": {"bbox": [987, 1788, 1181, 1840], "bbox_score": 0.5870230197906494, "text": "NA13MPU", "text_score": 0.17951069297607156}, "timestamp": {"ms": 16.666666666666668, "utc": "2025-12-15T16:09:24.207745+00:00"}}
{"frame_nmr": 1, "car_id": 4.0, "car_bbox": [2198, 1157, 2782, 1735], "license_plate": {"bbox": [2411, 1565, 2581, 1624], "bbox_score": 0.48742157220840454, "text": "MU51KSV", "text_score": 0.13863040322459963}, "timestamp": {"ms": 16.666666666666668, "utc": "2025-12-15T16:09:24.207745+00:00"}}
```

**🚀 Future Scope**

🔹 Performance Optimization

    * Convert YOLO models to OpenVINO IR
    * CPU-optimized inference for edge devices
    
🔹 Deployment

    * Dockerized pipeline
    * REST API for live camera feeds
    * Cloud deployment (AWS / GCP / Azure)

🔹 Intelligence & Analytics

    * Traffic rule violation detection (rule-based or ML-assisted)
    * Vehicle re-identification across cameras
    * Integration with dashboards and alerting systems

**📜 License & Acknowledgements**
* SORT tracker © Alex Bewley et al. (GPL)
* YOLO © Ultralytics
* EasyOCR © JaidedAI

This project is intended for academic, research, and learning purposes.

**⭐ Mentor Note**

This repository emphasizes:
 * Clear architecture
 * Clean separation of concerns
 * Reproducibility
 * Extendability for research and deployment
   
The project is intentionally kept readable and framework-light to support
onboarding, refactoring, and OSS-style collaboration.

