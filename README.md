**🚗 Automatic Number Plate Recognition (ANPR) Pipeline**

A modular video-based ANPR system that detects vehicles, tracks them across frames, extracts license plates, performs OCR, and stores results in CSV and JSON formats.

Designed with clean separation of concerns and pipeline-based architecture, making it suitable for research, deployment, and future optimization (e.g., OpenVINO).


**✨ Features**

* Vehicle detection using YOLO
* Vehicle tracking using SORT
* License plate detection using YOLO
* License plate OCR using EasyOCR
* Timestamped outputs in CSV and JSON
* Clean, modular, mentor-ready project structure

**📁 Project Structure**
```text
project-root/
├── src/
│   |
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

This project does NOT require any API keys or external credentials.

Ignored sensitive / runtime files:
* .env
* token.pickle
* model weights (*.pt)
  
  * license_plate_detector_3.pt : https://drive.google.com/file/d/1x8JX678ifE6xXMzTUSN509oEudFXQ0Ac/view?usp=drive_link
  * yolov8n.pt : https://drive.google.com/file/d/1v9J71JRUy7azjnhlpB3MvEMO0rpXfUhD/view?usp=drive_link
    
* generated CSV / JSON outputs
* Sample data to test working : https://drive.google.com/file/d/1D_OLOY5hkPcQfv93_3PZYAzyuvJsbLke/view?usp=drive_link
* All such files are excluded via .gitignore.

**📦 Directory Setup (Required)**
Before running the project, ensure the following structure exists:
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
3. Run the pipeline
   ```text
   python main.py
    ```
**✉️ Input Sample**

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/5b8ddcdd-c3ae-49c9-acaa-b8bf1e4da586" />


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

    * Traffic rule violation detection
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
