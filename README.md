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
* generated CSV / JSON outputs
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
**📤 Output Format**
1. CSV (` data/test.csv `) :- 
   Contains per-frame detections with:
   * Vehicle ID
   * Vehicle bounding box
   * License plate bounding box
   * OCR text + confidence
   * Timestamp (ms + UTC)

2. JSONL (` data/events.jsonl `) :- 
   Structured event-based output for:
   * Dashboards
   * Databases
   * Analytics pipelines

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
