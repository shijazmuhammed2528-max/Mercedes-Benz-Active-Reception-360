# Mercedes-Benz Active Reception — Automated 360° Vehicle Capture System

A desktop/web automotive service reception platform inspired by Mercedes-Benz design standards. The system simulates a four-camera vehicle reception bay using synchronized video feeds, performs real-time vehicle detection and tracking using Ultralytics YOLO, automatically evaluates vehicle presence, view completeness, and stationary conditions, and captures high-resolution 360° inspection records into an organized SQLite database and filesystem storage.

---

## 1. System Architecture & Project Structure

The project is structured under `D:\Project Costal star\Main`:

```
Mercedes_Active_Reception/
├── app.py                         # Flask web application & REST/MJPEG streaming API
├── config.py                      # Central configuration (YOLO, thresholds, camera sources)
├── app_config.json                # Persisted runtime settings
├── requirements.txt               # Python package dependencies
├── start.bat                      # 1-click Windows startup script
├── README.md                      # Complete system documentation
│
├── backend/
│   ├── capture/
│   │   ├── capture_controller.py  # Orchestrates synchronization, YOLO pipeline & state machine
│   │   ├── capture_state.py       # State machine definitions (MONITORING, CAPTURING, SAVED, etc.)
│   │   └── video_manager.py       # Thread-safe multi-camera video reader and MJPEG streamer
│   ├── database/
│   │   └── db_manager.py          # SQLite database CRUD operations and filesystem recovery
│   ├── services/
│   │   ├── image_service.py       # Image serving, path traversal defense, thumbnail generation
│   │   ├── storage_service.py     # Disk usage calculation and video integrity verification
│   │   └── zip_service.py         # In-memory ZIP packaging for vehicle image sets
│   └── vision/
│       ├── frame_selector.py      # Laplacian variance sharpness evaluation & best frame selection
│       ├── motion_detector.py     # Bounding box centroid drift & scale analysis for stationary detection
│       ├── plate_recognition.py   # OCR plate extraction & Indian format normalization (EasyOCR fallback)
│       ├── view_quality.py        # Frame boundary margin & aspect ratio quality validator
│       └── yolo_detector.py       # Ultralytics YOLO inference wrapper with CUDA/CPU auto-fallback
│
├── templates/
│   ├── base.html                  # Master layout with Mercedes-Benz dark UI sidebar navigation
│   ├── dashboard.html             # High-level metrics, system status indicators & recent captures
│   ├── live_capture.html          # 2x2 synchronized CCTV live feed with real-time condition checklist
│   ├── database.html              # Searchable archive with registration, date, and time filters
│   ├── vehicle_details.html       # Full 360° vehicle image inspection gallery, downloads & lightbox
│   └── settings.html              # Interactive configuration panel for YOLO and capture thresholds
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom Mercedes-Benz automotive aesthetic styling
│   ├── js/
│   │   ├── capture.js             # Live MJPEG control, status polling & capture event handler
│   │   ├── dashboard.js           # Live statistics and hardware status monitor
│   │   └── database.js            # Asynchronous database query engine with client-side sanitization
│   └── images/
│       └── placeholder.svg        # Fallback vehicle silhouette thumbnail
│
└── logs/
    └── app.log                    # Diagnostic log file
```

---

## 2. Hardware & Camera Mapping

The system processes four synchronized video cameras:

| Camera Name | View Angle | Source File Path |
|---|---|---|
| **CAM 01** | FRONT VIEW | `D:\Project Costal star\Videos\front.mp4` |
| **CAM 02** | REAR VIEW | `D:\Project Costal star\Videos\back.mp4` |
| **CAM 03** | LEFT VIEW | `D:\Project Costal star\Videos\left.mp4` |
| **CAM 04** | RIGHT VIEW | `D:\Project Costal star\Videos\right.mp4` |

All four feeds run at `1920x1080` resolution at `24 FPS`.

---

## 3. Computer Vision Engine — YOLO Detection

- **Model**: Ultralytics YOLO (`yolov8n.pt` by default; configurable to `yolov8s.pt` or `yolov8m.pt`).
- **Target Classes**: COCO Vehicle classes (Car: 2, Motorcycle: 3, Bus: 5, Truck: 7).
- **Inference Resolution**: Configurable (default 640px).
- **Execution Device**: Automatically detects NVIDIA CUDA GPU (e.g. GeForce RTX 4050 Laptop GPU); gracefully falls back to multi-threaded CPU without crashing.
- **Bounding Box Overlay**: Drawn in real-time with vibrant Mercedes-Benz Green (`#00FF66` / BGR `(0, 255, 102)`), displaying vehicle class, confidence percentage, camera label, and tracking status.
- **Unmodified Source Capture**: The display feed renders the green bounding boxes for live monitoring, but the saved inspection photographs are extracted directly from the raw, unmodified source frames.

---

## 4. Vehicle Stationary & View-Quality Detection

1. **Centroid Movement & Scale Tracking**:
   - The bounding box centroid $(c_x, c_y)$ is tracked across consecutive frames.
   - If displacement $||\Delta C|| < \text{threshold}$ (default 15 px) and relative scale variation $|\Delta S| < \text{threshold}$ (default 8%), the vehicle enters the `STOPPING` state.
   - If conditions remain continuously satisfied for the duration (default 2.0 seconds), the camera marks the vehicle as `STATIONARY`.

2. **Full Vehicle Visibility / View Quality**:
   - Boundary margin check: Ensures $(x_1, y_1)$ and $(x_2, y_2)$ do not touch the frame borders within a configurable margin ratio (default 3%).
   - Area ratio check: Ensures the vehicle occupies a reasonable percentage of the bay view (5% to 95%).
   - Aspect ratio sanity check: Confirms detection geometry matches an automobile profile (0.3 to 5.0).

---

## 5. Automatic Capture State Machine

Capture triggers **only** when all conditions are simultaneously satisfied across all four cameras:

$$\text{Ready} = \bigwedge_{c \in \{\text{front, back, left, right}\}} (\text{Detected}_c \land \text{ViewQualityOk}_c \land \text{Stationary}_c)$$

### Capture Flow:
1. **Selection of Sharpest Frame**: A rolling frame buffer retains recent frames during the stationary window; the frame with the highest Laplacian variance ($\sigma^2$) is chosen to eliminate any transitional motion blur.
2. **Registration Extraction (OCR)**: The front camera frame is analyzed using EasyOCR/pattern normalization for license plate extraction (e.g., `KL 01 AL 0009`). If unconfirmed, the operator can confirm or edit the registration via the web UI.
3. **Organized Storage Creation**:
   - Folder: `D:\Project Costal star\Data_base\VEHICLE_NUMBER_YYYY-MM-DD_HH-MM-SS_AM-PM`
   - Files: `front.jpg`, `back.jpg`, `left.jpg`, `right.jpg`, and `metadata.json`.
4. **SQLite Indexing**: All session attributes, image paths, detection confidence scores, and timestamps are recorded in `D:\Project Costal star\Data_base\vehicle_database.db`.

---

## 6. How to Run the Application

### Option A: 1-Click Windows Batch Script (Recommended)
Double-click `start.bat` in `D:\Project Costal star` or `D:\Project Costal star\Main`.
The script will:
1. Verify Python 3.10+ installation.
2. Ensure dependencies are satisfied.
3. Launch `app.py` on `http://127.0.0.1:5000`.
4. Automatically open the default web browser.

### Option B: Manual Command Line
```powershell
cd "D:\Project Costal star\Main"
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## 7. Web Application Features

1. **Dashboard (`/`)**:
   - Total capture sessions, today's captures, success count, and vehicle storage volume.
   - Live hardware and camera connection health indicators.
   - Recent capture list with clickable links to vehicle inspection dossiers.

2. **Live Capture (`/capture`)**:
   - 2x2 synchronized camera grid (FRONT, REAR, LEFT, RIGHT).
   - Real-time MJPEG streams with live YOLO green bounding boxes.
   - Playback controls: `START CAPTURE`, `PAUSE`, `RESUME`, `STOP SESSION`, `RESET`.
   - Real-time 9-point condition checklist.
   - Automatic capture popup with inline registration number editor and image previews.

3. **Vehicle Database (`/database`)**:
   - Search by vehicle registration number (normalized to ignore spacing, hyphens, and casing).
   - Optional date filter.
   - Optional time range filter (12-hour AM/PM).
   - Thumbnail previews and direct link to session inspection records.

4. **Vehicle Session Details (`/vehicle/<session_id>`)**:
   - High-resolution gallery of all four inspection views.
   - Single-image download buttons.
   - `DOWNLOAD ALL IMAGES` button producing an in-memory ZIP package containing all 4 images and `metadata.json`.
   - Interactive full-screen image lightbox.

5. **Settings (`/settings`)**:
   - Video file validation and disk usage statistics.
   - YOLO model selection, confidence threshold, inference size, and FPS controls.
   - CUDA GPU / CPU status.
   - Adjustable stationary duration and boundary margin thresholds.
