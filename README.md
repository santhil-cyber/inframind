# Inframind - Advanced Defect Scanner

Inframind is an AI-powered defect detection system designed for civil infrastructure inspection. It uses deep learning (YOLOv8) and computer vision techniques to detect cracks, corrosion, and spalling in both images and videos.

## Features

-   **Multi-Modal Analysis**:
    -   **Images**: High-resolution heatmap generation and severity classification.
    -   **Videos**: Frame-by-frame tracking with optimized H.264 playback.
-   **Dual Detection Engine**:
    -   **Deep Learning**: YOLOv8 specialized model for crack detection.
    -   **Computer Vision**: HSV/Morphological analysis for corrosion detection.
-   **Professional Dashboard**:
    -   Engineering-grade reporting metadata (Inspectors, IDs, Dates).
    -   Metric-rich defect cards (Confidence, Area Ratio, Action Priority).
    -   PDF Export capable.

## Tech Stack

-   **Backend**: Python, FastAPI, OpenCV, Ultralytics YOLO, SQLite
-   **Frontend**: React (via CDN), Tailwind CSS
-   **Visualization**: Matplotlib (Heatmaps), CV2 (Video Annotation)

## Setup

1.  **Backend**:
    ```bash
    cd backend
    pip install -r requirements.txt
    python3 -m uvicorn main:app --reload
    ```

2.  **Frontend**:
    ```bash
    cd frontend
    python3 server.py
    ```

3.  Access the dashboard at `http://localhost:3000`.

## Notes

-   The large `yolov8x.pt` model is excluded from the repo. It will be downloaded automatically if missing, or you can use the included lightweight `crack.pt`.
