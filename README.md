# 🎯 VisionTrack AI — Real-time Object Detection & Tracking

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A production-ready computer vision web app for real-time object detection and multi-object tracking.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Models](#-models)
- [Usage](#-usage)
- [Results Tab](#-results-tab)
- [Deployment](#-deployment)
- [Platform Notes](#-platform-notes)
- [Technology Stack](#-technology-stack)
- [How It Works](#-how-it-works)
- [License](#-license)

---

## 🌟 Overview

**VisionTrack AI** is a Streamlit-based web application wrapping YOLOv8's object detection with ByteTracker for seamless multi-object tracking. It supports three input modes — **live webcam**, **video file**, and **static image** — and automatically logs every detection session to a persistent **Results** history with CSV export.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 **Live Webcam** | Real-time detection & tracking with DirectShow (CAP_DSHOW) on Windows |
| 🎬 **Video File** | Upload and process MP4, AVI, MOV, MKV, WebM |
| 🖼️ **Image** | Instant inference on JPG, PNG, BMP, WebP with download |
| 📊 **Results Tab** | Persistent JSON history of all detection sessions |
| 🤖 **5 Model Tiers** | Good / Better / Great / Excellent / Best — friendly selector |
| 🎨 **Visual Model Picker** | Speed + accuracy bars, colour-coded tags, recommendations |
| 🏷️ **80 COCO Classes** | People, vehicles, animals, everyday objects |
| 🎨 **Motion Trails** | Coloured centroid path per tracked object |
| ⚡ **Live Stats** | FPS, total detections, unique IDs, frames processed |
| 🔍 **Class Filter** | Multiselect to focus on specific object types |
| ⬇️ **CSV Export** | Download full session results as a spreadsheet |
| 💾 **Video Export** | Save processed video with bounding boxes as `output.mp4` |
| 🌐 **Deployment Ready** | `requirements.txt` + `.streamlit/config.toml` included |

---

## 📁 Project Structure

```
OBJECT/
│
├── app.py                  # ✅ Main Streamlit application (UI + detection logic)
├── object_tracker.py       # Standalone OpenCV tracker (no Streamlit)
├── download_models.py      # Bulk download all 5 YOLOv8 model weights
├── requirements.txt        # Python dependencies for deployment
├── results_log.json        # Auto-generated — persists all detection sessions
│
├── yolov8n.pt              # ⚡ Good  — Nano  (6 MB)
├── yolov8s.pt              # 🚀 Better — Small (22 MB)
├── yolov8m.pt              # ⚖️ Great  — Medium (50 MB) ← default
├── yolov8l.pt              # 🎯 Excellent — Large (87 MB)
├── yolov8x.pt              # 🔬 Best  — XLarge (136 MB)
│
└── .streamlit/
    └── config.toml         # Dark purple theme + server settings
```

---

## 🔧 Installation

### 1. Clone / download the project

```bash
git clone https://github.com/yourusername/visiontrack-ai.git
cd visiontrack-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download all model weights

```bash
python download_models.py
```

> This downloads all 5 YOLOv8 `.pt` files (~305 MB total) into the project folder.
> Skip any models you don't need — the app auto-detects which are available.

### 5. Launch the app

```bash
streamlit run app.py
```

Opens automatically at **http://localhost:8501**

---

## 🤖 Models

All 5 YOLOv8 variants are supported. The app presents them with **friendly rating names** — no raw filenames shown to the user:

| Friendly Name | Technical | Size | Speed | Accuracy | Best For |
|---|---|---|---|---|---|
| ⚡ **Good** | yolov8n | 6 MB | 98% | 38% | Low-end hardware, max FPS |
| 🚀 **Better** | yolov8s | 22 MB | 80% | 58% | Laptops without GPU |
| ⚖️ **Great** ★ | yolov8m | 50 MB | 60% | 74% | **Most users (default)** |
| 🎯 **Excellent** | yolov8l | 87 MB | 38% | 88% | Decent GPU available |
| 🔬 **Best** | yolov8x | 136 MB | 18% | 96% | Powerful GPU, max accuracy |

The sidebar model picker shows **live speed and accuracy bars** plus a colour-coded tag (`FASTEST` / `FAST` / `RECOMMENDED` / `HIGH ACCURACY` / `MAX ACCURACY`) for each selection.

---

## 🖥️ Usage

### 📷 Webcam Live
1. Open the **Webcam Live** tab
2. Click **▶ Start Webcam** — YOLOv8 + ByteTracker runs on every frame
3. Click **⏹ Stop** — the session is automatically saved to the Results tab

### 🎬 Video File
1. Open the **Video File** tab
2. Upload any video (MP4, AVI, MOV, MKV, WebM)
3. Set max frames and frame-skip rate if needed
4. Click **🚀 Process Video** — results are saved on completion

### 🖼️ Image
1. Open the **Image** tab
2. Upload any image (JPG, PNG, BMP, WebP)
3. View original vs. annotated side-by-side
4. Click **💾 Save to Results** or **⬇️ Download Annotated Image**

### 📊 Results
- Dashboard of all-time stats: sessions, detections, IDs, frames
- Class frequency bar chart across all sessions
- Individual session history cards with class tags
- **⬇️ Export Results as CSV** — full spreadsheet download
- **🗑️ Clear All Results** — wipe the history

### ⚙️ Sidebar Controls
| Control | Effect |
|---|---|
| Model picker | Choose from 5 tiers with visual speed/accuracy bars |
| Show Class Labels | Toggle object name on bounding box |
| Show Confidence % | Toggle confidence score on bounding box |
| Show Motion Trails | Toggle centroid trail per tracked object |
| Class Filter | Show only selected COCO classes |
| Save Processed Video | Write `output.mp4` during video processing |

> **Note:** Confidence (50%) and IoU (45%) thresholds are fixed at optimal defaults.

---

## 📊 Results Tab

Every detection session is auto-saved to `results_log.json`:

```json
{
  "source": "Video: traffic.mp4",
  "timestamp": "2026-06-05 16:45:11",
  "model": "⚖️ Great (Medium · 50 MB)",
  "frames": 450,
  "total_detections": 1823,
  "unique_ids": 34,
  "avg_fps": 18.4,
  "classes_detected": {
    "car": 1201,
    "person": 512,
    "bus": 110
  }
}
```

---

## ☁️ Deployment

### Streamlit Community Cloud (Free)

1. Push your project to **GitHub** (exclude `.pt` files in `.gitignore` if size is an issue)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → select your repo → set `app.py` as the entrypoint
4. Click **Deploy** — live HTTPS URL in ~2 minutes

> On Streamlit Cloud, webcam access requires HTTPS — handled automatically.
> Add model downloads inside `download_models.py` or use `@st.cache_resource` to auto-fetch.

### Local Network / LAN

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Access from any device on the same network via `http://<your-ip>:8501`

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

```bash
docker build -t visiontrack-ai .
docker run -p 8501:8501 visiontrack-ai
```

---

## ⚠️ Platform Notes

### Windows — Webcam MSMF Crash Fix
On Windows, OpenCV's default **Media Foundation (MSMF)** backend causes crashes or hangs on `cap.read()`. Both `object_tracker.py` and `app.py` automatically switch to **DirectShow (CAP_DSHOW)**:

```python
# Applied automatically when sys.platform starts with "win"
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # Windows — stable
cap = cv2.VideoCapture(0)                   # Linux / macOS — default
```

### Dependency Warnings
- `langchainplus-sdk` pydantic conflict — **harmless**, does not affect YOLOv8 or Streamlit
- `torch.classes` path warning in Streamlit watcher — **harmless**, a known PyTorch + Streamlit quirk

---

## 🔬 Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| **Object Detection** | [YOLOv8](https://github.com/ultralytics/ultralytics) | ≥ 8.0 |
| **Multi-Object Tracking** | ByteTracker (built into YOLO) | — |
| **Video Processing** | OpenCV | ≥ 4.8 |
| **Web Framework** | Streamlit | ≥ 1.35 |
| **Deep Learning** | PyTorch | ≥ 2.0 |
| **HTTP / Networking** | Requests | ≥ 2.34.2 |
| **Data** | MS COCO (80 classes) | — |
| **Results Storage** | JSON file (local) | — |
| **Language** | Python | 3.9+ |

---

## ⚙️ How It Works

```
Input (Webcam / Video / Image)
        ↓
  OpenCV Frame Read
  (CAP_DSHOW on Windows)
        ↓
  YOLOv8 Inference
  conf ≥ 0.50 · iou = 0.45
        ↓
  Non-Max Suppression (NMS)
        ↓
  ByteTracker
  (Consistent Track IDs across frames)
        ↓
  Annotated Frame
  (Boxes · Labels · Confidence · Motion Trails)
        ↓
  Streamlit Live Display
        ↓
  Results Log (results_log.json)
```

---

## 📜 License

MIT License — free for personal and commercial use.

---

## 🙌 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web framework
- [MS COCO Dataset](https://cocodataset.org/) for the 80-class training set
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the multi-object tracking algorithm

---

<div align="center">
  Built with ❤️ using YOLOv8 + ByteTracker + Streamlit
</div>
