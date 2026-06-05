import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import time
import json
import datetime
import threading
import queue as _queue
from ultralytics import YOLO
from PIL import Image

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionTrack AI — Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Modern Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  :root {
    --primary: #7C3AED;
    --primary-light: #A78BFA;
    --accent: #06B6D4;
    --accent2: #10B981;
    --danger: #EF4444;
    --bg-deep: #050714;
    --bg-card: #0D1117;
    --bg-panel: #161B2E;
    --border: rgba(124,58,237,0.25);
    --text: #E2E8F0;
    --text-muted: #64748B;
    --glow: 0 0 30px rgba(124,58,237,0.35);
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-deep) !important;
    color: var(--text) !important;
  }
  .stApp { background-color: var(--bg-deep) !important; }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Hero ── */
  .hero-banner {
    background: linear-gradient(135deg, #0D1117 0%, #161B2E 40%, #1a0a3e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.2rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow);
  }
  .hero-banner::before {
    content: '';
    position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-banner::after {
    content: '';
    position: absolute; bottom: -30%; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(6,182,212,0.1) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #A78BFA 0%, #7C3AED 40%, #06B6D4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.1;
  }
  .hero-subtitle { color: var(--text-muted); font-size: 1rem; margin-top: 0.4rem; }
  .hero-badges { display:flex; gap:0.6rem; margin-top:1rem; flex-wrap:wrap; }
  .badge {
    background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.4);
    color: var(--primary-light); padding: 0.25rem 0.85rem; border-radius: 50px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
  }
  .badge.cyan { background:rgba(6,182,212,0.12); border-color:rgba(6,182,212,0.35); color:#67E8F9; }
  .badge.green { background:rgba(16,185,129,0.12); border-color:rgba(16,185,129,0.35); color:#6EE7B7; }
  .badge.red { background:rgba(239,68,68,0.12); border-color:rgba(239,68,68,0.35); color:#FCA5A5; }

  /* ── Stat Cards ── */
  .stats-grid {
    display:grid; grid-template-columns:repeat(4, 1fr); gap:1rem; margin-bottom:1.5rem;
  }
  .stat-card {
    background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
    padding:1.1rem 1.3rem; position:relative; overflow:hidden; transition:all 0.3s ease;
  }
  .stat-card:hover { box-shadow:var(--glow); transform:translateY(-2px); }
  .stat-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg, var(--primary), var(--accent));
    border-radius:14px 14px 0 0;
  }
  .stat-value { font-size:1.9rem; font-weight:800; color:#fff; line-height:1; }
  .stat-label { font-size:0.75rem; color:var(--text-muted); margin-top:0.3rem; font-weight:500; letter-spacing:0.05em; text-transform:uppercase; }
  .stat-icon { font-size:1.3rem; margin-bottom:0.35rem; }

  /* ── Detection Table ── */
  .detection-table { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; overflow:hidden; }
  .detection-row {
    display:flex; align-items:center; padding:0.65rem 1rem;
    border-bottom:1px solid rgba(124,58,237,0.1); transition:background 0.2s;
  }
  .detection-row:hover { background:rgba(124,58,237,0.07); }
  .detection-row:last-child { border-bottom:none; }
  .det-id { font-weight:700; color:var(--primary-light); width:42px; }
  .det-label { flex:1; font-weight:500; }
  .det-conf { color:var(--accent2); font-weight:600; font-size:0.88rem; }
  .conf-bar-wrap { flex:1; margin:0 0.8rem; height:6px; background:rgba(255,255,255,0.05); border-radius:50px; overflow:hidden; }
  .conf-bar-fill { height:100%; border-radius:50px; background:linear-gradient(90deg, var(--primary), var(--accent)); }

  /* ── Results Section ── */
  .results-header {
    font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700;
    background:linear-gradient(135deg,#A78BFA,#06B6D4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0 0 1rem 0;
  }
  .result-card {
    background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
    padding:1.2rem; margin-bottom:1rem; position:relative; overflow:hidden;
    transition:all 0.3s;
  }
  .result-card:hover { box-shadow:var(--glow); border-color:rgba(124,58,237,0.5); }
  .result-card::before {
    content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg, var(--primary), var(--accent));
  }
  .result-meta { font-size:0.78rem; color:var(--text-muted); margin-bottom:0.6rem; display:flex; gap:1rem; flex-wrap:wrap; }
  .result-meta span { display:flex; align-items:center; gap:0.3rem; }
  .result-tags { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.6rem; }
  .result-tag {
    font-size:0.72rem; font-weight:600; padding:0.2rem 0.6rem; border-radius:50px;
    background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.3); color:var(--primary-light);
  }

  /* ── Buttons ── */
  .stButton > button {
    background:linear-gradient(135deg, var(--primary), #5B21B6) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; font-size:0.92rem !important;
    padding:0.55rem 1.3rem !important; transition:all 0.3s ease !important;
    box-shadow:0 4px 15px rgba(124,58,237,0.4) !important;
  }
  .stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 6px 25px rgba(124,58,237,0.6) !important; }

  /* ── Info boxes ── */
  .info-box {
    background:rgba(6,182,212,0.07); border:1px solid rgba(6,182,212,0.25);
    border-left:3px solid var(--accent); border-radius:10px;
    padding:0.9rem 1.1rem; font-size:0.88rem; color:#BAE6FD; margin:0.7rem 0;
  }
  .warn-box {
    background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.25);
    border-left:3px solid #F59E0B; border-radius:10px;
    padding:0.9rem 1.1rem; font-size:0.88rem; color:#FDE68A; margin:0.7rem 0;
  }
  .success-box {
    background:rgba(16,185,129,0.07); border:1px solid rgba(16,185,129,0.25);
    border-left:3px solid var(--accent2); border-radius:10px;
    padding:0.9rem 1.1rem; font-size:0.88rem; color:#A7F3D0; margin:0.7rem 0;
  }

  /* ── Model Cards ── */
  .model-card {
    background: var(--bg-panel);
    border: 2px solid rgba(124,58,237,0.2);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
  }
  .model-card.selected {
    border-color: var(--primary);
    background: rgba(124,58,237,0.12);
    box-shadow: 0 0 16px rgba(124,58,237,0.3);
  }
  .model-card:hover { border-color: rgba(124,58,237,0.5); transform: translateX(3px); }
  .model-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem; font-weight: 700; color: #E2E8F0;
    display: flex; align-items: center; justify-content: space-between;
  }
  .model-tag {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.15rem 0.55rem; border-radius: 50px;
  }
  .tag-fastest { background:rgba(16,185,129,0.2); color:#6EE7B7; border:1px solid rgba(16,185,129,0.35); }
  .tag-fast    { background:rgba(6,182,212,0.2);  color:#67E8F9; border:1px solid rgba(6,182,212,0.35); }
  .tag-default { background:rgba(124,58,237,0.2); color:#A78BFA; border:1px solid rgba(124,58,237,0.35); }
  .tag-high    { background:rgba(245,158,11,0.2); color:#FDE68A; border:1px solid rgba(245,158,11,0.35); }
  .tag-max     { background:rgba(239,68,68,0.2);  color:#FCA5A5; border:1px solid rgba(239,68,68,0.35); }
  .model-bars { margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.3rem; }
  .bar-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.68rem; color: #64748B; }
  .bar-track { flex: 1; height: 5px; background: rgba(255,255,255,0.07); border-radius: 50px; overflow: hidden; }
  .bar-fill-speed    { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #10B981, #06B6D4); }
  .bar-fill-accuracy { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #7C3AED, #A78BFA); }
  .bar-label { width: 52px; text-align: right; font-weight: 600; }
  .model-meta { font-size: 0.7rem; color: #475569; margin-top: 0.35rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background:var(--bg-card) !important; border-right:1px solid var(--border) !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stCheckbox label { color:var(--text) !important; font-weight:500 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, #0D1117 0%, #161B2E 100%) !important;
    border-radius: 14px !important;
    padding: 0.4rem !important;
    gap: 0.3rem !important;
    border: 1px solid rgba(124,58,237,0.55) !important;
    box-shadow: 0 0 24px rgba(124,58,237,0.25), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    margin-bottom: 1.2rem !important;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.1rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.25s ease !important;
    border: 1px solid transparent !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #E2E8F0 !important;
    background: rgba(124,58,237,0.12) !important;
    border-color: rgba(124,58,237,0.3) !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED 0%, #5B21B6 60%, #06B6D4 100%) !important;
    color: white !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.55), 0 0 0 1px rgba(167,139,250,0.3) !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    transform: translateY(-1px) !important;
  }


  /* ── Live badge ── */
  .live-badge {
    margin-left:auto; background:rgba(239,68,68,0.2); border:1px solid rgba(239,68,68,0.5);
    color:#FCA5A5; padding:0.2rem 0.7rem; border-radius:50px;
    font-size:0.68rem; font-weight:700; letter-spacing:0.1em; animation:pulse 2s infinite;
  }
  .video-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .dot-red { background:#EF4444; }
  .dot-yellow { background:#F59E0B; }
  .dot-green { background:#10B981; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

  /* ── Footer ── */
  .custom-footer {
    text-align:center; color:var(--text-muted); font-size:0.78rem;
    padding:2rem 0 1rem; border-top:1px solid var(--border); margin-top:2rem;
  }
  .custom-footer span { color:var(--primary-light); font-weight:600; }

  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:var(--bg-deep); }
  ::-webkit-scrollbar-thumb { background:var(--primary); border-radius:3px; }

  /* ══════════════════════════════════════════════════
     MOBILE RESPONSIVE — max-width: 768px
  ══════════════════════════════════════════════════ */
  @media (max-width: 768px) {

    /* Hero shrink */
    .hero-banner { padding: 1.2rem 1rem; margin-bottom: 1rem; }
    .hero-title  { font-size: 1.55rem !important; }
    .hero-subtitle { font-size: 0.82rem; }
    .hero-badges { gap: 0.4rem; }
    .badge { font-size: 0.65rem; padding: 0.2rem 0.6rem; }

    /* Stats: 2 columns on mobile */
    .stats-grid {
      grid-template-columns: repeat(2, 1fr) !important;
      gap: 0.6rem !important;
    }
    .stat-value { font-size: 1.4rem !important; }
    .stat-label { font-size: 0.68rem !important; }
    .stat-card  { padding: 0.8rem !important; }

    /* Tabs — horizontal scroll on mobile */
    .stTabs [data-baseweb="tab-list"] {
      overflow-x: auto !important;
      flex-wrap: nowrap !important;
      -webkit-overflow-scrolling: touch !important;
      padding: 0.25rem !important;
      gap: 0.2rem !important;
    }
    .stTabs [data-baseweb="tab"] {
      font-size: 0.78rem !important;
      padding: 0.45rem 0.7rem !important;
      white-space: nowrap !important;
      min-width: fit-content !important;
    }

    /* Buttons — full width, large touch targets */
    .stButton > button {
      width: 100% !important;
      padding: 0.75rem 1rem !important;
      font-size: 1rem !important;
      min-height: 48px !important;
    }

    /* Detection table — smaller text */
    .detection-row { padding: 0.5rem 0.7rem; }
    .det-label { font-size: 0.85rem; }
    .det-conf  { font-size: 0.8rem; }
    .det-id    { width: 34px; font-size: 0.8rem; }

    /* Result cards */
    .result-card { padding: 0.9rem; }
    .result-meta { flex-direction: column; gap: 0.3rem; font-size: 0.74rem; }
    .result-tag  { font-size: 0.68rem; }

    /* Model card */
    .model-card { padding: 0.6rem 0.8rem; }
    .model-card-title { font-size: 0.88rem; }

    /* Info / warn boxes */
    .info-box, .warn-box, .success-box {
      font-size: 0.82rem;
      padding: 0.7rem 0.85rem;
    }

    /* Results header */
    .results-header { font-size: 1.2rem; }

    /* General layout padding */
    .main .block-container {
      padding-left: 0.8rem !important;
      padding-right: 0.8rem !important;
      padding-top: 1rem !important;
    }

    /* File uploader — larger touch zone */
    [data-testid="stFileUploader"] { min-height: 80px; }

    /* Column stacking override for small screens */
    [data-testid="stHorizontalBlock"] {
      flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
      width: 100% !important;
      min-width: 100% !important;
    }

    /* Sidebar toggle hint */
    [data-testid="collapsedControl"] {
      display: flex !important;
    }
  }

  /* Small phones */
  @media (max-width: 400px) {
    .hero-title  { font-size: 1.25rem !important; }
    .stats-grid  { grid-template-columns: repeat(2, 1fr) !important; }
    .stat-value  { font-size: 1.2rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.72rem !important; padding: 0.4rem 0.55rem !important; }
  }

  /* Tablet mid-range */
  @media (min-width: 769px) and (max-width: 1024px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr) !important; }
    .hero-title  { font-size: 2rem !important; }
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(MODEL_DIR, "results_log.json")

CONF_THRESHOLD = 0.50   # Ignore predictions below 50% confidence
IOU_THRESHOLD  = 0.45   # Non-max suppression IoU cutoff

# Use DirectShow on Windows to prevent MSMF webcam crashes (mirrors object_tracker.py)
IS_WINDOWS = sys.platform.startswith("win")

PALETTE = [
    (124, 58, 237), (6, 182, 212), (16, 185, 129), (245, 158, 11),
    (239, 68, 68),  (236, 72, 153),(14, 165, 233), (168, 85, 247),
    (251, 146, 60), (52, 211, 153),(34, 211, 238), (251, 191, 36),
]

def get_color(class_id: int):
    return PALETTE[class_id % len(PALETTE)]

# ─────────────────────────────────────────────────────────────────────────────
# Persistent Results Store (JSON file)
# ─────────────────────────────────────────────────────────────────────────────
def load_results() -> list:
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_result_entry(entry: dict):
    results = load_results()
    results.append(entry)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def clear_results():
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("total_detected", 0),
    ("frames_processed", 0),
    ("unique_ids", set()),
    ("fps_list", []),
    ("webcam_running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# Model Loader  ─  downloads into the app directory so the file persists
#                  across reruns within the same deployment session
# ─────────────────────────────────────────────────────────────────────────────

def _model_path(model_name: str) -> str:
    """Return the local path where the .pt file lives (or will be saved)."""
    return os.path.join(MODEL_DIR, model_name)

def _ensure_model(model_name: str) -> str:
    """
    Guarantee the .pt file exists in MODEL_DIR.
    If it is missing, let Ultralytics download it and then copy it
    from wherever YOLO stored it into MODEL_DIR.
    Returns the local path.
    """
    local = _model_path(model_name)
    if os.path.exists(local):
        return local

    # Let YOLO auto-download (goes to its own cache or cwd)
    tmp_model = YOLO(model_name)           # triggers download
    src = tmp_model.ckpt_path             # path YOLO actually used
    if src and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(local):
        import shutil
        shutil.copy2(src, local)
    return local if os.path.exists(local) else src

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load (and if necessary download) a YOLOv8 model, cached for the session."""
    local = _ensure_model(model_name)
    return YOLO(local)

# ─────────────────────────────────────────────────────────────────────────────
# Startup: pre-download the default (Nano) model so something is
# always ready immediately — runs once per deployment via cache.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _predownload_nano():
    """Silently ensure yolov8n.pt is present at startup."""
    try:
        _ensure_model("yolov8n.pt")
    except Exception:
        pass  # fail silently; user will see an error when they pick a model

_predownload_nano()

# ─────────────────────────────────────────────────────────────────────────────
# Core Detection + Tracking
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame, model, show_labels, show_conf, show_tracks,
                  track_history, selected_classes, class_names):
    detections = []
    annotated  = frame.copy()

    results = model.track(
        source=frame, persist=True,
        conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False,
    )

    if results and results[0].boxes is not None:
        boxes_data = results[0].boxes
        has_ids    = boxes_data.id is not None
        ids   = boxes_data.id.int().cpu().tolist() if has_ids else list(range(len(boxes_data)))
        xyxy  = boxes_data.xyxy.int().cpu().tolist()
        cls   = boxes_data.cls.int().cpu().tolist()
        confs = boxes_data.conf.float().cpu().tolist()

        for (x1, y1, x2, y2), class_id, track_id, conf in zip(xyxy, cls, ids, confs):
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            if selected_classes and label not in selected_classes:
                continue

            color = get_color(class_id)
            bgr   = (color[2], color[1], color[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if show_tracks:
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 40:
                    track_history[track_id].pop(0)
                pts = track_history[track_id]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    cv2.line(annotated, pts[i-1], pts[i], bgr, max(1, int(alpha * 3)))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

            parts = []
            if show_labels:
                parts.append(label)
            if has_ids:
                parts.append(f"#{track_id}")
            if show_conf:
                parts.append(f"{conf:.0%}")
            text = "  ".join(parts)

            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), bgr, -1)
                cv2.putText(annotated, text, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({"id": track_id, "label": label, "conf": conf, "box": (x1, y1, x2, y2)})

    return annotated, detections, track_history

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem;">
      <div style="font-size:2.2rem;">🎯</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;
                  background:linear-gradient(135deg,#A78BFA,#06B6D4);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        VisionTrack AI
      </div>
      <div style="font-size:0.7rem;color:#64748B;margin-top:0.2rem;letter-spacing:0.1em;">
        SETTINGS & CONTROLS
      </div>
    </div>
    <hr style="border-color:rgba(124,58,237,0.2);margin:0.8rem 0;">
    """, unsafe_allow_html=True)

    # ── Model Selector ──
    st.markdown("### 🤖 Choose Your Model")
    st.caption("Select based on your speed vs. accuracy needs")

    # All 5 model definitions — friendly name, file, speed%, accuracy%, size, tag, tag_cls, description
    ALL_MODELS = [
        {
            "file":  "yolov8n.pt",
            "name":  "⚡ Good",
            "label": "Nano · 6 MB",
            "desc":  "Ultra-fast. Great for real-time on low-end hardware.",
            "speed": 98, "accuracy": 38,
            "tag": "FASTEST", "tag_cls": "tag-fastest",
            "recommend": False,
        },
        {
            "file":  "yolov8s.pt",
            "name":  "🚀 Better",
            "label": "Small · 22 MB",
            "desc":  "Fast with noticeably improved detection quality.",
            "speed": 80, "accuracy": 58,
            "tag": "FAST", "tag_cls": "tag-fast",
            "recommend": False,
        },
        {
            "file":  "yolov8m.pt",
            "name":  "⚖️ Great",
            "label": "Medium · 50 MB",
            "desc":  "Best balance of speed and accuracy. Recommended.",
            "speed": 60, "accuracy": 74,
            "tag": "RECOMMENDED", "tag_cls": "tag-default",
            "recommend": True,
        },
        {
            "file":  "yolov8l.pt",
            "name":  "🎯 Excellent",
            "label": "Large · 87 MB",
            "desc":  "High accuracy, needs a decent GPU.",
            "speed": 38, "accuracy": 88,
            "tag": "HIGH ACCURACY", "tag_cls": "tag-high",
            "recommend": False,
        },
        {
            "file":  "yolov8x.pt",
            "name":  "🔬 Best",
            "label": "XLarge · 136 MB",
            "desc":  "Maximum precision. Requires a powerful GPU.",
            "speed": 18, "accuracy": 96,
            "tag": "MAX ACCURACY", "tag_cls": "tag-max",
            "recommend": False,
        },
    ]

    # Always show all 5 models — mark which are already cached on disk
    # (on Streamlit Cloud none will be cached until first use)
    def _cached(fname):
        return os.path.exists(os.path.join(MODEL_DIR, fname))

    # Build labels: cached models show ✅, others show ⬇️ (will auto-download)
    friendly_labels = [
        f"{m['name']}  —  {m['label']}  {'✅' if _cached(m['file']) else '⬇️'}"
        for m in ALL_MODELS
    ]

    # Default: prefer nano (smallest, always fast to download) on cloud;
    # prefer medium if it's already cached locally.
    cached_medium = _cached("yolov8m.pt")
    if cached_medium:
        default_idx = next(i for i, m in enumerate(ALL_MODELS) if m["file"] == "yolov8m.pt")
    else:
        default_idx = next(i for i, m in enumerate(ALL_MODELS) if m["file"] == "yolov8n.pt")

    st.caption("✅ = cached  ·  ⬇️ = will download on first use")

    chosen_label = st.radio(
        "Pick a model:",
        friendly_labels,
        index=default_idx,
        label_visibility="collapsed",
    )
    chosen_meta  = ALL_MODELS[friendly_labels.index(chosen_label)]
    model_choice = chosen_meta["file"]

    # Rich info card for the selected model
    st.markdown(f"""
    <div class="model-card selected">
      <div class="model-card-title">
        <span>{chosen_meta['name']}</span>
        <span class="model-tag {chosen_meta['tag_cls']}">{chosen_meta['tag']}</span>
      </div>
      <div class="model-meta">{chosen_meta['label']}</div>
      <div class="model-bars">
        <div class="bar-row">
          <span style="width:52px;">Speed</span>
          <div class="bar-track"><div class="bar-fill-speed" style="width:{chosen_meta['speed']}%;"></div></div>
          <span class="bar-label" style="color:#6EE7B7;">{chosen_meta['speed']}%</span>
        </div>
        <div class="bar-row">
          <span style="width:52px;">Accuracy</span>
          <div class="bar-track"><div class="bar-fill-accuracy" style="width:{chosen_meta['accuracy']}%;"></div></div>
          <span class="bar-label" style="color:#A78BFA;">{chosen_meta['accuracy']}%</span>
        </div>
      </div>
      <div style="font-size:0.73rem;color:#64748B;margin-top:0.5rem;font-style:italic;">
        {chosen_meta['desc']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Display Options ──
    st.markdown("### 🗂️ Display Options")
    show_labels = st.checkbox("Show Class Labels",  value=True)
    show_conf   = st.checkbox("Show Confidence %",  value=True)
    show_tracks = st.checkbox("Show Motion Trails", value=True)

    st.markdown("---")

    # ── Class Filter ──
    st.markdown("### 🏷️ Class Filter")
    st.caption("Leave empty to detect all 80 COCO classes")
    common_classes = ["person","car","truck","bus","motorcycle","bicycle",
                      "dog","cat","bottle","chair","laptop","cell phone"]
    selected_classes = st.multiselect("Filter Classes", common_classes, default=[],
                                      help="Show only these object types")

    st.markdown("---")

    # ── Output ──
    st.markdown("### 💾 Output")
    save_output = st.checkbox("Save Processed Video", value=False,
                              help="Write output.mp4 after video processing")

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.7rem;color:#64748B;text-align:center;padding:0.4rem 0;">
      Confidence: <b style="color:#A78BFA;">{CONF_THRESHOLD:.0%}</b> &nbsp;|&nbsp;
      IoU: <b style="color:#67E8F9;">{IOU_THRESHOLD:.2f}</b><br>
      Powered by YOLOv8 + ByteTracker
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🎯 VisionTrack AI</div>
  <div class="hero-subtitle">Real-time Object Detection & Multi-Object Tracking powered by YOLOv8</div>
  <div class="hero-badges">
    <span class="badge">YOLOv8</span>
    <span class="badge cyan">ByteTracker</span>
    <span class="badge green">80 COCO Classes</span>
    <span class="badge">Real-Time</span>
    <span class="badge cyan">Multi-Object</span>
    <span class="badge red">Results Saved</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
# Show whether this will load from cache or download fresh
_is_cached = os.path.exists(_model_path(model_choice))
_action_label = (
    f"⚡ Loading {chosen_meta['name']} from cache…"
    if _is_cached
    else f"⬇️ Downloading {chosen_meta['name']} ({chosen_meta['label']}) — please wait…"
)

with st.spinner(_action_label):
    try:
        model = load_model(model_choice)
        class_names = list(model.names.values())
        st.markdown(f"""
        <div class="success-box">
          ✅ <b>{chosen_meta['name']}</b> ({chosen_meta['label']}) ready — {len(class_names)} classes
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warn-box">
          ⚠️ Could not load <b>{chosen_meta['name']}</b>: {e}<br>
          <small>Try selecting a smaller model (⚡ Good / 🚀 Better) which downloads faster.</small>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Live Stats Row
# ─────────────────────────────────────────────────────────────────────────────
stat_row = st.empty()

def render_stats(frames, detections, unique_ids, avg_fps):
    stat_row.markdown(f"""
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon">🖼️</div>
        <div class="stat-value">{frames:,}</div>
        <div class="stat-label">Frames Processed</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🔍</div>
        <div class="stat-value">{detections:,}</div>
        <div class="stat-label">Total Detections</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🆔</div>
        <div class="stat-value">{unique_ids}</div>
        <div class="stat-label">Unique IDs Tracked</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-value">{avg_fps:.1f}</div>
        <div class="stat-label">Avg FPS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

render_stats(0, 0, 0, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_webcam, tab_video, tab_image, tab_results, tab_about = st.tabs([
    "📷  Webcam Live", "🎬  Video File", "🖼️  Image", "📊  Results", "ℹ️  About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.markdown("""
    <div class="info-box">
      📷 Uses your <b>browser's camera</b> via WebRTC — works on Streamlit Cloud, mobile &amp; desktop.
      Click <b>START</b> and allow camera permission when your browser asks.
    </div>
    """, unsafe_allow_html=True)

    if not WEBRTC_AVAILABLE:
        st.markdown("""
        <div class="warn-box">
          ⚠️ <b>streamlit-webrtc</b> is not installed. Run:<br>
          <code>pip install streamlit-webrtc av</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── RTC config (public Google STUN servers) ──────────────────────────
        RTC_CONFIG = RTCConfiguration({"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]})

        # ════════════════════════════════════════════════════════════════════
        #  YOLOProcessor  —  KEY ARCHITECTURE:
        #
        #   recv()  returns in <1 ms  (never blocks WebRTC)
        #   A dedicated worker thread runs YOLO at its own pace
        #   Frames are dropped when the worker is busy (maxsize=1 queue)
        #   The last annotated frame is returned until the next one is ready
        # ════════════════════════════════════════════════════════════════════
        class YOLOProcessor(VideoProcessorBase):
            def __init__(self):
                # --- queues & state ---
                self._in_q       = _queue.Queue(maxsize=1)   # 1 pending frame max
                self._lock       = threading.Lock()
                self._last_rgb   = None          # latest annotated frame (RGB)
                self._track_hist = {}
                # --- per-session stats ---
                self._frames   = 0
                self._all_dets = []
                self._ids      = set()
                self._fps_buf  = []
                # --- capture sidebar settings at construction time ---
                self._show_labels = show_labels
                self._show_conf   = show_conf
                self._show_tracks = show_tracks
                self._sel_cls     = selected_classes
                # --- start background YOLO worker ---
                self._running = True
                self._worker  = threading.Thread(
                    target=self._yolo_loop, daemon=True
                )
                self._worker.start()

            # ─── background thread: YOLO runs here, never touches WebRTC ───
            def _yolo_loop(self):
                while self._running:
                    try:
                        img = self._in_q.get(timeout=0.5)
                    except _queue.Empty:
                        continue
                    t0 = time.time()
                    try:
                        annotated, dets, self._track_hist = process_frame(
                            img, model,
                            self._show_labels, self._show_conf, self._show_tracks,
                            self._track_hist, self._sel_cls, class_names,
                        )
                        elapsed = time.time() - t0
                        fps = 1.0 / max(elapsed, 1e-6)

                        # FPS overlay on the annotated frame
                        cv2.rectangle(annotated, (0, 0), (160, 30), (13, 17, 23), -1)
                        cv2.putText(annotated, f"FPS: {fps:.1f}", (7, 21),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (124, 58, 237), 2)

                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                        with self._lock:
                            self._last_rgb = rgb
                            self._frames  += 1
                            self._all_dets.extend(dets)
                            for d in dets:
                                self._ids.add(d["id"])
                            self._fps_buf.append(fps)
                    except Exception:
                        pass   # keep running even if one frame fails

            # ─── WebRTC callback: MUST return in milliseconds ───
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Enqueue frame (non-blocking — drop if worker is still busy)
                try:
                    self._in_q.put_nowait(img)
                except _queue.Full:
                    pass  # worker busy — skip this frame, video stays smooth

                # Return the latest processed frame (or the raw frame if YOLO
                # hasn't finished its first inference yet)
                with self._lock:
                    out = self._last_rgb
                if out is None:
                    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                return av.VideoFrame.from_ndarray(out, format="rgb24")

            # ─── Called when the WebRTC stream stops ───
            def on_ended(self):
                self._running = False

            # ─── Thread-safe snapshot of current stats ───
            def snapshot(self):
                with self._lock:
                    return (
                        self._frames,
                        list(self._all_dets),
                        set(self._ids),
                        list(self._fps_buf),
                    )

        # ── Camera feed header bar ───────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(90deg,rgba(124,58,237,0.2),rgba(6,182,212,0.1));
             border-radius:12px 12px 0 0; padding:0.65rem 1.1rem;
             display:flex;align-items:center;gap:0.5rem;
             border:1px solid rgba(124,58,237,0.2);border-bottom:none;">
          <span class="video-dot dot-red"></span>
          <span class="video-dot dot-yellow"></span>
          <span class="video-dot dot-green"></span>
          <span style="color:#94A3B8;font-size:0.8rem;margin-left:0.4rem;">Live Camera Feed  (YOLO annotated)</span>
          <span class="live-badge">● LIVE</span>
        </div>
        """, unsafe_allow_html=True)

        # ── WebRTC streamer ──────────────────────────────────────────────────
        ctx = webrtc_streamer(
            key="visiontrack-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=YOLOProcessor,
            # Lower resolution + FPS → faster YOLO inference
            media_stream_constraints={
                "video": {
                    "width":     {"ideal": 640,  "max": 1280},
                    "height":    {"ideal": 480,  "max": 720},
                    "frameRate": {"ideal": 15,   "max": 30},
                },
                "audio": False,
            },
            # async_processing=False: recv() is called synchronously;
            # our queue makes it return in <1 ms so this is safe.
            async_processing=False,
        )

        # ── Live stats (reads processor state directly) ──────────────────────
        stats_ph  = st.empty()
        detect_ph = st.empty()

        if ctx.state.playing and ctx.video_processor:
            frames, all_dets, ids, fps_buf = ctx.video_processor.snapshot()
            avg_fps = float(np.mean(fps_buf[-30:])) if fps_buf else 0.0

            stats_ph.markdown(f"""
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-icon">🖼️</div>
                <div class="stat-value">{frames:,}</div>
                <div class="stat-label">Frames Processed</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">🔍</div>
                <div class="stat-value">{len(all_dets):,}</div>
                <div class="stat-label">Total Detections</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">🆔</div>
                <div class="stat-value">{len(ids)}</div>
                <div class="stat-label">Unique IDs Tracked</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">⚡</div>
                <div class="stat-value">{avg_fps:.1f}</div>
                <div class="stat-label">Avg FPS</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Show last 8 detections
            recent = all_dets[-8:] if all_dets else []
            if recent:
                rows = "".join(f"""
                <div class="detection-row">
                  <div class="det-id">#{d['id']}</div>
                  <div class="det-label">{d['label']}</div>
                  <div class="conf-bar-wrap">
                    <div class="conf-bar-fill" style="width:{int(d['conf']*100)}%;"></div>
                  </div>
                  <div class="det-conf">{int(d['conf']*100)}%</div>
                </div>""" for d in recent)
                detect_ph.markdown(
                    f'<div class="detection-table">{rows}</div>',
                    unsafe_allow_html=True
                )

        elif not ctx.state.playing:
            # ── Session ended: save results if we have any ───────────────────
            proc = getattr(ctx, "video_processor", None)
            if proc is not None:
                frames, all_dets, ids, fps_buf = proc.snapshot()
                if all_dets and frames > 0:
                    label_counts = {}
                    for d in all_dets:
                        label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
                    avg_fps = round(float(np.mean(fps_buf)), 1) if fps_buf else 0.0
                    save_result_entry({
                        "source": "Webcam (WebRTC)",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": f"{chosen_meta['name']} ({chosen_meta['label']})",
                        "frames": frames,
                        "total_detections": len(all_dets),
                        "unique_ids": len(ids),
                        "avg_fps": avg_fps,
                        "classes_detected": label_counts,
                    })
                    st.markdown(
                        '<div class="success-box">✅ Session saved to Results tab.</div>',
                        unsafe_allow_html=True
                    )
            else:
                # No processor yet — show idle placeholder
                st.markdown("""
                <div style="background:var(--bg-card);border:1px solid rgba(124,58,237,0.2);
                            border-radius:0 0 14px 14px;padding:3rem 2rem;text-align:center;">
                  <div style="font-size:3rem;margin-bottom:0.8rem;">📷</div>
                  <div style="font-size:1rem;font-weight:600;color:#64748B;">Camera not active</div>
                  <div style="font-size:0.83rem;margin-top:0.5rem;color:#475569;">
                    Click <b style="color:#A78BFA;">START</b> above and allow camera permission.
                  </div>
                </div>
                """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO FILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("""
    <div class="info-box">
      🎬 Upload any video file. Results are automatically saved to the <b>Results</b> tab after processing.
    </div>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Drop your video here", type=["mp4","avi","mov","mkv","webm"],
        label_visibility="collapsed",
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        video_path = tfile.name

        col_a, col_b = st.columns(2)
        with col_a:
            max_frames  = st.slider("Max frames (0 = all)", 0, 2000, 0, 50)
        with col_b:
            skip_frames = st.slider("Process every N-th frame", 1, 5, 1)

        if st.button("🚀 Process Video", key="proc_vid"):
            track_history = {}
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 30
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            limit        = max_frames if max_frames > 0 else total_frames

            st.markdown(f"""
            <div class="success-box">
              📊 <b>{uploaded_video.name}</b> — {total_frames} frames · {orig_fps:.1f} FPS · {width}×{height}
            </div>
            """, unsafe_allow_html=True)

            out_writer = None
            if save_output:
                out_path   = os.path.join(MODEL_DIR, "output.mp4")
                out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             orig_fps, (width, height))

            progress_bar = st.progress(0, text="Processing video…")
            vid_ph       = st.empty()

            frame_idx = processed = total_dets = 0
            unique_ids = set()
            fps_list   = []
            all_dets   = []

            while True:
                ret, frame = cap.read()
                if not ret or processed >= limit:
                    break
                frame_idx += 1
                if frame_idx % skip_frames != 0:
                    continue

                t0 = time.time()
                annotated, dets, track_history = process_frame(
                    frame, model, show_labels, show_conf, show_tracks,
                    track_history, selected_classes, class_names,
                )
                fps_val = 1.0 / (time.time() - t0) if (time.time() - t0) > 0 else 0

                processed  += 1
                total_dets += len(dets)
                all_dets.extend(dets)
                for d in dets:
                    unique_ids.add(d["id"])
                fps_list.append(fps_val)

                if out_writer:
                    out_writer.write(annotated)

                cv2.rectangle(annotated, (0, 0), (200, 30), (13, 17, 23), -1)
                cv2.putText(annotated, f"Frame {processed}/{limit}  {fps_val:.1f}fps",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (124, 58, 237), 1)

                vid_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             channels="RGB", use_container_width=True)
                progress_bar.progress(min(processed / limit, 1.0),
                                      text=f"Processing… {processed}/{limit} frames")
                render_stats(processed, total_dets, len(unique_ids),
                             float(np.mean(fps_list[-30:])))

            cap.release()
            if out_writer:
                out_writer.release()

            progress_bar.progress(1.0, text="✅ Processing complete!")

            # Save to results
            if all_dets:
                label_counts = {}
                for d in all_dets:
                    label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
                save_result_entry({
                    "source": f"Video: {uploaded_video.name}",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": f"{chosen_meta['name']} ({chosen_meta['label']})",
                    "frames": processed,
                    "total_detections": total_dets,
                    "unique_ids": len(unique_ids),
                    "avg_fps": round(float(np.mean(fps_list)), 1),
                    "classes_detected": label_counts,
                })
                st.markdown('<div class="success-box">✅ Results saved to the Results tab.</div>',
                            unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--bg-card);border:2px dashed rgba(124,58,237,0.3);
                    border-radius:14px;padding:4rem 2rem;text-align:center;">
          <div style="font-size:3.5rem;margin-bottom:1rem;">🎬</div>
          <div style="font-size:1.05rem;font-weight:600;color:#64748B;">No video uploaded</div>
          <div style="font-size:0.83rem;margin-top:0.4rem;color:#475569;">Supports MP4, AVI, MOV, MKV, WebM</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown("""
    <div class="info-box">
      🖼️ Upload an image for instant detection. Results are saved to the <b>Results</b> tab.
    </div>
    """, unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Drop image here", type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed", key="img_uploader"
    )

    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_det:
            st.markdown("**Detected**")
            results  = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            annotated = frame.copy()
            dets = []

            if results and results[0].boxes is not None:
                boxes_data = results[0].boxes
                xyxy  = boxes_data.xyxy.int().cpu().tolist()
                cls   = boxes_data.cls.int().cpu().tolist()
                confs = boxes_data.conf.float().cpu().tolist()

                for (x1, y1, x2, y2), class_id, conf in zip(xyxy, cls, confs):
                    label = class_names[class_id] if class_id < len(class_names) else str(class_id)
                    if selected_classes and label not in selected_classes:
                        continue
                    color = get_color(class_id)
                    bgr   = (color[2], color[1], color[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
                    text = f"{label}  {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                    cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+8, y1), bgr, -1)
                    cv2.putText(annotated, text, (x1+4, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
                    dets.append({"label": label, "conf": round(conf, 3)})

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if dets:
            st.markdown("### 📋 Detections")
            rows = "".join(f"""
            <div class="detection-row">
              <div class="det-id">#{i+1}</div>
              <div class="det-label">{d['label']}</div>
              <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{int(d['conf']*100)}%;"></div>
              </div>
              <div class="det-conf">{int(d['conf']*100)}%</div>
            </div>""" for i, d in enumerate(dets))
            st.markdown(f'<div class="detection-table">{rows}</div>', unsafe_allow_html=True)

            col_dl, col_sv = st.columns(2)
            with col_dl:
                success, buffer = cv2.imencode(".jpg", annotated)
                if success:
                    st.download_button("⬇️ Download Annotated Image",
                                       data=buffer.tobytes(), file_name="detected.jpg",
                                       mime="image/jpeg")
            with col_sv:
                if st.button("💾 Save to Results", key="save_img_result"):
                    label_counts = {}
                    for d in dets:
                        label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
                    save_result_entry({
                        "source": f"Image: {uploaded_img.name}",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model_choice,
                        "frames": 1,
                        "total_detections": len(dets),
                        "unique_ids": len(dets),
                        "avg_fps": 0,
                        "classes_detected": label_counts,
                    })
                    st.markdown('<div class="success-box">✅ Saved to Results tab!</div>',
                                unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">No objects detected at current threshold.</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--bg-card);border:2px dashed rgba(124,58,237,0.3);
                    border-radius:14px;padding:4rem 2rem;text-align:center;">
          <div style="font-size:3.5rem;margin-bottom:1rem;">🖼️</div>
          <div style="font-size:1.05rem;font-weight:600;color:#64748B;">No image uploaded</div>
          <div style="font-size:0.83rem;margin-top:0.4rem;color:#475569;">JPG, PNG, BMP, WebP supported</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown('<div class="results-header">📊 Detection Results History</div>', unsafe_allow_html=True)

    results_data = load_results()

    if not results_data:
        st.markdown("""
        <div style="background:var(--bg-card);border:2px dashed rgba(124,58,237,0.3);
                    border-radius:14px;padding:4rem 2rem;text-align:center;">
          <div style="font-size:3.5rem;margin-bottom:1rem;">📂</div>
          <div style="font-size:1.05rem;font-weight:600;color:#64748B;">No results yet</div>
          <div style="font-size:0.83rem;margin-top:0.4rem;color:#475569;">
            Run detection on a webcam, video or image — results appear here automatically
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Summary Stats ──
        total_sessions  = len(results_data)
        total_dets_all  = sum(r.get("total_detections", 0) for r in results_data)
        total_ids_all   = sum(r.get("unique_ids", 0) for r in results_data)
        total_frames_all= sum(r.get("frames", 0) for r in results_data)

        st.markdown(f"""
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-icon">🗂️</div>
            <div class="stat-value">{total_sessions}</div>
            <div class="stat-label">Total Sessions</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🔍</div>
            <div class="stat-value">{total_dets_all:,}</div>
            <div class="stat-label">All-Time Detections</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🆔</div>
            <div class="stat-value">{total_ids_all:,}</div>
            <div class="stat-label">Unique IDs (total)</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🖼️</div>
            <div class="stat-value">{total_frames_all:,}</div>
            <div class="stat-label">Frames Analysed</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Class Frequency across all sessions ──
        all_classes: dict = {}
        for r in results_data:
            for cls, cnt in r.get("classes_detected", {}).items():
                all_classes[cls] = all_classes.get(cls, 0) + cnt

        if all_classes:
            import pandas as pd
            st.markdown("### 🏷️ Most Detected Classes (all sessions)")
            df_cls = pd.DataFrame(list(all_classes.items()), columns=["Class","Count"])
            df_cls = df_cls.sort_values("Count", ascending=False).head(15)
            st.bar_chart(df_cls.set_index("Class"))

        st.markdown("---")
        st.markdown("### 📋 Session History")

        # ── Individual Result Cards (newest first) ──
        for entry in reversed(results_data):
            cls_tags = "".join(
                f'<span class="result-tag">{cls} <b>×{cnt}</b></span>'
                for cls, cnt in entry.get("classes_detected", {}).items()
            )
            fps_str = f"{entry.get('avg_fps', 0):.1f} FPS" if entry.get("avg_fps", 0) > 0 else "—"
            st.markdown(f"""
            <div class="result-card">
              <div style="font-weight:700;font-size:1rem;color:#E2E8F0;margin-bottom:0.4rem;">
                {entry.get('source', 'Unknown')}
              </div>
              <div class="result-meta">
                <span>🕐 {entry.get('timestamp', '—')}</span>
                <span>🤖 {entry.get('model', '—')}</span>
                <span>🖼️ {entry.get('frames', 0):,} frames</span>
                <span>🔍 {entry.get('total_detections', 0):,} detections</span>
                <span>🆔 {entry.get('unique_ids', 0)} IDs</span>
                <span>⚡ {fps_str}</span>
              </div>
              <div class="result-tags">{cls_tags if cls_tags else '<span style="color:#475569;font-size:0.78rem;">No objects detected</span>'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Export & Clear ──
        import pandas as pd
        col_exp, col_clr = st.columns([1, 1])
        with col_exp:
            flat_rows = []
            for r in results_data:
                for cls, cnt in r.get("classes_detected", {}).items():
                    flat_rows.append({
                        "Timestamp": r.get("timestamp"),
                        "Source":    r.get("source"),
                        "Model":     r.get("model"),
                        "Class":     cls,
                        "Count":     cnt,
                        "Frames":    r.get("frames"),
                        "Total Detections": r.get("total_detections"),
                        "Unique IDs": r.get("unique_ids"),
                        "Avg FPS":   r.get("avg_fps"),
                    })
            if flat_rows:
                df_export = pd.DataFrame(flat_rows)
                st.download_button(
                    "⬇️ Export Results as CSV",
                    data=df_export.to_csv(index=False).encode("utf-8"),
                    file_name="visiontrack_results.csv",
                    mime="text/csv",
                )
        with col_clr:
            if st.button("🗑️ Clear All Results", key="clear_results"):
                clear_results()
                st.markdown('<div class="warn-box">🗑️ All results cleared. Reload to refresh.</div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
        ## 🎯 About VisionTrack AI

        **VisionTrack AI** is a production-ready computer vision application built on
        [YOLOv8](https://github.com/ultralytics/ultralytics) combined with ByteTracker
        for real-time multi-object tracking.

        ### ✨ Features
        - **Real-time webcam** detection & tracking
        - **Video file** processing with optional export
        - **Single image** inference with download
        - **Results tab** — persistent history of all sessions
        - **80 COCO classes** — people, vehicles, animals & more
        - **Colour-coded** per-class bounding boxes
        - **Motion trails** showing object trajectories
        - **CSV export** of all detection results
        - **Class filter** to focus on specific object types

        ### 🏗️ Architecture
        | Component | Technology |
        |-----------|------------|
        | Detection | YOLOv8 (n / s / m / l / x) |
        | Tracking  | ByteTracker (built into YOLO) |
        | Backend   | Python + OpenCV |
        | UI        | Streamlit |
        | Dataset   | MS COCO (80 classes) |
        """)

    with col_r:
        st.markdown("""
        ### 📦 Model Guide
        | Rating | Name | Size | Speed | Accuracy |
        |--------|------|------|-------|----------|
        | ⚡ **Good** | Nano | 6 MB | ⚡⚡⚡⚡ 98% | ★☆☆☆☆ 38% |
        | 🚀 **Better** | Small | 22 MB | ⚡⚡⚡ 80% | ★★☆☆☆ 58% |
        | ⚖️ **Great** ★ | Medium | 50 MB | ⚡⚡ 60% | ★★★☆☆ 74% |
        | 🎯 **Excellent** | Large | 87 MB | ⚡ 38% | ★★★★☆ 88% |
        | 🔬 **Best** | XLarge | 136 MB | 🐢 18% | ★★★★★ 96% |

        ### 💡 Tips
        - **⚖️ Great (Medium)** — best for most users
        - **⚡ Good (Nano)** — use on laptops or no GPU
        - **🔬 Best (XLarge)** — use with a dedicated GPU
        - Use **Class Filter** to focus on specific objects
        - Check the **Results tab** for all-time stats
        """)

    st.markdown("""
    ---
    ### 🔬 How It Works
    ```
    Input Frame → YOLOv8 Inference → NMS → ByteTracker → Annotated Output → Results Log
    ```
    1. **Frame Capture** — OpenCV reads from webcam or video
    2. **Inference** — YOLOv8 predicts bounding boxes & class scores
    3. **NMS** — Non-Maximum Suppression removes duplicate detections
    4. **Tracking** — ByteTracker assigns consistent IDs across frames
    5. **Rendering** — Annotated frames displayed in real-time
    6. **Logging** — Every session saved to the Results tab
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
  <span>VisionTrack AI</span> · YOLOv8 + ByteTracker + Streamlit ·
  Real-time Object Detection & Tracking
</div>
""", unsafe_allow_html=True)
