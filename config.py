# config.py
# ─────────────────────────────────────────────────────────────
# Central configuration for the entire system.
# Every other module imports from here — never hardcode values.
# ─────────────────────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "data"
IMAGES_DIR      = DATA_DIR / "images"
ENCODINGS_DIR   = DATA_DIR / "encodings"
ENCODINGS_FILE  = ENCODINGS_DIR / "encodings.pkl"
DATABASE_FILE   = BASE_DIR / "attendance.db"
LOG_FILE        = BASE_DIR / "system.log"

# Auto-create directories if they don't exist
for directory in [IMAGES_DIR, ENCODINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX        = 0       # 0 = built-in webcam, 1 = external
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
FRAME_SKIP          = 3       # Process every Nth frame (performance)
DETECTION_SCALE     = 0.5     # Scale down for faster detection

# ── Face Recognition ─────────────────────────────────────────
# DeepFace model options (in order of accuracy vs speed):
# "ArcFace"   → Best accuracy  (recommended ✅)
# "Facenet512"→ Very accurate, slightly faster
# "Facenet"   → Fast, good accuracy
# "VGG-Face"  → Classic, decent accuracy
RECOGNITION_MODEL   = "ArcFace"
DETECTOR_BACKEND    = "mediapipe"   # Fast, accurate, no cmake needed

# Cosine distance threshold (lower = stricter matching)
# ArcFace:    0.40 is strict, 0.50 is balanced, 0.68 is lenient
# Facenet512: 0.30 strict, 0.40 balanced
SIMILARITY_THRESHOLD = 0.45

# Minimum images per person for reliable recognition
MIN_IMAGES_PER_PERSON = 5
# Images captured during data collection
CAPTURE_COUNT         = 10

# ── Attendance Window ────────────────────────────────────────
# Set these in .env or change here directly
ATTENDANCE_START    = os.getenv("ATTENDANCE_START", "00:00:00")
ATTENDANCE_END      = os.getenv("ATTENDANCE_END",   "23:59:59")

# Cooldown: seconds before same person can be marked again
# Prevents duplicate marks if person stays in frame
ATTENDANCE_COOLDOWN = 30

# ── Liveness Detection ───────────────────────────────────────
LIVENESS_ENABLED         = True
BLINK_THRESHOLD          = 0.25   # EAR below this = eye closed
BLINK_CONSECUTIVE_FRAMES = 2      # Frames eye must be closed
REQUIRED_BLINKS          = 2      # Blinks needed to pass liveness

# ── Email Notifications ──────────────────────────────────────
EMAIL_ENABLED   = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587

# ── Flask ────────────────────────────────────────────────────
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
FLASK_DEBUG      = os.getenv("FLASK_DEBUG", "false").lower() == "true"
FLASK_PORT       = int(os.getenv("FLASK_PORT", "5001"))

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL   = "DEBUG" if FLASK_DEBUG else "INFO"
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE    = "%Y-%m-%d %H:%M:%S"


def print_config():
    """Print current config — useful for debugging."""
    print("\n" + "="*55)
    print("   FACE ATTENDANCE SYSTEM — CONFIGURATION")
    print("="*55)
    print(f"  Base Dir         : {BASE_DIR}")
    print(f"  Images Dir       : {IMAGES_DIR}")
    print(f"  Encodings File   : {ENCODINGS_FILE}")
    print(f"  Database         : {DATABASE_FILE}")
    print(f"  Recognition Model: {RECOGNITION_MODEL}")
    print(f"  Detector Backend : {DETECTOR_BACKEND}")
    print(f"  Similarity Thresh: {SIMILARITY_THRESHOLD}")
    print(f"  Attendance Window: {ATTENDANCE_START} → {ATTENDANCE_END}")
    print(f"  Liveness Enabled : {LIVENESS_ENABLED}")
    print(f"  Email Enabled    : {EMAIL_ENABLED}")
    print(f"  Flask Port       : {FLASK_PORT}")
    print("="*55 + "\n")


if __name__ == "__main__":
    print_config()