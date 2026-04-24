import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "data"
IMAGES_DIR      = DATA_DIR / "images"
ENCODINGS_DIR   = DATA_DIR / "encodings"
ENCODINGS_FILE  = ENCODINGS_DIR / "encodings.pkl"
DATABASE_FILE   = BASE_DIR / "attendance.db"
LOG_FILE        = BASE_DIR / "system.log"
for directory in [IMAGES_DIR, ENCODINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
CAMERA_INDEX        = int(os.getenv("CAMERA_INDEX", "1"))
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
FRAME_SKIP          = 3       
DETECTION_SCALE     = 0.5     
RECOGNITION_MODEL   = "ArcFace"
DETECTOR_BACKEND    = "mediapipe"   
SIMILARITY_THRESHOLD = 0.45
MIN_IMAGES_PER_PERSON = 5
CAPTURE_COUNT         = 10
ATTENDANCE_START    = os.getenv("ATTENDANCE_START", "00:00:00")
ATTENDANCE_END      = os.getenv("ATTENDANCE_END",   "23:59:59")
ATTENDANCE_COOLDOWN = 30
LIVENESS_ENABLED         = True
BLINK_THRESHOLD          = 0.25   
BLINK_CONSECUTIVE_FRAMES = 2      
REQUIRED_BLINKS          = 2      
EMAIL_ENABLED   = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
FLASK_DEBUG      = os.getenv("FLASK_DEBUG", "false").lower() == "true"
FLASK_PORT       = int(os.getenv("FLASK_PORT", "5001"))
LOG_LEVEL   = "DEBUG" if FLASK_DEBUG else "INFO"
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE    = "%Y-%m-%d %H:%M:%S"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SESSION_LIFETIME_MINUTES = 60
SUBJECTS = [
    "Mathematics",
    "Data Structures",
    "Computer Networks",
    "Operating Systems",
    "Machine Learning",
    "Database Systems",
    "Software Engineering",
    "General",
]
DEFAULT_SUBJECT = "General"
RECOGNITION_VOTE_FRAMES = 5     
RECOGNITION_VOTE_THRESH = 3     
LBP_VARIANCE_THRESHOLD  = 150   
def print_config():
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