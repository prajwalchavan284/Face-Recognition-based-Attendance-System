import logging
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import cv2
import numpy as np
import config
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
logger = setup_logger(__name__)
def is_attendance_open() -> bool:
    try:
        now   = datetime.now().time()
        start = datetime.strptime(config.ATTENDANCE_START, "%H:%M:%S").time()
        end   = datetime.strptime(config.ATTENDANCE_END,   "%H:%M:%S").time()
        return start <= now <= end
    except Exception as e:
        logger.error("Time window check failed: %s", e)
        return False
def is_before_window() -> bool:
    try:
        now   = datetime.now().time()
        start = datetime.strptime(config.ATTENDANCE_START, "%H:%M:%S").time()
        return now < start
    except Exception:
        return False
def get_time_remaining() -> str:
    try:
        end = datetime.strptime(config.ATTENDANCE_END, "%H:%M:%S")
        end = end.replace(
            year=datetime.now().year,
            month=datetime.now().month,
            day=datetime.now().day
        )
        delta = end - datetime.now()
        if delta.total_seconds() <= 0:
            return "Closed"
        minutes = int(delta.total_seconds() // 60)
        seconds = int(delta.total_seconds() % 60)
        return f"{minutes}m {seconds}s remaining"
    except Exception:
        return "Unknown"
def preprocess_frame(frame: np.ndarray,
                     scale: float = None) -> np.ndarray:
    scale = scale or config.DETECTION_SCALE
    if scale != 1.0:
        h, w  = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame
def equalize_histogram(frame: np.ndarray) -> np.ndarray:
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
def align_face(frame: np.ndarray,
               landmarks: dict) -> np.ndarray:
    try:
        left_eye  = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        if left_eye is None or right_eye is None:
            return frame
        dx    = right_eye[0] - left_eye[0]
        dy    = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        h, w  = frame.shape[:2]
        center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(frame, M, (w, h),
                                 flags=cv2.INTER_CUBIC)
        return aligned
    except Exception as e:
        logger.warning("Face alignment failed: %s", e)
        return frame
def draw_face_box(frame: np.ndarray, bbox: tuple,
                  name: str, confidence: float,
                  color: tuple = None) -> np.ndarray:
    if color is None:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label     = f"{name} ({confidence:.0%})" if name != "Unknown" else "Unknown"
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    thickness  = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y2 - th - 10), (x1 + tw + 8, y2), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 4, y2 - 5),
                font, font_scale, (255, 255, 255), thickness)
    return frame
def draw_status_bar(frame: np.ndarray, message: str,
                    color: tuple = (0, 200, 0)) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, message, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
def draw_liveness_prompt(frame: np.ndarray,
                         blinks_done: int,
                         required: int) -> np.ndarray:
    h, w = frame.shape[:2]
    msg  = f"Liveness: Blink {blinks_done}/{required} times"
    cv2.putText(frame, msg, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return frame
def send_absent_email(absent_students: list) -> bool:
    if not config.EMAIL_ENABLED:
        logger.info("Email disabled. Skipping notification.")
        return False
    if not absent_students:
        logger.info("No absent students. Skipping email.")
        return True
    try:
        msg            = MIMEMultipart()
        msg["From"]    = config.EMAIL_SENDER
        msg["To"]      = config.EMAIL_RECIPIENT
        msg["Subject"] = f"Absent Students — {datetime.now().strftime('%d %b %Y')}"
        names = "\n".join(
            f"  • {s['name']} ({s['student_id']})" for s in absent_students
        )
        body = (
            f"Attendance Report — {datetime.now().strftime('%d %B %Y')}\n\n"
            f"The following {len(absent_students)} student(s) were absent:\n\n"
            f"{names}\n\n"
            f"— Face Attendance System"
        )
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            server.sendmail(config.EMAIL_SENDER, config.EMAIL_RECIPIENT,
                            msg.as_string())
        logger.info("✅ Absent student email sent to %s", config.EMAIL_RECIPIENT)
        return True
    except Exception as e:
        logger.error("❌ Failed to send email: %s", e)
        return False
class CooldownTracker:
    def __init__(self, cooldown_seconds: int = None):
        self.cooldown = cooldown_seconds or config.ATTENDANCE_COOLDOWN
        self._last_seen: dict[str, float] = {}
    def can_mark(self, student_id: str) -> bool:
        now  = time.time()
        last = self._last_seen.get(student_id, 0)
        return (now - last) >= self.cooldown
    def mark(self, student_id: str):
        self._last_seen[student_id] = time.time()
    def seconds_until_ready(self, student_id: str) -> int:
        now  = time.time()
        last = self._last_seen.get(student_id, 0)
        remaining = self.cooldown - (now - last)
        return max(0, int(remaining))
if __name__ == "__main__":
    print("Testing utils.py...\n")
    test_logger = setup_logger("test")
    test_logger.info("Logger is working")
    print(f"Attendance open   : {is_attendance_open()}")
    print(f"Before window     : {is_before_window()}")
    print(f"Time remaining    : {get_time_remaining()}")
    tracker = CooldownTracker(cooldown_seconds=5)
    print(f"\nCooldown can_mark : {tracker.can_mark('CS001')}")
    tracker.mark("CS001")
    print(f"After mark        : {tracker.can_mark('CS001')}")
    print(f"Seconds remaining : {tracker.seconds_until_ready('CS001')}")
    print("\n✅ utils.py working perfectly!")