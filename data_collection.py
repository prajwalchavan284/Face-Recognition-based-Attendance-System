# data_collection.py
# ─────────────────────────────────────────────────────────────
# Captures training images for a student using webcam.
# MediaPipe detects faces — only clean, sharp, centered,
# well-lit images are saved. Quality over quantity.
# ─────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

import config
import database
from utils import setup_logger, equalize_histogram, draw_status_bar

logger = setup_logger(__name__)

# ── MediaPipe Setup ───────────────────────────────────────────
_mp_face   = mp.solutions.face_detection
_mp_draw   = mp.solutions.drawing_utils


# ── Image Quality Checks ──────────────────────────────────────

def _blur_score(image: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Below 80 = reject."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _is_centered(bbox: tuple, fw: int, fh: int,
                 margin: float = 0.18) -> bool:
    """Face must not be too close to frame edges."""
    x1, y1, x2, y2 = bbox
    return (x1 > fw * margin and x2 < fw * (1 - margin) and
            y1 > fh * margin and y2 < fh * (1 - margin))


def _is_large_enough(bbox: tuple, fw: int, fh: int,
                     min_ratio: float = 0.12) -> bool:
    """Face must occupy at least min_ratio of frame area."""
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) * (y2 - y1)) / (fw * fh) >= min_ratio


def _extract_crop(frame: np.ndarray,
                  detection,
                  padding: float = 0.30) -> tuple:
    """
    Crop face from frame with padding for context.
    Returns (crop, bbox) or (None, None).
    """
    h, w = frame.shape[:2]
    b    = detection.location_data.relative_bounding_box
    x1   = max(0, int((b.xmin - padding * b.width)  * w))
    y1   = max(0, int((b.ymin - padding * b.height) * h))
    x2   = min(w, int((b.xmin + b.width  * (1 + padding)) * w))
    y2   = min(h, int((b.ymin + b.height * (1 + padding)) * h))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


# ── Main Collection Function ──────────────────────────────────

def collect_images(student_id: str,
                   name: str,
                   count: int = None,
                   email: str = None,
                   department: str = None,
                   register_to_db: bool = True) -> bool:
    """
    Opens webcam and captures `count` quality-verified face
    images for the given student. Saves to data/images/<id>/.

    Returns True on success (>= MIN_IMAGES_PER_PERSON saved).
    """
    count    = count or config.CAPTURE_COUNT
    save_dir = config.IMAGES_DIR / student_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Register in DB
    if register_to_db:
        try:
            database.add_student(
                student_id=student_id, name=name,
                email=email, department=department,
                image_folder=str(save_dir)
            )
            logger.info("Registered: %s (%s)", name, student_id)
        except ValueError as e:
            logger.warning("%s — continuing capture.", e)

    # Camera init
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error("Cannot open camera %d", config.CAMERA_INDEX)
        return False

    hints = [
        "Look straight at the camera",
        "Turn slightly LEFT",
        "Turn slightly RIGHT",
        "Tilt head UP slightly",
        "Tilt head DOWN slightly",
        "Move a little closer",
        "Move a little further back",
        "Neutral expression",
        "Slight smile",
        "Look straight again",
    ]

    saved       = 0
    frame_count = 0
    rejected    = dict(no_face=0, blur=0, position=0, size=0)

    print("\n" + "=" * 55)
    print(f"  CAPTURING: {name}  ({student_id})")
    print(f"  Target   : {count} images")
    print("  SPACE = manual capture  |  Q = quit early")
    print("=" * 55 + "\n")

    with _mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.75
    ) as detector:

        while saved < count:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera read failed.")
                break

            frame_count += 1
            h, w         = frame.shape[:2]
            display      = frame.copy()

            enhanced = equalize_histogram(frame)
            rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            results  = detector.process(rgb)

            valid        = False
            status_msg   = "⚠️  No face — check lighting"
            status_color = (0, 0, 255)

            if (results.detections and
                    len(results.detections) == 1):

                det              = results.detections[0]
                crop, bbox       = _extract_crop(frame, det)

                if crop is not None:
                    x1, y1, x2, y2 = bbox
                    blur           = _blur_score(crop)
                    centered       = _is_centered(bbox, w, h)
                    large          = _is_large_enough(bbox, w, h)

                    if blur < 80:
                        rejected["blur"] += 1
                        status_msg   = f"❌ Hold still — too blurry ({blur:.0f})"
                        status_color = (0, 100, 255)
                    elif not centered:
                        rejected["position"] += 1
                        status_msg   = "❌ Center your face in frame"
                        status_color = (0, 165, 255)
                    elif not large:
                        rejected["size"] += 1
                        status_msg   = "❌ Move closer to camera"
                        status_color = (0, 165, 255)
                    else:
                        valid        = True
                        status_msg   = (f"✅ {saved}/{count}  "
                                        f"blur={blur:.0f}  VALID")
                        status_color = (0, 220, 0)

                    color = (0, 220, 0) if valid else (0, 0, 255)
                    cv2.rectangle(display,
                                  (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, f"{blur:.0f}",
                                (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)
                else:
                    rejected["no_face"] += 1
            else:
                rejected["no_face"] += 1

            # ── Overlays ──────────────────────────────────────
            draw_status_bar(display, status_msg, status_color)

            # Progress bar
            prog = int((saved / count) * w)
            cv2.rectangle(display, (0, h - 8),
                          (w, h), (40, 40, 40), cv2.FILLED)
            cv2.rectangle(display, (0, h - 8),
                          (prog, h), (0, 200, 0), cv2.FILLED)

            # Hint
            if saved < len(hints):
                cv2.putText(display,
                            f"Hint: {hints[saved]}",
                            (10, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (255, 220, 0), 1)

            # Counter top-right
            cv2.putText(display,
                        f"{saved}/{count}",
                        (w - 90, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)

            cv2.imshow(f"Capture — {name}", display)
            key = cv2.waitKey(1) & 0xFF

            # Auto-save every 10 valid frames OR manual SPACE
            if valid and (frame_count % 10 == 0 or
                          key == ord(" ")):
                fname = save_dir / f"{student_id}_{saved+1:03d}.jpg"
                cv2.imwrite(str(fname), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                logger.info("Saved %d/%d → %s",
                            saved, count, fname.name)

            if key == ord("q"):
                logger.info("Quit early at %d/%d.", saved, count)
                break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print(f"  DONE: {saved}/{count} images saved → {save_dir}")
    print(f"  Rejected  blur={rejected['blur']}  "
          f"position={rejected['position']}  "
          f"size={rejected['size']}  "
          f"no_face={rejected['no_face']}")
    print("=" * 55 + "\n")

    if saved < config.MIN_IMAGES_PER_PERSON:
        logger.warning("Only %d images. Min %d needed. "
                       "Run again for better accuracy.",
                       saved, config.MIN_IMAGES_PER_PERSON)
        return False

    logger.info("✅ Collection complete for %s", student_id)
    return True


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎓 FACE ATTENDANCE — DATA COLLECTION")
    print("─" * 40)

    sid  = input("Student ID   (e.g. CS2021001) : ").strip().upper()
    name = input("Full Name                     : ").strip().title()
    mail = input("Email        (Enter to skip)  : ").strip() or None
    dept = input("Department   (Enter to skip)  : ").strip() or None

    if not sid or not name:
        print("❌ Student ID and Name are required.")
        raise SystemExit(1)

    ok = collect_images(sid, name, email=mail,
                        department=dept, register_to_db=True)
    if ok:
        print("✅ Done! Next run: python train_model.py")
    else:
        print("⚠️  Too few images captured. Run again.")