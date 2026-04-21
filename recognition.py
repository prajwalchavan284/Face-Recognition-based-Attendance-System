import os
import time
import threading
from collections import deque, Counter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import config
import database
from train_model import load_encodings
from utils import (
    setup_logger, equalize_histogram,
    draw_face_box, draw_status_bar,
    draw_liveness_prompt, is_attendance_open,
    is_before_window, get_time_remaining,
    CooldownTracker,
)
logger = setup_logger(__name__)
_mp_det  = mp.solutions.face_detection
_mp_mesh = mp.solutions.face_mesh
_L_EAR = [362, 385, 387, 263, 373, 380]
_R_EAR = [33,  160, 158, 133, 153, 144]
_state_lock    = threading.Lock()
_recognition_state = {
    "running"   : False,
    "present"   : [],   
    "last_event": "",   
    "fps"       : 0.0,
    "faces"     : 0,
    "subject"   : config.DEFAULT_SUBJECT,
}
def get_state() -> dict:
    with _state_lock:
        return dict(_recognition_state)
def set_subject(subject: str):
    with _state_lock:
        _recognition_state["subject"] = subject
def _lbp_variance(gray: np.ndarray) -> float:
    h, w   = gray.shape
    lbp    = np.zeros_like(gray, dtype=np.uint8)
    radius = 1
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),
                   (1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(gray, dy, axis=0),
                          dx, axis=1)
        lbp    += (gray >= shifted).astype(np.uint8)
    return float(np.var(lbp))
def _is_real_face(face_crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    var  = _lbp_variance(gray)
    logger.debug("LBP variance: %.1f (threshold: %d)",
                 var, config.LBP_VARIANCE_THRESHOLD)
    return var >= config.LBP_VARIANCE_THRESHOLD
def _ear(landmarks, pts, w, h) -> float:
    def p(i):
        lm = landmarks[i]
        return np.array([lm.x * w, lm.y * h],
                        dtype=np.float32)
    p1,p2,p3,p4,p5,p6 = (p(i) for i in pts)
    return ((np.linalg.norm(p2-p6) +
             np.linalg.norm(p3-p5)) /
            (2.0 * np.linalg.norm(p1-p4) + 1e-6))
def _avg_ear(landmarks, w, h) -> float:
    return (_ear(landmarks, _L_EAR, w, h) +
            _ear(landmarks, _R_EAR, w, h)) / 2.0
def _cosine(a, b) -> float:
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) *
                  np.linalg.norm(b) + 1e-6))
def match_face(query: np.ndarray,
               known: dict,
               threshold: float = None) -> tuple:
    threshold = threshold or config.SIMILARITY_THRESHOLD
    if not known:
        return "Unknown", 0.0
    qn   = query / (np.linalg.norm(query) + 1e-6)
    best_id, best_score = "Unknown", -1.0
    for sid, emb in known.items():
        score = _cosine(qn, emb)
        if score > best_score:
            best_score = score
            best_id    = sid
    return ((best_id, best_score)
            if best_score >= threshold
            else ("Unknown", best_score))
def _live_embedding(frame: np.ndarray,
                    bbox: tuple) -> np.ndarray | None:
    h, w    = frame.shape[:2]
    x1,y1,x2,y2 = [int(v) for v in bbox]
    pad     = 25
    x1      = max(0,  x1 - pad)
    y1      = max(0,  y1 - pad)
    x2      = min(w,  x2 + pad)
    y2      = min(h,  y2 + pad)
    crop    = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop    = equalize_histogram(crop)
    crop    = cv2.resize(crop, (112, 112),
                         interpolation=cv2.INTER_AREA)
    try:
        res = DeepFace.represent(
            img_path=crop,
            model_name=config.RECOGNITION_MODEL,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        if not res:
            return None
        emb  = np.array(res[0]["embedding"],
                        dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-6)
    except Exception as e:
        logger.error("Live embedding: %s", e)
        return None
class _LivenessState:
    def __init__(self):
        self.blinks = 0
        self.consec = 0
        self.passed = False
    def update(self, ear: float) -> bool:
        if ear < config.BLINK_THRESHOLD:
            self.consec += 1
        else:
            if self.consec >= config.BLINK_CONSECUTIVE_FRAMES:
                self.blinks += 1
            self.consec = 0
        if self.blinks >= config.REQUIRED_BLINKS:
            self.passed = True
        return self.passed
class _VoteBuffer:
    def __init__(self):
        self.votes  = deque(
            maxlen=config.RECOGNITION_VOTE_FRAMES)
        self.scores = deque(
            maxlen=config.RECOGNITION_VOTE_FRAMES)
    def add(self, sid: str, score: float):
        self.votes.append(sid)
        self.scores.append(score)
    def confirmed(self) -> tuple[str, float] | None:
        if len(self.votes) < config.RECOGNITION_VOTE_FRAMES:
            return None
        counter = Counter(self.votes)
        top_sid, top_count = counter.most_common(1)[0]
        if (top_sid != "Unknown" and
                top_count >= config.RECOGNITION_VOTE_THRESH):
            avg_conf = float(np.mean([
                s for s, v in zip(self.scores, self.votes)
                if v == top_sid
            ]))
            return top_sid, avg_conf
        return None
def run_recognition(subject: str = None):
    data = load_encodings()
    if not data or not data.get("embeddings"):
        logger.error("No encodings. Run train_model.py first.")
        return
    known_emb   = data["embeddings"]
    known_names = data["names"]
    active_subj = subject or config.DEFAULT_SUBJECT
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        logger.error("Cannot open camera.")
        with _state_lock:
            _recognition_state["running"] = False
        return
    with _state_lock:
        _recognition_state["running"] = True
        _recognition_state["subject"] = active_subj
    cooldown   = CooldownTracker()
    liveness   = {}
    vote_bufs  = {}
    last_faces = []
    frame_n    = 0
    fps        = 0.0
    t_fps      = time.time()
    logger.info("Recognition started | subject: %s | "
                "students: %d", active_subj, len(known_emb))
    with _mp_det.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.70
    ) as detector,    _mp_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    ) as mesher:
        while True:
            with _state_lock:
                if not _recognition_state["running"]:
                    break
                active_subj = _recognition_state["subject"]
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1
            h, w     = frame.shape[:2]
            display  = frame.copy()
            if frame_n % 30 == 0:
                fps   = 30.0 / (time.time() - t_fps + 1e-6)
                t_fps = time.time()
                with _state_lock:
                    _recognition_state["fps"] = round(fps, 1)
            if is_before_window():
                draw_status_bar(
                    display,
                    f"⏳ Opens at {config.ATTENDANCE_START}",
                    (0, 180, 255))
                cv2.imshow("Face Attendance", display)
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    break
                continue
            if not is_attendance_open():
                draw_status_bar(
                    display,
                    "🔒 Attendance window closed",
                    (0, 0, 220))
                cv2.imshow("Face Attendance", display)
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
                continue
            if frame_n % config.FRAME_SKIP == 0:
                enhanced = equalize_histogram(frame)
                rgb      = cv2.cvtColor(enhanced,
                                        cv2.COLOR_BGR2RGB)
                det_res  = detector.process(rgb)
                mesh_res = mesher.process(rgb)
                current = []
                n_det   = (len(det_res.detections)
                           if det_res.detections else 0)
                with _state_lock:
                    _recognition_state["faces"] = n_det
                for i in range(n_det):
                    det  = det_res.detections[i]
                    bb   = det.location_data                              .relative_bounding_box
                    x1   = max(0, int(bb.xmin * w))
                    y1   = max(0, int(bb.ymin * h))
                    x2   = min(w, int(
                        (bb.xmin + bb.width)  * w))
                    y2   = min(h, int(
                        (bb.ymin + bb.height) * h))
                    bbox = (x1, y1, x2, y2)
                    key  = f"face_{i}"
                    live_ok = not config.LIVENESS_ENABLED
                    blinks  = 0
                    if (config.LIVENESS_ENABLED and
                            mesh_res.multi_face_landmarks and
                            i < len(
                                mesh_res.multi_face_landmarks)):
                        if key not in liveness:
                            liveness[key] = _LivenessState()
                        lms  = (mesh_res
                                .multi_face_landmarks[i]
                                .landmark)
                        ear  = _avg_ear(lms, w, h)
                        st   = liveness[key]
                        live_ok = st.update(ear)
                        blinks  = st.blinks
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        texture_ok = _is_real_face(crop)
                    else:
                        texture_ok = False
                    spoof_blocked = (
                        config.LIVENESS_ENABLED and
                        not texture_ok)
                    sid  = "Unknown"
                    conf = 0.0
                    if live_ok and not spoof_blocked:
                        emb = _live_embedding(frame, bbox)
                        if emb is not None:
                            raw_sid, raw_conf = match_face(
                                emb, known_emb)
                            if key not in vote_bufs:
                                vote_bufs[key] = _VoteBuffer()
                            vote_bufs[key].add(
                                raw_sid, raw_conf)
                            confirmed = (vote_bufs[key]
                                         .confirmed())
                            if confirmed:
                                sid, conf = confirmed
                            else:
                                sid  = "Confirming..."
                                conf = raw_conf
                    current.append({
                        "bbox"        : bbox,
                        "sid"         : sid,
                        "conf"        : conf,
                        "live"        : live_ok,
                        "blinks"      : blinks,
                        "spoof_block" : spoof_blocked,
                    })
                    if (sid not in ("Unknown", "Confirming...")
                            and live_ok
                            and not spoof_blocked
                            and cooldown.can_mark(sid)):
                        nm  = known_names.get(sid, sid)
                        res = database.mark_attendance(
                            student_id=sid,
                            name=nm,
                            confidence=conf,
                            liveness_pass=True,
                            subject=active_subj,
                        )
                        database.add_log(
                            event_type="RECOGNIZED",
                            student_id=sid,
                            confidence=conf,
                            notes=res["message"],
                        )
                        if res["success"]:
                            cooldown.mark(sid)
                            with _state_lock:
                                _recognition_state[
                                    "last_event"] = (
                                    res["message"])
                                _recognition_state[
                                    "present"].append({
                                    "name": nm,
                                    "sid" : sid,
                                    "conf": conf,
                                    "time": datetime.now()
                                           .strftime(
                                           "%H:%M:%S")
                                })
                            logger.info(
                                "✅ %s | %s | conf=%.1f%%",
                                nm, active_subj, conf*100)
                            print(f"  ✅ {res['message']} "
                                  f"[{conf:.1%}]")
                    elif spoof_blocked:
                        database.add_log(
                            event_type="LIVENESS_FAIL",
                            student_id=sid,
                            confidence=conf,
                            notes="LBP spoof detected",
                        )
                    elif sid == "Unknown":
                        database.add_log(
                            event_type="UNKNOWN",
                            confidence=conf,
                            notes=f"best={conf:.3f}",
                        )
                active_keys = {f"face_{i}"
                               for i in range(n_det)}
                for k in list(liveness):
                    if k not in active_keys:
                        del liveness[k]
                for k in list(vote_bufs):
                    if k not in active_keys:
                        del vote_bufs[k]
                last_faces = current
            for f in last_faces:
                x1,y1,x2,y2 = f["bbox"]
                sid   = f["sid"]
                conf  = f["conf"]
                live  = f["live"]
                blks  = f["blinks"]
                spoof = f["spoof_block"]
                nm    = (known_names.get(sid, sid)
                         if sid not in
                         ("Unknown","Confirming...")
                         else sid)
                if spoof:
                    color = (0, 0, 180)
                    nm    = "SPOOF DETECTED"
                elif sid not in ("Unknown","Confirming...")                        and live:
                    color = (0, 230, 0)
                elif sid == "Confirming...":
                    color = (0, 200, 255)
                elif not live:
                    color = (0, 230, 230)
                else:
                    color = (0, 0, 220)
                draw_face_box(display,
                              (x1,y1,x2,y2),
                              nm, conf, color)
                if not live and config.LIVENESS_ENABLED:
                    draw_liveness_prompt(
                        display, blks,
                        config.REQUIRED_BLINKS)
                if spoof:
                    cv2.putText(
                        display,
                        "Anti-spoof: LBP texture fail",
                        (x1, y1 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 0, 200), 2)
            draw_status_bar(
                display,
                f"🟢 {active_subj} | "
                f"{get_time_remaining()} | "
                f"{len(last_faces)} face(s) | "
                f"{fps:.0f} fps",
                (0, 180, 0))
            cv2.imshow("Face Attendance", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                with _state_lock:
                    _recognition_state["running"] = False
                break
    cap.release()
    cv2.destroyAllWindows()
    with _state_lock:
        _recognition_state["running"] = False
    logger.info("Recognition stopped.")
    from datetime import date as date_
    from utils import send_absent_email
    absent = database.get_absent_students(
        date_.today(), active_subj)
    send_absent_email(absent)
_rec_thread: threading.Thread | None = None
def start_recognition_thread(subject: str = None):
    global _rec_thread
    with _state_lock:
        if _recognition_state["running"]:
            logger.info("Recognition already running.")
            return
    _rec_thread = threading.Thread(
        target=run_recognition,
        args=(subject,),
        daemon=True,
        name="RecognitionThread"
    )
    _rec_thread.start()
    logger.info("Recognition thread started.")
def stop_recognition_thread():
    with _state_lock:
        _recognition_state["running"] = False
    if _rec_thread and _rec_thread.is_alive():
        _rec_thread.join(timeout=5)
    logger.info("Recognition thread stopped.")
if __name__ == "__main__":
    run_recognition()