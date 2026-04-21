import os
import pickle
import time
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
from deepface import DeepFace
import config
import database
from utils import setup_logger, equalize_histogram
logger = setup_logger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def _load_image(path: Path) -> np.ndarray | None:
    try:
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            logger.warning("Cannot load: %s", path.name)
            return None
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            logger.warning("Too small (%dx%d): %s", w, h, path.name)
            return None
        return img
    except Exception as e:
        logger.error("Load error %s: %s", path.name, e)
        return None
def _get_embedding(image: np.ndarray) -> np.ndarray | None:
    try:
        result = DeepFace.represent(
            img_path          = image,
            model_name        = config.RECOGNITION_MODEL,
            detector_backend  = "skip",
            enforce_detection = False,
            align             = True,
        )
        if not result:
            return None
        emb = np.array(result[0]["embedding"], dtype=np.float32)
        if np.allclose(emb, 0, atol=1e-6):
            logger.warning("Zero embedding (no face): %s",
                           image_path.name)
            return None
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return None
        return emb / norm
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return None
def _representative(embeddings: list) -> np.ndarray:
    if len(embeddings) == 1:
        return embeddings[0]
    stack = np.stack(embeddings)          
    mean  = stack.mean(axis=0)
    mean /= (np.linalg.norm(mean) + 1e-6)
    sims    = stack @ mean                
    best    = int(np.argmax(sims))
    return embeddings[best]
def train_model(force_retrain: bool = False) -> bool:
    enc_file = config.ENCODINGS_FILE
    img_dir  = config.IMAGES_DIR
    existing = {}
    if enc_file.exists() and not force_retrain:
        try:
            with open(enc_file, "rb") as f:
                existing = pickle.load(f)
            logger.info("Loaded existing encodings: %d students",
                        len(existing.get("embeddings", {})))
        except Exception as e:
            logger.warning("Could not load existing: %s — "
                           "starting fresh.", e)
            existing = {}
    known_emb  = dict(existing.get("embeddings", {}))
    known_names = dict(existing.get("names",      {}))
    known_meta  = dict(existing.get("metadata",   {}))
    folders = sorted([
        d for d in img_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if not folders:
        print(f"❌ No student folders in {img_dir}")
        print("   Run: python data_collection.py first.")
        return False
    print("\n" + "=" * 55)
    print(f"  MODEL TRAINING  —  {config.RECOGNITION_MODEL}")
    print(f"  Detector: {config.DETECTOR_BACKEND}  |  "
          f"Threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  Students found: {len(folders)}")
    print("=" * 55)
    t0            = time.time()
    success_count = 0
    skip_count    = 0
    fail_count    = 0
    for folder in folders:
        sid = folder.name
        if sid in known_emb and not force_retrain:
            logger.info("Skip (already trained): %s", sid)
            skip_count += 1
            continue
        rec  = database.get_student_by_id(sid)
        name = rec["name"] if rec else sid
        imgs = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in IMG_EXTS
        ])
        if not imgs:
            logger.warning("No images in: %s", folder)
            fail_count += 1
            continue
        print(f"\n  ▶ {name} ({sid})  —  {len(imgs)} image(s)")
        embeddings   = []
        failed_count = 0
        for i, img_path in enumerate(imgs, 1):
            img = _load_image(img_path)
            if img is None:
                failed_count += 1
                continue
            img = equalize_histogram(img)
            img = cv2.resize(img, (112, 112),
                             interpolation=cv2.INTER_AREA)
            emb = _get_embedding(img)
            if emb is not None:
                embeddings.append(emb)
                print(f"    [{i:02d}/{len(imgs):02d}] ✅ {img_path.name}")
            else:
                failed_count += 1
                print(f"    [{i:02d}/{len(imgs):02d}] ❌ {img_path.name}"
                      f"  — no face detected")
        if not embeddings:
            logger.error("No valid embeddings for %s — skipping.",
                         name)
            print(f"  ❌ FAILED — {name}: no valid faces found")
            fail_count += 1
            continue
        if len(embeddings) < config.MIN_IMAGES_PER_PERSON:
            logger.warning("%s: only %d valid images (min %d). "
                           "Accuracy may suffer.",
                           name, len(embeddings),
                           config.MIN_IMAGES_PER_PERSON)
            print(f"  ⚠️  Only {len(embeddings)} valid — "
                  f"recommend {config.MIN_IMAGES_PER_PERSON}+")
        rep = _representative(embeddings)
        known_emb[sid]   = rep
        known_names[sid] = name
        known_meta[sid]  = {
            "image_count": len(imgs),
            "valid_count": len(embeddings),
            "fail_count" : failed_count,
            "trained_at" : time.strftime("%Y-%m-%d %H:%M:%S"),
            "model"      : config.RECOGNITION_MODEL,
        }
        print(f"  ✅ {name}  "
              f"{len(embeddings)}/{len(imgs)} embeddings → representative")
        success_count += 1
    if not known_emb:
        logger.error("No embeddings generated. Training failed.")
        return False
    payload = {
        "embeddings": known_emb,
        "names"     : known_names,
        "metadata"  : known_meta,
        "model"     : config.RECOGNITION_MODEL,
        "detector"  : config.DETECTOR_BACKEND,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version"   : 2,
    }
    try:
        with open(enc_file, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("✅ Saved encodings → %s", enc_file)
    except Exception as e:
        logger.error("Save failed: %s", e)
        return False
    elapsed = time.time() - t0
    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE  ({elapsed:.1f}s)")
    print(f"  ✅ Trained  : {success_count}")
    print(f"  ⏭  Skipped  : {skip_count}  (already trained)")
    print(f"  ❌ Failed   : {fail_count}")
    print(f"  👥 Total    : {len(known_emb)} student(s) known")
    print(f"  💾 Saved    : {enc_file}")
    print("=" * 55 + "\n")
    return True
def load_encodings() -> dict:
    if not config.ENCODINGS_FILE.exists():
        logger.error("Encodings not found: %s", config.ENCODINGS_FILE)
        logger.error("Run:  python train_model.py")
        return {}
    try:
        with open(config.ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        logger.info("✅ Encodings loaded: %d students | model: %s",
                    len(data.get("embeddings", {})),
                    data.get("model", "?"))
        return data
    except Exception as e:
        logger.error("Load encodings failed: %s", e)
        return {}
def get_encoding_stats() -> dict:
    data = load_encodings()
    if not data:
        return {"status": "not_found"}
    return {
        "status"         : "ok",
        "total_students" : len(data.get("embeddings", {})),
        "model"          : data.get("model"),
        "trained_at"     : data.get("trained_at"),
        "students"       : [
            {
                "id"    : sid,
                "name"  : data["names"].get(sid, sid),
                "images": data["metadata"].get(
                    sid, {}).get("valid_count", "?"),
            }
            for sid in data.get("embeddings", {})
        ],
    }
if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    if force:
        print("⚠️  --force: regenerating ALL embeddings\n")
    ok = train_model(force_retrain=force)
    if ok:
        stats = get_encoding_stats()
        print("📊 Encoding Stats:")
        print(f"   Students : {stats['total_students']}")
        print(f"   Model    : {stats['model']}")
        print(f"   Trained  : {stats['trained_at']}")
        print("\n✅ Ready — run: python recognition.py")
    else:
        print("❌ Training failed. Check output above.")
        sys.exit(1)