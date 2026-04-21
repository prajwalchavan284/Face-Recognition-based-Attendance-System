import sys
print(f"Python: {sys.version}")
packages = [
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("mediapipe", "MediaPipe"),
    ("deepface", "DeepFace"),
    ("flask", "Flask"),
    ("sqlalchemy", "SQLAlchemy"),
    ("dotenv", "python-dotenv"),
    ("PIL", "Pillow"),
    ("pandas", "Pandas"),
]
all_ok = True
for module, name in packages:
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError as e:
        print(f"  ❌ {name} — {e}")
        all_ok = False
print("\n✅ All good! Ready for Step 3." if all_ok else "\n❌ Fix errors before proceeding.")