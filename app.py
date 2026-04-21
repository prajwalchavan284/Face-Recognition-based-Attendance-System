import os
import io
import base64
from datetime import date, datetime, timedelta
from functools import wraps
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, Response,
    session as flask_session, abort
)
import config
import database
import attendance as att
from utils import setup_logger
import recognition as rec_engine
logger = setup_logger(__name__)
app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY
with app.app_context():
    database.init_db()
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in flask_session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in flask_session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        if flask_session["user"]["role"] != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated
def current_user():
    return flask_session.get("user")
def is_admin():
    u = current_user()
    return u and u.get("role") == "admin"
@app.context_processor
def inject_globals():
    return {
        "today"       : date.today().strftime("%d %B %Y"),
        "today_iso"   : date.today().isoformat(),
        "app_name"    : "Face Attendance System",
        "current_user": current_user(),
        "is_admin"    : is_admin(),
        "subjects"    : config.SUBJECTS,
        "rec_state"   : rec_engine.get_state(),
    }
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in flask_session:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user     = database.authenticate(username, password)
        if user:
            flask_session["user"]    = user
            flask_session.permanent  = True
            app.permanent_session_lifetime = timedelta(
                minutes=config.SESSION_LIFETIME_MINUTES)
            logger.info("Login: %s (%s)",
                        username, user["role"])
            flash(f"Welcome, {username}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")
@app.route("/logout")
def logout():
    flask_session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))
@app.route("/")
@login_required
def index():
    stats = att.get_dashboard_stats()
    rec   = rec_engine.get_state()
    return render_template("index.html",
                           stats=stats, rec=rec)
@app.route("/recognition/start", methods=["POST"])
@login_required
def start_recognition():
    subject = request.form.get(
        "subject", config.DEFAULT_SUBJECT)
    rec_engine.start_recognition_thread(subject)
    flash(f"✅ Recognition started for: {subject}", "success")
    return redirect(url_for("index"))
@app.route("/recognition/stop", methods=["POST"])
@login_required
def stop_recognition():
    rec_engine.stop_recognition_thread()
    flash("Recognition stopped.", "info")
    return redirect(url_for("index"))
@app.route("/students")
@admin_required
def students():
    return render_template(
        "students.html",
        students=database.get_all_students(),
        users=database.get_all_users())
@app.route("/students/add", methods=["GET", "POST"])
@admin_required
def add_student():
    if request.method == "POST":
        sid  = request.form.get(
            "student_id", "").strip().upper()
        name = request.form.get(
            "name", "").strip().title()
        mail = request.form.get(
            "email", "").strip() or None
        dept = request.form.get(
            "department", "").strip() or None
        uname = request.form.get(
            "username", "").strip() or None
        pwd   = request.form.get(
            "password", "").strip() or None
        if not sid or not name:
            flash("Student ID and Name required.", "danger")
            return redirect(url_for("add_student"))
        try:
            database.add_student(
                student_id=sid, name=name,
                email=mail, department=dept)
            if uname and pwd:
                database.create_user(
                    username=uname,
                    password=pwd,
                    role="student",
                    student_id=sid)
            flash(f"✅ {name} added! "
                  f"Now capture their face images.",
                  "success")
            return redirect(url_for(
                "capture_images",
                student_id=sid))
        except ValueError as e:
            flash(str(e), "danger")
    return render_template("add_student.html")
@app.route("/students/edit/<student_id>",
           methods=["GET", "POST"])
@admin_required
def edit_student(student_id):
    student = database.get_student_by_id(student_id)
    if not student:
        abort(404)
    if request.method == "POST":
        name = request.form.get(
            "name", "").strip().title()
        mail = request.form.get(
            "email", "").strip() or None
        dept = request.form.get(
            "department", "").strip() or None
        try:
            database.update_student(
                student_id, name, mail, dept)
            flash(f"✅ {name} updated.", "success")
            return redirect(url_for("students"))
        except ValueError as e:
            flash(str(e), "danger")
    return render_template(
        "edit_student.html", student=student)
@app.route("/students/delete/<student_id>",
           methods=["POST"])
@admin_required
def delete_student(student_id):
    s = database.get_student_by_id(student_id)
    if s:
        database.delete_student(student_id)
        flash(f"Student {student_id} deleted.", "info")
    return redirect(url_for("students"))
@app.route("/students/capture/<student_id>")
@admin_required
def capture_images(student_id):
    student = database.get_student_by_id(student_id)
    if not student:
        abort(404)
    return render_template(
        "capture.html", student=student)
@app.route("/api/capture/start/<student_id>",
           methods=["POST"])
@admin_required
def api_capture_start(student_id):
    import multiprocessing
    from data_collection import collect_images
    student = database.get_student_by_id(student_id)
    if not student:
        return jsonify({"error": "Student not found"}), 404
    p = multiprocessing.Process(
        target=collect_images,
        kwargs={
            "student_id": student_id,
            "name": student["name"],
            "register_to_db": False
        },
        daemon=True
    )
    p.start()
    return jsonify({
        "status": "started",
        "message": (f"Capturing images for "
                    f"{student['name']}. "
                    f"Watch the webcam window.")
    })
@app.route("/api/train", methods=["POST"])
@admin_required
def api_train():
    import threading
    from train_model import train_model
    def _train():
        train_model(force_retrain=True)
    threading.Thread(target=_train, daemon=True).start()
    return jsonify({
        "status": "started",
        "message": "Training started in background."
    })
@app.route("/attendance")
@app.route("/attendance/<string:date_str>")
@login_required
def attendance_view(date_str=None):
    try:
        target = (date.fromisoformat(date_str)
                  if date_str else date.today())
    except ValueError:
        target = date.today()
    subject = request.args.get(
        "subject", config.DEFAULT_SUBJECT)
    today  = date.today()
    if target > today:
        date_state = "future"
    elif target == today:
        date_state = "today"
    else:
        date_state = "past"
    user = current_user()
    if user["role"] == "student":
        sid = user.get("student_id")
        if not sid:
            flash("No student linked to your account.",
                  "warning")
            return redirect(url_for("index"))
        report = att.get_student_daily_report(
            sid, target, subject)
        return render_template(
            "my_attendance.html",
            report=report,
            date_state=date_state,
            target_date=target.isoformat(),
            subject=subject)
    report = att.get_daily_report(target, subject)
    return render_template(
        "attendance.html",
        report=report,
        date_state=date_state,
        target_date=target.isoformat(),
        subject=subject)
@app.route("/student/<student_id>")
@login_required
def student_detail(student_id):
    user = current_user()
    if (user["role"] == "student" and
            user.get("student_id") != student_id):
        flash("Access denied.", "danger")
        return redirect(url_for("index"))
    history = att.get_student_history(student_id)
    if "error" in history:
        abort(404)
    return render_template(
        "student_detail.html", history=history)
@app.route("/profile")
@login_required
def profile():
    user = current_user()
    if user["role"] == "admin":
        return redirect(url_for("index"))
    sid = user.get("student_id")
    if not sid:
        flash("No student record linked.", "warning")
        return redirect(url_for("index"))
    history = att.get_student_history(sid)
    return render_template(
        "student_detail.html", history=history)
@app.route("/export/csv")
@app.route("/export/csv/<string:date_str>")
@admin_required
def export_csv(date_str=None):
    try:
        target = (date.fromisoformat(date_str)
                  if date_str else date.today())
    except ValueError:
        target = date.today()
    subject  = request.args.get(
        "subject", config.DEFAULT_SUBJECT)
    csv_data = att.export_csv(target, subject)
    filename = (f"attendance_{target.isoformat()}"
                f"_{subject}.csv")
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename={filename}"})
@app.route("/logs")
@admin_required
def logs():
    return render_template(
        "logs.html",
        logs=database.get_recent_logs(100))
@app.route("/users/delete/<username>", methods=["POST"])
@admin_required
def delete_user(username):
    if username == config.ADMIN_USERNAME:
        flash("Cannot delete default admin.", "danger")
        return redirect(url_for("students"))
    database.delete_user(username)
    flash(f"User {username} deleted.", "info")
    return redirect(url_for("students"))
@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(att.get_dashboard_stats())
@app.route("/api/rec/state")
@login_required
def api_rec_state():
    return jsonify(rec_engine.get_state())
@app.route("/api/attendance")
@login_required
def api_attendance():
    date_str = request.args.get("date")
    subject  = request.args.get(
        "subject", config.DEFAULT_SUBJECT)
    try:
        target = (date.fromisoformat(date_str)
                  if date_str else date.today())
    except ValueError:
        target = date.today()
    return jsonify(
        database.get_attendance_by_date(target, subject))
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404
@app.errorhandler(500)
def server_error(e):
    logger.error("500: %s", e)
    return render_template("500.html"), 500
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  FACE ATTENDANCE SYSTEM")
    print(f"  http://127.0.0.1:{config.FLASK_PORT}")
    print(f"  Admin: {config.ADMIN_USERNAME} / "
          f"{config.ADMIN_PASSWORD}")
    print("  Ctrl+C to stop")
    print("=" * 55 + "\n")
    app.run(
        host="0.0.0.0",
        port=config.FLASK_PORT,
        debug=False,
        use_reloader=False)  