# attendance.py
import csv
import io
from datetime import date, datetime, timedelta

import config
import database
from utils import setup_logger, send_absent_email

logger = setup_logger(__name__)


def get_daily_report(target_date: date = None,
                     subject: str = None) -> dict:
    target_date = target_date or date.today()
    subject     = subject or config.DEFAULT_SUBJECT
    present     = database.get_attendance_by_date(
        target_date, subject)
    today       = date.today()

    if target_date > today:
        date_state = "future"
        absent     = []
    else:
        date_state = "today" if target_date == today \
                     else "past"
        absent     = database.get_absent_students(
            target_date, subject)

    total = len(database.get_all_students())
    return {
        "date"          : target_date.strftime("%Y-%m-%d"),
        "date_display"  : target_date.strftime("%d %B %Y"),
        "date_state"    : date_state,
        "subject"       : subject,
        "present"       : present,
        "absent"        : absent,
        "total_students": total,
        "present_count" : len(present),
        "absent_count"  : len(absent)
                          if date_state != "future" else 0,
        "attendance_pct": round(
            len(present) / total * 100, 1)
                          if total else 0,
    }


def get_student_daily_report(student_id: str,
                              target_date: date = None,
                              subject: str = None) -> dict:
    """Report for a single student — used by student role."""
    target_date = target_date or date.today()
    subject     = subject or config.DEFAULT_SUBJECT
    records     = database.get_attendance_by_date(
        target_date, subject)
    my_record   = next(
        (r for r in records
         if r["student_id"] == student_id), None)
    today       = date.today()
    date_state  = ("future" if target_date > today
                   else "today" if target_date == today
                   else "past")
    return {
        "date"        : target_date.strftime("%Y-%m-%d"),
        "date_display": target_date.strftime("%d %B %Y"),
        "date_state"  : date_state,
        "subject"     : subject,
        "my_record"   : my_record,
        "marked"      : my_record is not None,
    }


def get_weekly_summary(subject: str = None) -> list:
    subject = subject or config.DEFAULT_SUBJECT
    total   = len(database.get_all_students())
    summary = []
    for i in range(6, -1, -1):
        d       = date.today() - timedelta(days=i)
        present = database.get_attendance_by_date(d, subject)
        summary.append({
            "date"   : d.strftime("%Y-%m-%d"),
            "day"    : d.strftime("%a"),
            "present": len(present),
            "total"  : total,
            "pct"    : round(
                len(present) / total * 100, 1)
                       if total else 0,
        })
    return summary


def get_student_history(student_id: str,
                        subject: str = None) -> dict:
    student = database.get_student_by_id(student_id)
    if not student:
        return {"error": f"{student_id} not found"}
    records    = database.get_attendance_by_student(
        student_id, subject)
    reg_date   = student.get("registered_at", "")[:10]
    try:
        total_days = (date.today() -
                      date.fromisoformat(reg_date)).days + 1
    except Exception:
        total_days = 1
    return {
        "student"       : student,
        "records"       : records,
        "days_present"  : len(records),
        "total_days"    : total_days,
        "attendance_pct": round(
            len(records) / total_days * 100, 1)
                          if total_days else 0,
    }


def get_dashboard_stats(subject: str = None) -> dict:
    subject     = subject or config.DEFAULT_SUBJECT
    today       = date.today()
    report      = get_daily_report(today, subject)
    weekly      = get_weekly_summary(subject)
    recent_logs = database.get_recent_logs(limit=10)
    return {
        "today"         : report,
        "weekly"        : weekly,
        "recent_logs"   : recent_logs,
        "total_students": report["total_students"],
        "subject"       : subject,
    }


def export_csv(target_date: date = None,
               subject: str = None) -> str:
    target_date = target_date or date.today()
    subject     = subject or config.DEFAULT_SUBJECT
    report      = get_daily_report(target_date, subject)
    output      = io.StringIO()
    writer      = csv.writer(output)
    writer.writerow([
        "Attendance Report", report["date_display"],
        f"Subject: {subject}"])
    writer.writerow([])
    writer.writerow([
        "#", "Student ID", "Name",
        "Time In", "Confidence", "Status"])
    for i, r in enumerate(report["present"], 1):
        writer.writerow([
            i, r["student_id"], r["name"],
            r["time_in"],
            f"{r['confidence']:.1%}"
            if r["confidence"] else "—",
            r["status"]])
    writer.writerow([])
    writer.writerow(["ABSENT"])
    for i, s in enumerate(report["absent"], 1):
        writer.writerow([
            i, s["student_id"], s["name"],
            s.get("department", "—")])
    writer.writerow([])
    writer.writerow(
        ["Total", report["total_students"]])
    writer.writerow(
        ["Present", report["present_count"]])
    writer.writerow(
        ["Rate", f"{report['attendance_pct']}%"])
    return output.getvalue()


def send_daily_report(subject: str = None) -> bool:
    absent = database.get_absent_students(
        date.today(), subject)
    return send_absent_email(absent)