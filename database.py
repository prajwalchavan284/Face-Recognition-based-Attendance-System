import logging
from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, String,
    DateTime, Date, Float, Boolean,
    UniqueConstraint, text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from werkzeug.security import generate_password_hash, check_password_hash
import config
logger = logging.getLogger(__name__)
DATABASE_URL = f"sqlite:///{config.DATABASE_FILE}"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
SessionFactory = sessionmaker(bind=engine,
                              autocommit=False, autoflush=False)
Session = scoped_session(SessionFactory)
Base   = declarative_base()
class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(50), unique=True, nullable=False)
    password   = Column(String(255), nullable=False)
    role       = Column(String(10), nullable=False,
                        default="student")              
    student_id = Column(String(50), nullable=True)     
    created_at = Column(DateTime, default=datetime.now)
    def to_dict(self):
        return {
            "id": self.id, "username": self.username,
            "role": self.role, "student_id": self.student_id
        }
class Student(Base):
    __tablename__ = "students"
    id           = Column(Integer, primary_key=True,
                          autoincrement=True)
    student_id   = Column(String(50), unique=True, nullable=False)
    name         = Column(String(100), nullable=False)
    email        = Column(String(150), nullable=True)
    department   = Column(String(100), nullable=True)
    image_folder = Column(String(255), nullable=True)
    is_active    = Column(Boolean, default=True)
    registered_at = Column(DateTime, default=datetime.now)
    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "name": self.name,
            "email": self.email,
            "department": self.department,
            "is_active": self.is_active,
            "registered_at": (self.registered_at
                               .strftime("%Y-%m-%d %H:%M:%S")
                               if self.registered_at else None)
        }
class AttendanceRecord(Base):
    __tablename__ = "attendance"
    id            = Column(Integer, primary_key=True,
                           autoincrement=True)
    student_id    = Column(String(50), nullable=False)
    name          = Column(String(100), nullable=False)
    subject       = Column(String(100),
                           nullable=False, default="General")
    date          = Column(Date, nullable=False,
                           default=date.today)
    time_in       = Column(DateTime, nullable=False,
                           default=datetime.now)
    confidence    = Column(Float, nullable=True)
    status        = Column(String(20), default="Present")
    liveness_pass = Column(Boolean, default=True)
    __table_args__ = (
        UniqueConstraint("student_id", "date", "subject",
                         name="uq_student_date_subject"),
    )
    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "name": self.name,
            "subject": self.subject,
            "date": (self.date.strftime("%Y-%m-%d")
                     if self.date else None),
            "time_in": (self.time_in.strftime("%H:%M:%S")
                        if self.time_in else None),
            "confidence": (round(self.confidence, 4)
                           if self.confidence else None),
            "status": self.status,
            "liveness_pass": self.liveness_pass
        }
class SystemLog(Base):
    __tablename__ = "system_logs"
    id         = Column(Integer, primary_key=True,
                        autoincrement=True)
    event_type = Column(String(50))
    student_id = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    timestamp  = Column(DateTime, default=datetime.now)
    notes      = Column(String(255), nullable=True)
    def to_dict(self):
        return {
            "id": self.id,
            "event_type": self.event_type,
            "student_id": self.student_id,
            "confidence": self.confidence,
            "timestamp": self.timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"),
            "notes": self.notes
        }
class SubjectEnrollment(Base):
    __tablename__ = "subject_enrollments"
    id         = Column(Integer, primary_key=True,
                        autoincrement=True)
    student_id = Column(String(50), nullable=False)
    subject    = Column(String(100), nullable=False)
    enrolled_at = Column(DateTime, default=datetime.now)
    __table_args__ = (
        UniqueConstraint("student_id", "subject",
                         name="uq_student_subject"),
    )
    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "subject": self.subject,
            "enrolled_at": (self.enrolled_at
                            .strftime("%Y-%m-%d %H:%M:%S")
                            if self.enrolled_at else None)
        }
def init_db():
    Base.metadata.create_all(bind=engine)
    _seed_admin()
    logger.info("✅ Database ready: %s", config.DATABASE_FILE)
def _seed_admin():
    session = Session()
    try:
        exists = session.query(User).filter_by(
            role="admin").first()
        if not exists:
            session.add(User(
                username=config.ADMIN_USERNAME,
                password=generate_password_hash(config.ADMIN_PASSWORD),
                role="admin"
            ))
            session.commit()
            logger.info("Default admin created: %s",
                        config.ADMIN_USERNAME)
    finally:
        session.close()
def authenticate(username: str,
                 password: str) -> dict | None:
    session = Session()
    try:
        user = session.query(User).filter_by(
            username=username
        ).first()
        if user and check_password_hash(user.password, password):
            return user.to_dict()
        return None
    finally:
        session.close()
def create_user(username: str, password: str,
                role: str, student_id: str = None) -> dict:
    session = Session()
    try:
        if session.query(User).filter_by(
                username=username).first():
            raise ValueError(f"Username '{username}' taken.")
        hashed_pw = generate_password_hash(password)
        u = User(username=username, password=hashed_pw,
                 role=role, student_id=student_id)
        session.add(u)
        session.commit()
        session.refresh(u)
        return u.to_dict()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
def get_all_users() -> list:
    session = Session()
    try:
        return [u.to_dict() for u in session.query(User).all()]
    finally:
        session.close()
def delete_user(username: str):
    session = Session()
    try:
        u = session.query(User).filter_by(
            username=username).first()
        if u:
            session.delete(u)
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
def add_student(student_id: str, name: str,
                email: str = None, department: str = None,
                image_folder: str = None) -> dict:
    session = Session()
    try:
        if session.query(Student).filter_by(
                student_id=student_id).first():
            raise ValueError(
                f"Student ID '{student_id}' already exists.")
        s = Student(student_id=student_id, name=name,
                    email=email, department=department,
                    image_folder=image_folder)
        session.add(s)
        session.commit()
        session.refresh(s)
        return s.to_dict()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
def update_student(student_id: str, name: str = None,
                   email: str = None,
                   department: str = None) -> dict:
    session = Session()
    try:
        s = session.query(Student).filter_by(
            student_id=student_id).first()
        if not s:
            raise ValueError(f"Student '{student_id}' not found.")
        if name:
            s.name = name
        if email is not None:
            s.email = email
        if department is not None:
            s.department = department
        session.commit()
        session.refresh(s)
        return s.to_dict()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
def delete_student(student_id: str):
    session = Session()
    try:
        s = session.query(Student).filter_by(
            student_id=student_id).first()
        if s:
            session.delete(s)
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
def get_all_students() -> list:
    session = Session()
    try:
        return [s.to_dict() for s in
                session.query(Student)
                .filter_by(is_active=True).all()]
    finally:
        session.close()
def get_student_by_id(student_id: str) -> dict | None:
    session = Session()
    try:
        s = session.query(Student).filter_by(
            student_id=student_id).first()
        return s.to_dict() if s else None
    finally:
        session.close()
def student_exists(student_id: str) -> bool:
    session = Session()
    try:
        return session.query(Student).filter_by(
            student_id=student_id).count() > 0
    finally:
        session.close()
def mark_attendance(student_id: str, name: str,
                    confidence: float,
                    liveness_pass: bool = True,
                    status: str = "Present",
                    subject: str = None) -> dict:
    subject = subject or config.DEFAULT_SUBJECT
    session = Session()
    try:
        today   = date.today()
        existing = session.query(AttendanceRecord).filter_by(
            student_id=student_id,
            date=today,
            subject=subject
        ).first()
        if existing:
            return {
                "success": False,
                "message": (f"{name} already marked for "
                            f"{subject} today at "
                            f"{existing.time_in.strftime('%H:%M:%S')}"),
                "record": existing.to_dict()
            }
        r = AttendanceRecord(
            student_id=student_id, name=name,
            subject=subject, date=today,
            time_in=datetime.now(),
            confidence=confidence,
            status=status, liveness_pass=liveness_pass
        )
        session.add(r)
        session.commit()
        session.refresh(r)
        logger.info("✅ %s marked for %s (%.1f%%)",
                    name, subject, confidence * 100)
        return {
            "success": True,
            "message": f"✅ {name} marked Present for {subject}",
            "record": r.to_dict()
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
def get_attendance_by_date(target_date: date = None,
                           subject: str = None) -> list:
    session = Session()
    try:
        target_date = target_date or date.today()
        q = session.query(AttendanceRecord).filter_by(
            date=target_date)
        if subject:
            q = q.filter_by(subject=subject)
        return [r.to_dict() for r in q.all()]
    finally:
        session.close()
def get_attendance_by_student(student_id: str,
                               subject: str = None) -> list:
    session = Session()
    try:
        q = session.query(AttendanceRecord).filter_by(
            student_id=student_id)
        if subject:
            q = q.filter_by(subject=subject)
        return [r.to_dict() for r in
                q.order_by(AttendanceRecord.date.desc()).all()]
    finally:
        session.close()
def get_absent_students(target_date: date = None,
                        subject: str = None) -> list:
    session = Session()
    try:
        target_date = target_date or date.today()
        subject     = subject or config.DEFAULT_SUBJECT
        present_ids = {
            r.student_id for r in
            session.query(AttendanceRecord).filter_by(
                date=target_date, subject=subject).all()
        }
        return [s.to_dict() for s in
                session.query(Student)
                .filter_by(is_active=True).all()
                if s.student_id not in present_ids]
    finally:
        session.close()
def is_already_marked(student_id: str,
                       subject: str = None) -> bool:
    session = Session()
    try:
        subject = subject or config.DEFAULT_SUBJECT
        return session.query(AttendanceRecord).filter_by(
            student_id=student_id,
            date=date.today(),
            subject=subject
        ).count() > 0
    finally:
        session.close()
def add_log(event_type: str, student_id: str = None,
            confidence: float = None, notes: str = None):
    session = Session()
    try:
        session.add(SystemLog(
            event_type=event_type,
            student_id=student_id,
            confidence=confidence,
            notes=notes
        ))
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()
def get_recent_logs(limit: int = 50) -> list:
    session = Session()
    try:
        return [l.to_dict() for l in
                session.query(SystemLog)
                .order_by(SystemLog.timestamp.desc())
                .limit(limit).all()]
    finally:
        session.close()
def enroll_student_subject(student_id: str,
                           subject: str):
    session = Session()
    try:
        existing = session.query(SubjectEnrollment).filter_by(
            student_id=student_id, subject=subject).first()
        if not existing:
            session.add(SubjectEnrollment(
                student_id=student_id, subject=subject))
            session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()
def get_subjects_for_student(student_id: str) -> list:
    session = Session()
    try:
        enrollments = session.query(SubjectEnrollment).filter_by(
            student_id=student_id).order_by(
            SubjectEnrollment.subject).all()
        return [e.to_dict() for e in enrollments]
    finally:
        session.close()
def get_all_subjects() -> list:
    session = Session()
    try:
        results = session.query(
            SubjectEnrollment.subject
        ).distinct().order_by(
            SubjectEnrollment.subject).all()
        return [r[0] for r in results]
    finally:
        session.close()