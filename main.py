import os
from datetime import datetime, timezone, date, time
from decimal import Decimal
from typing import Optional, List

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, ForeignKey, DateTime, Boolean, Numeric, LargeBinary, Date, Time, func, select, \
    distinct
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

app = Flask(__name__)

load_dotenv()

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DB_URI")
app.secret_key = os.environ.get("SECRET_KEY")

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

class Student(db.Model):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    registration_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email:Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    department: Mapped[Optional[str]] = mapped_column(String(100))

    # relationships
    fingerprints: Mapped[List["FingerprintTemplate"]] = relationship(back_populates="student",
                                                      lazy="select", cascade="all, delete-orphan")
    registrations: Mapped[List["ExamRegistration"]] = relationship(back_populates="student", lazy="select")

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def has_fingerprint(self):
        return len(self.fingerprints) > 0

class FingerprintTemplate(db.Model):
    __tablename__ = "fingerprint_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(Integer, ForeignKey("students.id"), nullable=False)
    template_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    template_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    enrollment_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda : datetime.now(timezone.utc))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    quality_score: Mapped[Decimal] = mapped_column(Numeric(5, 2))

    # Relationships
    student: Mapped["Student"] = relationship(back_populates="fingerprints")
    verification_attempts: Mapped[List["VerificationAttempt"]] = relationship(
        back_populates="matched_template"
    )

class Exam(db.Model):
    __tablename__ = "exams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_code: Mapped[str] = mapped_column(String(20), nullable=False)
    exam_title: Mapped[str] = mapped_column(String(255), nullable=False)
    exam_date: Mapped[date] = mapped_column(Date, nullable=False)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    venue: Mapped[Optional[str]] = mapped_column(String(255))
    duration_minutes: Mapped[int] = mapped_column(Integer, default=180)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled, active, completed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    registrations: Mapped[List["ExamRegistration"]] = relationship(back_populates="exam", lazy="select")
    verifications: Mapped[List["VerificationAttempt"]] = relationship(back_populates="exam", lazy="select")


class ExamRegistration(db.Model):
    __tablename__ = "exam_registrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    registration_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    seat_number: Mapped[str | None] = mapped_column(String(20))

    # Relationships
    student: Mapped["Student"] = relationship(back_populates="registrations")
    exam: Mapped[Exam] = relationship(back_populates="registrations")


class VerificationAttempt(db.Model):
    __tablename__ = "verification_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    student_id: Mapped[int | None] = mapped_column(ForeignKey("students.id"), nullable=True)  # Null if failed
    verification_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    verification_status: Mapped[str] = mapped_column(String(20), nullable=False)  # success, failed
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    template_matched: Mapped[int | None] = mapped_column(ForeignKey("fingerprint_templates.id"))

    # Relationships
    exam: Mapped["Exam"] = relationship(back_populates="verifications")
    student: Mapped[Optional["Student"]] = relationship()  # No back_populates since it's optional
    matched_template: Mapped[Optional["FingerprintTemplate"]] = relationship(
        back_populates="verification_attempts")

with app.app_context():
    db.create_all()


@app.route("/")
def index():
    #Main Dashboard
    stats = {
        'total_students': db.session.execute(
            select(func.count()).select_from(Student)
        ).scalar(),

        'enrolled_fingerprints': db.session.execute(
            select(func.count(distinct(FingerprintTemplate.student_id)))
        ).scalar(),

        'total_exams': db.session.execute(
            select(func.count()).select_from(Exam)
        ).scalar(),

        'active_exams': db.session.execute(
            select(func.count()).select_from(Exam).where(Exam.status == 'active')
        ).scalar()
    }

    # Recent verifications query
    stmt = select(
        VerificationAttempt,
        Student.registration_number,
        Student.first_name,
        Student.last_name,
        Exam.exam_code
    ).outerjoin(Student).join(Exam).order_by(
        VerificationAttempt.verification_timestamp.desc()
    ).limit(5)

    recent_verifications = db.session.execute(stmt).all()

    return render_template("index.html", stats=stats, recent_verifications=recent_verifications)


@app.route("/enrollment")
def enrollment():
    return render_template("enrollment.html")


@app.route("/api/lookup-student", methods=["POST"])
def lookup_student():
    # Look up student by registration number
    reg_number = request.json.get("registration_number")

    if not reg_number:
        return jsonify({
            "error": "Registration number required"
        }), 400

    # student = Student.query.filter_by(registration_number=reg_number).first()
    student = db.session.execute(select(Student).where(Student.registration_number == reg_number)).scalar()

    if not student:
        return jsonify({
            "error": "Student not found"
        }), 404

    return jsonify({
        "id": student.id,
        "registration_number": student.registration_number,
        "first_name": student.first_name,
        "last_name": student.last_name,
        "department": student.department or 'N/A',
        "has_fingerprint": student.has_fingerprint,
        "fingerprint_count": len(student.fingerprints)
    })


@app.route("/verification")
def verification():
    # Live verification page
    active_exams = db.session.execute(
        select(Exam).where(
            Exam.status.in_(["scheduled", "active"])
        ).order_by(Exam.exam_date, Exam.start_time)
    ).scalars().all()

    return render_template("verification.html", active_exams=active_exams)


@app.route("/exams")
def exam_management():
    exams = Exam.query.order_by(Exam.exam_date.desc()).all()

    # Add registration counts
    exam_data = []
    for exam in exams:
        registered_count = ExamRegistration.query.filter_by(exam_id=exam.id).count()
        verified_count = VerificationAttempt.query.filter_by(
            exam_id=exam.id, verification_status='success'
        ).count()

        exam_data.append({
            'exam': exam,
            'registered_count': registered_count,
            'verified_count': verified_count
        })

    return render_template('exam_management.html', exam_data=exam_data)



@app.route('/api/create-exam', methods=['POST'])
def create_exam():
    """Create new exam"""
    try:
        data = request.json

        exam = Exam(
            exam_code=data['exam_code'],
            exam_title=data['exam_title'],
            exam_date=datetime.strptime(data['exam_date'], '%Y-%m-%d').date(),
            start_time=datetime.strptime(data['start_time'], '%H:%M').time(),
            venue=data.get('venue', ''),
            duration_minutes=int(data.get('duration_minutes', 180))
        )

        db.session.add(exam)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Exam created successfully',
            'exam_id': exam.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/update-exam-status/<int:exam_id>', methods=['POST'])
def update_exam_status(exam_id):
    """Update exam status"""
    try:
        new_status = request.json.get('status')

        exam = Exam.query.get(exam_id)
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404

        exam.status = new_status
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Exam status updated to {new_status}'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/register-student-for-exam', methods=['POST'])
def register_student_for_exam():
    """Register student for an exam"""
    try:
        data = request.json
        student_id = data.get('student_id')
        exam_id = data.get('exam_id')

        # Check if already registered
        existing = ExamRegistration.query.filter_by(
            student_id=student_id, exam_id=exam_id
        ).first()

        if existing:
            return jsonify({'error': 'Student already registered for this exam'}), 400

        registration = ExamRegistration(
            student_id=student_id,
            exam_id=exam_id,
            seat_number=data.get('seat_number')
        )

        db.session.add(registration)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Student registered successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)