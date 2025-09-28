import os
import hashlib
import base64
from datetime import datetime, timezone, date, time
from decimal import Decimal
from typing import Optional, List
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, url_for, flash
from flask_login import UserMixin, LoginManager, current_user, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, ForeignKey, DateTime, Boolean, Numeric, LargeBinary, Date, Time, func, select, \
    distinct
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from werkzeug.utils import redirect
from collections import defaultdict
from sqlalchemy import and_, or_
from fingerprint_matcher import FingerprintMatcher
from fingerprint_scanner import FingerprintScanner

app = Flask(__name__)

load_dotenv()

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DB_URI")
app.secret_key = os.environ.get("SECRET_KEY")

login_manager = LoginManager()
login_manager.init_app(app)

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


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)

class AttendanceRecord(db.Model):
    """Track student attendance for exams"""
    __tablename__ = "attendance_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False)
    verification_attempt_id: Mapped[int] = mapped_column(ForeignKey("verification_attempts.id"), nullable=False)
    check_in_time: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    seat_number: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="present")  # present, absent, late

    # Relationships
    exam: Mapped["Exam"] = relationship()
    student: Mapped["Student"] = relationship()
    verification_attempt: Mapped["VerificationAttempt"] = relationship()

class VerificationSession(db.Model):
    """Track verification sessions for exams"""
    __tablename__ = "verification_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    session_start: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    session_end: Mapped[datetime | None] = mapped_column(DateTime)
    operator_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)
    total_attempts: Mapped[int] = mapped_column(Integer, default=0)
    successful_verifications: Mapped[int] = mapped_column(Integer, default=0)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_attempts: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    exam: Mapped["Exam"] = relationship()
    operator: Mapped["User"] = relationship()

@login_manager.user_loader
def load_user(user_id):
    return db.get_or_404(User, user_id)


with app.app_context():
    db.create_all()

try:
    # Example for Digital Persona U.are.U SDK
    # You'll need to install the appropriate SDK for your scanner
    import digitalpersona as dp
    scanner = dp.UareUGlobal()

    # For ZKTeco devices
    # from zk import ZK
    # scanner = ZK('192.168.1.201', port=4370)

    # For camera-based fingerprint capture using OpenCV
    import cv2

    SCANNER_AVAILABLE = True

except ImportError as e:
    print(f"Scanner SDK not available: {e}")
    SCANNER_AVAILABLE = False

# Scanner Persona Configuration
SCANNER_PERSONA = {
    "name": "NEXUS-7",
    "version": "2.1.4",
    "responses": {
        "idle": "SYSTEM READY - AWAITING STUDENT DATA...",
        "student_found": "STUDENT VERIFIED - BIOMETRIC SCAN AUTHORIZED",
        "student_not_found": "ERROR: STUDENT RECORD NOT FOUND IN DATABASE",
        "already_enrolled": "WARNING: BIOMETRIC DATA ALREADY EXISTS FOR THIS STUDENT",
        "scanning_start": "INITIATING BIOMETRIC CAPTURE SEQUENCE...",
        "scanning_progress": "ANALYZING RIDGE PATTERNS AND MINUTIAE...",
        "scanning_complete": "BIOMETRIC TEMPLATE GENERATED SUCCESSFULLY",
        "enrollment_start": "ENCRYPTING AND STORING BIOMETRIC DATA...",
        "enrollment_success": "ENROLLMENT COMPLETE - BIOMETRIC DATA SECURED",
        "enrollment_error": "ENROLLMENT FAILED - PLEASE RETRY OPERATION",
        "quality_low": "SCAN QUALITY INSUFFICIENT - PLEASE RESCAN",
        "system_error": "SYSTEM ERROR - PLEASE CONTACT ADMINISTRATOR",
        "hardware_error": "SCANNER HARDWARE NOT RESPONDING - CHECK CONNECTION"
    }
}


# Initialize scanner
fingerprint_scanner = FingerprintScanner("opencv")  # Change to your scanner type

# Initialize matcher
fingerprint_matcher = FingerprintMatcher()

def generate_scanner_response(status, additional_data=None):
    """Generate standardized scanner persona responses"""
    response = {
        "scanner_name": SCANNER_PERSONA["name"],
        "scanner_version": SCANNER_PERSONA["version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "message": SCANNER_PERSONA["responses"].get(status, "UNKNOWN STATUS"),
        "data": additional_data or {}
    }
    return response


@app.route("/")
def index():
    return render_template("login.html")

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json

        username = data["username"]
        # password = data["password"]

        result = db.session.execute(db.select(User).where(User.email == username.lower()))
        user = result.scalars().first()
        if user:
            login_user(user)
            return jsonify({
                "success": True,
                "redirect_url": url_for("dashboard")  # dashboard route
            })
        else:
            return jsonify({"success": False, "error": "Invalid credentials"})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
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
    return render_template("dashboard.html",
                           stats=stats,
                           recent_verifications=recent_verifications,
                           current_user=current_user)


@app.route("/register-course", methods=["GET", "POST"])
def course_registration():
    available_courses = Exam.query.order_by(Exam.exam_date.desc()).all()
    registered_courses = ExamRegistration.query.filter_by(
        student_id=current_user.id
    ).join(Exam).order_by(ExamRegistration.registration_date.desc()).all()

    if request.method == "POST":
        selected_exam_codes = request.form.getlist("selected_exam_codes")

        # Check if any exams were selected
        if not selected_exam_codes:
            flash("Please select at least one exam to register for.", "warning")
            return render_template("course-registration.html", exams=available_courses)

        try:
            # Get exam IDs from the selected exam codes
            selected_exams = Exam.query.filter(Exam.exam_code.in_(selected_exam_codes)).all()
            exam_ids = [exam.id for exam in selected_exams]

            # Check for existing registrations to prevent duplicates
            existing_registrations = ExamRegistration.query.filter(
                ExamRegistration.student_id == current_user.id,
                ExamRegistration.exam_id.in_(exam_ids)
            ).all()

            existing_exam_ids = [reg.exam_id for reg in existing_registrations]
            new_exam_ids = [exam_id for exam_id in exam_ids if exam_id not in existing_exam_ids]

            if existing_registrations:
                existing_exam_codes = [exam.exam_code for exam in selected_exams if exam.id in existing_exam_ids]
                flash(f"You are already registered for: {', '.join(existing_exam_codes)}", "warning")

            if not new_exam_ids:
                flash("No new registrations to process.", "info")
                return render_template("course-registration.html", exams=available_courses)

            # Create new registrations
            successful_registrations = 0
            for exam in selected_exams:
                if exam.id in new_exam_ids:
                    # Create new ExamRegistration record
                    registration = ExamRegistration(
                        student_id=current_user.id,
                        exam_id=exam.id,
                        registration_date=datetime.now(timezone.utc),
                        seat_number=None  # Will be assigned later by admin
                    )
                    db.session.add(registration)
                    successful_registrations += 1

            # Commit all changes to database
            db.session.commit()

            flash(f"Successfully registered for {successful_registrations} exam(s)!", "success")
            print(f"Student {current_user.id} registered for {successful_registrations} exams")

            # Redirect to prevent form resubmission
            return redirect(url_for('course_registration'))

        except Exception as e:
            # Rollback in case of error
            db.session.rollback()
            flash(f"Registration failed: {str(e)}", "error")
            print(f"Registration error: {str(e)}")

    return render_template("course-registration.html", exams=available_courses, exam_data=registered_courses)

@app.route("/results")
def view_result():
    return render_template("result.html", current_user=current_user)

@app.route("/enrollment")
def enrollment():
    """Enhanced enrollment page with real scanner integration"""
    scanner_status = fingerprint_scanner.is_connected
    scanner_type = fingerprint_scanner.scanner_type

    return render_template("enrollment.html",
                           scanner_available=scanner_status,
                           scanner_type=scanner_type,
                           scanner_persona=SCANNER_PERSONA)


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


@app.route("/api/scanner-status")
def get_scanner_status():
    """Get current scanner system status"""
    try:
        scanner_data = {
            "operational": fingerprint_scanner.is_connected,
            "scanner_type": fingerprint_scanner.scanner_type,
            "temperature": "Normal",
            "last_calibration": "2024-01-15T10:30:00Z"
        }

        if fingerprint_scanner.is_connected:
            response = generate_scanner_response("idle", scanner_data)
        else:
            response = generate_scanner_response("hardware_error", scanner_data)

        return jsonify(response)

    except Exception as e:
        return jsonify(generate_scanner_response("system_error", {
            "error": str(e)
        })), 500


@app.route("/api/initiate-scan", methods=["POST"])
def initiate_fingerprint_scan():
    """Start fingerprint scanning process"""
    try:
        data = request.json
        student_id = data.get('student_id')

        if not student_id:
            return jsonify(generate_scanner_response("system_error", {
                "error": "Student ID required"
            })), 400

        # Verify student exists
        student = db.session.execute(
            select(Student).where(Student.id == student_id)
        ).scalar()

        if not student:
            return jsonify(generate_scanner_response("student_not_found"))

        # Check if already enrolled
        if student.has_fingerprint:
            return jsonify(generate_scanner_response("already_enrolled", {
                "student_name": student.full_name,
                "existing_templates": len(student.fingerprints)
            }))

        # Start scanning process
        return jsonify(generate_scanner_response("scanning_start", {
            "student_name": student.full_name,
            "student_reg": student.registration_number
        }))

    except Exception as e:
        return jsonify(generate_scanner_response("system_error", {
            "error": str(e)
        })), 500


@app.route("/api/capture-fingerprint", methods=["POST"])
def capture_fingerprint():
    """Capture fingerprint from scanner hardware"""
    try:
        data = request.json
        student_id = data.get('student_id')

        if not fingerprint_scanner.is_connected:
            return jsonify(generate_scanner_response("hardware_error")), 500

        # Capture fingerprint from hardware
        scan_result = fingerprint_scanner.capture_fingerprint()

        if scan_result['quality_score'] < 70:
            return jsonify(generate_scanner_response("quality_low", {
                "quality_score": scan_result['quality_score'],
                "retry_recommended": True
            }))

        # Return successful scan
        return jsonify(generate_scanner_response("scanning_complete", {
            "quality_score": scan_result['quality_score'],
            "minutiae_count": len(scan_result['template']),
            "scan_data": scan_result['image_data']  # Base64 encoded image
        }))

    except Exception as e:
        return jsonify(generate_scanner_response("system_error", {
            "error": str(e)
        })), 500


@app.route("/api/enroll-fingerprint", methods=["POST"])
def enroll_fingerprint():
    """Enroll captured fingerprint into database"""
    try:
        data = request.json
        student_id = data.get('student_id')
        scan_data = data.get('scan_data')  # Base64 encoded fingerprint data

        if not student_id or not scan_data:
            return jsonify(generate_scanner_response("enrollment_error", {
                "error": "Missing required data"
            })), 400

        # Get student
        student = db.session.execute(
            select(Student).where(Student.id == student_id)
        ).scalar()

        if not student:
            return jsonify(generate_scanner_response("student_not_found"))

        # Process fingerprint template
        template_data = base64.b64decode(scan_data)
        template_hash = hashlib.sha256(template_data).hexdigest()

        # Create fingerprint template record
        fingerprint_template = FingerprintTemplate(
            student_id=student_id,
            template_data=template_data,
            template_hash=template_hash,
            enrollment_date=datetime.now(timezone.utc),
            is_active=True,
            quality_score=Decimal(str(data.get('quality_score', 85.0)))
        )

        db.session.add(fingerprint_template)
        db.session.commit()

        return jsonify(generate_scanner_response("enrollment_success", {
            "student_name": student.full_name,
            "template_id": fingerprint_template.id,
            "quality_score": float(fingerprint_template.quality_score),
            "enrollment_time": fingerprint_template.enrollment_date.isoformat()
        }))

    except Exception as e:
        db.session.rollback()
        return jsonify(generate_scanner_response("enrollment_error", {
            "error": str(e)
        })), 500


@app.route("/api/start-verification-session", methods=["POST"])
def start_verification_session():
    """Start a new verification session for an exam"""
    try:
        data = request.json
        exam_id = data.get('exam_id')
        operator_id = current_user.id if current_user.is_authenticated else 1

        if not exam_id:
            return jsonify({"error": "Exam ID required"}), 400

        # Check if exam exists and is active
        exam = db.session.execute(
            select(Exam).where(Exam.id == exam_id)
        ).scalar()

        if not exam:
            return jsonify({"error": "Exam not found"}), 404

        if exam.status not in ['active', 'scheduled']:
            return jsonify({"error": "Exam is not available for verification"}), 400

        # End any existing active session for this exam
        existing_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar()

        if existing_session:
            existing_session.session_end = datetime.now(timezone.utc)
            existing_session.is_active = False

        # Create new session
        new_session = VerificationSession(
            exam_id=exam_id,
            operator_id=operator_id,
            session_start=datetime.now(timezone.utc),
            is_active=True
        )

        db.session.add(new_session)
        db.session.commit()

        return jsonify({
            "success": True,
            "session_id": new_session.id,
            "exam_title": exam.exam_title,
            "exam_code": exam.exam_code,
            "venue": exam.venue,
            "message": f"Verification session started for {exam.exam_code}"
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/verify-fingerprint", methods=["POST"])
def verify_fingerprint():
    """Enhanced fingerprint verification with duplicate detection"""
    try:
        data = request.json
        exam_id = data.get('exam_id')

        if not exam_id:
            return jsonify({"error": "Exam ID required"}), 400

        # Check if verification session is active
        active_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar()

        if not active_session:
            return jsonify({
                "success": False,
                "status": "error",
                "message": "No active verification session found"
            }), 400

        # Capture fingerprint from scanner
        if not fingerprint_scanner.is_connected:
            return jsonify({
                "success": False,
                "status": "error",
                "message": "Scanner hardware not available"
            }), 500

        # Capture current fingerprint
        try:
            current_scan = fingerprint_scanner.capture_fingerprint()
        except Exception as e:
            return jsonify({
                "success": False,
                "status": "error",
                "message": f"Fingerprint capture failed: {str(e)}"
            }), 500

        # Get all active fingerprint templates
        templates = db.session.execute(
            select(FingerprintTemplate).where(FingerprintTemplate.is_active == True)
        ).scalars().all()

        best_match = None
        best_score = 0
        match_details = None

        # Match against all templates
        for template in templates:
            try:
                # Decode stored template data
                if isinstance(template.template_data, bytes):
                    stored_template = template.template_data.decode('utf-8')
                else:
                    stored_template = template.template_data

                confidence_score, details = fingerprint_matcher.match_templates(
                    current_scan, stored_template
                )

                if confidence_score > best_score and confidence_score >= fingerprint_matcher.matching_threshold:
                    best_match = template
                    best_score = confidence_score
                    match_details = details

            except Exception as e:
                print(f"Error matching template {template.id}: {e}")
                continue

        # Update session statistics
        active_session.total_attempts += 1

        # Create verification attempt record
        verification = VerificationAttempt(
            exam_id=exam_id,
            student_id=best_match.student_id if best_match else None,
            verification_timestamp=datetime.now(timezone.utc),
            verification_status="success" if best_match else "failed",
            confidence_score=Decimal(str(best_score)) if best_match else None,
            template_matched=best_match.id if best_match else None
        )

        db.session.add(verification)
        db.session.flush()  # Get the ID

        if best_match:
            # Check for duplicate verification (student already verified for this exam)
            existing_attendance = db.session.execute(
                select(AttendanceRecord).where(
                    and_(AttendanceRecord.exam_id == exam_id,
                         AttendanceRecord.student_id == best_match.student_id)
                )
            ).scalar()

            if existing_attendance:
                # Duplicate verification
                active_session.duplicate_attempts += 1
                db.session.commit()

                return jsonify({
                    "success": False,
                    "status": "duplicate",
                    "message": f"Student {best_match.student.full_name} already verified for this exam",
                    "student": {
                        "id": best_match.student.id,
                        "name": best_match.student.full_name,
                        "registration_number": best_match.student.registration_number,
                        "department": best_match.student.department
                    },
                    "first_verification_time": existing_attendance.check_in_time.isoformat(),
                    "confidence_score": float(best_score)
                })

            # Check if student is registered for this exam
            registration = db.session.execute(
                select(ExamRegistration).where(
                    and_(ExamRegistration.exam_id == exam_id,
                         ExamRegistration.student_id == best_match.student_id)
                )
            ).scalar()

            if not registration:
                # Student not registered for this exam
                active_session.failed_attempts += 1
                verification.verification_status = "not_registered"
                db.session.commit()

                return jsonify({
                    "success": False,
                    "status": "not_registered",
                    "message": f"Student {best_match.student.full_name} is not registered for this exam",
                    "student": {
                        "id": best_match.student.id,
                        "name": best_match.student.full_name,
                        "registration_number": best_match.student.registration_number,
                        "department": best_match.student.department
                    },
                    "confidence_score": float(best_score)
                })

            # Successful verification - create attendance record
            attendance = AttendanceRecord(
                exam_id=exam_id,
                student_id=best_match.student_id,
                verification_attempt_id=verification.id,
                check_in_time=datetime.now(timezone.utc),
                seat_number=registration.seat_number,
                status="present"
            )

            db.session.add(attendance)
            active_session.successful_verifications += 1
            db.session.commit()

            return jsonify({
                "success": True,
                "status": "verified",
                "message": f"Welcome {best_match.student.full_name}!",
                "student": {
                    "id": best_match.student.id,
                    "name": best_match.student.full_name,
                    "registration_number": best_match.student.registration_number,
                    "department": best_match.student.department or "N/A",
                    "seat_number": registration.seat_number
                },
                "confidence_score": float(best_score),
                "verification_id": verification.id,
                "attendance_id": attendance.id,
                "check_in_time": attendance.check_in_time.isoformat(),
                "match_details": match_details
            })

        else:
            # No match found
            active_session.failed_attempts += 1
            db.session.commit()

            return jsonify({
                "success": False,
                "status": "no_match",
                "message": "Fingerprint not recognized. Please try again or contact administrator.",
                "verification_id": verification.id,
                "best_score": float(best_score)
            })

    except Exception as e:
        db.session.rollback()
        print(f"Verification error: {e}")
        return jsonify({
            "success": False,
            "status": "error",
            "message": f"System error during verification: {str(e)}"
        }), 500


@app.route("/api/end-verification-session", methods=["POST"])
def end_verification_session():
    """End the current verification session"""
    try:
        data = request.json
        exam_id = data.get('exam_id')

        # Find active session
        active_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar()

        if not active_session:
            return jsonify({"error": "No active session found"}), 404

        # End session
        active_session.session_end = datetime.now(timezone.utc)
        active_session.is_active = False
        db.session.commit()

        # Calculate session statistics
        session_duration = active_session.session_end - active_session.session_start

        return jsonify({
            "success": True,
            "message": "Verification session ended",
            "session_summary": {
                "duration_minutes": int(session_duration.total_seconds() / 60),
                "total_attempts": active_session.total_attempts,
                "successful_verifications": active_session.successful_verifications,
                "failed_attempts": active_session.failed_attempts,
                "duplicate_attempts": active_session.duplicate_attempts,
                "success_rate": round(
                    (active_session.successful_verifications / max(active_session.total_attempts, 1)) * 100, 2)
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/verification-session-stats/<int:exam_id>")
def get_verification_session_stats(exam_id):
    """Get current session statistics"""
    try:
        active_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar()

        if not active_session:
            return jsonify({"error": "No active session"}), 404

        # Get recent verifications
        recent_verifications = db.session.execute(
            select(VerificationAttempt, Student.first_name, Student.last_name, Student.registration_number)
            .outerjoin(Student)
            .where(
                and_(VerificationAttempt.exam_id == exam_id,
                     VerificationAttempt.verification_timestamp >= active_session.session_start)
            )
            .order_by(VerificationAttempt.verification_timestamp.desc())
            .limit(10)
        ).all()

        recent_list = []
        for verification, first_name, last_name, reg_number in recent_verifications:
            recent_list.append({
                "timestamp": verification.verification_timestamp.isoformat(),
                "status": verification.verification_status,
                "student_name": f"{first_name} {last_name}" if first_name else "Unknown",
                "registration_number": reg_number,
                "confidence": float(verification.confidence_score) if verification.confidence_score else 0
            })

        session_duration = datetime.now(timezone.utc) - active_session.session_start

        return jsonify({
            "session_active": True,
            "session_duration_minutes": int(session_duration.total_seconds() / 60),
            "total_attempts": active_session.total_attempts,
            "successful_verifications": active_session.successful_verifications,
            "failed_attempts": active_session.failed_attempts,
            "duplicate_attempts": active_session.duplicate_attempts,
            "success_rate": round(
                (active_session.successful_verifications / max(active_session.total_attempts, 1)) * 100, 2),
            "recent_verifications": recent_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/exam-attendance-report/<int:exam_id>")
def get_exam_attendance_report(exam_id):
    """Generate attendance report for an exam"""
    try:
        # Get exam details
        exam = db.session.execute(select(Exam).where(Exam.id == exam_id)).scalar()
        if not exam:
            return jsonify({"error": "Exam not found"}), 404

        # Get all registered students
        registered_students = db.session.execute(
            select(ExamRegistration, Student)
            .join(Student)
            .where(ExamRegistration.exam_id == exam_id)
        ).all()

        # Get attendance records
        attendance_records = db.session.execute(
            select(AttendanceRecord, Student)
            .join(Student)
            .where(AttendanceRecord.exam_id == exam_id)
        ).all()

        # Create attendance mapping
        attendance_map = {}
        for record, student in attendance_records:
            attendance_map[student.id] = {
                "check_in_time": record.check_in_time.isoformat(),
                "seat_number": record.seat_number,
                "status": record.status
            }

        # Build report
        attendance_list = []
        present_count = 0

        for registration, student in registered_students:
            is_present = student.id in attendance_map
            if is_present:
                present_count += 1

            attendance_list.append({
                "student_id": student.id,
                "registration_number": student.registration_number,
                "name": student.full_name,
                "department": student.department,
                "seat_number": registration.seat_number,
                "is_present": is_present,
                "attendance_details": attendance_map.get(student.id)
            })

        total_registered = len(registered_students)
        attendance_percentage = (present_count / total_registered * 100) if total_registered > 0 else 0

        return jsonify({
            "exam": {
                "id": exam.id,
                "code": exam.exam_code,
                "title": exam.exam_title,
                "date": exam.exam_date.isoformat(),
                "venue": exam.venue
            },
            "statistics": {
                "total_registered": total_registered,
                "present": present_count,
                "absent": total_registered - present_count,
                "attendance_percentage": round(attendance_percentage, 2)
            },
            "attendance_list": attendance_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Add this enhanced compare_fingerprints function to replace the simple one
def compare_fingerprints(template1, template2):
    """Enhanced fingerprint comparison using multiple algorithms"""
    try:
        confidence_score, match_details = fingerprint_matcher.match_templates(template1, template2)
        return confidence_score
    except Exception as e:
        print(f"Fingerprint comparison error: {e}")
        return 0.0


# Hardware-specific scanner configurations
SCANNER_CONFIGS = {
    "digital_persona": {
        "sdk_path": "/path/to/digitalpersona/sdk",
        "device_id": "default",
        "capture_timeout": 10000
    },
    "zkteco": {
        "ip_address": "192.168.1.201",
        "port": 4370,
        "timeout": 30
    },
    "pyfingerprint": {
        "port": "/dev/ttyUSB0",
        "baudrate": 57600,
        "address": 0xFFFFFFFF,
        "password": 0x00000000
    },
    "opencv": {
        "camera_index": 0,
        "resolution": (640, 480),
        "fps": 30
    }
}

if __name__ == "__main__":
    app.run(debug=True)