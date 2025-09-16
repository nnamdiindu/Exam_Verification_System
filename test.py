# app.py
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import base64
import hashlib
import json

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DB_URI")
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy()
db.init_app(app)


# Simple Models
class Student(db.Model):
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True)
    registration_number = db.Column(db.String(20), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100))
    email = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    fingerprints = db.relationship('FingerprintTemplate', backref='student', lazy=True, cascade='all, delete-orphan')
    registrations = db.relationship('ExamRegistration', backref='student', lazy=True)

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def has_fingerprint(self):
        return len(self.fingerprints) > 0


class FingerprintTemplate(db.Model):
    __tablename__ = 'fingerprint_templates'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    template_data = db.Column(db.LargeBinary, nullable=False)  # Encrypted template
    template_hash = db.Column(db.String(255), nullable=False)  # For quick matching
    quality_score = db.Column(db.Numeric(5, 2))
    enrollment_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)


class Exam(db.Model):
    __tablename__ = 'exams'

    id = db.Column(db.Integer, primary_key=True)
    exam_code = db.Column(db.String(20), nullable=False)
    exam_title = db.Column(db.String(255), nullable=False)
    exam_date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    venue = db.Column(db.String(255))
    duration_minutes = db.Column(db.Integer, default=180)
    status = db.Column(db.String(20), default='scheduled')  # scheduled, active, completed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    registrations = db.relationship('ExamRegistration', backref='exam', lazy=True)
    verifications = db.relationship('VerificationAttempt', backref='exam', lazy=True)


class ExamRegistration(db.Model):
    __tablename__ = 'exam_registrations'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)
    seat_number = db.Column(db.String(20))


class VerificationAttempt(db.Model):
    __tablename__ = 'verification_attempts'

    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=True)  # Null if failed
    verification_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    verification_status = db.Column(db.String(20), nullable=False)  # success, failed
    confidence_score = db.Column(db.Numeric(5, 2))
    template_matched = db.Column(db.Integer, db.ForeignKey('fingerprint_templates.id'))


# Simple Fingerprint Service
class FingerprintService:
    @staticmethod
    def process_fingerprint(fingerprint_data):
        """Simulate fingerprint processing"""
        # In real implementation, use actual fingerprint SDK
        raw_data = base64.b64decode(fingerprint_data)

        # Simulate quality score (60-95)
        quality_score = min(95.0, max(60.0, len(raw_data) % 30 + 65))

        # Create template hash
        template = hashlib.sha256(raw_data).digest()
        template_hash = hashlib.sha256(template).hexdigest()

        return {
            'template': base64.b64encode(template),
            'hash': template_hash,
            'quality': quality_score
        }

    @staticmethod
    def match_fingerprint(query_template, stored_templates):
        """Simple fingerprint matching simulation"""
        best_match = None
        best_confidence = 0.0

        for template_id, stored_hash in stored_templates:
            # Simulate matching (in real implementation, use proper matching algorithm)
            query_hash = hashlib.sha256(base64.b64decode(query_template)).hexdigest()

            if query_hash == stored_hash:
                confidence = 95.0 + (hash(query_hash) % 5)  # 95-99% for exact match
            else:
                # Simulate partial matching
                common_chars = sum(a == b for a, b in zip(query_hash[:20], stored_hash[:20]))
                confidence = (common_chars / 20) * 100

            if confidence > best_confidence and confidence >= 85.0:  # 85% threshold
                best_match = template_id
                best_confidence = confidence

        return best_match, best_confidence


# ROUTES

@app.route('/')
def index():
    """Main dashboard"""
    stats = {
        'total_students': Student.query.count(),
        'enrolled_fingerprints': db.session.query(db.func.count(db.distinct(FingerprintTemplate.student_id))).scalar(),
        'total_exams': Exam.query.count(),
        'active_exams': Exam.query.filter_by(status='active').count()
    }

    recent_verifications = db.session.query(
        VerificationAttempt, Student.registration_number, Student.first_name,
        Student.last_name, Exam.exam_code
    ).outerjoin(Student).join(Exam).order_by(
        VerificationAttempt.verification_timestamp.desc()
    ).limit(5).all()

    return render_template('index.html', stats=stats, recent_verifications=recent_verifications)


# FINGERPRINT ENROLLMENT ROUTES
@app.route('/enrollment')
def enrollment():
    """Fingerprint enrollment page"""
    return render_template('enrollment.html')


@app.route('/api/lookup-student', methods=['POST'])
def lookup_student():
    """Look up student by registration number"""
    reg_number = request.json.get('registration_number')

    if not reg_number:
        return jsonify({'error': 'Registration number required'}), 400

    student = Student.query.filter_by(registration_number=reg_number).first()

    if not student:
        return jsonify({'error': 'Student not found'}), 404

    return jsonify({
        'id': student.id,
        'registration_number': student.registration_number,
        'first_name': student.first_name,
        'last_name': student.last_name,
        'department': student.department or 'N/A',
        'has_fingerprint': student.has_fingerprint,
        'fingerprint_count': len(student.fingerprints)
    })


@app.route('/api/enroll-fingerprint', methods=['POST'])
def enroll_fingerprint():
    """Enroll student fingerprint"""
    try:
        data = request.json
        student_id = data.get('student_id')
        fingerprint_data = data.get('fingerprint_data')

        if not student_id or not fingerprint_data:
            return jsonify({'error': 'Student ID and fingerprint data required'}), 400

        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Student not found'}), 404

        # Process fingerprint
        result = FingerprintService.process_fingerprint(fingerprint_data)

        if result['quality'] < 65.0:
            return jsonify({'error': f'Fingerprint quality too low: {result["quality"]:.1f}%'}), 400

        # Deactivate existing fingerprints
        FingerprintTemplate.query.filter_by(student_id=student_id, is_active=True).update({'is_active': False})

        # Create new template
        template = FingerprintTemplate(
            student_id=student_id,
            template_data=result['template'],
            template_hash=result['hash'],
            quality_score=result['quality']
        )

        db.session.add(template)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Fingerprint enrolled successfully',
            'quality_score': result['quality']
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# LIVE VERIFICATION ROUTES
@app.route('/verification')
def verification():
    """Live verification page"""
    active_exams = Exam.query.filter(
        Exam.status.in_(['scheduled', 'active']),
        Exam.exam_date == date.today()
    ).all()

    return render_template('verification.html', active_exams=active_exams)


@app.route('/api/verify-fingerprint', methods=['POST'])
def verify_fingerprint():
    """Verify fingerprint for exam entry"""
    try:
        data = request.json
        exam_id = data.get('exam_id')
        fingerprint_data = data.get('fingerprint_data')

        if not exam_id or not fingerprint_data:
            return jsonify({'error': 'Exam ID and fingerprint data required'}), 400

        exam = Exam.query.get(exam_id)
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404

        # Get registered students for this exam
        registered_students = db.session.query(
            FingerprintTemplate.id,
            FingerprintTemplate.template_hash,
            FingerprintTemplate.student_id
        ).join(Student).join(ExamRegistration).filter(
            ExamRegistration.exam_id == exam_id,
            FingerprintTemplate.is_active == True
        ).all()

        if not registered_students:
            return jsonify({'error': 'No registered students found'}), 400

        # Process query fingerprint
        query_result = FingerprintService.process_fingerprint(fingerprint_data)

        # Match against enrolled templates
        templates_for_matching = [(t.id, t.template_hash) for t in registered_students]
        matched_template_id, confidence = FingerprintService.match_fingerprint(
            query_result['template'], templates_for_matching
        )

        # Create verification attempt record
        if matched_template_id:
            # Find student for matched template
            matched_student_id = next(t.student_id for t in registered_students if t.id == matched_template_id)

            # Check for duplicate verification
            existing_verification = VerificationAttempt.query.filter_by(
                exam_id=exam_id,
                student_id=matched_student_id,
                verification_status='success'
            ).first()

            if existing_verification:
                # Log duplicate attempt
                attempt = VerificationAttempt(
                    exam_id=exam_id,
                    student_id=matched_student_id,
                    verification_status='duplicate',
                    confidence_score=confidence,
                    template_matched=matched_template_id
                )
                db.session.add(attempt)
                db.session.commit()

                return jsonify({
                    'success': False,
                    'status': 'duplicate',
                    'message': 'Student already verified for this exam'
                })

            # Successful verification
            attempt = VerificationAttempt(
                exam_id=exam_id,
                student_id=matched_student_id,
                verification_status='success',
                confidence_score=confidence,
                template_matched=matched_template_id
            )
            db.session.add(attempt)
            db.session.commit()

            # Get student details
            student = Student.query.get(matched_student_id)

            return jsonify({
                'success': True,
                'status': 'success',
                'confidence': float(confidence),
                'student': {
                    'registration_number': student.registration_number,
                    'full_name': student.full_name,
                    'department': student.department or 'N/A'
                }
            })
        else:
            # Failed verification
            attempt = VerificationAttempt(
                exam_id=exam_id,
                verification_status='failed',
                confidence_score=0.0
            )
            db.session.add(attempt)
            db.session.commit()

            return jsonify({
                'success': False,
                'status': 'failed',
                'message': 'Fingerprint not recognized'
            })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# EXAM MANAGEMENT ROUTES
@app.route('/exams')
def exam_management():
    """Exam management page"""
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


# Initialize database
@app.before_request
def create_tables():
    db.create_all()

    # Add sample data if tables are empty
    if Student.query.count() == 0:
        # Sample students
        students = [
            Student(registration_number='2020/CS/001', first_name='Ada', last_name='Okafor',
                    department='Computer Science'),
            Student(registration_number='2020/CS/002', first_name='Chidi', last_name='Eze',
                    department='Computer Science'),
            Student(registration_number='2020/MT/045', first_name='Ngozi', last_name='Okoro', department='Mathematics')
        ]

        for student in students:
            db.session.add(student)

        db.session.commit()


if __name__ == '__main__':
    app.run(debug=True)