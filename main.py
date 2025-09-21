import os
import hashlib
import base64
import json
import numpy as np
import cv2
import threading
import time
import logging
from datetime import datetime, timezone, date, time
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple

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

# Setup logging
logging.basicConfig(level=logging.INFO)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
db.init_app(app)


# Your existing models remain unchanged
class Student(db.Model):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    registration_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    department: Mapped[Optional[str]] = mapped_column(String(100))

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
    enrollment_date: Mapped[datetime] = mapped_column(DateTime, nullable=False,
                                                      default=lambda: datetime.now(timezone.utc))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    quality_score: Mapped[Decimal] = mapped_column(Numeric(5, 2))

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
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    registrations: Mapped[List["ExamRegistration"]] = relationship(back_populates="exam", lazy="select")
    verifications: Mapped[List["VerificationAttempt"]] = relationship(back_populates="exam", lazy="select")


class ExamRegistration(db.Model):
    __tablename__ = "exam_registrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    registration_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    seat_number: Mapped[str | None] = mapped_column(String(20))

    student: Mapped["Student"] = relationship(back_populates="registrations")
    exam: Mapped[Exam] = relationship(back_populates="registrations")


class VerificationAttempt(db.Model):
    __tablename__ = "verification_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    student_id: Mapped[int | None] = mapped_column(ForeignKey("students.id"), nullable=True)
    verification_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    verification_status: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    template_matched: Mapped[int | None] = mapped_column(ForeignKey("fingerprint_templates.id"))

    exam: Mapped["Exam"] = relationship(back_populates="verifications")
    student: Mapped[Optional["Student"]] = relationship()
    matched_template: Mapped[Optional["FingerprintTemplate"]] = relationship(
        back_populates="verification_attempts")


# ================================
# BIOMETRIC SCANNER INTEGRATION
# ================================

class BiometricScannerInterface:
    """Base interface for biometric scanner integration"""

    def __init__(self):
        self.is_connected = False
        self.last_scan_data = None
        self.scan_callback = None

    def connect(self) -> bool:
        """Connect to the biometric scanner"""
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the scanner"""
        raise NotImplementedError

    def capture_fingerprint(self) -> Optional[bytes]:
        """Capture fingerprint image"""
        raise NotImplementedError

    def set_scan_callback(self, callback):
        """Set callback for automatic scan detection"""
        self.scan_callback = callback


class DigitalPersonaScanner(BiometricScannerInterface):
    """Integration for DigitalPersona scanners"""

    def __init__(self):
        super().__init__()
        self.sdk_available = False
        try:
            # Try to import DigitalPersona SDK
            # import digitalpersona
            # self.dp_scanner = digitalpersona.Scanner()
            # self.sdk_available = True
            pass
        except ImportError:
            logging.warning("DigitalPersona SDK not installed. Install from manufacturer.")

    def connect(self) -> bool:
        if not self.sdk_available:
            return False

        try:
            # Real implementation would be:
            # self.dp_scanner.connect()
            # self.is_connected = self.dp_scanner.is_connected()

            # For demo purposes, simulate connection
            self.is_connected = True
            logging.info("DigitalPersona scanner connected")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to DigitalPersona scanner: {e}")
            return False

    def capture_fingerprint(self) -> Optional[bytes]:
        if not self.is_connected:
            return None

        try:
            # Real implementation would be:
            # fingerprint_image = self.dp_scanner.capture()
            # return fingerprint_image.to_bytes()

            # Return None for demo (would return actual image bytes)
            return None
        except Exception as e:
            logging.error(f"Failed to capture fingerprint: {e}")
            return None


class FutronicScanner(BiometricScannerInterface):
    """Integration for Futronic scanners"""

    def __init__(self):
        super().__init__()
        self.sdk_available = False
        try:
            # Try to import Futronic SDK
            # import ftrScanAPI
            # self.futronic_device = ftrScanAPI.ftrScanAPI()
            # self.sdk_available = True
            pass
        except ImportError:
            logging.warning("Futronic SDK not installed")

    def connect(self) -> bool:
        if not self.sdk_available:
            return False

        try:
            # Real implementation would be:
            # result = self.futronic_device.ftrScanOpenDevice()
            # self.is_connected = (result == ftrScanAPI.FTR_RETCODE_OK)

            # Simulate connection
            self.is_connected = True
            logging.info("Futronic scanner connected")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Futronic scanner: {e}")
            return False

    def capture_fingerprint(self) -> Optional[bytes]:
        if not self.is_connected:
            return None

        try:
            # Real implementation would be:
            # image_buffer = (ftrScanAPI.UCHAR * 160000)()  # Buffer size for image
            # result = self.futronic_device.ftrScanGetImage(4, image_buffer)
            # if result == ftrScanAPI.FTR_RETCODE_OK:
            #     return bytes(image_buffer)
            return None
        except Exception as e:
            logging.error(f"Failed to capture fingerprint: {e}")
            return None


class WebcamFingerprint(BiometricScannerInterface):
    """Use webcam as fingerprint input (for testing/demo purposes)"""

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.cap = None

    def connect(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                # Set camera properties for better fingerprint capture
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                self.is_connected = True
                logging.info("Webcam connected for fingerprint capture")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to connect to webcam: {e}")
            return False

    def disconnect(self):
        if self.cap:
            self.cap.release()
        self.is_connected = False
        logging.info("Webcam disconnected")

    def capture_fingerprint(self) -> Optional[bytes]:
        if not self.is_connected or not self.cap:
            return None

        try:
            # Capture multiple frames and take the best quality one
            best_frame = None
            best_quality = 0

            for _ in range(5):
                ret, frame = self.cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Calculate frame quality (sharpness)
                    quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if quality > best_quality:
                        best_quality = quality
                        best_frame = gray

                time.sleep(0.1)  # Small delay between captures

            if best_frame is not None:
                # Crop to center region (assuming finger placement)
                h, w = best_frame.shape
                crop_size = min(h, w) // 2
                center_x, center_y = w // 2, h // 2
                cropped = best_frame[
                          center_y - crop_size // 2:center_y + crop_size // 2,
                          center_x - crop_size // 2:center_x + crop_size // 2
                          ]

                # Apply basic fingerprint enhancement
                cropped = cv2.equalizeHist(cropped)

                # Encode as PNG for better quality
                _, buffer = cv2.imencode('.png', cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return buffer.tobytes()
            return None
        except Exception as e:
            logging.error(f"Failed to capture from webcam: {e}")
            return None


class ScannerManager:
    """Manager class to handle different scanner types"""

    def __init__(self):
        self.scanners = []
        self.active_scanner = None
        self.auto_scan_thread = None
        self.auto_scan_active = False
        self.scan_lock = threading.Lock()

    def add_scanner(self, scanner: BiometricScannerInterface):
        """Add a scanner to the manager"""
        self.scanners.append(scanner)
        logging.info(f"Added scanner: {scanner.__class__.__name__}")

    def connect_scanners(self) -> Dict[str, bool]:
        """Try to connect to all available scanners"""
        results = {}
        for scanner in self.scanners:
            scanner_name = scanner.__class__.__name__
            results[scanner_name] = scanner.connect()
            if results[scanner_name] and not self.active_scanner:
                self.active_scanner = scanner
                logging.info(f"Set {scanner_name} as active scanner")
        return results

    def get_active_scanner_info(self) -> Dict[str, Any]:
        """Get information about the active scanner"""
        if not self.active_scanner:
            return {'connected': False, 'scanner_type': None}

        return {
            'connected': self.active_scanner.is_connected,
            'scanner_type': self.active_scanner.__class__.__name__,
            'supports_auto_scan': hasattr(self.active_scanner, 'start_auto_detection')
        }

    def capture_fingerprint(self) -> Optional[str]:
        """Capture fingerprint and return as base64 string"""
        if not self.active_scanner or not self.active_scanner.is_connected:
            logging.error("No active scanner available")
            return None

        with self.scan_lock:
            try:
                image_bytes = self.active_scanner.capture_fingerprint()
                if image_bytes:
                    return base64.b64encode(image_bytes).decode('utf-8')
                return None
            except Exception as e:
                logging.error(f"Error capturing fingerprint: {e}")
                return None

    def start_auto_scan(self, callback, interval=1.0):
        """Start automatic scanning in background thread"""
        if self.auto_scan_active:
            logging.warning("Auto-scan already active")
            return

        self.auto_scan_active = True
        self.auto_scan_thread = threading.Thread(
            target=self._auto_scan_worker,
            args=(callback, interval),
            daemon=True
        )
        self.auto_scan_thread.start()
        logging.info("Auto-scan started")

    def stop_auto_scan(self):
        """Stop automatic scanning"""
        self.auto_scan_active = False
        if self.auto_scan_thread and self.auto_scan_thread.is_alive():
            self.auto_scan_thread.join(timeout=3)
        logging.info("Auto-scan stopped")

    def _auto_scan_worker(self, callback, interval):
        """Background worker for automatic scanning"""
        while self.auto_scan_active:
            try:
                fingerprint_data = self.capture_fingerprint()
                if fingerprint_data:
                    callback(fingerprint_data)
                    time.sleep(interval * 2)  # Wait longer after successful capture
                else:
                    time.sleep(interval)
            except Exception as e:
                logging.error(f"Error in auto-scan: {e}")
                time.sleep(interval)

    def disconnect_all(self):
        """Disconnect all scanners"""
        for scanner in self.scanners:
            try:
                scanner.disconnect()
            except Exception as e:
                logging.error(f"Error disconnecting {scanner.__class__.__name__}: {e}")


# Global scanner manager
scanner_manager = ScannerManager()


# Your existing OpenCV fingerprint processor remains unchanged
class OpenCVFingerprintProcessor:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Invalid image data")
        image = cv2.resize(image, (256, 256))
        image = cv2.equalizeHist(image)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        if descriptors is None:
            raise ValueError("No features could be extracted from the image")
        keypoints_data = []
        for kp in keypoints:
            keypoints_data.append({
                'x': float(kp.pt[0]), 'y': float(kp.pt[1]),
                'angle': float(kp.angle), 'response': float(kp.response),
                'octave': int(kp.octave), 'size': float(kp.size)
            })
        return keypoints_data, descriptors

    def calculate_quality_score(self, image: np.ndarray, keypoints: List[Dict]) -> float:
        contrast = np.std(image)
        contrast_score = min(contrast / 50.0, 1.0)
        area = image.shape[0] * image.shape[1]
        density = len(keypoints) / (area / 10000)
        density_score = min(density / 5.0, 1.0)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        quality = (contrast_score * 0.3 + density_score * 0.4 + sharpness_score * 0.3) * 100
        return max(30.0, min(95.0, quality))

    def create_template(self, image_data: bytes) -> Dict[str, Any]:
        processed_image = self.preprocess_image(image_data)
        keypoints, descriptors = self.extract_features(processed_image)
        quality_score = self.calculate_quality_score(processed_image, keypoints)
        template = {
            'keypoints': keypoints,
            'descriptors': descriptors.tolist(),
            'image_shape': processed_image.shape,
            'quality_score': quality_score,
            'feature_count': len(keypoints),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        return template

    def match_templates(self, template1: Dict[str, Any], template2: Dict[str, Any]) -> float:
        try:
            desc1 = np.array(template1['descriptors'], dtype=np.uint8)
            desc2 = np.array(template2['descriptors'], dtype=np.uint8)
            if len(desc1) < 10 or len(desc2) < 10:
                return 0.0
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            match_ratio = len(good_matches) / min(len(desc1), len(desc2))
            confidence = min(match_ratio * 100, 95.0)
            if len(good_matches) >= 10:
                kp1 = template1['keypoints']
                kp2 = template2['keypoints']
                src_pts = np.float32([[kp1[m.queryIdx]['x'], kp1[m.queryIdx]['y']]
                                      for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([[kp2[m.trainIdx]['x'], kp2[m.trainIdx]['y']]
                                      for m in good_matches]).reshape(-1, 1, 2)
                if len(src_pts) >= 4:
                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers = np.sum(mask)
                        spatial_consistency = inliers / len(mask)
                        confidence *= spatial_consistency
            return confidence
        except Exception as e:
            logging.error(f"Error matching templates: {e}")
            return 0.0

    def encode_template(self, template: Dict[str, Any]) -> bytes:
        template_json = json.dumps(template, ensure_ascii=False)
        return template_json.encode('utf-8')

    def decode_template(self, template_bytes: bytes) -> Dict[str, Any]:
        template_json = template_bytes.decode('utf-8')
        return json.loads(template_json)


# Initialize fingerprint processor
fp_processor = OpenCVFingerprintProcessor()


def initialize_scanners():
    """Initialize available scanners"""
    global scanner_manager

    # Add different scanner types in order of preference
    scanner_manager.add_scanner(DigitalPersonaScanner())
    scanner_manager.add_scanner(FutronicScanner())
    scanner_manager.add_scanner(WebcamFingerprint())  # Fallback for testing

    # Try to connect
    results = scanner_manager.connect_scanners()
    logging.info(f"Scanner connection results: {results}")

    return any(results.values())


# Initialize database and scanners
with app.app_context():
    db.create_all()


# Initialize scanners on startup
@app.before_request
def setup_scanners():
    if initialize_scanners():
        logging.info("Biometric scanners initialized successfully")
    else:
        logging.warning("No biometric scanners connected - using mock mode")


# ================================
# NEW SCANNER API ROUTES
# ================================

@app.route('/api/scanner/status')
def scanner_status():
    """Get scanner connection status"""
    return jsonify(scanner_manager.get_active_scanner_info())


@app.route('/api/scanner/capture', methods=['POST'])
def capture_from_scanner():
    """Capture fingerprint from connected scanner"""
    fingerprint_data = scanner_manager.capture_fingerprint()
    if fingerprint_data:
        return jsonify({
            'success': True,
            'fingerprint_data': fingerprint_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to capture fingerprint from scanner'
        }), 500


@app.route('/api/scanner/start-auto', methods=['POST'])
def start_auto_scan():
    """Start automatic scanning mode"""

    def scan_callback(fingerprint_data):
        logging.info(f"Auto-captured fingerprint at {datetime.now()}")
        # Store the captured data temporarily
        # You could implement WebSocket here to push data to frontend

    scanner_manager.start_auto_scan(scan_callback, interval=0.5)
    return jsonify({'success': True, 'message': 'Auto-scan started'})


@app.route('/api/scanner/stop-auto', methods=['POST'])
def stop_auto_scan():
    """Stop automatic scanning mode"""
    scanner_manager.stop_auto_scan()
    return jsonify({'success': True, 'message': 'Auto-scan stopped'})


# ================================
# YOUR EXISTING ROUTES (UPDATED)
# ================================

@app.route("/")
def index():
    stats = {
        'total_students': db.session.execute(select(func.count()).select_from(Student)).scalar(),
        'enrolled_fingerprints': db.session.execute(
            select(func.count(distinct(FingerprintTemplate.student_id)))).scalar(),
        'total_exams': db.session.execute(select(func.count()).select_from(Exam)).scalar(),
        'active_exams': db.session.execute(
            select(func.count()).select_from(Exam).where(Exam.status == 'active')).scalar()
    }
    stmt = select(VerificationAttempt, Student.registration_number, Student.first_name, Student.last_name,
                  Exam.exam_code).outerjoin(Student).join(Exam).order_by(
        VerificationAttempt.verification_timestamp.desc()).limit(5)
    recent_verifications = db.session.execute(stmt).all()
    return render_template("index.html", stats=stats, recent_verifications=recent_verifications)


@app.route("/enrollment")
def enrollment():
    return render_template("enrollment.html")


@app.route("/api/lookup-student", methods=["POST"])
def lookup_student():
    reg_number = request.json.get("registration_number")
    if not reg_number:
        return jsonify({"error": "Registration number required"}), 400
    student = db.session.execute(select(Student).where(Student.registration_number == reg_number)).scalar()
    if not student:
        return jsonify({"error": "Student not found"}), 404
    return jsonify({
        "id": student.id,
        "registration_number": student.registration_number,
        "first_name": student.first_name,
        "last_name": student.last_name,
        "department": student.department or 'N/A',
        "has_fingerprint": student.has_fingerprint,
        "fingerprint_count": len(student.fingerprints)
    })


@app.route("/api/enroll-fingerprint", methods=["POST"])
def enroll_fingerprint():
    """Enroll fingerprint using OpenCV processing"""
    try:
        data = request.json
        student_id = data.get('student_id')
        fingerprint_data = data.get('fingerprint_data')

        if not student_id or not fingerprint_data:
            return jsonify({'error': 'Student ID and fingerprint data required'}), 400

        student = db.session.get(Student, student_id)
        if not student:
            return jsonify({'error': 'Student not found'}), 404

        existing_template = db.session.execute(
            select(FingerprintTemplate).where(
                FingerprintTemplate.student_id == student_id,
                FingerprintTemplate.is_active == True
            )
        ).scalar()

        if existing_template:
            return jsonify({'error': 'Student already has fingerprint enrolled'}), 400

        try:
            image_bytes = base64.b64decode(fingerprint_data)
        except Exception:
            return jsonify({'error': 'Invalid fingerprint image data'}), 400

        template = fp_processor.create_template(image_bytes)

        if template['quality_score'] < 40.0:
            return jsonify({
                'error': 'Fingerprint quality too low. Please try again.',
                'quality_score': template['quality_score']
            }), 400

        template_bytes = fp_processor.encode_template(template)
        template_hash = hashlib.sha256(template_bytes).hexdigest()

        duplicate = db.session.execute(
            select(FingerprintTemplate).where(FingerprintTemplate.template_hash == template_hash)
        ).scalar()

        if duplicate:
            return jsonify({'error': 'This fingerprint is already enrolled'}), 400

        fingerprint_template = FingerprintTemplate(
            student_id=student_id,
            template_data=template_bytes,
            template_hash=template_hash,
            quality_score=Decimal(str(template['quality_score']))
        )

        db.session.add(fingerprint_template)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Fingerprint enrolled successfully',
            'quality_score': template['quality_score'],
            'feature_count': template['feature_count']
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Enrollment failed: {str(e)}'}), 500


@app.route("/verification")
def verification():
    active_exams = db.session.execute(
        select(Exam).where(Exam.status.in_(["scheduled", "active"])).order_by(Exam.exam_date, Exam.start_time)
    ).scalars().all()
    return render_template("verification.html", active_exams=active_exams)


@app.route("/api/verify-fingerprint", methods=["POST"])
def verify_fingerprint():
    """Verify fingerprint against enrolled templates"""
    try:
        data = request.json
        exam_id = data.get('exam_id')
        fingerprint_data = data.get('fingerprint_data')

        if not exam_id or not fingerprint_data:
            return jsonify({'error': 'Exam ID and fingerprint data required'}), 400

        exam = db.session.get(Exam, exam_id)
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404

        try:
            image_bytes = base64.b64decode(fingerprint_data)
        except Exception:
            return jsonify({'error': 'Invalid fingerprint data'}), 400

        verify_template = fp_processor.create_template(image_bytes)

        if verify_template['quality_score'] < 30.0:
            return jsonify({
                'success': False,
                'status': 'failed',
                'message': 'Poor fingerprint quality. Please try again.'
            })

        enrolled_templates = db.session.execute(
            select(FingerprintTemplate).where(
                FingerprintTemplate.is_active == True
            ).join(Student).join(ExamRegistration).where(
                ExamRegistration.exam_id == exam_id
            )
        ).scalars().all()

        best_match = None
        best_confidence = 0.0

        for template_record in enrolled_templates:
            stored_template = fp_processor.decode_template(template_record.template_data)
            confidence = fp_processor.match_templates(verify_template, stored_template)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = template_record

        VERIFICATION_THRESHOLD = 60.0

        if best_match and best_confidence >= VERIFICATION_THRESHOLD:
            existing_verification = db.session.execute(
                select(VerificationAttempt).where(
                    VerificationAttempt.exam_id == exam_id,
                    VerificationAttempt.student_id == best_match.student_id,
                    VerificationAttempt.verification_status == 'success'
                )
            ).scalar()

            if existing_verification:
                return jsonify({
                    'success': False,
                    'status': 'duplicate',
                    'message': f'Student {best_match.student.full_name} already verified for this exam'
                })

            verification = VerificationAttempt(
                exam_id=exam_id,
                student_id=best_match.student_id,
                verification_status='success',
                confidence_score=Decimal(str(best_confidence)),
                template_matched=best_match.id
            )

            db.session.add(verification)
            db.session.commit()

            return jsonify({
                'success': True,
                'student': {
                    'full_name': best_match.student.full_name,
                    'registration_number': best_match.student.registration_number,
                    'department': best_match.student.department
                },
                'confidence': best_confidence
            })
        else:
            verification = VerificationAttempt(
                exam_id=exam_id,
                student_id=None,
                verification_status='failed',
                confidence_score=Decimal(str(best_confidence)) if best_confidence > 0 else None
            )

            db.session.add(verification)
            db.session.commit()

            return jsonify({
                'success': False,
                'status': 'failed',
                'message': 'Fingerprint not recognized. Please ensure you are registered for this exam.'
            })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'status': 'error',
            'message': f'Verification failed: {str(e)}'
        }), 500


# Your existing exam management routes
@app.route("/exams")
def exam_management():
    exams = Exam.query.order_by(Exam.exam_date.desc()).all()
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
    try:
        data = request.json
        student_id = data.get('student_id')
        exam_id = data.get('exam_id')

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


# Cleanup function for graceful shutdown
@app.teardown_appcontext
def cleanup_scanners(error):
    """Cleanup scanners on app shutdown"""
    if error:
        logging.error(f"App error: {error}")
    # Don't disconnect scanners on every request - only on actual shutdown


def shutdown_scanners():
    """Call this on application shutdown"""
    scanner_manager.stop_auto_scan()
    scanner_manager.disconnect_all()
    logging.info("All scanners disconnected")


if __name__ == "__main__":
    app.run(debug=True)
    # try:
    #     app.run(debug=True, host='0.0.0.0', port=5000)
    # except KeyboardInterrupt:
    #     logging.info("Shutting down application...")
    #     shutdown_scanners()
    # except Exception as e:
    #     logging.error(f"Application error: {e}")
    #     shutdown_scanners()
    # finally:
    #     logging.info("Application shutdown complete")