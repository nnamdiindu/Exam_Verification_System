import base64
import cv2
import numpy as np
# import digitalpersona as dp


class FingerprintScanner:
    """Real fingerprint scanner interface"""

    def __init__(self, scanner_type="opencv"):
        self.scanner_type = scanner_type
        self.is_connected = False
        self.scanner_device = None
        self.initialize_scanner()

    def initialize_scanner(self):
        """Initialize the fingerprint scanner hardware"""
        try:
            if self.scanner_type == "digital_persona":
                # Digital Persona U.are.U scanner initialization
                self.scanner_device = dp.UareUGlobal()
                self.is_connected = self.scanner_device.Initialize()
                # pass

            elif self.scanner_type == "zkteco":
                # ZKTeco device initialization
                # self.scanner_device = ZK('192.168.1.201', port=4370)
                # self.is_connected = self.scanner_device.connect()
                pass

            elif self.scanner_type == "opencv":
                # Camera-based fingerprint capture
                self.scanner_device = cv2.VideoCapture(0)
                self.is_connected = self.scanner_device.isOpened()

            elif self.scanner_type == "pyfingerprint":
                # Generic USB fingerprint scanner
                from pyfingerprint import PyFingerprint
                self.scanner_device = PyFingerprint('/dev/ttyUSB0', 57600, 0xFFFFFFFF, 0x00000000)
                self.is_connected = self.scanner_device.verifyPassword()
                # pass

            print(f"Scanner initialized: {self.is_connected}")

        except Exception as e:
            print(f"Scanner initialization failed: {e}")
            self.is_connected = False

    def capture_fingerprint(self):
        """Capture fingerprint from hardware scanner"""
        if not self.is_connected:
            raise Exception("Scanner not connected")

        try:
            if self.scanner_type == "digital_persona":
                # Digital Persona capture
                raw_image = self.scanner_device.CaptureImage()
                return self.process_dp_image(raw_image)
                # pass

            elif self.scanner_type == "zkteco":
                # ZKTeco capture
                # template = self.scanner_device.enroll_user()
                # return template
                pass

            elif self.scanner_type == "opencv":
                # Camera-based capture
                return self.capture_from_camera()

            elif self.scanner_type == "pyfingerprint":
                # USB scanner capture
                if self.scanner_device.readImage():
                    self.scanner_device.convertImage(0x01)
                    characteristics = self.scanner_device.downloadCharacteristics(0x01)
                    return characteristics
                # pass

        except Exception as e:
            raise Exception(f"Fingerprint capture failed: {e}")

    def capture_from_camera(self):
        """Capture fingerprint using camera (for development/testing)"""
        ret, frame = self.scanner_device.read()
        if not ret:
            raise Exception("Failed to capture image from camera")

        # Convert to grayscale for fingerprint processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply fingerprint enhancement techniques
        enhanced = self.enhance_fingerprint_image(gray)

        # Convert to base64 for storage
        _, buffer = cv2.imencode('.png', enhanced)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'image_data': img_base64,
            'quality_score': self.calculate_quality_score(enhanced),
            'template': self.extract_minutiae(enhanced)
        }

    def enhance_fingerprint_image(self, image):
        """Enhance fingerprint image quality"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        return enhanced

    def calculate_quality_score(self, image):
        """Calculate fingerprint quality score"""
        # Calculate image sharpness using Laplacian variance
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpness = laplacian.var()

        # Calculate contrast
        contrast = image.std()

        # Simple quality score based on sharpness and contrast
        quality = min(100, (sharpness * contrast) / 1000)
        return max(0, quality)

    def extract_minutiae(self, image):
        """Extract minutiae points from fingerprint"""
        # This is a simplified version - in production, use specialized libraries
        # like SourceAFIS, NIST BOZORTH3, or commercial SDKs

        # Apply binary threshold
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Find contours (simplified minutiae extraction)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract basic features
        minutiae_points = []
        for contour in contours[:20]:  # Limit to 20 points
            if cv2.contourArea(contour) > 10:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    minutiae_points.append({'x': cx, 'y': cy, 'type': 'ending'})

        return minutiae_points