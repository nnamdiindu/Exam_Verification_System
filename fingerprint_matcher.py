import json

# Enhanced fingerprint matching algorithms
class FingerprintMatcher:
    """Advanced fingerprint matching with multiple algorithms"""

    def __init__(self):
        self.matching_threshold = 80.0
        self.high_confidence_threshold = 95.0

    def match_templates(self, captured_template, stored_template):
        """
        Match captured fingerprint against stored template
        Returns: (confidence_score, match_details)
        """
        try:
            if isinstance(captured_template, str):
                captured_data = json.loads(captured_template)
            else:
                captured_data = captured_template

            if isinstance(stored_template, bytes):
                stored_data = json.loads(stored_template.decode('utf-8'))
            else:
                stored_data = stored_template

            # Method 1: Minutiae matching
            minutiae_score = self._match_minutiae(captured_data, stored_data)

            # Method 2: Ridge pattern matching (simplified)
            ridge_score = self._match_ridge_patterns(captured_data, stored_data)

            # Method 3: Quality-weighted scoring
            quality_weight = min(captured_data.get('quality_score', 80), stored_data.get('quality_score', 80)) / 100.0

            # Combined score
            combined_score = (minutiae_score * 0.6 + ridge_score * 0.4) * quality_weight

            match_details = {
                'minutiae_score': minutiae_score,
                'ridge_score': ridge_score,
                'quality_weight': quality_weight,
                'combined_score': combined_score,
                'threshold_met': combined_score >= self.matching_threshold
            }

            return combined_score, match_details

        except Exception as e:
            print(f"Template matching error: {e}")
            return 0.0, {'error': str(e)}

    def _match_minutiae(self, template1, template2):
        """Match minutiae points between templates"""
        minutiae1 = template1.get('minutiae', [])
        minutiae2 = template2.get('minutiae', [])

        if not minutiae1 or not minutiae2:
            return 0.0

        matches = 0
        tolerance = 10  # pixel tolerance

        for m1 in minutiae1:
            for m2 in minutiae2:
                distance = ((m1['x'] - m2['x']) ** 2 + (m1['y'] - m2['y']) ** 2) ** 0.5
                if distance <= tolerance:
                    matches += 1
                    break

        # Calculate score based on matching minutiae
        total_minutiae = max(len(minutiae1), len(minutiae2))
        if total_minutiae == 0:
            return 0.0

        return min(100.0, (matches / total_minutiae) * 150)

    def _match_ridge_patterns(self, template1, template2):
        """Match ridge patterns (simplified implementation)"""
        # In production, implement actual ridge pattern matching
        # For now, use image similarity if available

        quality1 = template1.get('quality_score', 0)
        quality2 = template2.get('quality_score', 0)

        # Simplified ridge matching based on quality correlation
        quality_diff = abs(quality1 - quality2)
        ridge_score = max(0, 100 - (quality_diff * 2))

        return ridge_score