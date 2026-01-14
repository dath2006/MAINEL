import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Dict, Any, Optional
import os

class PoseEstimator:
    """
    Robust Face Quality Assessment using MediaPipe Tasks API (FaceLandmarker).
    Implements ISO/IEC 29794-5 concepts (Pose, Sharpness, Illumination).
    """

    def __init__(self):
        # Path to model
        model_path = os.path.join("model_weights", "face_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face Landmarker model not found at {model_path}")
            
        # Initialize FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True, # Direct Pose Matrix!
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Generic 3D Model Points (approximate) - Still used for fallback/robustness if internal matrix fails
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left Eye Left Corner
            (225.0, 170.0, -135.0),      # Right Eye Right Corner
            (-150.0, -150.0, -125.0),    # Left Mouth Corner
            (150.0, -150.0, -125.0)      # Right Mouth Corner
        ])
        
        # MediaPipe Indices for the above points (New 478 landmark model is compatible with 468 indices)
        self.keypoint_indices = [1, 152, 33, 263, 61, 291]

    def get_pose_score(self, image: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Pose Score based on Yaw/Pitch.
        Returns (score 0.0-1.0, details dict).
        """
        if image is None or image.size == 0:
            return 0.0, {}

        # Convert to MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return 0.2, {"yaw": 999, "pitch": 999} # No face found -> Low score

        # 1. Try using the native Transformation Matrix if available (most accurate)
        if detection_result.facial_transformation_matrixes:
            matrix = detection_result.facial_transformation_matrixes[0]
            # Matrix is 4x4. We need Euler angles.
            # Only reliable if model supports it correctly. Often easier to re-Calculate PnP from landmarks
            # to be consistent with our specific "Frontal" definition. 
            pass

        # 2. Fallback/Standard: SolvePnP on landmarks (Standard ISO approach)
        landmarks = detection_result.face_landmarks[0]
        h, w, c = image.shape
        
        # Extract 2D image points
        image_points = []
        for idx in self.keypoint_indices:
            lm = landmarks[idx]
            # Landmarks are normalized [0, 1]
            x, y = int(lm.x * w), int(lm.y * h)
            image_points.append((x, y))
            
        image_points = np.array(image_points, dtype="double")
        
        # Camera Matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.2, {"error": "pnp_failed"}

        # Convert Rotation Vector to Euler Angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        
        pitch, yaw, roll = [float(x) for x in euler_angles]
        
        # Normalize scores
        yaw_score = max(0.0, 1.0 - (abs(yaw) / 30.0))  # 0 at 30 degrees
        pitch_score = max(0.0, 1.0 - (abs(pitch) / 30.0))
        
        pose_score = (yaw_score * 0.7 + pitch_score * 0.3)
        pose_score = min(max(pose_score, 0.0), 1.0)
        
        return pose_score, {"yaw": yaw, "pitch": pitch, "roll": roll}

    def get_sharpness_score(self, image: np.ndarray) -> float:
        """Tenengrad Sharpness (Gradient Magnitude)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        avg_mag = np.mean(magnitude)
        score = min(1.0, avg_mag / 15.0)
        return score

    def get_illumination_score(self, image: np.ndarray) -> float:
        """Brightness check."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 40: return 0.2
        if mean_brightness > 220: return 0.2
        return 1.0
