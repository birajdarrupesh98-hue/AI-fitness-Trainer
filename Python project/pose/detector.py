"""MediaPipe Tasks pose detector and overlay drawing."""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)

# Only these joints get their name printed; every landmark still
# gets a dot + index number (see draw_overlay below).
NAMED_JOINTS = {
    11: "L_SHO",
    12: "R_SHO",
    13: "L_ELB",
    14: "R_ELB",
    15: "L_WRI",
    16: "R_WRI",
    23: "L_HIP",
    24: "R_HIP",
    25: "L_KNE",
    26: "R_KNE",
    27: "L_ANK",
    28: "R_ANK",
}


class PoseDetector:
    """Detects body pose on a frame and draws a labeled overlay."""

    def __init__(
        self,
        model_path="models/pose_landmarker_lite.task",
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. "
                "Add pose_landmarker_lite.task inside the models folder."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """
        Takes a raw BGR frame (as read from OpenCV), runs pose detection,
        and returns (frame_rgb, results):
          - frame_rgb: RGB frame, safe to display in Streamlit
          - results: MediaPipe Tasks result object
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)

        return frame_rgb, results

    def draw_overlay(self, frame_rgb, results):
        """
        Draws skeleton + colored landmarks + index numbers + joint
        names directly onto frame_rgb (in place) and returns it.
        """
        if not results.pose_landmarks:
            return frame_rgb

        h, w, _ = frame_rgb.shape
        landmarks = results.pose_landmarks[0]

        for start_idx, end_idx in POSE_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_xy = (int(start.x * w), int(start.y * h))
            end_xy = (int(end.x * w), int(end.y * h))
            cv2.line(frame_rgb, start_xy, end_xy, (0, 200, 0), 3, cv2.LINE_AA)

        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_rgb, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(
                frame_rgb, str(idx), (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA,
            )
            if idx in NAMED_JOINTS:
                name = NAMED_JOINTS[idx]
                cv2.putText(
                    frame_rgb, name, (cx + 6, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
                )
        return frame_rgb

    def close(self):
        """Releases MediaPipe's internal resources."""
        self.landmarker.close()
