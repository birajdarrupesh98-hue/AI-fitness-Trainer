"""
camera/webcam.py

Wraps cv2.VideoCapture so app.py doesn't deal with raw OpenCV calls
or error handling directly. Reusable across future phases (e.g. if
we later support video file uploads instead of a live webcam).
"""

import cv2


class WebcamStream:
    """Thin wrapper around cv2.VideoCapture with safe open/read/release."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def open(self) -> bool:
        """
        Opens the webcam. Returns True on success, False on failure.
        Never raises — caller decides how to show the error in the UI.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def read_frame(self):
        """
        Reads one frame.
        Returns (True, frame) on success, (False, None) on failure
        (e.g. camera disconnected mid-stream).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self):
        """Releases the camera so other apps can use it."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
