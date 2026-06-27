"""Extract exercise landmarks from MediaPipe pose results."""

TARGET_JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def get_landmarks(result):
    """
    Args:
        result: PoseLandmarkerResult from PoseDetector.process().

    Returns:
        dict of {joint_name: {"x", "y", "z", "visibility"}} or None
        if no person was detected this frame.
    """
    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    extracted = {}

    for joint_name, idx in TARGET_JOINTS.items():
        lm = landmarks[idx]
        extracted[joint_name] = {
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility,
        }

    visible_points = [
        point for point in extracted.values() if point.get("visibility", 0) >= 0.45
    ]
    if visible_points:
        x_values = [point["x"] for point in visible_points]
        y_values = [point["y"] for point in visible_points]
        extracted["_body_box"] = {
            "width": max(x_values) - min(x_values),
            "height": max(y_values) - min(y_values),
        }

    return extracted
