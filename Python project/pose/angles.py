"""Angle math for exercise form checks."""

import numpy as np


def calculate_angle(a, b, c):
    """
    Calculates the angle ABC in degrees.

    Points are dictionaries with x/y/z keys from pose.landmarks.get_landmarks.
    The middle point, b, is the joint whose angle we want.
    """
    a_vec = np.array([a["x"], a["y"], a["z"]], dtype=float)
    b_vec = np.array([b["x"], b["y"], b["z"]], dtype=float)
    c_vec = np.array([c["x"], c["y"], c["z"]], dtype=float)

    ba = a_vec - b_vec
    bc = c_vec - b_vec

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(float(angle), 1)


def get_knee_angles(landmarks):
    """Returns left/right knee angles, or None values if landmarks are missing."""
    if landmarks is None:
        return {"left": None, "right": None}

    return {
        "left": calculate_angle(
            landmarks["left_hip"],
            landmarks["left_knee"],
            landmarks["left_ankle"],
        ),
        "right": calculate_angle(
            landmarks["right_hip"],
            landmarks["right_knee"],
            landmarks["right_ankle"],
        ),
    }


def get_elbow_angles(landmarks):
    """Returns left/right elbow angles, or None values if landmarks are missing."""
    if landmarks is None:
        return {"left": None, "right": None}

    return {
        "left": calculate_angle(
            landmarks["left_shoulder"],
            landmarks["left_elbow"],
            landmarks["left_wrist"],
        ),
        "right": calculate_angle(
            landmarks["right_shoulder"],
            landmarks["right_elbow"],
            landmarks["right_wrist"],
        ),
    }


def get_hip_angles(landmarks):
    """Returns left/right hip angles for sit-up style movement."""
    if landmarks is None:
        return {"left": None, "right": None}

    return {
        "left": calculate_angle(
            landmarks["left_shoulder"],
            landmarks["left_hip"],
            landmarks["left_knee"],
        ),
        "right": calculate_angle(
            landmarks["right_shoulder"],
            landmarks["right_hip"],
            landmarks["right_knee"],
        ),
    }


def get_shoulder_angles(landmarks):
    """Returns left/right shoulder angles for raise movements."""
    if landmarks is None:
        return {"left": None, "right": None}

    return {
        "left": calculate_angle(
            landmarks["left_hip"],
            landmarks["left_shoulder"],
            landmarks["left_elbow"],
        ),
        "right": calculate_angle(
            landmarks["right_hip"],
            landmarks["right_shoulder"],
            landmarks["right_elbow"],
        ),
    }


def get_average_angle(angles):
    """Average available left/right angles."""
    values = [angle for angle in angles.values() if angle is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def get_active_angle(angles):
    """
    Returns the most-bent visible side.

    This is more responsive for lunges, step-ups, curls, and split squats
    because one side usually works harder than the other.
    """
    values = [angle for angle in angles.values() if angle is not None]
    if not values:
        return None
    return round(min(values), 1)


def get_tracked_angle(angles, strategy="min"):
    """Choose the angle side that best represents the selected exercise."""
    values = [angle for angle in angles.values() if angle is not None]
    if not values:
        return None
    if strategy == "max":
        return round(max(values), 1)
    if strategy == "average":
        return round(sum(values) / len(values), 1)
    return round(min(values), 1)
