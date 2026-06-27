"""AI Fitness Trainer Streamlit app."""

import streamlit as st
import cv2
from pathlib import Path

from camera.webcam import WebcamStream
from pose.angles import (
    get_elbow_angles,
    get_hip_angles,
    get_knee_angles,
    get_shoulder_angles,
    get_tracked_angle,
)
from pose.detector import PoseDetector
from pose.feedback import EXERCISES, assess_position, get_camera_guide
from pose.landmarks import get_landmarks
from pose.tracker import ExerciseTracker

st.set_page_config(page_title="AI Fitness Trainer", layout="centered")
st.title("AI Fitness Trainer")
st.caption("Choose an exercise, start the webcam, and get live form feedback.")


STATUS_COLORS = {
    "good": (30, 170, 70),
    "warning": (255, 180, 40),
    "error": (220, 60, 60),
}
MODEL_PATH = Path("models/pose_landmarker_lite.task")


def draw_feedback_popup(frame_rgb, feedback):
    """Draw a popup-style feedback banner over the video frame."""
    color = STATUS_COLORS[feedback["status"]]
    message = feedback["message"]
    if len(message) > 64:
        message = message[:61] + "..."

    cv2.rectangle(frame_rgb, (16, 16), (frame_rgb.shape[1] - 16, 78), color, -1)
    cv2.putText(
        frame_rgb,
        message,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame_rgb


def draw_tracking_stats(frame_rgb, exercise_name, tracker_state):
    """Draw large tracking stats over the video frame."""
    settings = EXERCISES[exercise_name]
    if settings["mode"] == "timer":
        label = f"Hold: {tracker_state['hold_seconds']:.1f}s"
    else:
        label = f"Reps: {tracker_state['reps']}"

    cv2.putText(
        frame_rgb,
        label,
        (24, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_rgb,
        label,
        (24, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return frame_rgb


exercise_name = st.selectbox("Select exercise", list(EXERCISES.keys()))
st.caption(EXERCISES[exercise_name]["ready_message"])
with st.expander("i Camera guide"):
    st.write(get_camera_guide(exercise_name))
    st.write("Counting starts only after the required joints are visible and your body fits in frame.")

run = st.checkbox("Start / Stop exercise", key="Start camera")
st.caption("Untick this box to stop the exercise and release the webcam.")
frame_placeholder = st.empty()
feedback_placeholder = st.empty()
metric_cols = st.columns(4)

with metric_cols[0]:
    rep_metric = st.empty()
with metric_cols[1]:
    angle_metric = st.empty()
with metric_cols[2]:
    stage_metric = st.empty()
with metric_cols[3]:
    status_metric = st.empty()

if run:
    camera = WebcamStream()
    detector = None

    if not MODEL_PATH.exists():
        st.error(
            "Pose model file is missing. Add pose_landmarker_lite.task "
            "to the models folder, then restart the app."
        )
    elif not camera.open():
        st.error("Could not access webcam. Check camera permissions / index.")
    else:
        detector = PoseDetector(model_path=str(MODEL_PATH))
        tracker = ExerciseTracker(exercise_name, EXERCISES[exercise_name])
        frame_count = 0
        detected_count = 0

        while run:
            tracker.reset_if_exercise_changed(exercise_name, EXERCISES[exercise_name])
            ret, frame = camera.read_frame()
            if not ret:
                st.warning("Failed to grab frame.")
                break

            frame_rgb, results = detector.process(frame)
            frame_count += 1
            if results.pose_landmarks:
                detected_count += 1

            landmarks = get_landmarks(results)
            angle_source = EXERCISES[exercise_name]["angle_source"]
            angle_strategy = EXERCISES[exercise_name]["angle_strategy"]
            if angle_source == "elbow":
                angles = get_elbow_angles(landmarks)
            elif angle_source == "hip":
                angles = get_hip_angles(landmarks)
            elif angle_source == "shoulder":
                angles = get_shoulder_angles(landmarks)
            else:
                angles = get_knee_angles(landmarks)

            primary_angle = get_tracked_angle(angles, angle_strategy)
            feedback = assess_position(exercise_name, landmarks, primary_angle)
            tracker_state = tracker.update(primary_angle, feedback["ready"])

            frame_rgb = detector.draw_overlay(frame_rgb, results)
            frame_rgb = draw_feedback_popup(frame_rgb, feedback)
            frame_rgb = draw_tracking_stats(frame_rgb, exercise_name, tracker_state)

            frame_placeholder.image(frame_rgb, channels="RGB")

            if feedback["status"] == "good":
                feedback_placeholder.success(feedback["message"])
            elif feedback["status"] == "warning":
                feedback_placeholder.warning(feedback["message"])
            else:
                feedback_placeholder.error(feedback["message"])

            if EXERCISES[exercise_name]["mode"] == "timer":
                rep_metric.metric("Wall Sit Time", f"{tracker_state['hold_seconds']:.1f}s")
            else:
                rep_metric.metric("Reps", tracker_state["reps"])

            angle_value = tracker_state["angle"] or feedback["angle"]
            angle_label = f"{angle_source.title()} Angle"
            angle_metric.metric(
                angle_label,
                "--" if angle_value is None else f"{angle_value} deg",
            )
            if EXERCISES[exercise_name]["mode"] == "timer":
                stage_metric.metric("Position", "Holding" if tracker_state["stage"] == "holding" else "Adjust")
            else:
                stage_metric.metric("Stage", tracker_state["stage"].title())
            status_metric.metric("Detected Frames", f"{detected_count}/{frame_count}")

            run = st.session_state.get("Start camera", run)

        camera.release()
    if detector is not None:
        detector.close()
else:
    st.info("Tick the box above to start your webcam.")
