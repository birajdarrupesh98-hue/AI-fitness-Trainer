"""Exercise selection and basic position feedback rules."""

EXERCISES = {
    "Wall Sit": {
        "mode": "timer",
        "angle_source": "knee",
        "angle_strategy": "min",
        "ready_message": "Stand side-on near a wall with hips, knees, and ankles visible.",
        "start_message": "Ready. Slide into the wall sit to start the timer.",
        "target_angle": 115,
        "max_ready_angle": 170,
        "hold_min_angle": 70,
        "hold_max_angle": 120,
        "good_message": "Good wall sit position. Hold steady.",
        "shallow_message": "Slide lower until your knees are close to a right angle.",
    },
    "Biceps Curl": {
        "mode": "counter",
        "angle_source": "elbow",
        "angle_strategy": "min",
        "count_direction": "contract",
        "ready_message": "Face the camera with shoulders, elbows, and wrists visible.",
        "start_message": "Ready. Curl up to count one rep.",
        "target_angle": 65,
        "max_ready_angle": 155,
        "down_angle": 75,
        "up_angle": 145,
        "good_message": "Good curl. Keep your elbows close to your body.",
        "shallow_message": "Curl higher until your elbow closes more.",
    },
    "Sit-ups": {
        "mode": "counter",
        "angle_source": "hip",
        "angle_strategy": "min",
        "count_direction": "contract",
        "ready_message": "Lie side-on to the camera with shoulders, hips, and knees visible.",
        "start_message": "Ready. Sit up until your torso rises to count.",
        "target_angle": 105,
        "max_ready_angle": 150,
        "down_angle": 105,
        "up_angle": 145,
        "good_message": "Good sit-up. Lower back with control.",
        "shallow_message": "Bring your chest closer to your knees.",
    },
    "Shoulder Press": {
        "mode": "counter",
        "angle_source": "elbow",
        "angle_strategy": "max",
        "count_direction": "extend",
        "ready_message": "Face the camera with shoulders, elbows, and wrists visible.",
        "start_message": "Ready. Press overhead to count one rep.",
        "target_angle": 155,
        "max_ready_angle": 95,
        "down_angle": 105,
        "up_angle": 155,
        "good_message": "Good press. Finish with arms overhead.",
        "shallow_message": "Press higher until your elbows extend.",
    },
    "Lateral Raise": {
        "mode": "counter",
        "angle_source": "shoulder",
        "angle_strategy": "max",
        "count_direction": "extend",
        "ready_message": "Face the camera with arms by your sides and shoulders visible.",
        "start_message": "Ready. Raise arms out to the sides to count.",
        "target_angle": 80,
        "max_ready_angle": 35,
        "down_angle": 35,
        "up_angle": 80,
        "good_message": "Good raise. Keep shoulders relaxed.",
        "shallow_message": "Raise your arms closer to shoulder height.",
    },
    "Front Raise": {
        "mode": "counter",
        "angle_source": "shoulder",
        "angle_strategy": "max",
        "count_direction": "extend",
        "ready_message": "Stand side-on or slightly angled with shoulders, hips, and elbows visible.",
        "start_message": "Ready. Raise arms forward to count.",
        "target_angle": 80,
        "max_ready_angle": 35,
        "down_angle": 35,
        "up_angle": 80,
        "good_message": "Good front raise. Stop around shoulder height.",
        "shallow_message": "Raise your arms higher toward shoulder height.",
    },
}

CAMERA_GUIDES = {
    "knee": "Full lower body visible: hips, knees, and ankles in frame; stand 6-8 feet away.",
    "elbow": "Upper body visible: shoulders, elbows, and wrists in frame; camera around chest height.",
    "hip": "Side view works best: shoulders, hips, and knees visible while lying down.",
    "shoulder": "Upper body visible: shoulders, elbows, wrists, and hips in frame; avoid standing too close.",
}

REQUIRED_JOINTS = {
    "knee": (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
    "elbow": (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ),
    "hip": (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    ),
    "shoulder": (
        "left_hip",
        "right_hip",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
    ),
}


def get_required_joints(exercise_name):
    return REQUIRED_JOINTS[EXERCISES[exercise_name]["angle_source"]]


def get_camera_guide(exercise_name):
    return CAMERA_GUIDES[EXERCISES[exercise_name]["angle_source"]]


def assess_position(exercise_name, landmarks, primary_angle):
    """
    Returns a dict used by the UI:
      - status: "good", "warning", or "error"
      - message: short form cue
      - angle: current primary angle
      - ready: whether counting/timing may start
    """
    if landmarks is None:
        return {
            "status": "error",
            "message": "No pose detected. Step fully into the camera view.",
            "angle": None,
            "ready": False,
        }

    settings = EXERCISES[exercise_name]
    low_visibility = [
        joint_name
        for joint_name in get_required_joints(exercise_name)
        if landmarks[joint_name].get("visibility", 0) < 0.5
    ]
    if low_visibility:
        return {
            "status": "error",
            "message": "Move until the needed joints are clearly visible.",
            "angle": primary_angle,
            "ready": False,
        }

    body_box = landmarks.get("_body_box", {})
    if body_box.get("height", 0) > 0.94 or body_box.get("width", 0) > 0.94:
        return {
            "status": "error",
            "message": "Move back a little so your body fits inside the frame.",
            "angle": primary_angle,
            "ready": False,
        }

    if primary_angle is None:
        return {
            "status": "error",
            "message": "Cannot read joint angle. Turn slightly and keep joints visible.",
            "angle": None,
            "ready": False,
        }

    if settings["mode"] == "timer":
        if settings["hold_min_angle"] <= primary_angle <= settings["hold_max_angle"]:
            return {
                "status": "good",
                "message": settings["good_message"],
                "angle": primary_angle,
                "ready": True,
            }
        return {
            "status": "warning",
            "message": settings["shallow_message"],
            "angle": primary_angle,
            "ready": True,
        }

    if settings.get("count_direction") == "extend":
        if primary_angle < settings["down_angle"]:
            return {
                "status": "good",
                "message": settings["start_message"],
                "angle": primary_angle,
                "ready": True,
            }
        if primary_angle > settings["target_angle"]:
            return {
                "status": "good",
                "message": settings["good_message"],
                "angle": primary_angle,
                "ready": True,
            }
        return {
            "status": "warning",
            "message": settings["shallow_message"],
            "angle": primary_angle,
            "ready": True,
        }

    if primary_angle > settings["max_ready_angle"]:
        return {
            "status": "good",
            "message": settings["start_message"],
            "angle": primary_angle,
            "ready": True,
        }

    if primary_angle < settings["target_angle"]:
        return {
            "status": "good",
            "message": settings["good_message"],
            "angle": primary_angle,
            "ready": True,
        }

    return {
        "status": "warning",
        "message": settings["shallow_message"],
        "angle": primary_angle,
        "ready": True,
    }
