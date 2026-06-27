"""Stable rep counting and hold timing for exercise sessions."""

import time


class ExerciseTracker:
    """Tracks reps or hold time from smoothed joint angles."""

    def __init__(self, exercise_name, settings):
        self.exercise_name = exercise_name
        self.settings = settings
        self.reps = 0
        self.stage = "start"
        self.smoothed_angle = None
        self.hold_seconds = 0.0
        self.last_time = time.perf_counter()

    def reset_if_exercise_changed(self, exercise_name, settings):
        if exercise_name != self.exercise_name:
            self.__init__(exercise_name, settings)

    def update(self, angle, is_ready):
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now

        if angle is None or not is_ready:
            if self.stage == "start":
                self.stage = "setup"
            return self.snapshot()

        if self.smoothed_angle is None:
            self.smoothed_angle = angle
        else:
            self.smoothed_angle = (self.smoothed_angle * 0.45) + (angle * 0.55)

        if self.settings["mode"] == "timer":
            if self.settings["hold_min_angle"] <= angle <= self.settings["hold_max_angle"]:
                self.hold_seconds += elapsed
                self.stage = "holding"
            else:
                self.stage = "adjust"
            return self.snapshot()

        down_angle = self.settings["down_angle"]
        up_angle = self.settings["up_angle"]
        count_direction = self.settings.get("count_direction", "return_up")

        if count_direction == "contract":
            if angle > up_angle:
                self.stage = "ready"
            elif self.stage == "ready" and angle < down_angle:
                self.reps += 1
                self.stage = "contracted"
        elif count_direction == "extend":
            if angle < down_angle:
                self.stage = "ready"
            elif self.stage == "ready" and angle > up_angle:
                self.reps += 1
                self.stage = "extended"
        else:
            if angle > up_angle:
                if self.stage == "down":
                    self.reps += 1
                self.stage = "ready"
            elif angle < down_angle and self.stage in ("ready", "start", "setup"):
                self.stage = "down"

        return self.snapshot()

    def snapshot(self):
        return {
            "reps": self.reps,
            "stage": self.stage,
            "angle": None if self.smoothed_angle is None else round(self.smoothed_angle, 1),
            "hold_seconds": round(self.hold_seconds, 1),
        }
