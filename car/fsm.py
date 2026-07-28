import time

from car.config import OBSTACLE_COOLDOWN, TURN_TIME

FOLLOW_LANE = "FOLLOW_LANE"
AVOID_RED = "AVOID_RED"
AVOID_GREEN = "AVOID_GREEN"


class CarStateMachine:
    def __init__(self):
        self.state = FOLLOW_LANE
        self.last_obstacle_time = 0.0
        self.turn_start = 0.0

    def notify_obstacle(self, color, now=None):
        now = time.time() if now is None else now

        if now - self.last_obstacle_time < OBSTACLE_COOLDOWN:
            return

        if color == "RED":
            self.state = AVOID_RED
        elif color == "GREEN":
            self.state = AVOID_GREEN
        else:
            return

        self.last_obstacle_time = now
        self.turn_start = now

    def command(self, now=None):
        now = time.time() if now is None else now

        if self.state == AVOID_RED:
            if now - self.turn_start > TURN_TIME:
                self.state = FOLLOW_LANE
                return "FORWARD"
            return "RIGHT"

        if self.state == AVOID_GREEN:
            if now - self.turn_start > TURN_TIME:
                self.state = FOLLOW_LANE
                return "FORWARD"
            return "LEFT"

        return "FORWARD"
