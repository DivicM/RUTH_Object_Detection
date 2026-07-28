from car.config import OBSTACLE_COOLDOWN, TURN_TIME
from car.fsm import AVOID_GREEN, AVOID_RED, FOLLOW_LANE, CarStateMachine


def test_starts_in_follow_lane():
    fsm = CarStateMachine()
    assert fsm.state == FOLLOW_LANE
    assert fsm.command(now=0) == "FORWARD"


def test_red_obstacle_triggers_right_turn():
    fsm = CarStateMachine()
    fsm.notify_obstacle("RED", now=10.0)

    assert fsm.state == AVOID_RED
    assert fsm.command(now=10.1) == "RIGHT"


def test_green_obstacle_triggers_left_turn():
    fsm = CarStateMachine()
    fsm.notify_obstacle("GREEN", now=10.0)

    assert fsm.state == AVOID_GREEN
    assert fsm.command(now=10.1) == "LEFT"


def test_returns_to_follow_lane_after_turn_time():
    fsm = CarStateMachine()
    fsm.notify_obstacle("RED", now=10.0)

    assert fsm.command(now=10.0 + TURN_TIME + 0.01) == "FORWARD"
    assert fsm.state == FOLLOW_LANE


def test_cooldown_ignores_new_obstacle_too_soon():
    fsm = CarStateMachine()
    fsm.notify_obstacle("RED", now=10.0)
    fsm.notify_obstacle("GREEN", now=10.0 + OBSTACLE_COOLDOWN - 0.1)

    assert fsm.state == AVOID_RED


def test_new_obstacle_accepted_after_cooldown():
    fsm = CarStateMachine()
    fsm.notify_obstacle("RED", now=10.0)
    fsm.notify_obstacle("GREEN", now=10.0 + OBSTACLE_COOLDOWN + 0.1)

    assert fsm.state == AVOID_GREEN
