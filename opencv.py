import argparse

import cv2

from car.config import FRAME_HEIGHT, FRAME_WIDTH, STEERING_GAIN
from car.fsm import CarStateMachine
from car.logger import StateLogger
from car.motor import MotorController
from car.vision import find_lane_center, obstacle_detected, obstacle_masks


def parse_args():
    parser = argparse.ArgumentParser(description="RUTH lane-following + obstacle avoidance")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file (e.g. videos/video1.mp4)",
    )
    parser.add_argument(
        "--no-motor",
        action="store_true",
        help="Skip sending commands over serial, even if a port is configured",
    )
    return parser.parse_args()


def open_source(source):
    # A plain integer string means "camera index"; anything else is a file path.
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main():
    args = parse_args()

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"Could not open video source: {args.source}")
        return

    fsm = CarStateMachine()
    motor = None if args.no_motor else MotorController()
    logger = StateLogger()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lane_center = find_lane_center(frame)

            mask_red, mask_green = obstacle_masks(hsv)
            if obstacle_detected(mask_red):
                fsm.notify_obstacle("RED")
            if obstacle_detected(mask_green):
                fsm.notify_obstacle("GREEN")

            frame_center = FRAME_WIDTH // 2
            error = lane_center - frame_center
            steering = error * STEERING_GAIN

            command = fsm.command()

            if motor is not None:
                motor.send(command)

            logger.log(fsm.state, command, lane_center, steering)

            cv2.line(frame, (frame_center, 0), (frame_center, FRAME_HEIGHT), (255, 0, 0), 1)
            cv2.line(frame, (lane_center, 0), (lane_center, FRAME_HEIGHT), (0, 255, 0), 1)

            cv2.putText(
                frame,
                f"STATE: {fsm.state}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"CMD: {command}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("WRO CAR", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        if motor is not None:
            motor.close()


if __name__ == "__main__":
    main()
