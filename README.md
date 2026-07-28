# RUTH Object Detection

Computer-vision driving logic for an autonomous robot car (WRO-style), using OpenCV for lane following and colored-obstacle avoidance from a camera feed or video file.

## How it works

The entry point ([opencv.py](opencv.py)) runs a finite state machine (FSM) on each frame, built from the [car/](car/) package:

1. **Lane tracking** ([car/vision.py](car/vision.py)) — the bottom portion of the frame is converted to grayscale, blurred, and thresholded to find the lane contour. The center of the largest contour is tracked as the lane center, and the offset from the frame center gives a steering error.
2. **Obstacle detection** ([car/vision.py](car/vision.py)) — the frame is converted to HSV and masked to find red and green regions. Sufficiently large contours are treated as obstacles.
3. **State machine** ([car/fsm.py](car/fsm.py)) — based on detections, the car switches between states:
   - `FOLLOW_LANE` — drive forward, following the lane center.
   - `AVOID_RED` — turn right for a fixed duration.
   - `AVOID_GREEN` — turn left for a fixed duration.

   A cooldown timer prevents the obstacle detection from re-triggering repeatedly on the same object.
4. **Motor output** ([car/motor.py](car/motor.py)) — the computed command (`FORWARD` / `LEFT` / `RIGHT`) is sent to an Arduino over serial. If `pyserial` isn't installed or the configured port can't be opened, this becomes a no-op so the rest of the pipeline still runs (e.g. for testing without hardware attached).
5. **Logging** ([car/logger.py](car/logger.py)) — each frame's timestamp, state, command, lane center, and steering value are appended to `state_log.csv` for later debugging/analysis.

The current state, computed command, lane center, and frame center are also overlaid on the video window for live debugging.

All tunable constants (frame size, HSV ranges, thresholds, timings, serial port) live in [car/config.py](car/config.py).

## Requirements

- Python 3
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [NumPy](https://pypi.org/project/numpy/)
- [pyserial](https://pypi.org/project/pyserial/) (optional — only needed to actually send commands to an Arduino)
- [pytest](https://pypi.org/project/pytest/) (optional — only needed to run tests)

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Run against a live webcam (default is camera index `0`):

```bash
python opencv.py
```

Run against a video file instead (useful for testing without a robot/camera):

```bash
python opencv.py --source videos/video1.mp4
```

Run without sending anything over serial (e.g. no Arduino attached):

```bash
python opencv.py --no-motor
```

Press `q` in the video window to quit.

### Motor / serial output

By default, commands are sent to the Arduino configured by `SERIAL_PORT` / `SERIAL_BAUDRATE` in [car/config.py](car/config.py). If the port can't be opened (or `pyserial` isn't installed), `MotorController` silently falls back to a no-op — no error, no crash — so the vision/FSM pipeline can still be developed and tested on a machine with no hardware attached.

## Tests

The FSM ([car/fsm.py](car/fsm.py)) has no OpenCV dependency and is covered by unit tests in [tests/test_fsm.py](tests/test_fsm.py):

```bash
pytest tests/ -v
```

## Project structure

```
opencv.py             Entry point: capture loop wiring vision + FSM + motor + logging together
car/
  config.py           All tunable constants (frame size, HSV ranges, timings, serial port, log file)
  fsm.py               CarStateMachine — pure state machine, no OpenCV dependency
  vision.py            Lane detection and obstacle (red/green) detection
  motor.py             Serial output to an Arduino, with a safe no-op fallback
  logger.py            CSV logging of state/command history
tests/
  test_fsm.py          Unit tests for CarStateMachine
images/                Sample images used during development/testing
videos/                Sample video(s) used during development/testing
utils/coco.txt         COCO class labels (leftover from an earlier YOLO-based approach)
focal_length.npy       Saved focal length value (leftover from earlier distance-estimation experiments)
```

## Notes

This project previously used a YOLO model for object detection; it was replaced with a lighter, color-based HSV thresholding approach better suited for real-time performance on constrained hardware. Some files (`utils/coco.txt`, `focal_length.npy`) remain from that earlier iteration.
