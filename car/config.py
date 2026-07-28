import numpy as np

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Lane detection
LANE_ROI_START_RATIO = 0.6  # bottom 40% of the frame
LANE_BLUR_KERNEL = (5, 5)
LANE_THRESHOLD = 90

# Obstacle detection (HSV ranges)
LOWER_RED1 = np.array([0, 120, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 100])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_GREEN = np.array([25, 80, 50])
UPPER_GREEN = np.array([95, 255, 255])

OBSTACLE_MORPH_KERNEL = np.ones((5, 5), np.uint8)
OBSTACLE_MIN_AREA = 4000
OBSTACLE_COOLDOWN = 2.0  # seconds between obstacle re-triggers

# Steering
STEERING_GAIN = 0.01
TURN_TIME = 0.5  # seconds spent turning to avoid an obstacle

# Serial / motor output
SERIAL_PORT = "COM3"
SERIAL_BAUDRATE = 9600

# Logging
LOG_FILE = "state_log.csv"
