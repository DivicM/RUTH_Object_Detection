import cv2

from car.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LANE_BLUR_KERNEL,
    LANE_ROI_START_RATIO,
    LANE_THRESHOLD,
    LOWER_GREEN,
    LOWER_RED1,
    LOWER_RED2,
    OBSTACLE_MIN_AREA,
    OBSTACLE_MORPH_KERNEL,
    UPPER_GREEN,
    UPPER_RED1,
    UPPER_RED2,
)


def find_lane_center(frame):
    roi = frame[int(FRAME_HEIGHT * LANE_ROI_START_RATIO):FRAME_HEIGHT, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, LANE_BLUR_KERNEL, 0)

    _, thresh = cv2.threshold(blur, LANE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lane_center = FRAME_WIDTH // 2

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        lane_center = x + w // 2
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return lane_center


def obstacle_masks(hsv_frame):
    mask_red = cv2.inRange(hsv_frame, LOWER_RED1, UPPER_RED1) | cv2.inRange(
        hsv_frame, LOWER_RED2, UPPER_RED2
    )
    mask_green = cv2.inRange(hsv_frame, LOWER_GREEN, UPPER_GREEN)

    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, OBSTACLE_MORPH_KERNEL)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, OBSTACLE_MORPH_KERNEL)

    return mask_red, mask_green


def obstacle_detected(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) >= OBSTACLE_MIN_AREA:
            return True

    return False
