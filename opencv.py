import numpy as np
import cv2
import time

frame_width = 320
frame_height = 240


FOLLOW_LANE = "FOLLOW_LANE"
AVOID_RED = "AVOID_RED"
AVOID_GREEN = "AVOID_GREEN"

state = FOLLOW_LANE

last_obstacle_time = 0
OBSTACLE_COOLDOWN = 2.0  # sekunde

turn_start = 0
TURN_TIME = 0.5

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    roi = frame[int(frame_height*0.6):frame_height, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lane_center = frame_width // 2

    if len(contours) > 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)

        lane_center = x + w // 2

        cv2.rectangle(roi, (x,y), (x+w,y+h), (255,255,255), 2)

    lower_red1 = np.array([0,120,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,120,100])
    upper_red2 = np.array([180,255,255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([25,80,50])
    upper_green = np.array([95,255,255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)


    def detect_obstacle(mask, color):
        global state, last_obstacle_time, turn_start

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 4000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2

            # debounce (ne reagira stalno)
            if time.time() - last_obstacle_time < OBSTACLE_COOLDOWN:
                return

            last_obstacle_time = time.time()

            if color == "RED":
                state = AVOID_RED
                turn_start = time.time()

            elif color == "GREEN":
                state = AVOID_GREEN
                turn_start = time.time()

    detect_obstacle(mask_red, "RED")
    detect_obstacle(mask_green, "GREEN")
    
    frame_center = frame_width // 2
    error = lane_center - frame_center

    steering = error * 0.01

    command = "FORWARD"

    if state == FOLLOW_LANE:
        command = "FORWARD"

    elif state == AVOID_RED:
        command = "RIGHT"

        if time.time() - turn_start > TURN_TIME:
            state = FOLLOW_LANE

    elif state == AVOID_GREEN:
        command = "LEFT"

        if time.time() - turn_start > TURN_TIME:
            state = FOLLOW_LANE

    cv2.line(frame, (frame_center, 0), (frame_center, frame_height), (255,0,0), 1)
    cv2.line(frame, (lane_center, 0), (lane_center, frame_height), (0,255,0), 1)

    cv2.putText(frame,
                f"STATE: {state}",
                (10,20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2)

    cv2.putText(frame,
                f"CMD: {command}",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2)

    cv2.imshow("WRO CAR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()