import numpy as np
import cv2
import os

frame_width = 640
frame_height = 480

KNOWN_WIDTH = 6.0      # cm
KNOWN_DISTANCE = 40.0  # cm

FOCAL_FILE = "focal_length.npy"


if os.path.exists(FOCAL_FILE):
    FOCAL_LENGTH = np.load(FOCAL_FILE)
    print(f"Ucitan FOCAL_LENGTH: {FOCAL_LENGTH}")
else:
    FOCAL_LENGTH = None

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error opening video stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # CRVENA
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 100])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # ZELENA
    lower_green = np.array([25, 80, 50])
    upper_green = np.array([95, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    key = cv2.waitKey(1) & 0xFF  

    for color_name, mask, box_color in [
        ("RED", mask_red, (0, 0, 255)),
        ("GREEN", mask_green, (0, 255, 0))
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 1000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

                
                if key == ord('c'):
                    FOCAL_LENGTH = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
                    np.save(FOCAL_FILE, FOCAL_LENGTH)
                    print(f"FOCAL LENGTH spremljen: {FOCAL_LENGTH}")

                
                if FOCAL_LENGTH is not None:
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w
                    text = f"{color_name} {distance:.2f} cm"
                else:
                    text = f"{color_name} (Press C to calibrate)"

                cv2.putText(frame,
                            text,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            box_color,
                            2)

    cv2.imshow("ObjectDetection", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()