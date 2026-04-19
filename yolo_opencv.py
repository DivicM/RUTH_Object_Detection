import numpy as np  
import cv2
from ultralytics import YOLO   
import random


#model = YOLO('yolov8n.pt', "v8")

frame_width = 640
frame_height = 480

file=open("utils/coco.txt", "r")

data=file.read()

class_list=data.split("\n")
file.close()

detection_colors = []
for i in range(len(class_list)):
    r=random.randint(0, 255)
    g=random.randint(0, 255)
    b=random.randint(0, 255)
    detection_colors.append((r, g, b))


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("./videos/video1.mp4")


if not cap.isOpened():
    print("Error opening video stream or file")
    exit()


while True:
    ret, frame = cap.read()


    if not ret:
        print("Cant recieve frame")
        break

    # cv2.imwrite("images/output/frame.jpg", frame)

    # detect_params=model.predict(source="images/output/frame.jpg", conf=0.45, save=False)

    # print(detect_params[0].numpy())
    # detect_params=detect_params[0].numpy()

    # if len(detect_params) != 0:
    #     for result in detect_params:
    #         boxes = result.boxes
    #         for box in boxes:
    #             coords = box.xyxy[0].tolist()
    #             conf = float(box.conf[0])
    #             cls = int(box.cls[0])

    #             x1, y1, x2, y2 = map(int, coords)

    #             # Crtanje pravokutnika
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), detection_colors[cls], 3)

    #             # Pisanje teksta
    #             label = f"{class_list[cls]} {conf:.2f}"
    #             font = cv2.FONT_HERSHEY_COMPLEX
    #             cv2.putText(frame, label, (x1, y1 - 10), font, 1, (255, 255, 255), 2)

    # Detekcija crvenih i zelenih objekata koristeći HSV boju
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([25, 80, 50])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

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
                cv2.putText(frame, f"{color_name} OBJECT", (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, box_color, 2)

    cv2.imshow("ObjectDetection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()