from ultralytics import YOLO
import numpy as np 

model=YOLO('yolov8n.pt', "v8")

detection_output=model.predict(source="./images/car.jpg", conf=0.5, save=True)

print(detection_output)

print(detection_output[0].numpy())