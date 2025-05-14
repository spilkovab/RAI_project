# validate
from ultralytics import YOLO

# Load the model
model = YOLO("/home/student/Desktop/spilkova/RAI_project/runs/detect/offroad_yolov11s_v2/weights/best.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # mAP50-95