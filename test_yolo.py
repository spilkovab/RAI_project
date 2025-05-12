# validate
from ultralytics import YOLO

# Load the model
model = YOLO("C:/Users/spila/OneDrive/Plocha/bara/RAI_project/runs/detect/offroad_yolov11s_v12/weights/best.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # mAP50-95