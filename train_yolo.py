''' SCRIPT FOR TRAINING A YOLO MODEL'''
# TOTO NECO DELA!!!
# import yolo
from ultralytics import YOLO

if __name__ == "__main__":
    # load pretrained yolo model
    model = YOLO('yolo11s.pt')

    # train model -- BARO ALWAYS CHANGE NAME FOR A NEW MODEL!!!
    model.train(
        data='C:/Users/spila/OneDrive/Plocha/bara/RAI_project/dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='offroad_yolov11s_v2',  
        pretrained=True,
        multi_scale=True,
        patience=10,
        lr0=0.01,
        lrf=0.01
    )

    # validation
    metrics = model.val()
    print(metrics)


    