''' SCRIPT FOR TRAINING A YOLO MODEL'''
# TOTO NECO DELA!!!
# import yolo
from ultralytics import YOLO

if __name__ == "__main__":
    # load pretrained yolo model
    model = YOLO('yolo11s.pt')

    # train model -- BARO ALWAYS CHANGE NAME FOR A NEW MODEL!!!
    model.train(
        data='C:/Users/238750/bara/01_skul/4-letni/RAI_project/forest obstacle.v3i.yolov11/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='offroad_yolov11s_v1',
        pretrained=True
    )

    # validation
    metrics = model.val()
    print(metrics)


    