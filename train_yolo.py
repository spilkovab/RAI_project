''' SCRIPT FOR TRAINING A YOLO MODEL'''
# import yolo
from ultralytics import YOLO

if __name__ == "__main__":
    # load pretrained yolo model
    model = YOLO('yolo11n.pt')

    # train model -- ALWAYS CHANGE NAME FOR A NEW MODEL!!!
    model.train(
        data='/home/student/Desktop/spilkova/RAI_project/dataset/data.yaml',
        epochs=150,
        imgsz=640,
        batch=8,
        name='offroad_yolov11n_v1',  
        pretrained=True,
        multi_scale=True,
        patience=10,
        lr0=0.01,
        lrf=0.01
    )

    # validation
    metrics = model.val()
    print(metrics)


