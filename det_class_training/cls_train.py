from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n-cls.pt")  

    results = model.train(
        data='../../classify_dataset',
        epochs=130,
        imgsz=640,
        batch=16,
        device='0',
        val=True
    )
