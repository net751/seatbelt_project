from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8m.yaml")

    results = model.train(
        data='config.yaml',  
        epochs=100,            
        imgsz=640,             
        batch=16,             
        device='0',           
        flipud=0.5,            
        fliplr=0.5,            
        mosaic=1.0,            
        mixup=0.1,              
        hsv_h=0.015,         
        hsv_s=0.7,             
        hsv_v=0.4               
    )
