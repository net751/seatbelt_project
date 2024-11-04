import cv2
from ultralytics import YOLO
import numpy as np

#model = YOLO('../bc_prace_code/runs/detect/train10/weights/best.pt') 
model = YOLO('C:\\Users\\Netopus\\Desktop\\FatimaDataset\\bc_prace_code\\runs\\detect\\train10\\weights\\best.pt')

video_path = '../../VideosForTesting/detec2.MOV'
output_path = '../../TestingVideos/detection/detec_2_detection.mp4'

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0]) 
        confidence = detection.conf.item()
        class_id = detection.cls.item()
        class_name = results[0].names[class_id]

        label = f"{class_name}: {confidence:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)

    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_path}")
