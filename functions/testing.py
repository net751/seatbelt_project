import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('runs/classify/train14/weights/best.pt')


video_path = '../../VideosForTesting/2person_cut/person_passanger_ex2.mp4'

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = '../../TestingVideos/classification/person_passanger_ex2_14_classification.mp4'
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    try:
        top_class_index = results[0].probs.top1 
        top_class = results[0].names[top_class_index] 
        confidence = results[0].probs.top1conf.item()  

        message = f"{top_class}: {confidence:.2f}"

        cv2.putText(frame, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    except AttributeError:
        cv2.putText(frame, "Detection Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    out.write(frame)

    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_path}")
