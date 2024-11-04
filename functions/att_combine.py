import os
import cv2
import numpy as np
from ultralytics import YOLO

pose_model = YOLO('../yolov8x-pose.pt') 

classification_model = YOLO('runs/classify/train14/weights/best.pt')

#video_path = 'detec.MOV'
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_path = '../../VideosForTesting/2person_cut/person_passanger_ex2.mp4'

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = '../../TestingVideos/with_blue_lines/passanger_ex2_all.mp4'
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

pose_color = (0, 0, 255)  
bbox_color = (0, 255, 0)  
split_color = (255, 0, 0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pose_results = pose_model(frame)

    for r in pose_results:
        if r.keypoints.data.shape[0] > 0:
            keypoints = r.keypoints.data[0].clone().cpu().numpy()

            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            hip = keypoints[11]

            x_min = min(left_shoulder[0], right_shoulder[0])
            x_max = max(left_shoulder[0], right_shoulder[0])
            y_min = min(left_shoulder[1], right_shoulder[1])
            y_max = hip[1]

            img_height, img_width, _ = frame.shape
            if y_max < img_height * 0.9:
                y_max = img_height

            x_min = max(0, int(x_min - 80))
            x_max = min(img_width, int(x_max + 80))
            y_min = max(0, int(y_min - 50))
            y_max = min(img_height, int(y_max))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 2)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size > 0:
                class_results = classification_model(roi)

                try:
                    top_class_index = class_results[0].probs.top1 
                    top_class = class_results[0].names[top_class_index] 
                    confidence = class_results[0].probs.top1conf.item()  

                    message = f"{top_class}: {confidence:.2f}"
                    cv2.putText(frame, message, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                except AttributeError:
                    cv2.putText(frame, "Detection Error", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            for point in keypoints:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 5, pose_color, -1)

            num_lines = 5
            for i in range(1, num_lines + 1):
                start_x = x_min + (i * (x_max - x_min)) // (num_lines + 1)
                end_y = y_max - (i * (y_max - y_min)) // (num_lines + 1)
                cv2.line(frame, (start_x, y_min), (x_max, end_y), split_color, 2)

            for i in range(1, num_lines + 1):
                start_x = x_max - (i * (x_max - x_min)) // (num_lines + 1)
                end_y = y_max - (i * (y_max - y_min)) // (num_lines + 1)
                cv2.line(frame, (start_x, y_max), (x_min, end_y), split_color, 2)

    out.write(frame)

    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_path}")
