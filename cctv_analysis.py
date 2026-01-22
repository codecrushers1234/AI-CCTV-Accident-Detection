from ultralytics import YOLO
import cv2
import time

# Load AI model
model = YOLO("yolov8n.pt")

# CCTV video (for hackathon use recorded video)
cap = cv2.VideoCapture("cctv_video.mp4")

prev_positions = {}
speed_threshold = 40   # for rash driving

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:  # vehicles only
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if cls in prev_positions:
                prev_x, prev_y = prev_positions[cls]
                speed = abs(center_x - prev_x) + abs(center_y - prev_y)

                if speed > speed_threshold:
                    cv2.putText(frame, "RASH DRIVING",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

            prev_positions[cls] = (center_x, center_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("CCTV Analysis", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Analysis completed")
