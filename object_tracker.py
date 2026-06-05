import cv2
from ultralytics import YOLO

def run_tracker(source=0):
    """
    Runs real-time object detection and tracking with HIGH ACCURACY.
    Args:
        source: 0 for webcam, or a string path to a video file (e.g., 'video.mp4')
    """
    # Load the YOLOv8 Medium model for much higher accuracy.
    print("Loading highly accurate YOLOv8 Medium model...")
    model = YOLO("yolov8m.pt") 

    # Force DirectShow (CAP_DSHOW) if using a live webcam on Windows to prevent MSMF crashes
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or failed to read frame.")
            break

        # Added `conf=0.50` to ignore weak/wrong predictions.
        # The tracker will only register objects the model is 50%+ confident about.
        results = model.track(source=frame, persist=True, conf=0.50, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  
            class_ids = results[0].boxes.cls.int().cpu().tolist()  
            track_ids = results[0].boxes.id.int().cpu().tolist()  
            confidences = results[0].boxes.conf.float().cpu().tolist() 

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                x1, y1, x2, y2 = box
                label = model.names[class_id]
                
                text = f"{label} ID: {track_id} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(text) * 12, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("High-Accuracy Object Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker(source=0)