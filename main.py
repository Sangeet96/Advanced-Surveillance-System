import cv2
from detection.detector import Detector
from tracking.tracker import MultiTracker
from control.servo_controller import ServoController
from privacy.face_blur import FaceBlur
import time
start_time = time.time()

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

cap = cv2.VideoCapture("videos/test_video.mp4")  # use 0 for live webcam
detector = Detector()
tracker = MultiTracker()
servo = ServoController()
face_blur = FaceBlur()

frame_count = 0
DETECT_EVERY_N_FRAMES = 10  # adjust for speed/accuracy trade-off

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - start_time)
        print(f"FPS: {fps:.1f}")
        start_time = time.time()

    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        boxes = detector.detect(frame)
        tracked = tracker.update(boxes)
    else:
        tracked = tracker.tracker.update([])  # predict next positions

    if tracked:
        # choose target with largest bounding box (closest person)
        target = tracked[0]
        target_x, target_y, target_id = target
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        servo.update(target_x, target_y, center_x, center_y)

        # optionally blur other faces
        frame = face_blur.blur_faces(frame)

        cv2.circle(frame, (target_x, target_y), 6, (0, 255, 0), -1)
        cv2.putText(frame, f"Tracking ID: {target_id}",
                    (target_x - 50, target_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Advanced Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
