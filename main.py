import cv2
import threading
import queue
import time

from detection.detector import YOLODetector
from tracking.tracker import MultiTracker
from control.servo_controller import ServoController
from privacy.face_blur import blur_faces

# ---------------- QUEUES ---------------- #
frame_queue = queue.Queue(maxsize=2)
target_queue = queue.Queue(maxsize=2)

# ---------------- THREAD 1: FRAME CAPTURE ---------------- #
class FrameCaptureThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not frame_queue.full():
                frame_queue.put(frame)
        self.cap.release()

    def stop(self):
        self.running = False


# ---------------- THREAD 2: DETECTION + TRACKING ---------------- #
class DetectionThread(threading.Thread):
    def __init__(self, detector, tracker):
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        self.running = True

    def run(self):
        while self.running:
            if not frame_queue.empty():
                frame = frame_queue.get()
                boxes = self.detector.detect(frame)
                tracked = self.tracker.update(boxes)

                # pick main target (largest or center-most person)
                if tracked:
                    target = max(tracked, key=lambda t: t[0])
                    if not target_queue.full():
                        target_queue.put(target)

                # show bounding boxes and IDs
                for (x, y, tid) in tracked:
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {tid}", (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                frame = blur_faces(frame)
                cv2.imshow("Advanced Surveillance System", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

    def stop(self):
        self.running = False


# ---------------- THREAD 3: SERVO CONTROL ---------------- #
class ServoThread(threading.Thread):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True

    def run(self):
        while self.running:
            if not target_queue.empty():
                target = target_queue.get()
                self.controller.update(target)
            time.sleep(0.02)  # smoother servo response

    def stop(self):
        self.running = False


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    detector = YOLODetector()
    tracker = MultiTracker()
    controller = ServoController()

    capture_thread = FrameCaptureThread(src=0)  # 0 for webcam or path to video
    detection_thread = DetectionThread(detector, tracker)
    servo_thread = ServoThread(controller)

    capture_thread.start()
    detection_thread.start()
    servo_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        capture_thread.stop()
        detection_thread.stop()
        servo_thread.stop()
        cv2.destroyAllWindows()
