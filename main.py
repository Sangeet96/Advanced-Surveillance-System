import cv2
import numpy as np
import time

# ------------------- Load Model -------------------
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# ------------------- Video Source -------------------
cap = cv2.VideoCapture("videos/test_video.mp4")  # use 0 for webcam
cap.set(cv2.CAP_PROP_FPS, 30)

frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x, center_y = frame_w // 2, frame_h // 2

# ------------------- Simulation Params -------------------
servo_x, servo_y = 0.0, 0.0
sensitivity = 0.002
MIN_MOVE = 0.02  # print movement only if delta > this

tracker = None
tracking = False
frame_count = 0
DETECT_EVERY_N_FRAMES = 15  # run detection every N frames

fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # ---------- Run detection periodically ----------
    if (not tracking) or (frame_count % DETECT_EVERY_N_FRAMES == 0):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        max_conf, person_box = 0, None
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person" and conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")
                if conf > max_conf:
                    max_conf = conf
                    person_box = (sx, sy, ex - sx, ey - sy)

        if person_box:
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, person_box)
            tracking = True
        else:
            tracking = False

    elif tracking:
        ok, bbox = tracker.update(frame)
        if ok:
            (x, y, bw, bh) = [int(v) for v in bbox]
            px, py = x + bw // 2, y + bh // 2
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            dx, dy = px - center_x, py - center_y
            delta_x = -dx * sensitivity
            delta_y = dy * sensitivity

            servo_x = np.clip(servo_x + delta_x, -1.0, 1.0)
            servo_y = np.clip(servo_y + delta_y, -1.0, 1.0)

            # -------- Print movement to terminal --------
            if abs(delta_x) > MIN_MOVE or abs(delta_y) > MIN_MOVE:
                move_x = "Right" if delta_x > 0 else "Left"
                move_y = "Up" if delta_y > 0 else "Down"
                print(f"ServoX → {move_x} ({delta_x:+.3f}) | ServoY → {move_y} ({delta_y:+.3f})")

            cv2.putText(frame, f"ServoX:{servo_x:.2f} ServoY:{servo_y:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            tracking = False

    cv2.drawMarker(frame, (center_x, center_y), (255, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)

    # FPS monitor (optional)
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Surveillance System", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
