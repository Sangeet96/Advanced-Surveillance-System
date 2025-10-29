from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  # Auto-downloads if not found

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2))
        return boxes
