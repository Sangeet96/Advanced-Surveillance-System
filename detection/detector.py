# from ultralytics import YOLO
# import numpy as np

# class Detector:
#     def __init__(self, model_path="yolov8n.pt"):
#         self.model = YOLO(model_path)  # Auto-downloads if not found

#     def detect(self, frame):
#         results = self.model(frame, verbose=False)
#         boxes = []
#         for box in results[0].boxes:
#             cls_id = int(box.cls[0])
#             if cls_id == 0:  # 0 = person
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#                 boxes.append((x1, y1, x2, y2))
#         return boxes


from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        resized = frame.copy()
        results = self.model(resized, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls = int(box.cls)
            if cls == 0:  # only 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes
