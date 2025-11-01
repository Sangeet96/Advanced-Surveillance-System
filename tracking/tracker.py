from norfair import Detection, Tracker
import numpy as np


def euclidean(detection, tracked_object):
    """
    Compute Euclidean distance between a Detection and a TrackedObject.
    Compatible with all Norfair versions.
    """
    det_point = detection.points[0]
    obj_point = tracked_object.estimate[0]
    return np.linalg.norm(det_point - obj_point)


class MultiTracker:
    def __init__(self):
        # Basic tracker setup
        self.tracker = Tracker(
            distance_function=euclidean,
            distance_threshold=25
        )

    def update(self, boxes):
        # Convert YOLO boxes into detection points
        detections = [
            Detection(np.array([[ (x1 + x2) // 2, (y1 + y2) // 2 ]]))
            for (x1, y1, x2, y2) in boxes
        ]

        tracked_objects = self.tracker.update(detections)
        tracked_boxes = []

        for t in tracked_objects:
            point = t.estimate[0]
            tracked_boxes.append((int(point[0]), int(point[1]), t.id))

        return tracked_boxes
