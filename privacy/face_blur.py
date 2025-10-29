import cv2

class FaceBlur:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def blur_faces(self, frame, target_box=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            if not self._is_inside_target(x, y, w, h, target_box):
                face_region = frame[y:y+h, x:x+w]
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_region, (99, 99), 30)
        return frame

    def _is_inside_target(self, x, y, w, h, target_box):
        if target_box is None:
            return False
        (tx1, ty1, tx2, ty2) = target_box
        return (x > tx1 and y > ty1 and x + w < tx2 and y + h < ty2)
