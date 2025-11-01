import cv2

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def blur_faces(frame):
    """
    Detects faces in a frame and blurs them.
    Returns the frame with blurred faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        face_region = cv2.GaussianBlur(face_region, (35, 35), 30)
        frame[y:y+h, x:x+w] = face_region

    return frame
