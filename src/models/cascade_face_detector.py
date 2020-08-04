import cv2
import numpy as np

from src.models.face_detector import FaceDetector


class CascadeFaceDetector(FaceDetector):
    DEFAULT_HAAR_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self, pretrained_model: str = DEFAULT_HAAR_CASCADE_FILE, scale_factor: float = 1.3,
                 min_neighbours: int = 5):
        self.classifier = cv2.CascadeClassifier(pretrained_model)
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours

    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        faces = self.classifier.detectMultiScale(image, self.scale_factor, self.min_neighbours)
        return faces