import cv2
import numpy as np
from typing import List

from models.face_detector import FaceDetector
from models.bounding_box import BoundingBox


class CascadeFaceDetector(FaceDetector):
    DEFAULT_HAAR_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self, pretrained_model: str = DEFAULT_HAAR_CASCADE_FILE, scale_factor: float = 1.3,
                 min_neighbours: int = 5):
        self.classifier = cv2.CascadeClassifier(pretrained_model)
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        faces_raw = self.classifier.detectMultiScale(image, self.scale_factor, self.min_neighbours)
        faces = []
        for face in faces_raw:
            box = BoundingBox(face[0], face[1], face[2], face[3])
            faces.append(box)
        return faces
