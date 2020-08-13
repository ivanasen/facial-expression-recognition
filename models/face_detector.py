import numpy as np
from typing import List

from models.bounding_box import BoundingBox


class FaceDetector(object):
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        pass
