import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from models.bounding_box import BoundingBox
from models.yolo.model import decode_yolo_outputs, create_yolo_model
from models.yolo.utils import resize_image_with_borders
from models.face_detector import FaceDetector
import models.yolo.utils as utils


class YoloFaceDetector(FaceDetector):
    def __init__(
            self,
            model_path="data/wider_face_yolo.h5",
            anchors_path="data/wider_anchors.txt",
            classes_path="data/wider_classes.txt",
            score_threshold=0.3,
            iou_threshold=0.4,
            model_image_size=608):
        self._model_path = model_path
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._model_image_size = model_image_size
        self._class_names = utils.get_classes(classes_path)
        self._anchors = utils.get_anchors(anchors_path)

        self._init_model()

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        image_shape = image.shape[:2]
        resized_image = resize_image_with_borders(image, self._model_image_size)

        resized_image = tf.cast(resized_image, tf.float32)
        resized_image /= 255.
        resized_image = np.expand_dims(resized_image, 0)

        yolos = self._model.predict(resized_image)
        faces_raw, scores, _ = decode_yolo_outputs(
            yolos=yolos,
            anchors=self._anchors,
            classes_count=len(self._class_names),
            image_shape=image_shape,
            max_boxes=20,
            score_threshold=self._score_threshold,
            iou_threshold=self._iou_threshold)
        faces_raw = faces_raw.numpy()
        faces = utils.convert_ndarray_to_bboxes(faces_raw, scores, image_shape[0], image_shape[1])
        return faces

    def _init_model(self):
        model_path = os.path.expanduser(self._model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file."

        try:
            self._model = create_yolo_model(
                Input(shape=(None, None, 3)),
                anchors_count=len(self._anchors) // 3,
                classes_count=len(self._class_names))
            self._model.load_weights(self._model_path)
        except IOError:
            print(f"Failed loading model weights. Weights file {self._model_path} not found.")
        print(f"{model_path} model, anchors, and classes loaded.")
