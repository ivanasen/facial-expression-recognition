import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from models.bounding_box import BoundingBox
from models.yolo.model import decode_and_filter_yolo_outputs, create_yolo_model
from models.yolo.utils import resize_image_with_borders
from models.face_detector import FaceDetector
import models.yolo.utils as utils
import models.yolo.yolo_config as config


class YoloFaceDetector(FaceDetector):
    def __init__(
            self,
            model_path,
            anchors_path=config.ANCHORS_PATH,
            classes_path=config.CLASSES_PATH,
            score_threshold=config.SCORE_THRESHOLD,
            iou_threshold=config.IOU_THRESHOLD,
            model_image_size=config.INPUT_SIZE,
            max_boxes=config.MAX_BOXES_PER_SCALE):
        self._model_path = model_path
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._model_image_size = model_image_size
        self._class_names = utils.get_classes(classes_path)
        self._anchors = utils.get_anchors(anchors_path)
        self._max_boxes = max_boxes

        self._init_model()

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        image_shape = image.shape[:2]
        resized_image = resize_image_with_borders(
            image, self._model_image_size)

        resized_image = tf.cast(resized_image, tf.float32)
        resized_image /= 255.
        resized_image = np.expand_dims(resized_image, 0)

        yolos = self._model.predict(resized_image)
        faces_raw, scores, _ = decode_and_filter_yolo_outputs(
            yolos=yolos,
            anchors=self._anchors,
            classes_count=len(self._class_names),
            image_shape=image_shape,
            max_boxes=self._max_boxes,
            score_threshold=self._score_threshold,
            iou_threshold=self._iou_threshold)

        faces_raw = faces_raw.numpy()
        faces = utils.convert_ndarray_to_boxes(
            faces_raw, scores, image_shape[0], image_shape[1])
        return faces

    def _init_model(self):
        assert self._model_path.endswith(
            ".h5"), "Keras model or weights must be a .h5 file."

        input_tensor = Input(shape=(None, None, 3))
        outputs = create_yolo_model(
            input_tensor,
            anchors_count=len(self._anchors),
            classes_count=len(self._class_names))
        self._model = Model(input_tensor, outputs)
        try:
            self._model.load_weights(self._model_path)
        except IOError:
            print(
                f"Failed loading model weights. Weights file {self._model_path} not found.")
        # else:
            # box_tensors = []
            # for i, out in enumerate(outputs):
            #     box_tensor = decode_yolo_outputs(out, i)
            #     box_tensors.append(box_tensor)
            # self._model = Model(input_tensor, box_tensors)

        print(f"Model loaded with weights from {self._model_path}.")
