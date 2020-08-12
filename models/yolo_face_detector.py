import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input

from models.yolo.model import decode_yolo_outputs, create_yolo_model
from models.yolo.utils import letterbox_image
from models.face_detector import FaceDetector


class YoloFaceDetector(FaceDetector):
    def __init__(
            self,
            model_path="data/wider_face_yolo.h5",
            anchors_path="data/wider_anchors.txt",
            classes_path="data/wider_classes.txt",
            score_threshold=0.3,
            iou_threshold=0.4,
            model_image_size=(608, 608)):
        self._model_path = model_path
        self._anchors_path = anchors_path
        self._classes_path = classes_path
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._model_image_size = model_image_size

        self._class_names = self._get_class()
        self._anchors = self._get_anchors()
        self._boxes, self._scores, self._classes = self._init_model()

    def detect(self, image: np.ndarray) -> np.ndarray:
        if self._model_image_size != (None, None):
            boxed_image = letterbox_image(image, tuple(reversed(self._model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype="float32")

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        with tf.compat.v1.Session as sess:
            sess.run([self._boxes, self._scores, self._classes],
                     feed_dict={
                         self._yolo_model.input: image_data,
                         self._input_image_shape: [image.size[1], image.size[0]]
                     })
            out_boxes, out_scores, out_classes = self._yolo_model(image_data)

        return out_boxes

    def _get_class(self):
        classes_path = os.path.expanduser(self._classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self._anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1, 2)

    def _init_model(self):
        model_path = os.path.expanduser(self._model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file."

        anchors_count = len(self._anchors)
        classes_count = len(self._class_names)

        try:
            self._yolo_model = create_yolo_model(
                Input(shape=(None, None, 3)),
                anchors_count=anchors_count,
                classes_count=classes_count)
            self._yolo_model.load_weights(self._model_path)
        except IOError:
            print(f"Failed loading model weights. Weights file {self._model_path} not found.")

        print(f"{model_path} model, anchors, and classes loaded.")

        self._input_image_shape = keras.backend.placeholder(shape=(2,))
        boxes, scores, classes = decode_yolo_outputs(
            self._yolo_model.output,
            self._anchors,
            len(self._class_names),
            self._input_image_shape,
            max_boxes=20,
            score_threshold=self._score_threshold,
            iou_threshold=self._iou_threshold)

        return boxes, scores, classes
