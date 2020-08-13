from typing import List

import cv2
import numpy as np

from models.bounding_box import BoundingBox


def resize_image_with_borders(image, desired_size: int):
    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = [int(x * ratio) for x in old_size]

    new_image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    fill_color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(new_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)

    return new_image


def convert_ndarray_to_bboxes(
        bboxes_raw: np.ndarray, scores: np.ndarray, image_height, image_width) -> List[BoundingBox]:
    faces = []
    for i in range(len(bboxes_raw)):
        bbox = bboxes_raw[i]
        score = scores[i]

        top, left, bottom, right = bbox.astype('int32')
        top = max(0, top)
        left = max(0, left)
        bottom = min(image_height, bottom)
        right = min(image_width, right)
        faces.append(BoundingBox(left, top, abs(right - left), abs(bottom - top), score))
    return faces
