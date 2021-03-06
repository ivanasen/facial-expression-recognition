from typing import List
import os
import cv2
import numpy as np
import tensorflow as tf
import random

from models.bounding_box import BoundingBox


def resize_boxes(boxes, old_size, desired_size, max_boxes, classes_count):
    h = old_size[0]
    w = old_size[1]

    scale = float(desired_size) / max(old_size)
    new_size = [int(x * scale) for x in old_size]
    delta_y = abs(desired_size - new_size[0]) // 2
    delta_x = abs(desired_size - new_size[1]) // 2

    boxes = boxes.astype(np.float32)
    box_data = np.zeros((max_boxes, 4 + classes_count))
    if len(boxes) > max_boxes:
        boxes = boxes[:max_boxes]

    boxes[:, [0, 2]] = boxes[:, [0, 2]] * h * scale + delta_y
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * w * scale + delta_x
    box_data[:len(boxes), :4] = boxes

    box_data = box_data.astype(np.int32)

    return box_data


def resize_image_with_borders(image, desired_size: int):
    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = [int(x * ratio) for x in old_size]

    new_image = cv2.resize(
        image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    fill_color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(
        new_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)

    return new_image


def convert_ndarray_to_boxes(
        boxes_raw: np.ndarray, scores: np.ndarray, image_height, image_width) -> List[BoundingBox]:
    faces = []
    for i in range(len(boxes_raw)):
        box = boxes_raw[i]
        score = scores[i]

        top, left, bottom, right = box.astype('int32')
        top = max(0, top)
        left = max(0, left)
        bottom = min(image_height, bottom)
        right = min(image_width, right)
        faces.append(BoundingBox(left, top, abs(
            right - left), abs(bottom - top), score))
    return faces


def get_classes(classes_path: str):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    return np.array(anchors).reshape(3, 3, 2)


def shuffle_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def box_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def box_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def nms(boxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param boxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(boxes[:, 5]))
    best_boxes = []

    for cls in classes_in_img:
        cls_mask = (boxes[:, 5] == cls)
        cls_boxes = boxes[cls_mask]

        while len(cls_boxes) > 0:
            max_ind = np.argmax(cls_boxes[:, 4])
            best_box = cls_boxes[max_ind]
            best_boxes.append(best_box)
            cls_boxes = np.concatenate(
                [cls_boxes[: max_ind], cls_boxes[max_ind + 1:]])
            iou = bboxes_iou(best_box[np.newaxis, :4], cls_boxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_boxes[:, 4] = cls_boxes[:, 4] * weight
            score_mask = cls_boxes[:, 4] > 0.
            cls_boxes = cls_boxes[score_mask]

    return best_boxes


def random_horizontal_flip(image, boxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    return image, boxes


def random_crop(image, boxes):
    if random.random() < 0.5:
        return image, boxes

    h, w, _ = image.shape
    max_box = np.concatenate(
        [np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_box[0]
    max_u_trans = max_box[1]
    max_r_trans = w - max_box[2]
    max_d_trans = h - max_box[3]

    crop_xmin = max(
        0, int(max_box[0] - random.uniform(0, max_l_trans)))
    crop_ymin = max(
        0, int(max_box[1] - random.uniform(0, max_u_trans)))
    crop_xmax = max(
        w, int(max_box[2] + random.uniform(0, max_r_trans)))
    crop_ymax = max(
        h, int(max_box[3] + random.uniform(0, max_d_trans)))

    image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

    boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_xmin
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_ymin

    return image, boxes


def random_translate(image, boxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_box = np.concatenate(
            [np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_box[0]
        max_u_trans = max_box[1]
        max_r_trans = w - max_box[2]
        max_d_trans = h - max_box[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        boxes[:, [0, 2]] = boxes[:, [0, 2]] + tx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] + ty

    return image, boxes


def augment_data(image, boxes):
    # image, boxes = random_horizontal_flip(
    #     np.copy(image), np.copy(boxes))
    image, boxes = random_crop(np.copy(image), np.copy(boxes))
    # image, boxes = random_translate(
    #     np.copy(image), np.copy(boxes))
    image = image.astype(np.float32) / 255.0

    return image, boxes
