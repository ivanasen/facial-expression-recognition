import random
import cv2
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from models.yolo import utils
import models.yolo.yolo_config as config


class WiderDataset(object):
    def __init__(self, dataset_dir, dataset_size, dataset_split, classes_count, anchors, batch_size):
        self.ds_x, self.ds_y = get_wider_dataset(
            dataset_dir, dataset_size, split=dataset_split)

        self.input_size = config.INPUT_SIZE
        self.strides = config.STRIDES
        self.output_sizes = [self.input_size // x for x in self.strides]
        self.anchors = anchors
        self.classes_count = classes_count
        self.anchor_per_scale = len(anchors)
        self.batch_size = batch_size
        self.max_box_per_scale = config.MAX_BOXES_PER_SCALE
        self.samples_count = len(self.ds_x)
        self.batches_count = int(np.ceil(self.samples_count / self.batch_size))
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_count >= self.batches_count:
            self.batch_count = 0
            utils.shuffle_unison(self.ds_x, self.ds_y)
            raise StopIteration

        index = self.batch_count * self.batch_size
        images_batch = np.copy(self.ds_x[index:index + self.batch_size])
        boxes_batch = np.copy(self.ds_y[index:index + self.batch_size])

        batch_ltarget, batch_mtarget, batch_starget = transform_boxes_for_yolo_batch(
            boxes_batch,
            self.input_size,
            self.anchors,
            self.classes_count,
            self.anchor_per_scale,
            self.max_box_per_scale,
            self.strides)

        self.batch_count += 1

        return images_batch, (batch_ltarget, batch_mtarget, batch_starget)

    def __len__(self):
        return self.batches_count


def get_wider_dataset(data_dir, dataset_size, split="train"):
    ds = tfds.load(
        "wider_face",
        split=f"{split}[:{dataset_size}]",
        with_info=False,
        shuffle_files=True,
        data_dir=data_dir)
    ds_np = tfds.as_numpy(ds)

    ds_images = []
    ds_faces = []

    for d in ds_np:
        image_data = d["image"]
        face_data = d["faces"]["bbox"]
        np.random.shuffle(face_data)

        image_size = image_data.shape[:2]
        image_data = utils.resize_image_with_borders(
            image=image_data, desired_size=608)

        face_data = utils.resize_boxes(face_data, image_size, 608, 20)
        ds_images.append(image_data)
        ds_faces.append(face_data)

    ds_images = np.array(ds_images)
    ds_faces = np.array(ds_faces)

    return ds_images, ds_faces


def transform_boxes_for_yolo_batch(
        boxes_batch, input_size, anchors, classes_count, anchor_per_scale, max_box_per_scale, strides):
    batch_size = len(boxes_batch)
    output_sizes = [input_size // x for x in strides]
    batch_label_lbox = np.zeros((batch_size, output_sizes[0], output_sizes[0],
                                 anchor_per_scale, 5 + classes_count), dtype=np.float32)
    batch_label_mbox = np.zeros((batch_size, output_sizes[1], output_sizes[1],
                                 anchor_per_scale, 5 + classes_count), dtype=np.float32)
    batch_label_sbox = np.zeros((batch_size, output_sizes[2], output_sizes[2],
                                 anchor_per_scale, 5 + classes_count), dtype=np.float32)

    batch_lboxes = np.zeros(
        (batch_size, max_box_per_scale, 4), dtype=np.float32)
    batch_mboxes = np.zeros(
        (batch_size, max_box_per_scale, 4), dtype=np.float32)
    batch_sboxes = np.zeros(
        (batch_size, max_box_per_scale, 4), dtype=np.float32)

    for i, boxes in enumerate(boxes_batch):
        label_lbox, label_mbox, label_sbox, lboxes, mboxes, sboxes = transform_boxes_for_yolo(
            boxes=boxes,
            input_size=input_size,
            anchors=anchors,
            classes_count=classes_count,
            anchor_per_scale=anchor_per_scale,
            max_box_per_scale=max_box_per_scale,
            strides=strides)

        batch_label_lbox[i, :, :, :, :] = label_lbox
        batch_label_mbox[i, :, :, :, :] = label_mbox
        batch_label_sbox[i, :, :, :, :] = label_sbox
        batch_sboxes[i, :, :] = lboxes
        batch_mboxes[i, :, :] = mboxes
        batch_lboxes[i, :, :] = sboxes

    batch_ltarget = batch_label_lbox, batch_lboxes
    batch_mtarget = batch_label_mbox, batch_mboxes
    batch_starget = batch_label_sbox, batch_sboxes

    return batch_ltarget, batch_mtarget, batch_starget


def transform_boxes_for_yolo(boxes, input_size, anchors, classes_count, anchor_per_scale, max_box_per_scale, strides):
    output_sizes = [input_size // x for x in strides]

    label = [np.zeros((output_sizes[i], output_sizes[i], anchor_per_scale,
                       5 + classes_count)) for i in range(3)]
    boxes_xywh = [np.zeros((max_box_per_scale, 4))
                  for _ in range(3)]
    box_count = np.zeros((3,))

    for box in boxes:
        box_coor = box[:4]
        box_class_ind = box[4]

        onehot = np.ones(classes_count, dtype=np.float)

        box_xywh = np.concatenate(
            [(box_coor[2:] + box_coor[:2]) * 0.5, box_coor[2:] - box_coor[:2]], axis=-1)
        box_xywh_scaled = 1.0 * \
            box_xywh[np.newaxis, :] / strides[:, np.newaxis]

        iou = []
        exist_positive = False
        for i in range(3):
            anchors_xywh = np.zeros((anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = np.floor(
                box_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]

            iou_scale = utils.box_iou(
                box_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                xind, yind = np.floor(
                    box_xywh_scaled[i, 0:2]).astype(np.int32)

                label[i][yind, xind, iou_mask, :] = 0
                label[i][yind, xind, iou_mask, 0:4] = box_xywh
                label[i][yind, xind, iou_mask, 4:5] = 1.0
                label[i][yind, xind, iou_mask, 5:] = onehot

                box_ind = int(box_count[i] % max_box_per_scale)
                boxes_xywh[i][box_ind, :4] = box_xywh
                box_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / anchor_per_scale)
            best_anchor = int(best_anchor_ind % anchor_per_scale)
            xind, yind = np.floor(
                box_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = box_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = onehot

            box_ind = int(box_count[best_detect] %
                          max_box_per_scale)
            boxes_xywh[best_detect][box_ind, :4] = box_xywh
            box_count[best_detect] += 1
    label_lbox, label_mbox, label_sbox = label
    lboxes, mboxes, sboxes = boxes_xywh
    return label_lbox, label_mbox, label_sbox, lboxes, mboxes, sboxes
