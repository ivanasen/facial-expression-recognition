import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
import numpy as np

from models.yolo.blocks import conv_block
from models.yolo.darknet53 import darknet53
import models.yolo.yolo_config as config
from models.yolo import utils


def create_yolo_model(inputs, anchors_count, classes_count):
    route_1, route_2, x = darknet53(inputs)

    x = _conv_block_body(x, 1024)
    conv_large_branch, conv_large_box = _conv_box(
        x, 1024, anchors_count, classes_count)

    x = conv_block(x, filters=256, kernel_size=1)
    x = UpSampling2D()(x)
    x = tf.concat([x, route_2], axis=-1)

    x = _conv_block_body(x, 512)
    conv_medium_branch, conv_medium_box = _conv_box(
        x, 512, anchors_count, classes_count)

    x = conv_block(x, filters=128, kernel_size=1)
    x = UpSampling2D()(x)
    x = tf.concat([x, route_1], axis=-1)

    x = _conv_block_body(x, 256)
    conv_small_branch, conv_small_box = _conv_box(
        x, 256, anchors_count, classes_count)

    # return Model(inputs, [conv_large_box, conv_medium_box, conv_small_box])
    return [conv_large_box, conv_medium_box, conv_small_box]


def _conv_block_body(inputs, filters):
    half = filters // 2
    x = conv_block(inputs, filters=half, kernel_size=1)
    x = conv_block(x, filters=filters, kernel_size=3)
    x = conv_block(x, filters=half, kernel_size=1)
    x = conv_block(x, filters=filters, kernel_size=3)
    x = conv_block(x, filters=half, kernel_size=1)
    return x


def _conv_box(inputs, filters, num_anchors, num_classes):
    branch = conv_block(inputs, filters=filters, kernel_size=3)
    box = conv_block(branch,
                     filters=num_anchors * (num_classes + 5),
                     kernel_size=1,
                     activate=False,
                     batch_norm=False)
    return branch, box


def decode_yolo_outputs(conv_output, anchors, classes_count, i=0):
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + classes_count))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)
                [:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)
                [tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [
                      batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * config.STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * config.STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return pred_xywh, pred_conf, pred_prob


def decode_and_filter_yolo_outputs(yolos, anchors, classes_count, image_shape, max_boxes, score_threshold, iou_threshold):
    boxes, scores = _decode_initial_boxes(
        yolos, anchors, classes_count, image_shape)
    boxes, scores, classes = _filter_initial_boxes(
        boxes, scores, classes_count, max_boxes, score_threshold, iou_threshold)
    return boxes, scores, classes


def _decode_initial_boxes(yolos, anchors, classes_count, image_shape):
    yolos_count = len(yolos)

    input_shape = tf.shape(yolos[0])[1:3] * 32

    boxes = []
    box_scores = []
    for i in range(yolos_count):
        _boxes, _box_scores = _boxes_and_scores(
            yolos[i], anchors[i], classes_count, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)

    return boxes, box_scores


def _boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    num_anchors = len(anchors)
    anchors_tensor = tf.cast(tf.reshape(
        anchors, [1, 1, 1, num_anchors, 2]), feats.dtype)
    grid_shape = tf.shape(feats)[1:3]  # height, width
    grid_y = tf.tile(tf.reshape(
        tf.range(0, limit=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[1]), [
                     1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, feats.dtype)

    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Transform to bounding boxes
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / \
        tf.cast(grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / \
        tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    boxes = _correct_boxes_to_original_image(
        box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    
    return boxes, box_scores


# TODO: Fix strange bounding box offset we're getting with the pretrained weights
def _correct_boxes_to_original_image(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = tf.cast(input_shape, box_yx.dtype)
    image_shape = tf.cast(image_shape, box_yx.dtype)

    new_shape = tf.round(
        image_shape * tf.keras.backend.min(input_shape / image_shape))
    offset = ((input_shape - new_shape) / 2) / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    boxes = tf.concat([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes


def _convert_feats_to_boxes(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    anchors_tensor = tf.cast(tf.reshape(
        anchors, [1, 1, 1, num_anchors, 2]), feats.dtype)
    grid_shape = tf.shape(feats)[1:3]  # height, width
    grid_y = tf.tile(tf.reshape(
        tf.range(0, limit=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[1]), [
                     1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, feats.dtype)

    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Transform to bounding boxes
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / \
        tf.cast(grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / \
        tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def _filter_initial_boxes(boxes, scores, classes_count, max_boxes, score_threshold, iou_threshold):
    max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
    mask = scores >= score_threshold
    final_boxes = []
    final_scores = []
    final_classes = []

    for i in range(classes_count):
        class_boxes = tf.boolean_mask(boxes, mask[:, i])
        class_scores = tf.boolean_mask(scores[:, i], mask[:, i])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_scores = tf.gather(class_scores, nms_index)
        classes = tf.ones_like(class_scores, 'int32') * i
        final_boxes.append(class_boxes)
        final_scores.append(class_scores)
        final_classes.append(classes)

    final_boxes = tf.concat(final_boxes, axis=0)
    final_scores = tf.concat(final_scores, axis=0)
    final_classes = tf.concat(final_classes, axis=0)
    return final_boxes, final_scores, final_classes


def compute_loss(pred, conv, label, boxes, i=0):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = config.STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size,
                             output_size, 3, 5 + 1))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_box = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(utils.box_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    box_loss_scale = 2.0 - 1.0 * \
        label_xywh[:, :, :, :, 2:3] * \
        label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_box * box_loss_scale * (1 - giou)

    iou = utils.box_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                        boxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_box) * \
        tf.cast(max_iou < config.IOU_LOSS_THRESHOLD, tf.float32)

    conf_focal = tf.pow(respond_box - pred_conf, 2)

    conf_loss = conf_focal * (
        respond_box *
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_box, logits=conv_raw_conf)
        +
        respond_bgd *
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_box, logits=conv_raw_conf)
    )

    prob_loss = respond_box * \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
