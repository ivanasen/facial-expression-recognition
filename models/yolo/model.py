import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model

from models.yolo.blocks import conv_block
from models.yolo.darknet53 import darknet53


def create_yolo_model(inputs, anchors_count, classes_count):
    route_1, route_2, x = darknet53(inputs)

    x = _conv_block_body(x, 1024)
    conv_large_branch, conv_large_bbox = _conv_bbox(x, 1024, anchors_count, classes_count)

    x = conv_block(x, filters=256, kernel_size=1)
    x = UpSampling2D()(x)
    x = tf.concat([x, route_2], axis=-1)

    x = _conv_block_body(x, 512)
    conv_medium_branch, conv_medium_bbox = _conv_bbox(x, 512, anchors_count, classes_count)

    x = conv_block(x, filters=128, kernel_size=1)
    x = UpSampling2D()(x)
    x = tf.concat([x, route_1], axis=-1)

    x = _conv_block_body(x, 256)
    conv_small_branch, conv_small_bbox = _conv_bbox(x, 256, anchors_count, classes_count)

    return Model(inputs, [conv_small_bbox, conv_medium_bbox, conv_large_bbox])


def _conv_block_body(inputs, filters):
    half = filters // 2
    x = conv_block(inputs, filters=half, kernel_size=1)
    x = conv_block(x, filters=filters, kernel_size=3)
    x = conv_block(x, filters=half, kernel_size=1)
    x = conv_block(x, filters=filters, kernel_size=3)
    x = conv_block(x, filters=half, kernel_size=1)
    return x


def _conv_bbox(inputs, filters, num_anchors, num_classes):
    branch = conv_block(inputs, filters=filters, kernel_size=3)
    bbox = conv_block(branch,
                      filters=num_anchors * (num_classes + 5),
                      kernel_size=1,
                      activate=False,
                      batch_norm=False)
    return branch, bbox


# TODO: Finish feature to bounding box conversion
# def convert_feats_to_bboxes(feats, anchors, num_classes, input_shape, calc_loss=False):
#     num_anchors = len(anchors)
#
#     # Reshape to batch, height, width, num_anchors, box_params.
#     anchors_tensor = tf.reshape(anchors, [1, 1, 1, num_anchors, 2])
#
#     grid_shape = tf.shape(feats)[1:3]  # height, width
#     grid_y = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
#     grid_x = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
#     grid = tf.concat([grid_x, grid_y], axis=-1)
#     grid = tf.cast(grid, feats.dtype)
#
#     feats = tf.reshape(
#         feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
#
#     # Adjust preditions to each spatial grid point and anchor size.
#     box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], feats.dtype)
#     box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], feats.dtype)
#     box_confidence = tf.sigmoid(feats[..., 4:5])
#     box_class_probs = tf.sigmoid(feats[..., 5:])
#
#     if calc_loss:
#         return grid, feats, box_xy, box_wh
#     return box_xy, box_wh, box_confidence, box_class_probs
#
#
# def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
#     box_yx = box_xy[..., ::-1]
#     box_hw = box_wh[..., ::-1]
#     input_shape = tf.cast(input_shape, box_yx.dtype)
#     image_shape = tf.cast(image_shape, box_yx.dtype)
#     new_shape = tf.round(image_shape * tf.keras.backend.min(input_shape / image_shape))
#     offset = (input_shape - new_shape) / 2. / input_shape
#     scale = input_shape / new_shape
#     box_yx = (box_yx - offset) * scale
#     box_hw *= scale
#
#     box_mins = box_yx - (box_hw / 2.)
#     box_maxes = box_yx + (box_hw / 2.)
#     boxes = tf.concat([
#         box_mins[..., 0:1],  # y_min
#         box_mins[..., 1:2],  # x_min
#         box_maxes[..., 0:1],  # y_max
#         box_maxes[..., 1:2]  # x_max
#     ], axis=-1)
#
#     # Scale boxes back to original image shape.
#     boxes *= tf.concat([image_shape, image_shape], axis=-1)
#     return boxes
#
#
# def boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
#     box_xy, box_wh, box_confidence, box_class_probs = convert_feats_to_bboxes(
#         feats, anchors, num_classes, input_shape)
#     boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
#     boxes = tf.reshape(boxes, [-1, 4])
#     box_scores = box_confidence * box_class_probs
#     box_scores = tf.reshape(box_scores, [-1, num_classes])
#     return boxes, box_scores
#
#
def decode_yolo_outputs(outputs, anchors, classes_count, image_shape, max_boxes, score_threshold, iou_threshold):
    pass
#     num_layers = len(outputs)
#     anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#
#     input_shape = tf.shape(outputs[0])[1:3] * 32
#     boxes = []
#     box_scores = []
#     for i in range(num_layers):
#         _boxes, _box_scores = boxes_and_scores(
#             outputs[i], anchors[anchor_mask[i]], classes_count, input_shape, image_shape)
#         boxes.append(_boxes)
#         box_scores.append(_box_scores)
#     boxes = tf.concat(boxes, axis=0)
#     box_scores = tf.concat(box_scores, axis=0)
#
#     mask = box_scores >= score_threshold
#     max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
#     boxes_ = []
#     scores_ = []
#     classes_ = []
#     for c in range(classes_count):
#         class_boxes = tf.boolean_mask(boxes, mask[:, c])
#         class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
#         nms_index = tf.image.non_max_suppression(
#             class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
#         class_boxes = tf.gather(class_boxes, nms_index)
#         class_box_scores = tf.gather(class_box_scores, nms_index)
#         classes = tf.ones_like(class_box_scores, 'int32') * c
#         boxes_.append(class_boxes)
#         scores_.append(class_box_scores)
#         classes_.append(classes)
#     boxes_ = tf.concat(boxes_, axis=0)
#     scores_ = tf.concat(scores_, axis=0)
#     classes_ = tf.concat(classes_, axis=0)
#
#     return boxes_, scores_, classes_
