from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import numpy as np

from models.yolo import utils
import tensorflow as tf
from models.yolo.dataset import WiderDataset
from models.yolo.model import create_yolo_model, compute_loss, decode_yolo_outputs

import models.yolo.yolo_config as config


class YoloFaceDetectionTrainer(object):

    def __init__(self,
                 log_path=config.LOG_PATH,
                 anchors_path=config.ANCHORS_PATH,
                 classes_path=config.CLASSES_PATH,
                 input_shape=config.INPUT_SHAPE,
                 dataset_path=config.WIDER_DATASET_PATH,
                 dataset_size=config.WIDER_DATASET_SIZE,
                 batch_size=config.BATCH_SIZE,
                 model_save_path=config.FINAL_MODEL_SAVE_PATH,
                 train_epochs=config.TRAIN_EPOCHS):
        self.anchors = utils.get_anchors(anchors_path)
        self.classes = utils.get_classes(classes_path)
        self.classes_count = len(self.classes)
        self.input_shape = input_shape
        self.log_path = log_path
        self.model_save_path = model_save_path
        self.train_epochs = train_epochs

        self._load_dataset(dataset_path, dataset_size, batch_size)
        self.model = self._create_model(
            input_shape, self.anchors, self.classes_count)

        self.optimizer = tf.keras.optimizers.Adam()
        self.writer = tf.summary.create_file_writer(self.log_path)

        self.steps_per_epoch = len(self.ds_train)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = 2 * self.steps_per_epoch
        self.total_steps = 1 * self.steps_per_epoch

    def fit(self):
        for epoch in range(self.train_epochs):
            print(f"Start of epoch #{epoch + 1}")

            for image_data, target in self.ds_train:
                self._train_step(image_data, target)

            # Run a validation loop at the end of each epoch.
            # for x_batch_val, y_batch_val in val_dataset:
            #     val_logits = model(x_batch_val, training=False)
            #     # Update val metrics
            #     val_acc_metric.update_state(y_batch_val, val_logits)
            # val_acc = val_acc_metric.result()
            # val_acc_metric.reset_states()
            # print("Validation acc: %.4f" % (float(val_acc),))
            # print("Time taken: %.2fs" % (time.time() - start_time))

        self.model.save_weights(self.model_save_path)

    def _train_step(self, image_batch, target):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_batch, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(
                total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (self.global_steps, self.optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))

            # update learning rate
            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:
                lr = self.global_steps / self.warmup_steps * config.TRAIN_LR_INIT
            else:
                lr = config.TRAIN_LR_END + 0.5 * (config.TRAIN_LR_INIT - config.TRAIN_LR_END) * (
                    (1 + tf.cos((self.global_steps - self.warmup_steps) /
                                (self.total_steps - self.warmup_steps) * np.pi)))
            self.optimizer.lr.assign(lr.numpy())

            # writing summary data
            with self.writer.as_default():
                tf.summary.scalar("lr", self.optimizer.lr,
                                  step=self.global_steps)
                tf.summary.scalar("loss/total_loss",
                                  total_loss, step=self.global_steps)
                tf.summary.scalar("loss/giou_loss",
                                  giou_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss",
                                  conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss",
                                  prob_loss, step=self.global_steps)
            self.writer.flush()

    def _create_model(self, input_shape, anchors, classes_count):
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        anchors_count = len(anchors)

        yolos = create_yolo_model(
            image_input, anchors_count, classes_count)

        output_tensors = []
        for i, yolo in enumerate(yolos):
            pred_boxes, pred_scores, pred_classes = decode_yolo_outputs(
                yolo, anchors, classes_count, i)
            pred_tensor = tf.concat(
                [pred_boxes, pred_scores, pred_classes], axis=-1)
            output_tensors.append(yolo)
            output_tensors.append(pred_tensor)

        model = tf.keras.Model(image_input, output_tensors)

        return model

    def _load_dataset(self, ds_path, ds_size, batch_size):
        self.ds_train = WiderDataset(
            ds_path, ds_size, "train", self.classes_count, self.anchors, batch_size)
        self.ds_val = WiderDataset(
            ds_path, ds_size, "validation", self.classes_count, self.anchors, batch_size)

# def train_face_detection(log_dir: str, classes_path: str, anchors_path: str, data_dir: str, dataset_size: str):
#     anchors = utils.get_anchors(anchors_path)
#     self.classes = utils.get_classes(classes_path)
#     classes_count = len(self.classes)
#     input_shape = (608, 608)
#     model = create_model(input_shape, anchors, classes_count)

#     # print("Loading data...")
#     # train_x, train_y, val_x, val_y, test_x, test_y = load_data(
#     #     data_dir, dataset_size, input_shape, anchors, classes_count)
#     # print("Data loaded successfully!")
#     # print(
#     #     f"Dataset size: {len(train_x)} train, {len(val_x)} val, {len(test_x)} test")

#     # model.compile(optimizer=Adam(), loss=compute_loss(
#     #     anchors, classes_count=classes_count))

#     # tensorboard = TensorBoard(log_dir=log_dir)
#     # # checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
#     # #                              monitor="val_loss", save_weights_only=True, save_best_only=True, save_freq=3)
#     # reduce_lr = ReduceLROnPlateau(
#     #     monitor="val_loss", factor=0.1, patience=3, verbose=1)
#     # early_stopping = EarlyStopping(
#     #     monitor="val_loss", min_delta=0, patience=10, verbose=1)

#     # batch_size = 4

#     # model.fit(x=train_x,
#     #           y=train_y,
#     #           validation_data=(val_x, val_y),
#     #           epochs=20,
#     #           callbacks=[tensorboard, reduce_lr, early_stopping])
#     # model.save_weights(log_dir + "trained_weights_final.h5")
