from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.experimental import CosineDecay
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
                 train_epochs=config.TRAIN_EPOCHS,
                 model_path=None):
        self.anchors = utils.get_anchors(anchors_path)
        self.classes = utils.get_classes(classes_path)
        self.classes_count = len(self.classes)
        self.input_shape = input_shape
        self.log_path = log_path
        self.model_save_path = model_save_path
        self.train_epochs = train_epochs
        self.dataset_size = dataset_size

        self._load_dataset(dataset_path, dataset_size, batch_size)
        self.model = self._create_model(
            input_shape, self.anchors, self.classes_count, model_path)

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=CosineDecay(
        #     config.INIT_LEARNING_RATE, config.LEARNING_RATE_DECAY_STEPS))
        self.optimizer = tf.keras.optimizers.Adam()

        self.writer = tf.summary.create_file_writer(self.log_path)

        self.steps_per_epoch = len(self.ds_train)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = 2 * self.steps_per_epoch
        self.total_steps = 1 * self.steps_per_epoch

        self._log_info()

    def fit(self):
        for epoch in range(self.train_epochs):
            print(f"Start of epoch #{epoch + 1}")

            for image_data, target in self.ds_train:
                self._train_step(image_data, target)

            print(f"End of epoch {epoch} so testing on validation data")

            self._validation_step()

            if epoch % 5 == 0:
                self.model.save_weights(
                    self.model_save_path + f"wider_face_yolo_epoch_{epoch}.h5")
        model.save_weights(self.model_save_path + f"wider_face_yolo_final.h5")

    def _log_info(self):
        print("Dataset info")
        print("----------------------------------")
        print("Dataset size: ", self.dataset_size)
        print("Training size: ", self.ds_train.samples_count)
        print("Validation size: ", self.ds_val.samples_count)
        print("Number of train epochs: ", self.train_epochs)

    def _validation_step(self):
        loc_loss = conf_loss = prob_loss = 0
        for image_batch, target in self.ds_val:
            pred_result = self.model(image_batch, training=True)

            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                loc_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

        loc_loss /= len(self.ds_val)
        conf_loss /= len(self.ds_val)
        prob_loss /= len(self.ds_val)
        total_loss = loc_loss + conf_loss + prob_loss

        print("Validation loss:")
        tf.print("loc_loss: %4.2f conf_loss: %4.2f prob_loss: %4.2f total_loss: %4.2f"
                 % (loc_loss, conf_loss, prob_loss, total_loss))
        print("-------------------")

    def _train_step(self, image_batch, target):
        im_b = image_batch
        with tf.GradientTape() as tape:
            pred_result = self.model(image_batch, training=True)

            loc_loss = conf_loss = prob_loss = 0
            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss = compute_loss(pred, conv, *target[i], i)
                loc_loss += loss[0]
                conf_loss += loss[1]
                prob_loss += loss[2]

            total_loss = loc_loss + conf_loss + prob_loss
            gradients = tape.gradient(
                total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            tf.print("step %4d lr: %.6f loc_loss: %4.2f conf_loss: %4.2f "
                     "prob_loss: %4.2f total_loss: %4.2f"
                     % (self.global_steps, self.optimizer.lr.numpy(), loc_loss, conf_loss, prob_loss, total_loss))

            # Apply Cosine decay
            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:
                lr = self.global_steps / self.warmup_steps * config.TRAIN_LR_INIT
            else:
                lr = config.TRAIN_LR_END + 0.5 * (config.TRAIN_LR_INIT - config.TRAIN_LR_END) * (
                    (1 + tf.cos((self.global_steps - self.warmup_steps) /
                                (self.total_steps - self.warmup_steps) * np.pi)))
            self.optimizer.lr.assign(lr.numpy())

            with self.writer.as_default():
                tf.summary.scalar("lr", self.optimizer.lr,
                                  step=self.global_steps)
                tf.summary.scalar("loss/total_loss",
                                  total_loss, step=self.global_steps)
                tf.summary.scalar("loss/loc_loss",
                                  loc_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss",
                                  conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss",
                                  prob_loss, step=self.global_steps)
            self.writer.flush()

    def _create_model(self, input_shape, anchors, classes_count, model_path=None):
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        anchors_count = len(anchors)

        yolos = create_yolo_model(
            image_input, anchors_count, classes_count)

        output_tensors = []
        for i, yolo in enumerate(yolos):
            pred_tensor = decode_yolo_outputs(
                yolo, anchors, classes_count, i)
            output_tensors.append(yolo)
            output_tensors.append(pred_tensor)

        model = tf.keras.Model(image_input, output_tensors)

        if model_path != None:
            model.load_weights(model_path)
            print("Model loaded with pretrained weights from ", model_path)

        return model

    def _load_dataset(self, ds_path, ds_size, batch_size):
        self.ds_train = WiderDataset(
            ds_path, ds_size, "train", self.classes_count, self.anchors, batch_size)
        self.ds_val = WiderDataset(
            ds_path, ds_size, "validation", self.classes_count, self.anchors, batch_size)
