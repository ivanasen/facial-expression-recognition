import numpy as np


INPUT_SIZE = 608
INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE)
ANCHOR_PER_SCALE = 3
STRIDES = np.array([32, 16, 8])
MAX_BOXES_PER_SCALE = 20
TRAIN_LR_INIT = 1e-3
TRAIN_LR_END = 1e-6
SCORE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
IOU_LOSS_THRESHOLD = 0.5

ANCHORS_PATH = "model_config/wider_anchors.txt"
CLASSES_PATH = "model_config/wider_classes.txt"
LOG_PATH = "logs/"
WIDER_DATASET_PATH = "data/"
FINAL_MODEL_SAVE_PATH = LOG_PATH + "/models/"

# Use x% of the wider dataset for training, validation and testing
WIDER_DATASET_SIZE = "30%"

BATCH_SIZE = 4
TRAIN_EPOCHS = 10
