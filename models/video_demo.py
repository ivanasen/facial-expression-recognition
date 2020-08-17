from models.yolo_face_detector import YoloFaceDetector
from models.cascade_face_detector import CascadeFaceDetector
import os
import sys
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# load model which was trained for 25 epochs
model = model_from_json(
    open("./models/expression_recognition/weights/model2.json", "r").read())
model.load_weights("./models/expression_recognition/weights/weights2.h5")

# face_detector = CascadeFaceDetector()
face_detector = YoloFaceDetector(model_path="model_config/weights/wider_face_yolo.h5",
                                 anchors_path="model_config/wider_anchors.txt",
                                 classes_path="model_config/wider_classes.txt")

cap = cv2.VideoCapture(0)
PREDICTION_PERIOD = 30
prediction_count = 0
BORDER_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 0, 255)
predictions = []
emotions = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

while True:
    success, frame = cap.read()
    if not success:
        print("Error reading from camera!")
        break
    
    if prediction_count % PREDICTION_PERIOD == 0:
        faces = face_detector.detect(frame)

        predictions = []
        for face in faces:
            roi = frame[face.y:face.y +
                                face.width, face.x:face.x+face.height]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            img_np = image.img_to_array(roi)
            img_np = np.expand_dims(img_np, axis=0)
            emotion_predictions = model.predict(img_np)
            max_index = np.argmax(emotion_predictions[0])
            predicted_emotion = emotions[max_index]

            predictions.append((face, predicted_emotion))

    for face, emotion in predictions:
        cv2.rectangle(frame, (face.x, face.y),
                      (face.x + face.width, face.y + face.height), BORDER_COLOR, 7)
        cv2.putText(frame, emotion, (face.x, face.y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    prediction_count += 1

cap.release()
cv2.destroyAllWindows
