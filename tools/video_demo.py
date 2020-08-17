import os
import sys
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

sys.path.append("../")
from models.yolo_face_detector import YoloFaceDetector

#load model which was trained for 25 epochs
model = model_from_json(open("../models/expression_recognition/weights/model2.json", "r").read())
#load weights
model.load_weights('../models/expression_recognition/weights/weights2.h5')

face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# face_detector = YoloFaceDetector(model_path="../model_config/weights/wider_face_yolo.h5",
#                                  anchors_path="../model_config/wider_anchors.txt",
#                                  classes_path="../model_config/wider_classes.txt")

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # faces_detected = face_detector.detect(test_img)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x,y,w,h) in faces_detected:
        # cv2.rectangle(test_img,(face.x,face.y),(face.x+face.width,face.y+face.height),(255,0,0),thickness=7)
        # roi_gray=gray_img[face.y:face.y+face.width,face.x:face.x+face.height]#cropping region of interest i.e. face area from  image
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        # cv2.putText(test_img, predicted_emotion, (int(face.x), int(face.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows