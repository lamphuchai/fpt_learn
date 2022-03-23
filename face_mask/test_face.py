from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
people_list = []

video_capture = VideoStream(src=0).start()

while True:
    _, frame = video_capture.read()
    image = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    detections = faceCascade.detectMultiScale(gray, 1.15, 5)

    for i in range(len(detections)):
        face_i = detections[i]
        x, y, w, h = face_i

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 222, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        people_list.insert(len(people_list)+1,i)

        cv2.putText(frame, "id: "+str ( people_list[i]), (x, y), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)