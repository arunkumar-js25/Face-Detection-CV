import numpy as np
import cv2
from imutils.video import VideoStream
import imutils
import time

def EyeDetectImage():
    img = cv2.imread("Images/fam.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    path = "haarcascades/haarcascade_eye.xml"

    eye_cascade = cv2.CascadeClassifier(path)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2, minSize=(20,30))
    print(len(eyes))

    for (x, y, w, h) in eyes:
        cv2.circle(img, (int(x+(w/2)),int(y+(h/2))), int(w/2), (0, 0, 255),2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def EyeDetectVideo():
    cap = cv2.VideoCapture(0)
    color = (0, 255, 0)
    line_width = 3
    radius = 20
    point = (0, 0)

    #LOAD EYE DETECTION
    path = "haarcascades/haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(path)

    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(5, 5))

        for (x, y, w, h) in eyes:
            cv2.circle(frame, (int(x + (w / 2)), int(y + (h / 2))), int(w / 2), (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


EyeDetectVideo()