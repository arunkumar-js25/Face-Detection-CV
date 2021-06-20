import cv2
import numpy as np
import pyautogui
import face_recognition

#C:\Users\arunkumar.j06\Pictures\IMG_8740.jpg
print ('[INFO] Encoding Search Image...')
imgpath = 'Images/ak.jpg'
name = imgpath.split('/')[-1].split('.')[0]
print(name)
img = face_recognition.load_image_file(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
encode = face_recognition.face_encodings(img)[0]

def CheckFrame(imgS,name,frame):
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces([encode], encodeFace)
        faceDis = face_recognition.face_distance([encode], encodeFace)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex] :
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

def FaceRecognitionLiveCam():
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)

        frameS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameS = cv2.resize(frameS, (0, 0), fx=0.5, fy=0.5)
        CheckFrame(frameS,name,frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

def FaceRecognitionFromVideo():
    cap = cv2.VideoCapture('output.avi')
    while (cap.isOpened()):
        ret, frame = cap.read()
        frameS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameS = cv2.resize(frameS, (0, 0), fx=0.5, fy=0.5)
        CheckFrame(frameS,name,frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

def FaceRecognitionScreen():
    print(pyautogui.size()) #1024, 768
    # display screen resolution, get it from your OS settings
    SCREEN_SIZE = (1366, 768)

    # define the codec
    fourcc = cv2.VideoWriter_fourcc(*"XVID") #('m', 'p', '4', 'v') #

    # create the video write object
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, (SCREEN_SIZE))
    while True:
        img = pyautogui.screenshot()
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        # convert colors from BGR to RGB
        frameS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameS = cv2.resize(frameS, (0, 0), fx=0.5, fy=0.5)
        CheckFrame(frameS, "Culprit", frame)
        out.write(frame)
        if cv2.waitKey(1) == ord("q"):
            print('--')
            break
    cv2.destroyAllWindows()

FaceRecognitionFromVideo()
