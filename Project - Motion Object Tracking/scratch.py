import imutils
import cv2

frame = cv2.imread("Images/2.jpg")
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
#pts = deque(maxlen=buffer)

frame = imutils.resize(frame, width=600)
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",hsv)

mask = cv2.inRange(hsv, greenLower, greenUpper)
cv2.imshow("mask",mask)
cv2.waitKey(0)
