import numpy as np
import cv2

cap = np.ones([500, 500, 3], 'uint8')*255
color = (255, 0, 0)
rad = 3
pressed = False


# Click Callback
def click(event, x, y, flags, param):
    global cap , color, pressed                           # WHILE RUNNING, GIVEN GLOBAL VARIABLE OUTSIDE
    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        cv2.circle(cap, (x, y), rad, color, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if pressed:
            cv2.circle(cap, (x, y), rad, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        pressed = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        if color == (0, 255, 0):
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)


# Window Initialization and CallBack Assignment
cv2.namedWindow("Drawing")                      # We have to set the same name window as we use in video capture
cv2.setMouseCallback("Drawing", click)          # Mapping the click event to mouse click

while True:
    cv2.imshow("Drawing", cap)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
