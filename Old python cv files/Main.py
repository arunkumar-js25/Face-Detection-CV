import numpy as np
import cv2

# Useful Sites
'''
https://docs.opencv.org/3.4/index.html
'''


def ImageDetails():
    img: None = cv2.imread(r"C:\Users\arunkumar.j06\Desktop\logo.png", 1)
    # print(img)
    print(type(img))
    print(len(img))
    print(len(img[0]))
    print("no of channels in first row ( RGB ). if transparency, value will be x+1 :", len(img[0][0]))
    print(img.dtype)
    print("One Channel image with all pixel in it: ", img[:, :, 0])
    print("total num of pixels: ", img.size)


def Display_Image(img):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def Write_Image(img):
    cv2.imwrite("output.jpg", img)


def datastructure():
    black = np.zeros([200, 400, 1], 'uint8')  # height, width, channels
    cv2.imshow('Black', black)
    print('Pixels and channels in first row: ', black[0, 0, :])

    Ones = np.ones([200, 400, 3], 'uint8')
    cv2.imshow('Ones', Ones)
    print('Pixels and channels in first row: ', Ones[0, 0, :])

    white = Ones.copy()
    white *= 2 ** 16 - 1
    cv2.imshow('white', white)
    print('Pixels and channels in first row: ', white[0, 0, :])

    color = Ones.copy()
    color[:, :] = (255, 0, 0)
    cv2.imshow('color', color)
    print('Pixels and channels in first row: ', color[0, 0, :])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ImageColorChange():
    color = cv2.imread("output.jpg", 1)  # 1 should be the default
    cv2.imshow("Color", color)
    cv2.moveWindow("Color", 0, 0)

    print("Height, Width, Channels :", color.shape)
    height, width, channels = color.shape

    b, g, r = cv2.split(color)
    rgb_split = np.empty([height, width * 3, 3], 'uint8')
    rgb_split[:, 0:width] = cv2.merge([b, b, b])
    rgb_split[:, width:width * 2] = cv2.merge([g, g, g])
    rgb_split[:, width * 2:width * 3] = cv2.merge([r, r, r])
    cv2.imshow("Channels", rgb_split)
    cv2.moveWindow("Channels", 0, height + 100)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_split = np.concatenate((h, s, v), axis=1)
    cv2.imshow("Split HSV", hsv_split)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def PixelManipulation():
    color = cv2.imread("output.jpg", 1)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Gray", gray)

    b = color[:, :, 0]
    g = color[:, :, 1]
    r = color[:, :, 2]

    rgba = cv2.merge((b, g, r, g))
    cv2.imshow("RGBA", rgba)
    cv2.imwrite("rgba.png", rgba)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Blur_Dilation_Erosion():
    image = cv2.imread("output.jpg", 1)  # if 0 - load the image in black&white
    cv2.imshow("Original", image)

    blur = cv2.GaussianBlur(image, (5, 5), 5)
    cv2.imshow("Blur", blur)

    kernel = np.ones((2, 5), 'uint8')  # Kernel used to be small pixel size - size of brush
    dilate = cv2.dilate(image, kernel, iterations=1)
    erode = cv2.erode(image, kernel, iterations=1)

    cv2.imshow("Dilate", dilate)
    cv2.imshow("Erode", erode)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scaling_and_rotate():
    image = cv2.imread("output.jpg")

    # Scale
    img_half = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    img_stretch = cv2.resize(image, (600, 600))
    img_stretch_near = cv2.resize(image, (600, 600), interpolation=cv2.INTER_NEAREST)

    # cv2.imshow("Half", img_half)
    # cv2.imshow("Stretch", img_stretch)
    # cv2.imshow("Stretch_Near", img_stretch_near)

    # Rotation
    M = cv2.getRotationMatrix2D((0, 0), -30, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))  # width * Height
    cv2.imshow("Rotated", rotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def VideoCapture():
    cap = cv2.VideoCapture(0)

    color = (0, 255, 0)
    line_width = 3
    radius = 20
    point = (0, 0)

    def click(event, x, y, flags, param):
        global point  # WHILE RUNNING, GIVEN GLOBAL VARIABLE OUTSIDE
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Pressed: ', x, y)
            point = (x, y)

    cv2.namedWindow("Frame")  # We have to set the same name window as we use in video capture
    cv2.setMouseCallback("Frame", click)  # Mapping the click event to mouse click

    while (True):
        ret, frame = cap.read()

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.circle(frame, point, radius, color, line_width)
        cv2.imshow("Frame", frame)

        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def SimpleThresholding():  # make binary based on threshold color limit
    image = cv2.imread("output.jpg", 0)  # Open image in black and white ( 0 )
    height, width = image.shape[:2]
    cv2.imshow("Image", image)

    # 1. Threshold Comparison  - Slow Binary
    threshold = 85
    binary = np.zeros([height, width, 1], 'uint8')  # Using 1 channel becoz b&W
    for x in range(height):
        for y in range(width):
            if (image[x][y] > threshold):
                binary[x][y] = 255
    cv2.imshow("Binary", binary)

    # 2. Threshold Comparison Easily
    ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("CV Threshold", thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def AdaptiveThreshold():
    image = cv2.imread("output.jpg", 0)
    cv2.imshow("Image", image)

    ret, thresh_basic = cv2.threshold(image, 85, 255, cv2.THRESH_BINARY)
    cv2.imshow("Basic Binary Threshold", thresh_basic)

    thresh_adapt = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow("Adapt Threshold", thresh_adapt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def SkinDetection():
    image = cv2.imread(r"C:\Users\arunkumar.j06\Desktop\fam.png", 1)
    cv2.imshow("Image", image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hsv_split = np.concatenate((h, s, v), axis=1)
    # cv2.imshow("HSV", hsv_split)

    ret, min_sat = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Sat Filter", min_sat)

    ret, max_hue = cv2.threshold(h, 30, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("Hue Filter", max_hue)

    final = cv2.bitwise_and(min_sat, max_hue)
    cv2.imshow("Final Filter", final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contoursAlgo():  # An Iterative Algorithm - based on neighbourhood (N4,N6,N8) and Connectedness
    image = cv2.imread("output.jpg") #r"C:\Users\arunkumar.j06\Desktop\fam.png", 1)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow("Binary", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image2 = image.copy()
    index = -1
    thickness = 4
    color = (255, 0, 255)

    # Contour Drawing on photo
    # cv2.drawContours(image2, contours, index, color, thickness)
    # cv2.imshow("Contour", image2)

    objects = np.zeros([image.shape[0], image.shape[1], 3], 'uint8')
    for c in contours:
        cv2.drawContours(objects, [c], -1, color, -1)

        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        M = cv2.moments(c)
        try:
            cx = int(M['m10'] / M['m00'])
        except:
            cx = 0

        try:
            cy = int(M['m01'] / M['m00'])
        except:
            cy = 0
        cv2.circle(objects, (cx, cy), 3, (0, 0, 255), -1)

        print('Area: {}, parameters: {}'.format(area, perimeter))

    cv2.imshow("Contours: ", objects)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edgedetect():
    image = cv2.imread(r"C:\Users\arunkumar.j06\Desktop\fam.png", 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    res, thresh = cv2.threshold(hsv[:,:,0], 25,255, cv2.THRESH_BINARY_INV) # iNV - include hue value less than 25
    cv2.imshow("Threshold", thresh)

    edge = cv2.Canny(image, 100, 70)
    cv2.imshow("Canny", edge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# DETECTION != RECOGNITION
# Face Detection
'''
1. template matching
2. Face Detection - Haar Cascade Method
'''
def templatematching():
    template = cv2.imread(r"Images\GOT2.jpg",1)
    frame = cv2.imread(r"Images\emilia2.jpg", 1)

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    cv2.circle(result,max_loc,15,255,2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Template", template)
    cv2.imshow("Result",result)

    print(max_val, max_loc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def facedetection():
    img = cv2.imread("Images/fam.png",1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    path = "haarcascades/haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40,40))
    print(len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

templatematching()