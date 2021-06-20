import cv2
import face_recognition

img1 = face_recognition.load_image_file('Images/GOT2.jpg')
img2 = face_recognition.load_image_file('Images/fam.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img1)
encode1 = face_recognition.face_encodings(img1,faceLoc)
for faces in faceLoc:
    cv2.rectangle(img1,(faces[3],faces[0]),(faces[1],faces[2]),(255,0,255),2)

faceLocTestall = face_recognition.face_locations(img2)
encode2all = face_recognition.face_encodings(img2,faceLocTestall)
for faceLocTest,encode2 in zip(faceLocTestall,encode2all):
    results = face_recognition.compare_faces(encode1, encode2)
    faceDis = face_recognition.face_distance(encode1, encode2)
    print(results, faceDis)

    for x in results:
        if x:
            cv2.rectangle(img2, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
            cv2.putText(img2, f'{"Match Found"} {round(faceDis[0], 2)}', (faceLocTest[3], faceLocTest[0] - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)

cv2.imshow('Search: ', img1)
cv2.imshow('FindPicture: ', img2)
cv2.waitKey(0)