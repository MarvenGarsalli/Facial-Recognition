import numpy as np
import cv2
#from matplotlib import pyplot as plt
import time

face_cascade = cv2.CascadeClassifier('C:\Python27\Scripts\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Python27\Scripts\haarcascades\haarcascade_eye.xml')
#if you give 0 in second parameter while loading image using cv2.
#imread thenn no need to convert image using cvtColor, it is already loaded as grayscale image.
#img = cv2.imread('C:\im.jpg',0)
img = cv2.imread('C:\Python27\Scripts\Database\YasmieTchallenge.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #CV2.CV_FILLED :thickness
    roi_gray = gray[y:y+h, x:x+w] # Pick up a set of pixels
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
imgray = cv2.imread('C:\Python27\Scripts\Database\sachin.png',0)
cv2.imwrite("C:\Python27\Scripts\Database\YasmieTchallenge1.png",gray) 

cv2.waitKey(0)
cv2.destroyAllWindows()