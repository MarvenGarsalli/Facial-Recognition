import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

face_cascade = cv2.CascadeClassifier("C:\Python27\Scripts\haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:\Python27\Scripts\haarcascades\haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
print "Waiting the cam to be opened ..."
while(not cap.isOpened()):
	continue;

print "Cam successfully opened"
while(cap.isOpened()):
	now =  time.strftime("%d-%m-%Y_%H.%M.%S")
    #path = "C:\Python27\Scripts\Database\photo_"+now+".png"
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame, 1.3, 5)

	for (x,y,w,h) in faces :
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
		face_im_gray = gray[y:y+h, x:x+w]
		face_im      = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(face_im_gray)
		for (dx,dy,dw,dh) in eyes:
			cv2.rectangle(face_im,(dx,dy),(dx+dw,dy+dh),(0,255,0),2)
			cv2.rectangle(face_im_gray,(dx,dy),(dx+dw,dy+dh),(0,255,0),2)

	cv2.imshow('img', frame);
	#cv2.imshow('img', gray);
	if(cv2.waitKey(1) == ord('q') ):  #cv2.waitKey(1): non bloquante:create a thread(#waitKey(0))
		cv2.imwrite("C:\Python27\Scripts\Database\gray_"+now+".png", gray)
		break
	if(cv2.waitKey(2) == ord(' ') ):  #cv2.waitKey(1): non bloquante:create a thread(#waitKey(0))
		cv2.imwrite("C:\Python27\Scripts\Database\gray_"+now+".png",gray)
		print "New Image have just been registred under C:\Python27\Scripts\Database"

cv2.imwrite("C:\Python27\Scripts\Database\\"+now+".png", frame)
cap.release()
#cv2.waitKey(0)
cv2.destroyAllWindows();