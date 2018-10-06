# ********* All we need to do *************
# Create dataset(folder)  -> Dataset creator
# Train the recognizer    -> Trainner
# Detector                -> Detector
#******************************************

# ************** This is the Dataset creator  *******************
#################################################################
import numpy as np
import cv2

import time

face_cascade = cv2.CascadeClassifier('C:\Python27\Scripts\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

sampleNumber = 0
id = raw_input("Type identifier: ") #Fist w'll store the id<->Face, so next time we can identify the face 


while cam.isOpened() :
	ret, frame = cam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		sampleNumber = sampleNumber +1
		cv2.imwrite("C:\Python27\Scripts\dataset\User." + str(id)+"."+str(sampleNumber)+".jpg",gray[y:y+h, x:x+w])
		# *** datasetCreator created successfully at the end of the loop 
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

	cv2.imshow("Face", frame)
	if sampleNumber > 4 :
		break
	#if(cv2.waitKey(1) == ord('q') ):
		#break;
cam.release();
cv2.destroyAllWindows()