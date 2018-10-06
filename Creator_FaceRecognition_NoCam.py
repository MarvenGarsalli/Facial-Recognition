import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('C:\Python27\Scripts\haarcascades\haarcascade_frontalface_default.xml')

id = 0 #Name of the photo
frame = cv2.imread("Scripts\Database\Nada1.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("Face", frame)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print faces #<=> une liste de 4 elts
print faces #<=> une liste de 4 elts
for (x,y,w,h) in faces:
	#Take some samples of the gived photo
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
	id = raw_input("Please Choose an Id doesn't exist in the Dataset folder: ")
	name= raw_input("Type your Name: ")
	cv2.imwrite("C:\Python27\Scripts\Dataset\\"+ str(name)+"." +str(id)+".jpg",gray[y:y+h, x:x+w])
cv2.imshow("Face", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()