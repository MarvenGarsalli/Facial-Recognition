import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('C:\Python27\Scripts\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

rec = cv2.createLBPHFaceRecognizer(); # Pour la reconnaissance du frame donnee par le Haar
rec.load("C:\Python27\Scripts\\recognizer\\trainningData.xml")

id=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,3, 1, 0, 4)

# path = "C:\Python27\Scripts\dataset"
# imagePaths= [os.path.join(path,f) for f in os.listdir(path)]

while cam.isOpened() :
	ret, frame = cam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if ret:
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
			id, conf= rec.predict(gray[y:y+h, x:x+w])
			print id
			#for imagePath in imagePaths:
				#if id == int(os.path.split(imagePath)[-1].split('.')[2]) :
			if id == 1:
				id= "Marven"
			elif id== 2:
				id= "WM"
			elif id== 3:
				id= "WM_son"
			else:
				id= "N/A"

			cv2.cv.PutText(cv2.cv.fromarray(frame), str(id), (x,y+h),font, 255)

	cv2.imshow("Face", frame)
	if(cv2.waitKey(19) == ord('q') ):
		break;
cam.release();
cv2.destroyAllWindows()