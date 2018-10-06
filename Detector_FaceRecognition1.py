import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('C:\Python27\Scripts\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

EIGEN = cv2.createEigenFaceRecognizer(10, 5000)
EIGEN.load("C:\Python27\Scripts\Recognizer\\trainningData_EigenFace.xml")

id=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,3, 1, 0, 4)

nameFound= 'N/A'
IdentPerson= 0
DataBasePerson = [ ("Helmi", 0),(str("Marven"), 1),("W_Miller", 2),("Yakin", 3), ("Masmoudi", 4), ("Helmi", 5), ("Helmi", 6) ]
IdentPerson= 7
path = "C:\Python27\Scripts\Dataset"
#imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
ctrInconnu = 0
imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
while cam.isOpened() :
	ret, frame = cam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 4)
	nameFound= 'N/A'
	ctr=0
	for (x,y,w,h) in faces: #Utile le cas du +ieurs personnes
		Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110)) 
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		id, conf= EIGEN.predict(Face)
		print id
		if id == -1:
			nameFound= 'N/A'
			ctrInconnu= ctrInconnu+ 1
		else:
			for (fr, nb) in DataBasePerson:
				if nb == id :
			 		nameFound=fr
			 		ctr= ctr+1
			 		break;
			 	if ctr == len(DataBasePerson):
			 		for imagePath in imagePaths:
			 			id_im = int(os.path.split(imagePath)[-1].split('.')[1])
			 			if id_im == id :  # If the image is in the datasetFolder, but not stored into DataBasePerson
			 				print("The new id: "+ str(id_im))
			 				nameFound= str(os.path.split(imagePath)[-1].split('.')[0])
			 				DataBasePerson.append((nameFound, id_im))

		cv2.cv.PutText(cv2.cv.fromarray(frame), nameFound, (x,y+h),font, 255)
		if nameFound == 'N/A' and ctrInconnu == 40:
			ctrInconnu=0
			name= raw_input("Type the Name : ")
			DataBasePerson.append((str(name), IdentPerson))
			cv2.imwrite("C:\Python27\Scripts\Dataset\User." + str(IdentPerson)+".jpg",gray[y:y+h, x:x+w])
			IdentPerson= IdentPerson+ 1
			os.system("python.exe Scripts\Trainner_FaceRecognition1.py" )
			cv2.waitKey(10)

	cv2.imshow("Face", frame)
	if(cv2.waitKey(19) == ord('q') ):
		break;
cam.release();
cv2.destroyAllWindows()