import os
import cv2 
import numpy
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer();# Local Binary Patterns Histograms recognizer
path = "C:\Python27\Scripts\dataset"

def getImagesWithID(path):
	#Create a list of all image path in dataset
	imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
	#print imagePaths
	faces=[]
	IDs  =[]
	for imagePath in imagePaths:
		# convert color image to black & white: L = R * 299/1000 + G * 587/1000 + B * 114/1000
		faceImg = Image.open(imagePath).convert('L')#Image.open(imagePath,'L')
		#print faceImg #<PIL.Image.Image image mode=L size=181x181 at 0x3DE5550>
		faceNp = numpy.array(faceImg, "uint8")
		#cv2.imshow("nnn",faceNp)
		ID=int(os.path.split(imagePath)[-1].split('.')[1]) #Get the photo id # -1 to get the last elt of the imagePath
		#ID=os.path.split(imagePath)[-1].split('.')[1]
		faces.append(faceNp) #Store numpy array(im) in the list 
		print ID
		IDs.append(ID)
		cv2.imshow('trainning', faceNp)
		cv2.waitKey(10)
	return IDs, faces


Ids, faces = getImagesWithID(path)
# Guess the id of the current photo
recognizer.train(faces, numpy.array(Ids))#Transform the py array to numpy array(still simple array in this case)
recognizer.save('C:\Python27\Scripts\\recognizer\\trainningData.yml')
cv2.destroyAllWindows()