import os
import cv2 
import numpy
from PIL import Image

EIGEN = cv2.createEigenFaceRecognizer(15)   # creating EIGEN FACE RECOGNISER

path = "C:\Python27\Scripts\Dataset"

def getImagesWithID(path):
	imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
	faces=[]
	IDs  =[]

	for imagePath in imagePaths:
		# convert color image to black & white: L = R * 299/1000 + G * 587/1000 + B * 114/1000
		faceImg = Image.open(imagePath).convert('L')#Image.open(imagePath,'L')
		faceImg = faceImg.resize((110,110))         ## resize the image so the EIGEN recogniser can be trained
		#print faceImg #<PIL.Image.Image image mode=L size=181x181 at 0x3DE5550>
		faceNp = numpy.array(faceImg, "uint8")

		ID=int(os.path.split(imagePath)[-1].split('.')[1]) #Get the photo id # -1 to get the last elt of the imagePath
		print ID
		faces.append(faceNp) #Store numpy array(im) in the list 
		IDs.append(ID)
		cv2.imshow('trainning', faceNp)
		cv2.waitKey(1)
	return numpy.array(IDs), faces

print('TRAINING......')
Ids, faces = getImagesWithID(path)
EIGEN.train(faces, Ids)
print('EIGEN FACE RECOGNISER COMPLETE...')
EIGEN.save('Scripts\Recognizer\\trainningData_EigenFace.xml')
print('FILE SAVED..')

cv2.destroyAllWindows()