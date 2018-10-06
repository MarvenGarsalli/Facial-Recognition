# ------------------------------ FACE RECOGNISER FOR ALL THE ALGORITHMS  ---------------------------------
# ---------------------------------- BY LAHIRU DINALANKARA AKA SPIKE ------------------------------------

import cv2                  #   Importing the opencv
import numpy as np          #   Import Numarical Python


# --- import the Haar cascades for face and eye ditection

face_cascade = cv2.CascadeClassifier('C:\Python27\Scripts\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
spec_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

# FACE RECOGNISER OBJECT
LBPH = cv2.createLBPHFaceRecognizer(2, 2, 7, 7, 20)
EIGEN = cv2.createEigenFaceRecognizer(10, 5000)
FISHER = cv2.createFisherFaceRecognizer(5, 500)

# Load the training data from the trainer to recognise the faces
LBPH.load("Scripts/Recognizer/trainingDataLBPH.xml")
EIGEN.load("Scripts/Recognizer/trainingDataEigan.xml")
FISHER.load("Scripts/Recognizer/trainingDataFisher.xml")

# ------------------------------------  PHOTO INPUT  -----------------------------------------------------

img = cv2.imread('C:\Python27\Scripts\Database\Helmi_haddad.jpg')                  # ------->>> THE ADDRESS TO THE PHOTO

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Convert the Camera to gray
faces = face_cascade.detectMultiScale(gray, 1.3, 4)         # Detect the faces and store the positions
cam = cv2.VideoCapture(0)

print(faces)

while cam.isOpened() :
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:                                  # Frames  LOCATION X, Y  WIDTH, HEIGHT
    
        Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))   # The Face is isolated and cropped

        ID, conf = LBPH.predict(Face)                           # LBPH RECOGNITION
        print("id_LBPH: "+str(ID))
        
        ID, conf = EIGEN.predict(Face)                          # EIGEN FACE RECOGNITION
        print("id_EIGEN: "+str(ID))

        ID, conf = FISHER.predict(Face)                         # FISHER FACE RECOGNITION
        print ID

    cv2.imshow('LBPH Face Recognition System', gray)           # IMAGE DISPLAY
    if(cv2.waitKey(19) == ord('q') ):
        break;
cv2.waitKey(0)
# ****************** Conclusion ********************************
# The EIGEN algo is the best and most exact algo
################################################################
cv2.destroyAllWindows()
