from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2 
import sys

#dictionary of faces 
faces = {} 

if len(sys.argv) > 1:
    print(
        "Call this program like this:\n"
        "   ./facial_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

print("[INFO] loading facial landmark detector, facial shape detector and recognition model")

predictor_path = "dlib/predictors/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib/models/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# grab the frame from the threaded video stream, resize it to
# have a maximum width of 600 pixels, and convert it to
# grayscale
frame = cv2.imread("marc.jpg")
frame = imutils.resize(frame, width=600)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# detect faces in the grayayscale frame
rects = detector(gray, 0)

# loop over the faces that have been found
for rect in rects:
    #1.Draw rectangles over detected faces
    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)

    # Get the landmarks/parts for the face in box d.
    shape = sp(gray, rect)
    
    face_descriptor = facerec.compute_face_descriptor(frame, shape)
    text_file = open("myface.txt", "w")
    text_file.write("\n".join(str(face_descriptor)))
    text_file.close()


    #if(face_descriptor in faces):
    	#cv2.putText(frame, faces[face_descriptor], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
	#0.75, (0, 255, 0), 2)
    #else:
    	#name = input("Please add your name id : ")
    	#faces[face_descriptor] = name

    shape = face_utils.shape_to_np(shape)

    for(x,y) in shape:
        cv2.circle(frame,(x,y),1,(0,255,0),-1)

# show the frame
cv2.imshow("Frame", frame)
key = cv2.waitKey(1) & 0xFF

# if the `q` key was pressed, break from the loop
#if key == ord("q"):
    #break

# do a bit of cleanup
cv2.destroyAllWindows()

