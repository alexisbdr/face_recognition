from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2 
import sys
import numpy as np
import math


"""
SET GLOBAL PATHS AND VARIABLES HERE
"""

#Global variables
registered_users = {} 
array_of_faces = []
threshold = .8

#Set global path variables for predictor and model
predictor_path = "dlib/predictors/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib/models/dlib_face_recognition_resnet_model_v1.dat"

#Create detector, shape_predictor and model objects for dlib use 
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def euclidean_distance(vector_x, vector_y):
    """
    Takes in two face_descriptor vectors
    returns the euclidean distance between the two vectors
    """
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return math.sqrt(sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x))))

def norm_dist(face_vectors, f_vector):
    """
    Takes in a list of face_descriptors and a face_descriptor vector
    returns the euclidean distance between each existing face_descriptor and the vector
    """
    if len(face_vectors) == 0:
        return np.empty((0))
    return np.linalg.norm(face_vectors - f_vector, axis=1)

def compare_faces(face_vectors, f_vector, threshold):
    """
    Takes a list of face_descriptors, a face_descriptor vector and a threshold value
    returns index of known face_vector if there is a match, returns false if there is no match
    """
    return list(norm_dist(np.array(face_vectors), np.array(f_vector)) <= threshold)



print("[INFO] opening camera stream")
vs = VideoStream().start() 
time.sleep(2.0)

print("Place face in webcam area")

# Now process the stream 
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 600 pixels, and convert it to
    # grayscale
    frame = vs.read()
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
        
        backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        face_descriptor = facerec.compute_face_descriptor(backtorgb, shape)
        
        result = compare_faces(array_of_faces, face_descriptor, threshold)
        if True in result:
            ind = [i for i, x in enumerate(result) if x][0]
            cv2.putText(frame, str(registered_users[ind]), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
        else:
            registered_users[len(array_of_faces)] = str(input("Please add identifier: "))
            array_of_faces.append(face_descriptor)
            print("New face added to list, now have ", len(array_of_faces))    
        
        shape = face_utils.shape_to_np(shape)

        for(x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,255,0),-1)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

