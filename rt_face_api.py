"""
Import required packages
"""
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

#My Libraries
import facial_recognition_models
import distances


"""
Import models from my model library
"""
detector = dlib.get_frontal_face_detector()

shape_predictor5 = facial_recognition_models.shape_predictor_5_model_location()
sp5 = dlib.shape_predictor(shape_predictor5)

shape_predictor68 = facial_recognition_models.shape_predictor_68_model_location()
sp68 = dlib.shape_predictor(shape_predictor68)

face_rec_model_path = facial_recognition_models.face_recognition_model_location()
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#GPU accelerated model for face detection
#Useless on my machine
face_detect_model_path = facial_recognition_models.cnn_face_detector_model_location()
face_detector = dlib.cnn_face_detection_model_v1(face_detect_model_path)


"""
Set Global Variables and Containers
"""
existing_users = {} #Dictionary mapping of username to index
threshold = .65 #Threshold value for distance calculations
db_activate = False
face_path = "faces.npy"
dist = .1

"""
DB STUFF_activate with bd_activate bool
"""
if(db_activate):

    import facial_recognition_models
    from app.models import User 

    users = User.query.all()
    for u in users:
        array_of_faces.append(u.descriptor)
        registered_users[u.id-1] = u.username

    def add_to_db(name, face_descriptor):
        new = User(username=name,descriptor=face_descriptor)
        db.session.add(new)
        db.session.commit()
        return
    
    def add_face_descriptor(name,face_descriptor):
        find = User.query.get(name)
        find.descriptor[-1] = face_descriptor
        
"""
Distance measurements
"""
def euclidean_distance(vector_x, vector_y):
    """
    Takes in two face_descriptor vectors
    returns the euclidean distance between
    en the two vectors
    """
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return math.sqrt(sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x))))

def manhattan_distance(vector_x,vector_y):
    if len(vector_x != vector_y):
        raise Exception('Vectors must be the same dimensions')
    return sum(abs(a-b) for a,b in zip(vector_x,vector_y))


def norm_dist(face_vectors, f_vector):
    """
    Takes in a list of face_descriptors and a face_descriptor vector
    returns the euclidean distance between each existing face_descriptor and the vector
    """
    if len(face_vectors) == 0:
        return np.empty((0))
    return np.linalg.norm(face_vectors - f_vector, axis=1)


def cosine_similarity(vector_x, vector_y):
    """
    Ouput bounded by 0 and 1, 0 being absolute similarity
    """
    if(len(vector_x)!=len(vector_y)):
        raise Exception('Vectors must be the same dimensions')
        
    return 1-np.dot(vector_x,vector_y)/(np.linalg.norm(vector_x)*np.linalg.norm(vector_y))
    
def minkowski_distance(vector_x, vector_y,p):
    if(len(vector_x)!=len(vector_y)):
        raise Exception('Vectors must be the same dimensions')
    root_pow=1/p
    return pow(sum((vector_x[dim] - vector_y[dim]) ** p for dim in range(len(vector_x))),root_pow)

def load_image_file(filename):
    """
    Takes a filename and returns an RGB pic 
    """
    return cv2.cvtColor(cv2.imread(filename,0),COLOR_GRAY2RGB)

def load_faces():
    global existing_users
    try: 
        face_file = np.load(face_path).item()
        existing_users = face_file
        print(existing_users)
    except: 
        print("[INFO] No existing faces in numpy dump, initializing empty dict")
        existing_users={}
        
def compare_faces(f_vector, threshold):
    """
    Takes a list of face_descriptors, a face_descriptor vector and a threshold value
    returns index of known face_vector if there is a match, returns false if there is no match
    """
    global existing_users
    
    if(not existing_users):
        return [False]
    
    detected = [False] * len(existing_users)
    ind = 0
    for f in existing_users:
        r = list(norm_dist(np.array(existing_users[f]),np.array(f_vector)) <= threshold)
        if(any(r)):
           detected[ind] = True
        ind+=1
    
    return detected

def compute_descriptor(img):
    """
    Takes and rgb image, assumes face is always present
    computes and returns single face_descriptor
    """
    rects = detector(img,1)
    shape = sp68(img,rects[0])
    comp = facerec.compute_face_descriptor(img,shape)
    a = []
    for i in comp: 
        a.append(i)
    return a

def add_2dict(name,face_descriptor):
    a = []
    ind = 0
    for fd in existing_users['A']:
        s = cosine_similarity(face_descriptor,fd)     
        print(ind, s)
        ind+=1

print("[INFO] opening camera stream")
vs = VideoStream().start() 
time.sleep(2.0)

print("[INFO] Loading existing face descriptors")
load_faces()

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
        shape = sp68(frame, rect)
        
        #Compute face descriptor
        backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        face_descriptor = facerec.compute_face_descriptor(backtorgb, shape)
        a = []
        for i in face_descriptor: 
            a.append(i)
        face_descriptor = a

        #Compare to all other existing face_descriptors 
        detected= compare_faces(face_descriptor, threshold)
        
        if True in detected:
            ind = [i for i, x in enumerate(detected) if x][0]
            add_2dict(str((list)(existing_users)[ind]),face_descriptor)
            cv2.putText(frame, str(list(existing_users)[ind]), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
        else:
            name = str(input("Please enter your name: "))
            if(name in existing_users):
                print("[INFO] adding face_descriptor to your profile")
                existing_users[name].append(face_descriptor)   
                
            else:
                print("[INFO] adding your profile to the system")
                existing_users[name] = []
                existing_users[name].append(face_descriptor)
                
        shape = face_utils.shape_to_np(shape)
        for(x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,255,0),-1)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    #if 't' key is pressed, ask for new threshold
    if key == ord("t"):
        threshold = float(input("Please set new threshold: "))

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
np.save(face_path, existing_users)
