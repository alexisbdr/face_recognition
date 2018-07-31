"""
Import required packages
"""
import datetime
import argparse
import imutils
import time
import dlib
import cv2 
import sys
import numpy as np
import math
import facial_recognition_models


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
registered_users = {} #Dictionary mapping of username to index
array_of_faces = [] #Array of face_descriptors in index order

threshold = .65 #Threshold value for distance calculations

db_activate = False


"""
DB STUFF_activate with bd_activate bool
"""
if(db_activate):

    from app import db
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

def compare_faces(f_vector):
    """
    Takes a list of face_descriptors, a face_descriptor vector and a threshold value
    returns index of known face_vector if there is a match, returns false if there is no match
    """
    return list(norm_dist(np.array(array_of_faces), np.array(f_vector)) <= threshold)

def load_image_file(filename):
    """
    Takes a filename and returns an RGB pic 
    """
    return cv2.cvtColor(cv2.imread(filename,0),cv2.COLOR_GRAY2RGB)

def load_descriptors(face_desc):
    try:
        iterator = iter(face_desc)
    except TypeError: 
        array_of_faces = []
    else:
        for fd in face_desc:
            array_of_faces.append(fd)


def compute_descriptor(img_path):
    """
    Takes an image, assumes face is always present
    computes and returns single face_descriptor
    """
    img = load_image_file(img_path)
 
    rects = detector(img,1)
  
    shape = sp68(img,rects[0])
  
    comp = facerec.compute_face_descriptor(img,shape)
   
    a = ','.join(str(i) for i in comp)
    
    return a


def run(img_path, face_descriptors):

    start_time = time.time() 

    load_descriptors(face_descriptors)

    start_time = time.time()
    img = load_image_file(img_path)
    print("loading image from file took : " ,(time.time() - start_time))
    
    start_time = time.time()
    face_descriptor = compute_descriptor(img)
    print("building face descriptor took : "  ,(time.time() - start_time))

    start_time = time.time()
    result = compare_faces(face_descriptor)
    print("comparison to existing faces took : "  ,(time.time() - start_time))

    if True in result:
        ind = [i for i, x in enumerate(result) if x][0]
        return registered_users[ind] 
    else:
        return False
