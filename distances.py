import numpy as np
import math
import time


"""

Distance measurements
"""
def euclidean_distance(vector_x, vector_y):
    """
    L2 distance 
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
    """
    Lp distance
    """
    if(len(vector_x)!=len(vector_y)):
        raise Exception('Vectors must be the same dimensions')
    root_pow=1/p
    return pow(sum((vector_x[dim] - vector_y[dim]) ** p for dim in range(len(vector_x))),root_pow)

