from . import face_recog


class Setup():

	def __init__(self, app):
		self.db = app.db #main driver database info

		self.faces = [] #storage of all faces in caches array of face obj
		self.registered_users = {} #Dictionary of driver to key values
		self.array_of_faces = []#Array of face_predictors, position in array is key value of dictionary
		self.threshold #Threshold value for normal distance
		
		self.load_all()

	def load_all(self):
		#Here replace with what our db looks like
		#We will save a file name for each of the drivers-->file of photo to be abe to load face_descriptor
		results = self.db.select('SELECT faces.id, faces.user_id, faces.filename, faces.created FROM faces')

		for row in results:

			user_id = row[1]
			filename = row[2]

			face = {
				"id" : row[0],
				"user_id" : user_id,
				"filename" : filename,
				"created": row[3]
			}

			self.faces.append(face)

            #face_image = rt_facial_recognition.load_image_file(filename)
            #face_image_descriptor = rt_facial_recognition.compute_descriptor(face_image)[0]
            index_key = len(self.array_of_faces)

            self.array_of_faces.append(face_image_descriptor)
            self.registered_users[index_key] = user_id

    def add(self):
    	#create add function 
    	#Adds entry to the database

    def delete(self):
    	#create delete function
    	#Deletes entry from the database
    	

