import face_recognition
import cv2
import os
import glob
import numpy as np
import json

class Recognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        self.frame_resizing = 0.25
    def load_encoding_images(self):
        self.known_face_encodings=np.genfromtxt('face_code.csv',delimiter=',')
        namefile=open('face_name.txt','r')
        self.known_face_names=namefile.read().splitlines()
        print("Data Loaded Successfully")
    def save_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))


        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            print('Loading :- ' ,filename)
            img_encoding = face_recognition.face_encodings(rgb_img,num_jitters=100,model='large')[0]
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print(type(self.known_face_names))
        print(type(self.known_face_encodings))
        np.savetxt('face_code.csv',self.known_face_encodings,delimiter=',')
        namefile=open('face_name.txt','w')
        for name in self.known_face_names:
            namefile.write(name+" \n")
        namefile.close()
        print("Encoding images Saved")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding,0.4)
            name = "Unknown"


            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)


        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
