# import os
# import random
# from PIL import Image
# import numpy as np
# from mtcnn.mtcnn import MTCNN

# class FaceExtractor:
#     def __init__(self, required_size=(160, 160)):
#         self.required_size = required_size
#         self.detector = MTCNN()

#     def extract_face(self, filename):
#         image = Image.open(filename)
#         image = image.convert('RGB')
#         pixels = np.asarray(image)
#         results = self.detector.detect_faces(pixels)
#         if results:
#             x1, y1, width, height = results[0]['box']
#             x1, y1 = abs(x1), abs(y1)
#             x2, y2 = x1 + width, y1 + height
#             face = pixels[y1:y2, x1:x2]
#             image = Image.fromarray(face)
#             image = image.resize(self.required_size)
#             face_array = np.asarray(image)
#             return face_array
#         else:
#             return None

# class FaceDatasetLoader:
#     def __init__(self, directory):
#         self.directory = directory

#     def load_faces(self, subdir):
#         faces = []
#         path = os.path.join(self.directory, subdir)
#         file_list = os.listdir(path)
#         random.shuffle(file_list)
#         selected_files = file_list[:min(5, len(file_list))]
#         for filename in selected_files:
#             filepath = os.path.join(path, filename)
#             face = FaceExtractor().extract_face(filepath)
#             if face is not None:
#                 faces.append(face)
#         return faces

#     def load_dataset(self):
#         X, Y = [], []
#         for subdir in os.listdir(self.directory):
#             path = os.path.join(self.directory, subdir)
#             if os.path.isdir(path):  # Ensure it's a directory
#                 faces = self.load_faces(subdir)
#                 if faces:
#                     labels = [subdir for _ in range(len(faces))]
#                     print('>loaded %d examples for class: %s' % (len(faces), subdir))
#                     X.extend(faces)
#                     Y.extend(labels)
            
#         return np.asarray(X), np.asarray(Y)

# # Usage
# loader = FaceDatasetLoader('C:\\Users\\Saarthak\\Desktop\\Facial_recog_system\\data\\lfw-funneled')
# X, Y = loader.load_dataset()

# # Save the dataset
# np.savez_compressed('faces-dataset.npz', X, Y)


import os
import random
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

class FaceExtractor:
    def __init__(self, required_size=(160, 160)):
        self.required_size = required_size
        self.detector = MTCNN()

    def extract_face(self, filename):
        image = Image.open(filename)
        image = image.convert('RGB')
        pixels = np.asarray(image)
        results = self.detector.detect_faces(pixels)
        if results:
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize(self.required_size)
            face_array = np.asarray(image)
            return face_array
        else:
            return None

class FaceDatasetLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_faces(self, subdir):
        faces = []
        path = os.path.join(self.directory, subdir)
        file_list = os.listdir(path)
        random.shuffle(file_list)
        selected_files = file_list[:min(2, len(file_list))]
        for filename in selected_files:
            filepath = os.path.join(path, filename)
            face = FaceExtractor().extract_face(filepath)
            if face is not None:
                faces.append(face)
        return faces

    def load_dataset(self):
        X, Y = [], []
        for subdir in os.listdir(self.directory):
            path = os.path.join(self.directory, subdir)
            if os.path.isdir(path):  # Ensure it's a directory
                faces = self.load_faces(subdir)
                if faces:
                    labels = [subdir for _ in range(len(faces))]
                    print('>loaded %d examples for class: %s' % (len(faces), subdir))
                    X.extend(faces)
                    Y.extend(labels)
        return np.asarray(X), np.asarray(Y)

# Usage
loader = FaceDatasetLoader('C:\\Users\\Saarthak\\Desktop\\Facial_recog_system\\data\\lfw-funneled')
X, Y = loader.load_dataset()

# Save the dataset
np.savez_compressed('faces-dataset.npz', X, Y)

