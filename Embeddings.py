import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.model_selection import train_test_split

class FaceEmbeddingGenerator:
    def __init__(self, data_path, output_path,test_size):
        self.data_path = data_path
        self.output_path = output_path
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.mtcnn = MTCNN(keep_all=True)
        self.trainX = None
        self.trainy = None
        self.testX = None
        self.testy = None
        self.test_size = test_size

    def load_data(self):
        data = np.load(self.data_path, allow_pickle=True)
        self.trainX = data['arr_0']  # Face embeddings
        self.trainy = data['arr_1']  # Labels

        # Split the data into training and test sets
        self.trainX, self.testX, self.trainy, self.testy = train_test_split(
            self.trainX, self.trainy, test_size=self.test_size, random_state=42
        )
        
        print('Training set: ', self.trainX.shape, self.trainy.shape)
        print('Test set: ', self.testX.shape, self.testy.shape)

    def get_embedding(self, face_pixels):
        face_pixels = Image.fromarray(face_pixels)
        face_pixels = face_pixels.convert('RGB')
        
        # Convert to numpy array and normalize
        face_pixels = np.array(face_pixels).astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        
        # Change data format from HWC to CHW
        face_pixels = np.transpose(face_pixels, (2, 0, 1))
        
        # Transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        
        # Make prediction to get embedding
        samples_tensor = torch.from_numpy(samples).float()
        embedding = self.model(samples_tensor).detach().numpy()
        return embedding[0]

    def generate_embeddings(self):
        newtrainX = []
        for face_pixels in self.trainX:
            embedding = self.get_embedding(face_pixels)
            newtrainX.append(embedding)
        newtrainX = np.asarray(newtrainX)

        if self.testX is not None:
            newtestX = []
            for face_pixels in self.testX:
                embedding = self.get_embedding(face_pixels)
                newtestX.append(embedding)
            newtestX = np.asarray(newtestX)
        else:
            newtestX = None

        np.savez_compressed(self.output_path, newtrainX, self.trainy, newtestX, self.testy)
        print(f'Embeddings saved to {self.output_path}')

# Usage
data_path = r'C:\\Users\\Saarthak\\Desktop\\Facial_recog_system\\faces-dataset.npz'
output_path = 'faces-embeddings.npz'

face_embedding_generator = FaceEmbeddingGenerator(data_path, output_path,test_size = 0.1)
face_embedding_generator.load_data()
face_embedding_generator.generate_embeddings()
