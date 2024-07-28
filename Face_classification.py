import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot

class FaceDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_dataset()

    def load_dataset(self):
        data = np.load(self.dataset_path, allow_pickle=True)
        self.trainX = data['arr_0']  # Face embeddings for training
        self.trainy = data['arr_1']  # Labels for training
        self.testX = data['arr_2']   # Face embeddings for testing
        self.testy = data['arr_3']   # Labels for testing

        print('Training set: ', self.trainX.shape, self.trainy.shape)
        print('Test set: ', self.testX.shape, self.testy.shape)

    def normalize_input_vectors(self):
        input_encoder = Normalizer(norm='l2')
        self.trainX = input_encoder.fit_transform(self.trainX)
        self.testX = input_encoder.transform(self.testX)

    def label_encode_targets(self):
        # Fit the LabelEncoder on both training and testing labels
        combined_labels = np.concatenate((self.trainy, self.testy))
        output_encoder = LabelEncoder()
        output_encoder.fit(combined_labels)

        # Transform labels
        self.trainy = output_encoder.transform(self.trainy)
        self.testy = output_encoder.transform(self.testy)
        self.output_encoder = output_encoder  # Store for inverse transformation

class FaceRecognitionModel:
    def __init__(self, kernel='linear', probability=True):
        self.kernel = kernel
        self.probability = probability
        self.model = SVC(kernel=self.kernel, probability=self.probability)

    def fit(self, trainX, trainy):
        self.model.fit(trainX, trainy)

    def predict(self, testX):
        return self.model.predict(testX)

    def predict_proba(self, testX):
        return self.model.predict_proba(testX)

class FaceRecognitionUtilities:
    def __init__(self, output_encoder):
        self.output_encoder = output_encoder

    def evaluate_model(self, testX, testy, model):
        y_pred = model.predict(testX)
        accuracy = accuracy_score(testy, y_pred)
        print(f'Accuracy: {accuracy:.3f}')
        

# Usage
dataset_path = r'C:\Users\Saarthak\Desktop\Facial_recog_system\faces-embeddings.npz'
dataset = FaceDataset(dataset_path)
dataset.normalize_input_vectors()
dataset.label_encode_targets()

model = FaceRecognitionModel(kernel='linear', probability=True)
model.fit(dataset.trainX, dataset.trainy)

# Evaluate the model
utilities = FaceRecognitionUtilities(dataset.output_encoder)
utilities.evaluate_model(dataset.testX, dataset.testy, model)
