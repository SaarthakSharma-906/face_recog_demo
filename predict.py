import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from joblib import load

# Load the FaceNet model and MTCNN for face detection
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False)

# Load the trained SVM model and label encoder
svm_model = load('svm_model.joblib')
label_encoder = load('label_encoder.joblib')

def get_embedding(face_pixels):
    face_pixels = Image.fromarray(face_pixels.astype('uint8'))  # Convert to uint8 before creating Image
    face_pixels = face_pixels.convert('RGB')
    
    # Convert to numpy array and normalize
    face_pixels = np.array(face_pixels).astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    
    # Avoid division by zero
    if std == 0:
        std = 1.0
    
    face_pixels = (face_pixels - mean) / std
    
    # Change data format from HWC to CHW
    face_pixels = np.transpose(face_pixels, (2, 0, 1))
    
    # Transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    
    # Make prediction to get embedding
    samples_tensor = torch.from_numpy(samples).float()
    embedding = facenet_model(samples_tensor).detach().numpy()
    return embedding[0]

def predict_identity(image_path):
    # Read the image
    img = Image.open(image_path)
    
    # Detect face
    face, prob = mtcnn(img, return_prob=True)
    
    if face is not None:
        # Convert tensor to numpy array
        face_np = face.permute(1, 2, 0).int().numpy()
        
        # Generate embedding
        embedding = get_embedding(face_np)
        
        # Normalize embedding
        embedding = embedding.reshape(1, -1)
        
        # Check for NaN values in embedding
        if np.isnan(embedding).any():
            print("Embedding contains NaN values. Skipping prediction.")
            return
        
        # Predict the identity
        prediction = svm_model.predict(embedding)
        identity = label_encoder.inverse_transform(prediction)[0]
        
        print(f'Predicted identity: {identity}')
    else:
        print('No face detected')

# Usage example
image_path = r'C:\Users\Saarthak\Desktop\Facial_recog_system\data\lfw-funneled\Aaron_Peirsol\Aaron_Peirsol_0001.jpg'  # Replace with the path to your image
predict_identity(image_path)
