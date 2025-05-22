# src/ml_models/recognition_service.py
import numpy as np
import cv2 as cv
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import tensorflow as tf # Added for explicit TensorFlow operations if any, and to ensure it's listed

# --- Model and Asset Paths (relative to this file or absolute) ---
# Assuming models are in the same directory or a specified path accessible by the service
# For Flask, these paths should be relative to the application root or configured properly.
# When deploying with Flask, paths are tricky. Using paths relative to the current file is often safer for model loading.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACENET_WEIGHTS_PATH = os.path.join(BASE_DIR, "facenet_finetuned.weights.h5")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "face_embeddings_classes.npz")
HAARCASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
SVC_MODEL_PATH = os.path.join(BASE_DIR, "svc_model.pkl")

# --- Global Model Initialization ---
facenet_model = None
face_embeddings_data = None
label_encoder = None
haarcascade_classifier = None
svc_classifier = None
models_loaded = False

def load_recognition_models():
    global facenet_model, face_embeddings_data, label_encoder, haarcascade_classifier, svc_classifier, models_loaded
    if models_loaded:
        return

    try:
        # 1. FaceNet Model
        facenet_model = FaceNet().model
        if not os.path.exists(FACENET_WEIGHTS_PATH):
            raise FileNotFoundError(f"FaceNet weights not found at {FACENET_WEIGHTS_PATH}")
        facenet_model.load_weights(FACENET_WEIGHTS_PATH)
        print("FaceNet model and weights loaded successfully.")

        # 2. Embeddings and Label Encoder
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
        face_embeddings_data = np.load(EMBEDDINGS_PATH)
        label_encoder = LabelEncoder()
        label_encoder.fit(face_embeddings_data["arr_1"])
        print("Embeddings and LabelEncoder loaded successfully.")

        # 3. Haar Cascade Classifier
        if not os.path.exists(HAARCASCADE_PATH):
            raise FileNotFoundError(f"Haar Cascade XML not found at {HAARCASCADE_PATH}")
        haarcascade_classifier = cv.CascadeClassifier(HAARCASCADE_PATH)
        print("Haar Cascade classifier loaded successfully.")

        # 4. SVC Model
        if not os.path.exists(SVC_MODEL_PATH):
            raise FileNotFoundError(f"SVC model not found at {SVC_MODEL_PATH}")
        with open(SVC_MODEL_PATH, "rb") as f:
            svc_classifier = pickle.load(f)
        print("SVC model loaded successfully.")
        
        models_loaded = True
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        # Handle error appropriately, maybe raise an exception to stop the app if models are critical
        raise
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        raise

# Call loading function at module level, so it runs when the module is imported.
# This is common in Flask apps, but ensure it's handled correctly in the app context if issues arise.
load_recognition_models()

def recognize_face_from_image(image_np):
    """
    Recognizes a face from a given image (numpy array).
    Returns:
        - str: Name of the recognized person.
        - "Unknown": If a face is detected but not recognized.
        - "NO_FACE_DETECTED": If no face is detected in the image.
        - None: If models are not loaded or an error occurs during processing.
    """
    if not models_loaded:
        print("Error: Recognition models are not loaded.")
        return None # Or raise an exception

    try:
        frame_rgb = cv.cvtColor(image_np, cv.COLOR_BGR2RGB) # Ensure image is RGB for FaceNet
        frame_gray = cv.cvtColor(image_np, cv.COLOR_BGR2GRAY)
    except cv.error as e:
        print(f"OpenCV error during color conversion: {e}")
        return None # Indicate an error with image processing

    faces = haarcascade_classifier.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "NO_FACE_DETECTED"  # No faces detected

    # For simplicity, process the first detected face
    # In a real scenario, you might want to choose the largest face or handle multiple faces
    x, y, w, h = faces[0]
    face_roi_rgb = frame_rgb[y:y+h, x:x+w]

    try:
        # Preprocess for FaceNet
        face_resized = cv.resize(face_roi_rgb, (160, 160))
        face_normalized = (face_resized / 127.5) - 1 # Normalize to [-1, 1]
        face_expanded = np.expand_dims(face_normalized, axis=0) # Add batch dimension
    except cv.error as e:
        print(f"OpenCV error during face ROI processing: {e}")
        return None # Indicate an error with face processing

    # Get embeddings
    img_features = facenet_model.predict(face_expanded, verbose=0) # verbose=0 to suppress Keras logs per prediction

    # Predict with SVC
    face_pred_indices = svc_classifier.predict(img_features)
    
    # Convert prediction index to name
    # Ensure prediction is not empty and is valid for the encoder
    if len(face_pred_indices) > 0:
        try:
            predicted_name = label_encoder.inverse_transform(face_pred_indices)[0]
            # The original code implies that "Unknown" might be a class in your training data.
            # If "Unknown" is a specific class, it will be returned here.
            # If you want to add a confidence threshold, you'd use svc_classifier.predict_proba()
            # and then check if the max probability is above a certain threshold.
            # For now, we rely on the SVC model correctly classifying unknowns if trained to do so.
            return str(predicted_name) 
        except Exception as e:
            print(f"Error during label encoding inverse transform: {e}")
            return "Unknown" # Fallback if transform fails for some reason
    else:
        print("SVC prediction returned no result.")
        return "Unknown" # Fallback if SVC prediction is empty

# Example usage (for testing this module directly, not used by Flask app):
if __name__ == "__main__":
    # This part would require a sample image to test
    print("Recognition service module loaded. Models should be loaded if paths are correct.")
    # To test: 
    # Create a dummy numpy image or load one using cv.imread()
    # img = cv.imread("path_to_test_image.jpg")
    # if img is not None:
    #     result = recognize_face_from_image(img)
    #     print(f"Recognition Result: {result}")
    # else:
    #     print("Failed to load test image.")

