import numpy as np
import os
from PIL import Image
import tensorflow as tf
import tflite_runtime.interpreter as tflite

# Define possible diseases (replace with actual class names from your model)
DISEASE_CLASSES = [
    "Healthy",
    "African Swine Fever",
    "Foot and Mouth Disease",
    "Porcine Reproductive and Respiratory Syndrome",
    "Swine Influenza"
]

class PigDiseaseClassifier:
    def __init__(self, model_path):
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load the TFLite model
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            self.input_shape = self.input_details[0]['shape']
            self.img_height = self.input_shape[1]
            self.img_width = self.input_shape[2]
            
            print(f"Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            raise Exception(f"Failed to load the model: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess the image to match the model's input requirements"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize the image (scale pixel values to [0,1])
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def predict(self, image_path):
        """Make a prediction on the given image"""
        try:
            # Preprocess the image
            preprocessed_img = self.preprocess_image(image_path)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_img)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensor
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process the predictions
            prediction_probabilities = predictions[0]
            
            # Get the top prediction
            top_prediction_index = np.argmax(prediction_probabilities)
            top_prediction_confidence = prediction_probabilities[top_prediction_index] * 100
            
            # Get all predictions
            all_predictions = []
            for i, prob in enumerate(prediction_probabilities):
                if i < len(DISEASE_CLASSES):
                    disease_name = DISEASE_CLASSES[i]
                else:
                    disease_name = f"Class_{i}"
                all_predictions.append((disease_name, round(prob * 100, 2)))
            
            # Sort predictions by confidence (descending)
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get the predicted class name
            if top_prediction_index < len(DISEASE_CLASSES):
                predicted_disease = DISEASE_CLASSES[top_prediction_index]
            else:
                predicted_disease = f"Unknown Class {top_prediction_index}"
            
            return {
                "prediction": predicted_disease,
                "confidence": round(top_prediction_confidence, 2),
                "all_predictions": all_predictions
            }
        
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}") 