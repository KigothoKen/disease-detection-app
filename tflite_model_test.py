import os
import numpy as np
from PIL import Image

def print_section(title):
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

# Check if the model file exists
model_path = os.path.join('models', 'detect.tflite')
print_section("Checking model file")
if os.path.exists(model_path):
    print(f"Model file found at: {model_path}")
    print(f"File size: {os.path.getsize(model_path)} bytes")
else:
    print(f"Model file not found at: {model_path}")
    
# Try to load PIL and NumPy
print_section("Testing PIL and NumPy")
try:
    # Create a test image
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    img_array = np.array(img)
    print(f"Successfully created image with shape {img_array.shape}")
    print(f"Image array data type: {img_array.dtype}")
    print(f"Mean pixel value: {np.mean(img_array)}")
except Exception as e:
    print(f"Error with PIL/NumPy: {str(e)}")

# Try to load the model through different methods
print_section("Attempting to load TFLite model")

# Method 1: Try using native TensorFlow if available
try:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Successfully loaded model with TensorFlow")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Model input details: {input_details}")
    print(f"Model output details: {output_details}")
    
except ImportError:
    print("TensorFlow not available")
except Exception as e:
    print(f"Error loading model with TensorFlow: {str(e)}")

# Method 2: Try using tflite_runtime if available
try:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Successfully loaded model with tflite_runtime")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Model input details: {input_details}")
    print(f"Model output details: {output_details}")
    
except ImportError:
    print("tflite_runtime not available")
except Exception as e:
    print(f"Error loading model with tflite_runtime: {str(e)}")

print_section("Alternative methods")
print("Since we can't load the TFLite model directly, we could:")
print("1. Use a feature extraction approach (color histograms, etc.)")
print("2. Run a simple image classifier based on basic features")
print("3. Use a web API that supports TensorFlow models if available") 