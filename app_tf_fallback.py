import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Disease classes with treatment information
DISEASE_INFO = {
    "Healthy": {
        "description": "No disease detected. The pig appears to be in good health.",
        "treatment": "Continue regular care with proper nutrition, clean housing, and routine health monitoring."
    },
    "African Swine Fever": {
        "description": "A highly contagious hemorrhagic viral disease affecting domestic and wild pigs.",
        "treatment": "No effective treatment or vaccine exists. Control measures include isolation of infected animals, culling, strict biosecurity, and proper disposal of carcasses. Notify veterinary authorities immediately as this is a reportable disease."
    },
    "Foot and Mouth Disease": {
        "description": "A highly contagious viral disease affecting cloven-hoofed animals, causing fever and blisters on the feet, mouth, and teats.",
        "treatment": "Supportive care including soft food, antibiotics to prevent secondary infections, anti-inflammatory medication, and foot care. Strict quarantine is essential. Vaccination may be used in some regions. Report to authorities as this is a notifiable disease."
    },
    "Porcine Reproductive and Respiratory Syndrome": {
        "description": "A viral disease causing reproductive failure in breeding stock and respiratory issues in young pigs.",
        "treatment": "Supportive care including good nutrition, temperature control, reducing stress, and antibiotics for secondary bacterial infections. Modified live virus vaccines are available for prevention. Implement strict biosecurity measures."
    },
    "Swine Influenza": {
        "description": "A respiratory disease caused by influenza type A viruses that regularly cause outbreaks in pigs.",
        "treatment": "Supportive care including proper ventilation, maintaining appropriate temperature, providing plenty of fresh water, and antibiotics for secondary bacterial infections. Commercial vaccines are available. Separate sick animals from the herd."
    }
}

# Extract class names for compatibility
DISEASE_CLASSES = list(DISEASE_INFO.keys())

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_image(image_path):
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # Change to your model's input size if different
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="models/detect.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        top_idx = int(np.argmax(output_data))
        confidence = float(output_data[top_idx]) * 100

        # Map index to disease name
        disease = DISEASE_CLASSES[top_idx]
        all_predictions = [(DISEASE_CLASSES[i], float(output_data[i]) * 100) for i in range(len(DISEASE_CLASSES))]
        all_predictions.sort(key=lambda x: x[1], reverse=True)

        return {
            "prediction": disease,
            "confidence": round(confidence, 2),
            "all_predictions": [(d, round(c, 2)) for d, c in all_predictions],
            "treatment": DISEASE_INFO[disease]["treatment"],
            "description": DISEASE_INFO[disease]["description"],
            "all_treatments": {disease: info for disease, info in DISEASE_INFO.items()}
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "prediction": "Error in analysis",
            "confidence": 0.0,
            "all_predictions": [(cls, 0.0) for cls in DISEASE_CLASSES],
            "treatment": "Unable to provide treatment information due to analysis error.",
            "description": "Error occurred during image analysis.",
            "all_treatments": {disease: info for disease, info in DISEASE_INFO.items()}
        }

@app.route('/')
def index():
    """Render the main page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make predictions"""
    if 'image' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Analyze the image
        result = analyze_image(file_path)
        
        # Render the result page
        return render_template('result.html',
                              filename=filename,
                              prediction=result['prediction'],
                              confidence=result['confidence'],
                              all_predictions=result['all_predictions'],
                              treatment=result['treatment'],
                              description=result['description'],
                              all_treatments=result['all_treatments'])
    else:
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 