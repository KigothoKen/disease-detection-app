import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tflite_utils import TFLiteInterpreter

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Add Jinja template filter for symptom categories
@app.template_filter('get_symptom_categories')
def get_symptom_categories(symptom):
    """Categorize symptoms for filtering in the UI"""
    categories = ['all']
    
    symptom_text = symptom.lower()
    if any(word in symptom_text for word in ['fever', 'temperature', 'heat']):
        categories.append('fever')
    if any(word in symptom_text for word in ['skin', 'discoloration', 'lesion', 'purple', 'red', 'blue']):
        categories.append('skin')
    if any(word in symptom_text for word in ['breath', 'cough', 'respiratory', 'nasal', 'sneez']):
        categories.append('respiratory')
    if any(word in symptom_text for word in ['appetite', 'feed', 'eat', 'weight', 'food']):
        categories.append('appetite')
    
    return ' '.join(categories)

# Model path
MODEL_PATH = os.path.join('models', 'detect.tflite')

# Define symptoms for each disease
DISEASE_SYMPTOMS = {
    "Healthy": [],
    "African Swine Fever": [
        "High fever (40.5-42°C)",
        "Loss of appetite",
        "Red/purple discoloration of skin",
        "Diarrhea, sometimes bloody",
        "Vomiting",
        "Weakness or unwillingness to stand",
        "Abortion in pregnant sows"
    ],
    "Foot and Mouth Disease": [
        "Fever",
        "Blisters on feet, mouth, and teats",
        "Excessive salivation",
        "Lameness/reluctance to move",
        "Reduced appetite",
        "Depression",
        "Sudden death in young animals"
    ],
    "Porcine Reproductive and Respiratory Syndrome": [
        "Respiratory distress",
        "Poor growth performance",
        "Premature farrowing",
        "Abortion",
        "Stillbirths",
        "Weak piglets",
        "Blue discoloration of ears"
    ],
    "Swine Influenza": [
        "Sudden onset of fever",
        "Coughing",
        "Sneezing",
        "Nasal discharge",
        "Breathing difficulties",
        "Lethargy",
        "Reduced feed intake"
    ]
}

# Get all unique symptoms
ALL_SYMPTOMS = []
for symptoms in DISEASE_SYMPTOMS.values():
    ALL_SYMPTOMS.extend([s for s in symptoms if s not in ALL_SYMPTOMS])
ALL_SYMPTOMS.sort()

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

# Initialize the TFLite interpreter
try:
    interpreter = TFLiteInterpreter(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    interpreter = None

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_symptom_score(reported_symptoms):
    """Calculate symptom match score for each disease"""
    if not reported_symptoms:
        return {}
    
    scores = {}
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        if not symptoms:  # Skip "Healthy" as it has no symptoms
            continue
            
        # Count matching symptoms
        matches = sum(1 for s in reported_symptoms if s in symptoms)
        
        # Calculate match percentage (how many of the reported symptoms match this disease)
        if len(reported_symptoms) > 0:
            match_percentage = matches / len(reported_symptoms)
        else:
            match_percentage = 0
            
        # Calculate coverage percentage (how many of this disease's symptoms are reported)
        if len(symptoms) > 0:
            coverage_percentage = matches / len(symptoms)
        else:
            coverage_percentage = 0
            
        # Weighted score (match_percentage ensures reported symptoms are relevant,
        # coverage_percentage ensures enough symptoms are present for the disease)
        scores[disease] = 0.7 * match_percentage + 0.3 * coverage_percentage
    
    # Special case for Healthy - high score when no symptoms match any disease
    if sum(scores.values()) == 0:
        scores["Healthy"] = 1.0 * 0.5  # Reduce Healthy score by half
    else:
        # Inverse of the average of other disease scores
        non_healthy_avg = sum(scores.values()) / len(scores)
        scores["Healthy"] = max(0, (1.0 - non_healthy_avg) * 0.5)  # Reduce Healthy score by half
    
    return scores

def analyze_image(image_path, reported_symptoms=None):
    """
    Analyze the image using our TFLite model or fallback to image processing if model is not available
    """
    try:
        if interpreter is None:
            raise Exception("TFLite model not available")
        
        # Get predictions from the model
        predictions = interpreter.predict(image_path)
        
        # Adjust predictions based on reported symptoms
        if reported_symptoms:
            # Get symptom scores for each disease
            symptom_scores = calculate_symptom_score(reported_symptoms)
            
            # Combine image and symptom predictions
            # Original weights: 70% model, 30% symptoms (up to 50%)
            # New weights: 30% model, 70% symptoms (up to 80%)
            symptom_weight = min(0.8, 0.7 + (len(reported_symptoms) / 30))
            model_weight = 1.0 - symptom_weight
            
            # Create new combined predictions
            combined_predictions = np.zeros_like(predictions)
            for i, disease in enumerate(DISEASE_CLASSES):
                model_score = predictions[i]
                symptom_score = symptom_scores.get(disease, 0.0)
                combined_predictions[i] = (model_weight * model_score) + (symptom_weight * symptom_score)
            
            # Normalize to sum to 1
            if np.sum(combined_predictions) > 0:
                combined_predictions = combined_predictions / np.sum(combined_predictions)
            
            # Use the adjusted predictions
            predictions = combined_predictions
        
        # Further reduce weight of "Healthy" category
        healthy_idx = DISEASE_CLASSES.index("Healthy")
        predictions[healthy_idx] *= 0.5
        
        # Re-normalize predictions to sum to 1
        if np.sum(predictions) > 0:
            predictions = predictions / np.sum(predictions)
        
        # Get the top prediction
        top_idx = np.argmax(predictions)
        top_confidence = float(predictions[top_idx]) * 100
        top_disease = DISEASE_CLASSES[top_idx]
        
        # Format all predictions
        all_predictions = [(DISEASE_CLASSES[i], float(predictions[i]) * 100) for i in range(len(DISEASE_CLASSES))]
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Create matching symptoms dict for each disease
        matching_symptoms = {}
        if reported_symptoms:
            for disease, symptoms in DISEASE_SYMPTOMS.items():
                matching_symptoms[disease] = [s for s in reported_symptoms if s in symptoms]
        
        return {
            "prediction": top_disease,
            "confidence": round(top_confidence, 2),
            "all_predictions": [(d, round(c, 2)) for d, c in all_predictions],
            "treatment": DISEASE_INFO[top_disease]["treatment"],
            "description": DISEASE_INFO[top_disease]["description"],
            "all_treatments": {disease: info for disease, info in DISEASE_INFO.items()},
            "model_used": "TFLite Model" + (" with Symptom Analysis" if reported_symptoms else ""),
            "reported_symptoms": reported_symptoms or [],
            "matching_symptoms": matching_symptoms
        }
    except Exception as e:
        print(f"Error analyzing image with TFLite: {str(e)}")
        print("Falling back to image feature analysis")
        
        # Fallback to basic image analysis
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Basic image features
        r_channel = img_array[:,:,0]
        g_channel = img_array[:,:,1]
        b_channel = img_array[:,:,2]
        
        avg_r = np.mean(r_channel)
        avg_g = np.mean(g_channel)
        avg_b = np.mean(b_channel)
        
        # Brightness and ratios
        brightness = (avg_r + avg_g + avg_b) / 3
        red_ratio = avg_r / (avg_g + avg_b + 1e-5)
        
        # Simplified analysis - get scores for each disease
        scores = {}
        scores["Healthy"] = (1 - abs(red_ratio - 1.0)) * 0.5  # Reduce Healthy score by half
        scores["African Swine Fever"] = red_ratio - 0.8
        scores["Foot and Mouth Disease"] = abs(brightness / 255 - 0.4)
        scores["Porcine Reproductive and Respiratory Syndrome"] = abs(brightness / 255 - 0.6)
        scores["Swine Influenza"] = abs(1 - red_ratio)
        
        # Adjust scores based on reported symptoms
        if reported_symptoms:
            # Get symptom scores for each disease
            symptom_scores = calculate_symptom_score(reported_symptoms)
            
            # Weight: 30% image features, 70% symptoms (changed from 60:40)
            image_weight = 0.3
            symptom_weight = 0.7
            
            # Combine scores
            for disease in DISEASE_CLASSES:
                image_score = scores.get(disease, 0.0)
                symptom_score = symptom_scores.get(disease, 0.0)
                scores[disease] = (image_weight * image_score) + (symptom_weight * symptom_score)
        
        # Convert to probabilities
        total = sum(scores.values())
        probs = {k: v/total for k, v in scores.items()}
        
        # Further reduce "Healthy" probability by half
        if "Healthy" in probs:
            probs["Healthy"] *= 0.5
            
        # Re-normalize probabilities
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Sort and get top prediction
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_disease = sorted_probs[0][0]
        top_confidence = sorted_probs[0][1] * 100
        
        # Create matching symptoms dict for each disease
        matching_symptoms = {}
        if reported_symptoms:
            for disease, symptoms in DISEASE_SYMPTOMS.items():
                matching_symptoms[disease] = [s for s in reported_symptoms if s in symptoms]
        
        return {
            "prediction": top_disease,
            "confidence": round(top_confidence, 2),
            "all_predictions": [(d, round(p * 100, 2)) for d, p in sorted_probs],
            "treatment": DISEASE_INFO[top_disease]["treatment"],
            "description": DISEASE_INFO[top_disease]["description"],
            "all_treatments": {disease: info for disease, info in DISEASE_INFO.items()},
            "model_used": "Fallback Image Analysis" + (" with Symptom Analysis" if reported_symptoms else ""),
            "reported_symptoms": reported_symptoms or [],
            "matching_symptoms": matching_symptoms
        }

@app.route('/')
def index():
    """Render the main page with upload form and symptoms checklist"""
    return render_template('index_with_symptoms.html', symptoms=ALL_SYMPTOMS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make predictions with symptoms"""
    if 'image' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    # Get reported symptoms from form
    reported_symptoms = request.form.getlist('symptoms')
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Analyze the image with reported symptoms
        result = analyze_image(file_path, reported_symptoms)
        
        # Render the result page
        return render_template('result_with_symptoms.html',
                              filename=filename,
                              prediction=result['prediction'],
                              confidence=result['confidence'],
                              all_predictions=result['all_predictions'],
                              treatment=result['treatment'],
                              description=result['description'],
                              all_treatments=result['all_treatments'],
                              model_used=result.get('model_used', 'Unknown'),
                              reported_symptoms=result['reported_symptoms'],
                              matching_symptoms=result['matching_symptoms'])
    else:
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting Pig Disease Detection Web App...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Available symptoms: {len(ALL_SYMPTOMS)}")
    if interpreter:
        print("TFLite model loaded successfully")
    else:
        print("Warning: TFLite model could not be loaded, will use fallback analysis")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 