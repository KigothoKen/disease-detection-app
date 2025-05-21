import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and show placeholder results"""
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
        
        # Placeholder results since we don't have the TensorFlow model running
        placeholder_results = {
            "prediction": "Demo Result - TensorFlow model not loaded",
            "confidence": 99.99,
            "all_predictions": [
                ("Demo Disease Type 1", 99.99),
                ("Demo Disease Type 2", 0.01)
            ]
        }
        
        # Render the result page
        return render_template('result.html',
                              filename=filename,
                              prediction=placeholder_results['prediction'],
                              confidence=placeholder_results['confidence'],
                              all_predictions=placeholder_results['all_predictions'])
    else:
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    from werkzeug.utils import secure_filename
    app.run(debug=True, host='0.0.0.0', port=5000) 