# Pig Disease Classifier Web App

This is a web application that uses a TensorFlow Lite model to classify pig diseases from uploaded images.

## Features

- Easy-to-use web interface
- Upload images for disease classification
- View prediction results with confidence scores
- Preview images before submission

## Installation

1. Clone this repository
pip install -r requirements.txt
```

## Running the Application

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

## Features

- Upload images for disease classification
- Real-time image preview
- Disease prediction with confidence scores
- Clean and user-friendly interface

## Deployment

The application is deployed using Heroku. To deploy your own instance:

1. Create a Heroku account if you don't have one
2. Install the Heroku CLI
3. Login to Heroku:
```bash
heroku login
```
4. Create a new Heroku app:
```bash
heroku create
```
5. Deploy the application:
```bash
git push heroku main
```

## Model Information

The application uses a TensorFlow Lite model (`detect.tflite`) to classify pig diseases. 
The model classifies images into the following categories:
- Healthy
- African Swine Fever
- Foot and Mouth Disease
- Porcine Reproductive and Respiratory Syndrome
- Swine Influenza

**Note**: You may need to modify the `DISEASE_CLASSES` in `model.py` to match the actual classes your model is trained to detect.

## System Requirements

- Python 3.6+
- Flask
- TensorFlow or TensorFlow Lite
- PIL (Python Imaging Library)
- NumPy

## Troubleshooting

If you encounter issues with the TFLite runtime, you may need to install the appropriate version for your system:
```
pip install tflite-runtime
```

For ARM devices like Raspberry Pi, you may need a specific installation command. Check the TensorFlow website for instructions. 