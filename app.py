import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import io
import gc  # Garbage collector

# Limit TensorFlow memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
MODEL_DIR = 'model'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load model and label encoder
def load_prediction_model():
    print("Loading model...")
    try:
        model = load_model(os.path.join(MODEL_DIR, 'skin_cancer_model.h5'))
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        return model, le
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess input image
def preprocess_image(img_path):
    # Load and resize image
    img = Image.open(img_path)
    img = img.resize(IMG_SIZE)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Load model at startup if it exists
if os.path.exists(os.path.join(MODEL_DIR, 'skin_cancer_model.h5')):
    try:
        model, le = load_prediction_model()
        if model is not None:
            print("Model loaded successfully!")
    except:
        model, le = None, None
        print("Error loading model. Will try to load when needed.")
else:
    model, le = None, None
    print("No model found. Please run the training script first!")

@app.route('/')
def home():
    """Render the home page"""
    global model, le
    
    # Try loading the model if it's not loaded yet
    if model is None and os.path.exists(os.path.join(MODEL_DIR, 'skin_cancer_model.h5')):
        try:
            model, le = load_prediction_model()
        except:
            pass
            
    model_status = "Model loaded" if model is not None else "Model not loaded"
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make prediction"""
    global model, le
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Try loading the model if it's not loaded yet
        if model is None and os.path.exists(os.path.join(MODEL_DIR, 'skin_cancer_model.h5')):
            try:
                model, le = load_prediction_model()
            except:
                pass
                
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please run the training script first!'
            })
        
        try:
            # Preprocess the image
            img_array = preprocess_image(filename)
            
            # Make prediction
            predictions = model.predict(img_array)[0]
            predicted_class_index = np.argmax(predictions)
            predicted_class = le.inverse_transform([predicted_class_index])[0]
            confidence = float(predictions[predicted_class_index])
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3 = [
                {
                    'class': le.inverse_transform([idx])[0],
                    'probability': float(predictions[idx])
                }
                for idx in top_3_indices
            ]
            
            # Only calculate all probabilities if specifically requested
            # to save memory
            class_probabilities = {
                le.inverse_transform([i])[0]: float(predictions[i]) 
                for i in range(len(predictions))
            }
            
            # Free memory
            gc.collect()
            
            # Return predictions
            return jsonify({
                'success': True,
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'top_3': top_3,
                    'all_probabilities': class_probabilities
                },
                'image_path': filename.replace('\\', '/')
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Error during prediction: {str(e)}'
            })
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    # Run the app on 0.0.0.0 to make it accessible outside the container
    print("Starting Flask application...")
    print("Model directory:", MODEL_DIR)
    print("Model path:", os.path.join(MODEL_DIR, 'skin_cancer_model.h5'))
    print("Server will run on port 8080")
    app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)  # Changed to port 8080 