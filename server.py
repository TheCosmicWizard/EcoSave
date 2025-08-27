from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
# Enable CORS for all origins during development
CORS(app, origins=['*'])

model = None
classes = ["compostable", "recyclable", "non_recyclable"]

def load_model():
    global model
    try:
        model_path = os.path.join('models', 'waste_model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
            return True
        else:
            print("⚠️ Model file not found at:", model_path)
            print("⚠️ App will run without model for testing")
            return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("⚠️ App will run without model for testing")
        return False

# Status endpoint that frontend expects
@app.route('/', methods=['GET'])
def get_status():
    """Return backend status for frontend"""
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "message": "EcoSave Backend API"
    })

# Serve the main HTML file
@app.route('/app')
def serve_app():
    """Serve the main HTML application"""
    try:
        return send_file('index.html')
    except Exception as e:
        return jsonify({"error": f"Could not serve HTML file: {str(e)}"}), 404

# Serve about page
@app.route('/about')
def serve_about():
    """Serve the about HTML page"""
    try:
        return send_file('about.html')
    except Exception as e:
        return jsonify({"error": f"Could not serve about.html: {str(e)}"}), 404

# Serve static files from src directory
@app.route('/src/<path:filename>')
def serve_static_files(filename):
    """Serve static files from src directory"""
    try:
        return send_from_directory('src', filename)
    except Exception as e:
        return jsonify({"error": f"Could not serve file: {str(e)}"}), 404

@app.route('/analyze-waste', methods=['POST'])
def analyze_waste():
    try:
        print("🔥 Received analyze request")
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        print(f"📷 Processing image: {file.filename}")
        
        # If no model, return mock data for testing
        if model is None:
            print("⚠️ No model available, returning mock data")
            mock_result = {
                "category": "Recyclable",
                "confidence": 85.5,
                "color": "#0C3C01",
                "items": ["Plastic Bottle", "Test Item"]
            }
            print(f"🎯 Mock Result: {mock_result}")
            return jsonify(mock_result)
        
        # Process image
        image = Image.open(file.stream)
        print(f"📊 Image mode: {image.mode}, size: {image.size}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        prediction = model.predict(image_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]) * 100)
        category = classes[class_idx]
        
        print(f"🎯 Prediction: {category} ({confidence:.1f}%)")
        
        # Format response
        result = {
            "category": category.title(),
            "confidence": round(confidence, 1),
            "color": get_color(category),
            "items": get_items(category)
        }
        
        print(f"📤 Sending result to frontend: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

def get_color(category):
    colors = {
        "compostable": "#5B6D49",
        "recyclable": "#0C3C01", 
        "non_recyclable": "#2E2D1D"
    }
    return colors.get(category, "#808080")

def get_items(category):
    items = {
        "compostable": ["Food Scraps", "Organic Waste"],
        "recyclable": ["Plastic Bottles", "Aluminum Cans"],
        "non_recyclable": ["Mixed Materials", "Non-recyclable Items"]
    }
    return items.get(category, ["Unknown Items"])

if __name__ == '__main__':
    print("🚀 Starting EcoSave Backend...")
    print("🌐 Server will run on: http://localhost:8000")
    
    model_loaded = load_model()
    if model_loaded:
        print("✅ Ready with AI model")
    else:
        print("⚠️ Running in test mode without AI model")
    
    print("🔗 CORS enabled for frontend connection")
    print("=" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True)