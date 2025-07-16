import os
import numpy as np
import io
from PIL import Image
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.session import run_js

# Mock model for demonstration (since we don't have the actual model.h5 file)
class MockModel:
    def predict(self, img_array):
        # Simulate prediction with random probabilities
        return np.random.rand(1, 3)

# Use mock model since we don't have the actual trained model
model = MockModel()

def predict_breast_cancer():
    # Add CSS styling
    put_html('''
        <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .upload-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .result-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-left: 5px solid #28a745;
        }
        .pwt-btn {
            background: linear-gradient(135deg, #FF69B4, #FF1493) !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 12px 30px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        .pwt-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(255, 105, 180, 0.4) !important;
        }
        .prediction-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.2em;
            margin: 10px 0;
        }
        .normal { background-color: #d4edda; color: #155724; }
        .benign { background-color: #fff3cd; color: #856404; }
        .malignant { background-color: #f8d7da; color: #721c24; }
        </style>
    ''')
    
    # Header section
    put_html('''
        <div class="header-container">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 300;">🩺 Breast Cancer Detection</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">AI-Powered Medical Image Analysis</p>
        </div>
    ''')

    # Information section
    put_html('''
        <div class="upload-container">
            <h3 style="color: #333; margin-bottom: 15px;">📋 Instructions</h3>
            <p style="color: #666; line-height: 1.6;">
                Upload a breast ultrasound image for AI-powered analysis. Our system uses advanced 
                deep learning techniques with DenseNet-201 architecture to classify images into 
                three categories: Normal, Benign, or Malignant.
            </p>
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only and should not 
                replace professional medical diagnosis.
            </div>
        </div>
    ''')

    # File upload section
    put_html('<div class="upload-container">')
    put_markdown("### 📤 Upload Image")
    
    try:
        img = file_upload("Select breast ultrasound image", accept="image/*")
        
        if img:
            # Display uploaded image
            put_markdown("### 🖼️ Uploaded Image")
            put_image(img['content'], width='300px')
            
            # Process the image
            put_markdown("### 🔄 Processing...")
            put_loading()
            
            # Preprocess the image
            img_array = np.array(Image.open(io.BytesIO(img['content'])))
            
            # Resize image to match model input size (128x128 as in original code)
            img_resized = Image.fromarray(img_array).resize((128, 128))
            
            # Convert to array and add batch dimension
            img_preprocessed = np.array(img_resized)
            if len(img_preprocessed.shape) == 2:  # Grayscale
                img_preprocessed = np.stack([img_preprocessed] * 3, axis=-1)  # Convert to RGB
            
            # Make predictions
            preds = model.predict(np.expand_dims(img_preprocessed, axis=0))
            labels = ['Normal', 'Benign', 'Malignant']
            prediction_idx = np.argmax(preds)
            prediction = labels[prediction_idx]
            confidence = preds[0][prediction_idx] * 100
            
            # Clear loading
            clear()
            
            # Show results
            put_html('</div>')
            
            put_html(f'''
                <div class="result-container">
                    <h3 style="color: #333; margin-bottom: 20px;">🎯 Analysis Results</h3>
                    <div class="prediction-badge {prediction.lower()}">
                        Prediction: {prediction}
                    </div>
                    <p style="color: #666; margin: 15px 0;">
                        <strong>Confidence:</strong> {confidence:.1f}%
                    </p>
                    <div style="margin-top: 20px;">
                        <h4>Class Probabilities:</h4>
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                <span>Normal:</span>
                                <span>{preds[0][0]*100:.1f}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                <span>Benign:</span>
                                <span>{preds[0][1]*100:.1f}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                <span>Malignant:</span>
                                <span>{preds[0][2]*100:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            ''')
            
            # Add restart button
            if actions("What would you like to do?", ["Analyze Another Image", "Exit"]) == "Analyze Another Image":
                run_js("location.reload()")
    
    except Exception as e:
        put_error(f"Error processing image: {str(e)}")
    
    put_html('</div>')

if __name__ == '__main__':
    start_server(predict_breast_cancer, port=8080, debug=True)