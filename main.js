// Global variables
let currentImage = null;
let analysisStartTime = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    const loadingElement = document.getElementById('loading');
    const appElement = document.getElementById('app');
    
    // Show loading screen while model loads
    loadingElement.style.display = 'flex';
    appElement.style.display = 'none';
    
    try {
        // Load and train the real TensorFlow.js model
        console.log('Loading real TensorFlow.js model...');
        await model.loadModel();
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Failed to load model:', error);
        alert('Failed to load AI model. Please refresh the page.');
        return;
    }
    
    // Hide loading and show app
    setTimeout(() => {
        loadingElement.style.display = 'none';
        appElement.style.display = 'block';
        
        // Initialize drag and drop
        initializeDragAndDrop();
        initializeEventListeners();
    }, 1000); // Allow time for model to fully initialize
});

function initializeDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop events
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function initializeEventListeners() {
    // Change image button
    document.getElementById('changeImage').addEventListener('click', () => {
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('fileInput').value = '';
        currentImage = null;
    });

    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', analyzeImage);

    // New analysis button
    document.getElementById('newAnalysis').addEventListener('click', () => {
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('fileInput').value = '';
        currentImage = null;
    });

    // Download report button
    document.getElementById('downloadReport').addEventListener('click', downloadReport);
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB.');
        return;
    }

    currentImage = file;
    displayImagePreview(file);
}

function displayImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const img = document.getElementById('previewImg');
        img.src = e.target.result;
        
        // Update file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
        
        // Get image dimensions
        img.onload = () => {
            document.getElementById('imageDimensions').textContent = 
                `${img.naturalWidth} × ${img.naturalHeight}px`;
        };
        
        // Show preview
        document.getElementById('imagePreview').style.display = 'block';
        document.getElementById('imagePreview').classList.add('fade-in');
    };
    
    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!currentImage) return;

    // Check if model is loaded
    if (!model.isLoaded) {
        alert('AI model is still loading. Please wait a moment and try again.');
        return;
    }

    // Check if model is loaded
    if (!model.isLoaded) {
        alert('AI model is still loading. Please wait a moment and try again.');
        return;
    }

    analysisStartTime = Date.now();
    
    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalText = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running AI Analysis...';
    analyzeBtn.disabled = true;

    try {
        // Get image element for processing
        const img = document.getElementById('previewImg');
        
        // Ensure image is loaded
        if (!img.complete) {
            await new Promise(resolve => {
                img.onload = resolve;
            });
        }
        
        console.log('Running TensorFlow prediction on image:', {
            width: img.naturalWidth,
            height: img.naturalHeight,
            modelLoaded: model.isLoaded
        });
        
        // Run real TensorFlow.js CNN prediction
        const probabilities = await model.predict(img);
        
        console.log('TensorFlow.js prediction completed:', probabilities);

        // Display results
        displayResults(probabilities);
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Analysis failed: ${error.message}. Please try with a different image.`);
    } finally {
        // Reset button
        analyzeBtn.innerHTML = originalText;
        analyzeBtn.disabled = false;
    }
}

function displayResults(probabilities) {
    const labels = ['Normal', 'Benign', 'Malignant'];
    const classes = ['normal', 'benign', 'malignant'];
    const icons = ['fa-check-circle', 'fa-exclamation-circle', 'fa-times-circle'];
    
    // Find prediction
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const prediction = labels[maxIndex];
    const confidence = (probabilities[maxIndex] * 100).toFixed(1);
    
    // Update prediction display
    const predictionMain = document.querySelector('.prediction-main');
    predictionMain.className = `prediction-main ${classes[maxIndex]}`;
    
    document.getElementById('predictionIcon').innerHTML = `<i class="fas ${icons[maxIndex]}"></i>`;
    document.getElementById('predictionLabel').textContent = prediction;
    document.getElementById('confidenceScore').textContent = `${confidence}%`;
    
    // Update analysis time
    const analysisTime = ((Date.now() - analysisStartTime) / 1000).toFixed(1);
    document.getElementById('analysisTime').textContent = `Analysis completed in ${analysisTime}s`;
    
    // Update probability bars
    const probElements = ['normalProb', 'benignProb', 'malignantProb'];
    const barElements = ['normalBar', 'benignBar', 'malignantBar'];
    
    probabilities.forEach((prob, index) => {
        const percentage = (prob * 100).toFixed(1);
        document.getElementById(probElements[index]).textContent = `${percentage}%`;
        
        // Animate bars
        setTimeout(() => {
            document.getElementById(barElements[index]).style.width = `${percentage}%`;
        }, 100 * index);
    });
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').classList.add('fade-in');
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function downloadReport() {
    if (!currentImage) return;
    
    // Get current results
    const prediction = document.getElementById('predictionLabel').textContent;
    const confidence = document.getElementById('confidenceScore').textContent;
    const analysisTime = document.getElementById('analysisTime').textContent;
    
    const normalProb = document.getElementById('normalProb').textContent;
    const benignProb = document.getElementById('benignProb').textContent;
    const malignantProb = document.getElementById('malignantProb').textContent;
    
    // Create report content
    const reportContent = `
BREAST CANCER DETECTION ANALYSIS REPORT
=====================================
Generated by Real TensorFlow.js CNN Model

Patient Information:
- Analysis Date: ${new Date().toLocaleString()}
- Image File: ${currentImage.name}
- Image Size: ${document.getElementById('fileSize').textContent}
- Image Dimensions: ${document.getElementById('imageDimensions').textContent}

Analysis Results:
- Primary Prediction: ${prediction}
- Confidence Level: ${confidence}
- ${analysisTime}

Detailed Probability Breakdown:
- Normal: ${normalProb}
- Benign: ${benignProb}
- Malignant: ${malignantProb}

AI Model Information:
- Architecture: ${model.getModelInfo()?.architecture || 'Custom CNN'}
- Framework: ${model.getModelInfo()?.framework || 'TensorFlow.js'}
- Model Parameters: ${model.getModelInfo()?.totalParams || 'N/A'}
- Layers: ${model.getModelInfo()?.layers || 'N/A'}
- Input Size: ${model.getModelInfo()?.inputShape?.join('x') || '224x224x3'}
- Training: Synthetic medical imaging data
- Real-time Processing: Yes (Browser-based)
- Image Analysis: Contrast, Edge Detection, Symmetry, Uniformity

AI Model Information:
- Architecture: ${model.getModelInfo()?.architecture || 'Custom CNN'}
- Framework: ${model.getModelInfo()?.framework || 'TensorFlow.js'}
- Model Parameters: ${model.getModelInfo()?.totalParams || 'N/A'}
- Layers: ${model.getModelInfo()?.layers || 'N/A'}
- Input Size: ${model.getModelInfo()?.inputShape?.join('x') || '224x224x3'}
- Training: Synthetic medical imaging data
- Real-time Processing: Yes (Browser-based)
- Image Analysis: Contrast, Edge Detection, Symmetry, Uniformity

MEDICAL DISCLAIMER:
This AI analysis uses real CNN processing but is trained on synthetic data
for educational and research purposes only. 
for educational and research purposes only. 
Always consult with qualified healthcare professionals for medical 
diagnosis and treatment decisions.

Generated by MedAI Diagnostics System - Real TensorFlow.js Implementation
    `.trim();
    
    // Create and download file
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `breast_cancer_analysis_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}