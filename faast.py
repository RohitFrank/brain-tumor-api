from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn
from typing import Dict

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for detecting brain tumors from MRI scans using VGG19",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Setup ---
# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model architecture
model = models.vgg19(weights=None)
for param in model.features.parameters():
    param.requires_grad = False
num_ftrs = model.classifier[6].in_features
num_classes = 4
model.classifier[6] = nn.Linear(num_ftrs, num_classes)
model.to(device)

# Load the saved weights
weights_path = r'D:\hackothan\best_vgg19_model.pth'
try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the class names and data transformations
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess the uploaded image for model prediction"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_tumor(image_tensor: torch.Tensor) -> Dict[str, any]:
    """Make prediction on the preprocessed image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item() * 100
        
        # Format the prediction result
        if predicted_class == 'notumor':
            prediction_text = "No Tumor Detected"
        else:
            prediction_text = f"{predicted_class.title()} Tumor Detected"
        
        return {
            "prediction": prediction_text,
            "class": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": {
                class_names[i]: round(torch.nn.functional.softmax(outputs, dim=1)[0][i].item() * 100, 2)
                for i in range(len(class_names))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Brain Tumor Detection API",
        "description": "Upload an MRI scan image to detect brain tumors",
        "endpoints": {
            "/predict": "POST - Upload image for tumor detection",
            "/health": "GET - Check API health status",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "device": str(device),
        "model_status": model_status,
        "classes": class_names
    }

@app.post("/predict")
async def predict_brain_tumor(file: UploadFile = File(...)):
    """
    Predict brain tumor from uploaded MRI image
    
    - *file*: MRI scan image (jpg, jpeg, png)
    - Returns prediction, confidence, and probabilities for all classes
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    allowed_extensions = ['image/jpeg', 'image/jpg', 'image/png']
    if file.content_type not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file.content_type} not supported. Allowed types: {allowed_extensions}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_tumor(image_tensor)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "result": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "fasst:app",  # Replace 'main' with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )