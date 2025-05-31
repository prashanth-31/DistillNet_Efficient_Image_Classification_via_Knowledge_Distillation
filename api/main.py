import sys
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io
from PIL import Image
import time
from pydantic import BaseModel
import yaml

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config
from src.models import get_student_model
from src.utils import load_model
from torchvision import transforms

# Load configuration
config = load_config("config.yaml")

# Initialize FastAPI app
app = FastAPI(
    title="DistillNet API",
    description="API for efficient image classification using knowledge distillation",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Define response model
class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    inference_time_ms: float

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Model loading
def load_student_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_student_model(
        model_name=config["student"]["model"], 
        num_classes=10,
        pretrained=False
    )
    
    try:
        model = load_model(model, config["api"]["model_path"])
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully from {config['api']['model_path']}")
        return model, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, device

model, device = load_student_model()

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": config["student"]["model"]}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and process image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    # Make prediction
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image_tensor)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())
    
    # Return prediction
    return PredictionResponse(
        class_id=predicted_class,
        class_name=CLASSES[predicted_class],
        confidence=confidence,
        inference_time_ms=inference_time
    )

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_name": config["student"]["model"],
        "num_parameters": num_params,
        "input_shape": [3, 32, 32],
        "classes": CLASSES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=config["api"]["host"], 
        port=config["api"]["port"],
        reload=True
    )
