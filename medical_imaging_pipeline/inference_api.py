"""
Inference API module using FastAPI for model deployment.
Provides REST API endpoints for medical image classification.
"""
import io
import os
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

from config import InferenceConfig, get_config
from models import create_model
from data_preprocessing import get_val_transforms


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    image_base64: Optional[str] = None
    return_features: bool = False


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    model_version: str


class ModelServer:
    """
    Model server for inference.
    Handles model loading, preprocessing, and prediction.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.device = None
        self.transform = None
        self.class_names = ["NORMAL", "PNEUMONIA"]
        self.model_version = "v1.0.0"
        
    def load_model(self, model_path: Optional[str] = None):
        """Load model from checkpoint."""
        model_path = model_path or self.config.model_path
        
        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Create model architecture
        self.model = create_model(
            model_type='resnet50',
            num_classes=len(self.class_names),
            pretrained=False,
            freeze_backbone=False
        )
        
        # Load weights if checkpoint exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model checkpoint not found at {model_path}. Using random weights.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load quantized model if specified
        if self.config.use_quantized:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Using quantized model")
        
        # Setup transforms
        config = get_config()
        self.transform = get_val_transforms(config.data)
        
        logger.info("Model loaded successfully")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy and apply transforms
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        tensor = transformed['image'].unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        return_features: bool = False
    ) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image
            return_features: Whether to return intermediate features
        
        Returns:
            Prediction results
        """
        start_time = time.perf_counter()
        
        # Preprocess
        tensor = self.preprocess(image)
        
        # Forward pass
        if return_features and hasattr(self.model, 'get_features'):
            features = self.model.get_features(tensor)
            logits = self.model.classifier(features)
        else:
            logits = self.model(tensor)
            features = None
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        result = {
            'prediction': self.class_names[pred_idx],
            'confidence': confidence,
            'probabilities': {
                name: probs[i].item()
                for i, name in enumerate(self.class_names)
            },
            'inference_time_ms': inference_time,
            'model_version': self.model_version
        }
        
        if return_features and features is not None:
            result['features'] = features[0].cpu().numpy().tolist()
        
        return result
    
    def batch_predict(
        self,
        images: List[Image.Image]
    ) -> List[Dict]:
        """Run batch inference."""
        # Preprocess all images
        tensors = torch.cat([self.preprocess(img) for img in images])
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            logits = self.model(tensors)
            probs = F.softmax(logits, dim=1)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        results = []
        for i, prob in enumerate(probs):
            pred_idx = prob.argmax().item()
            results.append({
                'prediction': self.class_names[pred_idx],
                'confidence': prob[pred_idx].item(),
                'probabilities': {
                    name: prob[j].item()
                    for j, name in enumerate(self.class_names)
                },
                'inference_time_ms': inference_time / len(images),
                'model_version': self.model_version
            })
        
        return results


# Global model server instance
model_server: Optional[ModelServer] = None


def get_model_server() -> ModelServer:
    """Get the model server instance."""
    global model_server
    if model_server is None:
        raise RuntimeError("Model server not initialized")
    return model_server


# Create FastAPI app if available
if FASTAPI_AVAILABLE:
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for model loading."""
        global model_server
        config = get_config().inference
        model_server = ModelServer(config)
        model_server.load_model()
        yield
        # Cleanup
        model_server = None
    
    app = FastAPI(
        title="Medical Image Classification API",
        description="API for classifying medical images using deep learning",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        server = get_model_server()
        return HealthResponse(
            status="healthy",
            model_loaded=server.model is not None,
            device=str(server.device),
            model_version=server.model_version
        )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        return_features: bool = False
    ):
        """
        Classify a medical image.
        
        Args:
            file: Image file (JPEG, PNG)
            return_features: Whether to return intermediate features
        
        Returns:
            Classification result with confidence scores
        """
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Get prediction
            server = get_model_server()
            result = server.predict(image, return_features)
            
            return PredictionResponse(**result)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    @app.post("/predict/batch")
    async def predict_batch(
        files: List[UploadFile] = File(...)
    ):
        """
        Classify multiple medical images.
        
        Args:
            files: List of image files
        
        Returns:
            List of classification results
        """
        if len(files) > 32:
            raise HTTPException(
                status_code=400,
                detail="Maximum batch size is 32"
            )
        
        try:
            images = []
            for file in files:
                contents = await file.read()
                images.append(Image.open(io.BytesIO(contents)))
            
            server = get_model_server()
            results = server.batch_predict(images)
            
            return {"predictions": results}
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch prediction failed: {str(e)}"
            )
    
    @app.get("/model/info")
    async def model_info():
        """Get model information."""
        server = get_model_server()
        
        num_params = sum(p.numel() for p in server.model.parameters())
        
        return {
            "model_version": server.model_version,
            "model_type": "ResNet-50",
            "num_parameters": num_params,
            "classes": server.class_names,
            "device": str(server.device),
            "quantized": server.config.use_quantized
        }


def create_app(model_path: Optional[str] = None) -> 'FastAPI':
    """Create and configure FastAPI app."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")
    
    global model_server
    config = get_config().inference
    if model_path:
        config.model_path = model_path
    
    model_server = ModelServer(config)
    model_server.load_model()
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_path: Optional[str] = None,
    reload: bool = False
):
    """Run the inference server."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI is required. Install with: pip install fastapi uvicorn python-multipart")
        return
    
    # Initialize model server
    global model_server
    config = get_config().inference
    if model_path:
        config.model_path = model_path
    
    model_server = ModelServer(config)
    model_server.load_model()
    
    # Run server
    uvicorn.run(
        "inference_api:app",
        host=host,
        port=port,
        reload=reload,
        workers=1  # Single worker for GPU models
    )


# Simple inference function for non-API usage
def predict_image(
    image_path: str,
    model_path: str,
    device: str = 'cuda'
) -> Dict:
    """
    Simple function to predict a single image.
    
    Args:
        image_path: Path to image file
        model_path: Path to model checkpoint
        device: Device to use
    
    Returns:
        Prediction results
    """
    config = get_config().inference
    config.model_path = model_path
    
    server = ModelServer(config)
    server.load_model()
    
    image = Image.open(image_path)
    return server.predict(image)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Medical Image Classification API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model,
        reload=args.reload
    )
