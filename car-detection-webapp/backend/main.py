import base64
import io
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    try:
        # Load the pretrained model (or user's custom 'best.pt' if available)
        # Note: 'yolov8n.pt' is a pretrained YOLOv8 model that can detect cars (class 2 in COCO).
        # To use your custom trained model, replace 'yolov8n.pt' with the path to your 'best.pt' file.
        model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        logger.info(f"Loading YOLO model from {model_path}...")
        model = YOLO(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    # Clean up (if necessary)
    model = None

app = FastAPI(title="Car Object Detection API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run inference
        # Classes: 2 represents 'car' in the default COCO dataset used by yolov8n.pt
        # If using a custom model trained only on cars, you might not need the classes parameter or it might be 0.
        # We will just predict all and then filter, or let YOLO filter if possible.
        results = model.predict(img, conf=0.25)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Convert annotated image back to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Collect basic stats
        detections = []
        for box in results[0].boxes:
            # box.cls is a tensor, .item() gets the python number
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = model.names[class_id]
            detections.append({
                "class": class_name,
                "confidence": confidence
            })
        
        # Filter detections for cars for the summary
        cars_detected = [d for d in detections if d["class"].lower() in ["car", "vehicle", "truck", "bus"]]
        
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "car_count": len(cars_detected)
        })

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend static files last so it doesn't override API routes
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
