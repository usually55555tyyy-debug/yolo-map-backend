import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Blueprint Vectorizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_blueprint_cv(img, approx_epsilon=2.0, min_wall_length=15.0):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to isolate dark wall lines from background
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    # Morphological closing to bridge small gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find external and internal structural contours
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    walls = []
    
    for cnt in contours:
        # Ignore tiny artifacts/noise
        if cv2.contourArea(cnt) < 50:
            continue
            
        # Simplify contour into straight polylines
        approx = cv2.approxPolyDP(cnt, approx_epsilon, True)
        pts = approx.reshape(-1, 2)
        
        # Convert polylines into wall segment pairs (start x,y -> end x,y)
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            
            # Calculate segment length
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if length >= min_wall_length:
                walls.append({
                    "start": {"x": float(p1[0]), "y": float(p1[1])},
                    "end": {"x": float(p2[0]), "y": float(p2[1])},
                    "thickness": 0.2
                })
                
    return walls

@app.post("/extract-map")
async def extract_map(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid image file"})

    orig_h, orig_w = img.shape[:2]
    
    # Process image with computer vision
    walls = process_blueprint_cv(img)

    return JSONResponse(content={
        "status": "success",
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "walls": walls,
        "doors": []
    })

@app.get("/health")
def health():
    return {"status": "ok"}
