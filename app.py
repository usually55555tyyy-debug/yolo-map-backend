import io
import math
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="YOLO Blueprint Auto-Map Engine")

# 1. Enable CORS for cross-origin browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load YOLOv8 Model on Startup
# Ensure 'best.pt' is in the same directory as app.py on Render
try:
    model = YOLO("best.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load 'best.pt' ({e}). Make sure the weights file is present.")
    model = None


def simplify_contour_to_line(contour, epsilon_factor=0.02):
    """Simplifies OpenCV contours into clean line endpoints [x, y]."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
    pts = approx.reshape(-1, 2).tolist()
    return pts


def opencv_fallback_walls(img_np):
    """Fallback wall extraction using Canny edge detection if YOLO detects 0 walls."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological closing to join gaps in wall lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    walls = []
    wall_id = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        pts = simplify_contour_to_line(cnt)
        if len(pts) >= 2:
            walls.append({
                "id": f"w{wall_id}",
                "points": pts,
                "thickness": 12.0
            })
            wall_id += 1

    return walls


@app.get("/")
def health_check():
    """Root route to confirm backend is online."""
    return {"status": "online", "message": "YOLO Map Backend Running"}


@app.post("/extract-walls")
async def extract_walls(
    file: UploadFile = File(...),
    real_wall_meters: float = Form(15.0)
):
    """
    Main endpoint called by deepseek_html_20260926_ac09f0 (1).html
    Receives image + scale, returns vectorized wall coordinates.
    """
    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        h, w, _ = img_np.shape

        walls = []
        doors = []
        windows = []

        # Run YOLOv8 Inference
        if model is not None:
            results = model.predict(source=img_np, conf=0.25)
            res = results[0]

            # Parse segmentation masks if available
            if res.masks is not None:
                for i, mask in enumerate(res.masks.xy):
                    pts = mask.astype(int).tolist()
                    if len(pts) >= 2:
                        walls.append({
                            "id": f"w{i}",
                            "points": pts,
                            "thickness": 14.0
                        })
            # Fallback to bounding boxes converted to center line vectors
            elif res.boxes is not None and len(res.boxes) > 0:
                for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box[:4]
                    # Create wall segment from box diagonal/orientation
                    if (x2 - x1) > (y2 - y1):
                        pts = [[int(x1), int((y1 + y2) / 2)], [int(x2), int((y1 + y2) / 2)]]
                    else:
                        pts = [[int((x1 + x2) / 2), int(y1)], [int((x1 + x2) / 2), int(y2)]]
                    
                    walls.append({
                        "id": f"w{i}",
                        "points": pts,
                        "thickness": float(min(x2 - x1, y2 - y1))
                    })

        # Fallback to OpenCV if YOLO returned 0 walls
        if len(walls) == 0:
            print("Zero walls detected by YOLO. Running OpenCV contour fallback...")
            walls = opencv_fallback_walls(img_np)

        # Calculate scale factor (px_per_meter)
        max_len_px = 100.0  # default minimum threshold
        for w_item in walls:
            pts = w_item["points"]
            for k in range(len(pts) - 1):
                dx = pts[k+1][0] - pts[k][0]
                dy = pts[k+1][1] - pts[k][1]
                dist = math.hypot(dx, dy)
                if dist > max_len_px:
                    max_len_px = dist

        px_per_meter = max_len_px / max(real_wall_meters, 1.0)

        return {
            "image": {
                "width": int(w),
                "height": int(h)
            },
            "scale": {
                "px_per_meter": round(float(px_per_meter), 2)
            },
            "walls": walls,
            "doors": doors,
            "windows": windows
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
