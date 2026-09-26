import io
import numpy as np
import cv2
import onnxruntime as ort
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

MODEL_PATH = "yolov8n-seg.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

session = None
input_name = None
output_names = None


@app.on_event("startup")
def load_model():
    global session, input_name, output_names
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def preprocess(img):
    padded, r, pad_x, pad_y = letterbox(img, INPUT_SIZE)
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None, :, :, :]
    return np.ascontiguousarray(blob), r, pad_x, pad_y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_threshold]
    return keep


def process_mask(protos, mask_coef, box, shape_hw):
    c, mh, mw = protos.shape
    m = sigmoid(mask_coef @ protos.reshape(c, mh * mw)).reshape(mh, mw)
    m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = [int(v) for v in box]
    mask = np.zeros(shape_hw, dtype=np.uint8)
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, shape_hw[1]), min(y2, shape_hw[0])
    if x2c > x1c and y2c > y1c:
        region = (m[y1c:y2c, x1c:x2c] > 0.5).astype(np.uint8) * 255
        mask[y1c:y2c, x1c:x2c] = region
    return mask


def mask_to_segments(mask, r, pad_x, pad_y, orig_w, orig_h, epsilon=3.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = (pts[:, 0] - pad_x) / r
        pts[:, 1] = (pts[:, 1] - pad_y) / r
        pts[:, 0] = np.clip(pts[:, 0], 0, orig_w)
        pts[:, 1] = np.clip(pts[:, 1], 0, orig_h)
        segments.append(pts)
    return segments


def segments_to_walls(segments, thickness=0.2):
    walls = []
    for pts in segments:
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            walls.append({
                "start": [round(float(p1[0]), 2), round(float(p1[1]), 2)],
                "end": [round(float(p2[0]), 2), round(float(p2[1]), 2)],
                "thickness": thickness,
            })
    return walls


@app.post("/extract-map")
async def extract_map(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid image"})

    orig_h, orig_w = img.shape[:2]
    blob, r, pad_x, pad_y = preprocess(img)

    outputs = session.run(output_names, {input_name: blob})
    preds, protos = outputs[0], outputs[1]

    preds = preds[0].T
    boxes_raw = preds[:, :4]
    scores_raw = preds[:, 4:84]
    mask_coefs = preds[:, 84:]

    class_ids = np.argmax(scores_raw, axis=1)
    confidences = np.max(scores_raw, axis=1)
    keep_conf = confidences > CONF_THRESHOLD

    boxes_raw = boxes_raw[keep_conf]
    confidences = confidences[keep_conf]
    mask_coefs = mask_coefs[keep_conf]

    all_walls = []
    if boxes_raw.shape[0] > 0:
        boxes_xyxy = xywh2xyxy(boxes_raw)
        keep_idx = nms(boxes_xyxy, confidences, IOU_THRESHOLD)

        combined_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        for i in keep_idx:
            mask = process_mask(protos[0], mask_coefs[i], boxes_xyxy[i], (INPUT_SIZE, INPUT_SIZE))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        segments = mask_to_segments(combined_mask, r, pad_x, pad_y, orig_w, orig_h)
        all_walls = segments_to_walls(segments)

    return {
        "status": "success",
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "walls": all_walls,
        "doors": [],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
