from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
