from ultralytics import YOLO

def load_yolov11_model(model_path):
    try:
        # Load YOLOv8/YOLOv11 model using Ultralytics
        model = YOLO(model_path)  # Ultralytics library
        print(f"YOLOv11 model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        raise e
    return model
