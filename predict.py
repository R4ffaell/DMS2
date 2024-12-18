import torch
from ultralytics import YOLO
import cv2

def load_yolov11_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"YOLOv11 model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        raise e
    return model

# Load model
model_path = 'Models/yolov11n-face.pt'  # Update with your model path
model = load_yolov11_model(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare image for YOLO
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    print(f"Prepared image tensor shape: {img_tensor.shape}")
    print(f"Max pixel value: {img_tensor.max()} | Min pixel value: {img_tensor.min()}")

    # Run inference
    results = model(img_tensor)

    # Process detections
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy  # Bounding box coordinates
        confidences = results[0].boxes.conf  # Confidence scores
        class_ids = results[0].boxes.cls  # Class IDs

        # Draw boxes on the frame
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{results[0].names[int(cls_id)]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv11 Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
