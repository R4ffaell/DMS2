import cv2
import mediapipe as mp
import math
import os
from datetime import datetime
import time
import pygame
from ultralytics import YOLO

# Initialize Pygame sound
pygame.init()
pygame.mixer.init()

yawn_sound = pygame.mixer.Sound("yawn_alert.mp3")
drowsiness_sound = pygame.mixer.Sound("drowsiness_alert.mp3")
distraction_sound = pygame.mixer.Sound("distraction_alert.mp3")

# Initialize sound channels
yawn_channel = pygame.mixer.Channel(0)
drowsiness_channel = pygame.mixer.Channel(1)
distraction_channel = pygame.mixer.Channel(2)

# Load YOLO model
def load_yolo_model(model_path):
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
    return model

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Directory setup for screenshots
output_dir = 'driver_behavior_screenshots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

defector_folders = ['yawn', 'drowsiness', 'distraction']
for folder in defector_folders:
    folder_path = os.path.join(output_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Mediapipe confidence checks
def yawn_confidence(landmarks):
    mouth_upper = landmarks[13]
    mouth_lower = landmarks[14]
    return max(0, min(100, (calculate_distance(mouth_upper, mouth_lower) - 0.03) / 0.03 * 100))

def eyes_closed_confidence(landmarks):
    # Get the positions of the upper and lower eyelids for both eyes
    left_eye_upper = landmarks[386]  # Left eye upper eyelid
    left_eye_lower = landmarks[374]  # Left eye lower eyelid
    right_eye_upper = landmarks[159]  # Right eye upper eyelid
    right_eye_lower = landmarks[145]  # Right eye lower eyelid

    # Calculate distances between the upper and lower eyelids for both eyes
    left_eye_distance = calculate_distance(left_eye_upper, left_eye_lower)
    right_eye_distance = calculate_distance(right_eye_upper, right_eye_lower)

    # Average of both eye distances
    avg_eye_distance = (left_eye_distance + right_eye_distance) / 2

    # Debugging print statements to observe the distances
    print(f"Left eye distance: {left_eye_distance}")
    print(f"Right eye distance: {right_eye_distance}")
    print(f"Average eye distance: {avg_eye_distance}")

    # Adjust the threshold here if necessary
    return max(0, min(100, (0.02 - avg_eye_distance) / 0.02 * 100))



def head_pose_confidence(landmarks):
    nose_tip = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    ear_distance = calculate_distance(left_ear, right_ear)
    if calculate_distance(nose_tip, left_ear) > ear_distance * 1.2:
        return 100  # Looking left
    elif calculate_distance(nose_tip, right_ear) > ear_distance * 1.2:
        return 100  # Looking right
    else:
        return 0

def save_screenshot(frame, label):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'{label}_{timestamp}.png'
    filepath = os.path.join(output_dir, label, filename)
    cv2.imwrite(filepath, frame)
    print(f'Screenshot saved: {filepath}')

# Main function to combine YOLO and Mediapipe
def main():
    yolo_model_path = 'Models/yolov11n-face.pt'  # Replace with your YOLO model path
    yolo_model = load_yolo_model(yolo_model_path)

    cap = cv2.VideoCapture(0)

    start_time = time.time()  # Record the start time
    frame_count = 0  # Initialize frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = yolo_model(frame)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box  # Only unpack 4 values
                conf = result.boxes.conf.cpu().numpy()  # Confidence scores
                cls = result.boxes.cls.cpu().numpy()  # Class IDs

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers
                face = frame[y1:y2, x1:x2]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Face: {conf[0]:.2f}"  # Label with confidence score
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_mesh_results = face_mesh.process(rgb_face)

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

                        # Print confidence values for debugging
                        yawn_conf = yawn_confidence(landmarks)
                        eyes_conf = eyes_closed_confidence(landmarks)
                        head_pose_conf = head_pose_confidence(landmarks)

                        print(f"Yawn Confidence: {yawn_conf}")
                        print(f"Eyes Closed Confidence: {eyes_conf}")
                        print(f"Head Pose Confidence: {head_pose_conf}")

                        # Lower the threshold for debugging
                        if yawn_conf > 30:  # Lowered threshold for testing
                            save_screenshot(frame, "yawn")
                            yawn_channel.play(yawn_sound)
                        if eyes_conf > 30:  # Lowered threshold for testing
                            save_screenshot(frame, "drowsiness")
                            drowsiness_channel.play(drowsiness_sound)
                        if head_pose_conf > 30:  # Lowered threshold for testing
                            save_screenshot(frame, "distraction")
                            distraction_channel.play(distraction_sound)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Driver Monitoring System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
