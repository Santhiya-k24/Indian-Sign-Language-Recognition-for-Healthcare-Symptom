import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Paths
VIDEO_DIR = r"D:/ISL/datasets/ISL_Healthcare_Symptoms"
SAVE_DIR = r"D:/ISL/datasets/ISL_Healthcare_Symptoms_Landmarks"
os.makedirs(SAVE_DIR, exist_ok=True)

# Mediapipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to extract keypoints from a frame
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    return np.concatenate([pose, lh, rh, face])

# Process each symptom folder
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for symptom in os.listdir(VIDEO_DIR):
        symptom_path = os.path.join(VIDEO_DIR, symptom)
        if not os.path.isdir(symptom_path):
            continue

        save_path = os.path.join(SAVE_DIR, symptom)
        os.makedirs(save_path, exist_ok=True)

        for video_file in tqdm(os.listdir(symptom_path), desc=f"Processing {symptom}"):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(symptom_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                frame_data.append(keypoints)

            cap.release()
            frame_data = np.array(frame_data)

            # Save the landmarks array
            npy_filename = video_file.rsplit('.', 1)[0] + '.npy'
            np.save(os.path.join(save_path, npy_filename), frame_data)

print("✅ All landmarks extracted and saved.")
