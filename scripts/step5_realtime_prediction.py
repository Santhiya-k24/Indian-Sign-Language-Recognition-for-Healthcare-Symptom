import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

model = load_model(r'D:\ISL\models\best_lstm_model.h5')




mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.load('datasets/actions.npy')

def extract_landmarks(results):
    def get_landmarks(landmarks, count):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
        else:
            return np.zeros(count * 4)

    pose = get_landmarks(results.pose_landmarks, 33)
    face = get_landmarks(results.face_landmarks, 468)
    lh = get_landmarks(results.left_hand_landmarks, 21)
    rh = get_landmarks(results.right_hand_landmarks, 21)

    return np.concatenate([pose, face, lh, rh])  # ✅ Shape: (1662,)

sequence = []
seq_length = 30

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-seq_length:]

        if len(sequence) == seq_length:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            pred_class = actions[np.argmax(res)]
            confidence = np.max(res)

            if confidence > 0.6:
                cv2.putText(image, f'{pred_class} ({confidence:.2f})', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('ISL Healthcare Symptom Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
