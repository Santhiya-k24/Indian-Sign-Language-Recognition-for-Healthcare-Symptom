import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

landmark_root = 'datasets/ISL_Healthcare_Symptoms_Landmarks'
X, y = [], []
fixed_length = 30
num_features = 1662

for symptom in sorted(os.listdir(landmark_root)):
    symptom_folder = os.path.join(landmark_root, symptom)
    if not os.path.isdir(symptom_folder):
        continue
    for file in os.listdir(symptom_folder):
        if file.endswith(".npy"):
            file_path = os.path.join(symptom_folder, file)
            data = np.load(file_path)
            if len(data.shape) != 2 or data.shape[1] != num_features:
                print(f"⚠️ Skipped {file_path} due to invalid shape {data.shape}")
                continue
            if data.shape[0] < fixed_length:
                pad_len = fixed_length - data.shape[0]
                pad = np.zeros((pad_len, num_features))
                data = np.vstack([data, pad])
            elif data.shape[0] > fixed_length:
                data = data[:fixed_length]
            X.append(data)
            y.append(symptom)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("❌ No valid landmark sequences found.")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

np.save(os.path.join(landmark_root, 'X_train.npy'), X_train)
np.save(os.path.join(landmark_root, 'X_test.npy'), X_test)
np.save(os.path.join(landmark_root, 'y_train.npy'), y_train)
np.save(os.path.join(landmark_root, 'y_test.npy'), y_test)
np.save(os.path.join(landmark_root, 'label_classes.npy'), le.classes_)

print(f"✅ Dataset prepared and saved.\nSamples: {len(X)} | Classes: {len(le.classes_)} | Shape: {X.shape}")
