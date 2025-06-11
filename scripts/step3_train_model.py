import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

data_root = 'datasets/ISL_Healthcare_Symptoms_Landmarks'

X_train = np.load(os.path.join(data_root, 'X_train.npy'))
X_test = np.load(os.path.join(data_root, 'X_test.npy'))
y_train = np.load(os.path.join(data_root, 'y_train.npy'))
y_test = np.load(os.path.join(data_root, 'y_test.npy'))
label_classes = np.load(os.path.join(data_root, 'label_classes.npy'))

y_train_cat = to_categorical(y_train, num_classes=len(label_classes))
y_test_cat = to_categorical(y_test, num_classes=len(label_classes))

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(30, 1662)))
model.add(Dropout(0.4))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_classes), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("✅ Starting training...")
model.fit(X_train, y_train_cat, epochs=30, validation_data=(X_test, y_test_cat), batch_size=8)

model.save(os.path.join(data_root, 'isl_symptom_model.h5'))

print("✅ Model trained and saved successfully!")
