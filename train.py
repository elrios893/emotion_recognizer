import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.audio_features import extract_mfcc
from utils.label_features import extract_filenames
from tqdm import tqdm
import joblib

import librosa
import numpy as np

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# X = características (MFCC), y = etiquetas (emociones)
X = []
y = []

data = extract_filenames()
import pandas as pd

df = pd.DataFrame(data, columns=["file_path", "emotion"])
print(df.head())

for file_path, emotion in data:
    features = extract_features(file_path)
    X.append(features)
    y.append(emotion)
    
# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
print("Entrenando modelo...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluar
print("Evaluando...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar modelo
joblib.dump(model, "model/emotion_model.pkl")
print("Modelo guardado en model/emotion_model.pkl")
