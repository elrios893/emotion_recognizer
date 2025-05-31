import os
from glob import glob
# emotion_recognition_project/utils/label_features.py

def extract_emotion(filename):
    emotion_code = int(filename.split("-")[2])
    emotions = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotions.get(emotion_code, "unknown")


def extract_filenames():
    # Ruta a la carpeta donde están los audios
    DATASET_PATH = "emotion_recognition_project/audios"
    audio_files = glob(os.path.join(DATASET_PATH, "**/*.wav"), recursive=True)

# Crear una lista con tuplas: (ruta_archivo, emoción)
    data = []

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        emotion = extract_emotion(filename)
        data.append((file_path, emotion))
    return data
    

