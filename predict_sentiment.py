import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy import signal
import os

# ---------------- SETTINGS ----------------
MODEL_PATH = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\Sentiment-Analysis-From-Voice-\ser_model.h5"
SAMPLE_RATE = 8000

# ---------------- LABEL MAP ----------------
reverse_map = {
    0: "positive",
    1: "negative",
    2: "neutral"
}

# ---------------- AUDIO RESAMPLING ----------------
def resample_audio(data, orig_sr, target_sr=8000):
    if orig_sr == target_sr:
        return data, orig_sr

    duration = len(data) / orig_sr
    new_length = int(duration * target_sr)

    resampled = signal.resample(data, new_length)
    return resampled.astype(np.float32), target_sr

# ---------------- SPECTROGRAM EXTRACTION ----------------
def extract_spectrogram(filepath):
    sr, data = wavfile.read(filepath)

    # Convert stereo to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample
    data, sr = resample_audio(data, sr, SAMPLE_RATE)

    # Spectrogram
    _, _, Sxx = signal.spectrogram(data, sr)

    # Log scale + normalization
    Sxx = np.log(Sxx + 1e-10)
    Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-9)

    # Resize to model input size
    Sxx = tf.image.resize(Sxx[..., np.newaxis], [128, 128])

    return tf.squeeze(Sxx).numpy()

# ---------------- LOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    print("Model file not found!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully")

# ---------------- TESTING ----------------
path = input("Enter WAV file path: ")

if os.path.exists(path):

    spec = extract_spectrogram(path)

    # Add batch + channel dimension
    spec = spec[np.newaxis, ..., np.newaxis]

    # Prediction
    prediction = model.predict(spec)

    predicted_class = np.argmax(prediction)

    predicted_label = reverse_map[predicted_class]

    confidence = np.max(prediction) * 100

    print("\nPredicted Sentiment :", predicted_label)
    print("Confidence           :", round(confidence, 2), "%")

else:
    print("File not found!")