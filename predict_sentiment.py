import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd
import scipy.io.wavfile as wav_writer
import tempfile
import os
import sys

# ---------------- SETTINGS ----------------
MODEL_PATH = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\PROJECTS\HUMAN SENTIMENT ANALYSIS\MODELS\ser_model.h5"
SAMPLE_RATE = 8000
RECORD_SECONDS = 5          # Duration of each real-time recording
CHANNELS = 1                # Mono capture

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

# ---------------- EXTRACT SPECTROGRAM FROM NUMPY ARRAY ----------------
def extract_spectrogram_from_array(data, sr):
    """Same pipeline as extract_spectrogram but takes raw numpy array directly."""

    # Ensure float32
    data = data.astype(np.float32)

    # Flatten if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to model's expected sample rate
    data, sr = resample_audio(data, sr, SAMPLE_RATE)

    # Spectrogram
    _, _, Sxx = signal.spectrogram(data, sr)

    # Log scale + normalization
    Sxx = np.log(Sxx + 1e-10)
    Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-9)

    # Resize to model input size
    Sxx = tf.image.resize(Sxx[..., np.newaxis], [128, 128])

    return tf.squeeze(Sxx).numpy()

# ---------------- PREDICT FROM SPECTROGRAM ----------------
def predict_sentiment(spec):
    spec_input = spec[np.newaxis, ..., np.newaxis]
    prediction = model.predict(spec_input, verbose=0)
    predicted_class = np.argmax(prediction)
    predicted_label = reverse_map[predicted_class]
    confidence = np.max(prediction) * 100
    return predicted_label, round(confidence, 2), prediction[0]

# ---------------- REAL-TIME CAPTURE ----------------
def record_audio(duration=RECORD_SECONDS, sample_rate=16000):
    """Records audio from the default microphone."""
    print(f"\n🎙️  Recording for {duration} seconds... Speak now!")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype='float32'
    )
    sd.wait()  # Block until recording is done
    print("✅ Recording complete.")
    return audio_data.flatten(), sample_rate

def run_realtime_mode():
    """Continuously record from mic and predict sentiment."""
    print("\n" + "="*50)
    print("   REAL-TIME SENTIMENT ANALYSIS MODE")
    print("="*50)
    print(f"Recording {RECORD_SECONDS}s clips from your microphone.")
    print("Press Ctrl+C to exit.\n")

    while True:
        input("Press ENTER to record a new clip (or Ctrl+C to quit)...")

        try:
            # Capture from microphone
            audio_data, sr = record_audio(duration=RECORD_SECONDS, sample_rate=16000)

            # Check if audio has content (not silent)
            if np.max(np.abs(audio_data)) < 1e-4:
                print("⚠️  No audio detected — please check your microphone.")
                continue

            # Extract features and predict
            spec = extract_spectrogram_from_array(audio_data, sr)
            label, confidence, raw_scores = predict_sentiment(spec)

            # Display results
            print("\n" + "-"*40)
            print(f"  Predicted Sentiment : {label.upper()}")
            print(f"  Confidence          : {confidence}%")
            print(f"  Raw Scores          : positive={raw_scores[0]:.3f}, "
                  f"negative={raw_scores[1]:.3f}, neutral={raw_scores[2]:.3f}")
            print("-"*40 + "\n")

        except Exception as e:
            print(f"❌ Error during recording/prediction: {e}")

# ---------------- LOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully")

# ---------------- MAIN MENU ----------------
print("\nHow would you like to run?")
print("  [1] Real-time microphone capture")
print("  [2] Provide a WAV file path")

choice = input("\nEnter choice (1 or 2): ").strip()

if choice == "1":
    run_realtime_mode()

elif choice == "2":
    path = input("Enter WAV file path: ").strip()

    if os.path.exists(path):
        spec = extract_spectrogram(path)
        label, confidence, raw_scores = predict_sentiment(spec)

        print("\n" + "-"*40)
        print(f"  Predicted Sentiment : {label.upper()}")
        print(f"  Confidence          : {confidence}%")
        print(f"  Raw Scores          : positive={raw_scores[0]:.3f}, "
              f"negative={raw_scores[1]:.3f}, neutral={raw_scores[2]:.3f}")
        print("-"*40)
    else:
        print("❌ File not found!")

else:
    print("Invalid choice. Exiting.")