import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import os
import pandas as pd
import seaborn as sns
MODEL_PATH = "ser_model.h5"
EPOCH_FILE = "last_epoch.txt"
MORE_EPOCHS = 100
# ---------------- AUDIO RESAMPLING ----------------
def resample_audio(data, orig_sr, target_sr=8000):
    if orig_sr == target_sr:
        return data, orig_sr

    duration = len(data) / orig_sr
    new_length = int(duration * target_sr)

    resampled = signal.resample(data, new_length)
    return resampled.astype(np.float32), target_sr


# ---------------- DATASET PATHS ----------------
ravdess_path = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\RAVDESS DATA SET"
savee_path   = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\SAVEE DATASET\ALL"
crema_path   = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\CREMA Dataset\AudioWAV"
tess_path    = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\TESS Data set"


# ---------------- GET WAV FILES ----------------
def get_wav_files(folder):
    files = []
    for root, _, fns in os.walk(folder):
        for f in fns:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))
    return files


ravdess_files = get_wav_files(ravdess_path)
savee_files   = get_wav_files(savee_path)
crema_files   = get_wav_files(crema_path)
tess_files    = get_wav_files(tess_path)

all_files = ravdess_files + savee_files + crema_files + tess_files

# ---------------- EMOTION LABEL EXTRACTION ----------------
def extract_label(filepath):
    filename = os.path.basename(filepath)

    if "-" in filename:
        parts = filename.split("-")
        if len(parts) > 2:
            emotion_map = {
                "01": "neutral","02": "calm","03": "happy","04": "sad",
                "05": "angry","06": "fear","07": "disgust","08": "surprise"
            }
            return emotion_map.get(parts[2], "unknown")

    if filename.startswith("a"): return "angry"
    elif filename.startswith("d"): return "disgust"
    elif filename.startswith("f"): return "fear"
    elif filename.startswith("h"): return "happy"
    elif filename.startswith("n"): return "neutral"
    elif filename.startswith("sa"): return "sad"
    elif filename.startswith("su"): return "surprise"

    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 3:
            crema_map = {
                "ANG":"angry","DIS":"disgust","FEA":"fear",
                "HAP":"happy","NEU":"neutral","SAD":"sad"
            }
            return crema_map.get(parts[2], "unknown")

    emotions = ["angry","disgust","fear","happy","neutral","sad","surprise"]
    for emo in emotions:
        if emo in filename.lower():
            return emo

    return "unknown"


# ---------------- EMOTION → SENTIMENT ----------------
emotion_to_sentiment = {
    "happy": "positive","surprise": "positive","calm": "positive",
    "sad": "negative","angry": "negative","fear": "negative","disgust": "negative",
    "neutral": "neutral"
}


# ---------------- SPECTROGRAM ----------------
SAMPLE_RATE = 8000

def extract_spectrogram(filepath):
    sr, data = wavfile.read(filepath)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data, sr = resample_audio(data, sr, SAMPLE_RATE)

    _, _, Sxx = signal.spectrogram(data, sr)

    Sxx = np.log(Sxx + 1e-10)
    Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-9)

    Sxx = tf.image.resize(Sxx[..., np.newaxis], [128, 128])

    return tf.squeeze(Sxx).numpy()


# ---------------- CREATE DATASET ----------------
print("\nCreating dataset...")

data_list = []

for file in all_files:
    emotion = extract_label(file)

    if emotion != "unknown":
        sentiment = emotion_to_sentiment.get(emotion, "unknown")

        if sentiment != "unknown":
            data_list.append([file, sentiment])

df = pd.DataFrame(data_list, columns=["filepath", "sentiment"])

df.to_csv("combined_dataset_3class.csv", index=False)

print("Dataset saved. Total samples:", len(df))


# ---------------- BEFORE BALANCING ----------------
print("\nBefore Balancing:")
print(df["sentiment"].value_counts())


# ---------------- BALANCING ----------------
min_count = df["sentiment"].value_counts().min()
min_count=3*min_count

df_balanced = df.groupby("sentiment").sample(
    n=min_count,
    replace=True,   # IMPORTANT
    random_state=42
)

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter Balancing:")
print(df_balanced["sentiment"].value_counts())


# ---------------- PLOT ----------------
df_balanced["sentiment"].value_counts().plot(kind='bar')
plt.title("Balanced Sentiment Distribution")
plt.xlabel("Sentiment Class")
plt.ylabel("Number of Samples")
plt.show()


# ---------------- LOAD DATA ----------------
files = df_balanced["filepath"].values
labels = df_balanced["sentiment"].values


# ---------------- FEATURE EXTRACTION ----------------
print("\nExtracting features...")

X, y = [], []

for file, label in zip(files, labels):
    try:
        spec = extract_spectrogram(file)

        # original sample
        X.append(spec)
        y.append(label)

        # augmented sample (noise)
        noise = np.random.normal(0, 0.01, spec.shape)
        X.append(spec + noise)
        y.append(label)

    except Exception as e:
        print("Error:", file, "|", e)

X = np.array(X)[..., np.newaxis]
y = np.array(y)

# ---------------- ENCODING ----------------
# ---------------- LABEL ENCODING ----------------

# Fixed mapping
label_map = {
    "positive": 0,
    "negative": 1,
    "neutral": 2
}

# Encode labels
y_encoded = np.array([label_map[label] for label in y])

# Reverse mapping (for prediction output)
reverse_map = {v: k for k, v in label_map.items()}
class_names = ["positive", "negative", "neutral"]

# Print mapping
print("\nLabel Mapping:")
for k, v in label_map.items():
    print(k, "→", v)


# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ---------------- MODEL ----------------


inputs = tf.keras.Input(shape=(128, 128, 1))

x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ---------------- LOAD EPOCH ----------------
if os.path.exists(EPOCH_FILE):
    try:
        initial_epoch = int(open(EPOCH_FILE).read().strip())
    except:
        initial_epoch = 0
else:
    initial_epoch = 0

print("Starting from epoch:", initial_epoch)

# ---------------- CALLBACKS ----------------
'''early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=20,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor='accuracy',
    save_best_only=True,
    verbose=1

)'''
# ---------------- TRAIN ----------------
#model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
TARGET_EPOCH = initial_epoch + MORE_EPOCHS

print(f"Training from epoch {initial_epoch} to {TARGET_EPOCH}")

history = model.fit(
    X_train, y_train,
    epochs=TARGET_EPOCH,
    initial_epoch=initial_epoch,
    batch_size=32,
    validation_split=0.2,
    #callbacks=[early_stop, checkpoint]
)
# ---------------- SAVE EPOCH ----------------
trained_epochs = len(history.history['loss'])
final_epoch = initial_epoch + trained_epochs

with open(EPOCH_FILE, "w") as f:
    f.write(str(final_epoch))

print("Training completed till epoch:", final_epoch)


# ---------------- TEST ----------------
loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc*100, "%")
# ---------------- REPORT ----------------
y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(
    y_test, y_pred,
    target_names=["positive","negative","neutral"]
))
# ---------------- EVALUATION METRICS ----------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test

print("\n=== Classification Report ===\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))


# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------- PREDICTION ----------------
print("\n=== Sentiment Prediction ===")

# Load model if exists
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded trained model")

path = input("Enter WAV file path: ")

if os.path.exists(path):
    spec = extract_spectrogram(path)
    spec = spec[np.newaxis, ..., np.newaxis]

    pred = model.predict(spec)
    label = reverse_map[np.argmax(pred)]

    print("Predicted Sentiment:", label)
else:
    print("File not found")
