import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd
import numpy as np
import os


def resample_audio(data, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return data, orig_sr

    duration = len(data) / orig_sr
    new_length = int(duration * target_sr)
    resampled = signal.resample(data, new_length)
    return resampled.astype(np.float32), target_sr



ravdess_path = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\RAVDESS DATA SET"
savee_path   = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\SAVEE DATASET\ALL"
crema_path   = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\CREMA Dataset\AudioWAV"
tess_path    = r"C:\Users\SUDIPAN SARKAR\OneDrive\Desktop\Final Year Project\TESS Data set"



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



print("\nSelect a dataset:")
print("1. RAVDESS")
print("2. SAVEE")
print("3. CREMA-D")
print("4. TESS")

choice = input("\nEnter your choice (1-4): ")

if choice == "1":
    selected_dataset = "RAVDESS"
    audio_files = ravdess_files
elif choice == "2":
    selected_dataset = "SAVEE"
    audio_files = savee_files
elif choice == "3":
    selected_dataset = "CREMA-D"
    audio_files = crema_files
elif choice == "4":
    selected_dataset = "TESS"
    audio_files = tess_files
else:
    print("Invalid choice!")
    exit()

print(f"\nSelected Dataset: {selected_dataset}")
print("\nAvailable audio files:")

for i, file in enumerate(audio_files):
    print(f"{i+1}. {os.path.basename(file)}")



file_choice = int(input("\nEnter file number to generate spectrogram: "))

if file_choice < 1 or file_choice > len(audio_files):
    print("Invalid file choice!")
    exit()

audio_path = audio_files[file_choice - 1]
print("\nSelected File:", audio_path)



print("\n▶ Playing Audio (resampled to 16 kHz)...")
try:
    sr, data = wavfile.read(audio_path)

    
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

   
    data, sr = resample_audio(data, sr, 16000)

    sd.play(data, sr)
    sd.wait()
    print("✔ Audio playback finished.")

except Exception as e:
    print(" Error playing audio:", e)


    sr, data = wavfile.read(audio_path)
except:
    print("Error: Cannot read WAV file.")
    exit()

if len(data.shape) > 1:
    data = np.mean(data, axis=1)


data, sr = resample_audio(data, sr, 16000)



frequencies, times, Sxx = signal.spectrogram(data, sr)

plt.figure(figsize=(7, 5))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto', cmap="jet")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Spectrogram - {os.path.basename(audio_path)} (16 kHz)")
plt.colorbar(label="Intensity (dB)")
plt.tight_layout()
plt.show()

print("\nDisplayed spectrogram for:", audio_path)