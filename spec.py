import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import soundfile as sf
from torch import as_strided
import os
from sklearn.preprocessing import LabelEncoder

import json
from pathlib import Path
from warnings import warn
import torch.multiprocessing as mp
import h5py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# USER SET PARAMETERS
SET = "train_tiny"   # Which dataset to pull from
MAX_ITER = 200000    # How many waveforms to process

SAMPLE_RATE = 22050 # It seems that all the files are this sample rate
LAST_INDEX_PATH = "/work/cssema416/202610/19/data/" + SET + "_last_index.json"
H5_INTERMEDIATE_PATH = "/work/cssema416/202610/19/data/intermediate_data/" + SET + "_intermediate.hdf5"
H5_FINAL_PATH = "/work/cssema416/202610/19/data/" + SET + ".hdf5"
PT_INTERMEDIATE_WAVEFORMS_PATH = "/work/cssema416/202610/19/data/intermediate_data/" + SET + "_intermediate_waveforms.pt"
PT_INTERMEDIATE_LABELS_PATH = "/work/cssema416/202610/19/data/intermediate_data/" + SET + "_intermediate_labels.pt"
PT_FINAL_SPECTROGRAMS_PATH = "/work/cssema416/202610/19/data/" + SET + "_spectrogramss.pt"
PT_FINAL_LABELS_PATH = "/work/cssema416/202610/19/data/" + SET + "_labels.pt"
CUDA_ELEM_LIMIT = 134217728

def create_spec(wav_path, device=DEVICE):
    data, sample_rate = read_wav(wav_path)
    spec = audio_to_spectrogram(data, sample_rate, device)
    return spec

# Takes a .wav file and returns the data as a tensor and its sampling rate
def read_wav(wav_path):
    data, sample_rate = sf.read(wav_path)   # data: (num_samples, channels) or mono
    data = torch.from_numpy(data)

    if len(data.shape) == 2:
        data = data.mean(dim=1)

    return data, sample_rate

# Takes an audio data tensor and returns windows of specified length
# If there is a remainder of the audio data that is < the window length, it will pad the window with silence 
def audio_slide_windows(data, sample_rate, window_length_sec, stride_length_sec):
    # Calculate values relevant to striding
    window_size = int(window_length_sec * sample_rate)
    step = int(stride_length_sec * sample_rate)
    data_size = int(data.size(dim=0))

    # Pad if data is smaller than window size
    if data_size < window_size:
        pad_size = window_size - data_size
        data = torch.cat((data, torch.zeros(pad_size)))
        data_size = window_size

    num_windows = ((data_size - window_size) // step) + 1

    # Pad to ensure that whole file can be strided
    last_end_index = ((num_windows - 1) * step) + window_size
    if last_end_index < data_size:
        padding_amount = (num_windows * step) + window_size - data_size
        data = torch.cat((data, torch.zeros(padding_amount)))
        num_windows = num_windows + 1

    # Stride audio
    size = (num_windows, window_size)
    stride = (step, 1)
    windows = as_strided(data, size=size, stride=stride)

    # Return
    return windows

# Converts audio data to spectrogram image
def audio_to_spectrogram(data, sample_rate, device):
    # Convert tensor to float32
    waveform = data.detach().clone().to(dtype=torch.float32, device=device).unsqueeze(0)

    # Create spectrogram transform
    spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=64,
        f_min=0,
        f_max=sample_rate // 2,
    ).to(device)
    spec = spectrogram(waveform)
    
    # Only need one channel
    spec = spec[0]  # Now [64, time_steps]
    
    spec_db = T.AmplitudeToDB().to(device)(spec)
    spec = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)
    spec = spec.unsqueeze(0)

    return spec

# Parses training and testing files from directories and encodes labels
def parse_files(train_path='./train', test_path='./test', 
                       train_limit=60000, test_limit=1000):
    # train
    audio_train, labels_train = [], []
    for folder in os.listdir(train_path):
        folder_path = os.path.join(train_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                audio_train.append(os.path.join(folder_path, file))
                labels_train.append(folder[6:])
    audio_train, labels_train = audio_train[:train_limit], labels_train[:train_limit]

    # test
    audio_test, labels_test = [], []
    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                audio_test.append(os.path.join(folder_path, file))
                labels_test.append(folder[6:])
    audio_test, labels_test = audio_test[:test_limit], labels_test[:test_limit]

    # ecode the labels
    label_encoder = LabelEncoder()
    labels_train_numeric = label_encoder.fit_transform(labels_train)
    labels_train_numeric = torch.tensor(labels_train_numeric)

    num_classes = len(label_encoder.classes_)
    labels_train_numeric = labels_train_numeric.clamp(max=num_classes - 1)

    # Filter test set to only include labels seen in training set
    valid_test_indices = [i for i, l in enumerate(labels_test) if l in label_encoder.classes_]
    audio_test = [audio_test[i] for i in valid_test_indices]
    labels_test = [labels_test[i] for i in valid_test_indices]

    labels_test_numeric = label_encoder.transform(labels_test)
    labels_test_numeric = torch.tensor(labels_test_numeric).clamp(max=num_classes - 1)

    return audio_train, labels_train_numeric, audio_test, labels_test_numeric

# Preprocess dataset: read wav files, stride into windows, convert to spectrograms
def preprocess_dataset(audio_train, labels_train_numeric, audio_test, labels_test_numeric):
    device = "cpu"

    preprocessed_spectrograms_train = []
    preprocessed_labels_train = []
    preprocessed_spectrograms_test = []
    preprocessed_labels_test = []

    print("Preprocessing dataset...")
    # train
    for i in range(len(audio_train)):
        data, sample_rate = read_wav(audio_train[i])
        windows = audio_slide_windows(data, sample_rate, window_length_sec=3, stride_length_sec=1.5)
        for window in windows:
            spec = audio_to_spectrogram(window, sample_rate, device)
            if torch.isnan(spec).any():
                continue
            preprocessed_spectrograms_train.append(spec)
            preprocessed_labels_train.append(labels_train_numeric[i])

        if i % 50 == 0:
            print(f"Processed {i} / {len(audio_train)} training files")

    # test
    for i in range(len(audio_test)):
        data, sample_rate = read_wav(audio_test[i])
        hold = []
        windows = audio_slide_windows(data, sample_rate, window_length_sec=3, stride_length_sec=1.5)
        for window in windows:
            spec = audio_to_spectrogram(window, sample_rate, device)
            if torch.isnan(spec).any():
                continue
            hold.append(spec)
        if len(hold) > 0:
            preprocessed_spectrograms_test.append(hold)
            preprocessed_labels_test.append(labels_test_numeric[i])

    return (preprocessed_spectrograms_train, preprocessed_labels_train,
            preprocessed_spectrograms_test, preprocessed_labels_test)

# Plots spectrogram image
def plot_spectrogram(spec, title="Mel Spectrogram"):
    plt.figure(figsize=(8, 3))
    plt.imshow(spec.squeeze().numpy(), cmap='inferno', origin='lower', aspect='auto')
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.colorbar(format="%+2.f dB")
    plt.tight_layout()
    plt.show()

