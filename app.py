# website for presentation 
# To run this app, use the command:
# streamlit run app.py

import streamlit as st
import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import soundfile as sf
import json
from spec import create_spec
import torch
import torch.nn.functional as F
from CNN import CNNModel
import torch
import torchaudio.transforms as T
from collections import Counter
import json
import gdown
import os

MODEL_PATH = "cnn_model.pth"
# need compatable with github repo structure
DATA_DIR = "train_2/"

VALID_PATH = "data/label_metadata/train_tiny_id_to_valid.json" 
LABEL_DIR = "data/label_metadata/id_to_species.json"

GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1cik1DYKdagjUv0Jl42UrED3BMv13PFgz"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        r = requests.get(GDRIVE_URL, allow_redirects=True)
        r.raise_for_status()
        open(MODEL_PATH, 'wb').write(r.content)
        print("Model downloaded successfully.")
    except Exception as e:
        print("Failed to download model:", e)

# Load model once at the top of your app
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 212039  # original model
model = CNNModel(output_size=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Load species mapping
with open(LABEL_DIR, "r") as f:
    id_to_species = json.load(f)  # list of species

# Mel spectrogram transform (matches your training preprocessing)
mel_transform = T.MelSpectrogram(
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=64,
        f_min=0,
        f_max=22050 // 2,
    ).to(device)

st.markdown("""
    <style>
    /* Set background color */
    .stApp {
        background-color: #f9f9f9;
    }

    /* Optional: Make title and text more elegant */
    h1, h2, h3, h4, h5, h6 {
        color: #111111;
        font-family: 'Georgia', serif;
    }
    </style>
""", unsafe_allow_html=True)


st.image("title2.jpeg", use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: #333333;
              font-family: Georgia, serif; font-size: 1.1rem;'>
        Fall 2025 - CSSEMA 416 - Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: #333333;
              font-family: Georgia, serif; font-size: 1.1rem;'>
        Naziia Raitova, Ethan Shan, Lucas Watson, Shane Nguyen
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <p style='text-align: left; font-size: 1.05rem; color: #444444; text-indent: 40px;
              font-family: Georgia, serif; max-width: 800px; margin: 0 auto;'>
        This web app uses a Convolutional Neural Network (CNN) to identify animal species 
        from short audio clips. It visualizes the waveform and spectrogram of each sound, 
        then predicts the most likely species based on learned frequency patterns. 
        You can explore samples from the dataset or upload your own recordings to test the model.
    </p>
    <br><br>
    """,
    unsafe_allow_html=True
)

our_species = ['Pseudacris crucifer',
 'Cardinalis cardinalis',
 'Cyanocitta cristata',
 'Fringilla coelebs',
 'Agelaius phoeniceus',
 'Erithacus rubecula',
 'Parus major',
 'Poecile atricapillus',
 'Melospiza melodia',
 'Passer domesticus',
 'Phylloscopus collybita',
 'Sylvia atricapilla',
 'Thryothorus ludovicianus',
 'Troglodytes aedon',
 'Turdus merula',
 'Turdus migratorius',
 'Turdus philomelos',
 'Vireo olivaceus']

# Classes
with open(VALID_PATH, "r") as f:
    valid_info = json.load(f)  
valid_info = [v.lower() == true if isinstance(v, str) else v for v in valid_info]
valid_classes = sorted([c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))])
classes = valid_classes

# Select Input Source
st.write("#### Choose Input Source")
st.markdown("<br>", unsafe_allow_html=True)
input_mode = st.radio("**Select how you want to provide the audio:**", ["From Dataset", "Upload from Device"])
st.markdown("<br>", unsafe_allow_html=True)

# If dataset mode
if input_mode == "From Dataset":
    selected_class = st.selectbox("**Choose a Class to Explore:**", options=valid_classes, index=0)
    class_path = os.path.join(DATA_DIR, str(selected_class))
    audio_files = [f for f in os.listdir(class_path) if f.endswith((".wav", ".mp3"))]

    if not audio_files:
        st.warning("No valid files found for this class.")
        st.stop()

    selected_file = st.selectbox("**Choose an Audio Sample:**", options=audio_files)
    audio_path = os.path.join(class_path, selected_file)
    st.write(f"###### Selected Audio from class: `{selected_class}`")

# If upload mode
else:
    selected_label = st.selectbox("**Choose the Correct Species:**", options=our_species, index=0)
    uploaded_file = st.file_uploader("**Upload your own audio file:**", type=["wav"])
    if uploaded_file is not None:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        audio_path = uploaded_file.name
        st.write(f"###### Uploaded file: `{uploaded_file.name}`")
    else:
        st.info("Please upload an audio file to continue.")
        st.stop()

st.audio(audio_path)
st.markdown("<br><br>", unsafe_allow_html=True)

# Display waveform and spectrogram
def display_waveform_and_spectrogram(audio_path):
    st.write("#### Waveform")
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    ax.set_title("Waveform")
    st.pyplot(fig)
    st.markdown("<br>", unsafe_allow_html=True)

    st.write("#### Spectrogram")
    spec = create_spec(audio_path, device="cpu").squeeze().numpy()
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='mel', cmap='inferno', ax=ax2)
    ax2.set_title("Mel Spectrogram")
    st.pyplot(fig2)
    st.markdown("<br><br>", unsafe_allow_html=True)

display_waveform_and_spectrogram(audio_path)

# PREDICTIONS
st.write("#### Run Model Predictions")
st.markdown("<br>", unsafe_allow_html=True)

def compute_model_predictions(audio_path, top_k=10):
    try:
        spec = create_spec(audio_path, device=str(device))
        # make sure spec has 3 dimensions (1, n_mels, time_steps)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        
        # The model expects input of shape (batch, 1, 64, 130)
        # We need to split the spectrogram into chunks of 130 time steps
        target_time_steps = 130
        _, _, time_steps = spec.shape
        
        # Split into chunks
        chunks = []
        for i in range(0, time_steps, target_time_steps // 2):
            chunk = spec[:, :, i:i+target_time_steps]
            if chunk.shape[2] < target_time_steps:
                padding = target_time_steps - chunk.shape[2]
                chunk = F.pad(chunk, (0, padding))
            chunks.append(chunk)
        
        # Stack chunks and add channel dimension
        spec_chunks = torch.stack(chunks).squeeze(1).unsqueeze(1) 
        
        # Run model inference on all chunks
        all_predictions = []
        with torch.no_grad():
            for chunk in spec_chunks:
                chunk = chunk.unsqueeze(0)  # Add batch dimension
                outputs = model(chunk)
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities)
        
        # Average predictions across all chunks
        avg_probabilities = torch.mean(torch.cat(all_predictions, dim=0), dim=0)
        
        # Get top 10 predictions
        top_probs, top_indices = torch.topk(avg_probabilities, top_k)
        
        # Convert to CPU and numpy
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Map indices to species names
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            species_name = id_to_species[idx] if idx < len(id_to_species) else f"Unknown_{idx}"
            predictions.append((species_name, float(prob)))
        
        return predictions
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []
    
# Button to compute predictions
if st.button("Compute Predictions"):
    st.markdown("<br>", unsafe_allow_html=True)

    # Folder name looks like "00992_Animalia_Chordata_Amphibia_Anura_Hylidae_Pseudacris_crucifer" 
    if input_mode == "From Dataset": 
        index = int(selected_class[0:5])  # get the index from the folder name
        with open(LABEL_DIR, "r") as f: 
            id_to_species = json.load(f) 
            correct_label = id_to_species[index] 
    else: 
        correct_label = selected_label
    st.success(f"**Correct Species:** {correct_label}") 
    st.markdown("<br>", unsafe_allow_html=True) 

    st.write("#### Predictions:")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Computing predictions..."):
        predictions = compute_model_predictions(audio_path, top_k=10)
    
    if predictions:
        top_species, top_prob = predictions[0]
        if top_species == correct_label:
            # green for correct
            st.success(f"**Top Predicted Species:** {top_species} ({top_prob*100:.2f}%)")
        else:
            # yellow for incorrect
            st.warning(f"**Top Predicted Species:** {top_species} ({top_prob*100:.2f}%)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Complete Top 10 Predictions:**")
        
        for i, (species, prob) in enumerate(predictions, 1):
            percentage = prob * 100
            st.write(f"{i}. **{species}** — {percentage:.2f}%")
            
            # Add a progress bar for visual representation
            st.progress(float(prob))
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.error("Failed to compute predictions. Please check your audio file and model.")


st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
