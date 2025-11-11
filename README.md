# Nature-Noises-Classifier
Deep learning web app that classifies animal species from audio using CNNs and mel-spectrogram analysis.

This project is a Streamlit web app that identifies animal species based on short audio recordings.  
It uses a Convolutional Neural Network (CNN) trained on mel-spectrogram representations of sounds.

## Features
- Upload your own audio files or explore samples from our dataset  
- Visualize waveforms and mel spectrograms
- Run real-time model predictions and view top-10 species matches  
- Interactive, clean UI built with Streamlit 

## Model
- Architecture: Custom CNN built with PyTorch
- Input: Mel-spectrograms generated using Librosa
- Output: Predicted species from over 200,000 possible classes

## Run Locally
streamlit run app.py
