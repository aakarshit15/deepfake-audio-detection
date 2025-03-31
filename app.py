import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("deepfake_audio_model.h5")

# Function to extract features from uploaded audio file
def extract_features(file_path, max_time_steps=130):
    y, sr = librosa.load(file_path, sr=22050, duration=3.0)
    
    # Compute Mel-spectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, 
                                          n_fft=2048, hop_length=512)
    log_mels = librosa.power_to_db(mels, ref=np.max)
    
    # Fix time dimension
    if log_mels.shape[1] > max_time_steps:
        log_mels = log_mels[:, :max_time_steps]
    elif log_mels.shape[1] < max_time_steps:
        pad_width = max_time_steps - log_mels.shape[1]
        log_mels = np.pad(log_mels, ((0, 0), (0, pad_width)), mode='constant')
    
    # Compute deltas
    delta = librosa.feature.delta(log_mels)
    delta2 = librosa.feature.delta(log_mels, order=2)
    
    # Stack features
    features = np.stack([log_mels, delta, delta2], axis=-1)
    features = (features - np.mean(features)) / (np.std(features) + 1e-9)
    return np.expand_dims(features, axis=0)  # Add batch dimension

# Streamlit UI
st.title("Deepfake Audio Detector 🎵🤖")
st.write("Upload an audio file (.wav) to check if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    features = extract_features("temp_audio.wav")
    
    # Make prediction
    prediction = model.predict(features)[0, 0]
    result = "Fake" if prediction > 0.5 else "Real"
    
    st.write(f"**Prediction: {result}**")
    st.write(f"Confidence Score: {prediction:.4f}")
