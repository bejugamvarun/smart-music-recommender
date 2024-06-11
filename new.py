import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import librosa
import pyaudio
import wave
from tkinter import Tk, filedialog

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "./output.wav"

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* recording")
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def extract_feature(file_name, mfcc=False, chroma=False, mel=False, contrast=False, tonnetz=False):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if X.ndim > 1:
            X = np.mean(X, axis=1)
        
        result = np.array([])
        stft = np.abs(librosa.stft(X)) if chroma or contrast else None
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feature))
        if mel:
            mel_feature = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feature))
        if contrast:
            contrast_feature = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast_feature))
        if tonnetz:
            tonnetz_feature = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz_feature))
        
        return result
    
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 1)),  # Define input shape explicitly here
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(rate=0.1),

        tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(rate=0.1),

        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(rate=0.1),

        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=5, activation='softmax')
    ])
    return model


def load_model(model_path, weights_path):
    # with open(model_path, 'r') as json_file:
    #     loaded_model_json = json_file.read()
    # model = tf.keras.models.model_from_json(loaded_model_json)
    model = build_model()
    model.load_weights(weights_path)
    return model

def main():
    record_audio()  # Record and save audio
    features = extract_feature(WAVE_OUTPUT_FILENAME, mfcc=True, chroma=True, mel=True)  # Extract features
    model = load_model('./model1.json', './Emotion_Voice_Detection_Model1.h5')  # Load the model
    features = np.expand_dims(np.array(features), axis=0)  # Reshape for prediction
    predictions = model.predict(features)
    print(predictions)

if __name__ == "__main__":
    main()
