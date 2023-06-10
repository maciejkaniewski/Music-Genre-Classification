import pickle
import argparse
import numpy as np
import librosa
import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
import random
from collections import Counter
import matplotlib.pyplot as plt

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(58, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)


def extract_features(file_name):
    y, sr = librosa.load(file_name, mono=True)

    if len(y) > 30 * sr:
        start = random.randint(0, len(y) - 30 * sr)
        y = y[start: start + 30 * sr]
    else:
        y = y[:30 * sr]

    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
    rms_mean = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y=y))
    harmony_mean = np.mean(librosa.effects.harmonic(y=y))
    harmony_var = np.var(librosa.effects.harmonic(y=y))
    perceptr_mean = np.mean(librosa.effects.percussive(y=y))
    perceptr_var = np.var(librosa.effects.percussive(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    means = np.mean(mfccs, axis=1).tolist()
    vars = np.var(mfccs, axis=1).tolist()

    features = [len(y), chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
                spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var,
                rolloff_mean, rolloff_var, zero_crossing_rate_mean,
                zero_crossing_rate_var, harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo]

    features += means
    features += vars

    return np.array(features)


def predict_genre(model, file_name, iterations):
    genres_counter = Counter()

    for _ in range(iterations):
        features = extract_features(file_name)
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        features = scaler.transform(features.reshape(1,-1))
        features_tensor = torch.from_numpy(features).float()
        with torch.no_grad():
            prediction = model(features_tensor)
        with open('model/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        predicted_genre = encoder.inverse_transform([torch.argmax(prediction).item()])
        genres_counter[predicted_genre[0]] += 1

    return genres_counter


def main():
    parser = argparse.ArgumentParser(description='Predict music genre')
    parser.add_argument('filename', type=str, help='The filename of the song you\'d like to predict the genre of')
    parser.add_argument('iterations', type=int, help='Number of iterations to predict the genre')
    args = parser.parse_args()

    model = MyModel()
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()

    genres_counter = predict_genre(model, args.filename, args.iterations)

    # Visualization
    labels, values = zip(*genres_counter.items())
    indexes = np.arange(len(labels))
    plt.bar(indexes, values, align='center')
    plt.xticks(indexes, labels)
    plt.ylabel('Counts')
    plt.xlabel('Genres')
    plt.title(f'Genre Prediction Counts for {args.filename}')
    plt.show()


if __name__ == '__main__':
    main()
