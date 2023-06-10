import os
import csv
import librosa
import numpy as np
from tqdm import tqdm

data_path = 'data/genres_original/'

header = ['filename', 'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
          'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
          'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
          'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo']

for i in range(1, 21):
    header.append(f'mfcc{i}_mean')
    header.append(f'mfcc{i}_var')

header.append('label')

with open('data/music_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Loop through each subdirectory (genre) in the dataset
    for label in sorted(os.listdir(data_path)):

        # Ignore any non-directory files in the dataset
        if not os.path.isdir(os.path.join(data_path, label)):
            continue

        # Loop through each audio file in the current subdirectory
        for filename in tqdm(sorted(os.listdir(os.path.join(data_path, label)))):
            filepath = os.path.join(data_path, label, filename)

            # Load the audio file
            y, sr = librosa.load(filepath, mono=True, duration=30)

            # Extract the features
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
            mfcc_list = mfccs.tolist()
            means = np.mean(mfccs, axis=1).tolist()
            vars = np.var(mfccs, axis=1).tolist()

            # Append the features and label to the row
            row = [filename, len(y), chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
                   spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var,
                   rolloff_mean, rolloff_var, zero_crossing_rate_mean,
                   zero_crossing_rate_var, harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo]

            row += means
            row += vars

            row.append(label)

            # Append the row to the CSV file
            writer.writerow(row)
