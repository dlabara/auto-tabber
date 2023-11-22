import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler



class AudioProcessor:
    def __init__(self, file_path, n_mels=512, window_size=32, skip=1):
        self.n_mels = n_mels
        self.audio_data = self.process_audio_file(file_path)
        frames = range(self.audio_data.shape[1])
        hop_length_used = 512  # or whatever value you used
        time = librosa.frames_to_time(frames, sr=22050, hop_length=hop_length_used)
        self.time_ms = time * 1000
        self.windows = self.sliding_window(self.audio_data, window_size, skip)

    @staticmethod
    def resample_audio(file_path, target_sr):
        # Load the audio with its original sample rate
        y, original_sr = librosa.load(file_path, sr=None)

        # Resample the audio to the target sample rate
        y_resampled = librosa.resample(y, original_sr, target_sr)

        return y_resampled

    @staticmethod
    def sliding_window(audio, window_size, skip=1):
        audio_w = []
        #audio = np.transpose(audio, (0, 2, 1))

        for i in range(0, len(audio[0]) - window_size, skip):
            v = audio[:, i:(i + window_size)]
            audio_w.append(v)
        # audio_w = np.squeeze(audio_w, 1)
        return np.array(audio_w)

    # def sliding_window(X, window_size):
    #    Xs, Ys = [], []
    #    X = np.transpose(X, (0, 2, 1))
    #    for i in range(len(X[0]) - window_size):
    #        v = X[:, i:(i + window_size)]
    #        Xs.append(v)  # Predict the next frame after the window
    #    Xs = np.squeeze(Xs, 1)
    #    return np.array(Xs), np.array(Ys)

    def load_audios(self, mp3_directory):
        X = []
        for file in os.listdir(mp3_directory):
            if file.endswith('.mp3'):
                audio_data, time_mp3 = self.process_audio_file(os.path.join(mp3_directory, file))
                # Normalize data
                X.append(audio_data)
        return audio_data, time_mp3, X

    def process_audio_file(self, file_path):
        # (same content as the original function)
        y, sr = librosa.load(file_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512)
        log_S = librosa.power_to_db(S, ref=np.max)
        #scaled_log_S = scaler.fit_transform(log_S)

        # Fit and transform the audio data
        scaler = StandardScaler()  # Create a scaler object
        scaler = MinMaxScaler(feature_range=(-1, 1))  # Initialize the MinMaxScaler with desired range
        
        scaled_log_S = scaler.fit_transform(log_S.reshape(-1, 1))
        scaled_log_S = scaled_log_S.reshape(log_S.shape)
        # (remaining plotting code is omitted for brevity)
        return scaled_log_S

