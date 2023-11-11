import os
import numpy as np
import guitarpro
import librosa
import tensorflow as tf


class TabProcessor:
    def __init__(self, file_path, audio_data, depth=100, skip=1, window_size=32):
        self.raw_tabs = guitarpro.parse(file_path)  # Read gp5 file
        [self.prep_tabs, self.time] = self.read_gp5_file()  # Preprocess gp5 file

        self.tempo = self.raw_tabs.tempo  # Parse tempo

        # Audio info (move to function)
        frames = range(audio_data.shape[1])
        time = librosa.frames_to_time(frames, sr=22050)
        time_ms = time * 1000

        self.time_cast = self.cast_tabs(audio_data, time_ms, self.prep_tabs,
                                        self.time)  # Tabs cast to mp3 spectrogram timesteps

        mapping = {0: -1}
        for idx, num in enumerate(range(29, 128)):
            mapping[num] = idx

        self.custom_mapping = self.apply_custom_mapping(mapping)

        self.onehot = self.encode_to_onehot(depth)  # Tabs converted to one hot encoding
        self.windows = self.sliding_window(window_size, skip)  # Windowed tabs ready for NN training
        pass

    def encode_to_onehot(self, depth):
        onehot = tf.one_hot(self.custom_mapping, depth)  # N categories, from 0 to N-1
        onehot = np.max(onehot, axis=0)  # converts to shape (batch_size, time_steps, 32)
        return onehot

    def sliding_window(self, window_size, skip=1):
        tabs = self.onehot.T
        tabs_w = []

        # Ensure the loop starts at 0
        for i in range(0, len(tabs[0]) - window_size + 1, skip):
            w = tabs[:, i:(i + window_size)]
            tabs_w.append(w)

        return np.array(tabs_w)

    def apply_custom_mapping(self, mapping):
        ##
        # Custom mapping
        # tabs_processed = self.prep_tabs
        # missing_values = np.unique([i for i in tabs_processed.flatten() if i not in mapping.keys()])
        # default_value = -1
        # mapping = {**mapping, **{key: default_value for key in missing_values}}
        # tabs_m = np.vectorize(mapping.get)(tabs_processed)
        # tabs_m = tabs_m[:, np.newaxis]

        depth = len(mapping) - 1
        # tabs_m = np.squeeze(tabs_m, 1)
        # tabs_m = tabs_m.transpose(1, 0, 2)  # Swap the last two dimensions
        ##
        tabs_processed = self.time_cast
        missing_values = np.unique([i for i in tabs_processed.flatten() if i not in mapping.keys()])
        default_value = -1
        mapping = {**mapping, **{key: default_value for key in missing_values}}
        tabs_m = np.vectorize(mapping.get)(tabs_processed)
        return tabs_m

    @staticmethod
    def cast_tabs(audio_data, time_mp3, tabs, time):
        tabs_processed = np.zeros([4, audio_data.shape[1]])
        for idx, note in enumerate(tabs):
            time_ms = time[idx]
            idx_note = TabProcessor.find_nearest(time_mp3, time_ms)
            tabs_processed[:, idx_note] = note
        return tabs_processed

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    # def load_gp5_folder(self, gp5_directory):
    #    tabs = []
    #    raw_tabs = []
    #    tempo = []
    #    for file in os.listdir(gp5_directory):
    #        if file.endswith('.gp5'):
    #            tab_data, raw_tabs, tempo = self.read_gp5_file(os.path.join(gp5_directory, file))
    #            tabs.append(tab_data)
    #            raw_tabs.append(raw_tabs)
    #            tempo.append(tempo)
    #    self.prep_tabs = np.asarray(tabs)
    #    self.raw_tabs = np.asarray(raw_tabs)
    #    self.tempo = np.asarray(tempo)
    #    pass

    def read_gp5_file(self):
        # (same content as the original function with minor modifications)
        raw_song = self.raw_tabs
        tab_data = []
        time_ms = []
        offset = (60 / raw_song.tempo) * 1000
        offset = 0
        j = 0

        # Find the percussion track
        percussion_track = None
        for track in raw_song.tracks:
            if track.isPercussionTrack:
                percussion_track = track
                break

        # If no percussion track is found, return empty arrays
        if percussion_track is None:
            return [np.array([]), np.array([])]

        # Process the percussion track
        for measure in percussion_track.measures:
            for beat in measure.voices[0].beats:
                beat_data = {}
                buffer = [0] * 4
                i = 0
                for note in beat.notes:
                    buffer[i] = note.value
                    i += 1
                beat_data = buffer
                duration = beat.duration.value
                if j == 0:
                    time_data = (4 / duration) * (60 / raw_song.tempo) * 1000 + offset
                else:
                    time_data = (4 / duration) * (60 / raw_song.tempo) * 1000 + time_ms[-1]
                time_ms.append(time_data)
                tab_data.append(beat_data)
                j += 1

        return [np.array(tab_data), np.array(time_ms)]

# tab_processor = TabProcessor()  # Create an instance of the class
# print(tab_processor)  # Print the instance to verify
