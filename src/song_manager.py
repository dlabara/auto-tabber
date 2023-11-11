from data.audio_processor import AudioProcessor
from data.tab_processor import TabProcessor
import numpy as np
import os

class Song:
    def __init__(self, audio_file_path, tab_file_path, n_mels=512, skip=1, window_size=32):
        self.audio_processor = AudioProcessor(audio_file_path, n_mels=n_mels, window_size=window_size, skip=skip)
        self.tab_processor = TabProcessor(tab_file_path, self.audio_processor.audio_data, skip=skip,
                                          window_size=window_size)

        # if self.tab_processor.prep_tabs is not None:
        #    self.tabs_processed = self.tab_processor.cast_tabs(self.audio_data, self.time_mp3, self.tabs)



class SongManager:
    def __init__(self):
        self.songs = []

    def load_dataset(self, mp3_directory, gp5_directory, n_mels=512, skip=1, window_size=32):
        # List all files in each directory and strip the extensions
        mp3_files = set([os.path.splitext(f)[0] for f in os.listdir(mp3_directory) if f.endswith('.mp3')])
        gp5_files = set([os.path.splitext(f)[0] for f in os.listdir(gp5_directory) if f.endswith('.gp5')])

        # Find common files between the two sets
        common_files = mp3_files.intersection(gp5_files)
        for filename_without_ext in common_files:
            mp3_path = os.path.join(mp3_directory, filename_without_ext + '.mp3')
            gp5_path = os.path.join(gp5_directory, filename_without_ext + '.gp5')
            self.add_song(mp3_path, gp5_path, n_mels, skip, window_size)

    def add_song(self, audio_file_path, tab_file_path, n_mels, skip, window_size):
        song = Song(audio_file_path, tab_file_path, n_mels, skip, window_size=window_size)
        self.songs.append(song)

    def get_all_windows(self, window_size):
        # Initialize with the first song's windows
        audio_windows = np.transpose(self.songs[0].audio_processor.windows, (0, 2, 1))
        tab_windows = np.transpose(self.songs[0].tab_processor.windows, (0, 2, 1))

        # Continue concatenating from the second song onwards
        for song in self.songs[1:]:
            audio_windows = np.concatenate((audio_windows, np.transpose(song.audio_processor.windows, (0, 2, 1))), axis=0)
            tab_windows = np.concatenate((tab_windows, np.transpose(song.tab_processor.windows, (0, 2, 1))), axis=0)

        return [audio_windows, tab_windows]
