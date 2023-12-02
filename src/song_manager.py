from src.data.audio_processor import AudioProcessor
from src.data.tab_processor import TabProcessor
import numpy as np
import pandas as pd
import gzip
import os 
import tensorflow as tf

# Processes song and gets all of the audio and tab windows
class Song:
    def __init__(self, audio_file_path, tab_file_path, n_mels, skip, window_size, hop_length, sr):
        self.file_path = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.audio_processor = AudioProcessor(audio_file_path, n_mels=n_mels, window_size=window_size, skip=skip, hop_length=hop_length, sr=sr)
        self.tab_processor = TabProcessor(tab_file_path, self.audio_processor.audio_data, skip=skip,
                                          window_size=window_size)

        # if self.tab_processor.prep_tabs is not None:
        #    self.tabs_processed = self.tab_processor.cast_tabs(self.audio_data, self.time_mp3, self.tabs)



class SongManager:
    def __init__(self):
        self.songs = []

    def load_dataset(self, mp3_directory, gp5_directory, n_mels, skip, window_size, hop_length, sr):        
        # List all files in each directory and strip the extensions
        mp3_files = set([os.path.splitext(f)[0] for f in os.listdir(mp3_directory) if f.endswith('.mp3')])
        gp5_files = set([os.path.splitext(f)[0] for f in os.listdir(gp5_directory) if f.endswith('.gp5')])

        # Find common files between the two sets
        common_files = mp3_files.intersection(gp5_files)
        for filename_without_ext in common_files:
            mp3_path = os.path.join(mp3_directory, filename_without_ext + '.mp3')
            gp5_path = os.path.join(gp5_directory, filename_without_ext + '.gp5')
            self.add_song(mp3_path, gp5_path, n_mels, skip, window_size, hop_length, sr=sr)

    def add_song(self, audio_file_path, tab_file_path, n_mels, skip, window_size, hop_length, sr):
        song = Song(audio_file_path, tab_file_path, n_mels, skip, window_size=window_size, hop_length=hop_length, sr=sr)
        self.songs.append(song)

    def save_processed_songs(self, output_directory):
        os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

        for song in self.songs:
            audio_windows = np.transpose(song.audio_processor.windows, (0, 2, 1)).astype(np.float16)
            tab_windows = np.transpose(song.tab_processor.windows, (0, 2, 1)).astype(np.float16)

            song_name = song.file_path
            audio_file_path = os.path.join(output_directory, f"{song_name}_audio.npz")
            tab_file_path = os.path.join(output_directory, f"{song_name}_tab.npz")

            # Save audio and tab windows as .npz files for each song
            np.savez(audio_file_path, audio_windows=audio_windows)
            np.savez(tab_file_path, tab_windows=tab_windows)

            # Compress the .npz files using gzip
            with open(audio_file_path, 'rb') as f_in:
                with gzip.open(audio_file_path + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)

            with open(tab_file_path, 'rb') as f_in:
                with gzip.open(tab_file_path + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)

            # Remove the original .npz files
            os.remove(audio_file_path)
            os.remove(tab_file_path)
            
    @staticmethod       
    def join_session_data(input_directory, output_directory, session_name):
        audio_data = []
        tab_data = []

        # Find all .npz.gz files in the input directory
        npz_files = [f for f in os.listdir(input_directory) if f.endswith('.npz.gz')]

        for file_name in npz_files:
            if file_name.endswith('_audio.npz.gz'):
                with gzip.open(os.path.join(input_directory, file_name), 'rb') as f:
                    audio_npz = np.load(f)
                    audio_data.append(audio_npz['audio_windows'])

            elif file_name.endswith('_tab.npz.gz'):
                with gzip.open(os.path.join(input_directory, file_name), 'rb') as f:
                    tab_npz = np.load(f)
                    tab_data.append(tab_npz['tab_windows'])

        # Concatenate all audio and tab data
        combined_audio = np.concatenate(audio_data, axis=0)
        combined_tab = np.concatenate(tab_data, axis=0)

        session_file_path = os.path.join(output_directory, f"{session_name}.npz")
        
        # Save the combined data into a single .npz file (session_name.npz)
        np.savez(session_file_path, audio_windows=combined_audio, tab_windows=combined_tab,
                 total_length=len(combined_audio))  # Storing total length as metadata
        
    def get_batch_length(self, file_path, batch_size):
        data = np.load(file_path)  # Use np.load to directly load the .npz file
        total_samples = data['total_length']  # Read total samples from metadata
        
        total_batches = total_samples // batch_size
        return total_batches
            
    def data_generator(self, file_path, batch_size):
        while True:
            data = np.load(file_path)
            audio_windows = data['audio_windows']
            tab_windows = data['tab_windows']
            total_samples_audio = audio_windows.shape[0]
            total_samples_tab = tab_windows.shape[0]

            remainder_audio = total_samples_audio % batch_size
            remainder_tab = total_samples_tab % batch_size

            adjusted_batch_size_audio = batch_size - remainder_audio if remainder_audio != 0 else batch_size
            adjusted_batch_size_tab = batch_size - remainder_tab if remainder_tab != 0 else batch_size

            audio_slice_end = total_samples_audio - remainder_audio if remainder_audio != 0 else total_samples_audio
            tab_slice_end = total_samples_tab - remainder_tab if remainder_tab != 0 else total_samples_tab

            dataset = tf.data.Dataset.from_tensor_slices((
                audio_windows[:audio_slice_end],
                tab_windows[:tab_slice_end]
            ))
            dataset = dataset.batch(batch_size)

            for batch in dataset:
                yield batch


    
    # OLD-----------------------------------
    def get_all_windows(self, window_size):
         # Initialize with the first song's windows
         audio_windows = np.transpose(self.songs[0].audio_processor.windows, (0, 2, 1))
         tab_windows = np.transpose(self.songs[0].tab_processor.windows, (0, 2, 1))

        # Continue concatenating from the second song onwards
         for song in self.songs[1:]:
            audio_windows = np.concatenate((audio_windows, np.transpose(song.audio_processor.windows, (0, 2, 1))), axis=0)
            tab_windows = np.concatenate((tab_windows, np.transpose(song.tab_processor.windows, (0, 2, 1))), axis=0)

         return [audio_windows, tab_windows]