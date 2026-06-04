import os
import glob
import numpy as np
import pandas as pd
import librosa
import pickle
import argparse
from tqdm import tqdm

def get_log_mel_spectrogram(file_path, sr=44100, n_mels=128, n_fft=2048, hop_length=1024):
    """
    Compute Log-Mel Spectrogram for a given audio file.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.

    Returns:
        np.array: Log-Mel spectrogram.
    """
    y, _ = librosa.load(file_path, sr=sr)
    
    target_len = sr * 10 
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel.astype(np.float32)

def main(args):
    """
    Main preprocessing function to encode DCASE data into pickle files.
    """
    df = pd.read_csv(os.path.join(args.dataset_dir, "meta.csv"), sep="\t")
    
    labels = sorted(df['scene_label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"Classes: {label2id}")

    grouped = df.groupby('source_label') # a, b, c, s1...
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for source_name, group in grouped:
        print(f"Processing Source: {source_name} ({len(group)} files)...")
        
        data_list = []
        label_list = []
        
        for _, row in tqdm(group.iterrows(), total=len(group)):
            file_path = os.path.join(args.dataset_dir, row['filename']) # audio/airport-lisbon...
            
            try:
                log_mel = get_log_mel_spectrogram(file_path)
                
                data_list.append(log_mel)
                label_list.append(label2id[row['scene_label']])
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        save_path = os.path.join(args.output_dir, f"{source_name}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump((data_list, label_list), f)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./Datasets/dcase_data/TAU-urban-acoustic-scenes-2020-mobile-development")
    parser.add_argument("--output_dir", type=str, default="./Encoded_data_dcase")
    args = parser.parse_args()
    main(args)
