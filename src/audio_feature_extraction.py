from pathlib import Path
import librosa

#extracting mfcc using librosa
def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    n_fft = min(2048, len(signal))

    mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=13,n_fft=n_fft)
    return mfcc.T


#load the data
def load_data(data_path):
    files = list(Path(data_path).glob("*.wav"))

    if not files:
        raise ValueError(f"No .wav files found in {data_path}")

    train_data = []
    test_data = []

    for file in files:
        # extract mfcc
        mfcc = extract_mfcc(file)

        # parse digit & speaker
        filename = file.stem  # e.g. "3_speaker2"

        parts = filename.split("_")
        digit = int(parts[0])
        speaker = int(parts[1].replace("speaker", ""))

        if speaker <= 5:
            train_data.append((mfcc, digit))
        else:
            test_data.append((mfcc, digit))

    return train_data, test_data

