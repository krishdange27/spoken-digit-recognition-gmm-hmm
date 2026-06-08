from src.audio_feature_extraction import extract_mfcc, load_data
from src.hmm import train_hmm_models, predict_digit_hmm

# Train once
data_path = "recordings"
train_data, _ = load_data(data_path)

hmm_models = train_hmm_models(train_data)


def predict_from_file(file_path):
    mfcc = extract_mfcc(file_path)
    return predict_digit_hmm(mfcc, hmm_models)


if __name__ == "__main__":
    # Change this path to test another audio file
    file = "recordings/3_speaker6_1.wav"

    pred = predict_from_file(file)
    print("Input file:", file)
    print("Predicted digit:", pred)