# music_processor.py
#
# COMS 4995 - Final Project
# NOTE: The system is used by running music_processor.py
import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import csv


# Called when music_processor.py is run
def process_file(file_path, label):
    train, test, data = "train/", "test/", "data/"

    # Load file and separate into 15 second parts
    segments = []

    y, sr = librosa.load(file_path)
    segment_seconds = 15
    segment_length = sr * segment_seconds
    number_sections = int(np.ceil(len(y)/segment_length))

    for i in range(min(number_sections, 15)):
        s = y[int((number_sections/15)*i) * segment_length:(int((number_sections/15)*i)+1) * segment_length]
        segments.append(s)

    index = 0
    test_idx = random.randint(0, 14)
    for segment in segments:
        # Converts audio segment to spectrogram
        spec = librosa.feature.melspectrogram(y=segment, sr=sr)

        # Uncomment to visualize:
        # visualize_spectrogram(spec)

        # Saves spectrogram image to data folder
        spec = Image.fromarray(spec).convert("L")
        filename = file_path.split("/")[-1].split(".")[0]
        if index == test_idx or index == test_idx - 14:
            spec.save(data + test + "/" + filename + "-" + str(index) + ".jpeg")
        else:
            spec.save(data + train + "/" + filename + "-" + str(index) + ".jpeg")

        index += 1


# Displays visualization of spectrogram, if needed
def visualize_spectrogram(spec):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


# Takes numpy array from saved image of spectrogram, converts back to audio file and saves to results folder
def convert_to_audio(spec, label):
    audio = librosa.feature.inverse.mel_to_audio(spec)
    reduced_noise = nr.reduce_noise(y=audio, sr=22050, prop_decrease=.9)
    sf.write('results/' + label + '.wav', reduced_noise, 22050)


# Takes the audio files scraped from YouTube by label and calls process_file() to convert to spectrograms
def transform_music(labels):
    for label in labels:
        folder_name = label.replace(" ", "_")
        print("Starting: " + folder_name)
        music_files = list(sorted(os.listdir(folder_name)))
        for music_file in music_files:
            process_file(os.path.join(folder_name, music_file), label)
        print("Finished: " + folder_name)


# Generates metadata.csv file needed to match captions to spectrograms in dataset
def create_csv(paths):
    fields = ["file_name", "caption_column"]
    captions = []

    for path in paths:
        images = list(sorted(os.listdir(path)))
        for image in images:
            captions.append([path.split("/")[-1] + "/" + image, " ".join(image.split("_")[:2]) + " spectrogram"])

    with open('data/metadata.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(captions)


# Called when music_processor.py is run
def main():
    labels = ['dark music', 'somber music', 'gloomy music', 'sad music',
              'bright music', 'happy music', 'cheerful music',
              'techno music', 'night club music', 'party music',
              'calm music', 'peaceful music', 'relaxing music',
              'classical music', 'classic music']

    transform_music(labels)


# Calls main() when music_processor.py is run
if __name__ == "__main__":
    main()
    create_csv(["data/train", "data/test"])

    # Uncomment to visualize saved spectrogram images
    # examples = ["##INSERT-EXAMPLE-NAME-HERE##"]

    # for example in examples:
    #    img = Image.open(example).convert('F')
    #    spec = np.array(img)
    #    visualize_spectrogram(spec)

    # Uncomment to convert saved spectrogram back to audio
    # examples = ["##INSERT-EXAMPLE-NAME-HERE##"]

    # for example in examples:
    #    img = Image.open(example).convert('F')
    #    spec = np.array(img)
    #    convert_to_audio(spec, example.split(".")[0])
