# music_processor.py
#
# COMS 4995 - Final Project
# NOTE: The system is used by running music_processor.py
import os
import librosa
import soundfile as sf
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image


# Called when music_processor.py is run
def process_file(file_path, label):
    train, test, data = "train/", "test/", "data/"

    # Load file and separate into 15 second parts
    segments = []

    y, sr = librosa.load(file_path)
    segment_seconds = 15
    segment_length = sr * segment_seconds
    number_sections = int(np.ceil(len(y)/segment_length))

    for i in range(min(number_sections, 5)):
        s = y[int((number_sections/5)*i) * segment_length:(int((number_sections/5)*i)+1) * segment_length]
        segments.append(s)

    index = 0
    test_idx = random.randint(0, 4)
    for segment in segments:
        # Converts audio segment to spectrogram
        spec = librosa.feature.melspectrogram(y=segment, sr=sr)

        # Uncomment to visualize:
        # visualize_spectrogram(spec)

        # Saves spectrogram image to data folder
        spec = Image.fromarray(spec).convert("L")
        filename = file_path.split("/")[-1].split(".")[0]
        if index == test_idx:
            spec.save(data + test + label + "/" + filename + "-" + str(index) + ".jpeg")
        else:
            spec.save(data + train + label + "/" + filename + "-" + str(index) + ".jpeg")

        index += 1


# Displays visualization of spectrogram, if needed
def visualize_spectrogram(spec):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


def convert_to_audio(spec):
    audio = librosa.feature.inverse.mel_to_audio(spec)
    sf.write('stereo_file1.wav', audio, 22050)


def transform_music(labels):
    for label in labels:
        folder_name = label.replace(" ", "_")
        print("Starting: " + folder_name)
        music_files = list(sorted(os.listdir(folder_name)))
        for music_file in music_files:
            process_file(folder_name + "/" + music_file, label)
        print("Finished: " + folder_name)


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

    # Uncomment to visualize saved spectrogram images
    # examples = ["##INSERT-EXAMPLE-NAME-HERE##"]

    # for example in examples:
    #    img = Image.open(example).convert('F')
    #    spec = np.array(img)
    #    visualize_spectrogram(spec)

