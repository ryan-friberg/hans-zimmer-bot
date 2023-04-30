# Welcome to the Github Repo of this image-to-music pipeline! 

This repository is pretty simple and easy to follow. The pipeline leverages image-scraping of the internet to build the training datasets, and then trains the models using this gathered data. Please find below a description of each of the files and how to use them.

## Overview of the files

### image_dataset.py
This file is used to build a PyTorch dataset object which will scrape Bing.com for images if a data directory does not exist. The scraping and dataset building happen somewhat simultaneously as a result. To use this functionality, once can import ImageDataSet from this file and build it as any normal PyTorch initialization. The only difference is, the ImageDataSet object requires a set of data transformations as well as a list of labels or search queries used for image scraping. These labels are best kept to short descriptors such as "gloomy" or "bright" such that later on the in the pipeline, the Vision Transformer can predict a more labels per image however the scraper supporters arbitrarily long queries include those that are multiple words.

### youtube_scraper.py
This file is used to scrape the audio files from youtube given a list of labels (similar to image_dataset.py). The high-level functionality of the scraping is largely the same.

### music_processor.py
This file will convert the scraped audio files and convert them into a PyTorch dataset of spectrogram images.

### image_to_music.ipynb
This file is the full pipeline of image to music. In it, the image dataset is collected, the ViT is trained, the Stable Diffusion is trained, and the file ends with a full inference pass. As with any notebook, any one subsection of the code can be run individually (ie only generate images, or only train the model). Please keep in mind that as of right now, since the ViT training is quite fast on a GCP GPU, there was no implementation of checkpointing as of right now. Since Stable Diffusion is trained via a HuggingFace training script and it takes far longer to train, it will create a checkpoint directory.

### train_text_to_image.py & script.sh
These files are directly provided by Huggingface for the purpose of training stable diffusion on a new dataset. To run training, update script.sh with the appropriate directories and hyperparameters and simply run the file. Script.sh is essentially a wrapper for calling the python file.
