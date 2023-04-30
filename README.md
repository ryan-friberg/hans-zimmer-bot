# Welcome to the Github Repo of this image-to-music pipeline! 

This repository is pretty simple and easy to follow. The pipeline leverages image-scraping of the internet to build the training datasets, and then trains the models using this gathered data. Please find below a description of each of the files and how to use them.

## Overview of the files

### results
Examples of processed spectrogram data files into .wav format. Please download to hear examples!

### spectrogram_results
Examples of the raw output of the pipeline. These are the raw spectrogram generations from the fine-tuned stable diffusion. These images would then be able to be converted to .wav, or may need some denoising applied. Each file was generated with a unique string and you can see how this will greatly affect the general structure of the spectrogram.

### image_dataset.py
This file is used to build a PyTorch dataset object which will scrape Bing.com for images if a data directory does not exist. The scraping and dataset building happen somewhat simultaneously as a result. To use this functionality, once can import ImageDataSet from this file and build it as any normal PyTorch initialization. The only difference is, the ImageDataSet object requires a set of data transformations as well as a list of labels or search queries used for image scraping. These labels are best kept to short descriptors such as "gloomy" or "bright" such that later on the in the pipeline, the Vision Transformer can predict a more labels per image however the scraper supporters arbitrarily long queries include those that are multiple words.

### youtube_scraper.py
This file is used to scrape the audio files from youtube given a list of labels (similar to image_dataset.py). The high-level functionality of the scraping is largely the same.

### music_processor.py
This file will convert the scraped audio files into separate 15 second segments and convert those audio segments into mel-scaled spectrograms, generating a dataset of 3400+ spectrograms. Supplemental functions can visualize the spectrograms in program, convert spectrograms back to audio files, or produce the metadata.csv file that provides the necessary captions for the dataset.

### image_to_music.ipynb
This file is the full pipeline of image to music. In it, the image dataset is collected, the ViT is trained, the Stable Diffusion is trained, and the file ends with a full inference pass. As with any notebook, any one subsection of the code can be run individually (ie only generate images, or only train the model). Please keep in mind that as of right now, since the ViT training is quite fast on a GCP GPU, there was no implementation of checkpointing as of right now. Since Stable Diffusion is trained via a HuggingFace training script and it takes far longer to train, it will create a checkpoint directory.

### train_text_to_image.py & script.sh
These files are directly provided by Huggingface for the purpose of training stable diffusion on a new dataset. To run training, update script.sh with the appropriate directories and hyperparameters and simply run the file. Script.sh is essentially a wrapper for calling the python file. Limited modification has been made to these files to tailor training to our purposes.


# References
https://huggingface.co/docs/transformers/model_doc/vit

https://huggingface.co/blog/fine-tune-vit

https://www.riffusion.com/about

https://huggingface.co/docs/diffusers/training/text2image
