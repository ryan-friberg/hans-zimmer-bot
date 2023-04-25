from bs4 import BeautifulSoup
import os
import json
import urllib.request, urllib.error, urllib.parse

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import itertools
import glob

class ImageDataSet(Dataset):
    def __init__(self, data_dir, label_names, transforms=None):
        self.data_dir = data_dir
        self.label_names = label_names
        self.transforms = transforms

        self.supported_file_types = [".png", ".jpg", ".jpeg"]
        print("Supported files:", self.supported_file_types)
    
        # if the dataset has not been downloaded, initiate the scrape
        # NOTE: Repeat runs may have different images
        if not os.path.exists(self.data_dir):
            print("Making data directory...")
            os.mkdir(self.data_dir)
            self.scrape_images()
            self.prune_data()
        
        image_files, labels = self.get_image_filenames_with_labels(self.data_dir)
        self.image_files = np.array(image_files)
        self.labels = np.array(labels)
        self.num_images = len(self.image_files)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_files[idx]).convert('RGB')
            label = self.labels[idx]
            if self.transforms is not None:
                image = self.transforms(image)
            return image, label
        except:
            return None
        
    def get_image_filenames_with_labels(self, images_dir):
        image_files = []
        labels = []
        
        files = os.listdir(images_dir)
        for name in files:
            if name == ".DS_Store":
                continue
            image_class_dir = os.path.join(images_dir, name)
            image_class_files = list(itertools.chain.from_iterable(
                [glob.glob(image_class_dir + '/*' + file_type) for file_type in self.supported_file_types]))
            image_files += image_class_files
            labels += [int(name)] * len(image_class_files)
        return image_files, labels
    
    def print_label_dist(self):
        return np.unique(self.labels, return_counts=True)
        
    def prune_data(self):
        print("pruning data")
        # remove failed downloads
        for label_file in os.listdir(self.data_dir):
            for filename in os.listdir(self.data_dir + label_file + '/'):
                ext = os.path.splitext(filename)[1]
                if ext not in self.supported_file_types:
                    os.remove(self.data_dir + label_file + '/' + filename)

    def scrape_images(self):
        print("Scraping images...")
        seen_images = set()
        label_count = 0
        for i, label in enumerate(self.label_names):
            print("===> Extracting '" + label + "' images...")

            # handle search queries of multiple words 
            label = label.split()
            label = '+'.join(label)

            # set up the bing query using the label name as the search query
            search_url = "http://www.bing.com/images/search?q=" + label + "&FORM=HDRSC2"
            header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
            soup = BeautifulSoup(urllib.request.urlopen(urllib.request.Request(search_url,headers=header)), 'html.parser')

            # extract the image files from the bing search results
            for a_tag in soup.find_all("a",{"class":"iusc"}): # potentially want to manually limit how many images per class
                try:
                    m = json.loads(a_tag["m"])
                    turl = m["turl"]
                    murl = m["murl"]
                except:
                    continue

                image_name = urllib.parse.urlsplit(murl).path.split("/")[-1]

                # remove instances of duplicate images
                if image_name in seen_images:
                    continue
                seen_images.add(image_name)

                # create a dictionary for each label
                label_dir = self.data_dir + str(i)
                if not os.path.exists(label_dir):
                    os.mkdir(label_dir)

                # attempt to extract and download the file
                try:
                    img = urllib.request.urlopen(turl).read()

                    # NOTE: line 112 sometimes fails to find an extension but this is handled in 
                    # get_image_filenames_with_labels by giving valid file extensions
                    ext = os.path.splitext(image_name)[1] 
                    if ext not in self.supported_file_types:
                        continue

                    # rename the image to be more generic for better data organization
                    name = label_dir + '/' + str(i) + '_' + str(label_count) + ext
                    file = open(name, 'wb')
                    file.write(img)
                    file.close()

                    label_count += 1
                except Exception as e:
                    # if the image fails to download, skip it
                    print("Image: ", image_name, "failed to download!")
                    continue
        print("Finished scraping images!")

def collate_fn(batch):
    # Filter failed images first
    batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    images = torch.stack([b[0] for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    
    return images, labels