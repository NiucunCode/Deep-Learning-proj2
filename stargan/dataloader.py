"""
Data Loader
"""

import glob
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", attributes=None):
        self.transform = transforms.Compose(transforms_)

        self.selected_attrs = attributes
        self.files = []
        self.labels = []
        self.label_index = 0

        for attr in attributes:
            # load images
            cur_attr_file = sorted(glob.glob(os.path.join(root, "%s/" % attr, "%s/" % mode) + "*.*"))
            self.files.extend(cur_attr_file)
            # labels
            self.label_zero = torch.zeros(len(attributes))
            self.label_zero[self.label_index] = 1
            for i in range(len(cur_attr_file)):
                self.labels.append(self.label_zero)

            self.label_index += 1

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        image = Image.open(filepath)

        # Convert grayscale images to rgb
        if image.mode != "RGB":
            image = to_rgb(image)

        img = self.transform(image)
        label = self.labels[index % len(self.labels)]

        return img, label

    def __len__(self):
        return len(self.files)
