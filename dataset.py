import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class EmotionDataset(Dataset):
    def __init__(self, df, image_directory, transforms, has_label=True):
        self.df = df
        self.image_directory = image_directory
        self.transforms = transforms
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        # get the path to the time
        image_id = self.df['image_id'].iloc[i]
        image_path = os.path.join(self.image_directory, image_id + '.jpg')

        # load image and convert to RGB
        image = Image.open(image_path)
        image = image.convert('RGB')

        # preprocessing transformations + augmentations
        image = self.transforms(image)

        # return the image and label if specified
        if self.has_label:
            emotion = self.df['emotion'].iloc[i]
            emotion = torch.tensor(emotion, dtype=torch.long)
            return image, emotion
        else:
            return image
