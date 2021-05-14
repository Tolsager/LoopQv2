import pandas as pd
import torch
from torchvision import transforms
import timm
from torch.utils.data import DataLoader
from config import *
from utils import check_paths
from engine import infer_fn
from dataset import EmotionDataset


def get_inferences(model_name, transforms_test, augmentations, batch_size=64):
    """Computes the inferences and saves it to a csv

    The function uses the dataframe at CSV_TEST specified in config to get the image ids and then expect the images
    to be in the directory specified by IMAGE_DIRECTORY_TEST in config.
    The inferences is saved as 'inferences.csv' in the current directory.

    Parameters
    ----------
    model_name : str
        path to model to infer on

    transforms_test : torchvision.transforms.Compose
        torchvision transforms chained together with Compose given to the test dataset

    augmentations : list of callable transformations
        Transformations called on the image before prediction for test-time augmentation

    batch_size : int, default=64
        batch size for inference'
    """

    print()
    check_paths(infer=True)

    df = pd.read_csv(CSV_TEST, index_col=0)

    # create dataset
    ds = EmotionDataset(df, IMAGE_DIRECTORY_TEST, transforms_test, has_label=False)

    # create dataloader
    dl = DataLoader(ds, batch_size=batch_size)

    model = timm.create_model('rexnet_150', num_classes=7)
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_name))

    inferences = infer_fn(model, dl, augmentations)

    df['emotion'] = inferences
    df.to_csv('inferences.csv')


if __name__ == '__main__':
    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augmentations = [
        transforms.GaussianBlur(5),
        transforms.RandomRotation([15, 15]),
        transforms.RandomRotation([-15, -15]),
        transforms.RandomHorizontalFlip(p=1.0)
    ]

    model_name = 'RexNet9.pt'
    get_inferences(model_name, transforms_test, augmentations)
