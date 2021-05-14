import pandas as pd
import torch
from torchvision import transforms
import timm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *
from utils import seed_everything, check_paths
from engine import train_fn, eval_fn
from dataset import EmotionDataset


def train_model(model, transforms_train, transforms_cv, augmentations, lr=1e-3, weight_decay=1e-3, batch_size_train=32,
                batch_size_cv=64, max_epochs=20, patience=3, train_size=0.9, seed=SEED, model_name='best_model.pt',
                retrain_from=None):
    """Train pretrained ReXNet

    Parameters
    ----------
    model : torch.nn.Module
        pytorch model to be trained

    transforms_train : torchvision.transforms.Compose
        torchvision transforms chained together with Compose given to the training dataset

    transforms_cv : torchvision.transforms.Compose
        torchvision transforms chained together with Compose given to the cross validation dataset

    augmentations : list of callable transformations
        Transformations called on the image before prediction for test-time augmentation

    lr : float, default=1e-3
        learning rate of the optimizer

    weight_decay : float, default=1e-3
        L2 regularisation given to the optimizer

    batch_size_train : int, default=32
        batch size for training

    batch_size_cv : int, default=64
        batch size for evaluation

    max_epochs : int, default=30
        maximum number of training epochs

    patience : int, default=3
        number of epochs without improvement on the cross validation set before terminating

    train_size : float, default=0.9
        fraction of the training set used for training. The rest is used as the cross validation set

    seed : int, default=24
        used to seed the data split and the model

    model_name : str, default='best_model.pt'
        filename of the best model that will be saved to the local directory during training

    retrain_from : str, default=None
        if 'retrain_from' is a string, it's expected to be a filename of model weights which will be loaded and trained
        from. Setting this parameter, unfreezes the entire model
    """

    print()
    check_paths()
    seed_everything(seed)

    # split data
    df_train = pd.read_csv(CSV_TRAIN, index_col=0)
    df_train, df_cv = train_test_split(df_train, stratify=df_train['emotion'], train_size=train_size)

    # create datasets
    ds_train = EmotionDataset(df_train, IMAGE_DIRECTORY_TRAIN, transforms_train)
    ds_cv = EmotionDataset(df_cv, IMAGE_DIRECTORY_TRAIN, transforms_cv)

    # create dataloaders
    dl_train = DataLoader(ds_train, batch_size=batch_size_train)
    dl_cv = DataLoader(ds_cv, batch_size=batch_size_cv)

    model.to(DEVICE)

    # load model weights if given if given
    if retrain_from is not None:
        print(f"Retraining from {retrain_from}\n")
        model.load_state_dict(torch.load(retrain_from))
        unfrozen = True  # stops the freezing of layers
    else:
        unfrozen = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy_cv = 0
    best_epoch = 0
    current_patience = 0
    current_layer = 16  # specifies the layer to be unfrozen

    last_accuracy = 0  # used to check for divergence
    for epoch in range(max_epochs):
        # terminate training if divergence
        if current_patience == patience:
            print("Patience reached. Terminating training\n")
            break

        # freeze all but the last fc layer for the first epoch
        if epoch == 0 and not unfrozen:
            for param in model.parameters():
                param.requires_grad = False
            model.head.fc.weight.requires_grad = True
            model.head.fc.bias.requires_grad = True

        elif not unfrozen:
            # unfreeze the entire model when all the feature blocks has been unfrozen
            if current_layer == 0:
                for param in model.parameters():
                    param.requires_grad = True
                unfrozen = True
            else:
                # unfreeze four feature layers
                for i in range(4):
                    for name, param in model.named_parameters():
                        if str(current_layer) in name:
                            param.requires_grad = True
                    current_layer -= 1  # decrement to keep track of the next layers to unfreeze

        print(f"Epoch {epoch}:")
        loss_train, accuracy_train = train_fn(model, dl_train, optimizer, criterion)
        print(f"    Training loss: {loss_train}")
        print(f"    Training accuracy: {accuracy_train}")

        accuracy_cv = eval_fn(model, dl_cv, augmentations)
        print(f"    Cross validation accuracy: {accuracy_cv}\n")

        if accuracy_cv > best_accuracy_cv:
            torch.save(model.state_dict(), model_name)
            best_accuracy_cv = accuracy_cv
            accuracy_train_of_best_model = accuracy_train
            best_epoch = epoch

        if accuracy_cv > last_accuracy:
            current_patience = 0
        else:
            current_patience += 1
        last_accuracy = accuracy_cv

    print("Training complete\n")
    print(f"Best model achieved at epoch {best_epoch}")
    print(f"    Training accuracy: {accuracy_train_of_best_model}")
    print(f"    Cross validation accuracy: {best_accuracy_cv}\n")
    print(f"Model saved at: {model_name}")


if __name__ == '__main__':
    model = timm.create_model('rexnet_150', pretrained=True, num_classes=7)
    transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(5)]), 0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip()
    ])

    transforms_cv = transforms.Compose([
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

    train_model(model, transforms_train, transforms_cv, augmentations)
