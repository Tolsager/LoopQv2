import torch
import tqdm
from config import *


def train_fn(model, dl, optimizer, criterion):
    """Train the model for one epoch

    Parameters
    ----------
    model : torch.nn.Module
        Model to be updated

    dl : torch.utils.data.Dataloader
        Dataloader with the training data

    optimizer : torch.optim.Optimizer
        Optimizer used for training model

    criterion : torch.nn.loss
        Loss function

    Returns
    -------
    average_loss : float
        Average loss per sample

    accuracy : float
        Classification accuracy
    """

    # track total number of data points trained on
    n_samples = 0

    correct = 0
    total_loss = 0
    for X, y in tqdm.tqdm(dl):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # get model output
        predictions = model(X)

        # get loss and update weights
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update samples and total loss to compute accuracy at the end
        n_samples += X.shape[0]
        total_loss += loss.cpu().item()

        # add number of correct prediction sto correct
        predictions = torch.argmax(predictions, dim=1)
        correct += torch.sum(predictions == y).cpu().item()

    # return the average loss per sample along with the accuracy during training
    average_loss = total_loss / n_samples
    accuracy = correct / n_samples
    return average_loss, accuracy


def eval_fn(model, dl, augmentations):
    """Evaluate the model

    Parameters
    ----------
    model : torch.nn.Module
        Model to be updated

    dl : torch.utils.data.Dataloader
        Dataloader with the evaluation data

    augmentations : list of callable transformations
        Transformations called on the image before prediction for test-time augmentation

    Returns
    -------
    accuracy : float
        Classification accuracy
    """

    n_samples = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in tqdm.tqdm(dl):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # get the model prediction for each augmentation
            predictions = []
            for augmentation in augmentations:
                augmented = augmentation(X)
                prediction = torch.argmax(model(augmented), dim=1)
                predictions.append(prediction)

            predictions = torch.stack(predictions, dim=1)

            # decide the final prediction by majority-voting
            final_prediction = []
            for i in range(predictions.shape[0]):
                counts = torch.bincount(predictions[i, :])  # count the number of predictions of each class
                final_prediction.append(torch.argmax(counts))

            final_prediction = torch.stack(final_prediction)

            correct += torch.sum(final_prediction == y).detach().cpu().item()
            n_samples += X.shape[0]

        accuracy = correct / n_samples
        return accuracy


def infer_fn(model, dl, augmentations):
    """Get inferences from the model

        Parameters
        ----------
        model : torch.nn.Module
            Model to be updated

        dl : torch.utils.data.Dataloader
            Dataloader with the evaluation data

        augmentations : list of callable transformations
            Transformations called on the image before prediction for test-time augmentation

        Returns
        -------
        inferences : list
            The inferences in a list
        """

    inferences = []

    model.eval()
    with torch.no_grad():
        for X in tqdm.tqdm(dl):
            X = X.to(DEVICE)

            # get the model prediction for each augmentation
            predictions = []
            for augmentation in augmentations:
                augmented = augmentation(X)
                prediction = torch.argmax(model(augmented), dim=1)
                predictions.append(prediction)

            predictions = torch.stack(predictions, dim=1)

            # decide the final prediction by majority-voting
            # final_prediction = []
            for i in range(predictions.shape[0]):
                counts = torch.bincount(predictions[i, :])  # count the number of predictions of each class
                # final_prediction.append(torch.argmax(counts))
                inferences.append(torch.argmax(counts).detach().cpu().item())

        return inferences
