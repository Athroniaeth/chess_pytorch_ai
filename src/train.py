import logging
import statistics
from typing import Literal, Type

import torch
import typer
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import MODEL_PATH


def train_model(
        model: nn.Module,

        train_loader: DataLoader,
        tests_loader: DataLoader,
        validation_loader: DataLoader,

        num_epochs: int = 20,
        device: Literal['cpu', 'cuda'] = 'cuda',

        optimizer: Type[Optimizer] = torch.optim.Adam,
        criterion: Type[_Loss] = nn.MSELoss,

        lr: float = 1e-4,
        tolerance: float = 1e-2,

        scheduler: Type[LRScheduler] = LinearLR,
        start_factor: float = 1.0,
        end_factor: float = 1e-6,
        total_iters: int = 10
):
    train_score, tests_score, validation_score = 0, 0, 0

    # Crée le modèle
    model = model.to(device)
    logging.info(f"Device: {model.device}")

    # noinspection PyArgumentList
    optimizer = optimizer(model.parameters(), lr=lr)

    # noinspection PyArgumentList
    criterion = criterion()

    # noinspection PyArgumentList
    scheduler = scheduler(
        optimizer=optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters
    )

    # Entraîne le modèle
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)

        list_loss = []
        for x, y in pbar:
            x = x.to(device=device)
            y = y.to(device=device).view(-1, 1)  # Met 'y' en batch_size de 1

            predictions = model(x)

            loss = criterion(predictions, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epochs: {epoch+1}/{num_epochs}  -  Loss: {loss.item():.1e}")
            list_loss.append(loss.item())

        scheduler.step()
        # afficher sous forme 1e-3, 2e-3, 3e-3, etc.
        tqdm.write(f"Learning rate: {scheduler.get_last_lr()[0]:.1e}")
        train_score = check_accuracy(train_loader, model, tolerance)
        tests_score = check_accuracy(tests_loader, model, tolerance)

        avg_loss = statistics.mean(list_loss)
        tqdm.write(f"Epochs: {epoch+1}/{num_epochs}  -  Loss: {avg_loss:.1e}  -  Train Accuracy: {train_score:.2f}%  -  Tests Accuracy: {tests_score:.2f}%")

        # Sauvegarde le modèle
        path = MODEL_PATH / f"model_{epoch+1}.pth"
        torch.save(model.state_dict(), path)

    # Affiche les derniers scores sur tout le dataset (tout les batchs)
    validation_score = check_accuracy(validation_loader, model, tolerance)

    logging.info(f"[Score global train] : {train_score:.2f}%")
    logging.info(f"[Score global tests] : {tests_score:.2f}%")
    logging.info(f"[Score global validation] : {validation_score:.2f}%")


def check_accuracy(
        loader: DataLoader,
        model: nn.Module,
        tolerance: float = 1e-2,
):
    """
    Fonction pour calculer la précision du modèle
    """
    num_correct = 0
    num_samples = 0

    # Récupère le device du modèle
    device = model.device

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device).view(-1, 1)  # Met 'y' en batch_size de 1

            predictions = model(x)

            num_correct, num_samples = _calcul_accuracy(predictions, y, tolerance)

    calcul = num_correct / num_samples * 100

    return calcul


def _calcul_accuracy(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        tolerance: float = 1e-2):
    # Calculer le ratio entre les prédictions et les valeurs réelles
    ratio = predictions / labels

    # S'assurer que les ratios sont toujours >= 1
    mask = ratio < 1
    ratio[mask] = 1 / ratio[mask]

    # Vérifier si les prédictions sont dans la tolérance
    correct = (ratio <= (1 + tolerance)).sum().item()
    total = predictions.size(0)

    return correct, total
