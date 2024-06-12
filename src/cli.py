import logging
from enum import StrEnum
from logging import getLevelName
from types import SimpleNamespace
from typing import Literal

import polars
import torch
import typer
from torchsummary import summary

from src import DATA_PATH
from src.dataset import ChessDataset, split_dataset, preprocess_dataframe
from src.model import ChessModel
from src.train import train_model
from src.utils import load_dotenv_kaggle

cli = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)


class Level(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@cli.callback()
def callback(
        ctx: typer.Context,

        kaggle_username: str = typer.Option(None, envvar="KAGGLE_USERNAME", help="Nom d'utilisateur à l'API Kaggle."),
        kaggle_key: str = typer.Option(None, envvar="KAGGLE_KEY", help="Token d'accès à l'API Kaggle."),

        logging_level: Level = typer.Option(Level.DEBUG, help="Niveau de log pour les logs de l'application."),
):
    """
    Initialisation du contexte de l'application CLI.

    Args:
        ctx (typer.Context): Contexte de la commande.
        kaggle_username (str): Nom d'utilisateur à l'API Kaggle.
        kaggle_key (str): Token d'accès à l'API Kaggle.
        logging_level (Level): Niveau de log pour les logs de l'application.

    Raises:
        typer.Exit: Si le token Hugging Face est manquant.
        typer.Exit: Si le token GitLab est manquant.

    Returns:
        SimpleNamespace: Objet contenant les paramètres de l'application.
    """
    logging_level = getLevelName(logging_level)
    logging.basicConfig(level=logging_level)

    ctx.obj = SimpleNamespace(
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
    )


@cli.command()
def download(
        kaggle_dataset: str = typer.Option('ronakbadhe/chess-evaluations', help="Nom du jeu de données Kaggle à télécharger."),
        kaggle_username: str = typer.Option(None, envvar="KAGGLE_USERNAME", help="Nom d'utilisateur à l'API Kaggle."),
        kaggle_key: str = typer.Option(None, envvar="KAGGLE_KEY", help="Token d'accès à l'API Kaggle."),

        output_filename: str = typer.Option("preprocess.parquet", help="Nom du fichier de sortie."),
        force_download: bool = typer.Option(False, help="Force le téléchargement du jeu de données."),
        max_length: int = typer.Option(100_000, help="Taille maximale du dataset"),
):
    """
    Télécharge le jeu de données depuis Kaggle.

    Args:
        kaggle_dataset (str): Nom du jeu de données Kaggle à télécharger.
        kaggle_username (str): Nom d'utilisateur à l'API Kaggle.
        kaggle_key (str): Token d'accès à l'API Kaggle.

        output_filename (str): Nom du fichier de sortie.
        force_download (bool): Force le téléchargement du jeu de données.
        max_length (int): Taille maximale du dataset.
    """
    # Récupère le nom du fichier
    input_filename = kaggle_dataset.split('/')[1]
    input_filename = input_filename.replace('-', '_')

    # Crée les chemins
    input_path = DATA_PATH / f'{input_filename}.csv'
    output_path = DATA_PATH / f'{output_filename}'

    if not input_path.exists() or force_download:
        import kaggle
        logging.info(f"Téléchargement du jeu de données '{kaggle_dataset}'.")
        load_dotenv_kaggle(kaggle_username, kaggle_key)

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            kaggle_dataset,
            path=DATA_PATH,
            unzip=True
        )

    # Charge le dataset
    dataframe = polars.read_csv(input_path, has_header=False)
    dataframe = preprocess_dataframe(dataframe, max_length=max_length)
    dataframe.write_parquet(output_path)


@cli.command()
def train(
        filename: str = typer.Option('preprocess.parquet', help="Nom du fichier de données d'entraînement."),

        # Paramètres du d'entraînement
        num_epochs: int = typer.Option(15, help="Nombre d'époques."),
        lr: float = typer.Option(1e-4, help="Taux d'apprentissage."),

        start_factor: float = typer.Option(1.0, help="Facteur de départ pour le scheduler."),
        end_factor: float = typer.Option(0.25, help="Facteur de fin pour le scheduler."),
        total_iters: int = typer.Option(10, help="Nombre total d'itérations."),

        # Paramètres du dataset
        batch_size: int = typer.Option(256, help="Taille du batch."),
        shuffle_dataset: bool = typer.Option(True, help="Mélange le dataset."),
        random_seed: int = typer.Option(42, help="Graine aléatoire pour le mélange."),
        ratio_tests: float = typer.Option(0.2, help="Ratio du dataset pour les tests."),
        ratio_validation: float = typer.Option(0.2, help="Ratio du dataset pour la validation."),

        # Autres paramètres
        tolerance: float = typer.Option(1e-2, help="Tolérance pour l'accuracy de différence entre le label et la prédiction (en %)."),
        device: str = typer.Option('cuda', help="Device utilisé pour l'entraînement. (cpu, cuda)"),
):
    """
    Entraîne le modèle.

    Args:
        filename (str): Nom du fichier de données d'entraînement.

        num_epochs (int): Nombre d'époques.
        lr (float): Taux d'apprentissage.
        start_factor (float): Facteur de départ pour le scheduler (% du lr).
        end_factor (float): Facteur de fin pour le scheduler (% du lr).
        total_iters (int): Nombre total d'itérations pour le scheduler.

        batch_size (int): Taille du batch.
        shuffle_dataset (bool): Mélange le dataset.
        random_seed (int): Graine aléatoire pour le mélange.
        ratio_tests (float): Ratio du dataset pour les tests.
        ratio_validation (float): Ratio du dataset pour la validation.

        tolerance (float): Tolérance pour l'accuracy de différence entre le label et la prédiction (en %).
        device (Literal['cpu', 'cuda']): Device utilisé pour l'entraînement.
    """
    # Todo : Implémenter la fonction d'entraînement du modèle.
    logging.info("Entraînement du modèle.")

    path = DATA_PATH / f'{filename}'

    if not path.exists():
        raise FileNotFoundError(f"Le dataframe n'existe pas : '{path}'.")

    # Crée le modèle
    model = ChessModel(input_size=837).cpu()
    summary(model, input_size=(837,), device='cpu', batch_size=1)

    # Charge le dataset
    dataframe = polars.read_parquet(path)
    logging.info(f"Le dataset contiens '{dataframe.__len__()}' lignes.")
    dataset = ChessDataset(preprocess_df=dataframe)

    # Sépare le dataset en : train, tests, validation
    list_loaders = split_dataset(
        dataset=dataset,
        batch_size=batch_size,
        shuffle_dataset=shuffle_dataset,
        random_seed=random_seed,
        ratio_tests=ratio_tests,
        ratio_validation=ratio_validation
    )
    train_loader, tests_loader, validation_loader = list_loaders

    # Entraîne le modèle
    train_model(
        model=model,
        train_loader=train_loader,
        tests_loader=tests_loader,
        validation_loader=validation_loader,

        num_epochs=num_epochs,
        device=device,

        lr=lr,
        tolerance=tolerance,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters
    )


def main():
    """
    Fonction principale du programme.

    Returns:
        int: Code de retour du programme.
    """

    # Todo : Besoin d'un try, except, finally pour gérer les erreurs et les logs ?
    cli()
