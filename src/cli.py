import logging
from enum import StrEnum
from logging import getLevelName
from types import SimpleNamespace

import typer

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
def download_dataset(
        kaggle_username: str = typer.Option(None, envvar="KAGGLE_USERNAME", help="Nom d'utilisateur à l'API Kaggle."),
        kaggle_key: str = typer.Option(None, envvar="KAGGLE_KEY", help="Token d'accès à l'API Kaggle."),

        kaggle_dataset: str = typer.Argument('antoinecastel/fen-to-stockfish-evaluation',
                                             help="Nom du jeu de données Kaggle à télécharger."),
):
    """
    Télécharge le jeu de données depuis Kaggle.

    Args:
        kaggle_username (str): Nom d'utilisateur à l'API Kaggle.
        kaggle_key (str): Token d'accès à l'API Kaggle.
        kaggle_dataset (str): Nom du jeu de données Kaggle à télécharger.
    """
    load_dotenv_kaggle(kaggle_username, kaggle_key)

    import kaggle

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(
        kaggle_dataset,
        path=DATA_PATH,
        unzip=True
    )


@cli.command()
def train(
        kaggle_dataset: str = typer.Argument('antoinecastel/fen-to-stockfish-evaluation', help="Nom du jeu de données Kaggle à télécharger."),
):
    """
    Entraîne le modèle.

    Args:
        kaggle_dataset (str): Nom du jeu de données Kaggle à télécharger.
    """
    # Todo : Implémenter la fonction d'entraînement du modèle.
    logging.info("Entraînement du modèle.")

    # Récupère le nom du fichier
    filename = kaggle_dataset.split('/')[1]
    filename = filename.replace('-', '_')

    dataset_path = DATA_PATH / f'{filename}.csv'

    if not dataset_path.exists():
        raise FileNotFoundError(f"Le dataset n'existe pas. Veuillez le télécharger avec la commande 'download_dataset {kaggle_dataset}'. ('{dataset_path}")

    # Todo : Implémenter la fonction d'entraînement du modèle.


def main():
    """
    Fonction principale du programme.

    Returns:
        int: Code de retour du programme.
    """

    # Todo : Besoin d'un try, except, finally pour gérer les erreurs et les logs ?
    cli()
