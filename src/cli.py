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


def main():
    """
    Fonction principale du programme.

    Returns:
        int: Code de retour du programme.
    """

    # Todo : Besoin d'un try, except, finally pour gérer les erreurs et les logs ?
    cli()
