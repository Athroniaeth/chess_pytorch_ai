import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from src import ENV_PATH, KAGGLE_CONFIG_PATH


def load_dotenv_kaggle(
        kaggle_username: Optional[str] = None,
        kaggle_key: Optional[str] = None,
) -> None:
    """
    Load Kaggle API credentials from .env file.

    Args:
        kaggle_username (Optional[str], optional): Kaggle username. Defaults to None.
        kaggle_key (Optional[str], optional): Kaggle key. Defaults to None.

    Raises:
        ValueError: If Kaggle API credentials are not found in .env file.
    """
    logging.info("Loading Kaggle API credentials from .env file.")

    # Charge les variables d'environnement
    load_dotenv(dotenv_path=ENV_PATH)

    # Récupère les variables d'environnement
    kaggle_username = kaggle_username or os.getenv("KAGGLE_USERNAME")
    kaggle_key = kaggle_key or os.getenv("KAGGLE_KEY")

    # Vérifie si les variables d'environnement sont définies
    conditions = (
        kaggle_username is None,
        kaggle_key is None,
    )

    if any(conditions):
        raise ValueError("Kaggle API credentials not found in .env file.")

    # Crée le fichier de configuration Kaggle
    json_content = json.dumps({"username": kaggle_username, "key": kaggle_key})

    KAGGLE_CONFIG_PATH.touch()
    KAGGLE_CONFIG_PATH.write_text(json_content)
