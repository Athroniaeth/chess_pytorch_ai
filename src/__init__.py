from pathlib import Path

# Variables globales de projet
PROJECT_PATH = Path(__file__).parents[1].absolute()
SOURCE_PATH = PROJECT_PATH / 'src'
ENV_PATH = PROJECT_PATH / '.env'

# Variables globales du projet
DATA_PATH = PROJECT_PATH / 'data'
KAGGLE_CONFIG_PATH = Path.home() / '.kaggle' / 'kaggle.json'
