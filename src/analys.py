# Todo : Implémenter la fonction d'entraînement du modèle.
import logging

import polars
import torch

from src import MODEL_PATH
from src.dataset import ChessDataset
from src.model import ChessModel
from src.tokenizer import fen_to_array

logging.info("Entraînement du modèle.")

# Récupère le nom du fichier
"""kaggle_dataset = 'antoinecastel/fen-to-stockfish-evaluation'
filename = kaggle_dataset.split('/')[1]
filename = filename.replace('-', '_')

dataset_path = DATA_PATH / f'{filename}.csv'

if not dataset_path.exists():
    raise FileNotFoundError(f"Le dataset n'existe pas. Veuillez le télécharger avec la commande 'download_dataset {kaggle_dataset}'. ('{dataset_path}")
"""
# Charge le dataset
score_column = 'score'
dataframe = polars.read_parquet('../data/preprocess_500k.parquet')

min_val = dataframe[score_column].min()
max_val = dataframe[score_column].max()

# min_val, max_val = -15312, 15319
print(f"Min: {min_val}  -  Max: {max_val}")

torch.cuda.empty_cache()
# Crée le modèle
path = MODEL_PATH / 'model_1.pth'
dict_model = torch.load(path, map_location='cpu')
model = ChessModel().cpu()
model.load_state_dict(dict_model)

dataset = ChessDataset(preprocess_df=dataframe)
dataset.hist_score()

print(dataset.denormalize_score(0.539366058323611605))
print(dataset.denormalize_score(0.5))

def analys_model(fen: str):
    fen_array = fen_to_array(fen)
    fen_tensor = torch.tensor(fen_array, dtype=torch.float32).view(1, -1)
    prediction = model(fen_tensor).item()

    # Dénormalise les prédictions (min-max scaler)
    score_stockfish = dataset.denormalize_score(prediction)
    print(f"Prédiction du modèle: {prediction:.6f}")
    print(f"Score Stockfish: {score_stockfish/100:.2f}")
    print()


list_fen = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Position de base
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  # Mat berger, Mat en 1 pour blanc
    "4k3/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",  # Noir n'ont qu'un roi
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/4K3 w kq - 0 1",  # Blanc n'ont qu'un roi
    "r3rbk1/p4ppp/8/2Nqp3/1p1P2b1/1P3N2/P2P1PPP/2RQR1K1 w - - 1 21"  # Position égale éloignée GMI (reputé légèrement favorable noir)
    ]

for fen in list_fen:
    analys_model(fen)