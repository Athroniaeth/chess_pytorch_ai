import polars
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from src.tokenizer import fen_to_array

path = r"C:\Users\pierr\PycharmProjects\chess_pytorch_ai\data\fen_to_stockfish_evaluation.csv"


class FenScoreDataset(Dataset):
    """
    Dataset contenant les FEN et les scores d'évaluation de Stockfish.

    Args:
        dataframe (polars.DataFrame): DataFrame contenant les données.
        fen_column (str): Nom de la colonne contenant les FEN. Defaults to "column_1".
        score_column (str): Nom de la colonne contenant les scores. Defaults to "column_2".
        dtype (torch.dtype): Type de données des tenseurs. Defaults to torch.float32.
        max_length (int): Taille maximale du dataset

    Notes:
        - La colonne 'score' contient des espaces (erreur du créateur?), ils sont supprimés.
    """

    def __init__(
            self,
            dataframe: polars.DataFrame,
            fen_column: str = "column_1",
            score_column: str = "column_2",

            dtype: torch.dtype = torch.float32,
            max_length: int = 10_000,
            min_score: int = -20,
            max_score: int = 20
    ):
        # Supprime les lignes non désirées
        dataframe = dataframe.slice(0, max_length)

        # Convertit les FEN en tableau numpy
        expression = polars.col(fen_column).map_elements(fen_to_array)
        dataframe = dataframe.with_columns(expression)

        # Convertit le dtype string de 'score' en dtype float
        expression = polars.col(score_column).str.replace(" ", "").cast(polars.Float32)

        # Convertit le dtype string de 'score' en dtype float
        dataframe = dataframe.with_columns(expression)

        # Remplace les valeurs "aberrantes" pour la normalisation
        expression = polars.col(score_column).clip(min_score, max_score)
        dataframe = dataframe.with_columns(expression)

        # Renomme les colonnes en 'fen' et 'score'
        dataframe = dataframe.rename({fen_column: 'fen', score_column: 'score'})

        self.dataframe = dataframe
        self.dtype = dtype

    def __getitem__(self, index: int):
        """
        Récupère un élément du dataset.

        Args:
            index (int): Index de l'élément à récupérer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple contenant le tableau numpy de la FEN et le score d'évaluation.

        """
        # Récupère la FEN et le score
        array = self.dataframe['fen'][index]
        score = self.dataframe['score'][index]

        # Convertit le tableau numpy en tenseur PyTorch
        tensor_array = torch.tensor(array, dtype=self.dtype)
        tensor_score = torch.tensor(score, dtype=self.dtype)

        return tensor_array, tensor_score

    def __len__(self):
        return len(self.dataframe)

    def hist_score(self, bins: int = 51):
        """
        Affiche un histogramme de la colonne 'score'.

        Notes:
            Incrémente bins de 1 si celui-ci est pair, cela permet d'afficher le nombre de
            zeros dans la colonne 'score'. En impair, nous n'avons que pour > 0 ou < 0.

        Args:
            bins (int): Nombre de "bacs" pour l'histogramme. Defaults to 51.

        Returns:
            None
        """
        # Si le nombre de bacs est pair, on l'incrémente de 1
        bins = bins + 1 if bins % 2 == 0 else bins

        # Créer un histogramme de la colonne 'score'
        plt.hist(self.dataframe['score'], bins=bins)

        # Ajouter des étiquettes et un titre
        plt.xlabel('Valeur')
        plt.ylabel('Fréquence')
        plt.title('Distribution de la colonne "score"')

        # Afficher le graphique
        plt.show()


dataset = polars.read_csv(path, has_header=False)
dataset = FenScoreDataset(dataset)
dataset.hist_score()
