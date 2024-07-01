import logging

import numpy as np
import polars
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from src.tokenizer import fen_to_array
from src.utils import logging_time

path = r"C:\Users\pierr\PycharmProjects\chess_pytorch_ai\data\fen_to_stockfish_evaluation.csv"


@logging_time('preprocess_dataframe')
def preprocess_dataframe(
        dataframe: polars.DataFrame,
        fen_column: str = "column_1",
        score_column: str = "column_2",
        max_length: int = 100_000,
):
    """
    Pré-traite le DataFrame pour le transformer en un format utilisable par le modèle.

    Args:
        dataframe (polars.DataFrame): DataFrame à pré-traiter.
        fen_column (str): Nom de la colonne contenant les FEN. Defaults to "column_1".
        score_column (str): Nom de la colonne contenant les scores. Defaults to "column_2".
        max_length (int): Taille maximale du DataFrame. Defaults to 100_000.

    Notes:
        - La colonne 'score' contient des espaces (erreur du créateur?), ils sont supprimés.
    """
    # Supprime les lignes non désirées
    dataframe = dataframe.slice(0, max_length)

    # Convertit les FEN en tableau numpy / # MapWithoutReturnDtypeWarning
    expression = polars.col(fen_column).map_elements(fen_to_array, return_dtype=polars.List(polars.UInt8))
    dataframe = dataframe.with_columns(expression)

    # Le dataset contient '#+0' ou '#-0' dans score quand il y'a un mat, on remplace par le maximum de la couleur
    expression = polars.when(
        polars.col(score_column).str.contains('#+')).then(polars.lit('+15319')).when(
        polars.col(score_column).str.contains('#-')).then(polars.lit('-15312')).otherwise(polars.col(score_column)).alias(score_column)

    dataframe = dataframe.with_columns(expression)

    # Convertit le dtype string de 'score' en dtype float
    expression = polars.col(score_column).cast(polars.Int16)
    dataframe = dataframe.with_columns(expression)

    # Renomme les colonnes en 'fen' et 'score'
    dataframe = dataframe.rename({fen_column: 'fen', score_column: 'score'})

    return dataframe


class ChessDataset(Dataset):
    """
    Dataset contenant les FEN et les scores d'évaluation de Stockfish.

    Args:
        preprocess_df (polars.DataFrame): DataFrame contenant les données.
    """

    def __init__(self, preprocess_df: polars.DataFrame):
        """self.min_val = preprocess_df['score'].min()
        self.max_val = preprocess_df['score'].max()

        expression = (polars.col('score') - self.min_val) / (self.max_val - self.min_val)
        preprocess_df = preprocess_df.with_columns(expression)"""

        self.dataframe = preprocess_df

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
        tensor_array = torch.tensor(array, dtype=torch.float32)
        tensor_score = torch.tensor(score, dtype=torch.float32)

        return tensor_array, tensor_score

    def __len__(self):
        return len(self.dataframe)

    def normalize_score(self, score: float) -> float:
        return (score - self.min_val) / (self.max_val - self.min_val)

    def denormalize_score(self, score: float) -> float:
        return score * (self.max_val - self.min_val) + self.min_val

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


def split_dataset(
        dataset: Dataset,

        batch_size: int = 256,
        shuffle_dataset: bool = True,
        random_seed: int = 42,

        ratio_tests: float = 0.2,
        ratio_validation: float = 0.2
):
    """
    Sépare le dataset en trois parties : entraînement, tests et validation.

    Args:
        dataset (Dataset): Dataset PyTorch à splitter.
        batch_size (int): Taille des batchs. Defaults to 32.
        shuffle_dataset (bool): Mélange le dataset. Defaults to True.
        random_seed (int): Graine aléatoire pour le mélange. Defaults to 42.
        ratio_tests (float): Ratio du dataset pour les tests. Defaults to 0.1.
        ratio_validation (float): Ratio du dataset pour la validation. Defaults to 0.1.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Tuple contenant les DataLoaders pour l'entraînement, les tests et la validation.
    """
    ratio = round(1 - ratio_tests - ratio_validation, 2)
    logging.info(f"Splitting dataset, {ratio=}, {ratio_tests=}, {ratio_validation=}")

    # Prépare une liste d'indices du dataset
    dataset_size = len(dataset)  # noqa
    indices = [i for i in range(dataset_size)]

    # Calcule les slices pour chaque partie du dataset
    split_tests = dataset_size // (1 / ratio_tests)
    split_validation = dataset_size // (1 / ratio_validation)
    split_tests, split_validation = int(split_tests), int(split_validation)

    slice_train = slice(split_tests + split_validation, None)
    slice_tests = slice(split_tests, split_tests + split_validation)
    slice_validation = slice(None, split_tests)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Applique les slices aux indices
    train_indices = indices[slice_train]
    tests_indices = indices[slice_tests]
    validation_indices = indices[slice_validation]

    # Crée les samplers (sous-ensembles du dataset)
    train_sampler = SubsetRandomSampler(train_indices)
    tests_sampler = SubsetRandomSampler(tests_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    # Crée les DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    tests_loader = DataLoader(dataset, batch_size=batch_size, sampler=tests_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

    return train_loader, tests_loader, validation_loader
