import itertools
from typing import Tuple, Generator

import chess
import numpy
import numpy as np

ProductPiece = Tuple[chess.PieceType, chess.Color]


def generator_pieces() -> Generator[ProductPiece, None, None]:
    return itertools.product(chess.PIECE_TYPES, chess.COLORS)


def fen_to_array(fen: str) -> np.ndarray:
    """
    Convertit un code FEN en tableau numpy.

    Notes:
        Les 768 premiers éléments correspondent aux pièces sur le plateau.
        (Booléen correspondant à la présence d'une pièce sur une case, 8x8 cases x 6 pièces x 2 couleurs)
            - 1 - 64: Pions blancs
            - 65 - 128: Pions noirs
            - 129 - 192: Cavaliers blancs
            - 193 - 256: Cavaliers noirs
            - 257 - 320: Fous blancs
            - 321 - 384: Fous noirs
            - 385 - 448: Tours blanches
            - 449 - 512: Tours noires
            - 513 - 576: Reines blanches
            - 577 - 640: Reines noires
            - 641 - 704: Rois blancs
            - 705 - 768: Rois noirs

        Le 769e élément correspond à la couleur du joueur.
        (Booléen pour la couleur du joueur, Blanc = 0, Noir = 1)

        Les 770-773e éléments correspondent aux droits de roque.
        (Booléen pour chaque couleur et chaque type de roque)
            - 770: Blanc, petit roque
            - 771: Blanc, grand roque
            - 772: Noir, petit roque
            - 773: Noir, grand roque

        Les 774-837e éléments correspondent à la case en passant.
        (Booléen pour la case en passant, 1 seul case est possible)
            - 1ère case: A8
            - 2e case: B8
            ...

            - 63e case: G1
            - 64e case: H1

    Args:
        fen (str): Code FEN à convertir.

    Returns:
        np.ndarray: Tableau numpy.
    """
    board = chess.Board(fen=fen)

    array_board = _get_board_array(board)
    array_turn = _get_turn_array(board)
    array_castling = _get_castling_array(board)
    array_en_passant = _get_passant_array(board)

    array = np.concatenate([
        array_board,
        array_turn,
        array_castling,
        array_en_passant
    ])

    return array


def _get_board_array(board: chess.Board) -> np.ndarray:
    board_array = numpy.array([], ndmin=2)

    for piece_type, color in generator_pieces():
        # Obtiens les pièces du plateau
        square_set = board.pieces(piece_type, color)

        # Crée un masque (numpy va inverser les axes, donc on utilise mirror)
        mask = square_set.mirror().tolist()

        # Convertit le masque en tableau numpy et en one-hot
        piece_array = numpy.array(mask).astype(int)

        # Ajoute le tableau de pièces au tableau de plateau
        board_array = numpy.append(board_array, piece_array)

    return board_array


def _get_turn_array(board: chess.Board) -> np.ndarray:
    return np.zeros(1) if board.turn == chess.WHITE else np.ones(1)


def _get_castling_array(board: chess.Board) -> np.ndarray:
    castling_array = numpy.array([0, 0, 0, 0], ndmin=1)

    castling_array[0] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    castling_array[1] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0

    castling_array[2] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    castling_array[3] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

    return castling_array


def _get_passant_array(board: chess.Board) -> np.ndarray:
    en_passant = board.ep_square  # None ou 0-63

    if en_passant is None:
        return np.zeros(64)

    return np.array([1 if square == en_passant else 0 for square in range(64)])
