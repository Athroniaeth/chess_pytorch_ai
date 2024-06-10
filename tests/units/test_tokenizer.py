import pytest

from src.tokenizer import fen_to_array


def test_fen_to_array():
    result = fen_to_array("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    assert sum(result) == 36.0  # 36 pièces sur l'échiquier
    assert result.shape == (837,)  # 64 cases * 6 pièces * 2 couleurs + 1 couleur + 4 roques + 64 cases en passant


@pytest.mark.parametrize(("slice_start", "slice_end", "excepted"), [
    (0, 64, 8.0),  # 8 pions noirs
    (64, 128, 8.0),  # 8 pions blancs
    (128, 192, 2.0),  # 2 cavaliers noirs
    (192, 256, 2.0),  # 2 cavaliers blancs
    (256, 320, 2.0),  # 2 fous noirs
    (320, 384, 2.0),  # 2 fous blancs
    (384, 448, 2.0),  # 2 tours noires
    (448, 512, 2.0),  # 2 tours blanches
    (512, 576, 1.0),  # 1 reine noire
    (576, 640, 1.0),  # 1 reine blanche
    (640, 704, 1.0),  # 1 roi noir
    (704, 768, 1.0),  # 1 roi blanc
])
def test_location_fen_to_array(slice_start: int, slice_end: int, excepted: float):
    result = fen_to_array("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    test_location = result[slice_start:slice_end]
    assert sum(test_location) == excepted
