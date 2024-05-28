""" Entry point of the program. """
import sys
from pathlib import Path

path = Path(__file__).absolute().parents[1]
sys.path.append(f'{path}')

from src.cli import main  # noqa: PEP8 E402

if __name__ == "__main__":
    main()
