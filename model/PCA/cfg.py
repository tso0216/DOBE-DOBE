import os
import sys

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import PATCHES  # noqa: E402,F401

VERSION = "PCA"
LATENT_DIM = 2
SEED = 0
