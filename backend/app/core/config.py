import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Data Files
LIDAR_FILE = DATA_DIR / "MNS(terrain+batiment)_2015_1m_Mercier–Hochelaga-Maisonneuve.tif"
CADASTRE_FILE = DATA_DIR / "quebec_lots.gpkg"
