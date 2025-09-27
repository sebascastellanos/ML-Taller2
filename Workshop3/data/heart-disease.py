"""
settings.py
Configuración central del proyecto.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PREGUNTAS_DIR = PROJECT_ROOT / "preguntas"

# === Dataset UCI Cleveland via URL ===
TARGET_NAME = "target"        # nombre de la etiqueta
DATA_PATH = None              # usamos URL, así que mantenlo en None
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# columnas porque el CSV no trae header
COLUMNS = [
    'age','sex','cp','trestbps','chol','fbs','restecg','thalach',
    'exang','oldpeak','slope','ca','thal','target'
]

RANDOM_STATE = 42
