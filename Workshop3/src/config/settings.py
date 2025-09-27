"""
settings.py
Configuración central del proyecto.
Ajusta rutas y parámetros globales acá.
"""
from pathlib import Path

# .../Proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PREGUNTAS_DIR = PROJECT_ROOT / "preguntas"

# Semillas/constantes
RANDOM_STATE = 42
