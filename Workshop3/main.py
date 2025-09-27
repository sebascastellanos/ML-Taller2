from src.flow.pipeline import run_pipeline
from src.config.settings import DATA_URL, COLUMNS, TARGET_NAME

if __name__ == "__main__":
    # por defecto lee de settings.py
    run_pipeline()