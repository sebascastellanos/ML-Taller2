from pathlib import Path
from typing import Optional, List
from prefect import flow

from ..config.settings import DATA_PATH, DATA_URL, COLUMNS, TARGET_NAME, PROJECT_ROOT
from ..tasks.dataprep   import (
    load_dataframe, split_features_target, binarize_target,
    evaluate_imputers_models, save_results, plot_results
)

@flow(name="taller2-ml-pipeline", log_prints=True)
def run_pipeline(
    data_path: Optional[Path] = DATA_PATH,
    url: Optional[str] = DATA_URL,
    columns: Optional[List[str]] = COLUMNS,
    target: str = TARGET_NAME,
    binarize: bool = True,
    results_csv: Path = PROJECT_ROOT / "preguntas" / "resultados_imputers_models.csv"
):
    df = load_dataframe(data_path=data_path, url=url, columns=columns)
    X, y = split_features_target(df, target)

    if binarize:
        y = binarize_target(y)  # 0 vs >0

    results = evaluate_imputers_models(X, y)
    save_results(results, results_csv)
    plot_results(results)