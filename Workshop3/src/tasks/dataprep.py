from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from prefect import task, get_run_logger

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


@task(name="load_dataframe", log_prints=True)
def load_dataframe(
    data_path: Optional[Path] = None,
    url: Optional[str] = None,
    columns: Optional[List[str]] = None,
    na_values: str | list[str] = "?"
) -> pd.DataFrame:
    logger = get_run_logger()
    if data_path is None and url is None:
        raise ValueError("Debes pasar data_path o url.")

    if url:
        logger.info(f"Leyendo CSV desde URL: {url}")
        df = pd.read_csv(url, names=columns, na_values=na_values) if columns else pd.read_csv(url, na_values=na_values)
    else:
        logger.info(f"Leyendo CSV local: {data_path}")
        df = pd.read_csv(data_path, names=columns, na_values=na_values) if columns else pd.read_csv(data_path, na_values=na_values)

    logger.info(f"Shape: {df.shape}")
    logger.info(df.head(3).to_string())
    return df


@task(name="split_features_target")
def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"No existe la columna target '{target_col}' en {list(df.columns)}")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


@task(name="binarize_target", log_prints=True)
def binarize_target(y: pd.Series, positive_if_greater_than_zero: bool = True) -> pd.Series:
    """
    UCI Cleveland: target en {0,1,2,3,4}. Suele usarse binario: 0 = sano, >0 = enfermo.
    """
    if positive_if_greater_than_zero:
        y_bin = (y.astype(float) > 0).astype(int)
    else:
        y_bin = y.astype(int)
    print("Distribución binaria (0=sano,1=enfermo):")
    print(y_bin.value_counts(dropna=False).to_string())
    return y_bin


@task(name="evaluate_imputers_models", log_prints=True)
def evaluate_imputers_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Evalúa imputadores con RandomForest y XGBoost usando CV=5 y scoring='accuracy'.
    """
    imputers = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "most_frequent": SimpleImputer(strategy="most_frequent"),
        "knn": KNNImputer(n_neighbors=5),
    }

    results = []

    for imp_name, imputer in imputers.items():
        X_imp = imputer.fit_transform(X)

        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf_cv = cross_val_score(rf, X_imp, y, cv=5, scoring="accuracy")
        rf_acc_mean = float(np.mean(rf_cv))
        rf_acc_std = float(np.std(rf_cv))

        # XGBoost
        xgb = XGBClassifier(eval_metric="logloss", random_state=42)
        xgb_cv = cross_val_score(xgb, X_imp, y, cv=5, scoring="accuracy")
        xgb_acc_mean = float(np.mean(xgb_cv))
        xgb_acc_std = float(np.std(xgb_cv))

        results.append({
            "imputer": imp_name,
            "RF_acc_mean": rf_acc_mean,
            "RF_acc_std": rf_acc_std,
            "XGB_acc_mean": xgb_acc_mean,
            "XGB_acc_std": xgb_acc_std,
        })

        # Impresiones rápidas (tu formato)
        print(f"Modelo: RandomForest ({imp_name}) | Accuracy: {rf_acc_mean:.3f} | F1-macro: N/A")
        print(f"Modelo: XGBoost ({imp_name})     | Accuracy: {xgb_acc_mean:.3f} | F1-macro: N/A")

    results_df = pd.DataFrame(results)
    print("\nResumen por imputador:")
    print(results_df.to_string(index=False))
    return results_df


@task(name="save_results", log_prints=True)
def save_results(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Resultados guardados en: {out_path}")
