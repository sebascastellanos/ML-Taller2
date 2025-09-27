# heart_disease_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


# -------------------
# 1️⃣ Fase Pre-Workshop
# -------------------
def prepare_data():
    """Carga y prepara el dataset: X e y listos pero sin entrenar"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(url, names=columns, na_values='?')

    # Convertir target a binario: 0 = sano, 1 = enfermo
    df["target"] = (df["target"] > 0).astype(int)

    # Imputación inicial simple (media), como en el pre-workshop
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(df.drop("target", axis=1))
    y = df["target"].values

    print("✅ Datos preparados (fase pre-workshop)")
    print("Shape de X:", X.shape)
    print("Shape de y:", y.shape)

    return df, X, y


# -------------------
# 2️⃣ Fase Workshop
# -------------------
def evaluate_models(df: pd.DataFrame):
    """Evalúa estrategias de imputación y compara RandomForest vs XGBoost"""

    X_raw = df.drop("target", axis=1)
    y = df["target"]

    imputers = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "most_frequent": SimpleImputer(strategy="most_frequent"),
        "knn": KNNImputer(n_neighbors=5),
    }

    results = []

    for imp_name, imputer in imputers.items():
        print(f"\nUsando imputación: {imp_name}")

        X_imp = imputer.fit_transform(X_raw)

        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf_score = np.mean(cross_val_score(rf, X_imp, y, cv=5))
        print(f"  Random Forest Accuracy: {rf_score:.4f}")

        # XGBoost
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        xgb_score = np.mean(cross_val_score(xgb, X_imp, y, cv=5))
        print(f"  XGBoost Accuracy: {xgb_score:.4f}")

        results.append({
            "imputer": imp_name,
            "RandomForest": rf_score,
            "XGBoost": xgb_score,
        })

    results_df = pd.DataFrame(results)
    return results_df


def plot_results(results_df: pd.DataFrame):
    """Genera un gráfico comparando imputaciones y modelos"""
    results_df_melted = results_df.melt(
        id_vars="imputer", var_name="Model", value_name="Accuracy"
    )
    sns.barplot(data=results_df_melted, x="imputer", y="Accuracy", hue="Model")
    plt.title("Comparación de imputaciones y modelos (Workshop)")
    plt.show()


# -------------------
# 🚀 Main
# -------------------
if __name__ == "__main__":
    # Pre-workshop
    df, X, y = prepare_data()

    # Workshop
    results_df = evaluate_models(df)
    print("\nResultados finales:\n", results_df)

    plot_results(results_df)
