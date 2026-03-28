"""
train_models.py
---------------
Treinamento dos modelos avançados: Random Forest e XGBoost.

Random Forest:
  - Ensemble de árvores de decisão com bagging
  - Robusto a outliers e não requer normalização
  - Boa interpretabilidade via feature importance

XGBoost:
  - Gradient boosting otimizado para velocidade e performance
  - Regularização L1/L2 nativa (evita overfitting)
  - Padrão da indústria em competições de tabular data

Ambos usam class_weight / scale_pos_weight para lidar com
desbalanceamento de classes (clientes que dão default são minoria).

Saídas:
  - models/random_forest.pkl
  - models/xgboost.pkl
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators  = 300,
    max_depth     = 12,
    min_samples_split = 20,
    min_samples_leaf  = 10,
    max_features  = "sqrt",
    class_weight  = "balanced",
    random_state  = 42,
    n_jobs        = -1,
)

XGB_PARAMS = dict(
    n_estimators        = 400,
    max_depth           = 6,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    reg_alpha           = 0.1,      # L1
    reg_lambda          = 1.0,      # L2
    eval_metric         = "auc",
    use_label_encoder   = False,
    random_state        = 42,
    n_jobs              = -1,
    # scale_pos_weight calculado dinamicamente em train_xgboost()
)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
    save: bool = True,
) -> RandomForestClassifier:
    """
    Treina um Random Forest com os hiperparâmetros configurados.

    Args:
        X_train : Features de treino (pode ser não escalonado; RF é invariante).
        y_train : Target de treino.
        params  : Dicionário opcional para sobrescrever RF_PARAMS.
        save    : Persiste o modelo em models/random_forest.pkl.

    Returns:
        Modelo treinado.
    """
    p = {**RF_PARAMS, **(params or {})}
    logger.info("Treinando Random Forest | params=%s", p)

    model = RandomForestClassifier(**p)
    model.fit(X_train, y_train)

    if save:
        path = Path("models/random_forest.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("✅  Random Forest salvo em: %s", path)

    return model


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
    save: bool = True,
) -> XGBClassifier:
    """
    Treina um XGBoost com early stopping opcional.

    O scale_pos_weight é calculado automaticamente a partir do ratio de
    classes, dando mais peso aos exemplos de default (classe minoritária).

    Args:
        X_train : Features de treino.
        y_train : Target de treino.
        X_val   : Features de validação para early stopping (opcional).
        y_val   : Target de validação (opcional).
        params  : Dicionário opcional para sobrescrever XGB_PARAMS.
        save    : Persiste o modelo em models/xgboost.pkl.

    Returns:
        Modelo treinado.
    """
    # Calcula peso para balancear classes
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    p = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight, **(params or {})}
    logger.info(
        "Treinando XGBoost | scale_pos_weight=%.2f | params=%s",
        scale_pos_weight, {k: v for k, v in p.items() if k != "scale_pos_weight"}
    )

    # Remove parâmetro legado se presente
    p.pop("use_label_encoder", None)

    model = XGBClassifier(**p)

    fit_kwargs: dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"]         = [(X_val, y_val)]
        fit_kwargs["early_stopping_rounds"] = 30
        fit_kwargs["verbose"]          = False

    model.fit(X_train, y_train, **fit_kwargs)

    if save:
        path = Path("models/xgboost.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("✅  XGBoost salvo em: %s", path)

    return model


# ---------------------------------------------------------------------------
# Carregadores
# ---------------------------------------------------------------------------
def load_random_forest() -> RandomForestClassifier:
    return joblib.load("models/random_forest.pkl")

def load_xgboost() -> XGBClassifier:
    return joblib.load("models/xgboost.pkl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_preprocessing import load_data, clean_data, split_data
    from src.feature_engineering import build_features, scale_features
    from sklearn.metrics import roc_auc_score

    df = clean_data(load_data("data/synthetic/credit_data.csv"))
    X_tr, X_te, y_tr, y_te = split_data(df)
    X_tr_fe = build_features(X_tr)
    X_te_fe = build_features(X_te)
    X_tr_s, X_te_s, _ = scale_features(X_tr_fe, X_te_fe)

    rf  = train_random_forest(X_tr_s, y_tr)
    xgb = train_xgboost(X_tr_s, y_tr, X_te_s, y_te)

    for name, m in [("Random Forest", rf), ("XGBoost", xgb)]:
        auc = roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1])
        print(f"   {name:<18} ROC-AUC: {auc:.4f}")
