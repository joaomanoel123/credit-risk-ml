"""
train_baseline.py
-----------------
Treinamento do modelo baseline: Regressão Logística.

Por que Regressão Logística como baseline?
  - Rápida de treinar e interpretar
  - Coeficientes têm significado de negócio (odds ratio)
  - Probabilidades bem calibradas por padrão
  - Referência mínima: qualquer modelo mais complexo deve superá-la

Saída:
  - Modelo salvo em models/logistic_regression.pkl
  - Métricas impressas no console
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_features, scale_features

logger = logging.getLogger(__name__)

MODEL_PATH  = Path("models/logistic_regression.pkl")
SCALER_PATH = Path("models/scaler_baseline.pkl")

# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------
LR_PARAMS = dict(
    C             = 1.0,       # regularização inversa; menor C = mais regularizado
    max_iter      = 1_000,     # garantir convergência
    class_weight  = "balanced",# corrige desbalanceamento de classes
    solver        = "lbfgs",
    random_state  = 42,
    n_jobs        = -1,
)


# ---------------------------------------------------------------------------
# train_baseline
# ---------------------------------------------------------------------------
def train_baseline(
    X_train,
    y_train,
    save: bool = True,
) -> tuple[LogisticRegression, dict]:
    """
    Treina a Regressão Logística como modelo baseline.

    Args:
        X_train : Features de treino (já escalonadas).
        y_train : Target de treino.
        save    : Se True, persiste o modelo em disco.

    Returns:
        (modelo_treinado, params_usados)
    """
    logger.info("Treinando Logistic Regression baseline...")

    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        logger.info("Modelo salvo em: %s", MODEL_PATH)

    logger.info("✅  Baseline treinado.")
    return model, LR_PARAMS


def load_baseline() -> LogisticRegression:
    """Carrega o modelo baseline salvo em disco."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df     = clean_data(load_data("data/synthetic/credit_data.csv"))
    X_tr, X_te, y_tr, y_te = split_data(df)
    X_tr_fe = build_features(X_tr)
    X_te_fe = build_features(X_te)
    X_tr_s, X_te_s, scaler = scale_features(X_tr_fe, X_te_fe)
    joblib.dump(scaler, SCALER_PATH)

    model, params = train_baseline(X_tr_s, y_tr)

    # Avaliação rápida
    from sklearn.metrics import roc_auc_score, classification_report
    y_prob = model.predict_proba(X_te_s)[:, 1]
    y_pred = model.predict(X_te_s)
    auc    = roc_auc_score(y_te, y_prob)

    print("\n── Baseline: Logistic Regression ──────────────────")
    print(f"   ROC-AUC : {auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Pago", "Default"]))
