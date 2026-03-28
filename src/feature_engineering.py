"""
feature_engineering.py
-----------------------
Criação de novas features e transformações de escala.

Design decision: todas as transformações são encapsuladas em um
sklearn Pipeline para garantir zero data leakage — o scaler é
ajustado apenas no X_train e aplicado ao X_test via transform().

Novas features derivadas:
  - loan_to_income        : empréstimo relativo à renda
  - monthly_payment_ratio : parcela mensal / renda mensal
  - credit_utilization    : estimativa de utilização do crédito
  - risk_score            : combinação ponderada de indicadores de risco
  - age_income_ratio      : idade * renda (proxy de estabilidade)

Funções públicas:
  - build_features(X)         → pd.DataFrame (features enriquecidas)
  - scale_features(X_tr, X_te) → (X_tr_scaled, X_te_scaled, scaler)
  - get_preprocessing_pipeline() → sklearn Pipeline
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criação de features derivadas
# ---------------------------------------------------------------------------

def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia de features: cria variáveis derivadas a partir das originais.

    Todas as operações são vetorizadas (pandas/numpy) e não dependem de
    informação externa ao DataFrame — seguras para aplicar em treino e
    teste independentemente.

    Args:
        X: DataFrame de features brutas.

    Returns:
        DataFrame com features originais + novas variáveis.
    """
    X = X.copy()

    # Razão empréstimo/renda
    X["loan_to_income"] = (
        X["loan_amount"] / X["income"].replace(0, np.nan)
    ).fillna(0).round(4)

    # Parcela mensal estimada / renda mensal
    # Evita divisão por zero em loan_tenure
    monthly_payment = X["loan_amount"] / X["loan_tenure"].replace(0, np.nan)
    monthly_income  = X["income"] / 12
    X["monthly_payment_ratio"] = (
        monthly_payment / monthly_income.replace(0, np.nan)
    ).fillna(0).clip(0, 5).round(4)

    # Proxy de utilização de crédito: consultas * contas abertas
    X["credit_utilization"] = (
        X["num_credit_inq"] * X["num_open_accounts"]
    ).astype(float)

    # Score de risco composto (normalizado 0–1 internamente para esta feature)
    # Peso baseado em correlação típica com default na literatura de crédito
    credit_norm = 1 - (X["credit_score"] - 300) / 550  # baixo score = alto risco
    dti_norm    = (X["debt_to_income"]).clip(0, 2) / 2
    inq_norm    = (X["num_credit_inq"]) / 10
    X["risk_score"] = (
        0.5 * credit_norm + 0.3 * dti_norm + 0.2 * inq_norm
    ).round(4)

    # Indicador de estabilidade: tempo de emprego * renda
    X["stability_index"] = (
        np.log1p(X["employment_years"]) * np.log1p(X["income"] / 1_000)
    ).round(4)

    # Flag: cliente de alto risco (score < 580 e DTI > 0.5)
    X["high_risk_flag"] = (
        (X["credit_score"] < 580) & (X["debt_to_income"] > 0.5)
    ).astype(int)

    new_cols = [
        "loan_to_income", "monthly_payment_ratio", "credit_utilization",
        "risk_score", "stability_index", "high_risk_flag",
    ]
    logger.info("Features criadas: %s", new_cols)
    return X


# ---------------------------------------------------------------------------
# Escalonamento
# ---------------------------------------------------------------------------

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normaliza as features com StandardScaler.

    O scaler é FIT apenas no X_train e TRANSFORM aplicado em ambos,
    eliminando qualquer data leakage de escala.

    Args:
        X_train : Features de treino.
        X_test  : Features de teste.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    cols   = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=cols,
        index=X_test.index,
    )
    logger.info("Escalonamento concluído | features=%d", len(cols))
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Pipeline sklearn (para uso em produção / API)
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transformador compatível com sklearn Pipeline que encapsula
    build_features + StandardScaler em um único objeto.

    Uso:
        from feature_engineering import FeatureEngineer
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ('fe',    FeatureEngineer()),
            ('model', LogisticRegression()),
        ])
        pipe.fit(X_train, y_train)
        pipe.predict_proba(X_test)
    """

    from sklearn.base import BaseEstimator, TransformerMixin

    class _FeatureTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler_ = StandardScaler()

        def fit(self, X, y=None):
            X_fe = build_features(pd.DataFrame(X))
            self.scaler_.fit(X_fe)
            self.feature_names_out_ = X_fe.columns.tolist()
            return self

        def transform(self, X, y=None):
            X_fe = build_features(pd.DataFrame(X))
            return self.scaler_.transform(X_fe)

        def get_feature_names_out(self):
            return self.feature_names_out_

    @staticmethod
    def get_transformer():
        return FeatureEngineer._FeatureTransformer()


def get_preprocessing_pipeline(model) -> Pipeline:
    """
    Retorna uma Pipeline completa: FeatureEngineer → StandardScaler → modelo.

    Args:
        model: Estimador sklearn já instanciado.

    Returns:
        sklearn Pipeline pronto para .fit() / .predict_proba().
    """
    return Pipeline([
        ("feature_engineering", FeatureEngineer.get_transformer()),
        ("classifier",          model),
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sys.path.insert(0, ".")

    from src.data_preprocessing import load_data, clean_data, split_data

    df     = clean_data(load_data("data/synthetic/credit_data.csv"))
    X_tr, X_te, y_tr, y_te = split_data(df)

    X_tr_fe = build_features(X_tr)
    X_te_fe = build_features(X_te)
    X_tr_s, X_te_s, scaler = scale_features(X_tr_fe, X_te_fe)

    print("\n✅  Feature engineering concluído.")
    print(f"   Features finais: {X_tr_s.shape[1]}")
    print(X_tr_s.describe().round(3))
