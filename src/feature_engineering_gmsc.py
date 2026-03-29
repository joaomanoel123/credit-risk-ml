"""
feature_engineering_gmsc.py
----------------------------
Feature engineering específico para o schema do Give Me Some Credit.

O GMSC tem variáveis completamente diferentes do dataset sintético —
em especial, não há loan_amount, credit_score nem employment_years.
O risco é capturado principalmente pelo histórico de atrasos e pela
utilização de crédito rotativo.

Novas features criadas:
  - total_late       : soma de todos os atrasos (30-59 + 60-89 + 90+)
  - late_severity    : atrasos graves (90+) como proporção do total
  - util_x_debt      : utilização rotativa × debt ratio (pressão dupla)
  - income_per_dep   : renda por dependente (capacidade de pagamento)
  - high_util_flag   : utilização > 0.75 (flag de alto risco)
  - multi_late_flag  : ocorrências de atraso em mais de um bucket
  - debt_income_ratio: debt_ratio como proxy de DTI (já existe no dataset,
                       mas re-escalado para 0–1 para comparabilidade)

Funções públicas:
  build_features_gmsc(df)                  → pd.DataFrame
  scale_features(X_train, X_test)          → (X_tr, X_te, scaler)
  get_feature_names()                      → list[str]
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy  as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features base do GMSC após normalização pelo data_loader
BASE_FEATURES = [
    "age", "income", "num_open_accounts", "num_dependents",
    "revolving_util", "debt_ratio",
    "late_30_59", "late_60_89", "late_90",
    "num_real_estate",
]

# Features derivadas criadas por este módulo
DERIVED_FEATURES = [
    "total_late", "late_severity", "util_x_debt",
    "income_per_dep", "high_util_flag", "multi_late_flag",
    "log_income",
]


def build_features_gmsc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas específicas para o Give Me Some Credit.

    Todas as operações são vetorizadas e não dependem de estatísticas
    externas — seguras para aplicar em treino, teste e produção.

    Args:
        df: DataFrame com as colunas base do GMSC.

    Returns:
        DataFrame com features originais + derivadas.
    """
    df = df.copy()

    # ── 1. Total de atrasos ────────────────────────────────────────────────────
    # Qualquer atraso é sinal de risco; a soma captura o volume histórico
    df["total_late"] = (
        df["late_30_59"] + df["late_60_89"] + df["late_90"]
    ).astype(float)

    # ── 2. Severidade dos atrasos ──────────────────────────────────────────────
    # Atrasos de 90+ dias são muito mais graves (impacto direto no score)
    # Proporção de atrasos graves sobre o total; 0 quando total_late == 0
    df["late_severity"] = np.where(
        df["total_late"] > 0,
        df["late_90"] / df["total_late"],
        0.0
    ).round(4)

    # ── 3. Pressão de crédito dupla ────────────────────────────────────────────
    # Alta utilização rotativa + alto debt ratio = situação crítica
    # O produto amplifica a sinalização quando ambos são altos
    df["util_x_debt"] = (
        df["revolving_util"].clip(0, 1.5) * df["debt_ratio"].clip(0, 3.0)
    ).round(6)

    # ── 4. Renda por dependente ────────────────────────────────────────────────
    # Mais dependentes reduzem a capacidade de pagamento
    # Usando num_dependents + 1 para evitar divisão por zero
    df["income_per_dep"] = (
        df["income"] / (df["num_dependents"] + 1)
    ).round(2)

    # ── 5. Flag de alta utilização ─────────────────────────────────────────────
    # Utilização > 75% do crédito rotativo é sinal de alerta no mercado
    df["high_util_flag"] = (df["revolving_util"] > 0.75).astype(int)

    # ── 6. Flag de múltiplos buckets de atraso ────────────────────────────────
    # Ter atrasos em mais de um bucket indica padrão de inadimplência
    late_buckets = (
        (df["late_30_59"] > 0).astype(int)
        + (df["late_60_89"] > 0).astype(int)
        + (df["late_90"] > 0).astype(int)
    )
    df["multi_late_flag"] = (late_buckets > 1).astype(int)

    # ── 7. Log da renda ────────────────────────────────────────────────────────
    # Renda tem distribuição log-normal — transformação log reduz skewness
    # e melhora a performance de modelos lineares
    df["log_income"] = np.log1p(df["income"]).round(4)

    new_cols = DERIVED_FEATURES
    logger.info("Features GMSC criadas: %s", new_cols)

    # Validação rápida
    for col in new_cols:
        n_null = df[col].isnull().sum()
        if n_null:
            logger.warning("Feature '%s' tem %d nulos — verifique os dados base", col, n_null)

    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test : pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normaliza features com StandardScaler.

    Fit APENAS no X_train → zero data leakage.

    Args:
        X_train: Features de treino.
        X_test : Features de teste.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    cols   = X_train.columns.tolist()

    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=cols, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=cols, index=X_test.index
    )
    logger.info("Scale concluído | features=%d", len(cols))
    return X_train_sc, X_test_sc, scaler


def get_feature_names() -> list[str]:
    """Retorna a lista completa de features após feature engineering."""
    return BASE_FEATURES + DERIVED_FEATURES


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sys.path.insert(0, ".")

    from src.data_loader_gmsc import load_processed

    df = load_processed()
    df_fe = build_features_gmsc(df)

    print("\n✅  Feature engineering GMSC concluído")
    print(f"   Features base    : {len(BASE_FEATURES)}")
    print(f"   Features novas   : {len(DERIVED_FEATURES)}")
    print(f"   Features totais  : {len(BASE_FEATURES) + len(DERIVED_FEATURES)}")
    print(f"   Shape final      : {df_fe.shape}")

    print("\n   Correlação das novas features com default:")
    for col in DERIVED_FEATURES:
        corr_v = df_fe[col].corr(df_fe["default"])
        bar    = "█" * int(abs(corr_v) * 40)
        sinal  = "+" if corr_v > 0 else "-"
        print(f"   {col:<22} {sinal}{bar:<35} {corr_v:+.4f}")
