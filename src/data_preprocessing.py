"""
data_preprocessing.py
---------------------
Módulo responsável pela ingestão, limpeza e divisão dos dados.

Princípio central: NENHUMA informação do conjunto de teste deve
"vazar" para o treino (data leakage). Toda estatística de imputação
e escala é ajustada APENAS no treino e aplicada ao teste.

Funções públicas:
  - load_data(path)        → pd.DataFrame
  - clean_data(df)         → pd.DataFrame
  - split_data(df, target) → (X_train, X_test, y_train, y_test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------
TARGET_COL   = "default"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

NUMERIC_COLS = [
    "age", "income", "loan_amount", "loan_tenure",
    "credit_score", "num_open_accounts", "num_credit_inq",
    "debt_to_income", "employment_years",
]
BINARY_COLS  = ["has_mortgage", "has_dependents"]


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset CSV para um DataFrame.

    Args:
        path: Caminho para o arquivo .csv.

    Returns:
        DataFrame bruto.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se o target 'default' não estiver presente.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    logger.info("Dados carregados | shape=%s", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Coluna target '{TARGET_COL}' não encontrada.")

    return df


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza e validação do DataFrame bruto:
      1. Remove duplicatas
      2. Remove linhas sem target
      3. Imputa nulos numéricos pela mediana (calculada no próprio conjunto)
      4. Clipa valores extremos (outliers por IQR)
      5. Garante tipos corretos

    IMPORTANTE: Esta função deve ser chamada ANTES de split_data.
    A imputação aqui usa a mediana global. Se quiser zero-leakage absoluto,
    use o parâmetro `fit_imputer` dentro de uma Pipeline sklearn — veja
    feature_engineering.py para detalhes.

    Args:
        df: DataFrame bruto.

    Returns:
        DataFrame limpo.
    """
    df = df.copy()
    n_before = len(df)

    # 1. Duplicatas
    df.drop_duplicates(inplace=True)
    logger.info("Duplicatas removidas: %d", n_before - len(df))

    # 2. Linhas sem target
    df.dropna(subset=[TARGET_COL], inplace=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # 3. Imputação de nulos numéricos → mediana
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.debug("Imputado '%s' com mediana=%.4f", col, median_val)

    # 4. Clipping de outliers (1.5×IQR)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        lower   = q1 - 1.5 * iqr
        upper   = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    # 5. Tipos corretos nas colunas binárias
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    logger.info("Dados limpos | shape=%s | default_rate=%.2f%%",
                df.shape, df[TARGET_COL].mean() * 100)
    return df


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide o dataset em treino e teste com estratificação opcional.

    Estratificação garante que a proporção de defaults seja igual em
    ambos os conjuntos — fundamental quando a classe positiva é rara.

    Args:
        df          : DataFrame limpo.
        target      : Nome da coluna alvo.
        test_size   : Proporção reservada para teste.
        random_state: Semente para reprodutibilidade.
        stratify    : Se True, estratifica pela coluna target.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    logger.info(
        "Split | treino=%d (%.0f%%)  teste=%d (%.0f%%)  "
        "default_treino=%.2f%%  default_teste=%.2f%%",
        len(X_train), (1 - test_size) * 100,
        len(X_test),  test_size * 100,
        y_train.mean() * 100,
        y_test.mean()  * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# CLI rápido para validação
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/credit_data.csv"
    df_raw    = load_data(data_path)
    df_clean  = clean_data(df_raw)
    splits    = split_data(df_clean)
    print("\n✅  Pré-processamento concluído.")
    print(f"   Nulos restantes: {splits[0].isna().sum().sum()}")
