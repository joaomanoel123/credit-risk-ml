"""
data_loader_gmsc.py
-------------------
Carregador e adaptador para o dataset real
"Give Me Some Credit" (Kaggle — cs-training.csv / give_me_some_credit.csv).

O dataset original tem colunas com nomes e escalas diferentes do schema
interno do pipeline. Este módulo faz o mapeamento, limpeza pesada específica
do dataset e salva o resultado como Parquet em data/processed/.

Colunas do dataset original → schema interno do pipeline:
───────────────────────────────────────────────────────────────────────────────
  SeriousDlqin2yrs                        → default         (TARGET)
  RevolvingUtilizationOfUnsecuredLines    → revolving_util
  age                                     → age
  NumberOfTime30-59DaysPastDueNotWorse    → late_30_59
  DebtRatio                               → debt_ratio
  MonthlyIncome                           → income
  NumberOfOpenCreditLinesAndLoans         → num_open_accounts
  NumberOfTimes90DaysLate                 → late_90
  NumberRealEstateLoansOrLines            → num_real_estate
  NumberOfTime60-89DaysPastDueNotWorse    → late_60_89
  NumberOfDependents                      → num_dependents
───────────────────────────────────────────────────────────────────────────────

Problemas conhecidos do GMSC tratados aqui:
  - MonthlyIncome : ~20% nulos → imputação por mediana
  - NumberOfDependents: ~2.5% nulos → imputação por 0
  - age == 0      : entradas inválidas → removidas
  - RevolvingUtil > 1 : valores > 1 são possíveis (utilização > 100%)
                         → clipado em 1.5 (outliers extremos)
  - DebtRatio > 1  : possível (dívidas > renda) → clipado em 3.0
  - late_* > 90   : erros de digitação conhecidos → clipado em 20

Funções públicas:
  load_gmsc(path)           → pd.DataFrame (schema interno)
  load_and_save_gmsc(path)  → pd.DataFrame + salva Parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy  as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Mapeamento de colunas original → schema interno ───────────────────────────
COLUMN_MAP = {
    "SeriousDlqin2yrs"                    : "default",
    "RevolvingUtilizationOfUnsecuredLines" : "revolving_util",
    "age"                                  : "age",
    "NumberOfTime30-59DaysPastDueNotWorse" : "late_30_59",
    "DebtRatio"                            : "debt_ratio",
    "MonthlyIncome"                        : "income",
    "NumberOfOpenCreditLinesAndLoans"      : "num_open_accounts",
    "NumberOfTimes90DaysLate"              : "late_90",
    "NumberRealEstateLoansOrLines"         : "num_real_estate",
    "NumberOfTime60-89DaysPastDueNotWorse" : "late_60_89",
    "NumberOfDependents"                   : "num_dependents",
}

# ── Schema esperado pelo restante do pipeline ─────────────────────────────────
PIPELINE_SCHEMA = [
    "age", "income", "num_open_accounts", "num_dependents",
    "revolving_util", "debt_ratio",
    "late_30_59", "late_60_89", "late_90",
    "num_real_estate",
    "default",
]


def load_gmsc(path: str | Path) -> pd.DataFrame:
    """
    Carrega e normaliza o dataset Give Me Some Credit.

    Etapas:
      1. Lê o CSV (aceita cs-training.csv ou give_me_some_credit.csv)
      2. Renomeia colunas para o schema interno
      3. Remove coluna de índice se presente (primeira coluna numérica sem nome)
      4. Trata nulos específicos do GMSC
      5. Remove registros inválidos (age == 0)
      6. Clipa outliers documentados do dataset
      7. Garante tipos corretos

    Args:
        path: Caminho para o CSV original.

    Returns:
        DataFrame no schema interno do pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    # ── Leitura ────────────────────────────────────────────────────────────────
    df = pd.read_csv(path)
    logger.info("GMSC carregado | shape=%s", df.shape)

    # Remove coluna de índice Kaggle (coluna sem nome ou chamada 'Unnamed: 0')
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)
        logger.debug("Coluna de índice removida: %s", unnamed)

    # ── Renomeia colunas ────────────────────────────────────────────────────────
    # Aceita tanto o mapeamento exato quanto variações de maiúsculas
    rename_map = {}
    for orig, novo in COLUMN_MAP.items():
        # Busca case-insensitive
        match = [c for c in df.columns if c.strip() == orig]
        if match:
            rename_map[match[0]] = novo
    df.rename(columns=rename_map, inplace=True)

    missing_cols = [c for c in PIPELINE_SCHEMA if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Colunas não encontradas após renomear: {missing_cols}\n"
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    df = df[PIPELINE_SCHEMA].copy()

    # ── Target ─────────────────────────────────────────────────────────────────
    df["default"] = df["default"].astype(int)
    df.dropna(subset=["default"], inplace=True)

    n_before = len(df)

    # ── Nulos específicos do GMSC ──────────────────────────────────────────────
    # income (~20% nulos) — mediana é mais robusta que média para renda
    income_median = df["income"].median()
    df["income"].fillna(income_median, inplace=True)
    logger.info("income nulos imputados com mediana=%.0f", income_median)

    # num_dependents (~2.5% nulos) — 0 é a moda, assume sem dependentes
    df["num_dependents"].fillna(0, inplace=True)

    # ── Registros inválidos ────────────────────────────────────────────────────
    df = df[df["age"] > 0].copy()
    logger.info("Registros com age==0 removidos: %d", n_before - len(df))

    # ── Clipping de outliers documentados ─────────────────────────────────────
    # revolving_util: 0–1 é normal; até 1.5 aceitável; acima disso é erro
    df["revolving_util"] = df["revolving_util"].clip(0, 1.5)

    # debt_ratio: até 3× é possível; acima disso é ruído
    df["debt_ratio"] = df["debt_ratio"].clip(0, 3.0)

    # late_*: valores > 90 são erros de digitação conhecidos no GMSC
    for col in ["late_30_59", "late_60_89", "late_90"]:
        df[col] = df[col].clip(0, 20)

    # income: remove extremos (salário > 500k/mês é outlier)
    df["income"] = df["income"].clip(0, 500_000)

    # age: remove extremos improváveis
    df["age"] = df["age"].clip(18, 100)

    # ── Tipos finais ───────────────────────────────────────────────────────────
    int_cols   = ["age", "num_open_accounts", "num_dependents",
                  "late_30_59", "late_60_89", "late_90", "num_real_estate"]
    float_cols = ["income", "revolving_util", "debt_ratio"]

    for col in int_cols:
        df[col] = df[col].astype(int)
    for col in float_cols:
        df[col] = df[col].astype(float).round(6)

    logger.info(
        "GMSC processado | shape=%s | default_rate=%.2f%%",
        df.shape, df["default"].mean() * 100
    )
    return df


def load_and_save_gmsc(
    raw_path  : str | Path = "data/raw/give_me_some_credit.csv",
    save_path : str | Path = "data/processed/credit_clean.parquet",
) -> pd.DataFrame:
    """
    Carrega o GMSC, normaliza e salva como Parquet comprimido.

    Parquet é preferível a CSV para dados processados:
      - 3–10× menor no disco (compressão snappy)
      - Preserva tipos de dados sem ambiguidade
      - Leitura 5–20× mais rápida que CSV

    Args:
        raw_path  : Caminho do CSV original.
        save_path : Destino do Parquet processado.

    Returns:
        DataFrame normalizado.
    """
    df = load_gmsc(raw_path)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False, compression="snappy")

    logger.info("Parquet salvo: %s (%.1f MB)", save_path,
                save_path.stat().st_size / 1e6)
    return df


def load_processed(path: str | Path = "data/processed/credit_clean.parquet") -> pd.DataFrame:
    """Carrega o Parquet processado. Preferir este após o primeiro load_and_save_gmsc()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {path}\n"
            "Execute load_and_save_gmsc() primeiro."
        )
    df = pd.read_parquet(path)
    logger.info("Parquet carregado | shape=%s", df.shape)
    return df


def gmsc_summary(df: pd.DataFrame) -> None:
    """Imprime um resumo estatístico do dataset GMSC normalizado."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Give Me Some Credit — Resumo do Dataset             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Registros        : {len(df):>10,}                              ║")
    print(f"║  Features         : {len(df.columns)-1:>10}                              ║")
    print(f"║  Taxa de default  : {df['default'].mean():>9.1%}                              ║")
    print(f"║  Ratio bom:ruim   : {(df['default']==0).sum()/(df['default']==1).sum():>9.1f}:1                          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Estatísticas principais:                                   ║")
    print(f"║  income  mediana  : {df['income'].median():>10,.0f}                             ║")
    print(f"║  age     mediana  : {df['age'].median():>10.0f}                             ║")
    print(f"║  revolv. util med : {df['revolving_util'].median():>10.3f}                             ║")
    print(f"║  debt_ratio med   : {df['debt_ratio'].median():>10.3f}                             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("║  ⚠️  Nulos restantes:                                        ║")
        for col, n in nulls.items():
            print(f"║    {col:<22}: {n:>6,}                              ║")
    else:
        print("║  ✅  Nenhum nulo restante                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    raw  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/give_me_some_credit.csv"
    save = sys.argv[2] if len(sys.argv) > 2 else "data/processed/credit_clean.parquet"

    df = load_and_save_gmsc(raw, save)
    gmsc_summary(df)
    print(f"\n✅  Pronto. Use load_processed('{save}') nos próximos passos.")
