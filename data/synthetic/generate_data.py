"""
generate_data.py
----------------
Gerador de dados sintéticos para crédito financeiro.
Simula características reais de clientes com padrões de default.

Variáveis geradas:
  - age               : idade do cliente
  - income            : renda mensal
  - loan_amount       : valor do empréstimo solicitado
  - loan_tenure       : prazo do empréstimo (meses)
  - credit_score      : score de crédito (300–850)
  - num_open_accounts : número de contas abertas
  - num_credit_inq    : consultas de crédito nos últimos 6 meses
  - debt_to_income    : razão dívida/renda
  - employment_years  : tempo de emprego (anos)
  - has_mortgage      : possui hipoteca (0/1)
  - has_dependents    : possui dependentes (0/1)
  - default           : inadimplência (TARGET — 1 = default, 0 = pagou)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 50_000


def generate_credit_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age               = rng.integers(21, 70, size=n)
    income            = rng.lognormal(mean=8.5, sigma=0.6, size=n).round(2)
    loan_amount       = (income * rng.uniform(0.5, 5.0, size=n)).round(2)
    loan_tenure       = rng.choice([12, 24, 36, 48, 60, 72], size=n)
    credit_score      = np.clip(rng.normal(650, 100, size=n), 300, 850).round().astype(int)
    num_open_accounts = rng.integers(1, 15, size=n)
    num_credit_inq    = rng.integers(0, 10, size=n)
    employment_years  = np.clip(rng.exponential(scale=5, size=n), 0, 40).round(1)
    has_mortgage      = rng.integers(0, 2, size=n)
    has_dependents    = rng.integers(0, 2, size=n)

    # DTI: dívida sobre renda (quanto maior, mais arriscado)
    monthly_payment   = loan_amount / loan_tenure
    debt_to_income    = np.clip((monthly_payment / (income / 12)).round(4), 0.01, 2.0)

    # -------------------------------------------------------
    # Modelagem probabilística de default
    # Fatores de risco baseados em literatura de crédito:
    #   credit_score baixo, DTI alto, muitas consultas → maior risco
    # -------------------------------------------------------
    log_odds = (
        -2.5
        + (-0.006) * credit_score          # score alto reduz risco
        + ( 1.5)   * debt_to_income        # DTI alto aumenta risco
        + ( 0.12)  * num_credit_inq        # consultas indicam pressão financeira
        + (-0.04)  * employment_years      # emprego estável reduz risco
        + (-0.003) * (income / 1_000)      # renda alta reduz risco
        + ( 0.3)   * (loan_amount / loan_amount.mean())  # empréstimo relativo
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default      = rng.binomial(1, prob_default).astype(int)

    df = pd.DataFrame({
        "age":               age,
        "income":            income,
        "loan_amount":       loan_amount,
        "loan_tenure":       loan_tenure,
        "credit_score":      credit_score,
        "num_open_accounts": num_open_accounts,
        "num_credit_inq":    num_credit_inq,
        "debt_to_income":    debt_to_income,
        "employment_years":  employment_years,
        "has_mortgage":      has_mortgage,
        "has_dependents":    has_dependents,
        "default":           default,
    })

    return df


def introduce_noise(df: pd.DataFrame, missing_rate: float = 0.03, seed: int = SEED) -> pd.DataFrame:
    """Insere valores nulos aleatórios para simular dados reais imperfeitos."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    noisy_cols = ["income", "employment_years", "num_credit_inq", "credit_score"]
    for col in noisy_cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "credit_data.csv"

    print("⚙️  Gerando dataset sintético...")
    df = generate_credit_dataset(N_SAMPLES)
    df = introduce_noise(df)

    df.to_csv(output_path, index=False)
    print(f"✅  Dataset salvo em: {output_path}")
    print(f"   Shape  : {df.shape}")
    print(f"   Default: {df['default'].mean():.1%} dos clientes")
    print(df.describe().round(2))
