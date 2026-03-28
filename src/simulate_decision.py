"""
simulate_decision.py
--------------------
Camada de negócio: decisão de crédito e simulação financeira.

Lógica de decisão:
  score > threshold → APROVAR
  score ≤ threshold → REJEITAR

Simulação de lucro (por cliente):
  ┌─────────────────────────────────────────────────────┐
  │  Decisão      │  Cliente Real  │  Resultado         │
  ├─────────────────────────────────────────────────────┤
  │  APROVAR      │  Bom (0)       │  +PROFIT_GOOD      │
  │  APROVAR      │  Ruim (1)      │  -LOSS_BAD         │
  │  REJEITAR     │  Bom (0)       │  -OPPORTUNITY_COST │
  │  REJEITAR     │  Ruim (1)      │  0 (evitou perda)  │
  └─────────────────────────────────────────────────────┘

O custo de oportunidade de rejeitar um cliente bom é muitas vezes
ignorado em simulações simplistas — aqui ele é parametrizável.

Funções públicas:
  - make_decision(score, threshold)        → str ("APROVAR" | "REJEITAR")
  - simulate_business(model, X, y, cfg)    → SimulationResult
  - optimize_threshold(model, X, y, cfg)   → (best_threshold, profit_curve)
  - plot_profit_curve(thresholds, profits)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuração de negócio
# ---------------------------------------------------------------------------
@dataclass
class BusinessConfig:
    """Parâmetros financeiros configuráveis da simulação."""
    threshold:        float = 0.70   # ponto de corte de aprovação
    profit_good:      float = 100.0  # lucro por cliente bom aprovado
    loss_bad:         float = 500.0  # prejuízo por cliente ruim aprovado
    opportunity_cost: float = 10.0   # custo de rejeitar cliente bom
    currency:         str   = "BRL"

    def summary(self) -> str:
        return (
            f"  Threshold       : {self.threshold}\n"
            f"  Lucro (bom)     : +{self.profit_good} {self.currency}\n"
            f"  Prejuízo (ruim) : -{self.loss_bad} {self.currency}\n"
            f"  Oport. perdida  : -{self.opportunity_cost} {self.currency}\n"
        )


# ---------------------------------------------------------------------------
# Resultado da simulação
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    total_customers:   int
    approved:          int
    rejected:          int
    true_positives:    int   # ruim corretamente rejeitado
    false_positives:   int   # bom incorretamente rejeitado
    true_negatives:    int   # bom corretamente aprovado
    false_negatives:   int   # ruim incorretamente aprovado
    total_profit:      float
    profit_per_client: float
    approval_rate:     float
    config:            BusinessConfig = field(repr=False)

    def __str__(self) -> str:
        sign = "+" if self.total_profit >= 0 else ""
        return (
            f"\n{'═'*50}\n"
            f"  SIMULAÇÃO DE DECISÃO DE CRÉDITO\n"
            f"{'═'*50}\n"
            f"  Clientes avaliados  : {self.total_customers:,}\n"
            f"  Aprovados           : {self.approved:,} ({self.approval_rate:.1%})\n"
            f"  Rejeitados          : {self.rejected:,}\n"
            f"\n  Resultado por quadrante:\n"
            f"    ✅ Bom aprovado (TP)       : {self.true_negatives:,}  → +{self.config.profit_good:.0f} cada\n"
            f"    ❌ Ruim aprovado (FN)      : {self.false_negatives:,} → -{self.config.loss_bad:.0f} cada\n"
            f"    ⚠️  Bom rejeitado (FP)      : {self.false_positives:,} → -{self.config.opportunity_cost:.0f} cada\n"
            f"    🛡️  Ruim rejeitado (TN)     : {self.true_positives:,}  → 0\n"
            f"\n  Lucro Total         : {sign}{self.total_profit:,.2f} {self.config.currency}\n"
            f"  Lucro por cliente   : {sign}{self.profit_per_client:,.2f} {self.config.currency}\n"
            f"{'═'*50}"
        )


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------
def make_decision(score: float, threshold: float = 0.70) -> str:
    """
    Converte um score de probabilidade em decisão de crédito.

    Args:
        score     : Probabilidade de default (0–1).
        threshold : Ponto de corte (acima = REJEITAR).

    Returns:
        "APROVAR" ou "REJEITAR"
    """
    return "REJEITAR" if score > threshold else "APROVAR"


def simulate_business(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: BusinessConfig | None = None,
) -> SimulationResult:
    """
    Simula o impacto financeiro do modelo em um conjunto de clientes.

    Args:
        model   : Modelo treinado com predict_proba.
        X_test  : Features dos clientes.
        y_test  : Inadimplência real (1 = default, 0 = bom).
        config  : Configuração financeira (BusinessConfig).

    Returns:
        SimulationResult com métricas e lucro total.
    """
    cfg = config or BusinessConfig()
    logger.info("Simulando decisão | threshold=%.2f", cfg.threshold)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = np.array(y_test)

    approved_mask = y_prob <= cfg.threshold   # baixo score de default = aprovado
    rejected_mask = ~approved_mask

    # Quadrantes
    tn = int(np.sum(approved_mask & (y_true == 0)))   # bom aprovado  ✅
    fn = int(np.sum(approved_mask & (y_true == 1)))   # ruim aprovado ❌
    fp = int(np.sum(rejected_mask & (y_true == 0)))   # bom rejeitado ⚠️
    tp = int(np.sum(rejected_mask & (y_true == 1)))   # ruim rejeitado 🛡️

    # Cálculo do lucro
    total_profit = (
        tn * cfg.profit_good
        - fn * cfg.loss_bad
        - fp * cfg.opportunity_cost
    )

    n = len(y_true)
    result = SimulationResult(
        total_customers   = n,
        approved          = int(approved_mask.sum()),
        rejected          = int(rejected_mask.sum()),
        true_positives    = tp,
        false_positives   = fp,
        true_negatives    = tn,
        false_negatives   = fn,
        total_profit      = total_profit,
        profit_per_client = total_profit / n,
        approval_rate     = approved_mask.mean(),
        config            = cfg,
    )

    logger.info("Lucro total=%.2f | aprovação=%.1f%%",
                total_profit, result.approval_rate * 100)
    return result


def optimize_threshold(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: BusinessConfig | None = None,
    steps: int = 100,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Busca o threshold que maximiza o lucro total.

    Args:
        model   : Modelo treinado.
        X_test  : Features de teste.
        y_test  : Labels reais.
        config  : BusinessConfig (sem o threshold; será varrido).
        steps   : Número de thresholds testados entre 0.01 e 0.99.

    Returns:
        (best_threshold, thresholds_array, profits_array)
    """
    cfg = config or BusinessConfig()
    thresholds = np.linspace(0.01, 0.99, steps)
    profits    = []

    for t in thresholds:
        cfg_t  = BusinessConfig(
            threshold=t,
            profit_good=cfg.profit_good,
            loss_bad=cfg.loss_bad,
            opportunity_cost=cfg.opportunity_cost,
        )
        res = simulate_business(model, X_test, y_test, cfg_t)
        profits.append(res.total_profit)

    profits        = np.array(profits)
    best_idx       = np.argmax(profits)
    best_threshold = thresholds[best_idx]

    logger.info(
        "Threshold ótimo: %.3f → Lucro máximo: %.2f",
        best_threshold, profits[best_idx]
    )
    return best_threshold, thresholds, profits


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------
def plot_profit_curve(
    thresholds: np.ndarray,
    profits: np.ndarray,
    best_threshold: float | None = None,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """Plota a curva de lucro em função do threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, profits / 1_000, lw=2.5, color="#4C72B0", label="Lucro total")
    ax.axhline(0, color="gray", lw=1, linestyle="--")

    if best_threshold is not None:
        best_idx    = np.argmin(np.abs(thresholds - best_threshold))
        best_profit = profits[best_idx]
        ax.axvline(best_threshold, color="#C44E52", lw=1.8, linestyle="--",
                   label=f"Threshold ótimo = {best_threshold:.2f}")
        ax.scatter([best_threshold], [best_profit / 1_000],
                   color="#C44E52", zorder=5, s=80)
        ax.annotate(
            f"  {best_profit/1_000:,.1f}k",
            (best_threshold, best_profit / 1_000),
            fontsize=10, color="#C44E52",
        )

    ax.set_xlabel("Threshold de Aprovação", fontsize=12)
    ax.set_ylabel("Lucro Total (x1000 BRL)", fontsize=12)
    ax.set_title(f"Curva de Lucro × Threshold — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = REPORTS / f"profit_curve_{model_name.lower().replace(' ','_')}.png"
        fig.savefig(path, dpi=150)
        logger.info("Curva de lucro salva: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sys.path.insert(0, ".")

    from src.data_preprocessing import load_data, clean_data, split_data
    from src.feature_engineering import build_features, scale_features
    from src.train_baseline import train_baseline

    df = clean_data(load_data("data/synthetic/credit_data.csv"))
    X_tr, X_te, y_tr, y_te = split_data(df)
    X_tr_fe = build_features(X_tr)
    X_te_fe = build_features(X_te)
    X_tr_s, X_te_s, _ = scale_features(X_tr_fe, X_te_fe)
    model, _ = train_baseline(X_tr_s, y_tr, save=False)

    cfg    = BusinessConfig(threshold=0.70)
    result = simulate_business(model, X_te_s, y_te, cfg)
    print(result)

    best_t, thresholds, profits = optimize_threshold(model, X_te_s, y_te, cfg)
    plot_profit_curve(thresholds, profits, best_t, "Logistic Regression")
    print(f"\n🎯  Threshold ótimo encontrado: {best_t:.3f}")
