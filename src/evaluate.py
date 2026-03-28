"""
evaluate.py
-----------
Avaliação completa dos modelos treinados.

Métricas calculadas:
  - ROC-AUC
  - Accuracy / Precision / Recall / F1-score
  - Matriz de confusão
  - Curva ROC
  - Importância de features (para modelos baseados em árvores)
  - Comparação entre modelos (tabela e gráfico de barras)

Todas as visualizações são salvas em reports/.

Funções públicas:
  - evaluate_model(model, X_test, y_test, name) → dict
  - plot_roc_curves(models_dict, X_test, y_test)
  - plot_confusion_matrix(model, X_test, y_test, name)
  - plot_feature_importance(model, feature_names, name)
  - compare_models(results_list)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")           # sem display; salva em arquivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# Paleta visual consistente
PALETTE = {"Logistic Regression": "#4C72B0",
           "Random Forest":       "#55A868",
           "XGBoost":             "#C44E52"}


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model",
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Calcula e exibe as principais métricas de classificação.

    Args:
        model     : Modelo sklearn com predict_proba.
        X_test    : Features de teste.
        y_test    : Labels verdadeiros.
        name      : Nome amigável do modelo.
        threshold : Ponto de corte para classe positiva.
        verbose   : Imprime o relatório completo.

    Returns:
        Dicionário com todas as métricas.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "name":      name,
        "auc":       roc_auc_score(y_test, y_prob),
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "threshold": threshold,
    }

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  {name}")
        print(f"{'─'*55}")
        print(f"  ROC-AUC   : {metrics['auc']:.4f}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Pago','Default'])}")

    logger.info(
        "%s | AUC=%.4f F1=%.4f Recall=%.4f",
        name, metrics["auc"], metrics["f1"], metrics["recall"]
    )
    return metrics


# ---------------------------------------------------------------------------
# Curvas ROC
# ---------------------------------------------------------------------------
def plot_roc_curves(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    """
    Plota curvas ROC de múltiplos modelos no mesmo gráfico.

    Args:
        models_dict : {nome: modelo_treinado}
        X_test      : Features de teste.
        y_test      : Labels verdadeiros.
        save        : Salva em reports/roc_curves.png.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    for name, model in models_dict.items():
        y_prob        = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _   = roc_curve(y_test, y_prob)
        roc_auc       = auc(fpr, tpr)
        color         = PALETTE.get(name, None)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Curvas ROC — Comparação de Modelos", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = REPORTS / "roc_curves.png"
        fig.savefig(path, dpi=150)
        logger.info("ROC salvo em: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Matriz de confusão
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model",
    threshold: float = 0.5,
    save: bool = True,
) -> None:
    """Plota e salva a matriz de confusão normalizada."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=["Pago", "Default"],
        yticklabels=["Pago", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predito",  fontsize=11)
    ax.set_ylabel("Real",     fontsize=11)
    ax.set_title(f"Matriz de Confusão — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        safe_name = name.lower().replace(" ", "_")
        path = REPORTS / f"confusion_matrix_{safe_name}.png"
        fig.savefig(path, dpi=150)
        logger.info("Matriz de confusão salva: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Importância de features
# ---------------------------------------------------------------------------
def plot_feature_importance(
    model,
    feature_names: list[str],
    name: str = "Model",
    top_n: int = 15,
    save: bool = True,
) -> None:
    """
    Plota as top-N features mais importantes do modelo.

    Compatível com Random Forest (feature_importances_) e
    XGBoost (feature_importances_).
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("%s não possui feature_importances_. Skipping.", name)
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = [feature_names[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    bars = ax.barh(top_names[::-1], top_vals[::-1],
                   color=PALETTE.get(name, "#4C72B0"), edgecolor="white")
    ax.set_xlabel("Importância (Gini)", fontsize=11)
    ax.set_title(f"Top {top_n} Features — {name}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        safe_name = name.lower().replace(" ", "_")
        path = REPORTS / f"feature_importance_{safe_name}.png"
        fig.savefig(path, dpi=150)
        logger.info("Feature importance salvo: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparação entre modelos
# ---------------------------------------------------------------------------
def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Exibe e salva uma tabela comparativa dos resultados de todos os modelos.

    Args:
        results : Lista de dicionários retornados por evaluate_model().

    Returns:
        DataFrame comparativo.
    """
    df = pd.DataFrame(results).set_index("name")
    df = df[["auc", "accuracy", "precision", "recall", "f1"]].round(4)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          COMPARAÇÃO DE MODELOS                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(df.to_string())

    # Gráfico de barras agrupadas
    metrics_to_plot = ["auc", "precision", "recall", "f1"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, row) in enumerate(df.iterrows()):
        vals  = [row[m] for m in metrics_to_plot]
        color = PALETTE.get(model_name, f"C{i}")
        ax.bar(x + i * width, vals, width, label=model_name, color=color, alpha=0.9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Comparação de Métricas por Modelo", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = REPORTS / "model_comparison.png"
    fig.savefig(path, dpi=150)
    logger.info("Comparação salva: %s", path)
    plt.close(fig)

    # Salva CSV
    csv_path = REPORTS / "model_metrics.csv"
    df.to_csv(csv_path)
    logger.info("Métricas CSV salvo: %s", csv_path)

    best = df["auc"].idxmax()
    print(f"\n🏆  Melhor modelo (AUC): {best} ({df.loc[best,'auc']:.4f})")
    return df
