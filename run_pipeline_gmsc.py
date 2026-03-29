"""
run_pipeline_gmsc.py
--------------------
Orquestrador do pipeline usando o dataset real
"Give Me Some Credit" (Kaggle).

Diferenças em relação ao run_pipeline.py (dataset sintético):
  - Usa data_loader_gmsc para carregar e normalizar o CSV real
  - Usa feature_engineering_gmsc com features específicas do GMSC
  - Salva o Parquet processado em data/processed/credit_clean.parquet
  - Schema de features diferente (sem credit_score, loan_amount, etc.)

Uso:
  python run_pipeline_gmsc.py
  python run_pipeline_gmsc.py --raw data/raw/cs-training.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("pipeline_gmsc")

SEED      = 42
TEST_SIZE = 0.20


def _step(msg: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  {msg}")
    print(f"{'─'*62}")


def run(raw_path: str = "data/raw/give_me_some_credit.csv") -> None:
    start = time.time()
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # ── 1. CARREGAMENTO E NORMALIZAÇÃO ───────────────────────────────────────
    _step("ETAPA 1/8 — Carregamento do GMSC e normalização de colunas")

    from src.data_loader_gmsc import load_and_save_gmsc, gmsc_summary

    df = load_and_save_gmsc(
        raw_path  = raw_path,
        save_path = "data/processed/credit_clean.parquet",
    )
    gmsc_summary(df)

    # ── 2. FEATURE ENGINEERING ──────────────────────────────────────────────
    _step("ETAPA 2/8 — Feature Engineering (específico GMSC)")

    from src.feature_engineering_gmsc import (
        build_features_gmsc, scale_features, get_feature_names
    )

    df_fe    = build_features_gmsc(df)
    TARGET   = "default"
    FEATURES = [c for c in df_fe.columns if c != TARGET]

    X = df_fe[FEATURES]
    y = df_fe[TARGET]

    logger.info("Features totais: %d | Target rate: %.2f%%",
                len(FEATURES), y.mean() * 100)

    # ── 3. SPLIT ESTRATIFICADO ───────────────────────────────────────────────
    _step("ETAPA 3/8 — Split treino/teste (estratificado, sem data leakage)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    assert abs(y_train.mean() - y_test.mean()) < 0.01, "Estratificação falhou"
    logger.info("Treino=%d (default=%.2f%%) | Teste=%d (default=%.2f%%)",
                len(X_train), y_train.mean()*100,
                len(X_test),  y_test.mean()*100)

    # ── 4. NORMALIZAÇÃO ──────────────────────────────────────────────────────
    _step("ETAPA 4/8 — Normalização (StandardScaler fit-only no treino)")

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    joblib.dump(scaler, "models/scaler_gmsc.pkl")

    splits = {
        "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
        "y_train"   : y_train,    "y_test"   : y_test,
        "features"  : FEATURES,
    }
    joblib.dump(splits, "models/splits_gmsc.pkl")

    # ── 5. TREINAMENTO ───────────────────────────────────────────────────────
    _step("ETAPA 5/8 — Treinamento dos 3 modelos")

    from src.train_baseline import train_baseline
    from src.train_models   import train_random_forest, train_xgboost

    lr_model, _  = train_baseline(X_train_sc, y_train,
                                   save_path="models/lr_gmsc.pkl")
    rf_model     = train_random_forest(X_train_sc, y_train,
                                        save_path="models/rf_gmsc.pkl")
    xgb_model    = train_xgboost(X_train_sc, y_train, X_test_sc, y_test,
                                  save_path="models/xgb_gmsc.pkl")

    models = {
        "Logistic Regression": lr_model,
        "Random Forest"      : rf_model,
        "XGBoost"            : xgb_model,
    }

    # ── 6. AVALIAÇÃO ─────────────────────────────────────────────────────────
    _step("ETAPA 6/8 — Avaliação e comparação")

    from src.evaluate import (
        evaluate_model, plot_roc_curves, plot_confusion_matrix,
        plot_feature_importance, compare_models,
    )

    results = []
    for name, model in models.items():
        m = evaluate_model(model, X_test_sc, y_test, name=name)
        results.append(m)
        plot_confusion_matrix(model, X_test_sc, y_test, name=name)
        if name in ("Random Forest", "XGBoost"):
            plot_feature_importance(model, FEATURES, name=name)

    plot_roc_curves(models, X_test_sc, y_test)
    df_comp = compare_models(results)

    # ── 7. SIMULAÇÃO DE NEGÓCIO ──────────────────────────────────────────────
    _step("ETAPA 7/8 — Simulação de decisão de crédito")

    from src.simulate_decision import BusinessConfig, simulate_business, optimize_threshold, plot_profit_curve

    best_name  = df_comp["auc"].idxmax()
    best_model = models[best_name]
    joblib.dump(best_model, "models/best_model_gmsc.pkl")

    cfg    = BusinessConfig(threshold=0.70)
    result = simulate_business(best_model, X_test_sc, y_test, cfg)
    print(result)

    best_t, thresholds, profits = optimize_threshold(best_model, X_test_sc, y_test, cfg)
    plot_profit_curve(thresholds, profits, best_t, best_name)

    # ── 8. RESUMO ────────────────────────────────────────────────────────────
    _step("ETAPA 8/8 — Resumo final")

    elapsed = time.time() - start
    print(f"\n{'═'*62}")
    print(f"  ✅  Pipeline GMSC concluído em {elapsed:.1f}s")
    print(f"  📊  Relatórios  : reports/")
    print(f"  🤖  Modelos     : models/*_gmsc.pkl")
    print(f"  💾  Dados       : data/processed/credit_clean.parquet")
    print(f"  🏆  Melhor      : {best_name} (AUC={df_comp.loc[best_name,'auc']:.4f})")
    print(f"  🎯  Threshold ótimo: {best_t:.3f}")
    print(f"{'═'*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline GMSC — Credit Risk ML")
    parser.add_argument(
        "--raw", type=str,
        default="data/raw/give_me_some_credit.csv",
        help="Caminho para o CSV original do GMSC"
    )
    args = parser.parse_args()
    run(raw_path=args.raw)
