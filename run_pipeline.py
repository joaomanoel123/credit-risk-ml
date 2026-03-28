"""
run_pipeline.py
---------------
Orquestrador completo do pipeline de Credit Risk ML.

Executa sequencialmente:
  1. Geração do dataset sintético
  2. Pré-processamento e limpeza
  3. Feature engineering
  4. Treinamento dos 3 modelos
  5. Avaliação completa com gráficos
  6. Seleção do melhor modelo
  7. Simulação de decisão de negócio
  8. Otimização de threshold
  9. Persistência de artefatos (modelos + scaler)

Uso:
  python run_pipeline.py [--data caminho/para/dados.csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _step(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run(data_path: str | None = None) -> None:
    start = time.time()
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # ── 1. DADOS ────────────────────────────────────────────────────────────
    _step("ETAPA 1/8 — Preparação dos Dados")

    if data_path is None:
        synthetic_path = Path("data/synthetic/credit_data.csv")
        if not synthetic_path.exists():
            logger.info("Gerando dataset sintético...")
            from data.synthetic.generate_data import generate_credit_dataset, introduce_noise
            df_gen = introduce_noise(generate_credit_dataset())
            synthetic_path.parent.mkdir(parents=True, exist_ok=True)
            df_gen.to_csv(synthetic_path, index=False)
            logger.info("Dataset gerado: %s", synthetic_path)
        data_path = str(synthetic_path)

    from src.data_preprocessing import load_data, clean_data, split_data

    df_raw   = load_data(data_path)
    df_clean = clean_data(df_raw)
    X_tr, X_te, y_tr, y_te = split_data(df_clean)
    logger.info("Dados prontos | treino=%d | teste=%d", len(X_tr), len(X_te))

    # ── 2. FEATURE ENGINEERING ──────────────────────────────────────────────
    _step("ETAPA 2/8 — Feature Engineering")

    from src.feature_engineering import build_features, scale_features

    X_tr_fe = build_features(X_tr)
    X_te_fe = build_features(X_te)
    X_tr_s, X_te_s, scaler = scale_features(X_tr_fe, X_te_fe)

    joblib.dump(scaler, "models/scaler_final.pkl")
    logger.info("Features finais: %d colunas", X_tr_s.shape[1])

    # ── 3. TREINAMENTO ──────────────────────────────────────────────────────
    _step("ETAPA 3/8 — Treinamento dos Modelos")

    from src.train_baseline import train_baseline
    from src.train_models   import train_random_forest, train_xgboost

    lr_model, _  = train_baseline(X_tr_s, y_tr)
    rf_model     = train_random_forest(X_tr_s, y_tr)
    xgb_model    = train_xgboost(X_tr_s, y_tr, X_te_s, y_te)

    models = {
        "Logistic Regression": lr_model,
        "Random Forest":       rf_model,
        "XGBoost":             xgb_model,
    }

    # ── 4. AVALIAÇÃO ─────────────────────────────────────────────────────────
    _step("ETAPA 4/8 — Avaliação dos Modelos")

    from src.evaluate import (
        evaluate_model,
        plot_roc_curves,
        plot_confusion_matrix,
        plot_feature_importance,
        compare_models,
    )

    results = []
    feature_names = X_tr_s.columns.tolist()

    for name, model in models.items():
        metrics = evaluate_model(model, X_te_s, y_te, name=name)
        results.append(metrics)

        plot_confusion_matrix(model, X_te_s, y_te, name=name)

        if name in ("Random Forest", "XGBoost"):
            plot_feature_importance(model, feature_names, name=name)

    plot_roc_curves(models, X_te_s, y_te)
    df_comparison = compare_models(results)

    # ── 5. SELEÇÃO DO MELHOR MODELO ──────────────────────────────────────────
    _step("ETAPA 5/8 — Seleção do Melhor Modelo")

    best_name  = df_comparison["auc"].idxmax()
    best_model = models[best_name]
    joblib.dump(best_model, "models/best_model.pkl")
    logger.info("🏆  Melhor modelo: %s (AUC=%.4f)", best_name,
                df_comparison.loc[best_name, "auc"])

    # ── 6. SIMULAÇÃO DE NEGÓCIO ──────────────────────────────────────────────
    _step("ETAPA 6/8 — Simulação de Negócio (threshold=0.70)")

    from src.simulate_decision import BusinessConfig, simulate_business

    cfg    = BusinessConfig(threshold=0.70)
    result = simulate_business(best_model, X_te_s, y_te, cfg)
    print(result)

    # Comparação entre modelos na simulação
    print("\n  Lucro por modelo com threshold=0.70:")
    for name, model in models.items():
        r = simulate_business(model, X_te_s, y_te, cfg)
        print(f"    {name:<22} → {r.total_profit:>12,.2f} BRL | "
              f"aprovação={r.approval_rate:.1%}")

    # ── 7. OTIMIZAÇÃO DE THRESHOLD ───────────────────────────────────────────
    _step("ETAPA 7/8 — Otimização de Threshold")

    from src.simulate_decision import optimize_threshold, plot_profit_curve

    best_t, thresholds, profits = optimize_threshold(best_model, X_te_s, y_te, cfg)
    plot_profit_curve(thresholds, profits, best_t, best_name)
    print(f"\n  Threshold padrão  : 0.70  → lucro com best model")
    cfg_opt  = BusinessConfig(threshold=best_t)
    r_opt    = simulate_business(best_model, X_te_s, y_te, cfg_opt)
    print(f"  Threshold ótimo   : {best_t:.3f} → {r_opt.total_profit:,.2f} BRL")

    # ── 8. PERSISTÊNCIA FINAL ────────────────────────────────────────────────
    _step("ETAPA 8/8 — Persistência dos Artefatos")

    artifacts = {
        "models/logistic_regression.pkl": lr_model,
        "models/random_forest.pkl":       rf_model,
        "models/xgboost.pkl":             xgb_model,
        "models/best_model.pkl":          best_model,
        "models/scaler_final.pkl":        scaler,
    }

    for path, obj in artifacts.items():
        joblib.dump(obj, path)
        logger.info("Salvo: %s", path)

    # Log de métricas finais
    elapsed = time.time() - start
    print(f"\n{'═'*60}")
    print(f"  ✅  PIPELINE CONCLUÍDO em {elapsed:.1f}s")
    print(f"  📊  Relatórios  : reports/")
    print(f"  🤖  Modelos     : models/")
    print(f"  🚀  API         : uvicorn app.main:app --reload")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk ML Pipeline")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Caminho para CSV de dados (default: gera dataset sintético)"
    )
    args = parser.parse_args()
    run(data_path=args.data)
