"""
app/main.py
-----------
API REST para scoring de risco de crédito em tempo real.

Endpoints:
  POST /predict      → Score de probabilidade de default + decisão
  GET  /health       → Status da API
  GET  /model/info   → Metadados do modelo em produção

Decisão de crédito baseada no threshold configurado em .env ou padrão 0.70.

Segurança:
  - Validação de entrada via Pydantic (tipos, ranges)
  - Sem shell=True, sem eval()
  - Rate limiting recomendado via middleware nginx/cloudflare em produção

Execução local:
  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import build_features

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
MODEL_PATH    = Path(os.getenv("MODEL_PATH",  "models/xgboost.pkl"))
SCALER_PATH   = Path(os.getenv("SCALER_PATH", "models/scaler_final.pkl"))
THRESHOLD     = float(os.getenv("DECISION_THRESHOLD", "0.70"))
LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("credit_risk_api")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title       = "Credit Risk ML API",
    description = "Sistema de pontuação de risco de crédito com ML",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

# ---------------------------------------------------------------------------
# Carregamento do modelo (startup)
# ---------------------------------------------------------------------------
_model  = None
_scaler = None

@app.on_event("startup")
async def load_model():
    global _model, _scaler
    try:
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
        logger.info("Modelo carregado: %s", MODEL_PATH)
    except FileNotFoundError:
        logger.error(
            "Modelo não encontrado em %s. "
            "Execute run_pipeline.py primeiro.", MODEL_PATH
        )


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------
class CreditRequest(BaseModel):
    """Dados de entrada para scoring de crédito."""

    age:               int   = Field(..., ge=18,     le=100,   description="Idade do cliente")
    income:            float = Field(..., ge=0,                description="Renda mensal (BRL)")
    loan_amount:       float = Field(..., ge=100,              description="Valor do empréstimo (BRL)")
    loan_tenure:       int   = Field(..., ge=1,      le=360,   description="Prazo do empréstimo (meses)")
    credit_score:      int   = Field(..., ge=300,    le=850,   description="Score de crédito (300–850)")
    num_open_accounts: int   = Field(..., ge=0,      le=50,    description="Número de contas abertas")
    num_credit_inq:    int   = Field(..., ge=0,      le=20,    description="Consultas de crédito (últimos 6 meses)")
    debt_to_income:    float = Field(..., ge=0.0,    le=5.0,   description="Razão dívida/renda")
    employment_years:  float = Field(..., ge=0.0,   le=50.0,  description="Tempo de emprego (anos)")
    has_mortgage:      int   = Field(..., ge=0,      le=1,     description="Possui hipoteca (0/1)")
    has_dependents:    int   = Field(..., ge=0,      le=1,     description="Possui dependentes (0/1)")

    @field_validator("income", "loan_amount")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Deve ser maior que zero.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "income": 8000.0,
                "loan_amount": 20000.0,
                "loan_tenure": 36,
                "credit_score": 680,
                "num_open_accounts": 3,
                "num_credit_inq": 1,
                "debt_to_income": 0.35,
                "employment_years": 5.0,
                "has_mortgage": 0,
                "has_dependents": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Resposta do endpoint /predict."""
    default_probability: float
    decision:            Literal["APROVAR", "REJEITAR"]
    threshold_used:      float
    risk_band:           Literal["BAIXO", "MÉDIO", "ALTO"]
    model_version:       str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float


class ModelInfoResponse(BaseModel):
    model_path:   str
    model_type:   str
    threshold:    float
    input_fields: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _risk_band(prob: float) -> str:
    if prob < 0.30:
        return "BAIXO"
    elif prob < 0.60:
        return "MÉDIO"
    return "ALTO"


def _preprocess(request: CreditRequest) -> pd.DataFrame:
    """Converte o request em DataFrame e aplica feature engineering."""
    data = pd.DataFrame([request.model_dump()])
    data = build_features(data)

    if _scaler is not None:
        # Garante mesma ordem de colunas do treino
        try:
            scaled = _scaler.transform(data)
            data   = pd.DataFrame(scaled, columns=data.columns)
        except Exception as exc:
            logger.warning("Scaler error: %s — usando features raw.", exc)

    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Infraestrutura"])
async def health_check():
    """Verifica se a API está online e o modelo carregado."""
    return HealthResponse(
        status       = "ok",
        model_loaded = _model is not None,
        threshold    = THRESHOLD,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Infraestrutura"])
async def model_info():
    """Retorna metadados sobre o modelo em produção."""
    if _model is None:
        raise HTTPException(503, "Modelo não carregado.")
    return ModelInfoResponse(
        model_path   = str(MODEL_PATH),
        model_type   = type(_model).__name__,
        threshold    = THRESHOLD,
        input_fields = list(CreditRequest.model_fields.keys()),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict(request: CreditRequest):
    """
    Recebe dados de um cliente e retorna:
    - Probabilidade de default (0–1)
    - Decisão de crédito (APROVAR / REJEITAR)
    - Faixa de risco (BAIXO / MÉDIO / ALTO)

    **Regra de decisão:**
    score_default > threshold → REJEITAR | score_default ≤ threshold → APROVAR
    """
    if _model is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Modelo não disponível. Tente novamente em instantes.",
        )

    try:
        X    = _preprocess(request)
        prob = float(_model.predict_proba(X)[0, 1])
    except Exception as exc:
        logger.exception("Erro na predição: %s", exc)
        raise HTTPException(status_code=500, detail=f"Erro de predição: {exc}")

    decision = "REJEITAR" if prob > THRESHOLD else "APROVAR"

    logger.info(
        "Predict | credit_score=%d | income=%.0f | prob=%.4f | decision=%s",
        request.credit_score, request.income, prob, decision,
    )

    return PredictionResponse(
        default_probability = round(prob, 6),
        decision            = decision,
        threshold_used      = THRESHOLD,
        risk_band           = _risk_band(prob),
        model_version       = type(_model).__name__,
    )


# ---------------------------------------------------------------------------
# Entrypoint local
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
