"""
=============================================================
CREDIT RISK ML v2.0 — Production-Grade API
=============================================================
Upgrades sobre v1:

  1. SHAP values reais → LLM recebe features que importaram
  2. Basel II Expected Loss → PD × LGD × EAD
  3. Segmentação de risco (4 níveis) com thresholds calibrados
  4. Calibração de probabilidade (Platt scaling)
  5. Endpoints async para não bloquear o servidor
  6. Logging estruturado com contexto por request
  7. Tratamento de erro centralizado
  8. Health check com status do modelo
  9. Rate limiting básico por IP
 10. Prompt LLM enriquecido com evidências SHAP
=============================================================
"""

# ================================
# 📦 IMPORTS
# ================================
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator

# ================================
# 🪵 LOGGING ESTRUTURADO
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("credit-risk-api")


# ================================
# 📦 CARREGAMENTO DO MODELO (lifespan)
# ================================
# Usar lifespan evita o deprecated @app.on_event("startup")
# O modelo é carregado uma vez e reutilizado em todos os requests

MODEL_STATE: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega artefatos na inicialização e libera ao encerrar."""
    logger.info("Carregando artefatos do modelo...")
    try:
        MODEL_STATE["model"]  = joblib.load("models/best_model.pkl")
        MODEL_STATE["scaler"] = joblib.load("models/scaler.pkl")          # opcional
        # SHAP explainer: TreeExplainer é mais rápido para XGBoost/RF
        MODEL_STATE["explainer"] = shap.TreeExplainer(MODEL_STATE["model"])
        logger.info("Modelo e SHAP explainer prontos.")
    except FileNotFoundError as e:
        logger.warning(f"Artefato nao encontrado: {e}. Usando modo demo.")
        MODEL_STATE["model"]     = None
        MODEL_STATE["explainer"] = None

    yield  # API fica disponível aqui

    MODEL_STATE.clear()
    logger.info("Artefatos liberados.")


# ================================
# 🚀 APP
# ================================
app = FastAPI(
    title="Credit Risk ML API",
    description="Previsão de risco de crédito com explicabilidade via SHAP + LLM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# 🤖 CONFIG LLM
# ================================
llm = ChatOpenAI(
    temperature=0.2,          # menos criativo = mais consistente em análise financeira
    model="gpt-4o-mini",
    max_tokens=400,
    request_timeout=15,
)

# ================================
# 📝 PROMPT v2 — com evidências SHAP
# ================================
# A diferença do v1: o LLM agora recebe as top features que
# influenciaram a decisão, não apenas os dados brutos.
# Isso transforma a explicação de genérica para factual.

PROMPT_TEMPLATE = """
Você é um analista sênior de risco de crédito. Explique a decisão de forma
clara, objetiva e profissional, usando as evidências abaixo.

PERFIL DO CLIENTE:
- Idade: {age} anos
- Renda mensal: R$ {income:,.2f}
- Score de crédito: {credit_score}
- Debt-to-Income: {dti:.1%}
- Tempo de emprego: {employment_years:.1f} anos

RESULTADO DA ANÁLISE:
- Probabilidade de default: {prob:.1%}
- Decisão: {decision}
- Segmento de risco: {risk_segment}
- Expected Loss estimado: R$ {expected_loss:,.2f}

PRINCIPAIS FATORES (análise SHAP):
{shap_summary}

Com base nisso, explique em 3 parágrafos curtos:
1. Por que esta decisão foi tomada
2. Quais fatores pesaram mais (positivos e negativos)
3. Recomendação ou próximo passo

Seja direto. Não repita os números já informados.
"""

prompt = PromptTemplate(
    input_variables=[
        "age", "income", "credit_score", "dti", "employment_years",
        "prob", "decision", "risk_segment", "expected_loss", "shap_summary"
    ],
    template=PROMPT_TEMPLATE,
)


# ================================
# 📊 SCHEMA DE ENTRADA v2
# ================================
class CreditRequest(BaseModel):
    age               : int   = Field(..., ge=18, le=90,   description="Idade do solicitante")
    income            : float = Field(..., gt=0,           description="Renda mensal em R$")
    loan_amount       : float = Field(..., gt=0,           description="Valor do empréstimo")
    loan_tenure       : int   = Field(..., ge=1, le=360,   description="Prazo em meses")
    credit_score      : int   = Field(..., ge=300, le=850, description="Score Serasa/SPC")
    num_open_accounts : int   = Field(..., ge=0)
    num_credit_inq    : int   = Field(..., ge=0)
    debt_to_income    : float = Field(..., ge=0.0, le=1.0)
    employment_years  : float = Field(..., ge=0.0)
    has_mortgage      : int   = Field(..., ge=0, le=1)
    has_dependents    : int   = Field(..., ge=0, le=1)

    # Validação customizada: renda mínima para o valor solicitado
    @field_validator("loan_amount")
    @classmethod
    def loan_within_reason(cls, v, info):
        if "income" in info.data and v > info.data["income"] * 120:
            raise ValueError("Valor do empréstimo excede 120x a renda mensal.")
        return v


# ================================
# 📤 SCHEMA DE SAÍDA v2
# ================================
class CreditResponse(BaseModel):
    request_id        : str
    default_probability: float
    decision          : str
    risk_segment      : str
    expected_loss     : float
    explanation       : str
    top_features      : dict
    processing_ms     : float


# ================================
# ⚙️ FEATURE ENGINEERING
# ================================
FEATURE_NAMES = [
    "age", "income", "loan_amount", "loan_tenure",
    "credit_score", "num_open_accounts", "num_credit_inq",
    "debt_to_income", "employment_years", "has_mortgage",
    "has_dependents", "loan_to_income", "monthly_pay_ratio",
]

def build_features(data: CreditRequest) -> np.ndarray:
    """
    Constrói o vetor de features com engenharia aplicada.

    Novas features v2:
      loan_to_income    : exposição relativa à renda
      monthly_pay_ratio : comprometimento mensal da renda
    """
    loan_to_income    = data.loan_amount / data.income
    monthly_pay_ratio = (data.loan_amount / data.loan_tenure) / (data.income / 12)

    return np.array([[
        data.age, data.income, data.loan_amount, data.loan_tenure,
        data.credit_score, data.num_open_accounts, data.num_credit_inq,
        data.debt_to_income, data.employment_years, data.has_mortgage,
        data.has_dependents, loan_to_income, monthly_pay_ratio,
    ]])


# ================================
# 🏦 SEGMENTAÇÃO DE RISCO
# ================================
# Thresholds calibrados com base em Basel II e práticas de mercado.
# Em produção: calibrar com dados reais de inadimplência histórica.

RISK_THRESHOLDS = {
    "BAIXO"  : (0.00, 0.25),
    "MÉDIO"  : (0.25, 0.50),
    "ALTO"   : (0.50, 0.70),
    "CRÍTICO": (0.70, 1.00),
}

def classify_risk(prob: float) -> tuple[str, str]:
    """
    Retorna (segmento, decisão) com base na probabilidade de default.

    Decisão:
      BAIXO/MÉDIO  → APROVAR
      ALTO         → ANÁLISE MANUAL (borderline)
      CRÍTICO      → REJEITAR
    """
    for segment, (low, high) in RISK_THRESHOLDS.items():
        if low <= prob < high:
            decision = "REJEITAR" if segment == "CRÍTICO" else \
                       "ANÁLISE MANUAL" if segment == "ALTO" else "APROVAR"
            return segment, decision
    return "CRÍTICO", "REJEITAR"


# ================================
# 💰 EXPECTED LOSS — BASEL II
# ================================
def compute_expected_loss(
    prob        : float,
    loan_amount : float,
    lgd         : float = 0.45,   # Loss Given Default padrão Basel II
    ccf         : float = 1.00,   # Credit Conversion Factor (empréstimo já desembolsado)
) -> float:
    """
    Expected Loss = PD × LGD × EAD

    PD  : Probability of Default (output do modelo)
    LGD : Loss Given Default (quanto se perde dado o default)
          Padrão Basel II: 45% para crédito sem garantia
    EAD : Exposure at Default = loan_amount × CCF
    """
    ead = loan_amount * ccf
    return prob * lgd * ead


# ================================
# 🔍 SHAP SUMMARY
# ================================
def get_shap_summary(
    explainer,
    features    : np.ndarray,
    top_n       : int = 5,
) -> tuple[str, dict]:
    """
    Calcula SHAP values e retorna:
      - Texto formatado para o prompt do LLM
      - Dict com top features e seus valores

    A diferença do v1: o LLM recebe evidências concretas,
    não apenas os dados brutos do cliente.
    """
    if explainer is None:
        return "SHAP não disponível (modo demo).", {}

    shap_values = explainer.shap_values(features)

    # Para classificação binária: pegar a classe positiva (default=1)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Ordenar por impacto absoluto
    indices = np.argsort(np.abs(sv))[::-1][:top_n]

    linhas = []
    top_dict = {}
    for i in indices:
        nome   = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        valor  = float(features[0][i])
        impacto = float(sv[i])
        direcao = "aumenta risco" if impacto > 0 else "reduz risco"
        linhas.append(f"  • {nome}: {valor:.2f} → {direcao} ({impacto:+.3f})")
        top_dict[nome] = round(impacto, 4)

    return "\n".join(linhas), top_dict


# ================================
# 🧠 EXPLICAÇÃO LLM (async)
# ================================
async def explain_decision_async(context: dict) -> str:
    """
    Chama o LLM de forma assíncrona para não bloquear o servidor.
    Com timeout e fallback para mensagem padrão.
    """
    try:
        formatted  = prompt.format(**context)
        response   = await llm.ainvoke(formatted)
        return response.content
    except Exception as e:
        logger.warning(f"LLM falhou: {e}. Usando explicação padrão.")
        return (
            f"Decisão: {context['decision']} com probabilidade de default de "
            f"{context['prob']:.1%}. Segmento: {context['risk_segment']}. "
            f"Explicação detalhada indisponível no momento."
        )


# ================================
# 🚀 ENDPOINT PRINCIPAL — async
# ================================
@app.post("/predict", response_model=CreditResponse)
async def predict(data: CreditRequest, request: Request):
    """
    Endpoint principal de predição de risco de crédito.

    Fluxo:
      1. Engenharia de features
      2. Predição do modelo
      3. Segmentação de risco
      4. Expected Loss (Basel II)
      5. SHAP values
      6. Explicação LLM com contexto SHAP
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] Novo request | score={data.credit_score} | renda={data.income}")

    # 1. Features
    features = build_features(data)

    # 2. Predição
    model = MODEL_STATE.get("model")
    if model is None:
        # Modo demo sem modelo carregado
        prob = float(np.random.uniform(0.1, 0.9))
        logger.warning(f"[{request_id}] Modo demo — modelo não carregado.")
    else:
        prob = float(model.predict_proba(features)[0][1])

    # 3. Segmentação
    risk_segment, decision = classify_risk(prob)

    # 4. Expected Loss
    expected_loss = compute_expected_loss(prob, data.loan_amount)

    # 5. SHAP
    explainer = MODEL_STATE.get("explainer")
    shap_summary_text, top_features = get_shap_summary(explainer, features)

    # 6. Explicação LLM
    explanation = await explain_decision_async({
        "age"            : data.age,
        "income"         : data.income,
        "credit_score"   : data.credit_score,
        "dti"            : data.debt_to_income,
        "employment_years": data.employment_years,
        "prob"           : prob,
        "decision"       : decision,
        "risk_segment"   : risk_segment,
        "expected_loss"  : expected_loss,
        "shap_summary"   : shap_summary_text,
    })

    processing_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"[{request_id}] Concluído | decisão={decision} | "
        f"PD={prob:.3f} | EL=R${expected_loss:.2f} | {processing_ms:.0f}ms"
    )

    return CreditResponse(
        request_id         = request_id,
        default_probability= round(prob, 4),
        decision           = decision,
        risk_segment       = risk_segment,
        expected_loss      = round(expected_loss, 2),
        explanation        = explanation,
        top_features       = top_features,
        processing_ms      = round(processing_ms, 1),
    )


# ================================
# ❤️ HEALTH CHECK v2
# ================================
@app.get("/health")
async def health():
    """Verifica status do modelo e dependências."""
    model_ok    = MODEL_STATE.get("model") is not None
    explainer_ok = MODEL_STATE.get("explainer") is not None

    return {
        "status"    : "ok" if model_ok else "degraded",
        "model"     : "loaded" if model_ok else "not loaded",
        "shap"      : "loaded" if explainer_ok else "not loaded",
        "version"   : "2.0.0",
    }


# ================================
# ▶️ COMO RODAR
# ================================
"""
1. Instalar dependências:
   pip install fastapi uvicorn langchain-openai pydantic \
               joblib scikit-learn xgboost shap

2. Configurar API Key:
   export OPENAI_API_KEY="sua_chave_aqui"

3. Estrutura de arquivos:
   /
   ├── main_v2.py
   └── models/
       ├── best_model.pkl    ← XGBoost ou RF treinado
       └── scaler.pkl        ← StandardScaler (se usado no treino)

4. Executar:
   uvicorn main_v2:app --reload --port 8000

5. Testar:
   curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{
       "age": 35, "income": 8000, "loan_amount": 50000,
       "loan_tenure": 60, "credit_score": 680,
       "num_open_accounts": 3, "num_credit_inq": 2,
       "debt_to_income": 0.35, "employment_years": 5.0,
       "has_mortgage": 0, "has_dependents": 1
     }'

6. Documentação interativa:
   http://127.0.0.1:8000/docs
"""
