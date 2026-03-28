# Credit Risk ML 🏦

> Sistema de Machine Learning para previsão de risco de crédito.  
> Pipeline completo de ponta a ponta, pronto para portfólio e ambiente de produção.

---

## Índice

1. [Descrição do Problema](#descrição-do-problema)
2. [Abordagem Técnica](#abordagem-técnica)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Pipeline Detalhado](#pipeline-detalhado)
5. [Decisões Técnicas e Trade-offs](#decisões-técnicas-e-trade-offs)
6. [Resultados Esperados](#resultados-esperados)
7. [Como Rodar](#como-rodar)
8. [API Reference](#api-reference)
9. [Próximos Passos](#próximos-passos)

---

## Descrição do Problema

Instituições financeiras precisam avaliar automaticamente se um solicitante de crédito irá honrar sua dívida. Este sistema:

- **Prevê** a probabilidade de inadimplência (*default*) com base em 11 características do cliente.
- **Apoia a decisão** de aprovação ou rejeição de crédito com um threshold configurável.
- **Simula o impacto financeiro** das decisões — quantificando lucro, prejuízo e custo de oportunidade.

### Definição do Target

| Valor | Significado |
|-------|------------|
| `0`   | Cliente pagou (bom pagador) |
| `1`   | Cliente não pagou (default) |

---

## Abordagem Técnica

### Features de entrada

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `age` | int | Idade do cliente |
| `income` | float | Renda mensal (BRL) |
| `loan_amount` | float | Valor do empréstimo |
| `loan_tenure` | int | Prazo em meses |
| `credit_score` | int | Score de crédito (300–850) |
| `num_open_accounts` | int | Contas abertas |
| `num_credit_inq` | int | Consultas de crédito (6 meses) |
| `debt_to_income` | float | Razão dívida/renda |
| `employment_years` | float | Tempo de emprego |
| `has_mortgage` | 0/1 | Possui hipoteca |
| `has_dependents` | 0/1 | Possui dependentes |

### Features derivadas (engenharia)

| Feature | Fórmula | Intuição de negócio |
|---------|---------|-------------------|
| `loan_to_income` | `loan_amount / income` | Carga relativa do empréstimo |
| `monthly_payment_ratio` | `(loan/tenure) / (income/12)` | Comprometimento mensal |
| `credit_utilization` | `num_inq × num_accounts` | Pressão de crédito |
| `risk_score` | Combinação ponderada | Score composto 0–1 |
| `stability_index` | `log(emp_years) × log(income/1k)` | Proxy de estabilidade |
| `high_risk_flag` | `credit<580 AND dti>0.5` | Flag binária de alto risco |

### Modelos treinados

| Modelo | Tipo | Justificativa |
|--------|------|--------------|
| Logistic Regression | Linear | Baseline interpretável, probabilidades calibradas |
| Random Forest | Ensemble bagging | Robusto a outliers, importância de features |
| XGBoost | Ensemble boosting | Estado da arte em tabular data, regularização nativa |

---

## Estrutura do Projeto

```
credit-risk-ml/
├── data/
│   ├── raw/              # Dados brutos externos (CSV)
│   ├── processed/        # Dados limpos
│   └── synthetic/
│       └── generate_data.py   # Gerador de dados sintéticos
│
├── src/
│   ├── data_preprocessing.py  # Ingestão, limpeza, split
│   ├── feature_engineering.py # Novas features, escalonamento
│   ├── train_baseline.py      # Logistic Regression
│   ├── train_models.py        # Random Forest + XGBoost
│   ├── evaluate.py            # Métricas + visualizações
│   └── simulate_decision.py   # Decisão de crédito + simulação financeira
│
├── app/
│   └── main.py           # FastAPI: /predict, /health, /model/info
│
├── models/               # Artefatos .pkl (gerados pelo pipeline)
├── reports/              # Gráficos PNG + CSV de métricas
├── notebooks/            # Análise exploratória (EDA)
├── run_pipeline.py       # Orquestrador completo
└── requirements.txt
```

---

## Pipeline Detalhado

### 1. Ingestão de Dados
```python
from src.data_preprocessing import load_data
df = load_data("data/synthetic/credit_data.csv")
```
Valida existência do arquivo e presença da coluna `default`.

### 2. Limpeza
- Remove duplicatas
- Remove linhas sem target
- Imputa nulos numéricos pela **mediana** (robusta a outliers)
- Clipa outliers via **1.5×IQR**

### 3. Divisão sem Data Leakage
```
df_limpo → train_test_split(stratify=y, test_size=0.20)
```
**Por que estratificar?** Classes desbalanceadas (ex: 20% de default) exigem que treino e teste mantenham a mesma proporção. Sem estratificação, um split aleatório poderia ter 15% de default no treino e 35% no teste — tornando a avaliação irreal.

**Zero leakage:** o `StandardScaler` é ajustado **apenas** no `X_train` via `fit_transform` e aplicado ao `X_test` via `transform`. Jamais o contrário.

### 4. Feature Engineering
Criação de 6 variáveis derivadas com base em conhecimento de domínio de crédito. Todas são calculadas a partir das features brutas sem dependência de estatísticas externas — seguras para treino e inferência.

### 5. Treinamento
- **Baseline** (LR): `class_weight="balanced"`, `C=1.0`
- **Random Forest**: 300 árvores, `max_depth=12`, `class_weight="balanced"`
- **XGBoost**: 400 estimadores, LR=0.05, early stopping no validation set; `scale_pos_weight` calculado automaticamente

### 6. Avaliação

```
Métricas → ROC-AUC / Accuracy / Precision / Recall / F1
Gráficos → Curva ROC, Matriz de Confusão, Feature Importance, Comparação
```

**Por que ROC-AUC como métrica principal?**
Classes desbalanceadas tornam o *accuracy* enganoso — um modelo que sempre prevê "não default" tem 80% de accuracy mas AUC de 0.5. O AUC mede separabilidade independente do threshold.

### 7. Simulação de Negócio

```
threshold configurável (padrão: 0.70)
score_default > threshold → REJEITAR
score_default ≤ threshold → APROVAR

Lucro = (bons aprovados × +100) - (ruins aprovados × -500) - (bons rejeitados × -10)
```

A otimização do threshold varre 100 pontos entre 0.01 e 0.99 e encontra o ponto de máximo lucro.

---

## Decisões Técnicas e Trade-offs

| Decisão | Alternativa considerada | Por quê esta |
|---------|------------------------|--------------|
| XGBoost como modelo final | LightGBM, CatBoost | Maturidade, API sklearn-compatível, documentação |
| StandardScaler | MinMaxScaler, RobustScaler | RF/XGBoost são invariantes a escala; LR beneficia de normalização |
| Imputação por mediana | KNNImputer, IterativeImputer | Velocidade + robustez a outliers; missing rate < 3% |
| Estratificação no split | Split aleatório | Mantém proporção de classes em ambos os conjuntos |
| `class_weight="balanced"` | SMOTE, undersampling | Sem criar amostras sintéticas; mais simples e reprodutível |
| joblib para serialização | pickle | Thread-safe, compressão, suporte nativo a numpy |
| Matplotlib Agg backend | Display interativo | Compatível com ambientes sem display (CI, Docker, Render) |

### Sobre o threshold de 0.70

O threshold padrão de 0.70 é conservador — rejeita mais para proteger contra prejuízo. A função `optimize_threshold()` encontra o threshold ótimo para o contexto de negócio configurado. **Em produção, este valor deve ser validado com a equipe de risco.**

---

## Resultados Esperados

Com o dataset sintético de 50.000 clientes:

| Modelo | ROC-AUC esperado | F1 esperado |
|--------|-----------------|-------------|
| Logistic Regression | ~0.82 | ~0.65 |
| Random Forest | ~0.88 | ~0.72 |
| XGBoost | ~0.90 | ~0.75 |

*Valores aproximados — variam com o seed e configuração de hiperparâmetros.*

---

## Como Rodar

### Pré-requisitos
```bash
Python 3.10+
```

### Instalação
```bash
git clone https://github.com/seuuser/credit-risk-ml.git
cd credit-risk-ml
pip install -r requirements.txt
```

### Rodar o pipeline completo
```bash
# Com dataset sintético (gerado automaticamente)
python run_pipeline.py

# Com seu próprio CSV
python run_pipeline.py --data caminho/para/dados.csv
```

O pipeline irá:
1. Gerar/carregar os dados
2. Treinar os 3 modelos
3. Salvar gráficos em `reports/`
4. Salvar modelos em `models/`
5. Exibir comparação de métricas e simulação financeira

### Rodar a API
```bash
# Rodar o pipeline primeiro para gerar os modelos
python run_pipeline.py

# Iniciar a API
uvicorn app.main:app --reload --port 8000
```

Acesse a documentação interativa em: **http://localhost:8000/docs**

### Exemplo de chamada à API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "has_dependents": 1
  }'
```

Resposta esperada:
```json
{
  "default_probability": 0.182341,
  "decision": "APROVAR",
  "threshold_used": 0.7,
  "risk_band": "BAIXO",
  "model_version": "XGBClassifier"
}
```

### Variáveis de ambiente
```bash
MODEL_PATH=models/xgboost.pkl        # Modelo a usar na API
SCALER_PATH=models/scaler_final.pkl  # Scaler persistido
DECISION_THRESHOLD=0.70              # Threshold de aprovação
LOG_LEVEL=INFO                       # Nível de log
```

---

## API Reference

### `POST /predict`

**Body:** `CreditRequest` (JSON)  
**Response:** `PredictionResponse`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `default_probability` | float | Probabilidade de inadimplência (0–1) |
| `decision` | string | `"APROVAR"` ou `"REJEITAR"` |
| `threshold_used` | float | Threshold utilizado na decisão |
| `risk_band` | string | `"BAIXO"` / `"MÉDIO"` / `"ALTO"` |
| `model_version` | string | Tipo do modelo em produção |

### `GET /health`

Status da API e se o modelo está carregado.

### `GET /model/info`

Tipo do modelo, threshold e campos de entrada esperados.

---

## Próximos Passos

- [ ] Adicionar SHAP values para explicabilidade por cliente
- [ ] Implementar monitoramento de drift com Evidently AI
- [ ] Adicionar testes automatizados (`pytest tests/`)
- [ ] Containerizar com Docker
- [ ] Deploy no Render.com / Railway / GCP Cloud Run
- [ ] Versionamento de modelos com MLflow
- [ ] Endpoint de batch scoring (`POST /predict/batch`)

---

## Licença

MIT — livre para uso pessoal e comercial.
