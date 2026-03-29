# Credit Risk ML 
## Business impact 
---
- Random Forest gerou +R$853k
- Threshold otimizado aumentou lucro em ~7–10%
- Decisão baseada em impacto financeiro, não apenas métricas

> Sistema de Machine Learning para decisão de crédito orientada a lucro.
> Pipeline completo end-to-end com modelagem, explicabilidade, simulação de negócio e deploy em produção.
> Pipeline completo de ponta a ponta — pronto para portfólio e ambiente de produção.
> Projeto focado em decisões reais de crédito, onde o objetivo não é maximizar AUC, mas lucro financeiro.
---

## Índice

1. [Descrição do Problema](#descrição-do-problema)
2. [Abordagem Técnica](#abordagem-técnica)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Notebooks](#notebooks)
5. [Pipeline Detalhado](#pipeline-detalhado)
6. [Camada Avançada](#camada-avançada)
7. [Decisões Técnicas e Trade-offs](#decisões-técnicas-e-trade-offs)
8. [Resultados Esperados](#resultados-esperados)
9. [Como Rodar](#como-rodar)
10. [API Reference](#api-reference)
11. [Deploy](#deploy)
12. [Licença](#licença)

---

## Descrição do Problema

Instituições financeiras precisam avaliar automaticamente se um solicitante de crédito irá honrar sua dívida. Este sistema:

- **Prevê** a probabilidade de inadimplência (*default*) com base em 11 características do cliente
- **Apoia a decisão** de aprovação ou rejeição de crédito com threshold configurável
- **Simula o impacto financeiro** das decisões — quantificando lucro, prejuízo e custo de oportunidade
- **Explica** cada decisão individual via SHAP values
- **Monitora** degradação do modelo em produção via drift detection

### Definição do Target

| Valor | Significado |
|-------|------------|
| `0` | Cliente pagou (bom pagador) |
| `1` | Cliente não pagou (default) |

---

## Abordagem Técnica

### Features de entrada (11 originais)

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

### Features derivadas — engenharia de domínio (6 novas)

| Feature | Fórmula | Intuição de negócio |
|---------|---------|-------------------|
| `loan_to_income` | `loan_amount / income` | Carga relativa do empréstimo |
| `monthly_pay_ratio` | `(loan/tenure) / (income/12)` | Comprometimento mensal da renda |
| `risk_score` | Combinação ponderada (0–1) | Score composto de risco |
| `stability_index` | `log(emp_years) × log(income/1k)` | Proxy de estabilidade financeira |
| `high_risk_flag` | `credit < 580 AND dti > 0.5` | Flag binária de alto risco |
| `inq_per_account` | `num_inq / num_accounts` | Pressão de crédito por conta |

### Modelos treinados

| Modelo | Tipo | Papel no pipeline |
|--------|------|------------------|
| Logistic Regression | Linear | Baseline interpretável — referência mínima obrigatória |
| Random Forest | Ensemble bagging | Robusto a outliers, feature importance nativa |
| XGBoost | Ensemble boosting | Modelo principal — estado da arte em tabular data |
| XGBoost (Optuna) | Boosting tunado | Versão otimizada com busca bayesiana de hiperparâmetros |

---

## Estrutura do Projeto

```
credit-risk-ml/
│
├── data/
│   ├── raw/                          # Datasets externos (CSV do Kaggle, etc.)
│   ├── processed/                    # Dados limpos e transformados
│   └── synthetic/
│       └── generate_data.py          # Gerador de 50k clientes sintéticos
│
├── notebooks/                        # Pipeline em 5 notebooks sequenciais
│   ├── 01_eda.ipynb                  # Análise exploratória completa
│   ├── 02_feature_engineering.ipynb  # Limpeza, novas features, split, scaler
│   ├── 03_model_baseline.ipynb       # Logistic Regression + análise de erros
│   ├── 04_model_advanced.ipynb       # Random Forest + XGBoost + comparação
│   └── 05_decision_simulation.ipynb  # Simulação financeira + threshold ótimo
│
├── src/                              # Módulos Python reutilizáveis
│   ├── data_preprocessing.py         # load_data(), clean_data(), split_data()
│   ├── feature_engineering.py        # build_features(), scale_features()
│   ├── train_baseline.py             # train_baseline()
│   ├── train_models.py               # train_random_forest(), train_xgboost()
│   ├── evaluate.py                   # evaluate_model(), plot_roc_curves(), compare_models()
│   └── simulate_decision.py          # simulate_business(), optimize_threshold()
│
├── app/
│   └── main.py                       # FastAPI: /predict, /health, /model/info
│
├── models/                           # Artefatos .pkl gerados pelo pipeline
│   ├── scaler.pkl / scaler_final.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── best_model.pkl
│
├── reports/                          # Gráficos e métricas gerados
│   ├── roc_curves.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── confusion_matrix_*.png
│   ├── business_simulation.png
│   ├── profit_curves.png
│   └── model_metrics.csv
│
├── run_pipeline.py                   # Orquestrador — 1 comando roda tudo
├── credit_risk_kaggle_v2.ipynb       # Pipeline unificado para Kaggle
├── credit_risk_next_steps.ipynb      # Camada avançada (Optuna, SHAP, Evidently, Docker)
└── requirements.txt
```

---

## Notebooks

O projeto é organizado em **5 notebooks sequenciais** onde cada um persiste artefatos que o próximo consome, mais dois notebooks complementares.

### Fluxo entre notebooks

```
01_eda
  └─→ 02_feature_engineering  →  splits.pkl + scaler.pkl
        └─→ 03_model_baseline  →  logistic_regression.pkl + lr_metrics.pkl
              └─→ 04_model_advanced  →  random_forest.pkl + xgboost.pkl + best_model.pkl
                    └─→ 05_decision_simulation  →  reports/business_simulation.png
```

### Descrição de cada notebook

#### `01_eda.ipynb` — Análise Exploratória

- `df.info()`, `df.describe()`, mapa visual de valores nulos
- Distribuição do target e quantificação do desbalanceamento
- Histogramas por classe de default para todas as features numéricas
- Boxplots por classe com diferença percentual de medianas destacada
- Mapa de correlação completo + ranking de correlação com o target
- **Saída:** insights documentados que guiam todas as decisões do pipeline

#### `02_feature_engineering.ipynb` — Transformações

- Limpeza: remoção de duplicatas, imputação por mediana, clipping IQR
- Criação das 6 features derivadas com justificativa de negócio explícita
- Validação da correlação de cada nova feature com o target
- Split estratificado 80/20 com assert automático de verificação
- StandardScaler fitado **apenas** no treino
- **Saída:** `models/splits.pkl`, `models/scaler.pkl`

#### `03_model_baseline.ipynb` — Logistic Regression

- Justificativa para uso obrigatório de baseline antes de modelos complexos
- Avaliação: AUC, F1, Precision, Recall, Curva ROC, Curva Precision-Recall
- Matrizes de confusão absoluta e normalizada por linha
- Coeficientes brutos + Odds Ratio — interpretabilidade de negócio
- Análise de erros por quadrante: onde o modelo erra e qual o custo de cada erro
- **Saída:** `models/logistic_regression.pkl`, `models/lr_metrics.pkl`

#### `04_model_advanced.ipynb` — Random Forest + XGBoost

- Random Forest: 200 árvores, `max_depth=12`, `class_weight="balanced"`
- XGBoost: `scale_pos_weight` calculado automaticamente, `early_stopping_rounds` no construtor
- Ganho vs baseline calculado e exibido para cada modelo
- Curvas ROC sobrepostas + tabela comparativa de todas as métricas
- Feature importance lado a lado (RF vs XGBoost), com destaque nas features derivadas
- Seleção do modelo final baseada em impacto de negócio (lucro), não apenas métricas estatísticas
- **Saída:** `models/random_forest.pkl`, `models/xgboost.pkl`, `models/best_model.pkl`

#### `05_decision_simulation.ipynb` — Simulação de Negócio

- `BusinessConfig`: dataclass com parâmetros financeiros totalmente configuráveis
- `simulate_business()`: lucro por quadrante incluindo custo de oportunidade
- Simulação comparativa para os 3 modelos com threshold=0.70
- `optimize_threshold()`: varre 200 pontos entre 0.01–0.99 e maximiza lucro
- Tabela comparando threshold padrão vs threshold ótimo por modelo
- Exemplos de decisão individual (cliente bom, médio e alto risco)
- **Saída:** `reports/business_simulation.png`, `reports/profit_curves.png`

### `credit_risk_kaggle_v2.ipynb` — Pipeline Unificado para Kaggle

Pipeline completo em **um único arquivo** com 15 seções — ideal para submissão no Kaggle, demonstração em entrevista ou apresentação rápida. Inclui adicionalmente:

- Checklist **programático** de boas práticas com validação automática via asserts
- Análise de erros com distribuição de scores por quadrante
- Curva Precision-Recall (mais informativa que ROC para classes desbalanceadas)
- Grade 2×3 de matrizes de confusão — absoluta e normalizada para cada modelo

### `credit_risk_next_steps.ipynb` — Camada Avançada

Notebook de evolução com os 5 próximos passos implementados:

| # | Seção | O que entrega |
|---|-------|--------------|
| 13 | StratifiedKFold | CV com Pipeline sklearn — leakage zero por fold, boxplot de AUCs |
| 14 | Optuna | Busca bayesiana (TPE) sobre 9 hiperparâmetros, curva de otimização, importância dos HPs |
| 15 | SHAP Values | Summary plot global, waterfall individual, bar plot de importância |
| 16 | Evidently AI | Detecção de drift, relatório HTML interativo, visualização de divergência |
| 17 | FastAPI + Docker | Gera `deploy/main.py`, `Dockerfile` e `requirements.txt` prontos para uso |

---

## Pipeline Detalhado

### 1. Ingestão de Dados
```python
from src.data_preprocessing import load_data
df = load_data("data/synthetic/credit_data.csv")
# ou: df = pd.read_csv("/kaggle/input/seu-dataset/credit.csv")
```

### 2. Limpeza
- Remove duplicatas e linhas sem target
- Imputa nulos pela **mediana** — robusta a outliers, sem risco de leakage
- Clipa outliers via **1.5×IQR** — controla extremos sem remover amostras

### 3. Split sem Data Leakage
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

`stratify=y` garante que treino e teste tenham a mesma taxa de default. O `StandardScaler` usa `fit_transform` no treino e apenas `transform` no teste.

### 4. Feature Engineering
6 variáveis derivadas calculadas a partir das features brutas, sem dependência de estatísticas externas — seguras para aplicar igualmente em treino, teste e produção.

### 5. Treinamento
- **LR:** `class_weight="balanced"`, `C=1.0`, `max_iter=1000`
- **RF:** 200 árvores, `max_depth=12`, `class_weight="balanced"`
- **XGBoost:** 300 estimadores, `learning_rate=0.05`, `early_stopping_rounds=30` no construtor, `scale_pos_weight` automático

### 6. Avaliação
```
Métricas : ROC-AUC · Accuracy · Precision · Recall · F1
Gráficos : Curva ROC · Curva PR · Matriz de Confusão · Feature Importance · Comparação
```

**Por que ROC-AUC como métrica principal?** Classes desbalanceadas tornam o *accuracy* enganoso — um modelo que sempre prevê "sem default" tem 80% de accuracy com AUC de apenas 0.50.

### 7. Simulação de Negócio
```
threshold = 0.70  (configurável via BusinessConfig)

score > threshold  →  REJEITAR
score ≤ threshold  →  APROVAR

Lucro = (bons aprovados   × +100)
      − (ruins aprovados  × −500)
      − (bons rejeitados  × −10)   ← custo de oportunidade
```

`optimize_threshold()` varre 200 pontos e retorna o threshold que maximiza o lucro total para os parâmetros financeiros configurados.

---

## Camada Avançada

### Validação Cruzada com Pipeline sklearn

```python
pipe = Pipeline([('scaler', StandardScaler()), ('clf', modelo)])
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_validate(pipe, X, y, cv=skf, scoring=['roc_auc', 'f1'])
```

O scaler é re-fitado a cada fold isoladamente — zero leakage garantido mesmo dentro do CV.

### Tunagem com Optuna

```python
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30)
```

Busca bayesiana (TPE) sobre 9 hiperparâmetros do XGBoost. Mais eficiente que GridSearchCV para espaços grandes. Melhora típica de 0.01–0.03 no AUC.

### SHAP Values

```python
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_sc)
shap.plots.waterfall(exp)  # explicação individual
```

Três visualizações: summary plot global, waterfall de decisão individual e bar plot de importância por |mean SHAP|.

### Drift Monitoring com Evidently AI

```python
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=X_train, current_data=X_producao)
drift_report.save_html('drift_report.html')
```

Dashboard HTML interativo. Alerta quando > 30% das features estão com drift.

### Deploy com FastAPI + Docker

Pasta `deploy/` gerada automaticamente pelo notebook com `main.py`, `Dockerfile` e `requirements.txt` prontos.

---

## Decisões Técnicas e Trade-offs

| Decisão | Alternativa | Por quê esta |
|---------|-------------|--------------|
| XGBoost como modelo principal | LightGBM, CatBoost | API sklearn-compatível, amplamente usado em produção |
| `early_stopping_rounds` no construtor | No `.fit()` | Compatível com XGBoost ≥ 1.6 — evita `TypeError` |
| StandardScaler | MinMaxScaler, RobustScaler | RF/XGBoost são invariantes a escala; LR beneficia diretamente |
| Imputação por mediana | KNNImputer, IterativeImputer | Robusta a outliers, sem risco de leakage, missing rate < 3% |
| `class_weight="balanced"` | SMOTE, undersampling | Não cria amostras sintéticas; mais simples e reprodutível |
| Optuna TPE | GridSearchCV, RandomSearch | Aprende regiões promissoras; superior para espaços grandes |
| SHAP TreeExplainer | KernelExplainer | Nativo para árvores — ordens de magnitude mais rápido |
| joblib | pickle | Thread-safe, compressão nativa, suporte a arrays numpy |
| Matplotlib Agg backend | Display interativo | Compatível com Docker, CI/CD e Kaggle kernels sem display |

### Sobre o threshold de 0.70

Conservador por design — rejeita mais clientes para proteger contra o alto custo de −R$500 por default aprovado. `optimize_threshold()` encontra o ponto de máximo lucro para os parâmetros financeiros configurados. **Em produção, este valor deve ser validado com a equipe de risco e compliance.**

---

## Resultados Esperados

Com o dataset sintético de 50.000 clientes:

| Modelo | ROC-AUC | F1 | Recall |
|--------|---------|-----|--------|
| Logistic Regression | ~0.82 | ~0.65 | ~0.70 |
| Random Forest | ~0.88 | ~0.72 | ~0.74 |
| XGBoost (padrão) | ~0.90 | ~0.75 | ~0.76 |
| XGBoost (Optuna) | ~0.91–0.92 | ~0.77 | ~0.78 |

*Valores aproximados — variam com seed e configuração de hiperparâmetros.*

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

### Opção 1 — Pipeline completo (1 comando)

```bash
# Dataset sintético gerado automaticamente
python run_pipeline.py

# Com CSV próprio
python run_pipeline.py --data caminho/para/dados.csv
```

### Opção 2 — Notebooks sequenciais

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_model_baseline.ipynb
jupyter notebook notebooks/04_model_advanced.ipynb
jupyter notebook notebooks/05_decision_simulation.ipynb
```

Execute na ordem para aproveitar os artefatos persistidos entre notebooks.

### Opção 3 — Notebook Kaggle unificado

```bash
jupyter notebook credit_risk_kaggle_v2.ipynb
```

### Camada avançada

```bash
pip install optuna shap evidently
jupyter notebook credit_risk_next_steps.ipynb
```

Cobre: StratifiedKFold, Optuna, SHAP, Evidently e geração automática dos arquivos de deploy.

### API

```bash
python run_pipeline.py          # gera os modelos primeiro
uvicorn app.main:app --reload --port 8000
```

Documentação interativa: **http://localhost:8000/docs**

### Exemplo de chamada

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
MODEL_PATH=models/best_model.pkl     # Modelo em produção
SCALER_PATH=models/scaler_final.pkl  # Scaler persistido
DECISION_THRESHOLD=0.70              # Threshold de aprovação
LOG_LEVEL=INFO                       # Nível de log
```

---

## API Reference

### `POST /predict`

**Body:** `CreditRequest` — os 11 campos do cliente em JSON  
**Response:** `PredictionResponse`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `default_probability` | float | Probabilidade de inadimplência (0–1) |
| `decision` | string | `"APROVAR"` ou `"REJEITAR"` |
| `threshold_used` | float | Threshold utilizado na decisão |
| `risk_band` | string | `"BAIXO"` / `"MÉDIO"` / `"ALTO"` |
| `model_version` | string | Tipo do modelo em produção |

### `GET /health`

Status da API, confirmação de modelo carregado e threshold ativo.

### `GET /model/info`

Tipo do modelo, caminho, threshold e lista de campos esperados.

---

## Deploy

### Docker

```bash
cd deploy
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### Render.com (gratuito)

1. Suba a pasta `deploy/` em um repositório GitHub
2. New Web Service → conecta o repositório
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Suba `best_model.pkl` e `scaler.pkl` como arquivos do repositório

---

## Checklist de Boas Práticas

- [x] Estratificação no split — proporção de default igual em treino e teste
- [x] StandardScaler fit apenas no treino — zero data leakage
- [x] `class_weight="balanced"` na LR e RF
- [x] `scale_pos_weight` automático no XGBoost
- [x] `early_stopping_rounds` no construtor — compatível com XGBoost ≥ 1.6
- [x] Pipeline sklearn encapsulando scaler + modelo no cross-validation
- [x] Threshold otimizado por maximização de lucro
- [x] SHAP values para explicabilidade e compliance
- [x] Drift monitoring para detectar degradação em produção
- [x] API com validação Pydantic e healthcheck
- [x] Dockerfile com `python:3.11-slim` e healthcheck

---

## Licença

MIT — livre para uso pessoal e comercial.
