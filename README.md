# Deep Sequential Models for Finance

Projeto de pesquisa que explora a aplicação de modelos de aprendizado de máquina na otimização de portfólios financeiros, comparando abordagens clássicas com métodos modernos de previsão de retornos.

## Sobre o Projeto

Este projeto implementa e compara três abordagens distintas para otimização de portfólios utilizando a teoria de Markowitz:

1. **Markowitz Clássico**: Utiliza médias históricas simples para estimar retornos esperados
2. **Markowitz + Regressão Linear**: Emprega Regressão Linear Ridge com validação walk-forward para prever retornos
3. **Markowitz + MLP**: Utiliza Multi-Layer Perceptron (MLP) com arquitetura de 2 camadas para previsão de retornos

O objetivo é avaliar se modelos de aprendizado de máquina podem melhorar as estimativas de retornos esperados (μ) e, consequentemente, gerar portfólios com melhor desempenho ajustado ao risco.

## [X] O que foi Implementado

### Modelos e Otimização

- [X] **Otimização de Portfólios com Markowitz**: Implementação completa da teoria moderna de portfólios
- [X] **Regressão Linear (Ridge)**: Modelo de previsão com regularização L2 e validação walk-forward
- [X] **Multi-Layer Perceptron (MLP)**: Rede neural com 2 camadas ocultas (50 neurônios cada) para previsão de retornos
- [X] **Otimização por Máximo Sharpe Ratio**: Seleção de portfólios otimizados para melhor relação risco-retorno

### Funcionalidades Principais

- [X] **Seleção Automática de Ativos**: Algoritmo que seleciona ativos com baixa correlação e padrões estáveis
- [X] **Validação Walk-Forward**: Metodologia que evita data leakage, usando apenas dados históricos para previsões
- [X] **Blending de Previsões**: Combinação de previsões de ML com médias históricas (α = 0.3)
- [X] **Cálculo de Métricas Financeiras**: Sharpe Ratio, retorno anualizado, volatilidade anualizada e retorno acumulado
- [X] **Geração de Fronteiras Eficientes**: Visualização comparativa das fronteiras eficientes para cada modelo

### Visualizações e Exportação

- [X] **Gráficos Comparativos**: 5 visualizações (fronteiras eficientes, séries temporais, heatmaps, histogramas, comparação de Sharpe)
- [X] **Exportação de Resultados**: Métricas, pesos dos portfólios e previsões exportados em CSV
- [X] **Tabelas Formatadas**: Tabelas prontas para inclusão em artigos científicos

### Estrutura de Código

- [X] **Código Modular**: Organização em módulos reutilizáveis (`src/data`, `src/models`, `src/optimization`, `src/utils`)
- [X] **Notebook Comparativo**: Notebook principal (`00-compare_models.ipynb`) que executa toda a pipeline
- [X] **Documentação Completa**: Documentação detalhada da implementação e metodologia

## Resultados Observados

Com base na análise de 10 ações brasileiras no período de 2010-2025, os resultados obtidos foram:

| Modelo | Sharpe Ratio | Retorno Anual | Volatilidade Anual | Retorno Acumulado |
|--------|--------------|---------------|-------------------|-------------------|
| Markowitz Clássico | 0.5432 | 33.16% | 27.48% | 4.41x |
| **Markowitz + Regressão Linear** | **0.5911** | **35.73%** | 28.56% | **4.86x** |
| Markowitz + MLP | 0.5718 | 34.33% | 27.66% | 4.63x |

## Estrutura do Projeto

```
deep_learning_finance/
├── config/                          # YAML configuration — no hardcoded parameters
│   ├── pipeline.yaml                # Global pipeline parameters (assets, frequency, windows)
│   ├── models.yaml                  # ML model hyperparameters (Ridge α, MLP layers, etc.)
│   ├── optimization.yaml            # Markowitz parameters (risk-free rate, solver, bounds)
│   └── api.yaml                     # FastAPI server settings (host, port, CORS)
│
├── src/
│   ├── ingestion/                   # Stage 1: download and validate raw market data
│   │   ├── downloader.py            # Fetch OHLCV prices from yfinance
│   │   └── validators.py            # Check data completeness and integrity
│   ├── features/                    # Stage 2: feature engineering
│   │   ├── returns.py               # Compute returns at multiple frequencies
│   │   ├── asset_selection.py       # Select uncorrelated assets (4 strategies)
│   │   └── lag_features.py          # Build lag feature matrices for supervised learning
│   ├── models/                      # Stage 3: ML model training and prediction
│   │   ├── base.py                  # Abstract interface shared by all models
│   │   ├── ridge.py                 # Ridge Regression with walk-forward validation
│   │   ├── mlp.py                   # MLP with walk-forward validation
│   │   ├── rnn.py                   # LSTM/RNN (PyTorch) — implemented, not yet integrated
│   │   └── blending.py              # Blend ML predictions with historical mean
│   ├── optimization/                # Stage 4: portfolio optimization
│   │   ├── markowitz.py             # Markowitz formulation (return, covariance, constraints)
│   │   ├── sharpe.py                # Maximize Sharpe Ratio via SLSQP
│   │   └── evaluation.py            # Portfolio metrics (Sharpe, return, volatility, frontier)
│   ├── pipeline/                    # Orchestration: full or partial pipeline execution
│   │   ├── runner.py                # Entry point — runs all or selected stages
│   │   ├── stages.py                # Each pipeline stage as an independent function
│   │   └── context.py               # Shared state object passed between stages
│   ├── api/                         # FastAPI serving layer (Stage 3 of roadmap)
│   │   ├── main.py                  # App factory: middleware, routers, startup
│   │   ├── routers/
│   │   │   ├── health.py            # GET /health
│   │   │   ├── assets.py            # GET /assets
│   │   │   ├── predictions.py       # GET /predictions
│   │   │   ├── pipeline.py          # POST /pipeline/run
│   │   │   ├── portfolio.py         # GET /portfolio/weights, /portfolio/frontier
│   │   │   └── metrics.py           # GET /metrics/model, /metrics/portfolio
│   │   └── schemas/
│   │       ├── requests.py          # Pydantic input schemas
│   │       └── responses.py         # Pydantic output schemas
│   └── utils/
│       ├── config_loader.py         # Load and merge YAML configs
│       ├── visualization.py         # Comparative charts (matplotlib/seaborn)
│       └── export.py                # Persist artifacts to disk
│
├── artifacts/                       # Pipeline outputs (not committed to git)
│   ├── data/                        # Processed returns and feature datasets
│   ├── models/                      # Trained model parameters
│   ├── predictions/                 # Expected return predictions per run
│   ├── metrics/                     # Model and portfolio evaluation metrics
│   ├── weights/                     # Optimized portfolio weights
│   └── runs/                        # Execution logs (timestamp, config, status)
│
├── tests/
│   ├── conftest.py                  # Shared fixtures (synthetic data, mock configs)
│   ├── unit/                        # Fast, isolated function-level tests
│   │   ├── test_returns.py
│   │   ├── test_features.py
│   │   ├── test_markowitz.py
│   │   └── test_sharpe.py
│   └── integration/                 # End-to-end tests across multiple components
│       ├── test_pipeline.py
│       └── test_api.py
│
├── notebooks/                       # Exploratory analysis (not part of the pipeline)
│   ├── 00-compare_models.ipynb      # Legacy main pipeline — reference results
│   ├── 01-markowitz_optimization.ipynb
│   └── 02-linear_regretion.ipynb
│
├── article_official/
│   └── article.tex                  # Academic article in LaTeX
│
├── .github/workflows/ci.yml         # CI: test + Docker build on push/PR
├── Dockerfile                        # Production image (API + pipeline)
├── docker-compose.yml               # Local orchestration: API service
├── pyproject.toml                   # Project metadata, pytest and ruff config
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development and testing dependencies
└── README.md
```

## Como Usar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Executar o Notebook Comparativo

```bash
cd notebooks
jupyter notebook 00-compare_models.ipynb
```

Execute todas as células em ordem. O notebook irá:
- Carregar dados históricos automaticamente
- Treinar os modelos com validação walk-forward
- Gerar todos os gráficos comparativos
- Exportar resultados em CSV

### Resultados Gerados

Após a execução, você encontrará em `outputs/`:

- **5 Gráficos PNG**: Fronteiras eficientes, séries temporais, heatmaps, histogramas e comparação de Sharpe
- **Múltiplos CSVs**: Métricas dos portfólios, pesos otimizados e previsões de retornos
- **Tabelas Formatadas**: Prontas para inclusão em artigos científicos

## Metodologia

### Dados

- **Período**: 2010-2025 (15 anos de dados históricos)
- **Frequência**: Retornos mensais calculados a partir de preços diários
- **Ativos**: 10 ações brasileiras selecionadas automaticamente por baixa correlação
- **Fonte**: Dados históricos obtidos via `yfinance`

### Modelos de Previsão

- **Janela de Features**: 24 meses de retornos históricos
- **Validação Walk-Forward**: Re-treinamento mensal usando apenas dados históricos
- **Blending**: Combinação de previsões ML com média histórica (α = 0.3)

### Otimização

- **Critério**: Máximo Sharpe Ratio
- **Taxa Livre de Risco**: 15% ao ano (Selic 2025)
- **Matriz de Covariância**: Estimada a partir de retornos históricos

## Objetivos do Projeto

Este projeto foi desenvolvido para:

- **Educação**: Aprender e aplicar conceitos de otimização de portfólios e aprendizado de máquina
- **Pesquisa**: Investigar se modelos ML podem melhorar estimativas de retornos esperados
- **Portfólio**: Demonstrar habilidades em análise quantitativa e desenvolvimento de sistemas financeiros

## Status do Projeto

- [X] **Implementação Completa**: Todos os modelos e funcionalidades implementados
- [X] **Validação e Correções**: Problemas identificados e corrigidos (otimização por máximo Sharpe)
- [X] **Resultados Documentados**: Análise completa dos resultados com interpretação crítica
- [X] **Artigo Científico**: Artigo LaTeX completo com metodologia e discussão dos resultados

## Trabalhos Futuros

- [ ] Implementação de RNN para previsão de retornos
- [ ] Implementação de LSTM e BiLSTM
- [ ] GANs para geração de cenários sintéticos
- [ ] Expansão para outros mercados e períodos
- [ ] Implementação de restrições de diversificação mais sofisticadas

## Licença

Ver arquivo `LICENSE` para detalhes.

---

**Nota**: Este projeto é de natureza educacional e de pesquisa. Os resultados não constituem recomendações de investimento.
