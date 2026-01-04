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
├── notebooks/              # Experimentos e análises
│   ├── 00-compare_models.ipynb    # Notebook principal comparativo
│   ├── 01-markowitz_optimization.ipynb
│   └── 02-linear_regretion.ipynb
├── src/                    # Código modular e reutilizável
│   ├── data/               # Carregamento e processamento de dados
│   ├── models/             # Modelos de ML (LR, MLP)
│   ├── optimization/       # Otimização de portfólios (Markowitz, Sharpe)
│   └── utils/              # Utilitários (visualização, exportação)
├── outputs/                # Resultados gerados
│   ├── charts/             # Gráficos PNG de alta resolução
│   ├── models/             # Métricas e pesos dos portfólios (CSV)
│   └── predictions/        # Previsões de retornos (CSV)
├── reports/                # Documentação teórica
├── article_official/       # Artigo científico em LaTeX
├── scripts/                # Scripts auxiliares
└── requirements.txt        # Dependências do projeto
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
