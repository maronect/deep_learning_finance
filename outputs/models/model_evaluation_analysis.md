# Análise dos Resultados de Avaliação dos Modelos Preditivos

## Resumo Executivo

Os resultados mostram que **prever retornos de ações é extremamente desafiador**, com ambos os modelos (Ridge e MLP) apresentando desempenho fraco. Os valores de R² negativos indicam que os modelos estão pior do que simplesmente prever a média histórica.

## 1. Métricas Gerais

### Resumo por Modelo

| Modelo | MSE Médio | R² Médio | Correlação Média |
|--------|-----------|----------|------------------|
| **Ridge** | 0.0307 | -0.601 | 0.020 |
| **MLP** | 0.0326 | -1.006 | 0.053 |

### Principais Observações:

1. **Ridge tem melhor MSE médio** (0.0307 vs 0.0326) - 6% melhor
2. **Ridge tem melhor R² médio** (-0.60 vs -1.01) - menos pior
3. **MLP tem correlação média ligeiramente melhor** (0.053 vs 0.020) - mas ainda muito baixa

## 2. Interpretação das Métricas

### R² Negativo: O que significa?

- **R² negativo** indica que o modelo está **pior do que prever a média histórica**
- R² = 1 - (SS_res / SS_tot), onde SS_res é a soma dos quadrados dos resíduos
- Quando R² < 0, significa que SS_res > SS_tot, ou seja, os erros do modelo são maiores que a variância dos dados
- **Conclusão**: Os modelos não conseguem capturar padrões preditivos significativos nos retornos mensais

### MSE (Mean Squared Error)

- **Variação entre ativos**: 
  - Menor MSE: VIVT3.SA (Ridge) = 0.0043
  - Maior MSE: PCAR3.SA (Ridge) = 0.1250 (29x maior!)
- **Interpretação**: PCAR3.SA é muito mais volátil e difícil de prever

### Correlação

- **Valores muito baixos** (0.02 a 0.29) indicam que as previsões têm **pouca relação linear** com os retornos reais
- A melhor correlação é CSNA3.SA (MLP) = 0.286, ainda assim muito baixa
- **Correlações negativas** em alguns casos (ex: PINE4.SA Ridge = -0.19) indicam que o modelo está prevendo na direção oposta

## 3. Comparação Ridge vs MLP por Ativo

### Ridge é melhor em MSE em 8 de 10 ativos:
- CSNA3.SA, EMBR3.SA, PETR3.SA, RADL3.SA, TOTS3.SA, VALE3.SA, VIVT3.SA, WEGE3.SA

### MLP é melhor em MSE em apenas 2 ativos:
- PCAR3.SA, PINE4.SA

### Análise:
- **Ridge (Regressão Linear)** é mais consistente e tem melhor desempenho geral
- **MLP** não consegue superar o modelo linear simples, sugerindo que:
  - Não há padrões não-lineares significativos para capturar
  - O MLP pode estar sofrendo de overfitting
  - A complexidade adicional do MLP não compensa

## 4. Análise por Ativo

### Melhores Ativos para Previsão (menor MSE):
1. **VIVT3.SA** (Ridge): MSE = 0.0043, R² = -0.22, Corr = 0.24
2. **RADL3.SA** (Ridge): MSE = 0.0080, R² = -0.60, Corr = 0.05
3. **TOTS3.SA** (Ridge): MSE = 0.0121, R² = -0.32, Corr = 0.14

### Piores Ativos para Previsão (maior MSE):
1. **PCAR3.SA** (Ridge): MSE = 0.1250, R² = -0.08, Corr = -0.08
2. **PINE4.SA** (Ridge): MSE = 0.0501, R² = -0.63, Corr = -0.19
3. **CSNA3.SA** (Ridge): MSE = 0.0315, R² = -0.50, Corr = 0.19

### Observações:
- **VIVT3.SA** tem o melhor desempenho relativo, mas ainda com R² negativo
- **PCAR3.SA** é extremamente difícil de prever (alta volatilidade)
- Ativos com menor volatilidade tendem a ter melhor MSE

## 5. Implicações para o Portfólio

### Por que os modelos ainda geram portfólios com bom desempenho?

1. **Previsão de Média vs Previsão Ponto a Ponto**:
   - A avaliação aqui mede a capacidade de prever retornos mensais individuais
   - Para otimização de portfólio, o que importa é a **média esperada** (μ) ao longo do tempo
   - Mesmo com previsões ruins ponto a ponto, a média pode estar razoável

2. **Diversificação**:
   - Erros de previsão em ativos individuais podem se cancelar no portfólio
   - A otimização de Markowitz usa a matriz de covariância, que pode compensar erros nas médias

3. **Métricas de Portfólio vs Métricas de Previsão**:
   - Sharpe Ratio, retorno anualizado e volatilidade do portfólio são métricas diferentes
   - Um modelo pode ter previsões ruins mas ainda gerar portfólios eficientes se:
     - As previsões capturam diferenças relativas entre ativos
     - A estrutura de correlação está bem estimada

## 6. Conclusões e Recomendações

### Conclusões Principais:

1. **Prever retornos é muito difícil**: R² negativos confirmam que não há padrões preditivos claros nos dados mensais

2. **Ridge supera MLP**: O modelo linear simples é mais eficaz, sugerindo que:
   - Padrões lineares são mais relevantes que não-lineares
   - MLP pode estar overfitting
   - Complexidade adicional não compensa

3. **Variação entre ativos**: Alguns ativos (VIVT3.SA) são mais previsíveis que outros (PCAR3.SA)

4. **Correlações baixas**: As previsões têm pouca relação com retornos reais, mas ainda podem ser úteis para otimização de portfólio

### Recomendações:

1. **Focar em Ridge**: O modelo linear é mais simples, mais rápido e tem melhor desempenho

2. **Melhorar features**: 
   - Adicionar features macroeconômicas
   - Incluir indicadores técnicos
   - Considerar dados de sentimento

3. **Ajustar janela**: Testar diferentes valores de `window` (atualmente 24 meses)

4. **Regularização**: Ajustar o parâmetro `alpha` do Ridge para melhorar generalização

5. **Ensemble**: Combinar previsões de múltiplos modelos pode melhorar robustez

6. **Focar em portfólio**: Mesmo com previsões ruins, o desempenho do portfólio pode ser bom devido à diversificação

## 7. Limitações da Análise

1. **Período de teste**: Apenas ~71 meses de teste (após split de 70%)
2. **Dados mensais**: Retornos mensais têm menos sinal que dados diários
3. **Features simples**: Apenas lags de retornos, sem features macro ou técnicas
4. **Métrica de avaliação**: MSE, R² e correlação medem previsão ponto a ponto, não utilidade para portfólio

## 8. Próximos Passos Sugeridos

1. Avaliar se as previsões melhoram com features adicionais
2. Testar diferentes períodos de janela e regularização
3. Comparar com baseline simples (média móvel, momentum)
4. Avaliar se métricas de portfólio melhoram mesmo com previsões ruins
5. Considerar modelos de ensemble ou boosting

