# Deep Sequential Models for Finance

This project explores sequential models applied to financial time series:

- [ ] Portfolio optimization with Markowitz
- [ ] RNN
- [ ] LSTM
- [ ] BiLSTM
- [ ] GANs for synthetic scenarios

Organized for educational, research, and portfolio-building purposes.

## Project Structure

- `notebooks/`: experiments and visualizations
- `src/`: modular and reusable code
- `data/`: raw and processed datasets
- `outputs/`: trained models and charts
- `reports/`: theoretical notes

## Requirements

```bash
pip install -r requirements.txt

## LeoComments
- Classe abstrata p modelos
- POO
- Começar com Marko, encontrar ponto de falhha. Após isos ir para RedesN
- saber a media e a cov é quase impossivel, p 
-        o MI não é constante, crece com o tempo.
- centralizar e analisar melhor o montante dos portfolios formados dentro da fronteira
# Marone Ideias

# dificlidade alimentar be merkowitz. com uma boa media e uma boa covariancia. uma media que so considera retornos diarios, seria pior que o dos valores mensais

# para ter o retono mensal tem q esperar acabar o mes. o day tradoer nao tem essa informacao completa. # (marone: acho essa vertente de desenvolvimento a menos interessante

# teria uma forma alternativa de calcualr os retonros diarios

# rede neural p prever a media em outros pontos

# com poucas informacoes (menors que temhso ate ofim do mes) conseguimos bater a qualidade do retorno do mes

# como danados eu detemindo f ()? (Leo: (rede neural). Aconselho, usar o SQlearn. Entrada valores de hoje até o começo do mes atual.!! ISSO PODE SER PARA GERAR UMA MEDIA?? OU UM PREDITOR DO DIA SEGUINTE

### ???? usar depis passando de entrada os anos e prever o proximo (Marone: CAMINHO MAIS INTERESSANTE)