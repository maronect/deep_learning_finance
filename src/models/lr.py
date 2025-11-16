# src/predictors/expected_returns.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.data.loader import compute_returns
from sklearn.metrics import mean_squared_error, r2_score


def create_features(returns: pd.DataFrame, window: int = 5) -> pd.DataFrame: #OK
    """
    Cria features simples baseadas nos retornos passados.
    - média móvel
    - desvio padrão móvel
    - retorno defasado (lag)

    Retorna um DataFrame com as features para cada ativo.
    """
    features = pd.DataFrame(index=returns.index)

    for col in returns.columns: # Col = ativo, linha = data point (dia)
        features[f"{col}_lag1"] = returns[col].shift(1) # retorno do dia anterior (valor de t-1 aparece em t)
        features[f"{col}_mean_{window}"] = returns[col].rolling(window).mean() # média móvel (5 dias), incluindo o dia atual
        features[f"{col}_std_{window}"] = returns[col].rolling(window).std() # desvio padrão móvel (5 dias)

    # Se os retornos passados forem positivos e estáveis, a média e o lag serão positivos 
    #   \_ o modelo tende a prever retorno positivo (momentum).
    return features.dropna() # remove linhas com NaN gerados pelos shifts e rolling

def predict_mean_returns(prices: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Usa Regressão Linear para prever o retorno esperado (médio)
    de cada ativo com base em janelas passadas de retornos.

    Retorna um pd.Series com a média prevista (E[R_i]) para cada ativo.
    """
    returns = compute_returns(prices, freq='daily')
    features = create_features(returns, window=window)

    X = features
    y_dict = {} # usado para armazenar os modelos treinados (uso futuro)
    predicted_means = {} # dicionario p armazenar os retornos previstos para cada ativo

    scaler = StandardScaler() # padroniza as features (média 0, desvio 1)
    X_scaled = scaler.fit_transform(X) # Array com as mesmas dimensões que X, mas padronizado

    # Para cada ativo, treinamos um modelo simples
    for col in returns.columns:
        # alvo: retorno atual
        y = returns[col].loc[X.index] # vetor de saida y para o ativo atual / retorno do ativo em cada data (alinha com X)
        model = LinearRegression() # O X MARCA O TESOURO XXXXXXXXXXXXXXXXXXXXX (matar regressao antes!)

        model.fit(X_scaled, y.values) #treina com todas as linhas de X_scaled e tem y como alvo
        # \_“Dada a combinação dos retornos passados (lags / médias / desvios) de todos os ativos, qual é o retorno desse ativo hoje?”
        
        # previsão do último ponto conhecido (t+1)
        X_last = X_scaled[-1].reshape(1, -1)    # ultima janeala (último dia em que todas as features estão disponíveis), transforma em array 2D de uma linha
        y_pred = model.predict(X_last)[0] # pevisao do próximo retorno (o retorno de amanhã, t+1)

        predicted_means[col] = y_pred # armazena o retorno previsto no dicionário
        y_dict[col] = model # armazena o modelo treinado (uso futuro)

    return pd.Series(predicted_means, name="Predicted_Mean_Returns")# retorna a série com os retornos previstos para cada ativo
    # aprende a relacao linear do comportamento rescento dos retornos e o retorno seguinte(t+1)
    # e prevê o próximo retorno


def evaluate_linear_model(prices: pd.DataFrame, window: int = 5, test_size: float = 0.2):
    """
    Avalia o desempenho preditivo da regressão linear
    para cada ativo, usando uma divisão treino/teste.

    Retorna um DataFrame com MSE, R² e correlação real x previsto.
    """
    returns = compute_returns(prices, freq='daily')
    features = create_features(returns, window=window)
    X = features.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    split_idx = int(len(X_scaled) * (1 - test_size)) # índice para divisão treino/teste (separando temporalmente ate uma data)

    for col in returns.columns:
        y = returns[col].loc[X.index].values
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        corr = np.corrcoef(y_test, y_pred)[0, 1]

        results.append({
            "Ticker": col,
            "MSE": mse, # baixo = boa precisão relativa (escala depende da magnitude dos retornos)
            "R²": r2, # R² > 0 = o modelo tem algum poder de explicação sobre os retornos
            "Correlação": corr # correlação positiva (>0.1) = previsões estão alinhadas com a direção dos retornos
        })

    return pd.DataFrame(results)

def inspect_coefficients(prices: pd.DataFrame, window: int = 5):
    """
    Mostra os coeficientes da regressão linear para cada ativo,
    indicando o peso das features na previsão dos retornos.
    """
    returns = compute_returns(prices, freq='daily')
    features = create_features(returns, window=window)
    X = features.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    coefs = pd.DataFrame(index=X.columns)

    for col in returns.columns:
        y = returns[col].loc[X.index].values
        model = LinearRegression().fit(X_scaled, y)
        coefs[col] = model.coef_

    return coefs
