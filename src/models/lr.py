# src/predictors/expected_returns.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
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
    feature_dfs = []  # lista para armazenar DataFrames

    for col in returns.columns: # Col = ativo, linha = data point (dia)
        col_features = {}
        for i in range(1, window+1):
            col_features[f"{col}_lag_{i}"] = returns[col].shift(i) # retorno do dia anterior (valor de t-1 aparece em t)
        # features[f"{col}_mean_{window}"] = returns[col].shift(1).rolling(window).mean() # média móvel (5 dias), nao incluindo o dia atual - shift() no leakage
        # features[f"{col}_std_{window}"] = returns[col].shift(1).rolling(window).std() # desvio padrão móvel (5 dias) - - shift() no leakage

        feature_dfs.append(pd.DataFrame(col_features, index=returns.index))
    features = pd.concat(feature_dfs, axis=1)
    # Se os retornos passados forem positivos e estáveis, a média e o lag serão positivos 
    #   \_ o modelo tende a prever retorno positivo (momentum).
    return features.dropna() # remove linhas com NaN gerados pelos shifts e rolling

def predict_mean_returns(
    returns: pd.DataFrame,
    window: int,
    alpha_blend,
    model_machine_learning: str
) -> pd.Series:
    """
    Versão melhorada com:
    - Regularização Ridge
    - Clipping de previsões irreais
    - Shrink (blend histórico + previsão)
    """
    X = create_features(returns, window=window)

    predicted_means = {} # dicionario p armazenar os retornos previstos para cada ativo

    # scaler = StandardScaler() # padroniza as features (média 0, desvio 1)
    X_scaled = X # Array com as mesmas dimensões que X, mas padronizado
    # X_scaled = scaler.fit_transform(X) # Array com as mesmas dimensões que X, mas padronizado
    split_idx = int(len(X_scaled) * 0.7)

    historical_mean = returns.mean()

    returns_train = returns.loc[X.index].iloc[:split_idx]  # retornos alinhados com X, apenas treino
    cov_matrix = returns_train.cov()  
    mean_train = returns_train.mean()  

    # Para cada ativo, treinamos um modelo simples
    for col_i, col in enumerate(returns.columns):
        # alvo: retorno atual
        
        # divisao da base de dados em treino/teste
        y = returns[col].loc[X.index] # vetor de saida y para o ativo atual / retorno do ativo em cada data (alinha com X)
        X_train = X_scaled[:split_idx]
        y_train = y.values[:split_idx]
        
        X_test = X_scaled[split_idx:]   # nao usar, leakage
        y_test = y.values[split_idx:] 
        
        if model_machine_learning == 'Ridge':
            model = Ridge(alpha=1.0)

        elif model_machine_learning == 'LinearRegression':
            model = LinearRegression()

        elif model_machine_learning == 'MLPRegressor':
            model = MLPRegressor(
                hidden_layer_sizes=(50, 50),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=1000,
                random_state=42
            )
        
        model.fit(X_train, y_train) #treina com todas as linhas de X_scaled e tem y como alvo

        
        # Predição recursiva (sem leakage)
        y_pred = []
        returns_modified = returns.copy()  # cópia dos retornos para atualizar com previsões
        N_test = len(y_test)
        for i in range(N_test):
            # Recalcula features usando apenas dados até o ponto atual
            returns_available = returns_modified.iloc[:split_idx + i]
            X_current = create_features(returns_available, window=window)
                
            # Padroniza usando apenas dados de treino
            X_current_scaled = X_current.iloc[[-1]]  # última linha (mais recente)
            # X_current_scaled = scaler.transform(X_current.iloc[[-1]])  # última linha (mais recente)
            
            # Prediz próximo retorno
            y_pred_i = model.predict(X_current_scaled)[0]
            # y_pred_i = np.clip(y_pred_i, -0.1, 0.1)
            y_pred.append(y_pred_i)
            
            # Atualiza o DataFrame com a previsão para usar no próximo passo
            returns_modified.iloc[split_idx + i, col_i] = y_pred_i
        
        y_pred_mean = np.mean(y_pred)

        #     X_test_i = create_features(returns_modified[:split_idx + i], window=window)
        #     X_scaled_test_i = scaler.fit_transform(X_test_i)
        #     y_pred_i = model.predict(X_scaled_test_i[split_idx + i].reshape(1, -1))[0]
        #     y_pred.append(y_pred_i)
        #     returns_modified[split_idx + i] = y_pred_i  # atualiza o array com o retorno previsto
        # y_pred_mean = np.mean(y_pred)
        
        # blending da previsão com a média histórica para evitar previsões extremas
        y_pred_final = alpha_blend * y_pred_mean + (1 - alpha_blend) * mean_train[col]

        print(f"Ativo: {col} - Retorno Médio Previsto: {y_pred_final:.6f} - Retorno Médio Real: {historical_mean[col]:.6f}.")

        

        # X_last = X_scaled[-2].reshape(1, -1)    # penultima (not ultima janeala) (último dia em que todas as features estão disponíveis), transforma em array 2D de uma linha
        # y_pred = model.predict(X_last)[0] # pevisao do próximo retorno (o retorno de amanhã, t+1)

          ### ALTERAÇÃO 3: clipping para conter explosões
        # y_pred = np.clip(y_pred, -clip_value, clip_value)


        predicted_means[col] = y_pred_final # armazena o retorno previsto no dicionário
        # y_dict[col] = model # armazena o modelo treinado (uso futuro)

    return pd.Series(predicted_means, name="Predicted_Mean_Returns"), split_idx, mean_train, cov_matrix # retorna a série com os retornos previstos para cada ativo
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

        model = Ridge(alpha=1.0).fit(X_train, y_train)
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
        model = Ridge(alpha=1.0).fit(X_scaled, y)
        coefs[col] = model.coef_

    return coefs
