"""
Módulo de visualização e exportação de gráficos e resultados.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from src.optimization.markowitz import portfolio_return, portfolio_volatility


def plot_efficient_frontier_comparison(
    models: list,
    save_path: str = None,
    figsize: tuple = (12, 8)
):
    """
    Plota comparação de fronteiras eficientes para múltiplos modelos.
    
    Parâmetros:
    -----------
    models : list
        Lista de dicionários com:
        {
            "name": str,
            "mean_returns": pd.Series,
            "cov": pd.DataFrame,
            "color": str,
            "linestyle": str
        }
    save_path : str, optional
        Caminho para salvar a figura
    figsize : tuple
        Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    lambdas = np.arange(0, 1.01, 0.01)
    
    for model in models:
        mean = model["mean_returns"]
        cov = model["cov"]
        name = model["name"]
        color = model.get("color", "blue")
        linestyle = model.get("linestyle", "-")
        
        ret_curve = []
        vol_curve = []
        
        for lamb in lambdas:
            from src.optimization.markowitz import solve_markowitz
            w = solve_markowitz(mean, cov, lamb=lamb)
            if w is not None:
                ret = portfolio_return(w, mean)
                vol = portfolio_volatility(w, cov)
                ret_curve.append(ret)
                vol_curve.append(vol)
        
        plt.plot(
            vol_curve, ret_curve,
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=2
        )
    
    plt.xlabel("Volatilidade (Risco)")
    plt.ylabel("Retorno Esperado")
    plt.title("Comparação de Fronteiras Eficientes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_portfolio_timeseries(
    models: list,
    returns: pd.DataFrame,
    split_idx: int,
    target_risk: float = None,
    lambdas: np.ndarray = None,
    w_list: list = None,
    ret_curve: list = None,
    vol_curve: list = None,
    save_path: str = None,
    figsize: tuple = (14, 8)
):
    """
    Plota séries temporais comparativas dos portfólios.
    
    Parâmetros:
    -----------
    models : list
        Lista de modelos
    returns : pd.DataFrame
        Retornos mensais
    split_idx : int
        Índice de divisão treino/teste
    target_risk : float, optional
        Risco alvo para selecionar lambda
    lambdas : np.ndarray, optional
        Array de lambdas usados
    w_list : list, optional
        Lista de pesos para cada lambda
    ret_curve : list, optional
        Curva de retornos
    vol_curve : list, optional
        Curva de volatilidades
    save_path : str, optional
        Caminho para salvar
    figsize : tuple
        Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    test_returns = returns.iloc[split_idx:]
    
    for i, model in enumerate(models):
        name = model["name"]
        color = model.get("color", "blue")
        linestyle = model.get("linestyle", "-")
        
        if target_risk is not None and lambdas is not None and w_list is not None:
            # Encontra lambda mais próximo do risco alvo
            if vol_curve and len(vol_curve) > i:
                min_diff = float('inf')
                best_idx = 0
                for j, vol in enumerate(vol_curve[i]):
                    diff = abs(vol - target_risk)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = j
                weights = w_list[i][best_idx]
            else:
                # Usa lambda médio
                weights = w_list[i][len(w_list[i]) // 2] if w_list else None
        else:
            # Usa lambda padrão
            from src.optimization.markowitz import solve_markowitz
            weights = solve_markowitz(model["mean_returns"], model["cov"], lamb=0.5)
        
        if weights is not None:
            portfolio_ret = test_returns.dot(weights)
            acum = (1 + portfolio_ret).cumprod()
            
            plt.plot(
                acum.index, acum,
                label=name,
                linewidth=2,
                color=color,
                linestyle=linestyle
            )
    
    plt.title("Comparação Temporal dos Portfólios (Backtest)")
    plt.xlabel("Tempo (Mensal)")
    plt.ylabel("Crescimento Acumulado")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_covariance_heatmap(
    cov_matrix: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8)
):
    """
    Plota heatmap da matriz de covariância.
    
    Parâmetros:
    -----------
    cov_matrix : pd.DataFrame
        Matriz de covariância
    save_path : str, optional
        Caminho para salvar
    figsize : tuple
        Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cov_matrix,
        annot=True,
        fmt='.4f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title("Matriz de Covariância dos Ativos")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predicted_returns_histogram(
    lr_predictions: pd.Series,
    mlp_predictions: pd.Series,
    save_path: str = None,
    figsize: tuple = (12, 5)
):
    """
    Plota histograma comparativo dos retornos previstos por LR e MLP.
    
    Parâmetros:
    -----------
    lr_predictions : pd.Series
        Previsões da Regressão Linear
    mlp_predictions : pd.Series
        Previsões do MLP
    save_path : str, optional
        Caminho para salvar
    figsize : tuple
        Tamanho da figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].hist(lr_predictions.values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title("Distribuição dos Retornos Previstos - Regressão Linear")
    axes[0].set_xlabel("Retorno Previsto")
    axes[0].set_ylabel("Frequência")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(mlp_predictions.values, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title("Distribuição dos Retornos Previstos - MLP")
    axes[1].set_xlabel("Retorno Previsto")
    axes[1].set_ylabel("Frequência")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sharpe_comparison(
    metrics_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6)
):
    """
    Plota gráfico de barras comparando Sharpe Ratio dos modelos.
    
    Parâmetros:
    -----------
    metrics_df : pd.DataFrame
        DataFrame com coluna 'Model' e 'Sharpe'
    save_path : str, optional
        Caminho para salvar
    figsize : tuple
        Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    models = metrics_df['Model'].values
    sharpe_values = metrics_df['Sharpe'].values
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, sharpe_values, color=colors, edgecolor='black', alpha=0.7)
    
    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title("Comparativo de Índice de Sharpe por Modelo")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Modelo")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()
    
    plt.close()

