"""
Módulo de visualização e exportação de gráficos e resultados.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from src.optimization.markowitz import portfolio_return, portfolio_volatility
from src.utils.portfolio_utils import get_optimal_portfolio_max_sharpe


def plot_efficient_frontier_comparison(
    models: list,
    save_path: str = None,
    figsize: tuple = (12, 8),
    risk_free_rate: float = 0.0,
    highlight_sharpe: bool = True
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
    risk_free_rate : float, optional
        Taxa livre de risco (no período) para cálculo do Sharpe Ratio
    highlight_sharpe : bool, optional
        Se True, destaca o portfólio ótimo por Sharpe Ratio em cada modelo
    """
    # Configurar tamanho de fonte maior
    plt.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots(figsize=figsize)
    
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
        
        # Converter retornos para porcentagem
        ret_curve_pct = [r * 100 for r in ret_curve]
        
        ax.plot(
            vol_curve, ret_curve_pct,
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=2.5
        )
        
        # Destacar portfólio ótimo por Sharpe Ratio
        if highlight_sharpe:
            try:
                optimal_weights = get_optimal_portfolio_max_sharpe(
                    mean,
                    cov,
                    risk_free_rate=risk_free_rate
                )
                
                if optimal_weights is not None:
                    optimal_ret = portfolio_return(optimal_weights, mean)
                    optimal_vol = portfolio_volatility(optimal_weights, cov)
                    optimal_ret_pct = optimal_ret * 100
                    
                    # Marcar o ponto com uma estrela usando a cor do modelo
                    ax.scatter(
                        optimal_vol, optimal_ret_pct,
                        s=300,  # Tamanho grande do marcador
                        color='red',
                        edgecolors='black',
                        linewidths=1.5,
                        marker='*',  # Estrela para destacar
                        zorder=5  # Garantir que fique acima da linha
                    )
            except Exception as e:
                # Se houver erro ao calcular Sharpe, apenas continua sem destacar
                pass
    
    # Adicionar entrada na legenda para as estrelas (apenas uma vez)
    if highlight_sharpe:
        ax.scatter([], [], s=300, color='red', edgecolors='black', 
                   linewidths=1.5, marker='*', 
                   label='Maximum Sharpe Ratio Portfolio', zorder=5)
    
    ax.set_xlabel("Volatility (Risk)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Expected Return (%)", fontsize=16, fontweight='bold')
    ax.set_title("Efficient Frontier Comparison", fontsize=18, fontweight='bold', pad=20)
    
    # Grade mais visível
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legenda com fonte maior - inclui as fronteiras e a indicação das estrelas
    ax.legend(fontsize=13, framealpha=0.9, loc='best')
    
    # Ajustar tamanho dos ticks
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    if save_path:
        # Garantir que o caminho seja uma string e tenha extensão .png
        save_path = str(save_path)
        if not save_path.lower().endswith('.png'):
            save_path = save_path + '.png'
        
        # Salvar figura explicitamente como PNG
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Chart saved at: {save_path}")
    else:
        plt.show()
    
    plt.close()
    # Restaurar configuração padrão
    plt.rcParams.update({'font.size': 10})


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
    # Configurar tamanho de fonte maior
    plt.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots(figsize=figsize)
    
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
            
            ax.plot(
                acum.index, acum,
                label=name,
                linewidth=2.5,
                color=color,
                linestyle=linestyle
            )
    
    ax.set_title("Portfolio Time Series Comparison (Backtest)", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Time (Monthly)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Cumulative Growth", fontsize=16, fontweight='bold')
    
    # Grade mais visível
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legenda com fonte maior
    ax.legend(fontsize=13, framealpha=0.9, loc='best')
    
    # Ajustar tamanho dos ticks
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Formatar eixo Y para mostrar valores como múltiplos (ex: 1x, 2x, 3x)
    from matplotlib.ticker import FuncFormatter
    def format_multiplier(x, pos):
        return f'{x:.1f}x'
    ax.yaxis.set_major_formatter(FuncFormatter(format_multiplier))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Chart saved at: {save_path}")
    else:
        plt.show()
    
    plt.close()
    # Restaurar configuração padrão
    plt.rcParams.update({'font.size': 10})


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
    
    plt.title("Asset Covariance Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Chart saved at: {save_path}")
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
    axes[0].set_title("Predicted Returns Distribution - Linear Regression")
    axes[0].set_xlabel("Predicted Return")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(mlp_predictions.values, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title("Predicted Returns Distribution - MLP")
    axes[1].set_xlabel("Predicted Return")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Chart saved at: {save_path}")
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
    # Configurar tamanho de fonte maior
    plt.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = metrics_df['Model'].values
    sharpe_values = metrics_df['Sharpe'].values
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, sharpe_values, color=colors, edgecolor='black', alpha=0.7, linewidth=1.5)
    
    # Adiciona valores nas barras com fonte maior
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=14, fontweight='bold')
    
    ax.set_title("Sharpe Ratio Comparison by Model", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Sharpe Ratio", fontsize=16, fontweight='bold')
    ax.set_xlabel("Model", fontsize=16, fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=13)
    
    # Grade mais visível
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Ajustar tamanho dos ticks
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Chart saved at: {save_path}")
    else:
        plt.show()
    
    plt.close()
    # Restaurar configuração padrão
    plt.rcParams.update({'font.size': 10})

