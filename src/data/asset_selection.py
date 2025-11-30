"""
Diversified Asset Selection Module
----------------------------------

Este módulo implementa múltiplas estratégias de seleção de ativos visando
MÁXIMA DIVERSIFICAÇÃO para Markowitz, Regressão Linear (Ridge) e MLP.

Inclui:
 - Seleção por baixa correlação total
 - Seleção gulosa por menor correlação máxima
 - Seleção dos 5 pares menos correlacionados
 - Seleção de pares mais estáveis ao longo de 2010→2025
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from src.data.loader import load_prices, compute_returns


# ============================================================
# 1. Universo grande de ações brasileiras (mantido)
# ============================================================

def get_brazilian_stocks_universe() -> List[str]:
    energy = ["PETR4.SA","PETR3.SA","ELET3.SA","ELET6.SA","EQTL3.SA","CSAN3.SA","UGPA3.SA"]
    mining_steel = ["VALE3.SA","GGBR4.SA","CSNA3.SA","USIM5.SA","CMIG4.SA","GOAU4.SA"]
    banks = ["ITUB4.SA","BBDC4.SA","BBAS3.SA","SANB11.SA","BRSR6.SA","BPAN4.SA","ABCB4.SA","PINE4.SA"]
    retail_consumer = ["ABEV3.SA","VIVT3.SA","RENT3.SA","MGLU3.SA","PCAR3.SA","MRVE3.SA"]
    industrial_tech = ["WEGE3.SA","EMBR3.SA","RADL3.SA","TOTS3.SA","TIMS3.SA","CYRE3.SA","KLBN11.SA"]
    financial_services = ["B3SA3.SA","CAML3.SA","SUZB3.SA"]
    construction = ["CYRE3.SA","EZTC3.SA","JHSF3.SA","MRVE3.SA"]

    all_stocks = (
        energy + mining_steel + banks +
        retail_consumer + industrial_tech +
        financial_services + construction
    )

    return list(dict.fromkeys(all_stocks))


# ============================================================
# 2. Seleção clássica: menor soma de correlações absolutas
# ============================================================

def _select_by_sum_abs_correlation(returns: pd.DataFrame, n_assets: int) -> List[str]:
    corr_matrix = returns.corr()
    abs_corr_sum = (corr_matrix.abs().sum() - 1.0).sort_values()
    return abs_corr_sum.head(n_assets).index.tolist()


# ============================================================
# 3. Seleção gulosa: minimizar correlação máxima
# ============================================================

def _select_by_min_max_correlation(returns: pd.DataFrame, n_assets: int) -> List[str]:
    corr = returns.corr().abs()

    # inicializar com ativo menos correlacionado em média
    ranking = (corr.sum() - 1).sort_values()
    selected = [ranking.index[0]]

    for _ in range(n_assets - 1):
        remaining = [a for a in corr.columns if a not in selected]
        best_asset = None
        best_score = 999

        for asset in remaining:
            score = corr.loc[selected, asset].max()
            if score < best_score:
                best_score = score
                best_asset = asset

        selected.append(best_asset)

    return selected


# ============================================================
# 4. NOVO — Seleção dos 5 pares menos correlacionados
# ============================================================

def _select_lowest_corr_pairs(returns: pd.DataFrame, n_pairs: int = 5) -> List[str]:
    corr = returns.corr()

    # gerar lista de pares únicos (triângulo inferior)
    pairs = []
    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    # ordenar por correlação crescente
    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    selected = []
    used_assets = set()

    for a, b, c in pairs_sorted:
        if a not in used_assets and b not in used_assets:
            selected.extend([a, b])
            used_assets.update({a, b})

        if len(selected) >= 2 * n_pairs:
            break

    return selected


# ============================================================
# 5. NOVO — Seleção baseada em estabilidade temporal 2010→2025
# ============================================================

def _select_stable_pairs(
    prices: pd.DataFrame,
    start_year: int = 2010,
    end_year: int = 2025,
    n_pairs: int = 5,
    return_freq: str = "monthly"
) -> List[str]:
    """
    Seleção baseada em estabilidade temporal de correlações.
    
    IMPORTANTE: Usa retornos mensais para manter consistência com o resto do pipeline.
    Para retornos mensais, requer pelo menos 6 meses por ano (metade do ano).
    """
    # Converter para retornos mensais para manter consistência com o pipeline
    returns = compute_returns(prices, freq=return_freq)
    cols = returns.columns

    # armazenar séries de correlação ano a ano
    corr_time: Dict[Tuple[str, str], List[float]] = {}

    for year in range(start_year, end_year + 1):
        year_slice = returns[returns.index.year == year]

        # Para retornos mensais, requer pelo menos 6 meses (metade do ano)
        min_periods = 6 if return_freq == "monthly" else 50
        if len(year_slice) < min_periods:
            continue

        corr_year = year_slice.corr()

        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pair = (cols[i], cols[j])
                val = corr_year.iloc[i, j]

                if pair not in corr_time:
                    corr_time[pair] = []

                corr_time[pair].append(val)

    # calcular estatísticas
    stats = []
    for pair, corr_vals in corr_time.items():
        if len(corr_vals) > 5:  # mínimo de anos válidos
            mean_corr = np.mean(corr_vals)
            std_corr = np.std(corr_vals)
            max_corr = np.max(corr_vals)

            stats.append((pair[0], pair[1], mean_corr, std_corr, max_corr))

    df_stats = pd.DataFrame(stats, columns=["A", "B", "mean", "std", "max"])

    # queremos baixa média, baixa volatilidade e baixa correlação máxima
    df_stats["score"] = df_stats["mean"] + df_stats["std"] + df_stats["max"]
    df_sorted = df_stats.sort_values("score")

    selected = []
    used = set()

    for _, row in df_sorted.iterrows():
        a, b = row["A"], row["B"]
        if a not in used and b not in used:
            selected.extend([a, b])
            used.update({a, b})
        if len(selected) >= 2 * n_pairs:
            break

    return selected


# ============================================================
# 6. Função Master — escolhe a estratégia desejada
# ============================================================

def select_assets(
    start_date: str,
    end_date: str,
    method: str = "lowest_corr_pairs",
    n_assets: int = 10,
    return_freq: str = "monthly",
    min_data_coverage: float = 0.85
) -> List[str]:
    """
    Métodos disponíveis:

    - "sum_abs_correlation"   → menor correlação total
    - "min_max_correlation"   → menor correlação máxima
    - "lowest_corr_pairs"     → seleciona 5 pares menos correlacionados
    - "stable_corr_pairs"     → pares mais estáveis entre 2010–2025
    """

    universe = get_brazilian_stocks_universe()

    prices = load_prices(
        universe,
        start=start_date,
        end=end_date,
        min_data_coverage=min_data_coverage
    )

    returns = compute_returns(prices, freq=return_freq)

    # ---- chamar método desejado ----

    if method == "sum_abs_correlation":
        return _select_by_sum_abs_correlation(returns, n_assets)

    if method == "min_max_correlation":
        return _select_by_min_max_correlation(returns, n_assets)

    if method == "lowest_corr_pairs":
        return _select_lowest_corr_pairs(returns, n_pairs=n_assets // 2)

    if method == "stable_corr_pairs":
        return _select_stable_pairs(prices, n_pairs=n_assets // 2, return_freq=return_freq)

    raise ValueError(f"Método '{method}' inválido.")


# ============================================================
# 7. Utilitário extra: obter matriz de correlação
# ============================================================

def get_correlation_matrix(returns: pd.DataFrame, selected_assets: List[str] = None):
    if selected_assets:
        returns = returns[selected_assets]
    return returns.corr()
