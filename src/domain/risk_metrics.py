"""Métricas de riesgo: concentración, volatilidad, dependencia."""
import pandas as pd

from src.domain.constants import (
    COL_NET_WEIGHT_TON, COL_FOB_USD, COL_CIF_USD,
    COL_ORIGIN_COUNTRY, COL_SUPPLIER, COL_IMPORTER,
    COL_PRICE_USD_PER_TON, COL_SUBMISSION_DATE, COL_TRADE_AGREEMENT,
)
from src.domain.models import RiskMetrics


def calculate_hhi(df: pd.DataFrame, group_col: str, weight_col: str) -> float:
    """Calcula Herfindahl-Hirschman Index (escala 0-10000).

    >2500: mercado muy concentrado.
    1500-2500: moderadamente concentrado.
    <1500: diversificado.
    """
    if df.empty:
        return 0
    total = df[weight_col].sum()
    if total <= 0:
        return 0
    shares = df.groupby(group_col)[weight_col].sum() / total
    return float((shares ** 2).sum() * 10000)


def calculate_top_n_share(
    df: pd.DataFrame, group_col: str, weight_col: str, n: int = 5
) -> float:
    """% del peso total concentrado en los top N grupos."""
    if df.empty:
        return 0
    total = df[weight_col].sum()
    if total <= 0:
        return 0
    shares = df.groupby(group_col)[weight_col].sum().sort_values(ascending=False)
    return float(shares.head(n).sum() / total * 100)


def calculate_price_volatility(df: pd.DataFrame) -> float:
    """Coeficiente de variación de precio (std/mean) en %."""
    if df.empty or COL_PRICE_USD_PER_TON not in df.columns:
        return 0
    prices = df[COL_PRICE_USD_PER_TON].dropna()
    if prices.empty or prices.mean() == 0:
        return 0
    return float(prices.std() / prices.mean() * 100)


def calculate_price_trend(df: pd.DataFrame) -> float:
    """Variación % del precio promedio ponderado: últimos 3m vs 12m anteriores."""
    if df.empty or COL_SUBMISSION_DATE not in df.columns:
        return 0

    df = df.copy()
    df[COL_SUBMISSION_DATE] = pd.to_datetime(df[COL_SUBMISSION_DATE])
    max_date = df[COL_SUBMISSION_DATE].max()

    cutoff_recent = max_date - pd.DateOffset(months=3)
    cutoff_old = max_date - pd.DateOffset(months=15)

    recent = df[df[COL_SUBMISSION_DATE] >= cutoff_recent]
    previous = df[
        (df[COL_SUBMISSION_DATE] < cutoff_recent)
        & (df[COL_SUBMISSION_DATE] >= cutoff_old)
    ]

    def weighted_price(d: pd.DataFrame) -> float:
        ton = d[COL_NET_WEIGHT_TON].sum()
        return d[COL_FOB_USD].sum() / ton if ton > 0 else 0

    p_recent = weighted_price(recent)
    p_prev = weighted_price(previous)

    if p_prev == 0:
        return 0
    return float((p_recent / p_prev - 1) * 100)


def calculate_logistic_pct(df: pd.DataFrame) -> float:
    """% del costo logístico (CIF - FOB) sobre el FOB total."""
    if df.empty:
        return 0
    fob = df[COL_FOB_USD].sum()
    cif = df[COL_CIF_USD].sum()
    if fob <= 0:
        return 0
    return float((cif - fob) / fob * 100)


def calculate_pct_with_agreement(df: pd.DataFrame) -> float:
    """% de toneladas importadas bajo algún acuerdo comercial."""
    if df.empty or COL_TRADE_AGREEMENT not in df.columns:
        return 0
    total = df[COL_NET_WEIGHT_TON].sum()
    if total <= 0:
        return 0

    sin_acuerdo_labels = {"Sin Información", "Sin Acuerdo", "", "Nan"}
    with_agreement = df[~df[COL_TRADE_AGREEMENT].isin(sin_acuerdo_labels)]
    return float(with_agreement[COL_NET_WEIGHT_TON].sum() / total * 100)


def calculate_risk_metrics(df: pd.DataFrame) -> RiskMetrics:
    """Calcula todas las métricas de riesgo de un solo paso."""
    return RiskMetrics(
        hhi_origenes=calculate_hhi(df, COL_ORIGIN_COUNTRY, COL_NET_WEIGHT_TON),
        hhi_proveedores=calculate_hhi(df, COL_SUPPLIER, COL_NET_WEIGHT_TON),
        n_origenes=df[COL_ORIGIN_COUNTRY].nunique() if COL_ORIGIN_COUNTRY in df.columns else 0,
        n_proveedores=df[COL_SUPPLIER].nunique() if COL_SUPPLIER in df.columns else 0,
        top5_origenes_pct=calculate_top_n_share(df, COL_ORIGIN_COUNTRY, COL_NET_WEIGHT_TON, 5),
        top10_importadores_pct=calculate_top_n_share(df, COL_IMPORTER, COL_NET_WEIGHT_TON, 10),
        precio_cv_pct=calculate_price_volatility(df),
        delta_precio_3m_vs_12m_pct=calculate_price_trend(df),
        pct_logistico_sobre_fob=calculate_logistic_pct(df),
        pct_bajo_acuerdo=calculate_pct_with_agreement(df),
    )
