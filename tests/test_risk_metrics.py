import pandas as pd
from src.domain.risk_metrics import calculate_hhi, calculate_top_n_share
from src.domain.constants import COL_NET_WEIGHT_TON, COL_ORIGIN_COUNTRY


def test_hhi_monopolio():
    df = pd.DataFrame([
        {COL_ORIGIN_COUNTRY: "Argentina", COL_NET_WEIGHT_TON: 1000},
    ])
    assert calculate_hhi(df, COL_ORIGIN_COUNTRY, COL_NET_WEIGHT_TON) == 10000


def test_hhi_diversificado():
    df = pd.DataFrame([
        {COL_ORIGIN_COUNTRY: f"P{i}", COL_NET_WEIGHT_TON: 100} for i in range(10)
    ])
    # 10 países con share igual de 10% c/u: HHI = 10 * (10^2) = 1000
    assert round(calculate_hhi(df, COL_ORIGIN_COUNTRY, COL_NET_WEIGHT_TON), 6) == 1000


def test_top_n_share():
    df = pd.DataFrame([
        {COL_ORIGIN_COUNTRY: "A", COL_NET_WEIGHT_TON: 600},
        {COL_ORIGIN_COUNTRY: "B", COL_NET_WEIGHT_TON: 300},
        {COL_ORIGIN_COUNTRY: "C", COL_NET_WEIGHT_TON: 100},
    ])
    assert calculate_top_n_share(df, COL_ORIGIN_COUNTRY, COL_NET_WEIGHT_TON, 1) == 60
    assert calculate_top_n_share(df, COL_ORIGIN_COUNTRY, COL_NET_WEIGHT_TON, 2) == 90
