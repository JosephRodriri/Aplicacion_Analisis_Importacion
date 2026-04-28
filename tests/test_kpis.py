import pandas as pd
from src.domain.kpis import calculate_kpis
from src.domain.constants import (
    COL_NET_WEIGHT_TON, COL_FOB_USD, COL_CIF_USD, COL_FREIGHT_USD,
    COL_PRICE_USD_PER_TON, COL_ORIGIN_COUNTRY,
)


def make_df(rows):
    return pd.DataFrame(rows)


def test_kpis_empty():
    kpis = calculate_kpis(pd.DataFrame())
    assert kpis.declaraciones == 0
    assert kpis.toneladas == 0
    assert kpis.top_origen == "N/A"


def test_kpis_basic():
    df = make_df([
        {COL_NET_WEIGHT_TON: 100, COL_FOB_USD: 20000, COL_CIF_USD: 22000,
         COL_FREIGHT_USD: 1500, COL_PRICE_USD_PER_TON: 200,
         COL_ORIGIN_COUNTRY: "Argentina"},
        {COL_NET_WEIGHT_TON: 200, COL_FOB_USD: 50000, COL_CIF_USD: 55000,
         COL_FREIGHT_USD: 4000, COL_PRICE_USD_PER_TON: 250,
         COL_ORIGIN_COUNTRY: "Canadá"},
    ])
    kpis = calculate_kpis(df)
    assert kpis.declaraciones == 2
    assert kpis.toneladas == 300
    assert kpis.fob_total_musd == 0.07
    assert kpis.cif_total_musd == 0.077
    # FOB ponderado: 70000 / 300 = 233.33
    assert round(kpis.fob_per_ton, 2) == 233.33
    # CFR ponderado: (70000 + 5500) / 300 = 251.67
    assert round(kpis.cfr_per_ton, 2) == 251.67
    # Top origen: Canadá (200/300 = 66.67%)
    assert kpis.top_origen == "Canadá"
    assert round(kpis.top_origen_pct, 2) == 66.67
