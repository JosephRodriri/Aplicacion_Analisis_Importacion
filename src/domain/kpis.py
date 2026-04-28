"""Cálculo de KPIs principales."""
import pandas as pd

from src.domain.constants import (
    COL_NET_WEIGHT_TON, COL_FOB_USD, COL_CIF_USD, COL_FREIGHT_USD,
    COL_PRICE_USD_PER_TON, COL_ORIGIN_COUNTRY,
)
from src.domain.models import ImportKPIs


def calculate_kpis(df: pd.DataFrame) -> ImportKPIs:
    """Calcula KPIs principales sobre el DataFrame filtrado.

    Asume que el DataFrame ya está limpio y con columnas en formato canónico.
    """
    if df.empty:
        return ImportKPIs(
            declaraciones=0, toneladas=0,
            fob_total_musd=0, cif_total_musd=0,
            fob_per_ton=0, cfr_per_ton=0,
            precio_promedio_simple=0,
            top_origen="N/A", top_origen_pct=0,
        )

    ton = df[COL_NET_WEIGHT_TON].sum()
    fob = df[COL_FOB_USD].sum()
    cif = df[COL_CIF_USD].sum()
    freight = df[COL_FREIGHT_USD].sum() if COL_FREIGHT_USD in df.columns else 0

    origen_shares = (
        df.groupby(COL_ORIGIN_COUNTRY)[COL_NET_WEIGHT_TON].sum() / ton
        if ton > 0 else pd.Series(dtype=float)
    )
    top_origen = origen_shares.idxmax() if not origen_shares.empty else "N/A"
    top_origen_pct = origen_shares.max() * 100 if not origen_shares.empty else 0

    precio_simple = (
        df[COL_PRICE_USD_PER_TON]
        .replace([float("inf"), float("-inf")], pd.NA)
        .mean()
    )

    return ImportKPIs(
        declaraciones=len(df),
        toneladas=ton,
        fob_total_musd=fob / 1e6,
        cif_total_musd=cif / 1e6,
        fob_per_ton=fob / ton if ton > 0 else 0,
        cfr_per_ton=(fob + freight) / ton if ton > 0 else 0,
        precio_promedio_simple=float(precio_simple) if pd.notna(precio_simple) else 0,
        top_origen=top_origen,
        top_origen_pct=top_origen_pct,
    )
