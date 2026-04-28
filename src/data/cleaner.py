"""Limpieza y normalización de tipos del DataFrame de importaciones."""
import numpy as np
import pandas as pd

from src.domain.constants import (
    COL_SUBMISSION_DATE, COL_YEAR, COL_MONTH_NUM, COL_YEAR_MONTH,
    COL_NET_WEIGHT_KG, COL_NET_WEIGHT_TON,
    COL_FOB_USD, COL_CIF_USD, COL_FOB_MUSD, COL_CIF_MUSD,
    COL_PRICE_USD_PER_TON,
)


RENAME_MAP = {
    "Mes": "month",
    "Fecha de Presentación": "submission_date",
    "Aduana": "customs",
    "Régimen Importación": "regime",
    "Regimen Importación": "regime",
    "Tipo de importación": "import_type",
    "Razón Social del Importador": "importer",
    "NIT del Importador": "importer_nit",
    "Departamento del Importador": "importer_dept",
    "Descripción de la partida arancelaria": "tariff_description",
    "Código Partida": "tariff_code",
    "País de origen": "origin_country",
    "País de procedencia": "source_country",
    "País de compra": "purchase_country",
    "Continente Origen": "origin_continent",
    "Continente Compra": "purchase_continent",
    "Proveedor": "supplier",
    "Vía de transporte": "transport_mode",
    "Peso en kilos netos": "net_weight_kg",
    "Peso en kilos brutos": "gross_weight_kg",
    "Valor FOB (USD)": "fob_usd",
    "Valor CIF (USD)": "cif_usd",
    "Valor FOB (COP)": "fob_cop",
    "Valor CIF (COP)": "cif_cop",
    "Fletes (USD)": "freight_usd",
    "Valor seguro": "insurance_usd",
    "Tasa de Cambio": "fx_rate",
    "Forma de pago": "payment_type",
    "Acuerdo de Tratamiento Arancelario": "trade_agreement",
    "Total pagado": "total_paid_cop",
    "IVA pagado": "vat_paid",
    "Arancel Pagado": "tariff_paid",
}

NUMERIC_COLS = [
    "net_weight_kg", "gross_weight_kg",
    "fob_usd", "cif_usd", "fob_cop", "cif_cop",
    "freight_usd", "insurance_usd", "fx_rate",
    "total_paid_cop", "vat_paid", "tariff_paid",
]

CATEGORICAL_COLS = [
    "customs", "regime", "import_type", "importer", "importer_dept",
    "tariff_description", "tariff_code", "origin_country", "source_country",
    "purchase_country", "origin_continent", "purchase_continent",
    "supplier", "transport_mode", "payment_type", "trade_agreement",
]


def parse_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Parsea fechas y deriva year/month_num/year_month. Retorna (df, n_invalid)."""
    try:
        df[COL_SUBMISSION_DATE] = pd.to_datetime(
            df[COL_SUBMISSION_DATE], errors="coerce", format="mixed", dayfirst=False,
        )
    except (ValueError, TypeError):
        df[COL_SUBMISSION_DATE] = pd.to_datetime(df[COL_SUBMISSION_DATE], errors="coerce")

    invalid = int(df[COL_SUBMISSION_DATE].isna().sum())
    df[COL_YEAR] = df[COL_SUBMISSION_DATE].dt.year
    df[COL_MONTH_NUM] = df[COL_SUBMISSION_DATE].dt.month
    df[COL_YEAR_MONTH] = df[COL_SUBMISSION_DATE].dt.to_period("M").dt.to_timestamp()
    return df, invalid


def parse_numerics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convierte columnas numéricas. Retorna (df, dict de problemas por columna)."""
    issues: dict[str, int] = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        before_nulls = df[col].isna().sum()
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^\d\.\-,]", "", regex=True)
                .str.replace(",", "", regex=False)
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = df[col].isna().sum() - before_nulls
        if new_nulls > 0:
            issues[col] = int(new_nulls)
    return df, issues


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena nulos y normaliza strings de columnas categóricas."""
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = (
                df[col].fillna("Sin información").astype(str).str.strip().str.title()
            )
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas: net_weight_ton, fob_musd, cif_musd, price_usd_per_ton."""
    df[COL_NET_WEIGHT_TON] = (
        df[COL_NET_WEIGHT_KG] / 1000 if COL_NET_WEIGHT_KG in df.columns else np.nan
    )

    if COL_FOB_USD in df.columns:
        df[COL_FOB_MUSD] = df[COL_FOB_USD] / 1_000_000
    else:
        df[COL_FOB_USD] = np.nan
        df[COL_FOB_MUSD] = np.nan

    if COL_CIF_USD in df.columns:
        df[COL_CIF_MUSD] = df[COL_CIF_USD] / 1_000_000
    else:
        df[COL_CIF_USD] = np.nan
        df[COL_CIF_MUSD] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        df[COL_PRICE_USD_PER_TON] = np.where(
            df[COL_NET_WEIGHT_TON].fillna(0) > 0,
            df[COL_FOB_USD] / df[COL_NET_WEIGHT_TON].replace(0, np.nan),
            np.nan,
        )
    return df
