"""
Dashboard de Importaciones de Maíz y Trigo en Colombia (2012-2025)
-------------------------------------------------------------------
Estructura inspirada en el demo de violencia intrafamiliar:
- Carga cacheada + normalización de columnas
- Filtros en sidebar
- KPIs arriba
- Grillas de gráficas (Plotly) abajo
- Insights al final

Ejecutar con:
    streamlit run streamlit_demo_importaciones.py
"""

import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Logging (útil para ver errores también en la terminal)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("importaciones_dashboard")

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
# Ruta por defecto del dataset. Se puede sobreescribir desde el sidebar.
# Uso parent (no parents[1]) para que funcione tanto si el script está en la
# raíz del proyecto como dentro de una carpeta.
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent / "Consolidado_importaciones_2012_2025.csv"
)
FALLBACK_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "Consolidado_importaciones_2012_2025.csv"
)

# Ruta al archivo JSON con configuración de alias y agrupaciones.
CONFIG_JSON_PATH = Path(__file__).resolve().parent / "comp_principales.json"

# Columnas mínimas que el dataset DEBE tener para que el dashboard funcione.
REQUIRED_COLUMNS = {
    "Fecha de Presentación",
    "Descripción de la partida arancelaria",
    "Peso en kilos netos",
    "Valor FOB (USD)",
    "País de origen",
}

st.set_page_config(
    page_title="Análisis de Importaciones de Maíz y Trigo",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta consistente para todas las gráficas
COLOR_SEQ = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"


# -----------------------------------------------------------------------------
# Normalización de nombres (importadores y proveedores)
# -----------------------------------------------------------------------------
def _load_name_config() -> dict:
    """Carga el JSON de configuración de alias y agrupaciones.

    Retorna un dict con las claves:
      - nit_to_group: {nit_str: nombre_grupo}  (de comp_principales)
      - importer_alias: {variante_title: nombre_canónico}  (de alias_importadores)
      - supplier_alias: {variante_title: nombre_canónico}  (de alias_proveedores)
    """
    result: dict = {"nit_to_group": {}, "importer_alias": {}, "supplier_alias": {}}

    if not CONFIG_JSON_PATH.is_file():
        logger.warning("No se encontró %s — no se aplicará normalización.", CONFIG_JSON_PATH)
        return result

    try:
        with open(CONFIG_JSON_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error leyendo %s: %s", CONFIG_JSON_PATH, exc)
        return result

    # 1) comp_principales → mapa NIT→grupo
    for group_name, nit_list in raw.get("comp_principales", {}).items():
        for nit in nit_list:
            result["nit_to_group"][str(nit).strip()] = group_name

    # 2) alias_importadores → mapa variante→canónico
    for canonical, variants in raw.get("alias_importadores", {}).items():
        canonical_title = canonical.strip().title()
        for v in variants:
            result["importer_alias"][v.strip().title()] = canonical_title

    # 3) alias_proveedores → mapa variante→canónico
    for canonical, variants in raw.get("alias_proveedores", {}).items():
        canonical_title = canonical.strip().title()
        for v in variants:
            result["supplier_alias"][v.strip().title()] = canonical_title

    logger.info(
        "Config cargada: %d NITs, %d alias importadores, %d alias proveedores.",
        len(result["nit_to_group"]),
        len(result["importer_alias"]),
        len(result["supplier_alias"]),
    )
    return result


def _normalize_names(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aplica normalización de nombres de importadores y proveedores.

    Pasos:
      1. Reemplaza variantes conocidas de importador por el nombre canónico.
      2. Si el NIT del importador pertenece a un grupo (comp_principales),
         crea/actualiza la columna 'importer_group'.
      3. Reemplaza variantes conocidas de proveedor por el nombre canónico.
    """
    # --- Importadores: alias por nombre ---
    if "importer" in df.columns and config["importer_alias"]:
        df["importer"] = df["importer"].replace(config["importer_alias"])

    # --- Importadores: agrupación por NIT ---
    if "importer_nit" in df.columns and config["nit_to_group"]:
        nit_str = df["importer_nit"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        df["importer_group"] = nit_str.map(config["nit_to_group"]).fillna("Otros")
    else:
        df["importer_group"] = "Otros"

    # --- Proveedores: alias por nombre ---
    if "supplier" in df.columns and config["supplier_alias"]:
        df["supplier"] = df["supplier"].replace(config["supplier_alias"])

    return df


# -----------------------------------------------------------------------------
# Utilidades de error
# -----------------------------------------------------------------------------
class DataLoadError(Exception):
    """Error controlado al cargar/validar el dataset."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


def resolve_data_path(user_path: str | None) -> Path:
    """Busca el CSV en varias rutas posibles y devuelve la primera que exista."""
    candidates: list[Path] = []
    if user_path:
        candidates.append(Path(user_path).expanduser())
    candidates.extend([DEFAULT_DATA_PATH, FALLBACK_DATA_PATH])

    for c in candidates:
        if c.is_file():
            return c

    tried = "\n".join(f"  • {c}" for c in candidates)
    raise DataLoadError(
        "No se encontró el archivo CSV.",
        hint=(
            "Rutas probadas:\n"
            f"{tried}\n\n"
            "Coloca el CSV en alguna de esas rutas o escribe la ruta "
            "correcta en el campo 'Ruta del CSV' del sidebar."
        ),
    )


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Lee el CSV probando encodings y separadores típicos."""
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(
                    path,
                    low_memory=False,
                    encoding=encoding,
                    sep=sep,
                    on_bad_lines="warn",
                )
                # Si solo hay 1 columna, probablemente el separador está mal
                if df.shape[1] < 2:
                    continue
                logger.info(
                    "CSV leído OK con encoding=%s, sep=%r, shape=%s",
                    encoding, sep, df.shape,
                )
                return df
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}/{sep!r}: {exc.__class__.__name__}")
                continue
            except pd.errors.EmptyDataError as exc:
                raise DataLoadError(
                    "El archivo CSV está vacío.",
                    hint=f"Archivo: {path}",
                ) from exc
            except pd.errors.ParserError as exc:
                errors.append(f"{encoding}/{sep!r}: ParserError")
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{encoding}/{sep!r}: {exc.__class__.__name__}")
                continue

    raise DataLoadError(
        "No se pudo leer el CSV con ningún encoding/separador conocido.",
        hint=(
            "Intentos fallidos:\n"
            + "\n".join(f"  • {e}" for e in errors[:10])
            + "\n\nRevisa que el archivo sea un CSV válido y no esté corrupto."
        ),
    )


def validate_columns(df: pd.DataFrame) -> None:
    """Valida que estén las columnas mínimas."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataLoadError(
            f"Faltan columnas obligatorias en el CSV: {sorted(missing)}",
            hint=(
                "El dataset debe incluir al menos:\n"
                + "\n".join(f"  • {c}" for c in sorted(REQUIRED_COLUMNS))
                + f"\n\nColumnas detectadas ({len(df.columns)}): "
                + ", ".join(df.columns[:15].tolist())
                + ("..." if len(df.columns) > 15 else "")
            ),
        )


# -----------------------------------------------------------------------------
# Carga y limpieza
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando dataset de importaciones...")
def load_data(path_str: str) -> tuple[pd.DataFrame, dict]:
    """
    Carga, valida y limpia el CSV.

    Devuelve:
        - DataFrame listo para usar
        - dict con diagnóstico (rows_raw, rows_clean, dropped_dates, etc.)
    """
    path = Path(path_str)
    diagnostics: dict = {"path": str(path)}

    df = read_csv_safely(path)
    diagnostics["rows_raw"] = len(df)
    diagnostics["cols_raw"] = len(df.columns)

    validate_columns(df)

    # Renombro columnas relevantes a nombres cortos en inglés internos.
    # Mantengo los originales que no se renombran (por si se quieren mostrar).
    rename_map = {
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
    df = df.rename(columns=rename_map)

    # Fecha y año (tolerante a formatos mixtos)
    try:
        df["submission_date"] = pd.to_datetime(
            df["submission_date"], errors="coerce", format="mixed", dayfirst=False,
        )
    except (ValueError, TypeError):
        # Fallback para versiones antiguas de pandas sin format="mixed"
        df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")

    invalid_dates = df["submission_date"].isna().sum()
    diagnostics["invalid_dates"] = int(invalid_dates)
    if invalid_dates == len(df):
        raise DataLoadError(
            "Ninguna fecha pudo ser parseada en 'Fecha de Presentación'.",
            hint="Revisa que la columna tenga fechas en un formato reconocible (YYYY-MM-DD, DD/MM/YYYY, etc.).",
        )

    df["year"] = df["submission_date"].dt.year
    df["month_num"] = df["submission_date"].dt.month
    df["year_month"] = df["submission_date"].dt.to_period("M").dt.to_timestamp()

    # Numéricos: forzar tipo y limpiar (tolerante a strings con comas/puntos)
    numeric_cols = [
        "net_weight_kg", "gross_weight_kg",
        "fob_usd", "cif_usd", "fob_cop", "cif_cop",
        "freight_usd", "insurance_usd", "fx_rate",
        "total_paid_cop", "vat_paid", "tariff_paid",
    ]
    numeric_conversion_issues: dict[str, int] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        before_nulls = df[col].isna().sum()
        # Si es texto, limpio separadores de miles típicos antes de convertir
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^\d\.\-,]", "", regex=True)
                .str.replace(",", "", regex=False)
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = df[col].isna().sum() - before_nulls
        if new_nulls > 0:
            numeric_conversion_issues[col] = int(new_nulls)
    diagnostics["numeric_conversion_issues"] = numeric_conversion_issues

    # Categóricas: rellenar nulos y normalizar strings
    categorical_cols = [
        "customs", "regime", "import_type", "importer", "importer_dept",
        "tariff_description", "tariff_code", "origin_country", "source_country",
        "purchase_country", "origin_continent", "purchase_continent",
        "supplier", "transport_mode", "payment_type", "trade_agreement",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col].fillna("Sin información").astype(str).str.strip().str.title()
            )

    # Clasifico producto (maíz vs trigo vs otro) a partir de la descripción
    def classify_product(desc: str) -> str:
        d = str(desc).lower()
        if "maíz" in d or "maiz" in d or "corn" in d:
            return "Maíz"
        if "trigo" in d or "wheat" in d:
            return "Trigo"
        if "soya" in d or "soja" in d:
            return "Soya"
        return "Otro"

    if "tariff_description" in df.columns:
        df["product"] = df["tariff_description"].apply(classify_product)
    else:
        df["product"] = "Otro"

    # Normalización de nombres (alias + agrupación por NIT)
    name_config = _load_name_config()
    df = _normalize_names(df, name_config)

    # Derivados (con guardas para evitar división por cero / columnas ausentes)
    if "net_weight_kg" in df.columns:
        df["net_weight_ton"] = df["net_weight_kg"] / 1000
    else:
        df["net_weight_ton"] = np.nan

    if "fob_usd" in df.columns:
        df["fob_musd"] = df["fob_usd"] / 1_000_000
    else:
        df["fob_musd"] = np.nan
        df["fob_usd"] = np.nan

    if "cif_usd" in df.columns:
        df["cif_musd"] = df["cif_usd"] / 1_000_000
    else:
        df["cif_musd"] = np.nan
        df["cif_usd"] = np.nan

    # Precio implícito USD/ton (FOB) — evita div/0
    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_usd_per_ton"] = np.where(
            df["net_weight_ton"].fillna(0) > 0,
            df["fob_usd"] / df["net_weight_ton"].replace(0, np.nan),
            np.nan,
        )

    df_clean = df.dropna(subset=["year"]).copy()
    diagnostics["rows_clean"] = len(df_clean)
    diagnostics["rows_dropped"] = len(df) - len(df_clean)

    if df_clean.empty:
        raise DataLoadError(
            "Después de limpiar el dataset no quedó ninguna fila válida.",
            hint=(
                "Todas las filas tenían fecha inválida. "
                "Revisa la columna 'Fecha de Presentación'."
            ),
        )

    return df_clean, diagnostics


# -----------------------------------------------------------------------------
# Sidebar — Configuración de datos (antes que nada)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Datos")
    user_path = st.text_input(
        "Ruta del CSV (opcional)",
        value="",
        placeholder=str(DEFAULT_DATA_PATH),
        help="Déjalo vacío para usar la ruta por defecto.",
    )
    if st.button("🔄 Recargar datos"):
        load_data.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# Carga con manejo global de errores
# -----------------------------------------------------------------------------
try:
    resolved_path = resolve_data_path(user_path or None)
    df, diag = load_data(str(resolved_path))
except DataLoadError as exc:
    st.error(f"❌ Error cargando los datos: {exc}")
    if exc.hint:
        with st.expander("💡 Cómo solucionarlo", expanded=True):
            st.code(exc.hint, language="text")
    st.stop()
except Exception as exc:  # noqa: BLE001
    logger.exception("Fallo inesperado cargando los datos")
    st.error(f"❌ Error inesperado: {exc.__class__.__name__}: {exc}")
    with st.expander("🐞 Traceback completo (para depurar)"):
        st.code(traceback.format_exc(), language="python")
    st.stop()

# Aviso sutil si hubo problemas de parsing
warnings: list[str] = []
if diag.get("invalid_dates", 0) > 0:
    warnings.append(
        f"{diag['invalid_dates']:,} filas con fecha inválida fueron descartadas."
    )
if diag.get("numeric_conversion_issues"):
    issues = diag["numeric_conversion_issues"]
    total_issues = sum(issues.values())
    warnings.append(
        f"{total_issues:,} valores no numéricos fueron convertidos a nulo "
        f"en las columnas: {', '.join(issues.keys())}."
    )
if warnings:
    with st.sidebar.expander("⚠️ Avisos de calidad de datos"):
        for w in warnings:
            st.caption(f"• {w}")

# -----------------------------------------------------------------------------
# Sidebar — Filtros
# -----------------------------------------------------------------------------
st.sidebar.header("🔎 Filtros")

# Guardas para columnas derivadas (por si el dataset viene incompleto)
if df["year"].dropna().empty:
    st.error("❌ No hay años válidos en el dataset después del parseo de fechas.")
    st.stop()

year_min, year_max = int(df["year"].min()), int(df["year"].max())
if year_min == year_max:
    # Si solo hay un año, el slider de rango no aplica
    st.sidebar.info(f"Único año disponible: {year_min}")
    selected_years = [year_min]
else:
    all_years = list(range(year_min, year_max + 1))
    selected_years = st.sidebar.multiselect(
        "Años",
        options=all_years,
        default=all_years,
    )

MONTH_NAMES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}
all_months = list(MONTH_NAMES.values())
selected_month_names = st.sidebar.multiselect(
    "Meses",
    options=all_months,
    default=all_months,
)
# Convertir nombres seleccionados a números para filtrar
NAME_TO_MONTH = {v: k for k, v in MONTH_NAMES.items()}
selected_months = [NAME_TO_MONTH[m] for m in selected_month_names]

available_products = sorted(df["product"].unique())
selected_products = st.sidebar.multiselect(
    "Producto",
    options=available_products,
    default=available_products,
)

available_origins = (
    sorted(df["origin_country"].dropna().unique())
    if "origin_country" in df.columns else []
)
selected_origins = st.sidebar.multiselect(
    "País de origen",
    options=available_origins,
)

available_customs = (
    sorted(df["customs"].dropna().unique())
    if "customs" in df.columns else []
)
selected_customs = st.sidebar.multiselect(
    "Aduana de ingreso",
    options=available_customs,
)

available_transport = (
    sorted(df["transport_mode"].dropna().unique())
    if "transport_mode" in df.columns else []
)
selected_transport = st.sidebar.multiselect(
    "Vía de transporte",
    options=available_transport,
)

available_tariff_codes = (
    sorted(df["tariff_code"].dropna().unique())
    if "tariff_code" in df.columns else []
)

# Creamos un diccionario {código: descripción}
if "tariff_code" in df.columns and "tariff_description" in df.columns:
    # Agrupamos para tener pares únicos
    mapping_df = df[['tariff_code', 'tariff_description']].drop_duplicates().dropna()
    tariff_mapping = dict(zip(mapping_df['tariff_code'], mapping_df['tariff_description']))
    available_tariff_codes = sorted(tariff_mapping.keys())
else:
    tariff_mapping = {}
    available_tariff_codes = []


selected_tariff_codes = st.sidebar.multiselect(
    "Código Partida",
    options=available_tariff_codes,
    default=[],
    format_func=lambda x: f"{x} - {tariff_mapping.get(x, '')}",
    help="Filtra por código de partida arancelaria (ej. 1001991090 para 'Los demás trigos').",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(df):,} registros totales\n\n"
    f"Rango disponible: {year_min} – {year_max}"
)

# -----------------------------------------------------------------------------
# Aplicar filtros (solo aplica los que estén disponibles)
# -----------------------------------------------------------------------------
mask = df["year"].isin(selected_years)
if selected_months and "month_num" in df.columns:
    mask &= df["month_num"].isin(selected_months)
if selected_products and "product" in df.columns:
    mask &= df["product"].isin(selected_products)
if selected_origins and "origin_country" in df.columns:
    mask &= df["origin_country"].isin(selected_origins)
if selected_customs and "customs" in df.columns:
    mask &= df["customs"].isin(selected_customs)
if selected_transport and "transport_mode" in df.columns:
    mask &= df["transport_mode"].isin(selected_transport)
if selected_tariff_codes and "tariff_code" in df.columns:
    mask &= df["tariff_code"].isin(selected_tariff_codes)

fdf = df[mask].copy()

if fdf.empty:
    st.error("No hay registros que coincidan con los filtros seleccionados.")
    st.stop()

# -----------------------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------------------
st.title("🌽 Importaciones de Maíz y Trigo en Colombia")
st.caption(
    "Dashboard interactivo sobre las importaciones colombianas de cereales "
    "(2012 – 2025). Datos consolidados de declaraciones de importación."
)

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
total_declaraciones = len(fdf)
total_toneladas = fdf["net_weight_ton"].sum()
total_fob_musd = fdf["fob_usd"].sum() / 1_000_000
total_cif_musd = fdf["cif_usd"].sum() / 1_000_000
precio_promedio = fdf["price_usd_per_ton"].replace([np.inf, -np.inf], np.nan).mean()
top_origen = (
    fdf.groupby("origin_country")["net_weight_ton"].sum().idxmax()
    if not fdf.empty else "N/A"
)
fob_per_ton_kpi = fdf["fob_usd"].sum() / fdf["net_weight_ton"].sum()
cfr_kpi = (fdf["fob_usd"].sum() + fdf["freight_usd"].sum()) / fdf["net_weight_ton"].sum()

# --- Fila 1 de KPIs ---
row1_1, row1_2, row1_3, row1_4 = st.columns(4)
row1_1.metric("📄 Declaraciones", f"{total_declaraciones:,}")
row1_2.metric("⚖️ Toneladas netas", f"{total_toneladas:,.0f}")
row1_3.metric("💵 FOB total (M USD)", f"{total_fob_musd:,.1f}")
row1_4.metric("💰 CIF total (M USD)", f"{total_cif_musd:,.1f}")

# --- Fila 2 de KPIs ---
row2_1, row2_2, row2_3, row2_4 = st.columns(4)
row2_1.metric("📊 Precio prom. por declaración", f"{precio_promedio:,.1f}")
row2_2.metric("🌍 Top origen", top_origen)
row2_3.metric("📊 FOB ponderado t", f"{fob_per_ton_kpi:,.1f}")
row2_4.metric("🚢 CFR ponderado t", f"{cfr_kpi:,.1f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Pestañas de análisis
# -----------------------------------------------------------------------------
tab_tendencias, tab_geo, tab_actores, tab_precios, tab_logistica, tab_data = st.tabs(
    ["📈 Tendencias", "🌎 Geografía", "🏢 Actores", "💲 Precios", "🚢 Logística", "🗂️ Datos"]
)

# ---------------------------- Tendencias -------------------------------------
with tab_tendencias:
    c1, c2 = st.columns(2)

    with c1:
        yearly = (
            fdf.groupby(["year", "product"], as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"), fob_usd=("fob_usd", "sum"))
        )
        fig = px.line(
            yearly, x="year", y="toneladas", color="product",
            markers=True, template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
            title="Toneladas importadas por año y producto",
        )
        fig.update_layout(yaxis_title="Toneladas netas", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            yearly, x="year", y="fob_usd", color="product", barmode="group",
            template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
            title="Valor FOB (USD) importado por año y producto",
        )
        fig.update_layout(yaxis_title="USD FOB", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Estacionalidad mensual
    monthly = (
        fdf.groupby(["month_num", "product"], as_index=False)
        .agg(toneladas=("net_weight_ton", "sum"))
    )
    month_names = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
    }
    monthly["mes"] = monthly["month_num"].map(month_names)

    fig = px.bar(
        monthly.sort_values("month_num"), x="mes", y="toneladas",
        color="product", barmode="group",
        template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
        title="Estacionalidad: toneladas por mes (acumulado todos los años)",
    )
    fig.update_layout(yaxis_title="Toneladas netas", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # Serie mensual continua
    monthly_ts = (
        fdf.groupby(["year_month", "product"], as_index=False)
        .agg(toneladas=("net_weight_ton", "sum"))
    )
    fig = px.area(
        monthly_ts, x="year_month", y="toneladas", color="product",
        template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
        title="Evolución mensual de toneladas importadas",
    )
    fig.update_layout(yaxis_title="Toneladas netas", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Geografía --------------------------------------
with tab_geo:
    c1, c2 = st.columns(2)

    with c1:
        origin = (
            fdf.groupby("origin_country", as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"))
            .sort_values("toneladas", ascending=False)
            .head(10)
        )
        fig = px.bar(
            origin, x="toneladas", y="origin_country", orientation="h",
            template=TEMPLATE, color="origin_country",
            color_discrete_sequence=COLOR_SEQ,
            title="Top 10 países de origen por toneladas",
        )
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Toneladas")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        continent = (
            fdf.groupby("origin_continent", as_index=False)
            .agg(fob_usd=("fob_usd", "sum"))
            .sort_values("fob_usd", ascending=False)
        )
        fig = px.pie(
            continent, values="fob_usd", names="origin_continent",
            template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
            title="Participación FOB por continente de origen",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Mapa choropleth del mundo
    country_agg = (
        fdf.groupby("origin_country", as_index=False)
        .agg(toneladas=("net_weight_ton", "sum"), fob_usd=("fob_usd", "sum"))
    )
    fig = px.choropleth(
        country_agg,
        locations="origin_country",
        locationmode="country names",
        color="toneladas",
        hover_data={"toneladas": ":,.0f", "fob_usd": ":,.0f"},
        color_continuous_scale="YlGnBu",
        template=TEMPLATE,
        title="Mapa de importaciones por país de origen (toneladas)",
    )
    fig.update_geos(showcountries=True, showcoastlines=True, projection_type="natural earth")
    st.plotly_chart(fig, use_container_width=True)

    # Aduanas y departamentos destino
    c3, c4 = st.columns(2)
    with c3:
        customs_agg = (
            fdf.groupby("customs", as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"))
            .sort_values("toneladas", ascending=False)
            .head(10)
        )
        fig = px.bar(
            customs_agg, x="customs", y="toneladas",
            template=TEMPLATE, color="customs",
            color_discrete_sequence=COLOR_SEQ,
            title="Aduanas por volumen importado (top 10)",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Toneladas")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        if "importer_dept" in fdf.columns:
            dept_agg = (
                fdf.groupby("importer_dept", as_index=False)
                .agg(toneladas=("net_weight_ton", "sum"))
                .sort_values("toneladas", ascending=False)
                .head(10)
            )
            fig = px.bar(
                dept_agg, x="importer_dept", y="toneladas",
                template=TEMPLATE, color="importer_dept",
                color_discrete_sequence=COLOR_SEQ,
                title="Top 10 departamentos del importador",
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Toneladas")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Actores ----------------------------------------
with tab_actores:
    c1, c2 = st.columns(2)

    with c1:
        importer_agg = (
            fdf.groupby("importer", as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"), fob_usd=("fob_usd", "sum"))
            .sort_values("toneladas", ascending=False)
            .head(15)
        )
        fig = px.bar(
            importer_agg, x="toneladas", y="importer", orientation="h",
            template=TEMPLATE, color="importer",
            color_discrete_sequence=COLOR_SEQ,
            title="Top 15 importadores por toneladas",
            hover_data={"fob_usd": ":,.0f"},
        )
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Toneladas")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        supplier_agg = (
            fdf.groupby("supplier", as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"))
            .sort_values("toneladas", ascending=False)
            .head(15)
        )
        fig = px.bar(
            supplier_agg, x="toneladas", y="supplier", orientation="h",
            template=TEMPLATE, color="supplier",
            color_discrete_sequence=COLOR_SEQ,
            title="Top 15 proveedores internacionales",
        )
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Toneladas")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    # Concentración (HHI simplificado por tonelaje)
    st.subheader("Concentración del mercado importador")
    imp_share = (
        fdf.groupby("importer")["net_weight_ton"].sum().sort_values(ascending=False)
    )
    total_ton = imp_share.sum()
    if total_ton > 0:
        shares = imp_share / total_ton
        hhi = (shares ** 2).sum() * 10000  # escala 0-10000
        top5_share = shares.head(5).sum() * 100
        top10_share = shares.head(10).sum() * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("HHI (índice)", f"{hhi:,.0f}", help="0-10000. >2500 = mercado muy concentrado")
        m2.metric("% Top 5 importadores", f"{top5_share:.1f}%")
        m3.metric("% Top 10 importadores", f"{top10_share:.1f}%")

# ---------------------------- Precios ----------------------------------------
with tab_precios:
    price_df = fdf[(fdf["price_usd_per_ton"].notna())
                   & (fdf["price_usd_per_ton"].between(50, 2000))]  # filtro de outliers

    c1, c2 = st.columns(2)

    with c1:
        price_yearly = (
            price_df.groupby(["year", "product"], as_index=False)
            .agg(precio_prom=("price_usd_per_ton", "mean"))
        )
        fig = px.line(
            price_yearly, x="year", y="precio_prom", color="product",
            markers=True, template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
            title="Precio promedio USD/tonelada por año",
        )
        fig.update_layout(yaxis_title="USD / tonelada", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            price_df, x="product", y="price_usd_per_ton", color="product",
            template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
            title="Distribución de precios por producto",
            points="outliers",
        )
        fig.update_layout(showlegend=False, yaxis_title="USD / tonelada", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Precio por país de origen (top 10)
    top_origins_list = (
        price_df.groupby("origin_country")["net_weight_ton"].sum()
        .nlargest(10).index.tolist()
    )
    price_by_origin = price_df[price_df["origin_country"].isin(top_origins_list)]
    fig = px.box(
        price_by_origin, x="origin_country", y="price_usd_per_ton",
        color="origin_country",
        template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
        title="Distribución de precios por país de origen (top 10 en volumen)",
    )
    fig.update_layout(showlegend=False, yaxis_title="USD / tonelada", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # Relación flete / valor FOB
    if "freight_usd" in fdf.columns:
        scatter_df = fdf[(fdf["fob_usd"] > 0) & (fdf["freight_usd"] > 0)].sample(
            n=min(5000, len(fdf)), random_state=42,
        )
        fig = px.scatter(
            scatter_df, x="fob_usd", y="freight_usd",
            color="product", size="net_weight_ton",
            template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
            title="Relación entre Valor FOB y Flete (USD)",
            hover_data=["origin_country", "customs"],
            opacity=0.6,
        )
        fig.update_layout(xaxis_title="FOB (USD)", yaxis_title="Flete (USD)")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Logística --------------------------------------
with tab_logistica:
    c1, c2 = st.columns(2)

    with c1:
        transport_agg = (
            fdf.groupby("transport_mode", as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"))
            .sort_values("toneladas", ascending=False)
        )
        fig = px.pie(
            transport_agg, values="toneladas", names="transport_mode",
            template=TEMPLATE, color_discrete_sequence=COLOR_SEQ,
            title="Distribución por vía de transporte",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        regime_agg = (
            fdf.groupby("regime", as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"))
            .sort_values("toneladas", ascending=False)
            .head(8)
        )
        fig = px.bar(
            regime_agg, x="regime", y="toneladas",
            template=TEMPLATE, color="regime",
            color_discrete_sequence=COLOR_SEQ,
            title="Régimen de importación",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Toneladas")
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: aduana vs producto
    pivot = (
        fdf.pivot_table(
            index="customs", columns="product",
            values="net_weight_ton", aggfunc="sum", fill_value=0,
        )
        .sort_values(by=fdf["product"].unique()[0] if len(fdf["product"].unique()) > 0 else "product",
                     ascending=False)
        .head(15)
    )
    fig = px.imshow(
        pivot, text_auto=".2s", aspect="auto",
        color_continuous_scale="YlOrRd",
        template=TEMPLATE,
        title="Heatmap: toneladas por aduana y producto (top 15 aduanas)",
    )
    fig.update_layout(xaxis_title="Producto", yaxis_title="Aduana")
    st.plotly_chart(fig, use_container_width=True)

    # Acuerdos comerciales
    if "trade_agreement" in fdf.columns:
        agree_agg = (
            fdf.groupby("trade_agreement", as_index=False)
            .agg(toneladas=("net_weight_ton", "sum"), fob_usd=("fob_usd", "sum"))
            .sort_values("toneladas", ascending=False)
            .head(10)
        )
        fig = px.bar(
            agree_agg, x="trade_agreement", y="toneladas",
            template=TEMPLATE, color="trade_agreement",
            color_discrete_sequence=COLOR_SEQ,
            title="Top 10 acuerdos de tratamiento arancelario por volumen",
            hover_data={"fob_usd": ":,.0f"},
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Toneladas")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Datos crudos -----------------------------------
with tab_data:
    st.subheader("Explorador de registros filtrados")
    st.caption(f"Mostrando {len(fdf):,} registros con los filtros actuales.")

    show_cols = [c for c in [
        "submission_date", "product", "importer", "supplier",
        "origin_country", "customs", "transport_mode",
        "net_weight_ton", "fob_usd", "cif_usd", "price_usd_per_ton",
    ] if c in fdf.columns]

    st.dataframe(
        fdf[show_cols].sort_values("submission_date", ascending=False).head(500),
        use_container_width=True,
        height=420,
    )

    csv = fdf[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV filtrado",
        data=csv,
        file_name="importaciones_filtrado.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Footer / Insights
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📝 Qué buscar en este dashboard")
st.markdown(
    """
- **Tendencias**: detecta años con picos o caídas abruptas (pueden coincidir con shocks de precios o climáticos).
- **Estacionalidad**: los cereales suelen seguir patrones de cosecha en el hemisferio norte/sur.
- **Geografía**: ¿qué tan dependiente es Colombia de un solo país de origen? Revisa la concentración en el mapa.
- **Actores**: el HHI y el top 5/10 cuentan si el mercado importador está concentrado en pocos jugadores.
- **Precios**: compara precios implícitos (USD/ton) entre orígenes para identificar diferenciales sistemáticos.
- **Logística**: la vía de transporte y la aduana dominante reflejan decisiones de costo y cercanía al consumo.
    """
)