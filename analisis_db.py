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

import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui.ingestion_panel import render_ingestion_panel

from src.data.cleaner import RENAME_MAP, parse_dates, parse_numerics, normalize_categoricals, add_derived_columns
from src.data.loader import DataLoadError, read_csv_safely, validate_columns, resolve_data_path
from src.data.normalizer import load_name_config, apply_name_normalization
from src.domain.classifiers import load_product_rules, classify_product
from src.domain.kpis import calculate_kpis
from src.domain.risk_metrics import calculate_risk_metrics

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
CONFIG_JSON_PATH = Path(__file__).resolve().parent / "src" / "config" / "comp_principales.json"

# Ruta al JSON de reglas de clasificación de productos.
PRODUCTS_CONFIG_PATH = Path(__file__).resolve().parent / "src" / "config" / "products.json"

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
# Carga y limpieza
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando dataset de importaciones...")
def load_data(path_str: str) -> tuple[pd.DataFrame, dict]:
    """Carga, valida y limpia el CSV.

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

    df = df.rename(columns=RENAME_MAP)
    df, invalid_dates = parse_dates(df)
    diagnostics["invalid_dates"] = invalid_dates

    if invalid_dates == len(df):
        raise DataLoadError(
            "Ninguna fecha pudo ser parseada en 'Fecha de Presentación'.",
            hint="Revisa que la columna tenga fechas en un formato reconocible (YYYY-MM-DD, DD/MM/YYYY, etc.).",
        )

    df, numeric_issues = parse_numerics(df)
    diagnostics["numeric_conversion_issues"] = numeric_issues

    df = normalize_categoricals(df)

    # Clasificación de productos
    product_rules = load_product_rules(PRODUCTS_CONFIG_PATH)
    if "tariff_description" in df.columns:
        df["product"] = df["tariff_description"].apply(
            lambda d: classify_product(d, product_rules)
        )
    else:
        df["product"] = "Otro"

    # Normalización de nombres
    name_config = load_name_config(CONFIG_JSON_PATH)
    df = apply_name_normalization(df, name_config)

    df = add_derived_columns(df)

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

with st.sidebar:
    render_ingestion_panel(
        csv_path=DEFAULT_DATA_PATH,
        on_success_callback=load_data.clear,
    )


# -----------------------------------------------------------------------------
# Carga con manejo global de errores
# -----------------------------------------------------------------------------
try:
    resolved_path = resolve_data_path(user_path or None, DEFAULT_DATA_PATH, FALLBACK_DATA_PATH)
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
kpis = calculate_kpis(fdf)
risk = calculate_risk_metrics(fdf)

# --- Fila 1: Volumen y valor ---
row1_1, row1_2, row1_3, row1_4 = st.columns(4)
row1_1.metric("📄 Declaraciones", f"{kpis.declaraciones:,}")
row1_2.metric("⚖️ Toneladas netas", f"{kpis.toneladas:,.0f}")
row1_3.metric("💵 FOB total (M USD)", f"{kpis.fob_total_musd:,.1f}")
row1_4.metric("💰 CIF total (M USD)", f"{kpis.cif_total_musd:,.1f}")

# --- Fila 2: Precios ---
row2_1, row2_2, row2_3, row2_4 = st.columns(4)
row2_1.metric("📊 FOB ponderado/t", f"{kpis.fob_per_ton:,.1f}")
row2_2.metric("🚢 CFR ponderado/t", f"{kpis.cfr_per_ton:,.1f}")
row2_3.metric("📈 Δ precio 3m vs 12m", f"{risk.delta_precio_3m_vs_12m_pct:+.1f}%")
row2_4.metric("📉 Volatilidad (CV)", f"{risk.precio_cv_pct:.1f}%")

# --- Fila 3: Riesgo ---
row3_1, row3_2, row3_3, row3_4 = st.columns(4)
row3_1.metric(
    "🌍 Top origen",
    kpis.top_origen,
    f"{kpis.top_origen_pct:.1f}% del volumen",
)
row3_2.metric(
    "🎯 HHI orígenes",
    f"{risk.hhi_origenes:,.0f}",
    help="0-10000. >2500 = altamente concentrado.",
)
row3_3.metric("🚢 % logístico/FOB", f"{risk.pct_logistico_sobre_fob:.1f}%")
row3_4.metric("📜 % con acuerdo", f"{risk.pct_bajo_acuerdo:.1f}%")

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