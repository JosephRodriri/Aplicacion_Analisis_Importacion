"""
Componente de UI para ingesta de nuevos datos.

Encapsula la interacción con Streamlit. La lógica vive en
src/data/ingestion.py — este archivo solo orquesta el flujo visual.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data.ingestion import (
    IngestionError,
    apply_ingestion,
    plan_ingestion,
    read_excel_sheet,
)


def render_ingestion_panel(csv_path: Path, on_success_callback=None) -> None:
    """Renderiza el panel de ingesta dentro del sidebar.

    Args:
        csv_path: Ruta al CSV maestro donde se agregarán los datos.
        on_success_callback: Opcional. Función a ejecutar después de una
            ingesta exitosa (típicamente, invalidar caches).
    """
    st.header("➕ Agregar datos a la base")

    uploaded_file = st.file_uploader(
        "Subir archivo Excel (hoja 'Detalle')",
        type=["xlsx"],
        help=(
            "Se descartan automáticamente las columnas 'fila' y 'Año'. "
            "Las filas duplicadas (mismo número de declaración) se ignoran."
        ),
        key="ingestion_uploader",
    )

    if uploaded_file is None:
        return

    # 1) Leer Excel
    try:
        df_new = read_excel_sheet(uploaded_file, sheet_name="Detalle")
    except IngestionError as exc:
        st.error(f"❌ {exc}")
        if exc.hint:
            st.caption(exc.hint)
        return

    # 2) Construir plan
    try:
        plan = plan_ingestion(df_new, csv_path)
    except IngestionError as exc:
        st.error(f"❌ {exc}")
        if exc.hint:
            with st.expander("💡 Cómo solucionarlo", expanded=True):
                st.code(exc.hint, language="text")
        return

    # 3) Mostrar resumen
    st.markdown("### 📋 Resumen de la carga")

    c1, c2 = st.columns(2)
    c1.metric("Filas en Excel", f"{plan.rows_in_excel:,}")
    c2.metric(
        "Filas nuevas",
        f"{plan.rows_to_append:,}",
        delta=(
            f"-{plan.rows_in_excel - plan.rows_to_append:,} filtradas"
            if plan.rows_in_excel != plan.rows_to_append else None
        ),
        delta_color="off",
    )

    avisos = plan.summary_lines()
    if avisos:
        with st.expander("Ver detalles del procesamiento", expanded=False):
            for aviso in avisos:
                st.caption(aviso)

    # 4) Preview
    if plan.has_changes:
        st.caption("Vista previa (primeras 10 filas a agregar):")
        st.dataframe(
            plan.df_to_append.head(10),
            use_container_width=True,
            height=200,
        )

    # 5) Botón de confirmación
    if not plan.has_changes:
        st.info("ℹ️ No hay filas nuevas para agregar.")
        return

    confirm = st.button(
        f"✅ Confirmar y agregar {plan.rows_to_append:,} filas",
        type="primary",
        key="ingestion_confirm",
        use_container_width=True,
    )

    if not confirm:
        return

    # 6) Aplicar ingesta
    with st.spinner("Guardando datos..."):
        try:
            result = apply_ingestion(plan, csv_path)
        except IngestionError as exc:
            st.error(f"❌ {exc}")
            if exc.hint:
                st.caption(exc.hint)
            return

    st.success(
        f"✅ Se agregaron **{result.rows_appended:,} filas**. "
        f"Total ahora: **{result.total_rows_after:,}**."
    )
    st.caption(f"💾 Backup: `{result.backup_path.name}`")

    # 7) Invalidar caches y recargar
    if on_success_callback is not None:
        on_success_callback()

    st.info("Recargando dashboard...")
    st.rerun()