"""
Ingesta de nuevos datos al CSV maestro.

Lógica pura, sin dependencias de UI. Recibe DataFrames y rutas, devuelve
resultados estructurados (IngestionPlan, IngestionResult). La capa de UI
(Streamlit, Tauri, CLI, etc.) consume estas funciones.

Flujo típico:
    1. read_excel_sheet(file, "Detalle")        -> DataFrame crudo
    2. plan_ingestion(df_new, csv_path)         -> IngestionPlan (preview)
    3. apply_ingestion(plan, csv_path)          -> IngestionResult (escribe)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constantes de ingesta
# -----------------------------------------------------------------------------

# Columnas que el Excel descargado de fuentes externas suele traer y que NO
# pertenecen al esquema maestro. Se descartan silenciosamente.
COLS_AUTO_DROP = ("fila", "Año")

# Llave natural para detectar duplicados entre cargas.
DEDUP_KEY = "Número de declaración (llave)"

# Columnas mínimas que el Excel DEBE traer para considerar la carga válida.
# Debe coincidir con REQUIRED_COLUMNS de loader.py.
REQUIRED_INGESTION_COLUMNS = {
    "Fecha de Presentación",
    "Descripción de la partida arancelaria",
    "Peso en kilos netos",
    "Valor FOB (USD)",
    "País de origen",
}


# -----------------------------------------------------------------------------
# Errores
# -----------------------------------------------------------------------------

class IngestionError(Exception):
    """Error controlado durante el proceso de ingesta."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


# -----------------------------------------------------------------------------
# Modelos de resultado
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestionPlan:
    """Plan previo a la escritura. Permite a la UI mostrar un preview."""

    df_to_append: pd.DataFrame              # filas listas para concatenar
    rows_in_excel: int                       # filas originales del Excel
    rows_to_append: int                      # filas que efectivamente se sumarán
    cols_dropped_auto: list[str] = field(default_factory=list)
    cols_dropped_unknown: list[str] = field(default_factory=list)
    cols_filled_with_na: list[str] = field(default_factory=list)
    duplicates_within_excel: int = 0
    duplicates_vs_base: int = 0

    @property
    def has_changes(self) -> bool:
        return self.rows_to_append > 0

    def summary_lines(self) -> list[str]:
        """Líneas legibles para mostrar en UI."""
        lines = []
        if self.cols_dropped_auto:
            lines.append(
                f"🗑️ Columnas eliminadas automáticamente: "
                f"{', '.join(self.cols_dropped_auto)}"
            )
        if self.cols_dropped_unknown:
            preview = ", ".join(self.cols_dropped_unknown[:5])
            extra = (
                f" y {len(self.cols_dropped_unknown) - 5} más"
                if len(self.cols_dropped_unknown) > 5 else ""
            )
            lines.append(f"⚠️ Columnas descartadas (no están en la base): {preview}{extra}")
        if self.cols_filled_with_na:
            lines.append(
                f"➕ Columnas rellenadas con NaN (faltaban en el Excel): "
                f"{len(self.cols_filled_with_na)} columnas"
            )
        if self.duplicates_within_excel:
            lines.append(
                f"♻️ Duplicados dentro del Excel: {self.duplicates_within_excel:,}"
            )
        if self.duplicates_vs_base:
            lines.append(
                f"⏭️ Ya existían en la base: {self.duplicates_vs_base:,}"
            )
        return lines


@dataclass(frozen=True)
class IngestionResult:
    """Resultado de una escritura efectiva al CSV maestro."""

    rows_appended: int
    total_rows_after: int
    backup_path: Path
    csv_path: Path


# -----------------------------------------------------------------------------
# Funciones de ingesta
# -----------------------------------------------------------------------------

def read_excel_sheet(file_or_path, sheet_name: str = "Detalle") -> pd.DataFrame:
    """Lee una hoja específica de un Excel.

    Acepta tanto un Path como un objeto file-like (UploadedFile de Streamlit).
    Lanza IngestionError con mensajes claros si algo falla.
    """
    try:
        df = pd.read_excel(file_or_path, sheet_name=sheet_name)
    except ValueError as exc:
        # Pandas lanza ValueError cuando la hoja no existe
        raise IngestionError(
            f"No se encontró la hoja '{sheet_name}' en el archivo.",
            hint="Verifica que el Excel tenga una hoja llamada exactamente "
                 f"'{sheet_name}' (con esa misma capitalización).",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(
            f"No se pudo leer el Excel: {exc.__class__.__name__}",
            hint="Revisa que el archivo no esté corrupto o protegido.",
        ) from exc

    if df.empty:
        raise IngestionError(
            "La hoja está vacía.",
            hint=f"La hoja '{sheet_name}' no tiene datos.",
        )

    logger.info(
        "Excel leído OK: hoja=%r, filas=%d, columnas=%d",
        sheet_name, len(df), len(df.columns),
    )
    return df


def _drop_auto_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Elimina columnas que sabemos que sobran (fila, Año, etc.)."""
    cols_to_drop = [c for c in COLS_AUTO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Verifica que el DataFrame nuevo tenga las columnas mínimas."""
    missing = REQUIRED_INGESTION_COLUMNS - set(df.columns)
    if missing:
        raise IngestionError(
            f"Faltan columnas obligatorias en el Excel: {sorted(missing)}",
            hint=(
                "El Excel debe incluir al menos:\n"
                + "\n".join(f"  • {c}" for c in sorted(REQUIRED_INGESTION_COLUMNS))
            ),
        )


def _align_to_base_schema(
    df_new: pd.DataFrame, base_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Alinea las columnas del nuevo DataFrame al esquema de la base.

    - Descarta columnas que existen en el nuevo pero no en la base.
    - Rellena con NaN columnas que faltan en el nuevo.
    - Devuelve el DataFrame con el mismo orden de columnas que la base.
    """
    base_set = set(base_columns)
    new_set = set(df_new.columns)

    cols_dropped = sorted(new_set - base_set)
    cols_filled = sorted(base_set - new_set)

    df_aligned = df_new.copy()

    if cols_dropped:
        df_aligned = df_aligned.drop(columns=cols_dropped)

    for col in cols_filled:
        df_aligned[col] = pd.NA

    df_aligned = df_aligned[base_columns]  # mismo orden que la base

    return df_aligned, cols_dropped, cols_filled


def _deduplicate(
    df_new: pd.DataFrame, df_base: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Elimina duplicados internos del Excel y filas que ya existen en la base.

    Devuelve (df_filtrado, n_dup_internos, n_dup_vs_base).
    Si la columna llave no existe en alguno de los dos, no deduplica.
    """
    if DEDUP_KEY not in df_new.columns or DEDUP_KEY not in df_base.columns:
        logger.warning(
            "Columna llave %r no encontrada — no se aplicará deduplicación.",
            DEDUP_KEY,
        )
        return df_new, 0, 0

    # 1) Duplicados internos del Excel
    n_internal = int(df_new.duplicated(subset=[DEDUP_KEY]).sum())
    df_new = df_new.drop_duplicates(subset=[DEDUP_KEY], keep="first")

    # 2) Duplicados contra la base existente
    existing_keys = set(df_base[DEDUP_KEY].dropna().astype(str))
    new_keys_str = df_new[DEDUP_KEY].astype(str)
    mask_truly_new = ~new_keys_str.isin(existing_keys)
    n_vs_base = int((~mask_truly_new).sum())
    df_new = df_new[mask_truly_new]

    return df_new, n_internal, n_vs_base


def plan_ingestion(
    df_new: pd.DataFrame, csv_path: Path,
) -> IngestionPlan:
    """Construye un plan de ingesta sin escribir nada.

    Pasos:
        1. Quita columnas auto-descartables (fila, Año).
        2. Valida columnas mínimas.
        3. Lee la base existente para alinear esquema.
        4. Deduplica internamente y contra la base.
        5. Devuelve un IngestionPlan listo para inspeccionar o aplicar.
    """
    if not csv_path.is_file():
        raise IngestionError(
            f"No se encontró la base existente en {csv_path}.",
            hint="Verifica la ruta del CSV maestro antes de cargar nuevos datos.",
        )

    rows_in_excel = len(df_new)

    # 1) Drop columnas extra
    df_new, cols_dropped_auto = _drop_auto_columns(df_new)

    # 2) Validar mínimos
    _validate_required_columns(df_new)

    # 3) Cargar base y alinear
    try:
        df_base = pd.read_csv(csv_path, low_memory=False, nrows=0)  # solo headers
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(
            f"No se pudo abrir la base existente: {exc.__class__.__name__}",
            hint=f"Archivo: {csv_path}",
        ) from exc

    base_columns = list(df_base.columns)
    df_aligned, cols_unknown, cols_filled = _align_to_base_schema(df_new, base_columns)

    # 4) Deduplicar (necesita las llaves de la base completa)
    try:
        df_base_keys = pd.read_csv(
            csv_path, low_memory=False, usecols=[DEDUP_KEY],
        ) if DEDUP_KEY in base_columns else pd.DataFrame(columns=base_columns)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(
            f"No se pudieron leer las llaves de la base: {exc.__class__.__name__}",
        ) from exc

    df_aligned, n_dup_internal, n_dup_vs_base = _deduplicate(df_aligned, df_base_keys)

    return IngestionPlan(
        df_to_append=df_aligned,
        rows_in_excel=rows_in_excel,
        rows_to_append=len(df_aligned),
        cols_dropped_auto=cols_dropped_auto,
        cols_dropped_unknown=cols_unknown,
        cols_filled_with_na=cols_filled,
        duplicates_within_excel=n_dup_internal,
        duplicates_vs_base=n_dup_vs_base,
    )


def _make_backup(csv_path: Path) -> Path:
    """Crea una copia de seguridad con timestamp del CSV maestro."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = csv_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"{csv_path.stem}.backup_{timestamp}.csv"

    # Copiamos por streaming en lugar de cargar todo a memoria
    with open(csv_path, "rb") as src, open(backup_path, "wb") as dst:
        while chunk := src.read(1024 * 1024):  # 1 MB
            dst.write(chunk)

    logger.info("Backup creado en %s", backup_path)
    return backup_path


def apply_ingestion(plan: IngestionPlan, csv_path: Path) -> IngestionResult:
    """Aplica el plan: crea backup, concatena y guarda.

    Retorna un IngestionResult con la información de la escritura.
    Lanza IngestionError si algo falla (el backup queda como recuperación).
    """
    if not plan.has_changes:
        raise IngestionError(
            "El plan no tiene filas nuevas para agregar.",
            hint="Revisa el resumen del plan antes de aplicarlo.",
        )

    # 1) Backup
    backup_path = _make_backup(csv_path)

    # 2) Append en modo streaming (sin cargar toda la base a memoria)
    try:
        plan.df_to_append.to_csv(
            csv_path,
            mode="a",          # append
            header=False,      # no escribir headers de nuevo
            index=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(
            f"Error al escribir el CSV: {exc.__class__.__name__}",
            hint=f"El backup quedó disponible en {backup_path}",
        ) from exc

    # 3) Contar filas finales (sin cargar todo a memoria)
    with open(csv_path, encoding="utf-8") as fh:
        total_rows = sum(1 for _ in fh) - 1  # menos el header

    logger.info(
        "Ingesta aplicada: %d filas agregadas, total ahora %d",
        plan.rows_to_append, total_rows,
    )

    return IngestionResult(
        rows_appended=plan.rows_to_append,
        total_rows_after=total_rows,
        backup_path=backup_path,
        csv_path=csv_path,
    )