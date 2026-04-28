"""Carga y validación del dataset de importaciones."""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Fecha de Presentación",
    "Descripción de la partida arancelaria",
    "Peso en kilos netos",
    "Valor FOB (USD)",
    "País de origen",
}


class DataLoadError(Exception):
    """Error controlado al cargar/validar el dataset."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


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
            except pd.errors.ParserError:
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


def resolve_data_path(
    user_path: str | None,
    default_path: Path,
    fallback_path: Path,
) -> Path:
    """Busca el CSV en varias rutas posibles y devuelve la primera que exista."""
    candidates: list[Path] = []
    if user_path:
        candidates.append(Path(user_path).expanduser())
    candidates.extend([default_path, fallback_path])

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
