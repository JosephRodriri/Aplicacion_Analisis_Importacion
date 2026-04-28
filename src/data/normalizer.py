"""Normalización de nombres de importadores y proveedores."""
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_name_config(config_path: Path) -> dict:
    """Carga el JSON con aliases y agrupaciones."""
    result: dict = {"nit_to_group": {}, "importer_alias": {}, "supplier_alias": {}}
    if not config_path.is_file():
        logger.warning("No se encontró %s", config_path)
        return result

    try:
        with open(config_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error leyendo %s: %s", config_path, exc)
        return result

    for group_name, nit_list in raw.get("comp_principales", {}).items():
        for nit in nit_list:
            result["nit_to_group"][str(nit).strip()] = group_name

    for canonical, variants in raw.get("alias_importadores", {}).items():
        canonical_title = canonical.strip().title()
        for v in variants:
            result["importer_alias"][v.strip().title()] = canonical_title

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


def apply_name_normalization(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aplica aliases y agrupación por NIT."""
    if "importer" in df.columns and config["importer_alias"]:
        df["importer"] = df["importer"].replace(config["importer_alias"])

    if "importer_nit" in df.columns and config["nit_to_group"]:
        nit_str = (
            df["importer_nit"].astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        df["importer_group"] = nit_str.map(config["nit_to_group"]).fillna("Otros")
    else:
        df["importer_group"] = "Otros"

    if "supplier" in df.columns and config["supplier_alias"]:
        df["supplier"] = df["supplier"].replace(config["supplier_alias"])

    return df
