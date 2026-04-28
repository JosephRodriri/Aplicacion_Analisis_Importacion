"""Modelos de datos del dominio."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ImportKPIs:
    """KPIs principales del dashboard."""
    declaraciones: int
    toneladas: float
    fob_total_musd: float
    cif_total_musd: float
    fob_per_ton: float          # ponderado
    cfr_per_ton: float          # ponderado
    precio_promedio_simple: float  # promedio simple por declaración
    top_origen: str
    top_origen_pct: float       # % share del top origen


@dataclass(frozen=True)
class RiskMetrics:
    """Métricas de riesgo para el equipo de compras."""
    hhi_origenes: float
    hhi_proveedores: float
    n_origenes: int
    n_proveedores: int
    top5_origenes_pct: float
    top10_importadores_pct: float
    precio_cv_pct: float        # coeficiente de variación
    delta_precio_3m_vs_12m_pct: float
    pct_logistico_sobre_fob: float
    pct_bajo_acuerdo: float
