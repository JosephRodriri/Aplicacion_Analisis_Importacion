"""
Nombres canónicos de columnas usadas en todo el proyecto.
Importar desde aquí en lugar de usar strings literales.
"""

# --- Identificación / metadata ---
COL_SUBMISSION_DATE = "submission_date"
COL_YEAR = "year"
COL_MONTH_NUM = "month_num"
COL_YEAR_MONTH = "year_month"
COL_MONTH = "month"

# --- Aduana / logística ---
COL_CUSTOMS = "customs"
COL_REGIME = "regime"
COL_IMPORT_TYPE = "import_type"
COL_TRANSPORT_MODE = "transport_mode"
COL_TRADE_AGREEMENT = "trade_agreement"

# --- Importador ---
COL_IMPORTER = "importer"
COL_IMPORTER_NIT = "importer_nit"
COL_IMPORTER_DEPT = "importer_dept"
COL_IMPORTER_GROUP = "importer_group"

# --- Producto / partida ---
COL_TARIFF_DESCRIPTION = "tariff_description"
COL_TARIFF_CODE = "tariff_code"
COL_PRODUCT = "product"

# --- Geografía ---
COL_ORIGIN_COUNTRY = "origin_country"
COL_SOURCE_COUNTRY = "source_country"
COL_PURCHASE_COUNTRY = "purchase_country"
COL_ORIGIN_CONTINENT = "origin_continent"
COL_PURCHASE_CONTINENT = "purchase_continent"
COL_SUPPLIER = "supplier"

# --- Pesos ---
COL_NET_WEIGHT_KG = "net_weight_kg"
COL_GROSS_WEIGHT_KG = "gross_weight_kg"
COL_NET_WEIGHT_TON = "net_weight_ton"

# --- Valores monetarios USD ---
COL_FOB_USD = "fob_usd"
COL_CIF_USD = "cif_usd"
COL_FOB_MUSD = "fob_musd"
COL_CIF_MUSD = "cif_musd"
COL_FREIGHT_USD = "freight_usd"
COL_INSURANCE_USD = "insurance_usd"

# --- Valores monetarios COP ---
COL_FOB_COP = "fob_cop"
COL_CIF_COP = "cif_cop"
COL_TOTAL_PAID_COP = "total_paid_cop"
COL_VAT_PAID = "vat_paid"
COL_TARIFF_PAID = "tariff_paid"

# --- Otros ---
COL_FX_RATE = "fx_rate"
COL_PAYMENT_TYPE = "payment_type"
COL_PRICE_USD_PER_TON = "price_usd_per_ton"

# --- Productos ---
PRODUCT_MAIZ = "Maíz"
PRODUCT_TRIGO = "Trigo"
PRODUCT_SOYA = "Soya"
PRODUCT_OTRO = "Otro"

# --- Constantes de negocio ---
PRICE_OUTLIER_MIN = 50      # USD/ton, filtro para análisis de precios
PRICE_OUTLIER_MAX = 2000    # USD/ton
HHI_THRESHOLD_HIGH = 2500   # > este valor = mercado muy concentrado
HHI_THRESHOLD_MODERATE = 1500
