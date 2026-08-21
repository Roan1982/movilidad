# -*- coding: utf-8 -*-
"""
30 - VALIDACIÓN DE ESCENARIOS TERRITORIALES AMBA
================================================

Proceso 30 del pipeline de movilidad AMBA.

Objetivo
--------
Auditar independientemente el resultado producido por el Proceso 29:

    27 -> construcción de escenarios
    28 -> evaluación
    29 -> optimización
    30 -> validación final

Este proceso NO modifica asignaciones.

Validaciones:
    1. Existencia y lectura de entrada
    2. Integridad geométrica
    3. Integridad de proyectos
    4. Integridad de escenarios
    5. Cobertura de proyectos
    6. Distribución de tamaño
    7. Cohesión espacial
    8. Balance territorial
    9. Consistencia de indicadores
    10. Consistencia contra Proceso 27
    11. Consistencia contra Proceso 28
    12. Consistencia contra Proceso 29
    13. Detección de regresiones
    14. Dictamen final

Salida:
    validacion_final_escenarios_territoriales_amba.parquet
    validacion_final_escenarios_territoriales_amba.csv
    detalle_validacion_escenarios_territoriales_amba.csv
    comparacion_procesos_27_28_29_30.csv
    alertas_validacion_escenarios_territoriales_amba.csv
    resumen_validacion_escenarios_territoriales_amba.json

Autoría:
    Pipeline de análisis territorial AMBA
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V1.0"

MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

MIN_PROYECTOS_ESCENARIO = 8

CRS_GEOGRAFICO = "EPSG:4326"

# Buenos Aires / AMBA.
# EPSG:22185 = POSGAR 2007 / Argentina 5
# Se utiliza para cálculos métricos.
CRS_METRICO = "EPSG:22185"

TOLERANCIA_DUPLICADOS = 0

# Umbrales de validación
UMBRAL_COBERTURA = 1.0
UMBRAL_COHESION = 0.60
UMBRAL_BALANCE = 0.75
UMBRAL_INDICADORES = 0.70

# Umbrales de advertencia
UMBRAL_COHESION_WARNING = 0.45
UMBRAL_BALANCE_WARNING = 0.60
UMBRAL_INDICADORES_WARNING = 0.50

# Diferencia admisible entre procesos
TOLERANCIA_SCORE = 0.15


# =============================================================================
# RUTAS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

ESCENARIOS_DIR = DATA_PROCESSED / "escenarios_territoriales_amba"

INPUT_OPTIMIZADO = (
    ESCENARIOS_DIR
    / "escenarios_territoriales_amba_optimizado.parquet"
)

INPUT_ORIGINAL = (
    ESCENARIOS_DIR
    / "escenarios_territoriales_amba.parquet"
)

INPUT_EVALUACION_28 = (
    ESCENARIOS_DIR
    / "evaluacion_escenarios_territoriales_amba.parquet"
)

INPUT_EVALUACION_28_CSV = (
    ESCENARIOS_DIR
    / "evaluacion_escenarios_territoriales_amba.csv"
)

INPUT_EVALUACION_29 = (
    ESCENARIOS_DIR
    / "evaluacion_escenarios_optimizada.csv"
)

INPUT_RESUMEN_29 = (
    ESCENARIOS_DIR
    / "resumen_optimizacion_escenarios.csv"
)


# =============================================================================
# SALIDAS
# =============================================================================

OUTPUT_PARQUET = (
    ESCENARIOS_DIR
    / "validacion_final_escenarios_territoriales_amba.parquet"
)

OUTPUT_CSV = (
    ESCENARIOS_DIR
    / "validacion_final_escenarios_territoriales_amba.csv"
)

OUTPUT_DETALLE = (
    ESCENARIOS_DIR
    / "detalle_validacion_escenarios_territoriales_amba.csv"
)

OUTPUT_COMPARACION = (
    ESCENARIOS_DIR
    / "comparacion_procesos_27_28_29_30.csv"
)

OUTPUT_ALERTAS = (
    ESCENARIOS_DIR
    / "alertas_validacion_escenarios_territoriales_amba.csv"
)

OUTPUT_JSON = (
    ESCENARIOS_DIR
    / "resumen_validacion_escenarios_territoriales_amba.json"
)


# =============================================================================
# VARIABLES
# =============================================================================

INDICADORES_PRINCIPALES = [
    "indice_demanda_estructural",
    "deficit_infraestructura",
    "indice_conectividad_estructural",
    "indice_intermodalidad_estructural",
    "indice_integracion_territorial",
    "indice_centralidad_estructural",
    "impacto_potencial",
    "urgencia_intervencion",
    "score_prioridad_territorial",
    "score_cartera",
]

COLUMNAS_ESCENARIO = [
    "escenario_id",
    "cantidad_proyectos",
    "score_escenario",
    "tipo_escenario",
    "dimension_dominante",
    "prioridad_escenario",
]

CANDIDATOS_PROYECTO_ID = [
    "proyecto_id",
    "id_proyecto",
    "project_id",
    "id",
]

COLUMNAS_RELEVANTES = [
    "escenario_id",
    "proyecto_id",
    "score_escenario",
    "tipo_escenario",
    "dimension_dominante",
    "prioridad_escenario",
]


# =============================================================================
# UTILIDADES
# =============================================================================

def imprimir_titulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def imprimir_seccion(numero: int, texto: str) -> None:
    print()
    print("=" * 88)
    print(f"{numero}. {texto}")
    print("=" * 88)


def fmt(value, decimals: int = 4) -> str:
    if value is None:
        return "N/D"

    try:
        if pd.isna(value):
            return "N/D"
    except Exception:
        pass

    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}"

    return str(value)


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def normalizar_serie(serie: pd.Series) -> pd.Series:
    """
    Normalización min-max robusta.
    """
    x = pd.to_numeric(serie, errors="coerce")

    if x.notna().sum() == 0:
        return pd.Series(0.0, index=serie.index)

    minimo = x.min()
    maximo = x.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(0.0, index=serie.index)

    if math.isclose(float(minimo), float(maximo)):
        return pd.Series(1.0, index=serie.index)

    return ((x - minimo) / (maximo - minimo)).fillna(0.0)


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: List[str],
) -> Optional[str]:

    lower_map = {str(c).lower(): c for c in df.columns}

    for candidato in candidatos:
        if candidato.lower() in lower_map:
            return lower_map[candidato.lower()]

    return None


def safe_numeric_mean(
    df: pd.DataFrame,
    column: str,
) -> float:

    if column not in df.columns:
        return np.nan

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return np.nan

    return float(values.mean())


def porcentaje(valor: float) -> float:
    return float(valor) * 100.0


def guardar_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# ALERTAS
# =============================================================================

class Alertas:

    def __init__(self):
        self.items: List[Dict] = []

    def agregar(
        self,
        nivel: str,
        codigo: str,
        mensaje: str,
        escenario_id: Optional[str] = None,
    ):
        self.items.append(
            {
                "nivel": nivel,
                "codigo": codigo,
                "escenario_id": escenario_id,
                "mensaje": mensaje,
            }
        )

    def error(
        self,
        codigo: str,
        mensaje: str,
        escenario_id: Optional[str] = None,
    ):
        self.agregar(
            "ERROR",
            codigo,
            mensaje,
            escenario_id,
        )

    def warning(
        self,
        codigo: str,
        mensaje: str,
        escenario_id: Optional[str] = None,
    ):
        self.agregar(
            "WARNING",
            codigo,
            mensaje,
            escenario_id,
        )

    def info(
        self,
        codigo: str,
        mensaje: str,
        escenario_id: Optional[str] = None,
    ):
        self.agregar(
            "INFO",
            codigo,
            mensaje,
            escenario_id,
        )

    @property
    def errores(self) -> int:
        return sum(
            1
            for x in self.items
            if x["nivel"] == "ERROR"
        )

    @property
    def warnings(self) -> int:
        return sum(
            1
            for x in self.items
            if x["nivel"] == "WARNING"
        )

    def dataframe(self) -> pd.DataFrame:

        if not self.items:
            return pd.DataFrame(
                columns=[
                    "nivel",
                    "codigo",
                    "escenario_id",
                    "mensaje",
                ]
            )

        return pd.DataFrame(self.items)


# =============================================================================
# CARGA
# =============================================================================

def cargar_geoparquet(path: Path) -> gpd.GeoDataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada:\n{path}"
        )

    gdf = gpd.read_parquet(path)

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            "El archivo no fue leído como GeoDataFrame."
        )

    return gdf


# =============================================================================
# VALIDACIÓN GEOMÉTRICA
# =============================================================================

def validar_geometrias(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Dict:

    resultado = {}

    resultado["geometrias_nulas"] = int(
        gdf.geometry.isna().sum()
    )

    resultado["geometrias_vacias"] = int(
        gdf.geometry.is_empty.sum()
    )

    resultado["geometrias_invalidas"] = int(
        (~gdf.geometry.is_valid).sum()
    )

    if resultado["geometrias_nulas"] > 0:
        alertas.error(
            "GEOMETRIAS_NULAS",
            f"Se detectaron "
            f"{resultado['geometrias_nulas']} geometrías nulas.",
        )

    if resultado["geometrias_vacias"] > 0:
        alertas.error(
            "GEOMETRIAS_VACIAS",
            f"Se detectaron "
            f"{resultado['geometrias_vacias']} geometrías vacías.",
        )

    if resultado["geometrias_invalidas"] > 0:
        alertas.error(
            "GEOMETRIAS_INVALIDAS",
            f"Se detectaron "
            f"{resultado['geometrias_invalidas']} geometrías inválidas.",
        )

    return resultado


# =============================================================================
# IDENTIFICADOR DE PROYECTO
# =============================================================================

def resolver_proyecto_id(
    gdf: gpd.GeoDataFrame,
) -> Optional[str]:

    return encontrar_columna(
        gdf,
        CANDIDATOS_PROYECTO_ID,
    )


# =============================================================================
# VALIDACIÓN DE PROYECTOS
# =============================================================================

def validar_proyectos(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
    proyecto_col: Optional[str],
) -> Dict:

    resultado = {
        "proyectos_totales": len(gdf),
        "proyecto_id_disponible": proyecto_col is not None,
        "duplicados_proyecto": 0,
        "proyectos_nulos": 0,
    }

    if proyecto_col is None:
        alertas.error(
            "PROYECTO_ID_AUSENTE",
            "No se encontró una columna de identificador de proyecto.",
        )
        return resultado

    serie = gdf[proyecto_col]

    resultado["proyectos_nulos"] = int(
        serie.isna().sum()
    )

    if resultado["proyectos_nulos"] > 0:
        alertas.error(
            "PROYECTOS_SIN_ID",
            f"Se detectaron "
            f"{resultado['proyectos_nulos']} proyectos sin identificador.",
        )

    duplicados = int(
        serie.duplicated(keep=False).sum()
    )

    resultado["duplicados_proyecto"] = duplicados

    if duplicados > TOLERANCIA_DUPLICADOS:
        alertas.error(
            "PROYECTOS_DUPLICADOS",
            f"Se detectaron {duplicados} registros "
            f"duplicados por proyecto.",
        )

    return resultado


# =============================================================================
# VALIDACIÓN DE ESCENARIOS
# =============================================================================

def validar_escenarios(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Dict:

    resultado = {}

    if "escenario_id" not in gdf.columns:
        alertas.error(
            "ESCENARIO_ID_AUSENTE",
            "No existe la columna escenario_id.",
        )

        return {
            "escenarios": 0,
            "escenarios_validos": 0,
            "escenarios_fuera_rango": 0,
        }

    escenarios = (
        gdf["escenario_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    n = len(escenarios)

    resultado["escenarios"] = n
    resultado["escenarios_validos"] = int(
        MIN_ESCENARIOS <= n <= MAX_ESCENARIOS
    )
    resultado["escenarios_fuera_rango"] = int(
        not resultado["escenarios_validos"]
    )

    if not resultado["escenarios_validos"]:
        alertas.error(
            "CANTIDAD_ESCENARIOS",
            f"Se detectaron {n} escenarios. "
            f"El rango permitido es "
            f"{MIN_ESCENARIOS}-{MAX_ESCENARIOS}.",
        )

    # Escenarios nulos
    nulos = int(
        gdf["escenario_id"].isna().sum()
    )

    resultado["escenarios_nulos"] = nulos

    if nulos > 0:
        alertas.error(
            "ESCENARIOS_NULOS",
            f"Hay {nulos} proyectos sin escenario asignado.",
        )

    return resultado


# =============================================================================
# COBERTURA
# =============================================================================

def calcular_cobertura(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Dict:

    total = len(gdf)

    if total == 0:
        return {
            "proyectos_totales": 0,
            "proyectos_asignados": 0,
            "proyectos_sin_escenario": 0,
            "cobertura": 0.0,
        }

    asignados = int(
        gdf["escenario_id"].notna().sum()
    )

    sin_escenario = total - asignados

    cobertura = asignados / total

    resultado = {
        "proyectos_totales": total,
        "proyectos_asignados": asignados,
        "proyectos_sin_escenario": sin_escenario,
        "cobertura": cobertura,
    }

    if cobertura < UMBRAL_COBERTURA:
        alertas.error(
            "COBERTURA_INCOMPLETA",
            f"La cobertura es "
            f"{porcentaje(cobertura):.2f}%.",
        )

    return resultado


# =============================================================================
# TAMAÑO
# =============================================================================

def calcular_tamano(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Tuple[Dict, pd.DataFrame]:

    conteo = (
        gdf.groupby("escenario_id", dropna=False)
        .size()
        .rename("cantidad_proyectos")
        .reset_index()
    )

    if len(conteo) == 0:
        return {}, conteo

    minimo = int(conteo["cantidad_proyectos"].min())
    maximo = int(conteo["cantidad_proyectos"].max())
    promedio = float(conteo["cantidad_proyectos"].mean())
    desviacion = float(
        conteo["cantidad_proyectos"].std(ddof=0)
    )

    cumplen = int(
        (
            conteo["cantidad_proyectos"]
            >= MIN_PROYECTOS_ESCENARIO
        ).sum()
    )

    score_minimo = (
        cumplen / len(conteo)
        if len(conteo)
        else 0.0
    )

    # Penalización moderada por dispersión.
    cv = (
        desviacion / promedio
        if promedio > 0
        else 1.0
    )

    score_balance_tamano = clamp(
        1.0 - cv
    )

    score = (
        0.70 * score_minimo
        + 0.30 * score_balance_tamano
    )

    resultado = {
        "escenarios": len(conteo),
        "minimo_proyectos": minimo,
        "maximo_proyectos": maximo,
        "promedio_proyectos": promedio,
        "desvio_proyectos": desviacion,
        "escenarios_cumplen_minimo": cumplen,
        "score_tamano": score,
        "coeficiente_variacion": cv,
    }

    for _, row in conteo.iterrows():

        escenario = str(
            row["escenario_id"]
        )

        cantidad = int(
            row["cantidad_proyectos"]
        )

        if cantidad < MIN_PROYECTOS_ESCENARIO:

            alertas.error(
                "ESCENARIO_PEQUENO",
                f"El escenario contiene "
                f"{cantidad} proyectos. "
                f"Mínimo: "
                f"{MIN_PROYECTOS_ESCENARIO}.",
                escenario,
            )

    return resultado, conteo


# =============================================================================
# COHESIÓN TERRITORIAL
# =============================================================================

def calcular_cohesion(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Tuple[Dict, pd.DataFrame]:

    if gdf.empty:
        return {}, pd.DataFrame()

    # Siempre trabajar en CRS métrico.
    if gdf.crs is None:
        alertas.error(
            "CRS_AUSENTE",
            "El GeoDataFrame no posee CRS.",
        )
        return {}, pd.DataFrame()

    try:
        gdf_metric = gdf.to_crs(CRS_METRICO)
    except Exception as exc:
        alertas.error(
            "CRS_METRICO_ERROR",
            f"No se pudo transformar a {CRS_METRICO}: {exc}",
        )
        return {}, pd.DataFrame()

    gdf_metric = gdf_metric.copy()

    # Para puntos usamos la geometría.
    # Para geometrías complejas usamos representative_point,
    # evitando centroides fuera de geometría.
    puntos = gdf_metric.geometry.representative_point()

    gdf_metric["_x"] = puntos.x
    gdf_metric["_y"] = puntos.y

    filas = []

    distancias_todas = []

    for escenario_id, grupo in gdf_metric.groupby(
        "escenario_id",
        dropna=False,
    ):

        escenario_id = str(escenario_id)

        coords = grupo[
            ["_x", "_y"]
        ].to_numpy(dtype=float)

        if len(coords) == 0:
            continue

        centro_x = float(coords[:, 0].mean())
        centro_y = float(coords[:, 1].mean())

        distancias = np.sqrt(
            (
                coords[:, 0] - centro_x
            ) ** 2
            +
            (
                coords[:, 1] - centro_y
            ) ** 2
        )

        distancia_media = float(
            np.mean(distancias)
        )

        distancia_maxima = float(
            np.max(distancias)
        )

        dispersion = float(
            np.std(distancias)
        )

        distancias_todas.extend(
            distancias.tolist()
        )

        filas.append(
            {
                "escenario_id": escenario_id,
                "cantidad_proyectos": len(grupo),
                "distancia_media_m": distancia_media,
                "distancia_maxima_m": distancia_maxima,
                "dispersion_m": dispersion,
            }
        )

    detalle = pd.DataFrame(filas)

    if detalle.empty:
        return {}, detalle

    distancia_media_global = float(
        detalle["distancia_media_m"].mean()
    )

    # Convertimos dispersión territorial en score.
    #
    # No utilizamos una distancia absoluta rígida,
    # porque los escenarios pueden tener tamaños territoriales
    # diferentes.
    #
    # Se utiliza relación relativa respecto al escenario
    # más compacto.
    max_dist = float(
        detalle["distancia_media_m"].max()
    )

    min_dist = float(
        detalle["distancia_media_m"].min()
    )

    if max_dist <= 0:
        score_cohesion = 1.0
    elif math.isclose(max_dist, min_dist):
        score_cohesion = 1.0
    else:

        # Mediana como referencia robusta.
        mediana = float(
            detalle["distancia_media_m"].median()
        )

        ratio = (
            mediana / max_dist
            if max_dist > 0
            else 1.0
        )

        score_cohesion = clamp(
            ratio
        )

    # Segunda señal: variabilidad relativa entre escenarios.
    media_dist = detalle[
        "distancia_media_m"
    ].mean()

    std_dist = detalle[
        "distancia_media_m"
    ].std(ddof=0)

    if media_dist > 0:
        cv = float(
            std_dist / media_dist
        )
    else:
        cv = 0.0

    score_estabilidad = clamp(
        1.0 - cv
    )

    score_final = (
        0.75 * score_cohesion
        + 0.25 * score_estabilidad
    )

    resultado = {
        "distancia_media_m": distancia_media_global,
        "distancia_maxima_m": float(
            detalle["distancia_maxima_m"].max()
        ),
        "score_cohesion": score_final,
        "cv_distancias": cv,
    }

    if score_final < UMBRAL_COHESION:
        if score_final < UMBRAL_COHESION_WARNING:
            alertas.error(
                "COHESION_BAJA",
                f"Score de cohesión: "
                f"{score_final:.4f}.",
            )
        else:
            alertas.warning(
                "COHESION_MODERADA",
                f"Score de cohesión: "
                f"{score_final:.4f}.",
            )

    return resultado, detalle


# =============================================================================
# INDICADORES
# =============================================================================

def detectar_indicadores(
    gdf: gpd.GeoDataFrame,
) -> List[str]:

    return [
        column
        for column in INDICADORES_PRINCIPALES
        if column in gdf.columns
    ]


def evaluar_indicadores(
    gdf: gpd.GeoDataFrame,
    indicadores: List[str],
    alertas: Alertas,
) -> Tuple[Dict, pd.DataFrame]:

    if not indicadores:

        alertas.error(
            "SIN_INDICADORES",
            "No se encontraron indicadores estructurales.",
        )

        return {
            "indicadores_detectados": 0,
            "score_indicadores": 0.0,
        }, pd.DataFrame()

    detalle = []

    for indicador in indicadores:

        serie = pd.to_numeric(
            gdf[indicador],
            errors="coerce",
        )

        validos = int(
            serie.notna().sum()
        )

        nulos = int(
            serie.isna().sum()
        )

        if validos == 0:

            alertas.error(
                "INDICADOR_SIN_DATOS",
                f"El indicador {indicador} no posee "
                f"valores numéricos válidos.",
            )

            continue

        minimo = float(
            serie.min()
        )

        maximo = float(
            serie.max()
        )

        media = float(
            serie.mean()
        )

        detalle.append(
            {
                "indicador": indicador,
                "validos": validos,
                "nulos": nulos,
                "minimo": minimo,
                "maximo": maximo,
                "media": media,
            }
        )

    detalle_df = pd.DataFrame(detalle)

    if detalle_df.empty:

        return {
            "indicadores_detectados": len(indicadores),
            "indicadores_validos": 0,
            "score_indicadores": 0.0,
        }, detalle_df

    completitud = (
        detalle_df["validos"].mean()
        / len(gdf)
    )

    # Se evalúa además la variabilidad.
    indicadores_con_variacion = int(
        (
            detalle_df["maximo"]
            > detalle_df["minimo"]
        ).sum()
    )

    variabilidad = (
        indicadores_con_variacion
        / len(detalle_df)
    )

    score = clamp(
        0.70 * completitud
        + 0.30 * variabilidad
    )

    if score < UMBRAL_INDICADORES:

        if score < UMBRAL_INDICADORES_WARNING:
            alertas.error(
                "INDICADORES_DEBILES",
                f"Score de indicadores: "
                f"{score:.4f}.",
            )
        else:
            alertas.warning(
                "INDICADORES_MODERADOS",
                f"Score de indicadores: "
                f"{score:.4f}.",
            )

    return {
        "indicadores_detectados": len(indicadores),
        "indicadores_validos": len(detalle_df),
        "score_indicadores": score,
        "completitud_indicadores": completitud,
        "variabilidad_indicadores": variabilidad,
    }, detalle_df


# =============================================================================
# BALANCE
# =============================================================================

def evaluar_balance(
    conteo: pd.DataFrame,
) -> Dict:

    if conteo.empty:
        return {
            "promedio": 0.0,
            "desvio": 0.0,
            "cv": 1.0,
            "score_balance": 0.0,
        }

    valores = conteo[
        "cantidad_proyectos"
    ].astype(float)

    promedio = float(
        valores.mean()
    )

    desvio = float(
        valores.std(ddof=0)
    )

    if promedio > 0:
        cv = desvio / promedio
    else:
        cv = 1.0

    score = clamp(
        1.0 - cv
    )

    return {
        "promedio": promedio,
        "desvio": desvio,
        "cv": cv,
        "score_balance": score,
    }


# =============================================================================
# CONSISTENCIA DE CAMPOS DE ESCENARIO
# =============================================================================

def validar_consistencia_escenario(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> pd.DataFrame:

    escenarios = []

    for escenario_id, grupo in gdf.groupby(
        "escenario_id",
        dropna=False,
    ):

        escenario_id = str(
            escenario_id
        )

        fila = {
            "escenario_id": escenario_id,
            "cantidad_registros": len(grupo),
        }

        for column in [
            "cantidad_proyectos",
            "score_escenario",
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
        ]:

            if column not in grupo.columns:
                fila[
                    f"{column}_consistente"
                ] = False
                continue

            valores = (
                grupo[column]
                .dropna()
                .astype(str)
                .unique()
            )

            consistente = len(valores) <= 1

            fila[
                f"{column}_consistente"
            ] = consistente

            if not consistente:

                alertas.error(
                    "CAMPO_ESCENARIO_INCONSISTENTE",
                    f"El campo {column} contiene "
                    f"múltiples valores dentro del "
                    f"escenario.",
                    escenario_id,
                )

        # cantidad_proyectos
        if "cantidad_proyectos" in grupo.columns:

            declarada = pd.to_numeric(
                grupo["cantidad_proyectos"],
                errors="coerce",
            )

            declaradas = (
                declarada.dropna()
                .unique()
            )

            real = len(grupo)

            if len(declaradas) > 0:

                if not np.isclose(
                    declaradas[0],
                    real,
                ):

                    alertas.warning(
                        "CANTIDAD_ESCENARIO_INCONSISTENTE",
                        f"cantidad_proyectos declarada="
                        f"{declaradas[0]} "
                        f"pero registros reales={real}.",
                        escenario_id,
                    )

        escenarios.append(fila)

    return pd.DataFrame(escenarios)


# =============================================================================
# SCORE GLOBAL DE VALIDACIÓN
# =============================================================================

def calcular_score_validacion(
    cobertura: float,
    tamano: float,
    cohesion: float,
    indicadores: float,
    balance: float,
    estructura: float,
) -> float:

    pesos = {
        "cobertura": 0.20,
        "tamano": 0.15,
        "cohesion": 0.25,
        "indicadores": 0.20,
        "balance": 0.20,
    }

    score = (
        pesos["cobertura"] * cobertura
        +
        pesos["tamano"] * tamano
        +
        pesos["cohesion"] * cohesion
        +
        pesos["indicadores"] * indicadores
        +
        pesos["balance"] * balance
    )

    # La estructura funciona como condición de integridad.
    if estructura <= 0:
        return 0.0

    return clamp(score)


def clasificar_score(score: float) -> str:

    if score >= 0.85:
        return "EXCELENTE"

    if score >= 0.75:
        return "VALIDADO"

    if score >= 0.65:
        return "VALIDADO_CON_OBSERVACIONES"

    if score >= 0.50:
        return "REVISAR"

    return "RECHAZADO"


# =============================================================================
# VALIDACIÓN CONTRA PROCESO 27
# =============================================================================

def comparar_con_proceso_27(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Dict:

    resultado = {
        "archivo_disponible": INPUT_ORIGINAL.exists(),
        "proyectos_original": None,
        "proyectos_optimizado": len(gdf),
        "escenarios_original": None,
        "escenarios_optimizado": (
            gdf["escenario_id"]
            .nunique()
            if "escenario_id" in gdf.columns
            else 0
        ),
        "asignaciones_modificadas": None,
    }

    if not INPUT_ORIGINAL.exists():
        alertas.warning(
            "PROCESO_27_AUSENTE",
            "No se encontró la salida original del Proceso 27.",
        )
        return resultado

    try:

        original = gpd.read_parquet(
            INPUT_ORIGINAL
        )

        resultado["proyectos_original"] = len(
            original
        )

        resultado["escenarios_original"] = (
            original["escenario_id"]
            .nunique()
            if "escenario_id" in original.columns
            else None
        )

        proyecto_col_orig = resolver_proyecto_id(
            original
        )

        proyecto_col_opt = resolver_proyecto_id(
            gdf
        )

        if (
            proyecto_col_orig is not None
            and proyecto_col_opt is not None
            and "escenario_id" in original.columns
            and "escenario_id" in gdf.columns
        ):

            a = original[
                [
                    proyecto_col_orig,
                    "escenario_id",
                ]
            ].copy()

            b = gdf[
                [
                    proyecto_col_opt,
                    "escenario_id",
                ]
            ].copy()

            a.columns = [
                "proyecto_id",
                "escenario_original",
            ]

            b.columns = [
                "proyecto_id",
                "escenario_optimizado",
            ]

            merge = a.merge(
                b,
                on="proyecto_id",
                how="outer",
                indicator=True,
            )

            modificados = int(
                (
                    (
                        merge["escenario_original"]
                        != merge["escenario_optimizado"]
                    )
                    &
                    merge["escenario_original"].notna()
                    &
                    merge["escenario_optimizado"].notna()
                ).sum()
            )

            resultado[
                "asignaciones_modificadas"
            ] = modificados

            solo_orig = int(
                (
                    merge["_merge"]
                    == "left_only"
                ).sum()
            )

            solo_opt = int(
                (
                    merge["_merge"]
                    == "right_only"
                ).sum()
            )

            resultado[
                "proyectos_solo_original"
            ] = solo_orig

            resultado[
                "proyectos_solo_optimizado"
            ] = solo_opt

            if solo_orig > 0 or solo_opt > 0:

                alertas.error(
                    "PROYECTOS_CAMBIADOS",
                    "La composición de proyectos entre "
                    "Proceso 27 y Proceso 29 no coincide.",
                )

    except Exception as exc:

        alertas.warning(
            "ERROR_COMPARACION_27",
            f"No se pudo comparar contra Proceso 27: {exc}",
        )

    return resultado


# =============================================================================
# VALIDACIÓN CONTRA PROCESO 28
# =============================================================================

def comparar_con_proceso_28(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Dict:

    resultado = {
        "archivo_disponible": (
            INPUT_EVALUACION_28.exists()
            or INPUT_EVALUACION_28_CSV.exists()
        ),
        "score_global_28": None,
        "escenarios_28": None,
    }

    path = None

    if INPUT_EVALUACION_28.exists():
        path = INPUT_EVALUACION_28
    elif INPUT_EVALUACION_28_CSV.exists():
        path = INPUT_EVALUACION_28_CSV

    if path is None:

        alertas.warning(
            "PROCESO_28_AUSENTE",
            "No se encontró la evaluación del Proceso 28.",
        )

        return resultado

    try:

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        resultado["escenarios_28"] = (
            df["escenario_id"].nunique()
            if "escenario_id" in df.columns
            else None
        )

        score_candidates = [
            "score_global",
            "score_evaluacion",
        ]

        for column in score_candidates:

            if column in df.columns:

                values = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                if values.notna().any():

                    if column == "score_global":
                        score = float(
                            values.dropna().iloc[0]
                        )
                    else:
                        score = float(
                            values.mean()
                        )

                    resultado[
                        "score_global_28"
                    ] = score

                    break

    except Exception as exc:

        alertas.warning(
            "ERROR_COMPARACION_28",
            f"No se pudo leer Proceso 28: {exc}",
        )

    return resultado


# =============================================================================
# VALIDACIÓN CONTRA PROCESO 29
# =============================================================================

def comparar_con_proceso_29(
    gdf: gpd.GeoDataFrame,
    score_global: float,
    alertas: Alertas,
) -> Dict:

    resultado = {
        "archivo_disponible": INPUT_EVALUACION_29.exists(),
        "score_global_29": None,
        "score_global_30": score_global,
        "diferencia_score": None,
    }

    if not INPUT_EVALUACION_29.exists():

        alertas.warning(
            "PROCESO_29_EVALUACION_AUSENTE",
            "No se encontró evaluacion_escenarios_optimizada.csv.",
        )

        return resultado

    try:

        df = pd.read_csv(
            INPUT_EVALUACION_29
        )

        candidatos = [
            "score_global",
            "score_global_optimizado",
            "score_evaluacion",
        ]

        for column in candidatos:

            if column in df.columns:

                values = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                if values.notna().any():

                    score = float(
                        values.dropna().mean()
                    )

                    resultado[
                        "score_global_29"
                    ] = score

                    resultado[
                        "diferencia_score"
                    ] = score_global - score

                    break

    except Exception as exc:

        alertas.warning(
            "ERROR_COMPARACION_29",
            f"No se pudo leer evaluación del Proceso 29: {exc}",
        )

    return resultado


# =============================================================================
# DETALLE DE ESCENARIOS
# =============================================================================

def construir_detalle_escenarios(
    gdf: gpd.GeoDataFrame,
    cohesion_detalle: pd.DataFrame,
    indicadores: List[str],
) -> pd.DataFrame:

    filas = []

    for escenario_id, grupo in gdf.groupby(
        "escenario_id",
        dropna=False,
    ):

        escenario_id = str(
            escenario_id
        )

        fila = {
            "escenario_id": escenario_id,
            "cantidad_proyectos": len(grupo),
        }

        if "score_escenario" in grupo.columns:

            fila["score_escenario"] = safe_numeric_mean(
                grupo,
                "score_escenario",
            )

        for column in [
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
        ]:

            if column in grupo.columns:

                values = (
                    grupo[column]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                fila[column] = (
                    values[0]
                    if len(values) == 1
                    else "INCONSISTENTE"
                )

        if not cohesion_detalle.empty:

            row_cohesion = cohesion_detalle[
                cohesion_detalle["escenario_id"]
                == escenario_id
            ]

            if not row_cohesion.empty:

                r = row_cohesion.iloc[0]

                fila[
                    "distancia_media_m"
                ] = r["distancia_media_m"]

                fila[
                    "distancia_maxima_m"
                ] = r["distancia_maxima_m"]

                fila[
                    "dispersion_m"
                ] = r["dispersion_m"]

        # Indicadores promedio por escenario.
        for indicador in indicadores:

            fila[
                f"media_{indicador}"
            ] = safe_numeric_mean(
                grupo,
                indicador,
            )

        filas.append(fila)

    return pd.DataFrame(filas)


# =============================================================================
# SCORE POR ESCENARIO
# =============================================================================

def calcular_score_por_escenario(
    detalle: pd.DataFrame,
    indicadores: List[str],
) -> pd.DataFrame:

    if detalle.empty:
        return detalle

    df = detalle.copy()

    # 1. Tamaño
    if "cantidad_proyectos" in df.columns:

        tamaño = df[
            "cantidad_proyectos"
        ].astype(float)

        # Cercanía a la media.
        media = tamaño.mean()

        if media > 0:

            desviacion_relativa = (
                (tamaño - media).abs()
                / media
            )

            df[
                "score_tamano"
            ] = (
                1.0
                - desviacion_relativa
            ).clip(0, 1)

        else:
            df[
                "score_tamano"
            ] = 0.0

    else:
        df["score_tamano"] = 0.0

    # 2. Cohesión
    if "distancia_media_m" in df.columns:

        distancia = pd.to_numeric(
            df["distancia_media_m"],
            errors="coerce",
        )

        maximo = distancia.max()

        if maximo > 0:

            df[
                "score_cohesion"
            ] = (
                1.0
                - distancia / maximo
            ).clip(0, 1)

        else:
            df[
                "score_cohesion"
            ] = 1.0

    else:
        df["score_cohesion"] = 0.0

    # 3. Indicadores
    columnas_ind = [
        f"media_{x}"
        for x in indicadores
        if f"media_{x}" in df.columns
    ]

    if columnas_ind:

        normales = []

        for column in columnas_ind:

            normales.append(
                normalizar_serie(
                    df[column]
                )
            )

        matriz = pd.concat(
            normales,
            axis=1,
        )

        df[
            "score_indicadores"
        ] = matriz.mean(
            axis=1
        )

    else:
        df[
            "score_indicadores"
        ] = 0.0

    # 4. Score final
    df[
        "score_validacion_escenario"
    ] = (
        0.20 * df["score_tamano"]
        +
        0.40 * df["score_cohesion"]
        +
        0.40 * df["score_indicadores"]
    )

    df[
        "ranking_validacion"
    ] = (
        df[
            "score_validacion_escenario"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    df[
        "clasificacion_validacion"
    ] = df[
        "score_validacion_escenario"
    ].apply(
        lambda x: (
            "VALIDADO"
            if x >= 0.70
            else
            "VALIDADO_CON_OBSERVACIONES"
            if x >= 0.50
            else
            "REVISAR"
        )
    )

    return df.sort_values(
        "ranking_validacion"
    )


# =============================================================================
# VALIDACIÓN FINAL DE ESTRUCTURA
# =============================================================================

def validar_integridad_final(
    gdf: gpd.GeoDataFrame,
    alertas: Alertas,
) -> Dict:

    proyecto_col = resolver_proyecto_id(
        gdf
    )

    resultado = {
        "proyecto_id_resuelto": proyecto_col,
        "escenario_id": "escenario_id"
        if "escenario_id" in gdf.columns
        else None,
        "geometria": "geometry"
        if "geometry" in gdf.columns
        else None,
    }

    columnas_criticas = [
        "escenario_id",
    ]

    for column in columnas_criticas:

        if column not in gdf.columns:

            alertas.error(
                "COLUMNA_CRITICA_AUSENTE",
                f"No existe la columna crítica {column}.",
            )

    if gdf.empty:

        alertas.error(
            "DATASET_VACIO",
            "El dataset optimizado está vacío.",
        )

    return resultado


# =============================================================================
# DICTAMEN
# =============================================================================

def construir_dictamen(
    score_global: float,
    alertas: Alertas,
    cobertura: float,
    escenarios: int,
) -> Tuple[str, str]:

    if alertas.errores > 0:

        return (
            "RECHAZADO",
            "Se detectaron errores estructurales o "
            "de integridad que impiden validar el resultado.",
        )

    if cobertura < 1.0:

        return (
            "RECHAZADO",
            "No todos los proyectos poseen escenario asignado.",
        )

    if not (
        MIN_ESCENARIOS
        <= escenarios
        <= MAX_ESCENARIOS
    ):

        return (
            "RECHAZADO",
            "La cantidad de escenarios está fuera del rango permitido.",
        )

    if score_global >= 0.85:

        if alertas.warnings == 0:
            return (
                "VALIDADO",
                "El resultado optimizado supera los criterios "
                "de validación sin observaciones relevantes.",
            )

        return (
            "VALIDADO_CON_OBSERVACIONES",
            "El resultado cumple los criterios principales, "
            "pero presenta observaciones menores.",
        )

    if score_global >= 0.65:

        return (
            "VALIDADO_CON_OBSERVACIONES",
            "El resultado es estructuralmente válido, "
            "pero presenta oportunidades de mejora.",
        )

    if score_global >= 0.50:

        return (
            "REVISAR",
            "El resultado mantiene integridad estructural, "
            "pero no alcanza un nivel suficiente de calidad.",
        )

    return (
        "RECHAZADO",
        "El resultado no alcanza los criterios mínimos de validación.",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    inicio = time.perf_counter()

    warnings.filterwarnings(
        "ignore",
        message=".*unary_union.*",
    )

    alertas = Alertas()

    ESCENARIOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    imprimir_titulo(
        f"30 - VALIDACIÓN FINAL DE ESCENARIOS "
        f"TERRITORIALES AMBA - {VERSION}"
    )

    print(f"Proyecto : {PROJECT_ROOT}")
    print(f"Entrada  : {INPUT_OPTIMIZADO}")
    print(f"Salida   : {ESCENARIOS_DIR}")

    print()
    print("CONFIGURACIÓN")
    print(f"  Versión                  : {VERSION}")
    print(
        f"  Escenarios válidos       : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
    )
    print(
        f"  Mínimo proyectos         : "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )
    print(
        f"  CRS geográfico           : "
        f"{CRS_GEOGRAFICO}"
    )
    print(
        f"  CRS métrico              : "
        f"{CRS_METRICO}"
    )

    # -------------------------------------------------------------------------
    # 1
    # -------------------------------------------------------------------------

    imprimir_seccion(
        1,
        "CARGANDO RESULTADO OPTIMIZADO DEL PROCESO 29",
    )

    print(f"Entrada:\n{INPUT_OPTIMIZADO}")

    try:
        gdf = cargar_geoparquet(
            INPUT_OPTIMIZADO
        )
    except Exception as exc:

        print()
        print("ERROR FATAL")
        print(str(exc))

        return 1

    print(f"Registros : {len(gdf)}")
    print(f"Columnas  : {len(gdf.columns)}")
    print(f"CRS       : {gdf.crs}")

    # -------------------------------------------------------------------------
    # 2
    # -------------------------------------------------------------------------

    imprimir_seccion(
        2,
        "VALIDANDO ESTRUCTURA GEOGRÁFICA",
    )

    geometria = validar_geometrias(
        gdf,
        alertas,
    )

    print(
        f"Geometrías nulas      : "
        f"{geometria['geometrias_nulas']}"
    )

    print(
        f"Geometrías vacías     : "
        f"{geometria['geometrias_vacias']}"
    )

    print(
        f"Geometrías inválidas  : "
        f"{geometria['geometrias_invalidas']}"
    )

    if gdf.crs is None:

        alertas.error(
            "CRS_AUSENTE",
            "El dataset no tiene CRS definido.",
        )

    else:

        print(
            f"CRS geográfico        : "
            f"{gdf.crs}"
        )

    # -------------------------------------------------------------------------
    # 3
    # -------------------------------------------------------------------------

    imprimir_seccion(
        3,
        "VALIDANDO PROYECTOS",
    )

    proyecto_col = resolver_proyecto_id(
        gdf
    )

    proyectos = validar_proyectos(
        gdf,
        alertas,
        proyecto_col,
    )

    print(
        f"Identificador proyecto : "
        f"{proyecto_col}"
    )

    print(
        f"Proyectos totales       : "
        f"{proyectos['proyectos_totales']}"
    )

    print(
        f"Proyectos sin ID        : "
        f"{proyectos['proyectos_nulos']}"
    )

    print(
        f"Duplicados proyecto     : "
        f"{proyectos['duplicados_proyecto']}"
    )

    # -------------------------------------------------------------------------
    # 4
    # -------------------------------------------------------------------------

    imprimir_seccion(
        4,
        "VALIDANDO ESCENARIOS",
    )

    escenarios_validacion = validar_escenarios(
        gdf,
        alertas,
    )

    print(
        f"Escenarios detectados   : "
        f"{escenarios_validacion['escenarios']}"
    )

    print(
        f"Rango válido            : "
        f"{MIN_ESCENARIOS}-{MAX_ESCENARIOS}"
    )

    print(
        f"Escenarios nulos        : "
        f"{escenarios_validacion.get('escenarios_nulos', 0)}"
    )

    # -------------------------------------------------------------------------
    # 5
    # -------------------------------------------------------------------------

    imprimir_seccion(
        5,
        "VALIDANDO COBERTURA",
    )

    cobertura = calcular_cobertura(
        gdf,
        alertas,
    )

    print(
        f"Proyectos totales       : "
        f"{cobertura['proyectos_totales']}"
    )

    print(
        f"Proyectos asignados     : "
        f"{cobertura['proyectos_asignados']}"
    )

    print(
        f"Proyectos sin escenario : "
        f"{cobertura['proyectos_sin_escenario']}"
    )

    print(
        f"Cobertura               : "
        f"{porcentaje(cobertura['cobertura']):.2f}%"
    )

    # -------------------------------------------------------------------------
    # 6
    # -------------------------------------------------------------------------

    imprimir_seccion(
        6,
        "VALIDANDO TAMAÑO DE ESCENARIOS",
    )

    tamano, conteo = calcular_tamano(
        gdf,
        alertas,
    )

    print(
        f"Escenarios              : "
        f"{tamano.get('escenarios', 0)}"
    )

    print(
        f"Mínimo proyectos        : "
        f"{tamano.get('minimo_proyectos', 0)}"
    )

    print(
        f"Máximo proyectos        : "
        f"{tamano.get('maximo_proyectos', 0)}"
    )

    print(
        f"Promedio proyectos      : "
        f"{fmt(tamano.get('promedio_proyectos'))}"
    )

    print(
        f"Desvío                  : "
        f"{fmt(tamano.get('desvio_proyectos'))}"
    )

    print(
        f"Cumplen mínimo          : "
        f"{tamano.get('escenarios_cumplen_minimo', 0)}"
        f"/{tamano.get('escenarios', 0)}"
    )

    print(
        f"Score tamaño            : "
        f"{fmt(tamano.get('score_tamano'))}"
    )

    # -------------------------------------------------------------------------
    # 7
    # -------------------------------------------------------------------------

    imprimir_seccion(
        7,
        "VALIDANDO COHESIÓN TERRITORIAL",
    )

    cohesion, cohesion_detalle = calcular_cohesion(
        gdf,
        alertas,
    )

    print(
        f"Distancia media         : "
        f"{fmt(cohesion.get('distancia_media_m'), 2)} m"
    )

    print(
        f"Distancia máxima        : "
        f"{fmt(cohesion.get('distancia_maxima_m'), 2)} m"
    )

    print(
        f"Score cohesión          : "
        f"{fmt(cohesion.get('score_cohesion'))}"
    )

    # -------------------------------------------------------------------------
    # 8
    # -------------------------------------------------------------------------

    imprimir_seccion(
        8,
        "VALIDANDO BALANCE ENTRE ESCENARIOS",
    )

    balance = evaluar_balance(
        conteo
    )

    print(
        f"Promedio                : "
        f"{fmt(balance['promedio'])}"
    )

    print(
        f"Desvío estándar         : "
        f"{fmt(balance['desvio'])}"
    )

    print(
        f"Coeficiente variación   : "
        f"{fmt(balance['cv'])}"
    )

    print(
        f"Score balance           : "
        f"{fmt(balance['score_balance'])}"
    )

    if balance["score_balance"] < UMBRAL_BALANCE:

        if balance["score_balance"] < UMBRAL_BALANCE_WARNING:

            alertas.error(
                "BALANCE_BAJO",
                f"Score de balance: "
                f"{balance['score_balance']:.4f}.",
            )

        else:

            alertas.warning(
                "BALANCE_MODERADO",
                f"Score de balance: "
                f"{balance['score_balance']:.4f}.",
            )

    # -------------------------------------------------------------------------
    # 9
    # -------------------------------------------------------------------------

    imprimir_seccion(
        9,
        "VALIDANDO INDICADORES ESTRUCTURALES",
    )

    indicadores = detectar_indicadores(
        gdf
    )

    print(
        f"Indicadores detectados : "
        f"{len(indicadores)}"
    )

    for indicador in indicadores:
        print(f"  - {indicador}")

    evaluacion_indicadores, indicadores_detalle = (
        evaluar_indicadores(
            gdf,
            indicadores,
            alertas,
        )
    )

    print(
        f"Score indicadores      : "
        f"{fmt(evaluacion_indicadores.get('score_indicadores'))}"
    )

    # -------------------------------------------------------------------------
    # 10
    # -------------------------------------------------------------------------

    imprimir_seccion(
        10,
        "VALIDANDO CONSISTENCIA INTERNA DE ESCENARIOS",
    )

    consistencia = validar_consistencia_escenario(
        gdf,
        alertas,
    )

    inconsistencias = 0

    if not consistencia.empty:

        for column in consistencia.columns:

            if column.endswith(
                "_consistente"
            ):

                inconsistencias += int(
                    (
                        consistencia[column]
                        == False
                    ).sum()
                )

    print(
        f"Escenarios revisados    : "
        f"{len(consistencia)}"
    )

    print(
        f"Inconsistencias         : "
        f"{inconsistencias}"
    )

    # -------------------------------------------------------------------------
    # 11
    # -------------------------------------------------------------------------

    imprimir_seccion(
        11,
        "COMPARANDO CONTRA PROCESO 27",
    )

    comparacion_27 = comparar_con_proceso_27(
        gdf,
        alertas,
    )

    print(
        f"Archivo disponible      : "
        f"{comparacion_27['archivo_disponible']}"
    )

    print(
        f"Proyectos originales    : "
        f"{comparacion_27.get('proyectos_original')}"
    )

    print(
        f"Proyectos optimizados   : "
        f"{comparacion_27.get('proyectos_optimizado')}"
    )

    print(
        f"Escenarios originales   : "
        f"{comparacion_27.get('escenarios_original')}"
    )

    print(
        f"Escenarios optimizados  : "
        f"{comparacion_27.get('escenarios_optimizado')}"
    )

    print(
        f"Asignaciones modificadas: "
        f"{comparacion_27.get('asignaciones_modificadas')}"
    )

    # -------------------------------------------------------------------------
    # 12
    # -------------------------------------------------------------------------

    imprimir_seccion(
        12,
        "COMPARANDO CONTRA PROCESO 28",
    )

    comparacion_28 = comparar_con_proceso_28(
        gdf,
        alertas,
    )

    print(
        f"Archivo disponible      : "
        f"{comparacion_28['archivo_disponible']}"
    )

    print(
        f"Score referencia 28     : "
        f"{comparacion_28.get('score_global_28')}"
    )

    # -------------------------------------------------------------------------
    # 13
    # -------------------------------------------------------------------------

    imprimir_seccion(
        13,
        "COMPARANDO CONTRA PROCESO 29",
    )

    # Score preliminar.
    score_previo = calcular_score_validacion(
        cobertura=cobertura["cobertura"],
        tamano=tamano["score_tamano"],
        cohesion=cohesion["score_cohesion"],
        indicadores=evaluacion_indicadores[
            "score_indicadores"
        ],
        balance=balance["score_balance"],
        estructura=1.0,
    )

    comparacion_29 = comparar_con_proceso_29(
        gdf,
        score_previo,
        alertas,
    )

    print(
        f"Archivo disponible      : "
        f"{comparacion_29['archivo_disponible']}"
    )

    print(
        f"Score proceso 29        : "
        f"{comparacion_29.get('score_global_29')}"
    )

    print(
        f"Score validación 30     : "
        f"{score_previo:.4f}"
    )

    if comparacion_29.get(
        "diferencia_score"
    ) is not None:

        print(
            f"Diferencia              : "
            f"{comparacion_29['diferencia_score']:+.4f}"
        )

    # -------------------------------------------------------------------------
    # 14
    # -------------------------------------------------------------------------

    imprimir_seccion(
        14,
        "CALCULANDO SCORE FINAL DE VALIDACIÓN",
    )

    estructura_score = 1.0

    if alertas.errores > 0:
        estructura_score = 0.0

    score_global = calcular_score_validacion(
        cobertura=cobertura["cobertura"],
        tamano=tamano["score_tamano"],
        cohesion=cohesion["score_cohesion"],
        indicadores=evaluacion_indicadores[
            "score_indicadores"
        ],
        balance=balance["score_balance"],
        estructura=estructura_score,
    )

    print(
        f"Cobertura             : "
        f"{cobertura['cobertura']:.4f}"
    )

    print(
        f"Tamaño                : "
        f"{tamano['score_tamano']:.4f}"
    )

    print(
        f"Cohesión              : "
        f"{cohesion['score_cohesion']:.4f}"
    )

    print(
        f"Indicadores           : "
        f"{evaluacion_indicadores['score_indicadores']:.4f}"
    )

    print(
        f"Balance               : "
        f"{balance['score_balance']:.4f}"
    )

    print()
    print(
        f"SCORE VALIDACIÓN      : "
        f"{score_global:.4f}"
    )

    clasificacion_score = clasificar_score(
        score_global
    )

    print(
        f"CLASIFICACIÓN SCORE    : "
        f"{clasificacion_score}"
    )

    # -------------------------------------------------------------------------
    # 15
    # -------------------------------------------------------------------------

    imprimir_seccion(
        15,
        "CONSTRUYENDO DETALLE DE ESCENARIOS",
    )

    detalle_escenarios = construir_detalle_escenarios(
        gdf,
        cohesion_detalle,
        indicadores,
    )

    detalle_escenarios = calcular_score_por_escenario(
        detalle_escenarios,
        indicadores,
    )

    print()

    columnas_print = [
        column
        for column in [
            "escenario_id",
            "cantidad_proyectos",
            "score_escenario",
            "score_tamano",
            "score_cohesion",
            "score_indicadores",
            "score_validacion_escenario",
            "ranking_validacion",
            "clasificacion_validacion",
        ]
        if column in detalle_escenarios.columns
    ]

    if columnas_print:

        print(
            detalle_escenarios[
                columnas_print
            ].to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # 16
    # -------------------------------------------------------------------------

    imprimir_seccion(
        16,
        "CONSTRUYENDO DICTAMEN FINAL",
    )

    dictamen, fundamento = construir_dictamen(
        score_global=score_global,
        alertas=alertas,
        cobertura=cobertura["cobertura"],
        escenarios=escenarios_validacion[
            "escenarios"
        ],
    )

    print(
        f"Dictamen                : "
        f"{dictamen}"
    )

    print(
        f"Fundamento              : "
        f"{fundamento}"
    )

    print()
    print(
        f"Errores                 : "
        f"{alertas.errores}"
    )

    print(
        f"Advertencias            : "
        f"{alertas.warnings}"
    )

    # -------------------------------------------------------------------------
    # 17
    # -------------------------------------------------------------------------

    imprimir_seccion(
        17,
        "EXPORTANDO RESULTADOS",
    )

    # -------------------------------------------------------------------------
    # Tabla de resumen
    # -------------------------------------------------------------------------

    resumen = pd.DataFrame(
        [
            {
                "proceso": 30,
                "version": VERSION,
                "dictamen": dictamen,
                "fundamento": fundamento,
                "score_validacion": score_global,
                "clasificacion_score": clasificacion_score,
                "proyectos_totales": cobertura[
                    "proyectos_totales"
                ],
                "proyectos_asignados": cobertura[
                    "proyectos_asignados"
                ],
                "proyectos_sin_escenario": cobertura[
                    "proyectos_sin_escenario"
                ],
                "cobertura": cobertura[
                    "cobertura"
                ],
                "escenarios": escenarios_validacion[
                    "escenarios"
                ],
                "minimo_proyectos": tamano[
                    "minimo_proyectos"
                ],
                "maximo_proyectos": tamano[
                    "maximo_proyectos"
                ],
                "promedio_proyectos": tamano[
                    "promedio_proyectos"
                ],
                "desvio_proyectos": tamano[
                    "desvio_proyectos"
                ],
                "score_tamano": tamano[
                    "score_tamano"
                ],
                "distancia_media_m": cohesion[
                    "distancia_media_m"
                ],
                "distancia_maxima_m": cohesion[
                    "distancia_maxima_m"
                ],
                "score_cohesion": cohesion[
                    "score_cohesion"
                ],
                "score_indicadores": evaluacion_indicadores[
                    "score_indicadores"
                ],
                "score_balance": balance[
                    "score_balance"
                ],
                "errores": alertas.errores,
                "warnings": alertas.warnings,
            }
        ]
    )

    # -------------------------------------------------------------------------
    # Comparación de procesos
    # -------------------------------------------------------------------------

    comparacion_filas = []

    comparacion_filas.append(
        {
            "proceso": "27",
            "archivo": str(
                INPUT_ORIGINAL
            ),
            "disponible": comparacion_27[
                "archivo_disponible"
            ],
            "proyectos": comparacion_27.get(
                "proyectos_original"
            ),
            "escenarios": comparacion_27.get(
                "escenarios_original"
            ),
            "score": None,
        }
    )

    comparacion_filas.append(
        {
            "proceso": "28",
            "archivo": str(
                INPUT_EVALUACION_28
                if INPUT_EVALUACION_28.exists()
                else INPUT_EVALUACION_28_CSV
            ),
            "disponible": comparacion_28[
                "archivo_disponible"
            ],
            "proyectos": None,
            "escenarios": comparacion_28.get(
                "escenarios_28"
            ),
            "score": comparacion_28.get(
                "score_global_28"
            ),
        }
    )

    comparacion_filas.append(
        {
            "proceso": "29",
            "archivo": str(
                INPUT_EVALUACION_29
            ),
            "disponible": comparacion_29[
                "archivo_disponible"
            ],
            "proyectos": len(gdf),
            "escenarios": (
                gdf["escenario_id"].nunique()
                if "escenario_id" in gdf.columns
                else None
            ),
            "score": comparacion_29.get(
                "score_global_29"
            ),
        }
    )

    comparacion_filas.append(
        {
            "proceso": "30",
            "archivo": str(
                OUTPUT_PARQUET
            ),
            "disponible": True,
            "proyectos": len(gdf),
            "escenarios": (
                gdf["escenario_id"].nunique()
                if "escenario_id" in gdf.columns
                else None
            ),
            "score": score_global,
        }
    )

    comparacion_df = pd.DataFrame(
        comparacion_filas
    )

    # -------------------------------------------------------------------------
    # Guardar
    # -------------------------------------------------------------------------

    resumen.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    guardar_csv(
        resumen,
        OUTPUT_CSV,
    )

    guardar_csv(
        detalle_escenarios,
        OUTPUT_DETALLE,
    )

    guardar_csv(
        comparacion_df,
        OUTPUT_COMPARACION,
    )

    alertas_df = alertas.dataframe()

    guardar_csv(
        alertas_df,
        OUTPUT_ALERTAS,
    )

    metadata = {
        "proceso": 30,
        "version": VERSION,
        "fecha_ejecucion": pd.Timestamp.now().isoformat(),
        "proyecto": str(PROJECT_ROOT),
        "entrada": str(INPUT_OPTIMIZADO),
        "salida_principal": str(OUTPUT_PARQUET),
        "crs_original": str(gdf.crs),
        "crs_metrico": CRS_METRICO,
        "proyectos": int(len(gdf)),
        "escenarios": int(
            gdf["escenario_id"].nunique()
            if "escenario_id" in gdf.columns
            else 0
        ),
        "cobertura": cobertura["cobertura"],
        "score_validacion": score_global,
        "clasificacion_score": clasificacion_score,
        "dictamen": dictamen,
        "fundamento": fundamento,
        "errores": alertas.errores,
        "warnings": alertas.warnings,
        "comparacion_proceso_27": comparacion_27,
        "comparacion_proceso_28": comparacion_28,
        "comparacion_proceso_29": comparacion_29,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # -------------------------------------------------------------------------
    # 18
    # -------------------------------------------------------------------------

    imprimir_seccion(
        18,
        "PROCESO 30 FINALIZADO",
    )

    duracion = (
        time.perf_counter()
        - inicio
    )

    print(
        f"Proyectos evaluados      : "
        f"{len(gdf)}"
    )

    print(
        f"Escenarios evaluados     : "
        f"{escenarios_validacion['escenarios']}"
    )

    print(
        f"Cobertura                : "
        f"{porcentaje(cobertura['cobertura']):.2f}%"
    )

    print(
        f"Score validación         : "
        f"{score_global:.4f}"
    )

    print(
        f"Clasificación            : "
        f"{clasificacion_score}"
    )

    print(
        f"Dictamen final           : "
        f"{dictamen}"
    )

    print(
        f"Errores                  : "
        f"{alertas.errores}"
    )

    print(
        f"Advertencias             : "
        f"{alertas.warnings}"
    )

    print(
        f"Duración                 : "
        f"{duracion:.2f} segundos"
    )

    print()
    print(
        "SALIDAS"
    )

    print(
        f"Resumen Parquet          : "
        f"{OUTPUT_PARQUET}"
    )

    print(
        f"Resumen CSV              : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Detalle escenarios       : "
        f"{OUTPUT_DETALLE}"
    )

    print(
        f"Comparación procesos     : "
        f"{OUTPUT_COMPARACION}"
    )

    print(
        f"Alertas                  : "
        f"{OUTPUT_ALERTAS}"
    )

    print(
        f"Metadata                 : "
        f"{OUTPUT_JSON}"
    )

    print()

    print(
        "DICTAMEN FINAL"
    )

    print(
        f"  {dictamen}"
    )

    print(
        f"  {fundamento}"
    )

    print()
    print(
        "=" * 88
    )

    # Un error de validación hace que el proceso devuelva código 2.
    # Esto permite usarlo posteriormente en automatizaciones/CI.
    if dictamen == "RECHAZADO":
        return 2

    return 0


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    sys.exit(
        main()
    )