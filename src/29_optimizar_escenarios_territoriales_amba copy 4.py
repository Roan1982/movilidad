# -*- coding: utf-8 -*-

"""
================================================================================
29 - OPTIMIZACIÓN DE ESCENARIOS TERRITORIALES AMBA - V3.0
================================================================================

Proceso:
    27 -> Construcción de escenarios territoriales
    28 -> Evaluación de escenarios
    29 -> Optimización de escenarios

OBJETIVO
--------
Optimizar la asignación territorial de proyectos mediante una estrategia
multiobjetivo basada en evaluación real del score global.

A diferencia de V2:
    - NO decide movimientos únicamente por distancia al centroide.
    - Simula cada movimiento candidato.
    - Recalcula las métricas globales.
    - Acepta únicamente movimientos que mejoran el score objetivo.
    - Mantiene cobertura total.
    - Mantiene identidad de proyectos.
    - Mantiene cantidad de escenarios.
    - Respeta un mínimo de proyectos por escenario.
    - Penaliza movimientos innecesarios.
    - Registra trazabilidad completa.

ENTRADAS
--------
    escenarios_territoriales_amba.parquet
    evaluacion_escenarios_territoriales_amba.parquet
    recomendaciones_escenarios_territoriales_amba.csv

SALIDAS
-------
    escenarios_territoriales_amba_optimizado.parquet
    escenarios_territoriales_amba_optimizado.csv
    evaluacion_escenarios_optimizada.csv
    movimientos_optimizacion_escenarios.csv
    resumen_optimizacion_escenarios.csv
    metadata_optimizacion_escenarios.json

================================================================================
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

VERSION = "V3.0"

MIN_PROYECTOS = 8

MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

MAX_ITERACIONES = 100

# --------------------------------------------------------------------------
# Pesos del score global
# --------------------------------------------------------------------------

PESO_COHESION = 0.30
PESO_BALANCE = 0.20
PESO_INDICADORES = 0.20
PESO_SCORE_ORIGINAL = 0.15
PESO_ESTABILIDAD = 0.15

PESO_COBERTURA = 0.15
PESO_ESTRUCTURA = 0.10
PESO_TAMANO = 0.20

RADIO_COHESION_METROS = 50_000.0

# --------------------------------------------------------------------------
# Criterios de aceptación
# --------------------------------------------------------------------------

MEJORA_MINIMA = 0.00005

# Penalización por mover proyectos.
# Es deliberadamente pequeña: solo evita movimientos marginales.
PENALIZACION_MOVIMIENTO = 0.00001

# Cantidad máxima de candidatos que se evalúan por proyecto.
# None = todos los escenarios.
MAX_ESCENARIOS_CANDIDATOS = None

# Evita que un escenario quede con pocos proyectos.
MIN_MARGEN_ORIGEN = MIN_PROYECTOS

# No permitimos que un escenario receptor tenga una diferencia excesiva
# respecto del escenario de origen.
MAX_DIFERENCIA_TAMANO = 12

# Si el score mejora menos que esto, se considera una mejora marginal.
MEJORA_REPORTABLE = 0.0001


# ==============================================================================
# RUTAS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_ESCENARIOS = (
    INPUT_DIR
    / "escenarios_territoriales_amba.parquet"
)

INPUT_EVALUACION = (
    INPUT_DIR
    / "evaluacion_escenarios_territoriales_amba.parquet"
)

INPUT_RECOMENDACIONES = (
    INPUT_DIR
    / "recomendaciones_escenarios_territoriales_amba.csv"
)

OUTPUT_OPTIMIZADO = (
    INPUT_DIR
    / "escenarios_territoriales_amba_optimizado.parquet"
)

OUTPUT_CSV = (
    INPUT_DIR
    / "escenarios_territoriales_amba_optimizado.csv"
)

OUTPUT_EVALUACION = (
    INPUT_DIR
    / "evaluacion_escenarios_optimizada.csv"
)

OUTPUT_MOVIMIENTOS = (
    INPUT_DIR
    / "movimientos_optimizacion_escenarios.csv"
)

OUTPUT_RESUMEN = (
    INPUT_DIR
    / "resumen_optimizacion_escenarios.csv"
)

OUTPUT_METADATA = (
    INPUT_DIR
    / "metadata_optimizacion_escenarios.json"
)


# ==============================================================================
# UTILIDADES
# ==============================================================================

def encabezado(titulo: str) -> None:
    print()
    print("=" * 96)
    print(titulo)
    print("=" * 96)


def subtitulo(titulo: str) -> None:
    print()
    print("-" * 80)
    print(titulo)
    print("-" * 80)


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: List[str],
    obligatoria: bool = True,
) -> Optional[str]:

    for columna in candidatos:
        if columna in df.columns:
            return columna

    if obligatoria:
        raise KeyError(
            f"No se encontró ninguna de las columnas requeridas: "
            f"{candidatos}"
        )

    return None


def normalizar_serie(
    serie: pd.Series,
) -> pd.Series:

    valores = pd.to_numeric(
        serie,
        errors="coerce",
    )

    if valores.notna().sum() == 0:
        return pd.Series(
            np.full(
                len(serie),
                0.5,
            ),
            index=serie.index,
            dtype=float,
        )

    minimo = valores.min()
    maximo = valores.max()

    if (
        pd.isna(minimo)
        or pd.isna(maximo)
    ):
        return pd.Series(
            np.full(
                len(serie),
                0.5,
            ),
            index=serie.index,
            dtype=float,
        )

    if math.isclose(
        float(minimo),
        float(maximo),
    ):
        return pd.Series(
            np.full(
                len(serie),
                0.5,
            ),
            index=serie.index,
            dtype=float,
        )

    resultado = (
        valores - minimo
    ) / (
        maximo - minimo
    )

    return resultado.fillna(0.5)


def safe_float(
    valor,
    default: float = 0.0,
) -> float:

    try:
        resultado = float(valor)

        if math.isfinite(resultado):
            return resultado

    except Exception:
        pass

    return default


# ==============================================================================
# CARGA
# ==============================================================================

def cargar_escenarios() -> gpd.GeoDataFrame:

    encabezado(
        "1. CARGANDO ESCENARIOS DEL PROCESO 27"
    )

    if not INPUT_ESCENARIOS.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{INPUT_ESCENARIOS}"
        )

    gdf = gpd.read_parquet(
        INPUT_ESCENARIOS
    )

    print(
        f"Archivo     : {INPUT_ESCENARIOS}"
    )

    print(
        f"Registros   : {len(gdf):,}"
    )

    print(
        f"Columnas    : {len(gdf.columns)}"
    )

    print(
        f"CRS         : {gdf.crs}"
    )

    if len(gdf) == 0:
        raise ValueError(
            "El GeoParquet no contiene registros."
        )

    return gdf


def cargar_evaluacion() -> Optional[pd.DataFrame]:

    subtitulo(
        "Cargando evaluación del proceso 28"
    )

    if not INPUT_EVALUACION.exists():

        print(
            "Evaluación del proceso 28 no encontrada."
        )

        return None

    evaluacion = pd.read_parquet(
        INPUT_EVALUACION
    )

    print(
        f"Archivo     : {INPUT_EVALUACION}"
    )

    print(
        f"Registros   : {len(evaluacion):,}"
    )

    return evaluacion


def cargar_recomendaciones() -> Optional[pd.DataFrame]:

    if not INPUT_RECOMENDACIONES.exists():

        print(
            "Recomendaciones del proceso 28 no encontradas."
        )

        return None

    try:

        recomendaciones = pd.read_csv(
            INPUT_RECOMENDACIONES,
            encoding="utf-8-sig",
        )

        print(
            f"Recomendaciones encontradas: "
            f"{len(recomendaciones)}"
        )

        return recomendaciones

    except Exception as exc:

        print(
            "No se pudieron cargar las recomendaciones:"
        )

        print(exc)

        return None


# ==============================================================================
# VALIDACIÓN
# ==============================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> Tuple[str, str]:

    encabezado(
        "2. VALIDANDO ENTRADA"
    )

    escenario_col = encontrar_columna(
        gdf,
        [
            "escenario_id",
            "id_escenario",
        ],
    )

    proyecto_col = encontrar_columna(
        gdf,
        [
            "proyecto_id",
            "id_proyecto",
        ],
    )

    if "geometry" not in gdf.columns:
        raise ValueError(
            "La entrada no contiene columna geometry."
        )

    geometria = gdf.geometry

    nulas = int(
        geometria.isna().sum()
    )

    vacias = int(
        geometria.is_empty.sum()
    )

    try:
        invalidas = int(
            (~geometria.is_valid).sum()
        )
    except Exception:
        invalidas = 0

    proyectos_nulos = int(
        gdf[proyecto_col].isna().sum()
    )

    escenarios_nulos = int(
        gdf[escenario_col].isna().sum()
    )

    duplicados_proyecto = int(
        gdf[proyecto_col].duplicated().sum()
    )

    print(
        f"Geometrías nulas       : {nulas}"
    )

    print(
        f"Geometrías vacías      : {vacias}"
    )

    print(
        f"Geometrías inválidas   : {invalidas}"
    )

    print(
        f"Proyectos nulos        : {proyectos_nulos}"
    )

    print(
        f"Escenarios nulos       : {escenarios_nulos}"
    )

    print(
        f"Escenarios             : "
        f"{gdf[escenario_col].nunique()}"
    )

    print(
        f"Proyectos              : "
        f"{gdf[proyecto_col].nunique()}"
    )

    print(
        f"Duplicados proyecto    : "
        f"{duplicados_proyecto}"
    )

    if nulas > 0:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if vacias > 0:
        raise ValueError(
            "Existen geometrías vacías."
        )

    if invalidas > 0:
        raise ValueError(
            "Existen geometrías inválidas."
        )

    if proyectos_nulos > 0:
        raise ValueError(
            "Existen proyectos sin identificador."
        )

    if escenarios_nulos > 0:
        raise ValueError(
            "Existen proyectos sin escenario."
        )

    if duplicados_proyecto > 0:
        raise ValueError(
            "Un mismo proyecto aparece más de una vez."
        )

    print(
        "Validación de entrada: OK"
    )

    return (
        escenario_col,
        proyecto_col,
    )


# ==============================================================================
# INDICADORES
# ==============================================================================

def detectar_indicadores(
    gdf: pd.DataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> List[str]:

    subtitulo(
        "Detectando indicadores territoriales"
    )

    excluir = {
        escenario_col,
        proyecto_col,
        "geometry",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
        "ranking_escenario",
        "ranking",
    }

    preferidos = [
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

    indicadores = []

    for columna in preferidos:

        if columna not in gdf.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            gdf[columna]
        ):
            continue

        indicadores.append(
            columna
        )

    if len(indicadores) < 3:

        for columna in gdf.columns:

            if columna in excluir:
                continue

            if columna.startswith(
                "ranking_"
            ):
                continue

            if columna.startswith(
                "nivel_"
            ):
                continue

            if not pd.api.types.is_numeric_dtype(
                gdf[columna]
            ):
                continue

            if columna not in indicadores:
                indicadores.append(
                    columna
                )

    print(
        f"Indicadores seleccionados: "
        f"{len(indicadores)}"
    )

    for indicador in indicadores:
        print(
            f"  - {indicador}"
        )

    score_col = encontrar_columna(
        gdf,
        [
            "score_prioridad_territorial",
            "score_cartera",
            "score_territorial",
            "score_prioridad",
        ],
        obligatoria=False,
    )

    print()

    print(
        "Score territorial utilizado: "
        f"{score_col if score_col else 'NO DISPONIBLE'}"
    )

    return indicadores


def construir_matriz_indicadores(
    gdf: pd.DataFrame,
    indicadores: List[str],
) -> np.ndarray:

    if not indicadores:
        return np.zeros(
            (
                len(gdf),
                1,
            ),
            dtype=float,
        )

    columnas = []

    for indicador in indicadores:

        columnas.append(
            normalizar_serie(
                gdf[indicador]
            )
            .fillna(0.5)
            .to_numpy(
                dtype=float
            )
        )

    return np.column_stack(
        columnas
    )


# ==============================================================================
# GEOMETRÍA
# ==============================================================================

def preparar_geometria_metrica(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    if gdf.crs is None:
        raise ValueError(
            "La capa no tiene CRS."
        )

    if gdf.crs.is_geographic:

        return gdf.to_crs(
            3857
        )

    return gdf


def obtener_centroides_escenarios(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> Dict:

    resultado = {}

    for escenario, grupo in gdf_metric.groupby(
        escenario_col
    ):

        if len(grupo) == 0:
            continue

        try:
            centroide = (
                grupo.geometry
                .union_all()
                .centroid
            )

        except AttributeError:

            centroide = (
                grupo.geometry
                .unary_union
                .centroid
            )

        resultado[
            escenario
        ] = centroide

    return resultado


# ==============================================================================
# MÉTRICAS
# ==============================================================================

def calcular_balance(
    cantidades: np.ndarray,
) -> float:

    if len(cantidades) == 0:
        return 0.0

    promedio = float(
        np.mean(cantidades)
    )

    if promedio <= 0:
        return 0.0

    cv = float(
        np.std(
            cantidades,
            ddof=0,
        )
        / promedio
    )

    return max(
        0.0,
        min(
            1.0,
            1.0 - cv,
        ),
    )


def calcular_cohesion(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> float:

    centroides = (
        gdf_metric.geometry.centroid
    )

    dispersiones = []

    for escenario in sorted(
        gdf_metric[
            escenario_col
        ]
        .dropna()
        .unique()
    ):

        mascara = (
            gdf_metric[
                escenario_col
            ]
            == escenario
        )

        puntos = centroides[
            mascara
        ]

        if len(puntos) < 2:
            continue

        try:

            centro = (
                puntos
                .union_all()
                .centroid
            )

        except AttributeError:

            centro = (
                puntos
                .unary_union
                .centroid
            )

        distancias = np.array(
            [
                float(
                    punto.distance(
                        centro
                    )
                )
                for punto in puntos
            ],
            dtype=float,
        )

        if len(distancias):
            dispersiones.append(
                float(
                    np.mean(
                        distancias
                    )
                )
            )

    if not dispersiones:
        return 0.0

    distancia_media = float(
        np.mean(
            dispersiones
        )
    )

    score = (
        1.0
        - (
            distancia_media
            / RADIO_COHESION_METROS
        )
    )

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def calcular_concentracion_indicadores(
    gdf: pd.DataFrame,
    escenario_col: str,
    indicadores: List[str],
) -> float:

    if not indicadores:
        return 0.0

    scores = []

    for indicador in indicadores:

        valores = pd.to_numeric(
            gdf[indicador],
            errors="coerce",
        )

        if valores.notna().sum() < 2:
            continue

        media_global = float(
            valores.mean()
        )

        if math.isclose(
            media_global,
            0.0,
        ):
            continue

        tmp = pd.DataFrame(
            {
                "__escenario":
                    gdf[
                        escenario_col
                    ],
                "__valor":
                    valores,
            }
        )

        medias = (
            tmp
            .groupby(
                "__escenario"
            )[
                "__valor"
            ]
            .mean()
        )

        dispersion = float(
            medias.std(
                ddof=0
            )
        )

        score = 1.0 / (
            1.0
            + abs(
                dispersion
                / (
                    abs(
                        media_global
                    )
                    + 1e-9
                )
            )
        )

        scores.append(
            score
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


def calcular_score_original(
    gdf: pd.DataFrame,
    score_col: Optional[str],
) -> float:

    if score_col is None:
        return 0.5

    valores = pd.to_numeric(
        gdf[score_col],
        errors="coerce",
    )

    if valores.notna().sum() == 0:
        return 0.5

    return float(
        normalizar_serie(
            valores
        )
        .mean()
    )


def calcular_estabilidad(
    gdf: pd.DataFrame,
    gdf_original: pd.DataFrame,
    proyecto_col: str,
    escenario_col: str,
) -> float:

    if len(gdf) == 0:
        return 0.0

    original = (
        gdf_original[
            [
                proyecto_col,
                escenario_col,
            ]
        ]
        .copy()
    )

    actual = (
        gdf[
            [
                proyecto_col,
                escenario_col,
            ]
        ]
        .copy()
    )

    comparacion = actual.merge(
        original,
        on=proyecto_col,
        how="left",
        suffixes=(
            "_actual",
            "_original",
        ),
    )

    iguales = (
        comparacion[
            f"{escenario_col}_actual"
        ]
        ==
        comparacion[
            f"{escenario_col}_original"
        ]
    )

    return float(
        iguales.mean()
    )


# ==============================================================================
# SCORE GLOBAL
# ==============================================================================

def evaluar_estructura(
    gdf: gpd.GeoDataFrame,
    gdf_original: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
) -> Dict[str, float]:

    gdf_metric = preparar_geometria_metrica(
        gdf
    )

    cantidades = (
        gdf
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .to_numpy()
    )

    cantidad_escenarios = len(
        cantidades
    )

    # --------------------------------------------------------------------------
    # Cobertura
    # --------------------------------------------------------------------------

    cobertura = 1.0

    # --------------------------------------------------------------------------
    # Estructura de escenarios
    # --------------------------------------------------------------------------

    if (
        MIN_ESCENARIOS
        <= cantidad_escenarios
        <= MAX_ESCENARIOS
    ):

        estructura = 1.0

    else:

        if cantidad_escenarios < MIN_ESCENARIOS:

            distancia = (
                MIN_ESCENARIOS
                - cantidad_escenarios
            )

        else:

            distancia = (
                cantidad_escenarios
                - MAX_ESCENARIOS
            )

        estructura = max(
            0.0,
            1.0
            - (
                0.25
                * distancia
            ),
        )

    # --------------------------------------------------------------------------
    # Tamaño
    # --------------------------------------------------------------------------

    if len(cantidades) == 0:

        tamano = 0.0

    else:

        tamano = float(
            np.mean(
                [
                    1.0
                    if n >= MIN_PROYECTOS
                    else (
                        n
                        / MIN_PROYECTOS
                    )
                    for n in cantidades
                ]
            )
        )

    # --------------------------------------------------------------------------
    # Balance
    # --------------------------------------------------------------------------

    balance = calcular_balance(
        cantidades
    )

    # --------------------------------------------------------------------------
    # Cohesión
    # --------------------------------------------------------------------------

    cohesion = calcular_cohesion(
        gdf_metric,
        escenario_col,
    )

    # --------------------------------------------------------------------------
    # Indicadores
    # --------------------------------------------------------------------------

    indicadores_score = (
        calcular_concentracion_indicadores(
            gdf,
            escenario_col,
            indicadores,
        )
    )

    # --------------------------------------------------------------------------
    # Score territorial original
    # --------------------------------------------------------------------------

    score_original = (
        calcular_score_original(
            gdf,
            score_col,
        )
    )

    # --------------------------------------------------------------------------
    # Estabilidad
    # --------------------------------------------------------------------------

    estabilidad = calcular_estabilidad(
        gdf,
        gdf_original,
        proyecto_col,
        escenario_col,
    )

    # --------------------------------------------------------------------------
    # Score global
    # --------------------------------------------------------------------------

    score_global = (
        PESO_COBERTURA
        * cobertura

        + PESO_ESTRUCTURA
        * estructura

        + PESO_TAMANO
        * tamano

        + PESO_COHESION
        * cohesion

        + PESO_BALANCE
        * balance

        + PESO_INDICADORES
        * indicadores_score

        + PESO_SCORE_ORIGINAL
        * score_original

        + PESO_ESTABILIDAD
        * estabilidad
    )

    return {
        "cobertura":
            float(cobertura),

        "estructura_escenarios":
            float(estructura),

        "tamano":
            float(tamano),

        "cohesion":
            float(cohesion),

        "balance":
            float(balance),

        "indicadores":
            float(indicadores_score),

        "score_original":
            float(score_original),

        "estabilidad":
            float(estabilidad),

        "score_global":
            float(
                max(
                    0.0,
                    min(
                        1.0,
                        score_global,
                    ),
                )
            ),
    }


# ==============================================================================
# RESTRICCIONES
# ==============================================================================

def movimiento_valido(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    idx,
    escenario_origen,
    escenario_destino,
) -> bool:

    if (
        escenario_origen
        == escenario_destino
    ):
        return False

    cantidad_origen = int(
        (
            gdf[
                escenario_col
            ]
            == escenario_origen
        ).sum()
    )

    cantidad_destino = int(
        (
            gdf[
                escenario_col
            ]
            == escenario_destino
        ).sum()
    )

    # Nunca vaciar ni dejar debajo del mínimo
    # el escenario de origen.
    if (
        cantidad_origen
        - 1
        < MIN_MARGEN_ORIGEN
    ):
        return False

    # El receptor debe existir.
    if cantidad_destino <= 0:
        return False

    # Evitar desequilibrios extremos.
    diferencia_despues = abs(
        (
            cantidad_origen
            - 1
        )
        - (
            cantidad_destino
            + 1
        )
    )

    if (
        diferencia_despues
        > MAX_DIFERENCIA_TAMANO
    ):
        return False

    return True


# ==============================================================================
# CANDIDATOS
# ==============================================================================

def generar_candidatos(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    gdf_metric: gpd.GeoDataFrame,
) -> List[Tuple]:

    escenarios = sorted(
        gdf[
            escenario_col
        ]
        .dropna()
        .unique()
    )

    centroides = (
        obtener_centroides_escenarios(
            gdf_metric,
            escenario_col,
        )
    )

    candidatos = []

    for idx, fila in gdf.iterrows():

        escenario_origen = fila[
            escenario_col
        ]

        if escenario_origen not in centroides:
            continue

        geometria = gdf_metric.loc[
            idx,
            "geometry",
        ]

        centro_origen = centroides[
            escenario_origen
        ]

        distancia_origen = float(
            geometria.distance(
                centro_origen
            )
        )

        posibles = []

        for escenario_destino in escenarios:

            if (
                escenario_destino
                == escenario_origen
            ):
                continue

            if (
                escenario_destino
                not in centroides
            ):
                continue

            if not movimiento_valido(
                gdf,
                escenario_col,
                proyecto_col,
                idx,
                escenario_origen,
                escenario_destino,
            ):
                continue

            distancia_destino = float(
                geometria.distance(
                    centroides[
                        escenario_destino
                    ]
                )
            )

            proximidad = (
                distancia_origen
                - distancia_destino
            )

            posibles.append(
                (
                    escenario_destino,
                    distancia_destino,
                    proximidad,
                )
            )

        posibles.sort(
            key=lambda x: x[1]
        )

        if (
            MAX_ESCENARIOS_CANDIDATOS
            is not None
        ):

            posibles = posibles[
                :MAX_ESCENARIOS_CANDIDATOS
            ]

        for (
            destino,
            distancia_destino,
            proximidad,
        ) in posibles:

            candidatos.append(
                (
                    idx,
                    escenario_origen,
                    destino,
                    distancia_origen,
                    distancia_destino,
                    proximidad,
                )
            )

    return candidatos


# ==============================================================================
# EVALUACIÓN DE MOVIMIENTO
# ==============================================================================

def evaluar_movimiento(
    gdf_actual: gpd.GeoDataFrame,
    gdf_original: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
    idx,
    escenario_origen,
    escenario_destino,
    score_actual: Dict[str, float],
) -> Optional[Dict]:

    if not movimiento_valido(
        gdf_actual,
        escenario_col,
        proyecto_col,
        idx,
        escenario_origen,
        escenario_destino,
    ):
        return None

    # --------------------------------------------------------------------------
    # Simulación
    # --------------------------------------------------------------------------

    gdf_simulado = gdf_actual.copy()

    gdf_simulado.loc[
        idx,
        escenario_col,
    ] = escenario_destino

    # --------------------------------------------------------------------------
    # Evaluación completa
    # --------------------------------------------------------------------------

    score_nuevo = evaluar_estructura(
        gdf_simulado,
        gdf_original,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
    )

    mejora_bruta = (
        score_nuevo[
            "score_global"
        ]
        - score_actual[
            "score_global"
        ]
    )

    mejora_neta = (
        mejora_bruta
        - PENALIZACION_MOVIMIENTO
    )

    return {
        "idx": idx,
        "proyecto_id":
            gdf_actual.loc[
                idx,
                proyecto_col,
            ],
        "escenario_origen":
            escenario_origen,
        "escenario_destino":
            escenario_destino,
        "score_antes":
            score_actual[
                "score_global"
            ],
        "score_despues":
            score_nuevo[
                "score_global"
            ],
        "mejora_bruta":
            mejora_bruta,
        "mejora_neta":
            mejora_neta,
        "metricas_antes":
            score_actual,
        "metricas_despues":
            score_nuevo,
    }


# ==============================================================================
# OPTIMIZACIÓN MULTIOBJETIVO
# ==============================================================================

def optimizar_multiobjetivo(
    gdf_original: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
) -> Tuple[
    gpd.GeoDataFrame,
    List[Dict],
    Dict[str, float],
]:

    encabezado(
        "5. OPTIMIZACIÓN MULTIOBJETIVO"
    )

    gdf_actual = (
        gdf_original.copy()
    )

    score_actual = evaluar_estructura(
        gdf_actual,
        gdf_original,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
    )

    score_inicial = (
        score_actual[
            "score_global"
        ]
    )

    print(
        f"Score inicial : {score_inicial:.6f}"
    )

    print(
        f"Escenarios    : "
        f"{gdf_actual[escenario_col].nunique()}"
    )

    movimientos = []

    for iteracion in range(
        1,
        MAX_ITERACIONES + 1,
    ):

        gdf_metric = (
            preparar_geometria_metrica(
                gdf_actual
            )
        )

        candidatos = generar_candidatos(
            gdf_actual,
            escenario_col,
            proyecto_col,
            gdf_metric,
        )

        if not candidatos:

            print(
                f"Iteración {iteracion:03d} "
                f"| sin candidatos"
            )

            break

        mejor_movimiento = None

        mejor_mejora = (
            MEJORA_MINIMA
        )

        # ----------------------------------------------------------------------
        # Evaluar candidatos
        # ----------------------------------------------------------------------

        for (
            idx,
            origen,
            destino,
            distancia_origen,
            distancia_destino,
            proximidad,
        ) in candidatos:

            resultado = evaluar_movimiento(
                gdf_actual,
                gdf_original,
                escenario_col,
                proyecto_col,
                indicadores,
                score_col,
                idx,
                origen,
                destino,
                score_actual,
            )

            if resultado is None:
                continue

            mejora = resultado[
                "mejora_neta"
            ]

            if mejora > mejor_mejora:

                mejor_mejora = mejora

                mejor_movimiento = (
                    resultado
                )

                mejor_movimiento[
                    "distancia_origen"
                ] = distancia_origen

                mejor_movimiento[
                    "distancia_destino"
                ] = distancia_destino

                mejor_movimiento[
                    "proximidad_centroidal"
                ] = proximidad

        # ----------------------------------------------------------------------
        # Sin mejora
        # ----------------------------------------------------------------------

        if mejor_movimiento is None:

            print(
                f"Iteración {iteracion:03d} "
                f"| candidatos evaluados: "
                f"{len(candidatos):,} "
                f"| sin mejora"
            )

            break

        # ----------------------------------------------------------------------
        # Aceptar movimiento
        # ----------------------------------------------------------------------

        idx = mejor_movimiento[
            "idx"
        ]

        origen = mejor_movimiento[
            "escenario_origen"
        ]

        destino = mejor_movimiento[
            "escenario_destino"
        ]

        proyecto = mejor_movimiento[
            "proyecto_id"
        ]

        score_antes = mejor_movimiento[
            "score_antes"
        ]

        score_despues = mejor_movimiento[
            "score_despues"
        ]

        gdf_actual.loc[
            idx,
            escenario_col,
        ] = destino

        movimiento = {
            "iteracion":
                iteracion,

            "proyecto_id":
                proyecto,

            "escenario_origen":
                origen,

            "escenario_destino":
                destino,

            "distancia_origen_m":
                mejor_movimiento[
                    "distancia_origen"
                ],

            "distancia_destino_m":
                mejor_movimiento[
                    "distancia_destino"
                ],

            "mejora_distancia_m":
                (
                    mejor_movimiento[
                        "distancia_origen"
                    ]
                    -
                    mejor_movimiento[
                        "distancia_destino"
                    ]
                ),

            "proximidad_centroidal":
                mejor_movimiento[
                    "proximidad_centroidal"
                ],

            "score_antes":
                score_antes,

            "score_despues":
                score_despues,

            "mejora_bruta":
                mejor_movimiento[
                    "mejora_bruta"
                ],

            "mejora_neta":
                mejor_movimiento[
                    "mejora_neta"
                ],

            "tipo_movimiento":
                "MULTIOBJETIVO",
        }

        # ----------------------------------------------------------------------
        # Registrar cambios de métricas
        # ----------------------------------------------------------------------

        metricas_antes = (
            mejor_movimiento[
                "metricas_antes"
            ]
        )

        metricas_despues = (
            mejor_movimiento[
                "metricas_despues"
            ]
        )

        for metrica in metricas_antes:

            movimiento[
                f"{metrica}_antes"
            ] = metricas_antes[
                metrica
            ]

            movimiento[
                f"{metrica}_despues"
            ] = metricas_despues[
                metrica
            ]

            movimiento[
                f"{metrica}_cambio"
            ] = (
                metricas_despues[
                    metrica
                ]
                -
                metricas_antes[
                    metrica
                ]
            )

        movimientos.append(
            movimiento
        )

        score_actual = (
            score_despues
            and metricas_despues
        )

        print(
            f"Iteración {iteracion:03d} "
            f"| proyecto={proyecto} "
            f"| {origen} -> {destino} "
            f"| score "
            f"{score_antes:.6f} -> "
            f"{score_despues:.6f} "
            f"| mejora "
            f"{mejor_movimiento['mejora_bruta']:+.6f}"
        )

    score_final = (
        evaluar_estructura(
            gdf_actual,
            gdf_original,
            escenario_col,
            proyecto_col,
            indicadores,
            score_col,
        )
    )

    print()
    print(
        f"Movimientos aceptados : "
        f"{len(movimientos)}"
    )

    print(
        f"Score final            : "
        f"{score_final['score_global']:.6f}"
    )

    print(
        f"Mejora total           : "
        f"{score_final['score_global'] - score_inicial:+.6f}"
    )

    return (
        gdf_actual,
        movimientos,
        score_final,
    )


# ==============================================================================
# EVALUACIÓN DETALLADA
# ==============================================================================

def evaluar_escenarios_detalladamente(
    gdf: gpd.GeoDataFrame,
    gdf_original: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
) -> pd.DataFrame:

    encabezado(
        "6. EVALUACIÓN DETALLADA DE ESCENARIOS"
    )

    filas = []

    conteos = (
        gdf
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .sort_values(
            ascending=False
        )
    )

    ranking = 0

    for escenario, cantidad in conteos.items():

        ranking += 1

        grupo = gdf[
            gdf[
                escenario_col
            ]
            == escenario
        ]

        metricas = evaluar_estructura(
            gdf,
            gdf_original,
            escenario_col,
            proyecto_col,
            indicadores,
            score_col,
        )

        gdf_metric = (
            preparar_geometria_metrica(
                gdf
            )
        )

        grupo_metric = (
            gdf_metric[
                gdf[
                    escenario_col
                ]
                == escenario
            ]
        )

        if len(grupo_metric) > 0:

            try:

                centro = (
                    grupo_metric.geometry
                    .union_all()
                    .centroid
                )

            except AttributeError:

                centro = (
                    grupo_metric.geometry
                    .unary_union
                    .centroid
                )

            distancias = (
                grupo_metric.geometry
                .distance(
                    centro
                )
            )

            distancia_media = float(
                distancias.mean()
            )

            distancia_maxima = float(
                distancias.max()
            )

        else:

            distancia_media = 0.0
            distancia_maxima = 0.0

        score_territorial = (
            calcular_score_original(
                grupo,
                score_col,
            )
        )

        filas.append(
            {
                "escenario_id":
                    escenario,

                "cantidad_proyectos":
                    int(cantidad),

                "score_territorial":
                    score_territorial,

                "distancia_media_centroide_m":
                    distancia_media,

                "distancia_maxima_centroide_m":
                    distancia_maxima,

                "ranking_escenario":
                    ranking,

                "cumple_minimo":
                    bool(
                        cantidad
                        >= MIN_PROYECTOS
                    ),

                "score_global":
                    metricas[
                        "score_global"
                    ],
            }
        )

    resultado = pd.DataFrame(
        filas
    )

    return resultado


# ==============================================================================
# VALIDACIÓN
# ==============================================================================

def validar_resultado(
    gdf_original: gpd.GeoDataFrame,
    gdf_optimizado: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> Dict:

    encabezado(
        "7. VALIDACIÓN FINAL"
    )

    proyectos_originales = set(
        gdf_original[
            proyecto_col
        ]
        .astype(str)
    )

    proyectos_optimizados = set(
        gdf_optimizado[
            proyecto_col
        ]
        .astype(str)
    )

    escenarios_originales = set(
        gdf_original[
            escenario_col
        ]
        .astype(str)
    )

    escenarios_optimizados = set(
        gdf_optimizado[
            escenario_col
        ]
        .astype(str)
    )

    duplicados = int(
        gdf_optimizado[
            proyecto_col
        ]
        .duplicated()
        .sum()
    )

    sin_escenario = int(
        gdf_optimizado[
            escenario_col
        ]
        .isna()
        .sum()
    )

    conteos = (
        gdf_optimizado
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
    )

    if len(conteos) > 0:

        minimo = int(
            conteos.min()
        )

        maximo = int(
            conteos.max()
        )

    else:

        minimo = 0
        maximo = 0

    validaciones = {
        "mismos_proyectos":
            proyectos_originales
            == proyectos_optimizados,

        "mismos_escenarios":
            escenarios_originales
            == escenarios_optimizados,

        "duplicados_proyecto":
            duplicados,

        "proyectos_sin_escenario":
            sin_escenario,

        "escenarios":
            int(
                gdf_optimizado[
                    escenario_col
                ]
                .nunique()
            ),

        "min_proyectos":
            minimo,

        "max_proyectos":
            maximo,

        "proyectos_totales":
            int(
                len(
                    gdf_original
                )
            ),

        "proyectos_optimizados":
            int(
                len(
                    gdf_optimizado
                )
            ),
    }

    print(
        f"Proyectos originales    : "
        f"{validaciones['proyectos_totales']}"
    )

    print(
        f"Proyectos optimizados   : "
        f"{validaciones['proyectos_optimizados']}"
    )

    print(
        f"Mismos proyectos        : "
        f"{validaciones['mismos_proyectos']}"
    )

    print(
        f"Mismos escenarios       : "
        f"{validaciones['mismos_escenarios']}"
    )

    print(
        f"Duplicados proyecto     : "
        f"{validaciones['duplicados_proyecto']}"
    )

    print(
        f"Sin escenario           : "
        f"{validaciones['proyectos_sin_escenario']}"
    )

    print(
        f"Escenarios              : "
        f"{validaciones['escenarios']}"
    )

    print(
        f"Mínimo proyectos       : "
        f"{validaciones['min_proyectos']}"
    )

    print(
        f"Máximo proyectos       : "
        f"{validaciones['max_proyectos']}"
    )

    if not validaciones[
        "mismos_proyectos"
    ]:

        raise ValueError(
            "La optimización alteró "
            "el conjunto de proyectos."
        )

    if not validaciones[
        "mismos_escenarios"
    ]:

        raise ValueError(
            "La optimización alteró "
            "el conjunto de escenarios."
        )

    if validaciones[
        "duplicados_proyecto"
    ] > 0:

        raise ValueError(
            "La optimización generó "
            "proyectos duplicados."
        )

    if validaciones[
        "proyectos_sin_escenario"
    ] > 0:

        raise ValueError(
            "Existen proyectos "
            "sin escenario."
        )

    if validaciones[
        "min_proyectos"
    ] < MIN_PROYECTOS:

        raise ValueError(
            "Existe un escenario "
            "por debajo del mínimo permitido."
        )

    print()
    print(
        "Validación final: OK"
    )

    return validaciones


# ==============================================================================
# EXPORTACIÓN
# ==============================================================================

def exportar_resultados(
    gdf_optimizado: gpd.GeoDataFrame,
    evaluacion: pd.DataFrame,
    movimientos: List[Dict],
    metricas_originales: Dict,
    metricas_optimizadas: Dict,
    validacion: Dict,
    indicadores: List[str],
    tiempo: float,
) -> None:

    encabezado(
        "8. EXPORTANDO RESULTADOS"
    )

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------------------
    # GeoParquet
    # --------------------------------------------------------------------------

    gdf_optimizado.to_parquet(
        OUTPUT_OPTIMIZADO,
        index=False,
    )

    # --------------------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------------------

    gdf_export = gdf_optimizado.copy()

    if "geometry" in gdf_export.columns:

        gdf_export = pd.DataFrame(
            gdf_export.drop(
                columns=[
                    "geometry"
                ]
            )
        )

    gdf_export.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------------------
    # Evaluación
    # --------------------------------------------------------------------------

    evaluacion.to_csv(
        OUTPUT_EVALUACION,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------------------
    # Movimientos
    # --------------------------------------------------------------------------

    movimientos_df = pd.DataFrame(
        movimientos
    )

    if movimientos_df.empty:

        movimientos_df = pd.DataFrame(
            columns=[
                "iteracion",
                "proyecto_id",
                "escenario_origen",
                "escenario_destino",
                "score_antes",
                "score_despues",
                "mejora_bruta",
                "mejora_neta",
                "tipo_movimiento",
            ]
        )

    movimientos_df.to_csv(
        OUTPUT_MOVIMIENTOS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------------------
    # Resumen
    # --------------------------------------------------------------------------

    filas_resumen = []

    for clave in metricas_originales:

        original = (
            metricas_originales[
                clave
            ]
        )

        optimizado = (
            metricas_optimizadas[
                clave
            ]
        )

        filas_resumen.append(
            {
                "metrica":
                    clave,

                "original":
                    original,

                "optimizado":
                    optimizado,

                "cambio":
                    optimizado
                    - original,
            }
        )

    resumen = pd.DataFrame(
        filas_resumen
    )

    resumen.to_csv(
        OUTPUT_RESUMEN,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------------------

    escenario_col = (
        "escenario_id"
        if "escenario_id"
        in gdf_optimizado.columns
        else "id_escenario"
    )

    proyecto_col = (
        "proyecto_id"
        if "proyecto_id"
        in gdf_optimizado.columns
        else "id_proyecto"
    )

    conteos = (
        gdf_optimizado
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .sort_values(
            ascending=False
        )
    )

    metadata = {
        "proceso": 29,

        "nombre":
            "Optimización de escenarios "
            "territoriales AMBA",

        "version":
            VERSION,

        "fecha_ejecucion":
            pd.Timestamp.now(
                tz="America/Argentina/Buenos_Aires"
            ).isoformat(),

        "proyectos":
            int(
                len(
                    gdf_optimizado
                )
            ),

        "escenarios":
            int(
                gdf_optimizado[
                    escenario_col
                ]
                .nunique()
            ),

        "distribucion_escenarios":
            {
                str(
                    indice
                ):
                    int(
                        valor
                    )
                for indice, valor
                in conteos.items()
            },

        "movimientos":
            int(
                len(
                    movimientos
                )
            ),

        "indicadores":
            indicadores,

        "configuracion": {
            "min_proyectos":
                MIN_PROYECTOS,

            "min_escenarios":
                MIN_ESCENARIOS,

            "max_escenarios":
                MAX_ESCENARIOS,

            "max_iteraciones":
                MAX_ITERACIONES,

            "mejora_minima":
                MEJORA_MINIMA,

            "penalizacion_movimiento":
                PENALIZACION_MOVIMIENTO,

            "radio_cohesion_m":
                RADIO_COHESION_METROS,

            "peso_cohesion":
                PESO_COHESION,

            "peso_balance":
                PESO_BALANCE,

            "peso_indicadores":
                PESO_INDICADORES,

            "peso_score_original":
                PESO_SCORE_ORIGINAL,

            "peso_estabilidad":
                PESO_ESTABILIDAD,
        },

        "metricas_originales":
            metricas_originales,

        "metricas_optimizadas":
            metricas_optimizadas,

        "mejora_global":
            (
                metricas_optimizadas[
                    "score_global"
                ]
                -
                metricas_originales[
                    "score_global"
                ]
            ),

        "validacion":
            validacion,

        "duracion_segundos":
            tiempo,
    }

    OUTPUT_METADATA.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"GeoParquet : {OUTPUT_OPTIMIZADO}"
    )

    print(
        f"CSV        : {OUTPUT_CSV}"
    )

    print(
        f"Evaluación : {OUTPUT_EVALUACION}"
    )

    print(
        f"Movimientos: {OUTPUT_MOVIMIENTOS}"
    )

    print(
        f"Resumen    : {OUTPUT_RESUMEN}"
    )

    print(
        f"Metadata   : {OUTPUT_METADATA}"
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:

    inicio = time.perf_counter()

    encabezado(
        f"29 - OPTIMIZACIÓN DE ESCENARIOS "
        f"TERRITORIALES AMBA - {VERSION}"
    )

    print(
        f"Proyecto : {PROJECT_ROOT}"
    )

    print(
        f"Entrada  : {INPUT_ESCENARIOS}"
    )

    print(
        f"Salida   : {INPUT_DIR}"
    )

    subtitulo(
        "Configuración"
    )

    print(
        f"Mínimo proyectos          : "
        f"{MIN_PROYECTOS}"
    )

    print(
        f"Rango escenarios          : "
        f"{MIN_ESCENARIOS} - "
        f"{MAX_ESCENARIOS}"
    )

    print(
        f"Máximo iteraciones        : "
        f"{MAX_ITERACIONES}"
    )

    print(
        f"Mejora mínima             : "
        f"{MEJORA_MINIMA}"
    )

    print(
        f"Penalización movimiento   : "
        f"{PENALIZACION_MOVIMIENTO}"
    )

    print(
        f"Radio cohesión            : "
        f"{RADIO_COHESION_METROS:,.0f} m"
    )

    print()
    print(
        "PESOS DE OPTIMIZACIÓN"
    )

    print(
        f"  Cohesión territorial    : "
        f"{PESO_COHESION:.0%}"
    )

    print(
        f"  Balance                 : "
        f"{PESO_BALANCE:.0%}"
    )

    print(
        f"  Indicadores             : "
        f"{PESO_INDICADORES:.0%}"
    )

    print(
        f"  Score original          : "
        f"{PESO_SCORE_ORIGINAL:.0%}"
    )

    print(
        f"  Estabilidad             : "
        f"{PESO_ESTABILIDAD:.0%}"
    )

    print()
    print(
        "COMPONENTES ESTRUCTURALES"
    )

    print(
        f"  Cobertura               : "
        f"{PESO_COBERTURA:.0%}"
    )

    print(
        f"  Estructura escenarios   : "
        f"{PESO_ESTRUCTURA:.0%}"
    )

    print(
        f"  Tamaño                  : "
        f"{PESO_TAMANO:.0%}"
    )

    # ==========================================================================
    # 1
    # ==========================================================================

    gdf_original = (
        cargar_escenarios()
    )

    # ==========================================================================
    # 2
    # ==========================================================================

    evaluacion_28 = (
        cargar_evaluacion()
    )

    recomendaciones_28 = (
        cargar_recomendaciones()
    )

    # ==========================================================================
    # 3
    # ==========================================================================

    (
        escenario_col,
        proyecto_col,
    ) = validar_entrada(
        gdf_original
    )

    # ==========================================================================
    # 4
    # ==========================================================================

    encabezado(
        "3. PREPARANDO INDICADORES"
    )

    indicadores = (
        detectar_indicadores(
            gdf_original,
            escenario_col,
            proyecto_col,
        )
    )

    score_col = encontrar_columna(
        gdf_original,
        [
            "score_prioridad_territorial",
            "score_cartera",
            "score_territorial",
            "score_prioridad",
        ],
        obligatoria=False,
    )

    if evaluacion_28 is not None:

        print(
            "Evaluación del proceso 28: "
            "DISPONIBLE"
        )

    else:

        print(
            "Evaluación del proceso 28: "
            "NO DISPONIBLE"
        )

    if recomendaciones_28 is not None:

        print(
            "Recomendaciones proceso 28: "
            "DISPONIBLES"
        )

    else:

        print(
            "Recomendaciones proceso 28: "
            "NO DISPONIBLES"
        )

    # ==========================================================================
    # 5
    # ==========================================================================

    encabezado(
        "4. EVALUACIÓN BASE"
    )

    metricas_originales = (
        evaluar_estructura(
            gdf_original,
            gdf_original,
            escenario_col,
            proyecto_col,
            indicadores,
            score_col,
        )
    )

    for clave, valor in (
        metricas_originales.items()
    ):

        print(
            f"{clave:<25}: "
            f"{valor:.6f}"
        )

    # ==========================================================================
    # 6
    # ==========================================================================

    (
        gdf_optimizado,
        movimientos,
        metricas_optimizadas,
    ) = optimizar_multiobjetivo(
        gdf_original,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
    )

    # ==========================================================================
    # 7
    # ==========================================================================

    evaluacion = (
        evaluar_escenarios_detalladamente(
            gdf_optimizado,
            gdf_original,
            escenario_col,
            proyecto_col,
            indicadores,
            score_col,
        )
    )

    # ==========================================================================
    # 8
    # ==========================================================================

    validacion = validar_resultado(
        gdf_original,
        gdf_optimizado,
        escenario_col,
        proyecto_col,
    )

    # ==========================================================================
    # 9
    # ==========================================================================

    tiempo = (
        time.perf_counter()
        - inicio
    )

    exportar_resultados(
        gdf_optimizado,
        evaluacion,
        movimientos,
        metricas_originales,
        metricas_optimizadas,
        validacion,
        indicadores,
        tiempo,
    )

    # ==========================================================================
    # 10
    # ==========================================================================

    encabezado(
        "10. PROCESO 29 FINALIZADO CORRECTAMENTE"
    )

    print(
        f"Proyectos procesados     : "
        f"{len(gdf_optimizado)}"
    )

    print(
        f"Escenarios               : "
        f"{gdf_optimizado[escenario_col].nunique()}"
    )

    print(
        f"Movimientos realizados   : "
        f"{len(movimientos)}"
    )

    print()
    print(
        f"Score global original    : "
        f"{metricas_originales['score_global']:.6f}"
    )

    print(
        f"Score global optimizado  : "
        f"{metricas_optimizadas['score_global']:.6f}"
    )

    mejora = (
        metricas_optimizadas[
            "score_global"
        ]
        -
        metricas_originales[
            "score_global"
        ]
    )

    print(
        f"Mejora global            : "
        f"{mejora:+.6f}"
    )

    print(
        f"Duración                 : "
        f"{tiempo:.2f} segundos"
    )

    print()
    print(
        "DISTRIBUCIÓN FINAL DE ESCENARIOS"
    )

    conteos = (
        gdf_optimizado
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .sort_values(
            ascending=False
        )
    )

    print(
        conteos.to_string()
    )

    print()
    print(
        "CONTROL DE OPTIMIZACIÓN"
    )

    if mejora > MEJORA_REPORTABLE:

        print(
            "Resultado: MEJORA REAL"
        )

    elif mejora > 0:

        print(
            "Resultado: MEJORA MARGINAL"
        )

    else:

        print(
            "Resultado: SIN MEJORA"
        )

    print()
    print(
        "Salida principal:"
    )

    print(
        OUTPUT_OPTIMIZADO
    )

    print()
    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()