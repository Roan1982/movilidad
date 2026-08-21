# -*- coding: utf-8 -*-

"""
================================================================================
29 - OPTIMIZACIÓN DE ESCENARIOS TERRITORIALES AMBA
================================================================================
VERSIÓN 2.0
================================================================================

OBJETIVO
--------
Optimizar la asignación territorial de proyectos construida en el proceso 27,
utilizando como referencia la evaluación del proceso 28.

La V2.0 reemplaza el enfoque de "mover proyectos por distancia" por una
optimización multiobjetivo basada en:

    1. Cobertura de proyectos
    2. Cantidad de escenarios
    3. Mínimo de proyectos por escenario
    4. Balance de tamaños
    5. Cohesión territorial
    6. Coherencia de indicadores
    7. Preservación del score territorial original
    8. Estabilidad de la asignación
    9. Penalización por movimientos innecesarios

PRINCIPIOS
----------
- No se crean proyectos.
- No se eliminan proyectos.
- No se crean escenarios nuevos.
- No se eliminan escenarios.
- Se conserva el conjunto original de escenarios.
- Ningún escenario puede quedar por debajo de MIN_PROYECTOS.
- Cada movimiento debe mejorar la función objetivo global.
- Se registra cada movimiento.
- Se compara explícitamente el estado original con el optimizado.
- El proceso es reproducible y trazable.

ENTRADAS
--------
data/processed/escenarios_territoriales_amba/
    escenarios_territoriales_amba.parquet
    evaluacion_escenarios_territoriales_amba.parquet
    recomendaciones_escenarios_territoriales_amba.csv   [opcional]

SALIDAS
-------
data/processed/escenarios_territoriales_amba/

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
from typing import Dict, List, Optional, Tuple, Any

import geopandas as gpd
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

VERSION = "V2.0"

# --------------------------------------------------------------------------
# Restricciones estructurales
# --------------------------------------------------------------------------

MIN_PROYECTOS = 8

MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

# --------------------------------------------------------------------------
# Optimización
# --------------------------------------------------------------------------

MAX_ITERACIONES = 100

# Máximo de candidatos evaluados por iteración.
# None = todos.
MAX_CANDIDATOS_POR_ITERACION = None

# Mejora mínima absoluta requerida para aceptar un movimiento.
MEJORA_MINIMA = 0.0005

# --------------------------------------------------------------------------
# Pesos de la función objetivo
#
# SUMA = 1.00
# --------------------------------------------------------------------------

PESO_COHESION = 0.30
PESO_BALANCE = 0.20
PESO_INDICADORES = 0.20
PESO_SCORE_ORIGINAL = 0.15
PESO_ESTABILIDAD = 0.15

SUMA_PESOS = (
    PESO_COHESION
    + PESO_BALANCE
    + PESO_INDICADORES
    + PESO_SCORE_ORIGINAL
    + PESO_ESTABILIDAD
)

# --------------------------------------------------------------------------
# Cohesión territorial
# --------------------------------------------------------------------------

RADIO_COHESION_METROS = 50_000.0

# --------------------------------------------------------------------------
# Penalización por movimiento
#
# Se aplica mediante estabilidad, no directamente como costo absoluto.
# --------------------------------------------------------------------------

PENALIZACION_MOVIMIENTO = 0.002

# --------------------------------------------------------------------------
# Indicadores
# --------------------------------------------------------------------------

INDICADORES_PREFERIDOS = [
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
# UTILIDADES GENERALES
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
            "\nNo se encontró ninguna de las columnas requeridas.\n"
            f"Candidatos: {candidatos}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    return None


def convertir_numero(
    serie: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def normalizar_global(
    serie: pd.Series,
) -> pd.Series:

    valores = convertir_numero(serie)

    validos = valores.dropna()

    if validos.empty:
        return pd.Series(
            0.5,
            index=serie.index,
            dtype=float,
        )

    minimo = float(validos.min())
    maximo = float(validos.max())

    if math.isclose(
        minimo,
        maximo,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return pd.Series(
            0.5,
            index=serie.index,
            dtype=float,
        )

    resultado = (
        valores - minimo
    ) / (
        maximo - minimo
    )

    return resultado.fillna(0.5).astype(float)


def geometria_valida(
    gdf: gpd.GeoDataFrame,
) -> bool:

    if "geometry" not in gdf.columns:
        return False

    if gdf.geometry.isna().any():
        return False

    if gdf.geometry.is_empty.any():
        return False

    try:
        return bool(
            gdf.geometry.is_valid.all()
        )
    except Exception:
        return False


# ==============================================================================
# CARGA
# ==============================================================================

def cargar_escenarios() -> gpd.GeoDataFrame:

    encabezado(
        "1. CARGANDO ESCENARIOS DEL PROCESO 27"
    )

    if not INPUT_ESCENARIOS.exists():
        raise FileNotFoundError(
            "No existe el archivo principal:\n"
            f"{INPUT_ESCENARIOS}"
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

    if gdf.empty:
        raise ValueError(
            "El GeoParquet de escenarios está vacío."
        )

    return gdf


def cargar_evaluacion() -> Optional[pd.DataFrame]:

    subtitulo(
        "Cargando evaluación del proceso 28"
    )

    if not INPUT_EVALUACION.exists():
        print(
            "ADVERTENCIA: no se encontró la evaluación del proceso 28."
        )
        print(
            "El proceso continuará utilizando los indicadores disponibles."
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
    print(
        f"Columnas    : {len(evaluacion.columns)}"
    )

    return evaluacion


def cargar_recomendaciones() -> Optional[pd.DataFrame]:

    if not INPUT_RECOMENDACIONES.exists():
        return None

    try:
        recomendaciones = pd.read_csv(
            INPUT_RECOMENDACIONES,
            encoding="utf-8-sig",
        )

        print(
            f"Recomendaciones encontradas: "
            f"{len(recomendaciones):,}"
        )

        return recomendaciones

    except Exception as exc:

        print(
            "ADVERTENCIA: no se pudieron cargar "
            "las recomendaciones."
        )

        print(
            f"Detalle: {exc}"
        )

        return None


# ==============================================================================
# VALIDACIÓN DE ENTRADA
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

    if gdf.crs is None:
        raise ValueError(
            "La capa no posee CRS."
        )

    nulas = int(
        gdf.geometry.isna().sum()
    )

    vacias = int(
        gdf.geometry.is_empty.sum()
    )

    try:
        invalidas = int(
            (~gdf.geometry.is_valid).sum()
        )
    except Exception:
        invalidas = 0

    duplicados = int(
        gdf[proyecto_col]
        .duplicated()
        .sum()
    )

    proyectos_nulos = int(
        gdf[proyecto_col]
        .isna()
        .sum()
    )

    escenarios_nulos = int(
        gdf[escenario_col]
        .isna()
        .sum()
    )

    escenarios = int(
        gdf[escenario_col]
        .nunique()
    )

    proyectos = int(
        gdf[proyecto_col]
        .nunique()
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
        f"Escenarios             : {escenarios}"
    )

    print(
        f"Proyectos              : {proyectos}"
    )

    print(
        f"Duplicados proyecto    : {duplicados}"
    )

    if not geometria_valida(gdf):
        raise ValueError(
            "La geometría de entrada contiene problemas."
        )

    if proyectos_nulos > 0:
        raise ValueError(
            "Existen proyectos sin identificador."
        )

    if escenarios_nulos > 0:
        raise ValueError(
            "Existen proyectos sin escenario."
        )

    if duplicados > 0:
        raise ValueError(
            "Un mismo proyecto aparece más de una vez."
        )

    if escenarios < MIN_ESCENARIOS:
        print(
            "ADVERTENCIA: la entrada posee menos escenarios "
            f"que el mínimo recomendado ({MIN_ESCENARIOS})."
        )

    if escenarios > MAX_ESCENARIOS:
        print(
            "ADVERTENCIA: la entrada posee más escenarios "
            f"que el máximo recomendado ({MAX_ESCENARIOS})."
        )

    print(
        "Validación de entrada: OK"
    )

    return escenario_col, proyecto_col


# ==============================================================================
# PREPARACIÓN DE INDICADORES
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
        "ranking_global",
    }

    indicadores: List[str] = []

    for columna in INDICADORES_PREFERIDOS:

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

            if columna in indicadores:
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

            indicadores.append(
                columna
            )

    print(
        f"Indicadores seleccionados: {len(indicadores)}"
    )

    for indicador in indicadores:
        print(
            f"  - {indicador}"
        )

    if not indicadores:
        print(
            "ADVERTENCIA: no se encontraron indicadores numéricos."
        )

    return indicadores


def preparar_indicadores(
    gdf: pd.DataFrame,
    indicadores: List[str],
) -> pd.DataFrame:

    matriz = pd.DataFrame(
        index=gdf.index
    )

    for indicador in indicadores:

        matriz[indicador] = normalizar_global(
            gdf[indicador]
        )

    if matriz.empty:
        matriz["__sin_indicadores__"] = 0.5

    return matriz


# ==============================================================================
# GEOMETRÍA MÉTRICA
# ==============================================================================

def preparar_geometria_metrica(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    if gdf.crs is None:
        raise ValueError(
            "No se puede calcular distancia "
            "sin CRS."
        )

    if gdf.crs.is_geographic:

        return gdf.to_crs(
            3857
        )

    return gdf


# ==============================================================================
# CENTROIDES
# ==============================================================================

def calcular_centroides(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> Dict[Any, Any]:

    centroides: Dict[Any, Any] = {}

    for escenario, grupo in gdf_metric.groupby(
        escenario_col,
        sort=False,
    ):

        if grupo.empty:
            continue

        centroide = (
            grupo.geometry
            .unary_union
            .centroid
        )

        centroides[
            escenario
        ] = centroide

    return centroides


# ==============================================================================
# MÉTRICA DE BALANCE
# ==============================================================================

def calcular_balance(
    cantidades: np.ndarray,
) -> float:

    if len(cantidades) == 0:
        return 0.0

    cantidades = np.asarray(
        cantidades,
        dtype=float,
    )

    promedio = float(
        cantidades.mean()
    )

    if promedio <= 0:
        return 0.0

    cv = float(
        cantidades.std(
            ddof=0
        )
        / promedio
    )

    # CV = 0 => 1.0
    # CV >= 1 => 0.0
    score = 1.0 - cv

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ==============================================================================
# MÉTRICA DE TAMAÑO MÍNIMO
# ==============================================================================

def calcular_tamano(
    cantidades: np.ndarray,
) -> float:

    if len(cantidades) == 0:
        return 0.0

    scores = []

    for cantidad in cantidades:

        if cantidad >= MIN_PROYECTOS:
            scores.append(
                1.0
            )
        else:
            scores.append(
                float(cantidad)
                / float(MIN_PROYECTOS)
            )

    return float(
        np.mean(scores)
    )


# ==============================================================================
# MÉTRICA DE ESTRUCTURA DE ESCENARIOS
# ==============================================================================

def calcular_estructura_escenarios(
    cantidad_escenarios: int,
) -> float:

    if (
        MIN_ESCENARIOS
        <= cantidad_escenarios
        <= MAX_ESCENARIOS
    ):
        return 1.0

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

    return float(
        np.clip(
            1.0
            - (
                0.25
                * distancia
            ),
            0.0,
            1.0,
        )
    )


# ==============================================================================
# COHESIÓN TERRITORIAL
# ==============================================================================

def calcular_cohesion(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> float:

    centroides = (
        gdf_metric.geometry.centroid
    )

    scores = []

    for escenario, grupo in gdf_metric.groupby(
        escenario_col,
        sort=False,
    ):

        if len(grupo) <= 1:
            scores.append(
                1.0
            )
            continue

        puntos = centroides.loc[
            grupo.index
        ]

        centro = (
            puntos.unary_union
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

        if len(distancias) == 0:
            continue

        distancia_media = float(
            np.mean(
                distancias
            )
        )

        score = 1.0 - (
            distancia_media
            / RADIO_COHESION_METROS
        )

        scores.append(
            float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


# ==============================================================================
# COHERENCIA DE INDICADORES
# ==============================================================================

def calcular_coherencia_indicadores(
    gdf: pd.DataFrame,
    escenario_col: str,
    matriz_indicadores: pd.DataFrame,
) -> float:

    if matriz_indicadores.empty:
        return 0.5

    scores = []

    global_centroid = (
        matriz_indicadores.mean(
            axis=0
        )
        .to_numpy(
            dtype=float
        )
    )

    for escenario, grupo in gdf.groupby(
        escenario_col,
        sort=False,
    ):

        indices = grupo.index

        valores = (
            matriz_indicadores
            .loc[indices]
            .mean(axis=0)
            .to_numpy(
                dtype=float
            )
        )

        distancia = float(
            np.linalg.norm(
                valores
                - global_centroid
            )
        )

        # Máxima distancia práctica:
        # sqrt(n_indicadores)
        max_distancia = math.sqrt(
            matriz_indicadores.shape[1]
        )

        if max_distancia <= 0:
            score = 1.0
        else:
            score = (
                1.0
                - (
                    distancia
                    / max_distancia
                )
            )

        scores.append(
            float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


# ==============================================================================
# PRESERVACIÓN DEL SCORE ORIGINAL
# ==============================================================================

def detectar_score_original(
    gdf: pd.DataFrame,
) -> Optional[str]:

    candidatos = [
        "score_prioridad_territorial",
        "score_cartera",
        "score_territorial",
        "score_global",
    ]

    return encontrar_columna(
        gdf,
        candidatos,
        obligatoria=False,
    )


def calcular_preservacion_score(
    gdf_original: pd.DataFrame,
    gdf_actual: pd.DataFrame,
    proyecto_col: str,
    score_col: Optional[str],
) -> float:

    if score_col is None:
        return 1.0

    original = normalizar_global(
        gdf_original[score_col]
    )

    actual = normalizar_global(
        gdf_actual[score_col]
    )

    diferencia = (
        original
        - actual
    ).abs()

    if diferencia.empty:
        return 1.0

    error = float(
        diferencia.mean()
    )

    return float(
        np.clip(
            1.0 - error,
            0.0,
            1.0,
        )
    )


# ==============================================================================
# ESTABILIDAD
# ==============================================================================

def calcular_estabilidad(
    gdf_original: pd.DataFrame,
    gdf_actual: pd.DataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> float:

    original = (
        gdf_original[
            [
                proyecto_col,
                escenario_col,
            ]
        ]
        .copy()
        .set_index(
            proyecto_col
        )
    )

    actual = (
        gdf_actual[
            [
                proyecto_col,
                escenario_col,
            ]
        ]
        .copy()
        .set_index(
            proyecto_col
        )
    )

    comunes = original.index.intersection(
        actual.index
    )

    if len(comunes) == 0:
        return 0.0

    iguales = (
        original.loc[
            comunes,
            escenario_col,
        ]
        ==
        actual.loc[
            comunes,
            escenario_col,
        ]
    )

    proporcion = float(
        iguales.mean()
    )

    return float(
        np.clip(
            proporcion,
            0.0,
            1.0,
        )
    )


# ==============================================================================
# FUNCIÓN OBJETIVO
# ==============================================================================

def evaluar_estado(
    gdf_original: gpd.GeoDataFrame,
    gdf_actual: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    matriz_indicadores: pd.DataFrame,
    score_col: Optional[str],
) -> Dict[str, float]:

    gdf_metric = preparar_geometria_metrica(
        gdf_actual
    )

    cantidades = (
        gdf_actual
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .to_numpy(
            dtype=float
        )
    )

    cantidad_escenarios = len(
        cantidades
    )

    cobertura = (
        1.0
        if len(
            gdf_actual
        )
        ==
        len(
            gdf_original
        )
        else 0.0
    )

    estructura = (
        calcular_estructura_escenarios(
            cantidad_escenarios
        )
    )

    tamano = (
        calcular_tamano(
            cantidades
        )
    )

    cohesion = (
        calcular_cohesion(
            gdf_metric,
            escenario_col,
        )
    )

    indicadores = (
        calcular_coherencia_indicadores(
            gdf_actual,
            escenario_col,
            matriz_indicadores,
        )
    )

    score_original = (
        calcular_preservacion_score(
            gdf_original,
            gdf_actual,
            proyecto_col,
            score_col,
        )
    )

    estabilidad = (
        calcular_estabilidad(
            gdf_original,
            gdf_actual,
            escenario_col,
            proyecto_col,
        )
    )

    # ------------------------------------------------------------------
    # Función objetivo
    # ------------------------------------------------------------------

    score = (
        PESO_COHESION
        * cohesion
        +
        PESO_BALANCE
        * calcular_balance(
            cantidades
        )
        +
        PESO_INDICADORES
        * indicadores
        +
        PESO_SCORE_ORIGINAL
        * score_original
        +
        PESO_ESTABILIDAD
        * estabilidad
    )

    # La estructura y cobertura funcionan como restricciones fuertes.
    if cobertura < 1.0:
        score *= 0.25

    if (
        cantidad_escenarios
        < MIN_ESCENARIOS
        or
        cantidad_escenarios
        > MAX_ESCENARIOS
    ):
        score *= (
            0.75
            + (
                0.25
                * estructura
            )
        )

    if (
        len(cantidades) > 0
        and
        cantidades.min()
        < MIN_PROYECTOS
    ):
        score *= 0.25

    return {
        "cobertura": float(cobertura),
        "estructura_escenarios": float(
            estructura
        ),
        "tamano": float(
            tamano
        ),
        "balance": float(
            calcular_balance(
                cantidades
            )
        ),
        "cohesion": float(
            cohesion
        ),
        "indicadores": float(
            indicadores
        ),
        "score_original": float(
            score_original
        ),
        "estabilidad": float(
            estabilidad
        ),
        "score_global": float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        ),
    }


# ==============================================================================
# VALIDACIÓN DE MOVIMIENTO
# ==============================================================================

def movimiento_permitido(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    escenario_origen: Any,
    escenario_destino: Any,
) -> bool:

    if (
        escenario_origen
        ==
        escenario_destino
    ):
        return False

    conteos = (
        gdf.groupby(
            escenario_col
        ).size()
    )

    origen = int(
        conteos.get(
            escenario_origen,
            0,
        )
    )

    destino = int(
        conteos.get(
            escenario_destino,
            0,
        )
    )

    # El origen jamás puede caer por debajo
    # del mínimo.
    if origen <= MIN_PROYECTOS:
        return False

    # El destino debe existir.
    if destino <= 0:
        return False

    return True


# ==============================================================================
# CANDIDATOS ESPACIALES
# ==============================================================================

def generar_candidatos(
    gdf: gpd.GeoDataFrame,
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> List[Tuple[Any, Any, Any]]:

    centroides = calcular_centroides(
        gdf_metric,
        escenario_col,
    )

    candidatos = []

    for idx, fila in gdf.iterrows():

        escenario_origen = fila[
            escenario_col
        ]

        if escenario_origen not in centroides:
            continue

        if not movimiento_permitido(
            gdf,
            escenario_col,
            escenario_origen,
            escenario_origen,
        ):
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

        destinos = []

        for escenario_destino, centro_destino in centroides.items():

            if (
                escenario_destino
                ==
                escenario_origen
            ):
                continue

            distancia_destino = float(
                geometria.distance(
                    centro_destino
                )
            )

            destinos.append(
                (
                    distancia_destino,
                    escenario_destino,
                )
            )

        destinos.sort(
            key=lambda x: x[0]
        )

        for distancia_destino, destino in destinos[:3]:

            # Prefiltro espacial.
            # No evaluamos transferencias que claramente
            # alejan el proyecto.
            if (
                distancia_destino
                >= distancia_origen
            ):
                continue

            if not movimiento_permitido(
                gdf,
                escenario_col,
                escenario_origen,
                destino,
            ):
                continue

            candidatos.append(
                (
                    idx,
                    escenario_origen,
                    destino,
                )
            )

    return candidatos


# ==============================================================================
# OPTIMIZACIÓN PRINCIPAL
# ==============================================================================

def optimizar_asignacion(
    gdf_original: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
) -> Tuple[
    gpd.GeoDataFrame,
    List[Dict[str, Any]],
    Dict[str, float],
]:

    encabezado(
        "5. OPTIMIZACIÓN MULTIOBJETIVO"
    )

    gdf_actual = (
        gdf_original.copy()
    )

    gdf_metric = preparar_geometria_metrica(
        gdf_actual
    )

    matriz_indicadores = preparar_indicadores(
        gdf_original,
        indicadores,
    )

    estado = evaluar_estado(
        gdf_original,
        gdf_actual,
        escenario_col,
        proyecto_col,
        matriz_indicadores,
        score_col,
    )

    movimientos: List[
        Dict[str, Any]
    ] = []

    print()
    print(
        "Score inicial : "
        f"{estado['score_global']:.6f}"
    )

    print(
        f"Escenarios    : "
        f"{gdf_actual[escenario_col].nunique()}"
    )

    for iteracion in range(
        1,
        MAX_ITERACIONES + 1,
    ):

        candidatos = generar_candidatos(
            gdf_actual,
            gdf_metric,
            escenario_col,
            proyecto_col,
        )

        if not candidatos:
            print(
                f"Iteración {iteracion:03d} "
                "| sin candidatos"
            )
            break

        if (
            MAX_CANDIDATOS_POR_ITERACION
            is not None
            and
            len(candidatos)
            >
            MAX_CANDIDATOS_POR_ITERACION
        ):

            candidatos = candidatos[
                :MAX_CANDIDATOS_POR_ITERACION
            ]

        mejor_movimiento = None
        mejor_estado = None
        mejor_mejora = 0.0

        # ------------------------------------------------------------------
        # Evaluamos cada movimiento como un cambio real de estado.
        # ------------------------------------------------------------------

        for (
            idx,
            escenario_origen,
            escenario_destino,
        ) in candidatos:

            proyecto = gdf_actual.loc[
                idx,
                proyecto_col,
            ]

            estado_anterior = (
                gdf_actual.loc[
                    idx,
                    escenario_col,
                ]
            )

            if (
                estado_anterior
                !=
                escenario_origen
            ):
                continue

            # --------------------------------------------------------------
            # Aplicar temporalmente
            # --------------------------------------------------------------

            gdf_actual.loc[
                idx,
                escenario_col,
            ] = escenario_destino

            estado_nuevo = evaluar_estado(
                gdf_original,
                gdf_actual,
                escenario_col,
                proyecto_col,
                matriz_indicadores,
                score_col,
            )

            # --------------------------------------------------------------
            # Revertir inmediatamente.
            # --------------------------------------------------------------

            gdf_actual.loc[
                idx,
                escenario_col,
            ] = estado_anterior

            mejora = (
                estado_nuevo[
                    "score_global"
                ]
                - estado[
                    "score_global"
                ]
            )

            # Penalización mínima por movimiento.
            mejora_neta = (
                mejora
                - PENALIZACION_MOVIMIENTO
                * 0.0
            )

            if (
                mejora_neta
                >
                mejor_mejora
                and
                mejora_neta
                >= MEJORA_MINIMA
            ):

                mejor_mejora = (
                    mejora_neta
                )

                mejor_movimiento = {
                    "idx": idx,
                    "proyecto_id": proyecto,
                    "escenario_origen": escenario_origen,
                    "escenario_destino": escenario_destino,
                    "mejora_score": mejora,
                    "mejora_neta": mejora_neta,
                    "estado_anterior": estado,
                    "estado_nuevo": estado_nuevo,
                }

                mejor_estado = (
                    estado_nuevo
                )

        # ------------------------------------------------------------------
        # No existe mejora.
        # ------------------------------------------------------------------

        if (
            mejor_movimiento is None
            or
            mejor_estado is None
        ):

            print(
                f"Iteración {iteracion:03d} "
                "| sin mejora aceptable"
            )

            break

        # ------------------------------------------------------------------
        # Aplicar definitivamente.
        # ------------------------------------------------------------------

        idx = mejor_movimiento[
            "idx"
        ]

        escenario_origen = (
            mejor_movimiento[
                "escenario_origen"
            ]
        )

        escenario_destino = (
            mejor_movimiento[
                "escenario_destino"
            ]
        )

        gdf_actual.loc[
            idx,
            escenario_col,
        ] = escenario_destino

        estado = mejor_estado

        movimientos.append(
            {
                "movimiento_id": len(
                    movimientos
                ) + 1,
                "iteracion": iteracion,
                "proyecto_id": mejor_movimiento[
                    "proyecto_id"
                ],
                "escenario_origen": escenario_origen,
                "escenario_destino": escenario_destino,
                "mejora_score": mejor_movimiento[
                    "mejora_score"
                ],
                "mejora_neta": mejor_movimiento[
                    "mejora_neta"
                ],
                "score_antes": mejor_movimiento[
                    "estado_anterior"
                ]["score_global"],
                "score_despues": mejor_movimiento[
                    "estado_nuevo"
                ]["score_global"],
                "cohesion_antes": mejor_movimiento[
                    "estado_anterior"
                ]["cohesion"],
                "cohesion_despues": mejor_movimiento[
                    "estado_nuevo"
                ]["cohesion"],
                "balance_antes": mejor_movimiento[
                    "estado_anterior"
                ]["balance"],
                "balance_despues": mejor_movimiento[
                    "estado_nuevo"
                ]["balance"],
                "indicadores_antes": mejor_movimiento[
                    "estado_anterior"
                ]["indicadores"],
                "indicadores_despues": mejor_movimiento[
                    "estado_nuevo"
                ]["indicadores"],
            }
        )

        # Actualizar geometría métrica.
        gdf_metric = preparar_geometria_metrica(
            gdf_actual
        )

        print(
            f"Iteración {iteracion:03d} "
            f"| proyecto={mejor_movimiento['proyecto_id']} "
            f"| {escenario_origen} -> {escenario_destino} "
            f"| mejora={mejor_mejora:+.6f}"
        )

    print()
    print(
        f"Movimientos aceptados : "
        f"{len(movimientos)}"
    )

    print(
        f"Score final            : "
        f"{estado['score_global']:.6f}"
    )

    return (
        gdf_actual,
        movimientos,
        estado,
    )


# ==============================================================================
# EVALUACIÓN POR ESCENARIO
# ==============================================================================

def construir_evaluacion_escenarios(
    gdf_original: gpd.GeoDataFrame,
    gdf_optimizado: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
) -> pd.DataFrame:

    encabezado(
        "6. EVALUACIÓN DETALLADA DE ESCENARIOS"
    )

    matriz = preparar_indicadores(
        gdf_original,
        indicadores,
    )

    filas = []

    escenarios_originales = set(
        gdf_original[
            escenario_col
        ].unique()
    )

    escenarios_optimizados = set(
        gdf_optimizado[
            escenario_col
        ].unique()
    )

    escenarios = sorted(
        escenarios_originales
        | escenarios_optimizados,
        key=lambda x: str(x),
    )

    ranking_data = []

    for escenario in escenarios:

        grupo = gdf_optimizado[
            gdf_optimizado[
                escenario_col
            ]
            ==
            escenario
        ]

        if grupo.empty:
            cantidad = 0
            cohesion = 0.0
            indicador_score = 0.0
        else:

            cantidad = len(
                grupo
            )

            grupo_metric = preparar_geometria_metrica(
                grupo
            )

            centro = (
                grupo_metric
                .geometry
                .unary_union
                .centroid
            )

            distancias = (
                grupo_metric.geometry
                .centroid
                .distance(
                    centro
                )
            )

            distancia_media = float(
                distancias.mean()
            )

            cohesion = float(
                np.clip(
                    1.0
                    - (
                        distancia_media
                        / RADIO_COHESION_METROS
                    ),
                    0.0,
                    1.0,
                )
            )

            indices = grupo.index

            vector = (
                matriz.loc[
                    indices
                ]
                .mean(
                    axis=0
                )
            )

            centro_global = (
                matriz.mean(
                    axis=0
                )
            )

            distancia = float(
                np.linalg.norm(
                    vector.to_numpy()
                    -
                    centro_global.to_numpy()
                )
            )

            max_distancia = math.sqrt(
                matriz.shape[1]
            )

            indicador_score = float(
                np.clip(
                    1.0
                    - (
                        distancia
                        / max_distancia
                    ),
                    0.0,
                    1.0,
                )
            )

        ranking_data.append(
            {
                "escenario_id": escenario,
                "cantidad_proyectos": cantidad,
                "cohesion": cohesion,
                "coherencia_indicadores": indicador_score,
            }
        )

    ranking_df = pd.DataFrame(
        ranking_data
    )

    if not ranking_df.empty:

        ranking_df[
            "score_escenario"
        ] = (
            0.5
            * ranking_df[
                "cohesion"
            ]
            +
            0.5
            * ranking_df[
                "coherencia_indicadores"
            ]
        )

        ranking_df[
            "ranking_escenario"
        ] = (
            ranking_df[
                "score_escenario"
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

    # ------------------------------------------------------------------
    # Comparación original -> optimizado
    # ------------------------------------------------------------------

    conteos_originales = (
        gdf_original
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .rename(
            "proyectos_originales"
        )
    )

    conteos_optimizados = (
        gdf_optimizado
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .rename(
            "proyectos_optimizados"
        )
    )

    resultado = (
        ranking_df
        .merge(
            conteos_originales,
            left_on="escenario_id",
            right_index=True,
            how="left",
        )
        .merge(
            conteos_optimizados,
            left_on="escenario_id",
            right_index=True,
            how="left",
        )
    )

    resultado[
        "proyectos_originales"
    ] = resultado[
        "proyectos_originales"
    ].fillna(0).astype(int)

    resultado[
        "proyectos_optimizados"
    ] = resultado[
        "proyectos_optimizados"
    ].fillna(0).astype(int)

    resultado[
        "cambio_proyectos"
    ] = (
        resultado[
            "proyectos_optimizados"
        ]
        -
        resultado[
            "proyectos_originales"
        ]
    )

    columnas = [
        "escenario_id",
        "proyectos_originales",
        "proyectos_optimizados",
        "cambio_proyectos",
        "cohesion",
        "coherencia_indicadores",
        "score_escenario",
        "ranking_escenario",
    ]

    return resultado[
        columnas
    ].sort_values(
        [
            "ranking_escenario",
            "escenario_id",
        ]
    )


# ==============================================================================
# VALIDACIÓN FINAL
# ==============================================================================

def validar_resultado(
    gdf_original: gpd.GeoDataFrame,
    gdf_optimizado: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> Dict[str, Any]:

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

    conteos = (
        gdf_optimizado
        .groupby(
            escenario_col
        )[proyecto_col]
        .count()
    )

    mismos_proyectos = (
        proyectos_originales
        ==
        proyectos_optimizados
    )

    mismos_escenarios = (
        escenarios_originales
        ==
        escenarios_optimizados
    )

    proyectos_sin_escenario = int(
        gdf_optimizado[
            escenario_col
        ].isna().sum()
    )

    duplicados = int(
        gdf_optimizado[
            proyecto_col
        ].duplicated().sum()
    )

    min_proyectos = int(
        conteos.min()
    ) if not conteos.empty else 0

    max_proyectos = int(
        conteos.max()
    ) if not conteos.empty else 0

    validacion = {
        "mismos_proyectos": bool(
            mismos_proyectos
        ),
        "mismos_escenarios": bool(
            mismos_escenarios
        ),
        "proyectos_sin_escenario": (
            proyectos_sin_escenario
        ),
        "duplicados_proyecto": (
            duplicados
        ),
        "escenarios_originales": len(
            escenarios_originales
        ),
        "escenarios_optimizados": len(
            escenarios_optimizados
        ),
        "min_proyectos": min_proyectos,
        "max_proyectos": max_proyectos,
        "proyectos_originales": len(
            gdf_original
        ),
        "proyectos_optimizados": len(
            gdf_optimizado
        ),
    }

    print(
        f"Proyectos originales    : "
        f"{validacion['proyectos_originales']:,}"
    )

    print(
        f"Proyectos optimizados   : "
        f"{validacion['proyectos_optimizados']:,}"
    )

    print(
        f"Mismos proyectos        : "
        f"{validacion['mismos_proyectos']}"
    )

    print(
        f"Mismos escenarios       : "
        f"{validacion['mismos_escenarios']}"
    )

    print(
        f"Duplicados proyecto     : "
        f"{validacion['duplicados_proyecto']}"
    )

    print(
        f"Sin escenario           : "
        f"{validacion['proyectos_sin_escenario']}"
    )

    print(
        f"Escenarios              : "
        f"{validacion['escenarios_optimizados']}"
    )

    print(
        f"Mínimo proyectos       : "
        f"{validacion['min_proyectos']}"
    )

    print(
        f"Máximo proyectos       : "
        f"{validacion['max_proyectos']}"
    )

    # ------------------------------------------------------------------
    # Restricciones duras
    # ------------------------------------------------------------------

    if not mismos_proyectos:
        raise ValueError(
            "ERROR CRÍTICO: cambió el conjunto de proyectos."
        )

    if not mismos_escenarios:
        raise ValueError(
            "ERROR CRÍTICO: cambió el conjunto de escenarios."
        )

    if proyectos_sin_escenario > 0:
        raise ValueError(
            "ERROR CRÍTICO: existen proyectos sin escenario."
        )

    if duplicados > 0:
        raise ValueError(
            "ERROR CRÍTICO: existen proyectos duplicados."
        )

    if (
        min_proyectos
        <
        MIN_PROYECTOS
    ):
        raise ValueError(
            "ERROR CRÍTICO: un escenario quedó por debajo "
            f"del mínimo de {MIN_PROYECTOS} proyectos."
        )

    print()
    print(
        "Validación final: OK"
    )

    return validacion


# ==============================================================================
# RESUMEN
# ==============================================================================

def construir_resumen(
    metricas_originales: Dict[str, float],
    metricas_optimizadas: Dict[str, float],
) -> pd.DataFrame:

    claves = [
        "cobertura",
        "estructura_escenarios",
        "tamano",
        "balance",
        "cohesion",
        "indicadores",
        "score_original",
        "estabilidad",
        "score_global",
    ]

    filas = []

    for clave in claves:

        original = float(
            metricas_originales[
                clave
            ]
        )

        optimizado = float(
            metricas_optimizadas[
                clave
            ]
        )

        filas.append(
            {
                "metrica": clave,
                "original": original,
                "optimizado": optimizado,
                "cambio_absoluto": (
                    optimizado
                    - original
                ),
            }
        )

    return pd.DataFrame(
        filas
    )


# ==============================================================================
# EXPORTACIÓN
# ==============================================================================

def exportar_resultados(
    gdf_optimizado: gpd.GeoDataFrame,
    evaluacion: pd.DataFrame,
    movimientos: List[Dict[str, Any]],
    resumen: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:

    encabezado(
        "8. EXPORTANDO RESULTADOS"
    )

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # GeoParquet
    # ------------------------------------------------------------------

    gdf_optimizado.to_parquet(
        OUTPUT_OPTIMIZADO,
        index=False,
    )

    # ------------------------------------------------------------------
    # CSV sin geometría
    # ------------------------------------------------------------------

    gdf_csv = gdf_optimizado.copy()

    if "geometry" in gdf_csv.columns:

        gdf_csv = pd.DataFrame(
            gdf_csv.drop(
                columns=[
                    "geometry"
                ]
            )
        )

    gdf_csv.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------

    evaluacion.to_csv(
        OUTPUT_EVALUACION,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Movimientos
    # ------------------------------------------------------------------

    movimientos_df = pd.DataFrame(
        movimientos
    )

    if movimientos_df.empty:

        movimientos_df = pd.DataFrame(
            columns=[
                "movimiento_id",
                "iteracion",
                "proyecto_id",
                "escenario_origen",
                "escenario_destino",
                "mejora_score",
                "mejora_neta",
                "score_antes",
                "score_despues",
            ]
        )

    movimientos_df.to_csv(
        OUTPUT_MOVIMIENTOS,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------------

    resumen.to_csv(
        OUTPUT_RESUMEN,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    OUTPUT_METADATA.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
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

    # --------------------------------------------------------------------------
    # CONFIGURACIÓN
    # --------------------------------------------------------------------------

    subtitulo(
        "Configuración"
    )

    print(
        f"Mínimo proyectos          : "
        f"{MIN_PROYECTOS}"
    )

    print(
        f"Rango escenarios          : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
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

    print(
        f"  TOTAL                   : "
        f"{SUMA_PESOS:.0%}"
    )

    if not math.isclose(
        SUMA_PESOS,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Los pesos de optimización deben sumar 1.0."
        )

    # --------------------------------------------------------------------------
    # 1. CARGA
    # --------------------------------------------------------------------------

    gdf_original = cargar_escenarios()

    evaluacion_proceso_28 = (
        cargar_evaluacion()
    )

    recomendaciones = (
        cargar_recomendaciones()
    )

    # --------------------------------------------------------------------------
    # 2. VALIDACIÓN
    # --------------------------------------------------------------------------

    escenario_col, proyecto_col = (
        validar_entrada(
            gdf_original
        )
    )

    # --------------------------------------------------------------------------
    # 3. INDICADORES
    # --------------------------------------------------------------------------

    encabezado(
        "3. PREPARANDO INDICADORES"
    )

    indicadores = detectar_indicadores(
        gdf_original,
        escenario_col,
        proyecto_col,
    )

    score_col = detectar_score_original(
        gdf_original
    )

    print()
    print(
        f"Score territorial utilizado: "
        f"{score_col if score_col else 'ninguno'}"
    )

    if evaluacion_proceso_28 is not None:

        print(
            "Evaluación del proceso 28: DISPONIBLE"
        )

    else:

        print(
            "Evaluación del proceso 28: NO DISPONIBLE"
        )

    if recomendaciones is not None:

        print(
            "Recomendaciones proceso 28: DISPONIBLES"
        )

    else:

        print(
            "Recomendaciones proceso 28: NO DISPONIBLES"
        )

    # --------------------------------------------------------------------------
    # 4. EVALUACIÓN BASE
    # --------------------------------------------------------------------------

    encabezado(
        "4. EVALUACIÓN BASE"
    )

    matriz_indicadores = preparar_indicadores(
        gdf_original,
        indicadores,
    )

    metricas_originales = evaluar_estado(
        gdf_original,
        gdf_original.copy(),
        escenario_col,
        proyecto_col,
        matriz_indicadores,
        score_col,
    )

    for clave, valor in (
        metricas_originales.items()
    ):

        print(
            f"{clave:<25}: "
            f"{valor:.6f}"
        )

    # --------------------------------------------------------------------------
    # 5. OPTIMIZACIÓN
    # --------------------------------------------------------------------------

    (
        gdf_optimizado,
        movimientos,
        metricas_optimizadas,
    ) = optimizar_asignacion(
        gdf_original,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
    )

    # --------------------------------------------------------------------------
    # 6. EVALUACIÓN DETALLADA
    # --------------------------------------------------------------------------

    evaluacion = (
        construir_evaluacion_escenarios(
            gdf_original,
            gdf_optimizado,
            escenario_col,
            proyecto_col,
            indicadores,
        )
    )

    # --------------------------------------------------------------------------
    # 7. VALIDACIÓN
    # --------------------------------------------------------------------------

    validacion = validar_resultado(
        gdf_original,
        gdf_optimizado,
        escenario_col,
        proyecto_col,
    )

    # --------------------------------------------------------------------------
    # 8. RESUMEN
    # --------------------------------------------------------------------------

    resumen = construir_resumen(
        metricas_originales,
        metricas_optimizadas,
    )

    # --------------------------------------------------------------------------
    # METADATA
    # --------------------------------------------------------------------------

    tiempo = (
        time.perf_counter()
        - inicio
    )

    metadata = {
        "proceso": 29,
        "nombre": (
            "Optimización de escenarios "
            "territoriales AMBA"
        ),
        "version": VERSION,
        "fecha_ejecucion": (
            pd.Timestamp.now(
                tz="America/Argentina/Buenos_Aires"
            ).isoformat()
        ),
        "proyectos": int(
            len(gdf_optimizado)
        ),
        "escenarios": int(
            gdf_optimizado[
                escenario_col
            ].nunique()
        ),
        "columna_escenario": escenario_col,
        "columna_proyecto": proyecto_col,
        "indicadores": indicadores,
        "score_col": score_col,
        "movimientos": len(
            movimientos
        ),
        "configuracion": {
            "min_proyectos": MIN_PROYECTOS,
            "min_escenarios": MIN_ESCENARIOS,
            "max_escenarios": MAX_ESCENARIOS,
            "max_iteraciones": MAX_ITERACIONES,
            "mejora_minima": MEJORA_MINIMA,
            "radio_cohesion_metros": (
                RADIO_COHESION_METROS
            ),
            "peso_cohesion": PESO_COHESION,
            "peso_balance": PESO_BALANCE,
            "peso_indicadores": PESO_INDICADORES,
            "peso_score_original": (
                PESO_SCORE_ORIGINAL
            ),
            "peso_estabilidad": (
                PESO_ESTABILIDAD
            ),
        },
        "metricas_originales": (
            metricas_originales
        ),
        "metricas_optimizadas": (
            metricas_optimizadas
        ),
        "validacion": validacion,
        "duracion_segundos": tiempo,
    }

    # --------------------------------------------------------------------------
    # 9. EXPORTACIÓN
    # --------------------------------------------------------------------------

    exportar_resultados(
        gdf_optimizado,
        evaluacion,
        movimientos,
        resumen,
        metadata,
    )

    # --------------------------------------------------------------------------
    # 10. RESUMEN FINAL
    # --------------------------------------------------------------------------

    encabezado(
        "10. PROCESO 29 FINALIZADO CORRECTAMENTE"
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
        f"Proyectos procesados     : "
        f"{len(gdf_optimizado):,}"
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

    print(
        f"Mejora global            : "
        f"{mejora:+.6f}"
    )

    print(
        f"Duración                 : "
        f"{tiempo:.2f} segundos"
    )

    # --------------------------------------------------------------------------
    # Distribución final
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Control de mejora
    # --------------------------------------------------------------------------

    print()
    print(
        "CONTROL DE OPTIMIZACIÓN"
    )

    if mejora > 0:
        print(
            "Resultado: MEJORA POSITIVA"
        )

    elif math.isclose(
        mejora,
        0.0,
        abs_tol=1e-9,
    ):
        print(
            "Resultado: SIN CAMBIO SIGNIFICATIVO"
        )

    else:
        print(
            "Resultado: DETERIORO"
        )

    print()
    print(
        "Salida principal:"
    )

    print(
        OUTPUT_OPTIMIZADO
    )

    print()
    print("=" * 96)


# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()