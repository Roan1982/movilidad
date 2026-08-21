# -*- coding: utf-8 -*-

"""
================================================================================
29 - OPTIMIZACIÓN DE ESCENARIOS TERRITORIALES AMBA
================================================================================
VERSIÓN 2.0

Proceso:
    27 -> Construcción de escenarios territoriales
    28 -> Evaluación de escenarios
    29 -> Optimización de escenarios

OBJETIVO
--------
Optimizar la asignación de proyectos a escenarios territoriales manteniendo:

    - cobertura total de proyectos
    - identidad de cada proyecto
    - cantidad razonable de escenarios
    - mínimo de proyectos por escenario
    - cohesión territorial
    - balance de tamaños
    - coherencia de indicadores
    - conservación del score territorial original
    - estabilidad de la asignación
    - trazabilidad completa de movimientos

PRINCIPIO DE OPTIMIZACIÓN
-------------------------
La V2 no realiza movimientos únicamente por distancia.

Cada movimiento candidato se evalúa mediante una función objetivo global:

    SCORE =
          cohesión territorial
        + balance de tamaños
        + coherencia de indicadores
        + conservación del score territorial
        + estabilidad de asignación

Un movimiento solamente se acepta cuando:

    1. respeta las restricciones duras
    2. mejora el score objetivo
    3. no deteriora significativamente el score territorial
    4. no viola el mínimo de proyectos
    5. mantiene la cobertura total
    6. mejora o mantiene la coherencia espacial

ENTRADAS
--------
    data/processed/escenarios_territoriales_amba/
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
from typing import Dict, List, Optional, Tuple, Any

import geopandas as gpd
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================

VERSION = "V2.0"

PROCESO = 29

# ------------------------------------------------------------------------------
# Restricciones estructurales
# ------------------------------------------------------------------------------

MIN_PROYECTOS = 8

MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

MAX_ITERACIONES = 50

# ------------------------------------------------------------------------------
# Parámetros de optimización
# ------------------------------------------------------------------------------

# Pesos de la función objetivo.
#
# La suma debe ser 1.0.

PESO_COHESION = 0.30
PESO_BALANCE = 0.20
PESO_INDICADORES = 0.25
PESO_SCORE_ORIGINAL = 0.15
PESO_ESTABILIDAD = 0.10

# ------------------------------------------------------------------------------
# Umbrales
# ------------------------------------------------------------------------------

# Mejora mínima del objetivo global para aceptar movimiento.
MEJORA_MINIMA = 0.0005

# Deterioro máximo permitido del score territorial original.
MAX_DETERIORO_SCORE_ORIGINAL = 0.015

# Deterioro máximo permitido de cohesión en un movimiento.
MAX_DETERIORO_COHESION = 0.020

# Máxima diferencia de tamaño entre escenarios.
MAX_DIFERENCIA_TAMANO = 8

# Cantidad de candidatos espaciales evaluados por proyecto.
MAX_DESTINOS_CANDIDATOS = 5

# Distancia máxima relativa para considerar una transferencia.
#
# 1.0 significa que el destino no puede estar más lejos que el
# centroide actual del proyecto.
FACTOR_DISTANCIA_DESTINO = 1.0

# ------------------------------------------------------------------------------
# Escala espacial
# ------------------------------------------------------------------------------

# EPSG métrico utilizado cuando la capa viene en coordenadas geográficas.
CRS_METRICO = 3857

# ------------------------------------------------------------------------------
# Indicadores preferidos
# ------------------------------------------------------------------------------

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
# UTILIDADES
# ==============================================================================

def encabezado(titulo: str) -> None:
    print()
    print("=" * 90)
    print(titulo)
    print("=" * 90)


def imprimir_metrica(
    nombre: str,
    valor: float,
) -> None:
    print(
        f"{nombre:<35}: {valor:.6f}"
    )


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
            "No se encontró ninguna de las columnas requeridas: "
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
            0.5,
            index=serie.index,
            dtype=float,
        )

    minimo = float(valores.min())
    maximo = float(valores.max())

    if math.isclose(
        minimo,
        maximo,
        rel_tol=1e-12,
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

    return resultado.fillna(0.5)


def validar_pesos() -> None:

    pesos = [
        PESO_COHESION,
        PESO_BALANCE,
        PESO_INDICADORES,
        PESO_SCORE_ORIGINAL,
        PESO_ESTABILIDAD,
    ]

    suma = sum(pesos)

    if not math.isclose(
        suma,
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Los pesos de optimización deben sumar 1.0. "
            f"Suma actual: {suma}"
        )


# ==============================================================================
# CARGA DE DATOS
# ==============================================================================

def cargar_escenarios() -> gpd.GeoDataFrame:

    encabezado(
        "1. CARGA DE ESCENARIOS"
    )

    if not INPUT_ESCENARIOS.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada:\n"
            f"{INPUT_ESCENARIOS}"
        )

    gdf = gpd.read_parquet(
        INPUT_ESCENARIOS
    )

    if len(gdf) == 0:
        raise ValueError(
            "El GeoParquet de escenarios está vacío."
        )

    print(
        f"Archivo      : {INPUT_ESCENARIOS}"
    )

    print(
        f"Registros    : {len(gdf):,}"
    )

    print(
        f"Columnas     : {len(gdf.columns)}"
    )

    print(
        f"CRS          : {gdf.crs}"
    )

    return gdf


def cargar_evaluacion() -> Optional[pd.DataFrame]:

    encabezado(
        "2. CARGA DE EVALUACIÓN DEL PROCESO 28"
    )

    if not INPUT_EVALUACION.exists():

        print(
            "Evaluación del proceso 28 no encontrada."
        )

        print(
            "Se continuará sin esta entrada."
        )

        return None

    evaluacion = pd.read_parquet(
        INPUT_EVALUACION
    )

    print(
        f"Archivo      : {INPUT_EVALUACION}"
    )

    print(
        f"Registros    : {len(evaluacion):,}"
    )

    return evaluacion


def cargar_recomendaciones() -> Optional[pd.DataFrame]:

    encabezado(
        "3. CARGA DE RECOMENDACIONES DEL PROCESO 28"
    )

    if not INPUT_RECOMENDACIONES.exists():

        print(
            "Archivo de recomendaciones no encontrado."
        )

        print(
            "Se continuará sin recomendaciones externas."
        )

        return None

    try:

        recomendaciones = pd.read_csv(
            INPUT_RECOMENDACIONES,
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:

        recomendaciones = pd.read_csv(
            INPUT_RECOMENDACIONES,
            encoding="latin-1",
        )

    print(
        f"Archivo      : {INPUT_RECOMENDACIONES}"
    )

    print(
        f"Registros    : {len(recomendaciones):,}"
    )

    print(
        f"Columnas     : {len(recomendaciones.columns)}"
    )

    return recomendaciones


# ==============================================================================
# VALIDACIÓN DE ENTRADA
# ==============================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> Tuple[str, str]:

    encabezado(
        "4. VALIDACIÓN DE ENTRADA"
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
            "La entrada no contiene geometría."
        )

    if gdf.crs is None:
        raise ValueError(
            "La capa no posee CRS."
        )

    geometria_nula = int(
        gdf.geometry.isna().sum()
    )

    geometria_vacia = int(
        gdf.geometry.is_empty.sum()
    )

    try:

        geometria_invalida = int(
            (~gdf.geometry.is_valid).sum()
        )

    except Exception:

        geometria_invalida = 0

    duplicados = int(
        gdf[proyecto_col]
        .duplicated()
        .sum()
    )

    proyectos_sin_escenario = int(
        gdf[escenario_col]
        .isna()
        .sum()
    )

    escenarios = int(
        gdf[escenario_col]
        .nunique(
            dropna=True
        )
    )

    proyectos = int(
        gdf[proyecto_col]
        .nunique()
    )

    print(
        f"Columna escenario       : {escenario_col}"
    )

    print(
        f"Columna proyecto        : {proyecto_col}"
    )

    print(
        f"Proyectos               : {proyectos:,}"
    )

    print(
        f"Escenarios              : {escenarios}"
    )

    print(
        f"Geometrías nulas        : {geometria_nula}"
    )

    print(
        f"Geometrías vacías       : {geometria_vacia}"
    )

    print(
        f"Geometrías inválidas    : {geometria_invalida}"
    )

    print(
        f"Proyectos duplicados    : {duplicados}"
    )

    print(
        f"Sin escenario           : {proyectos_sin_escenario}"
    )

    if geometria_nula > 0:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if geometria_vacia > 0:
        raise ValueError(
            "Existen geometrías vacías."
        )

    if geometria_invalida > 0:
        raise ValueError(
            "Existen geometrías inválidas."
        )

    if duplicados > 0:
        raise ValueError(
            "Existen proyectos duplicados."
        )

    if proyectos_sin_escenario > 0:
        raise ValueError(
            "Existen proyectos sin escenario en la entrada."
        )

    if escenarios < MIN_ESCENARIOS:
        print(
            "ADVERTENCIA: la entrada posee menos "
            f"de {MIN_ESCENARIOS} escenarios."
        )

    print(
        "Validación de entrada: OK"
    )

    return (
        escenario_col,
        proyecto_col,
    )


# ==============================================================================
# PREPARACIÓN GEOMÉTRICA
# ==============================================================================

def preparar_geometria_metrica(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    if gdf.crs is None:
        raise ValueError(
            "No se puede realizar análisis espacial sin CRS."
        )

    if gdf.crs.is_geographic:

        return gdf.to_crs(
            CRS_METRICO
        )

    return gdf.copy()


# ==============================================================================
# INDICADORES
# ==============================================================================

def detectar_indicadores(
    gdf: pd.DataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> List[str]:

    encabezado(
        "5. DETECCIÓN DE INDICADORES"
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

    indicadores = []

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

    # Si hay menos de 3 indicadores relevantes,
    # completar automáticamente con columnas numéricas.
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
        f"Indicadores utilizados : {len(indicadores)}"
    )

    for indicador in indicadores:

        print(
            f"  - {indicador}"
        )

    if not indicadores:

        print(
            "ADVERTENCIA: no se encontraron indicadores."
        )

    return indicadores


def construir_matriz_indicadores(
    gdf: pd.DataFrame,
    indicadores: List[str],
) -> pd.DataFrame:

    matriz = pd.DataFrame(
        index=gdf.index
    )

    for indicador in indicadores:

        matriz[
            indicador
        ] = normalizar_serie(
            gdf[indicador]
        )

    if matriz.empty:

        matriz[
            "__indicador_neutro"
        ] = 0.5

    return matriz


# ==============================================================================
# SCORE TERRITORIAL ORIGINAL
# ==============================================================================

def detectar_score_original(
    gdf: pd.DataFrame,
) -> Optional[str]:

    candidatos = [
        "score_cartera",
        "score_prioridad_territorial",
        "score_territorial",
        "score",
    ]

    for columna in candidatos:

        if columna in gdf.columns:

            if pd.api.types.is_numeric_dtype(
                gdf[columna]
            ):
                return columna

    return None


# ==============================================================================
# MÉTRICA DE TAMAÑO
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

    desviacion = float(
        np.std(
            cantidades,
            ddof=0,
        )
    )

    cv = (
        desviacion
        / promedio
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                1.0 - cv,
            ),
        )
    )


def calcular_score_tamano(
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
                cantidad
                / MIN_PROYECTOS
            )

    return float(
        np.mean(scores)
    )


# ==============================================================================
# COHESIÓN TERRITORIAL
# ==============================================================================

def calcular_cohesion(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> float:

    centroides = gdf_metric.geometry.centroid

    dispersiones = []

    for escenario in sorted(
        gdf_metric[
            escenario_col
        ].dropna().unique()
    ):

        mask = (
            gdf_metric[
                escenario_col
            ]
            == escenario
        )

        puntos = centroides[
            mask
        ]

        if len(puntos) <= 1:
            dispersiones.append(
                0.0
            )
            continue

        centro = puntos.unary_union.centroid

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

    dispersion_media = float(
        np.mean(
            dispersiones
        )
    )

    # 50 km = dispersión considerada muy alta.
    score = (
        1.0
        - dispersion_media / 50000.0
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                score,
            ),
        )
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
        return 0.0

    scores = []

    for indicador in matriz_indicadores.columns:

        valores = matriz_indicadores[
            indicador
        ]

        if valores.notna().sum() < 2:
            continue

        media_global = float(
            valores.mean()
        )

        medias = (
            pd.DataFrame(
                {
                    escenario_col:
                        gdf[
                            escenario_col
                        ],
                    "__valor":
                        valores,
                },
                index=gdf.index,
            )
            .groupby(
                escenario_col
            )[
                "__valor"
            ]
            .mean()
        )

        if len(medias) <= 1:
            scores.append(
                1.0
            )
            continue

        dispersion = float(
            medias.std(
                ddof=0
            )
        )

        # Buscamos escenarios internamente
        # coherentes sin generar concentraciones
        # extremas artificiales.
        score = 1.0 / (
            1.0
            + (
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


# ==============================================================================
# CONSERVACIÓN DEL SCORE TERRITORIAL
# ==============================================================================

def calcular_conservacion_score(
    gdf_original: pd.DataFrame,
    gdf_actual: pd.DataFrame,
    proyecto_col: str,
    score_col: Optional[str],
) -> float:

    if score_col is None:
        return 1.0

    original = pd.to_numeric(
        gdf_original[
            score_col
        ],
        errors="coerce",
    )

    actual = pd.to_numeric(
        gdf_actual[
            score_col
        ],
        errors="coerce",
    )

    if original.notna().sum() == 0:
        return 1.0

    # Normalización común.
    minimo = float(
        original.min()
    )

    maximo = float(
        original.max()
    )

    if math.isclose(
        minimo,
        maximo,
    ):
        return 1.0

    original_norm = (
        original - minimo
    ) / (
        maximo - minimo
    )

    actual_norm = (
        actual - minimo
    ) / (
        maximo - minimo
    )

    # Como el valor de score pertenece al proyecto,
    # debería permanecer invariado. Esta métrica
    # mide que no se haya alterado el campo.
    diferencia = float(
        np.mean(
            np.abs(
                original_norm
                - actual_norm
            )
        )
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                1.0 - diferencia,
            ),
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
    )

    actual = (
        gdf_actual[
            [
                proyecto_col,
                escenario_col,
            ]
        ]
        .copy()
    )

    original = original.set_index(
        proyecto_col
    )

    actual = actual.set_index(
        proyecto_col
    )

    if len(original) == 0:
        return 1.0

    iguales = (
        original[
            escenario_col
        ].astype(str)
        ==
        actual[
            escenario_col
        ].astype(str)
    )

    return float(
        iguales.mean()
    )


# ==============================================================================
# ESTRUCTURA DE ESCENARIOS
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
        max(
            0.0,
            1.0
            - 0.25
            * distancia,
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
        )[
            proyecto_col
        ]
        .count()
        .to_numpy(
            dtype=int
        )
    )

    cantidad_escenarios = len(
        cantidades
    )

    cobertura = 1.0

    estructura = (
        calcular_estructura_escenarios(
            cantidad_escenarios
        )
    )

    tamano = (
        calcular_score_tamano(
            cantidades
        )
    )

    balance = (
        calcular_balance(
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
        calcular_conservacion_score(
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

    score_global = (
        PESO_COHESION
        * cohesion
        +
        PESO_BALANCE
        * balance
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

    return {
        "cobertura": float(
            cobertura
        ),
        "estructura_escenarios": float(
            estructura
        ),
        "tamano": float(
            tamano
        ),
        "cohesion": float(
            cohesion
        ),
        "balance": float(
            balance
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
# CENTROIDES DE ESCENARIOS
# ==============================================================================

def obtener_centroides(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> Dict[Any, Any]:

    resultado = {}

    for escenario, grupo in gdf_metric.groupby(
        escenario_col
    ):

        if len(grupo) == 0:
            continue

        # Se utiliza el centroid de la unión geométrica.
        centro = (
            grupo.geometry
            .union_all()
            .centroid
        )

        resultado[
            escenario
        ] = centro

    return resultado


# ==============================================================================
# DISTANCIAS ESPACIALES
# ==============================================================================

def obtener_destinos_candidatos(
    geometria,
    centroides: Dict[Any, Any],
    escenario_actual: Any,
) -> List[Tuple[Any, float]]:

    candidatos = []

    for escenario, centro in centroides.items():

        if escenario == escenario_actual:
            continue

        try:

            distancia = float(
                geometria.distance(
                    centro
                )
            )

        except Exception:

            continue

        if math.isfinite(
            distancia
        ):

            candidatos.append(
                (
                    escenario,
                    distancia,
                )
            )

    candidatos.sort(
        key=lambda x: x[1]
    )

    return candidatos[
        :MAX_DESTINOS_CANDIDATOS
    ]


# ==============================================================================
# RESTRICCIONES DE MOVIMIENTO
# ==============================================================================

def movimiento_permitido(
    gdf_actual: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    idx,
    escenario_origen,
    escenario_destino,
) -> Tuple[bool, str]:

    conteos = (
        gdf_actual
        .groupby(
            escenario_col
        )[
            proyecto_col
        ]
        .count()
        .to_dict()
    )

    origen_n = int(
        conteos.get(
            escenario_origen,
            0,
        )
    )

    destino_n = int(
        conteos.get(
            escenario_destino,
            0,
        )
    )

    # Nunca vaciar un escenario por debajo
    # del mínimo estructural.
    if (
        origen_n
        <= MIN_PROYECTOS
    ):

        return (
            False,
            "ORIGEN_MINIMO",
        )

    # El destino no debe quedar excesivamente
    # más grande que el origen.
    if (
        destino_n
        >
        origen_n
        + MAX_DIFERENCIA_TAMANO
    ):

        return (
            False,
            "DESTINO_DESBALANCEADO",
        )

    return (
        True,
        "OK",
    )


# ==============================================================================
# OPTIMIZACIÓN
# ==============================================================================

def optimizar_asignacion(
    gdf_original: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    matriz_indicadores: pd.DataFrame,
    score_col: Optional[str],
) -> Tuple[
    gpd.GeoDataFrame,
    List[Dict[str, Any]],
]:

    encabezado(
        "6. OPTIMIZACIÓN ITERATIVA DE ASIGNACIONES"
    )

    gdf_opt = (
        gdf_original
        .copy()
    )

    gdf_metric = preparar_geometria_metrica(
        gdf_opt
    )

    movimientos = []

    estado_actual = evaluar_estado(
        gdf_original,
        gdf_opt,
        escenario_col,
        proyecto_col,
        matriz_indicadores,
        score_col,
    )

    print()
    print(
        "SCORE INICIAL"
    )

    for clave, valor in estado_actual.items():

        imprimir_metrica(
            clave,
            valor,
        )

    print()

    for iteracion in range(
        1,
        MAX_ITERACIONES + 1,
    ):

        cambios = 0

        escenarios = sorted(
            gdf_opt[
                escenario_col
            ]
            .dropna()
            .unique()
        )

        centroides = obtener_centroides(
            gdf_metric,
            escenario_col,
        )

        # --------------------------------------------------------------
        # Evaluar todos los proyectos.
        #
        # En lugar de aplicar inmediatamente el primer movimiento
        # espacialmente favorable, se busca el mejor movimiento
        # disponible según la función objetivo.
        # --------------------------------------------------------------

        mejor_movimiento = None
        mejor_estado = None
        mejor_mejora = 0.0

        for idx in gdf_opt.index:

            escenario_origen = (
                gdf_opt.loc[
                    idx,
                    escenario_col,
                ]
            )

            geometria = gdf_metric.loc[
                idx,
                "geometry",
            ]

            destinos = obtener_destinos_candidatos(
                geometria,
                centroides,
                escenario_origen,
            )

            if not destinos:
                continue

            centro_origen = centroides.get(
                escenario_origen
            )

            if centro_origen is None:
                continue

            distancia_origen = float(
                geometria.distance(
                    centro_origen
                )
            )

            for (
                escenario_destino,
                distancia_destino,
            ) in destinos:

                # ------------------------------------------------------
                # Restricciones duras
                # ------------------------------------------------------

                permitido, motivo = (
                    movimiento_permitido(
                        gdf_opt,
                        escenario_col,
                        proyecto_col,
                        idx,
                        escenario_origen,
                        escenario_destino,
                    )
                )

                if not permitido:
                    continue

                # El destino debe ser espacialmente razonable.
                if (
                    distancia_origen > 0
                    and
                    distancia_destino
                    >
                    distancia_origen
                    * (
                        1.0
                        / FACTOR_DISTANCIA_DESTINO
                    )
                ):
                    continue

                # ------------------------------------------------------
                # Simulación del movimiento
                # ------------------------------------------------------

                candidato = (
                    gdf_opt
                    .copy()
                )

                candidato.loc[
                    idx,
                    escenario_col,
                ] = escenario_destino

                estado_candidato = evaluar_estado(
                    gdf_original,
                    candidato,
                    escenario_col,
                    proyecto_col,
                    matriz_indicadores,
                    score_col,
                )

                # ------------------------------------------------------
                # Diferencias
                # ------------------------------------------------------

                mejora = (
                    estado_candidato[
                        "score_global"
                    ]
                    -
                    estado_actual[
                        "score_global"
                    ]
                )

                deterioro_score = (
                    estado_candidato[
                        "score_original"
                    ]
                    -
                    estado_actual[
                        "score_original"
                    ]
                )

                deterioro_cohesion = (
                    estado_candidato[
                        "cohesion"
                    ]
                    -
                    estado_actual[
                        "cohesion"
                    ]
                )

                # ------------------------------------------------------
                # Restricciones de calidad
                # ------------------------------------------------------

                if mejora < MEJORA_MINIMA:
                    continue

                if (
                    deterioro_score
                    <
                    -MAX_DETERIORO_SCORE_ORIGINAL
                ):
                    continue

                if (
                    deterioro_cohesion
                    <
                    -MAX_DETERIORO_COHESION
                ):
                    continue

                # ------------------------------------------------------
                # Guardar mejor candidato
                # ------------------------------------------------------

                if mejora > mejor_mejora:

                    mejor_mejora = mejora

                    mejor_movimiento = {
                        "idx": idx,
                        "proyecto_id": gdf_opt.loc[
                            idx,
                            proyecto_col,
                        ],
                        "escenario_origen":
                            escenario_origen,
                        "escenario_destino":
                            escenario_destino,
                        "distancia_origen":
                            distancia_origen,
                        "distancia_destino":
                            distancia_destino,
                        "mejora_distancia":
                            (
                                distancia_origen
                                -
                                distancia_destino
                            ),
                        "mejora_score_global":
                            mejora,
                        "deterioro_score_original":
                            deterioro_score,
                        "deterioro_cohesion":
                            deterioro_cohesion,
                        "iteracion":
                            iteracion,
                        "motivo":
                            "MEJORA_OBJETIVO_GLOBAL",
                    }

                    mejor_estado = (
                        estado_candidato
                    )

        # --------------------------------------------------------------
        # No hubo movimiento mejorador
        # --------------------------------------------------------------

        if (
            mejor_movimiento is None
            or mejor_estado is None
        ):

            print(
                f"Iteración {iteracion:02d} "
                f"| sin movimientos aceptables"
            )

            break

        # --------------------------------------------------------------
        # Aplicar mejor movimiento
        # --------------------------------------------------------------

        idx = mejor_movimiento[
            "idx"
        ]

        gdf_opt.loc[
            idx,
            escenario_col,
        ] = (
            mejor_movimiento[
                "escenario_destino"
            ]
        )

        estado_anterior = estado_actual

        estado_actual = mejor_estado

        movimientos.append(
            mejor_movimiento
        )

        cambios += 1

        print(
            f"Iteración {iteracion:02d} "
            f"| proyecto="
            f"{mejor_movimiento['proyecto_id']} "
            f"| "
            f"{mejor_movimiento['escenario_origen']}"
            f" -> "
            f"{mejor_movimiento['escenario_destino']} "
            f"| "
            f"Δscore="
            f"{mejor_movimiento['mejora_score_global']:+.6f}"
        )

        # --------------------------------------------------------------
        # Actualizar geometría métrica.
        #
        # La geometría no cambia, solamente cambia su asignación.
        # Por eso no necesitamos reproyectar nuevamente.
        # --------------------------------------------------------------

        if cambios == 0:
            break

    print()
    print(
        f"Movimientos espaciales aceptados : "
        f"{len(movimientos)}"
    )

    print(
        f"Score final de esta etapa        : "
        f"{estado_actual['score_global']:.6f}"
    )

    return (
        gdf_opt,
        movimientos,
    )


# ==============================================================================
# SEGUNDA ETAPA: REFINAMIENTO DE BALANCE
# ==============================================================================

def optimizar_balance(
    gdf_original: gpd.GeoDataFrame,
    gdf_actual: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    matriz_indicadores: pd.DataFrame,
    score_col: Optional[str],
) -> Tuple[
    gpd.GeoDataFrame,
    List[Dict[str, Any]],
]:

    encabezado(
        "7. REFINAMIENTO DE BALANCE"
    )

    gdf_opt = (
        gdf_actual
        .copy()
    )

    movimientos = []

    estado_actual = evaluar_estado(
        gdf_original,
        gdf_opt,
        escenario_col,
        proyecto_col,
        matriz_indicadores,
        score_col,
    )

    for iteracion in range(
        1,
        MAX_ITERACIONES + 1,
    ):

        conteos = (
            gdf_opt
            .groupby(
                escenario_col
            )[
                proyecto_col
            ]
            .count()
            .sort_values()
        )

        if len(conteos) <= 1:
            break

        escenario_pequeno = (
            conteos.index[0]
        )

        escenario_grande = (
            conteos.index[-1]
        )

        cantidad_pequeno = int(
            conteos.iloc[0]
        )

        cantidad_grande = int(
            conteos.iloc[-1]
        )

        diferencia = (
            cantidad_grande
            - cantidad_pequeno
        )

        if (
            diferencia
            <= MAX_DIFERENCIA_TAMANO
        ):
            break

        if (
            cantidad_grande
            <= MIN_PROYECTOS
        ):
            break

        candidatos = gdf_opt[
            gdf_opt[
                escenario_col
            ]
            ==
            escenario_grande
        ]

        if candidatos.empty:
            break

        gdf_metric = preparar_geometria_metrica(
            gdf_opt
        )

        centro_pequeno = (
            gdf_metric[
                gdf_opt[
                    escenario_col
                ]
                ==
                escenario_pequeno
            ]
            .geometry
            .union_all()
            .centroid
        )

        distancias = (
            gdf_metric.loc[
                candidatos.index
            ]
            .geometry
            .distance(
                centro_pequeno
            )
        )

        candidatos_ordenados = (
            distancias
            .sort_values()
            .index
        )

        mejor_candidato = None
        mejor_estado = None
        mejor_mejora = 0.0

        for idx in candidatos_ordenados:

            permitido, motivo = (
                movimiento_permitido(
                    gdf_opt,
                    escenario_col,
                    proyecto_col,
                    idx,
                    escenario_grande,
                    escenario_pequeno,
                )
            )

            if not permitido:
                continue

            candidato = (
                gdf_opt
                .copy()
            )

            candidato.loc[
                idx,
                escenario_col,
            ] = escenario_pequeno

            estado_candidato = evaluar_estado(
                gdf_original,
                candidato,
                escenario_col,
                proyecto_col,
                matriz_indicadores,
                score_col,
            )

            mejora = (
                estado_candidato[
                    "score_global"
                ]
                -
                estado_actual[
                    "score_global"
                ]
            )

            if mejora < MEJORA_MINIMA:
                continue

            deterioro_score = (
                estado_candidato[
                    "score_original"
                ]
                -
                estado_actual[
                    "score_original"
                ]
            )

            if (
                deterioro_score
                <
                -MAX_DETERIORO_SCORE_ORIGINAL
            ):
                continue

            if mejora > mejor_mejora:

                mejor_mejora = mejora

                mejor_candidato = idx

                mejor_estado = (
                    estado_candidato
                )

        if (
            mejor_candidato is None
            or mejor_estado is None
        ):

            print(
                f"Iteración balance {iteracion:02d} "
                f"| no se encontró movimiento mejorador"
            )

            break

        proyecto_id = gdf_opt.loc[
            mejor_candidato,
            proyecto_col,
        ]

        gdf_opt.loc[
            mejor_candidato,
            escenario_col,
        ] = escenario_pequeno

        movimientos.append(
            {
                "iteracion":
                    iteracion,
                "proyecto_id":
                    proyecto_id,
                "escenario_origen":
                    escenario_grande,
                "escenario_destino":
                    escenario_pequeno,
                "tipo_movimiento":
                    "BALANCE_OBJETIVO",
                "mejora_score_global":
                    mejor_mejora,
                "score_global_antes":
                    estado_actual[
                        "score_global"
                    ],
                "score_global_despues":
                    mejor_estado[
                        "score_global"
                    ],
                "cantidad_origen_antes":
                    cantidad_grande,
                "cantidad_destino_antes":
                    cantidad_pequeno,
            }
        )

        estado_actual = (
            mejor_estado
        )

        print(
            f"Iteración balance {iteracion:02d} "
            f"| proyecto={proyecto_id} "
            f"| "
            f"{escenario_grande} -> "
            f"{escenario_pequeno} "
            f"| "
            f"Δscore={mejor_mejora:+.6f}"
        )

    print()
    print(
        f"Movimientos de balance aceptados : "
        f"{len(movimientos)}"
    )

    return (
        gdf_opt,
        movimientos,
    )


# ==============================================================================
# EVALUACIÓN POR ESCENARIO
# ==============================================================================

def evaluar_escenarios(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    score_col: Optional[str],
    matriz_indicadores: pd.DataFrame,
) -> pd.DataFrame:

    filas = []

    conteos = (
        gdf
        .groupby(
            escenario_col
        )[
            proyecto_col
        ]
        .count()
        .sort_values(
            ascending=False
        )
    )

    total = len(gdf)

    for ranking, (
        escenario,
        cantidad,
    ) in enumerate(
        conteos.items(),
        start=1,
    ):

        grupo = gdf[
            gdf[
                escenario_col
            ]
            ==
            escenario
        ]

        porcentaje = (
            cantidad / total
            if total > 0
            else 0.0
        )

        if score_col is not None:

            score_valores = pd.to_numeric(
                grupo[
                    score_col
                ],
                errors="coerce",
            )

            score_medio = float(
                score_valores.mean()
            )

        else:

            score_medio = 0.5

        indicadores_score = []

        for indicador in matriz_indicadores.columns:

            if indicador.startswith(
                "__"
            ):
                continue

            valores = matriz_indicadores.loc[
                grupo.index,
                indicador,
            ]

            if len(valores):

                indicadores_score.append(
                    float(
                        valores.mean()
                    )
                )

        score_indicadores = (
            float(
                np.mean(
                    indicadores_score
                )
            )
            if indicadores_score
            else 0.5
        )

        filas.append(
            {
                "escenario_id":
                    escenario,
                "cantidad_proyectos":
                    int(cantidad),
                "porcentaje_proyectos":
                    porcentaje,
                "score_medio":
                    score_medio,
                "score_indicadores":
                    score_indicadores,
                "cumple_minimo":
                    bool(
                        cantidad
                        >= MIN_PROYECTOS
                    ),
                "ranking_escenario":
                    ranking,
            }
        )

    return pd.DataFrame(
        filas
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
        "8. VALIDACIÓN FINAL"
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

    sin_escenario = int(
        gdf_optimizado[
            escenario_col
        ]
        .isna()
        .sum()
    )

    duplicados = int(
        gdf_optimizado[
            proyecto_col
        ]
        .duplicated()
        .sum()
    )

    conteos = (
        gdf_optimizado
        .groupby(
            escenario_col
        )[
            proyecto_col
        ]
        .count()
    )

    cantidad_escenarios = int(
        len(conteos)
    )

    min_proyectos = (
        int(
            conteos.min()
        )
        if len(conteos)
        else 0
    )

    max_proyectos = (
        int(
            conteos.max()
        )
        if len(conteos)
        else 0
    )

    cobertura = (
        proyectos_originales
        ==
        proyectos_optimizados
    )

    mismos_registros = (
        len(gdf_original)
        ==
        len(gdf_optimizado)
    )

    validacion = {
        "mismos_proyectos":
            bool(cobertura),

        "mismos_registros":
            bool(mismos_registros),

        "duplicados_proyecto":
            duplicados,

        "proyectos_sin_escenario":
            sin_escenario,

        "escenarios":
            cantidad_escenarios,

        "min_proyectos":
            min_proyectos,

        "max_proyectos":
            max_proyectos,

        "proyectos_totales":
            int(len(gdf_original)),

        "proyectos_asignados":
            int(
                len(gdf_optimizado)
                - sin_escenario
            ),

        "cantidad_escenarios_valida":
            bool(
                MIN_ESCENARIOS
                <= cantidad_escenarios
                <= MAX_ESCENARIOS
            ),

        "todos_los_escenarios_cumplen_minimo":
            bool(
                min_proyectos
                >= MIN_PROYECTOS
            ),
    }

    print(
        f"Proyectos totales            : "
        f"{validacion['proyectos_totales']:,}"
    )

    print(
        f"Proyectos asignados          : "
        f"{validacion['proyectos_asignados']:,}"
    )

    print(
        f"Proyectos sin escenario      : "
        f"{sin_escenario}"
    )

    print(
        f"Duplicados                   : "
        f"{duplicados}"
    )

    print(
        f"Escenarios                   : "
        f"{cantidad_escenarios}"
    )

    print(
        f"Mínimo proyectos             : "
        f"{min_proyectos}"
    )

    print(
        f"Máximo proyectos             : "
        f"{max_proyectos}"
    )

    print(
        f"Cobertura proyectos          : "
        f"{cobertura}"
    )

    print(
        f"Cantidad escenarios válida   : "
        f"{validacion['cantidad_escenarios_valida']}"
    )

    print(
        f"Mínimo por escenario válido  : "
        f"{validacion['todos_los_escenarios_cumplen_minimo']}"
    )

    if not cobertura:

        raise ValueError(
            "La optimización alteró el conjunto de proyectos."
        )

    if not mismos_registros:

        raise ValueError(
            "La optimización alteró la cantidad de registros."
        )

    if duplicados > 0:

        raise ValueError(
            "La optimización generó proyectos duplicados."
        )

    if sin_escenario > 0:

        raise ValueError(
            "Existen proyectos sin escenario."
        )

    if min_proyectos < MIN_PROYECTOS:

        raise ValueError(
            "Existe un escenario por debajo del mínimo "
            f"de {MIN_PROYECTOS} proyectos."
        )

    print()
    print(
        "VALIDACIÓN FINAL: OK"
    )

    return validacion


# ==============================================================================
# COMPARACIÓN ORIGINAL VS OPTIMIZADO
# ==============================================================================

def construir_resumen_metricas(
    original: Dict[str, float],
    optimizado: Dict[str, float],
) -> pd.DataFrame:

    filas = []

    claves = [
        "cobertura",
        "estructura_escenarios",
        "tamano",
        "cohesion",
        "balance",
        "indicadores",
        "score_original",
        "estabilidad",
        "score_global",
    ]

    for clave in claves:

        valor_original = float(
            original.get(
                clave,
                np.nan,
            )
        )

        valor_optimizado = float(
            optimizado.get(
                clave,
                np.nan,
            )
        )

        filas.append(
            {
                "metrica":
                    clave,
                "original":
                    valor_original,
                "optimizado":
                    valor_optimizado,
                "cambio":
                    valor_optimizado
                    -
                    valor_original,
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
    evaluacion_escenarios: pd.DataFrame,
    movimientos: List[Dict[str, Any]],
    resumen_metricas: pd.DataFrame,
    metricas_originales: Dict[str, float],
    metricas_optimizadas: Dict[str, float],
    validacion: Dict[str, Any],
    tiempo_segundos: float,
    escenario_col: str,
) -> None:

    encabezado(
        "9. EXPORTACIÓN DE RESULTADOS"
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

    # --------------------------------------------------------------------------
    # Evaluación por escenario
    # --------------------------------------------------------------------------

    evaluacion_escenarios.to_csv(
        OUTPUT_EVALUACION,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------------------
    # Movimientos
    # --------------------------------------------------------------------------

    if movimientos:

        movimientos_df = pd.DataFrame(
            movimientos
        )

    else:

        movimientos_df = pd.DataFrame(
            columns=[
                "iteracion",
                "proyecto_id",
                "escenario_origen",
                "escenario_destino",
                "mejora_score_global",
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

    resumen_metricas.to_csv(
        OUTPUT_RESUMEN,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------------------

    cantidad_escenarios = int(
        gdf_optimizado[
            escenario_col
        ]
        .nunique()
    )

    metadata = {
        "proceso":
            PROCESO,

        "nombre":
            "Optimización de escenarios "
            "territoriales AMBA",

        "version":
            VERSION,

        "fecha_ejecucion":
            pd.Timestamp.now(
                tz="America/Argentina/Buenos_Aires"
            ).isoformat(),

        "project_root":
            str(PROJECT_ROOT),

        "input":
            str(INPUT_ESCENARIOS),

        "output":
            str(OUTPUT_OPTIMIZADO),

        "proyectos":
            int(len(gdf_optimizado)),

        "escenarios":
            cantidad_escenarios,

        "movimientos":
            len(movimientos),

        "configuracion": {
            "min_proyectos":
                MIN_PROYECTOS,

            "min_escenarios":
                MIN_ESCENARIOS,

            "max_escenarios":
                MAX_ESCENARIOS,

            "max_iteraciones":
                MAX_ITERACIONES,

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

            "mejora_minima":
                MEJORA_MINIMA,

            "max_deterioro_score_original":
                MAX_DETERIORO_SCORE_ORIGINAL,

            "max_deterioro_cohesion":
                MAX_DETERIORO_COHESION,

            "max_diferencia_tamano":
                MAX_DIFERENCIA_TAMANO,
        },

        "metricas_originales":
            metricas_originales,

        "metricas_optimizadas":
            metricas_optimizadas,

        "validacion":
            validacion,

        "duracion_segundos":
            float(tiempo_segundos),
    }

    OUTPUT_METADATA.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"GeoParquet : {OUTPUT_OPTIMIZADO}"
    )

    print(
        f"CSV        : {OUTPUT_CSV}"
    )

    print(
        f"Evaluación  : {OUTPUT_EVALUACION}"
    )

    print(
        f"Movimientos : {OUTPUT_MOVIMIENTOS}"
    )

    print(
        f"Resumen     : {OUTPUT_RESUMEN}"
    )

    print(
        f"Metadata    : {OUTPUT_METADATA}"
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
    # Validar configuración
    # --------------------------------------------------------------------------

    validar_pesos()

    print()
    print(
        "CONFIGURACIÓN V2.0"
    )

    print(
        f"  Mínimo proyectos              : "
        f"{MIN_PROYECTOS}"
    )

    print(
        f"  Escenarios válidos             : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
    )

    print(
        f"  Máx. iteraciones               : "
        f"{MAX_ITERACIONES}"
    )

    print(
        f"  Peso cohesión                  : "
        f"{PESO_COHESION:.0%}"
    )

    print(
        f"  Peso balance                   : "
        f"{PESO_BALANCE:.0%}"
    )

    print(
        f"  Peso indicadores               : "
        f"{PESO_INDICADORES:.0%}"
    )

    print(
        f"  Peso score original            : "
        f"{PESO_SCORE_ORIGINAL:.0%}"
    )

    print(
        f"  Peso estabilidad               : "
        f"{PESO_ESTABILIDAD:.0%}"
    )

    print(
        f"  Mejora mínima                  : "
        f"{MEJORA_MINIMA}"
    )

    print(
        f"  Deterioro máximo score original: "
        f"{MAX_DETERIORO_SCORE_ORIGINAL}"
    )

    # ==========================================================================
    # 1. CARGA
    # ==========================================================================

    gdf_original = cargar_escenarios()

    # ==========================================================================
    # 2. EVALUACIÓN PROCESO 28
    # ==========================================================================

    evaluacion_proceso_28 = (
        cargar_evaluacion()
    )

    # ==========================================================================
    # 3. RECOMENDACIONES
    # ==========================================================================

    recomendaciones = (
        cargar_recomendaciones()
    )

    # ==========================================================================
    # 4. VALIDACIÓN
    # ==========================================================================

    escenario_col, proyecto_col = (
        validar_entrada(
            gdf_original
        )
    )

    # ==========================================================================
    # 5. INDICADORES
    # ==========================================================================

    indicadores = detectar_indicadores(
        gdf_original,
        escenario_col,
        proyecto_col,
    )

    matriz_indicadores = (
        construir_matriz_indicadores(
            gdf_original,
            indicadores,
        )
    )

    score_col = detectar_score_original(
        gdf_original
    )

    print()

    if score_col:

        print(
            f"Score territorial utilizado : "
            f"{score_col}"
        )

    else:

        print(
            "Score territorial utilizado : "
            "NO DISPONIBLE"
        )

    # ==========================================================================
    # 6. EVALUACIÓN BASE
    # ==========================================================================

    encabezado(
        "6. EVALUACIÓN DE ESTADO ORIGINAL"
    )

    metricas_originales = evaluar_estado(
        gdf_original,
        gdf_original,
        escenario_col,
        proyecto_col,
        matriz_indicadores,
        score_col,
    )

    for clave, valor in (
        metricas_originales.items()
    ):

        imprimir_metrica(
            clave,
            valor,
        )

    # ==========================================================================
    # 7. OPTIMIZACIÓN ESPACIAL + FUNCIÓN OBJETIVO
    # ==========================================================================

    gdf_optimizado, movimientos_espaciales = (
        optimizar_asignacion(
            gdf_original,
            escenario_col,
            proyecto_col,
            matriz_indicadores,
            score_col,
        )
    )

    # ==========================================================================
    # 8. REFINAMIENTO DE BALANCE
    # ==========================================================================

    gdf_optimizado, movimientos_balance = (
        optimizar_balance(
            gdf_original,
            gdf_optimizado,
            escenario_col,
            proyecto_col,
            matriz_indicadores,
            score_col,
        )
    )

    movimientos = (
        movimientos_espaciales
        +
        movimientos_balance
    )

    # ==========================================================================
    # 9. EVALUACIÓN FINAL
    # ==========================================================================

    encabezado(
        "10. EVALUACIÓN DEL RESULTADO"
    )

    metricas_optimizadas = evaluar_estado(
        gdf_original,
        gdf_optimizado,
        escenario_col,
        proyecto_col,
        matriz_indicadores,
        score_col,
    )

    print()
    print(
        f"{'MÉTRICA':<35}"
        f"{'ORIGINAL':>15}"
        f"{'OPTIMIZADO':>15}"
        f"{'CAMBIO':>15}"
    )

    print(
        "-" * 80
    )

    for clave in [
        "cobertura",
        "estructura_escenarios",
        "tamano",
        "cohesion",
        "balance",
        "indicadores",
        "score_original",
        "estabilidad",
        "score_global",
    ]:

        original = metricas_originales[
            clave
        ]

        optimizado = metricas_optimizadas[
            clave
        ]

        cambio = (
            optimizado
            -
            original
        )

        print(
            f"{clave:<35}"
            f"{original:>15.6f}"
            f"{optimizado:>15.6f}"
            f"{cambio:>+15.6f}"
        )

    # ==========================================================================
    # 10. EVALUACIÓN POR ESCENARIO
    # ==========================================================================

    evaluacion_final = evaluar_escenarios(
        gdf_optimizado,
        escenario_col,
        proyecto_col,
        score_col,
        matriz_indicadores,
    )

    # ==========================================================================
    # 11. VALIDACIÓN FINAL
    # ==========================================================================

    validacion = validar_resultado(
        gdf_original,
        gdf_optimizado,
        escenario_col,
        proyecto_col,
    )

    # ==========================================================================
    # 12. RESUMEN
    # ==========================================================================

    resumen_metricas = (
        construir_resumen_metricas(
            metricas_originales,
            metricas_optimizadas,
        )
    )

    # ==========================================================================
    # 13. EXPORTACIÓN
    # ==========================================================================

    tiempo = (
        time.perf_counter()
        -
        inicio
    )

    exportar_resultados(
        gdf_optimizado,
        evaluacion_final,
        movimientos,
        resumen_metricas,
        metricas_originales,
        metricas_optimizadas,
        validacion,
        tiempo,
        escenario_col,
    )

    # ==========================================================================
    # 14. RESULTADO FINAL
    # ==========================================================================

    encabezado(
        "29 - PROCESO FINALIZADO"
    )

    print(
        f"Versión                   : {VERSION}"
    )

    print(
        f"Proyectos procesados      : "
        f"{len(gdf_optimizado):,}"
    )

    print(
        f"Escenarios                : "
        f"{gdf_optimizado[escenario_col].nunique()}"
    )

    print(
        f"Movimientos espaciales    : "
        f"{len(movimientos_espaciales)}"
    )

    print(
        f"Movimientos balance       : "
        f"{len(movimientos_balance)}"
    )

    print(
        f"Movimientos totales       : "
        f"{len(movimientos)}"
    )

    print()

    print(
        f"Score global original     : "
        f"{metricas_originales['score_global']:.6f}"
    )

    print(
        f"Score global optimizado   : "
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
        f"Mejora global              : "
        f"{mejora:+.6f}"
    )

    print()

    print(
        "ESCENARIOS OPTIMIZADOS"
    )

    conteos = (
        gdf_optimizado
        .groupby(
            escenario_col
        )[
            proyecto_col
        ]
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
        "SALIDAS"
    )

    print(
        f"  {OUTPUT_OPTIMIZADO}"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {OUTPUT_EVALUACION}"
    )

    print(
        f"  {OUTPUT_MOVIMIENTOS}"
    )

    print(
        f"  {OUTPUT_RESUMEN}"
    )

    print(
        f"  {OUTPUT_METADATA}"
    )

    print()

    print(
        f"Duración                  : "
        f"{tiempo:.2f} segundos"
    )

    print()

    print(
        "=" * 90
    )


# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()