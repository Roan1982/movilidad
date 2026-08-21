# -*- coding: utf-8 -*-

"""
23 - CONSTRUCCIÓN DEL ÍNDICE DE CENTRALIDAD ESTRUCTURAL AMBA

V1

Construye un índice compuesto de centralidad estructural para las
144 centralidades de movilidad SUBE del AMBA.

Entrada principal:
    data/processed/validacion_infraestructura_centralidades_amba/
        validacion_infraestructura_centralidades_amba.parquet

Componentes:

    DEMANDA                         30%
    INFRAESTRUCTURA                 25%
    INTERMODALIDAD                  20%
    CONECTIVIDAD                    15%
    INTEGRACIÓN TERRITORIAL        10%

El script genera:

    - índice estructural 0-100
    - índice estructural robusto 0-100
    - ranking
    - categorías
    - índices por dimensión
    - diagnóstico demanda / infraestructura
    - déficit relativo de infraestructura
    - mapas
    - gráficos
    - CSV
    - Parquet
    - GeoPackage
    - JSON

IMPORTANTE:

Este script no modifica los procesos 21 ni 22.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V1"

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "validacion_infraestructura_centralidades_amba"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "indice_centralidad_estructural_amba"
)

INPUT_FILE = (
    INPUT_DIR
    / "validacion_infraestructura_centralidades_amba.parquet"
)

CRS_GEOGRAFICO = "EPSG:4326"

CRS_METRICO = "EPSG:22185"


# =============================================================================
# PESOS
# =============================================================================

PESO_DEMANDA = 0.30
PESO_INFRAESTRUCTURA = 0.25
PESO_INTERMODALIDAD = 0.20
PESO_CONECTIVIDAD = 0.15
PESO_INTEGRACION = 0.10


# =============================================================================
# ARCHIVOS DE SALIDA
# =============================================================================

SALIDA_PARQUET = (
    OUTPUT_DIR
    / "indice_centralidad_estructural_amba.parquet"
)

SALIDA_CSV = (
    OUTPUT_DIR
    / "indice_centralidad_estructural_amba.csv"
)

SALIDA_GPKG = (
    OUTPUT_DIR
    / "indice_centralidad_estructural_amba.gpkg"
)

SALIDA_JSON = (
    OUTPUT_DIR
    / "indice_centralidad_estructural_amba_resumen.json"
)


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto: str) -> None:

    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def subtitulo(texto: str) -> None:

    print()
    print("-" * 78)
    print(texto)
    print("-" * 78)


def buscar_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = False,
) -> str | None:

    for columna in candidatos:

        if columna in df.columns:
            return columna

    if obligatoria:

        raise ValueError(
            "No se encontró ninguna columna requerida: "
            + ", ".join(candidatos)
        )

    return None


def numerico(
    serie: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        serie,
        errors="coerce",
    ).fillna(0.0)


def seguro_porcentaje(
    serie: pd.Series,
) -> pd.Series:

    return numerico(
        serie
    ).clip(
        0,
        100,
    )


def normalizar_minmax(
    serie: pd.Series,
) -> pd.Series:

    valores = numerico(serie)

    minimo = valores.min()
    maximo = valores.max()

    if maximo == minimo:

        return pd.Series(
            50.0,
            index=serie.index,
        )

    return (
        (
            valores - minimo
        )
        /
        (
            maximo - minimo
        )
        * 100
    ).clip(
        0,
        100,
    )


def normalizar_percentil(
    serie: pd.Series,
) -> pd.Series:

    """
    Normalización por rango percentil.

    Evita que un outlier extremo domine la escala.
    """

    valores = numerico(serie)

    if len(valores) <= 1:

        return pd.Series(
            100.0,
            index=serie.index,
        )

    rangos = valores.rank(
        method="average",
        pct=True,
    )

    return (
        rangos * 100
    ).clip(
        0,
        100,
    )


def normalizar_log(
    serie: pd.Series,
) -> pd.Series:

    """
    Normalización logarítmica + min-max.
    """

    valores = numerico(serie)

    transformados = np.log1p(
        valores.clip(
            lower=0
        )
    )

    minimo = transformados.min()
    maximo = transformados.max()

    if maximo == minimo:

        return pd.Series(
            50.0,
            index=serie.index,
        )

    return (
        (
            transformados - minimo
        )
        /
        (
            maximo - minimo
        )
        * 100
    ).clip(
        0,
        100,
    )


def valor_json(
    valor: Any,
) -> Any:

    if isinstance(valor, np.integer):
        return int(valor)

    if isinstance(valor, np.floating):

        if np.isnan(valor):
            return None

        return float(valor)

    if isinstance(valor, np.bool_):
        return bool(valor)

    if pd.isna(valor):
        return None

    return valor


# =============================================================================
# CARGA
# =============================================================================

def cargar_datos() -> gpd.GeoDataFrame:

    titulo(
        "1. CARGANDO RESULTADOS DEL PROCESO 22"
    )

    print(
        f"Archivo:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "No existe el resultado del proceso 22:\n"
            f"{INPUT_FILE}"
        )

    gdf = gpd.read_parquet(
        INPUT_FILE
    )

    print()
    print(
        f"Registros: {len(gdf):,}"
    )

    print(
        f"Columnas: {len(gdf.columns)}"
    )

    print(
        f"CRS: {gdf.crs}"
    )

    if gdf.crs is None:

        raise ValueError(
            "El archivo de entrada no tiene CRS."
        )

    return gdf


# =============================================================================
# VALIDACIÓN DE ENTRADA
# =============================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> str:

    titulo(
        "2. VALIDANDO RESULTADOS DEL PROCESO 22"
    )

    # -------------------------------------------------------------------------
    # GEOMETRÍA
    # -------------------------------------------------------------------------

    nulas = int(
        gdf.geometry.isna().sum()
    )

    vacias = int(
        gdf.geometry.is_empty.sum()
    )

    invalidas = int(
        (~gdf.geometry.is_valid).sum()
    )

    print(
        f"Geometrías nulas: {nulas}"
    )

    print(
        f"Geometrías vacías: {vacias}"
    )

    print(
        f"Geometrías inválidas: {invalidas}"
    )

    if nulas or vacias or invalidas:

        raise ValueError(
            "La geometría de entrada no es válida."
        )

    # -------------------------------------------------------------------------
    # NODO
    # -------------------------------------------------------------------------

    nodo_id = buscar_columna(
        gdf,
        [
            "nodo_id",
            "id_nodo",
            "node_id",
        ],
        obligatoria=True,
    )

    duplicados = int(
        gdf[nodo_id]
        .duplicated()
        .sum()
    )

    print(
        f"Columna identificadora: {nodo_id}"
    )

    print(
        f"IDs duplicados: {duplicados}"
    )

    if duplicados:

        raise ValueError(
            "Existen identificadores de centralidad duplicados."
        )

    # -------------------------------------------------------------------------
    # CANTIDAD
    # -------------------------------------------------------------------------

    if len(gdf) != 144:

        print()
        print(
            "ADVERTENCIA:"
        )

        print(
            "Se esperaban 144 centralidades."
        )

        print(
            f"Se encontraron {len(gdf)}."
        )

    else:

        print(
            "Centralidades esperadas: 144"
        )

    # -------------------------------------------------------------------------
    # COLUMNAS DISPONIBLES
    # -------------------------------------------------------------------------

    print()
    print(
        "Columnas relevantes encontradas:"
    )

    columnas_interesantes = [
        "score_demanda",
        "score_densidad",
        "score_conectividad",
        "score_alcance",
        "score_integracion",
        "score_intermodalidad",
        "score_intermodalidad_500m",
        "instalaciones_250m",
        "instalaciones_500m",
        "instalaciones_1000m",
        "intercambiadores_250m",
        "intercambiadores_500m",
        "intercambiadores_1000m",
        "cantidad_modos_500m",
        "cantidad_modos_1000m",
        "ferrocarril_500m",
        "subte_500m",
        "autobus_500m",
        "fluvial_500m",
        "tranvia_500m",
        "soporte_fisico_preliminar",
    ]

    for columna in columnas_interesantes:

        if columna in gdf.columns:

            print(
                f"  OK  {columna}"
            )

    return nodo_id


# =============================================================================
# DEMANDA
# =============================================================================

def calcular_demanda(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "3. CALCULANDO COMPONENTE DE DEMANDA"
    )

    gdf = gdf.copy()

    columna = buscar_columna(
        gdf,
        [
            "score_demanda",
            "demanda_score",
        ],
    )

    if columna is not None:

        print(
            f"Fuente principal: {columna}"
        )

        gdf[
            "demanda_base_23"
        ] = numerico(
            gdf[columna]
        )

    else:

        print(
            "No existe score_demanda."
        )

        candidatos = [
            "operaciones",
            "operaciones_2025",
            "cantidad_operaciones",
            "viajes",
            "demanda",
        ]

        columna = buscar_columna(
            gdf,
            candidatos,
        )

        if columna is None:

            raise ValueError(
                "No se encontró una variable de demanda "
                "para construir el componente."
            )

        print(
            f"Fuente alternativa: {columna}"
        )

        gdf[
            "demanda_base_23"
        ] = numerico(
            gdf[columna]
        )

    gdf[
        "indice_demanda_estructural"
    ] = normalizar_percentil(
        gdf[
            "demanda_base_23"
        ]
    )

    return gdf


# =============================================================================
# INFRAESTRUCTURA
# =============================================================================

def calcular_infraestructura(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "4. CALCULANDO COMPONENTE DE INFRAESTRUCTURA"
    )

    gdf = gdf.copy()

    # -------------------------------------------------------------------------
    # INSTALACIONES
    # -------------------------------------------------------------------------

    columna = buscar_columna(
        gdf,
        [
            "instalaciones_500m",
        ],
        obligatoria=True,
    )

    print(
        f"Instalaciones: {columna}"
    )

    instalaciones = normalizar_log(
        gdf[columna]
    )

    # -------------------------------------------------------------------------
    # INTERCAMBIADORES
    # -------------------------------------------------------------------------

    columna = buscar_columna(
        gdf,
        [
            "intercambiadores_500m",
        ],
        obligatoria=True,
    )

    print(
        f"Intercambiadores: {columna}"
    )

    intercambiadores = normalizar_log(
        gdf[columna]
    )

    # -------------------------------------------------------------------------
    # DIVERSIDAD MODAL
    # -------------------------------------------------------------------------

    columna = buscar_columna(
        gdf,
        [
            "cantidad_modos_500m",
        ],
        obligatoria=True,
    )

    print(
        f"Diversidad modal: {columna}"
    )

    diversidad = (
        numerico(
            gdf[columna]
        )
        / 5.0
        * 100
    ).clip(
        0,
        100,
    )

    # -------------------------------------------------------------------------
    # SOPORTE FÍSICO
    # -------------------------------------------------------------------------

    soporte = None

    if (
        "soporte_fisico_preliminar"
        in gdf.columns
    ):

        soporte = seguro_porcentaje(
            gdf[
                "soporte_fisico_preliminar"
            ]
        )

    else:

        soporte = (
            instalaciones
            * 0.45
            +
            intercambiadores
            * 0.30
            +
            diversidad
            * 0.25
        )

    # -------------------------------------------------------------------------
    # COMPONENTE
    # -------------------------------------------------------------------------

    gdf[
        "infraestructura_instalaciones"
    ] = instalaciones.round(4)

    gdf[
        "infraestructura_intercambiadores"
    ] = intercambiadores.round(4)

    gdf[
        "infraestructura_diversidad_modal"
    ] = diversidad.round(4)

    gdf[
        "indice_infraestructura_estructural"
    ] = (
        instalaciones * 0.35
        +
        intercambiadores * 0.30
        +
        diversidad * 0.20
        +
        soporte * 0.15
    ).round(4)

    return gdf


# =============================================================================
# INTERMODALIDAD
# =============================================================================

def calcular_intermodalidad(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "5. CALCULANDO COMPONENTE DE INTERMODALIDAD"
    )

    gdf = gdf.copy()

    variables = []

    # -------------------------------------------------------------------------
    # SCORE PROCESO 21
    # -------------------------------------------------------------------------

    for columna in [
        "score_intermodalidad",
        "score_intermodalidad_500m",
    ]:

        if columna in gdf.columns:

            print(
                f"Encontrado: {columna}"
            )

            variables.append(
                normalizar_percentil(
                    gdf[columna]
                )
            )

    # -------------------------------------------------------------------------
    # DIVERSIDAD MODAL
    # -------------------------------------------------------------------------

    if (
        "cantidad_modos_500m"
        in gdf.columns
    ):

        variables.append(
            (
                numerico(
                    gdf[
                        "cantidad_modos_500m"
                    ]
                )
                / 5
                * 100
            ).clip(
                0,
                100,
            )
        )

    # -------------------------------------------------------------------------
    # INTERCAMBIADORES
    # -------------------------------------------------------------------------

    if (
        "intercambiadores_500m"
        in gdf.columns
    ):

        variables.append(
            normalizar_log(
                gdf[
                    "intercambiadores_500m"
                ]
            )
        )

    if not variables:

        raise ValueError(
            "No existen variables suficientes "
            "para calcular intermodalidad."
        )

    matriz = pd.concat(
        variables,
        axis=1,
    )

    gdf[
        "indice_intermodalidad_estructural"
    ] = matriz.mean(
        axis=1
    ).round(4)

    return gdf


# =============================================================================
# CONECTIVIDAD
# =============================================================================

def calcular_conectividad(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "6. CALCULANDO COMPONENTE DE CONECTIVIDAD"
    )

    gdf = gdf.copy()

    variables = []

    for columna in [
        "score_conectividad",
        "score_alcance",
    ]:

        if columna in gdf.columns:

            print(
                f"Utilizando: {columna}"
            )

            variables.append(
                normalizar_percentil(
                    gdf[columna]
                )
            )

    # -------------------------------------------------------------------------
    # ALCANCE ESPACIAL DE INFRAESTRUCTURA
    # -------------------------------------------------------------------------

    if (
        "instalaciones_1000m"
        in gdf.columns
        and
        "instalaciones_500m"
        in gdf.columns
    ):

        alcance = (
            numerico(
                gdf[
                    "instalaciones_1000m"
                ]
            )
            /
            (
                numerico(
                    gdf[
                        "instalaciones_500m"
                    ]
                )
                + 1
            )
        )

        variables.append(
            normalizar_percentil(
                alcance
            )
        )

    # -------------------------------------------------------------------------
    # FERROCARRIL
    # -------------------------------------------------------------------------

    if (
        "ferrocarril_500m"
        in gdf.columns
    ):

        variables.append(
            numerico(
                gdf[
                    "ferrocarril_500m"
                ]
            )
            .gt(0)
            .astype(float)
            * 100
        )

    if not variables:

        raise ValueError(
            "No existen variables suficientes "
            "para calcular conectividad."
        )

    matriz = pd.concat(
        variables,
        axis=1,
    )

    gdf[
        "indice_conectividad_estructural"
    ] = matriz.mean(
        axis=1
    ).round(4)

    return gdf


# =============================================================================
# INTEGRACIÓN TERRITORIAL
# =============================================================================

def calcular_integracion(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "7. CALCULANDO COMPONENTE DE INTEGRACIÓN TERRITORIAL"
    )

    gdf = gdf.copy()

    variables = []

    for columna in [
        "score_integracion",
        "score_densidad",
    ]:

        if columna in gdf.columns:

            print(
                f"Utilizando: {columna}"
            )

            variables.append(
                normalizar_percentil(
                    gdf[columna]
                )
            )

    # -------------------------------------------------------------------------
    # CATEGORÍA / SCORE INTERMODALIDAD
    # -------------------------------------------------------------------------

    if (
        "score_intermodalidad"
        in gdf.columns
    ):

        variables.append(
            normalizar_percentil(
                gdf[
                    "score_intermodalidad"
                ]
            )
        )

    if not variables:

        # Si no existen los componentes originales,
        # utilizamos una medida física de respaldo.

        if (
            "instalaciones_1000m"
            in gdf.columns
        ):

            variables.append(
                normalizar_log(
                    gdf[
                        "instalaciones_1000m"
                    ]
                )
            )

    if not variables:

        raise ValueError(
            "No existen variables suficientes "
            "para calcular integración territorial."
        )

    matriz = pd.concat(
        variables,
        axis=1,
    )

    gdf[
        "indice_integracion_territorial"
    ] = matriz.mean(
        axis=1
    ).round(4)

    return gdf


# =============================================================================
# ÍNDICE COMPUESTO
# =============================================================================

def construir_indice(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "8. CONSTRUYENDO ÍNDICE DE CENTRALIDAD ESTRUCTURAL"
    )

    print()
    print(
        f"Demanda:          {PESO_DEMANDA:.0%}"
    )

    print(
        f"Infraestructura:  {PESO_INFRAESTRUCTURA:.0%}"
    )

    print(
        f"Intermodalidad:   {PESO_INTERMODALIDAD:.0%}"
    )

    print(
        f"Conectividad:     {PESO_CONECTIVIDAD:.0%}"
    )

    print(
        f"Integración:      {PESO_INTEGRACION:.0%}"
    )

    suma_pesos = (
        PESO_DEMANDA
        + PESO_INFRAESTRUCTURA
        + PESO_INTERMODALIDAD
        + PESO_CONECTIVIDAD
        + PESO_INTEGRACION
    )

    if not math.isclose(
        suma_pesos,
        1.0,
        abs_tol=0.0001,
    ):

        raise ValueError(
            "Los pesos del índice no suman 100%."
        )

    # -------------------------------------------------------------------------
    # ÍNDICE
    # -------------------------------------------------------------------------

    gdf[
        "indice_centralidad_estructural"
    ] = (
        gdf[
            "indice_demanda_estructural"
        ]
        * PESO_DEMANDA

        +

        gdf[
            "indice_infraestructura_estructural"
        ]
        * PESO_INFRAESTRUCTURA

        +

        gdf[
            "indice_intermodalidad_estructural"
        ]
        * PESO_INTERMODALIDAD

        +

        gdf[
            "indice_conectividad_estructural"
        ]
        * PESO_CONECTIVIDAD

        +

        gdf[
            "indice_integracion_territorial"
        ]
        * PESO_INTEGRACION
    ).round(2)

    # -------------------------------------------------------------------------
    # ÍNDICE ROBUSTO
    #
    # Se aplica percentil al índice compuesto.
    # Esto produce una segunda lectura relativa del sistema.
    # -------------------------------------------------------------------------

    gdf[
        "indice_centralidad_estructural_robusto"
    ] = normalizar_percentil(
        gdf[
            "indice_centralidad_estructural"
        ]
    ).round(2)

    # -------------------------------------------------------------------------
    # RANKING
    # -------------------------------------------------------------------------

    gdf[
        "ranking_centralidad_estructural"
    ] = (
        gdf[
            "indice_centralidad_estructural"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    gdf[
        "ranking_centralidad_estructural_robusto"
    ] = (
        gdf[
            "indice_centralidad_estructural_robusto"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return gdf


# =============================================================================
# CATEGORÍAS
# =============================================================================

def categoria_indice(
    valor: float,
) -> str:

    if valor >= 80:
        return "CENTRALIDAD_ESTRUCTURAL_MUY_ALTA"

    if valor >= 60:
        return "CENTRALIDAD_ESTRUCTURAL_ALTA"

    if valor >= 40:
        return "CENTRALIDAD_ESTRUCTURAL_MEDIA"

    if valor >= 20:
        return "CENTRALIDAD_ESTRUCTURAL_BAJA"

    return "CENTRALIDAD_ESTRUCTURAL_MUY_BAJA"


def clasificar_indice(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "9. CLASIFICANDO CENTRALIDADES"
    )

    gdf = gdf.copy()

    gdf[
        "categoria_centralidad_estructural"
    ] = gdf[
        "indice_centralidad_estructural"
    ].apply(
        categoria_indice
    )

    gdf[
        "categoria_centralidad_estructural_robusta"
    ] = gdf[
        "indice_centralidad_estructural_robusto"
    ].apply(
        categoria_indice
    )

    return gdf


# =============================================================================
# DIAGNÓSTICO DEMANDA / INFRAESTRUCTURA
# =============================================================================

def clasificar_demanda_infraestructura(
    demanda: float,
    infraestructura: float,
) -> str:

    demanda_alta = demanda >= 50
    infraestructura_alta = infraestructura >= 50

    if demanda_alta and infraestructura_alta:

        return "CENTRALIDAD_ESTRATEGICA"

    if demanda_alta and not infraestructura_alta:

        return "DEFICIT_INFRAESTRUCTURAL"

    if not demanda_alta and infraestructura_alta:

        return "OFERTA_INFRAESTRUCTURAL_RELATIVA"

    return "CENTRALIDAD_SECUNDARIA"


def calcular_diagnostico(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "10. DIAGNÓSTICO DEMANDA VS INFRAESTRUCTURA"
    )

    gdf = gdf.copy()

    gdf[
        "brecha_demanda_infraestructura"
    ] = (
        gdf[
            "indice_demanda_estructural"
        ]
        -
        gdf[
            "indice_infraestructura_estructural"
        ]
    ).round(2)

    gdf[
        "deficit_infraestructura"
    ] = (
        gdf[
            "brecha_demanda_infraestructura"
        ]
        .clip(
            lower=0
        )
    ).round(2)

    gdf[
        "diagnostico_demanda_infraestructura"
    ] = [
        clasificar_demanda_infraestructura(
            demanda,
            infraestructura,
        )
        for demanda, infraestructura in zip(
            gdf[
                "indice_demanda_estructural"
            ],
            gdf[
                "indice_infraestructura_estructural"
            ],
        )
    ]

    return gdf


# =============================================================================
# PRIORIDAD DE INTERVENCIÓN
# =============================================================================

def calcular_prioridad(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo(
        "11. CALCULANDO PRIORIDAD DE INTERVENCIÓN"
    )

    gdf = gdf.copy()

    # Una prioridad alta surge cuando:
    #
    # - la demanda es alta
    # - el déficit de infraestructura es alto
    #
    # Esto evita priorizar solamente por tamaño actual.

    gdf[
        "prioridad_intervencion"
    ] = (
        gdf[
            "indice_demanda_estructural"
        ]
        * 0.60
        +
        gdf[
            "deficit_infraestructura"
        ]
        * 0.40
    ).round(2)

    gdf[
        "ranking_prioridad_intervencion"
    ] = (
        gdf[
            "prioridad_intervencion"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return gdf


# =============================================================================
# TOP 20
# =============================================================================

def imprimir_top20(
    gdf: gpd.GeoDataFrame,
) -> None:

    titulo(
        "TOP 20 CENTRALIDADES POR ÍNDICE ESTRUCTURAL"
    )

    columnas = [
        "nodo_id",
        "indice_centralidad_estructural",
        "indice_centralidad_estructural_robusto",
        "ranking_centralidad_estructural",
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
        "diagnostico_demanda_infraestructura",
        "categoria_centralidad_estructural",
    ]

    columnas = [
        c
        for c in columnas
        if c in gdf.columns
    ]

    top = (
        gdf[
            columnas
        ]
        .sort_values(
            "ranking_centralidad_estructural"
        )
        .head(20)
    )

    print(
        top.to_string(
            index=False
        )
    )


def imprimir_top_prioridad(
    gdf: gpd.GeoDataFrame,
) -> None:

    titulo(
        "TOP 20 CENTRALIDADES POR PRIORIDAD DE INTERVENCIÓN"
    )

    columnas = [
        "nodo_id",
        "prioridad_intervencion",
        "ranking_prioridad_intervencion",
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "deficit_infraestructura",
        "diagnostico_demanda_infraestructura",
    ]

    columnas = [
        c
        for c in columnas
        if c in gdf.columns
    ]

    top = (
        gdf[
            columnas
        ]
        .sort_values(
            "ranking_prioridad_intervencion"
        )
        .head(20)
    )

    print(
        top.to_string(
            index=False
        )
    )


# =============================================================================
# RESUMEN
# =============================================================================

def construir_resumen(
    gdf: gpd.GeoDataFrame,
) -> dict:

    resumen = {
        "script": (
            "23_construir_indice_centralidad_"
            "estructural_amba.py"
        ),
        "version": VERSION,
        "centralidades": int(
            len(gdf)
        ),
        "pesos": {
            "demanda": PESO_DEMANDA,
            "infraestructura": PESO_INFRAESTRUCTURA,
            "intermodalidad": PESO_INTERMODALIDAD,
            "conectividad": PESO_CONECTIVIDAD,
            "integracion": PESO_INTEGRACION,
        },
    }

    # -------------------------------------------------------------------------
    # CATEGORÍAS
    # -------------------------------------------------------------------------

    categorias = (
        gdf[
            "categoria_centralidad_estructural"
        ]
        .value_counts()
        .to_dict()
    )

    resumen[
        "centralidades_por_categoria"
    ] = {
        str(k): int(v)
        for k, v in categorias.items()
    }

    # -------------------------------------------------------------------------
    # DIAGNÓSTICO
    # -------------------------------------------------------------------------

    diagnosticos = (
        gdf[
            "diagnostico_demanda_infraestructura"
        ]
        .value_counts()
        .to_dict()
    )

    resumen[
        "centralidades_por_diagnostico"
    ] = {
        str(k): int(v)
        for k, v in diagnosticos.items()
    }

    # -------------------------------------------------------------------------
    # ÍNDICE
    # -------------------------------------------------------------------------

    indice = numerico(
        gdf[
            "indice_centralidad_estructural"
        ]
    )

    resumen[
        "indice_centralidad_estructural"
    ] = {
        "min": float(
            indice.min()
        ),
        "max": float(
            indice.max()
        ),
        "media": float(
            indice.mean()
        ),
        "mediana": float(
            indice.median()
        ),
    }

    # -------------------------------------------------------------------------
    # TOP 20
    # -------------------------------------------------------------------------

    columnas = [
        "nodo_id",
        "indice_centralidad_estructural",
        "ranking_centralidad_estructural",
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
        "diagnostico_demanda_infraestructura",
        "prioridad_intervencion",
    ]

    columnas = [
        c
        for c in columnas
        if c in gdf.columns
    ]

    top = (
        gdf[
            columnas
        ]
        .sort_values(
            "ranking_centralidad_estructural"
        )
        .head(20)
    )

    resumen[
        "top_20_centralidades"
    ] = [
        {
            str(k): valor_json(v)
            for k, v in registro.items()
        }
        for registro in top.to_dict(
            orient="records"
        )
    ]

    # -------------------------------------------------------------------------
    # TOP PRIORIDAD
    # -------------------------------------------------------------------------

    top_prioridad = (
        gdf[
            columnas
        ]
        .sort_values(
            "prioridad_intervencion",
            ascending=False,
        )
        .head(20)
    )

    resumen[
        "top_20_prioridad_intervencion"
    ] = [
        {
            str(k): valor_json(v)
            for k, v in registro.items()
        }
        for registro in top_prioridad.to_dict(
            orient="records"
        )
    ]

    return resumen


# =============================================================================
# GUARDADO
# =============================================================================

def guardar(
    gdf: gpd.GeoDataFrame,
    resumen: dict,
) -> None:

    titulo(
        "12. GUARDANDO ARCHIVOS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # PARQUET
    # -------------------------------------------------------------------------

    gdf.to_parquet(
        SALIDA_PARQUET,
        index=False,
    )

    print(
        f"Parquet:\n{SALIDA_PARQUET}"
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv = gdf.drop(
        columns=["geometry"],
        errors="ignore",
    )

    csv.to_csv(
        SALIDA_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"CSV:\n{SALIDA_CSV}"
    )

    # -------------------------------------------------------------------------
    # GEOPACKAGE
    # -------------------------------------------------------------------------

    if SALIDA_GPKG.exists():
        SALIDA_GPKG.unlink()

    gdf.to_file(
        SALIDA_GPKG,
        layer="centralidades_estructurales",
        driver="GPKG",
    )

    print(
        f"GeoPackage:\n{SALIDA_GPKG}"
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    with open(
        SALIDA_JSON,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"JSON:\n{SALIDA_JSON}"
    )


# =============================================================================
# MAPAS
# =============================================================================

def generar_mapa(
    gdf: gpd.GeoDataFrame,
    columna: str,
    titulo_mapa: str,
    archivo: str,
) -> None:

    mapa = gdf.to_crs(
        CRS_GEOGRAFICO
    )

    figura, ax = plt.subplots(
        figsize=(12, 12)
    )

    mapa.plot(
        ax=ax,
        column=columna,
        legend=True,
        markersize=50,
        alpha=0.85,
    )

    ax.set_title(
        titulo_mapa
    )

    ax.set_axis_off()

    figura.tight_layout()

    figura.savefig(
        OUTPUT_DIR / archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figura)


# =============================================================================
# GRÁFICOS
# =============================================================================

def generar_histograma(
    gdf: gpd.GeoDataFrame,
    columna: str,
    titulo_grafico: str,
    xlabel: str,
    archivo: str,
) -> None:

    figura, ax = plt.subplots(
        figsize=(11, 7)
    )

    valores = numerico(
        gdf[columna]
    )

    ax.hist(
        valores,
        bins=20,
    )

    ax.set_title(
        titulo_grafico
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Cantidad de centralidades"
    )

    figura.tight_layout()

    figura.savefig(
        OUTPUT_DIR / archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figura)


def generar_diagnostico(
    gdf: gpd.GeoDataFrame,
) -> None:

    figura, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.scatter(
        gdf[
            "indice_demanda_estructural"
        ],
        gdf[
            "indice_infraestructura_estructural"
        ],
        alpha=0.75,
    )

    ax.axvline(
        50,
        linestyle="--",
    )

    ax.axhline(
        50,
        linestyle="--",
    )

    ax.set_xlabel(
        "Índice de demanda"
    )

    ax.set_ylabel(
        "Índice de infraestructura"
    )

    ax.set_title(
        "Demanda vs infraestructura "
        "de las centralidades AMBA"
    )

    figura.tight_layout()

    figura.savefig(
        OUTPUT_DIR
        / "05_demanda_vs_infraestructura.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figura)


def generar_categorias(
    gdf: gpd.GeoDataFrame,
) -> None:

    figura, ax = plt.subplots(
        figsize=(11, 7)
    )

    valores = (
        gdf[
            "categoria_centralidad_estructural"
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    ax.bar(
        valores.index,
        valores.values,
    )

    ax.set_title(
        "Centralidades por categoría estructural"
    )

    ax.set_ylabel(
        "Cantidad de centralidades"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    figura.tight_layout()

    figura.savefig(
        OUTPUT_DIR
        / "06_centralidades_por_categoria.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figura)


def generar_visualizaciones(
    gdf: gpd.GeoDataFrame,
) -> None:

    titulo(
        "13. GENERANDO MAPAS Y GRÁFICOS"
    )

    print(
        "Mapa: 01_mapa_indice_centralidad_estructural.png"
    )

    generar_mapa(
        gdf,
        "indice_centralidad_estructural",
        (
            "AMBA - Índice de Centralidad "
            "Estructural"
        ),
        "01_mapa_indice_centralidad_estructural.png",
    )

    print(
        "Mapa: 02_mapa_indice_estructural_robusto.png"
    )

    generar_mapa(
        gdf,
        "indice_centralidad_estructural_robusto",
        (
            "AMBA - Índice de Centralidad "
            "Estructural Robusto"
        ),
        "02_mapa_indice_estructural_robusto.png",
    )

    print(
        "Mapa: 03_mapa_demanda.png"
    )

    generar_mapa(
        gdf,
        "indice_demanda_estructural",
        (
            "AMBA - Índice de Demanda "
            "Estructural"
        ),
        "03_mapa_demanda.png",
    )

    print(
        "Mapa: 04_mapa_deficit_infraestructura.png"
    )

    generar_mapa(
        gdf,
        "deficit_infraestructura",
        (
            "AMBA - Déficit relativo "
            "de infraestructura"
        ),
        "04_mapa_deficit_infraestructura.png",
    )

    print(
        "Gráfico: 05_demanda_vs_infraestructura.png"
    )

    generar_diagnostico(
        gdf
    )

    print(
        "Gráfico: 06_centralidades_por_categoria.png"
    )

    generar_categorias(
        gdf
    )

    print(
        "Gráfico: 07_distribucion_indice.png"
    )

    generar_histograma(
        gdf,
        "indice_centralidad_estructural",
        (
            "Distribución del Índice de "
            "Centralidad Estructural"
        ),
        "Índice 0-100",
        "07_distribucion_indice.png",
    )

    print(
        "Gráfico: 08_distribucion_prioridad.png"
    )

    generar_histograma(
        gdf,
        "prioridad_intervencion",
        (
            "Distribución de prioridad "
            "de intervención"
        ),
        "Prioridad 0-100",
        "08_distribucion_prioridad.png",
    )


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

def validar_resultado(
    gdf: gpd.GeoDataFrame,
) -> None:

    titulo(
        "14. VALIDACIÓN FINAL"
    )

    columnas_clave = [
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
        "indice_centralidad_estructural",
        "indice_centralidad_estructural_robusto",
        "prioridad_intervencion",
    ]

    for columna in columnas_clave:

        if columna not in gdf.columns:

            raise ValueError(
                f"Falta columna calculada: {columna}"
            )

        nulos = int(
            gdf[columna]
            .isna()
            .sum()
        )

        print(
            f"{columna}: "
            f"{nulos} nulos"
        )

        if nulos:

            raise ValueError(
                f"La columna {columna} "
                "contiene valores nulos."
            )

    # -------------------------------------------------------------------------
    # RANGOS
    # -------------------------------------------------------------------------

    indices = [
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
        "indice_centralidad_estructural",
        "indice_centralidad_estructural_robusto",
        "prioridad_intervencion",
    ]

    for columna in indices:

        minimo = float(
            gdf[columna].min()
        )

        maximo = float(
            gdf[columna].max()
        )

        print(
            f"{columna}: "
            f"{minimo:.2f} - {maximo:.2f}"
        )

        if minimo < 0 or maximo > 100:

            raise ValueError(
                f"{columna} fuera del rango 0-100."
            )

    print()
    print(
        "Validación final: OK"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    titulo(
        "23 - CONSTRUCCIÓN DEL ÍNDICE "
        f"DE CENTRALIDAD ESTRUCTURAL AMBA - {VERSION}"
    )

    print(
        f"Proyecto : {PROJECT_DIR}"
    )

    print(
        f"Entrada  : {INPUT_FILE}"
    )

    print(
        f"Salida   : {OUTPUT_DIR}"
    )

    print(
        f"CRS      : {CRS_GEOGRAFICO}"
    )

    print(
        f"CRS métrico: {CRS_METRICO}"
    )

    try:

        # =====================================================================
        # 1. CARGA
        # =====================================================================

        gdf = cargar_datos()

        # =====================================================================
        # 2. VALIDACIÓN
        # =====================================================================

        nodo_id = validar_entrada(
            gdf
        )

        # =====================================================================
        # 3. DEMANDA
        # =====================================================================

        gdf = calcular_demanda(
            gdf
        )

        # =====================================================================
        # 4. INFRAESTRUCTURA
        # =====================================================================

        gdf = calcular_infraestructura(
            gdf
        )

        # =====================================================================
        # 5. INTERMODALIDAD
        # =====================================================================

        gdf = calcular_intermodalidad(
            gdf
        )

        # =====================================================================
        # 6. CONECTIVIDAD
        # =====================================================================

        gdf = calcular_conectividad(
            gdf
        )

        # =====================================================================
        # 7. INTEGRACIÓN
        # =====================================================================

        gdf = calcular_integracion(
            gdf
        )

        # =====================================================================
        # 8. ÍNDICE
        # =====================================================================

        gdf = construir_indice(
            gdf
        )

        # =====================================================================
        # 9. CATEGORÍAS
        # =====================================================================

        gdf = clasificar_indice(
            gdf
        )

        # =====================================================================
        # 10. DIAGNÓSTICO
        # =====================================================================

        gdf = calcular_diagnostico(
            gdf
        )

        # =====================================================================
        # 11. PRIORIDAD
        # =====================================================================

        gdf = calcular_prioridad(
            gdf
        )

        # =====================================================================
        # 12. VALIDACIÓN
        # =====================================================================

        validar_resultado(
            gdf
        )

        # =====================================================================
        # TOP
        # =====================================================================

        imprimir_top20(
            gdf
        )

        imprimir_top_prioridad(
            gdf
        )

        # =====================================================================
        # RESUMEN
        # =====================================================================

        titulo(
            "15. CONSTRUYENDO RESUMEN JSON"
        )

        resumen = construir_resumen(
            gdf
        )

        # =====================================================================
        # GUARDADO
        # =====================================================================

        guardar(
            gdf,
            resumen,
        )

        # =====================================================================
        # VISUALIZACIONES
        # =====================================================================

        generar_visualizaciones(
            gdf
        )

        # =====================================================================
        # FINAL
        # =====================================================================

        titulo(
            "23 - PROCESO FINALIZADO"
        )

        print(
            f"Centralidades analizadas: "
            f"{len(gdf):,}"
        )

        print()
        print(
            "ARCHIVOS GENERADOS"
        )

        for archivo in sorted(
            OUTPUT_DIR.iterdir()
        ):

            if archivo.is_file():

                print(
                    f"  {archivo.name}"
                )

        print()
        print(
            "SIGUIENTE ETAPA"
        )

        print(
            "Construir la tipología de centralidades "
            "y clasificar los nodos según demanda, "
            "infraestructura, intermodalidad, "
            "conectividad y déficit estructural."
        )

        return 0

    except Exception as exc:

        titulo(
            "23 - ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "El proceso fue detenido para evitar "
            "generar resultados incompletos."
        )

        return 1


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )