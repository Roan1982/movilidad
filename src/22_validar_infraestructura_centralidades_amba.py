# -*- coding: utf-8 -*-

"""
22 - VALIDACIÓN DE INFRAESTRUCTURA CONTRA CENTRALIDADES AMBA

Versión: V2

Objetivo
--------
Validar espacialmente las instalaciones de transporte y los
intercambiadores construidos en el proceso 21 contra las
144 centralidades de movilidad SUBE.

El proceso calcula:

- instalaciones a 250 m
- instalaciones a 500 m
- instalaciones a 1.000 m
- intercambiadores a 250 m
- intercambiadores a 500 m
- intercambiadores a 1.000 m
- diversidad modal
- presencia ferroviaria
- presencia de subte
- presencia de colectivo/autobús
- presencia fluvial
- presencia de tranvía
- densidad de infraestructura
- indicadores normalizados
- soporte físico preliminar
- ranking de soporte físico
- categoría de soporte físico

IMPORTANTE
----------
Este script NO modifica los archivos generados por el proceso 21.

Entradas
--------
data/processed/infraestructura_intermodal_amba/

    instalaciones_transporte_amba.parquet
    intercambiadores_intermodales_amba.parquet
    centralidades_intermodalidad_amba.parquet

Salida
------
data/processed/validacion_infraestructura_centralidades_amba/
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

VERSION = "V2"

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "infraestructura_intermodal_amba"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "validacion_infraestructura_centralidades_amba"
)

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

RADIOS = {
    "250m": 250,
    "500m": 500,
    "1000m": 1000,
}

ARCHIVO_INSTALACIONES = (
    INPUT_DIR
    / "instalaciones_transporte_amba.parquet"
)

ARCHIVO_INTERCAMBIADORES = (
    INPUT_DIR
    / "intercambiadores_intermodales_amba.parquet"
)

ARCHIVO_CENTRALIDADES = (
    INPUT_DIR
    / "centralidades_intermodalidad_amba.parquet"
)

SALIDA_GPKG = (
    OUTPUT_DIR
    / "validacion_infraestructura_centralidades_amba.gpkg"
)

SALIDA_PARQUET = (
    OUTPUT_DIR
    / "validacion_infraestructura_centralidades_amba.parquet"
)

SALIDA_CSV = (
    OUTPUT_DIR
    / "validacion_infraestructura_centralidades_amba.csv"
)

SALIDA_JSON = (
    OUTPUT_DIR
    / "validacion_infraestructura_centralidades_amba_resumen.json"
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


def normalizar_columnas(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    gdf = gdf.copy()

    nuevas = []

    for columna in gdf.columns:

        nombre = str(columna).strip().lower()

        nombre = (
            nombre
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        nuevas.append(nombre)

    gdf.columns = nuevas

    return gdf


def buscar_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = False,
) -> str | None:

    for candidato in candidatos:

        if candidato in df.columns:
            return candidato

    if obligatoria:

        raise ValueError(
            "No se encontró ninguna de las columnas requeridas: "
            + ", ".join(candidatos)
        )

    return None


def convertir_numerico(
    serie: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        serie,
        errors="coerce",
    ).fillna(0)


def normalizar_minmax(
    serie: pd.Series,
) -> pd.Series:

    valores = convertir_numerico(serie)

    minimo = valores.min()
    maximo = valores.max()

    if maximo == minimo:

        return pd.Series(
            0.0,
            index=serie.index,
        )

    return (
        (valores - minimo)
        / (maximo - minimo)
    )


def normalizar_log(
    serie: pd.Series,
) -> pd.Series:

    valores = convertir_numerico(serie)

    transformados = np.log1p(
        valores
    )

    minimo = transformados.min()
    maximo = transformados.max()

    if maximo == minimo:

        return pd.Series(
            0.0,
            index=serie.index,
        )

    return (
        (transformados - minimo)
        / (maximo - minimo)
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

def cargar_datos():

    titulo(
        "1. CARGANDO DATOS DEL PROCESO 21"
    )

    archivos = [
        ARCHIVO_INSTALACIONES,
        ARCHIVO_INTERCAMBIADORES,
        ARCHIVO_CENTRALIDADES,
    ]

    for archivo in archivos:

        print(
            f"Archivo: {archivo}"
        )

        if not archivo.exists():

            raise FileNotFoundError(
                f"No existe el archivo:\n{archivo}"
            )

    instalaciones = gpd.read_parquet(
        ARCHIVO_INSTALACIONES
    )

    intercambiadores = gpd.read_parquet(
        ARCHIVO_INTERCAMBIADORES
    )

    centralidades = gpd.read_parquet(
        ARCHIVO_CENTRALIDADES
    )

    instalaciones = normalizar_columnas(
        instalaciones
    )

    intercambiadores = normalizar_columnas(
        intercambiadores
    )

    centralidades = normalizar_columnas(
        centralidades
    )

    if instalaciones.crs is None:

        instalaciones = instalaciones.set_crs(
            CRS_GEOGRAFICO
        )

    if intercambiadores.crs is None:

        intercambiadores = intercambiadores.set_crs(
            CRS_GEOGRAFICO
        )

    if centralidades.crs is None:

        centralidades = centralidades.set_crs(
            CRS_GEOGRAFICO
        )

    print()
    print(
        f"Instalaciones: {len(instalaciones):,}"
    )

    print(
        f"Intercambiadores: {len(intercambiadores):,}"
    )

    print(
        f"Centralidades: {len(centralidades):,}"
    )

    print()
    print(
        f"CRS instalaciones: "
        f"{instalaciones.crs.to_string()}"
    )

    print(
        f"CRS intercambiadores: "
        f"{intercambiadores.crs.to_string()}"
    )

    print(
        f"CRS centralidades: "
        f"{centralidades.crs.to_string()}"
    )

    return (
        instalaciones,
        intercambiadores,
        centralidades,
    )


# =============================================================================
# VALIDACIÓN
# =============================================================================

def validar_geometrias(
    gdf: gpd.GeoDataFrame,
    nombre: str,
) -> None:

    subtitulo(
        f"VALIDACIÓN GEOMÉTRICA - {nombre}"
    )

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
        f"Registros: {len(gdf):,}"
    )

    print(
        f"Geometrías nulas: {nulas:,}"
    )

    print(
        f"Geometrías vacías: {vacias:,}"
    )

    print(
        f"Geometrías inválidas: {invalidas:,}"
    )

    if nulas:
        raise ValueError(
            f"{nombre}: geometrías nulas."
        )

    if vacias:
        raise ValueError(
            f"{nombre}: geometrías vacías."
        )

    if invalidas:
        raise ValueError(
            f"{nombre}: geometrías inválidas."
        )


def validar_entradas(
    instalaciones,
    intercambiadores,
    centralidades,
) -> None:

    titulo(
        "2. VALIDANDO DATOS DE ENTRADA"
    )

    validar_geometrias(
        instalaciones,
        "INSTALACIONES",
    )

    validar_geometrias(
        intercambiadores,
        "INTERCAMBIADORES",
    )

    validar_geometrias(
        centralidades,
        "CENTRALIDADES",
    )

    nodo_id = buscar_columna(
        centralidades,
        [
            "nodo_id",
            "id_nodo",
            "node_id",
        ],
        obligatoria=True,
    )

    duplicados = int(
        centralidades[nodo_id]
        .duplicated()
        .sum()
    )

    print()
    print(
        f"Columna identificadora: {nodo_id}"
    )

    print(
        f"nodo_id duplicados: {duplicados}"
    )

    if duplicados:

        raise ValueError(
            "Las centralidades contienen "
            "identificadores duplicados."
        )

    if len(centralidades) != 144:

        print()
        print(
            "ADVERTENCIA: se esperaban "
            "144 centralidades."
        )

        print(
            f"Cantidad encontrada: "
            f"{len(centralidades)}"
        )


# =============================================================================
# MODOS
# =============================================================================

def detectar_columna_modo(
    gdf: gpd.GeoDataFrame,
) -> str | None:

    return buscar_columna(
        gdf,
        [
            "modo",
            "modos",
            "modo_transporte",
            "modos_transporte",
            "modo_principal",
        ],
    )


def extraer_modos(
    valor: Any,
) -> set[str]:

    if valor is None:
        return set()

    if isinstance(valor, float):

        if np.isnan(valor):
            return set()

    texto = str(valor).upper().strip()

    if not texto:
        return set()

    texto = (
        texto
        .replace(",", "|")
        .replace(";", "|")
        .replace("/", "|")
        .replace(" ", "")
    )

    resultado = set()

    for parte in texto.split("|"):

        if not parte:
            continue

        if (
            "AUTOBUS" in parte
            or "BUS" in parte
        ):
            resultado.add(
                "AUTOBUS"
            )

        if (
            "FERROCARRIL" in parte
            or "FERROVIARIO" in parte
            or "TREN" in parte
        ):
            resultado.add(
                "FERROCARRIL"
            )

        if (
            "SUBTE" in parte
            or "METRO" in parte
        ):
            resultado.add(
                "SUBTE"
            )

        if "FLUVIAL" in parte:

            resultado.add(
                "FLUVIAL"
            )

        if (
            "TRANVIA" in parte
            or "TRAM" in parte
        ):
            resultado.add(
                "TRANVIA"
            )

    return resultado


def preparar_modos(
    instalaciones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    instalaciones = instalaciones.copy()

    columna = detectar_columna_modo(
        instalaciones
    )

    if columna is None:

        print(
            "ADVERTENCIA: no se encontró "
            "columna de modo."
        )

        instalaciones[
            "_modos_normalizados"
        ] = [
            set()
            for _ in range(
                len(instalaciones)
            )
        ]

    else:

        print(
            f"Columna de modo utilizada: "
            f"{columna}"
        )

        instalaciones[
            "_modos_normalizados"
        ] = [
            extraer_modos(valor)
            for valor in instalaciones[
                columna
            ]
        ]

    return instalaciones


# =============================================================================
# PREPARACIÓN ESPACIAL
# =============================================================================

def preparar_centralidades_metricas(
    centralidades: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    centrales = centralidades.to_crs(
        CRS_METRICO
    ).copy()

    centrales[
        "indice_centralidad_22"
    ] = np.arange(
        len(centrales),
        dtype=int,
    )

    return centrales


def preparar_instalaciones_metricas(
    instalaciones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    return instalaciones.to_crs(
        CRS_METRICO
    ).copy()


def preparar_intercambiadores_metricos(
    intercambiadores: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    return intercambiadores.to_crs(
        CRS_METRICO
    ).copy()


# =============================================================================
# CONTEOS ESPACIALES
# =============================================================================

def contar_elementos(
    centralidades_m: gpd.GeoDataFrame,
    elementos_m: gpd.GeoDataFrame,
    radio: int,
) -> np.ndarray:

    """
    Cuenta elementos dentro del buffer de cada centralidad.

    Se evita itertuples() y el acceso por atributos dinámicos.
    """

    buffers = centralidades_m[
        [
            "indice_centralidad_22",
            "geometry",
        ]
    ].copy()

    buffers["geometry"] = (
        buffers.geometry.buffer(
            radio
        )
    )

    elementos = elementos_m[
        ["geometry"]
    ].copy()

    joined = gpd.sjoin(
        elementos,
        buffers,
        how="inner",
        predicate="within",
    )

    resultado = np.zeros(
        len(centralidades_m),
        dtype=np.int64,
    )

    if joined.empty:
        return resultado

    conteos = (
        joined[
            "indice_centralidad_22"
        ]
        .value_counts()
    )

    for indice, cantidad in conteos.items():

        resultado[
            int(indice)
        ] = int(cantidad)

    return resultado


# =============================================================================
# MODOS POR RADIO
# =============================================================================

def calcular_modos_por_radio(
    centralidades_m: gpd.GeoDataFrame,
    instalaciones_m: gpd.GeoDataFrame,
    radio: int,
) -> list[set[str]]:

    buffers = centralidades_m[
        [
            "indice_centralidad_22",
            "geometry",
        ]
    ].copy()

    buffers["geometry"] = (
        buffers.geometry.buffer(
            radio
        )
    )

    elementos = instalaciones_m[
        [
            "geometry",
            "_modos_normalizados",
        ]
    ].copy()

    joined = gpd.sjoin(
        elementos,
        buffers,
        how="inner",
        predicate="within",
    )

    resultado = [
        set()
        for _ in range(
            len(centralidades_m)
        )
    ]

    if joined.empty:
        return resultado

    for indice, modos in zip(
        joined[
            "indice_centralidad_22"
        ],
        joined[
            "_modos_normalizados"
        ],
    ):

        indice = int(indice)

        if isinstance(modos, set):

            resultado[
                indice
            ].update(modos)

    return resultado


# =============================================================================
# INSTALACIONES
# =============================================================================

def calcular_instalaciones(
    resultado: pd.DataFrame,
    centralidades_m: gpd.GeoDataFrame,
    instalaciones_m: gpd.GeoDataFrame,
) -> pd.DataFrame:

    titulo(
        "3. VALIDANDO INSTALACIONES"
    )

    for nombre_radio, radio in RADIOS.items():

        print(
            f"Calculando instalaciones "
            f"dentro de {nombre_radio}..."
        )

        resultado[
            f"instalaciones_{nombre_radio}"
        ] = contar_elementos(
            centralidades_m,
            instalaciones_m,
            radio,
        )

    return resultado


# =============================================================================
# INTERCAMBIADORES
# =============================================================================

def calcular_intercambiadores(
    resultado: pd.DataFrame,
    centralidades_m: gpd.GeoDataFrame,
    intercambiadores_m: gpd.GeoDataFrame,
) -> pd.DataFrame:

    titulo(
        "4. VALIDANDO INTERCAMBIADORES"
    )

    for nombre_radio, radio in RADIOS.items():

        print(
            f"Calculando intercambiadores "
            f"dentro de {nombre_radio}..."
        )

        resultado[
            f"intercambiadores_{nombre_radio}"
        ] = contar_elementos(
            centralidades_m,
            intercambiadores_m,
            radio,
        )

    return resultado


# =============================================================================
# DIVERSIDAD MODAL
# =============================================================================

def calcular_modalidad(
    resultado: pd.DataFrame,
    centralidades_m: gpd.GeoDataFrame,
    instalaciones_m: gpd.GeoDataFrame,
) -> pd.DataFrame:

    titulo(
        "5. CALCULANDO DIVERSIDAD MODAL"
    )

    for nombre_radio, radio in RADIOS.items():

        print(
            f"Calculando modos dentro de "
            f"{nombre_radio}..."
        )

        conjuntos = calcular_modos_por_radio(
            centralidades_m,
            instalaciones_m,
            radio,
        )

        resultado[
            f"modos_{nombre_radio}"
        ] = [
            "|".join(
                sorted(modos)
            )
            for modos in conjuntos
        ]

        resultado[
            f"cantidad_modos_{nombre_radio}"
        ] = [
            len(modos)
            for modos in conjuntos
        ]

        for modo, nombre_columna in [
            (
                "FERROCARRIL",
                "ferrocarril",
            ),
            (
                "SUBTE",
                "subte",
            ),
            (
                "AUTOBUS",
                "autobus",
            ),
            (
                "FLUVIAL",
                "fluvial",
            ),
            (
                "TRANVIA",
                "tranvia",
            ),
        ]:

            resultado[
                f"{nombre_columna}_{nombre_radio}"
            ] = [
                int(
                    modo in modos
                )
                for modos in conjuntos
            ]

    return resultado


# =============================================================================
# INDICADORES ESTRUCTURALES
# =============================================================================

def calcular_indicadores(
    resultado: pd.DataFrame,
) -> pd.DataFrame:

    titulo(
        "6. CALCULANDO INDICADORES ESTRUCTURALES"
    )

    # -------------------------------------------------------------------------
    # DENSIDAD
    # -------------------------------------------------------------------------

    for nombre_radio, radio in RADIOS.items():

        area_km2 = (
            math.pi
            * (
                radio / 1000
            ) ** 2
        )

        columna = (
            f"instalaciones_{nombre_radio}"
        )

        resultado[
            f"densidad_instalaciones_{nombre_radio}"
        ] = (
            convertir_numerico(
                resultado[columna]
            )
            / area_km2
        )

    # -------------------------------------------------------------------------
    # NORMALIZACIONES
    # -------------------------------------------------------------------------

    resultado[
        "instalaciones_500m_normalizadas"
    ] = normalizar_minmax(
        resultado[
            "instalaciones_500m"
        ]
    )

    resultado[
        "densidad_instalaciones_500m_normalizada"
    ] = normalizar_log(
        resultado[
            "densidad_instalaciones_500m"
        ]
    )

    resultado[
        "intercambiadores_500m_normalizados"
    ] = normalizar_log(
        resultado[
            "intercambiadores_500m"
        ]
    )

    resultado[
        "diversidad_modal_500m_normalizada"
    ] = (
        convertir_numerico(
            resultado[
                "cantidad_modos_500m"
            ]
        )
        / 5.0
    ).clip(
        0,
        1,
    )

    # -------------------------------------------------------------------------
    # MODOS ESTRUCTURANTES
    # -------------------------------------------------------------------------

    resultado[
        "modo_ferroviario_estructurante"
    ] = (
        convertir_numerico(
            resultado[
                "ferrocarril_500m"
            ]
        ) > 0
    ).astype(int)

    resultado[
        "modo_subterraneo_estructurante"
    ] = (
        convertir_numerico(
            resultado[
                "subte_500m"
            ]
        ) > 0
    ).astype(int)

    resultado[
        "modo_fluvial_estructurante"
    ] = (
        convertir_numerico(
            resultado[
                "fluvial_500m"
            ]
        ) > 0
    ).astype(int)

    # -------------------------------------------------------------------------
    # CONECTIVIDAD MODAL
    # -------------------------------------------------------------------------

    resultado[
        "conectividad_modal_500m"
    ] = (
        resultado[
            "diversidad_modal_500m_normalizada"
        ]
        * 0.7
        +
        resultado[
            "intercambiadores_500m_normalizados"
        ]
        * 0.3
    )

    # -------------------------------------------------------------------------
    # SOPORTE FÍSICO PRELIMINAR
    #
    # Se utiliza densidad logarítmica para evitar que una
    # concentración extrema de objetos OSM domine el índice.
    # -------------------------------------------------------------------------

    resultado[
        "soporte_fisico_preliminar"
    ] = (
        0.35
        * resultado[
            "densidad_instalaciones_500m_normalizada"
        ]
        +
        0.25
        * resultado[
            "diversidad_modal_500m_normalizada"
        ]
        +
        0.25
        * resultado[
            "intercambiadores_500m_normalizados"
        ]
        +
        0.15
        * resultado[
            "instalaciones_500m_normalizadas"
        ]
    )

    resultado[
        "soporte_fisico_preliminar"
    ] = (
        resultado[
            "soporte_fisico_preliminar"
        ]
        * 100
    ).round(2)

    return resultado


# =============================================================================
# CLASIFICACIÓN
# =============================================================================

def clasificar_soporte(
    valor: float,
) -> str:

    if valor >= 80:
        return "SOPORTE_FISICO_MUY_ALTO"

    if valor >= 60:
        return "SOPORTE_FISICO_ALTO"

    if valor >= 40:
        return "SOPORTE_FISICO_MEDIO"

    if valor >= 20:
        return "SOPORTE_FISICO_BAJO"

    return "SOPORTE_FISICO_MUY_BAJO"


def clasificar_resultado(
    resultado: pd.DataFrame,
) -> pd.DataFrame:

    resultado[
        "categoria_soporte_fisico"
    ] = resultado[
        "soporte_fisico_preliminar"
    ].apply(
        clasificar_soporte
    )

    resultado[
        "ranking_soporte_fisico"
    ] = (
        resultado[
            "soporte_fisico_preliminar"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return resultado


# =============================================================================
# INTEGRACIÓN CON EL 21
# =============================================================================

def integrar_21(
    resultado: pd.DataFrame,
    centralidades: gpd.GeoDataFrame,
) -> pd.DataFrame:

    titulo(
        "7. INTEGRANDO RESULTADOS DEL PROCESO 21"
    )

    nodo_id = buscar_columna(
        centralidades,
        [
            "nodo_id",
            "id_nodo",
            "node_id",
        ],
        obligatoria=True,
    )

    resultado = resultado.copy()

    resultado[
        "_merge_nodo_id"
    ] = resultado[
        "nodo_id"
    ].astype(str)

    centrales = centralidades.copy()

    centrales[
        "_merge_nodo_id"
    ] = centrales[
        nodo_id
    ].astype(str)

    candidatos = []

    for columna in centrales.columns:

        if columna in [
            "geometry",
            "_merge_nodo_id",
            nodo_id,
        ]:
            continue

        nombre = columna.lower()

        if (
            "intermodal" in nombre
            or "score_" in nombre
            or "ranking_" in nombre
            or "categoria_intermodalidad" in nombre
        ):

            candidatos.append(
                columna
            )

    if candidatos:

        print(
            "Indicadores del proceso 21 encontrados:"
        )

        for columna in candidatos:
            print(
                f"  {columna}"
            )

        subset = centrales[
            [
                "_merge_nodo_id",
                *candidatos,
            ]
        ].copy()

        resultado = resultado.merge(
            subset,
            on="_merge_nodo_id",
            how="left",
            suffixes=(
                "",
                "_21",
            ),
        )

    else:

        print(
            "No se encontraron indicadores "
            "adicionales del proceso 21."
        )

    resultado = resultado.drop(
        columns=[
            "_merge_nodo_id"
        ],
        errors="ignore",
    )

    return resultado


# =============================================================================
# RESUMEN
# =============================================================================

def construir_resumen(
    resultado: pd.DataFrame,
    instalaciones: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
    centralidades: gpd.GeoDataFrame,
) -> dict:

    resumen = {
        "script": (
            "22_validar_infraestructura_"
            "centralidades_amba.py"
        ),
        "version": VERSION,
        "crs_geografico": CRS_GEOGRAFICO,
        "crs_metrico": CRS_METRICO,
        "instalaciones": int(
            len(instalaciones)
        ),
        "intercambiadores": int(
            len(intercambiadores)
        ),
        "centralidades": int(
            len(centralidades)
        ),
        "radios": RADIOS,
    }

    # -------------------------------------------------------------------------
    # SOPORTE FÍSICO
    # -------------------------------------------------------------------------

    categorias = (
        resultado[
            "categoria_soporte_fisico"
        ]
        .value_counts()
        .to_dict()
    )

    resumen[
        "centralidades_por_categoria_soporte_fisico"
    ] = {
        str(k): int(v)
        for k, v in categorias.items()
    }

    # -------------------------------------------------------------------------
    # MODOS
    # -------------------------------------------------------------------------

    cantidad_modos = (
        resultado[
            "cantidad_modos_500m"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    resumen[
        "centralidades_por_cantidad_modos_500m"
    ] = {
        str(int(k)): int(v)
        for k, v in cantidad_modos.items()
    }

    # -------------------------------------------------------------------------
    # INSTALACIONES
    # -------------------------------------------------------------------------

    instalaciones_500 = (
        convertir_numerico(
            resultado[
                "instalaciones_500m"
            ]
        )
    )

    resumen[
        "instalaciones_500m"
    ] = {
        "min": int(
            instalaciones_500.min()
        ),
        "max": int(
            instalaciones_500.max()
        ),
        "media": float(
            instalaciones_500.mean()
        ),
        "mediana": float(
            instalaciones_500.median()
        ),
    }

    # -------------------------------------------------------------------------
    # INTERCAMBIADORES
    # -------------------------------------------------------------------------

    intercambiadores_500 = (
        convertir_numerico(
            resultado[
                "intercambiadores_500m"
            ]
        )
    )

    resumen[
        "intercambiadores_500m"
    ] = {
        "min": int(
            intercambiadores_500.min()
        ),
        "max": int(
            intercambiadores_500.max()
        ),
        "media": float(
            intercambiadores_500.mean()
        ),
        "mediana": float(
            intercambiadores_500.median()
        ),
    }

    # -------------------------------------------------------------------------
    # TOP 20
    # -------------------------------------------------------------------------

    columnas_top = [
        "nodo_id",
        "soporte_fisico_preliminar",
        "ranking_soporte_fisico",
        "instalaciones_250m",
        "instalaciones_500m",
        "instalaciones_1000m",
        "cantidad_modos_500m",
        "modos_500m",
        "intercambiadores_500m",
        "categoria_soporte_fisico",
    ]

    columnas_top = [
        c
        for c in columnas_top
        if c in resultado.columns
    ]

    top = (
        resultado[
            columnas_top
        ]
        .sort_values(
            "ranking_soporte_fisico"
        )
        .head(20)
    )

    resumen[
        "top_20_soporte_fisico"
    ] = [
        {
            str(k): valor_json(v)
            for k, v in registro.items()
        }
        for registro in top.to_dict(
            orient="records"
        )
    ]

    return resumen


# =============================================================================
# GUARDADO
# =============================================================================

def guardar(
    resultado_gdf: gpd.GeoDataFrame,
    resumen: dict,
) -> None:

    titulo(
        "8. GUARDANDO ARCHIVOS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # PARQUET
    # -------------------------------------------------------------------------

    resultado_gdf.to_parquet(
        SALIDA_PARQUET,
        index=False,
    )

    print(
        f"Parquet:\n{SALIDA_PARQUET}"
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv = resultado_gdf.drop(
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

    resultado_gdf.to_file(
        SALIDA_GPKG,
        layer="centralidades_validacion",
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

    figura, ax = plt.subplots(
        figsize=(12, 12)
    )

    mapa = gdf.to_crs(
        CRS_GEOGRAFICO
    )

    mapa.plot(
        ax=ax,
        column=columna,
        legend=True,
        markersize=45,
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
    resultado: pd.DataFrame,
    columna: str,
    titulo_grafico: str,
    xlabel: str,
    archivo: str,
) -> None:

    figura, ax = plt.subplots(
        figsize=(11, 7)
    )

    valores = convertir_numerico(
        resultado[columna]
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


def generar_grafico_modal(
    resultado: pd.DataFrame,
) -> None:

    figura, ax = plt.subplots(
        figsize=(10, 7)
    )

    valores = (
        resultado[
            "cantidad_modos_500m"
        ]
        .value_counts()
        .sort_index()
    )

    ax.bar(
        valores.index.astype(str),
        valores.values,
    )

    ax.set_title(
        "Cantidad de modos de transporte "
        "por centralidad - 500 m"
    )

    ax.set_xlabel(
        "Cantidad de modos"
    )

    ax.set_ylabel(
        "Cantidad de centralidades"
    )

    figura.tight_layout()

    figura.savefig(
        OUTPUT_DIR
        / "06_diversidad_modal.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figura)


def generar_visualizaciones(
    resultado_gdf: gpd.GeoDataFrame,
    resultado: pd.DataFrame,
) -> None:

    titulo(
        "9. GENERANDO MAPAS Y GRÁFICOS"
    )

    print(
        "Mapa: 01_mapa_validacion_centralidades.png"
    )

    generar_mapa(
        resultado_gdf,
        "soporte_fisico_preliminar",
        (
            "AMBA - Soporte físico preliminar "
            "de las centralidades SUBE"
        ),
        "01_mapa_validacion_centralidades.png",
    )

    print(
        "Mapa: 02_mapa_infraestructura_500m.png"
    )

    generar_mapa(
        resultado_gdf,
        "instalaciones_500m",
        (
            "AMBA - Instalaciones de transporte "
            "dentro de 500 m"
        ),
        "02_mapa_infraestructura_500m.png",
    )

    print(
        "Mapa: 03_mapa_intercambiadores_500m.png"
    )

    generar_mapa(
        resultado_gdf,
        "intercambiadores_500m",
        (
            "AMBA - Intercambiadores intermodales "
            "dentro de 500 m"
        ),
        "03_mapa_intercambiadores_500m.png",
    )

    print(
        "Gráfico: 04_distribucion_infraestructura_500m.png"
    )

    generar_histograma(
        resultado,
        "instalaciones_500m",
        (
            "Distribución de instalaciones de "
            "transporte dentro de 500 m"
        ),
        "Instalaciones dentro de 500 m",
        "04_distribucion_infraestructura_500m.png",
    )

    print(
        "Gráfico: 05_distribucion_intercambiadores_500m.png"
    )

    generar_histograma(
        resultado,
        "intercambiadores_500m",
        (
            "Distribución de intercambiadores "
            "dentro de 500 m"
        ),
        "Intercambiadores dentro de 500 m",
        "05_distribucion_intercambiadores_500m.png",
    )

    print(
        "Gráfico: 06_diversidad_modal.png"
    )

    generar_grafico_modal(
        resultado
    )


# =============================================================================
# TOP 20
# =============================================================================

def imprimir_top20(
    resultado: pd.DataFrame,
) -> None:

    titulo(
        "TOP 20 CENTRALIDADES POR SOPORTE FÍSICO"
    )

    columnas = [
        "nodo_id",
        "soporte_fisico_preliminar",
        "ranking_soporte_fisico",
        "instalaciones_250m",
        "instalaciones_500m",
        "instalaciones_1000m",
        "cantidad_modos_500m",
        "modos_500m",
        "intercambiadores_500m",
        "categoria_soporte_fisico",
    ]

    columnas = [
        c
        for c in columnas
        if c in resultado.columns
    ]

    top = (
        resultado[
            columnas
        ]
        .sort_values(
            "ranking_soporte_fisico"
        )
        .head(20)
    )

    print(
        top.to_string(
            index=False
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    titulo(
        "22 - VALIDACIÓN DE INFRAESTRUCTURA "
        f"CONTRA CENTRALIDADES AMBA - {VERSION}"
    )

    print(
        f"Proyecto : {PROJECT_DIR}"
    )

    print(
        f"Entrada  : {INPUT_DIR}"
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

    print(
        "Radios   : 250 m / 500 m / 1.000 m"
    )

    try:

        # =====================================================================
        # 1. CARGA
        # =====================================================================

        (
            instalaciones,
            intercambiadores,
            centralidades,
        ) = cargar_datos()

        # =====================================================================
        # 2. VALIDACIÓN
        # =====================================================================

        validar_entradas(
            instalaciones,
            intercambiadores,
            centralidades,
        )

        # =====================================================================
        # 3. MODOS
        # =====================================================================

        instalaciones = preparar_modos(
            instalaciones
        )

        # =====================================================================
        # 4. CRS MÉTRICO
        # =====================================================================

        centrales_m = preparar_centralidades_metricas(
            centralidades
        )

        instalaciones_m = preparar_instalaciones_metricas(
            instalaciones
        )

        intercambiadores_m = (
            preparar_intercambiadores_metricos(
                intercambiadores
            )
        )

        # =====================================================================
        # 5. DATAFRAME BASE
        # =====================================================================

        nodo_id = buscar_columna(
            centralidades,
            [
                "nodo_id",
                "id_nodo",
                "node_id",
            ],
            obligatoria=True,
        )

        resultado = pd.DataFrame(
            {
                "nodo_id": centralidades[
                    nodo_id
                ].values,
            }
        )

        # =====================================================================
        # IDENTIFICADORES COMPLEMENTARIOS
        # =====================================================================

        columnas_extra = [
            "h3",
            "h3_index",
            "localidad",
            "municipio",
            "partido",
            "nombre",
            "nombre_nodo",
        ]

        for columna in columnas_extra:

            if columna in centralidades.columns:

                resultado[
                    columna
                ] = centralidades[
                    columna
                ].values

        # =====================================================================
        # 6. INSTALACIONES
        # =====================================================================

        resultado = calcular_instalaciones(
            resultado,
            centrales_m,
            instalaciones_m,
        )

        # =====================================================================
        # 7. INTERCAMBIADORES
        # =====================================================================

        resultado = calcular_intercambiadores(
            resultado,
            centrales_m,
            intercambiadores_m,
        )

        # =====================================================================
        # 8. MODALIDAD
        # =====================================================================

        resultado = calcular_modalidad(
            resultado,
            centrales_m,
            instalaciones_m,
        )

        # =====================================================================
        # 9. INDICADORES
        # =====================================================================

        resultado = calcular_indicadores(
            resultado
        )

        # =====================================================================
        # 10. CLASIFICACIÓN
        # =====================================================================

        resultado = clasificar_resultado(
            resultado
        )

        # =====================================================================
        # 11. INTEGRACIÓN CON EL 21
        # =====================================================================

        resultado = integrar_21(
            resultado,
            centralidades,
        )

        # =====================================================================
        # GEOMETRÍA
        # =====================================================================

        resultado_gdf = gpd.GeoDataFrame(
            resultado,
            geometry=centralidades.geometry.values,
            crs=centralidades.crs,
        )

        # =====================================================================
        # TOP
        # =====================================================================

        imprimir_top20(
            resultado
        )

        # =====================================================================
        # RESUMEN
        # =====================================================================

        titulo(
            "10. CONSTRUYENDO RESUMEN JSON"
        )

        resumen = construir_resumen(
            resultado,
            instalaciones,
            intercambiadores,
            centralidades,
        )

        # =====================================================================
        # GUARDADO
        # =====================================================================

        guardar(
            resultado_gdf,
            resumen,
        )

        # =====================================================================
        # VISUALIZACIONES
        # =====================================================================

        generar_visualizaciones(
            resultado_gdf,
            resultado,
        )

        # =====================================================================
        # FINAL
        # =====================================================================

        titulo(
            "22 - PROCESO FINALIZADO"
        )

        print(
            f"Instalaciones analizadas: "
            f"{len(instalaciones):,}"
        )

        print(
            f"Intercambiadores analizados: "
            f"{len(intercambiadores):,}"
        )

        print(
            f"Centralidades analizadas: "
            f"{len(centralidades):,}"
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
            "Construir el índice de centralidad "
            "estructural integrando demanda SUBE, "
            "infraestructura, intermodalidad, "
            "conectividad y jerarquía territorial."
        )

        return 0

    except Exception as exc:

        titulo(
            "22 - ERROR"
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


if __name__ == "__main__":
    sys.exit(
        main()
    )