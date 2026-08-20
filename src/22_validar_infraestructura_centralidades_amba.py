# -*- coding: utf-8 -*-
"""
22 - VALIDACIÓN DE INFRAESTRUCTURA CONTRA CENTRALIDADES AMBA

Objetivo
--------
Validar y caracterizar la infraestructura intermodal construida en el
script 21 contra las 144 centralidades de movilidad SUBE.

El script NO modifica los archivos generados por el proceso 21.

Entradas principales
--------------------
data/processed/infraestructura_intermodal_amba/
    instalaciones_transporte_amba.parquet
    intercambiadores_intermodales_amba.parquet
    centralidades_intermodalidad_amba.parquet

Salidas principales
-------------------
data/processed/validacion_infraestructura_centralidades_amba/
    validacion_infraestructura_centralidades_amba.gpkg
    validacion_infraestructura_centralidades_amba.parquet
    validacion_infraestructura_centralidades_amba.csv
    validacion_infraestructura_centralidades_amba_resumen.json

Además genera:
    01_mapa_validacion_centralidades.png
    02_mapa_infraestructura_500m.png
    03_mapa_intercambiadores_500m.png
    04_distribucion_infraestructura_500m.png
    05_distribucion_intercambiadores_500m.png
    06_diversidad_modal.png

CRS geográfico:
    EPSG:4326

CRS métrico AMBA:
    EPSG:22185
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
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SCRIPT_VERSION = "V1"

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

ARCHIVO_INSTALACIONES = INPUT_DIR / "instalaciones_transporte_amba.parquet"
ARCHIVO_INTERCAMBIADORES = (
    INPUT_DIR / "intercambiadores_intermodales_amba.parquet"
)
ARCHIVO_CENTRALIDADES = (
    INPUT_DIR / "centralidades_intermodalidad_amba.parquet"
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

def imprimir_titulo(texto: str) -> None:
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def imprimir_subtitulo(texto: str) -> None:
    print()
    print("-" * 78)
    print(texto)
    print("-" * 78)


def normalizar_nombre_columna(nombre: str) -> str:
    return (
        str(nombre)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalizar_columnas(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf.columns = [
        normalizar_nombre_columna(c)
        for c in gdf.columns
    ]
    return gdf


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = False,
) -> str | None:
    columnas = set(df.columns)

    for candidato in candidatos:
        if candidato in columnas:
            return candidato

    if obligatoria:
        raise ValueError(
            "No se encontró ninguna de las columnas requeridas: "
            + ", ".join(candidatos)
        )

    return None


def convertir_numerico(
    df: pd.DataFrame,
    columnas: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for columna in columnas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

    return df


def serie_numerica(
    df: pd.DataFrame,
    columna: str,
) -> pd.Series:
    if columna not in df.columns:
        return pd.Series(
            np.zeros(len(df)),
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[columna],
        errors="coerce",
    ).fillna(0)


def validar_geometrias(
    gdf: gpd.GeoDataFrame,
    nombre: str,
) -> None:
    imprimir_subtitulo(f"VALIDACIÓN GEOMÉTRICA - {nombre}")

    n = len(gdf)

    geometria_nula = int(gdf.geometry.isna().sum())
    geometria_vacia = int(gdf.geometry.is_empty.sum())
    geometria_invalida = int((~gdf.geometry.is_valid).sum())

    print(f"Registros: {n:,}")
    print(f"Geometrías nulas: {geometria_nula:,}")
    print(f"Geometrías vacías: {geometria_vacia:,}")
    print(f"Geometrías inválidas: {geometria_invalida:,}")

    if geometria_nula > 0:
        raise ValueError(
            f"{nombre}: existen geometrías nulas."
        )

    if geometria_vacia > 0:
        raise ValueError(
            f"{nombre}: existen geometrías vacías."
        )

    if geometria_invalida > 0:
        raise ValueError(
            f"{nombre}: existen geometrías inválidas."
        )


def percentil_normalizado(
    serie: pd.Series,
) -> pd.Series:
    """
    Normalización robusta por rango.

    0 = mínimo
    1 = máximo

    Si todos los valores son iguales:
        devuelve 0.0
    """

    s = pd.to_numeric(
        serie,
        errors="coerce",
    ).fillna(0.0)

    minimo = float(s.min())
    maximo = float(s.max())

    if math.isclose(minimo, maximo):
        return pd.Series(
            np.zeros(len(s)),
            index=s.index,
            dtype=float,
        )

    return (s - minimo) / (maximo - minimo)


def log_normalizado(
    serie: pd.Series,
) -> pd.Series:
    """
    Normalización logarítmica.

    Reduce el impacto de valores extremadamente altos,
    particularmente útil para densidades de instalaciones.
    """

    s = pd.to_numeric(
        serie,
        errors="coerce",
    ).fillna(0.0)

    transformada = np.log1p(s)

    minimo = float(transformada.min())
    maximo = float(transformada.max())

    if math.isclose(minimo, maximo):
        return pd.Series(
            np.zeros(len(s)),
            index=s.index,
            dtype=float,
        )

    return (transformada - minimo) / (maximo - minimo)


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if pd.isna(value):
        return None

    return value


# =============================================================================
# CARGA
# =============================================================================

def cargar_archivos() -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    imprimir_titulo("1. CARGANDO DATOS DEL PROCESO 21")

    archivos = [
        ARCHIVO_INSTALACIONES,
        ARCHIVO_INTERCAMBIADORES,
        ARCHIVO_CENTRALIDADES,
    ]

    for archivo in archivos:
        print(f"Archivo: {archivo}")

        if not archivo.exists():
            raise FileNotFoundError(
                f"No existe el archivo requerido:\n{archivo}"
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

    instalaciones = normalizar_columnas(instalaciones)
    intercambiadores = normalizar_columnas(intercambiadores)
    centralidades = normalizar_columnas(centralidades)

    if instalaciones.crs is None:
        instalaciones = instalaciones.set_crs(
            CRS_GEOGRAFICO,
            allow_override=True,
        )

    if intercambiadores.crs is None:
        intercambiadores = intercambiadores.set_crs(
            CRS_GEOGRAFICO,
            allow_override=True,
        )

    if centralidades.crs is None:
        centralidades = centralidades.set_crs(
            CRS_GEOGRAFICO,
            allow_override=True,
        )

    print()
    print(f"Instalaciones: {len(instalaciones):,}")
    print(f"Intercambiadores: {len(intercambiadores):,}")
    print(f"Centralidades: {len(centralidades):,}")

    print()
    print(f"CRS instalaciones: {instalaciones.crs}")
    print(f"CRS intercambiadores: {intercambiadores.crs}")
    print(f"CRS centralidades: {centralidades.crs}")

    return (
        instalaciones,
        intercambiadores,
        centralidades,
    )


# =============================================================================
# VALIDACIÓN ESTRUCTURAL
# =============================================================================

def validar_entradas(
    instalaciones: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
    centralidades: gpd.GeoDataFrame,
) -> None:
    imprimir_titulo("2. VALIDANDO DATOS DE ENTRADA")

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

    nodo_id = encontrar_columna(
        centralidades,
        [
            "nodo_id",
            "id_nodo",
            "node_id",
        ],
        obligatoria=True,
    )

    duplicados = int(
        centralidades[nodo_id].duplicated().sum()
    )

    print()
    print(f"Columna identificadora: {nodo_id}")
    print(f"nodo_id duplicados: {duplicados}")

    if duplicados > 0:
        raise ValueError(
            "Las centralidades contienen nodo_id duplicados."
        )

    if len(centralidades) != 144:
        print(
            "ADVERTENCIA: se esperaban 144 centralidades."
        )
        print(
            f"Cantidad encontrada: {len(centralidades)}"
        )


# =============================================================================
# NORMALIZACIÓN DE MODOS
# =============================================================================

def detectar_columna_modo(
    gdf: gpd.GeoDataFrame,
) -> str | None:
    return encontrar_columna(
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

    if isinstance(valor, float) and np.isnan(valor):
        return set()

    texto = str(valor).upper().strip()

    if not texto:
        return set()

    texto = (
        texto
        .replace(" ", "")
        .replace(",", "|")
        .replace(";", "|")
        .replace("/", "|")
    )

    partes = [
        p.strip()
        for p in texto.split("|")
        if p.strip()
    ]

    resultado: set[str] = set()

    for parte in partes:
        if "AUTOBUS" in parte or "BUS" in parte:
            resultado.add("AUTOBUS")

        if (
            "FERROCARRIL" in parte
            or "FERROVIARIO" in parte
            or "TREN" in parte
        ):
            resultado.add("FERROCARRIL")

        if "SUBTE" in parte or "METRO" in parte:
            resultado.add("SUBTE")

        if "FLUVIAL" in parte:
            resultado.add("FLUVIAL")

        if "TRANVIA" in parte or "TRAM" in parte:
            resultado.add("TRANVIA")

    return resultado


def preparar_modos(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    columna_modo = detectar_columna_modo(gdf)

    if columna_modo is None:
        gdf["_modos_normalizados"] = [
            set()
            for _ in range(len(gdf))
        ]
    else:
        gdf["_modos_normalizados"] = [
            extraer_modos(valor)
            for valor in gdf[columna_modo]
        ]

    return gdf


# =============================================================================
# CONTEO ESPACIAL
# =============================================================================

def contar_instalaciones_por_radio(
    centralidades: gpd.GeoDataFrame,
    instalaciones: gpd.GeoDataFrame,
    radio: int,
) -> np.ndarray:
    """
    Cuenta instalaciones dentro del radio indicado.

    Se utiliza sjoin para evitar construir una matriz
    completa centralidad x instalación.
    """

    centrales_m = centralidades.to_crs(CRS_METRICO)
    instalaciones_m = instalaciones.to_crs(CRS_METRICO)

    buffer = centrales_m[
        ["_indice_centralidad", "geometry"]
    ].copy()

    buffer["geometry"] = buffer.geometry.buffer(
        radio
    )

    joined = gpd.sjoin(
        instalaciones_m[
            ["geometry"]
        ],
        buffer,
        how="inner",
        predicate="within",
    )

    conteos = (
        joined.groupby(
            "_indice_centralidad"
        )
        .size()
    )

    resultado = np.zeros(
        len(centralidades),
        dtype=int,
    )

    for indice, cantidad in conteos.items():
        resultado[int(indice)] = int(cantidad)

    return resultado


def contar_intercambiadores_por_radio(
    centralidades: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
    radio: int,
) -> np.ndarray:
    centrales_m = centralidades.to_crs(CRS_METRICO)
    intercambiadores_m = intercambiadores.to_crs(
        CRS_METRICO
    )

    buffer = centrales_m[
        ["_indice_centralidad", "geometry"]
    ].copy()

    buffer["geometry"] = buffer.geometry.buffer(
        radio
    )

    joined = gpd.sjoin(
        intercambiadores_m[
            ["geometry"]
        ],
        buffer,
        how="inner",
        predicate="within",
    )

    conteos = (
        joined.groupby(
            "_indice_centralidad"
        )
        .size()
    )

    resultado = np.zeros(
        len(centralidades),
        dtype=int,
    )

    for indice, cantidad in conteos.items():
        resultado[int(indice)] = int(cantidad)

    return resultado


# =============================================================================
# MODOS POR RADIO
# =============================================================================

def modos_por_radio(
    centralidades: gpd.GeoDataFrame,
    instalaciones: gpd.GeoDataFrame,
    radio: int,
) -> list[set[str]]:
    centrales_m = centralidades.to_crs(CRS_METRICO)
    instalaciones_m = instalaciones.to_crs(
        CRS_METRICO
    )

    buffer = centrales_m[
        ["_indice_centralidad", "geometry"]
    ].copy()

    buffer["geometry"] = buffer.geometry.buffer(
        radio
    )

    instalaciones_tmp = instalaciones_m[
        [
            "geometry",
            "_modos_normalizados",
        ]
    ].copy()

    joined = gpd.sjoin(
        instalaciones_tmp,
        buffer,
        how="inner",
        predicate="within",
    )

    resultado = [
        set()
        for _ in range(len(centralidades))
    ]

    for fila in joined.itertuples():
        indice = int(
            getattr(
                fila,
                "_indice_centralidad",
            )
        )

        modos = getattr(
            fila,
            "_modos_normalizados",
        )

        if isinstance(modos, set):
            resultado[indice].update(modos)

    return resultado


def calcular_indicadores_modalidad(
    resultado: pd.DataFrame,
    centralidades: gpd.GeoDataFrame,
    instalaciones: gpd.GeoDataFrame,
) -> pd.DataFrame:
    imprimir_titulo(
        "3. CALCULANDO DIVERSIDAD MODAL"
    )

    for radio_nombre, radio in RADIOS.items():
        print(
            f"Calculando modos dentro de {radio_nombre}..."
        )

        conjuntos = modos_por_radio(
            centralidades,
            instalaciones,
            radio,
        )

        resultado[
            f"modos_{radio_nombre}"
        ] = [
            "|".join(sorted(modos))
            for modos in conjuntos
        ]

        resultado[
            f"cantidad_modos_{radio_nombre}"
        ] = [
            len(modos)
            for modos in conjuntos
        ]

        resultado[
            f"ferrocarril_{radio_nombre}"
        ] = [
            int("FERROCARRIL" in modos)
            for modos in conjuntos
        ]

        resultado[
            f"subte_{radio_nombre}"
        ] = [
            int("SUBTE" in modos)
            for modos in conjuntos
        ]

        resultado[
            f"autobus_{radio_nombre}"
        ] = [
            int("AUTOBUS" in modos)
            for modos in conjuntos
        ]

        resultado[
            f"fluvial_{radio_nombre}"
        ] = [
            int("FLUVIAL" in modos)
            for modos in conjuntos
        ]

        resultado[
            f"tranvia_{radio_nombre}"
        ] = [
            int("TRANVIA" in modos)
            for modos in conjuntos
        ]

    return resultado


# =============================================================================
# INTERCAMBIADORES
# =============================================================================

def calcular_intercambiadores(
    resultado: pd.DataFrame,
    centralidades: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
) -> pd.DataFrame:
    imprimir_titulo(
        "4. VALIDANDO INTERCAMBIADORES"
    )

    for radio_nombre, radio in RADIOS.items():
        print(
            f"Calculando intercambiadores dentro de "
            f"{radio_nombre}..."
        )

        conteos = contar_intercambiadores_por_radio(
            centralidades,
            intercambiadores,
            radio,
        )

        resultado[
            f"intercambiadores_{radio_nombre}"
        ] = conteos

    return resultado


# =============================================================================
# INSTALACIONES
# =============================================================================

def calcular_instalaciones(
    resultado: pd.DataFrame,
    centralidades: gpd.GeoDataFrame,
    instalaciones: gpd.GeoDataFrame,
) -> pd.DataFrame:
    imprimir_titulo(
        "5. VALIDANDO INSTALACIONES"
    )

    for radio_nombre, radio in RADIOS.items():
        print(
            f"Calculando instalaciones dentro de "
            f"{radio_nombre}..."
        )

        conteos = contar_instalaciones_por_radio(
            centralidades,
            instalaciones,
            radio,
        )

        resultado[
            f"instalaciones_{radio_nombre}"
        ] = conteos

    return resultado


# =============================================================================
# DENSIDAD Y ESTRUCTURA
# =============================================================================

def calcular_indicadores_estructurales(
    resultado: pd.DataFrame,
) -> pd.DataFrame:
    imprimir_titulo(
        "6. CALCULANDO INDICADORES ESTRUCTURALES"
    )

    # -------------------------------------------------------------------------
    # DENSIDAD
    # -------------------------------------------------------------------------

    # Área de un círculo:
    # A = pi * r²
    #
    # La densidad se expresa en instalaciones/km².

    for radio_nombre, radio in RADIOS.items():
        area_km2 = (
            math.pi
            * (radio / 1000.0) ** 2
        )

        columna = (
            f"instalaciones_{radio_nombre}"
        )

        resultado[
            f"densidad_instalaciones_{radio_nombre}"
        ] = (
            serie_numerica(
                resultado,
                columna,
            )
            / area_km2
        )

    # -------------------------------------------------------------------------
    # DENSIDAD LOGARÍTMICA
    # -------------------------------------------------------------------------

    resultado[
        "densidad_instalaciones_500m_normalizada"
    ] = log_normalizado(
        resultado[
            "densidad_instalaciones_500m"
        ]
    )

    resultado[
        "instalaciones_500m_normalizadas"
    ] = percentil_normalizado(
        resultado[
            "instalaciones_500m"
        ]
    )

    # -------------------------------------------------------------------------
    # DIVERSIDAD MODAL
    # -------------------------------------------------------------------------

    resultado[
        "diversidad_modal_500m_normalizada"
    ] = (
        serie_numerica(
            resultado,
            "cantidad_modos_500m",
        )
        / 5.0
    ).clip(
        lower=0,
        upper=1,
    )

    # -------------------------------------------------------------------------
    # PRESENCIA DE MODOS ESTRUCTURANTES
    # -------------------------------------------------------------------------

    resultado[
        "modo_ferroviario_estructurante"
    ] = (
        serie_numerica(
            resultado,
            "ferrocarril_500m",
        )
        > 0
    ).astype(int)

    resultado[
        "modo_subterraneo_estructurante"
    ] = (
        serie_numerica(
            resultado,
            "subte_500m",
        )
        > 0
    ).astype(int)

    resultado[
        "modo_fluvial_estructurante"
    ] = (
        serie_numerica(
            resultado,
            "fluvial_500m",
        )
        > 0
    ).astype(int)

    # -------------------------------------------------------------------------
    # INTERCAMBIADORES
    # -------------------------------------------------------------------------

    resultado[
        "intercambiadores_500m_normalizados"
    ] = log_normalizado(
        resultado[
            "intercambiadores_500m"
        ]
    )

    # -------------------------------------------------------------------------
    # CONECTIVIDAD MULTIMODAL
    # -------------------------------------------------------------------------

    resultado[
        "conectividad_modal_500m"
    ] = (
        serie_numerica(
            resultado,
            "cantidad_modos_500m",
        )
        + serie_numerica(
            resultado,
            "intercambiadores_500m",
        ).clip(upper=10)
        / 10.0
    )

    # -------------------------------------------------------------------------
    # INDICADOR PRELIMINAR DE SOPORTE FÍSICO
    #
    # NO ES EL ÍNDICE FINAL DE CENTRALIDAD.
    # -------------------------------------------------------------------------

    resultado[
        "soporte_fisico_preliminar"
    ] = (
        0.35
        * resultado[
            "densidad_instalaciones_500m_normalizada"
        ]
        + 0.25
        * resultado[
            "diversidad_modal_500m_normalizada"
        ]
        + 0.25
        * resultado[
            "intercambiadores_500m_normalizados"
        ]
        + 0.15
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
# INTEGRACIÓN CON RESULTADOS DEL 21
# =============================================================================

def integrar_resultados_21(
    resultado: pd.DataFrame,
    centralidades_original: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Conserva indicadores calculados en el proceso 21 cuando existen.

    No depende de nombres exactos más allá de nodo_id.
    """

    imprimir_titulo(
        "7. INTEGRANDO INDICADORES DEL PROCESO 21"
    )

    centralidades = centralidades_original.copy()

    nodo_id = encontrar_columna(
        centralidades,
        [
            "nodo_id",
            "id_nodo",
            "node_id",
        ],
        obligatoria=True,
    )

    resultado = resultado.copy()

    resultado["_nodo_id_merge"] = resultado[
        "nodo_id"
    ].astype(str)

    centralidades["_nodo_id_merge"] = centralidades[
        nodo_id
    ].astype(str)

    columnas_21 = [
        columna
        for columna in centralidades.columns
        if (
            columna != "geometry"
            and columna != "_nodo_id_merge"
            and (
                "intermodal" in columna.lower()
                or "score_" in columna.lower()
                or "ranking_" in columna.lower()
                or "categoria_" in columna.lower()
            )
        )
    ]

    if columnas_21:
        print(
            "Indicadores encontrados del proceso 21:"
        )

        for columna in columnas_21:
            print(f"  {columna}")

        subset = centralidades[
            [
                "_nodo_id_merge",
                *columnas_21,
            ]
        ].copy()

        resultado = resultado.merge(
            subset,
            on="_nodo_id_merge",
            how="left",
            suffixes=(
                "",
                "_21",
            ),
        )

    else:
        print(
            "No se encontraron columnas de indicadores "
            "del proceso 21."
        )

    resultado = resultado.drop(
        columns=["_nodo_id_merge"],
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
) -> dict[str, Any]:
    resumen: dict[str, Any] = {
        "script": "22_validar_infraestructura_centralidades_amba.py",
        "version": SCRIPT_VERSION,
        "crs_geografico": CRS_GEOGRAFICO,
        "crs_metrico": CRS_METRICO,
        "cantidad_instalaciones": int(
            len(instalaciones)
        ),
        "cantidad_intercambiadores": int(
            len(intercambiadores)
        ),
        "cantidad_centralidades": int(
            len(centralidades)
        ),
        "radios_metros": RADIOS,
    }

    # -------------------------------------------------------------------------
    # SOPORTE FÍSICO
    # -------------------------------------------------------------------------

    categoria = (
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
        for k, v in categoria.items()
    }

    # -------------------------------------------------------------------------
    # MODOS
    # -------------------------------------------------------------------------

    resumen[
        "centralidades_por_cantidad_de_modos_500m"
    ] = {
        str(int(k)): int(v)
        for k, v in resultado[
            "cantidad_modos_500m"
        ]
        .value_counts()
        .sort_index()
        .items()
    }

    # -------------------------------------------------------------------------
    # INSTALACIONES
    # -------------------------------------------------------------------------

    resumen[
        "instalaciones_500m"
    ] = {
        "min": int(
            resultado[
                "instalaciones_500m"
            ].min()
        ),
        "max": int(
            resultado[
                "instalaciones_500m"
            ].max()
        ),
        "media": float(
            resultado[
                "instalaciones_500m"
            ].mean()
        ),
        "mediana": float(
            resultado[
                "instalaciones_500m"
            ].median()
        ),
    }

    # -------------------------------------------------------------------------
    # INTERCAMBIADORES
    # -------------------------------------------------------------------------

    resumen[
        "intercambiadores_500m"
    ] = {
        "min": int(
            resultado[
                "intercambiadores_500m"
            ].min()
        ),
        "max": int(
            resultado[
                "intercambiadores_500m"
            ].max()
        ),
        "media": float(
            resultado[
                "intercambiadores_500m"
            ].mean()
        ),
        "mediana": float(
            resultado[
                "intercambiadores_500m"
            ].median()
        ),
    }

    # -------------------------------------------------------------------------
    # TOP 20
    # -------------------------------------------------------------------------

    top = (
        resultado[
            [
                "nodo_id",
                "soporte_fisico_preliminar",
                "instalaciones_250m",
                "instalaciones_500m",
                "instalaciones_1000m",
                "cantidad_modos_500m",
                "intercambiadores_500m",
                "modos_500m",
            ]
        ]
        .sort_values(
            "soporte_fisico_preliminar",
            ascending=False,
        )
        .head(20)
    )

    resumen["top_20_soporte_fisico"] = [
        {
            str(k): safe_json_value(v)
            for k, v in fila.items()
        }
        for fila in top.to_dict(
            orient="records"
        )
    ]

    return resumen


# =============================================================================
# GUARDADO
# =============================================================================

def guardar_resultados(
    resultado_gdf: gpd.GeoDataFrame,
    resumen: dict[str, Any],
) -> None:
    imprimir_titulo(
        "8. GUARDANDO RESULTADOS"
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

    resultado_csv = resultado_gdf.copy()

    if "geometry" in resultado_csv.columns:
        resultado_csv = resultado_csv.drop(
            columns=["geometry"]
        )

    resultado_csv.to_csv(
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

def generar_mapa_validacion(
    resultado_gdf: gpd.GeoDataFrame,
) -> None:
    print(
        "Mapa: 01_mapa_validacion_centralidades.png"
    )

    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    base = resultado_gdf.to_crs(
        CRS_GEOGRAFICO
    )

    base.plot(
        ax=ax,
        column="soporte_fisico_preliminar",
        legend=True,
        markersize=45,
        alpha=0.85,
    )

    ax.set_title(
        "AMBA - Validación de infraestructura física "
        "en centralidades SUBE"
    )

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_mapa_validacion_centralidades.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def generar_mapa_infraestructura(
    resultado_gdf: gpd.GeoDataFrame,
) -> None:
    print(
        "Mapa: 02_mapa_infraestructura_500m.png"
    )

    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    base = resultado_gdf.to_crs(
        CRS_GEOGRAFICO
    )

    base.plot(
        ax=ax,
        column="instalaciones_500m",
        legend=True,
        markersize=45,
        alpha=0.85,
    )

    ax.set_title(
        "AMBA - Instalaciones de transporte "
        "dentro de 500 m de centralidades"
    )

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_mapa_infraestructura_500m.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def generar_mapa_intercambiadores(
    resultado_gdf: gpd.GeoDataFrame,
) -> None:
    print(
        "Mapa: 03_mapa_intercambiadores_500m.png"
    )

    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    base = resultado_gdf.to_crs(
        CRS_GEOGRAFICO
    )

    base.plot(
        ax=ax,
        column="intercambiadores_500m",
        legend=True,
        markersize=45,
        alpha=0.85,
    )

    ax.set_title(
        "AMBA - Intercambiadores intermodales "
        "dentro de 500 m"
    )

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "03_mapa_intercambiadores_500m.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# GRÁFICOS
# =============================================================================

def generar_grafico_distribucion(
    resultado: pd.DataFrame,
    columna: str,
    titulo: str,
    nombre_archivo: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    serie = pd.to_numeric(
        resultado[columna],
        errors="coerce",
    ).fillna(0)

    ax.hist(
        serie,
        bins=20,
    )

    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cantidad de centralidades")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / nombre_archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def generar_grafico_modal(
    resultado: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
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
        "por centralidad - radio 500 m"
    )

    ax.set_xlabel(
        "Cantidad de modos"
    )

    ax.set_ylabel(
        "Cantidad de centralidades"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "06_diversidad_modal.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def generar_graficos(
    resultado: pd.DataFrame,
) -> None:
    imprimir_titulo(
        "9. GENERANDO MAPAS Y GRÁFICOS"
    )

    generar_mapa_validacion(
        resultado
    )

    generar_mapa_infraestructura(
        resultado
    )

    generar_mapa_intercambiadores(
        resultado
    )

    generar_grafico_distribucion(
        resultado,
        "instalaciones_500m",
        "Distribución de instalaciones de transporte "
        "dentro de 500 m",
        "04_distribucion_infraestructura_500m.png",
        "Instalaciones dentro de 500 m",
    )

    generar_grafico_distribucion(
        resultado,
        "intercambiadores_500m",
        "Distribución de intercambiadores "
        "dentro de 500 m",
        "05_distribucion_intercambiadores_500m.png",
        "Intercambiadores dentro de 500 m",
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
    imprimir_titulo(
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
    imprimir_titulo(
        f"22 - VALIDACIÓN DE INFRAESTRUCTURA "
        f"CONTRA CENTRALIDADES AMBA - {SCRIPT_VERSION}"
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
        # ---------------------------------------------------------------------
        # 1
        # ---------------------------------------------------------------------

        (
            instalaciones,
            intercambiadores,
            centralidades,
        ) = cargar_archivos()

        # ---------------------------------------------------------------------
        # 2
        # ---------------------------------------------------------------------

        validar_entradas(
            instalaciones,
            intercambiadores,
            centralidades,
        )

        # ---------------------------------------------------------------------
        # 3
        # ---------------------------------------------------------------------

        instalaciones = preparar_modos(
            instalaciones
        )

        # ---------------------------------------------------------------------
        # PREPARAR CENTRALIDADES
        # ---------------------------------------------------------------------

        nodo_id = encontrar_columna(
            centralidades,
            [
                "nodo_id",
                "id_nodo",
                "node_id",
            ],
            obligatoria=True,
        )

        centralidades = centralidades.copy()

        centralidades[
            "_indice_centralidad"
        ] = np.arange(
            len(centralidades)
        )

        # ---------------------------------------------------------------------
        # DATAFRAME BASE
        # ---------------------------------------------------------------------

        resultado = pd.DataFrame(
            {
                "nodo_id": centralidades[
                    nodo_id
                ].values,
            }
        )

        # ---------------------------------------------------------------------
        # IDENTIFICADORES EXTRA
        # ---------------------------------------------------------------------

        for columna in [
            "h3",
            "h3_index",
            "localidad",
            "municipio",
            "partido",
            "nombre",
            "nombre_nodo",
            "categoria",
        ]:
            if columna in centralidades.columns:
                resultado[columna] = (
                    centralidades[
                        columna
                    ].values
                )

        # ---------------------------------------------------------------------
        # 4
        # ---------------------------------------------------------------------

        resultado = calcular_instalaciones(
            resultado,
            centralidades,
            instalaciones,
        )

        # ---------------------------------------------------------------------
        # 5
        # ---------------------------------------------------------------------

        resultado = calcular_intercambiadores(
            resultado,
            centralidades,
            intercambiadores,
        )

        # ---------------------------------------------------------------------
        # 6
        # ---------------------------------------------------------------------

        resultado = calcular_indicadores_modalidad(
            resultado,
            centralidades,
            instalaciones,
        )

        # ---------------------------------------------------------------------
        # 7
        # ---------------------------------------------------------------------

        resultado = calcular_indicadores_estructurales(
            resultado
        )

        # ---------------------------------------------------------------------
        # 8
        # ---------------------------------------------------------------------

        resultado = clasificar_resultado(
            resultado
        )

        # ---------------------------------------------------------------------
        # 9
        # ---------------------------------------------------------------------

        resultado = integrar_resultados_21(
            resultado,
            centralidades,
        )

        # ---------------------------------------------------------------------
        # GEOMETRÍA
        # ---------------------------------------------------------------------

        resultado_gdf = gpd.GeoDataFrame(
            resultado,
            geometry=centralidades.geometry.values,
            crs=centralidades.crs,
        )

        # ---------------------------------------------------------------------
        # TOP
        # ---------------------------------------------------------------------

        imprimir_top20(
            resultado
        )

        # ---------------------------------------------------------------------
        # RESUMEN
        # ---------------------------------------------------------------------

        imprimir_titulo(
            "10. CONSTRUYENDO RESUMEN JSON"
        )

        resumen = construir_resumen(
            resultado,
            instalaciones,
            intercambiadores,
            centralidades,
        )

        # ---------------------------------------------------------------------
        # GUARDAR
        # ---------------------------------------------------------------------

        guardar_resultados(
            resultado_gdf,
            resumen,
        )

        # ---------------------------------------------------------------------
        # MAPAS / GRÁFICOS
        # ---------------------------------------------------------------------

        generar_graficos(
            resultado
        )

        # ---------------------------------------------------------------------
        # FINAL
        # ---------------------------------------------------------------------

        imprimir_titulo(
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
            "Construir el índice de centralidad estructural "
            "integrando demanda SUBE, infraestructura, "
            "intermodalidad, conectividad y jerarquía "
            "territorial."
        )

        return 0

    except Exception as exc:
        imprimir_titulo(
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