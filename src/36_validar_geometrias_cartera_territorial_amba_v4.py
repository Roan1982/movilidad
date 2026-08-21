# -*- coding: utf-8 -*-
"""
36_validar_geometrias_cartera_territorial_amba_v4.py

PROCESO 36
Validación geoespacial y control de geometrías
de la cartera territorial AMBA V4.

Entrada principal:
    data/processed/escenarios_territoriales_amba/cartera_proyectos_v4.csv

Fuente canónica para:
    - escenario_id
    - geometría
    - indicadores originales

    data/processed/escenarios_territoriales_amba/
        escenarios_territoriales_amba_v4.parquet

Salidas:
    validacion_geoespacial_cartera_v4.csv
    auditoria_36_geometrias_cartera_territorial_amba.csv
    geometria_cartera_proyectos_v4.gpkg
    geometria_escenarios_cartera_v4.gpkg
    resumen_36_geometrias_cartera_territorial_amba.json
    sintesis_geoespacial_cartera_v4.md

IMPORTANTE:
    Este archivo está diseñado para ejecutarse directamente desde /src.
    NO genera ni reescribe su propio código.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SCRIPT_VERSION = "V4.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_CARTERA = DATA_DIR / "cartera_proyectos_v4.csv"

INPUT_CANONICO = (
    DATA_DIR
    / "escenarios_territoriales_amba_v4.parquet"
)

OUTPUT_VALIDACION = (
    DATA_DIR
    / "validacion_geoespacial_cartera_v4.csv"
)

OUTPUT_AUDITORIA = (
    DATA_DIR
    / "auditoria_36_geometrias_cartera_territorial_amba.csv"
)

OUTPUT_GPKG_PROYECTOS = (
    DATA_DIR
    / "geometria_cartera_proyectos_v4.gpkg"
)

OUTPUT_GPKG_ESCENARIOS = (
    DATA_DIR
    / "geometria_escenarios_cartera_v4.gpkg"
)

OUTPUT_RESUMEN = (
    DATA_DIR
    / "resumen_36_geometrias_cartera_territorial_amba.json"
)

OUTPUT_MARKDOWN = (
    DATA_DIR
    / "sintesis_geoespacial_cartera_v4.md"
)

DEFAULT_CRS = "EPSG:4326"

# CRS métrico habitual para AMBA.
# Se utiliza únicamente para cálculos geométricos.
METRIC_CRS = "EPSG:22185"


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def subtitulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def normalizar_texto(value) -> Optional[str]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    texto = str(value).strip()

    if not texto:
        return None

    return texto


def resolver_columna(
    df: pd.DataFrame,
    candidatos: Iterable[str],
    obligatorio: bool = True,
) -> Optional[str]:
    """
    Busca una columna usando coincidencia exacta primero y
    luego coincidencia case-insensitive.
    """

    columnas = list(df.columns)

    # Exacta
    for candidato in candidatos:
        if candidato in columnas:
            return candidato

    # Case insensitive
    mapa = {
        str(col).strip().lower(): col
        for col in columnas
    }

    for candidato in candidatos:
        key = str(candidato).strip().lower()
        if key in mapa:
            return mapa[key]

    if obligatorio:
        raise KeyError(
            "No se encontró ninguna de las columnas esperadas: "
            f"{list(candidatos)}"
        )

    return None


def eliminar_columnas_duplicadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserva la primera aparición de cada nombre de columna.
    """
    if not df.columns.duplicated().any():
        return df

    return df.loc[:, ~df.columns.duplicated()].copy()


def asegurar_directorio_salida() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def geometry_is_valid_object(value) -> bool:
    return isinstance(value, BaseGeometry)


def limpiar_geometria(value):
    """
    Intenta recuperar una geometría válida.

    No modifica geometrías válidas.
    Para geometrías inválidas intenta make_valid / buffer(0).
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if not geometry_is_valid_object(value):
        return None

    try:
        if value.is_empty:
            return value

        if value.is_valid:
            return value

        # Shapely 2.x
        try:
            from shapely.validation import make_valid

            repaired = make_valid(value)

            if repaired is not None and not repaired.is_empty:
                return repaired
        except Exception:
            pass

        # Fallback clásico
        try:
            repaired = value.buffer(0)

            if repaired is not None and not repaired.is_empty:
                return repaired
        except Exception:
            pass

    except Exception:
        return None

    return value


def serializar_geometry_wkt(value) -> Optional[str]:
    if not geometry_is_valid_object(value):
        return None

    try:
        return value.wkt
    except Exception:
        return None


def porcentaje(n, total):
    if total == 0:
        return 0.0
    return float(n) * 100.0 / float(total)


def coeficiente_variacion(series: pd.Series) -> float:
    valores = pd.to_numeric(series, errors="coerce").dropna()

    if len(valores) == 0:
        return 0.0

    media = valores.mean()

    if media == 0:
        return 0.0

    return float(valores.std(ddof=0) / media)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_cartera() -> pd.DataFrame:
    if not INPUT_CARTERA.exists():
        raise FileNotFoundError(
            f"No existe la cartera de proyectos:\n{INPUT_CARTERA}"
        )

    subtitulo("CARGANDO CARTERA DE PROYECTOS DEL PROCESO 35")

    print(f"Cargando: {INPUT_CARTERA}")

    df = pd.read_csv(
        INPUT_CARTERA,
        low_memory=False,
    )

    df = eliminar_columnas_duplicadas(df)

    print(f"Registros : {len(df)}")
    print(f"Columnas : {len(df.columns)}")

    return df


def cargar_fuente_canonica() -> gpd.GeoDataFrame:
    if not INPUT_CANONICO.exists():
        raise FileNotFoundError(
            f"No existe la fuente canónica:\n{INPUT_CANONICO}"
        )

    subtitulo("CARGANDO FUENTE CANÓNICA V4")

    print(f"Cargando: {INPUT_CANONICO}")

    gdf = gpd.read_parquet(INPUT_CANONICO)

    gdf = eliminar_columnas_duplicadas(gdf)

    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_CRS)

    print(f"Registros : {len(gdf)}")
    print(f"Columnas : {len(gdf.columns)}")
    print(f"CRS       : {gdf.crs}")

    return gdf


# =============================================================================
# RESOLUCIÓN DE CAMPOS
# =============================================================================

def resolver_campos_cartera(
    cartera: pd.DataFrame,
) -> dict:

    subtitulo("RESOLUCIÓN DE CAMPOS")

    campos = {}

    campos["proyecto"] = resolver_columna(
        cartera,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    campos["escenario"] = resolver_columna(
        cartera,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
        obligatorio=False,
    )

    campos["geometria"] = resolver_columna(
        cartera,
        [
            "geometry",
            "geom",
            "geometria",
        ],
        obligatorio=False,
    )

    print(
        f"proyecto                    : "
        f"{campos['proyecto']}"
    )

    print(
        f"escenario                   : "
        f"{campos['escenario'] or 'NO DISPONIBLE'}"
    )

    print(
        f"geometria                   : "
        f"{campos['geometria'] or 'NO DISPONIBLE'}"
    )

    return campos


def resolver_campos_canonicos(
    gdf: gpd.GeoDataFrame,
) -> dict:

    campos = {}

    campos["proyecto"] = resolver_columna(
        gdf,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    campos["escenario"] = resolver_columna(
        gdf,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
        obligatorio=False,
    )

    campos["geometria"] = (
        gdf.geometry.name
        if gdf.geometry is not None
        else resolver_columna(
            gdf,
            ["geometry", "geom", "geometria"],
            obligatorio=False,
        )
    )

    return campos


# =============================================================================
# RECUPERACIÓN DE ESCENARIO
# =============================================================================

def recuperar_escenarios(
    cartera: pd.DataFrame,
    fuente: gpd.GeoDataFrame,
    campo_proyecto_cartera: str,
    campo_escenario_cartera: Optional[str],
) -> tuple[pd.DataFrame, dict]:

    subtitulo(
        "RESOLVIENDO ASIGNACIÓN PROYECTO -> ESCENARIO"
    )

    resultado = cartera.copy()

    # -------------------------------------------------------------------------
    # Si ya existe escenario_id en cartera, se conserva.
    # -------------------------------------------------------------------------

    if campo_escenario_cartera is not None:

        resultado["escenario_id"] = (
            resultado[campo_escenario_cartera]
            .apply(normalizar_texto)
        )

        disponibles = (
            resultado["escenario_id"]
            .notna()
            .sum()
        )

        print(
            f"Asignaciones existentes : {disponibles}"
        )

    else:

        print(
            "No existe escenario_id en cartera_proyectos_v4.csv."
        )

        print(
            "Buscando asignación en "
            "escenarios_territoriales_amba_v4.parquet..."
        )

        campos_fuente = resolver_campos_canonicos(
            fuente
        )

        if campos_fuente["escenario"] is None:
            raise KeyError(
                "La fuente canónica tampoco contiene "
                "escenario_id."
            )

        mapa = (
            fuente[
                [
                    campos_fuente["proyecto"],
                    campos_fuente["escenario"],
                ]
            ]
            .dropna(subset=[campos_fuente["proyecto"]])
            .drop_duplicates(
                subset=[campos_fuente["proyecto"]]
            )
            .copy()
        )

        mapa["_proyecto_key"] = (
            mapa[campos_fuente["proyecto"]]
            .astype(str)
            .str.strip()
        )

        mapa["_escenario_value"] = (
            mapa[campos_fuente["escenario"]]
            .apply(normalizar_texto)
        )

        lookup = dict(
            zip(
                mapa["_proyecto_key"],
                mapa["_escenario_value"],
            )
        )

        resultado["_proyecto_key"] = (
            resultado[campo_proyecto_cartera]
            .astype(str)
            .str.strip()
        )

        resultado["escenario_id"] = (
            resultado["_proyecto_key"]
            .map(lookup)
        )

        resultado.drop(
            columns=["_proyecto_key"],
            inplace=True,
        )

        recuperadas = (
            resultado["escenario_id"]
            .notna()
            .sum()
        )

        faltantes = (
            resultado["escenario_id"]
            .isna()
            .sum()
        )

        print(
            f"Asignaciones recuperadas : {recuperadas}"
        )

        print(
            f"Asignaciones faltantes   : {faltantes}"
        )

    resultado["escenario_id"] = (
        resultado["escenario_id"]
        .apply(normalizar_texto)
    )

    return resultado, {
        "campo_escenario_original": campo_escenario_cartera,
        "asignaciones_faltantes": int(
            resultado["escenario_id"].isna().sum()
        ),
    }


# =============================================================================
# RECUPERACIÓN DE GEOMETRÍAS
# =============================================================================

def recuperar_geometrias(
    cartera: pd.DataFrame,
    fuente: gpd.GeoDataFrame,
    campo_proyecto_cartera: str,
    campo_geometria_cartera: Optional[str],
) -> gpd.GeoDataFrame:

    subtitulo(
        "RECUPERANDO GEOMETRÍAS CANÓNICAS"
    )

    resultado = cartera.copy()

    # -------------------------------------------------------------------------
    # Primero intentamos utilizar geometría ya existente en cartera.
    # -------------------------------------------------------------------------

    geometria_existente = None

    if campo_geometria_cartera is not None:

        try:
            serie = resultado[campo_geometria_cartera]

            objetos = serie.apply(
                geometry_is_valid_object
            )

            if objetos.any():
                geometria_existente = serie
        except Exception:
            geometria_existente = None

    # -------------------------------------------------------------------------
    # Fuente canónica
    # -------------------------------------------------------------------------

    campos_fuente = resolver_campos_canonicos(
        fuente
    )

    campo_proyecto_fuente = (
        campos_fuente["proyecto"]
    )

    geometria_fuente = fuente.geometry

    # -------------------------------------------------------------------------
    # Construimos lookup proyecto -> geometría.
    # -------------------------------------------------------------------------

    fuente_tmp = pd.DataFrame(
        {
            "_proyecto_key": (
                fuente[campo_proyecto_fuente]
                .astype(str)
                .str.strip()
            ),
            "_geometry": geometria_fuente,
        }
    )

    fuente_tmp = (
        fuente_tmp
        .dropna(subset=["_proyecto_key"])
        .drop_duplicates(
            subset=["_proyecto_key"]
        )
    )

    geometry_lookup = dict(
        zip(
            fuente_tmp["_proyecto_key"],
            fuente_tmp["_geometry"],
        )
    )

    keys = (
        resultado[campo_proyecto_cartera]
        .astype(str)
        .str.strip()
    )

    geometria_recuperada = keys.map(
        geometry_lookup
    )

    # -------------------------------------------------------------------------
    # Si la geometría de cartera es válida, tiene prioridad.
    # En caso contrario se utiliza la canónica.
    # -------------------------------------------------------------------------

    geometria_final = []

    for i in range(len(resultado)):

        geom = None

        if geometria_existente is not None:
            candidato = geometria_existente.iloc[i]

            if geometry_is_valid_object(candidato):
                geom = candidato

        if geom is None:
            candidato = geometria_recuperada.iloc[i]

            if geometry_is_valid_object(candidato):
                geom = candidato

        geometria_final.append(
            limpiar_geometria(geom)
        )

    resultado["geometry"] = geometria_final

    gdf = gpd.GeoDataFrame(
        resultado,
        geometry="geometry",
        crs=fuente.crs or DEFAULT_CRS,
    )

    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_CRS)

    print(
        f"CRS       : {gdf.crs}"
    )

    validas = gdf.geometry.apply(
        geometry_is_valid_object
    ).sum()

    print(
        f"Geometrías recuperadas : {validas}"
    )

    print(
        f"Geometrías faltantes   : "
        f"{len(gdf) - validas}"
    )

    return gdf


# =============================================================================
# VALIDACIÓN BASE
# =============================================================================

def validar_base(
    gdf: gpd.GeoDataFrame,
) -> dict:

    subtitulo("VALIDACIÓN BASE DE ENTRADA")

    proyecto_nulos = int(
        gdf["proyecto_id"]
        .isna()
        .sum()
    )

    proyecto_duplicados = int(
        gdf["proyecto_id"]
        .duplicated()
        .sum()
    )

    escenario_nulos = int(
        gdf["escenario_id"]
        .isna()
        .sum()
    )

    escenarios = (
        gdf["escenario_id"]
        .dropna()
        .nunique()
    )

    print(
        f"Registros              : {len(gdf)}"
    )

    print(
        f"Proyectos únicos       : "
        f"{gdf['proyecto_id'].nunique(dropna=True)}"
    )

    print(
        f"Proyectos nulos        : "
        f"{proyecto_nulos}"
    )

    print(
        f"Proyectos duplicados   : "
        f"{proyecto_duplicados}"
    )

    print(
        f"Escenarios             : "
        f"{escenarios}"
    )

    print(
        f"Escenarios nulos       : "
        f"{escenario_nulos}"
    )

    return {
        "registros": int(len(gdf)),
        "proyectos_unicos": int(
            gdf["proyecto_id"].nunique(
                dropna=True
            )
        ),
        "proyectos_nulos": proyecto_nulos,
        "proyectos_duplicados": proyecto_duplicados,
        "escenarios": int(escenarios),
        "escenarios_nulos": escenario_nulos,
    }


# =============================================================================
# VALIDACIÓN GEOMÉTRICA
# =============================================================================

def validar_geometrias(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:

    subtitulo(
        "VALIDACIÓN DE GEOMETRÍAS"
    )

    n = len(gdf)

    nulas = int(
        gdf.geometry.isna().sum()
    )

    vacias = int(
        gdf.geometry
        .apply(
            lambda g:
                geometry_is_valid_object(g)
                and g.is_empty
        )
        .sum()
    )

    validas_objeto = (
        gdf.geometry
        .apply(geometry_is_valid_object)
    )

    invalidas = int(
        gdf.geometry
        .apply(
            lambda g:
                geometry_is_valid_object(g)
                and not g.is_valid
        )
        .sum()
    )

    tipos = (
        gdf.geometry
        .apply(
            lambda g:
                g.geom_type
                if geometry_is_valid_object(g)
                else None
        )
    )

    tipo_principal = (
        tipos.dropna()
        .value_counts()
        .index[0]
        if tipos.notna().any()
        else None
    )

    geometria_valida = (
        validas_objeto
        & (~gdf.geometry.isna())
        & (
            ~gdf.geometry
            .apply(
                lambda g:
                    geometry_is_valid_object(g)
                    and g.is_empty
            )
        )
        & (
            gdf.geometry
            .apply(
                lambda g:
                    geometry_is_valid_object(g)
                    and g.is_valid
            )
        )
    )

    validas = int(
        geometria_valida.sum()
    )

    cobertura = porcentaje(
        validas,
        n,
    )

    print(
        f"Registros               : {n}"
    )

    print(
        f"Geometrías válidas      : {validas}"
    )

    print(
        f"Geometrías nulas        : {nulas}"
    )

    print(
        f"Geometrías vacías       : {vacias}"
    )

    print(
        f"Geometrías inválidas    : {invalidas}"
    )

    print(
        f"Cobertura geométrica    : "
        f"{cobertura:.2f}%"
    )

    print(
        f"Tipo geométrico         : "
        f"{tipo_principal or 'N/D'}"
    )

    return gdf, {
        "geometrias_validas": validas,
        "geometrias_nulas": nulas,
        "geometrias_vacias": vacias,
        "geometrias_invalidas": invalidas,
        "cobertura_geometrica_pct": cobertura,
        "tipo_geometrico_principal": tipo_principal,
    }


# =============================================================================
# VALIDACIÓN TERRITORIAL
# =============================================================================

def validar_consistencia_territorial(
    gdf: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict]:

    subtitulo(
        "VALIDACIÓN DE CONSISTENCIA TERRITORIAL"
    )

    conteos = (
        gdf.groupby(
            "escenario_id",
            dropna=False,
        )
        .size()
        .reset_index(
            name="cantidad_proyectos"
        )
    )

    conteos = conteos.sort_values(
        "escenario_id",
        na_position="last",
    )

    if len(conteos):

        minimo = int(
            conteos["cantidad_proyectos"].min()
        )

        maximo = int(
            conteos["cantidad_proyectos"].max()
        )

        promedio = float(
            conteos["cantidad_proyectos"].mean()
        )

        cv = coeficiente_variacion(
            conteos["cantidad_proyectos"]
        )

    else:

        minimo = 0
        maximo = 0
        promedio = 0.0
        cv = 0.0

    duplicados_multiescenario = (
        gdf.groupby(
            "proyecto_id"
        )["escenario_id"]
        .nunique()
    )

    multiescenario = int(
        (duplicados_multiescenario > 1)
        .sum()
    )

    print(
        f"Escenarios detectados              : "
        f"{len(conteos)}"
    )

    print(
        f"Proyectos con múltiples escenarios : "
        f"{multiescenario}"
    )

    print(
        f"Mínimo proyectos/escenario        : "
        f"{minimo}"
    )

    print(
        f"Máximo proyectos/escenario        : "
        f"{maximo}"
    )

    print(
        f"Promedio proyectos/escenario      : "
        f"{promedio:.2f}"
    )

    print(
        f"CV tamaño escenarios              : "
        f"{cv:.4f}"
    )

    return conteos, {
        "escenarios": int(len(conteos)),
        "multiescenario": multiescenario,
        "minimo_proyectos_escenario": minimo,
        "maximo_proyectos_escenario": maximo,
        "promedio_proyectos_escenario": promedio,
        "cv_tamano_escenarios": cv,
    }


# =============================================================================
# GEOMETRÍAS DE ESCENARIOS
# =============================================================================

def construir_geometrias_escenarios(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    subtitulo(
        "CONSTRUYENDO GEOMETRÍAS DE ESCENARIOS"
    )

    registros = []

    for escenario_id, grupo in gdf.groupby(
        "escenario_id",
        dropna=False,
    ):

        geometrias = [
            geom
            for geom in grupo.geometry
            if geometry_is_valid_object(geom)
            and not geom.is_empty
        ]

        geometry = None

        if geometrias:

            try:
                from shapely.ops import unary_union

                geometry = unary_union(
                    geometrias
                )

            except Exception:

                geometry = geometrias[0]

        registros.append(
            {
                "escenario_id": escenario_id,
                "cantidad_proyectos": len(grupo),
                "geometry": geometry,
            }
        )

    escenarios = gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=gdf.crs,
    )

    return escenarios


# =============================================================================
# TABLA DE VALIDACIÓN
# =============================================================================

def construir_validacion(
    gdf: gpd.GeoDataFrame,
    conteos: pd.DataFrame,
) -> pd.DataFrame:

    validacion = gdf.copy()

    validacion["geometria_nula"] = (
        validacion.geometry.isna()
    )

    validacion["geometria_vacia"] = (
        validacion.geometry
        .apply(
            lambda g:
                geometry_is_valid_object(g)
                and g.is_empty
        )
    )

    validacion["geometria_invalida"] = (
        validacion.geometry
        .apply(
            lambda g:
                geometry_is_valid_object(g)
                and not g.is_valid
        )
    )

    validacion["geometria_valida"] = (
        validacion.geometry
        .apply(
            lambda g:
                geometry_is_valid_object(g)
                and not g.is_empty
                and g.is_valid
        )
    )

    validacion["escenario_proyectos"] = (
        validacion["escenario_id"]
        .map(
            dict(
                zip(
                    conteos["escenario_id"],
                    conteos["cantidad_proyectos"],
                )
            )
        )
    )

    validacion["proyecto_multiescenario"] = (
        validacion["proyecto_id"]
        .map(
            gdf.groupby(
                "proyecto_id"
            )["escenario_id"]
            .nunique()
        )
        .fillna(0)
        .gt(1)
    )

    columnas = [
        "proyecto_id",
        "escenario_id",
        "geometria_valida",
        "geometria_nula",
        "geometria_vacia",
        "geometria_invalida",
        "escenario_proyectos",
        "proyecto_multiescenario",
    ]

    columnas = [
        c for c in columnas
        if c in validacion.columns
    ]

    salida = validacion[columnas].copy()

    return salida


# =============================================================================
# AUDITORÍA
# =============================================================================

def construir_auditoria(
    base: dict,
    geometria: dict,
    territorial: dict,
) -> pd.DataFrame:

    filas = [
        {
            "control": "registros",
            "valor": base["registros"],
            "estado": (
                "OK"
                if base["registros"] > 0
                else "ERROR"
            ),
        },
        {
            "control": "proyectos_unicos",
            "valor": base["proyectos_unicos"],
            "estado": (
                "OK"
                if base["proyectos_unicos"] > 0
                else "ERROR"
            ),
        },
        {
            "control": "proyectos_nulos",
            "valor": base["proyectos_nulos"],
            "estado": (
                "OK"
                if base["proyectos_nulos"] == 0
                else "OBSERVADO"
            ),
        },
        {
            "control": "proyectos_duplicados",
            "valor": base["proyectos_duplicados"],
            "estado": (
                "OK"
                if base["proyectos_duplicados"] == 0
                else "OBSERVADO"
            ),
        },
        {
            "control": "escenarios_nulos",
            "valor": base["escenarios_nulos"],
            "estado": (
                "OK"
                if base["escenarios_nulos"] == 0
                else "OBSERVADO"
            ),
        },
        {
            "control": "geometrias_validas",
            "valor": geometria[
                "geometrias_validas"
            ],
            "estado": (
                "OK"
                if geometria[
                    "geometrias_validas"
                ] == base["registros"]
                else "OBSERVADO"
            ),
        },
        {
            "control": "geometrias_nulas",
            "valor": geometria[
                "geometrias_nulas"
            ],
            "estado": (
                "OK"
                if geometria[
                    "geometrias_nulas"
                ] == 0
                else "OBSERVADO"
            ),
        },
        {
            "control": "geometrias_invalidas",
            "valor": geometria[
                "geometrias_invalidas"
            ],
            "estado": (
                "OK"
                if geometria[
                    "geometrias_invalidas"
                ] == 0
                else "OBSERVADO"
            ),
        },
        {
            "control": "proyectos_multiescenario",
            "valor": territorial[
                "multiescenario"
            ],
            "estado": (
                "OK"
                if territorial[
                    "multiescenario"
                ] == 0
                else "OBSERVADO"
            ),
        },
    ]

    return pd.DataFrame(filas)


# =============================================================================
# RESUMEN
# =============================================================================

def construir_resumen(
    base: dict,
    geometria: dict,
    territorial: dict,
) -> dict:

    cobertura = geometria[
        "cobertura_geometrica_pct"
    ]

    auditoria_ok = (
        base["proyectos_nulos"] == 0
        and base["proyectos_duplicados"] == 0
        and base["escenarios_nulos"] == 0
        and geometria["geometrias_nulas"] == 0
        and geometria["geometrias_vacias"] == 0
        and geometria["geometrias_invalidas"] == 0
        and territorial["multiescenario"] == 0
    )

    if auditoria_ok:
        dictamen = "VALIDADO"
        estado = "OK"

    elif (
        cobertura >= 95.0
        and territorial["multiescenario"] == 0
    ):
        dictamen = "VALIDADO_CON_OBSERVACIONES"
        estado = "OBSERVADO"

    else:
        dictamen = "OBSERVADO"
        estado = "OBSERVADO"

    return {
        "proceso": 36,
        "version": SCRIPT_VERSION,
        "proyecto": str(PROJECT_ROOT),
        "entrada_cartera": str(INPUT_CARTERA),
        "entrada_canonica": str(INPUT_CANONICO),
        "registros": base["registros"],
        "proyectos_unicos": base["proyectos_unicos"],
        "proyectos_nulos": base["proyectos_nulos"],
        "proyectos_duplicados": base["proyectos_duplicados"],
        "escenarios": base["escenarios"],
        "escenarios_nulos": base["escenarios_nulos"],
        "geometrias_validas": geometria[
            "geometrias_validas"
        ],
        "geometrias_nulas": geometria[
            "geometrias_nulas"
        ],
        "geometrias_vacias": geometria[
            "geometrias_vacias"
        ],
        "geometrias_invalidas": geometria[
            "geometrias_invalidas"
        ],
        "cobertura_geometrica_pct": cobertura,
        "proyectos_multiescenario": territorial[
            "multiescenario"
        ],
        "minimo_proyectos_escenario": territorial[
            "minimo_proyectos_escenario"
        ],
        "maximo_proyectos_escenario": territorial[
            "maximo_proyectos_escenario"
        ],
        "promedio_proyectos_escenario": territorial[
            "promedio_proyectos_escenario"
        ],
        "cv_tamano_escenarios": territorial[
            "cv_tamano_escenarios"
        ],
        "auditoria": estado,
        "dictamen": dictamen,
    }


# =============================================================================
# EXPORTACIÓN CSV
# =============================================================================

def exportar_validacion(
    validacion: pd.DataFrame,
) -> None:

    export = validacion.copy()

    # Nunca intentamos escribir objetos Shapely en CSV.
    if "geometry" in export.columns:
        export["geometry"] = export[
            "geometry"
        ].apply(
            serializar_geometry_wkt
        )

    export.to_csv(
        OUTPUT_VALIDACION,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# EXPORTACIÓN GEOPACKAGE
# =============================================================================

def exportar_geopackage(
    gdf: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
) -> None:

    subtitulo(
        "EXPORTANDO GEOMETRÍAS A GEOPACKAGE"
    )

    # -------------------------------------------------------------------------
    # PROYECTOS
    # -------------------------------------------------------------------------

    try:

        proyectos = gdf.copy()

        columnas_problematicas = [
            c
            for c in proyectos.columns
            if c != "geometry"
            and isinstance(
                proyectos[c].iloc[0]
                if len(proyectos)
                else None,
                (dict, list, tuple),
            )
        ]

        if columnas_problematicas:
            for col in columnas_problematicas:
                proyectos[col] = (
                    proyectos[col]
                    .astype(str)
                )

        # Eliminar columnas con tipos no soportados
        # por Fiona/GDAL cuando corresponda.
        proyectos.to_file(
            OUTPUT_GPKG_PROYECTOS,
            layer="proyectos",
            driver="GPKG",
        )

        print(
            f"Proyectos : {OUTPUT_GPKG_PROYECTOS}"
        )

    except Exception as exc:

        print(
            "ADVERTENCIA GPKG PROYECTOS: "
            f"{exc}"
        )

    # -------------------------------------------------------------------------
    # ESCENARIOS
    # -------------------------------------------------------------------------

    try:

        escenarios_out = escenarios.copy()

        escenarios_out.to_file(
            OUTPUT_GPKG_ESCENARIOS,
            layer="escenarios",
            driver="GPKG",
        )

        print(
            f"Escenarios: {OUTPUT_GPKG_ESCENARIOS}"
        )

    except Exception as exc:

        print(
            "ADVERTENCIA GPKG ESCENARIOS: "
            f"{exc}"
        )


# =============================================================================
# MARKDOWN
# =============================================================================

def generar_markdown(
    resumen: dict,
    conteos: pd.DataFrame,
    escenarios: gpd.GeoDataFrame,
) -> str:

    lines = []

    lines.append(
        "# Síntesis geoespacial de la cartera territorial AMBA V4"
    )

    lines.append("")

    lines.append(
        "## Proceso"
    )

    lines.append("")

    lines.append(
        "Validación geoespacial y control de geometrías "
        "del proceso 36."
    )

    lines.append("")

    lines.append(
        "## Resultado"
    )

    lines.append("")

    lines.append(
        f"- Proyectos: **{resumen['registros']}**"
    )

    lines.append(
        f"- Proyectos únicos: **{resumen['proyectos_unicos']}**"
    )

    lines.append(
        f"- Escenarios: **{resumen['escenarios']}**"
    )

    lines.append(
        f"- Cobertura geométrica: "
        f"**{resumen['cobertura_geometrica_pct']:.2f}%**"
    )

    lines.append(
        f"- Geometrías válidas: "
        f"**{resumen['geometrias_validas']}**"
    )

    lines.append(
        f"- Geometrías nulas: "
        f"**{resumen['geometrias_nulas']}**"
    )

    lines.append(
        f"- Geometrías vacías: "
        f"**{resumen['geometrias_vacias']}**"
    )

    lines.append(
        f"- Geometrías inválidas: "
        f"**{resumen['geometrias_invalidas']}**"
    )

    lines.append(
        f"- Proyectos multiescenario: "
        f"**{resumen['proyectos_multiescenario']}**"
    )

    lines.append(
        f"- Dictamen: **{resumen['dictamen']}**"
    )

    lines.append("")

    lines.append(
        "## Distribución territorial"
    )

    lines.append("")

    lines.append(
        "| Escenario | Proyectos |"
    )

    lines.append(
        "|---|---:|"
    )

    for _, row in conteos.iterrows():

        escenario_id = row.get(
            "escenario_id",
            "",
        )

        cantidad = safe_int(
            row.get(
                "cantidad_proyectos",
                0,
            )
        )

        lines.append(
            f"| {escenario_id} | {cantidad} |"
        )

    lines.append("")

    lines.append(
        "## Geometrías de escenario"
    )

    lines.append("")

    for _, row in escenarios.iterrows():

        escenario_id = row.get(
            "escenario_id",
            "",
        )

        cantidad = safe_int(
            row.get(
                "cantidad_proyectos",
                0,
            )
        )

        geom = row.get(
            "geometry"
        )

        if geometry_is_valid_object(geom):
            geom_tipo = geom.geom_type
            area = safe_float(
                geom.area,
                0.0,
            )
        else:
            geom_tipo = "N/D"
            area = 0.0

        lines.append(
            f"- **{escenario_id}**: "
            f"{cantidad} proyectos; "
            f"geometría `{geom_tipo}`."
        )

    lines.append("")

    lines.append(
        "## Conclusión"
    )

    lines.append("")

    if resumen["dictamen"] == "VALIDADO":

        lines.append(
            "La cartera territorial presenta cobertura "
            "geoespacial completa, geometrías válidas y "
            "asignación territorial consistente."
        )

        lines.append("")

        lines.append(
            "La salida queda preparada para la etapa "
            "final de integración territorial, programación "
            "de inversiones y elaboración del informe AMBA."
        )

    elif (
        resumen["dictamen"]
        == "VALIDADO_CON_OBSERVACIONES"
    ):

        lines.append(
            "La cartera presenta cobertura geoespacial "
            "suficiente, aunque persisten observaciones "
            "menores que deberán quedar documentadas."
        )

    else:

        lines.append(
            "La cartera presenta observaciones geoespaciales "
            "que deben ser corregidas antes de considerar "
            "cerrada la etapa de validación."
        )

    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    titulo(
        "36 - VALIDACIÓN GEOESPACIAL Y CONTROL DE GEOMETRÍAS - V4.0"
    )

    print(
        f"Proyecto : {PROJECT_ROOT}"
    )

    print(
        f"Entrada  : {DATA_DIR}"
    )

    print(
        f"Salida   : {DATA_DIR}"
    )

    asegurar_directorio_salida()

    # -------------------------------------------------------------------------
    # CARGA
    # -------------------------------------------------------------------------

    cartera = cargar_cartera()

    fuente = cargar_fuente_canonica()

    # -------------------------------------------------------------------------
    # RESOLUCIÓN DE CAMPOS
    # -------------------------------------------------------------------------

    campos = resolver_campos_cartera(
        cartera
    )

    # -------------------------------------------------------------------------
    # PROYECTO
    # -------------------------------------------------------------------------

    cartera["proyecto_id"] = (
        cartera[campos["proyecto"]]
        .apply(normalizar_texto)
    )

    # -------------------------------------------------------------------------
    # ESCENARIO
    # -------------------------------------------------------------------------

    cartera, info_escenario = recuperar_escenarios(
        cartera=cartera,
        fuente=fuente,
        campo_proyecto_cartera=campos["proyecto"],
        campo_escenario_cartera=campos["escenario"],
    )

    # -------------------------------------------------------------------------
    # GEOMETRÍA
    # -------------------------------------------------------------------------

    gdf = recuperar_geometrias(
        cartera=cartera,
        fuente=fuente,
        campo_proyecto_cartera=campos["proyecto"],
        campo_geometria_cartera=campos["geometria"],
    )

    # -------------------------------------------------------------------------
    # VALIDACIÓN BASE
    # -------------------------------------------------------------------------

    base = validar_base(
        gdf
    )

    # -------------------------------------------------------------------------
    # VALIDACIÓN GEOMÉTRICA
    # -------------------------------------------------------------------------

    gdf, geometria = validar_geometrias(
        gdf
    )

    # -------------------------------------------------------------------------
    # VALIDACIÓN TERRITORIAL
    # -------------------------------------------------------------------------

    conteos, territorial = (
        validar_consistencia_territorial(
            gdf
        )
    )

    # -------------------------------------------------------------------------
    # GEOMETRÍAS DE ESCENARIOS
    # -------------------------------------------------------------------------

    escenarios = construir_geometrias_escenarios(
        gdf
    )

    # -------------------------------------------------------------------------
    # TABLA DE VALIDACIÓN
    # -------------------------------------------------------------------------

    validacion = construir_validacion(
        gdf,
        conteos,
    )

    # -------------------------------------------------------------------------
    # AUDITORÍA
    # -------------------------------------------------------------------------

    subtitulo(
        "CONSTRUYENDO AUDITORÍA DEL PROCESO 36"
    )

    auditoria = construir_auditoria(
        base,
        geometria,
        territorial,
    )

    # -------------------------------------------------------------------------
    # RESUMEN
    # -------------------------------------------------------------------------

    resumen = construir_resumen(
        base,
        geometria,
        territorial,
    )

    # -------------------------------------------------------------------------
    # EXPORTACIÓN
    # -------------------------------------------------------------------------

    subtitulo(
        "EXPORTANDO RESULTADOS DEL PROCESO 36"
    )

    exportar_validacion(
        validacion
    )

    auditoria.to_csv(
        OUTPUT_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Validación : {OUTPUT_VALIDACION}"
    )

    print(
        f"Auditoría  : {OUTPUT_AUDITORIA}"
    )

    # -------------------------------------------------------------------------
    # GEOPACKAGES
    # -------------------------------------------------------------------------

    exportar_geopackage(
        gdf,
        escenarios,
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    with OUTPUT_RESUMEN.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            resumen,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Resumen    : {OUTPUT_RESUMEN}"
    )

    # -------------------------------------------------------------------------
    # MARKDOWN
    # -------------------------------------------------------------------------

    subtitulo(
        "GENERANDO SÍNTESIS GEOESPACIAL"
    )

    markdown = generar_markdown(
        resumen,
        conteos,
        escenarios,
    )

    OUTPUT_MARKDOWN.write_text(
        markdown,
        encoding="utf-8",
    )

    print(
        f"Markdown   : {OUTPUT_MARKDOWN}"
    )

    # -------------------------------------------------------------------------
    # RESULTADO
    # -------------------------------------------------------------------------

    titulo(
        "RESULTADO FINAL DEL PROCESO 36"
    )

    print(
        f"Proyectos                 : "
        f"{base['registros']}"
    )

    print(
        f"Proyectos únicos          : "
        f"{base['proyectos_unicos']}"
    )

    print(
        f"Escenarios                : "
        f"{base['escenarios']}"
    )

    print(
        f"Cobertura geométrica      : "
        f"{geometria['cobertura_geometrica_pct']:.2f}%"
    )

    print(
        f"Geometrías válidas        : "
        f"{geometria['geometrias_validas']}"
    )

    print(
        f"Geometrías nulas          : "
        f"{geometria['geometrias_nulas']}"
    )

    print(
        f"Geometrías vacías         : "
        f"{geometria['geometrias_vacias']}"
    )

    print(
        f"Geometrías inválidas      : "
        f"{geometria['geometrias_invalidas']}"
    )

    print(
        f"Proyectos multiescenario  : "
        f"{territorial['multiescenario']}"
    )

    print(
        f"CV tamaño escenarios      : "
        f"{territorial['cv_tamano_escenarios']:.4f}"
    )

    print(
        f"Auditoría                 : "
        f"{resumen['auditoria']}"
    )

    print(
        f"Dictamen                  : "
        f"{resumen['dictamen']}"
    )

    print()

    if resumen["dictamen"] == "VALIDADO":

        print(
            "La cartera territorial presenta cobertura "
            "geoespacial completa y geometrías válidas."
        )

        print(
            "La asignación proyecto -> escenario es consistente."
        )

        print(
            "La salida queda preparada para la integración "
            "territorial final y la elaboración del informe AMBA."
        )

    elif (
        resumen["dictamen"]
        == "VALIDADO_CON_OBSERVACIONES"
    ):

        print(
            "La validación geoespacial es suficiente, "
            "pero existen observaciones menores."
        )

    else:

        print(
            "La cartera presenta observaciones geoespaciales "
            "que deben ser revisadas antes de la etapa final."
        )

    print()

    return 0


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nProceso interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 88)
        print(
            "ERROR FATAL EN EL PROCESO 36"
        )
        print("=" * 88)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        traceback.print_exc()

        sys.exit(1)