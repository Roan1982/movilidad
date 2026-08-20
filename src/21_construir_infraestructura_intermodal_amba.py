# -*- coding: utf-8 -*-
"""
21_construir_infraestructura_intermodal_amba.py

INVENTARIO DE INFRAESTRUCTURA DE TRANSPORTE E INTERMODALIDAD DEL AMBA

Versión mejorada.

Objetivos:
1. Descargar infraestructura de transporte desde OpenStreetMap / Overpass.
2. Clasificar la infraestructura por modo y jerarquía.
3. Separar infraestructura estructural de paradas y plataformas.
4. Detectar posibles intercambiadores intermodales.
5. Deduplicar elementos OSM sin destruir información relevante.
6. Asociar infraestructura a las 144 centralidades SUBE.
7. Calcular indicadores de intermodalidad.
8. Generar GeoParquet, GeoPackage, CSV y JSON.
9. Generar mapas y gráficos de control.

No requiere scipy.
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from shapely.geometry import Point


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "infraestructura_intermodal_amba"
)

CENTRALIDADES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sube_2025_validacion_centralidades.parquet"
)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# BBOX:
# sur, oeste, norte, este
AMBA_BBOX = "-35.20,-59.40,-34.20,-57.80"

CRS_WGS84 = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3

# Radios de análisis.
RADIOS_INTERMODALIDAD_M = [250, 500, 1000]

# Distancia máxima para detectar elementos del mismo lugar físico.
DEDUP_TOLERANCIA_M = 75

# Distancia para considerar dos infraestructuras como potencialmente
# pertenecientes al mismo intercambiador.
INTERCAMBIADOR_TOLERANCIA_M = 150


# ======================================================================
# UTILIDADES
# ======================================================================

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


def safe_int(valor: Any) -> int | None:
    try:
        if valor is None:
            return None
        return int(valor)
    except Exception:
        return None


def safe_float(valor: Any) -> float | None:
    try:
        if valor is None:
            return None
        return float(valor)
    except Exception:
        return None


def normalizar_texto(valor: Any) -> str:
    if valor is None:
        return ""

    texto = str(valor).strip().lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    return " ".join(texto.split())


def valor_tag(tags: dict, campo: str) -> str | None:
    valor = tags.get(campo)

    if valor is None:
        return None

    valor = str(valor).strip()

    return valor if valor else None


def extraer_nombre(tags: dict) -> str | None:

    for campo in [
        "name",
        "official_name",
        "short_name",
        "alt_name",
        "loc_name",
    ]:

        valor = valor_tag(tags, campo)

        if valor:
            return valor

    return None


def extraer_operador(tags: dict) -> str | None:

    for campo in [
        "operator",
        "network",
        "brand",
        "operator:wikidata",
    ]:

        valor = valor_tag(tags, campo)

        if valor:
            return valor

    return None


# ======================================================================
# OVERPASS
# ======================================================================

def construir_query_overpass() -> str:
    """
    Infraestructura de transporte relevante.

    Se incluyen:

    FERROCARRIL
    - station
    - halt
    - stop
    - tram_stop
    - subway
    - light_rail

    AUTOBUS
    - bus_station
    - public_transport=station
    - public_transport=stop_position
    - public_transport=platform

    FLUVIAL
    - ferry_terminal
    - route=ferry

    También se consultan elementos way cuando OSM los modela como
    infraestructura física.
    """

    return f"""
[out:json][timeout:240];

(
    node["railway"="station"]({AMBA_BBOX});
    way["railway"="station"]({AMBA_BBOX});

    node["railway"="halt"]({AMBA_BBOX});
    way["railway"="halt"]({AMBA_BBOX});

    node["railway"="stop"]({AMBA_BBOX});
    way["railway"="stop"]({AMBA_BBOX});

    node["railway"="tram_stop"]({AMBA_BBOX});
    way["railway"="tram_stop"]({AMBA_BBOX});

    node["railway"="subway"]({AMBA_BBOX});
    way["railway"="subway"]({AMBA_BBOX});

    node["railway"="light_rail"]({AMBA_BBOX});
    way["railway"="light_rail"]({AMBA_BBOX});

    node["amenity"="bus_station"]({AMBA_BBOX});
    way["amenity"="bus_station"]({AMBA_BBOX});

    node["amenity"="ferry_terminal"]({AMBA_BBOX});
    way["amenity"="ferry_terminal"]({AMBA_BBOX});

    node["public_transport"="station"]({AMBA_BBOX});
    way["public_transport"="station"]({AMBA_BBOX});

    node["public_transport"="stop_position"]({AMBA_BBOX});
    way["public_transport"="stop_position"]({AMBA_BBOX});

    node["public_transport"="platform"]({AMBA_BBOX});
    way["public_transport"="platform"]({AMBA_BBOX});

    way["route"="ferry"]({AMBA_BBOX});
);

out center tags;
"""


def consultar_overpass() -> dict:

    query = construir_query_overpass()

    ultimo_error: Exception | None = None

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for intento in range(
        1,
        MAX_RETRIES + 1,
    ):

        print(
            f"Intento {intento}/{MAX_RETRIES}"
        )

        for endpoint in OVERPASS_URLS:

            print(
                f"Endpoint: {endpoint}"
            )

            try:

                respuesta = requests.post(
                    endpoint,
                    data=query.encode("utf-8"),
                    headers={
                        "Content-Type":
                            "application/x-www-form-urlencoded; "
                            "charset=UTF-8",
                        "User-Agent":
                            "analisis-movilidad-amba/2.0",
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                print(
                    f"HTTP: {respuesta.status_code}"
                )

                respuesta.raise_for_status()

                datos = respuesta.json()

                elementos = datos.get(
                    "elements"
                )

                if not isinstance(
                    elementos,
                    list,
                ):
                    raise RuntimeError(
                        "La respuesta de Overpass "
                        "no contiene una lista 'elements'."
                    )

                print(
                    f"Elementos recibidos: "
                    f"{len(elementos):,}"
                )

                raw_path = (
                    OUTPUT_DIR
                    / "overpass_infraestructura_intermodal_amba.json"
                )

                with open(
                    raw_path,
                    "w",
                    encoding="utf-8",
                ) as archivo:

                    json.dump(
                        datos,
                        archivo,
                        ensure_ascii=False,
                        indent=2,
                    )

                print(
                    f"Respuesta cruda guardada:\n{raw_path}"
                )

                return datos

            except Exception as exc:

                ultimo_error = exc

                print(
                    f"Error: {type(exc).__name__}: {exc}"
                )

                time.sleep(3)

    raise RuntimeError(
        "No fue posible consultar Overpass. "
        f"Último error: {ultimo_error}"
    )


# ======================================================================
# GEOMETRÍA
# ======================================================================

def geometry_from_element(
    element: dict,
):
    """
    Convierte elementos OSM en puntos.

    Para ways se utiliza center porque la consulta Overpass
    devuelve out center tags.
    """

    tipo = element.get("type")

    if tipo == "node":

        lat = element.get("lat")
        lon = element.get("lon")

        if lat is None or lon is None:
            return None

        return Point(
            float(lon),
            float(lat),
        )

    center = element.get("center")

    if center:

        lat = center.get("lat")
        lon = center.get("lon")

        if lat is not None and lon is not None:

            return Point(
                float(lon),
                float(lat),
            )

    return None


# ======================================================================
# CLASIFICACIÓN
# ======================================================================

def clasificar_elemento(
    tags: dict,
) -> dict[str, Any]:
    """
    Clasificación jerárquica.

    Retorna:

    modo_principal
    tipo_infraestructura
    jerarquia
    categoria
    es_parada
    es_estacion
    es_terminal
    es_plataforma
    es_intercambiador_potencial
    """

    railway = normalizar_texto(
        tags.get("railway")
    )

    amenity = normalizar_texto(
        tags.get("amenity")
    )

    public_transport = normalizar_texto(
        tags.get("public_transport")
    )

    route = normalizar_texto(
        tags.get("route")
    )

    # --------------------------------------------------------------
    # FERROCARRIL
    # --------------------------------------------------------------

    if railway == "station":

        return {
            "modo_principal": "FERROCARRIL",
            "tipo_infraestructura":
                "ESTACION_FERROVIARIA",
            "jerarquia": "ESTRUCTURAL",
            "categoria": "FERROVIARIO",
            "es_parada": 0,
            "es_estacion": 1,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 1,
        }

    if railway == "halt":

        return {
            "modo_principal": "FERROCARRIL",
            "tipo_infraestructura":
                "PARADA_FERROVIARIA",
            "jerarquia": "PARADA",
            "categoria": "FERROVIARIO",
            "es_parada": 1,
            "es_estacion": 0,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 0,
        }

    if railway == "stop":

        return {
            "modo_principal": "FERROCARRIL",
            "tipo_infraestructura":
                "PARADA_FERROVIARIA",
            "jerarquia": "PARADA",
            "categoria": "FERROVIARIO",
            "es_parada": 1,
            "es_estacion": 0,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 0,
        }

    # --------------------------------------------------------------
    # SUBTE
    # --------------------------------------------------------------

    if railway == "subway":

        return {
            "modo_principal": "SUBTE",
            "tipo_infraestructura":
                "ESTACION_SUBTE",
            "jerarquia": "ESTRUCTURAL",
            "categoria": "FERROVIARIO",
            "es_parada": 0,
            "es_estacion": 1,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 1,
        }

    # --------------------------------------------------------------
    # TRANVÍA
    # --------------------------------------------------------------

    if railway == "tram_stop":

        return {
            "modo_principal": "TRANVIA",
            "tipo_infraestructura":
                "PARADA_TRANVIA",
            "jerarquia": "PARADA",
            "categoria": "FERROVIARIO",
            "es_parada": 1,
            "es_estacion": 0,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 0,
        }

    # --------------------------------------------------------------
    # LIGHT RAIL
    # --------------------------------------------------------------

    if railway == "light_rail":

        return {
            "modo_principal": "FERROCARRIL",
            "tipo_infraestructura":
                "ESTACION_LIGERO",
            "jerarquia": "ESTRUCTURAL",
            "categoria": "FERROVIARIO",
            "es_parada": 0,
            "es_estacion": 1,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 1,
        }

    # --------------------------------------------------------------
    # TERMINAL DE AUTOBUSES
    # --------------------------------------------------------------

    if amenity == "bus_station":

        return {
            "modo_principal": "AUTOBUS",
            "tipo_infraestructura":
                "TERMINAL_AUTOBUS",
            "jerarquia": "ESTRUCTURAL",
            "categoria": "AUTOMOTOR",
            "es_parada": 0,
            "es_estacion": 0,
            "es_terminal": 1,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 1,
        }

    # --------------------------------------------------------------
    # TERMINAL FLUVIAL
    # --------------------------------------------------------------

    if amenity == "ferry_terminal":

        return {
            "modo_principal": "FLUVIAL",
            "tipo_infraestructura":
                "TERMINAL_FLUVIAL",
            "jerarquia": "ESTRUCTURAL",
            "categoria": "FLUVIAL",
            "es_parada": 0,
            "es_estacion": 0,
            "es_terminal": 1,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 1,
        }

    # --------------------------------------------------------------
    # ESTACIÓN DE TRANSPORTE PÚBLICO
    # --------------------------------------------------------------

    if public_transport == "station":

        return {
            "modo_principal": "MULTIMODAL",
            "tipo_infraestructura":
                "ESTACION_TRANSPORTE_PUBLICO",
            "jerarquia": "INTERCAMBIO",
            "categoria": "INTERMODAL",
            "es_parada": 0,
            "es_estacion": 1,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 1,
        }

    # --------------------------------------------------------------
    # PARADA DE TRANSPORTE PÚBLICO
    # --------------------------------------------------------------

    if public_transport == "stop_position":

        return {
            "modo_principal": "AUTOBUS",
            "tipo_infraestructura":
                "PARADA_TRANSPORTE_PUBLICO",
            "jerarquia": "PARADA",
            "categoria": "AUTOMOTOR",
            "es_parada": 1,
            "es_estacion": 0,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 0,
        }

    # --------------------------------------------------------------
    # PLATAFORMA
    # --------------------------------------------------------------

    if public_transport == "platform":

        return {
            "modo_principal": "AUTOBUS",
            "tipo_infraestructura":
                "PLATAFORMA_TRANSPORTE_PUBLICO",
            "jerarquia": "SOPORTE",
            "categoria": "AUTOMOTOR",
            "es_parada": 1,
            "es_estacion": 0,
            "es_terminal": 0,
            "es_plataforma": 1,
            "es_intercambiador_potencial": 0,
        }

    # --------------------------------------------------------------
    # RUTA FLUVIAL
    # --------------------------------------------------------------

    if route == "ferry":

        return {
            "modo_principal": "FLUVIAL",
            "tipo_infraestructura":
                "RUTA_FLUVIAL",
            "jerarquia": "RED",
            "categoria": "FLUVIAL",
            "es_parada": 0,
            "es_estacion": 0,
            "es_terminal": 0,
            "es_plataforma": 0,
            "es_intercambiador_potencial": 0,
        }

    # --------------------------------------------------------------
    # OTRO
    # --------------------------------------------------------------

    return {
        "modo_principal": "OTRO",
        "tipo_infraestructura":
            "OTRA_INFRAESTRUCTURA",
        "jerarquia": "OTRO",
        "categoria": "OTRO",
        "es_parada": 0,
        "es_estacion": 0,
        "es_terminal": 0,
        "es_plataforma": 0,
        "es_intercambiador_potencial": 0,
    }


# ======================================================================
# CONSTRUCCIÓN DEL GDF
# ======================================================================

def construir_gdf(
    datos: dict,
) -> gpd.GeoDataFrame:

    registros = []

    elementos = datos.get(
        "elements",
        [],
    )

    for element in elementos:

        tags = element.get(
            "tags"
        ) or {}

        geometry = geometry_from_element(
            element
        )

        if geometry is None:
            continue

        if geometry.is_empty:
            continue

        clasificacion = clasificar_elemento(
            tags
        )

        registro = {

            "osm_type":
                element.get("type"),

            "osm_id":
                safe_int(element.get("id")),

            "nombre":
                extraer_nombre(tags),

            "operador":
                extraer_operador(tags),

            "ref":
                valor_tag(tags, "ref"),

            "wikidata":
                valor_tag(tags, "wikidata"),

            "website":
                valor_tag(tags, "website"),

            "railway":
                valor_tag(tags, "railway"),

            "amenity":
                valor_tag(tags, "amenity"),

            "public_transport":
                valor_tag(tags, "public_transport"),

            "route":
                valor_tag(tags, "route"),

            "network":
                valor_tag(tags, "network"),

            "brand":
                valor_tag(tags, "brand"),

            "modo_principal":
                clasificacion[
                    "modo_principal"
                ],

            "tipo_infraestructura":
                clasificacion[
                    "tipo_infraestructura"
                ],

            "jerarquia":
                clasificacion[
                    "jerarquia"
                ],

            "categoria_intermodal":
                clasificacion[
                    "categoria"
                ],

            "es_parada":
                clasificacion[
                    "es_parada"
                ],

            "es_estacion":
                clasificacion[
                    "es_estacion"
                ],

            "es_terminal":
                clasificacion[
                    "es_terminal"
                ],

            "es_plataforma":
                clasificacion[
                    "es_plataforma"
                ],

            "es_intercambiador_potencial":
                clasificacion[
                    "es_intercambiador_potencial"
                ],

            "geometry":
                geometry,
        }

        registros.append(
            registro
        )

    if not registros:

        raise RuntimeError(
            "No se pudieron construir geometrías."
        )

    gdf = gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=CRS_WGS84,
    )

    return gdf


# ======================================================================
# NORMALIZACIÓN
# ======================================================================

def normalizar_infraestructura(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    gdf = gdf.copy()

    gdf["osm_id"] = pd.to_numeric(
        gdf["osm_id"],
        errors="coerce",
    ).astype("Int64")

    columnas_texto = [
        "nombre",
        "operador",
        "ref",
        "wikidata",
        "website",
        "railway",
        "amenity",
        "public_transport",
        "route",
        "network",
        "brand",
        "modo_principal",
        "tipo_infraestructura",
        "jerarquia",
        "categoria_intermodal",
    ]

    for columna in columnas_texto:

        if columna in gdf.columns:

            gdf[columna] = (
                gdf[columna]
                .astype("string")
            )

    gdf["nombre_normalizado"] = (
        gdf["nombre"]
        .fillna("")
        .map(normalizar_texto)
        .astype("string")
    )

    return gdf


# ======================================================================
# VALIDACIÓN
# ======================================================================

def validar_infraestructura(
    gdf: gpd.GeoDataFrame,
    nombre: str,
) -> None:

    subtitulo(
        f"VALIDACIÓN DE INFRAESTRUCTURA {nombre}"
    )

    print(
        f"Registros: {len(gdf):,}"
    )

    geometria_nula = (
        gdf.geometry.isna()
    )

    geometria_vacia = (
        gdf.geometry.is_empty
    )

    geometria_invalida = (
        ~gdf.geometry.is_valid
    )

    duplicados = (
        gdf["osm_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Geometrías nulas: "
        f"{int(geometria_nula.sum()):,}"
    )

    print(
        f"Geometrías vacías: "
        f"{int(geometria_vacia.sum()):,}"
    )

    print(
        f"Geometrías inválidas: "
        f"{int(geometria_invalida.sum()):,}"
    )

    print(
        f"Duplicados OSM: "
        f"{int(duplicados):,}"
    )

    if geometria_nula.any():
        raise RuntimeError(
            "Existen geometrías nulas."
        )

    if geometria_vacia.any():
        raise RuntimeError(
            "Existen geometrías vacías."
        )

    if geometria_invalida.any():
        raise RuntimeError(
            "Existen geometrías inválidas."
        )

    if duplicados:
        raise RuntimeError(
            "Existen OSM IDs duplicados."
        )


# ======================================================================
# ELIMINAR DUPLICADOS OSM
# ======================================================================

def eliminar_duplicados_osm(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    antes = len(gdf)

    gdf = gdf.drop_duplicates(
        subset=[
            "osm_type",
            "osm_id",
        ],
        keep="first",
    ).copy()

    eliminados = (
        antes - len(gdf)
    )

    print(
        f"Duplicados OSM eliminados: "
        f"{eliminados:,}"
    )

    return gdf


# ======================================================================
# DEDUPLICACIÓN ESPACIAL
# ======================================================================

def deduplicar_espacialmente(
    gdf: gpd.GeoDataFrame,
    tolerancia_m: float = DEDUP_TOLERANCIA_M,
) -> gpd.GeoDataFrame:
    """
    Deduplicación conservadora.

    REGLAS:

    1. No se mezclan modos diferentes.
    2. No se mezclan jerarquías diferentes.
    3. Elementos sin nombre no se deduplican entre sí.
    4. Elementos con el mismo nombre, modo y jerarquía
       pueden agruparse si están dentro de la tolerancia.
    """

    if gdf.empty:
        return gdf

    metric = (
        gdf
        .to_crs(CRS_METRICO)
        .copy()
    )

    metric["geometry"] = (
        metric.geometry.centroid
    )

    sindex = metric.sindex

    usados: set[int] = set()

    resultado = []

    for idx, row in metric.iterrows():

        if idx in usados:
            continue

        candidatos = list(
            sindex.query(
                row.geometry.buffer(
                    tolerancia_m
                ),
                predicate="intersects",
            )
        )

        grupo = []

        for candidato in candidatos:

            if candidato in usados:
                continue

            otra = metric.loc[
                candidato
            ]

            mismo_nombre = (
                row["nombre_normalizado"]
                != ""
                and
                row["nombre_normalizado"]
                ==
                otra["nombre_normalizado"]
            )

            mismo_modo = (
                row["modo_principal"]
                ==
                otra["modo_principal"]
            )

            misma_jerarquia = (
                row["jerarquia"]
                ==
                otra["jerarquia"]
            )

            distancia = (
                row.geometry.distance(
                    otra.geometry
                )
            )

            if (
                distancia <= tolerancia_m
                and mismo_nombre
                and mismo_modo
                and misma_jerarquia
            ):
                grupo.append(
                    candidato
                )

        if not grupo:
            grupo = [idx]

        usados.update(
            grupo
        )

        base = metric.loc[
            grupo[0]
        ].copy()

        ids = sorted(
            {
                str(
                    metric.loc[i]["osm_id"]
                )
                for i in grupo
                if pd.notna(
                    metric.loc[i]["osm_id"]
                )
            }
        )

        nombres = sorted(
            {
                str(
                    metric.loc[i]["nombre"]
                )
                for i in grupo
                if pd.notna(
                    metric.loc[i]["nombre"]
                )
                and str(
                    metric.loc[i]["nombre"]
                ).strip()
            }
        )

        base["osm_ids"] = ",".join(
            ids
        )

        base["cantidad_elementos_osm"] = (
            len(grupo)
        )

        base["nombres_osm"] = " | ".join(
            nombres
        )

        resultado.append(
            base
        )

    salida = gpd.GeoDataFrame(
        resultado,
        geometry="geometry",
        crs=CRS_METRICO,
    ).to_crs(
        CRS_WGS84
    )

    return salida


# ======================================================================
# DETECCIÓN DE INTERCAMBIADORES
# ======================================================================

def detectar_intercambiadores(
    gdf: gpd.GeoDataFrame,
    tolerancia_m: float = INTERCAMBIADOR_TOLERANCIA_M,
) -> gpd.GeoDataFrame:
    """
    Detecta lugares donde existen dos o más modos distintos
    próximos entre sí.

    Esto es una aproximación espacial.

    Ejemplo:

        estación ferroviaria
              +
        parada de colectivo
              ↓
        INTERCAMBIADOR POTENCIAL
    """

    if gdf.empty:
        return gdf

    metric = (
        gdf
        .to_crs(CRS_METRICO)
        .copy()
    )

    sindex = metric.sindex

    metric["cantidad_modos_cercanos"] = 0
    metric["modos_cercanos"] = ""
    metric["es_intercambiador"] = 0
    metric["score_intercambio"] = 0.0

    for idx, row in metric.iterrows():

        candidatos = list(
            sindex.query(
                row.geometry.buffer(
                    tolerancia_m
                ),
                predicate="intersects",
            )
        )

        modos = set()

        for candidato in candidatos:

            otro = metric.loc[
                candidato
            ]

            modo = otro[
                "modo_principal"
            ]

            if pd.notna(modo):

                modo = str(modo)

                if modo != "OTRO":

                    modos.add(
                        modo
                    )

        metric.at[
            idx,
            "cantidad_modos_cercanos"
        ] = len(modos)

        metric.at[
            idx,
            "modos_cercanos"
        ] = "|".join(
            sorted(modos)
        )

        if len(modos) >= 2:

            metric.at[
                idx,
                "es_intercambiador"
            ] = 1

            score = min(
                100.0,
                50.0
                + (
                    len(modos) - 2
                )
                * 20.0,
            )

            metric.at[
                idx,
                "score_intercambio"
            ] = score

        else:

            metric.at[
                idx,
                "es_intercambiador"
            ] = 0

            metric.at[
                idx,
                "score_intercambio"
            ] = 0.0

    return metric.to_crs(
        CRS_WGS84
    )


# ======================================================================
# INDICADORES
# ======================================================================

def calcular_indicadores(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    gdf = gdf.copy()

    modos = [
        "FERROCARRIL",
        "SUBTE",
        "AUTOBUS",
        "FLUVIAL",
        "TRANVIA",
    ]

    for modo in modos:

        nombre = normalizar_texto(
            modo
        ).replace(
            " ",
            "_",
        )

        gdf[
            f"modo_bin_{nombre}"
        ] = (
            gdf["modo_principal"]
            == modo
        ).astype(int)

    gdf["es_estructural"] = (
        gdf["jerarquia"]
        == "ESTRUCTURAL"
    ).astype(int)

    gdf["es_intercambio"] = (
        gdf["jerarquia"]
        == "INTERCAMBIO"
    ).astype(int)

    gdf["es_parada"] = (
        gdf["es_parada"]
        .fillna(0)
        .astype(int)
    )

    gdf["es_terminal"] = (
        gdf["es_terminal"]
        .fillna(0)
        .astype(int)
    )

    gdf["es_estacion"] = (
        gdf["es_estacion"]
        .fillna(0)
        .astype(int)
    )

    gdf["es_plataforma"] = (
        gdf["es_plataforma"]
        .fillna(0)
        .astype(int)
    )

    # Peso estructural.
    pesos = {
        "ESTRUCTURAL": 1.0,
        "INTERCAMBIO": 0.9,
        "PARADA": 0.4,
        "SOPORTE": 0.2,
        "RED": 0.1,
        "OTRO": 0.0,
    }

    gdf["peso_jerarquia"] = (
        gdf["jerarquia"]
        .map(pesos)
        .fillna(0.0)
    )

    # Peso modal.
    pesos_modal = {
        "FERROCARRIL": 1.0,
        "SUBTE": 1.0,
        "FLUVIAL": 1.0,
        "AUTOBUS": 0.7,
        "TRANVIA": 0.8,
        "MULTIMODAL": 1.0,
        "OTRO": 0.0,
    }

    gdf["peso_modal"] = (
        gdf["modo_principal"]
        .map(pesos_modal)
        .fillna(0.0)
    )

    gdf["score_infraestructura"] = (
        gdf["peso_jerarquia"]
        * gdf["peso_modal"]
        * 100.0
    )

    return gdf


# ======================================================================
# CENTRALIDADES
# ======================================================================

def cargar_centralidades() -> gpd.GeoDataFrame | None:

    if not CENTRALIDADES_PATH.exists():

        print(
            "ADVERTENCIA: no existe el archivo:"
        )

        print(
            CENTRALIDADES_PATH
        )

        return None

    try:

        centralidades = (
            gpd.read_parquet(
                CENTRALIDADES_PATH
            )
        )

    except Exception as exc:

        print(
            "ADVERTENCIA: error cargando "
            f"centralidades: {exc}"
        )

        return None

    if centralidades.empty:

        return None

    if centralidades.crs is None:

        centralidades = (
            centralidades
            .set_crs(CRS_WGS84)
        )

    centralidades = (
        centralidades
        .to_crs(CRS_METRICO)
        .copy()
    )

    if "nodo_id" not in centralidades.columns:

        centralidades["nodo_id"] = (
            np.arange(
                1,
                len(centralidades) + 1,
            )
        )

    duplicados = (
        centralidades["nodo_id"]
        .duplicated()
        .sum()
    )

    subtitulo(
        "VALIDACIÓN DE CENTRALIDADES"
    )

    print(
        f"Centralidades: {len(centralidades):,}"
    )

    print(
        f"nodo_id duplicados: {duplicados:,}"
    )

    if duplicados:

        raise RuntimeError(
            "Hay nodo_id duplicados."
        )

    return centralidades


# ======================================================================
# ANÁLISIS DE CENTRALIDADES
# ======================================================================

def analizar_centralidades(
    centralidades: gpd.GeoDataFrame,
    infraestructura: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    centros = centralidades.copy()

    infra = (
        infraestructura
        .to_crs(CRS_METRICO)
        .copy()
    )

    sindex = infra.sindex

    resultados = []

    for posicion, (_, nodo) in enumerate(
        centros.iterrows(),
        start=1,
    ):

        if (
            posicion == 1
            or posicion % 25 == 0
            or posicion == len(centros)
        ):

            print(
                f"Centralidad "
                f"{posicion}/{len(centros)}"
            )

        punto = nodo.geometry

        registro = {
            "nodo_id":
                nodo["nodo_id"],
        }

        for radio in RADIOS_INTERMODALIDAD_M:

            candidatos = list(
                sindex.query(
                    punto.buffer(radio),
                    predicate="intersects",
                )
            )

            if not candidatos:

                registro[
                    f"infra_{radio}m"
                ] = 0

                registro[
                    f"infra_estructural_{radio}m"
                ] = 0

                registro[
                    f"infra_paradas_{radio}m"
                ] = 0

                registro[
                    f"intercambiadores_{radio}m"
                ] = 0

                registro[
                    f"modos_{radio}m"
                ] = 0

                for modo in [
                    "FERROCARRIL",
                    "SUBTE",
                    "AUTOBUS",
                    "FLUVIAL",
                    "TRANVIA",
                ]:

                    registro[
                        f"{modo.lower()}_{radio}m"
                    ] = 0

                continue

            sub = infra.iloc[
                candidatos
            ]

            registro[
                f"infra_{radio}m"
            ] = len(sub)

            registro[
                f"infra_estructural_{radio}m"
            ] = int(
                sub[
                    "es_estructural"
                ].sum()
            )

            registro[
                f"infra_paradas_{radio}m"
            ] = int(
                sub[
                    "es_parada"
                ].sum()
            )

            registro[
                f"intercambiadores_{radio}m"
            ] = int(
                sub[
                    "es_intercambiador"
                ].sum()
            )

            modos = set(
                sub[
                    "modo_principal"
                ]
                .dropna()
                .astype(str)
            )

            modos.discard(
                "OTRO"
            )

            registro[
                f"modos_{radio}m"
            ] = len(modos)

            for modo in [
                "FERROCARRIL",
                "SUBTE",
                "AUTOBUS",
                "FLUVIAL",
                "TRANVIA",
            ]:

                registro[
                    f"{modo.lower()}_{radio}m"
                ] = int(
                    modo in modos
                )

        # ----------------------------------------------------------
        # INFRAESTRUCTURA ESTRUCTURAL A 1000 M
        # ----------------------------------------------------------

        candidatos = list(
            sindex.query(
                punto.buffer(1000),
                predicate="intersects",
            )
        )

        if candidatos:

            sub = infra.iloc[
                candidatos
            ]

            distancias = (
                sub.geometry.distance(
                    punto
                )
            )

            minimo = float(
                distancias.min()
            )

            posicion_minima = int(
                np.argmin(
                    distancias.to_numpy()
                )
            )

            mas_cercano = sub.iloc[
                posicion_minima
            ]

            registro[
                "distancia_infraestructura_m"
            ] = minimo

            nombre = (
                mas_cercano["nombre"]
                if pd.notna(
                    mas_cercano["nombre"]
                )
                else None
            )

            if not nombre:

                nombre = (
                    mas_cercano[
                        "tipo_infraestructura"
                    ]
                )

            registro[
                "infraestructura_mas_cercana"
            ] = str(nombre)

            registro[
                "modo_infraestructura_mas_cercana"
            ] = str(
                mas_cercano[
                    "modo_principal"
                ]
            )

        else:

            registro[
                "distancia_infraestructura_m"
            ] = np.nan

            registro[
                "infraestructura_mas_cercana"
            ] = None

            registro[
                "modo_infraestructura_mas_cercana"
            ] = None

        # ----------------------------------------------------------
        # SCORE DE INTERMODALIDAD
        # ----------------------------------------------------------

        modos_500 = registro[
            "modos_500m"
        ]

        infra_estructural_500 = registro[
            "infra_estructural_500m"
        ]

        intercambiadores_500 = registro[
            "intercambiadores_500m"
        ]

        presencia_modal = (
            registro["ferrocarril_500m"]
            + registro["subte_500m"]
            + registro["autobus_500m"]
            + registro["fluvial_500m"]
            + registro["tranvia_500m"]
        )

        # ----------------------------------------------------------
        # COMPONENTES DEL SCORE
        #
        # Diversidad modal: 40 puntos
        # Infraestructura estructural: 30 puntos
        # Intercambiadores: 20 puntos
        # Presencia modal: 10 puntos
        # ----------------------------------------------------------

        componente_modos = min(
            40.0,
            modos_500 / 4.0 * 40.0,
        )

        componente_estructural = min(
            30.0,
            infra_estructural_500
            / 5.0
            * 30.0,
        )

        componente_intercambio = min(
            20.0,
            intercambiadores_500
            / 3.0
            * 20.0,
        )

        componente_presencia = min(
            10.0,
            presencia_modal
            / 4.0
            * 10.0,
        )

        score = (
            componente_modos
            + componente_estructural
            + componente_intercambio
            + componente_presencia
        )

        registro[
            "score_intermodalidad_500m"
        ] = round(
            min(100.0, score),
            2,
        )

        # ----------------------------------------------------------
        # CATEGORÍA
        # ----------------------------------------------------------

        if score >= 75:

            categoria = (
                "INTERMODALIDAD_MUY_ALTA"
            )

        elif score >= 50:

            categoria = (
                "INTERMODALIDAD_ALTA"
            )

        elif score >= 25:

            categoria = (
                "INTERMODALIDAD_MEDIA"
            )

        elif score > 0:

            categoria = (
                "INTERMODALIDAD_BAJA"
            )

        else:

            categoria = (
                "SIN_INFRAESTRUCTURA"
            )

        registro[
            "categoria_intermodalidad_500m"
        ] = categoria

        resultados.append(
            registro
        )

    indicadores = pd.DataFrame(
        resultados
    )

    centros = centros.merge(
        indicadores,
        on="nodo_id",
        how="left",
    )

    centros[
        "ranking_intermodalidad"
    ] = (
        centros[
            "score_intermodalidad_500m"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype("Int64")
    )

    return centros


# ======================================================================
# RESUMEN
# ======================================================================

def construir_resumen(
    infraestructura: gpd.GeoDataFrame,
    centralidades: gpd.GeoDataFrame | None,
) -> dict:

    resumen = {

        "proyecto":
            "Análisis de movilidad SUBE AMBA",

        "script":
            "21_construir_infraestructura_intermodal_amba.py",

        "version":
            "2.0",

        "fecha_ejecucion":
            pd.Timestamp.now().isoformat(),

        "bbox":
            AMBA_BBOX,

        "crs":
            CRS_WGS84,

        "crs_metrico":
            CRS_METRICO,

        "infraestructuras_finales":
            int(len(infraestructura)),

        "osm_ids_unicos":
            int(
                infraestructura[
                    "osm_id"
                ].nunique()
            ),
    }

    for columna, clave in [
        (
            "modo_principal",
            "por_modo",
        ),
        (
            "tipo_infraestructura",
            "por_tipo",
        ),
        (
            "jerarquia",
            "por_jerarquia",
        ),
        (
            "categoria_intermodal",
            "por_categoria",
        ),
    ]:

        if columna in infraestructura.columns:

            conteo = (
                infraestructura[
                    columna
                ]
                .fillna("SIN_DATO")
                .value_counts()
                .to_dict()
            )

            resumen[clave] = {
                str(k): int(v)
                for k, v in conteo.items()
            }

    resumen[
        "intercambiadores_detectados"
    ] = int(
        infraestructura[
            "es_intercambiador"
        ].sum()
    )

    resumen[
        "infraestructura_estructural"
    ] = int(
        infraestructura[
            "es_estructural"
        ].sum()
    )

    resumen[
        "paradas"
    ] = int(
        infraestructura[
            "es_parada"
        ].sum()
    )

    resumen[
        "plataformas"
    ] = int(
        infraestructura[
            "es_plataforma"
        ].sum()
    )

    if centralidades is not None:

        resumen[
            "centralidades_analizadas"
        ] = int(
            len(centralidades)
        )

        if (
            "categoria_intermodalidad_500m"
            in centralidades.columns
        ):

            conteo = (
                centralidades[
                    "categoria_intermodalidad_500m"
                ]
                .value_counts()
                .to_dict()
            )

            resumen[
                "centralidades_por_intermodalidad"
            ] = {
                str(k): int(v)
                for k, v in conteo.items()
            }

        ranking = (
            centralidades
            .sort_values(
                "score_intermodalidad_500m",
                ascending=False,
            )
            .head(10)
        )

        resumen[
            "top_10_centralidades"
        ] = []

        for _, row in ranking.iterrows():

            resumen[
                "top_10_centralidades"
            ].append({

                "nodo_id":
                    safe_int(
                        row["nodo_id"]
                    ),

                "score":
                    safe_float(
                        row[
                            "score_intermodalidad_500m"
                        ]
                    ),

                "infra_500m":
                    safe_int(
                        row[
                            "infra_500m"
                        ]
                    ),

                "infra_estructural_500m":
                    safe_int(
                        row[
                            "infra_estructural_500m"
                        ]
                    ),

                "modos_500m":
                    safe_int(
                        row[
                            "modos_500m"
                        ]
                    ),

                "intercambiadores_500m":
                    safe_int(
                        row[
                            "intercambiadores_500m"
                        ]
                    ),

                "categoria":
                    str(
                        row[
                            "categoria_intermodalidad_500m"
                        ]
                    ),
            })

    return resumen


# ======================================================================
# SALIDAS
# ======================================================================

def guardar_infraestructura(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.parquet"
    )

    gpkg_path = (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.gpkg"
    )

    print(
        "Guardando GeoParquet..."
    )

    infraestructura.to_parquet(
        parquet_path,
        index=False,
    )

    print(
        parquet_path
    )

    print(
        "Guardando GeoPackage..."
    )

    infraestructura.to_file(
        gpkg_path,
        layer="infraestructura",
        driver="GPKG",
    )

    print(
        gpkg_path
    )


def guardar_centralidades(
    centralidades: gpd.GeoDataFrame,
) -> None:

    parquet_path = (
        OUTPUT_DIR
        / "centralidades_intermodalidad_amba.parquet"
    )

    centralidades.to_parquet(
        parquet_path,
        index=False,
    )

    gpkg_path = (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.gpkg"
    )

    centralidades.to_file(
        gpkg_path,
        layer="centralidades_intermodalidad",
        driver="GPKG",
    )

    print(
        f"Centralidades guardadas:\n"
        f"{parquet_path}"
    )


def guardar_csv(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    tabla = (
        infraestructura
        .drop(
            columns="geometry",
            errors="ignore",
        )
        .copy()
    )

    path = (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.csv"
    )

    tabla.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"CSV:\n{path}"
    )


def guardar_resumen(
    resumen: dict,
) -> None:

    path = (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba_resumen.json"
    )

    with open(
        path,
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
        f"JSON:\n{path}"
    )


# ======================================================================
# MAPAS
# ======================================================================

def generar_mapa_infraestructura(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "Matplotlib no disponible."
        )

        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    infraestructura.plot(
        ax=ax,
        column="modo_principal",
        legend=True,
        markersize=7,
        alpha=0.65,
    )

    ax.set_title(
        "Infraestructura de transporte - AMBA",
        fontsize=15,
    )

    ax.set_axis_off()

    path = (
        OUTPUT_DIR
        / "01_mapa_infraestructura_intermodal.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Mapa:\n{path}"
    )


def generar_mapa_centralidades(
    centralidades: gpd.GeoDataFrame,
) -> None:

    if centralidades is None:
        return

    if centralidades.empty:
        return

    if (
        "score_intermodalidad_500m"
        not in centralidades.columns
    ):
        return

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    centralidades.plot(
        ax=ax,
        column="score_intermodalidad_500m",
        cmap="viridis",
        markersize=35,
        legend=True,
    )

    ax.set_title(
        "Centralidades SUBE - "
        "Intermodalidad a 500 m",
        fontsize=15,
    )

    ax.set_axis_off()

    path = (
        OUTPUT_DIR
        / "02_centralidades_intermodalidad_500m.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Mapa:\n{path}"
    )


def generar_grafico_modos(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        return

    conteo = (
        infraestructura[
            "modo_principal"
        ]
        .fillna("SIN_DATO")
        .value_counts()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    conteo.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Infraestructura por modo",
        fontsize=14,
    )

    ax.set_xlabel(
        "Cantidad"
    )

    ax.set_ylabel(
        "Modo"
    )

    path = (
        OUTPUT_DIR
        / "03_infraestructura_por_modo.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico:\n{path}"
    )


def generar_grafico_jerarquia(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        return

    conteo = (
        infraestructura[
            "jerarquia"
        ]
        .fillna("SIN_DATO")
        .value_counts()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    conteo.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Infraestructura por jerarquía",
        fontsize=14,
    )

    ax.set_xlabel(
        "Cantidad"
    )

    ax.set_ylabel(
        "Jerarquía"
    )

    path = (
        OUTPUT_DIR
        / "04_infraestructura_por_jerarquia.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico:\n{path}"
    )


def generar_grafico_intermodalidad(
    centralidades: gpd.GeoDataFrame,
) -> None:

    if centralidades is None:
        return

    if (
        "score_intermodalidad_500m"
        not in centralidades.columns
    ):
        return

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        return

    valores = (
        centralidades[
            "score_intermodalidad_500m"
        ]
        .dropna()
    )

    if valores.empty:
        return

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.hist(
        valores,
        bins=15,
    )

    ax.set_title(
        "Distribución del score de intermodalidad",
        fontsize=14,
    )

    ax.set_xlabel(
        "Score"
    )

    ax.set_ylabel(
        "Cantidad de centralidades"
    )

    path = (
        OUTPUT_DIR
        / "05_distribucion_score_intermodalidad.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico:\n{path}"
    )


# ======================================================================
# RESUMEN DE CONSOLA
# ======================================================================

def imprimir_resumen_inventario(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    subtitulo(
        "RESUMEN DEL INVENTARIO"
    )

    print(
        f"Total: {len(infraestructura):,}"
    )

    print()
    print(
        "POR MODO"
    )

    for modo, cantidad in (
        infraestructura[
            "modo_principal"
        ]
        .value_counts()
        .items()
    ):

        print(
            f"  {str(modo):20s}"
            f"{int(cantidad):8,d}"
        )

    print()
    print(
        "POR JERARQUÍA"
    )

    for jerarquia, cantidad in (
        infraestructura[
            "jerarquia"
        ]
        .value_counts()
        .items()
    ):

        print(
            f"  {str(jerarquia):20s}"
            f"{int(cantidad):8,d}"
        )

    print()
    print(
        "POR TIPO"
    )

    for tipo, cantidad in (
        infraestructura[
            "tipo_infraestructura"
        ]
        .value_counts()
        .items()
    ):

        print(
            f"  {str(tipo):40s}"
            f"{int(cantidad):8,d}"
        )

    print()
    print(
        "INDICADORES"
    )

    print(
        "  Infraestructura estructural:",
        int(
            infraestructura[
                "es_estructural"
            ].sum()
        ),
    )

    print(
        "  Estaciones:",
        int(
            infraestructura[
                "es_estacion"
            ].sum()
        ),
    )

    print(
        "  Terminales:",
        int(
            infraestructura[
                "es_terminal"
            ].sum()
        ),
    )

    print(
        "  Paradas:",
        int(
            infraestructura[
                "es_parada"
            ].sum()
        ),
    )

    print(
        "  Plataformas:",
        int(
            infraestructura[
                "es_plataforma"
            ].sum()
        ),
    )

    print(
        "  Intercambiadores potenciales:",
        int(
            infraestructura[
                "es_intercambiador"
            ].sum()
        ),
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    titulo(
        "21 - CONSTRUCCIÓN DE INFRAESTRUCTURA "
        "INTERMODAL AMBA"
    )

    print(
        f"Proyecto : {PROJECT_ROOT}"
    )

    print(
        f"Salida   : {OUTPUT_DIR}"
    )

    print(
        f"BBOX     : {AMBA_BBOX}"
    )

    print(
        f"CRS      : {CRS_WGS84}"
    )

    print(
        f"CRS métrico: {CRS_METRICO}"
    )

    # ==============================================================
    # 1
    # ==============================================================

    titulo(
        "1. CONSULTANDO OPENSTREETMAP / OVERPASS"
    )

    datos = consultar_overpass()

    # ==============================================================
    # 2
    # ==============================================================

    titulo(
        "2. CONSTRUYENDO INVENTARIO"
    )

    infraestructura = construir_gdf(
        datos
    )

    print(
        f"Elementos construidos: "
        f"{len(infraestructura):,}"
    )

    infraestructura = (
        normalizar_infraestructura(
            infraestructura
        )
    )

    validar_infraestructura(
        infraestructura,
        "INICIAL",
    )

    # ==============================================================
    # 3
    # ==============================================================

    titulo(
        "3. ELIMINANDO DUPLICADOS OSM"
    )

    infraestructura = (
        eliminar_duplicados_osm(
            infraestructura
        )
    )

    validar_infraestructura(
        infraestructura,
        "POST OSM",
    )

    # ==============================================================
    # 4
    # ==============================================================

    titulo(
        "4. DEDUPLICACIÓN ESPACIAL"
    )

    antes = len(
        infraestructura
    )

    infraestructura = (
        deduplicar_espacialmente(
            infraestructura
        )
    )

    despues = len(
        infraestructura
    )

    print(
        f"Antes     : {antes:,}"
    )

    print(
        f"Después   : {despues:,}"
    )

    print(
        f"Reducidos : {antes - despues:,}"
    )

    # ==============================================================
    # 5
    # ==============================================================

    titulo(
        "5. DETECTANDO INTERCAMBIADORES"
    )

    infraestructura = (
        detectar_intercambiadores(
            infraestructura
        )
    )

    print(
        "Intercambiadores potenciales:",
        int(
            infraestructura[
                "es_intercambiador"
            ].sum()
        ),
    )

    # ==============================================================
    # 6
    # ==============================================================

    titulo(
        "6. CALCULANDO INDICADORES"
    )

    infraestructura = (
        calcular_indicadores(
            infraestructura
        )
    )

    validar_infraestructura(
        infraestructura,
        "FINAL",
    )

    # ==============================================================
    # 7
    # ==============================================================

    imprimir_resumen_inventario(
        infraestructura
    )

    # ==============================================================
    # 8
    # ==============================================================

    titulo(
        "7. CARGANDO CENTRALIDADES SUBE"
    )

    centralidades = (
        cargar_centralidades()
    )

    if centralidades is not None:

        print(
            f"Centralidades cargadas: "
            f"{len(centralidades):,}"
        )

        titulo(
            "8. CALCULANDO INTERMODALIDAD "
            "DE CENTRALIDADES"
        )

        centralidades = (
            analizar_centralidades(
                centralidades,
                infraestructura,
            )
        )

        print()
        print(
            "TOP 15 CENTRALIDADES "
            "POR INTERMODALIDAD"
        )

        columnas = [

            "nodo_id",

            "score_intermodalidad_500m",

            "ranking_intermodalidad",

            "infra_250m",

            "infra_500m",

            "infra_estructural_500m",

            "intercambiadores_500m",

            "infra_1000m",

            "modos_500m",

            "ferrocarril_500m",

            "subte_500m",

            "autobus_500m",

            "fluvial_500m",

            "categoria_intermodalidad_500m",
        ]

        disponibles = [
            columna
            for columna in columnas
            if columna
            in centralidades.columns
        ]

        print(
            centralidades[
                disponibles
            ]
            .sort_values(
                "score_intermodalidad_500m",
                ascending=False,
            )
            .head(15)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No se realizará el cruce."
        )

    # ==============================================================
    # 9
    # ==============================================================

    titulo(
        "9. CONSTRUYENDO RESUMEN JSON"
    )

    resumen = construir_resumen(
        infraestructura,
        centralidades,
    )

    # ==============================================================
    # 10
    # ==============================================================

    titulo(
        "10. GUARDANDO ARCHIVOS"
    )

    guardar_infraestructura(
        infraestructura
    )

    guardar_csv(
        infraestructura
    )

    if centralidades is not None:

        guardar_centralidades(
            centralidades
        )

    guardar_resumen(
        resumen
    )

    # ==============================================================
    # 11
    # ==============================================================

    titulo(
        "11. GENERANDO MAPAS Y GRÁFICOS"
    )

    generar_mapa_infraestructura(
        infraestructura
    )

    generar_grafico_modos(
        infraestructura
    )

    generar_grafico_jerarquia(
        infraestructura
    )

    if centralidades is not None:

        generar_mapa_centralidades(
            centralidades
        )

        generar_grafico_intermodalidad(
            centralidades
        )

    # ==============================================================
    # 12
    # ==============================================================

    titulo(
        "21 - PROCESO FINALIZADO"
    )

    print(
        f"Infraestructuras finales: "
        f"{len(infraestructura):,}"
    )

    if centralidades is not None:

        print(
            f"Centralidades analizadas: "
            f"{len(centralidades):,}"
        )

    print()

    print(
        "ARCHIVOS GENERADOS"
    )

    if OUTPUT_DIR.exists():

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
        "Cruzar:"
    )

    print(
        "  SUBE 2025"
    )

    print(
        "  + centralidades"
    )

    print(
        "  + infraestructura intermodal"
    )

    print(
        "  + red vial"
    )

    print(
        "para construir el índice "
        "de centralidad estructural."
    )


# ======================================================================
# EJECUCIÓN
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nProceso cancelado por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "=" * 78
        )

        print(
            "ERROR FATAL"
        )

        print(
            "=" * 78
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        print(
            "Revisá el paso anterior "
            "para identificar el origen del error."
        )

        sys.exit(1)