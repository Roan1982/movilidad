# -*- coding: utf-8 -*-
"""
21_construir_infraestructura_intermodal_amba.py

Construcción de inventario de infraestructura de transporte
intermodal del Área Metropolitana de Buenos Aires (AMBA).

Fuentes:
    - OpenStreetMap
    - Overpass API
    - Centralidades SUBE 2025 previamente construidas

El script:

1. Consulta infraestructura de transporte en OSM.
2. Construye geometrías válidas.
3. Normaliza atributos.
4. Elimina duplicados OSM.
5. Realiza deduplicación espacial conservadora.
6. Calcula indicadores por infraestructura.
7. Asocia infraestructura a las centralidades SUBE.
8. Calcula indicadores de intermodalidad a 250, 500 y 1000 m.
9. Genera GeoParquet.
10. Genera GeoPackage.
11. Genera CSV.
12. Genera JSON de resumen.
13. Genera mapas de control.

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
from shapely.ops import transform


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

RAW_OVERPASS_PATH = (
    OUTPUT_DIR
    / "overpass_infraestructura_intermodal_amba.json"
)

CRS_WGS84 = "EPSG:4326"

# Gauss-Krüger / Argentina 5
CRS_METRICO = "EPSG:22185"

# BBOX:
# sur, oeste, norte, este
AMBA_BBOX = "-35.20,-59.40,-34.20,-57.80"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

RADIOS_INTERMODALIDAD_M = (
    250,
    500,
    1000,
)

TOLERANCIA_DEDUPLICACION_M = 75


# ======================================================================
# CONSTANTES DE CLASIFICACIÓN
# ======================================================================

MODOS_VALIDOS = (
    "FERROCARRIL",
    "SUBTE",
    "AUTOBUS",
    "TRANVIA",
    "FLUVIAL",
    "MULTIMODAL",
    "OTRO",
)

CATEGORIAS_VALIDAS = (
    "FERROVIARIO",
    "AUTOMOTOR",
    "FLUVIAL",
    "INTERMODAL",
    "OTRO",
)


# ======================================================================
# UTILIDADES GENERALES
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


def normalizar_texto(valor: Any) -> str:
    """
    Normaliza texto para comparación.

    Ejemplo:
        "Estación Constitución"
        -> "estacion constitucion"
    """

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

    return texto


def texto_o_none(valor: Any) -> str | None:
    if valor is None:
        return None

    texto = str(valor).strip()

    if not texto:
        return None

    return texto


def safe_int(valor: Any) -> int | None:
    try:
        if valor is None:
            return None

        if pd.isna(valor):
            return None

        return int(valor)

    except Exception:
        return None


def safe_float(valor: Any) -> float | None:
    try:
        if valor is None:
            return None

        if pd.isna(valor):
            return None

        return float(valor)

    except Exception:
        return None


def asegurar_directorio_salida() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ======================================================================
# OVERPASS
# ======================================================================

def construir_query_overpass() -> str:
    """
    Construye la consulta Overpass.

    Se consulta infraestructura y no rutas de transporte.

    Incluye:

    FERROCARRIL
        railway=station
        railway=halt
        railway=stop
        railway=tram_stop
        railway=subway
        railway=light_rail

    TRANSPORTE PÚBLICO
        public_transport=station
        public_transport=stop_position
        public_transport=platform

    AUTOBÚS
        amenity=bus_station

    FLUVIAL
        amenity=ferry_terminal
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
);

out geom tags;
"""


def consultar_overpass() -> dict:
    """
    Ejecuta la consulta Overpass con varios endpoints y reintentos.
    """

    query = construir_query_overpass()

    ultimo_error: Exception | None = None

    for intento in range(
        1,
        MAX_RETRIES + 1,
    ):

        for endpoint in OVERPASS_URLS:

            print()
            print(
                f"Intento {intento}/{MAX_RETRIES}"
            )
            print(
                f"Endpoint: {endpoint}"
            )

            try:

                respuesta = requests.post(
                    endpoint,
                    data={
                        "data": query,
                    },
                    headers={
                        "User-Agent": (
                            "analisis-movilidad-amba/"
                            "21-infraestructura-intermodal"
                        ),
                        "Accept": "application/json",
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                print(
                    f"HTTP: {respuesta.status_code}"
                )

                respuesta.raise_for_status()

                datos = respuesta.json()

                elementos = datos.get(
                    "elements",
                    [],
                )

                if not isinstance(
                    elementos,
                    list,
                ):
                    raise RuntimeError(
                        "La respuesta de Overpass tiene "
                        "un formato inválido."
                    )

                print(
                    f"Elementos recibidos: "
                    f"{len(elementos):,}"
                )

                return datos

            except Exception as exc:

                ultimo_error = exc

                print(
                    f"Error: {type(exc).__name__}: {exc}"
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        "No fue posible consultar Overpass.\n"
        f"Último error: {ultimo_error}"
    )


def guardar_respuesta_overpass(
    datos: dict,
) -> None:

    asegurar_directorio_salida()

    with open(
        RAW_OVERPASS_PATH,
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
        f"Respuesta cruda guardada:\n"
        f"{RAW_OVERPASS_PATH}"
    )


# ======================================================================
# CLASIFICACIÓN OSM
# ======================================================================

def clasificar_elemento(
    tags: dict[str, Any],
) -> tuple[str, str, str]:
    """
    Clasifica un elemento OSM.

    Retorna:

        tipo_infraestructura
        modo_principal
        categoria_intermodal
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

    # --------------------------------------------------------------
    # FERROCARRIL
    # --------------------------------------------------------------

    if railway == "station":
        return (
            "ESTACION_FERROVIARIA",
            "FERROCARRIL",
            "FERROVIARIO",
        )

    if railway == "halt":
        return (
            "PARADA_FERROVIARIA",
            "FERROCARRIL",
            "FERROVIARIO",
        )

    if railway == "stop":
        return (
            "PARADA_FERROVIARIA",
            "FERROCARRIL",
            "FERROVIARIO",
        )

    # --------------------------------------------------------------
    # SUBTE
    # --------------------------------------------------------------

    if railway == "subway":
        return (
            "ESTACION_SUBTE",
            "SUBTE",
            "FERROVIARIO",
        )

    # --------------------------------------------------------------
    # TRANVÍA
    # --------------------------------------------------------------

    if railway == "tram_stop":
        return (
            "PARADA_TRANVIA",
            "TRANVIA",
            "FERROVIARIO",
        )

    # --------------------------------------------------------------
    # LIGHT RAIL
    # --------------------------------------------------------------

    if railway == "light_rail":
        return (
            "ESTACION_LIGERO",
            "FERROCARRIL",
            "FERROVIARIO",
        )

    # --------------------------------------------------------------
    # AUTOBÚS
    # --------------------------------------------------------------

    if amenity == "bus_station":
        return (
            "TERMINAL_AUTOBUS",
            "AUTOBUS",
            "AUTOMOTOR",
        )

    # --------------------------------------------------------------
    # FLUVIAL
    # --------------------------------------------------------------

    if amenity == "ferry_terminal":
        return (
            "TERMINAL_FLUVIAL",
            "FLUVIAL",
            "FLUVIAL",
        )

    # --------------------------------------------------------------
    # TRANSPORTE PÚBLICO
    # --------------------------------------------------------------

    if public_transport == "station":

        # Una station genérica puede tener información de
        # otros tags que permitan identificar su modo.
        if railway:
            return (
                "ESTACION_TRANSPORTE_PUBLICO",
                "MULTIMODAL",
                "INTERMODAL",
            )

        return (
            "ESTACION_TRANSPORTE_PUBLICO",
            "MULTIMODAL",
            "INTERMODAL",
        )

    if public_transport == "stop_position":

        # Si no hay modo explícito, no inventamos que es colectivo.
        return (
            "PARADA_TRANSPORTE_PUBLICO",
            "OTRO",
            "OTRO",
        )

    if public_transport == "platform":

        return (
            "PLATAFORMA_TRANSPORTE_PUBLICO",
            "OTRO",
            "OTRO",
        )

    return (
        "OTRA_INFRAESTRUCTURA",
        "OTRO",
        "OTRO",
    )


# ======================================================================
# ATRIBUTOS OSM
# ======================================================================

def extraer_nombre(
    tags: dict[str, Any],
) -> str | None:

    campos = (
        "name",
        "official_name",
        "short_name",
        "alt_name",
    )

    for campo in campos:

        valor = texto_o_none(
            tags.get(campo)
        )

        if valor:
            return valor

    return None


def extraer_operador(
    tags: dict[str, Any],
) -> str | None:

    campos = (
        "operator",
        "network",
        "brand",
    )

    for campo in campos:

        valor = texto_o_none(
            tags.get(campo)
        )

        if valor:
            return valor

    return None


def extraer_capacidad(
    tags: dict[str, Any],
) -> float | None:

    for campo in (
        "capacity",
        "capacity:persons",
    ):

        valor = safe_float(
            tags.get(campo)
        )

        if valor is not None:
            return valor

    return None


# ======================================================================
# GEOMETRÍA
# ======================================================================

def geometry_from_element(
    element: dict[str, Any],
):
    """
    Construye una geometría representativa.

    Node:
        Point

    Way:
        Si posee geometry, se utiliza el centroide.

    La infraestructura se trabaja como puntos representativos
    porque posteriormente se calculan radios de influencia.
    """

    tipo = element.get("type")

    # --------------------------------------------------------------
    # NODE
    # --------------------------------------------------------------

    if tipo == "node":

        lat = element.get("lat")
        lon = element.get("lon")

        if lat is None or lon is None:
            return None

        try:
            return Point(
                float(lon),
                float(lat),
            )
        except Exception:
            return None

    # --------------------------------------------------------------
    # WAY
    # --------------------------------------------------------------

    if tipo == "way":

        geometry = element.get(
            "geometry"
        )

        if isinstance(
            geometry,
            list,
        ) and geometry:

            puntos = []

            for punto in geometry:

                lat = punto.get("lat")
                lon = punto.get("lon")

                if lat is None or lon is None:
                    continue

                try:
                    puntos.append(
                        Point(
                            float(lon),
                            float(lat),
                        )
                    )
                except Exception:
                    continue

            if puntos:

                if len(puntos) == 1:
                    return puntos[0]

                try:

                    from shapely.geometry import LineString

                    linea = LineString(
                        [
                            (
                                p.x,
                                p.y,
                            )
                            for p in puntos
                        ]
                    )

                    return linea.centroid

                except Exception:
                    return None

        # Fallback al centro provisto por Overpass.
        center = element.get(
            "center"
        )

        if isinstance(
            center,
            dict,
        ):

            lat = center.get("lat")
            lon = center.get("lon")

            if lat is not None and lon is not None:

                try:
                    return Point(
                        float(lon),
                        float(lat),
                    )
                except Exception:
                    return None

    return None


# ======================================================================
# CONSTRUCCIÓN DEL INVENTARIO
# ======================================================================

def construir_gdf(
    datos: dict,
) -> gpd.GeoDataFrame:

    registros: list[dict[str, Any]] = []

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

        tipo, modo, categoria = (
            clasificar_elemento(tags)
        )

        osm_type = texto_o_none(
            element.get("type")
        )

        osm_id = safe_int(
            element.get("id")
        )

        if osm_type is None or osm_id is None:
            continue

        registro = {

            "osm_type": osm_type,

            "osm_id": osm_id,

            "osm_key": (
                f"{osm_type}/{osm_id}"
            ),

            "nombre": extraer_nombre(
                tags
            ),

            "nombre_normalizado": normalizar_texto(
                extraer_nombre(tags)
            ),

            "operador": extraer_operador(
                tags
            ),

            "tipo_infraestructura": tipo,

            "modo_principal": modo,

            "categoria_intermodal": categoria,

            "capacidad": extraer_capacidad(
                tags
            ),

            "railway": texto_o_none(
                tags.get("railway")
            ),

            "amenity": texto_o_none(
                tags.get("amenity")
            ),

            "public_transport": texto_o_none(
                tags.get("public_transport")
            ),

            "network": texto_o_none(
                tags.get("network")
            ),

            "brand": texto_o_none(
                tags.get("brand")
            ),

            "operator": texto_o_none(
                tags.get("operator")
            ),

            "ref": texto_o_none(
                tags.get("ref")
            ),

            "wikidata": texto_o_none(
                tags.get("wikidata")
            ),

            "website": texto_o_none(
                tags.get("website")
            ),

            "geometry": geometry,
        }

        registros.append(
            registro
        )

    if not registros:
        raise RuntimeError(
            "Overpass devolvió elementos, "
            "pero ninguno pudo convertirse "
            "en una geometría válida."
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

    # --------------------------------------------------------------
    # TIPOS
    # --------------------------------------------------------------

    gdf["osm_id"] = pd.to_numeric(
        gdf["osm_id"],
        errors="coerce",
    ).astype("Int64")

    gdf["osm_type"] = (
        gdf["osm_type"]
        .astype("string")
    )

    gdf["osm_key"] = (
        gdf["osm_key"]
        .astype("string")
    )

    # --------------------------------------------------------------
    # TEXTOS
    # --------------------------------------------------------------

    columnas_texto = [
        "nombre",
        "nombre_normalizado",
        "operador",
        "tipo_infraestructura",
        "modo_principal",
        "categoria_intermodal",
        "railway",
        "amenity",
        "public_transport",
        "network",
        "brand",
        "operator",
        "ref",
        "wikidata",
        "website",
    ]

    for columna in columnas_texto:

        if columna in gdf.columns:

            gdf[columna] = (
                gdf[columna]
                .astype("string")
            )

    # --------------------------------------------------------------
    # VALORES POR DEFECTO
    # --------------------------------------------------------------

    gdf["modo_principal"] = (
        gdf["modo_principal"]
        .fillna("OTRO")
    )

    gdf["categoria_intermodal"] = (
        gdf["categoria_intermodal"]
        .fillna("OTRO")
    )

    gdf["tipo_infraestructura"] = (
        gdf["tipo_infraestructura"]
        .fillna("OTRA_INFRAESTRUCTURA")
    )

    # --------------------------------------------------------------
    # CAPACIDAD
    # --------------------------------------------------------------

    gdf["capacidad"] = pd.to_numeric(
        gdf["capacidad"],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # ORDEN
    # --------------------------------------------------------------

    columnas_principales = [
        "osm_type",
        "osm_id",
        "osm_key",
        "nombre",
        "nombre_normalizado",
        "operador",
        "tipo_infraestructura",
        "modo_principal",
        "categoria_intermodal",
        "capacidad",
        "railway",
        "amenity",
        "public_transport",
        "network",
        "brand",
        "operator",
        "ref",
        "wikidata",
        "website",
        "geometry",
    ]

    existentes = [
        columna
        for columna in columnas_principales
        if columna in gdf.columns
    ]

    restantes = [
        columna
        for columna in gdf.columns
        if columna not in existentes
    ]

    gdf = gdf[
        existentes + restantes
    ]

    return gdf


# ======================================================================
# DEDUPLICACIÓN OSM
# ======================================================================

def eliminar_duplicados_osm(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    antes = len(gdf)

    gdf = gdf.copy()

    gdf = gdf.drop_duplicates(
        subset=[
            "osm_type",
            "osm_id",
        ],
        keep="first",
    )

    despues = len(gdf)

    print(
        f"Duplicados OSM eliminados: "
        f"{antes - despues:,}"
    )

    return gdf.reset_index(
        drop=True
    )


# ======================================================================
# DEDUPLICACIÓN ESPACIAL
# ======================================================================

def deduplicar_espacialmente(
    gdf: gpd.GeoDataFrame,
    tolerancia_m: float = TOLERANCIA_DEDUPLICACION_M,
) -> gpd.GeoDataFrame:
    """
    Deduplicación conservadora.

    Se agrupan elementos solamente cuando:

    1. Están dentro de la tolerancia espacial.
    2. Tienen el mismo modo.
    3. Tienen el mismo nombre normalizado.

    Si no tienen nombre, NO se fusionan.

    Esto evita destruir infraestructura distinta
    ubicada en el mismo entorno.
    """

    if gdf.empty:
        return gdf.copy()

    metric = (
        gdf
        .to_crs(CRS_METRICO)
        .copy()
    )

    metric["geometry"] = (
        metric.geometry
        .centroid
    )

    metric = metric.reset_index(
        drop=True
    )

    sindex = metric.sindex

    usados: set[int] = set()

    resultado: list[pd.Series] = []

    for idx in range(
        len(metric)
    ):

        if idx in usados:
            continue

        fila = metric.iloc[idx]

        candidatos = list(
            sindex.query(
                fila.geometry.buffer(
                    tolerancia_m
                ),
                predicate="intersects",
            )
        )

        grupo: list[int] = []

        for candidato in candidatos:

            if candidato in usados:
                continue

            otra = metric.iloc[
                candidato
            ]

            distancia = (
                fila.geometry.distance(
                    otra.geometry
                )
            )

            if distancia > tolerancia_m:
                continue

            mismo_modo = (
                str(
                    fila["modo_principal"]
                )
                ==
                str(
                    otra["modo_principal"]
                )
            )

            nombre_fila = normalizar_texto(
                fila.get(
                    "nombre"
                )
            )

            nombre_otra = normalizar_texto(
                otra.get(
                    "nombre"
                )
            )

            mismo_nombre = (
                nombre_fila != ""
                and
                nombre_fila == nombre_otra
            )

            if (
                mismo_modo
                and
                mismo_nombre
            ):
                grupo.append(
                    candidato
                )

        if not grupo:
            grupo = [idx]

        usados.update(
            grupo
        )

        base = metric.iloc[
            grupo[0]
        ].copy()

        osm_ids = []

        for posicion in grupo:

            osm_id = metric.iloc[
                posicion
            ]["osm_id"]

            if pd.notna(
                osm_id
            ):
                osm_ids.append(
                    str(
                        int(osm_id)
                    )
                )

        base["osm_ids"] = ",".join(
            sorted(
                set(osm_ids)
            )
        )

        base["cantidad_elementos_osm"] = (
            len(grupo)
        )

        resultado.append(
            base
        )

    salida = gpd.GeoDataFrame(
        resultado,
        geometry="geometry",
        crs=CRS_METRICO,
    )

    salida = salida.to_crs(
        CRS_WGS84
    )

    salida = salida.reset_index(
        drop=True
    )

    return salida


# ======================================================================
# INDICADORES DE INFRAESTRUCTURA
# ======================================================================

def calcular_indicadores(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    gdf = gdf.copy()

    gdf["es_ferrocarril"] = (
        gdf["modo_principal"]
        == "FERROCARRIL"
    ).astype("int8")

    gdf["es_subte"] = (
        gdf["modo_principal"]
        == "SUBTE"
    ).astype("int8")

    gdf["es_autobus"] = (
        gdf["modo_principal"]
        == "AUTOBUS"
    ).astype("int8")

    gdf["es_fluvial"] = (
        gdf["modo_principal"]
        == "FLUVIAL"
    ).astype("int8")

    gdf["es_tranvia"] = (
        gdf["modo_principal"]
        == "TRANVIA"
    ).astype("int8")

    gdf["es_multimodal"] = (
        gdf["modo_principal"]
        == "MULTIMODAL"
    ).astype("int8")

    # Puntaje descriptivo del elemento.
    pesos = {
        "FERROCARRIL": 1.0,
        "SUBTE": 1.0,
        "AUTOBUS": 0.8,
        "FLUVIAL": 1.0,
        "TRANVIA": 0.8,
        "MULTIMODAL": 1.2,
        "OTRO": 0.0,
    }

    gdf["score_infraestructura"] = (
        gdf["modo_principal"]
        .map(pesos)
        .fillna(0.0)
        .astype(float)
    )

    return gdf


# ======================================================================
# VALIDACIÓN
# ======================================================================

def validar_infraestructura(
    gdf: gpd.GeoDataFrame,
    etapa: str = "",
) -> None:

    subtitulo(
        f"VALIDACIÓN DE INFRAESTRUCTURA {etapa}"
    )

    if gdf.empty:
        raise RuntimeError(
            "El GeoDataFrame está vacío."
        )

    if gdf.crs is None:
        raise RuntimeError(
            "La infraestructura no tiene CRS."
        )

    if gdf.crs.to_string() != CRS_WGS84:
        raise RuntimeError(
            f"CRS inesperado: {gdf.crs}"
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

    print(
        f"Registros: "
        f"{len(gdf):,}"
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

    duplicados_osm = (
        gdf[
            ["osm_type", "osm_id"]
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicados OSM: "
        f"{int(duplicados_osm):,}"
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

    if duplicados_osm > 0:
        raise RuntimeError(
            "Existen identificadores OSM duplicados."
        )


# ======================================================================
# CENTRALIDADES
# ======================================================================

def cargar_centralidades() -> (
    gpd.GeoDataFrame | None
):

    if not CENTRALIDADES_PATH.exists():

        print()
        print(
            "ADVERTENCIA"
        )
        print(
            "No se encontró el archivo:"
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
            "No se pudieron cargar "
            f"las centralidades: {exc}"
        )

        return None

    if centralidades.empty:

        print(
            "El archivo de centralidades "
            "está vacío."
        )

        return None

    if centralidades.crs is None:

        print(
            "ADVERTENCIA: las centralidades "
            "no tienen CRS."
        )

        centralidades = (
            centralidades
            .set_crs(
                CRS_WGS84
            )
        )

    # --------------------------------------------------------------
    # nodo_id
    # --------------------------------------------------------------

    if "nodo_id" not in centralidades.columns:

        centralidades["nodo_id"] = (
            np.arange(
                1,
                len(centralidades) + 1,
            )
        )

    centralidades["nodo_id"] = (
        centralidades["nodo_id"]
        .astype("string")
    )

    # --------------------------------------------------------------
    # Geometría
    # --------------------------------------------------------------

    centralidades = (
        centralidades
        .to_crs(CRS_METRICO)
        .copy()
    )

    centralidades["geometry_centro"] = (
        centralidades.geometry
        .centroid
    )

    return centralidades


# ======================================================================
# ANÁLISIS DE CENTRALIDADES
# ======================================================================

def calcular_modos(
    sub: gpd.GeoDataFrame,
) -> set[str]:

    if sub.empty:
        return set()

    return set(
        sub["modo_principal"]
        .dropna()
        .astype(str)
    )


def agregar_indicadores_radio(
    registro: dict[str, Any],
    sub: gpd.GeoDataFrame,
    radio: int,
) -> None:

    sufijo = f"{radio}m"

    modos = calcular_modos(
        sub
    )

    registro[
        f"infra_{sufijo}"
    ] = int(
        len(sub)
    )

    registro[
        f"modos_{sufijo}"
    ] = int(
        len(modos)
    )

    registro[
        f"ferrocarril_{sufijo}"
    ] = int(
        "FERROCARRIL" in modos
    )

    registro[
        f"subte_{sufijo}"
    ] = int(
        "SUBTE" in modos
    )

    registro[
        f"autobus_{sufijo}"
    ] = int(
        "AUTOBUS" in modos
    )

    registro[
        f"fluvial_{sufijo}"
    ] = int(
        "FLUVIAL" in modos
    )

    registro[
        f"tranvia_{sufijo}"
    ] = int(
        "TRANVIA" in modos
    )

    registro[
        f"multimodal_{sufijo}"
    ] = int(
        "MULTIMODAL" in modos
    )


def calcular_score_intermodalidad(
    registro: dict[str, Any],
) -> float:

    modos = registro[
        "modos_500m"
    ]

    infraestructura = registro[
        "infra_500m"
    ]

    presencia = (
        registro["ferrocarril_500m"]
        + registro["subte_500m"]
        + registro["autobus_500m"]
        + registro["fluvial_500m"]
        + registro["tranvia_500m"]
    )

    # --------------------------------------------------------------
    # Componentes:
    #
    # diversidad de modos = 60 puntos
    # cantidad infraestructura = 20 puntos
    # presencia de modos = 20 puntos
    # --------------------------------------------------------------

    componente_modos = (
        min(
            modos,
            4,
        )
        / 4.0
        * 60.0
    )

    componente_infra = (
        min(
            infraestructura,
            10,
        )
        / 10.0
        * 20.0
    )

    componente_presencia = (
        min(
            presencia,
            4,
        )
        / 4.0
        * 20.0
    )

    return round(
        min(
            100.0,
            (
                componente_modos
                + componente_infra
                + componente_presencia
            ),
        ),
        3,
    )


def categorizar_intermodalidad(
    score: float,
    modos: int,
) -> str:

    if modos == 0:
        return "SIN_INFRAESTRUCTURA_500M"

    if score >= 75:
        return "INTERMODALIDAD_MUY_ALTA"

    if score >= 50:
        return "INTERMODALIDAD_ALTA"

    if score >= 25:
        return "INTERMODALIDAD_MEDIA"

    return "INTERMODALIDAD_BAJA"


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

    # --------------------------------------------------------------
    # Usamos puntos representativos de la infraestructura.
    # --------------------------------------------------------------

    infra["geometry"] = (
        infra.geometry
        .centroid
    )

    sindex = infra.sindex

    resultados = []

    total = len(centros)

    for posicion, (
        idx,
        nodo,
    ) in enumerate(
        centros.iterrows(),
        start=1,
    ):

        if (
            posicion == 1
            or posicion % 25 == 0
            or posicion == total
        ):

            print(
                f"Centralidad "
                f"{posicion}/{total}"
            )

        # ----------------------------------------------------------
        # Centroide H3 / centralidad
        # ----------------------------------------------------------

        punto = nodo[
            "geometry_centro"
        ]

        if punto is None:
            continue

        registro = {
            "nodo_id": str(
                nodo["nodo_id"]
            )
        }

        # ----------------------------------------------------------
        # Radios
        # ----------------------------------------------------------

        for radio in RADIOS_INTERMODALIDAD_M:

            candidatos = list(
                sindex.query(
                    punto.buffer(
                        radio
                    ),
                    predicate="intersects",
                )
            )

            if candidatos:

                sub = infra.iloc[
                    candidatos
                ]

                # Distancia exacta al punto central.
                distancias = (
                    sub.geometry.distance(
                        punto
                    )
                )

                sub = sub.copy()

                sub[
                    "_distancia_m"
                ] = distancias.values

                sub = sub[
                    sub["_distancia_m"]
                    <= radio
                ]

            else:

                sub = infra.iloc[
                    0:0
                ]

            agregar_indicadores_radio(
                registro,
                sub,
                radio,
            )

        # ----------------------------------------------------------
        # Infraestructura más cercana
        # ----------------------------------------------------------

        candidatos = list(
            sindex.query(
                punto.buffer(
                    1000
                ),
                predicate="intersects",
            )
        )

        if candidatos:

            sub = infra.iloc[
                candidatos
            ].copy()

            sub[
                "_distancia_m"
            ] = sub.geometry.distance(
                punto
            )

            sub = sub[
                sub["_distancia_m"]
                <= 1000
            ]

            if not sub.empty:

                sub = sub.sort_values(
                    "_distancia_m"
                )

                mas_cercana = sub.iloc[
                    0
                ]

                registro[
                    "distancia_infraestructura_m"
                ] = round(
                    float(
                        mas_cercana[
                            "_distancia_m"
                        ]
                    ),
                    2,
                )

                nombre = mas_cercana[
                    "nombre"
                ]

                if (
                    pd.notna(nombre)
                    and str(nombre).strip()
                ):

                    registro[
                        "infraestructura_mas_cercana"
                    ] = str(
                        nombre
                    )

                else:

                    registro[
                        "infraestructura_mas_cercana"
                    ] = str(
                        mas_cercana[
                            "tipo_infraestructura"
                        ]
                    )

                registro[
                    "modo_infraestructura_mas_cercana"
                ] = str(
                    mas_cercana[
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
        # Score
        # ----------------------------------------------------------

        score = calcular_score_intermodalidad(
            registro
        )

        registro[
            "score_intermodalidad_500m"
        ] = score

        registro[
            "categoria_intermodalidad_500m"
        ] = categorizar_intermodalidad(
            score,
            registro[
                "modos_500m"
            ],
        )

        resultados.append(
            registro
        )

    indicadores = pd.DataFrame(
        resultados
    )

    if indicadores.empty:

        raise RuntimeError(
            "No fue posible calcular "
            "indicadores para las centralidades."
        )

    indicadores["nodo_id"] = (
        indicadores["nodo_id"]
        .astype("string")
    )

    centros["nodo_id"] = (
        centros["nodo_id"]
        .astype("string")
    )

    centros = centros.merge(
        indicadores,
        on="nodo_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------------
    # Ranking
    # --------------------------------------------------------------

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
) -> dict[str, Any]:

    resumen: dict[str, Any] = {

        "proyecto": (
            "Análisis de movilidad SUBE AMBA"
        ),

        "script": (
            "21_construir_infraestructura_"
            "intermodal_amba.py"
        ),

        "fecha_ejecucion": (
            pd.Timestamp.now(
                tz="America/Argentina/Buenos_Aires"
            ).isoformat()
        ),

        "bbox": AMBA_BBOX,

        "crs": CRS_WGS84,

        "crs_metrico": CRS_METRICO,

        "infraestructuras": int(
            len(infraestructura)
        ),

        "osm_ids_unicos": int(
            infraestructura[
                "osm_id"
            ].nunique()
        ),

        "tipos": {},

        "modos": {},

        "categorias": {},
    }

    for columna, clave in [
        (
            "tipo_infraestructura",
            "tipos",
        ),
        (
            "modo_principal",
            "modos",
        ),
        (
            "categoria_intermodal",
            "categorias",
        ),
    ]:

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

    # --------------------------------------------------------------
    # Centralidades
    # --------------------------------------------------------------

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
                .fillna("SIN_DATO")
                .value_counts()
                .to_dict()
            )

            resumen[
                "centralidades_por_intermodalidad"
            ] = {
                str(k): int(v)
                for k, v in conteo.items()
            }

        if (
            "score_intermodalidad_500m"
            in centralidades.columns
        ):

            top = (
                centralidades
                .sort_values(
                    "score_intermodalidad_500m",
                    ascending=False,
                )
                .head(10)
            )

            resumen[
                "top_10_centralidades_intermodalidad"
            ] = []

            for _, fila in top.iterrows():

                resumen[
                    "top_10_centralidades_intermodalidad"
                ].append(
                    {
                        "nodo_id": str(
                            fila[
                                "nodo_id"
                            ]
                        ),

                        "score": safe_float(
                            fila[
                                "score_intermodalidad_500m"
                            ]
                        ),

                        "infra_500m": safe_int(
                            fila[
                                "infra_500m"
                            ]
                        ),

                        "modos_500m": safe_int(
                            fila[
                                "modos_500m"
                            ]
                        ),

                        "categoria": str(
                            fila[
                                "categoria_intermodalidad_500m"
                            ]
                        ),
                    }
                )

    return resumen


# ======================================================================
# SALIDAS
# ======================================================================

def guardar_infraestructura(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    asegurar_directorio_salida()

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

    gpkg_path = (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.gpkg"
    )

    # No necesitamos guardar geometry_centro
    # como segunda geometría en GeoPackage.
    salida = centralidades.copy()

    geometry_centro = None

    if "geometry_centro" in salida.columns:

        geometry_centro = (
            salida["geometry_centro"]
            .copy()
        )

        salida = salida.drop(
            columns=[
                "geometry_centro"
            ]
        )

    salida.to_parquet(
        parquet_path,
        index=False,
    )

    salida.to_file(
        gpkg_path,
        layer="centralidades_intermodalidad",
        driver="GPKG",
    )

    print(
        "Centralidades guardadas:"
    )

    print(
        parquet_path
    )


def guardar_csv(
    infraestructura: gpd.GeoDataFrame,
) -> None:

    tabla = infraestructura.drop(
        columns=[
            "geometry"
        ],
        errors="ignore",
    ).copy()

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
    resumen: dict[str, Any],
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
            "Matplotlib no disponible. "
            "Se omite mapa."
        )

        return

    if infraestructura.empty:
        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    infraestructura.plot(
        ax=ax,
        column="modo_principal",
        markersize=8,
        alpha=0.7,
        legend=True,
    )

    ax.set_title(
        "Infraestructura de transporte "
        "intermodal - AMBA",
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

    if (
        centralidades is None
        or centralidades.empty
    ):
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

    if conteo.empty:
        return

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    conteo.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Infraestructura por modo de transporte",
        fontsize=14,
    )

    ax.set_xlabel(
        "Cantidad de elementos"
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


def generar_grafico_intermodalidad(
    centralidades: gpd.GeoDataFrame,
) -> None:

    if (
        centralidades is None
        or centralidades.empty
    ):
        return

    columna = (
        "score_intermodalidad_500m"
    )

    if columna not in centralidades.columns:
        return

    try:

        import matplotlib.pyplot as plt

    except ImportError:
        return

    valores = (
        pd.to_numeric(
            centralidades[
                columna
            ],
            errors="coerce",
        )
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
        "Distribución del score "
        "de intermodalidad",
        fontsize=14,
    )

    ax.set_xlabel(
        "Score de intermodalidad"
    )

    ax.set_ylabel(
        "Cantidad de centralidades"
    )

    path = (
        OUTPUT_DIR
        / "04_distribucion_score_intermodalidad.png"
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
# RESUMEN POR CONSOLA
# ======================================================================

def imprimir_resumen_infraestructura(
    gdf: gpd.GeoDataFrame,
) -> None:

    subtitulo(
        "RESUMEN DEL INVENTARIO"
    )

    print(
        f"Total: {len(gdf):,}"
    )

    print()
    print(
        "POR MODO"
    )

    for modo, cantidad in (
        gdf[
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
        "POR TIPO"
    )

    for tipo, cantidad in (
        gdf[
            "tipo_infraestructura"
        ]
        .value_counts()
        .items()
    ):

        print(
            f"  {str(tipo):35s}"
            f"{int(cantidad):8,d}"
        )


# ======================================================================
# VALIDACIÓN DE CENTRALIDADES
# ======================================================================

def validar_centralidades(
    centralidades: gpd.GeoDataFrame,
) -> None:

    subtitulo(
        "VALIDACIÓN DE CENTRALIDADES"
    )

    if centralidades.empty:
        raise RuntimeError(
            "Las centralidades están vacías."
        )

    print(
        f"Centralidades: "
        f"{len(centralidades):,}"
    )

    if "nodo_id" not in centralidades.columns:

        raise RuntimeError(
            "Las centralidades no tienen nodo_id."
        )

    duplicados = (
        centralidades[
            "nodo_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"nodo_id duplicados: "
        f"{int(duplicados):,}"
    )

    if duplicados:
        raise RuntimeError(
            "Existen nodo_id duplicados."
        )

    if (
        "geometry_centro"
        not in centralidades.columns
    ):

        raise RuntimeError(
            "No se pudo construir "
            "geometry_centro."
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
    # 0. PREPARACIÓN
    # ==============================================================

    asegurar_directorio_salida()

    # ==============================================================
    # 1. OVERPASS
    # ==============================================================

    titulo(
        "1. CONSULTANDO OPENSTREETMAP / OVERPASS"
    )

    datos = consultar_overpass()

    guardar_respuesta_overpass(
        datos
    )

    # ==============================================================
    # 2. CONSTRUCCIÓN
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
    # 3. DEDUPLICACIÓN OSM
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
    # 4. DEDUPLICACIÓN ESPACIAL
    # ==============================================================

    titulo(
        "4. DEDUPLICACIÓN ESPACIAL"
    )

    cantidad_antes = len(
        infraestructura
    )

    infraestructura = (
        deduplicar_espacialmente(
            infraestructura,
            TOLERANCIA_DEDUPLICACION_M,
        )
    )

    cantidad_despues = len(
        infraestructura
    )

    print(
        f"Antes     : {cantidad_antes:,}"
    )

    print(
        f"Después   : {cantidad_despues:,}"
    )

    print(
        f"Reducidos: "
        f"{cantidad_antes - cantidad_despues:,}"
    )

    # ==============================================================
    # 5. INDICADORES
    # ==============================================================

    titulo(
        "5. CALCULANDO INDICADORES"
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

    imprimir_resumen_infraestructura(
        infraestructura
    )

    # ==============================================================
    # 6. CENTRALIDADES
    # ==============================================================

    titulo(
        "6. CARGANDO CENTRALIDADES SUBE"
    )

    centralidades = (
        cargar_centralidades()
    )

    if centralidades is not None:

        validar_centralidades(
            centralidades
        )

        print(
            f"Centralidades cargadas: "
            f"{len(centralidades):,}"
        )

        titulo(
            "7. CALCULANDO INTERMODALIDAD "
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
            if columna in centralidades.columns
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
            "No se realizó el análisis "
            "de centralidades."
        )

    # ==============================================================
    # 8. RESUMEN
    # ==============================================================

    titulo(
        "8. CONSTRUYENDO RESUMEN JSON"
    )

    resumen = construir_resumen(
        infraestructura,
        centralidades,
    )

    # ==============================================================
    # 9. ARCHIVOS
    # ==============================================================

    titulo(
        "9. GUARDANDO ARCHIVOS"
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
    # 10. MAPAS
    # ==============================================================

    titulo(
        "10. GENERANDO MAPAS Y GRÁFICOS"
    )

    generar_mapa_infraestructura(
        infraestructura
    )

    generar_grafico_modos(
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
    # 11. FINAL
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
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Proceso cancelado "
            "por el usuario."
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
            "El proceso fue detenido."
        )

        sys.exit(1)