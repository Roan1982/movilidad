# -*- coding: utf-8 -*-
"""
21_construir_infraestructura_intermodal_amba.py
VERSION 3.0 - CONSOLIDACIÓN FÍSICA

Construye un inventario de infraestructura de transporte del AMBA a partir
de OpenStreetMap / Overpass.

Esta versión separa explícitamente:
    1) objetos OSM
    2) instalaciones físicas
    3) intercambiadores intermodales
    4) centralidades SUBE

Objetivo metodológico:
no contar cada platform / stop_position como una infraestructura
independiente. Los objetos OSM se consolidan en instalaciones físicas
antes de calcular densidad e intermodalidad.

No requiere scipy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from shapely.ops import unary_union


# ============================================================================
# CONFIGURACION
# ============================================================================

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

OVERPASS_CACHE = (
    OUTPUT_DIR / "overpass_infraestructura_intermodal_amba.json"
)

AMBA_BBOX = "-35.20,-59.40,-34.20,-57.80"

CRS_WGS84 = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3

RADIOS_INTERMODALIDAD_M = [250, 500, 1000]

# Consolidacion de objetos OSM.
RADIO_INSTALACION_NOMBRADA_M = 100
RADIO_INSTALACION_SIN_NOMBRE_M = 60

# Elementos auxiliares (platform / stop_position) se absorben dentro
# de instalaciones existentes si estan suficientemente cerca.
RADIO_ADHESION_A_INSTALACION_M = 100

# Intercambiador: dos o mas instalaciones de modos distintos.
RADIO_INTERCAMBIADOR_M = 150
MIN_MODOS_INTERCAMBIADOR = 2

MODOS_VALIDOS = {
    "FERROCARRIL",
    "SUBTE",
    "AUTOBUS",
    "FLUVIAL",
    "TRANVIA",
}

TIPOS_ANCLA = {
    "ESTACION_FERROVIARIA",
    "PARADA_FERROVIARIA",
    "ESTACION_SUBTE",
    "PARADA_TRANVIA",
    "ESTACION_LIGERO",
    "TERMINAL_AUTOBUS",
    "TERMINAL_FLUVIAL",
    "ESTACION_TRANSPORTE_PUBLICO",
    "RUTA_FLUVIAL",
}

TIPOS_AUXILIARES = {
    "PLATAFORMA_TRANSPORTE_PUBLICO",
    "PARADA_TRANSPORTE_PUBLICO",
}

# Pesos metodologicos: la diversidad modal domina sobre la cantidad bruta
# de objetos OSM.
PESO_MODO = {
    "FERROCARRIL": 25.0,
    "SUBTE": 25.0,
    "AUTOBUS": 15.0,
    "FLUVIAL": 25.0,
    "TRANVIA": 15.0,
}


# ============================================================================
# UTILIDADES
# ============================================================================

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


def resetear_indices(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.copy().reset_index(drop=True)


def safe_int(valor: Any) -> int | None:
    try:
        if valor is None or pd.isna(valor):
            return None
        return int(valor)
    except Exception:
        return None


def safe_float(valor: Any) -> float | None:
    try:
        if valor is None or pd.isna(valor):
            return None
        return float(valor)
    except Exception:
        return None


def normalizar_texto(valor: Any) -> str:
    if valor is None:
        return ""

    try:
        if pd.isna(valor):
            return ""
    except Exception:
        pass

    texto = str(valor).strip().lower()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)

    return texto


def nombres_unicos(serie: pd.Series, limite: int = 12) -> str:
    salida = []
    vistos = set()

    for valor in serie.dropna().astype(str):
        valor = valor.strip()
        clave = normalizar_texto(valor)

        if not clave or clave in vistos:
            continue

        vistos.add(clave)
        salida.append(valor)

        if len(salida) >= limite:
            break

    return " | ".join(salida)


def consultar_vecinos(sindex, geometry, radio_m: float) -> list[int]:
    return [int(x) for x in sindex.query(
        geometry.buffer(radio_m),
        predicate="intersects",
    )]


# ============================================================================
# OVERPASS
# ============================================================================

def construir_query_overpass() -> str:
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
  node["public_transport"="platform"]({AMBA_BBOX});

  way["route"="ferry"]({AMBA_BBOX});
);
out center tags;
"""


def guardar_cache_overpass(datos: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OVERPASS_CACHE.open("w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False)


def cargar_cache_overpass() -> dict | None:
    if not OVERPASS_CACHE.exists():
        return None

    try:
        with OVERPASS_CACHE.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)

        if isinstance(datos, dict) and "elements" in datos:
            return datos
    except Exception as exc:
        print(f"ADVERTENCIA: cache inválida: {exc}")

    return None


def consultar_overpass() -> dict:
    query = construir_query_overpass()
    ultimo_error: Exception | None = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for intento in range(1, MAX_RETRIES + 1):
        print(f"Intento {intento}/{MAX_RETRIES}")

        for endpoint in OVERPASS_URLS:
            print(f"Endpoint: {endpoint}")

            try:
                respuesta = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={
                        "User-Agent": "analisis-movilidad-amba/3.0",
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                print(f"HTTP: {respuesta.status_code}")
                respuesta.raise_for_status()

                datos = respuesta.json()

                if "elements" not in datos:
                    raise RuntimeError(
                        "La respuesta de Overpass no contiene 'elements'."
                    )

                print(
                    f"Elementos recibidos: "
                    f"{len(datos['elements']):,}"
                )

                guardar_cache_overpass(datos)
                print(f"Respuesta cruda guardada:\n{OVERPASS_CACHE}")

                return datos

            except Exception as exc:
                ultimo_error = exc
                print(
                    f"Error: {type(exc).__name__}: {exc}"
                )
                time.sleep(2)

    cache = cargar_cache_overpass()

    if cache is not None:
        print()
        print(
            "ADVERTENCIA: Overpass no respondió correctamente."
        )
        print(
            "Se utilizará la última respuesta cacheada válida."
        )
        print(
            f"Elementos cacheados: "
            f"{len(cache.get('elements', [])):,}"
        )
        return cache

    raise RuntimeError(
        "No fue posible consultar Overpass y no existe cache válida. "
        f"Último error: {ultimo_error}"
    )


# ============================================================================
# CLASIFICACION OSM
# ============================================================================

def clasificar_elemento(
    tags: dict,
) -> tuple[str, str, str]:
    railway = normalizar_texto(tags.get("railway"))
    amenity = normalizar_texto(tags.get("amenity"))
    public_transport = normalizar_texto(
        tags.get("public_transport")
    )
    route = normalizar_texto(tags.get("route"))

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

    if railway == "subway":
        return (
            "ESTACION_SUBTE",
            "SUBTE",
            "FERROVIARIO",
        )

    if railway == "tram_stop":
        return (
            "PARADA_TRANVIA",
            "TRANVIA",
            "FERROVIARIO",
        )

    if railway == "light_rail":
        return (
            "ESTACION_LIGERO",
            "FERROCARRIL",
            "FERROVIARIO",
        )

    if railway == "stop":
        return (
            "PARADA_FERROVIARIA",
            "FERROCARRIL",
            "FERROVIARIO",
        )

    if amenity == "bus_station":
        return (
            "TERMINAL_AUTOBUS",
            "AUTOBUS",
            "AUTOMOTOR",
        )

    if amenity == "ferry_terminal":
        return (
            "TERMINAL_FLUVIAL",
            "FLUVIAL",
            "FLUVIAL",
        )

    if route == "ferry":
        return (
            "RUTA_FLUVIAL",
            "FLUVIAL",
            "FLUVIAL",
        )

    if public_transport == "station":
        return (
            "ESTACION_TRANSPORTE_PUBLICO",
            "MULTIMODAL",
            "INTERMODAL",
        )

    if public_transport == "stop_position":
        return (
            "PARADA_TRANSPORTE_PUBLICO",
            "AUTOBUS",
            "AUTOMOTOR",
        )

    if public_transport == "platform":
        return (
            "PLATAFORMA_TRANSPORTE_PUBLICO",
            "AUTOBUS",
            "AUTOMOTOR",
        )

    return (
        "OTRA_INFRAESTRUCTURA",
        "OTRO",
        "OTRO",
    )


def extraer_nombre(tags: dict) -> str | None:
    for campo in (
        "name",
        "official_name",
        "short_name",
        "alt_name",
    ):
        valor = tags.get(campo)

        if valor:
            return str(valor).strip()

    return None


def extraer_operador(tags: dict) -> str | None:
    for campo in (
        "operator",
        "network",
        "brand",
    ):
        valor = tags.get(campo)

        if valor:
            return str(valor).strip()

    return None


def geometry_from_element(element: dict) -> Point | None:
    if element.get("type") == "node":
        lat = element.get("lat")
        lon = element.get("lon")

        if lat is None or lon is None:
            return None

        return Point(float(lon), float(lat))

    center = element.get("center") or {}

    lat = center.get("lat")
    lon = center.get("lon")

    if lat is None or lon is None:
        return None

    return Point(float(lon), float(lat))


# ============================================================================
# CONSTRUCCION
# ============================================================================

def construir_gdf(datos: dict) -> gpd.GeoDataFrame:
    registros = []

    for element in datos.get("elements", []):
        tags = element.get("tags") or {}
        geometry = geometry_from_element(element)

        if geometry is None or geometry.is_empty:
            continue

        tipo, modo, categoria = clasificar_elemento(tags)

        registros.append(
            {
                "osm_type": element.get("type"),
                "osm_id": safe_int(element.get("id")),
                "nombre": extraer_nombre(tags),
                "operador": extraer_operador(tags),
                "tipo_infraestructura": tipo,
                "modo_principal": modo,
                "categoria_intermodal": categoria,
                "railway": tags.get("railway"),
                "amenity": tags.get("amenity"),
                "public_transport": tags.get(
                    "public_transport"
                ),
                "route": tags.get("route"),
                "network": tags.get("network"),
                "ref": tags.get("ref"),
                "wikidata": tags.get("wikidata"),
                "website": tags.get("website"),
                "geometry": geometry,
            }
        )

    if not registros:
        raise RuntimeError(
            "No se pudieron construir geometrías."
        )

    return gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=CRS_WGS84,
    ).reset_index(drop=True)


def normalizar_infraestructura(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    gdf = resetear_indices(gdf)

    gdf["osm_id"] = pd.to_numeric(
        gdf["osm_id"],
        errors="coerce",
    ).astype("Int64")

    columnas = [
        "nombre",
        "operador",
        "tipo_infraestructura",
        "modo_principal",
        "categoria_intermodal",
        "railway",
        "amenity",
        "public_transport",
        "route",
        "network",
        "ref",
        "wikidata",
        "website",
    ]

    for columna in columnas:
        if columna in gdf.columns:
            gdf[columna] = gdf[columna].astype("string")

    gdf["nombre_normalizado"] = (
        gdf["nombre"]
        .fillna("")
        .map(normalizar_texto)
        .astype("string")
    )

    gdf["modo_principal"] = (
        gdf["modo_principal"]
        .fillna("OTRO")
        .astype("string")
    )

    return resetear_indices(gdf)


def eliminar_duplicados_osm(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    antes = len(gdf)

    salida = gdf.drop_duplicates(
        subset=["osm_type", "osm_id"],
        keep="first",
    )

    salida = resetear_indices(salida)

    print(
        f"Duplicados OSM eliminados: "
        f"{antes - len(salida):,}"
    )

    return salida


# ============================================================================
# VALIDACION
# ============================================================================

def validar_gdf(
    gdf: gpd.GeoDataFrame,
    etiqueta: str,
    validar_osm: bool = True,
) -> None:
    subtitulo(etiqueta)

    nulos = int(gdf.geometry.isna().sum())
    vacios = int(gdf.geometry.is_empty.sum())
    invalidos = int((~gdf.geometry.is_valid).sum())

    print(f"Registros: {len(gdf):,}")
    print(f"Geometrías nulas: {nulos:,}")
    print(f"Geometrías vacías: {vacios:,}")
    print(f"Geometrías inválidas: {invalidos:,}")

    if validar_osm and "osm_id" in gdf.columns:
        duplicados = int(gdf["osm_id"].duplicated().sum())
        print(f"Duplicados OSM: {duplicados:,}")
    else:
        duplicados = 0

    if nulos or vacios or invalidos:
        raise RuntimeError(
            f"{etiqueta}: existen geometrías inválidas."
        )

    if validar_osm and duplicados:
        raise RuntimeError(
            f"{etiqueta}: existen OSM IDs duplicados."
        )


# ============================================================================
# CONSOLIDACION DE INSTALACIONES
# ============================================================================

def componentes_por_proximidad(
    gdf: gpd.GeoDataFrame,
    radio_m: float,
) -> list[list[int]]:
    """
    Componentes conexas espaciales.

    El spatial index devuelve posiciones; siempre se usa iloc.
    """

    gdf = resetear_indices(gdf)

    n = len(gdf)

    if n == 0:
        return []

    sindex = gdf.sindex
    padre = list(range(n))

    def find(x: int) -> int:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)

        if ra != rb:
            padre[rb] = ra

    for i in range(n):
        geom = gdf.iloc[i].geometry

        candidatos = sindex.query(
            geom.buffer(radio_m),
            predicate="intersects",
        )

        for j_raw in candidatos:
            j = int(j_raw)

            if j <= i:
                continue

            if geom.distance(gdf.iloc[j].geometry) <= radio_m:
                union(i, j)

    grupos = {}

    for i in range(n):
        raiz = find(i)
        grupos.setdefault(raiz, []).append(i)

    return list(grupos.values())


def construir_instalacion(
    sub: gpd.GeoDataFrame,
    instalacion_id: int,
    metodo: str,
) -> dict[str, Any]:
    modos = sorted(
        set(sub["modo_principal"].dropna().astype(str))
        & MODOS_VALIDOS
    )

    tipos = sorted(
        set(sub["tipo_infraestructura"].dropna().astype(str))
    )

    nombres = nombres_unicos(sub["nombre"])

    union = unary_union(
        list(sub.geometry)
    )

    centroide = union.centroid

    conteo_por_modo = {
        modo: int(
            (sub["modo_principal"] == modo).sum()
        )
        for modo in MODOS_VALIDOS
    }

    return {
        "instalacion_id": instalacion_id,
        "metodo_consolidacion": metodo,
        "cantidad_objetos_osm": int(len(sub)),
        "cantidad_modos": len(modos),
        "modos": "|".join(modos),
        "nombre_principal": (
            nombres.split(" | ")[0]
            if nombres
            else None
        ),
        "nombres_referencias": nombres,
        "tipos": "|".join(tipos),
        "ferrocarril": conteo_por_modo["FERROCARRIL"],
        "subte": conteo_por_modo["SUBTE"],
        "autobus": conteo_por_modo["AUTOBUS"],
        "fluvial": conteo_por_modo["FLUVIAL"],
        "tranvia": conteo_por_modo["TRANVIA"],
        "geometry": centroide,
    }


def consolidar_instalaciones(
    infraestructura: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Convierte objetos OSM en instalaciones físicas.

    Estrategia:

    A) elementos ancla:
       estaciones, terminales, paradas ferroviarias, etc.

       Se agrupan por mismo nombre + modo dentro de 100 m.

    B) elementos auxiliares:
       platform / stop_position.

       Se absorben en la instalación más cercana dentro de 100 m,
       preferentemente de su mismo modo.

    C) elementos restantes:
       se agrupan espacialmente por modo dentro de 60 m.

    D) elementos sin modo válido:
       se conservan como instalaciones aisladas.

    Esto evita que miles de objetos OSM auxiliares dominen el indicador.
    """

    titulo("CONSOLIDANDO OBJETOS OSM EN INSTALACIONES FÍSICAS")

    original = resetear_indices(
        infraestructura.to_crs(CRS_METRICO)
    )

    original["_asignado"] = False
    original["_instalacion_tmp"] = pd.NA

    instalaciones = []
    siguiente_id = 1

    # ------------------------------------------------------------------
    # A. ANCLAS NOMBRADAS
    # ------------------------------------------------------------------

    anclas = original[
        original["tipo_infraestructura"].isin(TIPOS_ANCLA)
        & original["nombre_normalizado"].fillna("").ne("")
        & original["modo_principal"].isin(MODOS_VALIDOS)
    ].copy()

    anclas = resetear_indices(anclas)

    print(
        f"Elementos ancla nombrados: {len(anclas):,}"
    )

    if not anclas.empty:
        grupos_nombre = []

        # Agrupar por nombre + modo.
        for _, bloque in anclas.groupby(
            ["nombre_normalizado", "modo_principal"],
            sort=False,
        ):
            bloque = resetear_indices(bloque)

            grupos = componentes_por_proximidad(
                bloque,
                RADIO_INSTALACION_NOMBRADA_M,
            )

            for grupo in grupos:
                indices_originales = (
                    bloque.iloc[grupo].index.to_list()
                )
                grupos_nombre.append(
                    indices_originales
                )

        for indices in grupos_nombre:
            sub = original.iloc[indices].copy()

            instalaciones.append(
                construir_instalacion(
                    sub,
                    siguiente_id,
                    "ANCLA_NOMBRE_MODO",
                )
            )

            original.loc[
                indices,
                "_asignado"
            ] = True

            original.loc[
                indices,
                "_instalacion_tmp"
            ] = siguiente_id

            siguiente_id += 1

    # ------------------------------------------------------------------
    # B. ANCLAS SIN NOMBRE
    # ------------------------------------------------------------------

    anclas_sin_nombre = original[
        (~original["_asignado"])
        & original["tipo_infraestructura"].isin(TIPOS_ANCLA)
        & original["modo_principal"].isin(MODOS_VALIDOS)
    ].copy()

    anclas_sin_nombre = resetear_indices(
        anclas_sin_nombre
    )

    print(
        f"Anclas sin nombre: {len(anclas_sin_nombre):,}"
    )

    if not anclas_sin_nombre.empty:
        grupos = componentes_por_proximidad(
            anclas_sin_nombre,
            RADIO_INSTALACION_SIN_NOMBRE_M,
        )

        for grupo in grupos:
            sub = anclas_sin_nombre.iloc[grupo].copy()

            ids_originales = sub.index.to_list()

            instalaciones.append(
                construir_instalacion(
                    original.iloc[ids_originales],
                    siguiente_id,
                    "ANCLA_ESPACIAL",
                )
            )

            original.loc[
                ids_originales,
                "_asignado"
            ] = True

            original.loc[
                ids_originales,
                "_instalacion_tmp"
            ] = siguiente_id

            siguiente_id += 1

    # ------------------------------------------------------------------
    # C. AUXILIARES -> INSTALACION EXISTENTE
    # ------------------------------------------------------------------

    auxiliares = original[
        (~original["_asignado"])
        & original["tipo_infraestructura"].isin(
            TIPOS_AUXILIARES
        )
        & original["modo_principal"].isin(
            MODOS_VALIDOS
        )
    ].copy()

    auxiliares = resetear_indices(auxiliares)

    print(
        f"Elementos auxiliares pendientes: "
        f"{len(auxiliares):,}"
    )

    # GeoDataFrame temporal de instalaciones.
    if instalaciones:
        instalaciones_gdf = gpd.GeoDataFrame(
            instalaciones,
            geometry="geometry",
            crs=CRS_METRICO,
        )
    else:
        instalaciones_gdf = gpd.GeoDataFrame(
            columns=["instalacion_id", "geometry"],
            geometry="geometry",
            crs=CRS_METRICO,
        )

    instalaciones_gdf = resetear_indices(
        instalaciones_gdf
    )

    if not auxiliares.empty and not instalaciones_gdf.empty:
        sindex = instalaciones_gdf.sindex

        for pos in range(len(auxiliares)):
            aux = auxiliares.iloc[pos]

            candidatos = consultar_vecinos(
                sindex,
                aux.geometry,
                RADIO_ADHESION_A_INSTALACION_M,
            )

            if not candidatos:
                continue

            mejor = None
            mejor_distancia = float("inf")

            for cand in candidatos:
                instalacion = instalaciones_gdf.iloc[cand]

                # Preferencia fuerte por mismo modo.
                mismo_modo = (
                    aux["modo_principal"]
                    == instalacion["modos"]
                )

                distancia = aux.geometry.distance(
                    instalacion.geometry
                )

                penalizacion = (
                    0
                    if mismo_modo
                    else 1000
                )

                distancia_ponderada = (
                    distancia + penalizacion
                )

                if distancia_ponderada < mejor_distancia:
                    mejor_distancia = distancia_ponderada
                    mejor = cand

            if mejor is not None:
                instalacion_id = int(
                    instalaciones_gdf.iloc[mejor][
                        "instalacion_id"
                    ]
                )

                original_idx = int(
                    auxiliares.index[pos]
                )

                original.loc[
                    original_idx,
                    "_asignado"
                ] = True

                original.loc[
                    original_idx,
                    "_instalacion_tmp"
                ] = instalacion_id

    # ------------------------------------------------------------------
    # D. RESTANTES -> CLUSTERS POR MODO
    # ------------------------------------------------------------------

    restantes = original[
        ~original["_asignado"]
        & original["modo_principal"].isin(
            MODOS_VALIDOS
        )
    ].copy()

    restantes = resetear_indices(restantes)

    print(
        f"Elementos restantes para clustering: "
        f"{len(restantes):,}"
    )

    if not restantes.empty:
        for modo, bloque in restantes.groupby(
            "modo_principal",
            sort=False,
        ):
            bloque = resetear_indices(bloque)

            grupos = componentes_por_proximidad(
                bloque,
                RADIO_INSTALACION_SIN_NOMBRE_M,
            )

            for grupo in grupos:
                ids_originales = bloque.iloc[
                    grupo
                ].index.to_list()

                sub = original.iloc[
                    ids_originales
                ].copy()

                instalaciones.append(
                    construir_instalacion(
                        sub,
                        siguiente_id,
                        "CLUSTER_MODO",
                    )
                )

                original.loc[
                    ids_originales,
                    "_asignado"
                ] = True

                original.loc[
                    ids_originales,
                    "_instalacion_tmp"
                ] = siguiente_id

                siguiente_id += 1

    # ------------------------------------------------------------------
    # E. OBJETOS SIN MODO
    # ------------------------------------------------------------------

    otros = original[
        ~original["_asignado"]
    ].copy()

    otros = resetear_indices(otros)

    print(
        f"Objetos sin instalación asignada: "
        f"{len(otros):,}"
    )

    for pos in range(len(otros)):
        sub = otros.iloc[[pos]].copy()

        instalaciones.append(
            construir_instalacion(
                sub,
                siguiente_id,
                "AISLADO",
            )
        )

        siguiente_id += 1

    # ------------------------------------------------------------------
    # F. FINAL
    # ------------------------------------------------------------------

    if not instalaciones:
        raise RuntimeError(
            "No se pudieron construir instalaciones."
        )

    salida = gpd.GeoDataFrame(
        instalaciones,
        geometry="geometry",
        crs=CRS_METRICO,
    )

    salida = salida.to_crs(CRS_WGS84)
    salida = resetear_indices(salida)

    # Indicadores.
    salida["es_intermodal"] = (
        salida["cantidad_modos"] >= 2
    ).astype(int)

    salida["score_instalacion"] = (
        salida["cantidad_modos"].clip(
            upper=4
        )
        / 4.0
        * 70.0
        + salida["cantidad_objetos_osm"].clip(
            upper=10
        )
        / 10.0
        * 30.0
    ).round(2)

    print()
    print(
        f"Instalaciones físicas consolidadas: "
        f"{len(salida):,}"
    )

    print(
        f"Reducción respecto de objetos OSM: "
        f"{len(infraestructura) - len(salida):,}"
    )

    print()
    print("INSTALACIONES POR MODO")

    for modo, cantidad in (
        salida["modos"]
        .replace("", "SIN_MODO")
        .value_counts()
        .head(20)
        .items()
    ):
        print(
            f"  {str(modo):35s} {int(cantidad):8,d}"
        )

    return salida


# ============================================================================
# INTERCAMBIADORES
# ============================================================================

def detectar_intercambiadores(
    instalaciones: gpd.GeoDataFrame,
    radio_m: float = RADIO_INTERCAMBIADOR_M,
) -> gpd.GeoDataFrame:
    """
    Detecta clusters de INSTALACIONES, no de objetos OSM.

    Cada instalación cuenta una sola vez por modo.
    """

    titulo(
        "DETECTANDO INTERCAMBIADORES ENTRE INSTALACIONES"
    )

    infra = resetear_indices(
        instalaciones.to_crs(CRS_METRICO)
    )

    infra = infra[
        infra["cantidad_modos"] > 0
    ].copy()

    infra = resetear_indices(infra)

    if infra.empty:
        return gpd.GeoDataFrame(
            columns=[
                "intercambiador_id",
                "cantidad_instalaciones",
                "cantidad_modos",
                "modos",
                "geometry",
            ],
            geometry="geometry",
            crs=CRS_WGS84,
        )

    grupos = componentes_por_proximidad(
        infra,
        radio_m,
    )

    registros = []
    intercambiador_id = 1

    for grupo in grupos:
        sub = infra.iloc[grupo].copy()

        modos = sorted(
            set(
                modo
                for valor in sub["modos"].dropna().astype(str)
                for modo in valor.split("|")
                if modo in MODOS_VALIDOS
            )
        )

        if len(modos) < MIN_MODOS_INTERCAMBIADOR:
            continue

        union = unary_union(
            list(sub.geometry)
        )

        centroide = union.centroid

        distancias = sub.geometry.distance(
            centroide
        )

        radio_real = (
            float(distancias.max())
            if len(distancias)
            else 0.0
        )

        conteo_instalaciones = {
            modo: int(
                sub["modos"]
                .fillna("")
                .str.contains(
                    modo,
                    regex=False,
                )
                .sum()
            )
            for modo in MODOS_VALIDOS
        }

        nombres = nombres_unicos(
            sub["nombre_principal"]
        )

        # Score independiente de la cantidad bruta de objetos OSM.
        score_diversidad = min(
            len(modos),
            4,
        ) / 4.0 * 70.0

        score_instalaciones = min(
            len(sub),
            6,
        ) / 6.0 * 30.0

        registros.append(
            {
                "intercambiador_id": intercambiador_id,
                "cantidad_instalaciones": int(len(sub)),
                "cantidad_modos": len(modos),
                "modos": "|".join(modos),
                "nombre_referencias": nombres,
                "radio_cluster_m": round(
                    radio_real,
                    2,
                ),
                "ferrocarril": conteo_instalaciones[
                    "FERROCARRIL"
                ],
                "subte": conteo_instalaciones[
                    "SUBTE"
                ],
                "autobus": conteo_instalaciones[
                    "AUTOBUS"
                ],
                "fluvial": conteo_instalaciones[
                    "FLUVIAL"
                ],
                "tranvia": conteo_instalaciones[
                    "TRANVIA"
                ],
                "score_intercambiador": round(
                    min(
                        100.0,
                        score_diversidad
                        + score_instalaciones,
                    ),
                    2,
                ),
                "geometry": centroide,
            }
        )

        intercambiador_id += 1

    if not registros:
        return gpd.GeoDataFrame(
            columns=[
                "intercambiador_id",
                "cantidad_instalaciones",
                "cantidad_modos",
                "modos",
                "geometry",
            ],
            geometry="geometry",
            crs=CRS_METRICO,
        ).to_crs(CRS_WGS84)

    salida = gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=CRS_METRICO,
    )

    salida = salida.to_crs(CRS_WGS84)

    print(
        f"Intercambiadores detectados: "
        f"{len(salida):,}"
    )

    return resetear_indices(salida)


# ============================================================================
# CENTRALIDADES
# ============================================================================

def cargar_centralidades() -> gpd.GeoDataFrame | None:
    if not CENTRALIDADES_PATH.exists():
        print(
            "ADVERTENCIA: no se encontró:"
        )
        print(CENTRALIDADES_PATH)
        return None

    try:
        centralidades = gpd.read_parquet(
            CENTRALIDADES_PATH
        )
    except Exception as exc:
        print(
            f"ADVERTENCIA: no se pudieron cargar "
            f"centralidades: {exc}"
        )
        return None

    if centralidades.empty:
        return None

    if centralidades.crs is None:
        centralidades = centralidades.set_crs(
            CRS_WGS84
        )

    centralidades = centralidades.to_crs(
        CRS_METRICO
    )

    centralidades = resetear_indices(
        centralidades
    )

    if "nodo_id" not in centralidades.columns:
        centralidades["nodo_id"] = np.arange(
            1,
            len(centralidades) + 1,
        )

    return centralidades


def validar_centralidades(
    centralidades: gpd.GeoDataFrame,
) -> None:
    subtitulo(
        "VALIDACIÓN DE CENTRALIDADES"
    )

    duplicados = int(
        centralidades["nodo_id"].duplicated().sum()
    )

    invalidas = int(
        (~centralidades.geometry.is_valid).sum()
    )

    print(
        f"Centralidades: {len(centralidades):,}"
    )
    print(
        f"nodo_id duplicados: {duplicados:,}"
    )
    print(
        f"Geometrías inválidas: {invalidas:,}"
    )

    if duplicados or invalidas:
        raise RuntimeError(
            "Las centralidades no superan la validación."
        )


def analizar_centralidades(
    centralidades: gpd.GeoDataFrame,
    instalaciones: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Calcula indicadores de intermodalidad usando INSTALACIONES FISICAS.

    Esto es importante: la densidad ya no depende de la cantidad de
    platform / stop_position individuales de OSM.
    """

    titulo(
        "CALCULANDO INTERMODALIDAD DE CENTRALIDADES"
    )

    centros = resetear_indices(
        centralidades.to_crs(CRS_METRICO)
    )

    infra = resetear_indices(
        instalaciones.to_crs(CRS_METRICO)
    )

    interc = resetear_indices(
        intercambiadores.to_crs(CRS_METRICO)
    )

    sindex_infra = infra.sindex
    sindex_interc = (
        interc.sindex
        if not interc.empty
        else None
    )

    resultados = []

    for pos in range(len(centros)):
        nodo = centros.iloc[pos]
        punto = nodo.geometry

        if (
            pos == 0
            or (pos + 1) % 25 == 0
            or pos + 1 == len(centros)
        ):
            print(
                f"Centralidad {pos + 1}/"
                f"{len(centros)}"
            )

        registro = {
            "nodo_id": nodo["nodo_id"],
        }

        # --------------------------------------------------------------
        # RADIOS
        # --------------------------------------------------------------

        for radio in RADIOS_INTERMODALIDAD_M:
            candidatos = consultar_vecinos(
                sindex_infra,
                punto,
                radio,
            )

            if candidatos:
                sub = infra.iloc[
                    candidatos
                ]

                modos = {
                    modo
                    for valor in (
                        sub["modos"]
                        .dropna()
                        .astype(str)
                    )
                    for modo in valor.split("|")
                    if modo in MODOS_VALIDOS
                }

            else:
                sub = infra.iloc[[]]
                modos = set()

            registro[
                f"instalaciones_{radio}m"
            ] = int(len(sub))

            # Alias para compatibilidad.
            registro[
                f"infra_{radio}m"
            ] = int(len(sub))

            registro[
                f"modos_{radio}m"
            ] = int(len(modos))

            for modo, nombre in [
                ("FERROCARRIL", "ferrocarril"),
                ("SUBTE", "subte"),
                ("AUTOBUS", "autobus"),
                ("FLUVIAL", "fluvial"),
                ("TRANVIA", "tranvia"),
            ]:
                registro[
                    f"{nombre}_{radio}m"
                ] = int(modo in modos)

        # --------------------------------------------------------------
        # INFRAESTRUCTURA MAS CERCANA
        # --------------------------------------------------------------

        candidatos_1000 = consultar_vecinos(
            sindex_infra,
            punto,
            1000,
        )

        if candidatos_1000:
            sub = infra.iloc[
                candidatos_1000
            ]

            distancias = sub.geometry.distance(
                punto
            )

            pos_min = int(
                np.argmin(
                    distancias.to_numpy()
                )
            )

            cercano = sub.iloc[pos_min]

            registro[
                "distancia_infraestructura_m"
            ] = float(
                distancias.iloc[pos_min]
            )

            registro[
                "infraestructura_mas_cercana"
            ] = (
                str(cercano["nombre_principal"])
                if pd.notna(
                    cercano["nombre_principal"]
                )
                and str(
                    cercano["nombre_principal"]
                ).strip()
                else str(
                    cercano["tipos"]
                )
            )

            registro[
                "modo_infraestructura_mas_cercana"
            ] = str(
                cercano["modos"]
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

        # --------------------------------------------------------------
        # INTERCAMBIADORES
        # --------------------------------------------------------------

        if sindex_interc is not None:
            cand500 = consultar_vecinos(
                sindex_interc,
                punto,
                500,
            )

            cand1000 = consultar_vecinos(
                sindex_interc,
                punto,
                1000,
            )
        else:
            cand500 = []
            cand1000 = []

        registro[
            "intercambiadores_500m"
        ] = len(cand500)

        registro[
            "intercambiadores_1000m"
        ] = len(cand1000)

        if cand500:
            sub_i = interc.iloc[
                cand500
            ]

            distancias_i = (
                sub_i.geometry.distance(punto)
            )

            pos_min_i = int(
                np.argmin(
                    distancias_i.to_numpy()
                )
            )

            cercano_i = sub_i.iloc[
                pos_min_i
            ]

            registro[
                "distancia_intercambiador_m"
            ] = float(
                distancias_i.iloc[pos_min_i]
            )

            registro[
                "intercambiador_mas_cercano_id"
            ] = safe_int(
                cercano_i[
                    "intercambiador_id"
                ]
            )

        else:
            registro[
                "distancia_intercambiador_m"
            ] = np.nan

            registro[
                "intercambiador_mas_cercano_id"
            ] = None

        # --------------------------------------------------------------
        # SCORE
        # --------------------------------------------------------------

        modos_500 = registro[
            "modos_500m"
        ]

        presencia = sum(
            registro[
                f"{nombre}_500m"
            ]
            for nombre in (
                "ferrocarril",
                "subte",
                "autobus",
                "fluvial",
                "tranvia",
            )
        )

        instalaciones_500 = registro[
            "instalaciones_500m"
        ]

        # Diversidad modal: 60%.
        score_diversidad = (
            min(modos_500, 4)
            / 4.0
            * 60.0
        )

        # Presencia modal ponderada: 25%.
        score_presencia = 0.0

        for modo, nombre in [
            ("FERROCARRIL", "ferrocarril"),
            ("SUBTE", "subte"),
            ("AUTOBUS", "autobus"),
            ("FLUVIAL", "fluvial"),
            ("TRANVIA", "tranvia"),
        ]:
            if registro[
                f"{nombre}_500m"
            ]:
                score_presencia += PESO_MODO[
                    modo
                ]

        score_presencia = min(
            score_presencia,
            25.0,
        )

        # Intensidad física: 10%, capada.
        score_intensidad = (
            min(
                instalaciones_500,
                10,
            )
            / 10.0
            * 10.0
        )

        # Intercambiador: 5%.
        score_intercambiador = (
            min(
                registro[
                    "intercambiadores_500m"
                ],
                1,
            )
            * 5.0
        )

        score = min(
            100.0,
            score_diversidad
            + score_presencia
            + score_intensidad
            + score_intercambiador,
        )

        registro[
            "score_intermodalidad_500m"
        ] = round(score, 2)

        # --------------------------------------------------------------
        # CATEGORIA
        # --------------------------------------------------------------

        if (
            modos_500 >= 4
            or (
                modos_500 >= 3
                and registro[
                    "intercambiadores_500m"
                ] >= 1
            )
        ):
            categoria = (
                "INTERMODALIDAD_MUY_ALTA"
            )

        elif modos_500 >= 3:
            categoria = (
                "INTERMODALIDAD_ALTA"
            )

        elif modos_500 == 2:
            categoria = (
                "INTERMODALIDAD_MEDIA"
            )

        elif modos_500 == 1:
            categoria = (
                "INTERMODALIDAD_BAJA"
            )

        else:
            categoria = (
                "SIN_INFRAESTRUCTURA_500M"
            )

        registro[
            "categoria_intermodalidad_500m"
        ] = categoria

        resultados.append(registro)

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

    return resetear_indices(centros)


# ============================================================================
# RESUMEN
# ============================================================================

def construir_resumen(
    objetos_osm: gpd.GeoDataFrame,
    instalaciones: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
    centralidades: gpd.GeoDataFrame | None,
) -> dict[str, Any]:
    resumen = {
        "proyecto": (
            "Análisis de movilidad SUBE AMBA"
        ),
        "script": (
            "21_construir_infraestructura_intermodal_amba.py"
        ),
        "version": "3.0",
        "fecha_ejecucion": (
            pd.Timestamp.now().isoformat()
        ),
        "bbox": AMBA_BBOX,
        "crs_original": CRS_WGS84,
        "crs_metrico": CRS_METRICO,
        "objetos_osm": int(len(objetos_osm)),
        "instalaciones_fisicas": int(
            len(instalaciones)
        ),
        "intercambiadores": int(
            len(intercambiadores)
        ),
        "radio_instalacion_nombrada_m": (
            RADIO_INSTALACION_NOMBRADA_M
        ),
        "radio_instalacion_sin_nombre_m": (
            RADIO_INSTALACION_SIN_NOMBRE_M
        ),
        "radio_adhesion_m": (
            RADIO_ADHESION_A_INSTALACION_M
        ),
        "radio_intercambiador_m": (
            RADIO_INTERCAMBIADOR_M
        ),
    }

    resumen[
        "instalaciones_por_cantidad_modos"
    ] = {
        str(k): int(v)
        for k, v in (
            instalaciones[
                "cantidad_modos"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        )
    }

    resumen[
        "intercambiadores_por_cantidad_modos"
    ] = {
        str(k): int(v)
        for k, v in (
            intercambiadores[
                "cantidad_modos"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        )
    }

    if centralidades is not None:
        resumen[
            "centralidades_analizadas"
        ] = int(len(centralidades))

        top = (
            centralidades
            .sort_values(
                "score_intermodalidad_500m",
                ascending=False,
            )
            .head(15)
        )

        resumen[
            "top_15_centralidades"
        ] = []

        for _, row in top.iterrows():
            resumen[
                "top_15_centralidades"
            ].append(
                {
                    "nodo_id": safe_int(
                        row["nodo_id"]
                    ),
                    "score": safe_float(
                        row[
                            "score_intermodalidad_500m"
                        ]
                    ),
                    "instalaciones_500m": safe_int(
                        row[
                            "instalaciones_500m"
                        ]
                    ),
                    "modos_500m": safe_int(
                        row["modos_500m"]
                    ),
                    "intercambiadores_500m": (
                        safe_int(
                            row[
                                "intercambiadores_500m"
                            ]
                        )
                    ),
                    "categoria": str(
                        row[
                            "categoria_intermodalidad_500m"
                        ]
                    ),
                }
            )

    return resumen


# ============================================================================
# SALIDAS
# ============================================================================

def guardar_gpkg_layer(
    gdf: gpd.GeoDataFrame,
    path: Path,
    layer: str,
) -> None:
    gdf.to_file(
        path,
        layer=layer,
        driver="GPKG",
    )


def guardar_salidas(
    objetos_osm: gpd.GeoDataFrame,
    instalaciones: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
    centralidades: gpd.GeoDataFrame | None,
    resumen: dict[str, Any],
) -> None:
    titulo("GUARDANDO ARCHIVOS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gpkg = (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.gpkg"
    )

    # Objetos OSM normalizados.
    objetos_osm.to_parquet(
        OUTPUT_DIR
        / "objetos_osm_infraestructura_amba.parquet",
        index=False,
    )

    # Instalaciones fisicas.
    instalaciones.to_parquet(
        OUTPUT_DIR
        / "instalaciones_transporte_amba.parquet",
        index=False,
    )

    instalaciones.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        OUTPUT_DIR
        / "instalaciones_transporte_amba.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Intercambiadores.
    intercambiadores.to_parquet(
        OUTPUT_DIR
        / "intercambiadores_intermodales_amba.parquet",
        index=False,
    )

    intercambiadores.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        OUTPUT_DIR
        / "intercambiadores_intermodales_amba.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Mantener nombres historicos para compatibilidad.
    instalaciones.to_parquet(
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.parquet",
        index=False,
    )

    instalaciones.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        OUTPUT_DIR
        / "infraestructura_intermodal_amba.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # GeoPackage: se sobreescribe limpiamente.
    if gpkg.exists():
        try:
            gpkg.unlink()
        except Exception:
            pass

    guardar_gpkg_layer(
        objetos_osm,
        gpkg,
        "objetos_osm",
    )

    guardar_gpkg_layer(
        instalaciones,
        gpkg,
        "instalaciones",
    )

    guardar_gpkg_layer(
        intercambiadores,
        gpkg,
        "intercambiadores",
    )

    if centralidades is not None:
        centralidades.to_parquet(
            OUTPUT_DIR
            / "centralidades_intermodalidad_amba.parquet",
            index=False,
        )

        guardar_gpkg_layer(
            centralidades,
            gpkg,
            "centralidades",
        )

    with (
        OUTPUT_DIR
        / "infraestructura_intermodal_amba_resumen.json"
    ).open(
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
        f"GeoPackage:\n{gpkg}"
    )


# ============================================================================
# GRAFICOS
# ============================================================================

def importar_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print(
            "ADVERTENCIA: Matplotlib no disponible."
        )
        return None


def generar_mapa(
    gdf: gpd.GeoDataFrame,
    columna: str,
    titulo_mapa: str,
    archivo: str,
    markersize: float = 15,
) -> None:
    if gdf.empty:
        return

    plt = importar_matplotlib()

    if plt is None:
        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    gdf.plot(
        ax=ax,
        column=columna,
        markersize=markersize,
        legend=True,
        alpha=0.7,
    )

    ax.set_title(
        titulo_mapa,
        fontsize=15,
    )

    ax.set_axis_off()

    path = OUTPUT_DIR / archivo

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Mapa:\n{path}")


def generar_graficos(
    instalaciones: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
    centralidades: gpd.GeoDataFrame | None,
) -> None:
    titulo(
        "GENERANDO MAPAS Y GRÁFICOS"
    )

    plt = importar_matplotlib()

    if plt is None:
        return

    # 01 instalaciones.
    generar_mapa(
        instalaciones,
        "cantidad_modos",
        "Instalaciones físicas de transporte - AMBA",
        "01_mapa_instalaciones_transporte.png",
        12,
    )

    # 02 intercambiadores.
    if not intercambiadores.empty:
        generar_mapa(
            intercambiadores,
            "cantidad_modos",
            "Intercambiadores intermodales - AMBA",
            "02_mapa_intercambiadores_intermodales.png",
            35,
        )

    # 03 cantidad de instalaciones por modos.
    conteo = {}

    for modo in MODOS_VALIDOS:
        conteo[modo] = int(
            instalaciones["modos"]
            .fillna("")
            .str.contains(
                modo,
                regex=False,
            )
            .sum()
        )

    serie = (
        pd.Series(conteo)
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    serie.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Instalaciones físicas por modo"
    )

    ax.set_xlabel(
        "Cantidad de instalaciones"
    )

    path = (
        OUTPUT_DIR
        / "03_instalaciones_por_modo.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Gráfico:\n{path}")

    # 04 intercambiadores por cantidad de modos.
    if not intercambiadores.empty:
        serie_i = (
            intercambiadores[
                "cantidad_modos"
            ]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        serie_i.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_title(
            "Intercambiadores por cantidad de modos"
        )

        ax.set_xlabel(
            "Cantidad de modos"
        )

        ax.set_ylabel(
            "Cantidad de intercambiadores"
        )

        path = (
            OUTPUT_DIR
            / "04_intercambiadores_por_modos.png"
        )

        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Gráfico:\n{path}")

    # 05 centralidades.
    if (
        centralidades is not None
        and not centralidades.empty
    ):
        generar_mapa(
            centralidades,
            "score_intermodalidad_500m",
            "Centralidades SUBE - Intermodalidad a 500 m",
            "05_centralidades_intermodalidad_500m.png",
            35,
        )

        valores = (
            centralidades[
                "score_intermodalidad_500m"
            ]
            .dropna()
        )

        if not valores.empty:
            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            ax.hist(
                valores,
                bins=15,
            )

            ax.set_title(
                "Distribución del score de intermodalidad"
            )

            ax.set_xlabel("Score")
            ax.set_ylabel(
                "Cantidad de centralidades"
            )

            path = (
                OUTPUT_DIR
                / "06_distribucion_score_intermodalidad.png"
            )

            fig.savefig(
                path,
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(f"Gráfico:\n{path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    titulo(
        "21 - CONSOLIDACIÓN DE INFRAESTRUCTURA INTERMODAL AMBA - V3"
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

    print(
        f"Radio instalación nombrada: "
        f"{RADIO_INSTALACION_NOMBRADA_M} m"
    )

    print(
        f"Radio instalación sin nombre: "
        f"{RADIO_INSTALACION_SIN_NOMBRE_M} m"
    )

    print(
        f"Radio adhesión: "
        f"{RADIO_ADHESION_A_INSTALACION_M} m"
    )

    print(
        f"Radio intercambiador: "
        f"{RADIO_INTERCAMBIADOR_M} m"
    )

    # ------------------------------------------------------------------
    # 1. OVERPASS
    # ------------------------------------------------------------------

    titulo(
        "1. CONSULTANDO OPENSTREETMAP / OVERPASS"
    )

    datos = consultar_overpass()

    # ------------------------------------------------------------------
    # 2. OBJETOS OSM
    # ------------------------------------------------------------------

    titulo(
        "2. CONSTRUYENDO OBJETOS OSM NORMALIZADOS"
    )

    objetos_osm = construir_gdf(
        datos
    )

    print(
        f"Elementos construidos: "
        f"{len(objetos_osm):,}"
    )

    objetos_osm = normalizar_infraestructura(
        objetos_osm
    )

    validar_gdf(
        objetos_osm,
        "VALIDACIÓN DE OBJETOS OSM",
    )

    # ------------------------------------------------------------------
    # 3. DUPLICADOS
    # ------------------------------------------------------------------

    titulo(
        "3. ELIMINANDO DUPLICADOS OSM"
    )

    objetos_osm = eliminar_duplicados_osm(
        objetos_osm
    )

    validar_gdf(
        objetos_osm,
        "VALIDACIÓN POST OSM",
    )

    # ------------------------------------------------------------------
    # 4. INSTALACIONES FISICAS
    # ------------------------------------------------------------------

    instalaciones = consolidar_instalaciones(
        objetos_osm
    )

    validar_gdf(
        instalaciones,
        "VALIDACIÓN DE INSTALACIONES FÍSICAS",
        validar_osm=False,
    )

    # ------------------------------------------------------------------
    # 5. INTERCAMBIADORES
    # ------------------------------------------------------------------

    intercambiadores = detectar_intercambiadores(
        instalaciones,
        RADIO_INTERCAMBIADOR_M,
    )

    # ------------------------------------------------------------------
    # 6. CENTRALIDADES
    # ------------------------------------------------------------------

    titulo(
        "6. CARGANDO CENTRALIDADES SUBE"
    )

    centralidades = cargar_centralidades()

    if centralidades is not None:
        validar_centralidades(
            centralidades
        )

        print(
            f"Centralidades cargadas: "
            f"{len(centralidades):,}"
        )

        centralidades = analizar_centralidades(
            centralidades,
            instalaciones,
            intercambiadores,
        )

        print()
        print(
            "TOP 20 CENTRALIDADES POR INTERMODALIDAD"
        )

        columnas = [
            "nodo_id",
            "score_intermodalidad_500m",
            "ranking_intermodalidad",
            "instalaciones_250m",
            "instalaciones_500m",
            "instalaciones_1000m",
            "modos_500m",
            "ferrocarril_500m",
            "subte_500m",
            "autobus_500m",
            "fluvial_500m",
            "intercambiadores_500m",
            "categoria_intermodalidad_500m",
        ]

        disponibles = [
            c
            for c in columnas
            if c in centralidades.columns
        ]

        print(
            centralidades[
                disponibles
            ]
            .sort_values(
                "score_intermodalidad_500m",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )

    else:
        print(
            "No se realizó el cruce con centralidades."
        )

    # ------------------------------------------------------------------
    # 7. RESUMEN
    # ------------------------------------------------------------------

    titulo(
        "7. CONSTRUYENDO RESUMEN JSON"
    )

    resumen = construir_resumen(
        objetos_osm,
        instalaciones,
        intercambiadores,
        centralidades,
    )

    # ------------------------------------------------------------------
    # 8. GUARDAR
    # ------------------------------------------------------------------

    guardar_salidas(
        objetos_osm,
        instalaciones,
        intercambiadores,
        centralidades,
        resumen,
    )

    # ------------------------------------------------------------------
    # 9. GRAFICOS
    # ------------------------------------------------------------------

    generar_graficos(
        instalaciones,
        intercambiadores,
        centralidades,
    )

    # ------------------------------------------------------------------
    # 10. FINAL
    # ------------------------------------------------------------------

    titulo(
        "21 - PROCESO FINALIZADO"
    )

    print(
        f"Objetos OSM: "
        f"{len(objetos_osm):,}"
    )

    print(
        f"Instalaciones físicas: "
        f"{len(instalaciones):,}"
    )

    print(
        f"Intercambiadores: "
        f"{len(intercambiadores):,}"
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
        "Validar las instalaciones e intercambiadores "
        "contra las 144 centralidades y luego construir "
        "el índice de centralidad estructural."
    )


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
        print("=" * 78)
        print("ERROR FATAL")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "Revisá el mensaje anterior para "
            "identificar el paso donde ocurrió el error."
        )
        sys.exit(1)