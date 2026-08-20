# -*- coding: utf-8 -*-
"""
21_construir_infraestructura_intermodal_amba.py

Construye un inventario de infraestructura de transporte del AMBA a partir
 de OpenStreetMap / Overpass y calcula intermodalidad alrededor de las
144 centralidades SUBE.

Mejoras de esta versión:
- Índices GeoPandas completamente normalizados para evitar KeyError del sindex.
- Cache local de la respuesta Overpass: si la consulta falla, se reutiliza la
  última respuesta válida.
- Deduplicación OSM y espacial conservadora.
- Detección explícita de intercambiadores intermodales.
- No se cuenta OTRO como modo de transporte.
- Indicadores a 250 / 500 / 1000 m.
- Capa independiente de intercambiadores.
- Enriquecimiento de las 144 centralidades SUBE.
- GeoParquet, GeoPackage, CSV y JSON.
- Mapas y gráficos de control.

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
# CONFIGURACIÓN
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "infraestructura_intermodal_amba"

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

OVERPASS_CACHE = OUTPUT_DIR / "overpass_infraestructura_intermodal_amba.json"

# sur, oeste, norte, este
AMBA_BBOX = "-35.20,-59.40,-34.20,-57.80"

CRS_WGS84 = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

RADIOS_INTERMODALIDAD_M = [250, 500, 1000]
RADIO_CENTRALIDAD_M = 1000

# Radio para agrupar infraestructura físicamente próxima.
RADIO_INTERCAMBIADOR_M = 150

# Mínimo de modos de transporte distintos para considerar un intercambiador.
MIN_MODOS_INTERCAMBIADOR = 2

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3

MODOS_VALIDOS = {
    "FERROCARRIL",
    "SUBTE",
    "AUTOBUS",
    "FLUVIAL",
    "TRANVIA",
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
    if valor is None or pd.isna(valor):
        return ""

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


def resetear_indices(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Garantiza que el índice coincida con posiciones iloc 0..n-1."""
    salida = gdf.copy()
    salida = salida.reset_index(drop=True)
    return salida


def valores_unicos_texto(serie: pd.Series) -> list[str]:
    return sorted(
        {
            str(v)
            for v in serie.dropna()
            if str(v).strip() and str(v) != "nan"
        }
    )


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
    try:
        with OVERPASS_CACHE.open("w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False)
        print(f"Respuesta cruda guardada:\n{OVERPASS_CACHE}")
    except Exception as exc:
        print(f"ADVERTENCIA: no se pudo guardar cache Overpass: {exc}")


def cargar_cache_overpass() -> dict | None:
    if not OVERPASS_CACHE.exists():
        return None

    try:
        with OVERPASS_CACHE.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)

        if not isinstance(datos, dict) or "elements" not in datos:
            return None

        return datos
    except Exception as exc:
        print(f"ADVERTENCIA: cache Overpass inválida: {exc}")
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
                        "User-Agent": "analisis-movilidad-amba/2.0",
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                print(f"HTTP: {respuesta.status_code}")
                respuesta.raise_for_status()

                datos = respuesta.json()

                if "elements" not in datos:
                    raise RuntimeError("La respuesta no contiene 'elements'.")

                print(f"Elementos recibidos: {len(datos['elements']):,}")
                guardar_cache_overpass(datos)
                return datos

            except Exception as exc:
                ultimo_error = exc
                print(f"Error: {type(exc).__name__}: {exc}")
                time.sleep(2)

    # Fallback importante: no perdemos una ejecución anterior válida.
    cache = cargar_cache_overpass()
    if cache is not None:
        print()
        print("ADVERTENCIA: Overpass no respondió correctamente.")
        print("Se utilizará la última respuesta cacheada válida.")
        print(f"Elementos cacheados: {len(cache.get('elements', [])):,}")
        return cache

    raise RuntimeError(
        "No fue posible consultar Overpass y no existe una cache válida. "
        f"Último error: {ultimo_error}"
    )


# ============================================================================
# CLASIFICACIÓN OSM
# ============================================================================


def clasificar_elemento(tags: dict) -> tuple[str, str, str]:
    railway = normalizar_texto(tags.get("railway"))
    amenity = normalizar_texto(tags.get("amenity"))
    public_transport = normalizar_texto(tags.get("public_transport"))
    route = normalizar_texto(tags.get("route"))

    if railway == "station":
        return "ESTACION_FERROVIARIA", "FERROCARRIL", "FERROVIARIO"

    if railway == "halt":
        return "PARADA_FERROVIARIA", "FERROCARRIL", "FERROVIARIO"

    if railway == "subway":
        return "ESTACION_SUBTE", "SUBTE", "FERROVIARIO"

    if railway == "tram_stop":
        return "PARADA_TRANVIA", "TRANVIA", "FERROVIARIO"

    if railway == "light_rail":
        return "ESTACION_LIGERO", "FERROCARRIL", "FERROVIARIO"

    if railway == "stop":
        return "PARADA_FERROVIARIA", "FERROCARRIL", "FERROVIARIO"

    if amenity == "bus_station":
        return "TERMINAL_AUTOBUS", "AUTOBUS", "AUTOMOTOR"

    if amenity == "ferry_terminal":
        return "TERMINAL_FLUVIAL", "FLUVIAL", "FLUVIAL"

    if route == "ferry":
        return "RUTA_FLUVIAL", "FLUVIAL", "FLUVIAL"

    # public_transport=station puede ser ferroviario, subte o terminal.
    # Sin información adicional se conserva como MULTIMODAL.
    if public_transport == "station":
        return "ESTACION_TRANSPORTE_PUBLICO", "MULTIMODAL", "INTERMODAL"

    if public_transport == "stop_position":
        return "PARADA_TRANSPORTE_PUBLICO", "AUTOBUS", "AUTOMOTOR"

    if public_transport == "platform":
        return "PLATAFORMA_TRANSPORTE_PUBLICO", "AUTOBUS", "AUTOMOTOR"

    return "OTRA_INFRAESTRUCTURA", "OTRO", "OTRO"


def extraer_nombre(tags: dict) -> str | None:
    for campo in ("name", "official_name", "short_name", "alt_name"):
        valor = tags.get(campo)
        if valor:
            return str(valor).strip()
    return None


def extraer_operador(tags: dict) -> str | None:
    for campo in ("operator", "network", "brand"):
        valor = tags.get(campo)
        if valor:
            return str(valor).strip()
    return None


def geometry_from_element(element: dict) -> Point | None:
    tipo = element.get("type")

    if tipo == "node":
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
# CONSTRUCCIÓN Y NORMALIZACIÓN
# ============================================================================


def construir_gdf(datos: dict) -> gpd.GeoDataFrame:
    registros: list[dict[str, Any]] = []

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
                "public_transport": tags.get("public_transport"),
                "route": tags.get("route"),
                "network": tags.get("network"),
                "ref": tags.get("ref"),
                "wikidata": tags.get("wikidata"),
                "website": tags.get("website"),
                "geometry": geometry,
            }
        )

    if not registros:
        raise RuntimeError("No se pudieron construir geometrías.")

    gdf = gpd.GeoDataFrame(registros, geometry="geometry", crs=CRS_WGS84)
    return resetear_indices(gdf)


def normalizar_infraestructura(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = resetear_indices(gdf)

    gdf["osm_id"] = pd.to_numeric(gdf["osm_id"], errors="coerce").astype("Int64")

    columnas_texto = [
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

    for columna in columnas_texto:
        if columna in gdf.columns:
            gdf[columna] = gdf[columna].astype("string")

    gdf["nombre_normalizado"] = (
        gdf["nombre"].fillna("").map(normalizar_texto).astype("string")
    )

    gdf["modo_principal"] = gdf["modo_principal"].fillna("OTRO")
    gdf["tipo_infraestructura"] = gdf["tipo_infraestructura"].fillna(
        "OTRA_INFRAESTRUCTURA"
    )

    return resetear_indices(gdf)


def eliminar_duplicados_osm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = resetear_indices(gdf)

    antes = len(gdf)
    gdf = gdf.drop_duplicates(subset=["osm_type", "osm_id"], keep="first")
    gdf = resetear_indices(gdf)

    print(f"Duplicados OSM eliminados: {antes - len(gdf):,}")
    return gdf


# ============================================================================
# DEDUPLICACIÓN ESPACIAL
# ============================================================================


def deduplicar_espacialmente(
    gdf: gpd.GeoDataFrame,
    tolerancia_m: float = 75,
) -> gpd.GeoDataFrame:
    """
    Deduplicación conservadora.

    Se fusionan elementos próximos solamente cuando comparten nombre y modo.
    Si no tienen nombre, no se fusionan: de esta manera no se destruyen
    paradas diferentes que están físicamente próximas.
    """

    if gdf.empty:
        return resetear_indices(gdf)

    metric = resetear_indices(gdf.to_crs(CRS_METRICO))
    metric["geometry"] = metric.geometry.centroid
    metric = resetear_indices(metric)

    sindex = metric.sindex
    usados: set[int] = set()
    resultado: list[pd.Series] = []

    for idx in range(len(metric)):
        if idx in usados:
            continue

        row = metric.iloc[idx]
        candidatos = list(
            sindex.query(row.geometry.buffer(tolerancia_m), predicate="intersects")
        )

        grupo: list[int] = []

        for candidato in candidatos:
            candidato = int(candidato)
            if candidato in usados:
                continue

            otra = metric.iloc[candidato]
            distancia = row.geometry.distance(otra.geometry)

            mismo_modo = row["modo_principal"] == otra["modo_principal"]
            nombre_a = str(row["nombre_normalizado"] or "")
            nombre_b = str(otra["nombre_normalizado"] or "")
            mismo_nombre = bool(nombre_a) and nombre_a == nombre_b

            if distancia <= tolerancia_m and mismo_modo and mismo_nombre:
                grupo.append(candidato)

        if not grupo:
            grupo = [idx]

        usados.update(grupo)
        base = metric.iloc[grupo[0]].copy()

        ids = sorted(
            {
                str(metric.iloc[i]["osm_id"])
                for i in grupo
                if pd.notna(metric.iloc[i]["osm_id"])
            }
        )

        base["osm_ids"] = ",".join(ids)
        base["cantidad_elementos_osm"] = len(grupo)
        resultado.append(base)

    salida = gpd.GeoDataFrame(resultado, geometry="geometry", crs=CRS_METRICO)
    salida = salida.to_crs(CRS_WGS84)
    return resetear_indices(salida)


# ============================================================================
# INDICADORES DE INFRAESTRUCTURA
# ============================================================================


def calcular_indicadores(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = resetear_indices(gdf)

    for modo in MODOS_VALIDOS:
        nombre_columna = "modo_bin_" + normalizar_texto(modo)
        gdf[nombre_columna] = (gdf["modo_principal"] == modo).astype(int)

    gdf["es_modo_transporte"] = gdf["modo_principal"].isin(MODOS_VALIDOS).astype(int)
    gdf["es_intermodal"] = (gdf["categoria_intermodal"] == "INTERMODAL").astype(int)

    pesos = {
        "FERROCARRIL": 1.0,
        "SUBTE": 1.0,
        "AUTOBUS": 0.8,
        "FLUVIAL": 1.0,
        "TRANVIA": 0.8,
    }

    gdf["score_infraestructura"] = gdf["modo_principal"].map(pesos).fillna(0.0)

    return resetear_indices(gdf)


# ============================================================================
# INTERCAMBIADORES
# ============================================================================


def _componentes_proximos(
    infra: gpd.GeoDataFrame,
    radio_m: float,
) -> list[list[int]]:
    """
    Componentes conexas de elementos a <= radio_m.

    Importante: el resultado del spatial index se trata como posición iloc,
    nunca como etiqueta loc. Esto elimina el KeyError observado.
    """

    infra = resetear_indices(infra)
    n = len(infra)

    if n == 0:
        return []

    sindex = infra.sindex
    padre = list(range(n))

    def encontrar(x: int) -> int:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a: int, b: int) -> None:
        ra = encontrar(a)
        rb = encontrar(b)
        if ra != rb:
            padre[rb] = ra

    for i in range(n):
        geom = infra.iloc[i].geometry
        candidatos = list(
            sindex.query(geom.buffer(radio_m), predicate="intersects")
        )

        for j_raw in candidatos:
            j = int(j_raw)
            if j <= i:
                continue

            distancia = geom.distance(infra.iloc[j].geometry)
            if distancia <= radio_m:
                unir(i, j)

    grupos: dict[int, list[int]] = {}
    for i in range(n):
        raiz = encontrar(i)
        grupos.setdefault(raiz, []).append(i)

    return list(grupos.values())


def detectar_intercambiadores(
    infraestructura: gpd.GeoDataFrame,
    radio_m: float = RADIO_INTERCAMBIADOR_M,
) -> gpd.GeoDataFrame:
    """
    Detecta clusters de infraestructura donde existen al menos dos modos
    válidos distintos dentro del radio configurado.
    """

    infra = resetear_indices(infraestructura.to_crs(CRS_METRICO))
    infra = infra[infra["modo_principal"].isin(MODOS_VALIDOS)].copy()
    infra = resetear_indices(infra)

    if infra.empty:
        return gpd.GeoDataFrame(
            columns=["intercambiador_id", "modos", "cantidad_infraestructura", "geometry"],
            geometry="geometry",
            crs=CRS_WGS84,
        )

    grupos = _componentes_proximos(infra, radio_m)
    registros: list[dict[str, Any]] = []
    intercambiador_id = 0

    for grupo in grupos:
        sub = infra.iloc[grupo].copy()
        modos = sorted(set(sub["modo_principal"].dropna().astype(str)) & MODOS_VALIDOS)

        if len(modos) < MIN_MODOS_INTERCAMBIADOR:
            continue

        intercambiador_id += 1

        union = unary_union(list(sub.geometry))
        centroide = union.centroid

        distancias = sub.geometry.distance(centroide)
        radio_real = float(distancias.max()) if len(distancias) else 0.0

        conteo_por_modo = {
            modo: int((sub["modo_principal"] == modo).sum())
            for modo in MODOS_VALIDOS
            if int((sub["modo_principal"] == modo).sum()) > 0
        }

        nombres = []
        for nombre in sub["nombre"].dropna().astype(str):
            if nombre.strip() and nombre not in nombres:
                nombres.append(nombre)

        registros.append(
            {
                "intercambiador_id": intercambiador_id,
                "cantidad_infraestructura": len(sub),
                "cantidad_modos": len(modos),
                "modos": "|".join(modos),
                "nombre_referencias": " | ".join(nombres[:10]),
                "radio_cluster_m": radio_real,
                "ferrocarril": int("FERROCARRIL" in modos),
                "subte": int("SUBTE" in modos),
                "autobus": int("AUTOBUS" in modos),
                "fluvial": int("FLUVIAL" in modos),
                "tranvia": int("TRANVIA" in modos),
                "score_intercambiador": min(
                    100.0,
                    len(modos) / 4.0 * 70.0
                    + min(len(sub), 15) / 15.0 * 30.0,
                ),
                "geometry": centroide,
            }
        )

    if not registros:
        return gpd.GeoDataFrame(
            columns=["intercambiador_id", "modos", "geometry"],
            geometry="geometry",
            crs=CRS_METRICO,
        ).to_crs(CRS_WGS84)

    intercambiadores = gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=CRS_METRICO,
    )

    intercambiadores = intercambiadores.to_crs(CRS_WGS84)
    return resetear_indices(intercambiadores)


# ============================================================================
# CENTRALIDADES
# ============================================================================


def cargar_centralidades() -> gpd.GeoDataFrame | None:
    if not CENTRALIDADES_PATH.exists():
        print("ADVERTENCIA: no se encontró el archivo de centralidades:")
        print(CENTRALIDADES_PATH)
        return None

    try:
        centralidades = gpd.read_parquet(CENTRALIDADES_PATH)
    except Exception as exc:
        print(f"ADVERTENCIA: no se pudieron cargar centralidades: {exc}")
        return None

    if centralidades.empty:
        return None

    if centralidades.crs is None:
        centralidades = centralidades.set_crs(CRS_WGS84)

    centralidades = centralidades.to_crs(CRS_METRICO)
    centralidades = resetear_indices(centralidades)

    if "nodo_id" not in centralidades.columns:
        centralidades["nodo_id"] = np.arange(1, len(centralidades) + 1)

    return centralidades


def consultar_vecinos(
    sindex,
    geometry,
    radio_m: float,
) -> list[int]:
    """Devuelve posiciones iloc, no etiquetas de índice."""
    return [int(x) for x in sindex.query(geometry.buffer(radio_m), predicate="intersects")]


def analizar_centralidades(
    centralidades: gpd.GeoDataFrame,
    infraestructura: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    centros = resetear_indices(centralidades.to_crs(CRS_METRICO))
    infra = resetear_indices(infraestructura.to_crs(CRS_METRICO))
    interc = resetear_indices(intercambiadores.to_crs(CRS_METRICO))

    sindex_infra = infra.sindex
    sindex_interc = interc.sindex if not interc.empty else None

    resultados: list[dict[str, Any]] = []

    for pos in range(len(centros)):
        nodo = centros.iloc[pos]
        punto = nodo.geometry

        if pos == 0 or (pos + 1) % 25 == 0 or pos + 1 == len(centros):
            print(f"Centralidad {pos + 1}/{len(centros)}")

        registro: dict[str, Any] = {"nodo_id": nodo["nodo_id"]}

        for radio in RADIOS_INTERMODALIDAD_M:
            candidatos = consultar_vecinos(sindex_infra, punto, radio)

            if candidatos:
                sub = infra.iloc[candidatos]
                modos = set(sub["modo_principal"].dropna().astype(str)) & MODOS_VALIDOS
            else:
                sub = infra.iloc[[]]
                modos = set()

            registro[f"infra_{radio}m"] = len(sub)
            registro[f"modos_{radio}m"] = len(modos)

            for modo, nombre in [
                ("FERROCARRIL", "ferrocarril"),
                ("SUBTE", "subte"),
                ("AUTOBUS", "autobus"),
                ("FLUVIAL", "fluvial"),
                ("TRANVIA", "tranvia"),
            ]:
                registro[f"{nombre}_{radio}m"] = int(modo in modos)

        # Distancia a infraestructura válida más cercana dentro de 1000 m.
        candidatos_1000 = consultar_vecinos(sindex_infra, punto, RADIO_CENTRALIDAD_M)

        if candidatos_1000:
            sub = infra.iloc[candidatos_1000]
            distancias = sub.geometry.distance(punto)
            pos_min = int(np.argmin(distancias.to_numpy()))
            mas_cercano = sub.iloc[pos_min]

            registro["infraestructuras_1000m"] = len(sub)
            registro["distancia_infraestructura_m"] = float(distancias.iloc[pos_min])
            registro["infraestructura_mas_cercana"] = (
                str(mas_cercano["nombre"])
                if pd.notna(mas_cercano["nombre"]) and str(mas_cercano["nombre"]).strip()
                else str(mas_cercano["tipo_infraestructura"])
            )
            registro["modo_infraestructura_mas_cercana"] = str(
                mas_cercano["modo_principal"]
            )
        else:
            registro["infraestructuras_1000m"] = 0
            registro["distancia_infraestructura_m"] = np.nan
            registro["infraestructura_mas_cercana"] = None
            registro["modo_infraestructura_mas_cercana"] = None

        # Intercambiadores dentro de 1000 m y 500 m.
        if sindex_interc is not None:
            candidatos_interc_500 = consultar_vecinos(sindex_interc, punto, 500)
            candidatos_interc_1000 = consultar_vecinos(sindex_interc, punto, 1000)
        else:
            candidatos_interc_500 = []
            candidatos_interc_1000 = []

        registro["intercambiadores_500m"] = len(candidatos_interc_500)
        registro["intercambiadores_1000m"] = len(candidatos_interc_1000)

        if candidatos_interc_500:
            sub_i = interc.iloc[candidatos_interc_500]
            distancias_i = sub_i.geometry.distance(punto)
            pos_min_i = int(np.argmin(distancias_i.to_numpy()))
            cercano_i = sub_i.iloc[pos_min_i]
            registro["distancia_intercambiador_m"] = float(distancias_i.iloc[pos_min_i])
            registro["intercambiador_mas_cercano_id"] = safe_int(
                cercano_i["intercambiador_id"]
            )
        else:
            registro["distancia_intercambiador_m"] = np.nan
            registro["intercambiador_mas_cercano_id"] = None

        # Score 0..100.
        modos_500 = registro["modos_500m"]
        infra_500 = registro["infra_500m"]
        presencia_500 = sum(
            registro[f"{nombre}_500m"]
            for nombre in ("ferrocarril", "subte", "autobus", "fluvial", "tranvia")
        )

        score_modos = min(modos_500, 4) / 4.0 * 50.0
        score_densidad = min(infra_500, 20) / 20.0 * 20.0
        score_presencia = min(presencia_500, 4) / 4.0 * 20.0
        score_intercambiador = min(registro["intercambiadores_500m"], 2) / 2.0 * 10.0

        registro["score_intermodalidad_500m"] = round(
            min(
                100.0,
                score_modos
                + score_densidad
                + score_presencia
                + score_intercambiador,
            ),
            2,
        )

        if modos_500 >= 4 or registro["intercambiadores_500m"] >= 2:
            categoria = "INTERMODALIDAD_MUY_ALTA"
        elif modos_500 >= 3:
            categoria = "INTERMODALIDAD_ALTA"
        elif modos_500 == 2:
            categoria = "INTERMODALIDAD_MEDIA"
        elif modos_500 == 1:
            categoria = "INTERMODALIDAD_BAJA"
        else:
            categoria = "SIN_INFRAESTRUCTURA_500M"

        registro["categoria_intermodalidad_500m"] = categoria
        resultados.append(registro)

    indicadores = pd.DataFrame(resultados)

    centros = centros.merge(indicadores, on="nodo_id", how="left")

    centros["ranking_intermodalidad"] = (
        centros["score_intermodalidad_500m"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    return resetear_indices(centros)


# ============================================================================
# RESUMEN
# ============================================================================


def construir_resumen(
    infraestructura: gpd.GeoDataFrame,
    intercambiadores: gpd.GeoDataFrame,
    centralidades: gpd.GeoDataFrame | None,
) -> dict[str, Any]:
    resumen: dict[str, Any] = {
        "proyecto": "Análisis de movilidad SUBE AMBA",
        "script": "21_construir_infraestructura_intermodal_amba.py",
        "version": "2.0",
        "fecha_ejecucion": pd.Timestamp.now().isoformat(),
        "bbox": AMBA_BBOX,
        "crs_original": CRS_WGS84,
        "crs_metrico": CRS_METRICO,
        "radio_intercambiador_m": RADIO_INTERCAMBIADOR_M,
        "infraestructuras": int(len(infraestructura)),
        "intercambiadores": int(len(intercambiadores)),
        "osm_ids_unicos": int(infraestructura["osm_id"].nunique()),
    }

    for columna, clave in [
        ("tipo_infraestructura", "tipos"),
        ("modo_principal", "modos"),
        ("categoria_intermodal", "categorias"),
    ]:
        conteo = infraestructura[columna].fillna("SIN_DATO").value_counts().to_dict()
        resumen[clave] = {str(k): int(v) for k, v in conteo.items()}

    if not intercambiadores.empty:
        resumen["intercambiadores_por_cantidad_de_modos"] = {
            str(k): int(v)
            for k, v in intercambiadores["cantidad_modos"].value_counts().to_dict().items()
        }
        resumen["top_10_intercambiadores"] = []
        for _, row in intercambiadores.sort_values(
            "score_intercambiador", ascending=False
        ).head(10).iterrows():
            resumen["top_10_intercambiadores"].append(
                {
                    "intercambiador_id": safe_int(row["intercambiador_id"]),
                    "score": safe_float(row["score_intercambiador"]),
                    "cantidad_infraestructura": safe_int(row["cantidad_infraestructura"]),
                    "cantidad_modos": safe_int(row["cantidad_modos"]),
                    "modos": str(row["modos"]),
                    "nombre_referencias": str(row["nombre_referencias"]),
                }
            )

    if centralidades is not None:
        resumen["centralidades_analizadas"] = int(len(centralidades))

        if "categoria_intermodalidad_500m" in centralidades.columns:
            resumen["centralidades_por_intermodalidad"] = {
                str(k): int(v)
                for k, v in centralidades["categoria_intermodalidad_500m"]
                .value_counts()
                .to_dict()
                .items()
            }

        if "score_intermodalidad_500m" in centralidades.columns:
            resumen["top_10_centralidades_intermodalidad"] = []
            top = centralidades.sort_values(
                "score_intermodalidad_500m", ascending=False
            ).head(10)

            for _, row in top.iterrows():
                resumen["top_10_centralidades_intermodalidad"].append(
                    {
                        "nodo_id": safe_int(row["nodo_id"]),
                        "score": safe_float(row["score_intermodalidad_500m"]),
                        "infra_500m": safe_int(row["infra_500m"]),
                        "modos_500m": safe_int(row["modos_500m"]),
                        "intercambiadores_500m": safe_int(row["intercambiadores_500m"]),
                        "categoria": str(row["categoria_intermodalidad_500m"]),
                    }
                )

    return resumen


# ============================================================================
# VALIDACIONES
# ============================================================================


def validar_infraestructura(gdf: gpd.GeoDataFrame, etiqueta: str) -> None:
    subtitulo(etiqueta)

    gdf = resetear_indices(gdf)

    nulos = int(gdf.geometry.isna().sum())
    vacios = int(gdf.geometry.is_empty.sum())
    invalidos = int((~gdf.geometry.is_valid).sum())
    duplicados = int(gdf["osm_id"].duplicated().sum())

    print(f"Registros: {len(gdf):,}")
    print(f"Geometrías nulas: {nulos:,}")
    print(f"Geometrías vacías: {vacios:,}")
    print(f"Geometrías inválidas: {invalidos:,}")
    print(f"Duplicados OSM: {duplicados:,}")

    if nulos or vacios or invalidos:
        raise RuntimeError("La infraestructura contiene geometrías inválidas.")


def validar_centralidades(centralidades: gpd.GeoDataFrame) -> None:
    subtitulo("VALIDACIÓN DE CENTRALIDADES")
    duplicados = int(centralidades["nodo_id"].duplicated().sum())
    invalidas = int((~centralidades.geometry.is_valid).sum())
    print(f"Centralidades: {len(centralidades):,}")
    print(f"nodo_id duplicados: {duplicados:,}")
    print(f"Geometrías inválidas: {invalidas:,}")

    if duplicados or invalidas:
        raise RuntimeError("Las centralidades no superan la validación.")


# ============================================================================
# SALIDAS
# ============================================================================


def guardar_infraestructura(infraestructura: gpd.GeoDataFrame) -> None:
    parquet_path = OUTPUT_DIR / "infraestructura_intermodal_amba.parquet"
    gpkg_path = OUTPUT_DIR / "infraestructura_intermodal_amba.gpkg"

    print("Guardando GeoParquet...")
    infraestructura.to_parquet(parquet_path, index=False)
    print(parquet_path)

    print("Guardando GeoPackage...")
    infraestructura.to_file(
        gpkg_path,
        layer="infraestructura",
        driver="GPKG",
    )
    print(gpkg_path)


def guardar_intercambiadores(intercambiadores: gpd.GeoDataFrame) -> None:
    parquet_path = OUTPUT_DIR / "intercambiadores_intermodales_amba.parquet"
    gpkg_path = OUTPUT_DIR / "infraestructura_intermodal_amba.gpkg"

    intercambiadores.to_parquet(parquet_path, index=False)
    intercambiadores.to_file(
        gpkg_path,
        layer="intercambiadores_intermodales",
        driver="GPKG",
    )

    print(f"Intercambiadores:\n{parquet_path}")


def guardar_centralidades(centralidades: gpd.GeoDataFrame) -> None:
    parquet_path = OUTPUT_DIR / "centralidades_intermodalidad_amba.parquet"
    gpkg_path = OUTPUT_DIR / "infraestructura_intermodal_amba.gpkg"

    centralidades.to_parquet(parquet_path, index=False)
    centralidades.to_file(
        gpkg_path,
        layer="centralidades_intermodalidad",
        driver="GPKG",
    )

    print(f"Centralidades:\n{parquet_path}")


def guardar_csv(infraestructura: gpd.GeoDataFrame) -> None:
    tabla = infraestructura.drop(columns="geometry", errors="ignore")
    path = OUTPUT_DIR / "infraestructura_intermodal_amba.csv"
    tabla.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV:\n{path}")


def guardar_csv_intercambiadores(intercambiadores: gpd.GeoDataFrame) -> None:
    tabla = intercambiadores.drop(columns="geometry", errors="ignore")
    path = OUTPUT_DIR / "intercambiadores_intermodales_amba.csv"
    tabla.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV intercambiadores:\n{path}")


def guardar_resumen(resumen: dict[str, Any]) -> None:
    path = OUTPUT_DIR / "infraestructura_intermodal_amba_resumen.json"
    with path.open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)
    print(f"JSON:\n{path}")


# ============================================================================
# MAPAS Y GRÁFICOS
# ============================================================================


def importar_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ADVERTENCIA: Matplotlib no disponible. Se omiten gráficos.")
        return None


def generar_mapa_infraestructura(infraestructura: gpd.GeoDataFrame) -> None:
    plt = importar_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(14, 11))
    infraestructura.plot(
        ax=ax,
        markersize=5,
        alpha=0.55,
        column="modo_principal",
        legend=True,
    )
    ax.set_title("Infraestructura de transporte - AMBA", fontsize=15)
    ax.set_axis_off()

    path = OUTPUT_DIR / "01_mapa_infraestructura_intermodal.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa:\n{path}")


def generar_mapa_intercambiadores(intercambiadores: gpd.GeoDataFrame) -> None:
    if intercambiadores.empty:
        print("No hay intercambiadores para mapear.")
        return

    plt = importar_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(14, 11))
    intercambiadores.plot(
        ax=ax,
        column="cantidad_modos",
        markersize=30,
        legend=True,
    )
    ax.set_title("Intercambiadores intermodales - AMBA", fontsize=15)
    ax.set_axis_off()

    path = OUTPUT_DIR / "05_mapa_intercambiadores_intermodales.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa:\n{path}")


def generar_mapa_centralidades(centralidades: gpd.GeoDataFrame) -> None:
    if centralidades.empty or "score_intermodalidad_500m" not in centralidades.columns:
        return

    plt = importar_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(14, 11))
    centralidades.plot(
        ax=ax,
        column="score_intermodalidad_500m",
        cmap="viridis",
        markersize=35,
        legend=True,
    )
    ax.set_title("Centralidades SUBE - Intermodalidad a 500 m", fontsize=15)
    ax.set_axis_off()

    path = OUTPUT_DIR / "02_centralidades_intermodalidad_500m.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa:\n{path}")


def generar_grafico_modos(infraestructura: gpd.GeoDataFrame) -> None:
    plt = importar_matplotlib()
    if plt is None:
        return

    conteo = (
        infraestructura[infraestructura["modo_principal"].isin(MODOS_VALIDOS)]["modo_principal"]
        .value_counts()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    conteo.plot(kind="barh", ax=ax)
    ax.set_title("Infraestructura por modo de transporte", fontsize=14)
    ax.set_xlabel("Cantidad de elementos")
    ax.set_ylabel("Modo")

    path = OUTPUT_DIR / "03_infraestructura_por_modo.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico:\n{path}")


def generar_grafico_intermodalidad(centralidades: gpd.GeoDataFrame) -> None:
    if centralidades.empty or "score_intermodalidad_500m" not in centralidades.columns:
        return

    plt = importar_matplotlib()
    if plt is None:
        return

    valores = centralidades["score_intermodalidad_500m"].dropna()
    if valores.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(valores, bins=15)
    ax.set_title("Distribución del score de intermodalidad", fontsize=14)
    ax.set_xlabel("Score")
    ax.set_ylabel("Cantidad de nodos")

    path = OUTPUT_DIR / "04_distribucion_score_intermodalidad.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico:\n{path}")


def generar_grafico_intercambiadores(intercambiadores: gpd.GeoDataFrame) -> None:
    if intercambiadores.empty:
        return

    plt = importar_matplotlib()
    if plt is None:
        return

    conteo = intercambiadores["cantidad_modos"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    conteo.plot(kind="bar", ax=ax)
    ax.set_title("Intercambiadores por cantidad de modos", fontsize=14)
    ax.set_xlabel("Cantidad de modos")
    ax.set_ylabel("Cantidad de intercambiadores")

    path = OUTPUT_DIR / "06_intercambiadores_por_modos.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico:\n{path}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    titulo("21 - CONSTRUCCIÓN DE INFRAESTRUCTURA INTERMODAL AMBA")
    print(f"Proyecto : {PROJECT_ROOT}")
    print(f"Salida   : {OUTPUT_DIR}")
    print(f"BBOX     : {AMBA_BBOX}")
    print(f"CRS      : {CRS_WGS84}")
    print(f"CRS métrico: {CRS_METRICO}")
    print(f"Radio intercambiador: {RADIO_INTERCAMBIADOR_M} m")

    # ------------------------------------------------------------------
    # 1. OVERPASS
    # ------------------------------------------------------------------
    titulo("1. CONSULTANDO OPENSTREETMAP / OVERPASS")
    datos = consultar_overpass()

    # ------------------------------------------------------------------
    # 2. CONSTRUCCIÓN
    # ------------------------------------------------------------------
    titulo("2. CONSTRUYENDO INVENTARIO")
    infraestructura = construir_gdf(datos)
    print(f"Elementos construidos: {len(infraestructura):,}")

    infraestructura = normalizar_infraestructura(infraestructura)
    validar_infraestructura(infraestructura, "VALIDACIÓN DE INFRAESTRUCTURA INICIAL")

    # ------------------------------------------------------------------
    # 3. DUPLICADOS OSM
    # ------------------------------------------------------------------
    titulo("3. ELIMINANDO DUPLICADOS OSM")
    infraestructura = eliminar_duplicados_osm(infraestructura)
    validar_infraestructura(infraestructura, "VALIDACIÓN DE INFRAESTRUCTURA POST OSM")

    # ------------------------------------------------------------------
    # 4. DEDUPLICACIÓN ESPACIAL
    # ------------------------------------------------------------------
    titulo("4. DEDUPLICACIÓN ESPACIAL")
    antes = len(infraestructura)
    infraestructura = deduplicar_espacialmente(infraestructura, tolerancia_m=75)
    despues = len(infraestructura)
    print(f"Antes     : {antes:,}")
    print(f"Después   : {despues:,}")
    print(f"Reducidos : {antes - despues:,}")

    infraestructura = resetear_indices(infraestructura)

    # ------------------------------------------------------------------
    # 5. INDICADORES
    # ------------------------------------------------------------------
    titulo("5. CALCULANDO INDICADORES")
    infraestructura = calcular_indicadores(infraestructura)
    validar_infraestructura(infraestructura, "VALIDACIÓN DE INFRAESTRUCTURA FINAL")

    # ------------------------------------------------------------------
    # 6. RESUMEN
    # ------------------------------------------------------------------
    subtitulo("RESUMEN DEL INVENTARIO")
    print(f"Total: {len(infraestructura):,}")

    print("\nPOR MODO")
    for modo, cantidad in infraestructura["modo_principal"].value_counts().items():
        print(f"  {str(modo):20s} {int(cantidad):8,d}")

    print("\nPOR TIPO")
    for tipo, cantidad in infraestructura["tipo_infraestructura"].value_counts().items():
        print(f"  {str(tipo):35s} {int(cantidad):8,d}")

    # ------------------------------------------------------------------
    # 7. INTERCAMBIADORES
    # ------------------------------------------------------------------
    titulo("7. DETECTANDO INTERCAMBIADORES INTERMODALES")
    intercambiadores = detectar_intercambiadores(
        infraestructura,
        radio_m=RADIO_INTERCAMBIADOR_M,
    )

    print(f"Intercambiadores detectados: {len(intercambiadores):,}")

    if not intercambiadores.empty:
        print("\nTOP 20 INTERCAMBIADORES")
        columnas = [
            "intercambiador_id",
            "score_intercambiador",
            "cantidad_infraestructura",
            "cantidad_modos",
            "modos",
            "nombre_referencias",
        ]
        disponibles = [c for c in columnas if c in intercambiadores.columns]
        print(
            intercambiadores[disponibles]
            .sort_values("score_intercambiador", ascending=False)
            .head(20)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 8. CENTRALIDADES
    # ------------------------------------------------------------------
    titulo("8. CARGANDO CENTRALIDADES SUBE")
    centralidades = cargar_centralidades()

    if centralidades is not None:
        validar_centralidades(centralidades)
        print(f"Centralidades cargadas: {len(centralidades):,}")

        titulo("9. CALCULANDO INTERMODALIDAD DE CENTRALIDADES")
        centralidades = analizar_centralidades(
            centralidades,
            infraestructura,
            intercambiadores,
        )

        print("\nTOP 15 CENTRALIDADES POR INTERMODALIDAD")
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
            "intercambiadores_500m",
            "categoria_intermodalidad_500m",
        ]
        disponibles = [c for c in columnas if c in centralidades.columns]
        print(
            centralidades[disponibles]
            .sort_values("score_intermodalidad_500m", ascending=False)
            .head(15)
            .to_string(index=False)
        )
    else:
        print("No se realizó el cruce con centralidades.")

    # ------------------------------------------------------------------
    # 10. RESUMEN JSON
    # ------------------------------------------------------------------
    titulo("10. CONSTRUYENDO RESUMEN JSON")
    resumen = construir_resumen(
        infraestructura,
        intercambiadores,
        centralidades,
    )

    # ------------------------------------------------------------------
    # 11. GUARDAR
    # ------------------------------------------------------------------
    titulo("11. GUARDANDO ARCHIVOS")
    guardar_infraestructura(infraestructura)
    guardar_intercambiadores(intercambiadores)
    guardar_csv(infraestructura)
    guardar_csv_intercambiadores(intercambiadores)

    if centralidades is not None:
        guardar_centralidades(centralidades)

    guardar_resumen(resumen)

    # ------------------------------------------------------------------
    # 12. GRÁFICOS
    # ------------------------------------------------------------------
    titulo("12. GENERANDO MAPAS Y GRÁFICOS")
    generar_mapa_infraestructura(infraestructura)
    generar_grafico_modos(infraestructura)
    generar_mapa_intercambiadores(intercambiadores)
    generar_grafico_intercambiadores(intercambiadores)

    if centralidades is not None:
        generar_mapa_centralidades(centralidades)
        generar_grafico_intermodalidad(centralidades)

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------
    titulo("21 - PROCESO FINALIZADO")
    print(f"Infraestructuras finales: {len(infraestructura):,}")
    print(f"Intercambiadores: {len(intercambiadores):,}")

    if centralidades is not None:
        print(f"Centralidades analizadas: {len(centralidades):,}")

    print("\nARCHIVOS GENERADOS")
    for archivo in sorted(OUTPUT_DIR.iterdir()):
        if archivo.is_file():
            print(f"  {archivo.name}")

    print("\nSIGUIENTE ETAPA")
    print("Cruzar:")
    print("  SUBE 2025")
    print("  + centralidades")
    print("  + infraestructura intermodal")
    print("  + intercambiadores")
    print("  + red vial")
    print("para construir el índice de centralidad estructural.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProceso cancelado por el usuario.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 78)
        print("ERROR FATAL")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Revisá el mensaje anterior para identificar el paso donde ocurrió el error.")
        sys.exit(1)