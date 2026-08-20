import json
from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "red_transporte_amba"
)

OUTPUT_PARQUET = OUTPUT_DIR / "red_vial_amba.parquet"
OUTPUT_GPKG = OUTPUT_DIR / "red_vial_amba.gpkg"
OUTPUT_RESUMEN = OUTPUT_DIR / "red_vial_amba_resumen.json"

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:3857"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Área aproximada del AMBA mediante bounding box.
# Sur, Oeste, Norte, Este.
AMBA_BBOX = "-35.20,-59.40,-34.20,-57.80"

# Jerarquía vial que utilizaremos para el análisis estructural.
HIGHWAY_TYPES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
]

JERARQUIA_VIAL = {
    "motorway": 5,
    "trunk": 4,
    "primary": 3,
    "secondary": 2,
    "tertiary": 1,
}

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================


def construir_query_overpass():
    tipos = "|".join(HIGHWAY_TYPES)

    return f"""
[out:json][timeout:300];
(
  way["highway"~"^({tipos})$"]({AMBA_BBOX});
);
out geom tags;
"""


def descargar_red():
    print()
    print("Consultando OpenStreetMap / Overpass...")
    print(f"Endpoint: {OVERPASS_URL}")
    print(f"BBOX AMBA: {AMBA_BBOX}")
    print(f"Jerarquías: {', '.join(HIGHWAY_TYPES)}")

    query = construir_query_overpass()

    try:
        respuesta = requests.post(
            OVERPASS_URL,
            data=query.encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "analisis-sube-2025/1.0",
            },
            timeout=360,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "No fue posible conectarse con Overpass.\n"
            f"Motivo: {error}"
        ) from error

    print(f"HTTP: {respuesta.status_code}")

    if respuesta.status_code != 200:
        texto = respuesta.text[:2000]
        raise RuntimeError(
            "Overpass devolvió un error HTTP.\n"
            f"Status: {respuesta.status_code}\n"
            f"Respuesta: {texto}"
        )

    try:
        datos = respuesta.json()
    except ValueError as error:
        raise RuntimeError(
            "La respuesta de Overpass no contiene JSON válido."
        ) from error

    elementos = datos.get("elements", [])

    print(f"Elementos recibidos: {len(elementos):,}")

    if not elementos:
        raise ValueError(
            "Overpass no devolvió elementos de red vial."
        )

    return elementos


def convertir_elementos_a_geodataframe(elementos):
    print()
    print("Construyendo GeoDataFrame...")

    registros = []

    for elemento in elementos:
        if elemento.get("type") != "way":
            continue

        way_id = elemento.get("id")
        tags = elemento.get("tags", {}) or {}
        geometry = elemento.get("geometry", []) or []

        if not way_id or len(geometry) < 2:
            continue

        coordenadas = []

        for punto in geometry:
            lat = punto.get("lat")
            lon = punto.get("lon")

            if lat is None or lon is None:
                continue

            try:
                coordenadas.append((float(lon), float(lat)))
            except (TypeError, ValueError):
                continue

        if len(coordenadas) < 2:
            continue

        try:
            geometria = LineString(coordenadas)
        except Exception:
            continue

        if geometria.is_empty or not geometria.is_valid:
            continue

        highway = str(tags.get("highway", "")).strip().lower()

        if highway not in HIGHWAY_TYPES:
            continue

        registros.append(
            {
                "osm_id": int(way_id),
                "highway": highway,
                "name": tags.get("name"),
                "ref": tags.get("ref"),
                "maxspeed": tags.get("maxspeed"),
                "lanes": tags.get("lanes"),
                "oneway": tags.get("oneway"),
                "surface": tags.get("surface"),
                "bridge": tags.get("bridge"),
                "tunnel": tags.get("tunnel"),
                "access": tags.get("access"),
                "geometry": geometria,
            }
        )

    if not registros:
        raise ValueError(
            "No fue posible construir geometrías LineString."
        )

    gdf = gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=CRS_GEOGRAFICO,
    )

    print(f"Segmentos construidos: {len(gdf):,}")

    return gdf


def normalizar_atributos(gdf):
    print()
    print("Normalizando atributos...")

    gdf["jerarquia_vial"] = (
        gdf["highway"]
        .map(JERARQUIA_VIAL)
        .fillna(0)
        .astype("int16")
    )

    gdf["longitud_km"] = (
        gdf.to_crs(CRS_METRICO)
        .geometry.length
        .div(1000.0)
    )

    # Conversión conservadora de atributos numéricos.
    gdf["lanes"] = pd.to_numeric(
        gdf["lanes"],
        errors="coerce",
    )

    # maxspeed puede contener valores como:
    # "80", "60 km/h", "30 mph", etc.
    gdf["maxspeed_kmh"] = (
        gdf["maxspeed"]
        .astype("string")
        .str.extract(r"(\d+(?:[.,]\d+)?)", expand=False)
    )

    gdf["maxspeed_kmh"] = pd.to_numeric(
        gdf["maxspeed_kmh"],
        errors="coerce",
    )

    # Longitud válida.
    gdf = gdf[
        gdf["longitud_km"].notna()
        & (gdf["longitud_km"] > 0)
    ].copy()

    return gdf


def eliminar_duplicados(gdf):
    print()
    print("Validando duplicados...")

    duplicados_osm = int(
        gdf["osm_id"].duplicated().sum()
    )

    print(f"OSM IDs duplicados: {duplicados_osm:,}")

    if duplicados_osm:
        # Un mismo OSM way debería aparecer una sola vez.
        gdf = gdf.drop_duplicates(
            subset=["osm_id"],
            keep="first",
        ).copy()

        print(
            "Duplicados eliminados: "
            f"{duplicados_osm:,}"
        )

    return gdf


def validar_geometrias(gdf):
    print()
    print("Validando geometrías...")

    mascara = (
        gdf["geometry"].notna()
        & ~gdf["geometry"].is_empty
        & gdf["geometry"].is_valid
    )

    validas = int(mascara.sum())
    invalidas = int((~mascara).sum())

    print(f"Geometrías válidas: {validas:,}")
    print(f"Geometrías inválidas: {invalidas:,}")

    gdf = gdf[mascara].copy()

    if gdf.empty:
        raise ValueError(
            "No quedaron geometrías válidas."
        )

    return gdf


def generar_resumen(gdf, elementos_originales):
    resumen_por_tipo = {}

    for tipo in HIGHWAY_TYPES:
        grupo = gdf[gdf["highway"] == tipo]

        resumen_por_tipo[tipo] = {
            "segmentos": int(len(grupo)),
            "longitud_km": float(
                grupo["longitud_km"].sum()
            ),
        }

    return {
        "fuente": {
            "openstreetmap": True,
            "overpass_url": OVERPASS_URL,
            "bbox": AMBA_BBOX,
            "elementos_recibidos": int(
                len(elementos_originales)
            ),
        },
        "red": {
            "segmentos": int(len(gdf)),
            "longitud_total_km": float(
                gdf["longitud_km"].sum()
            ),
            "osm_ids_unicos": int(
                gdf["osm_id"].nunique()
            ),
            "jerarquia_maxima": int(
                gdf["jerarquia_vial"].max()
            ),
        },
        "por_tipo": resumen_por_tipo,
        "crs": CRS_GEOGRAFICO,
        "crs_metrico_calculo_longitud": CRS_METRICO,
    }


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    print("=" * 70)
    print("CONSTRUCCIÓN DE RED DE TRANSPORTE AMBA")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    elementos = descargar_red()

    red = convertir_elementos_a_geodataframe(elementos)
    red = validar_geometrias(red)
    red = normalizar_atributos(red)
    red = eliminar_duplicados(red)
    red = validar_geometrias(red)

    columnas = [
        "osm_id",
        "highway",
        "jerarquia_vial",
        "name",
        "ref",
        "maxspeed",
        "maxspeed_kmh",
        "lanes",
        "oneway",
        "surface",
        "bridge",
        "tunnel",
        "access",
        "longitud_km",
        "geometry",
    ]

    red = red[[c for c in columnas if c in red.columns]].copy()

    red = red.sort_values(
        ["jerarquia_vial", "longitud_km", "osm_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    resumen = generar_resumen(red, elementos)

    print()
    print("=" * 70)
    print("RESUMEN DE RED")
    print("=" * 70)

    print(f"\nSegmentos: {len(red):,}")
    print(f"Longitud total: {red['longitud_km'].sum():,.2f} km")
    print(f"OSM IDs únicos: {red['osm_id'].nunique():,}")

    print("\nPor jerarquía:")
    for tipo in HIGHWAY_TYPES:
        grupo = red[red["highway"] == tipo]
        print(
            f"  {tipo:<10} "
            f"{len(grupo):>7,} segmentos  "
            f"{grupo['longitud_km'].sum():>10,.2f} km"
        )

    print()
    print("Guardando GeoParquet...")
    red.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"GeoParquet guardado:\n{OUTPUT_PARQUET}")

    print()
    print("Guardando GeoPackage...")
    try:
        red.to_file(OUTPUT_GPKG, layer="red_vial_amba", driver="GPKG")
        print(f"GeoPackage guardado:\n{OUTPUT_GPKG}")
    except Exception as error:
        print("ADVERTENCIA: no se pudo guardar el GeoPackage.")
        print(f"Motivo: {error}")

    print()
    print("Guardando resumen JSON...")
    with open(OUTPUT_RESUMEN, "w", encoding="utf-8") as archivo:
        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    print(f"Resumen guardado:\n{OUTPUT_RESUMEN}")

    print()
    print("=" * 70)
    print("CONSTRUCCIÓN DE RED FINALIZADA")
    print("=" * 70)

    print(f"\nSegmentos analizados: {len(red):,}")
    print(f"Longitud total: {red['longitud_km'].sum():,.2f} km")

    print("\nArchivos generados:")
    print(f"  {OUTPUT_PARQUET}")
    if OUTPUT_GPKG.exists():
        print(f"  {OUTPUT_GPKG}")
    print(f"  {OUTPUT_RESUMEN}")


if __name__ == "__main__":
    main()