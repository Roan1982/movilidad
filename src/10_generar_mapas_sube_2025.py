from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_perfil_h3.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "maps_sube_2025"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CARGA
# ============================================================

print("=" * 70)
print("GENERACIÓN DE MAPAS SUBE 2025")
print("=" * 70)

print("\n" + "=" * 70)
print("1. CARGANDO DATOS")
print("=" * 70)

print(f"\nArchivo:")
print(INPUT_FILE)

gdf = gpd.read_parquet(INPUT_FILE)

print(f"\nH3 cargados: {len(gdf):,}")
print(f"CRS: {gdf.crs}")


# ============================================================
# VALIDACIÓN
# ============================================================

print("\n" + "=" * 70)
print("2. VALIDACIÓN")
print("=" * 70)

required_columns = [
    "id_h3",
    "geometry",
    "operaciones_totales",
    "categoria_demanda",
    "hora_pico",
    "modo_dominante",
    "perfil_temporal",
    "prioridad_analitica",
    "perfil_nodo",
]

missing_columns = [
    column
    for column in required_columns
    if column not in gdf.columns
]

if missing_columns:
    raise ValueError(
        "Faltan columnas requeridas:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )

print("Columnas requeridas: OK")

null_geometry = gdf.geometry.isna().sum()

print(f"Geometrías nulas: {null_geometry:,}")

if null_geometry > 0:
    print(
        "\nADVERTENCIA:"
        f" se eliminarán {null_geometry:,} registros sin geometría."
    )

    gdf = gdf.loc[
        gdf.geometry.notna()
    ].copy()

print(
    f"H3 utilizados para mapas: {len(gdf):,}"
)


# ============================================================
# VALIDACIÓN CRS
# ============================================================

if gdf.crs is None:
    raise ValueError(
        "El dataset no tiene CRS definido."
    )

print(f"CRS: {gdf.crs}")


# ============================================================
# EXTENSIÓN DEL MAPA
# ============================================================

print("\n" + "=" * 70)
print("3. PREPARANDO EXTENSIÓN CARTOGRÁFICA")
print("=" * 70)

minx, miny, maxx, maxy = gdf.total_bounds

margin_x = (
    maxx - minx
) * 0.03

margin_y = (
    maxy - miny
) * 0.03

MAP_XLIM = (
    minx - margin_x,
    maxx + margin_x,
)

MAP_YLIM = (
    miny - margin_y,
    maxy + margin_y,
)

print(
    f"Extensión X: {MAP_XLIM}"
)

print(
    f"Extensión Y: {MAP_YLIM}"
)


# ============================================================
# FUNCIÓN GENERAL
# ============================================================

def guardar_mapa(
    data,
    column,
    title,
    filename,
    cmap="viridis",
    legend=True,
    categorical=False,
    figsize=(14, 12),
):
    print(
        f"\nGenerando mapa: {filename}"
    )

    if column not in data.columns:
        print(
            f"ADVERTENCIA: columna "
            f"'{column}' no existe."
        )
        return

    fig, ax = plt.subplots(
        figsize=figsize
    )

    data.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        legend=legend,
        categorical=categorical,
        linewidth=0,
        edgecolor="none",
    )

    ax.set_title(
        title,
        fontsize=16,
        pad=15,
    )

    ax.set_xlim(
        MAP_XLIM
    )

    ax.set_ylim(
        MAP_YLIM
    )

    ax.set_axis_off()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Guardado: {output}"
    )


# ============================================================
# 4. DEMANDA TOTAL
# ============================================================

print("\n" + "=" * 70)
print("4. MAPA DE DEMANDA TOTAL")
print("=" * 70)

guardar_mapa(
    gdf,
    "operaciones_totales",
    "SUBE 2025 — Demanda total por H3",
    "01_demanda_total.png",
    cmap="YlOrRd",
)


# ============================================================
# 5. CATEGORÍA DE DEMANDA
# ============================================================

print("\n" + "=" * 70)
print("5. MAPA DE HOTSPOTS")
print("=" * 70)

guardar_mapa(
    gdf,
    "categoria_demanda",
    "SUBE 2025 — Clasificación de demanda espacial",
    "02_hotspots.png",
    cmap="YlOrRd",
    categorical=True,
)


# ============================================================
# 6. HORA PICO
# ============================================================

print("\n" + "=" * 70)
print("6. MAPA DE HORA PICO")
print("=" * 70)

guardar_mapa(
    gdf,
    "hora_pico",
    "SUBE 2025 — Hora pico por H3",
    "03_hora_pico.png",
    cmap="twilight",
)


# ============================================================
# 7. MODO DOMINANTE
# ============================================================

print("\n" + "=" * 70)
print("7. MAPA DE MODO DOMINANTE")
print("=" * 70)

guardar_mapa(
    gdf,
    "modo_dominante",
    "SUBE 2025 — Modo de transporte dominante",
    "04_modo_dominante.png",
    cmap="Set2",
    categorical=True,
)


# ============================================================
# 8. PERFIL TEMPORAL
# ============================================================

print("\n" + "=" * 70)
print("8. MAPA DE PERFIL TEMPORAL")
print("=" * 70)

guardar_mapa(
    gdf,
    "perfil_temporal",
    "SUBE 2025 — Perfil temporal dominante",
    "05_perfil_temporal.png",
    cmap="Set3",
    categorical=True,
)


# ============================================================
# 9. PRIORIDAD ANALÍTICA
# ============================================================

print("\n" + "=" * 70)
print("9. MAPA DE PRIORIDAD")
print("=" * 70)

guardar_mapa(
    gdf,
    "prioridad_analitica",
    "SUBE 2025 — Prioridad territorial",
    "06_prioridad.png",
    cmap="YlOrRd",
    categorical=True,
)


# ============================================================
# 10. PERFIL COMBINADO
# ============================================================

print("\n" + "=" * 70)
print("10. MAPA DE PERFIL COMBINADO")
print("=" * 70)

# Para evitar una leyenda gigantesca,
# mostramos únicamente los perfiles más frecuentes.

perfil_counts = (
    gdf["perfil_nodo"]
    .value_counts()
)

top_perfiles = (
    perfil_counts
    .head(10)
    .index
)

gdf["perfil_nodo_mapa"] = (
    gdf["perfil_nodo"]
    .where(
        gdf["perfil_nodo"]
        .isin(top_perfiles),
        "OTROS",
    )
)

guardar_mapa(
    gdf,
    "perfil_nodo_mapa",
    "SUBE 2025 — Perfil territorial de demanda",
    "07_perfil_nodo.png",
    cmap="tab20",
    categorical=True,
)


# ============================================================
# 11. CONCENTRACIÓN MODAL
# ============================================================

print("\n" + "=" * 70)
print("11. MAPA DE DOMINANCIA MODAL")
print("=" * 70)

guardar_mapa(
    gdf,
    "indice_dominancia_modal",
    "SUBE 2025 — Concentración del modo dominante",
    "08_dominancia_modal.png",
    cmap="viridis",
)


# ============================================================
# 12. DIVERSIDAD MODAL
# ============================================================

print("\n" + "=" * 70)
print("12. MAPA DE DIVERSIDAD MODAL")
print("=" * 70)

guardar_mapa(
    gdf,
    "cantidad_modos",
    "SUBE 2025 — Cantidad de modos por H3",
    "09_diversidad_modal.png",
    cmap="Blues",
)


# ============================================================
# 13. TOP 100
# ============================================================

print("\n" + "=" * 70)
print("13. TOP 100 H3")
print("=" * 70)

top100 = (
    gdf
    .sort_values(
        "operaciones_totales",
        ascending=False,
    )
    .head(100)
    .copy()
)

top100_columns = [
    "id_h3",
    "operaciones_totales",
    "ranking_perfil",
    "percentil_demanda",
    "categoria_demanda",
    "hora_pico",
    "operaciones_hora_pico",
    "pct_hora_pico",
    "franja_dominante",
    "perfil_temporal",
    "modo_dominante",
    "pct_modo_dominante",
    "cantidad_modos",
    "indice_dominancia_modal",
    "perfil_nodo",
    "prioridad_analitica",
]

top100_columns = [
    column
    for column in top100_columns
    if column in top100.columns
]

top100[
    top100_columns
].to_csv(
    OUTPUT_DIR
    / "top100_h3.csv",
    index=False,
)

print(
    "Top 100 guardado."
)


# ============================================================
# 14. EXPORTACIÓN GEOJSON
# ============================================================

print("\n" + "=" * 70)
print("14. EXPORTANDO GEOJSON")
print("=" * 70)

geojson_columns = [
    "id_h3",
    "operaciones_totales",
    "ranking_perfil",
    "percentil_demanda",
    "categoria_demanda",
    "hora_pico",
    "operaciones_hora_pico",
    "pct_hora_pico",
    "franja_dominante",
    "perfil_temporal",
    "modo_dominante",
    "pct_modo_dominante",
    "cantidad_modos",
    "indice_dominancia_modal",
    "perfil_nodo",
    "prioridad_analitica",
    "geometry",
]

geojson_columns = [
    column
    for column in geojson_columns
    if column in gdf.columns
]

gdf_geojson = gdf[
    geojson_columns
].copy()

geojson_output = (
    OUTPUT_DIR
    / "h3_sube_2025.geojson"
)

gdf_geojson.to_file(
    geojson_output,
    driver="GeoJSON",
)

print(
    f"GeoJSON guardado: "
    f"{geojson_output}"
)


# ============================================================
# 15. GEOJSON HOTSPOTS
# ============================================================

print("\nExportando hotspots...")

hotspots = gdf.loc[
    gdf["categoria_demanda"]
    .isin(
        [
            "HOTSPOT_EXTREMO",
            "HOTSPOT_ALTO",
        ]
    )
].copy()

hotspots_output = (
    OUTPUT_DIR
    / "h3_hotspots_sube_2025.geojson"
)

hotspots[
    geojson_columns
].to_file(
    hotspots_output,
    driver="GeoJSON",
)

print(
    f"Hotspots: {len(hotspots):,}"
)

print(
    f"Archivo: {hotspots_output}"
)


# ============================================================
# 16. GEOJSON PRIORIDAD
# ============================================================

print("\nExportando zonas prioritarias...")

prioridad = gdf.loc[
    gdf["prioridad_analitica"]
    .isin(
        [
            "PRIORIDAD_MUY_ALTA",
            "PRIORIDAD_ALTA",
        ]
    )
].copy()

prioridad_output = (
    OUTPUT_DIR
    / "h3_prioridad_sube_2025.geojson"
)

prioridad[
    geojson_columns
].to_file(
    prioridad_output,
    driver="GeoJSON",
)

print(
    f"Zonas prioritarias: "
    f"{len(prioridad):,}"
)

print(
    f"Archivo: {prioridad_output}"
)


# ============================================================
# 17. RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("17. RESUMEN DE MAPAS")
print("=" * 70)

print(
    f"""
H3 cartografiados:
  {len(gdf):,}

Operaciones cartografiadas:
  {gdf["operaciones_totales"].sum():,.0f}

Hotspots extremos:
  {
      (
          gdf["categoria_demanda"]
          == "HOTSPOT_EXTREMO"
      ).sum():,}

Hotspots altos:
  {
      (
          gdf["categoria_demanda"]
          == "HOTSPOT_ALTO"
      ).sum():,}

Prioridad muy alta:
  {
      (
          gdf["prioridad_analitica"]
          == "PRIORIDAD_MUY_ALTA"
      ).sum():,}

Prioridad alta:
  {
      (
          gdf["prioridad_analitica"]
          == "PRIORIDAD_ALTA"
      ).sum():,}
"""
)


# ============================================================
# 18. ARCHIVOS GENERADOS
# ============================================================

print("=" * 70)
print("18. ARCHIVOS GENERADOS")
print("=" * 70)

for file in sorted(
    OUTPUT_DIR.iterdir()
):
    if file.is_file():
        print(
            f"  - {file.name}"
        )


print("\n" + "=" * 70)
print("MAPAS SUBE 2025 GENERADOS CORRECTAMENTE")
print("=" * 70)