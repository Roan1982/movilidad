import geopandas as gpd
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# HEXÁGONOS H3
# ============================================================

archivo_h3 = DATA_DIR / "2" / "Hexagonos h3.shp"

print("=" * 70)
print("HEXÁGONOS H3")
print("=" * 70)

h3 = gpd.read_file(archivo_h3)

print(f"Archivo: {archivo_h3}")
print(f"Filas: {len(h3):,}")
print(f"Columnas: {len(h3.columns)}")

print("\nColumnas:")
for columna in h3.columns:
    print(f"  - {columna}")

print("\nSistema de coordenadas:")
print(h3.crs)

print("\nPrimeras filas:")
print(h3.head())

print("\nTipos de geometría:")
print(h3.geometry.geom_type.value_counts())

print("\nExtensión geográfica:")
print(h3.total_bounds)


# ============================================================
# GRILLA 600 METROS
# ============================================================

archivo_grilla = DATA_DIR / "4" / "GrillaHexPB.shp"

print("\n")
print("=" * 70)
print("GRILLA HEXAGONAL")
print("=" * 70)

grilla = gpd.read_file(archivo_grilla)

print(f"Archivo: {archivo_grilla}")
print(f"Filas: {len(grilla):,}")
print(f"Columnas: {len(grilla.columns)}")

print("\nColumnas:")
for columna in grilla.columns:
    print(f"  - {columna}")

print("\nSistema de coordenadas:")
print(grilla.crs)

print("\nPrimeras filas:")
print(grilla.head())

print("\nTipos de geometría:")
print(grilla.geometry.geom_type.value_counts())

print("\nExtensión geográfica:")
print(grilla.total_bounds)