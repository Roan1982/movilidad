import pandas as pd
import geopandas as gpd


print("Cargando datos SUBE 2025...")

sube_2025 = pd.read_csv(
    "data/raw/sube_2025.csv"
)

print(sube_2025.head())
print()
print(sube_2025.info())


print("\nCargando geometría H3...")

hexagonos = gpd.read_file(
    "data/raw/hexagonos_h3/hexagonos.shp"
)

print(hexagonos.head())
print()
print(hexagonos.info())