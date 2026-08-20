from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# 2025
# ============================================================

archivo_2025 = DATA_DIR / "1.csv"

print("=" * 70)
print("SUBE 2025")
print("=" * 70)

df_2025 = pd.read_csv(archivo_2025)

print(f"Archivo: {archivo_2025}")
print(f"Filas: {len(df_2025):,}")
print(f"Columnas: {len(df_2025.columns)}")

print("\nColumnas:")
for columna in df_2025.columns:
    print(f"  - {columna}")

print("\nPrimeras filas:")
print(df_2025.head())

print("\nTipos de datos:")
print(df_2025.dtypes)


# ============================================================
# 2019
# ============================================================

archivo_2019 = DATA_DIR / "3.csv"

print("\n")
print("=" * 70)
print("SUBE 2019")
print("=" * 70)

df_2019 = pd.read_csv(
    archivo_2019,
    sep=";"
)
print(f"Archivo: {archivo_2019}")
print(f"Filas: {len(df_2019):,}")
print(f"Columnas: {len(df_2019.columns)}")

print("\nColumnas:")
for columna in df_2019.columns:
    print(f"  - {columna}")

print("\nPrimeras filas:")
print(df_2019.head())

print("\nTipos de datos:")
print(df_2019.dtypes)