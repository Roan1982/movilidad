from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# CARGAR 2025
# ============================================================

archivo_2025 = DATA_DIR / "1.csv"

df_2025 = pd.read_csv(archivo_2025)


# ============================================================
# CARGAR 2019
# ============================================================

archivo_2019 = DATA_DIR / "3.csv"

df_2019 = pd.read_csv(
    archivo_2019,
    sep=";"
)


# ============================================================
# CARGAR H3
# ============================================================

archivo_h3 = DATA_DIR / "2" / "Hexagonos h3.shp"

h3 = gpd.read_file(archivo_h3)


# ============================================================
# CARGAR GRILLA 2019
# ============================================================

archivo_grilla = DATA_DIR / "4" / "GrillaHexPB.shp"

grilla = gpd.read_file(archivo_grilla)


# ============================================================
# RESUMEN GENERAL
# ============================================================

print("=" * 70)
print("RESUMEN GENERAL")
print("=" * 70)

print(f"2025 - registros:       {len(df_2025):,}")
print(f"2025 - H3:              {df_2025['id_h3'].nunique():,}")

print()

print(f"2019 - registros:       {len(df_2019):,}")
print(f"2019 - hexágonos:       {df_2019['ID_HEXAGONO'].nunique():,}")

print()

print(f"H3 disponibles:         {len(h3):,}")
print(f"Grilla disponible:      {len(grilla):,}")


# ============================================================
# MODOS 2025
# ============================================================

print()
print("=" * 70)
print("MODOS 2025")
print("=" * 70)

print(df_2025["modo"].value_counts().sort_index())


# ============================================================
# MODOS 2019
# ============================================================

print()
print("=" * 70)
print("MODOS 2019")
print("=" * 70)

print(df_2019["MODO"].value_counts().sort_index())


# ============================================================
# HORAS 2025
# ============================================================

print()
print("=" * 70)
print("HORAS 2025")
print("=" * 70)

print(
    df_2025["hora"]
    .value_counts()
    .sort_index()
)


# ============================================================
# HORAS 2019
# ============================================================

print()
print("=" * 70)
print("HORAS 2019")
print("=" * 70)

print(
    df_2019["HORA"]
    .value_counts()
    .sort_index()
)


# ============================================================
# COMPROBAR IDs 2025 CONTRA H3
# ============================================================

print()
print("=" * 70)
print("VALIDACIÓN ESPACIAL 2025")
print("=" * 70)

ids_2025 = set(df_2025["id_h3"].dropna().unique())
ids_h3 = set(h3["id_h3"].dropna().unique())

ids_2025_sin_geometria = ids_2025 - ids_h3
ids_h3_sin_datos = ids_h3 - ids_2025

print(f"IDs H3 utilizados en 2025:       {len(ids_2025):,}")
print(f"IDs H3 disponibles:              {len(ids_h3):,}")
print(f"IDs 2025 sin geometría:          {len(ids_2025_sin_geometria):,}")
print(f"Hexágonos sin operaciones:       {len(ids_h3_sin_datos):,}")


# ============================================================
# COMPROBAR IDs 2019 CONTRA GRILLA
# ============================================================

print()
print("=" * 70)
print("VALIDACIÓN ESPACIAL 2019")
print("=" * 70)

ids_2019 = set(
    df_2019["ID_HEXAGONO"]
    .dropna()
    .unique()
)

ids_grilla = set(
    grilla["id"]
    .dropna()
    .astype(int)
    .unique()
)

ids_2019_sin_grilla = ids_2019 - ids_grilla
ids_grilla_sin_datos = ids_grilla - ids_2019

print(f"IDs de hexágonos usados 2019:   {len(ids_2019):,}")
print(f"IDs disponibles en grilla:      {len(ids_grilla):,}")
print(f"IDs 2019 sin geometría:         {len(ids_2019_sin_grilla):,}")
print(f"Hexágonos sin operaciones:      {len(ids_grilla_sin_datos):,}")


# ============================================================
# DUPLICADOS / NULOS
# ============================================================

print()
print("=" * 70)
print("CALIDAD DE DATOS")
print("=" * 70)

print("\nNulos 2025:")
print(df_2025.isna().sum())

print("\nNulos 2019:")
print(df_2019.isna().sum())

print("\nValores negativos 2025:")
print(
    (df_2025["cantidad_trx"] < 0).sum()
)

print("\nValores negativos 2019:")
print(
    (df_2019["OPERACIONES"] < 0).sum()
)


# ============================================================
# DUPLICADOS LÓGICOS
# ============================================================

print()
print("=" * 70)
print("DUPLICADOS LÓGICOS")
print("=" * 70)

duplicados_2025 = df_2025.duplicated(
    subset=["id_h3", "hora", "modo"]
).sum()

duplicados_2019 = df_2019.duplicated(
    subset=["ID_HEXAGONO", "HORA", "MODO"]
).sum()

print(
    f"2025 - duplicados "
    f"(id_h3 + hora + modo): {duplicados_2025:,}"
)

print(
    f"2019 - duplicados "
    f"(ID_HEXAGONO + HORA + MODO): {duplicados_2019:,}"
)