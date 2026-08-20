from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# CARGAR DATOS
# ============================================================

df_2025 = pd.read_csv(
    DATA_DIR / "1.csv"
)

df_2019 = pd.read_csv(
    DATA_DIR / "3.csv",
    sep=";"
)

h3 = gpd.read_file(
    DATA_DIR / "2" / "Hexagonos h3.shp"
)

grilla = gpd.read_file(
    DATA_DIR / "4" / "GrillaHexPB.shp"
)


# ============================================================
# 1. H3 DE 2025 SIN GEOMETRÍA
# ============================================================

print("=" * 70)
print("H3 2025 SIN GEOMETRÍA")
print("=" * 70)

ids_2025 = set(
    df_2025["id_h3"].unique()
)

ids_h3 = set(
    h3["id_h3"].unique()
)

ids_sin_geometria = sorted(
    ids_2025 - ids_h3
)

print(
    f"Cantidad: {len(ids_sin_geometria):,}"
)

print("\nPrimeros 50:")

for h3_id in ids_sin_geometria[:50]:
    print(h3_id)


# ============================================================
# 2. ¿CUÁNTAS OPERACIONES TIENEN ESOS H3?
# ============================================================

print()
print("=" * 70)
print("OPERACIONES ASOCIADAS A H3 SIN GEOMETRÍA")
print("=" * 70)

df_sin_geometria = df_2025[
    df_2025["id_h3"].isin(ids_sin_geometria)
].copy()

print(
    f"Registros: {len(df_sin_geometria):,}"
)

print(
    f"Transacciones: "
    f"{df_sin_geometria['cantidad_trx'].sum():,}"
)

print("\nPor modo:")

print(
    df_sin_geometria
    .groupby("modo")["cantidad_trx"]
    .agg(["count", "sum"])
    .sort_values("sum", ascending=False)
)


# ============================================================
# 3. UBICACIÓN DE LOS H3 SIN GEOMETRÍA
# ============================================================

print()
print("=" * 70)
print("HORAS DE LOS H3 SIN GEOMETRÍA")
print("=" * 70)

print(
    df_sin_geometria
    .groupby("hora")["cantidad_trx"]
    .sum()
    .sort_index()
)


# ============================================================
# 4. VALORES NEGATIVOS 2019
# ============================================================

print()
print("=" * 70)
print("VALORES NEGATIVOS 2019")
print("=" * 70)

negativos_2019 = df_2019[
    df_2019["OPERACIONES"] < 0
].copy()

print(negativos_2019.to_string(index=False))


# ============================================================
# 5. MAGNITUD DE LOS NEGATIVOS
# ============================================================

print()
print("=" * 70)
print("RESUMEN NEGATIVOS")
print("=" * 70)

print(
    negativos_2019["OPERACIONES"].describe()
)


# ============================================================
# 6. TOTALES GENERALES
# ============================================================

print()
print("=" * 70)
print("TOTALES DE OPERACIONES")
print("=" * 70)

total_2025 = df_2025["cantidad_trx"].sum()
total_2019 = df_2019["OPERACIONES"].sum()

print(f"2025: {total_2025:,}")
print(f"2019: {total_2019:,}")


# ============================================================
# 7. TOTALES POR MODO
# ============================================================

print()
print("=" * 70)
print("2025 POR MODO")
print("=" * 70)

print(
    df_2025
    .groupby("modo")["cantidad_trx"]
    .agg(["count", "sum"])
    .sort_values("sum", ascending=False)
)


print()
print("=" * 70)
print("2019 POR MODO")
print("=" * 70)

print(
    df_2019
    .groupby("MODO")["OPERACIONES"]
    .agg(["count", "sum"])
    .sort_values("sum", ascending=False)
)


# ============================================================
# 8. TOTALES POR HORA
# ============================================================

print()
print("=" * 70)
print("2025 POR HORA")
print("=" * 70)

print(
    df_2025
    .groupby("hora")["cantidad_trx"]
    .sum()
    .sort_index()
)


print()
print("=" * 70)
print("2019 POR HORA")
print("=" * 70)

print(
    df_2019
    .groupby("HORA")["OPERACIONES"]
    .sum()
    .sort_index()
)