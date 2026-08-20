from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3.parquet"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_concentracion.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_concentracion_resumen.csv"
)


# ============================================================
# CONFIGURACIÓN DE ANÁLISIS
# ============================================================

TOP_PERCENTAGES = [
    1,
    5,
    10,
    20,
    30,
    40,
    50,
]

TARGET_PERCENTAGES = [
    25,
    50,
    75,
    90,
    95,
    99,
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_h3_necesarios(
    serie_acumulada: pd.Series,
    target: float,
) -> int:
    """
    Devuelve la cantidad mínima de H3 necesarios
    para alcanzar el porcentaje objetivo de operaciones.
    """

    posiciones = serie_acumulada.ge(target)

    if not posiciones.any():
        return len(serie_acumulada)

    return int(posiciones.to_numpy().argmax()) + 1


def clasificar_concentracion(pct_top_10: float) -> str:
    """
    Clasificación descriptiva de concentración.

    No representa un estándar estadístico oficial.
    Sirve como indicador exploratorio.
    """

    if pct_top_10 >= 40:
        return "MUY_ALTA"

    if pct_top_10 >= 30:
        return "ALTA"

    if pct_top_10 >= 20:
        return "MEDIA"

    return "BAJA"


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ANÁLISIS DE CONCENTRACIÓN ESPACIAL SUBE 2025")
print("=" * 70)


# ============================================================
# CARGA
# ============================================================

print("\n" + "=" * 70)
print("1. CARGANDO DATOS")
print("=" * 70)

print(f"\nArchivo:")
print(INPUT_FILE)

gdf = gpd.read_parquet(INPUT_FILE)

print(f"\nH3 cargados: {len(gdf):,}")

# ------------------------------------------------------------
# Validaciones
# ------------------------------------------------------------

required_columns = [
    "id_h3",
    "operaciones_totales",
    "geometry",
]

missing_columns = [
    column
    for column in required_columns
    if column not in gdf.columns
]

if missing_columns:
    raise ValueError(
        f"Faltan columnas requeridas: {missing_columns}"
    )

if gdf["id_h3"].isna().any():
    raise ValueError(
        "Existen H3 nulos."
    )

if gdf["id_h3"].duplicated().any():
    duplicados = (
        gdf["id_h3"]
        .duplicated()
        .sum()
    )

    raise ValueError(
        f"Existen H3 duplicados: {duplicados:,}"
    )

if gdf["operaciones_totales"].isna().any():
    raise ValueError(
        "Existen operaciones_totales nulas."
    )

if (gdf["operaciones_totales"] < 0).any():
    raise ValueError(
        "Existen operaciones_totales negativas."
    )


# ============================================================
# TOTAL
# ============================================================

total_operaciones = (
    gdf["operaciones_totales"]
    .sum()
)

total_h3 = len(gdf)

print(
    f"\nOperaciones totales: "
    f"{total_operaciones:,.0f}"
)

print(
    f"H3 analizados: "
    f"{total_h3:,}"
)


# ============================================================
# ORDENAR POR DEMANDA
# ============================================================

print("\n" + "=" * 70)
print("2. ORDENANDO H3 POR DEMANDA")
print("=" * 70)

gdf = (
    gdf
    .sort_values(
        "operaciones_totales",
        ascending=False,
        kind="stable",
    )
    .reset_index(drop=True)
)


# ============================================================
# RANKING DE CONCENTRACIÓN
# ============================================================

gdf["ranking_concentracion"] = (
    gdf.index + 1
)

gdf["pct_h3"] = (
    gdf["ranking_concentracion"]
    / total_h3
    * 100
)


# ============================================================
# PARTICIPACIÓN INDIVIDUAL
# ============================================================

gdf["pct_operaciones"] = (
    gdf["operaciones_totales"]
    / total_operaciones
    * 100
)


# ============================================================
# PARTICIPACIÓN ACUMULADA
# ============================================================

gdf["operaciones_acumuladas"] = (
    gdf["operaciones_totales"]
    .cumsum()
)

gdf["pct_operaciones_acumulado"] = (
    gdf["operaciones_acumuladas"]
    / total_operaciones
    * 100
)


# ============================================================
# CONCENTRACIÓN
# ============================================================

print("\n" + "=" * 70)
print("3. CONCENTRACIÓN POR PERCENTILES")
print("=" * 70)

resumen_concentracion = []

for percentage in TOP_PERCENTAGES:

    n = max(
        1,
        int(total_h3 * percentage / 100),
    )

    operations = (
        gdf.iloc[:n]["operaciones_totales"]
        .sum()
    )

    share = (
        operations
        / total_operaciones
        * 100
    )

    print(
        f"Top {percentage:>2}% H3 "
        f"({n:>5,} hexágonos): "
        f"{operations:>12,.0f} operaciones "
        f"({share:>6.2f}%)"
    )

    resumen_concentracion.append(
        {
            "tipo": "TOP",
            "objetivo_pct": percentage,
            "h3": n,
            "pct_h3": (
                n
                / total_h3
                * 100
            ),
            "operaciones": operations,
            "pct_operaciones": share,
        }
    )


# ============================================================
# H3 NECESARIOS PARA ALCANZAR OBJETIVOS
# ============================================================

print("\n" + "=" * 70)
print("4. H3 NECESARIOS PARA EXPLICAR LA DEMANDA")
print("=" * 70)

for target in TARGET_PERCENTAGES:

    h3_needed = calcular_h3_necesarios(
        gdf["pct_operaciones_acumulado"],
        target,
    )

    operations = (
        gdf.iloc[:h3_needed]["operaciones_totales"]
        .sum()
    )

    pct_h3_needed = (
        h3_needed
        / total_h3
        * 100
    )

    pct_operations = (
        operations
        / total_operaciones
        * 100
    )

    print(
        f"Para alcanzar {target:>2}% de operaciones: "
        f"{h3_needed:>5,} H3 "
        f"({pct_h3_needed:>6.2f}% de los H3)"
    )

    resumen_concentracion.append(
        {
            "tipo": "OBJETIVO",
            "objetivo_pct": target,
            "h3": h3_needed,
            "pct_h3": pct_h3_needed,
            "operaciones": operations,
            "pct_operaciones": pct_operations,
        }
    )


# ============================================================
# HHI
# ============================================================

print("\n" + "=" * 70)
print("5. ÍNDICE DE CONCENTRACIÓN HHI")
print("=" * 70)

shares = (
    gdf["pct_operaciones"]
    / 100
)

hhi = (
    (shares ** 2)
    .sum()
    * 10_000
)

print(
    f"\nHHI espacial: {hhi:,.2f}"
)

print(
    "\nNota:"
)

print(
    "El HHI se utiliza aquí como indicador "
    "exploratorio de concentración espacial."
)

print(
    "No debe interpretarse automáticamente "
    "con los umbrales regulatorios de mercados."
)


# ============================================================
# TOP 20
# ============================================================

print("\n" + "=" * 70)
print("6. TOP 20 H3")
print("=" * 70)

top_columns = [
    "ranking_concentracion",
    "id_h3",
    "operaciones_totales",
    "pct_operaciones",
    "pct_operaciones_acumulado",
]

optional_columns = [
    "hora_pico",
    "operaciones_hora_pico",
    "ranking_operaciones",
]

for column in optional_columns:

    if column in gdf.columns:
        top_columns.append(column)


top20 = (
    gdf[
        top_columns
    ]
    .head(20)
)

print(
    top20.to_string(
        index=False
    )
)


# ============================================================
# TOP 10
# ============================================================

print("\n" + "=" * 70)
print("7. TOP 10 H3")
print("=" * 70)

top10 = (
    gdf
    .head(10)
    [
        [
            "ranking_concentracion",
            "id_h3",
            "operaciones_totales",
            "pct_operaciones",
            "pct_operaciones_acumulado",
        ]
    ]
)

print(
    top10.to_string(
        index=False
    )
)


# ============================================================
# CLASIFICACIÓN DE CONCENTRACIÓN
# ============================================================

top10_operations = (
    gdf
    .head(
        max(
            1,
            int(total_h3 * 10 / 100),
        )
    )
    ["operaciones_totales"]
    .sum()
)

pct_top10 = (
    top10_operations
    / total_operaciones
    * 100
)

clasificacion = (
    clasificar_concentracion(
        pct_top10
    )
)

print("\n" + "=" * 70)
print("8. CLASIFICACIÓN")
print("=" * 70)

print(
    f"\nParticipación Top 10% H3: "
    f"{pct_top10:.2f}%"
)

print(
    f"Concentración espacial: "
    f"{clasificacion}"
)


# ============================================================
# RESUMEN ESTADÍSTICO
# ============================================================

print("\n" + "=" * 70)
print("9. RESUMEN ESTADÍSTICO")
print("=" * 70)

print(
    f"""
H3:
  Total:                       {total_h3:,}

Operaciones:
  Total:                       {total_operaciones:,.0f}
  Promedio por H3:             {gdf["operaciones_totales"].mean():,.2f}
  Mediana por H3:              {gdf["operaciones_totales"].median():,.2f}
  Máximo por H3:               {gdf["operaciones_totales"].max():,.0f}
  Mínimo por H3:               {gdf["operaciones_totales"].min():,.0f}

Concentración:
  Top 1%:                      {(
        resumen_concentracion[0]["pct_operaciones"]
    ):,.2f}%

  Top 5%:                      {(
        resumen_concentracion[1]["pct_operaciones"]
    ):,.2f}%

  Top 10%:                     {pct_top10:,.2f}%

  HHI espacial:                {hhi:,.2f}

Clasificación:
  {clasificacion}
"""
)


# ============================================================
# VALIDACIÓN FINAL
# ============================================================

print("\n" + "=" * 70)
print("10. VALIDACIÓN FINAL")
print("=" * 70)

suma_operaciones = (
    gdf["operaciones_totales"]
    .sum()
)

if suma_operaciones != total_operaciones:
    raise ValueError(
        "La suma de operaciones cambió durante el análisis."
    )

if len(gdf) != total_h3:
    raise ValueError(
        "La cantidad de H3 cambió durante el análisis."
    )

if (
    gdf["pct_operaciones_acumulado"]
    .iloc[-1]
    < 99.999999
):
    raise ValueError(
        "La participación acumulada no alcanza 100%."
    )

print(
    "Integridad del análisis: OK"
)


# ============================================================
# EXPORTAR DATASET
# ============================================================

print("\n" + "=" * 70)
print("11. GUARDANDO DATASET")
print("=" * 70)

gdf.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nArchivo:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# EXPORTAR RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("12. GUARDANDO RESUMEN")
print("=" * 70)

resumen_df = pd.DataFrame(
    resumen_concentracion
)

resumen_general = pd.DataFrame(
    [
        {
            "indicador": "total_h3",
            "valor": total_h3,
        },
        {
            "indicador": "operaciones_totales",
            "valor": total_operaciones,
        },
        {
            "indicador": "operaciones_promedio_h3",
            "valor": gdf[
                "operaciones_totales"
            ].mean(),
        },
        {
            "indicador": "operaciones_mediana_h3",
            "valor": gdf[
                "operaciones_totales"
            ].median(),
        },
        {
            "indicador": "operaciones_max_h3",
            "valor": gdf[
                "operaciones_totales"
            ].max(),
        },
        {
            "indicador": "hhi_espacial",
            "valor": hhi,
        },
        {
            "indicador": "pct_top_10",
            "valor": pct_top10,
        },
        {
            "indicador": "clasificacion_concentracion",
            "valor": clasificacion,
        },
    ]
)

resumen_final = pd.concat(
    [
        resumen_general,
        resumen_df,
    ],
    ignore_index=True,
)

resumen_final.to_csv(
    OUTPUT_RESUMEN,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"\nArchivo:"
)

print(
    OUTPUT_RESUMEN
)


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS DE CONCENTRACIÓN FINALIZADO CORRECTAMENTE")
print("=" * 70)