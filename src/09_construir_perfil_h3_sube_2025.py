from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_H3 = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3.parquet"
)

INPUT_H3_HORA = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_hora.parquet"
)

INPUT_H3_MODO = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_modo.parquet"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_perfil_h3.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_perfil_h3_resumen.csv"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

FRANJAS = [
    "MADRUGADA",
    "MANANA",
    "MEDIODIA",
    "TARDE",
    "NOCHE",
]


# ============================================================
# FUNCIONES
# ============================================================

def clasificar_demanda(percentil):
    """
    Clasificación relativa de la demanda espacial.
    """

    if percentil >= 99:
        return "HOTSPOT_EXTREMO"

    if percentil >= 95:
        return "HOTSPOT_ALTO"

    if percentil >= 75:
        return "DEMANDA_ALTA"

    if percentil >= 25:
        return "DEMANDA_MEDIA"

    return "DEMANDA_BAJA"


def clasificar_perfil_temporal(franja):
    """
    Clasificación simplificada del comportamiento temporal.
    """

    if franja == "MANANA":
        return "MATUTINO"

    if franja == "TARDE":
        return "VESPERTINO"

    if franja == "MEDIODIA":
        return "DIURNO"

    if franja == "NOCHE":
        return "NOCTURNO"

    return "MADRUGADA"


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("PERFIL DE DEMANDA POR H3 - SUBE 2025")
print("=" * 70)


# ============================================================
# 1. CARGA
# ============================================================

print("\n" + "=" * 70)
print("1. CARGANDO INDICADORES")
print("=" * 70)

print(f"\nArchivo H3:")
print(INPUT_H3)

h3 = gpd.read_parquet(INPUT_H3)

print(f"\nH3 cargados: {len(h3):,}")


# ============================================================
# VALIDACIÓN
# ============================================================

required_h3 = [
    "id_h3",
    "operaciones_totales",
    "hora_pico",
    "operaciones_hora_pico",
    "geometry",
]

missing_h3 = [
    column
    for column in required_h3
    if column not in h3.columns
]

if missing_h3:
    raise ValueError(
        f"Faltan columnas requeridas en H3: {missing_h3}"
    )


# ============================================================
# 2. PERCENTIL DE DEMANDA
# ============================================================

print("\n" + "=" * 70)
print("2. CLASIFICACIÓN DE DEMANDA")
print("=" * 70)

print("\nCalculando percentiles...")

h3["percentil_demanda"] = (
    h3["operaciones_totales"]
    .rank(
        method="average",
        pct=True,
    )
    * 100
)

h3["categoria_demanda"] = (
    h3["percentil_demanda"]
    .apply(clasificar_demanda)
)


# ============================================================
# 3. CONCENTRACIÓN EN HORA PICO
# ============================================================

print("\nCalculando concentración de hora pico...")

h3["pct_hora_pico"] = np.where(
    h3["operaciones_totales"] > 0,
    (
        h3["operaciones_hora_pico"]
        / h3["operaciones_totales"]
        * 100
    ),
    0,
)


# ============================================================
# 4. FRANJA DOMINANTE
# ============================================================

print("\n" + "=" * 70)
print("4. PERFIL TEMPORAL")
print("=" * 70)

franja_columns = [
    f"operaciones_{franja.lower()}"
    for franja in FRANJAS
]

available_franjas = [
    column
    for column in franja_columns
    if column in h3.columns
]

if not available_franjas:
    raise ValueError(
        "No se encontraron columnas de franjas horarias."
    )


franja_mapping = {
    f"operaciones_{franja.lower()}": franja
    for franja in FRANJAS
}


def obtener_franja_dominante(row):

    valores = {
        column: row[column]
        for column in available_franjas
    }

    if not valores:
        return None

    return franja_mapping[
        max(
            valores,
            key=valores.get,
        )
    ]


h3["franja_dominante"] = h3.apply(
    obtener_franja_dominante,
    axis=1,
)


h3["perfil_temporal"] = (
    h3["franja_dominante"]
    .apply(clasificar_perfil_temporal)
)


# ============================================================
# 5. MODO DOMINANTE
# ============================================================

print("\n" + "=" * 70)
print("5. PERFIL MODAL")
print("=" * 70)

print("\nCargando H3 + modo...")

h3_modo = gpd.read_parquet(INPUT_H3_MODO)

print(
    f"Registros H3 + modo: "
    f"{len(h3_modo):,}"
)


required_modo = [
    "id_h3",
    "modo",
    "operaciones",
]

missing_modo = [
    column
    for column in required_modo
    if column not in h3_modo.columns
]

if missing_modo:
    raise ValueError(
        f"Faltan columnas en H3 + modo: {missing_modo}"
    )


# ============================================================
# TOTALES POR MODO
# ============================================================

modo_totales = (
    h3_modo
    .groupby(
        ["id_h3", "modo"],
        as_index=False,
    )["operaciones"]
    .sum()
)


# ============================================================
# MODO DOMINANTE
# ============================================================

idx_modo = (
    modo_totales
    .groupby("id_h3")["operaciones"]
    .idxmax()
)


modo_dominante = (
    modo_totales
    .loc[idx_modo]
    .set_index("id_h3")
)


modo_dominante = (
    modo_dominante
    .rename(
        columns={
            "modo": "modo_dominante",
            "operaciones": "operaciones_modo_dominante",
        }
    )
)


# ============================================================
# INCORPORAR MODO
# ============================================================

h3 = h3.set_index("id_h3")

h3 = h3.join(
    modo_dominante[
        [
            "modo_dominante",
            "operaciones_modo_dominante",
        ]
    ],
    how="left",
)


# ============================================================
# PORCENTAJE DEL MODO DOMINANTE
# ============================================================

h3["pct_modo_dominante"] = np.where(
    h3["operaciones_totales"] > 0,
    (
        h3["operaciones_modo_dominante"]
        / h3["operaciones_totales"]
        * 100
    ),
    0,
)


# ============================================================
# ÍNDICE DE DOMINANCIA MODAL
# ============================================================

h3["indice_dominancia_modal"] = (
    h3["pct_modo_dominante"]
    / 100
)


# ============================================================
# 6. DIVERSIDAD MODAL
# ============================================================

print("\n" + "=" * 70)
print("6. DIVERSIDAD MODAL")
print("=" * 70)

modo_counts = (
    h3_modo
    .groupby("id_h3")["modo"]
    .nunique()
)


h3["cantidad_modos"] = (
    modo_counts
    .reindex(h3.index)
    .fillna(0)
    .astype(int)
)


# ============================================================
# 7. ÍNDICE DE DIVERSIDAD MODAL
# ============================================================

print("Calculando diversidad modal relativa...")

h3["diversidad_modal"] = np.where(
    h3["cantidad_modos"] > 0,
    h3["cantidad_modos"]
    / h3_modo["modo"].nunique(),
    0,
)


# ============================================================
# 8. PERFIL COMBINADO
# ============================================================

print("\n" + "=" * 70)
print("8. PERFIL COMBINADO")
print("=" * 70)


def construir_perfil(row):

    demanda = row["categoria_demanda"]
    temporal = row["perfil_temporal"]

    return (
        f"{demanda}_{temporal}"
    )


h3["perfil_nodo"] = h3.apply(
    construir_perfil,
    axis=1,
)


# ============================================================
# 9. INDICADOR DE PRIORIDAD
# ============================================================

print("Calculando prioridad analítica...")


def calcular_prioridad(row):

    categoria = row["categoria_demanda"]
    pct_pico = row["pct_hora_pico"]

    if categoria == "HOTSPOT_EXTREMO":
        if pct_pico >= 20:
            return "PRIORIDAD_CRITICA"

        return "PRIORIDAD_MUY_ALTA"

    if categoria == "HOTSPOT_ALTO":
        if pct_pico >= 20:
            return "PRIORIDAD_MUY_ALTA"

        return "PRIORIDAD_ALTA"

    if categoria == "DEMANDA_ALTA":
        return "PRIORIDAD_MEDIA"

    return "PRIORIDAD_BAJA"


h3["prioridad_analitica"] = h3.apply(
    calcular_prioridad,
    axis=1,
)


# ============================================================
# 10. RANKING FINAL
# ============================================================

h3["ranking_perfil"] = (
    h3["operaciones_totales"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("Int64")
)


# ============================================================
# 11. LIMPIEZA
# ============================================================

h3 = h3.reset_index()

h3 = h3.sort_values(
    "operaciones_totales",
    ascending=False,
)


# ============================================================
# 12. VALIDACIÓN
# ============================================================

print("\n" + "=" * 70)
print("12. VALIDACIÓN")
print("=" * 70)

if h3["id_h3"].duplicated().any():

    raise ValueError(
        "Existen H3 duplicados."
    )


if h3["operaciones_totales"].isna().any():

    raise ValueError(
        "Existen operaciones_totales nulas."
    )


if h3["categoria_demanda"].isna().any():

    raise ValueError(
        "Existen categorías de demanda nulas."
    )


if h3["modo_dominante"].isna().any():

    print(
        "ADVERTENCIA: existen H3 sin modo dominante."
    )


print("Integridad: OK")


# ============================================================
# 13. GUARDAR DATASET
# ============================================================

print("\n" + "=" * 70)
print("13. GUARDANDO DATASET")
print("=" * 70)

print(f"\nArchivo:")
print(OUTPUT_FILE)

h3.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 14. RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("14. RESUMEN")
print("=" * 70)

print("\nCategorías de demanda:")

print(
    h3[
        "categoria_demanda"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nPerfiles temporales:")

print(
    h3[
        "perfil_temporal"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nModos dominantes:")

print(
    h3[
        "modo_dominante"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nPrioridad analítica:")

print(
    h3[
        "prioridad_analitica"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nPerfiles combinados:")

print(
    h3[
        "perfil_nodo"
    ]
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# 15. TOP 20
# ============================================================

print("\n" + "=" * 70)
print("15. TOP 20 H3 - PERFIL")
print("=" * 70)


columns = [
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


print(
    h3[
        columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 16. RESUMEN CSV
# ============================================================

print("\n" + "=" * 70)
print("16. GUARDANDO RESUMEN CSV")
print("=" * 70)


resumen = pd.DataFrame(
    [
        {
            "indicador": "h3",
            "valor": len(h3),
        },
        {
            "indicador": "operaciones",
            "valor": h3[
                "operaciones_totales"
            ].sum(),
        },
        {
            "indicador": "hotspot_extremo",
            "valor": (
                h3["categoria_demanda"]
                == "HOTSPOT_EXTREMO"
            ).sum(),
        },
        {
            "indicador": "hotspot_alto",
            "valor": (
                h3["categoria_demanda"]
                == "HOTSPOT_ALTO"
            ).sum(),
        },
        {
            "indicador": "demanda_alta",
            "valor": (
                h3["categoria_demanda"]
                == "DEMANDA_ALTA"
            ).sum(),
        },
        {
            "indicador": "demanda_media",
            "valor": (
                h3["categoria_demanda"]
                == "DEMANDA_MEDIA"
            ).sum(),
        },
        {
            "indicador": "demanda_baja",
            "valor": (
                h3["categoria_demanda"]
                == "DEMANDA_BAJA"
            ).sum(),
        },
        {
            "indicador": "hora_pico_global",
            "valor": h3.groupby(
                "hora_pico"
            )[
                "operaciones_hora_pico"
            ].sum().idxmax(),
        },
        {
            "indicador": "operaciones_hora_pico_global",
            "valor": h3.groupby(
                "hora_pico"
            )[
                "operaciones_hora_pico"
            ].sum().max(),
        },
    ]
)


resumen.to_csv(
    OUTPUT_RESUMEN,
    index=False,
    encoding="utf-8-sig",
)


print(f"\nArchivo:")
print(OUTPUT_RESUMEN)


# ============================================================
# 17. RESULTADO FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESULTADO FINAL")
print("=" * 70)

print(
    f"""
H3 analizados:
  {len(h3):,}

Operaciones:
  {h3["operaciones_totales"].sum():,.0f}

Hotspots extremos:
  {(h3["categoria_demanda"] == "HOTSPOT_EXTREMO").sum():,}

Hotspots altos:
  {(h3["categoria_demanda"] == "HOTSPOT_ALTO").sum():,}

Demanda alta:
  {(h3["categoria_demanda"] == "DEMANDA_ALTA").sum():,}

H3 con múltiples modos:
  {(h3["cantidad_modos"] > 1).sum():,}

H3 con un único modo:
  {(h3["cantidad_modos"] == 1).sum():,}

H3 prioridad crítica:
  {(h3["prioridad_analitica"] == "PRIORIDAD_CRITICA").sum():,}

H3 prioridad muy alta:
  {(h3["prioridad_analitica"] == "PRIORIDAD_MUY_ALTA").sum():,}

H3 prioridad alta:
  {(h3["prioridad_analitica"] == "PRIORIDAD_ALTA").sum():,}
"""
)


print("=" * 70)
print("PERFIL H3 CONSTRUIDO CORRECTAMENTE")
print("=" * 70)