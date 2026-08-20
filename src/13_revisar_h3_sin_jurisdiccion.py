from pathlib import Path
import json

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
    / "sube_2025_h3_territorio.parquet"
)

OUTPUT_AUDITORIA = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_auditoria_territorial.csv"
)

OUTPUT_NEAREST = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_nearest_auditoria.csv"
)

OUTPUT_CALIDAD = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_calidad_territorial.json"
)


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("AUDITORÍA TERRITORIAL SUBE 2025")
print("=" * 70)


# ============================================================
# CARGAR
# ============================================================

print("\nCargando dataset territorial...")

gdf = gpd.read_parquet(INPUT_FILE)

print(
    f"H3 cargados: {len(gdf):,}"
)


# ============================================================
# VALIDACIONES ESTRUCTURALES
# ============================================================

print("\n" + "=" * 70)
print("VALIDACIONES ESTRUCTURALES")
print("=" * 70)


errores = []


# ------------------------------------------------------------
# CANTIDAD H3
# ------------------------------------------------------------

if len(gdf) != 6785:

    errores.append(
        f"Cantidad H3 incorrecta: {len(gdf):,}"
    )

else:

    print("OK    6.785 H3")


# ------------------------------------------------------------
# H3 DUPLICADOS
# ------------------------------------------------------------

duplicados = (
    gdf["id_h3"]
    .duplicated()
    .sum()
)

if duplicados > 0:

    errores.append(
        f"H3 duplicados: {duplicados:,}"
    )

else:

    print("OK    H3 únicos")


# ------------------------------------------------------------
# OPERACIONES
# ------------------------------------------------------------

operaciones = (
    pd.to_numeric(
        gdf["operaciones_totales"],
        errors="coerce"
    )
    .fillna(0)
    .sum()
)

print(
    f"OK    Operaciones: {operaciones:,.0f}"
)


# ------------------------------------------------------------
# JURISDICCIONES
# ------------------------------------------------------------

sin_jurisdiccion = (
    gdf["jurisdiccion"]
    .isna()
    .sum()
)

if sin_jurisdiccion > 0:

    errores.append(
        f"H3 sin jurisdicción: {sin_jurisdiccion:,}"
    )

else:

    print("OK    Todos los H3 tienen jurisdicción")


# ------------------------------------------------------------
# PROVINCIAS
# ------------------------------------------------------------

sin_provincia = (
    gdf["provincia"]
    .isna()
    .sum()
)

if sin_provincia > 0:

    errores.append(
        f"H3 sin provincia: {sin_provincia:,}"
    )

else:

    print("OK    Todos los H3 tienen provincia")


# ------------------------------------------------------------
# MÉTODO
# ------------------------------------------------------------

sin_metodo = (
    gdf["metodo_asignacion"]
    .isna()
    .sum()
)

if sin_metodo > 0:

    errores.append(
        f"H3 sin método de asignación: {sin_metodo:,}"
    )

else:

    print("OK    Todos los H3 tienen método")


# ============================================================
# MÉTODOS DE ASIGNACIÓN
# ============================================================

print("\n" + "=" * 70)
print("MÉTODOS DE ASIGNACIÓN")
print("=" * 70)


metodos = (
    gdf
    .groupby(
        "metodo_asignacion",
        dropna=False
    )
    .agg(
        h3=(
            "id_h3",
            "nunique"
        ),

        operaciones=(
            "operaciones_totales",
            "sum"
        )
    )
    .reset_index()
)


metodos["pct_h3"] = (
    metodos["h3"]
    / len(gdf)
    * 100
)


metodos["pct_operaciones"] = (
    metodos["operaciones"]
    / operaciones
    * 100
)


print(
    metodos.to_string(
        index=False,
        formatters={
            "pct_h3": "{:.2f}".format,
            "pct_operaciones": "{:.2f}".format,
        }
    )
)


# ============================================================
# NEAREST
# ============================================================

print("\n" + "=" * 70)
print("AUDITORÍA DE ASIGNACIONES NEAREST")
print("=" * 70)


nearest = gdf[
    gdf["metodo_asignacion"] == "NEAREST"
].copy()


print(
    f"H3 asignados por NEAREST: "
    f"{len(nearest):,}"
)


if len(nearest) > 0:

    nearest["distancia_asignacion_m"] = (
        pd.to_numeric(
            nearest["distancia_asignacion_m"],
            errors="coerce"
        )
    )


    print(
        "\nEstadísticas de distancia:"
    )

    print(
        nearest[
            "distancia_asignacion_m"
        ]
        .describe()
        .to_string()
    )


    print(
        "\nDetalle de los H3:"
    )


    columnas_nearest = [
        "id_h3",
        "provincia",
        "jurisdiccion",
        "operaciones_totales",
        "categoria_demanda",
        "hora_pico",
        "modo_dominante",
        "distancia_asignacion_m",
    ]


    columnas_nearest = [
        c
        for c in columnas_nearest
        if c in nearest.columns
    ]


    print(
        nearest[
            columnas_nearest
        ]
        .sort_values(
            "distancia_asignacion_m"
        )
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # CLASIFICACIÓN
    # --------------------------------------------------------

    def clasificar_distancia(distancia):

        if pd.isna(distancia):
            return "SIN_DISTANCIA"

        if distancia <= 100:
            return "EXCELENTE"

        if distancia <= 500:
            return "BUENA"

        if distancia <= 1000:
            return "ACEPTABLE"

        if distancia <= 2500:
            return "REVISAR"

        return "CRITICA"


    nearest[
        "clasificacion_distancia"
    ] = (
        nearest[
            "distancia_asignacion_m"
        ]
        .apply(
            clasificar_distancia
        )
    )


    print(
        "\nClasificación:"
    )

    print(
        nearest[
            "clasificacion_distancia"
        ]
        .value_counts()
        .to_string()
    )


else:

    print(
        "No existen asignaciones NEAREST."
    )


# ============================================================
# DISTANCIAS NEAREST
# ============================================================

print("\n" + "=" * 70)
print("CONTROL DE DISTANCIAS")
print("=" * 70)


if len(nearest) > 0:

    for limite in [
        100,
        500,
        1000,
        2500,
    ]:

        cantidad = (
            nearest[
                "distancia_asignacion_m"
            ]
            <= limite
        ).sum()


        porcentaje = (
            cantidad
            / len(nearest)
            * 100
        )


        print(
            f"<= {limite:>4} m: "
            f"{cantidad:>4} H3 "
            f"({porcentaje:6.2f}%)"
        )


# ============================================================
# IMPACTO DE NEAREST
# ============================================================

print("\n" + "=" * 70)
print("IMPACTO DE NEAREST")
print("=" * 70)


operaciones_nearest = (
    nearest[
        "operaciones_totales"
    ]
    .sum()
    if len(nearest) > 0
    else 0
)


pct_operaciones_nearest = (
    operaciones_nearest
    / operaciones
    * 100
    if operaciones > 0
    else 0
)


print(
    f"H3 NEAREST: "
    f"{len(nearest):,}"
)


print(
    f"Operaciones NEAREST: "
    f"{operaciones_nearest:,.0f}"
)


print(
    f"% de H3: "
    f"{len(nearest) / len(gdf) * 100:.4f}%"
)


print(
    f"% de operaciones: "
    f"{pct_operaciones_nearest:.4f}%"
)


# ============================================================
# INTERSECCIÓN
# ============================================================

print("\n" + "=" * 70)
print("AUDITORÍA DE INTERSECCIONES")
print("=" * 70)


interseccion = gdf[
    gdf["metodo_asignacion"]
    == "INTERSECCION_MAYOR_AREA"
].copy()


print(
    f"H3 asignados por intersección: "
    f"{len(interseccion):,}"
)


operaciones_interseccion = (
    interseccion[
        "operaciones_totales"
    ]
    .sum()
)


print(
    f"Operaciones involucradas: "
    f"{operaciones_interseccion:,.0f}"
)


print(
    f"% H3: "
    f"{len(interseccion) / len(gdf) * 100:.2f}%"
)


print(
    f"% operaciones: "
    f"{operaciones_interseccion / operaciones * 100:.2f}%"
)


# ============================================================
# REPRESENTATIVE POINT
# ============================================================

print("\n" + "=" * 70)
print("REPRESENTATIVE POINT")
print("=" * 70)


representative = gdf[
    gdf["metodo_asignacion"]
    == "REPRESENTATIVE_POINT"
]


operaciones_representative = (
    representative[
        "operaciones_totales"
    ]
    .sum()
)


print(
    f"H3: "
    f"{len(representative):,}"
)


print(
    f"Operaciones: "
    f"{operaciones_representative:,.0f}"
)


print(
    f"% H3: "
    f"{len(representative) / len(gdf) * 100:.2f}%"
)


print(
    f"% operaciones: "
    f"{operaciones_representative / operaciones * 100:.2f}%"
)


# ============================================================
# JURISDICCIONES
# ============================================================

print("\n" + "=" * 70)
print("COBERTURA TERRITORIAL")
print("=" * 70)


jurisdicciones = (
    gdf[
        "jurisdiccion"
    ]
    .dropna()
    .nunique()
)


provincias = (
    gdf[
        "provincia"
    ]
    .dropna()
    .nunique()
)


print(
    f"Jurisdicciones: "
    f"{jurisdicciones:,}"
)


print(
    f"Provincias: "
    f"{provincias:,}"
)


# ============================================================
# CALIDAD TERRITORIAL
# ============================================================

print("\n" + "=" * 70)
print("ÍNDICE DE CALIDAD TERRITORIAL")
print("=" * 70)


# Porcentaje de H3 resueltos mediante
# representative point.
#
pct_representative = (
    len(representative)
    / len(gdf)
    * 100
)


# Porcentaje de operaciones que utilizan
# representative point.
#
pct_operaciones_representative = (
    operaciones_representative
    / operaciones
    * 100
)


# Penalización por NEAREST.
#
# El NEAREST no es necesariamente incorrecto,
# pero indica una asignación menos directa.
#
penalizacion_nearest = (
    len(nearest)
    / len(gdf)
    * 100
)


# Penalización por distancias > 500 m.
#
if len(nearest) > 0:

    nearest_mayor_500 = (
        nearest[
            "distancia_asignacion_m"
        ]
        > 500
    ).sum()

else:

    nearest_mayor_500 = 0


penalizacion_distancia = (
    nearest_mayor_500
    / len(gdf)
    * 100
)


indice_calidad = (
    100
    - penalizacion_nearest
    - penalizacion_distancia
)


indice_calidad = max(
    0,
    min(
        100,
        indice_calidad
    )
)


print(
    f"% H3 REPRESENTATIVE_POINT: "
    f"{pct_representative:.2f}%"
)


print(
    f"% operaciones REPRESENTATIVE_POINT: "
    f"{pct_operaciones_representative:.2f}%"
)


print(
    f"H3 NEAREST: "
    f"{len(nearest):,}"
)


print(
    f"H3 NEAREST > 500 m: "
    f"{nearest_mayor_500:,}"
)


print(
    f"\nÍNDICE DE CALIDAD TERRITORIAL: "
    f"{indice_calidad:.2f}/100"
)


# ============================================================
# ERRORES
# ============================================================

print("\n" + "=" * 70)
print("RESULTADO DE AUDITORÍA")
print("=" * 70)


if errores:

    print(
        "\nSE DETECTARON PROBLEMAS:"
    )

    for error in errores:

        print(
            f"ERROR  {error}"
        )

else:

    print(
        "\nOK    TODAS LAS VALIDACIONES PASARON"
    )


# ============================================================
# GUARDAR AUDITORÍA COMPLETA
# ============================================================

auditoria = (
    gdf[
        [
            "id_h3",
            "provincia",
            "jurisdiccion",
            "operaciones_totales",
            "categoria_demanda",
            "hora_pico",
            "modo_dominante",
            "metodo_asignacion",
            "distancia_asignacion_m",
        ]
    ]
    .copy()
)


auditoria.to_csv(
    OUTPUT_AUDITORIA,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# GUARDAR NEAREST
# ============================================================

if len(nearest) > 0:

    nearest.to_csv(
        OUTPUT_NEAREST,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# JSON DE CALIDAD
# ============================================================

resumen = {
    "h3": int(len(gdf)),

    "operaciones": int(
        operaciones
    ),

    "jurisdicciones": int(
        jurisdicciones
    ),

    "provincias": int(
        provincias
    ),

    "h3_representative_point": int(
        len(representative)
    ),

    "h3_interseccion": int(
        len(interseccion)
    ),

    "h3_nearest": int(
        len(nearest)
    ),

    "operaciones_representative_point": int(
        operaciones_representative
    ),

    "operaciones_interseccion": int(
        operaciones_interseccion
    ),

    "operaciones_nearest": int(
        operaciones_nearest
    ),

    "pct_h3_representative_point": round(
        pct_representative,
        4
    ),

    "pct_operaciones_representative_point": round(
        pct_operaciones_representative,
        4
    ),

    "pct_h3_nearest": round(
        len(nearest) / len(gdf) * 100,
        4
    ),

    "pct_operaciones_nearest": round(
        pct_operaciones_nearest,
        4
    ),

    "nearest_mayor_500_m": int(
        nearest_mayor_500
    ),

    "indice_calidad_territorial": round(
        indice_calidad,
        2
    ),

    "errores": errores,

    "estado": (
        "OK"
        if not errores
        else "REVISAR"
    ),
}


if len(nearest) > 0:

    resumen["nearest_distancia_min_m"] = float(
        nearest[
            "distancia_asignacion_m"
        ].min()
    )

    resumen["nearest_distancia_max_m"] = float(
        nearest[
            "distancia_asignacion_m"
        ].max()
    )

    resumen["nearest_distancia_promedio_m"] = float(
        nearest[
            "distancia_asignacion_m"
        ].mean()
    )


with open(
    OUTPUT_CALIDAD,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        resumen,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# ARCHIVOS
# ============================================================

print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)


print(
    f"\nAuditoría completa:\n"
    f"{OUTPUT_AUDITORIA}"
)


print(
    f"\nAuditoría NEAREST:\n"
    f"{OUTPUT_NEAREST}"
)


print(
    f"\nResumen de calidad:\n"
    f"{OUTPUT_CALIDAD}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("AUDITORÍA TERRITORIAL FINALIZADA")
print("=" * 70)

print(
    f"H3: {len(gdf):,}"
)

print(
    f"Operaciones: {operaciones:,.0f}"
)

print(
    f"Índice calidad territorial: "
    f"{indice_calidad:.2f}/100"
)

print(
    f"Estado: "
    f"{'OK' if not errores else 'REVISAR'}"
)