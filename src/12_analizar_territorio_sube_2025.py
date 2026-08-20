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
    / "sube_2025_h3_territorio.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_JURISDICCIONES = (
    OUTPUT_DIR
    / "sube_2025_jurisdicciones.parquet"
)

OUTPUT_PROVINCIAS = (
    OUTPUT_DIR
    / "sube_2025_provincias.parquet"
)

OUTPUT_HOTSPOTS = (
    OUTPUT_DIR
    / "sube_2025_hotspots_territoriales.parquet"
)

OUTPUT_H3_RANKING = (
    OUTPUT_DIR
    / "sube_2025_ranking_h3.parquet"
)

OUTPUT_CATEGORIAS = (
    OUTPUT_DIR
    / "sube_2025_categorias_demanda.parquet"
)

OUTPUT_MODOS = (
    OUTPUT_DIR
    / "sube_2025_modos.parquet"
)

OUTPUT_HORAS = (
    OUTPUT_DIR
    / "sube_2025_horas_pico.parquet"
)

OUTPUT_RESUMEN = (
    OUTPUT_DIR
    / "sube_2025_resumen_territorial.json"
)


# ============================================================
# CONFIGURACIÓN ANALÍTICA
# ============================================================

EXPECTED_H3 = 6785
EXPECTED_OPERACIONES = 11919434

CATEGORIAS = [
    "HOTSPOT_EXTREMO",
    "HOTSPOT_ALTO",
    "DEMANDA_ALTA",
    "DEMANDA_MEDIA",
    "DEMANDA_BAJA",
]


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ANÁLISIS TERRITORIAL SUBE 2025")
print("=" * 70)


# ============================================================
# CARGAR
# ============================================================

print("\nCargando:")
print(INPUT_FILE)

gdf = gpd.read_parquet(INPUT_FILE)

print(f"H3 cargados: {len(gdf):,}")


# ============================================================
# VALIDACIÓN ESTRUCTURAL
# ============================================================

columnas_requeridas = [
    "id_h3",
    "operaciones_totales",
    "jurisdiccion",
    "provincia",
    "categoria_demanda",
    "hora_pico",
    "modo_dominante",
    "operaciones_hora_pico",
]

faltantes = [
    columna
    for columna in columnas_requeridas
    if columna not in gdf.columns
]

if faltantes:
    raise ValueError(
        "Faltan columnas requeridas:\n"
        + "\n".join(f"  - {c}" for c in faltantes)
    )


# ============================================================
# VALIDAR CANTIDAD H3
# ============================================================

if len(gdf) != EXPECTED_H3:

    raise ValueError(
        f"Se esperaban {EXPECTED_H3:,} H3, "
        f"pero se encontraron {len(gdf):,}."
    )


# ============================================================
# VALIDAR DUPLICADOS
# ============================================================

duplicados = (
    gdf["id_h3"]
    .duplicated()
    .sum()
)

if duplicados > 0:

    raise ValueError(
        f"Existen {duplicados:,} H3 duplicados."
    )


# ============================================================
# NORMALIZAR OPERACIONES
# ============================================================

gdf["operaciones_totales"] = pd.to_numeric(
    gdf["operaciones_totales"],
    errors="coerce",
).fillna(0)


gdf["operaciones_hora_pico"] = pd.to_numeric(
    gdf["operaciones_hora_pico"],
    errors="coerce",
).fillna(0)


# ============================================================
# TOTAL OPERACIONES
# ============================================================

TOTAL_OPERACIONES = (
    gdf["operaciones_totales"]
    .sum()
)

print(
    f"\nOperaciones totales: "
    f"{TOTAL_OPERACIONES:,.0f}"
)


# ============================================================
# VALIDAR OPERACIONES
# ============================================================

if abs(
    TOTAL_OPERACIONES - EXPECTED_OPERACIONES
) > 0.01:

    print(
        "\nADVERTENCIA:"
    )

    print(
        f"Se esperaban "
        f"{EXPECTED_OPERACIONES:,} operaciones."
    )

    print(
        f"Se encontraron "
        f"{TOTAL_OPERACIONES:,.0f}."
    )


# ============================================================
# FUNCIÓN MODA SEGURA
# ============================================================

def moda_segura(series):

    series = series.dropna()

    if len(series) == 0:
        return None

    moda = series.mode()

    if len(moda) == 0:
        return None

    return moda.iloc[0]


# ============================================================
# FUNCIÓN PORCENTUAL
# ============================================================

def porcentaje(valor, total):

    if total == 0:
        return 0.0

    return valor / total * 100


# ============================================================
# PREPARAR CATEGORÍAS
# ============================================================

gdf["categoria_demanda"] = (
    gdf["categoria_demanda"]
    .fillna("SIN_CATEGORIA")
)


# ============================================================
# ============================================================
# JURISDICCIONES
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS POR JURISDICCIÓN")
print("=" * 70)


gdf_jur = gdf[
    gdf["jurisdiccion"].notna()
].copy()


jurisdicciones = (
    gdf_jur
    .groupby(
        [
            "provincia",
            "jurisdiccion",
        ],
        dropna=False,
    )
    .agg(
        h3=(
            "id_h3",
            "nunique",
        ),

        operaciones=(
            "operaciones_totales",
            "sum",
        ),

        operaciones_hora_pico=(
            "operaciones_hora_pico",
            "sum",
        ),

        hotspots_extremos=(
            "categoria_demanda",
            lambda x:
                (
                    x == "HOTSPOT_EXTREMO"
                ).sum(),
        ),

        hotspots_altos=(
            "categoria_demanda",
            lambda x:
                (
                    x == "HOTSPOT_ALTO"
                ).sum(),
        ),

        demanda_alta=(
            "categoria_demanda",
            lambda x:
                (
                    x == "DEMANDA_ALTA"
                ).sum(),
        ),

        demanda_media=(
            "categoria_demanda",
            lambda x:
                (
                    x == "DEMANDA_MEDIA"
                ).sum(),
        ),

        demanda_baja=(
            "categoria_demanda",
            lambda x:
                (
                    x == "DEMANDA_BAJA"
                ).sum(),
        ),

        hora_pico=(
            "hora_pico",
            moda_segura,
        ),

        modo_dominante=(
            "modo_dominante",
            moda_segura,
        ),
    )
    .reset_index()
)


# ============================================================
# MÉTRICAS DERIVADAS
# ============================================================

jurisdicciones["pct_operaciones"] = (
    jurisdicciones["operaciones"]
    / TOTAL_OPERACIONES
    * 100
)

jurisdicciones["pct_h3"] = (
    jurisdicciones["h3"]
    / len(gdf)
    * 100
)

jurisdicciones["operaciones_por_h3"] = (
    jurisdicciones["operaciones"]
    / jurisdicciones["h3"]
)

jurisdicciones["h3_hotspot"] = (
    jurisdicciones["hotspots_extremos"]
    + jurisdicciones["hotspots_altos"]
)

jurisdicciones["pct_h3_hotspot"] = (
    jurisdicciones["h3_hotspot"]
    / jurisdicciones["h3"]
    * 100
)

jurisdicciones["operaciones_hotspot"] = (
    gdf_jur[
        gdf_jur["categoria_demanda"].isin(
            [
                "HOTSPOT_EXTREMO",
                "HOTSPOT_ALTO",
            ]
        )
    ]
    .groupby(
        [
            "provincia",
            "jurisdiccion",
        ]
    )["operaciones_totales"]
    .sum()
    .reindex(
        pd.MultiIndex.from_frame(
            jurisdicciones[
                [
                    "provincia",
                    "jurisdiccion",
                ]
            ]
        )
    )
    .fillna(0)
    .values
)

jurisdicciones["pct_operaciones_hotspot"] = (
    jurisdicciones["operaciones_hotspot"]
    / jurisdicciones["operaciones"]
    * 100
)


# ============================================================
# RANKING POR OPERACIONES
# ============================================================

jurisdicciones = (
    jurisdicciones
    .sort_values(
        [
            "operaciones",
            "jurisdiccion",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

jurisdicciones["ranking"] = (
    jurisdicciones.index + 1
)


# ============================================================
# PARTICIPACIÓN ACUMULADA
# ============================================================

jurisdicciones["pct_operaciones_acumulado"] = (
    jurisdicciones["pct_operaciones"]
    .cumsum()
)


# ============================================================
# REORDENAR
# ============================================================

columnas_jurisdicciones = [
    "ranking",
    "provincia",
    "jurisdiccion",
    "h3",
    "operaciones",
    "pct_operaciones",
    "pct_operaciones_acumulado",
    "pct_h3",
    "operaciones_por_h3",
    "h3_hotspot",
    "pct_h3_hotspot",
    "operaciones_hotspot",
    "pct_operaciones_hotspot",
    "hotspots_extremos",
    "hotspots_altos",
    "demanda_alta",
    "demanda_media",
    "demanda_baja",
    "hora_pico",
    "operaciones_hora_pico",
    "modo_dominante",
]


jurisdicciones = jurisdicciones[
    columnas_jurisdicciones
]


# ============================================================
# MOSTRAR TOP JURISDICCIONES
# ============================================================

print("\nTOP 30 JURISDICCIONES")

print(
    jurisdicciones
    .head(30)
    .to_string(
        index=False,
        formatters={
            "pct_operaciones":
                "{:.2f}".format,

            "pct_operaciones_acumulado":
                "{:.2f}".format,

            "pct_h3":
                "{:.2f}".format,

            "operaciones_por_h3":
                "{:.1f}".format,

            "pct_h3_hotspot":
                "{:.2f}".format,

            "pct_operaciones_hotspot":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# ============================================================
# PROVINCIAS
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS POR PROVINCIA")
print("=" * 70)


provincias = (
    gdf
    .groupby(
        "provincia",
        dropna=False,
    )
    .agg(
        h3=(
            "id_h3",
            "nunique",
        ),

        operaciones=(
            "operaciones_totales",
            "sum",
        ),

        operaciones_hora_pico=(
            "operaciones_hora_pico",
            "sum",
        ),

        hotspots_extremos=(
            "categoria_demanda",
            lambda x:
                (
                    x == "HOTSPOT_EXTREMO"
                ).sum(),
        ),

        hotspots_altos=(
            "categoria_demanda",
            lambda x:
                (
                    x == "HOTSPOT_ALTO"
                ).sum(),
        ),

        demanda_alta=(
            "categoria_demanda",
            lambda x:
                (
                    x == "DEMANDA_ALTA"
                ).sum(),
        ),

        demanda_media=(
            "categoria_demanda",
            lambda x:
                (
                    x == "DEMANDA_MEDIA"
                ).sum(),
        ),

        demanda_baja=(
            "categoria_demanda",
            lambda x:
                (
                    x == "DEMANDA_BAJA"
                ).sum(),
        ),

        hora_pico=(
            "hora_pico",
            moda_segura,
        ),

        modo_dominante=(
            "modo_dominante",
            moda_segura,
        ),
    )
    .reset_index()
)


provincias["pct_operaciones"] = (
    provincias["operaciones"]
    / TOTAL_OPERACIONES
    * 100
)

provincias["pct_h3"] = (
    provincias["h3"]
    / len(gdf)
    * 100
)

provincias["operaciones_por_h3"] = (
    provincias["operaciones"]
    / provincias["h3"]
)

provincias["h3_hotspot"] = (
    provincias["hotspots_extremos"]
    + provincias["hotspots_altos"]
)

provincias["pct_h3_hotspot"] = (
    provincias["h3_hotspot"]
    / provincias["h3"]
    * 100
)


# ============================================================
# OPERACIONES HOTSPOT POR PROVINCIA
# ============================================================

hotspot_provincia = (
    gdf[
        gdf["categoria_demanda"].isin(
            [
                "HOTSPOT_EXTREMO",
                "HOTSPOT_ALTO",
            ]
        )
    ]
    .groupby("provincia")[
        "operaciones_totales"
    ]
    .sum()
)

provincias["operaciones_hotspot"] = (
    provincias["provincia"]
    .map(hotspot_provincia)
    .fillna(0)
)

provincias["pct_operaciones_hotspot"] = (
    provincias["operaciones_hotspot"]
    / provincias["operaciones"]
    * 100
)


# ============================================================
# RANKING PROVINCIAS
# ============================================================

provincias = (
    provincias
    .sort_values(
        "operaciones",
        ascending=False,
    )
    .reset_index(drop=True)
)

provincias["ranking"] = (
    provincias.index + 1
)

provincias["pct_operaciones_acumulado"] = (
    provincias["pct_operaciones"]
    .cumsum()
)


columnas_provincias = [
    "ranking",
    "provincia",
    "h3",
    "operaciones",
    "pct_operaciones",
    "pct_operaciones_acumulado",
    "pct_h3",
    "operaciones_por_h3",
    "h3_hotspot",
    "pct_h3_hotspot",
    "operaciones_hotspot",
    "pct_operaciones_hotspot",
    "hotspots_extremos",
    "hotspots_altos",
    "demanda_alta",
    "demanda_media",
    "demanda_baja",
    "hora_pico",
    "operaciones_hora_pico",
    "modo_dominante",
]

provincias = provincias[
    columnas_provincias
]


print(
    provincias.to_string(
        index=False,
        formatters={
            "pct_operaciones":
                "{:.2f}".format,

            "pct_operaciones_acumulado":
                "{:.2f}".format,

            "pct_h3":
                "{:.2f}".format,

            "operaciones_por_h3":
                "{:.1f}".format,

            "pct_h3_hotspot":
                "{:.2f}".format,

            "pct_operaciones_hotspot":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# ============================================================
# CONCENTRACIÓN TERRITORIAL
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("CONCENTRACIÓN TERRITORIAL")
print("=" * 70)


for n in [1, 3, 5, 10, 20, 30]:

    operaciones = (
        jurisdicciones
        .head(n)["operaciones"]
        .sum()
    )

    pct = porcentaje(
        operaciones,
        TOTAL_OPERACIONES,
    )

    print(
        f"Top {n:2d} jurisdicciones: "
        f"{operaciones:>12,.0f} operaciones "
        f"({pct:6.2f}%)"
    )


# ============================================================
# ============================================================
# HOTSPOTS TERRITORIALES
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("HOTSPOTS TERRITORIALES")
print("=" * 70)


hotspots = gdf[
    gdf["categoria_demanda"].isin(
        [
            "HOTSPOT_EXTREMO",
            "HOTSPOT_ALTO",
        ]
    )
].copy()


hotspots = hotspots.sort_values(
    "operaciones_totales",
    ascending=False,
).reset_index(drop=True)


hotspots["ranking_hotspot"] = (
    hotspots.index + 1
)


hotspots["pct_operaciones"] = (
    hotspots["operaciones_totales"]
    / TOTAL_OPERACIONES
    * 100
)


hotspots["pct_operaciones_acumulado"] = (
    hotspots["pct_operaciones"]
    .cumsum()
)


print(
    f"H3 HOTSPOT: "
    f"{len(hotspots):,}"
)


print(
    f"Operaciones en HOTSPOTS: "
    f"{hotspots['operaciones_totales'].sum():,.0f}"
)


pct_hotspot_total = porcentaje(
    hotspots["operaciones_totales"].sum(),
    TOTAL_OPERACIONES,
)


print(
    f"% operaciones en HOTSPOTS: "
    f"{pct_hotspot_total:.2f}%"
)


print("\nTOP 30 HOTSPOTS")

print(
    hotspots[
        [
            "ranking_hotspot",
            "id_h3",
            "provincia",
            "jurisdiccion",
            "operaciones_totales",
            "pct_operaciones",
            "categoria_demanda",
            "hora_pico",
            "modo_dominante",
        ]
    ]
    .head(30)
    .to_string(
        index=False,
        formatters={
            "pct_operaciones":
                "{:.3f}".format,
        },
    )
)


# ============================================================
# ============================================================
# RANKING DE TODOS LOS H3
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("RANKING DE H3")
print("=" * 70)


ranking_h3 = gdf[
    [
        "id_h3",
        "provincia",
        "jurisdiccion",
        "operaciones_totales",
        "categoria_demanda",
        "hora_pico",
        "operaciones_hora_pico",
        "modo_dominante",
        "metodo_asignacion",
        "distancia_asignacion_m",
    ]
].copy()


ranking_h3 = ranking_h3.sort_values(
    "operaciones_totales",
    ascending=False,
).reset_index(drop=True)


ranking_h3["ranking"] = (
    ranking_h3.index + 1
)


ranking_h3["pct_operaciones"] = (
    ranking_h3["operaciones_totales"]
    / TOTAL_OPERACIONES
    * 100
)


ranking_h3["pct_operaciones_acumulado"] = (
    ranking_h3["pct_operaciones"]
    .cumsum()
)


ranking_h3 = ranking_h3[
    [
        "ranking",
        "id_h3",
        "provincia",
        "jurisdiccion",
        "operaciones_totales",
        "pct_operaciones",
        "pct_operaciones_acumulado",
        "categoria_demanda",
        "hora_pico",
        "operaciones_hora_pico",
        "modo_dominante",
        "metodo_asignacion",
        "distancia_asignacion_m",
    ]
]


print(
    ranking_h3
    .head(30)
    .to_string(
        index=False,
        formatters={
            "pct_operaciones":
                "{:.3f}".format,

            "pct_operaciones_acumulado":
                "{:.3f}".format,
        },
    )
)


# ============================================================
# ============================================================
# DISTRIBUCIÓN POR CATEGORÍA
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUCIÓN POR CATEGORÍA DE DEMANDA")
print("=" * 70)


categorias = (
    gdf
    .groupby(
        "categoria_demanda",
        dropna=False,
    )
    .agg(
        h3=(
            "id_h3",
            "nunique",
        ),

        operaciones=(
            "operaciones_totales",
            "sum",
        ),

        operaciones_hora_pico=(
            "operaciones_hora_pico",
            "sum",
        ),
    )
    .reset_index()
)


categorias["pct_h3"] = (
    categorias["h3"]
    / len(gdf)
    * 100
)


categorias["pct_operaciones"] = (
    categorias["operaciones"]
    / TOTAL_OPERACIONES
    * 100
)


categorias = categorias.sort_values(
    "operaciones",
    ascending=False,
).reset_index(drop=True)


categorias["ranking"] = (
    categorias.index + 1
)


categorias = categorias[
    [
        "ranking",
        "categoria_demanda",
        "h3",
        "pct_h3",
        "operaciones",
        "pct_operaciones",
        "operaciones_hora_pico",
    ]
]


print(
    categorias.to_string(
        index=False,
        formatters={
            "pct_h3":
                "{:.2f}".format,

            "pct_operaciones":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# ============================================================
# MODOS DOMINANTES
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("MODOS DOMINANTES")
print("=" * 70)


modos = (
    gdf
    .groupby(
        "modo_dominante",
        dropna=False,
    )
    .agg(
        h3=(
            "id_h3",
            "nunique",
        ),

        operaciones=(
            "operaciones_totales",
            "sum",
        ),
    )
    .reset_index()
)


modos["pct_h3"] = (
    modos["h3"]
    / len(gdf)
    * 100
)


modos["pct_operaciones"] = (
    modos["operaciones"]
    / TOTAL_OPERACIONES
    * 100
)


modos = modos.sort_values(
    "operaciones",
    ascending=False,
).reset_index(drop=True)


modos["ranking"] = (
    modos.index + 1
)


print(
    modos.to_string(
        index=False,
        formatters={
            "pct_h3":
                "{:.2f}".format,

            "pct_operaciones":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# ============================================================
# HORAS PICO
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("HORAS PICO")
print("=" * 70)


horas = (
    gdf
    .groupby(
        "hora_pico",
        dropna=False,
    )
    .agg(
        h3=(
            "id_h3",
            "nunique",
        ),

        operaciones=(
            "operaciones_totales",
            "sum",
        ),

        operaciones_hora_pico=(
            "operaciones_hora_pico",
            "sum",
        ),
    )
    .reset_index()
)


horas["pct_h3"] = (
    horas["h3"]
    / len(gdf)
    * 100
)


horas["pct_operaciones"] = (
    horas["operaciones"]
    / TOTAL_OPERACIONES
    * 100
)


horas = horas.sort_values(
    "operaciones",
    ascending=False,
).reset_index(drop=True)


horas["ranking"] = (
    horas.index + 1
)


print(
    horas.to_string(
        index=False,
        formatters={
            "pct_h3":
                "{:.2f}".format,

            "pct_operaciones":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# ============================================================
# CABA VS BUENOS AIRES
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("CABA VS BUENOS AIRES")
print("=" * 70)


comparacion = (
    gdf
    .groupby(
        "provincia",
        dropna=False,
    )
    .agg(
        h3=(
            "id_h3",
            "nunique",
        ),

        operaciones=(
            "operaciones_totales",
            "sum",
        ),

        hotspots=(
            "categoria_demanda",
            lambda x:
                x.isin(
                    [
                        "HOTSPOT_EXTREMO",
                        "HOTSPOT_ALTO",
                    ]
                ).sum(),
        ),
    )
    .reset_index()
)


comparacion["pct_h3"] = (
    comparacion["h3"]
    / len(gdf)
    * 100
)


comparacion["pct_operaciones"] = (
    comparacion["operaciones"]
    / TOTAL_OPERACIONES
    * 100
)


comparacion["operaciones_por_h3"] = (
    comparacion["operaciones"]
    / comparacion["h3"]
)


comparacion["pct_h3_hotspot"] = (
    comparacion["hotspots"]
    / comparacion["h3"]
    * 100
)


print(
    comparacion.to_string(
        index=False,
        formatters={
            "pct_h3":
                "{:.2f}".format,

            "pct_operaciones":
                "{:.2f}".format,

            "operaciones_por_h3":
                "{:.1f}".format,

            "pct_h3_hotspot":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# ============================================================
# MÉTODOS DE ASIGNACIÓN
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("MÉTODOS DE ASIGNACIÓN TERRITORIAL")
print("=" * 70)


metodos = (
    gdf
    .groupby(
        "metodo_asignacion",
        dropna=False,
    )
    .agg(
        h3=(
            "id_h3",
            "nunique",
        ),

        operaciones=(
            "operaciones_totales",
            "sum",
        ),
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
    / TOTAL_OPERACIONES
    * 100
)


print(
    metodos.to_string(
        index=False,
        formatters={
            "pct_h3":
                "{:.2f}".format,

            "pct_operaciones":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# ============================================================
# VALIDACIÓN ANALÍTICA
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("VALIDACIÓN ANALÍTICA")
print("=" * 70)


checks = []


# Cantidad H3
checks.append(
    (
        "Cantidad H3",
        len(gdf) == EXPECTED_H3,
    )
)


# H3 únicos
checks.append(
    (
        "H3 únicos",
        not gdf["id_h3"].duplicated().any(),
    )
)


# Operaciones
checks.append(
    (
        "Operaciones conservadas",
        abs(
            TOTAL_OPERACIONES
            - EXPECTED_OPERACIONES
        ) <= 0.01,
    )
)


# Jurisdicciones
checks.append(
    (
        "Jurisdicciones completas",
        gdf["jurisdiccion"].notna().all(),
    )
)


# Provincias
checks.append(
    (
        "Provincias completas",
        gdf["provincia"].notna().all(),
    )
)


# Categorías
checks.append(
    (
        "Categoría de demanda",
        gdf["categoria_demanda"].notna().all(),
    )
)


# Método
checks.append(
    (
        "Método de asignación",
        gdf["metodo_asignacion"].notna().all(),
    )
)


for nombre, ok in checks:

    estado = "OK" if ok else "ERROR"

    print(
        f"{estado:>7}  {nombre}"
    )


errores = [
    nombre
    for nombre, ok in checks
    if not ok
]


if errores:

    raise ValueError(
        "La validación analítica falló en: "
        + ", ".join(errores)
    )


# ============================================================
# RESUMEN EJECUTIVO
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN EJECUTIVO")
print("=" * 70)


total_h3_hotspot = len(hotspots)

operaciones_hotspot = (
    hotspots[
        "operaciones_totales"
    ]
    .sum()
)


top_10_operaciones = (
    jurisdicciones
    .head(10)["operaciones"]
    .sum()
)


top_10_pct = porcentaje(
    top_10_operaciones,
    TOTAL_OPERACIONES,
)


jurisdiccion_principal = (
    jurisdicciones.iloc[0]
)


provincia_principal = (
    provincias.iloc[0]
)


print(
    f"\nH3 analizados: "
    f"{len(gdf):,}"
)

print(
    f"Operaciones: "
    f"{TOTAL_OPERACIONES:,.0f}"
)

print(
    f"Jurisdicciones: "
    f"{len(jurisdicciones):,}"
)

print(
    f"Provincias: "
    f"{len(provincias):,}"
)

print(
    f"H3 hotspot: "
    f"{total_h3_hotspot:,}"
)

print(
    f"Operaciones hotspot: "
    f"{operaciones_hotspot:,.0f}"
)

print(
    f"% operaciones hotspot: "
    f"{porcentaje(operaciones_hotspot, TOTAL_OPERACIONES):.2f}%"
)

print(
    f"\nJurisdicción con más operaciones: "
    f"{jurisdiccion_principal['jurisdiccion']}"
)

print(
    f"Operaciones: "
    f"{jurisdiccion_principal['operaciones']:,.0f}"
)

print(
    f"Participación: "
    f"{jurisdiccion_principal['pct_operaciones']:.2f}%"
)

print(
    f"\nProvincia con más operaciones: "
    f"{provincia_principal['provincia']}"
)

print(
    f"Operaciones: "
    f"{provincia_principal['operaciones']:,.0f}"
)

print(
    f"Participación: "
    f"{provincia_principal['pct_operaciones']:.2f}%"
)

print(
    f"\nTop 10 jurisdicciones: "
    f"{top_10_operaciones:,.0f} operaciones"
)

print(
    f"Participación Top 10: "
    f"{top_10_pct:.2f}%"
)


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("GUARDANDO RESULTADOS")
print("=" * 70)


jurisdicciones.to_parquet(
    OUTPUT_JURISDICCIONES,
    index=False,
)


provincias.to_parquet(
    OUTPUT_PROVINCIAS,
    index=False,
)


hotspots.to_parquet(
    OUTPUT_HOTSPOTS,
    index=False,
)


ranking_h3.to_parquet(
    OUTPUT_H3_RANKING,
    index=False,
)


categorias.to_parquet(
    OUTPUT_CATEGORIAS,
    index=False,
)


modos.to_parquet(
    OUTPUT_MODOS,
    index=False,
)


horas.to_parquet(
    OUTPUT_HORAS,
    index=False,
)


# ============================================================
# RESUMEN JSON
# ============================================================

import json


resumen = {
    "h3": int(len(gdf)),

    "operaciones": int(
        TOTAL_OPERACIONES
    ),

    "jurisdicciones": int(
        len(jurisdicciones)
    ),

    "provincias": int(
        len(provincias)
    ),

    "h3_hotspot": int(
        total_h3_hotspot
    ),

    "operaciones_hotspot": int(
        operaciones_hotspot
    ),

    "pct_operaciones_hotspot": float(
        porcentaje(
            operaciones_hotspot,
            TOTAL_OPERACIONES,
        )
    ),

    "top_10_operaciones": int(
        top_10_operaciones
    ),

    "top_10_pct": float(
        top_10_pct
    ),

    "jurisdiccion_principal": str(
        jurisdiccion_principal[
            "jurisdiccion"
        ]
    ),

    "provincia_principal": str(
        provincia_principal[
            "provincia"
        ]
    ),
}


with open(
    OUTPUT_RESUMEN,
    "w",
    encoding="utf-8",
) as archivo:

    json.dump(
        resumen,
        archivo,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# MOSTRAR ARCHIVOS
# ============================================================

print("\nArchivos generados:")

print(
    f"\nJurisdicciones:"
    f"\n{OUTPUT_JURISDICCIONES}"
)

print(
    f"\nProvincias:"
    f"\n{OUTPUT_PROVINCIAS}"
)

print(
    f"\nHotspots:"
    f"\n{OUTPUT_HOTSPOTS}"
)

print(
    f"\nRanking H3:"
    f"\n{OUTPUT_H3_RANKING}"
)

print(
    f"\nCategorías:"
    f"\n{OUTPUT_CATEGORIAS}"
)

print(
    f"\nModos:"
    f"\n{OUTPUT_MODOS}"
)

print(
    f"\nHoras:"
    f"\n{OUTPUT_HORAS}"
)

print(
    f"\nResumen:"
    f"\n{OUTPUT_RESUMEN}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS TERRITORIAL FINALIZADO")
print("=" * 70)

print(
    f"H3 analizados: "
    f"{len(gdf):,}"
)

print(
    f"Operaciones: "
    f"{TOTAL_OPERACIONES:,.0f}"
)

print(
    f"Jurisdicciones: "
    f"{len(jurisdicciones):,}"
)

print(
    f"Hotspots: "
    f"{len(hotspots):,}"
)

print(
    f"Top 10 concentración: "
    f"{top_10_pct:.2f}%"
)