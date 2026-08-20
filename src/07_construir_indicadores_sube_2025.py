from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# IMPORTANTE:
# Ahora usamos el dataset territorial ya validado.
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_jurisdicciones.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_H3 = (
    OUTPUT_DIR
    / "sube_2025_h3.parquet"
)

OUTPUT_H3_HORA = (
    OUTPUT_DIR
    / "sube_2025_h3_hora.parquet"
)

OUTPUT_H3_MODO = (
    OUTPUT_DIR
    / "sube_2025_h3_modo.parquet"
)

OUTPUT_JURISDICCIONES = (
    OUTPUT_DIR
    / "sube_2025_indicadores_jurisdicciones.parquet"
)

OUTPUT_JURISDICCIONES_HORA = (
    OUTPUT_DIR
    / "sube_2025_jurisdicciones_hora.parquet"
)

OUTPUT_JURISDICCIONES_MODO = (
    OUTPUT_DIR
    / "sube_2025_jurisdicciones_modo.parquet"
)

OUTPUT_FRANJAS = (
    OUTPUT_DIR
    / "sube_2025_jurisdicciones_franjas.parquet"
)

OUTPUT_RESUMEN = (
    OUTPUT_DIR
    / "sube_2025_indicadores_resumen.csv"
)


# ============================================================
# FRANJAS HORARIAS
# ============================================================

def clasificar_franja(hora):
    if pd.isna(hora):
        return "SIN_CLASIFICAR"

    hora = int(hora)

    if 0 <= hora <= 5:
        return "MADRUGADA"

    if 6 <= hora <= 9:
        return "MANANA"

    if 10 <= hora <= 15:
        return "MEDIODIA"

    if 16 <= hora <= 19:
        return "TARDE"

    if 20 <= hora <= 23:
        return "NOCHE"

    return "SIN_CLASIFICAR"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_columnas_modo(df):
    """
    Normaliza los nombres de columnas generados
    a partir del campo modo.
    """

    nuevas = []

    for column in df.columns:
        if isinstance(column, tuple):
            column = "_".join(
                str(x)
                for x in column
                if str(x) != ""
            )

        nuevas.append(str(column).lower())

    df.columns = nuevas

    return df


def agregar_porcentajes_modal(df):
    """
    Agrega porcentajes de participación por modo
    cuando existen las columnas correspondientes.
    """

    modos = [
        "colectivo",
        "tren",
        "subte",
        "lanchas",
    ]

    for modo in modos:

        column = f"operaciones_{modo}"

        if column in df.columns:

            df[f"pct_{modo}"] = (
                df[column]
                / df["operaciones_totales"]
                .replace(0, pd.NA)
                * 100
            )

    return df


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("CONSTRUCCIÓN DE INDICADORES SUBE 2025")
print("=" * 70)

print(f"\nArchivo de entrada:")
print(INPUT_FILE)


# ============================================================
# CARGA
# ============================================================

print("\n" + "=" * 70)
print("1. CARGANDO DATASET TERRITORIAL")
print("=" * 70)

gdf = gpd.read_parquet(INPUT_FILE)

print(f"Registros: {len(gdf):,}")
print(f"Columnas: {len(gdf.columns)}")

print("\nColumnas disponibles:")

for column in gdf.columns:
    print(f"  - {column}")


# ============================================================
# VALIDACIÓN
# ============================================================

print("\n" + "=" * 70)
print("2. VALIDACIÓN")
print("=" * 70)

required_columns = [
    "id",
    "id_h3",
    "hora",
    "modo",
    "cantidad_trx",
]

missing_columns = [
    column
    for column in required_columns
    if column not in gdf.columns
]

if missing_columns:

    raise ValueError(
        "Faltan columnas requeridas: "
        f"{missing_columns}"
    )

print("Columnas básicas: OK")


# ============================================================
# VALIDACIÓN H3
# ============================================================

print("\nValidando H3...")

print(
    f"H3 distintos: "
    f"{gdf['id_h3'].nunique():,}"
)

print(
    f"H3 nulos: "
    f"{gdf['id_h3'].isna().sum():,}"
)


# ============================================================
# VALIDACIÓN OPERACIONES
# ============================================================

print("\nValidando operaciones...")

gdf["cantidad_trx"] = pd.to_numeric(
    gdf["cantidad_trx"],
    errors="coerce",
)

if gdf["cantidad_trx"].isna().any():

    print(
        "ADVERTENCIA: existen valores nulos "
        "en cantidad_trx."
    )

    gdf["cantidad_trx"] = (
        gdf["cantidad_trx"]
        .fillna(0)
    )

print(
    f"Operaciones totales: "
    f"{gdf['cantidad_trx'].sum():,.0f}"
)


# ============================================================
# FRANJA HORARIA
# ============================================================

print("\nClasificando franjas horarias...")

gdf["franja"] = (
    gdf["hora"]
    .apply(clasificar_franja)
)

print(
    gdf["franja"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# DATOS CON GEOMETRÍA
# ============================================================

if "geometry" in gdf.columns:

    gdf_geo = gdf.loc[
        gdf["geometry"].notna()
    ].copy()

else:

    raise ValueError(
        "El dataset no contiene columna geometry."
    )


print(
    f"\nRegistros con geometría: "
    f"{len(gdf_geo):,}"
)

print(
    f"Registros sin geometría: "
    f"{len(gdf) - len(gdf_geo):,}"
)

print(
    f"Operaciones con geometría: "
    f"{gdf_geo['cantidad_trx'].sum():,.0f}"
)

print(
    f"Operaciones sin geometría: "
    f"{gdf.loc[gdf['geometry'].isna(), 'cantidad_trx'].sum():,.0f}"
)


# ============================================================
# INFORMACIÓN TERRITORIAL
# ============================================================

territorial_columns = [
    "id_jurisdiccion",
    "jurisdiccion",
    "provincia_jurisdiccion",
    "metodo_asignacion",
    "confianza_asignacion",
]

available_territorial_columns = [
    column
    for column in territorial_columns
    if column in gdf.columns
]

print("\nCampos territoriales disponibles:")

for column in available_territorial_columns:
    print(f"  - {column}")


# ============================================================
# ============================================================
# 3. INDICADORES POR H3
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("3. INDICADORES POR H3")
print("=" * 70)


# ------------------------------------------------------------
# Base general
# ------------------------------------------------------------

h3_base = (
    gdf_geo
    .groupby("id_h3")
    .agg(
        registros=("id", "size"),
        operaciones_totales=(
            "cantidad_trx",
            "sum",
        ),
        hora_promedio=(
            "hora",
            "mean",
        ),
    )
)


# ------------------------------------------------------------
# Operaciones por modo
# ------------------------------------------------------------

modo_pivot = (
    gdf_geo
    .pivot_table(
        index="id_h3",
        columns="modo",
        values="cantidad_trx",
        aggfunc="sum",
        fill_value=0,
    )
)

modo_pivot.columns = [
    f"operaciones_{str(column).lower()}"
    for column in modo_pivot.columns
]


# ------------------------------------------------------------
# Operaciones por franja
# ------------------------------------------------------------

franja_pivot = (
    gdf_geo
    .pivot_table(
        index="id_h3",
        columns="franja",
        values="cantidad_trx",
        aggfunc="sum",
        fill_value=0,
    )
)

franja_pivot.columns = [
    f"operaciones_{str(column).lower()}"
    for column in franja_pivot.columns
]


# ------------------------------------------------------------
# Unión
# ------------------------------------------------------------

h3 = (
    h3_base
    .join(
        modo_pivot,
        how="left",
    )
    .join(
        franja_pivot,
        how="left",
    )
)


# ------------------------------------------------------------
# Rellenar indicadores
# ------------------------------------------------------------

indicator_columns = [
    column
    for column in h3.columns
    if column.startswith("operaciones_")
]

h3[indicator_columns] = (
    h3[indicator_columns]
    .fillna(0)
)


# ============================================================
# HORA PICO
# ============================================================

print("Calculando hora pico por H3...")

hora_h3 = (
    gdf_geo
    .groupby(
        [
            "id_h3",
            "hora",
        ]
    )["cantidad_trx"]
    .sum()
    .reset_index()
)

idx_peak = (
    hora_h3
    .groupby("id_h3")["cantidad_trx"]
    .idxmax()
)

hora_pico = (
    hora_h3
    .loc[idx_peak]
    .set_index("id_h3")
    .rename(
        columns={
            "hora": "hora_pico",
            "cantidad_trx":
                "operaciones_hora_pico",
        }
    )
)

h3 = h3.join(
    hora_pico,
    how="left",
)


# ============================================================
# PORCENTAJES MODALES
# ============================================================

print("Calculando composición modal...")

h3 = agregar_porcentajes_modal(h3)


# ============================================================
# RANKING
# ============================================================

print("Calculando ranking espacial...")

h3["ranking_operaciones"] = (
    h3["operaciones_totales"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("Int64")
)


# ============================================================
# INFORMACIÓN TERRITORIAL DEL H3
# ============================================================

territory_h3_columns = [
    "id_h3"
] + [
    column
    for column in available_territorial_columns
]


territory_h3 = (
    gdf[
        territory_h3_columns
    ]
    .drop_duplicates(
        "id_h3"
    )
    .set_index("id_h3")
)


h3 = territory_h3.join(
    h3,
    how="right",
)


# ============================================================
# GEOMETRÍA
# ============================================================

geometry = (
    gdf_geo[
        [
            "id_h3",
            "geometry",
        ]
    ]
    .drop_duplicates(
        "id_h3"
    )
    .set_index("id_h3")
)

h3 = geometry.join(
    h3,
    how="right",
)

h3 = gpd.GeoDataFrame(
    h3,
    geometry="geometry",
    crs=gdf.crs,
)

h3 = h3.reset_index()


# ============================================================
# ORDEN
# ============================================================

h3 = h3.sort_values(
    "operaciones_totales",
    ascending=False,
)


# ============================================================
# GUARDAR H3
# ============================================================

print(
    f"\nGuardando:"
    f"\n{OUTPUT_H3}"
)

h3.to_parquet(
    OUTPUT_H3,
    index=False,
)


# ============================================================
# ============================================================
# 4. H3 + HORA
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("4. INDICADORES H3 + HORA")
print("=" * 70)

h3_hora = (
    gdf_geo
    .groupby(
        [
            "id_h3",
            "hora",
            "franja",
        ],
        as_index=False,
    )
    .agg(
        registros=("id", "size"),
        operaciones=(
            "cantidad_trx",
            "sum",
        ),
    )
)

h3_hora = h3_hora.merge(
    geometry.reset_index(),
    on="id_h3",
    how="left",
)

h3_hora = gpd.GeoDataFrame(
    h3_hora,
    geometry="geometry",
    crs=gdf.crs,
)

print(
    f"Registros H3 + hora: "
    f"{len(h3_hora):,}"
)

print(
    f"Guardando:"
    f"\n{OUTPUT_H3_HORA}"
)

h3_hora.to_parquet(
    OUTPUT_H3_HORA,
    index=False,
)


# ============================================================
# ============================================================
# 5. H3 + MODO
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("5. INDICADORES H3 + MODO")
print("=" * 70)

h3_modo = (
    gdf_geo
    .groupby(
        [
            "id_h3",
            "modo",
        ],
        as_index=False,
    )
    .agg(
        registros=("id", "size"),
        operaciones=(
            "cantidad_trx",
            "sum",
        ),
    )
)

h3_modo = h3_modo.merge(
    geometry.reset_index(),
    on="id_h3",
    how="left",
)

h3_modo = gpd.GeoDataFrame(
    h3_modo,
    geometry="geometry",
    crs=gdf.crs,
)

print(
    f"Registros H3 + modo: "
    f"{len(h3_modo):,}"
)

print(
    f"Guardando:"
    f"\n{OUTPUT_H3_MODO}"
)

h3_modo.to_parquet(
    OUTPUT_H3_MODO,
    index=False,
)


# ============================================================
# ============================================================
# 6. INDICADORES POR JURISDICCIÓN
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("6. INDICADORES POR JURISDICCIÓN")
print("=" * 70)


if "id_jurisdiccion" not in gdf_geo.columns:

    raise ValueError(
        "El dataset territorial no contiene "
        "'id_jurisdiccion'."
    )


# ------------------------------------------------------------
# Base general
# ------------------------------------------------------------

jur_base = (
    gdf_geo
    .groupby(
        [
            "id_jurisdiccion",
            "jurisdiccion",
        ],
        as_index=False,
    )
    .agg(
        registros=(
            "id",
            "size",
        ),
        h3_distintos=(
            "id_h3",
            "nunique",
        ),
        operaciones_totales=(
            "cantidad_trx",
            "sum",
        ),
        hora_promedio=(
            "hora",
            "mean",
        ),
    )
)


# ------------------------------------------------------------
# Operaciones por modo
# ------------------------------------------------------------

jur_modo = (
    gdf_geo
    .pivot_table(
        index=[
            "id_jurisdiccion",
            "jurisdiccion",
        ],
        columns="modo",
        values="cantidad_trx",
        aggfunc="sum",
        fill_value=0,
    )
)

jur_modo.columns = [
    f"operaciones_{str(column).lower()}"
    for column in jur_modo.columns
]


# ------------------------------------------------------------
# Operaciones por franja
# ------------------------------------------------------------

jur_franja = (
    gdf_geo
    .pivot_table(
        index=[
            "id_jurisdiccion",
            "jurisdiccion",
        ],
        columns="franja",
        values="cantidad_trx",
        aggfunc="sum",
        fill_value=0,
    )
)

jur_franja.columns = [
    f"operaciones_{str(column).lower()}"
    for column in jur_franja.columns
]


# ------------------------------------------------------------
# Unión
# ------------------------------------------------------------

jur = (
    jur_base
    .set_index(
        [
            "id_jurisdiccion",
            "jurisdiccion",
        ]
    )
    .join(
        jur_modo,
        how="left",
    )
    .join(
        jur_franja,
        how="left",
    )
)


# ------------------------------------------------------------
# Rellenar
# ------------------------------------------------------------

indicator_columns = [
    column
    for column in jur.columns
    if column.startswith("operaciones_")
]

jur[indicator_columns] = (
    jur[indicator_columns]
    .fillna(0)
)


# ------------------------------------------------------------
# Porcentajes modales
# ------------------------------------------------------------

jur = agregar_porcentajes_modal(
    jur
)


# ============================================================
# HORA PICO POR JURISDICCIÓN
# ============================================================

print(
    "Calculando hora pico por jurisdicción..."
)

hora_jur = (
    gdf_geo
    .groupby(
        [
            "id_jurisdiccion",
            "hora",
        ]
    )["cantidad_trx"]
    .sum()
    .reset_index()
)

idx_peak_jur = (
    hora_jur
    .groupby(
        "id_jurisdiccion"
    )["cantidad_trx"]
    .idxmax()
)

hora_pico_jur = (
    hora_jur
    .loc[idx_peak_jur]
    .set_index(
        "id_jurisdiccion"
    )
    .rename(
        columns={
            "hora":
                "hora_pico",
            "cantidad_trx":
                "operaciones_hora_pico",
        }
    )
)

jur = jur.join(
    hora_pico_jur,
    how="left",
)


# ============================================================
# RANKING JURISDICCIONAL
# ============================================================

print(
    "Calculando ranking por jurisdicción..."
)

jur["ranking_operaciones"] = (
    jur["operaciones_totales"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("Int64")
)


# ============================================================
# PORCENTAJE DEL TOTAL
# ============================================================

total_operaciones_geo = (
    gdf_geo["cantidad_trx"]
    .sum()
)

jur["pct_operaciones_total"] = (
    jur["operaciones_totales"]
    / total_operaciones_geo
    * 100
)


# ============================================================
# REINICIAR ÍNDICE
# ============================================================

jur = jur.reset_index()


# ============================================================
# ORDEN
# ============================================================

jur = jur.sort_values(
    "operaciones_totales",
    ascending=False,
)


# ============================================================
# GEOMETRÍA TERRITORIAL
# ============================================================

if "geometry" in gdf.columns:

    geometria_jur = (
        gdf_geo[
            [
                "id_jurisdiccion",
                "jurisdiccion",
                "geometry",
            ]
        ]
        .drop_duplicates(
            "id_jurisdiccion"
        )
    )

    # Disolver los H3 de cada jurisdicción.
    # Esto crea una geometría territorial
    # basada en los H3 utilizados por SUBE.

    print(
        "\nConstruyendo geometría agregada "
        "por jurisdicción..."
    )

    geometria_jur = (
        geometria_jur
        .dissolve(
            by=[
                "id_jurisdiccion",
                "jurisdiccion",
            ],
            as_index=False,
        )
    )

    jur = jur.merge(
        geometria_jur,
        on=[
            "id_jurisdiccion",
            "jurisdiccion",
        ],
        how="left",
    )

    jur = gpd.GeoDataFrame(
        jur,
        geometry="geometry",
        crs=gdf.crs,
    )


# ============================================================
# GUARDAR JURISDICCIONES
# ============================================================

print(
    f"\nJurisdicciones con operaciones: "
    f"{len(jur):,}"
)

print(
    f"Guardando:"
    f"\n{OUTPUT_JURISDICCIONES}"
)

jur.to_parquet(
    OUTPUT_JURISDICCIONES,
    index=False,
)


# ============================================================
# ============================================================
# 7. JURISDICCIÓN + HORA
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("7. JURISDICCIÓN + HORA")
print("=" * 70)

jur_hora = (
    gdf_geo
    .groupby(
        [
            "id_jurisdiccion",
            "jurisdiccion",
            "hora",
            "franja",
        ],
        as_index=False,
    )
    .agg(
        registros=(
            "id",
            "size",
        ),
        operaciones=(
            "cantidad_trx",
            "sum",
        ),
    )
)

print(
    f"Registros jurisdicción + hora: "
    f"{len(jur_hora):,}"
)

print(
    f"Guardando:"
    f"\n{OUTPUT_JURISDICCIONES_HORA}"
)

jur_hora.to_parquet(
    OUTPUT_JURISDICCIONES_HORA,
    index=False,
)


# ============================================================
# ============================================================
# 8. JURISDICCIÓN + MODO
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("8. JURISDICCIÓN + MODO")
print("=" * 70)

jur_modo_detalle = (
    gdf_geo
    .groupby(
        [
            "id_jurisdiccion",
            "jurisdiccion",
            "modo",
        ],
        as_index=False,
    )
    .agg(
        registros=(
            "id",
            "size",
        ),
        operaciones=(
            "cantidad_trx",
            "sum",
        ),
    )
)

print(
    f"Registros jurisdicción + modo: "
    f"{len(jur_modo_detalle):,}"
)

print(
    f"Guardando:"
    f"\n{OUTPUT_JURISDICCIONES_MODO}"
)

jur_modo_detalle.to_parquet(
    OUTPUT_JURISDICCIONES_MODO,
    index=False,
)


# ============================================================
# ============================================================
# 9. JURISDICCIÓN + FRANJA
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("9. JURISDICCIÓN + FRANJA HORARIA")
print("=" * 70)

jur_franjas = (
    gdf_geo
    .groupby(
        [
            "id_jurisdiccion",
            "jurisdiccion",
            "franja",
        ],
        as_index=False,
    )
    .agg(
        registros=(
            "id",
            "size",
        ),
        operaciones=(
            "cantidad_trx",
            "sum",
        ),
    )
)

print(
    f"Registros jurisdicción + franja: "
    f"{len(jur_franjas):,}"
)

print(
    f"Guardando:"
    f"\n{OUTPUT_FRANJAS}"
)

jur_franjas.to_parquet(
    OUTPUT_FRANJAS,
    index=False,
)


# ============================================================
# ============================================================
# 10. RESUMEN GENERAL
# ============================================================
# ============================================================

print("\n" + "=" * 70)
print("10. RESUMEN GENERAL")
print("=" * 70)


total_registros = len(gdf)

registros_geo = len(gdf_geo)

registros_sin_geo = (
    total_registros
    - registros_geo
)

operaciones_total = (
    gdf["cantidad_trx"]
    .sum()
)

operaciones_geo = (
    gdf_geo["cantidad_trx"]
    .sum()
)

operaciones_sin_geo = (
    operaciones_total
    - operaciones_geo
)

cobertura_operaciones = (
    operaciones_geo
    / operaciones_total
    * 100
)


# ============================================================
# H3
# ============================================================

h3_utilizados = (
    gdf["id_h3"]
    .nunique()
)

h3_con_geo = (
    gdf_geo["id_h3"]
    .nunique()
)


# ============================================================
# JURISDICCIONES
# ============================================================

jurisdicciones = (
    gdf_geo["id_jurisdiccion"]
    .nunique()
)


# ============================================================
# HORA PICO GLOBAL
# ============================================================

hora_global = (
    gdf_geo
    .groupby("hora")["cantidad_trx"]
    .sum()
)

hora_pico_global = (
    hora_global.idxmax()
)

operaciones_hora_pico_global = (
    hora_global.max()
)


# ============================================================
# MODO PRINCIPAL GLOBAL
# ============================================================

modo_global = (
    gdf_geo
    .groupby("modo")["cantidad_trx"]
    .sum()
    .sort_values(
        ascending=False
    )
)

modo_principal = (
    modo_global.index[0]
)

operaciones_modo_principal = (
    modo_global.iloc[0]
)


# ============================================================
# JURISDICCIÓN PRINCIPAL
# ============================================================

jurisdiccion_principal = (
    jur.iloc[0]["jurisdiccion"]
)

operaciones_jurisdiccion_principal = (
    jur.iloc[0]["operaciones_totales"]
)


# ============================================================
# RESUMEN DATAFRAME
# ============================================================

resumen = pd.DataFrame(
    [
        {
            "indicador":
                "registros_totales",
            "valor":
                total_registros,
        },
        {
            "indicador":
                "registros_con_geometria",
            "valor":
                registros_geo,
        },
        {
            "indicador":
                "registros_sin_geometria",
            "valor":
                registros_sin_geo,
        },
        {
            "indicador":
                "h3_utilizados",
            "valor":
                h3_utilizados,
        },
        {
            "indicador":
                "h3_con_geometria",
            "valor":
                h3_con_geo,
        },
        {
            "indicador":
                "jurisdicciones_con_operaciones",
            "valor":
                jurisdicciones,
        },
        {
            "indicador":
                "operaciones_totales",
            "valor":
                operaciones_total,
        },
        {
            "indicador":
                "operaciones_con_geometria",
            "valor":
                operaciones_geo,
        },
        {
            "indicador":
                "operaciones_sin_geometria",
            "valor":
                operaciones_sin_geo,
        },
        {
            "indicador":
                "cobertura_operaciones_pct",
            "valor":
                cobertura_operaciones,
        },
        {
            "indicador":
                "hora_pico_global",
            "valor":
                hora_pico_global,
        },
        {
            "indicador":
                "operaciones_hora_pico_global",
            "valor":
                operaciones_hora_pico_global,
        },
        {
            "indicador":
                "modo_principal",
            "valor":
                modo_principal,
        },
        {
            "indicador":
                "operaciones_modo_principal",
            "valor":
                operaciones_modo_principal,
        },
        {
            "indicador":
                "jurisdiccion_principal",
            "valor":
                jurisdiccion_principal,
        },
        {
            "indicador":
                "operaciones_jurisdiccion_principal",
            "valor":
                operaciones_jurisdiccion_principal,
        },
    ]
)


# ============================================================
# GUARDAR RESUMEN
# ============================================================

print(
    f"\nGuardando resumen:"
    f"\n{OUTPUT_RESUMEN}"
)

resumen.to_csv(
    OUTPUT_RESUMEN,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# RESUMEN EN CONSOLA
# ============================================================

print("\n" + "=" * 70)
print("RESULTADO FINAL")
print("=" * 70)

print(
    f"""
Registros totales:
  {total_registros:,}

Registros con geometría:
  {registros_geo:,}

Registros sin geometría:
  {registros_sin_geo:,}

H3 utilizados:
  {h3_utilizados:,}

H3 con geometría:
  {h3_con_geo:,}

Jurisdicciones con operaciones:
  {jurisdicciones:,}

Operaciones totales:
  {operaciones_total:,.0f}

Operaciones con geometría:
  {operaciones_geo:,.0f}

Operaciones sin geometría:
  {operaciones_sin_geo:,.0f}

Cobertura de operaciones:
  {cobertura_operaciones:.2f}%

Hora pico global:
  {hora_pico_global:02d}:00

Operaciones hora pico:
  {operaciones_hora_pico_global:,.0f}

Modo principal:
  {modo_principal}

Operaciones modo principal:
  {operaciones_modo_principal:,.0f}

Jurisdicción principal:
  {jurisdiccion_principal}

Operaciones jurisdicción principal:
  {operaciones_jurisdiccion_principal:,.0f}
"""
)


# ============================================================
# FIN
# ============================================================

print("=" * 70)
print("INDICADORES SUBE 2025 CONSTRUIDOS CORRECTAMENTE")
print("=" * 70)