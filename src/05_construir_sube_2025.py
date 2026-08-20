from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUBE_PATH = RAW_DIR / "1.csv"
H3_PATH = RAW_DIR / "2" / "Hexagonos h3.shp"

OUTPUT_PATH = (
    PROCESSED_DIR
    / "sube_2025_geo.parquet"
)

OUTPUT_MISSING_H3 = (
    PROCESSED_DIR
    / "sube_2025_h3_sin_geometria.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED_DIR
    / "sube_2025_geo_resumen.csv"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def separador():
    print("=" * 70)


def porcentaje(parte, total):
    if total == 0:
        return 0.0

    return parte / total * 100


# ============================================================
# INICIO
# ============================================================

separador()
print("CONSTRUCCIÓN GEOGRÁFICA SUBE 2025")
separador()


# ============================================================
# CARGAR OPERACIONES SUBE
# ============================================================

print("\nCargando operaciones SUBE 2025...")

sube = pd.read_csv(
    SUBE_PATH
)

print(
    f"Registros SUBE: "
    f"{len(sube):,}"
)


# ============================================================
# VALIDAR COLUMNAS SUBE
# ============================================================

print("\nValidando columnas SUBE...")

columnas_requeridas = {
    "id_h3",
    "hora",
    "modo",
    "cantidad_trx",
}

faltantes = (
    columnas_requeridas
    - set(sube.columns)
)

if faltantes:
    raise ValueError(
        "Faltan columnas obligatorias en SUBE: "
        + ", ".join(sorted(faltantes))
    )

print("Columnas SUBE OK.")


# ============================================================
# NORMALIZAR ID H3
# ============================================================

print("\nNormalizando identificadores H3...")

sube["id_h3"] = (
    sube["id_h3"]
    .astype("string")
    .str.strip()
    .str.lower()
)

h3_nulos = sube["id_h3"].isna().sum()

print(
    f"id_h3 nulos en SUBE: "
    f"{h3_nulos:,}"
)

if h3_nulos > 0:
    raise ValueError(
        "Existen registros SUBE sin id_h3."
    )


# ============================================================
# VALIDAR H3 REPETIDOS
# ============================================================

print("\nValidando identificadores H3 en SUBE...")

h3_repetidos = (
    sube["id_h3"]
    .duplicated()
    .sum()
)

print(
    f"id_h3 repetidos en SUBE: "
    f"{h3_repetidos:,}"
)

if h3_repetidos > 0:
    print(
        "Los duplicados en SUBE son esperables "
        "porque existen múltiples observaciones "
        "por H3/hora/modo."
    )


# ============================================================
# VALIDAR OPERACIONES
# ============================================================

print("\nValidando cantidad de operaciones...")

sube["cantidad_trx"] = pd.to_numeric(
    sube["cantidad_trx"],
    errors="coerce"
)

operaciones_nulas = (
    sube["cantidad_trx"]
    .isna()
    .sum()
)

if operaciones_nulas > 0:
    raise ValueError(
        f"Hay {operaciones_nulas:,} registros "
        "con cantidad_trx inválida."
    )


operaciones_negativas = (
    sube["cantidad_trx"] < 0
).sum()

if operaciones_negativas > 0:
    raise ValueError(
        f"Hay {operaciones_negativas:,} "
        "registros con operaciones negativas."
    )


operaciones_totales = (
    sube["cantidad_trx"]
    .sum()
)

print(
    f"Operaciones totales: "
    f"{operaciones_totales:,.0f}"
)


# ============================================================
# CARGAR GEOMETRÍA H3
# ============================================================

print("\nCargando geometría H3...")

h3 = gpd.read_file(
    H3_PATH
)

print(
    f"Hexágonos: "
    f"{len(h3):,}"
)

print(
    f"CRS original: "
    f"{h3.crs}"
)


# ============================================================
# VALIDAR COLUMNA ID H3
# ============================================================

if "id_h3" not in h3.columns:
    raise ValueError(
        "La geometría H3 no contiene la columna id_h3."
    )


# ============================================================
# NORMALIZAR ID H3 GEOMETRÍA
# ============================================================

print("\nNormalizando identificadores H3 de geometría...")

h3["id_h3"] = (
    h3["id_h3"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# ============================================================
# VALIDAR CRS
# ============================================================

if h3.crs is None:

    print(
        "ADVERTENCIA: la geometría H3 no tiene CRS."
    )

    print(
        "Asignando EPSG:4326..."
    )

    h3 = h3.set_crs(
        "EPSG:4326"
    )

else:

    h3 = h3.to_crs(
        "EPSG:4326"
    )


print(
    f"CRS normalizado: "
    f"{h3.crs}"
)


# ============================================================
# VALIDAR GEOMETRÍAS
# ============================================================

print("\nValidando geometrías H3...")

geometrias_nulas = (
    h3.geometry.isna()
    .sum()
)

geometrias_invalidas = (
    (~h3.geometry.is_valid)
    & h3.geometry.notna()
).sum()

print(
    f"Geometrías nulas: "
    f"{geometrias_nulas:,}"
)

print(
    f"Geometrías inválidas: "
    f"{geometrias_invalidas:,}"
)

if geometrias_nulas > 0:
    raise ValueError(
        "Existen geometrías H3 nulas."
    )

if geometrias_invalidas > 0:

    print(
        "Intentando corregir geometrías..."
    )

    h3["geometry"] = (
        h3.geometry
        .make_valid()
    )

    invalidas_despues = (
        (~h3.geometry.is_valid)
        & h3.geometry.notna()
    ).sum()

    if invalidas_despues > 0:

        raise ValueError(
            "No fue posible corregir todas "
            "las geometrías H3."
        )

    print(
        "Geometrías H3 corregidas."
    )


# ============================================================
# VALIDAR IDS H3
# ============================================================

print(
    "\nValidando identificadores H3 "
    "de la geometría..."
)

h3_nulos = (
    h3["id_h3"]
    .isna()
    .sum()
)

h3_duplicados = (
    h3["id_h3"]
    .duplicated()
    .sum()
)

print(
    f"id_h3 nulos: "
    f"{h3_nulos:,}"
)

print(
    f"H3 duplicados en geometría: "
    f"{h3_duplicados:,}"
)

if h3_nulos > 0:
    raise ValueError(
        "La geometría contiene H3 sin identificador."
    )

if h3_duplicados > 0:
    raise ValueError(
        "La geometría H3 contiene IDs duplicados."
    )


# ============================================================
# COMPARAR H3
# ============================================================

print(
    "\nComparando H3 de SUBE "
    "contra geometría..."
)

h3_sube = set(
    sube["id_h3"]
    .dropna()
    .unique()
)

h3_geometria = set(
    h3["id_h3"]
    .dropna()
    .unique()
)

h3_sin_geometria = (
    h3_sube
    - h3_geometria
)

h3_sin_operaciones = (
    h3_geometria
    - h3_sube
)

print(
    f"H3 utilizados por SUBE: "
    f"{len(h3_sube):,}"
)

print(
    f"H3 disponibles en geometría: "
    f"{len(h3_geometria):,}"
)

print(
    f"H3 SUBE sin geometría: "
    f"{len(h3_sin_geometria):,}"
)

print(
    f"H3 geometría sin operaciones: "
    f"{len(h3_sin_operaciones):,}"
)


# ============================================================
# DIAGNÓSTICO H3 SIN GEOMETRÍA
# ============================================================

if h3_sin_geometria:

    print(
        "\n" + "=" * 70
    )

    print(
        "H3 SIN GEOMETRÍA"
    )

    print(
        "=" * 70
    )

    faltantes = (
        sube[
            sube["id_h3"]
            .isin(h3_sin_geometria)
        ]
        .groupby("id_h3", as_index=False)
        .agg(
            registros=(
                "id_h3",
                "size"
            ),
            operaciones=(
                "cantidad_trx",
                "sum"
            ),
        )
        .sort_values(
            "operaciones",
            ascending=False
        )
    )

    print(
        f"H3 únicos sin geometría: "
        f"{len(faltantes):,}"
    )

    print(
        "\nPrimeros 20:"
    )

    print(
        faltantes
        .head(20)
        .to_string(index=False)
    )

    faltantes.to_csv(
        OUTPUT_MISSING_H3,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nDetalle guardado en:"
    )

    print(
        OUTPUT_MISSING_H3
    )


# ============================================================
# PREPARAR GEOMETRÍA PARA JOIN
# ============================================================

print(
    "\nPreparando geometría H3..."
)

h3_join = h3[
    [
        "id_h3",
        "geometry",
    ]
].copy()


# ============================================================
# JOIN
# ============================================================

print(
    "\nRealizando JOIN por id_h3..."
)

resultado = sube.merge(
    h3_join,
    on="id_h3",
    how="left",
    validate="many_to_one",
)


# ============================================================
# INDICADOR DE GEOMETRÍA
# ============================================================

resultado["geometry_missing"] = (
    resultado["geometry"]
    .isna()
)


# ============================================================
# CONVERTIR A GEODATAFRAME
# ============================================================

resultado = gpd.GeoDataFrame(
    resultado,
    geometry="geometry",
    crs="EPSG:4326",
)


# ============================================================
# VALIDACIÓN DEL JOIN
# ============================================================

print()
separador()
print("VALIDACIÓN DEL JOIN")
separador()

registros_originales = len(sube)

registros_resultantes = len(resultado)

registros_con_geo = (
    ~resultado["geometry_missing"]
).sum()

registros_sin_geo = (
    resultado["geometry_missing"]
).sum()

operaciones_con_geo = (
    resultado.loc[
        ~resultado["geometry_missing"],
        "cantidad_trx"
    ].sum()
)

operaciones_sin_geo = (
    resultado.loc[
        resultado["geometry_missing"],
        "cantidad_trx"
    ].sum()
)

print(
    f"\nRegistros SUBE originales: "
    f"{registros_originales:,}"
)

print(
    f"Registros resultantes: "
    f"{registros_resultantes:,}"
)

print(
    f"Registros con geometría: "
    f"{registros_con_geo:,}"
)

print(
    f"Registros sin geometría: "
    f"{registros_sin_geo:,}"
)

print(
    f"\nOperaciones totales: "
    f"{operaciones_totales:,.0f}"
)

print(
    f"Operaciones con geometría: "
    f"{operaciones_con_geo:,.0f}"
)

print(
    f"Operaciones sin geometría: "
    f"{operaciones_sin_geo:,.0f}"
)

cobertura = porcentaje(
    operaciones_con_geo,
    operaciones_totales
)

sin_cobertura = porcentaje(
    operaciones_sin_geo,
    operaciones_totales
)

print(
    f"\nCobertura de operaciones "
    f"por geometría: {cobertura:.2f}%"
)

print(
    f"Operaciones sin cobertura "
    f"geográfica: {sin_cobertura:.2f}%"
)


# ============================================================
# VALIDAR QUE EL JOIN NO ALTERÓ REGISTROS
# ============================================================

if registros_originales != registros_resultantes:

    raise ValueError(
        "El JOIN alteró la cantidad de registros."
    )

print(
    "\nIntegridad del JOIN: OK."
)


# ============================================================
# RESUMEN POR H3 SIN GEOMETRÍA
# ============================================================

if registros_sin_geo > 0:

    print()
    separador()
    print("IMPACTO DE LOS H3 SIN GEOMETRÍA")
    separador()

    h3_faltantes_impacto = (
        resultado[
            resultado["geometry_missing"]
        ]
        .groupby(
            "id_h3",
            as_index=False
        )
        .agg(
            registros=(
                "id_h3",
                "size"
            ),
            operaciones=(
                "cantidad_trx",
                "sum"
            ),
        )
        .sort_values(
            "operaciones",
            ascending=False
        )
    )

    print(
        f"H3 afectados: "
        f"{len(h3_faltantes_impacto):,}"
    )

    print(
        f"Registros afectados: "
        f"{registros_sin_geo:,}"
    )

    print(
        f"Operaciones afectadas: "
        f"{operaciones_sin_geo:,.0f}"
    )


# ============================================================
# GUARDAR PARQUET
# ============================================================

print()
separador()
print("GUARDANDO RESULTADO")
separador()

print(
    f"\nArchivo:"
)

print(
    OUTPUT_PATH
)

resultado.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# VALIDAR ARCHIVO GENERADO
# ============================================================

print(
    "\nValidando archivo generado..."
)

validacion = gpd.read_parquet(
    OUTPUT_PATH
)

print(
    f"Registros guardados: "
    f"{len(validacion):,}"
)

print(
    f"CRS guardado: "
    f"{validacion.crs}"
)

geometrias_guardadas = (
    validacion.geometry.notna()
).sum()

print(
    f"Geometrías guardadas: "
    f"{geometrias_guardadas:,}"
)


# ============================================================
# RESUMEN FINAL CSV
# ============================================================

resumen = pd.DataFrame(
    [
        {
            "indicador": "h3_geometria",
            "valor": len(h3_geometria),
        },
        {
            "indicador": "h3_utilizados_sube",
            "valor": len(h3_sube),
        },
        {
            "indicador": "h3_sin_geometria",
            "valor": len(h3_sin_geometria),
        },
        {
            "indicador": "h3_geometria_sin_operaciones",
            "valor": len(h3_sin_operaciones),
        },
        {
            "indicador": "registros_sube",
            "valor": registros_originales,
        },
        {
            "indicador": "registros_con_geometria",
            "valor": registros_con_geo,
        },
        {
            "indicador": "registros_sin_geometria",
            "valor": registros_sin_geo,
        },
        {
            "indicador": "operaciones_totales",
            "valor": operaciones_totales,
        },
        {
            "indicador": "operaciones_con_geometria",
            "valor": operaciones_con_geo,
        },
        {
            "indicador": "operaciones_sin_geometria",
            "valor": operaciones_sin_geo,
        },
        {
            "indicador": "cobertura_geografica_pct",
            "valor": cobertura,
        },
        {
            "indicador": "sin_cobertura_geografica_pct",
            "valor": sin_cobertura,
        },
    ]
)

resumen.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
separador()
print("RESUMEN FINAL")
separador()

print(
    f"\nH3 en geometría: "
    f"{len(h3_geometria):,}"
)

print(
    f"H3 utilizados por SUBE: "
    f"{len(h3_sube):,}"
)

print(
    f"H3 SUBE sin geometría: "
    f"{len(h3_sin_geometria):,}"
)

print(
    f"\nRegistros SUBE: "
    f"{registros_originales:,}"
)

print(
    f"Registros con geometría: "
    f"{registros_con_geo:,}"
)

print(
    f"Registros sin geometría: "
    f"{registros_sin_geo:,}"
)

print(
    f"\nOperaciones totales: "
    f"{operaciones_totales:,.0f}"
)

print(
    f"Operaciones con geometría: "
    f"{operaciones_con_geo:,.0f}"
)

print(
    f"Operaciones sin geometría: "
    f"{operaciones_sin_geo:,.0f}"
)

print(
    f"\nCobertura geográfica: "
    f"{cobertura:.2f}%"
)

print(
    "\nArchivo generado:"
)

print(
    OUTPUT_PATH
)

print(
    "\nResumen:"
)

print(
    OUTPUT_SUMMARY
)

print()
separador()
print("PROCESO FINALIZADO CORRECTAMENTE")
separador()