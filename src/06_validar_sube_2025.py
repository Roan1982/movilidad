from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

INPUT_FILE = PROCESSED_DIR / "sube_2025_geo.parquet"

GOBIERNOS_FILE = (
    RAW_DIR
    / "georef"
    / "gobiernos-locales.geojson"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "sube_2025_jurisdicciones.parquet"
)

AUDITORIA_FILE = (
    PROCESSED_DIR
    / "sube_2025_jurisdicciones_auditoria.csv"
)

SIN_JURISDICCION_FILE = (
    PROCESSED_DIR
    / "sube_2025_h3_sin_jurisdiccion.csv"
)

RESUMEN_JURISDICCIONES_FILE = (
    PROCESSED_DIR
    / "sube_2025_jurisdicciones_resumen.csv"
)

RESUMEN_METODOS_FILE = (
    PROCESSED_DIR
    / "sube_2025_jurisdicciones_metodos.csv"
)

RESUMEN_GENERAL_FILE = (
    PROCESSED_DIR
    / "sube_2025_jurisdicciones_resumen_general.csv"
)

H3_SIN_GEOMETRIA_FILE = (
    PROCESSED_DIR
    / "sube_2025_h3_sin_geometria_detalle.csv"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURACIÓN DE ASIGNACIÓN
# ============================================================

DISTANCIA_MAXIMA_M = 1000


# ============================================================
# UTILIDADES
# ============================================================

def porcentaje(valor, total):
    if total == 0:
        return 0.0

    return valor / total * 100


def imprimir_separador(titulo):
    print()
    print("=" * 70)
    print(titulo)
    print("=" * 70)


def convertir_a_texto(valor):
    """
    Convierte valores complejos provenientes de GeoJSON
    a texto para evitar problemas con groupby.
    """

    if valor is None:
        return None

    if isinstance(valor, dict):
        return str(valor)

    if isinstance(valor, list):
        return str(valor)

    return valor


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ASIGNACIÓN TERRITORIAL SUBE 2025")
print("=" * 70)


# ============================================================
# 1. CARGAR SUBE
# ============================================================

imprimir_separador("1. CARGANDO SUBE 2025")

print(f"Archivo: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"No existe el archivo:\n{INPUT_FILE}"
    )

gdf = gpd.read_parquet(INPUT_FILE)

print(f"Registros: {len(gdf):,}")
print(f"Columnas: {len(gdf.columns)}")
print(f"CRS: {gdf.crs}")


# ============================================================
# 2. VALIDAR ESTRUCTURA
# ============================================================

imprimir_separador("2. VALIDANDO SUBE")

columnas_requeridas = {
    "id_h3",
    "cantidad_trx",
    "geometry",
}

faltantes = (
    columnas_requeridas
    - set(gdf.columns)
)

if faltantes:
    raise ValueError(
        "Faltan columnas requeridas en SUBE: "
        f"{sorted(faltantes)}"
    )

print("Columnas requeridas: OK")


# ============================================================
# 3. NORMALIZAR H3
# ============================================================

h3_nulos = gdf["id_h3"].isna().sum()

print(f"H3 nulos: {h3_nulos:,}")

if h3_nulos > 0:
    raise ValueError(
        "Existen registros SUBE con id_h3 nulo."
    )

gdf["id_h3"] = (
    gdf["id_h3"]
    .astype(str)
    .str.strip()
    .str.lower()
)

h3_unicos_sube = gdf["id_h3"].nunique()

print(
    f"H3 distintos utilizados por SUBE: "
    f"{h3_unicos_sube:,}"
)


# ============================================================
# 4. VALIDAR OPERACIONES
# ============================================================

gdf["cantidad_trx"] = pd.to_numeric(
    gdf["cantidad_trx"],
    errors="coerce"
)

if gdf["cantidad_trx"].isna().any():
    raise ValueError(
        "Existen valores inválidos en cantidad_trx."
    )

operaciones_totales = gdf["cantidad_trx"].sum()

print(
    f"Operaciones totales: "
    f"{operaciones_totales:,.0f}"
)


# ============================================================
# 5. COBERTURA GEOGRÁFICA
# ============================================================

imprimir_separador("3. COBERTURA GEOGRÁFICA")

tiene_geometria = gdf.geometry.notna()

registros_con_geometria = tiene_geometria.sum()
registros_sin_geometria = (~tiene_geometria).sum()

operaciones_con_geometria = (
    gdf.loc[
        tiene_geometria,
        "cantidad_trx"
    ].sum()
)

operaciones_sin_geometria = (
    gdf.loc[
        ~tiene_geometria,
        "cantidad_trx"
    ].sum()
)

h3_con_geometria = (
    gdf.loc[
        tiene_geometria,
        "id_h3"
    ]
    .nunique()
)

h3_sin_geometria = (
    gdf.loc[
        ~tiene_geometria,
        "id_h3"
    ]
    .nunique()
)

print(
    f"H3 con geometría:             "
    f"{h3_con_geometria:,}"
)

print(
    f"H3 sin geometría:             "
    f"{h3_sin_geometria:,}"
)

print(
    f"Registros con geometría:      "
    f"{registros_con_geometria:,}"
)

print(
    f"Registros sin geometría:      "
    f"{registros_sin_geometria:,}"
)

print(
    f"Operaciones con geometría:    "
    f"{operaciones_con_geometria:,.0f}"
)

print(
    f"Operaciones sin geometría:    "
    f"{operaciones_sin_geometria:,.0f}"
)

print(
    f"Cobertura de operaciones:     "
    f"{porcentaje(operaciones_con_geometria, operaciones_totales):.2f}%"
)

print(
    f"Operaciones sin cobertura:     "
    f"{porcentaje(operaciones_sin_geometria, operaciones_totales):.2f}%"
)


# ============================================================
# 6. H3 SIN GEOMETRÍA
# ============================================================

imprimir_separador("4. H3 SIN GEOMETRÍA")

sin_geometria = gdf.loc[
    ~tiene_geometria
].copy()

if len(sin_geometria) > 0:

    h3_sin_geometria_detalle = (
        sin_geometria
        .groupby("id_h3")
        .agg(
            registros=("id_h3", "size"),
            operaciones=("cantidad_trx", "sum")
        )
        .sort_values(
            "operaciones",
            ascending=False
        )
        .reset_index()
    )

    print(
        f"H3 únicos sin geometría: "
        f"{len(h3_sin_geometria_detalle):,}"
    )

    print("\nPrimeros 20:")

    print(
        h3_sin_geometria_detalle
        .head(20)
        .to_string(index=False)
    )

else:

    h3_sin_geometria_detalle = pd.DataFrame(
        columns=[
            "id_h3",
            "registros",
            "operaciones",
        ]
    )

    print(
        "No existen H3 sin geometría."
    )

h3_sin_geometria_detalle.to_csv(
    H3_SIN_GEOMETRIA_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. CARGAR GOBIERNOS LOCALES
# ============================================================

imprimir_separador("5. CARGANDO GOBIERNOS LOCALES")

print(
    f"Archivo: {GOBIERNOS_FILE}"
)

if not GOBIERNOS_FILE.exists():
    raise FileNotFoundError(
        f"No existe el archivo:\n{GOBIERNOS_FILE}"
    )

territorio = gpd.read_file(
    GOBIERNOS_FILE
)

print(
    f"Gobiernos locales: "
    f"{len(territorio):,}"
)

print(
    f"CRS original: "
    f"{territorio.crs}"
)


# ============================================================
# 8. NORMALIZAR TERRITORIO
# ============================================================

imprimir_separador("6. NORMALIZANDO TERRITORIO")

if territorio.crs is None:
    territorio = territorio.set_crs(
        "EPSG:4326"
    )

territorio = territorio.to_crs(
    "EPSG:4326"
)


# ------------------------------------------------------------
# CORREGIR GEOMETRÍAS
# ------------------------------------------------------------

invalidas = (
    ~territorio.geometry.is_valid
).sum()

print(
    f"Geometrías inválidas: "
    f"{invalidas:,}"
)

if invalidas > 0:

    print(
        "Intentando corregir geometrías..."
    )

    territorio["geometry"] = (
        territorio.geometry
        .buffer(0)
    )

    invalidas_despues = (
        ~territorio.geometry.is_valid
    ).sum()

    if invalidas_despues > 0:
        raise ValueError(
            "Persisten geometrías territoriales inválidas: "
            f"{invalidas_despues}"
        )

    print(
        "Todas las geometrías fueron corregidas."
    )


# ============================================================
# 9. VALIDAR IDS
# ============================================================

if "id" not in territorio.columns:
    raise ValueError(
        "El archivo territorial no contiene columna 'id'."
    )

territorio["id"] = (
    territorio["id"]
    .astype(str)
    .str.strip()
)

duplicados_territorio = (
    territorio["id"]
    .duplicated()
    .sum()
)

print(
    f"IDs territoriales duplicados: "
    f"{duplicados_territorio:,}"
)

if duplicados_territorio > 0:
    raise ValueError(
        "Existen IDs territoriales duplicados."
    )


# ============================================================
# 10. NORMALIZAR CAMPOS TERRITORIALES
# ============================================================

columnas_territorio = [
    "id",
    "nombre",
    "nombre_completo",
    "provincia",
    "categoria",
]

for columna in columnas_territorio:

    if columna not in territorio.columns:
        territorio[columna] = None


territorio = territorio[
    columnas_territorio
    + ["geometry"]
].copy()


territorio = territorio.rename(
    columns={
        "id": "id_jurisdiccion",
        "nombre": "jurisdiccion",
        "nombre_completo": "jurisdiccion_completa",
        "provincia": "provincia_jurisdiccion",
        "categoria": "categoria_jurisdiccion",
    }
)


print("\nNormalizando campos territoriales...")

for columna in [
    "id_jurisdiccion",
    "jurisdiccion",
    "jurisdiccion_completa",
    "provincia_jurisdiccion",
    "categoria_jurisdiccion",
]:

    territorio[columna] = (
        territorio[columna]
        .map(convertir_a_texto)
    )


# ============================================================
# 11. PREPARAR H3
# ============================================================

imprimir_separador(
    "7. PREPARANDO GEOMETRÍAS H3"
)

h3_gdf = (
    gdf.loc[
        gdf.geometry.notna()
    ]
    [
        ["id_h3", "geometry"]
    ]
    .drop_duplicates(
        subset=["id_h3"]
    )
    .copy()
)

h3_gdf = gpd.GeoDataFrame(
    h3_gdf,
    geometry="geometry",
    crs=gdf.crs
)

print(
    f"H3 con geometría para asignación: "
    f"{len(h3_gdf):,}"
)


# ============================================================
# 12. PROYECTAR H3 A CRS MÉTRICO
# ============================================================

h3_metric = h3_gdf.to_crs(
    "EPSG:3857"
).copy()


# ------------------------------------------------------------
# ÁREA H3
# ------------------------------------------------------------

h3_metric["area_h3_m2"] = (
    h3_metric.geometry.area
)


# ------------------------------------------------------------
# PUNTO REPRESENTATIVO
# ------------------------------------------------------------

h3_metric["punto"] = (
    h3_metric.geometry
    .representative_point()
)


# ============================================================
# 13. PREPARAR TERRITORIO MÉTRICO
# ============================================================

territorio_metric = (
    territorio
    .to_crs("EPSG:3857")
    .copy()
)


# ============================================================
# 14. ASIGNACIÓN POR INTERSECCIÓN
# ============================================================

imprimir_separador(
    "8. ASIGNACIÓN POR INTERSECCIÓN"
)

print(
    "Buscando jurisdicciones que intersecten cada H3..."
)

intersecciones = gpd.sjoin(
    h3_metric[
        [
            "id_h3",
            "geometry",
            "area_h3_m2",
        ]
    ],
    territorio_metric,
    how="left",
    predicate="intersects",
)

intersecciones = (
    intersecciones
    .loc[
        intersecciones[
            "id_jurisdiccion"
        ].notna()
    ]
    .copy()
)

print(
    f"Registros generados por intersección: "
    f"{len(intersecciones):,}"
)


# ============================================================
# 15. CALCULAR INTERSECCIÓN REAL
# ============================================================

if len(intersecciones) > 0:

    territorio_geometrias = (
        territorio_metric
        .set_index("id_jurisdiccion")
        .geometry
    )

    def calcular_interseccion(row):

        jurisdiccion_id = row[
            "id_jurisdiccion"
        ]

        try:

            geom_jurisdiccion = (
                territorio_geometrias
                .loc[jurisdiccion_id]
            )

            geom_interseccion = (
                row.geometry
                .intersection(
                    geom_jurisdiccion
                )
            )

            return geom_interseccion.area

        except Exception:

            return 0.0


    intersecciones[
        "interseccion_m2"
    ] = intersecciones.apply(
        calcular_interseccion,
        axis=1
    )

    intersecciones[
        "interseccion_pct"
    ] = (
        intersecciones[
            "interseccion_m2"
        ]
        / intersecciones[
            "area_h3_m2"
        ]
        * 100
    )

else:

    intersecciones[
        "interseccion_m2"
    ] = pd.Series(
        dtype=float
    )

    intersecciones[
        "interseccion_pct"
    ] = pd.Series(
        dtype=float
    )


# ============================================================
# 16. ELEGIR MEJOR JURISDICCIÓN
# ============================================================

if len(intersecciones) > 0:

    intersecciones = (
        intersecciones
        .sort_values(
            [
                "id_h3",
                "interseccion_pct",
            ],
            ascending=[
                True,
                False,
            ]
        )
    )

    asignaciones = (
        intersecciones
        .drop_duplicates(
            subset=["id_h3"],
            keep="first"
        )
        .copy()
    )

else:

    asignaciones = pd.DataFrame(
        columns=[
            "id_h3",
            "id_jurisdiccion",
            "jurisdiccion",
            "jurisdiccion_completa",
            "provincia_jurisdiccion",
            "categoria_jurisdiccion",
            "interseccion_m2",
            "interseccion_pct",
        ]
    )


columnas_asignacion_base = [
    "id_h3",
    "id_jurisdiccion",
    "jurisdiccion",
    "jurisdiccion_completa",
    "provincia_jurisdiccion",
    "categoria_jurisdiccion",
    "interseccion_m2",
    "interseccion_pct",
]

asignaciones = asignaciones[
    columnas_asignacion_base
].copy()

asignaciones[
    "metodo_asignacion"
] = "INTERSECCION"

asignaciones[
    "distancia_asignacion_m"
] = 0.0


print(
    f"H3 asignados por intersección: "
    f"{len(asignaciones):,}"
)


# ============================================================
# 17. H3 PENDIENTES
# ============================================================

h3_asignados = set(
    asignaciones["id_h3"]
)

h3_pendientes = h3_gdf.loc[
    ~h3_gdf["id_h3"].isin(
        h3_asignados
    )
].copy()

print(
    f"H3 pendientes después de intersección: "
    f"{len(h3_pendientes):,}"
)


# ============================================================
# 18. FALLBACK: PUNTO REPRESENTATIVO
# ============================================================

imprimir_separador(
    "9. FALLBACK: PUNTO REPRESENTATIVO"
)

asignaciones_punto = []

if len(h3_pendientes) > 0:

    puntos_pendientes = (
        h3_metric[
            h3_metric["id_h3"].isin(
                h3_pendientes["id_h3"]
            )
        ]
        [
            ["id_h3", "punto"]
        ]
        .copy()
    )

    puntos_pendientes = gpd.GeoDataFrame(
        puntos_pendientes,
        geometry="punto",
        crs=h3_metric.crs
    )

    puntos_pendientes = (
        puntos_pendientes
        .rename(
            columns={
                "punto": "geometry"
            }
        )
    )

    puntos_pendientes = gpd.GeoDataFrame(
        puntos_pendientes,
        geometry="geometry",
        crs=h3_metric.crs
    )

    punto_intersecciones = gpd.sjoin(
        puntos_pendientes,
        territorio_metric,
        how="left",
        predicate="within"
    )

    punto_intersecciones = (
        punto_intersecciones
        .loc[
            punto_intersecciones[
                "id_jurisdiccion"
            ].notna()
        ]
        .copy()
    )

    punto_intersecciones = (
        punto_intersecciones
        .drop_duplicates(
            subset=["id_h3"],
            keep="first"
        )
    )

    for _, row in punto_intersecciones.iterrows():

        asignaciones_punto.append(
            {
                "id_h3": row["id_h3"],
                "id_jurisdiccion": row[
                    "id_jurisdiccion"
                ],
                "jurisdiccion": row[
                    "jurisdiccion"
                ],
                "jurisdiccion_completa": row[
                    "jurisdiccion_completa"
                ],
                "provincia_jurisdiccion": row[
                    "provincia_jurisdiccion"
                ],
                "categoria_jurisdiccion": row[
                    "categoria_jurisdiccion"
                ],
                "interseccion_m2": 0.0,
                "interseccion_pct": 0.0,
                "metodo_asignacion": (
                    "PUNTO_REPRESENTATIVO"
                ),
                "distancia_asignacion_m": 0.0,
            }
        )

if asignaciones_punto:

    asignaciones_punto_df = pd.DataFrame(
        asignaciones_punto
    )

    asignaciones = pd.concat(
        [
            asignaciones,
            asignaciones_punto_df,
        ],
        ignore_index=True
    )

else:

    asignaciones_punto_df = pd.DataFrame()

    print(
        "No se asignaron H3 mediante punto representativo."
    )


print(
    f"H3 asignados por punto representativo: "
    f"{len(asignaciones_punto_df):,}"
)


# ============================================================
# 19. ACTUALIZAR H3 PENDIENTES
# ============================================================

h3_asignados = set(
    asignaciones["id_h3"]
)

h3_pendientes = h3_gdf.loc[
    ~h3_gdf["id_h3"].isin(
        h3_asignados
    )
].copy()


# ============================================================
# 20. FALLBACK: JURISDICCIÓN MÁS CERCANA
# ============================================================

imprimir_separador(
    "10. FALLBACK: JURISDICCIÓN MÁS CERCANA"
)

print(
    f"H3 pendientes: "
    f"{len(h3_pendientes):,}"
)

asignaciones_cercania = []

if len(h3_pendientes) > 0:

    puntos_pendientes = (
        h3_metric[
            h3_metric["id_h3"].isin(
                h3_pendientes["id_h3"]
            )
        ]
        [
            ["id_h3", "punto"]
        ]
        .copy()
    )

    puntos_pendientes = gpd.GeoDataFrame(
        puntos_pendientes,
        geometry="punto",
        crs=h3_metric.crs
    )

    puntos_pendientes = (
        puntos_pendientes
        .rename(
            columns={
                "punto": "geometry"
            }
        )
    )

    puntos_pendientes = gpd.GeoDataFrame(
        puntos_pendientes,
        geometry="geometry",
        crs=h3_metric.crs
    )

    territorio_cercania = territorio_metric[
        [
            "id_jurisdiccion",
            "jurisdiccion",
            "jurisdiccion_completa",
            "provincia_jurisdiccion",
            "categoria_jurisdiccion",
            "geometry",
        ]
    ].copy()

    nearest = gpd.sjoin_nearest(
        puntos_pendientes,
        territorio_cercania,
        how="left",
        distance_col="distancia_m"
    )

    nearest = (
        nearest.loc[
            nearest["distancia_m"]
            <= DISTANCIA_MAXIMA_M
        ]
        .copy()
    )

    print(
        f"H3 encontrados dentro de "
        f"{DISTANCIA_MAXIMA_M:,} m: "
        f"{nearest['id_h3'].nunique():,}"
    )

    nearest = (
        nearest
        .sort_values(
            "distancia_m"
        )
        .drop_duplicates(
            subset=["id_h3"]
        )
    )

    for _, row in nearest.iterrows():

        asignaciones_cercania.append(
            {
                "id_h3": row["id_h3"],
                "id_jurisdiccion": row[
                    "id_jurisdiccion"
                ],
                "jurisdiccion": row[
                    "jurisdiccion"
                ],
                "jurisdiccion_completa": row[
                    "jurisdiccion_completa"
                ],
                "provincia_jurisdiccion": row[
                    "provincia_jurisdiccion"
                ],
                "categoria_jurisdiccion": row[
                    "categoria_jurisdiccion"
                ],
                "interseccion_m2": 0.0,
                "interseccion_pct": 0.0,
                "metodo_asignacion": (
                    "JURISDICCION_MAS_CERCANA"
                ),
                "distancia_asignacion_m": row[
                    "distancia_m"
                ],
            }
        )

if asignaciones_cercania:

    asignaciones_cercania_df = pd.DataFrame(
        asignaciones_cercania
    )

    asignaciones = pd.concat(
        [
            asignaciones,
            asignaciones_cercania_df,
        ],
        ignore_index=True
    )

else:

    asignaciones_cercania_df = pd.DataFrame()

    print(
        "No se asignaron H3 mediante cercanía."
    )


print(
    f"H3 asignados por cercanía: "
    f"{len(asignaciones_cercania_df):,}"
)


# ============================================================
# 21. VALIDACIÓN DE ASIGNACIONES
# ============================================================

imprimir_separador(
    "11. VALIDACIÓN DE ASIGNACIONES"
)

asignaciones = (
    asignaciones
    .drop_duplicates(
        subset=["id_h3"],
        keep="first"
    )
    .copy()
)

h3_total = len(h3_gdf)

h3_asignados_total = (
    asignaciones["id_h3"]
    .nunique()
)

h3_sin_asignar = (
    h3_total
    - h3_asignados_total
)

print(
    f"H3 con geometría: "
    f"{h3_total:,}"
)

print(
    f"H3 asignados:     "
    f"{h3_asignados_total:,}"
)

print(
    f"H3 sin asignar:   "
    f"{h3_sin_asignar:,}"
)

print(
    f"Cobertura territorial: "
    f"{porcentaje(h3_asignados_total, h3_total):.2f}%"
)

print("\nMétodos de asignación:")

print(
    asignaciones[
        "metodo_asignacion"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# VALIDAR DUPLICADOS
# ------------------------------------------------------------

duplicados_asignacion = (
    asignaciones["id_h3"]
    .duplicated()
    .sum()
)

if duplicados_asignacion > 0:

    raise ValueError(
        "Existen H3 duplicados en las asignaciones."
    )

print(
    "\nIntegridad de asignaciones: OK."
)


# ============================================================
# 22. CALCULAR CONFIANZA
# ============================================================

def calcular_confianza(row):

    metodo = row[
        "metodo_asignacion"
    ]

    if metodo == "INTERSECCION":

        pct = row[
            "interseccion_pct"
        ]

        if pct >= 50:
            return "ALTA"

        if pct >= 10:
            return "MEDIA"

        return "BAJA"

    if metodo == "PUNTO_REPRESENTATIVO":

        return "MEDIA"

    if metodo == "JURISDICCION_MAS_CERCANA":

        distancia = row[
            "distancia_asignacion_m"
        ]

        if distancia <= 100:
            return "MEDIA"

        if distancia <= 500:
            return "BAJA"

        return "MUY_BAJA"

    return "NO_DETERMINADA"


asignaciones[
    "confianza_asignacion"
] = asignaciones.apply(
    calcular_confianza,
    axis=1
)


# ============================================================
# 23. PREPARAR DATASET TERRITORIAL H3
# ============================================================

imprimir_separador(
    "12. PREPARANDO DATASET TERRITORIAL"
)


# ------------------------------------------------------------
# IMPORTANTE:
# NO HACEMOS MERGE CON area_h3_m2 DESDE ASIGNACIONES.
#
# El área pertenece a h3_metric/h3_gdf.
# Esto evita el problema area_h3_m2_x / area_h3_m2_y.
# ------------------------------------------------------------

columnas_asignacion_h3 = [
    "id_h3",
    "id_jurisdiccion",
    "jurisdiccion",
    "jurisdiccion_completa",
    "provincia_jurisdiccion",
    "categoria_jurisdiccion",
    "interseccion_m2",
    "interseccion_pct",
    "metodo_asignacion",
    "distancia_asignacion_m",
    "confianza_asignacion",
]

h3_base = h3_metric[
    [
        "id_h3",
        "geometry",
        "area_h3_m2",
    ]
].copy()


h3_final = h3_base.merge(
    asignaciones[
        columnas_asignacion_h3
    ],
    on="id_h3",
    how="left",
    validate="one_to_one"
)


h3_final = gpd.GeoDataFrame(
    h3_final,
    geometry="geometry",
    crs=h3_metric.crs
)

h3_final["tiene_geometria"] = True

h3_final["tiene_jurisdiccion"] = (
    h3_final[
        "id_jurisdiccion"
    ].notna()
)


# ============================================================
# 24. VALIDAR DATASET H3
# ============================================================

if len(h3_final) != len(h3_gdf):

    raise ValueError(
        "El JOIN territorial alteró la cantidad de H3."
    )

if h3_final["id_h3"].duplicated().any():

    raise ValueError(
        "Existen H3 duplicados en h3_final."
    )

if "area_h3_m2" not in h3_final.columns:

    raise ValueError(
        "No existe area_h3_m2 en h3_final."
    )


# ============================================================
# 25. GENERAR DATASET FINAL POR REGISTRO SUBE
# ============================================================

imprimir_separador(
    "13. GENERANDO DATASET FINAL"
)


gdf_final = gdf.merge(
    h3_final[
        columnas_asignacion_h3
    ],
    on="id_h3",
    how="left",
    validate="many_to_one"
)


# ------------------------------------------------------------
# INDICADORES
# ------------------------------------------------------------

gdf_final["tiene_geometria"] = (
    gdf_final.geometry.notna()
)

gdf_final["tiene_jurisdiccion"] = (
    gdf_final[
        "id_jurisdiccion"
    ].notna()
)

gdf_final["operacion_geografica"] = (
    gdf_final["tiene_geometria"]
)

gdf_final["operacion_territorial"] = (
    gdf_final["tiene_jurisdiccion"]
)


# ============================================================
# 26. VALIDACIÓN DATASET FINAL
# ============================================================

registros_finales = len(gdf_final)

if registros_finales != len(gdf):

    raise ValueError(
        "El JOIN final alteró la cantidad de registros."
    )

if gdf_final["id_h3"].isna().any():

    raise ValueError(
        "Existen registros sin id_h3 después del JOIN."
    )

print(
    "Integridad del dataset final: OK."
)

print(
    f"Registros finales: "
    f"{registros_finales:,}"
)


# ============================================================
# 27. MÉTRICAS FINALES
# ============================================================

operaciones_con_jurisdiccion = (
    gdf_final.loc[
        gdf_final[
            "tiene_jurisdiccion"
        ],
        "cantidad_trx"
    ].sum()
)

operaciones_sin_jurisdiccion = (
    gdf_final.loc[
        ~gdf_final[
            "tiene_jurisdiccion"
        ],
        "cantidad_trx"
    ].sum()
)

h3_con_jurisdiccion = (
    gdf_final.loc[
        gdf_final[
            "tiene_jurisdiccion"
        ],
        "id_h3"
    ]
    .nunique()
)


# ============================================================
# 28. GUARDAR DATASET
# ============================================================

imprimir_separador(
    "14. GUARDANDO DATASET"
)

print(
    f"Archivo: {OUTPUT_FILE}"
)

gdf_final.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 29. AUDITORÍA
# ============================================================

imprimir_separador(
    "15. GENERANDO AUDITORÍA"
)


# ------------------------------------------------------------
# IMPORTANTE:
# area_h3_m2 sale directamente de h3_final.
# Ya no depende de asignaciones.
# ------------------------------------------------------------

columnas_auditoria = [
    "id_h3",
    "area_h3_m2",
    "id_jurisdiccion",
    "jurisdiccion",
    "jurisdiccion_completa",
    "provincia_jurisdiccion",
    "categoria_jurisdiccion",
    "interseccion_m2",
    "interseccion_pct",
    "metodo_asignacion",
    "distancia_asignacion_m",
    "confianza_asignacion",
]


auditoria = (
    h3_final[
        columnas_auditoria
    ]
    .copy()
)


# ------------------------------------------------------------
# OPERACIONES POR H3
# ------------------------------------------------------------

operaciones_h3 = (
    gdf_final
    .groupby("id_h3")
    .agg(
        registros=("id_h3", "size"),
        operaciones=("cantidad_trx", "sum")
    )
    .reset_index()
)


auditoria = auditoria.merge(
    operaciones_h3,
    on="id_h3",
    how="left",
    validate="one_to_one"
)


# ------------------------------------------------------------
# GUARDAR AUDITORÍA
# ------------------------------------------------------------

auditoria.to_csv(
    AUDITORIA_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Auditoría: {AUDITORIA_FILE}"
)


# ============================================================
# 30. H3 SIN JURISDICCIÓN
# ============================================================

imprimir_separador(
    "16. H3 SIN JURISDICCIÓN"
)

h3_sin_jurisdiccion = (
    auditoria.loc[
        auditoria[
            "id_jurisdiccion"
        ].isna()
    ]
    .copy()
)

print(
    f"H3 sin jurisdicción: "
    f"{len(h3_sin_jurisdiccion):,}"
)

h3_sin_jurisdiccion.to_csv(
    SIN_JURISDICCION_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Archivo: {SIN_JURISDICCION_FILE}"
)


# ============================================================
# 31. RESUMEN POR JURISDICCIÓN
# ============================================================

imprimir_separador(
    "17. RESUMEN POR JURISDICCIÓN"
)

con_jurisdiccion = gdf_final.loc[
    gdf_final[
        "tiene_jurisdiccion"
    ]
].copy()


# ------------------------------------------------------------
# NORMALIZAR CAMPOS ANTES DEL GROUPBY
# ------------------------------------------------------------

for columna in [
    "id_jurisdiccion",
    "jurisdiccion",
    "jurisdiccion_completa",
    "provincia_jurisdiccion",
    "categoria_jurisdiccion",
]:

    con_jurisdiccion[columna] = (
        con_jurisdiccion[columna]
        .map(convertir_a_texto)
    )


if len(con_jurisdiccion) > 0:

    resumen_jurisdicciones = (
        con_jurisdiccion
        .groupby(
            [
                "id_jurisdiccion",
                "jurisdiccion",
                "jurisdiccion_completa",
                "provincia_jurisdiccion",
                "categoria_jurisdiccion",
            ],
            dropna=False,
            sort=False,
        )
        .agg(
            registros=(
                "id_h3",
                "size"
            ),
            h3_distintos=(
                "id_h3",
                "nunique"
            ),
            operaciones=(
                "cantidad_trx",
                "sum"
            )
        )
        .reset_index()
        .sort_values(
            "operaciones",
            ascending=False
        )
        .reset_index(drop=True)
    )


    total_operaciones_resumen = (
        resumen_jurisdicciones[
            "operaciones"
        ].sum()
    )

    resumen_jurisdicciones[
        "pct_operaciones"
    ] = (
        resumen_jurisdicciones[
            "operaciones"
        ]
        / total_operaciones_resumen
        * 100
    )

    resumen_jurisdicciones[
        "ranking_operaciones"
    ] = (
        resumen_jurisdicciones.index
        + 1
    )


else:

    resumen_jurisdicciones = pd.DataFrame()


print(
    f"Jurisdicciones con operaciones: "
    f"{len(resumen_jurisdicciones):,}"
)


if len(resumen_jurisdicciones) > 0:

    print(
        "\nTop 20 jurisdicciones por operaciones:"
    )

    print(
        resumen_jurisdicciones[
            [
                "ranking_operaciones",
                "id_jurisdiccion",
                "jurisdiccion",
                "provincia_jurisdiccion",
                "h3_distintos",
                "operaciones",
                "pct_operaciones",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            formatters={
                "pct_operaciones":
                "{:.2f}%".format
            }
        )
    )


resumen_jurisdicciones.to_csv(
    RESUMEN_JURISDICCIONES_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nResumen guardado:"
    f"\n{RESUMEN_JURISDICCIONES_FILE}"
)


# ============================================================
# 32. RESUMEN POR MÉTODO
# ============================================================

imprimir_separador(
    "18. RESUMEN POR MÉTODO DE ASIGNACIÓN"
)

resumen_metodos = (
    auditoria
    .groupby(
        "metodo_asignacion",
        dropna=False
    )
    .agg(
        h3=(
            "id_h3",
            "nunique"
        ),
        registros=(
            "registros",
            "sum"
        ),
        operaciones=(
            "operaciones",
            "sum"
        ),
        distancia_min_m=(
            "distancia_asignacion_m",
            "min"
        ),
        distancia_promedio_m=(
            "distancia_asignacion_m",
            "mean"
        ),
        distancia_max_m=(
            "distancia_asignacion_m",
            "max"
        )
    )
    .reset_index()
)


print(
    resumen_metodos.to_string(
        index=False
    )
)


resumen_metodos.to_csv(
    RESUMEN_METODOS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 33. RESUMEN GENERAL
# ============================================================

imprimir_separador(
    "19. RESUMEN GENERAL"
)

h3_total_sube = h3_unicos_sube

h3_con_geometria_pct = porcentaje(
    h3_con_geometria,
    h3_total_sube
)

h3_sin_geometria_pct = porcentaje(
    h3_sin_geometria,
    h3_total_sube
)

h3_asignados_pct = porcentaje(
    h3_con_jurisdiccion,
    h3_con_geometria
)

operaciones_geograficas_pct = porcentaje(
    operaciones_con_geometria,
    operaciones_totales
)

operaciones_territoriales_pct = porcentaje(
    operaciones_con_jurisdiccion,
    operaciones_totales
)


resumen_general = pd.DataFrame(
    [
        {
            "h3_total_sube":
                h3_total_sube,

            "h3_con_geometria":
                h3_con_geometria,

            "h3_sin_geometria":
                h3_sin_geometria,

            "h3_con_jurisdiccion":
                h3_con_jurisdiccion,

            "h3_sin_jurisdiccion":
                (
                    h3_con_geometria
                    - h3_con_jurisdiccion
                ),

            "h3_cobertura_geometrica_pct":
                h3_con_geometria_pct,

            "h3_sin_geometria_pct":
                h3_sin_geometria_pct,

            "h3_cobertura_territorial_pct":
                h3_asignados_pct,

            "registros_totales":
                len(gdf_final),

            "registros_con_geometria":
                registros_con_geometria,

            "registros_sin_geometria":
                registros_sin_geometria,

            "operaciones_totales":
                operaciones_totales,

            "operaciones_con_geometria":
                operaciones_con_geometria,

            "operaciones_sin_geometria":
                operaciones_sin_geometria,

            "operaciones_con_jurisdiccion":
                operaciones_con_jurisdiccion,

            "operaciones_sin_jurisdiccion":
                operaciones_sin_jurisdiccion,

            "cobertura_geografica_operaciones_pct":
                operaciones_geograficas_pct,

            "cobertura_territorial_operaciones_pct":
                operaciones_territoriales_pct,
        }
    ]
)


resumen_general.to_csv(
    RESUMEN_GENERAL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 34. VALIDACIÓN FINAL DE INTEGRIDAD
# ============================================================

imprimir_separador(
    "20. VALIDACIÓN FINAL"
)


# ------------------------------------------------------------
# REGISTROS
# ------------------------------------------------------------

if len(gdf_final) != len(gdf):

    raise ValueError(
        "ERROR FINAL: cantidad de registros alterada."
    )


# ------------------------------------------------------------
# H3
# ------------------------------------------------------------

if (
    gdf_final["id_h3"]
    .nunique()
    != h3_unicos_sube
):

    raise ValueError(
        "ERROR FINAL: cantidad de H3 alterada."
    )


# ------------------------------------------------------------
# OPERACIONES
# ------------------------------------------------------------

operaciones_finales = (
    gdf_final["cantidad_trx"].sum()
)

if operaciones_finales != operaciones_totales:

    raise ValueError(
        "ERROR FINAL: cantidad de operaciones alterada."
    )


# ------------------------------------------------------------
# ASIGNACIONES
# ------------------------------------------------------------

if (
    gdf_final["id_h3"]
    .isna()
    .any()
):

    raise ValueError(
        "ERROR FINAL: existen registros sin H3."
    )


print(
    "Integridad final: OK."
)


# ============================================================
# 35. RESUMEN FINAL
# ============================================================

imprimir_separador(
    "21. RESUMEN FINAL"
)

print(
    f"H3 utilizados por SUBE:             "
    f"{h3_total_sube:,}"
)

print(
    f"H3 con geometría:                    "
    f"{h3_con_geometria:,}"
)

print(
    f"H3 sin geometría:                    "
    f"{h3_sin_geometria:,}"
)

print(
    f"H3 con jurisdicción:                 "
    f"{h3_con_jurisdiccion:,}"
)

print(
    f"H3 sin jurisdicción:                 "
    f"{h3_con_geometria - h3_con_jurisdiccion:,}"
)

print(
    f"Cobertura geométrica H3:             "
    f"{h3_con_geometria_pct:.2f}%"
)

print(
    f"Cobertura territorial H3:            "
    f"{h3_asignados_pct:.2f}%"
)

print()

print(
    f"Registros SUBE:                       "
    f"{len(gdf_final):,}"
)

print(
    f"Registros con geometría:              "
    f"{registros_con_geometria:,}"
)

print(
    f"Registros sin geometría:              "
    f"{registros_sin_geometria:,}"
)

print()

print(
    f"Operaciones totales:                  "
    f"{operaciones_totales:,.0f}"
)

print(
    f"Operaciones con geometría:            "
    f"{operaciones_con_geometria:,.0f}"
)

print(
    f"Operaciones sin geometría:            "
    f"{operaciones_sin_geometria:,.0f}"
)

print(
    f"Operaciones con jurisdicción:         "
    f"{operaciones_con_jurisdiccion:,.0f}"
)

print(
    f"Operaciones sin jurisdicción:         "
    f"{operaciones_sin_jurisdiccion:,.0f}"
)

print(
    f"Cobertura geográfica operaciones:     "
    f"{operaciones_geograficas_pct:.2f}%"
)

print(
    f"Cobertura territorial operaciones:    "
    f"{operaciones_territoriales_pct:.2f}%"
)

print()

print(
    "Métodos de asignación:"
)

print(
    auditoria[
        "metodo_asignacion"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()

print(
    "Confianza de asignación:"
)

print(
    auditoria[
        "confianza_asignacion"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# 36. ARCHIVOS GENERADOS
# ============================================================

print()

print(
    "Archivos generados:"
)

print(
    f"\nDataset:"
    f"\n{OUTPUT_FILE}"
)

print(
    f"\nAuditoría:"
    f"\n{AUDITORIA_FILE}"
)

print(
    f"\nH3 sin jurisdicción:"
    f"\n{SIN_JURISDICCION_FILE}"
)

print(
    f"\nH3 sin geometría:"
    f"\n{H3_SIN_GEOMETRIA_FILE}"
)

print(
    f"\nResumen por jurisdicción:"
    f"\n{RESUMEN_JURISDICCIONES_FILE}"
)

print(
    f"\nResumen por método:"
    f"\n{RESUMEN_METODOS_FILE}"
)

print(
    f"\nResumen general:"
    f"\n{RESUMEN_GENERAL_FILE}"
)

print()
print("=" * 70)
print("ASIGNACIÓN TERRITORIAL FINALIZADA")
print("=" * 70)