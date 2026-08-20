from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_H3 = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_perfil_h3.parquet"
)

INPUT_TERRITORIO = (
    BASE_DIR
    / "data"
    / "raw"
    / "georef"
    / "gobiernos-locales.geojson"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_territorio.parquet"
)

# CRS métrico para áreas y distancias.
CRS_METRICO = "EPSG:3857"

# Distancia máxima para el método NEAREST.
MAX_DISTANCIA_M = 1000


# ============================================================
# CONSTANTES
# ============================================================

COLUMNAS_TERRITORIO = [
    "id",
    "nombre",
    "nombre_completo",
    "provincia_nombre",
    "categoria",
    "fuente",
    "geometry",
]

COLUMNAS_ASIGNACION = [
    "id_jurisdiccion",
    "jurisdiccion",
    "jurisdiccion_completa",
    "provincia",
    "categoria_jurisdiccion",
    "fuente_jurisdiccion",
    "metodo_asignacion",
    "distancia_asignacion_m",
]


# ============================================================
# FUNCIONES
# ============================================================

def imprimir_titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def validar_columnas(df, columnas, nombre):
    faltantes = [
        columna
        for columna in columnas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"{nombre} no contiene las columnas requeridas: "
            f"{faltantes}"
        )


def preparar_asignaciones(
    df,
    metodo,
    distancia_col=None,
):
    """
    Convierte el resultado de un spatial join en una tabla
    única por id_h3, lista para hacer merge con el resultado.

    IMPORTANTE:
    No utiliza índices de pandas para identificar H3.
    Todo se relaciona exclusivamente mediante id_h3.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "id_h3",
                *COLUMNAS_ASIGNACION,
            ]
        )

    df = df.copy()

    # --------------------------------------------------------
    # Eliminar registros sin jurisdicción
    # --------------------------------------------------------

    if "id" in df.columns:
        df = df[df["id"].notna()].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "id_h3",
                *COLUMNAS_ASIGNACION,
            ]
        )

    # --------------------------------------------------------
    # Distancia
    # --------------------------------------------------------

    if distancia_col is not None:
        if distancia_col in df.columns:
            df["distancia_asignacion_m"] = pd.to_numeric(
                df[distancia_col],
                errors="coerce",
            )
        else:
            df["distancia_asignacion_m"] = pd.NA
    else:
        df["distancia_asignacion_m"] = pd.NA

    # --------------------------------------------------------
    # Orden determinístico
    # --------------------------------------------------------

    columnas_orden = ["id_h3"]

    if "area_interseccion_m2" in df.columns:
        columnas_orden.extend(
            [
                "area_interseccion_m2",
            ]
        )

    if "distancia_asignacion_m" in df.columns:
        columnas_orden.append(
            "distancia_asignacion_m"
        )

    if "id" in df.columns:
        columnas_orden.append("id")

    ascending = []

    for columna in columnas_orden:

        if columna == "area_interseccion_m2":
            ascending.append(False)

        elif columna == "distancia_asignacion_m":
            ascending.append(True)

        else:
            ascending.append(True)

    df = df.sort_values(
        columnas_orden,
        ascending=ascending,
        na_position="last",
    )

    # --------------------------------------------------------
    # Una sola jurisdicción por H3
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset="id_h3",
        keep="first",
    )

    # --------------------------------------------------------
    # Renombrar
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "id": "id_jurisdiccion",
            "nombre": "jurisdiccion",
            "nombre_completo": "jurisdiccion_completa",
            "provincia_nombre": "provincia",
            "categoria": "categoria_jurisdiccion",
            "fuente": "fuente_jurisdiccion",
        }
    )

    # --------------------------------------------------------
    # Método
    # --------------------------------------------------------

    df["metodo_asignacion"] = metodo

    # --------------------------------------------------------
    # Seleccionar columnas
    # --------------------------------------------------------

    columnas_finales = [
        "id_h3",
        "id_jurisdiccion",
        "jurisdiccion",
        "jurisdiccion_completa",
        "provincia",
        "categoria_jurisdiccion",
        "fuente_jurisdiccion",
        "metodo_asignacion",
        "distancia_asignacion_m",
    ]

    for columna in columnas_finales:

        if columna not in df.columns:

            if columna == "id_h3":
                raise ValueError(
                    "Las asignaciones no contienen id_h3."
                )

            df[columna] = pd.NA

    return df[columnas_finales].copy()


def obtener_pendientes(resultado):
    return resultado[
        resultado["jurisdiccion"].isna()
    ][
        [
            "id_h3",
            "geometry",
        ]
    ].copy()


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ASIGNACIÓN TERRITORIAL SUBE 2025")
print("=" * 70)


# ============================================================
# CARGAR H3
# ============================================================

print("\nCargando H3...")

h3 = gpd.read_parquet(INPUT_H3)

print(
    f"H3 cargados: {len(h3):,}"
)

print(
    f"CRS H3: {h3.crs}"
)


# ============================================================
# VALIDAR H3
# ============================================================

print("\nValidando H3...")

validar_columnas(
    h3,
    [
        "id_h3",
        "operaciones_totales",
        "geometry",
    ],
    "Archivo H3",
)

if h3.crs is None:
    raise ValueError(
        "Los H3 no tienen CRS definido."
    )

h3["id_h3"] = (
    h3["id_h3"]
    .astype(str)
    .str.strip()
)

duplicados_h3 = (
    h3["id_h3"]
    .duplicated()
    .sum()
)

print(
    f"H3 duplicados: {duplicados_h3:,}"
)

if duplicados_h3 > 0:
    raise ValueError(
        "El archivo H3 contiene identificadores duplicados."
    )


# ============================================================
# OPERACIONES ORIGINALES
# ============================================================

operaciones_originales = (
    pd.to_numeric(
        h3["operaciones_totales"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

print(
    f"Operaciones totales: "
    f"{operaciones_originales:,.0f}"
)


# ============================================================
# CARGAR TERRITORIO
# ============================================================

print("\nCargando gobiernos locales...")

territorio = gpd.read_file(
    INPUT_TERRITORIO
)

print(
    f"Gobiernos locales: "
    f"{len(territorio):,}"
)

print(
    f"CRS territorio original: "
    f"{territorio.crs}"
)


# ============================================================
# VALIDAR TERRITORIO
# ============================================================

validar_columnas(
    territorio,
    [
        "id",
        "nombre",
        "nombre_completo",
        "provincia",
        "categoria",
        "fuente",
        "geometry",
    ],
    "Archivo de gobiernos locales",
)

if territorio.crs is None:
    raise ValueError(
        "El territorio no tiene CRS definido."
    )


# ============================================================
# NORMALIZAR CRS
# ============================================================

if territorio.crs != h3.crs:

    print(
        "\nAjustando CRS del territorio..."
    )

    territorio = territorio.to_crs(
        h3.crs
    )

print(
    f"CRS territorio normalizado: "
    f"{territorio.crs}"
)


# ============================================================
# PREPARAR TERRITORIO
# ============================================================

territorio = territorio[
    [
        "id",
        "nombre",
        "nombre_completo",
        "provincia",
        "categoria",
        "fuente",
        "geometry",
    ]
].copy()


# ============================================================
# CORREGIR GEOMETRÍAS
# ============================================================

geometrias_invalidas = (
    ~territorio.geometry.is_valid
).sum()

geometrias_vacias = (
    territorio.geometry.is_empty
).sum()

print(
    f"\nGeometrías territoriales inválidas: "
    f"{geometrias_invalidas:,}"
)

print(
    f"Geometrías territoriales vacías: "
    f"{geometrias_vacias:,}"
)


if geometrias_invalidas > 0:

    print(
        "Intentando corregir geometrías..."
    )

    territorio["geometry"] = (
        territorio.geometry.make_valid()
    )


# ------------------------------------------------------------
# Eliminar geometrías que siguieran siendo inválidas/vacías
# ------------------------------------------------------------

territorio = territorio[
    territorio.geometry.notna()
    & ~territorio.geometry.is_empty
    & territorio.geometry.is_valid
].copy()


print(
    f"Geometrías territoriales preparadas: "
    f"{len(territorio):,}"
)


# ============================================================
# VALIDAR IDs TERRITORIALES
# ============================================================

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
# PROVINCIA
# ============================================================

territorio["provincia_nombre"] = (
    territorio["provincia"]
    .apply(
        lambda x:
            x.get("nombre")
            if isinstance(x, dict)
            else None
    )
)


# ============================================================
# RESULTADO BASE
# ============================================================

resultado = h3.copy()

resultado[
    "id_jurisdiccion"
] = pd.NA

resultado[
    "jurisdiccion"
] = pd.NA

resultado[
    "jurisdiccion_completa"
] = pd.NA

resultado[
    "provincia"
] = pd.NA

resultado[
    "categoria_jurisdiccion"
] = pd.NA

resultado[
    "fuente_jurisdiccion"
] = pd.NA

resultado[
    "metodo_asignacion"
] = pd.NA

resultado[
    "distancia_asignacion_m"
] = pd.NA


# ============================================================
# MÉTODO 1
# PUNTO REPRESENTATIVO
# ============================================================

imprimir_titulo(
    "MÉTODO 1: PUNTO REPRESENTATIVO"
)


# ------------------------------------------------------------
# Crear puntos
# ------------------------------------------------------------

puntos = h3[
    [
        "id_h3",
        "geometry",
    ]
].copy()

puntos["geometry"] = (
    puntos.geometry
    .representative_point()
)

puntos = gpd.GeoDataFrame(
    puntos,
    geometry="geometry",
    crs=h3.crs,
)


# ------------------------------------------------------------
# Spatial join
# ------------------------------------------------------------

join_punto = gpd.sjoin(
    puntos,
    territorio[COLUMNAS_TERRITORIO],
    how="left",
    predicate="within",
)


# ------------------------------------------------------------
# Detectar múltiples jurisdicciones
# ------------------------------------------------------------

conteo_puntos = (
    join_punto
    .groupby("id_h3")
    .size()
)

multiples_punto = (
    conteo_puntos[
        conteo_puntos > 1
    ]
)

print(
    "H3 con múltiples jurisdicciones "
    "por punto representativo: "
    f"{len(multiples_punto):,}"
)


# ------------------------------------------------------------
# Preparar asignaciones
# ------------------------------------------------------------

asignaciones_1 = preparar_asignaciones(
    join_punto,
    "REPRESENTATIVE_POINT",
)


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

resultado = resultado.merge(
    asignaciones_1,
    on="id_h3",
    how="left",
    suffixes=(
        "",
        "_nuevo",
    ),
)


# ------------------------------------------------------------
# Aplicar columnas nuevas
# ------------------------------------------------------------

for columna in COLUMNAS_ASIGNACION:

    nueva = f"{columna}_nuevo"

    if nueva in resultado.columns:

        resultado[columna] = (
            resultado[nueva]
        )

        resultado = resultado.drop(
            columns=nueva
        )


asignados_1 = (
    resultado["jurisdiccion"]
    .notna()
    .sum()
)

print(
    f"H3 asignados: "
    f"{asignados_1:,}"
)


# ============================================================
# MÉTODO 2
# INTERSECCIÓN POR ÁREA
# ============================================================

sin_asignar = obtener_pendientes(
    resultado
)

print(
    f"H3 sin asignar: "
    f"{len(sin_asignar):,}"
)

imprimir_titulo(
    "MÉTODO 2: INTERSECCIÓN POR ÁREA"
)


asignados_2 = 0


if len(sin_asignar) > 0:

    # --------------------------------------------------------
    # Proyección métrica
    # --------------------------------------------------------

    h3_intersect = gpd.GeoDataFrame(
        sin_asignar,
        geometry="geometry",
        crs=h3.crs,
    ).to_crs(
        CRS_METRICO
    )

    territorio_m = (
        territorio
        .to_crs(CRS_METRICO)
    )


    # --------------------------------------------------------
    # Spatial join
    # --------------------------------------------------------

    inter = gpd.sjoin(
        h3_intersect,
        territorio_m[
            COLUMNAS_TERRITORIO
        ],
        how="inner",
        predicate="intersects",
    )


    if len(inter) > 0:

        # ----------------------------------------------------
        # Recuperar geometría territorial
        # ----------------------------------------------------

        geometria_territorio = (
            territorio_m[
                [
                    "id",
                    "geometry",
                ]
            ]
            .drop_duplicates(
                "id"
            )
            .set_index("id")[
                "geometry"
            ]
        )

        inter[
            "geometry_territorio"
        ] = (
            inter["id"]
            .map(
                geometria_territorio
            )
        )


        # ----------------------------------------------------
        # Área de intersección
        # ----------------------------------------------------

        inter[
            "area_interseccion_m2"
        ] = (
            inter.geometry
            .intersection(
                inter[
                    "geometry_territorio"
                ]
            )
            .area
        )


        # ----------------------------------------------------
        # Eliminar intersecciones nulas
        # ----------------------------------------------------

        inter = inter[
            inter[
                "area_interseccion_m2"
            ] > 0
        ].copy()


        if len(inter) > 0:

            # -----------------------------------------------
            # Elegir mayor área
            # -----------------------------------------------

            inter = inter.sort_values(
                [
                    "id_h3",
                    "area_interseccion_m2",
                    "id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
                na_position="last",
            )

            inter = (
                inter
                .drop_duplicates(
                    subset="id_h3",
                    keep="first",
                )
            )


            # -----------------------------------------------
            # Preparar asignaciones
            # -----------------------------------------------

            asignaciones_2_df = (
                preparar_asignaciones(
                    inter,
                    "INTERSECCION_MAYOR_AREA",
                )
            )


            # -----------------------------------------------
            # Merge
            # -----------------------------------------------

            resultado = resultado.merge(
                asignaciones_2_df,
                on="id_h3",
                how="left",
                suffixes=(
                    "",
                    "_nuevo",
                ),
            )


            # -----------------------------------------------
            # Aplicar solamente a pendientes
            # -----------------------------------------------

            mascara_pendiente = (
                resultado[
                    "jurisdiccion"
                ].isna()
            )

            for columna in COLUMNAS_ASIGNACION:

                nueva = (
                    f"{columna}_nuevo"
                )

                if nueva in resultado.columns:

                    resultado.loc[
                        mascara_pendiente,
                        columna,
                    ] = (
                        resultado.loc[
                            mascara_pendiente,
                            nueva,
                        ]
                    )

                    resultado = (
                        resultado.drop(
                            columns=nueva
                        )
                    )

            asignados_2 = (
                asignaciones_2_df[
                    "id_h3"
                ].nunique()
            )


print(
    f"H3 asignados por intersección: "
    f"{asignados_2:,}"
)


# ============================================================
# MÉTODO 3
# NEAREST
# ============================================================

sin_asignar = obtener_pendientes(
    resultado
)

print(
    f"H3 pendientes después de intersección: "
    f"{len(sin_asignar):,}"
)

imprimir_titulo(
    "MÉTODO 3: JURISDICCIÓN MÁS CERCANA"
)

print(
    f"H3 pendientes: "
    f"{len(sin_asignar):,}"
)


asignados_3 = 0
rechazados_nearest = 0


if len(sin_asignar) > 0:

    # --------------------------------------------------------
    # Punto representativo
    # --------------------------------------------------------

    puntos_restantes = (
        gpd.GeoDataFrame(
            sin_asignar,
            geometry="geometry",
            crs=h3.crs,
        )
    )

    puntos_restantes[
        "geometry"
    ] = (
        puntos_restantes.geometry
        .representative_point()
    )


    # --------------------------------------------------------
    # Proyectar
    # --------------------------------------------------------

    puntos_m = (
        puntos_restantes
        .to_crs(CRS_METRICO)
    )

    territorio_m = (
        territorio
        .to_crs(CRS_METRICO)
    )


    # --------------------------------------------------------
    # Nearest
    # --------------------------------------------------------

    nearest = gpd.sjoin_nearest(
        puntos_m,
        territorio_m[
            COLUMNAS_TERRITORIO
        ],
        how="left",
        distance_col=(
            "distancia_asignacion_m"
        ),
    )


    # --------------------------------------------------------
    # Resolver empates
    # --------------------------------------------------------

    nearest = nearest.sort_values(
        [
            "id_h3",
            "distancia_asignacion_m",
            "id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        na_position="last",
    )

    nearest = (
        nearest
        .drop_duplicates(
            subset="id_h3",
            keep="first",
        )
    )


    # --------------------------------------------------------
    # Distancia válida
    # --------------------------------------------------------

    nearest_validos = nearest[
        nearest[
            "distancia_asignacion_m"
        ].notna()
        &
        (
            nearest[
                "distancia_asignacion_m"
            ]
            <= MAX_DISTANCIA_M
        )
    ].copy()

    rechazados_nearest = (
        len(nearest)
        -
        len(nearest_validos)
    )

    print(
        f"H3 encontrados por nearest: "
        f"{len(nearest_validos):,}"
    )

    print(
        f"H3 rechazados por superar "
        f"{MAX_DISTANCIA_M:,} m: "
        f"{rechazados_nearest:,}"
    )


    if len(nearest_validos) > 0:

        asignaciones_3_df = (
            preparar_asignaciones(
                nearest_validos,
                "NEAREST",
                distancia_col=(
                    "distancia_asignacion_m"
                ),
            )
        )


        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        resultado = resultado.merge(
            asignaciones_3_df,
            on="id_h3",
            how="left",
            suffixes=(
                "",
                "_nuevo",
            ),
        )


        # ----------------------------------------------------
        # Aplicar a pendientes
        # ----------------------------------------------------

        mascara_pendiente = (
            resultado[
                "jurisdiccion"
            ].isna()
        )

        for columna in COLUMNAS_ASIGNACION:

            nueva = (
                f"{columna}_nuevo"
            )

            if nueva in resultado.columns:

                resultado.loc[
                    mascara_pendiente,
                    columna,
                ] = (
                    resultado.loc[
                        mascara_pendiente,
                        nueva,
                    ]
                )

                resultado = (
                    resultado.drop(
                        columns=nueva
                    )
                )


        asignados_3 = (
            asignaciones_3_df[
                "id_h3"
            ].nunique()
        )


print(
    f"H3 asignados por nearest: "
    f"{asignados_3:,}"
)


# ============================================================
# NORMALIZAR TIPOS
# ============================================================

resultado[
    "distancia_asignacion_m"
] = pd.to_numeric(
    resultado[
        "distancia_asignacion_m"
    ],
    errors="coerce",
)


# ============================================================
# VALIDACIÓN TERRITORIAL
# ============================================================

imprimir_titulo(
    "VALIDACIÓN TERRITORIAL"
)

total = len(resultado)

con_jurisdiccion = (
    resultado[
        "jurisdiccion"
    ]
    .notna()
    .sum()
)

sin_jurisdiccion = (
    resultado[
        "jurisdiccion"
    ]
    .isna()
    .sum()
)

print(
    f"\nH3 originales:       "
    f"{len(h3):,}"
)

print(
    f"H3 resultado:        "
    f"{total:,}"
)

print(
    f"H3 con jurisdicción: "
    f"{con_jurisdiccion:,}"
)

print(
    f"H3 sin jurisdicción: "
    f"{sin_jurisdiccion:,}"
)

print(
    f"% asignados:         "
    f"{con_jurisdiccion / total * 100:.2f}%"
)


# ============================================================
# DISTRIBUCIÓN POR MÉTODO
# ============================================================

print(
    "\nDistribución por método:"
)

print(
    resultado[
        "metodo_asignacion"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# OPERACIONES SIN JURISDICCIÓN
# ============================================================

operaciones_sin_jurisdiccion = (
    resultado.loc[
        resultado[
            "jurisdiccion"
        ].isna(),
        "operaciones_totales",
    ]
    .fillna(0)
    .sum()
)

print(
    "\nOperaciones sin jurisdicción:"
)

print(
    f"{operaciones_sin_jurisdiccion:,.0f}"
)

porcentaje_operaciones_sin = (
    operaciones_sin_jurisdiccion
    /
    operaciones_originales
    *
    100
)

print(
    f"% operaciones sin jurisdicción: "
    f"{porcentaje_operaciones_sin:.2f}%"
)


# ============================================================
# VALIDAR OPERACIONES
# ============================================================

operaciones_finales = (
    resultado[
        "operaciones_totales"
    ]
    .fillna(0)
    .sum()
)

diferencia_operaciones = (
    operaciones_finales
    -
    operaciones_originales
)

print(
    "\nValidación de operaciones:"
)

print(
    f"Originales: "
    f"{operaciones_originales:,.0f}"
)

print(
    f"Finales:    "
    f"{operaciones_finales:,.0f}"
)

print(
    f"Diferencia: "
    f"{diferencia_operaciones:,.0f}"
)

if abs(diferencia_operaciones) > 0.01:

    raise ValueError(
        "ERROR: las operaciones totales "
        "cambiaron durante la asignación."
    )


# ============================================================
# VALIDAR DUPLICADOS
# ============================================================

duplicados_finales = (
    resultado[
        "id_h3"
    ]
    .duplicated()
    .sum()
)

print(
    f"\nH3 duplicados finales: "
    f"{duplicados_finales:,}"
)

if total != len(h3):

    raise ValueError(
        f"ERROR: se esperaban "
        f"{len(h3):,} H3 "
        f"pero quedaron {total:,}."
    )

if duplicados_finales > 0:

    raise ValueError(
        "ERROR: quedaron H3 duplicados."
    )


# ============================================================
# H3 SIN JURISDICCIÓN
# ============================================================

if sin_jurisdiccion > 0:

    imprimir_titulo(
        "ADVERTENCIA: H3 SIN JURISDICCIÓN"
    )

    pendientes = resultado[
        resultado[
            "jurisdiccion"
        ].isna()
    ].copy()

    print(
        pendientes[
            [
                "id_h3",
                "operaciones_totales",
                "categoria_demanda",
            ]
        ]
        .sort_values(
            "operaciones_totales",
            ascending=False,
        )
        .head(30)
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nTodos los H3 tienen jurisdicción."
    )


# ============================================================
# ESTADÍSTICAS NEAREST
# ============================================================

nearest_asignados = resultado[
    resultado[
        "metodo_asignacion"
    ]
    ==
    "NEAREST"
]

if len(nearest_asignados) > 0:

    imprimir_titulo(
        "ESTADÍSTICAS DE ASIGNACIONES NEAREST"
    )

    print(
        nearest_asignados[
            "distancia_asignacion_m"
        ]
        .describe()
        .to_string()
    )


# ============================================================
# PROVINCIAS
# ============================================================

print(
    "\nProvincias encontradas:"
)

print(
    resultado[
        "provincia"
    ]
    .value_counts(
        dropna=False
    )
    .head(20)
    .to_string()
)


# ============================================================
# JURISDICCIONES POR H3
# ============================================================

print(
    "\nPrincipales jurisdicciones por cantidad de H3:"
)

print(
    resultado[
        "jurisdiccion"
    ]
    .value_counts(
        dropna=True
    )
    .head(30)
    .to_string()
)


# ============================================================
# OPERACIONES POR JURISDICCIÓN
# ============================================================

print(
    "\nPrincipales jurisdicciones por operaciones:"
)

ranking = (
    resultado
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
        hotspots_extremos=(
            "categoria_demanda",
            lambda x:
                (
                    x
                    ==
                    "HOTSPOT_EXTREMO"
                ).sum(),
        ),
        hotspots_altos=(
            "categoria_demanda",
            lambda x:
                (
                    x
                    ==
                    "HOTSPOT_ALTO"
                ).sum(),
        ),
    )
    .sort_values(
        "operaciones",
        ascending=False,
    )
)

print(
    ranking
    .head(30)
    .to_string()
)


# ============================================================
# DISTRIBUCIÓN TERRITORIAL
# ============================================================

print(
    "\nDistribución territorial:"
)

distribucion = (
    resultado
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
    )
)

distribucion[
    "pct_operaciones"
] = (
    distribucion[
        "operaciones"
    ]
    /
    operaciones_finales
    *
    100
)

distribucion[
    "pct_h3"
] = (
    distribucion[
        "h3"
    ]
    /
    total
    *
    100
)

print(
    distribucion
    .sort_values(
        "operaciones",
        ascending=False,
    )
    .to_string()
)


# ============================================================
# VALIDACIÓN FINAL ESTRICTA
# ============================================================

imprimir_titulo(
    "VALIDACIÓN FINAL"
)

validaciones = []


# ------------------------------------------------------------
# Cantidad H3
# ------------------------------------------------------------

ok_h3 = (
    total == len(h3)
)

validaciones.append(
    (
        "Cantidad H3",
        ok_h3,
    )
)


# ------------------------------------------------------------
# H3 únicos
# ------------------------------------------------------------

ok_unicos = (
    duplicados_finales == 0
)

validaciones.append(
    (
        "H3 únicos",
        ok_unicos,
    )
)


# ------------------------------------------------------------
# Operaciones
# ------------------------------------------------------------

ok_operaciones = (
    abs(diferencia_operaciones)
    <= 0.01
)

validaciones.append(
    (
        "Operaciones conservadas",
        ok_operaciones,
    )
)


# ------------------------------------------------------------
# Jurisdicciones
# ------------------------------------------------------------

ok_jurisdicciones = (
    sin_jurisdiccion == 0
)

validaciones.append(
    (
        "Sin jurisdicciones faltantes",
        ok_jurisdicciones,
    )
)


# ------------------------------------------------------------
# Método
# ------------------------------------------------------------

ok_metodo = (
    resultado[
        "metodo_asignacion"
    ]
    .notna()
    .all()
)

validaciones.append(
    (
        "Método para cada H3",
        ok_metodo,
    )
)


# ------------------------------------------------------------
# Mostrar
# ------------------------------------------------------------

for nombre, ok in validaciones:

    print(
        f"   {'OK' if ok else 'ERROR':5} "
        f"{nombre}"
    )


# ============================================================
# DECISIÓN FINAL
# ============================================================

fallos = [
    nombre
    for nombre, ok in validaciones
    if not ok
]

if fallos:

    raise ValueError(
        "La validación final falló en: "
        +
        ", ".join(fallos)
    )


# ============================================================
# GUARDAR
# ============================================================

imprimir_titulo(
    "GUARDANDO RESULTADOS"
)

print(
    OUTPUT_FILE
)


resultado.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

imprimir_titulo(
    "ASIGNACIÓN TERRITORIAL FINALIZADA CORRECTAMENTE"
)

print(
    f"H3 procesados: "
    f"{len(resultado):,}"
)

print(
    f"H3 asignados: "
    f"{con_jurisdiccion:,}"
)

print(
    f"Operaciones conservadas: "
    f"{operaciones_finales:,.0f}"
)

print(
    f"Archivo: "
    f"{OUTPUT_FILE}"
)