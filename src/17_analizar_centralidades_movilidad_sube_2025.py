from pathlib import Path
import json

import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

H3_CLUSTERS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_clusters.parquet"
)

CLUSTERS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_clusters_movilidad.parquet"
)

CORREDORES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_corredores_movilidad.parquet"
)

NODOS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_nodos_movilidad.parquet"
)

H3_NODOS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_nodos.parquet"
)

OUTPUT_CENTRALIDADES = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_centralidades_movilidad.parquet"
)

OUTPUT_H3_CENTRALIDADES = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_centralidades.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_centralidades_resumen.json"
)


CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:3857"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def percentil_normalizado(serie):

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    )

    if serie.notna().sum() <= 1:

        resultado = pd.Series(
            50.0,
            index=serie.index
        )

    else:

        resultado = (
            serie.rank(
                pct=True,
                method="average"
            )
            * 100
        )

    return resultado.fillna(0)


def modo_dominante(serie):

    serie = (
        serie
        .dropna()
        .astype(str)
    )

    if serie.empty:
        return None

    return serie.value_counts().index[0]


def hora_dominante(serie):

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    ).dropna()

    if serie.empty:
        return None

    return int(
        serie.value_counts().index[0]
    )


def safe_int(valor):

    if pd.isna(valor):
        return None

    try:
        return int(valor)
    except Exception:
        return None


def safe_float(valor):

    if pd.isna(valor):
        return None

    try:
        return float(valor)
    except Exception:
        return None


def valores_unicos(serie):

    if serie is None:
        return []

    valores = (
        serie
        .dropna()
        .astype(str)
        .str.strip()
    )

    valores = [
        v
        for v in valores.unique().tolist()
        if v not in ("", "nan", "None")
    ]

    valores.sort()

    return valores


def texto_unicos(serie):

    return " | ".join(
        valores_unicos(serie)
    )


def normalizar_id_h3(df):

    if "id_h3" not in df.columns:
        return df

    df = df.copy()

    df["id_h3"] = (
        df["id_h3"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ANÁLISIS DE CENTRALIDADES DE MOVILIDAD SUBE 2025")
print("=" * 70)


# ============================================================
# CARGAR LIBRERÍA H3
# ============================================================

print("\nCargando librería H3...")

try:

    import h3

except ImportError:

    raise ImportError(
        "No está instalada la librería h3.\n"
        "Instalar con:\n"
        "pip install h3"
    )


print(
    f"Versión H3: "
    f"{getattr(h3, '__version__', 'desconocida')}"
)


# ============================================================
# CARGAR H3
# ============================================================

print("\nCargando H3...")

h3_base = pd.read_parquet(
    H3_CLUSTERS_PATH
)

h3_base = normalizar_id_h3(
    h3_base
)

print(
    f"H3 cargados: "
    f"{len(h3_base):,}"
)

print("Columnas:")
print(h3_base.columns.tolist())


# ============================================================
# VALIDACIÓN H3
# ============================================================

requeridas_h3 = [
    "id_h3",
    "operaciones_totales",
]

faltantes = [
    columna
    for columna in requeridas_h3
    if columna not in h3_base.columns
]

if faltantes:

    raise ValueError(
        "Faltan columnas obligatorias en H3: "
        + ", ".join(faltantes)
    )


# ============================================================
# VALIDAR DUPLICADOS H3
# ============================================================

duplicados_h3 = (
    h3_base["id_h3"]
    .duplicated()
    .sum()
)

print(
    f"\nH3 duplicados: "
    f"{duplicados_h3:,}"
)

if duplicados_h3 > 0:

    raise ValueError(
        "El archivo H3 contiene identificadores duplicados."
    )


# ============================================================
# RECONSTRUIR GEOMETRÍAS
# ============================================================

print(
    "\nReconstruyendo geometrías H3..."
)


def h3_a_poligono(h3_id):

    try:

        limites = h3.cell_to_boundary(
            str(h3_id)
        )

        coordenadas = [
            (lon, lat)
            for lat, lon in limites
        ]

        return Polygon(
            coordenadas
        )

    except Exception:

        return None


h3_base["geometry"] = (
    h3_base["id_h3"]
    .apply(h3_a_poligono)
)


h3_geo = gpd.GeoDataFrame(
    h3_base,
    geometry="geometry",
    crs=CRS_GEOGRAFICO
)


h3_geo = h3_geo[
    h3_geo.geometry.notna()
].copy()


h3_geo = h3_geo[
    ~h3_geo.geometry.is_empty
].copy()


print(
    f"Geometrías reconstruidas: "
    f"{len(h3_geo):,}"
)


# ============================================================
# SUPERFICIES
# ============================================================

print(
    "\nCalculando superficies..."
)


h3_m = h3_geo.to_crs(
    CRS_METRICO
)


h3_geo[
    "superficie_km2_calculada"
] = (
    h3_m.geometry.area
    / 1_000_000
)


print(
    f"Superficie H3 total calculada: "
    f"{h3_geo['superficie_km2_calculada'].sum():,.2f} km²"
)


# ============================================================
# CARGAR NODOS
# ============================================================

print(
    "\nCargando nodos..."
)


nodos = pd.read_parquet(
    NODOS_PATH
)


print(
    f"Nodos cargados: "
    f"{len(nodos):,}"
)


print("Columnas:")
print(nodos.columns.tolist())


if "nodo_id" not in nodos.columns:

    raise ValueError(
        "El archivo de nodos no contiene 'nodo_id'."
    )


# ============================================================
# CARGAR RELACIÓN H3 → NODO
# ============================================================

print(
    "\nCargando relación H3 → nodo..."
)


h3_nodos = pd.read_parquet(
    H3_NODOS_PATH
)


h3_nodos = normalizar_id_h3(
    h3_nodos
)


print(
    f"H3 → nodo cargados: "
    f"{len(h3_nodos):,}"
)


print("Columnas:")
print(h3_nodos.columns.tolist())


requeridas_h3_nodos = [
    "id_h3",
    "nodo_id",
]


faltantes = [
    columna
    for columna in requeridas_h3_nodos
    if columna not in h3_nodos.columns
]


if faltantes:

    raise ValueError(
        "Faltan columnas en H3 → nodo: "
        + ", ".join(faltantes)
    )


# ============================================================
# VALIDAR DUPLICADOS H3 → NODO
# ============================================================

duplicados_h3_nodo = (
    h3_nodos["id_h3"]
    .duplicated()
    .sum()
)


print(
    f"\nH3 duplicados en relación H3 → nodo: "
    f"{duplicados_h3_nodo:,}"
)


if duplicados_h3_nodo > 0:

    print(
        "ADVERTENCIA: un H3 aparece más de una vez "
        "en la relación H3 → nodo."
    )

    h3_nodos = (
        h3_nodos
        .drop_duplicates(
            subset=["id_h3"],
            keep="first"
        )
        .copy()
    )


# ============================================================
# RELACIONES DISPONIBLES
# ============================================================

print(
    "\nRelaciones disponibles:"
)

print(
    "  H3 → nodo: SI"
)

print(
    "  H3 → cluster: "
    + (
        "SI"
        if "cluster_id" in h3_nodos.columns
        else "NO"
    )
)

print(
    "  H3 → corredor: "
    + (
        "SI"
        if "corredor_id" in h3_nodos.columns
        else "NO"
    )
)


# ============================================================
# CARGAR CORREDORES
# ============================================================

print(
    "\nCargando información agregada de corredores..."
)


corredores = pd.read_parquet(
    CORREDORES_PATH
)


print(
    f"Registros de corredores: "
    f"{len(corredores):,}"
)


print("Columnas:")
print(corredores.columns.tolist())


# ============================================================
# IMPORTANTE:
# EL PARQUET DE CORREDORES ES AGREGADO POR CLUSTER
# ============================================================

if "cluster_id" in corredores.columns:

    corredores["cluster_id"] = (
        corredores["cluster_id"]
        .astype(str)
        .str.strip()
    )


# ============================================================
# PREPARAR MAPA DE CORREDOR
# ============================================================

corredor_por_cluster = {}

if (
    "cluster_id" in corredores.columns
    and "indice_corredor" in corredores.columns
):

    columnas_corredor = [
        "cluster_id",
        "indice_corredor",
        "categoria_corredor",
        "ranking_operaciones",
    ]

    columnas_corredor = [
        c
        for c in columnas_corredor
        if c in corredores.columns
    ]

    corredor_por_cluster = (
        corredores[
            columnas_corredor
        ]
        .drop_duplicates(
            subset=["cluster_id"]
        )
        .set_index(
            "cluster_id"
        )
        .to_dict(
            orient="index"
        )
    )


# ============================================================
# PREPARAR H3
# ============================================================

print(
    "\nPreparando indicadores H3..."
)


# ------------------------------------------------------------
# ASEGURAR COLUMNAS BASE
# ------------------------------------------------------------

columnas_base = [
    "id_h3",
    "cluster_id",
    "provincia",
    "jurisdiccion",
    "operaciones_totales",
    "categoria_demanda",
    "hora_pico",
    "modo_dominante",
    "superficie_km2_calculada",
]


for columna in columnas_base:

    if columna not in h3_geo.columns:

        h3_geo[columna] = None


# ============================================================
# CREAR MAPAS H3
# ============================================================

print(
    "\nConstruyendo mapas H3..."
)


# ============================================================
# MAPA H3 → NODO
# ============================================================

mapa_h3_nodo = (
    h3_nodos[
        [
            "id_h3",
            "nodo_id",
        ]
    ]
    .copy()
)


mapa_h3_nodo["id_h3"] = (
    mapa_h3_nodo["id_h3"]
    .astype(str)
)


mapa_h3_nodo = (
    mapa_h3_nodo
    .drop_duplicates(
        subset=["id_h3"]
    )
)


# ============================================================
# MAPA H3 → CLUSTER
# ============================================================

if "cluster_id" in h3_nodos.columns:

    mapa_h3_cluster = (
        h3_nodos[
            [
                "id_h3",
                "cluster_id",
            ]
        ]
        .copy()
    )

    mapa_h3_cluster["id_h3"] = (
        mapa_h3_cluster["id_h3"]
        .astype(str)
    )

    mapa_h3_cluster["cluster_id"] = (
        mapa_h3_cluster["cluster_id"]
        .astype(str)
    )

    mapa_h3_cluster = (
        mapa_h3_cluster
        .drop_duplicates(
            subset=["id_h3"]
        )
    )

else:

    mapa_h3_cluster = pd.DataFrame(
        columns=[
            "id_h3",
            "cluster_id",
        ]
    )


# ============================================================
# MAPA H3 → CORREDOR
# ============================================================

if "corredor_id" in h3_nodos.columns:

    mapa_h3_corredor = (
        h3_nodos[
            [
                "id_h3",
                "corredor_id",
            ]
        ]
        .copy()
    )

    mapa_h3_corredor["id_h3"] = (
        mapa_h3_corredor["id_h3"]
        .astype(str)
    )

    mapa_h3_corredor = (
        mapa_h3_corredor
        .drop_duplicates(
            subset=["id_h3"]
        )
    )

else:

    mapa_h3_corredor = pd.DataFrame(
        columns=[
            "id_h3",
            "corredor_id",
        ]
    )


# ============================================================
# MERGE H3 → NODO
# ============================================================

h3_geo = h3_geo.drop(
    columns=[
        "nodo_id",
        "cluster_id_h3_nodo",
        "cluster_id_x",
        "cluster_id_y",
        "corredor_id",
    ],
    errors="ignore"
)


h3_geo = h3_geo.merge(
    mapa_h3_nodo,
    on="id_h3",
    how="left",
)


# ============================================================
# MERGE H3 → CLUSTER
# ============================================================

# Si el H3 original ya tenía cluster_id,
# se utiliza como fuente primaria.

if "cluster_id" in h3_geo.columns:

    h3_geo["cluster_id"] = (
        h3_geo["cluster_id"]
        .astype("string")
    )

else:

    h3_geo["cluster_id"] = (
        pd.Series(
            pd.NA,
            index=h3_geo.index,
            dtype="string"
        )
    )


if not mapa_h3_cluster.empty:

    h3_geo = h3_geo.merge(
        mapa_h3_cluster.rename(
            columns={
                "cluster_id":
                    "cluster_id_relacion"
            }
        ),
        on="id_h3",
        how="left",
    )

    h3_geo[
        "cluster_id"
    ] = h3_geo[
        "cluster_id"
    ].fillna(
        h3_geo[
            "cluster_id_relacion"
        ]
    )

    h3_geo = h3_geo.drop(
        columns=[
            "cluster_id_relacion"
        ],
        errors="ignore"
    )


# ============================================================
# MERGE H3 → CORREDOR
# ============================================================

if not mapa_h3_corredor.empty:

    h3_geo = h3_geo.merge(
        mapa_h3_corredor,
        on="id_h3",
        how="left",
    )

else:

    h3_geo[
        "corredor_id"
    ] = None


# ============================================================
# ESTADÍSTICAS DE RELACIÓN
# ============================================================

h3_con_nodo = (
    h3_geo["nodo_id"]
    .notna()
    .sum()
)

h3_con_cluster = (
    h3_geo["cluster_id"]
    .notna()
    .sum()
)

h3_con_corredor = (
    h3_geo["corredor_id"]
    .notna()
    .sum()
)


print(
    f"\nH3 asociados a nodos: "
    f"{h3_con_nodo:,}"
)

print(
    f"H3 sin nodo: "
    f"{len(h3_geo) - h3_con_nodo:,}"
)

print(
    f"H3 asociados a clusters: "
    f"{h3_con_cluster:,}"
)

print(
    f"H3 asociados a corredores: "
    f"{h3_con_corredor:,}"
)


# ============================================================
# VALIDACIÓN ESTRUCTURAL
# ============================================================

print(
    "\nValidando estructura H3..."
)


columnas_importantes = [
    "id_h3",
    "cluster_id",
    "nodo_id",
    "corredor_id",
]


for columna in columnas_importantes:

    if columna not in h3_geo.columns:

        raise RuntimeError(
            f"No se pudo construir la columna "
            f"'{columna}' en h3_geo."
        )


print(
    "Estructura H3 validada correctamente."
)


# ============================================================
# DEMANDA
# ============================================================

h3_geo["operaciones"] = (
    pd.to_numeric(
        h3_geo[
            "operaciones_totales"
        ],
        errors="coerce"
    )
    .fillna(0)
)


# ============================================================
# CANDIDATOS A CENTRALIDAD
# ============================================================

h3_nodo = h3_geo[
    h3_geo["nodo_id"].notna()
].copy()


print(
    f"\nH3 asociados a nodos: "
    f"{len(h3_nodo):,}"
)


print(
    f"Operaciones asociadas a nodos: "
    f"{h3_nodo['operaciones'].sum():,.0f}"
)


# ============================================================
# CONVERTIR NODO_ID
# ============================================================

h3_nodo["nodo_id"] = pd.to_numeric(
    h3_nodo["nodo_id"],
    errors="coerce"
)


h3_nodo = h3_nodo[
    h3_nodo["nodo_id"].notna()
].copy()


h3_nodo["nodo_id"] = (
    h3_nodo["nodo_id"]
    .astype(int)
)


# ============================================================
# CARACTERIZAR CENTRALIDADES
# ============================================================

print(
    "\nCalculando centralidades..."
)


registros = []


for nodo_id, grupo in h3_nodo.groupby(
    "nodo_id"
):

    operaciones = (
        grupo["operaciones"]
        .sum()
    )


    superficie = (
        grupo[
            "superficie_km2_calculada"
        ]
        .sum()
    )


    if superficie > 0:

        densidad = (
            operaciones
            / superficie
        )

    else:

        densidad = 0


    # --------------------------------------------------------
    # CORREDORES
    # --------------------------------------------------------

    corredores_unicos = valores_unicos(
        grupo["corredor_id"]
    )


    # --------------------------------------------------------
    # CLUSTERS
    # --------------------------------------------------------

    clusters_unicos = valores_unicos(
        grupo["cluster_id"]
    )


    # --------------------------------------------------------
    # JURISDICCIONES
    # --------------------------------------------------------

    jurisdicciones = valores_unicos(
        grupo["jurisdiccion"]
    )


    # --------------------------------------------------------
    # PROVINCIAS
    # --------------------------------------------------------

    provincias = valores_unicos(
        grupo["provincia"]
    )


    # --------------------------------------------------------
    # MODOS
    # --------------------------------------------------------

    if "modo_dominante" in grupo.columns:

        modo = modo_dominante(
            grupo[
                "modo_dominante"
            ]
        )

    else:

        modo = None


    # --------------------------------------------------------
    # HORA
    # --------------------------------------------------------

    if "hora_pico" in grupo.columns:

        hora = hora_dominante(
            grupo[
                "hora_pico"
            ]
        )

    else:

        hora = None


    # --------------------------------------------------------
    # CATEGORÍA DEMANDA
    # --------------------------------------------------------

    if "categoria_demanda" in grupo.columns:

        categoria = modo_dominante(
            grupo[
                "categoria_demanda"
            ]
        )

    else:

        categoria = None


    # --------------------------------------------------------
    # INTERMODALIDAD
    # --------------------------------------------------------

    modos = set()

    if "modo_dominante" in grupo.columns:

        modos = set(
            grupo[
                "modo_dominante"
            ]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
        )


    tiene_colectivo = (
        any(
            "COLECT" in modo
            for modo in modos
        )
    )


    tiene_tren = (
        any(
            "TREN" in modo
            or "FERRO" in modo
            for modo in modos
        )
    )


    if tiene_colectivo and tiene_tren:

        tipo_intermodalidad = (
            "INTERMODAL"
        )

        score_intermodalidad = 100

    elif tiene_colectivo or tiene_tren:

        tipo_intermodalidad = (
            "MONOMODAL"
        )

        score_intermodalidad = 50

    else:

        tipo_intermodalidad = (
            "SIN_DATOS"
        )

        score_intermodalidad = 0


    # --------------------------------------------------------
    # ALCANCE TERRITORIAL
    # --------------------------------------------------------

    cantidad_jurisdicciones = (
        len(jurisdicciones)
    )


    if cantidad_jurisdicciones >= 8:

        alcance = "METROPOLITANO"

    elif cantidad_jurisdicciones >= 3:

        alcance = "INTERJURISDICCIONAL"

    elif cantidad_jurisdicciones == 2:

        alcance = "BIJURISDICCIONAL"

    else:

        alcance = "LOCAL"


    # --------------------------------------------------------
    # CENTROIDE
    # --------------------------------------------------------

    union = grupo.geometry.union_all()

    centroide = union.centroid


    # --------------------------------------------------------
    # ÍNDICE CORREDOR PROMEDIO
    # --------------------------------------------------------

    indices_corredor = []

    if "cluster_id" in grupo.columns:

        for cluster_id in clusters_unicos:

            info = corredor_por_cluster.get(
                str(cluster_id)
            )

            if info:

                valor = info.get(
                    "indice_corredor"
                )

                if valor is not None:

                    try:

                        indices_corredor.append(
                            float(valor)
                        )

                    except Exception:

                        pass


    if indices_corredor:

        indice_corredor_promedio = (
            sum(indices_corredor)
            / len(indices_corredor)
        )

    else:

        indice_corredor_promedio = 0


    # --------------------------------------------------------
    # REGISTRO
    # --------------------------------------------------------

    registros.append({

        "nodo_id":
            int(nodo_id),

        "h3":
            int(len(grupo)),

        "operaciones":
            float(operaciones),

        "superficie_km2":
            float(superficie),

        "operaciones_por_km2":
            float(densidad),

        "cantidad_corredores":
            int(len(corredores_unicos)),

        "cantidad_clusters":
            int(len(clusters_unicos)),

        "cantidad_jurisdicciones":
            int(cantidad_jurisdicciones),

        "cantidad_provincias":
            int(len(provincias)),

        "jurisdicciones":
            " | ".join(
                jurisdicciones
            ),

        "provincias":
            " | ".join(
                provincias
            ),

        "corredores":
            " | ".join(
                corredores_unicos
            ),

        "clusters":
            " | ".join(
                clusters_unicos
            ),

        "categoria_dominante":
            categoria,

        "hora_pico":
            hora,

        "modo_dominante":
            modo,

        "tipo_intermodalidad":
            tipo_intermodalidad,

        "score_intermodalidad":
            float(
                score_intermodalidad
            ),

        "indice_corredor_promedio":
            float(
                indice_corredor_promedio
            ),

        "alcance_territorial":
            alcance,

        "geometry":
            centroide,

    })


# ============================================================
# CREAR GEODATAFRAME
# ============================================================

centralidades = gpd.GeoDataFrame(
    registros,
    geometry="geometry",
    crs=CRS_GEOGRAFICO
)


if centralidades.empty:

    raise RuntimeError(
        "No se pudieron construir centralidades."
    )


print(
    f"Centralidades construidas: "
    f"{len(centralidades):,}"
)


# ============================================================
# ÍNDICES NORMALIZADOS
# ============================================================

print(
    "\nCalculando componentes del índice..."
)


centralidades[
    "score_demanda"
] = percentil_normalizado(
    centralidades[
        "operaciones"
    ]
)


centralidades[
    "score_densidad"
] = percentil_normalizado(
    centralidades[
        "operaciones_por_km2"
    ]
)


centralidades[
    "score_conectividad"
] = percentil_normalizado(
    centralidades[
        "cantidad_corredores"
    ]
)


centralidades[
    "score_alcance"
] = percentil_normalizado(
    centralidades[
        "cantidad_jurisdicciones"
    ]
)


centralidades[
    "score_integracion"
] = percentil_normalizado(
    centralidades[
        "cantidad_clusters"
    ]
)


# ============================================================
# ÍNDICE DE CENTRALIDAD
# ============================================================

centralidades[
    "indice_centralidad"
] = (

    centralidades[
        "score_demanda"
    ] * 0.30

    +

    centralidades[
        "score_densidad"
    ] * 0.15

    +

    centralidades[
        "score_conectividad"
    ] * 0.20

    +

    centralidades[
        "score_intermodalidad"
    ] * 0.15

    +

    centralidades[
        "score_alcance"
    ] * 0.10

    +

    centralidades[
        "score_integracion"
    ] * 0.10

)


centralidades[
    "indice_centralidad"
] = (
    centralidades[
        "indice_centralidad"
    ]
    .clip(
        0,
        100
    )
)


# ============================================================
# CATEGORÍA CENTRALIDAD
# ============================================================

def clasificar_centralidad(indice):

    if indice >= 85:
        return "CENTRALIDAD_CRITICA"

    if indice >= 70:
        return "CENTRALIDAD_ALTA"

    if indice >= 50:
        return "CENTRALIDAD_MEDIA"

    return "CENTRALIDAD_BAJA"


centralidades[
    "categoria_centralidad"
] = (
    centralidades[
        "indice_centralidad"
    ]
    .apply(
        clasificar_centralidad
    )
)


# ============================================================
# TIPO FUNCIONAL
# ============================================================

umbral_densidad_alta = (
    centralidades[
        "operaciones_por_km2"
    ]
    .quantile(
        0.75
    )
)


def clasificar_tipo(row):

    indice = row[
        "indice_centralidad"
    ]

    jurisdicciones = row[
        "cantidad_jurisdicciones"
    ]

    corredores = row[
        "cantidad_corredores"
    ]

    intermodalidad = row[
        "tipo_intermodalidad"
    ]

    densidad = row[
        "operaciones_por_km2"
    ]


    if (
        indice >= 85
        and jurisdicciones >= 8
    ):

        return "CENTRALIDAD_METROPOLITANA"


    if (
        intermodalidad == "INTERMODAL"
        and indice >= 70
    ):

        return "CENTRALIDAD_INTERMODAL"


    if (
        corredores >= 3
        and indice >= 65
    ):

        return "CENTRALIDAD_CONECTORA"


    if (
        jurisdicciones >= 3
        and indice >= 60
    ):

        return "CENTRALIDAD_TERRITORIAL"


    if (
        densidad >= umbral_densidad_alta
    ):

        return "CENTRALIDAD_DENSIDAD"


    return "CENTRALIDAD_LOCAL"


centralidades[
    "tipo_centralidad"
] = (
    centralidades
    .apply(
        clasificar_tipo,
        axis=1
    )
)


# ============================================================
# RANKING
# ============================================================

centralidades = (
    centralidades
    .sort_values(
        [
            "indice_centralidad",
            "operaciones",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(
        drop=True
    )
)


centralidades[
    "ranking_centralidad"
] = (
    centralidades.index
    + 1
)


# ============================================================
# PARTICIPACIÓN
# ============================================================

operaciones_total_nodos = (
    centralidades[
        "operaciones"
    ]
    .sum()
)


if operaciones_total_nodos > 0:

    centralidades[
        "pct_operaciones"
    ] = (

        centralidades[
            "operaciones"
        ]
        /
        operaciones_total_nodos
        * 100

    )

else:

    centralidades[
        "pct_operaciones"
    ] = 0


centralidades[
    "pct_operaciones_acumulado"
] = (
    centralidades[
        "pct_operaciones"
    ]
    .cumsum()
)


# ============================================================
# TOP 30
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TOP 30 CENTRALIDADES DE MOVILIDAD"
)

print(
    "=" * 70
)


columnas_top = [

    "ranking_centralidad",

    "nodo_id",

    "h3",

    "operaciones",

    "pct_operaciones",

    "pct_operaciones_acumulado",

    "superficie_km2",

    "operaciones_por_km2",

    "indice_centralidad",

    "categoria_centralidad",

    "tipo_centralidad",

    "tipo_intermodalidad",

    "alcance_territorial",

    "cantidad_corredores",

    "cantidad_clusters",

    "cantidad_jurisdicciones",

    "jurisdicciones",

    "hora_pico",

    "modo_dominante",

]


print(
    centralidades[
        columnas_top
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# DISTRIBUCIÓN
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "DISTRIBUCIÓN DE CENTRALIDADES"
)

print(
    "=" * 70
)


print(
    centralidades[
        "categoria_centralidad"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# TIPOS FUNCIONALES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TIPOS FUNCIONALES DE CENTRALIDAD"
)

print(
    "=" * 70
)


print(
    centralidades[
        "tipo_centralidad"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# INTERMODALIDAD
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "INTERMODALIDAD"
)

print(
    "=" * 70
)


print(
    centralidades[
        "tipo_intermodalidad"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# ALCANCE TERRITORIAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ALCANCE TERRITORIAL"
)

print(
    "=" * 70
)


print(
    centralidades[
        "alcance_territorial"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# HORAS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "HORAS PICO DE LAS CENTRALIDADES"
)

print(
    "=" * 70
)


print(
    centralidades[
        "hora_pico"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# MODOS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MODOS DOMINANTES"
)

print(
    "=" * 70
)


print(
    centralidades[
        "modo_dominante"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# CENTRALIDADES METROPOLITANAS
# ============================================================

metropolitanas = centralidades[
    centralidades[
        "tipo_centralidad"
    ]
    == "CENTRALIDAD_METROPOLITANA"
]


print(
    "\n"
    + "=" * 70
)

print(
    "CENTRALIDADES METROPOLITANAS"
)

print(
    "=" * 70
)


if metropolitanas.empty:

    print(
        "No se detectaron centralidades metropolitanas."
    )

else:

    print(
        metropolitanas[
            [
                "ranking_centralidad",
                "nodo_id",
                "operaciones",
                "indice_centralidad",
                "cantidad_corredores",
                "cantidad_jurisdicciones",
                "jurisdicciones",
                "hora_pico",
                "modo_dominante",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# CENTRALIDADES INTERMODALES
# ============================================================

intermodales = centralidades[
    centralidades[
        "tipo_centralidad"
    ]
    == "CENTRALIDAD_INTERMODAL"
]


print(
    "\n"
    + "=" * 70
)

print(
    "CENTRALIDADES INTERMODALES"
)

print(
    "=" * 70
)


if intermodales.empty:

    print(
        "No se detectaron centralidades intermodales "
        "con los criterios actuales."
    )

else:

    print(
        intermodales[
            [
                "ranking_centralidad",
                "nodo_id",
                "operaciones",
                "indice_centralidad",
                "cantidad_corredores",
                "cantidad_jurisdicciones",
                "jurisdicciones",
                "hora_pico",
                "modo_dominante",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# RESUMEN EJECUTIVO
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RESUMEN EJECUTIVO"
)

print(
    "=" * 70
)


print(
    f"\nNodos analizados: "
    f"{len(centralidades):,}"
)


print(
    f"Operaciones analizadas: "
    f"{operaciones_total_nodos:,.0f}"
)


print(
    f"Centralidades críticas: "
    f"{(
        centralidades[
            'categoria_centralidad'
        ]
        == 'CENTRALIDAD_CRITICA'
    ).sum():,}"
)


print(
    f"Centralidades altas: "
    f"{(
        centralidades[
            'categoria_centralidad'
        ]
        == 'CENTRALIDAD_ALTA'
    ).sum():,}"
)


print(
    f"Centralidades medias: "
    f"{(
        centralidades[
            'categoria_centralidad'
        ]
        == 'CENTRALIDAD_MEDIA'
    ).sum():,}"
)


print(
    f"Centralidades bajas: "
    f"{(
        centralidades[
            'categoria_centralidad'
        ]
        == 'CENTRALIDAD_BAJA'
    ).sum():,}"
)


# ============================================================
# CENTRALIDAD PRINCIPAL
# ============================================================

principal = (
    centralidades
    .iloc[0]
)


print(
    "\n"
    + "=" * 70
)

print(
    "CENTRALIDAD PRINCIPAL"
)

print(
    "=" * 70
)


print(
    f"Nodo: "
    f"{principal['nodo_id']}"
)


print(
    f"Ranking: "
    f"{principal['ranking_centralidad']}"
)


print(
    f"Operaciones: "
    f"{principal['operaciones']:,.0f}"
)


print(
    f"Participación: "
    f"{principal['pct_operaciones']:.2f}%"
)


print(
    f"Índice de centralidad: "
    f"{principal['indice_centralidad']:.2f}/100"
)


print(
    f"Categoría: "
    f"{principal['categoria_centralidad']}"
)


print(
    f"Tipo: "
    f"{principal['tipo_centralidad']}"
)


print(
    f"Intermodalidad: "
    f"{principal['tipo_intermodalidad']}"
)


print(
    f"Jurisdicciones: "
    f"{principal['cantidad_jurisdicciones']}"
)


print(
    f"Corredores: "
    f"{principal['cantidad_corredores']}"
)


print(
    f"Hora pico: "
    f"{principal['hora_pico']}"
)


print(
    f"Modo: "
    f"{principal['modo_dominante']}"
)


print(
    "\nJurisdicciones:"
)


print(
    principal["jurisdicciones"]
)


# ============================================================
# PREPARAR H3 → CENTRALIDAD
# ============================================================

print(
    "\nPreparando relación H3 → centralidad..."
)


mapa_centralidad = (
    centralidades[
        [
            "nodo_id",
            "ranking_centralidad",
            "indice_centralidad",
            "categoria_centralidad",
            "tipo_centralidad",
        ]
    ]
    .copy()
)


h3_centralidades = h3_geo.merge(
    mapa_centralidad,
    on="nodo_id",
    how="left",
)


# ============================================================
# INDICAR H3 SIN CENTRALIDAD
# ============================================================

h3_centralidades[
    "es_centralidad"
] = (
    h3_centralidades[
        "nodo_id"
    ]
    .notna()
)


# ============================================================
# ELIMINAR GEOMETRÍA DEL PARQUET
# ============================================================

h3_centralidades_salida = (
    h3_centralidades
    .drop(
        columns=[
            "geometry"
        ],
        errors="ignore"
    )
    .copy()
)


# ============================================================
# GUARDAR H3 → CENTRALIDAD
# ============================================================

print(
    "\nGuardando H3 → centralidad..."
)


h3_centralidades_salida.to_parquet(
    OUTPUT_H3_CENTRALIDADES,
    index=False,
)


# ============================================================
# GUARDAR CENTRALIDADES
# ============================================================

print(
    "Guardando centralidades..."
)


centralidades_salida = (
    centralidades
    .drop(
        columns=[
            "geometry"
        ],
        errors="ignore"
    )
    .copy()
)


centralidades_salida.to_parquet(
    OUTPUT_CENTRALIDADES,
    index=False,
)


# ============================================================
# RESUMEN JSON
# ============================================================

resumen = {

    "h3_analizados":
        int(len(h3_geo)),

    "h3_con_nodo":
        int(h3_con_nodo),

    "h3_con_cluster":
        int(h3_con_cluster),

    "h3_con_corredor":
        int(h3_con_corredor),

    "nodos_analizados":
        int(len(centralidades)),

    "operaciones_analizadas":
        float(operaciones_total_nodos),

    "centralidades_criticas":
        int(
            (
                centralidades[
                    "categoria_centralidad"
                ]
                == "CENTRALIDAD_CRITICA"
            ).sum()
        ),

    "centralidades_altas":
        int(
            (
                centralidades[
                    "categoria_centralidad"
                ]
                == "CENTRALIDAD_ALTA"
            ).sum()
        ),

    "centralidades_medias":
        int(
            (
                centralidades[
                    "categoria_centralidad"
                ]
                == "CENTRALIDAD_MEDIA"
            ).sum()
        ),

    "centralidades_bajas":
        int(
            (
                centralidades[
                    "categoria_centralidad"
                ]
                == "CENTRALIDAD_BAJA"
            ).sum()
        ),

    "tipos_centralidad":
        {
            str(k): int(v)
            for k, v
            in centralidades[
                "tipo_centralidad"
            ]
            .value_counts()
            .items()
        },

    "intermodalidad":
        {
            str(k): int(v)
            for k, v
            in centralidades[
                "tipo_intermodalidad"
            ]
            .value_counts()
            .items()
        },

    "alcance_territorial":
        {
            str(k): int(v)
            for k, v
            in centralidades[
                "alcance_territorial"
            ]
            .value_counts()
            .items()
        },

    "horas_pico":
        {
            str(k): int(v)
            for k, v
            in centralidades[
                "hora_pico"
            ]
            .value_counts()
            .sort_index()
            .items()
        },

    "modos_dominantes":
        {
            str(k): int(v)
            for k, v
            in centralidades[
                "modo_dominante"
            ]
            .value_counts()
            .items()
        },
}


# ============================================================
# INFORMACIÓN CENTRALIDAD PRINCIPAL
# ============================================================

resumen[
    "centralidad_principal"
] = {

    "nodo_id":
        int(
            principal[
                "nodo_id"
            ]
        ),

    "ranking":
        int(
            principal[
                "ranking_centralidad"
            ]
        ),

    "operaciones":
        float(
            principal[
                "operaciones"
            ]
        ),

    "pct_operaciones":
        float(
            principal[
                "pct_operaciones"
            ]
        ),

    "indice_centralidad":
        float(
            principal[
                "indice_centralidad"
            ]
        ),

    "categoria":
        principal[
            "categoria_centralidad"
        ],

    "tipo":
        principal[
            "tipo_centralidad"
        ],

    "intermodalidad":
        principal[
            "tipo_intermodalidad"
        ],

    "jurisdicciones":
        principal[
            "jurisdicciones"
        ],

    "corredores":
        int(
            principal[
                "cantidad_corredores"
            ]
        ),

    "hora_pico":
        safe_int(
            principal[
                "hora_pico"
            ]
        ),

    "modo":
        principal[
            "modo_dominante"
        ],
}


# ============================================================
# GUARDAR JSON
# ============================================================

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
# ARCHIVOS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ARCHIVOS GENERADOS"
)

print(
    "=" * 70
)


print(
    "\nCentralidades:"
)

print(
    OUTPUT_CENTRALIDADES
)


print(
    "\nH3 → centralidad:"
)

print(
    OUTPUT_H3_CENTRALIDADES
)


print(
    "\nResumen:"
)

print(
    OUTPUT_RESUMEN
)


# ============================================================
# FIN
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ANÁLISIS DE CENTRALIDADES FINALIZADO"
)

print(
    "=" * 70
)


print(
    f"Nodos analizados: "
    f"{len(centralidades):,}"
)


print(
    f"Centralidad principal: "
    f"Nodo {principal['nodo_id']}"
)


print(
    f"Índice principal: "
    f"{principal['indice_centralidad']:.2f}/100"
)


print(
    f"Operaciones analizadas: "
    f"{operaciones_total_nodos:,.0f}"
)
