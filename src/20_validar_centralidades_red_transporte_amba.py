from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.ops import unary_union


# ============================================================
# CONFIGURACIÓN
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

CENTRALIDADES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_validacion_centralidades.parquet"
)

RED_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "red_transporte_amba"
    / "red_vial_amba.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "validacion_red_transporte"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "sube_2025_centralidades_red_validacion.parquet"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "sube_2025_centralidades_red_resumen.json"
)

OUTPUT_MAPA = (
    OUTPUT_DIR
    / "01_mapa_centralidad_estructural.png"
)

OUTPUT_DISCREPANCIAS = (
    OUTPUT_DIR
    / "02_discrepancias_sube_red.png"
)

OUTPUT_JERARQUIA = (
    OUTPUT_DIR
    / "03_jerarquia_vial_nodos.png"
)

OUTPUT_DENSIDAD = (
    OUTPUT_DIR
    / "04_densidad_red.png"
)

OUTPUT_BOX = (
    OUTPUT_DIR
    / "05_distribucion_indicadores.png"
)

OUTPUT_SCATTER = (
    OUTPUT_DIR
    / "06_sube_vs_red.png"
)

OUTPUT_RANKING = (
    OUTPUT_DIR
    / "07_ranking_sube_vs_red.png"
)

OUTPUT_MATRIZ = (
    OUTPUT_DIR
    / "08_matriz_sube_red.png"
)

CRS_GEOGRAFICO = "EPSG:4326"

# Argentina Gauss-Krüger faja 5.
# Se utiliza para trabajar distancias y longitudes en metros.
CRS_METRICO = "EPSG:22185"

BUFFER_ANALISIS_M = 500


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def validar_archivo(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido:\n{path}"
        )


def convertir_numerico(df, columnas):
    for columna in columnas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )


def correlacion_pearson(a, b):
    datos = pd.DataFrame(
        {
            "a": pd.to_numeric(a, errors="coerce"),
            "b": pd.to_numeric(b, errors="coerce"),
        }
    ).dropna()

    if len(datos) < 2:
        return None

    if datos["a"].nunique() <= 1:
        return None

    if datos["b"].nunique() <= 1:
        return None

    valor = datos["a"].corr(
        datos["b"],
        method="pearson"
    )

    if pd.isna(valor):
        return None

    return float(valor)


def correlacion_spearman(a, b):
    datos = pd.DataFrame(
        {
            "a": pd.to_numeric(a, errors="coerce"),
            "b": pd.to_numeric(b, errors="coerce"),
        }
    ).dropna()

    if len(datos) < 2:
        return None

    if datos["a"].nunique() <= 1:
        return None

    if datos["b"].nunique() <= 1:
        return None

    valor = datos["a"].corr(
        datos["b"],
        method="spearman"
    )

    if pd.isna(valor):
        return None

    return float(valor)


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def safe_string(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def normalizar_highway(valor):
    if valor is None:
        return "unknown"

    texto = str(valor).strip().lower()

    return texto


def jerarquia_numerica(valor):
    mapping = {
        "motorway": 5,
        "trunk": 4,
        "primary": 3,
        "secondary": 2,
        "tertiary": 1,
    }

    return mapping.get(
        normalizar_highway(valor),
        0
    )


def clasificar_matriz_sub_red(row):
    sube = safe_float(
        row.get("indice_centralidad")
    )

    red = safe_float(
        row.get("score_estructural_red")
    )

    if sube is None or red is None:
        return "SIN_DATOS"

    if sube >= 50 and red >= 50:
        return "ALTA_SUBE_ALTA_RED"

    if sube >= 50 and red < 50:
        return "ALTA_SUBE_BAJA_RED"

    if sube < 50 and red >= 50:
        return "BAJA_SUBE_ALTA_RED"

    return "BAJA_SUBE_BAJA_RED"


def clasificar_discrepancia(row):
    diferencia = safe_float(
        row.get("diferencia_ranking_sube_red")
    )

    if diferencia is None:
        return "SIN_DATOS"

    if diferencia >= 50:
        return "CENTRALIDAD_SUBE_MUCHO_MAYOR_QUE_RED"

    if diferencia >= 20:
        return "CENTRALIDAD_SUBE_MAYOR_QUE_RED"

    if diferencia <= -50:
        return "CENTRALIDAD_RED_MUCHO_MAYOR_QUE_SUBE"

    if diferencia <= -20:
        return "CENTRALIDAD_RED_MAYOR_QUE_SUBE"

    return "ALINEADO"


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print(
    "VALIDACIÓN DE CENTRALIDADES CONTRA RED DE TRANSPORTE AMBA"
)
print("=" * 70)


# ============================================================
# VALIDAR ARCHIVOS
# ============================================================

print("\nValidando archivos de entrada...")

validar_archivo(
    CENTRALIDADES_PATH
)

validar_archivo(
    RED_PATH
)

print(
    "Archivos encontrados correctamente."
)


# ============================================================
# CARGAR CENTRALIDADES
# ============================================================

print("\nCargando centralidades SUBE...")

centralidades = gpd.read_parquet(
    CENTRALIDADES_PATH
)

print(
    f"Centralidades cargadas: "
    f"{len(centralidades):,}"
)

print("Columnas:")
print(
    centralidades.columns.tolist()
)


# ============================================================
# VALIDAR GEOMETRÍAS CENTRALIDADES
# ============================================================

print(
    "\nValidando geometrías de centralidades..."
)

if centralidades.crs is None:
    centralidades = centralidades.set_crs(
        CRS_GEOGRAFICO
    )

print(
    f"CRS centralidades: "
    f"{centralidades.crs}"
)

geometrias_validas = (
    centralidades.geometry.notna()
    &
    ~centralidades.geometry.is_empty
)

print(
    f"Geometrías válidas: "
    f"{int(geometrias_validas.sum()):,}"
)

print(
    f"Geometrías inválidas: "
    f"{int((~geometrias_validas).sum()):,}"
)

centralidades = centralidades[
    geometrias_validas
].copy()

if centralidades.empty:
    raise ValueError(
        "No existen geometrías válidas de centralidades."
    )


# ============================================================
# CARGAR RED VIAL
# ============================================================

print("\nCargando red vial AMBA...")

red = gpd.read_parquet(
    RED_PATH
)

print(
    f"Segmentos cargados: "
    f"{len(red):,}"
)

print("Columnas:")
print(
    red.columns.tolist()
)


# ============================================================
# VALIDAR GEOMETRÍAS RED
# ============================================================

print(
    "\nValidando geometrías de la red..."
)

if red.crs is None:
    red = red.set_crs(
        CRS_GEOGRAFICO
    )

print(
    f"CRS red vial: "
    f"{red.crs}"
)

geometrias_red_validas = (
    red.geometry.notna()
    &
    ~red.geometry.is_empty
)

print(
    f"Geometrías válidas: "
    f"{int(geometrias_red_validas.sum()):,}"
)

print(
    f"Geometrías inválidas: "
    f"{int((~geometrias_red_validas).sum()):,}"
)

red = red[
    geometrias_red_validas
].copy()


# ============================================================
# NORMALIZAR RED
# ============================================================

print(
    "\nNormalizando atributos de red..."
)

if "highway" not in red.columns:
    raise ValueError(
        "La red no contiene la columna 'highway'."
    )

red["highway"] = (
    red["highway"]
    .astype(str)
    .str.strip()
    .str.lower()
)

jerarquias_validas = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
]

red = red[
    red["highway"].isin(
        jerarquias_validas
    )
].copy()

print(
    f"Segmentos después de filtrar "
    f"jerarquías: {len(red):,}"
)


# ============================================================
# TRANSFORMAR A CRS MÉTRICO
# ============================================================

print(
    "\nTransformando a CRS métrico..."
)

centralidades_m = centralidades.to_crs(
    CRS_METRICO
)

red_m = red.to_crs(
    CRS_METRICO
)

print(
    f"CRS métrico utilizado: "
    f"{CRS_METRICO}"
)


# ============================================================
# LONGITUDES DE RED
# ============================================================

print(
    "\nCalculando longitudes de segmentos..."
)

red_m["longitud_m"] = (
    red_m.geometry.length
)

red_m["longitud_km"] = (
    red_m["longitud_m"]
    / 1000.0
)

longitud_total_red = float(
    red_m["longitud_km"].sum()
)

print(
    f"Longitud total de red: "
    f"{longitud_total_red:,.2f} km"
)


# ============================================================
# PREPARAR RED ESPACIAL
# ============================================================

print(
    "\nPreparando índice espacial..."
)

# GeoPandas utiliza STRtree internamente
# para los joins espaciales.

red_m = red_m.reset_index(
    drop=True
)


# ============================================================
# ANÁLISIS ESPACIAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ANÁLISIS ESPACIAL DE LOS NODOS"
)

print(
    "=" * 70
)

resultados = []


cantidad_nodos = len(
    centralidades_m
)


for posicion, (_, nodo) in enumerate(
    centralidades_m.iterrows(),
    start=1
):

    nodo_id = nodo.get(
        "nodo_id"
    )

    if (
        posicion == 1
        or posicion % 10 == 0
        or posicion == cantidad_nodos
    ):
        print(
            f"Procesando nodo "
            f"{posicion}/{cantidad_nodos} "
            f"(nodo {nodo_id})..."
        )

    geometria_nodo = nodo.geometry

    # --------------------------------------------------------
    # PUNTO REPRESENTATIVO
    # --------------------------------------------------------

    try:
        punto = (
            geometria_nodo
            .representative_point()
        )
    except Exception:
        punto = (
            geometria_nodo
            .centroid
        )

    # --------------------------------------------------------
    # DISTANCIA A LA RED
    # --------------------------------------------------------

    distancias = red_m.geometry.distance(
        punto
    )

    if len(distancias) > 0:
        distancia_red_m = float(
            distancias.min()
        )
    else:
        distancia_red_m = np.nan

    # --------------------------------------------------------
    # RED EN BUFFER DE 500 M
    # --------------------------------------------------------

    buffer = geometria_nodo.buffer(
        BUFFER_ANALISIS_M
    )

    try:
        candidatos = red_m[
            red_m.geometry.intersects(
                buffer
            )
        ].copy()
    except Exception:
        candidatos = red_m.copy()

    # --------------------------------------------------------
    # INDICADORES
    # --------------------------------------------------------

    if candidatos.empty:

        longitud_red_500m_km = 0.0

        cantidad_segmentos = 0

        cantidad_jerarquias = 0

        jerarquias = ""

        longitud_motorway = 0.0
        longitud_trunk = 0.0
        longitud_primary = 0.0
        longitud_secondary = 0.0
        longitud_tertiary = 0.0

    else:

        cantidad_segmentos = int(
            len(candidatos)
        )

        # ----------------------------------------------------
        # IMPORTANTE:
        # usamos la longitud completa de los segmentos
        # seleccionados. Esto evita obtener 0 km cuando
        # el segmento intersecta el buffer.
        # ----------------------------------------------------

        longitud_red_500m_km = float(
            candidatos["longitud_km"].sum()
        )

        jerarquias_presentes = sorted(
            candidatos["highway"]
            .dropna()
            .unique()
            .tolist()
        )

        cantidad_jerarquias = int(
            len(jerarquias_presentes)
        )

        jerarquias = "|".join(
            jerarquias_presentes
        )

        def longitud_por_jerarquia(
            nombre
        ):
            return float(
                candidatos.loc[
                    candidatos["highway"] == nombre,
                    "longitud_km"
                ].sum()
            )

        longitud_motorway = (
            longitud_por_jerarquia(
                "motorway"
            )
        )

        longitud_trunk = (
            longitud_por_jerarquia(
                "trunk"
            )
        )

        longitud_primary = (
            longitud_por_jerarquia(
                "primary"
            )
        )

        longitud_secondary = (
            longitud_por_jerarquia(
                "secondary"
            )
        )

        longitud_tertiary = (
            longitud_por_jerarquia(
                "tertiary"
            )
        )

    # --------------------------------------------------------
    # HIERARQUÍA VIAL PREDOMINANTE
    # --------------------------------------------------------

    if candidatos.empty:

        jerarquia_dominante = (
            "SIN_RED"
        )

    else:

        longitudes_jerarquia = (
            candidatos
            .groupby("highway")[
                "longitud_km"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if len(
            longitudes_jerarquia
        ) > 0:

            jerarquia_dominante = (
                longitudes_jerarquia
                .index[0]
            )

        else:

            jerarquia_dominante = (
                "SIN_RED"
            )

    # --------------------------------------------------------
    # SCORE DE JERARQUÍA
    # --------------------------------------------------------

    score_jerarquia = (
        jerarquia_numerica(
            jerarquia_dominante
        )
    )

    # --------------------------------------------------------
    # LONGITUD NORMALIZADA
    # --------------------------------------------------------

    resultados.append(
        {
            "nodo_id": nodo_id,

            "distancia_red_m":
                distancia_red_m,

            "cantidad_segmentos_500m":
                cantidad_segmentos,

            "longitud_red_500m_km":
                longitud_red_500m_km,

            "cantidad_jerarquias_viales":
                cantidad_jerarquias,

            "jerarquias_viales":
                jerarquias,

            "jerarquia_vial_dominante":
                jerarquia_dominante,

            "score_jerarquia_vial":
                score_jerarquia,

            "longitud_motorway_km":
                longitud_motorway,

            "longitud_trunk_km":
                longitud_trunk,

            "longitud_primary_km":
                longitud_primary,

            "longitud_secondary_km":
                longitud_secondary,

            "longitud_tertiary_km":
                longitud_tertiary,
        }
    )


# ============================================================
# CONSTRUIR DATAFRAME ESPACIAL
# ============================================================

estructura = pd.DataFrame(
    resultados
)

estructura["nodo_id"] = pd.to_numeric(
    estructura["nodo_id"],
    errors="coerce"
)

centralidades_m["nodo_id"] = pd.to_numeric(
    centralidades_m["nodo_id"],
    errors="coerce"
)

validacion = centralidades_m.merge(
    estructura,
    on="nodo_id",
    how="left",
    suffixes=("", "_red")
)


# ============================================================
# NORMALIZAR INDICADORES
# ============================================================

print(
    "\nNormalizando indicadores estructurales..."
)


def normalizar_0_100(serie):

    valores = pd.to_numeric(
        serie,
        errors="coerce"
    )

    if valores.notna().sum() == 0:
        return pd.Series(
            0.0,
            index=serie.index
        )

    minimo = valores.min()
    maximo = valores.max()

    if (
        pd.isna(minimo)
        or pd.isna(maximo)
        or maximo == minimo
    ):
        return pd.Series(
            50.0,
            index=serie.index
        )

    resultado = (
        (valores - minimo)
        /
        (maximo - minimo)
        * 100.0
    )

    return resultado.fillna(0.0)


validacion[
    "score_distancia_red"
] = normalizar_0_100(
    -validacion[
        "distancia_red_m"
    ]
)

validacion[
    "score_longitud_red"
] = normalizar_0_100(
    validacion[
        "longitud_red_500m_km"
    ]
)

validacion[
    "score_jerarquia_red"
] = normalizar_0_100(
    validacion[
        "score_jerarquia_vial"
    ]
)

validacion[
    "score_segmentos_red"
] = normalizar_0_100(
    validacion[
        "cantidad_segmentos_500m"
    ]
)

validacion[
    "score_diversidad_red"
] = normalizar_0_100(
    validacion[
        "cantidad_jerarquias_viales"
    ]
)


# ============================================================
# SCORE ESTRUCTURAL
# ============================================================

# Pesos:
#
# 30% longitud de red
# 25% diversidad de jerarquías
# 20% jerarquía vial
# 15% cantidad de segmentos
# 10% proximidad
#
# La ponderación busca evitar que la mera proximidad
# domine el indicador.

validacion[
    "score_estructural_red"
] = (

    validacion[
        "score_longitud_red"
    ] * 0.30

    +

    validacion[
        "score_diversidad_red"
    ] * 0.25

    +

    validacion[
        "score_jerarquia_red"
    ] * 0.20

    +

    validacion[
        "score_segmentos_red"
    ] * 0.15

    +

    validacion[
        "score_distancia_red"
    ] * 0.10

)


validacion[
    "score_estructural_red"
] = (
    validacion[
        "score_estructural_red"
    ]
    .clip(0, 100)
)


# ============================================================
# RANKINGS
# ============================================================

print(
    "\nCalculando ranking estructural..."
)

validacion[
    "ranking_estructural_red"
] = (
    validacion[
        "score_estructural_red"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


if "ranking_centralidad_validacion" not in validacion.columns:

    validacion[
        "ranking_centralidad_validacion"
    ] = (
        validacion[
            "indice_centralidad"
        ]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )


if "ranking_operaciones_validacion" not in validacion.columns:

    validacion[
        "ranking_operaciones_validacion"
    ] = (
        validacion[
            "operaciones"
        ]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )


validacion[
    "diferencia_ranking_sube_red"
] = (
    validacion[
        "ranking_estructural_red"
    ]
    -
    validacion[
        "ranking_centralidad_validacion"
    ]
)

validacion[
    "diferencia_ranking_sube_red_abs"
] = (
    validacion[
        "diferencia_ranking_sube_red"
    ]
    .abs()
)


# ============================================================
# MATRIZ
# ============================================================

validacion[
    "matriz_centralidad_estructural"
] = (
    validacion.apply(
        clasificar_matriz_sub_red,
        axis=1
    )
)

validacion[
    "tipo_discrepancia_sube_red"
] = (
    validacion.apply(
        clasificar_discrepancia,
        axis=1
    )
)


# ============================================================
# RANKING ESTRUCTURAL NORMALIZADO
# ============================================================

validacion[
    "percentil_estructural_red"
] = (
    100
    *
    (
        len(validacion)
        -
        validacion[
            "ranking_estructural_red"
        ]
        + 1
    )
    /
    len(validacion)
)


# ============================================================
# CORRELACIONES
# ============================================================

print(
    "\nCalculando correlaciones..."
)

correlaciones = {}


variables_red = {

    "score_estructural_red": (
        "Score estructural red"
    ),

    "longitud_red_500m_km": (
        "Longitud red 500 m"
    ),

    "cantidad_jerarquias_viales": (
        "Cantidad jerarquías viales"
    ),

    "score_jerarquia_red": (
        "Score jerarquía vial"
    ),

    "score_diversidad_red": (
        "Score diversidad red"
    ),

    "score_segmentos_red": (
        "Score segmentos red"
    ),
}


for columna, nombre in variables_red.items():

    correlaciones[columna] = {

        "nombre":
            nombre,

        "pearson":
            correlacion_pearson(
                validacion[
                    "indice_centralidad"
                ],
                validacion[
                    columna
                ]
            ),

        "spearman":
            correlacion_spearman(
                validacion[
                    "indice_centralidad"
                ],
                validacion[
                    columna
                ]
            ),
    }


# ============================================================
# RESUMEN RED
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RESUMEN RED DE TRANSPORTE"
)

print(
    "=" * 70
)

print(
    f"\nNodos analizados: "
    f"{len(validacion):,}"
)

print(
    f"Segmentos de red: "
    f"{len(red_m):,}"
)

print(
    f"Longitud red: "
    f"{longitud_total_red:,.2f} km"
)

print(
    f"\nDistancia mediana a red: "
    f"{validacion['distancia_red_m'].median():,.2f} m"
)

print(
    f"Distancia máxima a red: "
    f"{validacion['distancia_red_m'].max():,.2f} m"
)


# ============================================================
# TOP 20
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TOP 20 NODOS POR CENTRALIDAD ESTRUCTURAL"
)

print(
    "=" * 70
)

top20 = (
    validacion
    .sort_values(
        "score_estructural_red",
        ascending=False
    )
    .head(20)
)

columnas_top = [
    "nodo_id",
    "score_estructural_red",
    "ranking_estructural_red",
    "indice_centralidad",
    "ranking_centralidad_validacion",
    "operaciones",
    "distancia_red_m",
    "longitud_red_500m_km",
    "cantidad_jerarquias_viales",
    "matriz_centralidad_estructural",
]

print(
    top20[
        columnas_top
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# MAYORES DISCREPANCIAS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MAYORES DISCREPANCIAS SUBE VS RED"
)

print(
    "=" * 70
)

top_discrepancias = (
    validacion
    .sort_values(
        "diferencia_ranking_sube_red_abs",
        ascending=False
    )
    .head(20)
)

columnas_discrepancia = [
    "nodo_id",
    "indice_centralidad",
    "ranking_centralidad_validacion",
    "score_estructural_red",
    "ranking_estructural_red",
    "diferencia_ranking_sube_red",
    "operaciones",
    "distancia_red_m",
    "matriz_centralidad_estructural",
]

print(
    top_discrepancias[
        columnas_discrepancia
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# GRÁFICO 1
# MAPA CENTRALIDAD ESTRUCTURAL
# ============================================================

print(
    "\nGenerando mapa de centralidad estructural..."
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

red_m.plot(
    ax=ax,
    linewidth=0.25,
    alpha=0.35
)

validacion.plot(
    ax=ax,
    column="score_estructural_red",
    legend=True,
    markersize=30,
    alpha=0.85
)

ax.set_title(
    "Centralidad estructural de nodos SUBE vs red vial AMBA"
)

ax.set_axis_off()

plt.tight_layout()

plt.savefig(
    OUTPUT_MAPA,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRÁFICO 2
# DISCREPANCIAS
# ============================================================

print(
    "Generando gráfico de discrepancias..."
)

grafico_discrepancias = (
    validacion
    .sort_values(
        "diferencia_ranking_sube_red"
    )
    .tail(30)
)

fig, ax = plt.subplots(
    figsize=(14, 8)
)

ax.bar(
    grafico_discrepancias[
        "nodo_id"
    ].astype(str),
    grafico_discrepancias[
        "diferencia_ranking_sube_red"
    ]
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "Discrepancia de ranking: SUBE vs red"
)

ax.set_xlabel(
    "Nodo"
)

ax.set_ylabel(
    "Ranking red - ranking SUBE"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DISCREPANCIAS,
    dpi=200
)

plt.close()


# ============================================================
# GRÁFICO 3
# JERARQUÍA VIAL
# ============================================================

print(
    "Generando gráfico de jerarquía vial..."
)

jerarquia_counts = (
    validacion[
        "jerarquia_vial_dominante"
    ]
    .value_counts()
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.bar(
    jerarquia_counts.index.astype(str),
    jerarquia_counts.values
)

ax.set_title(
    "Jerarquía vial dominante por nodo"
)

ax.set_xlabel(
    "Jerarquía vial"
)

ax.set_ylabel(
    "Cantidad de nodos"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_JERARQUIA,
    dpi=200
)

plt.close()


# ============================================================
# GRÁFICO 4
# DENSIDAD
# ============================================================

print(
    "Generando gráfico de densidad..."
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.scatter(
    validacion[
        "operaciones"
    ],
    validacion[
        "longitud_red_500m_km"
    ],
    alpha=0.7
)

ax.set_title(
    "Demanda SUBE vs longitud de red próxima"
)

ax.set_xlabel(
    "Operaciones SUBE"
)

ax.set_ylabel(
    "Longitud red en 500 m (km)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DENSIDAD,
    dpi=200
)

plt.close()


# ============================================================
# GRÁFICO 5
# BOXPLOT
#
# CORRECCIÓN IMPORTANTE:
# Matplotlib moderno utiliza tick_labels=
# en lugar de labels=
# ============================================================

print(
    "Generando distribución de indicadores..."
)

datos_boxplot = [

    validacion[
        "indice_centralidad"
    ]
    .dropna()
    .values,

    validacion[
        "score_estructural_red"
    ]
    .dropna()
    .values,

    validacion[
        "score_jerarquia_red"
    ]
    .dropna()
    .values,

    validacion[
        "score_diversidad_red"
    ]
    .dropna()
    .values,
]

fig, ax = plt.subplots(
    figsize=(12, 8)
)

ax.boxplot(
    datos_boxplot,

    # ========================================================
    # CORRECCIÓN DEL ERROR:
    #
    # ANTES:
    # labels=[...]
    #
    # AHORA:
    # tick_labels=[...]
    # ========================================================

    tick_labels=[
        "Centralidad SUBE",
        "Estructural red",
        "Jerarquía vial",
        "Diversidad red",
    ]
)

ax.set_title(
    "Distribución de indicadores de centralidad"
)

ax.set_ylabel(
    "Valor"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_BOX,
    dpi=200
)

plt.close()


# ============================================================
# GRÁFICO 6
# SUBE VS RED
# ============================================================

print(
    "Generando gráfico SUBE vs red..."
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.scatter(
    validacion[
        "indice_centralidad"
    ],
    validacion[
        "score_estructural_red"
    ],
    alpha=0.75
)

for _, fila in validacion.iterrows():

    try:
        ax.annotate(
            str(
                int(
                    fila[
                        "nodo_id"
                    ]
                )
            ),
            (
                fila[
                    "indice_centralidad"
                ],
                fila[
                    "score_estructural_red"
                ]
            ),
            fontsize=7
        )
    except Exception:
        pass

ax.set_title(
    "Centralidad SUBE vs centralidad estructural de red"
)

ax.set_xlabel(
    "Índice centralidad SUBE"
)

ax.set_ylabel(
    "Score estructural red"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_SCATTER,
    dpi=200
)

plt.close()


# ============================================================
# GRÁFICO 7
# RANKINGS
# ============================================================

print(
    "Generando comparación de rankings..."
)

fig, ax = plt.subplots(
    figsize=(10, 10)
)

ax.scatter(
    validacion[
        "ranking_centralidad_validacion"
    ],
    validacion[
        "ranking_estructural_red"
    ],
    alpha=0.7
)

limite = max(
    len(validacion),
    int(
        validacion[
            "ranking_centralidad_validacion"
        ].max()
    ),
    int(
        validacion[
            "ranking_estructural_red"
        ].max()
    )
)

ax.plot(
    [1, limite],
    [1, limite],
    linestyle="--"
)

ax.set_title(
    "Ranking centralidad SUBE vs ranking estructural red"
)

ax.set_xlabel(
    "Ranking centralidad SUBE"
)

ax.set_ylabel(
    "Ranking estructural red"
)

ax.invert_xaxis()
ax.invert_yaxis()

plt.tight_layout()

plt.savefig(
    OUTPUT_RANKING,
    dpi=200
)

plt.close()


# ============================================================
# GRÁFICO 8
# MATRIZ SUBE / RED
# ============================================================

print(
    "Generando matriz SUBE vs red..."
)

matriz_counts = (
    validacion[
        "matriz_centralidad_estructural"
    ]
    .value_counts()
)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.bar(
    matriz_counts.index.astype(str),
    matriz_counts.values
)

ax.set_title(
    "Matriz de centralidad SUBE vs red vial"
)

ax.set_xlabel(
    "Clasificación"
)

ax.set_ylabel(
    "Cantidad de nodos"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_MATRIZ,
    dpi=200
)

plt.close()


# ============================================================
# DISTRIBUCIONES
# ============================================================

distribucion_matriz = (
    validacion[
        "matriz_centralidad_estructural"
    ]
    .value_counts()
    .to_dict()
)

distribucion_discrepancias = (
    validacion[
        "tipo_discrepancia_sube_red"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# NODO PRINCIPAL SUBE
# ============================================================

principal_sube = (
    validacion
    .sort_values(
        "indice_centralidad",
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# NODO PRINCIPAL RED
# ============================================================

principal_red = (
    validacion
    .sort_values(
        "score_estructural_red",
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# MAYOR DISCREPANCIA
# ============================================================

mayor_discrepancia = (
    validacion
    .sort_values(
        "diferencia_ranking_sube_red_abs",
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# CONCENTRACIÓN
# ============================================================

operaciones_ordenadas = (
    validacion[
        "operaciones"
    ]
    .sort_values(
        ascending=False
    )
)

operaciones_total = float(
    operaciones_ordenadas.sum()
)

concentracion = {}

for cantidad in [
    1,
    5,
    10,
    20,
    30,
    50,
]:

    if operaciones_total > 0:

        acumulado = float(
            operaciones_ordenadas
            .head(cantidad)
            .sum()
        )

        porcentaje = (
            acumulado
            /
            operaciones_total
            *
            100
        )

    else:

        porcentaje = 0.0

    concentracion[
        f"top_{cantidad}"
    ] = porcentaje


# ============================================================
# CORRELACIONES EN CONSOLA
# ============================================================

print(
    "\nCorrelaciones:"
)

for columna, datos in correlaciones.items():

    print(
        f"  {datos['nombre']}: "
        f"Pearson={datos['pearson']}, "
        f"Spearman={datos['spearman']}"
    )


# ============================================================
# DISTRIBUCIÓN MATRIZ
# ============================================================

print(
    "\nMatriz SUBE-red:"
)

for clave, valor in (
    distribucion_matriz.items()
):

    print(
        f"  {clave}: "
        f"{valor:,}"
    )


# ============================================================
# INTERPRETACIÓN NODO PRINCIPAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "INTERPRETACIÓN DE LOS NODOS PRINCIPALES"
)

print(
    "=" * 70
)

print(
    f"\nNodo principal SUBE: "
    f"{int(principal_sube['nodo_id'])}"
)

print(
    f"  Índice centralidad: "
    f"{principal_sube['indice_centralidad']:.2f}"
)

print(
    f"  Ranking SUBE: "
    f"{int(principal_sube['ranking_centralidad_validacion'])}"
)

print(
    f"  Operaciones: "
    f"{principal_sube['operaciones']:,.0f}"
)

print(
    f"\nNodo principal red: "
    f"{int(principal_red['nodo_id'])}"
)

print(
    f"  Score estructural: "
    f"{principal_red['score_estructural_red']:.2f}"
)

print(
    f"  Ranking red: "
    f"{int(principal_red['ranking_estructural_red'])}"
)

print(
    f"  Longitud red 500 m: "
    f"{principal_red['longitud_red_500m_km']:.2f} km"
)

print(
    f"\nNodo con mayor discrepancia: "
    f"{int(mayor_discrepancia['nodo_id'])}"
)

print(
    f"  Diferencia absoluta: "
    f"{int(mayor_discrepancia['diferencia_ranking_sube_red_abs'])}"
)


# ============================================================
# RESUMEN JSON
# ============================================================

resumen = {

    "fuentes": {

        "centralidades":
            str(CENTRALIDADES_PATH),

        "red":
            str(RED_PATH),
    },

    "analisis": {

        "nodos":
            int(len(validacion)),

        "segmentos_red":
            int(len(red_m)),

        "longitud_red_km":
            float(longitud_total_red),

        "distancia_mediana_red_m":
            safe_float(
                validacion[
                    "distancia_red_m"
                ].median()
            ),

        "distancia_maxima_red_m":
            safe_float(
                validacion[
                    "distancia_red_m"
                ].max()
            ),
    },

    "nodo_principal_sube": {

        "nodo_id":
            safe_int(
                principal_sube[
                    "nodo_id"
                ]
            ),

        "indice_centralidad":
            safe_float(
                principal_sube[
                    "indice_centralidad"
                ]
            ),

        "ranking":
            safe_int(
                principal_sube[
                    "ranking_centralidad_validacion"
                ]
            ),

        "operaciones":
            safe_float(
                principal_sube[
                    "operaciones"
                ]
            ),
    },

    "nodo_principal_red": {

        "nodo_id":
            safe_int(
                principal_red[
                    "nodo_id"
                ]
            ),

        "score_estructural":
            safe_float(
                principal_red[
                    "score_estructural_red"
                ]
            ),

        "ranking":
            safe_int(
                principal_red[
                    "ranking_estructural_red"
                ]
            ),

        "longitud_red_500m_km":
            safe_float(
                principal_red[
                    "longitud_red_500m_km"
                ]
            ),

        "jerarquia_dominante":
            safe_string(
                principal_red[
                    "jerarquia_vial_dominante"
                ]
            ),
    },

    "mayor_discrepancia": {

        "nodo_id":
            safe_int(
                mayor_discrepancia[
                    "nodo_id"
                ]
            ),

        "diferencia_ranking":
            safe_int(
                mayor_discrepancia[
                    "diferencia_ranking_sube_red"
                ]
            ),

        "diferencia_absoluta":
            safe_int(
                mayor_discrepancia[
                    "diferencia_ranking_sube_red_abs"
                ]
            ),

        "tipo":
            safe_string(
                mayor_discrepancia[
                    "tipo_discrepancia_sube_red"
                ]
            ),
    },

    "correlaciones":
        correlaciones,

    "concentracion":
        concentracion,

    "distribucion_matriz":
        {
            str(k): int(v)
            for k, v
            in distribucion_matriz.items()
        },

    "distribucion_discrepancias":
        {
            str(k): int(v)
            for k, v
            in distribucion_discrepancias.items()
        },

    "archivos_generados": {

        "parquet":
            str(OUTPUT_PARQUET),

        "mapa":
            str(OUTPUT_MAPA),

        "discrepancias":
            str(OUTPUT_DISCREPANCIAS),

        "jerarquia":
            str(OUTPUT_JERARQUIA),

        "densidad":
            str(OUTPUT_DENSIDAD),

        "boxplot":
            str(OUTPUT_BOX),

        "scatter":
            str(OUTPUT_SCATTER),

        "ranking":
            str(OUTPUT_RANKING),

        "matriz":
            str(OUTPUT_MATRIZ),
    },
}


# ============================================================
# GUARDAR GEOPARQUET
# ============================================================

print(
    "\nGuardando validación..."
)

# Volvemos a EPSG:4326 para almacenar
# el resultado geoespacial final.

validacion_salida = (
    validacion
    .to_crs(CRS_GEOGRAFICO)
)

validacion_salida.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

print(
    "Validación guardada:"
)

print(
    OUTPUT_PARQUET
)


# ============================================================
# GUARDAR JSON
# ============================================================

print(
    "\nGuardando resumen..."
)

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        resumen,
        archivo,
        indent=2,
        ensure_ascii=False
    )

print(
    "Resumen guardado:"
)

print(
    OUTPUT_JSON
)


# ============================================================
# ARCHIVOS GENERADOS
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
    f"\nParquet:"
)

print(
    OUTPUT_PARQUET
)

print(
    f"\nResumen:"
)

print(
    OUTPUT_JSON
)

print(
    "\nGráficos:"
)

for path in [

    OUTPUT_MAPA,

    OUTPUT_DISCREPANCIAS,

    OUTPUT_JERARQUIA,

    OUTPUT_DENSIDAD,

    OUTPUT_BOX,

    OUTPUT_SCATTER,

    OUTPUT_RANKING,

    OUTPUT_MATRIZ,

]:

    print(
        f"  {path}"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VALIDACIÓN DE CENTRALIDADES VS RED FINALIZADA"
)

print(
    "=" * 70
)

print(
    f"\nNodos analizados: "
    f"{len(validacion):,}"
)

print(
    f"Segmentos red: "
    f"{len(red_m):,}"
)

print(
    f"Longitud red: "
    f"{longitud_total_red:,.2f} km"
)

print(
    f"Centralidad SUBE principal: "
    f"Nodo {int(principal_sube['nodo_id'])}"
)

print(
    f"Centralidad red principal: "
    f"Nodo {int(principal_red['nodo_id'])}"
)

print(
    f"Mayor discrepancia: "
    f"Nodo {int(mayor_discrepancia['nodo_id'])}"
)

print(
    "\nSiguiente etapa:"
)

print(
    "Cruzar las centralidades estructurales con "
    "la infraestructura intermodal, estaciones, "
    "terminales, ferrocarriles y corredores de "
    "transporte público."
)

print()