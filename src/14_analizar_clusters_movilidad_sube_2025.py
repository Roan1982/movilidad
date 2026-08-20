from pathlib import Path
from collections import defaultdict, deque
import json

import pandas as pd
import geopandas as gpd
import h3


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_territorio.parquet"
)

OUTPUT_CLUSTERS = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_clusters_movilidad.parquet"
)

OUTPUT_H3 = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_clusters.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_clusters_resumen.json"
)


# ============================================================
# PARÁMETROS ANALÍTICOS
# ============================================================

# Categorías consideradas suficientemente importantes
# para formar clusters.
CATEGORIAS_CLUSTER = {
    "HOTSPOT_EXTREMO",
    "HOTSPOT_ALTO",
    "DEMANDA_ALTA",
}

# Mínimo de H3 para considerar un cluster territorial.
MIN_H3_CLUSTER = 2

# Operaciones mínimas acumuladas.
MIN_OPERACIONES_CLUSTER = 5000

# H3 por encima de este umbral pueden iniciar/participar
# en clusters independientemente de la categoría.
#
# Se calcula dinámicamente sobre el dataset.
PERCENTIL_OPERACIONES = 0.75


# ============================================================
# FUNCIONES
# ============================================================

def convertir_h3(valor):
    """
    Normaliza el identificador H3 a string.
    """
    if pd.isna(valor):
        return None

    return str(valor)


def obtener_vecinos(h3_id):
    """
    Obtiene los vecinos directos del H3 utilizando la
    librería oficial h3.
    """

    try:
        return set(
            h3.grid_disk(
                h3_id,
                1
            )
        )

    except Exception:
        return set()


def combinar_modos(series):
    """
    Determina el modo dominante del cluster.
    """

    valores = (
        series
        .dropna()
        .astype(str)
    )

    if len(valores) == 0:
        return None

    return valores.mode().iloc[0]


def combinar_horas(series):
    """
    Determina la hora pico dominante del cluster.
    """

    valores = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(valores) == 0:
        return None

    return int(
        valores.mode().iloc[0]
    )


def calcular_bbox(geometrias):
    """
    Calcula bounding box del cluster.
    """

    union = geometrias.union_all()

    minx, miny, maxx, maxy = union.bounds

    return {
        "minx": float(minx),
        "miny": float(miny),
        "maxx": float(maxx),
        "maxy": float(maxy),
    }


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ANÁLISIS DE CLUSTERS DE MOVILIDAD SUBE 2025")
print("=" * 70)


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset territorial...")

df = pd.read_parquet(
    INPUT_PATH
)

print(
    f"H3 cargados: {len(df):,}"
)


# ============================================================
# VALIDACIONES
# ============================================================

columnas_requeridas = [
    "id_h3",
    "operaciones_totales",
    "provincia",
    "jurisdiccion",
    "categoria_demanda",
    "hora_pico",
    "modo_dominante",
    "geometry",
]

faltantes = [
    c
    for c in columnas_requeridas
    if c not in df.columns
]

if faltantes:

    raise ValueError(
        "Faltan columnas requeridas: "
        + ", ".join(faltantes)
    )


# ============================================================
# NORMALIZAR DATOS
# ============================================================

df["id_h3"] = (
    df["id_h3"]
    .apply(convertir_h3)
)

df["operaciones_totales"] = (
    pd.to_numeric(
        df["operaciones_totales"],
        errors="coerce"
    )
    .fillna(0)
)

df["hora_pico"] = (
    pd.to_numeric(
        df["hora_pico"],
        errors="coerce"
    )
)


# ============================================================
# RECONSTRUIR GEOMETRÍAS
# ============================================================

print("\nReconstruyendo geometrías...")

def reconstruir_geometry(valor):

    if isinstance(
        valor,
        (bytes, bytearray)
    ):
        from shapely import wkb

        return wkb.loads(valor)

    return valor


df["geometry"] = (
    df["geometry"]
    .apply(reconstruir_geometry)
)


gdf = gpd.GeoDataFrame(
    df,
    geometry="geometry",
    crs="EPSG:4326"
)


gdf = gdf[
    gdf.geometry.notna()
].copy()

gdf = gdf[
    ~gdf.geometry.is_empty
].copy()


print(
    f"Geometrías válidas: {len(gdf):,}"
)


# ============================================================
# SUPERFICIE
# ============================================================

print("\nCalculando superficies...")

gdf_m = gdf.to_crs(
    "EPSG:3857"
)

gdf["superficie_m2"] = (
    gdf_m.geometry.area
)

gdf["superficie_km2"] = (
    gdf["superficie_m2"]
    / 1_000_000
)


# ============================================================
# UMBRAL DINÁMICO DE DEMANDA
# ============================================================

umbral_operaciones = (
    gdf["operaciones_totales"]
    .quantile(
        PERCENTIL_OPERACIONES
    )
)

print(
    "\nUmbral de operaciones "
    f"(percentil {PERCENTIL_OPERACIONES:.0%}): "
    f"{umbral_operaciones:,.0f}"
)


# ============================================================
# SELECCIÓN DE H3 CANDIDATOS
# ============================================================

gdf["es_categoria_cluster"] = (
    gdf["categoria_demanda"]
    .isin(
        CATEGORIAS_CLUSTER
    )
)


gdf["es_h3_alta_demanda"] = (
    gdf["operaciones_totales"]
    >= umbral_operaciones
)


gdf["es_candidato_cluster"] = (
    gdf["es_categoria_cluster"]
    |
    gdf["es_h3_alta_demanda"]
)


candidatos = gdf[
    gdf["es_candidato_cluster"]
].copy()


print(
    f"H3 candidatos a cluster: "
    f"{len(candidatos):,}"
)

print(
    f"Operaciones candidatos: "
    f"{candidatos['operaciones_totales'].sum():,.0f}"
)


# ============================================================
# ÍNDICE ESPACIAL H3
# ============================================================

print("\nConstruyendo índice espacial H3...")

ids_candidatos = set(
    candidatos["id_h3"]
)

h3_a_indice = {
    row["id_h3"]: idx
    for idx, row
    in candidatos.reset_index(drop=True).iterrows()
}


# ============================================================
# DETECTAR COMPONENTES CONEXOS
# ============================================================

print(
    "\nDetectando clusters territoriales..."
)


visitados = set()

clusters = []

cluster_id = 0


for h3_inicial in ids_candidatos:

    if h3_inicial in visitados:
        continue

    cluster_id += 1

    cola = deque(
        [h3_inicial]
    )

    visitados.add(
        h3_inicial
    )

    miembros = []

    while cola:

        actual = cola.popleft()

        miembros.append(
            actual
        )

        vecinos = obtener_vecinos(
            actual
        )

        vecinos_candidatos = (
            vecinos
            & ids_candidatos
        )

        for vecino in vecinos_candidatos:

            if vecino not in visitados:

                visitados.add(
                    vecino
                )

                cola.append(
                    vecino
                )

    clusters.append(
        {
            "cluster_id": cluster_id,
            "h3_ids": miembros,
        }
    )


print(
    f"Componentes territoriales detectados: "
    f"{len(clusters):,}"
)


# ============================================================
# CREAR TABLA H3 → CLUSTER
# ============================================================

h3_cluster_rows = []

for cluster in clusters:

    cid = cluster["cluster_id"]

    for h3_id in cluster["h3_ids"]:

        h3_cluster_rows.append(
            {
                "id_h3": h3_id,
                "cluster_id": cid,
            }
        )


h3_clusters = pd.DataFrame(
    h3_cluster_rows
)


# ============================================================
# UNIR CLUSTERS AL DATASET
# ============================================================

gdf = gdf.merge(
    h3_clusters,
    on="id_h3",
    how="left"
)


# ============================================================
# ELIMINAR CLUSTERS DEMASIADO PEQUEÑOS
# ============================================================

tamano_clusters = (
    h3_clusters
    .groupby("cluster_id")
    .size()
    .rename("h3_cluster")
)


operaciones_clusters = (
    gdf
    .groupby("cluster_id")[
        "operaciones_totales"
    ]
    .sum()
    .rename("operaciones_cluster")
)


estadisticas_clusters = pd.concat(
    [
        tamano_clusters,
        operaciones_clusters,
    ],
    axis=1
).reset_index()


estadisticas_clusters = (
    estadisticas_clusters[
        (
            estadisticas_clusters[
                "h3_cluster"
            ]
            >= MIN_H3_CLUSTER
        )
        &
        (
            estadisticas_clusters[
                "operaciones_cluster"
            ]
            >= MIN_OPERACIONES_CLUSTER
        )
    ]
)


clusters_validos = set(
    estadisticas_clusters[
        "cluster_id"
    ]
)


gdf["cluster_id"] = (
    gdf["cluster_id"]
    .where(
        gdf["cluster_id"]
        .isin(clusters_validos)
    )
)


print(
    f"Clusters válidos: "
    f"{len(clusters_validos):,}"
)


# ============================================================
# ANALIZAR CLUSTERS
# ============================================================

print(
    "\nCaracterizando clusters..."
)


resultados = []


for cid in sorted(
    clusters_validos
):

    grupo = gdf[
        gdf["cluster_id"] == cid
    ].copy()

    if grupo.empty:
        continue

    operaciones = (
        grupo[
            "operaciones_totales"
        ].sum()
    )

    h3_count = len(
        grupo
    )

    superficie = (
        grupo[
            "superficie_km2"
        ].sum()
    )

    jurisdicciones = sorted(
        grupo[
            "jurisdiccion"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    provincias = sorted(
        grupo[
            "provincia"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    categorias = (
        grupo[
            "categoria_demanda"
        ]
        .value_counts()
    )

    categoria_dominante = (
        categorias.index[0]
        if len(categorias)
        else None
    )

    hora_pico = combinar_horas(
        grupo["hora_pico"]
    )

    modo_dominante = combinar_modos(
        grupo["modo_dominante"]
    )

    h3_hotspot = int(
        grupo[
            "categoria_demanda"
        ]
        .isin(
            [
                "HOTSPOT_EXTREMO",
                "HOTSPOT_ALTO",
            ]
        )
        .sum()
    )

    operaciones_hotspot = (
        grupo.loc[
            grupo[
                "categoria_demanda"
            ]
            .isin(
                [
                    "HOTSPOT_EXTREMO",
                    "HOTSPOT_ALTO",
                ]
            ),
            "operaciones_totales",
        ]
        .sum()
    )

    densidad = (
        operaciones
        / superficie
        if superficie > 0
        else 0
    )

    bbox = calcular_bbox(
        grupo.geometry
    )

    resultados.append(
        {
            "cluster_id": cid,

            "h3": h3_count,

            "operaciones": int(
                operaciones
            ),

            "superficie_km2": float(
                superficie
            ),

            "operaciones_por_km2": float(
                densidad
            ),

            "h3_hotspot": h3_hotspot,

            "pct_h3_hotspot": (
                h3_hotspot
                / h3_count
                * 100
                if h3_count
                else 0
            ),

            "operaciones_hotspot": int(
                operaciones_hotspot
            ),

            "pct_operaciones_hotspot": (
                operaciones_hotspot
                / operaciones
                * 100
                if operaciones
                else 0
            ),

            "categoria_dominante":
                categoria_dominante,

            "hora_pico":
                hora_pico,

            "modo_dominante":
                modo_dominante,

            "jurisdicciones":
                " | ".join(
                    jurisdicciones
                ),

            "cantidad_jurisdicciones":
                len(jurisdicciones),

            "provincias":
                " | ".join(
                    provincias
                ),

            "cantidad_provincias":
                len(provincias),

            "minx":
                bbox["minx"],

            "miny":
                bbox["miny"],

            "maxx":
                bbox["maxx"],

            "maxy":
                bbox["maxy"],
        }
    )


clusters_df = pd.DataFrame(
    resultados
)


# ============================================================
# RANKINGS
# ============================================================

clusters_df = clusters_df.sort_values(
    [
        "operaciones",
        "h3",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(
    drop=True
)


clusters_df["ranking_operaciones"] = (
    clusters_df.index + 1
)


clusters_df["pct_operaciones"] = (
    clusters_df["operaciones"]
    / clusters_df["operaciones"].sum()
    * 100
)


clusters_df[
    "pct_operaciones_acumulado"
] = (
    clusters_df[
        "pct_operaciones"
    ].cumsum()
)


# ============================================================
# CLASIFICACIÓN DEL CLUSTER
# ============================================================

def clasificar_cluster(row):

    operaciones = row[
        "operaciones"
    ]

    densidad = row[
        "operaciones_por_km2"
    ]

    h3 = row[
        "h3"
    ]

    hotspot_pct = row[
        "pct_operaciones_hotspot"
    ]

    if (
        operaciones >= 500_000
        or hotspot_pct >= 80
    ):
        return "CLUSTER_CRITICO"

    if (
        operaciones >= 200_000
        or hotspot_pct >= 60
    ):
        return "CLUSTER_MUY_ALTO"

    if (
        operaciones >= 100_000
        or hotspot_pct >= 40
    ):
        return "CLUSTER_ALTO"

    if (
        operaciones >= 25_000
        or h3 >= 10
    ):
        return "CLUSTER_MEDIO"

    return "CLUSTER_BAJO"


clusters_df[
    "categoria_cluster"
] = (
    clusters_df
    .apply(
        clasificar_cluster,
        axis=1
    )
)


# ============================================================
# GUARDAR CLUSTERS
# ============================================================

clusters_df.to_parquet(
    OUTPUT_CLUSTERS,
    index=False
)


# ============================================================
# GUARDAR H3 → CLUSTER
# ============================================================

gdf_h3 = gdf[
    gdf["cluster_id"].notna()
].copy()


columnas_h3_salida = [
    "id_h3",
    "cluster_id",
    "provincia",
    "jurisdiccion",
    "operaciones_totales",
    "categoria_demanda",
    "hora_pico",
    "modo_dominante",
    "superficie_km2",
]


columnas_h3_salida = [
    c
    for c in columnas_h3_salida
    if c in gdf_h3.columns
]


gdf_h3[
    columnas_h3_salida
].to_parquet(
    OUTPUT_H3,
    index=False
)


# ============================================================
# RESUMEN
# ============================================================

operaciones_clusterizadas = (
    clusters_df[
        "operaciones"
    ].sum()
    if not clusters_df.empty
    else 0
)

operaciones_totales = (
    gdf[
        "operaciones_totales"
    ].sum()
)


pct_clusterizado = (
    operaciones_clusterizadas
    / operaciones_totales
    * 100
    if operaciones_totales
    else 0
)


resumen = {

    "h3_totales":
        int(len(gdf)),

    "operaciones_totales":
        int(operaciones_totales),

    "h3_candidatos":
        int(len(candidatos)),

    "clusters_detectados":
        int(len(clusters)),

    "clusters_validos":
        int(len(clusters_df)),

    "h3_clusterizados":
        int(len(gdf_h3)),

    "operaciones_clusterizadas":
        int(operaciones_clusterizadas),

    "pct_operaciones_clusterizadas":
        float(pct_clusterizado),

    "parametros": {

        "categorias_cluster":
            sorted(
                CATEGORIAS_CLUSTER
            ),

        "min_h3_cluster":
            MIN_H3_CLUSTER,

        "min_operaciones_cluster":
            MIN_OPERACIONES_CLUSTER,

        "percentil_operaciones":
            PERCENTIL_OPERACIONES,

        "umbral_operaciones":
            float(
                umbral_operaciones
            ),
    },

    "categorias_clusters":
        (
            clusters_df[
                "categoria_cluster"
            ]
            .value_counts()
            .to_dict()
            if not clusters_df.empty
            else {}
        ),
}


with open(
    OUTPUT_RESUMEN,
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
# SALIDA
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TOP 30 CLUSTERS DE MOVILIDAD"
)

print(
    "=" * 70
)


columnas_top = [
    "ranking_operaciones",
    "cluster_id",
    "h3",
    "operaciones",
    "pct_operaciones",
    "pct_operaciones_acumulado",
    "superficie_km2",
    "operaciones_por_km2",
    "categoria_cluster",
    "categoria_dominante",
    "hora_pico",
    "modo_dominante",
    "cantidad_jurisdicciones",
    "jurisdicciones",
]


print(
    clusters_df[
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
    "DISTRIBUCIÓN DE CLUSTERS"
)

print(
    "=" * 70
)


if not clusters_df.empty:

    print(
        clusters_df[
            "categoria_cluster"
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
    "HORAS PICO DE LOS CLUSTERS"
)

print(
    "=" * 70
)


print(
    clusters_df[
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
    clusters_df[
        "modo_dominante"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# RESUMEN FINAL
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
    f"\nH3 analizados: "
    f"{len(gdf):,}"
)

print(
    f"Operaciones: "
    f"{operaciones_totales:,.0f}"
)

print(
    f"H3 candidatos: "
    f"{len(candidatos):,}"
)

print(
    f"Clusters detectados: "
    f"{len(clusters):,}"
)

print(
    f"Clusters válidos: "
    f"{len(clusters_df):,}"
)

print(
    f"H3 dentro de clusters: "
    f"{len(gdf_h3):,}"
)

print(
    f"Operaciones clusterizadas: "
    f"{operaciones_clusterizadas:,.0f}"
)

print(
    f"% operaciones clusterizadas: "
    f"{pct_clusterizado:.2f}%"
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
    "\nClusters:"
)

print(
    OUTPUT_CLUSTERS
)

print(
    "\nH3 → cluster:"
)

print(
    OUTPUT_H3
)

print(
    "\nResumen:"
)

print(
    OUTPUT_RESUMEN
)


print(
    "\n"
    + "=" * 70
)

print(
    "ANÁLISIS DE CLUSTERS FINALIZADO"
)

print(
    "=" * 70
)