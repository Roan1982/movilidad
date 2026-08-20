from pathlib import Path
import json

import pandas as pd
import geopandas as gpd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

H3_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_clusters.parquet"
)

H3_CORREDORES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_corredores.parquet"
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

OUTPUT_NODOS = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_nodos_movilidad.parquet"
)

OUTPUT_H3_NODOS = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_nodos.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_nodos_resumen.json"
)

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:3857"

# ------------------------------------------------------------
# PARÁMETROS DEL ANÁLISIS
# ------------------------------------------------------------

PERCENTIL_DEMANDA = 0.75

# Radio H3 utilizado para buscar continuidad.
#
# 1 = vecinos inmediatos.
# 2 = vecinos hasta distancia 2.
#
# Mantenemos 1 para no sobreagrupar.
RADIO_CONECTIVIDAD = 1

# IMPORTANTE:
# Los nodos de un solo H3 son válidos.
#
# Esto evita perder hotspots puntuales de movilidad.
MIN_H3_NODO = 1


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ANÁLISIS DE NODOS DE MOVILIDAD SUBE 2025")
print("=" * 70)


# ============================================================
# CARGAR H3
# ============================================================

print("\nCargando librería H3...")

try:

    import h3

except ImportError:

    raise ImportError(
        "No está instalada la librería h3. "
        "Instalar con: pip install h3"
    )


print(
    f"Versión H3: "
    f"{getattr(h3, '__version__', 'desconocida')}"
)


# ============================================================
# CARGAR H3 → CLUSTER
# ============================================================

print(
    "\nCargando relación H3 → cluster..."
)

h3_df = pd.read_parquet(
    H3_PATH
)

print(
    f"H3 cargados: "
    f"{len(h3_df):,}"
)

print(
    "Columnas H3:"
)

print(
    h3_df.columns.tolist()
)


# ============================================================
# VALIDACIONES
# ============================================================

columnas_h3_requeridas = [
    "id_h3",
    "operaciones_totales",
    "jurisdiccion",
    "provincia",
]

faltantes = [
    columna
    for columna in columnas_h3_requeridas
    if columna not in h3_df.columns
]

if faltantes:

    raise ValueError(
        "Faltan columnas en H3: "
        + ", ".join(faltantes)
    )


print(
    f"\nColumna H3: id_h3"
)

print(
    f"Columna operaciones: operaciones_totales"
)

print(
    f"Columna jurisdicción: jurisdiccion"
)

print(
    f"Columna provincia: provincia"
)


# ============================================================
# NORMALIZAR ID H3
# ============================================================

h3_df["id_h3"] = (
    h3_df["id_h3"]
    .astype(str)
    .str.strip()
)


# ============================================================
# RECONSTRUIR GEOMETRÍAS H3
# ============================================================

print(
    "\nReconstruyendo geometrías H3..."
)


def construir_geometria_h3(h3_id):

    try:

        # H3 4.x
        limite = h3.cell_to_boundary(
            h3_id
        )

        # H3 devuelve:
        # [(lat, lon), ...]

        from shapely.geometry import Polygon

        return Polygon(
            [
                (lon, lat)
                for lat, lon
                in limite
            ]
        )

    except Exception:

        return None


h3_df["geometry"] = (
    h3_df["id_h3"]
    .apply(
        construir_geometria_h3
    )
)


h3_geo = gpd.GeoDataFrame(
    h3_df,
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
# VALIDAR GEOMETRÍAS
# ============================================================

invalidas = (
    ~h3_geo.geometry.is_valid
)

if invalidas.any():

    print(
        f"Corrigiendo "
        f"{invalidas.sum():,} geometrías..."
    )

    try:

        from shapely.validation import make_valid

        h3_geo.loc[
            invalidas,
            "geometry"
        ] = (
            h3_geo.loc[
                invalidas,
                "geometry"
            ]
            .apply(make_valid)
        )

    except Exception:

        h3_geo.loc[
            invalidas,
            "geometry"
        ] = (
            h3_geo.loc[
                invalidas,
                "geometry"
            ]
            .buffer(0)
        )


# ============================================================
# SUPERFICIE
# ============================================================

print(
    "\nCalculando superficies..."
)


h3_m = h3_geo.to_crs(
    CRS_METRICO
)


h3_geo["superficie_km2"] = (
    h3_m.geometry.area
    / 1_000_000
)


# ============================================================
# CARGAR H3 → CORREDOR
# ============================================================

print(
    "\nCargando relación H3 → corredor..."
)


h3_corredores = pd.read_parquet(
    H3_CORREDORES_PATH
)


print(
    f"H3 → corredor cargados: "
    f"{len(h3_corredores):,}"
)


print(
    "Columnas:"
)

print(
    h3_corredores.columns.tolist()
)


if "id_h3" not in h3_corredores.columns:

    raise ValueError(
        "El archivo H3 → corredor "
        "no contiene 'id_h3'."
    )


# ============================================================
# NORMALIZAR RELACIÓN CORREDOR
# ============================================================

h3_corredores["id_h3"] = (
    h3_corredores["id_h3"]
    .astype(str)
    .str.strip()
)


# ============================================================
# DETECTAR ID DEL CORREDOR
# ============================================================

if "corredor_id" in h3_corredores.columns:

    columna_corredor = "corredor_id"

elif "cluster_id" in h3_corredores.columns:

    columna_corredor = "cluster_id"

else:

    columna_corredor = None


if columna_corredor is None:

    print(
        "ADVERTENCIA: no se encontró "
        "identificador de corredor."
    )

else:

    print(
        f"Columna de corredor detectada: "
        f"{columna_corredor}"
    )


# ============================================================
# PREPARAR RELACIÓN CORREDOR
# ============================================================

columnas_corredor = [
    "id_h3"
]


if columna_corredor:

    columnas_corredor.append(
        columna_corredor
    )


for columna in [
    "indice_corredor",
    "categoria_corredor",
]:

    if columna in h3_corredores.columns:

        columnas_corredor.append(
            columna
        )


h3_corredores = (
    h3_corredores[
        columnas_corredor
    ]
    .drop_duplicates(
        subset=["id_h3"]
    )
    .copy()
)


if columna_corredor:

    h3_corredores = (
        h3_corredores.rename(
            columns={
                columna_corredor:
                    "corredor_id"
            }
        )
    )


# ============================================================
# MERGE H3 → CORREDOR
# ============================================================

h3_geo = h3_geo.merge(
    h3_corredores,
    on="id_h3",
    how="left",
    suffixes=("", "_corredor")
)


# ============================================================
# CARGAR CORREDORES
# ============================================================

print(
    "\nValidando archivo de corredores..."
)


corredores = pd.read_parquet(
    CORREDORES_PATH
)


print(
    f"Corredores cargados: "
    f"{len(corredores):,}"
)


# ============================================================
# CARGAR CLUSTERS
# ============================================================

print(
    "\nCargando información de clusters..."
)


clusters = pd.read_parquet(
    CLUSTERS_PATH
)


print(
    f"Clusters cargados: "
    f"{len(clusters):,}"
)


if "cluster_id" not in h3_geo.columns:

    print(
        "ADVERTENCIA: "
        "no existe cluster_id en H3."
    )


# ============================================================
# CANDIDATOS A NODO
# ============================================================

print(
    "\nCalculando candidatos a nodo..."
)


h3_geo["operaciones_totales"] = (
    pd.to_numeric(
        h3_geo["operaciones_totales"],
        errors="coerce"
    )
    .fillna(0)
)


operaciones = (
    h3_geo["operaciones_totales"]
)


umbral_demanda = (
    operaciones.quantile(
        PERCENTIL_DEMANDA
    )
)


print(
    f"Percentil 75 de operaciones: "
    f"{umbral_demanda:,.0f}"
)


h3_geo["candidato_demanda"] = (
    operaciones >= umbral_demanda
)


# ============================================================
# CRITERIO ESTRUCTURAL
# ============================================================

h3_geo["tiene_cluster"] = False

if "cluster_id" in h3_geo.columns:

    h3_geo["tiene_cluster"] = (
        h3_geo["cluster_id"]
        .notna()
    )


h3_geo["tiene_corredor"] = False

if "corredor_id" in h3_geo.columns:

    h3_geo["tiene_corredor"] = (
        h3_geo["corredor_id"]
        .notna()
    )


# Un nodo requiere alta demanda y además
# pertenencia territorial a cluster o corredor.

h3_geo["candidato_nodo"] = (
    h3_geo["candidato_demanda"]
    &
    (
        h3_geo["tiene_cluster"]
        |
        h3_geo["tiene_corredor"]
    )
)


candidatos = h3_geo[
    h3_geo["candidato_nodo"]
].copy()


print(
    f"H3 candidatos a nodo: "
    f"{len(candidatos):,}"
)


print(
    f"Operaciones candidatos: "
    f"{candidatos['operaciones_totales'].sum():,.0f}"
)


# ============================================================
# ÍNDICE H3
# ============================================================

indice_h3 = {

    str(row["id_h3"]): idx

    for idx, row
    in candidatos.iterrows()

}


# ============================================================
# CONECTIVIDAD H3
# ============================================================

print(
    "\nConstruyendo conectividad H3..."
)


def obtener_vecinos(
    h3_id,
    radio=RADIO_CONECTIVIDAD
):

    try:

        vecinos = h3.grid_disk(
            h3_id,
            radio
        )

        return [
            str(v)
            for v in vecinos
        ]

    except Exception:

        return []


# ============================================================
# COMPONENTES
# ============================================================

print(
    "\nDetectando componentes nodales..."
)


visitados = set()

componentes = []


for h3_id in indice_h3:

    if h3_id in visitados:

        continue


    cola = [
        h3_id
    ]

    componente = []

    visitados.add(
        h3_id
    )


    while cola:

        actual = cola.pop()

        componente.append(
            actual
        )


        vecinos = obtener_vecinos(
            actual
        )


        for vecino in vecinos:

            if vecino == actual:

                continue


            if vecino not in indice_h3:

                continue


            if vecino in visitados:

                continue


            visitados.add(
                vecino
            )

            cola.append(
                vecino
            )


    componentes.append(
        componente
    )


print(
    f"Componentes nodales detectados: "
    f"{len(componentes):,}"
)


# ============================================================
# VALIDAR COMPONENTES
# ============================================================

# A diferencia del script anterior,
# NO descartamos componentes de un solo H3.

componentes_validos = [

    componente

    for componente in componentes

    if len(componente) >= MIN_H3_NODO

]


print(
    f"Nodos válidos: "
    f"{len(componentes_validos):,}"
)


# ============================================================
# H3 → NODO
# ============================================================

h3_nodo = {}


for nodo_id, componente in enumerate(
    componentes_validos,
    start=1
):

    for h3_id in componente:

        h3_nodo[
            str(h3_id)
        ] = nodo_id


candidatos["nodo_id"] = (
    candidatos["id_h3"]
    .astype(str)
    .map(h3_nodo)
)


candidatos = candidatos[
    candidatos["nodo_id"].notna()
].copy()


candidatos["nodo_id"] = (
    candidatos["nodo_id"]
    .astype(int)
)


print(
    f"H3 dentro de nodos: "
    f"{len(candidatos):,}"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def modo_dominante_serie(
    serie
):

    serie = (
        serie
        .dropna()
        .astype(str)
    )

    if serie.empty:

        return None

    return (
        serie
        .value_counts()
        .index[0]
    )


def hora_pico_serie(
    serie
):

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    ).dropna()

    if serie.empty:

        return None

    return int(
        serie
        .value_counts()
        .index[0]
    )


def percentil_normalizado(
    serie
):

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    )


    if serie.notna().sum() <= 1:

        return pd.Series(
            50.0,
            index=serie.index
        )


    return (
        serie
        .rank(
            pct=True,
            method="average"
        )
        * 100
    ).fillna(0)


# ============================================================
# CARACTERIZAR NODOS
# ============================================================

print(
    "\nCaracterizando nodos..."
)


agrupaciones = []


for nodo_id, grupo in candidatos.groupby(
    "nodo_id"
):

    operaciones_nodo = (
        pd.to_numeric(
            grupo[
                "operaciones_totales"
            ],
            errors="coerce"
        )
        .fillna(0)
    )


    operaciones_total = (
        operaciones_nodo.sum()
    )


    superficie = (
        pd.to_numeric(
            grupo[
                "superficie_km2"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    if superficie > 0:

        densidad = (
            operaciones_total
            / superficie
        )

    else:

        densidad = 0


    # --------------------------------------------------------
    # CORREDORES
    # --------------------------------------------------------

    corredores_unicos = []

    if "corredor_id" in grupo.columns:

        corredores_unicos = (
            grupo[
                "corredor_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


    # --------------------------------------------------------
    # CLUSTERS
    # --------------------------------------------------------

    clusters_unicos = []

    if "cluster_id" in grupo.columns:

        clusters_unicos = (
            grupo[
                "cluster_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


    # --------------------------------------------------------
    # JURISDICCIONES
    # --------------------------------------------------------

    jurisdicciones = (
        grupo[
            "jurisdiccion"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    jurisdicciones.sort()


    # --------------------------------------------------------
    # PROVINCIAS
    # --------------------------------------------------------

    provincias = (
        grupo[
            "provincia"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    provincias.sort()


    # --------------------------------------------------------
    # CATEGORÍA DEMANDA
    # --------------------------------------------------------

    categoria = None

    if "categoria_demanda" in grupo.columns:

        categoria = (
            modo_dominante_serie(
                grupo[
                    "categoria_demanda"
                ]
            )
        )


    # --------------------------------------------------------
    # HORA PICO
    # --------------------------------------------------------

    hora = None

    if "hora_pico" in grupo.columns:

        hora = (
            hora_pico_serie(
                grupo[
                    "hora_pico"
                ]
            )
        )


    # --------------------------------------------------------
    # MODO
    # --------------------------------------------------------

    modo = None

    if "modo_dominante" in grupo.columns:

        modo = (
            modo_dominante_serie(
                grupo[
                    "modo_dominante"
                ]
            )
        )


    # --------------------------------------------------------
    # GEOMETRÍA
    # --------------------------------------------------------

    geometria = (
        grupo
        .geometry
        .union_all()
    )


    punto = (
        geometria
        .centroid
    )


    # --------------------------------------------------------
    # INDICADORES
    # --------------------------------------------------------

    cantidad_corredores = (
        len(
            corredores_unicos
        )
    )


    cantidad_clusters = (
        len(
            clusters_unicos
        )
    )


    cantidad_jurisdicciones = (
        len(
            jurisdicciones
        )
    )


    cantidad_provincias = (
        len(
            provincias
        )
    )


    # --------------------------------------------------------
    # REGISTRO
    # --------------------------------------------------------

    agrupaciones.append({

        "nodo_id":
            int(nodo_id),

        "h3":
            int(len(grupo)),

        "operaciones":
            float(operaciones_total),

        "superficie_km2":
            float(superficie),

        "operaciones_por_km2":
            float(densidad),

        "cantidad_corredores":
            int(cantidad_corredores),

        "cantidad_clusters":
            int(cantidad_clusters),

        "cantidad_jurisdicciones":
            int(cantidad_jurisdicciones),

        "cantidad_provincias":
            int(cantidad_provincias),

        "jurisdicciones":
            " | ".join(
                jurisdicciones
            ),

        "provincias":
            " | ".join(
                provincias
            ),

        "categoria_dominante":
            categoria,

        "hora_pico":
            hora,

        "modo_dominante":
            modo,

        "geometry":
            punto,

    })


# ============================================================
# CREAR GEODATAFRAME
# ============================================================

if agrupaciones:

    nodos = gpd.GeoDataFrame(
        agrupaciones,
        geometry="geometry",
        crs=CRS_GEOGRAFICO
    )

else:

    # Protección adicional.
    # Nunca debería ocurrir porque aceptamos nodos
    # individuales, pero evita errores posteriores.

    nodos = gpd.GeoDataFrame(
        columns=[
            "nodo_id",
            "h3",
            "operaciones",
            "superficie_km2",
            "operaciones_por_km2",
            "cantidad_corredores",
            "cantidad_clusters",
            "cantidad_jurisdicciones",
            "cantidad_provincias",
            "jurisdicciones",
            "provincias",
            "categoria_dominante",
            "hora_pico",
            "modo_dominante",
            "geometry",
        ],
        geometry="geometry",
        crs=CRS_GEOGRAFICO
    )


# ============================================================
# ÍNDICES DE NODO
# ============================================================

print(
    "\nCalculando índices de nodos..."
)


if not nodos.empty:

    nodos[
        "score_demanda"
    ] = percentil_normalizado(
        nodos[
            "operaciones"
        ]
    )


    nodos[
        "score_densidad"
    ] = percentil_normalizado(
        nodos[
            "operaciones_por_km2"
        ]
    )


    nodos[
        "score_corredores"
    ] = percentil_normalizado(
        nodos[
            "cantidad_corredores"
        ]
    )


    nodos[
        "score_clusters"
    ] = percentil_normalizado(
        nodos[
            "cantidad_clusters"
        ]
    )


    nodos[
        "score_jurisdicciones"
    ] = percentil_normalizado(
        nodos[
            "cantidad_jurisdicciones"
        ]
    )


else:

    for columna in [
        "score_demanda",
        "score_densidad",
        "score_corredores",
        "score_clusters",
        "score_jurisdicciones",
    ]:

        nodos[columna] = pd.Series(
            dtype=float
        )


# ============================================================
# ÍNDICE FINAL
# ============================================================

if not nodos.empty:

    nodos[
        "indice_nodo"
    ] = (

        nodos[
            "score_demanda"
        ] * 0.35

        +

        nodos[
            "score_densidad"
        ] * 0.20

        +

        nodos[
            "score_corredores"
        ] * 0.25

        +

        nodos[
            "score_clusters"
        ] * 0.10

        +

        nodos[
            "score_jurisdicciones"
        ] * 0.10

    )


    nodos[
        "indice_nodo"
    ] = (
        nodos[
            "indice_nodo"
        ]
        .clip(0, 100)
    )

else:

    nodos[
        "indice_nodo"
    ] = pd.Series(
        dtype=float
    )


# ============================================================
# CATEGORÍA
# ============================================================

def clasificar_nodo(
    indice
):

    if pd.isna(indice):

        return "NODO_BAJO"


    if indice >= 85:

        return "NODO_CRITICO"


    if indice >= 70:

        return "NODO_ALTO"


    if indice >= 50:

        return "NODO_MEDIO"


    return "NODO_BAJO"


nodos[
    "categoria_nodo"
] = (
    nodos[
        "indice_nodo"
    ]
    .apply(
        clasificar_nodo
    )
)


# ============================================================
# RANKING
# ============================================================

nodos = nodos.sort_values(
    [
        "operaciones",
        "indice_nodo",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(
    drop=True
)


nodos[
    "ranking_operaciones"
] = (
    nodos.index + 1
)


# ============================================================
# OPERACIONES TOTALES
# ============================================================

operaciones_totales = (
    h3_geo[
        "operaciones_totales"
    ]
    .sum()
)


# ============================================================
# PARTICIPACIÓN
# ============================================================

if operaciones_totales > 0:

    nodos[
        "pct_operaciones"
    ] = (
        nodos[
            "operaciones"
        ]
        / operaciones_totales
        * 100
    )

else:

    nodos[
        "pct_operaciones"
    ] = 0


nodos[
    "pct_operaciones_acumulado"
] = (
    nodos[
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
    "TOP 30 NODOS DE MOVILIDAD"
)

print(
    "=" * 70
)


columnas_top = [

    "ranking_operaciones",

    "nodo_id",

    "h3",

    "operaciones",

    "pct_operaciones",

    "pct_operaciones_acumulado",

    "superficie_km2",

    "operaciones_por_km2",

    "indice_nodo",

    "categoria_nodo",

    "categoria_dominante",

    "hora_pico",

    "modo_dominante",

    "cantidad_corredores",

    "cantidad_clusters",

    "cantidad_jurisdicciones",

    "jurisdicciones",

]


if not nodos.empty:

    print(
        nodos[
            columnas_top
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No se detectaron nodos."
    )


# ============================================================
# DISTRIBUCIÓN
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "DISTRIBUCIÓN DE NODOS"
)

print(
    "=" * 70
)


if not nodos.empty:

    print(
        nodos[
            "categoria_nodo"
        ]
        .value_counts()
        .to_string()
    )

else:

    print(
        "Sin datos."
    )


# ============================================================
# HORAS PICO
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "HORAS PICO DE LOS NODOS"
)

print(
    "=" * 70
)


if not nodos.empty:

    print(
        nodos[
            "hora_pico"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

else:

    print(
        "Sin datos."
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


if not nodos.empty:

    print(
        nodos[
            "modo_dominante"
        ]
        .value_counts()
        .to_string()
    )

else:

    print(
        "Sin datos."
    )


# ============================================================
# JURISDICCIONES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "JURISDICCIONES MÁS PRESENTES EN NODOS"
)

print(
    "=" * 70
)


jurisdicciones_nodos = {}


for _, fila in nodos.iterrows():

    jurisdicciones = (
        str(
            fila[
                "jurisdicciones"
            ]
        )
        .split(" | ")
    )


    for jurisdiccion in jurisdicciones:

        if not jurisdiccion:

            continue


        jurisdicciones_nodos[
            jurisdiccion
        ] = (
            jurisdicciones_nodos.get(
                jurisdiccion,
                0
            )
            + 1
        )


if jurisdicciones_nodos:

    serie_jurisdicciones = (
        pd.Series(
            jurisdicciones_nodos,
            name="nodos"
        )
        .sort_values(
            ascending=False
        )
    )


    print(
        serie_jurisdicciones
        .head(30)
        .to_string()
    )

else:

    print(
        "Sin datos."
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


operaciones_nodos = (
    nodos[
        "operaciones"
    ]
    .sum()
)


pct_nodos = (

    operaciones_nodos
    / operaciones_totales
    * 100

    if operaciones_totales > 0

    else 0

)


print(
    f"\nH3 analizados: "
    f"{len(h3_geo):,}"
)


print(
    f"Operaciones totales: "
    f"{operaciones_totales:,.0f}"
)


print(
    f"H3 candidatos: "
    f"{len(candidatos):,}"
)


print(
    f"Nodos detectados: "
    f"{len(nodos):,}"
)


print(
    f"H3 dentro de nodos: "
    f"{len(candidatos):,}"
)


print(
    f"Operaciones en nodos: "
    f"{operaciones_nodos:,.0f}"
)


print(
    f"% operaciones en nodos: "
    f"{pct_nodos:.2f}%"
)


# ============================================================
# NODO PRINCIPAL
# ============================================================

principal = None


if not nodos.empty:

    principal = (
        nodos
        .sort_values(
            "operaciones",
            ascending=False
        )
        .iloc[0]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "NODO PRINCIPAL"
    )

    print(
        "=" * 70
    )


    print(
        f"Nodo: "
        f"{principal['nodo_id']}"
    )


    print(
        f"H3: "
        f"{principal['h3']}"
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
        f"Índice: "
        f"{principal['indice_nodo']:.2f}/100"
    )


    print(
        f"Categoría: "
        f"{principal['categoria_nodo']}"
    )


    print(
        f"Corredores: "
        f"{principal['cantidad_corredores']}"
    )


    print(
        f"Clusters: "
        f"{principal['cantidad_clusters']}"
    )


    print(
        f"Jurisdicciones: "
        f"{principal['cantidad_jurisdicciones']}"
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
        principal[
            "jurisdicciones"
        ]
    )


# ============================================================
# PREPARAR H3 → NODO
# ============================================================

print(
    "\nPreparando relación H3 → nodo..."
)


columnas_h3_salida = [

    "id_h3",

    "nodo_id",

    "cluster_id",

    "corredor_id",

    "operaciones_totales",

    "jurisdiccion",

    "provincia",

    "categoria_demanda",

    "hora_pico",

    "modo_dominante",

    "superficie_km2",

]


columnas_h3_salida = [

    columna

    for columna in columnas_h3_salida

    if columna in candidatos.columns

]


h3_resultado = (
    candidatos[
        columnas_h3_salida
    ]
    .copy()
)


# ============================================================
# GUARDAR H3 → NODO
# ============================================================

print(
    "\nGuardando H3 → nodo..."
)


h3_resultado.to_parquet(
    OUTPUT_H3_NODOS,
    index=False
)


# ============================================================
# GUARDAR NODOS
# ============================================================

print(
    "Guardando nodos..."
)


nodos.to_parquet(
    OUTPUT_NODOS,
    index=False
)


# ============================================================
# RESUMEN JSON
# ============================================================

resumen = {

    "h3_analizados":
        int(
            len(h3_geo)
        ),

    "operaciones_totales":
        float(
            operaciones_totales
        ),

    "percentil_demanda":
        float(
            PERCENTIL_DEMANDA
        ),

    "percentil_75_operaciones":
        float(
            umbral_demanda
        ),

    "radio_conectividad_h3":
        int(
            RADIO_CONECTIVIDAD
        ),

    "min_h3_nodo":
        int(
            MIN_H3_NODO
        ),

    "h3_candidatos":
        int(
            len(candidatos)
        ),

    "nodos_detectados":
        int(
            len(nodos)
        ),

    "h3_en_nodos":
        int(
            len(candidatos)
        ),

    "operaciones_en_nodos":
        float(
            operaciones_nodos
        ),

    "pct_operaciones_en_nodos":
        float(
            pct_nodos
        ),

    "distribucion_nodos":
        {
            str(k): int(v)

            for k, v

            in nodos[
                "categoria_nodo"
            ]
            .value_counts()
            .items()
        },

    "horas_pico":
        {
            str(k): int(v)

            for k, v

            in nodos[
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

            in nodos[
                "modo_dominante"
            ]
            .value_counts()
            .items()
        },

}


# ============================================================
# NODO PRINCIPAL JSON
# ============================================================

if principal is not None:

    resumen[
        "nodo_principal"
    ] = {

        "nodo_id":
            int(
                principal[
                    "nodo_id"
                ]
            ),

        "h3":
            int(
                principal[
                    "h3"
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

        "indice_nodo":
            float(
                principal[
                    "indice_nodo"
                ]
            ),

        "categoria":
            principal[
                "categoria_nodo"
            ],

        "corredores":
            int(
                principal[
                    "cantidad_corredores"
                ]
            ),

        "clusters":
            int(
                principal[
                    "cantidad_clusters"
                ]
            ),

        "jurisdicciones":
            principal[
                "jurisdicciones"
            ],

        "hora_pico":
            (
                None

                if pd.isna(
                    principal[
                        "hora_pico"
                    ]
                )

                else int(
                    principal[
                        "hora_pico"
                    ]
                )
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
    encoding="utf-8"
) as archivo:

    json.dump(
        resumen,
        archivo,
        ensure_ascii=False,
        indent=2
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
    "\nNodos:"
)

print(
    OUTPUT_NODOS
)


print(
    "\nH3 → nodo:"
)

print(
    OUTPUT_H3_NODOS
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
    "ANÁLISIS DE NODOS FINALIZADO"
)

print(
    "=" * 70
)


print(
    f"H3 analizados: "
    f"{len(h3_geo):,}"
)


print(
    f"Nodos: "
    f"{len(nodos):,}"
)


print(
    f"Operaciones en nodos: "
    f"{operaciones_nodos:,.0f}"
)


print(
    f"% operaciones: "
    f"{pct_nodos:.2f}%"
)