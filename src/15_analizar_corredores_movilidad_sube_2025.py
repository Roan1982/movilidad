from pathlib import Path
import json
import math

import pandas as pd
import geopandas as gpd
from shapely import wkb


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

H3_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_territorio.parquet"
)

CLUSTERS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_clusters.parquet"
)

OUTPUT_CORREDORES = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_corredores_movilidad.parquet"
)

OUTPUT_H3_CORREDORES = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_corredores.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_corredores_resumen.json"
)

OUTPUT_JURISDICCIONES = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_corredores_jurisdicciones.parquet"
)


# ============================================================
# PARÁMETROS ANALÍTICOS
# ============================================================

# Percentil mínimo para considerar un H3 candidato.
PERCENTIL_DEMANDA = 75

# Cantidad mínima absoluta de operaciones.
OPERACIONES_MINIMAS = 1000

# Máxima diferencia horaria para considerar
# compatibles dos H3.
DIFERENCIA_HORARIA_MAXIMA = 2

# Peso de cada componente del índice.
PESO_DEMANDA = 0.30
PESO_CONTINUIDAD_ESPACIAL = 0.30
PESO_CONTINUIDAD_HORARIA = 0.20
PESO_CONTINUIDAD_MODAL = 0.20

# Longitud mínima de un corredor.
MIN_H3_CORREDOR = 3

# Cantidad máxima de corredores a mostrar.
TOP_CORREDORES = 30


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def reconstruir_wkb(valor):
    """
    Reconstruye geometrías almacenadas como WKB.
    """

    if isinstance(valor, (bytes, bytearray)):
        return wkb.loads(valor)

    return valor


def diferencia_horaria(hora_a, hora_b):
    """
    Calcula diferencia circular entre horas.

    Ejemplo:
        23 y 1 -> 2 horas
    """

    if pd.isna(hora_a) or pd.isna(hora_b):
        return 24

    a = int(hora_a)
    b = int(hora_b)

    diferencia = abs(a - b)

    return min(
        diferencia,
        24 - diferencia
    )


def compatibilidad_modal(modo_a, modo_b):
    """
    Compatibilidad modal:

    1.0 -> mismo modo
    0.5 -> uno de los valores es desconocido
    0.0 -> modos diferentes
    """

    if pd.isna(modo_a) or pd.isna(modo_b):
        return 0.5

    if str(modo_a) == str(modo_b):
        return 1.0

    return 0.0


def clasificar_corredor(indice):
    """
    Clasificación final del corredor.
    """

    if indice >= 80:
        return "CORREDOR_CRITICO"

    if indice >= 65:
        return "CORREDOR_ALTO"

    if indice >= 50:
        return "CORREDOR_MEDIO"

    return "CORREDOR_BAJO"


def normalizar_serie_minmax(serie):
    """
    Normalización 0-1.
    """

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    ).fillna(0)

    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return pd.Series(
            1.0,
            index=serie.index
        )

    return (
        (serie - minimo)
        / (maximo - minimo)
    )


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ANÁLISIS DE CORREDORES DE MOVILIDAD SUBE 2025")
print("=" * 70)


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset territorial...")

h3 = pd.read_parquet(
    H3_PATH
)

print(
    f"H3 cargados: {len(h3):,}"
)


# ============================================================
# VALIDACIONES
# ============================================================

columnas_requeridas = [
    "id_h3",
    "operaciones_totales",
    "jurisdiccion",
    "provincia",
    "categoria_demanda",
    "hora_pico",
    "modo_dominante",
    "geometry",
]

faltantes = [
    c
    for c in columnas_requeridas
    if c not in h3.columns
]

if faltantes:

    raise ValueError(
        "Faltan columnas requeridas: "
        + ", ".join(faltantes)
    )


# ============================================================
# RECONSTRUIR GEOMETRÍAS
# ============================================================

print("\nReconstruyendo geometrías...")

h3["geometry"] = (
    h3["geometry"]
    .apply(reconstruir_wkb)
)

h3_geo = gpd.GeoDataFrame(
    h3,
    geometry="geometry",
    crs="EPSG:4326"
)

h3_geo = h3_geo[
    h3_geo.geometry.notna()
].copy()

h3_geo = h3_geo[
    ~h3_geo.geometry.is_empty
].copy()

print(
    f"Geometrías válidas: {len(h3_geo):,}"
)


# ============================================================
# VALIDAR H3
# ============================================================

duplicados = h3_geo["id_h3"].duplicated().sum()

if duplicados > 0:

    raise ValueError(
        f"Se encontraron {duplicados:,} H3 duplicados."
    )


# ============================================================
# PREPARAR VARIABLES
# ============================================================

h3_geo["operaciones_totales"] = (
    pd.to_numeric(
        h3_geo["operaciones_totales"],
        errors="coerce"
    )
    .fillna(0)
)


h3_geo["hora_pico"] = (
    pd.to_numeric(
        h3_geo["hora_pico"],
        errors="coerce"
    )
)


# ============================================================
# SUPERFICIE
# ============================================================

print("\nCalculando superficies...")

h3_m = h3_geo.to_crs(
    "EPSG:3857"
)

h3_geo["superficie_km2"] = (
    h3_m.geometry.area
    / 1_000_000
)


h3_geo["operaciones_por_km2"] = (
    h3_geo["operaciones_totales"]
    / h3_geo["superficie_km2"]
    .replace(0, float("nan"))
)


# ============================================================
# UMBRAL DE DEMANDA
# ============================================================

umbral_percentil = h3_geo[
    "operaciones_totales"
].quantile(
    PERCENTIL_DEMANDA / 100
)

umbral_demanda = max(
    umbral_percentil,
    OPERACIONES_MINIMAS
)


print(
    f"\nPercentil {PERCENTIL_DEMANDA}: "
    f"{umbral_percentil:,.0f}"
)

print(
    f"Umbral utilizado: "
    f"{umbral_demanda:,.0f}"
)


# ============================================================
# H3 CANDIDATOS
# ============================================================

candidatos = h3_geo[
    h3_geo["operaciones_totales"]
    >= umbral_demanda
].copy()


print(
    f"H3 candidatos a corredor: "
    f"{len(candidatos):,}"
)


operaciones_candidatos = candidatos[
    "operaciones_totales"
].sum()


print(
    f"Operaciones candidatos: "
    f"{operaciones_candidatos:,.0f}"
)


# ============================================================
# ÍNDICE H3
# ============================================================

candidatos = candidatos.reset_index(
    drop=True
)

id_to_idx = {
    h3_id: idx
    for idx, h3_id in enumerate(
        candidatos["id_h3"]
    )
}


# ============================================================
# IMPORTAR H3
# ============================================================

print("\nCargando librería H3...")

try:

    import h3

except ImportError:

    raise ImportError(
        "\nNo está instalada la librería h3.\n"
        "Instalar con:\n\n"
        "pip install h3"
    )


# ============================================================
# FUNCIÓN DE VECINOS
# ============================================================

def obtener_vecinos(h3_id):
    """
    Obtiene vecinos inmediatos del H3.

    Compatible con distintas versiones
    de la librería h3.
    """

    try:

        return list(
            h3.grid_disk(
                h3_id,
                1
            )
        )

    except AttributeError:

        try:

            return list(
                h3.k_ring(
                    h3_id,
                    1
                )
            )

        except AttributeError:

            return []


# ============================================================
# CONSTRUIR GRAFO DE CORREDORES
# ============================================================

print(
    "\nConstruyendo grafo de continuidad..."
)


adyacencias = {
    idx: set()
    for idx in range(
        len(candidatos)
    )
}


for idx, fila in candidatos.iterrows():

    h3_id = fila["id_h3"]

    vecinos = obtener_vecinos(
        h3_id
    )

    for vecino in vecinos:

        if vecino == h3_id:
            continue

        vecino_idx = id_to_idx.get(
            vecino
        )

        if vecino_idx is None:
            continue

        fila_vecino = candidatos.iloc[
            vecino_idx
        ]

        # ----------------------------------------------------
        # COMPATIBILIDAD HORARIA
        # ----------------------------------------------------

        diff_hora = diferencia_horaria(
            fila["hora_pico"],
            fila_vecino["hora_pico"]
        )

        if (
            diff_hora
            > DIFERENCIA_HORARIA_MAXIMA
        ):
            continue

        # ----------------------------------------------------
        # COMPATIBILIDAD MODAL
        # ----------------------------------------------------

        compat_modal = (
            compatibilidad_modal(
                fila["modo_dominante"],
                fila_vecino["modo_dominante"]
            )
        )

        # No conectamos modos completamente diferentes.
        if compat_modal == 0:
            continue

        # ----------------------------------------------------
        # CONECTAR
        # ----------------------------------------------------

        adyacencias[idx].add(
            vecino_idx
        )

        adyacencias[
            vecino_idx
        ].add(idx)


# ============================================================
# COMPONENTES CONEXAS
# ============================================================

print(
    "\nDetectando componentes de continuidad..."
)


visitados = set()

componentes = []


for inicio in range(
    len(candidatos)
):

    if inicio in visitados:
        continue

    cola = [
        inicio
    ]

    componente = []

    visitados.add(
        inicio
    )

    while cola:

        actual = cola.pop()

        componente.append(
            actual
        )

        for vecino in adyacencias[
            actual
        ]:

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
    f"Componentes detectados: "
    f"{len(componentes):,}"
)


# ============================================================
# FILTRAR COMPONENTES
# ============================================================

componentes_validos = [
    componente
    for componente in componentes
    if len(componente)
    >= MIN_H3_CORREDOR
]


print(
    f"Corredores candidatos: "
    f"{len(componentes_validos):,}"
)


# ============================================================
# CREAR CORREDORES
# ============================================================

corredores = []

h3_corredores = []

cluster_id = 0


operaciones_totales = (
    h3_geo[
        "operaciones_totales"
    ].sum()
)


for componente in componentes_validos:

    cluster_id += 1

    datos = candidatos.iloc[
        componente
    ].copy()

    # --------------------------------------------------------
    # DEMANDA
    # --------------------------------------------------------

    operaciones = (
        datos[
            "operaciones_totales"
        ].sum()
    )

    h3_count = len(
        datos
    )

    superficie = (
        datos[
            "superficie_km2"
        ].sum()
    )

    densidad = (
        operaciones / superficie
        if superficie > 0
        else 0
    )

    # --------------------------------------------------------
    # HORA PICO
    # --------------------------------------------------------

    hora_pico = (
        datos
        .groupby("hora_pico")[
            "operaciones_totales"
        ]
        .sum()
        .idxmax()
    )

    # --------------------------------------------------------
    # MODO DOMINANTE
    # --------------------------------------------------------

    modo_pico = (
        datos
        .groupby("modo_dominante")[
            "operaciones_totales"
        ]
        .sum()
        .idxmax()
    )

    # --------------------------------------------------------
    # CATEGORÍA DOMINANTE
    # --------------------------------------------------------

    categoria = (
        datos
        .groupby("categoria_demanda")[
            "operaciones_totales"
        ]
        .sum()
        .idxmax()
    )

    # --------------------------------------------------------
    # JURISDICCIONES
    # --------------------------------------------------------

    jurisdicciones = sorted(
        datos[
            "jurisdiccion"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    provincias = sorted(
        datos[
            "provincia"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # HORAS COMPATIBLES
    # --------------------------------------------------------

    horas = (
        datos["hora_pico"]
        .dropna()
        .astype(int)
    )

    if len(horas) > 0:

        dispersion_horaria = (
            horas.max()
            - horas.min()
        )

    else:

        dispersion_horaria = None

    # --------------------------------------------------------
    # CONTINUIDAD ESPACIAL
    # --------------------------------------------------------

    posibles_conexiones = (
        h3_count * 6
    )

    conexiones_reales = 0

    for idx in componente:

        conexiones_reales += len(
            adyacencias[idx]
            .intersection(componente)
        )

    conexiones_reales = (
        conexiones_reales / 2
    )

    if posibles_conexiones > 0:

        continuidad_espacial = min(
            conexiones_reales
            / h3_count,
            1
        )

    else:

        continuidad_espacial = 0

    # --------------------------------------------------------
    # CONTINUIDAD HORARIA
    # --------------------------------------------------------

    pares = []

    for idx in componente:

        for vecino in adyacencias[
            idx
        ]:

            if vecino <= idx:
                continue

            if vecino not in componente:
                continue

            hora_a = candidatos.iloc[
                idx
            ]["hora_pico"]

            hora_b = candidatos.iloc[
                vecino
            ]["hora_pico"]

            diff = diferencia_horaria(
                hora_a,
                hora_b
            )

            pares.append(
                diff
            )

    if pares:

        continuidad_horaria = (
            1
            - (
                sum(pares)
                / len(pares)
            )
            / (
                DIFERENCIA_HORARIA_MAXIMA
                + 1
            )
        )

        continuidad_horaria = max(
            0,
            min(
                continuidad_horaria,
                1
            )
        )

    else:

        continuidad_horaria = 0

    # --------------------------------------------------------
    # CONTINUIDAD MODAL
    # --------------------------------------------------------

    pares_modal = []

    for idx in componente:

        for vecino in adyacencias[
            idx
        ]:

            if vecino <= idx:
                continue

            if vecino not in componente:
                continue

            modo_a = candidatos.iloc[
                idx
            ]["modo_dominante"]

            modo_b = candidatos.iloc[
                vecino
            ]["modo_dominante"]

            pares_modal.append(
                compatibilidad_modal(
                    modo_a,
                    modo_b
                )
            )

    if pares_modal:

        continuidad_modal = (
            sum(pares_modal)
            / len(pares_modal)
        )

    else:

        continuidad_modal = 0

    # --------------------------------------------------------
    # DENSIDAD NORMALIZADA
    # --------------------------------------------------------

    corredores.append(
        {
            "cluster_id": cluster_id,
            "h3": h3_count,
            "operaciones": operaciones,
            "superficie_km2": superficie,
            "operaciones_por_km2": densidad,
            "categoria_dominante": categoria,
            "hora_pico": int(hora_pico)
                if not pd.isna(hora_pico)
                else None,
            "modo_dominante": modo_pico,
            "cantidad_jurisdicciones":
                len(jurisdicciones),
            "jurisdicciones":
                " | ".join(jurisdicciones),
            "cantidad_provincias":
                len(provincias),
            "provincias":
                " | ".join(provincias),
            "dispersion_horaria":
                dispersion_horaria,
            "continuidad_espacial":
                continuidad_espacial,
            "continuidad_horaria":
                continuidad_horaria,
            "continuidad_modal":
                continuidad_modal,
        }
    )

    # --------------------------------------------------------
    # H3 → CORREDOR
    # --------------------------------------------------------

    for idx in componente:

        fila = candidatos.iloc[
            idx
        ]

        h3_corredores.append(
            {
                "id_h3":
                    fila["id_h3"],

                "cluster_id":
                    cluster_id,

                "operaciones_totales":
                    fila[
                        "operaciones_totales"
                    ],

                "jurisdiccion":
                    fila["jurisdiccion"],

                "provincia":
                    fila["provincia"],

                "categoria_demanda":
                    fila[
                        "categoria_demanda"
                    ],

                "hora_pico":
                    fila["hora_pico"],

                "modo_dominante":
                    fila[
                        "modo_dominante"
                    ],

                "superficie_km2":
                    fila[
                        "superficie_km2"
                    ],
            }
        )


# ============================================================
# DATAFRAME CORREDORES
# ============================================================

corredores_df = pd.DataFrame(
    corredores
)


if corredores_df.empty:

    raise RuntimeError(
        "No se detectaron corredores válidos."
    )


# ============================================================
# ÍNDICES NORMALIZADOS
# ============================================================

corredores_df[
    "demanda_normalizada"
] = normalizar_serie_minmax(
    corredores_df[
        "operaciones"
    ]
)


corredores_df[
    "densidad_normalizada"
] = normalizar_serie_minmax(
    corredores_df[
        "operaciones_por_km2"
    ]
)


# ============================================================
# ÍNDICE DE CORREDOR
# ============================================================

# Demanda:
# combinación entre operaciones absolutas y densidad.

componente_demanda = (
    0.70
    * corredores_df[
        "demanda_normalizada"
    ]
    +
    0.30
    * corredores_df[
        "densidad_normalizada"
    ]
)


corredores_df[
    "indice_corredor"
] = (
    (
        componente_demanda
        * PESO_DEMANDA
    )
    +
    (
        corredores_df[
            "continuidad_espacial"
        ]
        * PESO_CONTINUIDAD_ESPACIAL
    )
    +
    (
        corredores_df[
            "continuidad_horaria"
        ]
        * PESO_CONTINUIDAD_HORARIA
    )
    +
    (
        corredores_df[
            "continuidad_modal"
        ]
        * PESO_CONTINUIDAD_MODAL
    )
) * 100


# ============================================================
# CLASIFICACIÓN
# ============================================================

corredores_df[
    "categoria_corredor"
] = (
    corredores_df[
        "indice_corredor"
    ]
    .apply(
        clasificar_corredor
    )
)


# ============================================================
# ORDENAMIENTO
# ============================================================

corredores_df = (
    corredores_df
    .sort_values(
        [
            "operaciones",
            "indice_corredor",
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


corredores_df[
    "ranking_operaciones"
] = (
    corredores_df.index + 1
)


corredores_df[
    "pct_operaciones"
] = (
    corredores_df[
        "operaciones"
    ]
    / operaciones_totales
    * 100
)


corredores_df[
    "pct_operaciones_acumulado"
] = (
    corredores_df[
        "pct_operaciones"
    ]
    .cumsum()
)


# ============================================================
# H3 → CORREDOR
# ============================================================

h3_corredores_df = pd.DataFrame(
    h3_corredores
)


h3_corredores_df = (
    h3_corredores_df
    .merge(
        corredores_df[
            [
                "cluster_id",
                "indice_corredor",
                "categoria_corredor",
            ]
        ],
        on="cluster_id",
        how="left"
    )
)


# ============================================================
# JURISDICCIONES
# ============================================================

print(
    "\nCaracterizando jurisdicciones..."
)


jurisdicciones = (
    h3_corredores_df
    .groupby(
        [
            "cluster_id",
            "jurisdiccion",
            "provincia",
        ],
        dropna=False
    )
    .agg(
        h3=(
            "id_h3",
            "count"
        ),

        operaciones=(
            "operaciones_totales",
            "sum"
        ),

        hora_pico=(
            "hora_pico",
            lambda x:
                x.mode().iloc[0]
                if not x.mode().empty
                else None
        ),

        modo_dominante=(
            "modo_dominante",
            lambda x:
                x.mode().iloc[0]
                if not x.mode().empty
                else None
        ),
    )
    .reset_index()
)


jurisdicciones[
    "pct_operaciones_corredor"
] = (
    jurisdicciones
    .groupby(
        "cluster_id"
    )["operaciones"]
    .transform(
        lambda x:
            x / x.sum() * 100
    )
)


jurisdicciones = (
    jurisdicciones
    .sort_values(
        [
            "cluster_id",
            "operaciones",
        ],
        ascending=[
            True,
            False,
        ]
    )
)


# ============================================================
# TOP 30
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TOP 30 CORREDORES DE MOVILIDAD"
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
    "indice_corredor",
    "categoria_corredor",
    "categoria_dominante",
    "hora_pico",
    "modo_dominante",
    "cantidad_jurisdicciones",
    "jurisdicciones",
]


print(
    corredores_df[
        columnas_top
    ]
    .head(TOP_CORREDORES)
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
    "DISTRIBUCIÓN DE CORREDORES"
)

print(
    "=" * 70
)


print(
    corredores_df[
        "categoria_corredor"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# HORAS PICO
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "HORAS PICO DE LOS CORREDORES"
)

print(
    "=" * 70
)


print(
    corredores_df[
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
    corredores_df[
        "modo_dominante"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# JURISDICCIONES MÁS ATRAVESADAS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "JURISDICCIONES MÁS PRESENTES EN CORREDORES"
)

print(
    "=" * 70
)


jurisdicciones_count = (
    h3_corredores_df[
        [
            "jurisdiccion",
            "provincia",
        ]
    ]
    .value_counts()
    .head(30)
)


print(
    jurisdicciones_count.to_string()
)


# ============================================================
# RESUMEN EJECUTIVO
# ============================================================

h3_clusterizados = len(
    h3_corredores_df
)

operaciones_clusterizadas = (
    h3_corredores_df[
        "operaciones_totales"
    ].sum()
)


pct_clusterizadas = (
    operaciones_clusterizadas
    / operaciones_totales
    * 100
    if operaciones_totales > 0
    else 0
)


corredor_principal = (
    corredores_df.iloc[0]
)


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
    f"Corredores detectados: "
    f"{len(corredores_df):,}"
)


print(
    f"H3 dentro de corredores: "
    f"{h3_clusterizados:,}"
)


print(
    f"Operaciones en corredores: "
    f"{operaciones_clusterizadas:,.0f}"
)


print(
    f"% operaciones en corredores: "
    f"{pct_clusterizadas:.2f}%"
)


print(
    "\nCORREDOR PRINCIPAL"
)


print(
    f"Cluster: "
    f"{int(corredor_principal['cluster_id'])}"
)


print(
    f"H3: "
    f"{int(corredor_principal['h3']):,}"
)


print(
    f"Operaciones: "
    f"{corredor_principal['operaciones']:,.0f}"
)


print(
    f"Participación: "
    f"{corredor_principal['pct_operaciones']:.2f}%"
)


print(
    f"Índice: "
    f"{corredor_principal['indice_corredor']:.2f}/100"
)


print(
    f"Categoría: "
    f"{corredor_principal['categoria_corredor']}"
)


print(
    f"Hora pico: "
    f"{int(corredor_principal['hora_pico'])}"
)


print(
    f"Modo: "
    f"{corredor_principal['modo_dominante']}"
)


print(
    f"Jurisdicciones: "
    f"{int(corredor_principal['cantidad_jurisdicciones'])}"
)


print(
    f"\nJurisdicciones:"
)


print(
    corredor_principal[
        "jurisdicciones"
    ]
)


# ============================================================
# GUARDAR CORREDORES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "GUARDANDO RESULTADOS"
)

print(
    "=" * 70
)


corredores_df.to_parquet(
    OUTPUT_CORREDORES,
    index=False
)


h3_corredores_df.to_parquet(
    OUTPUT_H3_CORREDORES,
    index=False
)


jurisdicciones.to_parquet(
    OUTPUT_JURISDICCIONES,
    index=False
)


# ============================================================
# RESUMEN JSON
# ============================================================

top_corredores_json = []

for _, fila in (
    corredores_df
    .head(20)
    .iterrows()
):

    top_corredores_json.append(
        {
            "ranking":
                int(
                    fila[
                        "ranking_operaciones"
                    ]
                ),

            "cluster_id":
                int(
                    fila[
                        "cluster_id"
                    ]
                ),

            "h3":
                int(
                    fila["h3"]
                ),

            "operaciones":
                float(
                    fila[
                        "operaciones"
                    ]
                ),

            "pct_operaciones":
                float(
                    fila[
                        "pct_operaciones"
                    ]
                ),

            "indice_corredor":
                float(
                    fila[
                        "indice_corredor"
                    ]
                ),

            "categoria":
                fila[
                    "categoria_corredor"
                ],

            "hora_pico":
                int(
                    fila[
                        "hora_pico"
                    ]
                )
                if not pd.isna(
                    fila[
                        "hora_pico"
                    ]
                )
                else None,

            "modo":
                fila[
                    "modo_dominante"
                ],

            "jurisdicciones":
                fila[
                    "jurisdicciones"
                ],
        }
    )


resumen = {

    "dataset":
        "SUBE 2025",

    "h3_analizados":
        int(
            len(h3_geo)
        ),

    "operaciones_totales":
        float(
            operaciones_totales
        ),

    "percentil_demanda":
        PERCENTIL_DEMANDA,

    "umbral_demanda":
        float(
            umbral_demanda
        ),

    "h3_candidatos":
        int(
            len(candidatos)
        ),

    "corredores_detectados":
        int(
            len(corredores_df)
        ),

    "h3_en_corredores":
        int(
            h3_clusterizados
        ),

    "operaciones_en_corredores":
        float(
            operaciones_clusterizadas
        ),

    "pct_operaciones_en_corredores":
        float(
            pct_clusterizadas
        ),

    "parametros": {

        "diferencia_horaria_maxima":
            DIFERENCIA_HORARIA_MAXIMA,

        "min_h3_corredor":
            MIN_H3_CORREDOR,

        "peso_demanda":
            PESO_DEMANDA,

        "peso_continuidad_espacial":
            PESO_CONTINUIDAD_ESPACIAL,

        "peso_continuidad_horaria":
            PESO_CONTINUIDAD_HORARIA,

        "peso_continuidad_modal":
            PESO_CONTINUIDAD_MODAL,
    },

    "distribucion_categorias":
        {
            str(k): int(v)
            for k, v in (
                corredores_df[
                    "categoria_corredor"
                ]
                .value_counts()
                .items()
            )
        },

    "distribucion_horas":
        {
            str(k): int(v)
            for k, v in (
                corredores_df[
                    "hora_pico"
                ]
                .value_counts()
                .items()
            )
        },

    "distribucion_modos":
        {
            str(k): int(v)
            for k, v in (
                corredores_df[
                    "modo_dominante"
                ]
                .value_counts()
                .items()
            )
        },

    "top_corredores":
        top_corredores_json,
}


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
    "\nCorredores:"
)

print(
    OUTPUT_CORREDORES
)


print(
    "\nH3 → corredor:"
)

print(
    OUTPUT_H3_CORREDORES
)


print(
    "\nJurisdicciones:"
)

print(
    OUTPUT_JURISDICCIONES
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
    "ANÁLISIS DE CORREDORES FINALIZADO"
)

print(
    "=" * 70
)


print(
    f"H3 analizados: "
    f"{len(h3_geo):,}"
)


print(
    f"Corredores: "
    f"{len(corredores_df):,}"
)


print(
    f"Operaciones en corredores: "
    f"{operaciones_clusterizadas:,.0f}"
)


print(
    f"% operaciones: "
    f"{pct_clusterizadas:.2f}%"
)