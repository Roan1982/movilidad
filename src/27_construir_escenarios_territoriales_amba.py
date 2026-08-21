# -*- coding: utf-8 -*-

"""
========================================================================================
27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - V7.2
========================================================================================

Entrada:
    data/processed/cartera_proyectos_amba/cartera_proyectos_amba.parquet

Salida:
    data/processed/escenarios_territoriales_amba/

Objetivo:
    Construir escenarios territoriales a partir de la cartera consolidada
    del proceso 26.

Principios:
    - No crear valores artificiales.
    - Utilizar indicadores reales de la cartera.
    - Separar componente espacial y territorial.
    - Evaluar K entre 6 y 12.
    - Garantizar asignación completa de escenarios.
    - No utilizar columnas antes de crearlas.
    - Mantener trazabilidad del proceso.
    - Generar salidas reproducibles.
========================================================================================
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ======================================================================================
# CONFIGURACIÓN
# ======================================================================================

VERSION = "V7.2"

K_MIN = 6
K_MAX = 12

MIN_PROYECTOS = 8

PESO_ESPACIAL = 0.55
PESO_TERRITORIAL = 0.45

PESO_NECESIDAD = 0.40
PESO_ESTRATEGICA = 0.60

RANDOM_STATE = 42
N_INIT = 30

PROYECTO = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROYECTO
    / "data"
    / "processed"
    / "cartera_proyectos_amba"
    / "cartera_proyectos_amba.parquet"
)

OUTPUT_DIR = (
    PROYECTO
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================================
# UTILIDADES
# ======================================================================================

def titulo(texto: str, numero: Optional[int] = None) -> None:
    print()
    print("=" * 88)

    if numero is not None:
        print(f"{numero}. {texto}")
    else:
        print(texto)

    print("=" * 88)


def normalizar_nombre(nombre: str) -> str:
    """
    Normaliza nombres de columnas para comparación robusta.
    """
    return (
        str(nombre)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def encontrar_columna(
    df: pd.DataFrame,
    aliases: List[str],
    obligatorio: bool = True,
) -> Optional[str]:
    """
    Busca una columna utilizando aliases.
    """

    mapa = {
        normalizar_nombre(col): col
        for col in df.columns
    }

    for alias in aliases:
        clave = normalizar_nombre(alias)

        if clave in mapa:
            return mapa[clave]

    if obligatorio:
        raise ValueError(
            "\nNo se encontró una columna requerida.\n"
            f"Aliases buscados: {aliases}\n"
            f"Columnas disponibles:\n{list(df.columns)}"
        )

    return None


def convertir_numerico(
    serie: pd.Series,
    nombre: str,
) -> pd.Series:

    resultado = pd.to_numeric(
        serie,
        errors="coerce",
    )

    if resultado.isna().all():
        raise ValueError(
            f"La columna '{nombre}' no contiene valores numéricos utilizables."
        )

    return resultado


def normalizar_minmax(
    serie: pd.Series,
) -> pd.Series:

    serie = convertir_numerico(
        serie,
        str(serie.name),
    )

    minimo = serie.min()
    maximo = serie.max()

    if pd.isna(minimo) or pd.isna(maximo):
        raise ValueError(
            f"No fue posible normalizar '{serie.name}'."
        )

    if math.isclose(float(minimo), float(maximo)):
        return pd.Series(
            np.full(len(serie), 0.5),
            index=serie.index,
            dtype=float,
        )

    return (
        (serie - minimo)
        / (maximo - minimo)
    ).astype(float)


def calcular_equidad(tamanos: np.ndarray) -> float:
    """
    Equidad basada en el tamaño de clusters.

    1 = distribución perfectamente uniforme.
    0 = distribución extremadamente desigual.
    """

    if len(tamanos) == 0:
        return 0.0

    esperado = len(tamanos) / len(tamanos)

    if esperado == 0:
        return 0.0

    desviacion = np.std(tamanos)

    return float(
        max(
            0.0,
            1.0 - desviacion / esperado,
        )
    )


# ======================================================================================
# RESOLUCIÓN DE INDICADORES
# ======================================================================================

INDICADORES = {

    "demanda": [
        "indice_demanda_estructural",
        "demanda",
        "score_demanda",
        "demanda_estructural",
    ],

    "deficit": [
        "deficit_infraestructura",
        "deficit",
        "score_deficit",
    ],

    "conectividad": [
        "indice_conectividad_estructural",
        "conectividad",
        "score_conectividad",
    ],

    "intermodalidad": [
        "indice_intermodalidad_estructural",
        "intermodalidad",
        "score_intermodalidad",
    ],

    "integracion": [
        "indice_integracion_territorial",
        "integracion_territorial",
        "integracion",
        "score_integracion_territorial",
    ],

    "centralidad": [
        "indice_centralidad_estructural",
        "centralidad",
        "score_centralidad",
    ],

    "impacto": [
        "impacto_potencial",
        "impacto",
        "score_impacto",
    ],

    "urgencia": [
        "urgencia_intervencion",
        "urgencia",
        "score_urgencia",
    ],

    "score_cartera": [
        "score_cartera",
        "score_global",
        "score_prioridad",
    ],
}


def resolver_indicadores(
    gdf: gpd.GeoDataFrame,
) -> Dict[str, str]:

    titulo("RESOLVIENDO INDICADORES REALES", 3)

    columnas = {}

    for indicador, aliases in INDICADORES.items():

        columna = encontrar_columna(
            gdf,
            aliases,
            obligatorio=True,
        )

        columnas[indicador] = columna

        print(
            f"  {indicador:<22} OK -> {columna}"
        )

    return columnas


# ======================================================================================
# VALIDACIÓN
# ======================================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo("VALIDANDO DATOS DE ENTRADA", 2)

    if gdf.empty:
        raise ValueError(
            "La cartera está vacía."
        )

    nulos = int(
        gdf.geometry.isna().sum()
    )

    vacios = int(
        gdf.geometry.is_empty.sum()
    )

    invalidos = int(
        (~gdf.geometry.is_valid).sum()
    )

    print(
        f"Geometrías nulas     : {nulos}"
    )

    print(
        f"Geometrías vacías    : {vacios}"
    )

    print(
        f"Geometrías inválidas : {invalidos}"
    )

    if nulos:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if vacios:
        raise ValueError(
            "Existen geometrías vacías."
        )

    if invalidos:
        raise ValueError(
            "Existen geometrías inválidas."
        )

    print(
        f"Proyectos válidos : {len(gdf)}"
    )

    print(
        "Validación de entrada: OK"
    )

    return gdf


# ======================================================================================
# COMPONENTE ESPACIAL
# ======================================================================================

def preparar_componente_espacial(
    gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:

    titulo("PREPARANDO COMPONENTE ESPACIAL", 5)

    if gdf.crs is None:
        raise ValueError(
            "La capa no tiene CRS definido."
        )

    if gdf.crs.is_geographic:

        gdf_metric = gdf.to_crs(
            "EPSG:3857"
        )

    else:

        gdf_metric = gdf.copy()

    centroids = gdf_metric.geometry.centroid

    x = centroids.x.to_numpy(
        dtype=float
    )

    y = centroids.y.to_numpy(
        dtype=float
    )

    print(
        f"X: {x.min():,.2f} -> {x.max():,.2f}"
    )

    print(
        f"Y: {y.min():,.2f} -> {y.max():,.2f}"
    )

    coords = np.column_stack(
        [
            x,
            y,
        ]
    )

    return gdf_metric, coords


# ======================================================================================
# MATRIZ MULTICRITERIO
# ======================================================================================

def construir_matriz_multicriterio(
    gdf: gpd.GeoDataFrame,
    columnas: Dict[str, str],
    coords: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:

    titulo(
        "CONSTRUYENDO MATRIZ MULTICRITERIO V7.2",
        6,
    )

    n = len(gdf)

    # ------------------------------------------------------------------
    # Indicadores territoriales
    # ------------------------------------------------------------------

    demanda = normalizar_minmax(
        gdf[columnas["demanda"]]
    ).to_numpy()

    deficit = normalizar_minmax(
        gdf[columnas["deficit"]]
    ).to_numpy()

    conectividad = normalizar_minmax(
        gdf[columnas["conectividad"]]
    ).to_numpy()

    intermodalidad = normalizar_minmax(
        gdf[columnas["intermodalidad"]]
    ).to_numpy()

    integracion = normalizar_minmax(
        gdf[columnas["integracion"]]
    ).to_numpy()

    centralidad = normalizar_minmax(
        gdf[columnas["centralidad"]]
    ).to_numpy()

    impacto = normalizar_minmax(
        gdf[columnas["impacto"]]
    ).to_numpy()

    urgencia = normalizar_minmax(
        gdf[columnas["urgencia"]]
    ).to_numpy()

    score_cartera = normalizar_minmax(
        gdf[columnas["score_cartera"]]
    ).to_numpy()

    # ------------------------------------------------------------------
    # Normalización espacial
    # ------------------------------------------------------------------

    coords_std = StandardScaler().fit_transform(
        coords
    )

    # Magnitud espacial.
    #
    # No reemplaza la ubicación real.
    # Se utiliza para permitir que KMeans considere
    # simultáneamente posición y atributos.
    #
    # El componente espacial entra con el peso global
    # PESO_ESPACIAL.
    #
    spatial_x = coords_std[:, 0]
    spatial_y = coords_std[:, 1]

    # ------------------------------------------------------------------
    # Matriz territorial
    # ------------------------------------------------------------------

    pesos = {
        "demanda": 0.0900,
        "deficit": 0.0900,
        "conectividad": 0.05625,
        "intermodalidad": 0.05625,
        "integracion": 0.0450,
        "centralidad": 0.03375,
        "impacto": 0.03375,
        "urgencia": 0.0450,
    }

    matriz_territorial = np.column_stack(
        [
            demanda,
            deficit,
            conectividad,
            intermodalidad,
            integracion,
            centralidad,
            impacto,
            urgencia,
        ]
    )

    # ------------------------------------------------------------------
    # Matriz final
    # ------------------------------------------------------------------

    # Se mantiene la componente espacial explícita.
    matriz = np.column_stack(
        [
            spatial_x * PESO_ESPACIAL,
            spatial_y * PESO_ESPACIAL,
            matriz_territorial * PESO_TERRITORIAL,
        ]
    )

    print(
        f"Proyectos             : {n}"
    )

    print(
        f"Variables             : {matriz.shape[1]}"
    )

    print(
        f"Peso espacial         : {PESO_ESPACIAL:.0%}"
    )

    print(
        f"Peso territorial      : {PESO_TERRITORIAL:.0%}"
    )

    print()

    for nombre, peso in pesos.items():

        print(
            f"  {nombre:<20} {peso:.4f}"
        )

    return matriz, pesos


# ======================================================================================
# SELECCIÓN DE K
# ======================================================================================

def seleccionar_k(
    matriz: np.ndarray,
) -> Tuple[int, pd.DataFrame, np.ndarray]:

    titulo("SELECCIONANDO K V7.2", 7)

    resultados = []

    modelos = {}

    for k in range(
        K_MIN,
        K_MAX + 1,
    ):

        modelo = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )

        labels = modelo.fit_predict(
            matriz
        )

        tamanos = np.bincount(
            labels,
            minlength=k,
        )

        minimo = int(
            tamanos.min()
        )

        maximo = int(
            tamanos.max()
        )

        if len(
            np.unique(labels)
        ) > 1:

            silhouette = float(
                silhouette_score(
                    matriz,
                    labels,
                )
            )

            db = float(
                davies_bouldin_score(
                    matriz,
                    labels,
                )
            )

        else:

            silhouette = -1.0
            db = float("inf")

        equidad = calcular_equidad(
            tamanos
        )

        # Penalización por clusters que incumplen mínimo.
        penalizacion_minimo = (
            0.0
            if minimo >= MIN_PROYECTOS
            else 0.20
        )

        # Mejor silhouette y mejor DB.
        sil_score = (
            (silhouette + 1.0)
            / 2.0
        )

        db_score = (
            1.0
            / (1.0 + db)
        )

        score = (
            0.45 * sil_score
            + 0.25 * db_score
            + 0.30 * equidad
            - penalizacion_minimo
        )

        resultados.append(
            {
                "k_solicitado": k,
                "k_final": k,
                "silhouette": silhouette,
                "davies_bouldin": db,
                "min_proyectos": minimo,
                "max_proyectos": maximo,
                "equidad": equidad,
                "score": score,
            }
        )

        modelos[k] = (
            modelo,
            labels,
        )

        print(
            f"K={k:2d} | "
            f"K final={k:2d} | "
            f"sil={silhouette:.4f} | "
            f"DB={db:.4f} | "
            f"mín={minimo:2d} | "
            f"máx={maximo:2d} | "
            f"eq={equidad:.3f} | "
            f"score={score:.4f}"
        )

    resultados_df = pd.DataFrame(
        resultados
    )

    candidatos = resultados_df[
        resultados_df["min_proyectos"]
        >= MIN_PROYECTOS
    ]

    if candidatos.empty:

        raise ValueError(
            "Ningún K cumple el mínimo de proyectos por cluster."
        )

    mejor = candidatos.sort_values(
        [
            "score",
            "silhouette",
            "equidad",
        ],
        ascending=False,
    ).iloc[0]

    k_seleccionado = int(
        mejor["k_solicitado"]
    )

    modelo, labels = modelos[
        k_seleccionado
    ]

    print()

    print(
        f"K solicitado seleccionado : "
        f"{k_seleccionado}"
    )

    print(
        f"K final                   : "
        f"{k_seleccionado}"
    )

    print(
        f"Score                     : "
        f"{mejor['score']:.4f}"
    )

    return (
        k_seleccionado,
        resultados_df,
        labels,
    )


# ======================================================================================
# MÉTRICAS DE COHESIÓN
# ======================================================================================

def construir_metricas_cohesion(
    matriz: np.ndarray,
    labels: np.ndarray,
    modelo: Optional[KMeans] = None,
) -> pd.DataFrame:

    titulo("MÉTRICAS DE COHESIÓN", 10)

    k = len(
        np.unique(labels)
    )

    if k > 1:

        silhouette_individual = (
            __import__(
                "sklearn.metrics",
                fromlist=["silhouette_samples"],
            )
            .silhouette_samples(
                matriz,
                labels,
            )
        )

    else:

        silhouette_individual = np.zeros(
            len(labels)
        )

    rows = []

    if modelo is not None:

        centroides = modelo.cluster_centers_

    else:

        centroides = np.vstack(
            [
                matriz[labels == cluster].mean(axis=0)
                for cluster in sorted(
                    np.unique(labels)
                )
            ]
        )

    for cluster in sorted(
        np.unique(labels)
    ):

        mask = labels == cluster

        datos = matriz[
            mask
        ]

        centroide = centroides[
            cluster
        ]

        distancias = np.linalg.norm(
            datos - centroide,
            axis=1,
        )

        sil = silhouette_individual[
            mask
        ]

        rows.append(
            {
                "cluster_territorial": cluster + 1,
                "cantidad_proyectos": int(
                    mask.sum()
                ),
                "silhouette_promedio": float(
                    np.mean(sil)
                ),
                "silhouette_minima": float(
                    np.min(sil)
                ),
                "silhouette_maxima": float(
                    np.max(sil)
                ),
                "distancia_centroide_promedio": float(
                    np.mean(distancias)
                ),
                "distancia_centroide_maxima": float(
                    np.max(distancias)
                ),
                "dispersion_std": float(
                    np.std(distancias)
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    print(
        df.to_string(
            index=False
        )
    )

    return df


# ======================================================================================
# SCORE TERRITORIAL
# ======================================================================================

def calcular_scores_proyecto(
    gdf: gpd.GeoDataFrame,
    columnas: Dict[str, str],
) -> pd.DataFrame:

    demanda = normalizar_minmax(
        gdf[columnas["demanda"]]
    )

    deficit = normalizar_minmax(
        gdf[columnas["deficit"]]
    )

    conectividad = normalizar_minmax(
        gdf[columnas["conectividad"]]
    )

    intermodalidad = normalizar_minmax(
        gdf[columnas["intermodalidad"]]
    )

    integracion = normalizar_minmax(
        gdf[columnas["integracion"]]
    )

    centralidad = normalizar_minmax(
        gdf[columnas["centralidad"]]
    )

    impacto = normalizar_minmax(
        gdf[columnas["impacto"]]
    )

    urgencia = normalizar_minmax(
        gdf[columnas["urgencia"]]
    )

    score_cartera = normalizar_minmax(
        gdf[columnas["score_cartera"]]
    )

    necesidad = (
        0.40 * demanda
        + 0.30 * deficit
        + 0.20 * urgencia
        + 0.10 * integracion
    )

    estrategica = (
        0.30 * conectividad
        + 0.20 * intermodalidad
        + 0.15 * centralidad
        + 0.15 * impacto
        + 0.20 * score_cartera
    )

    score_final = (
        PESO_NECESIDAD * necesidad
        + PESO_ESTRATEGICA * estrategica
    )

    return pd.DataFrame(
        {
            "_demanda_norm": demanda,
            "_deficit_norm": deficit,
            "_conectividad_norm": conectividad,
            "_intermodalidad_norm": intermodalidad,
            "_integracion_norm": integracion,
            "_centralidad_norm": centralidad,
            "_impacto_norm": impacto,
            "_urgencia_norm": urgencia,
            "_score_cartera_norm": score_cartera,
            "score_necesidad": necesidad,
            "score_estrategico": estrategica,
            "score_proyecto": score_final,
        },
        index=gdf.index,
    )


# ======================================================================================
# DIMENSIÓN DOMINANTE
# ======================================================================================

def calcular_dimension_dominante(
    df: pd.DataFrame,
) -> pd.Series:

    dimensiones = {
        "DEMANDA": "_demanda_norm",
        "DEFICIT": "_deficit_norm",
        "CONECTIVIDAD": "_conectividad_norm",
        "INTERMODALIDAD": "_intermodalidad_norm",
        "INTEGRACION": "_integracion_norm",
        "CENTRALIDAD": "_centralidad_norm",
        "IMPACTO": "_impacto_norm",
        "URGENCIA": "_urgencia_norm",
    }

    matriz = pd.DataFrame(
        {
            nombre: df[columna]
            for nombre, columna in dimensiones.items()
        },
        index=df.index,
    )

    return matriz.idxmax(
        axis=1
    )


# ======================================================================================
# CLASIFICACIÓN DE PRIORIDAD
# ======================================================================================

def clasificar_prioridad(
    score: float,
) -> str:

    if score >= 0.80:
        return "PRIORIDAD_1_MUY_ALTA"

    if score >= 0.65:
        return "PRIORIDAD_2_MEDIA_ALTA"

    if score >= 0.50:
        return "PRIORIDAD_3_MEDIA"

    return "PRIORIDAD_4_MEDIA_BAJA"


# ======================================================================================
# CONSTRUCCIÓN DE ESCENARIOS
# ======================================================================================

def construir_escenarios(
    gdf: gpd.GeoDataFrame,
    labels: np.ndarray,
    scores: pd.DataFrame,
) -> Tuple[
    gpd.GeoDataFrame,
    pd.DataFrame,
]:

    titulo(
        "CONSTRUYENDO ESCENARIOS TERRITORIALES",
        8,
    )

    trabajo = gdf.copy()

    # ------------------------------------------------------------------
    # ESTA ES LA CORRECCIÓN FUNDAMENTAL DEL ERROR ANTERIOR
    #
    # Primero se crea el cluster territorial.
    # Después se crea escenario_id.
    # Nunca se accede a escenario_id antes de crearlo.
    # ------------------------------------------------------------------

    trabajo["_cluster_territorial"] = (
        labels.astype(int) + 1
    )

    # Incorporar scores.
    for columna in scores.columns:
        trabajo[columna] = scores[
            columna
        ]

    trabajo[
        "dimension_dominante_proyecto"
    ] = calcular_dimension_dominante(
        trabajo
    )

    # ------------------------------------------------------------------
    # Estadísticas por cluster
    # ------------------------------------------------------------------

    escenarios = []

    for cluster in sorted(
        trabajo["_cluster_territorial"].unique()
    ):

        subset = trabajo[
            trabajo["_cluster_territorial"]
            == cluster
        ]

        cantidad = len(
            subset
        )

        score_promedio = float(
            subset[
                "score_proyecto"
            ].mean()
        )

        necesidad_promedio = float(
            subset[
                "score_necesidad"
            ].mean()
        )

        estrategico_promedio = float(
            subset[
                "score_estrategico"
            ].mean()
        )

        dimensiones = (
            subset[
                "dimension_dominante_proyecto"
            ]
            .value_counts()
        )

        if len(dimensiones):

            dimension_dominante = (
                dimensiones.index[0]
            )

        else:

            dimension_dominante = (
                "SIN_DIMENSION"
            )

        score_escenario = (
            100.0
            * (
                0.40
                * necesidad_promedio
                + 0.60
                * estrategico_promedio
            )
        )

        prioridad = clasificar_prioridad(
            score_promedio
        )

        if (
            necesidad_promedio >= 0.60
            and estrategico_promedio >= 0.60
        ):

            tipo = (
                "ESCENARIO_INTEGRADO"
            )

        elif necesidad_promedio >= estrategico_promedio:

            tipo = (
                "ESCENARIO_DE_NECESIDAD"
            )

        else:

            tipo = (
                "ESCENARIO_ESTRATEGICO"
            )

        escenarios.append(
            {
                "cluster_territorial": cluster,
                "cantidad_proyectos": cantidad,
                "score_escenario": score_escenario,
                "score_necesidad": necesidad_promedio,
                "score_estrategico": estrategico_promedio,
                "tipo_escenario": tipo,
                "dimension_dominante": dimension_dominante,
                "prioridad_escenario": prioridad,
            }
        )

    escenarios_df = pd.DataFrame(
        escenarios
    )

    # ------------------------------------------------------------------
    # Orden definitivo de escenarios
    #
    # Se ordenan por score descendente.
    # Esto produce AMBA-E001 como escenario
    # de mayor prioridad.
    # ------------------------------------------------------------------

    escenarios_df = (
        escenarios_df
        .sort_values(
            [
                "score_escenario",
                "cantidad_proyectos",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    escenarios_df[
        "escenario_id"
    ] = [
        f"AMBA-E{i:03d}"
        for i in range(
            1,
            len(escenarios_df) + 1,
        )
    ]

    # ------------------------------------------------------------------
    # Mapa cluster -> escenario
    # ------------------------------------------------------------------

    mapa_escenarios = dict(
        zip(
            escenarios_df[
                "cluster_territorial"
            ],
            escenarios_df[
                "escenario_id"
            ],
        )
    )

    # ------------------------------------------------------------------
    # ASIGNACIÓN A PROYECTOS
    # ------------------------------------------------------------------

    trabajo[
        "escenario_id"
    ] = trabajo[
        "_cluster_territorial"
    ].map(
        mapa_escenarios
    )

    # ------------------------------------------------------------------
    # Validación inmediata
    # ------------------------------------------------------------------

    if trabajo[
        "escenario_id"
    ].isna().any():

        faltantes = trabajo[
            trabajo[
                "escenario_id"
            ].isna()
        ]

        raise RuntimeError(
            "Hay proyectos sin escenario asignado. "
            f"Cantidad: {len(faltantes)}"
        )

    # ------------------------------------------------------------------
    # Incorporar atributos del escenario
    # ------------------------------------------------------------------

    atributos = escenarios_df[
        [
            "escenario_id",
            "score_escenario",
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
        ]
    ].copy()

    trabajo = trabajo.merge(
        atributos,
        on="escenario_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_escenario"),
    )

    # ------------------------------------------------------------------
    # Renombrar dimensión dominante
    # ------------------------------------------------------------------

    trabajo[
        "dimension_dominante"
    ] = trabajo[
        "dimension_dominante_escenario"
    ]

    trabajo.drop(
        columns=[
            "dimension_dominante_escenario"
        ],
        inplace=True,
        errors="ignore",
    )

    # ------------------------------------------------------------------
    # Orden de salida
    # ------------------------------------------------------------------

    escenarios_df = escenarios_df[
        [
            "escenario_id",
            "cluster_territorial",
            "cantidad_proyectos",
            "score_escenario",
            "score_necesidad",
            "score_estrategico",
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
        ]
    ]

    escenarios_df = escenarios_df.sort_values(
        "score_escenario",
        ascending=False,
    ).reset_index(
        drop=True
    )

    print()

    print(
        escenarios_df[
            [
                "escenario_id",
                "cantidad_proyectos",
                "score_escenario",
                "tipo_escenario",
                "dimension_dominante",
                "prioridad_escenario",
            ]
        ].to_string(
            index=False
        )
    )

    return (
        trabajo,
        escenarios_df,
    )


# ======================================================================================
# VALIDACIÓN DE ASIGNACIÓN
# ======================================================================================

def validar_asignacion(
    gdf: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
) -> None:

    titulo(
        "VALIDANDO ASIGNACIÓN DE ESCENARIOS",
        9,
    )

    if "escenario_id" not in gdf.columns:

        raise RuntimeError(
            "La columna escenario_id no existe."
        )

    total = len(
        gdf
    )

    asignados = int(
        gdf[
            "escenario_id"
        ].notna().sum()
    )

    unicos = (
        gdf[
            "escenario_id"
        ]
        .dropna()
        .nunique()
    )

    esperados = len(
        escenarios
    )

    print(
        f"Proyectos totales       : {total}"
    )

    print(
        f"Proyectos asignados     : {asignados}"
    )

    print(
        f"Proyectos sin escenario : {total - asignados}"
    )

    print(
        f"Escenarios construidos  : {esperados}"
    )

    print(
        f"Escenarios utilizados   : {unicos}"
    )

    if total != asignados:

        raise RuntimeError(
            "NO todos los proyectos tienen escenario asignado."
        )

    if unicos != esperados:

        raise RuntimeError(
            "No todos los escenarios tienen proyectos asignados."
        )

    conteo = (
        gdf[
            "escenario_id"
        ]
        .value_counts()
        .sort_index()
    )

    for escenario_id, cantidad in conteo.items():

        if cantidad < MIN_PROYECTOS:

            raise RuntimeError(
                f"{escenario_id} tiene solamente "
                f"{cantidad} proyectos."
            )

    print(
        "Validación de asignación: OK"
    )


# ======================================================================================
# LIMPIEZA DE COLUMNAS TÉCNICAS
# ======================================================================================

def limpiar_columnas_tecnicas(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    columnas_tecnicas = [
        columna
        for columna in gdf.columns
        if columna.startswith("_")
    ]

    return gdf.drop(
        columns=columnas_tecnicas,
        errors="ignore",
    )


# ======================================================================================
# EXPORTACIÓN
# ======================================================================================

def exportar_resultados(
    gdf: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
    cohesion: pd.DataFrame,
    columnas_indicadores: Dict[str, str],
) -> None:

    titulo(
        "EXPORTANDO RESULTADOS",
        11,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # GeoParquet principal
    # ------------------------------------------------------------------

    gdf_final = limpiar_columnas_tecnicas(
        gdf
    )

    salida_gpkg = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.parquet"
    )

    gdf_final.to_parquet(
        salida_gpkg,
        index=False,
    )

    print(
        f"GeoParquet             : {salida_gpkg}"
    )

    # ------------------------------------------------------------------
    # Cartera de escenarios
    # ------------------------------------------------------------------

    salida_escenarios = (
        OUTPUT_DIR
        / "cartera_escenarios_territoriales_amba.parquet"
    )

    escenarios.to_parquet(
        salida_escenarios,
        index=False,
    )

    print(
        f"Cartera escenarios     : {salida_escenarios}"
    )

    # ------------------------------------------------------------------
    # CSV escenarios
    # ------------------------------------------------------------------

    salida_csv = (
        OUTPUT_DIR
        / "cartera_escenarios_territoriales_amba.csv"
    )

    escenarios.to_csv(
        salida_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"CSV escenarios          : {salida_csv}"
    )

    # ------------------------------------------------------------------
    # Evaluación K
    # ------------------------------------------------------------------

    salida_k = (
        OUTPUT_DIR
        / "evaluacion_k.csv"
    )

    evaluacion_k.to_csv(
        salida_k,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Evaluación K            : {salida_k}"
    )

    # ------------------------------------------------------------------
    # Cohesión
    # ------------------------------------------------------------------

    salida_cohesion = (
        OUTPUT_DIR
        / "metricas_cohesion.csv"
    )

    cohesion.to_csv(
        salida_cohesion,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Métricas cohesión      : {salida_cohesion}"
    )

    # ------------------------------------------------------------------
    # JSON de configuración / trazabilidad
    # ------------------------------------------------------------------

    metadata = {
        "proceso": 27,
        "version": VERSION,
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_DIR),
        "parametros": {
            "k_min": K_MIN,
            "k_max": K_MAX,
            "min_proyectos": MIN_PROYECTOS,
            "peso_espacial": PESO_ESPACIAL,
            "peso_territorial": PESO_TERRITORIAL,
            "peso_necesidad": PESO_NECESIDAD,
            "peso_estrategica": PESO_ESTRATEGICA,
            "random_state": RANDOM_STATE,
        },
        "indicadores": columnas_indicadores,
        "proyectos": int(len(gdf_final)),
        "escenarios": int(len(escenarios)),
    }

    salida_json = (
        OUTPUT_DIR
        / "metadata.json"
    )

    salida_json.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Metadata                : {salida_json}"
    )


# ======================================================================================
# MAIN
# ======================================================================================

def main() -> None:

    print()
    print(
        "=" * 88
    )

    print(
        "27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - V7.2"
    )

    print(
        "=" * 88
    )

    print(
        f"Proyecto : {PROYECTO}"
    )

    print(
        f"Entrada  : {INPUT_PATH}"
    )

    print(
        f"Salida   : {OUTPUT_DIR}"
    )

    print()

    print(
        "CONFIGURACIÓN"
    )

    print(
        f"  Versión          : {VERSION}"
    )

    print(
        f"  K                : {K_MIN} - {K_MAX}"
    )

    print(
        f"  Mínimo proyectos : {MIN_PROYECTOS}"
    )

    print(
        f"  Espacial         : {PESO_ESPACIAL:.0%}"
    )

    print(
        f"  Territorial      : {PESO_TERRITORIAL:.0%}"
    )

    print(
        f"  Necesidad        : {PESO_NECESIDAD:.0%}"
    )

    print(
        f"  Estratégica      : {PESO_ESTRATEGICA:.0%}"
    )

    # ==========================================================================
    # 1
    # ==========================================================================

    titulo(
        "CARGANDO CARTERA DEL PROCESO 26",
        1,
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"No existe la entrada:\n{INPUT_PATH}"
        )

    print(
        f"Entrada:\n{INPUT_PATH}"
    )

    gdf = gpd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Registros : {len(gdf)}"
    )

    print(
        f"Columnas  : {len(gdf.columns)}"
    )

    print(
        f"CRS       : {gdf.crs}"
    )

    # ==========================================================================
    # 2
    # ==========================================================================

    gdf = validar_entrada(
        gdf
    )

    # ==========================================================================
    # 3
    # ==========================================================================

    columnas_indicadores = (
        resolver_indicadores(
            gdf
        )
    )

    # ==========================================================================
    # 4
    # ==========================================================================

    titulo(
        "NORMALIZACIÓN DE INDICADORES",
        4,
    )

    for nombre, columna in columnas_indicadores.items():

        # Se valida que haya datos numéricos reales.
        serie = pd.to_numeric(
            gdf[columna],
            errors="coerce",
        )

        validos = int(
            serie.notna().sum()
        )

        if validos == 0:

            raise ValueError(
                f"El indicador '{nombre}' "
                f"no tiene valores numéricos."
            )

        print(
            f"  {nombre:<22} OK"
        )

    print()
    print(
        "No se utilizaron valores artificiales."
    )

    # ==========================================================================
    # 5
    # ==========================================================================

    gdf_metric, coords = (
        preparar_componente_espacial(
            gdf
        )
    )

    # ==========================================================================
    # 6
    # ==========================================================================

    matriz, pesos = (
        construir_matriz_multicriterio(
            gdf_metric,
            columnas_indicadores,
            coords,
        )
    )

    # ==========================================================================
    # 7
    # ==========================================================================

    (
        k_seleccionado,
        evaluacion_k,
        labels,
    ) = seleccionar_k(
        matriz
    )

    # ==========================================================================
    # 8
    # ==========================================================================

    # Modelo final para obtener centroides.
    modelo_final = KMeans(
        n_clusters=k_seleccionado,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    labels_finales = (
        modelo_final.fit_predict(
            matriz
        )
    )

    # ==========================================================================
    # 9
    # ==========================================================================

    scores = calcular_scores_proyecto(
        gdf,
        columnas_indicadores,
    )

    gdf_resultado, escenarios = (
        construir_escenarios(
            gdf,
            labels_finales,
            scores,
        )
    )

    # ==========================================================================
    # 10
    # ==========================================================================

    cohesion = construir_metricas_cohesion(
        matriz,
        labels_finales,
        modelo_final,
    )

    # ==========================================================================
    # 11
    # ==========================================================================

    validar_asignacion(
        gdf_resultado,
        escenarios,
    )

    # ==========================================================================
    # 12
    # ==========================================================================

    exportar_resultados(
        gdf_resultado,
        escenarios,
        evaluacion_k,
        cohesion,
        columnas_indicadores,
    )

    # ==========================================================================
    # RESUMEN
    # ==========================================================================

    titulo(
        "PROCESO 27 FINALIZADO CORRECTAMENTE",
        12,
    )

    print(
        f"Proyectos procesados : {len(gdf_resultado)}"
    )

    print(
        f"Escenarios generados : {len(escenarios)}"
    )

    print(
        f"K seleccionado       : {k_seleccionado}"
    )

    print()

    print(
        "ESCENARIOS"
    )

    print(
        escenarios[
            [
                "escenario_id",
                "cantidad_proyectos",
                "score_escenario",
                "tipo_escenario",
                "dimension_dominante",
                "prioridad_escenario",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Salida principal:"
    )

    print(
        OUTPUT_DIR
        / "escenarios_territoriales_amba.parquet"
    )

    print()

    print(
        "========================================================================================"
    )


# ======================================================================================
# EJECUCIÓN
# ======================================================================================

if __name__ == "__main__":
    main()