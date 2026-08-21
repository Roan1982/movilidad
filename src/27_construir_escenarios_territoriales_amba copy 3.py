# -*- coding: utf-8 -*-

"""
===============================================================================
27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA
===============================================================================

Objetivo
--------
Construir escenarios territoriales de inversión a partir de la cartera
consolidada generada por el proceso 26.

Principios del modelo
---------------------
1. La cartera del proceso 26 es la única fuente de proyectos.
2. La componente espacial representa el 55% del modelo.
3. La componente territorial representa el 45%.
4. Las variables se normalizan globalmente antes del clustering.
5. K se evalúa sobre la solución FINAL.
6. Todo escenario debe contener al menos MIN_PROYECTOS_ESCENARIO proyectos.
7. Los clusters pequeños se reparan de manera determinística.
8. Los indicadores de escenario se calculan sobre la escala global.
9. Las geometrías de escenario se construyen a partir de sus proyectos.
10. La salida mantiene compatibilidad con los productos del V3/V4.

Entrada
-------
data/processed/cartera_proyectos_amba/cartera_proyectos_amba.parquet

Salidas
-------
data/processed/escenarios_territoriales_amba/

Autor
------
Pipeline AMBA - Movilidad
===============================================================================
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "cartera_proyectos_amba"
    / "cartera_proyectos_amba.parquet"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

RANDOM_STATE = 42

K_MIN = 6
K_MAX = 12

MIN_PROYECTOS_ESCENARIO = 8

# =============================================================================
# PESOS
# =============================================================================
#
# El modelo tiene dos dimensiones principales:
#
#   ESPACIAL    -> 55%
#   TERRITORIAL -> 45%
#
# Los pesos internos de la dimensión territorial suman 100%.
#
# =============================================================================

PESO_ESPACIAL = 0.55
PESO_TERRITORIAL = 0.45

PESO_DEMANDA = 0.20
PESO_DEFICIT = 0.20
PESO_CONECTIVIDAD = 0.15
PESO_INTERMODALIDAD = 0.15
PESO_INTEGRACION = 0.10
PESO_CENTRALIDAD = 0.10
PESO_IMPACTO = 0.10

# Variables secundarias utilizadas para construir el diagnóstico
PESO_URGENCIA = 0.50
PESO_SCORE_CARTERA = 0.50


# =============================================================================
# COLUMNAS
# =============================================================================

COLUMNAS_ID = [
    "proyecto_id",
    "id_proyecto",
    "codigo_proyecto",
]

COLUMNAS_NOMBRE = [
    "proyecto_nombre",
    "nombre_proyecto",
    "nombre",
]

COLUMNAS_SCORE = [
    "score_cartera",
]

COLUMNAS_PRIORIDAD = [
    "score_prioridad_territorial",
]

COLUMNAS_IMPACTO = [
    "impacto_potencial",
]

COLUMNAS_URGENCIA = [
    "urgencia_intervencion",
]

COLUMNAS_DEMANDA = [
    "indice_demanda_estructural",
    "indice_demanda",
]

COLUMNAS_INFRA = [
    "indice_infraestructura_estructural",
]

COLUMNAS_DEFICIT = [
    "deficit_infraestructura",
    "deficit_estructural_promedio",
]

COLUMNAS_CONECTIVIDAD = [
    "indice_conectividad_estructural",
]

COLUMNAS_INTERMODALIDAD = [
    "indice_intermodalidad_estructural",
]

COLUMNAS_INTEGRACION = [
    "indice_integracion_territorial",
]

COLUMNAS_CENTRALIDAD = [
    "indice_centralidad_estructural",
]


# =============================================================================
# WARNINGS
# =============================================================================

warnings.filterwarnings("ignore")


# =============================================================================
# UTILIDADES
# =============================================================================

def imprimir_titulo(texto: str) -> None:
    print()
    print("=" * 80)
    print(texto)
    print("=" * 80)


def imprimir_subtitulo(numero: str, texto: str) -> None:
    print()
    print("=" * 80)
    print(f"{numero}. {texto}")
    print("=" * 80)


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = False,
) -> str | None:

    for columna in candidatos:
        if columna in df.columns:
            return columna

    if obligatoria:
        raise ValueError(
            "No se encontró ninguna de las columnas requeridas:\n"
            + "\n".join(f"  - {x}" for x in candidatos)
        )

    return None


def convertir_numerico(
    serie: pd.Series,
    default: float = 0.0,
) -> pd.Series:

    resultado = pd.to_numeric(
        serie,
        errors="coerce",
    )

    resultado = resultado.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    resultado = resultado.fillna(default)

    return resultado.astype(float)


def normalizar_global(
    serie: pd.Series,
    valor_neutro: float = 0.5,
) -> pd.Series:

    x = convertir_numerico(serie)

    if len(x) == 0:
        return x

    minimo = float(x.min())
    maximo = float(x.max())

    if not np.isfinite(minimo) or not np.isfinite(maximo):
        return pd.Series(
            np.full(len(x), valor_neutro),
            index=x.index,
            dtype=float,
        )

    if math.isclose(minimo, maximo):
        return pd.Series(
            np.full(len(x), valor_neutro),
            index=x.index,
            dtype=float,
        )

    resultado = (
        (x - minimo)
        / (maximo - minimo)
    )

    return resultado.clip(
        lower=0.0,
        upper=1.0,
    )


def safe_float(valor: Any) -> float:

    try:
        x = float(valor)

        if np.isfinite(x):
            return x

    except Exception:
        pass

    return 0.0


def safe_int(valor: Any) -> int:

    try:
        return int(round(float(valor)))
    except Exception:
        return 0


# =============================================================================
# CARGA
# =============================================================================

def cargar_cartera() -> gpd.GeoDataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "No existe la cartera generada por el proceso 26:\n"
            f"{INPUT_PATH}"
        )

    print("Archivo de entrada:")
    print(INPUT_PATH)

    cartera = gpd.read_parquet(
        INPUT_PATH
    )

    if cartera.empty:
        raise ValueError(
            "La cartera del proceso 26 está vacía."
        )

    if "geometry" not in cartera.columns:
        raise ValueError(
            "La cartera no contiene columna geometry."
        )

    if cartera.crs is None:

        print(
            "ADVERTENCIA: la cartera no posee CRS. "
            f"Se asignará {CRS_GEOGRAFICO}."
        )

        cartera = cartera.set_crs(
            CRS_GEOGRAFICO,
            allow_override=True,
        )

    return cartera


# =============================================================================
# VALIDACIÓN DE ENTRADA
# =============================================================================

def validar_entrada(
    cartera: gpd.GeoDataFrame,
) -> str:

    imprimir_subtitulo(
        "2",
        "VALIDANDO DATOS DE ENTRADA",
    )

    print(
        f"Registros : {len(cartera):,}"
    )

    print(
        f"Columnas  : {len(cartera.columns):,}"
    )

    print(
        f"CRS       : {cartera.crs}"
    )

    # -------------------------------------------------------------------------
    # Geometrías
    # -------------------------------------------------------------------------

    nulas = int(
        cartera.geometry.isna().sum()
    )

    vacias = int(
        cartera.geometry.is_empty.sum()
    )

    print(
        f"Geometrías nulas : {nulas}"
    )

    print(
        f"Geometrías vacías: {vacias}"
    )

    if nulas > 0:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if vacias > 0:
        raise ValueError(
            "Existen geometrías vacías."
        )

    invalidas = int(
        (~cartera.geometry.is_valid).sum()
    )

    print(
        f"Geometrías inválidas: {invalidas}"
    )

    if invalidas > 0:

        print(
            "Intentando reparar geometrías inválidas..."
        )

        cartera["geometry"] = (
            cartera.geometry.make_valid()
        )

        restantes = int(
            (~cartera.geometry.is_valid).sum()
        )

        print(
            f"Geometrías inválidas restantes: {restantes}"
        )

        if restantes > 0:
            raise ValueError(
                "No fue posible reparar todas las geometrías."
            )

    # -------------------------------------------------------------------------
    # ID
    # -------------------------------------------------------------------------

    columna_id = encontrar_columna(
        cartera,
        COLUMNAS_ID,
        obligatoria=True,
    )

    cartera[columna_id] = (
        cartera[columna_id]
        .astype(str)
        .str.strip()
    )

    duplicados = int(
        cartera[columna_id]
        .duplicated()
        .sum()
    )

    print(
        f"ID utilizado: {columna_id}"
    )

    print(
        f"IDs duplicados: {duplicados}"
    )

    if duplicados > 0:
        raise ValueError(
            f"La columna {columna_id} contiene IDs duplicados."
        )

    # -------------------------------------------------------------------------
    # Cantidad mínima
    # -------------------------------------------------------------------------

    minimo_total = (
        K_MIN
        * MIN_PROYECTOS_ESCENARIO
    )

    if len(cartera) < minimo_total:

        raise ValueError(
            "No hay suficientes proyectos para construir "
            f"{K_MIN} escenarios con al menos "
            f"{MIN_PROYECTOS_ESCENARIO} proyectos cada uno.\n"
            f"Proyectos disponibles: {len(cartera)}\n"
            f"Proyectos requeridos: {minimo_total}"
        )

    print(
        f"Proyectos válidos: {len(cartera):,}"
    )

    print(
        "Validación de entrada: OK"
    )

    return columna_id


# =============================================================================
# VALIDACIÓN DE COMPONENTES
# =============================================================================

def validar_componentes(
    cartera: gpd.GeoDataFrame,
) -> dict[str, str | None]:

    imprimir_subtitulo(
        "3",
        "VALIDANDO COMPONENTES TERRITORIALES",
    )

    grupos = {
        "score_cartera": COLUMNAS_SCORE,
        "score_prioridad_territorial": COLUMNAS_PRIORIDAD,
        "impacto_potencial": COLUMNAS_IMPACTO,
        "urgencia_intervencion": COLUMNAS_URGENCIA,
        "indice_demanda_estructural": COLUMNAS_DEMANDA,
        "indice_infraestructura_estructural": COLUMNAS_INFRA,
        "deficit_infraestructura": COLUMNAS_DEFICIT,
        "indice_conectividad_estructural": COLUMNAS_CONECTIVIDAD,
        "indice_intermodalidad_estructural": COLUMNAS_INTERMODALIDAD,
        "indice_integracion_territorial": COLUMNAS_INTEGRACION,
        "indice_centralidad_estructural": COLUMNAS_CENTRALIDAD,
    }

    columnas: dict[str, str | None] = {}

    for nombre, candidatos in grupos.items():

        columna = encontrar_columna(
            cartera,
            candidatos,
            obligatoria=False,
        )

        columnas[nombre] = columna

        if columna is None:

            print(
                f"  {nombre:<42} NO DISPONIBLE -> valor neutro"
            )

        else:

            cartera[columna] = convertir_numerico(
                cartera[columna]
            )

            print(
                f"  {nombre:<42} OK -> {columna}"
            )

    disponibles = sum(
        x is not None
        for x in columnas.values()
    )

    print()
    print(
        f"Componentes disponibles: "
        f"{disponibles} / {len(columnas)}"
    )

    return columnas


# =============================================================================
# PREPARACIÓN ESPACIAL
# =============================================================================

def preparar_variables_espaciales(
    cartera: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    imprimir_subtitulo(
        "4",
        "PREPARANDO COMPONENTE ESPACIAL",
    )

    trabajo = cartera.copy()

    metric = trabajo.to_crs(
        CRS_METRICO
    )

    centroides = metric.geometry.centroid

    trabajo["x_m"] = centroides.x
    trabajo["y_m"] = centroides.y

    if (
        trabajo["x_m"].isna().any()
        or trabajo["y_m"].isna().any()
    ):
        raise ValueError(
            "No fue posible calcular todos los centroides."
        )

    print(
        f"X mínimo: {trabajo['x_m'].min():.2f} m"
    )

    print(
        f"X máximo: {trabajo['x_m'].max():.2f} m"
    )

    print(
        f"Y mínimo: {trabajo['y_m'].min():.2f} m"
    )

    print(
        f"Y máximo: {trabajo['y_m'].max():.2f} m"
    )

    print(
        "Componente espacial: OK"
    )

    return trabajo


# =============================================================================
# MATRIZ MULTICRITERIO
# =============================================================================

def construir_matriz_multicriterio(
    cartera: gpd.GeoDataFrame,
    columnas: dict[str, str | None],
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    dict[str, float],
]:

    imprimir_subtitulo(
        "5",
        "CONSTRUYENDO MATRIZ MULTICRITERIO",
    )

    # -------------------------------------------------------------------------
    # Variables territoriales
    # -------------------------------------------------------------------------

    variables_territoriales = [
        (
            "demanda",
            "indice_demanda_estructural",
            PESO_DEMANDA,
        ),
        (
            "deficit",
            "deficit_infraestructura",
            PESO_DEFICIT,
        ),
        (
            "conectividad",
            "indice_conectividad_estructural",
            PESO_CONECTIVIDAD,
        ),
        (
            "intermodalidad",
            "indice_intermodalidad_estructural",
            PESO_INTERMODALIDAD,
        ),
        (
            "integracion",
            "indice_integracion_territorial",
            PESO_INTEGRACION,
        ),
        (
            "centralidad",
            "indice_centralidad_estructural",
            PESO_CENTRALIDAD,
        ),
        (
            "impacto",
            "impacto_potencial",
            PESO_IMPACTO,
        ),
    ]

    # -------------------------------------------------------------------------
    # Componente espacial
    # -------------------------------------------------------------------------

    scaler_xy = RobustScaler()

    xy = scaler_xy.fit_transform(
        cartera[
            [
                "x_m",
                "y_m",
            ]
        ]
    )

    xy = np.nan_to_num(
        xy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # -------------------------------------------------------------------------
    # Escala robusta espacial
    # -------------------------------------------------------------------------

    matriz = pd.DataFrame(
        index=cartera.index
    )

    matriz["espacial_x"] = xy[:, 0]
    matriz["espacial_y"] = xy[:, 1]

    pesos: dict[str, float] = {}

    pesos["espacial_x"] = (
        PESO_ESPACIAL / 2.0
    )

    pesos["espacial_y"] = (
        PESO_ESPACIAL / 2.0
    )

    # -------------------------------------------------------------------------
    # Variables territoriales normalizadas globalmente
    # -------------------------------------------------------------------------

    for nombre, componente, peso_interno in (
        variables_territoriales
    ):

        columna = columnas.get(
            componente
        )

        nombre_matriz = (
            f"territorial_{nombre}"
        )

        if columna is None:

            matriz[nombre_matriz] = 0.0

            pesos[nombre_matriz] = (
                PESO_TERRITORIAL
                * peso_interno
            )

            print(
                f"  {nombre:<20} "
                f"NO DISPONIBLE -> 0"
            )

            continue

        normalizada = normalizar_global(
            cartera[columna]
        )

        matriz[nombre_matriz] = (
            normalizada.values
        )

        pesos[nombre_matriz] = (
            PESO_TERRITORIAL
            * peso_interno
        )

        print(
            f"  {nombre:<20} "
            f"OK -> {columna}"
        )

    # -------------------------------------------------------------------------
    # Estandarización espacial
    # -------------------------------------------------------------------------

    # Las coordenadas ya fueron escaladas robustamente.
    #
    # Las variables territoriales están en [0,1].
    #
    # No se vuelve a aplicar RobustScaler a toda la matriz porque hacerlo
    # eliminaría parte del significado de los pesos explícitos.
    #
    # Se aplica una escala común mediante multiplicación por sqrt(peso).
    # De esta forma los pesos actúan sobre la distancia euclídea.
    # -------------------------------------------------------------------------

    X = matriz.copy().astype(float)

    for columna in X.columns:

        peso = max(
            float(
                pesos.get(
                    columna,
                    0.0,
                )
            ),
            0.000001,
        )

        X[columna] = (
            X[columna]
            * math.sqrt(peso)
        )

    X_array = np.nan_to_num(
        X.values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    suma_pesos = sum(
        pesos.values()
    )

    print()
    print(
        f"Proyectos : {len(X_array):,}"
    )

    print(
        f"Variables : {X_array.shape[1]}"
    )

    print(
        f"Peso espacial    : {PESO_ESPACIAL:.0%}"
    )

    print(
        f"Peso territorial : {PESO_TERRITORIAL:.0%}"
    )

    print(
        f"Suma de pesos    : {suma_pesos:.4f}"
    )

    if not math.isclose(
        suma_pesos,
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Los pesos del modelo no suman 1."
        )

    return (
        X_array,
        matriz,
        pesos,
    )


# =============================================================================
# REPARACIÓN DE CLUSTERS
# =============================================================================

def calcular_centroides(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:

    centroides: dict[int, np.ndarray] = {}

    for cluster in sorted(
        np.unique(labels)
    ):

        indices = np.where(
            labels == cluster
        )[0]

        if len(indices) == 0:
            continue

        centroides[int(cluster)] = (
            X[indices].mean(axis=0)
        )

    return centroides


def reparar_clusters_pequenos(
    X: np.ndarray,
    labels: np.ndarray,
    minimo: int,
) -> np.ndarray:

    labels = labels.astype(int).copy()

    while True:

        conteo = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        pequenos = (
            conteo[
                conteo < minimo
            ]
            .sort_values(
                ascending=True
            )
        )

        if pequenos.empty:
            break

        cluster_pequeno = int(
            pequenos.index[0]
        )

        indices_pequenos = np.where(
            labels == cluster_pequeno
        )[0]

        conteo_actual = (
            pd.Series(labels)
            .value_counts()
        )

        receptores = [
            int(cluster)
            for cluster, cantidad
            in conteo_actual.items()
            if (
                int(cluster)
                != cluster_pequeno
                and int(cantidad) >= minimo
            )
        ]

        if not receptores:

            raise RuntimeError(
                "No existe un cluster receptor válido "
                "para reparar el cluster pequeño."
            )

        centroides = calcular_centroides(
            X,
            labels,
        )

        # ---------------------------------------------------------------------
        # Orden determinístico de proyectos
        # ---------------------------------------------------------------------

        indices_pequenos = sorted(
            indices_pequenos.tolist()
        )

        for idx in indices_pequenos:

            # -------------------------------------------------------------
            # Recalcular receptores en cada movimiento.
            # -------------------------------------------------------------

            conteo_actual = (
                pd.Series(labels)
                .value_counts()
            )

            receptores = [
                int(cluster)
                for cluster, cantidad
                in conteo_actual.items()
                if (
                    int(cluster)
                    != cluster_pequeno
                    and int(cantidad) >= minimo
                )
            ]

            if not receptores:
                raise RuntimeError(
                    "Se quedó sin cluster receptor durante "
                    "la reparación."
                )

            centroides = calcular_centroides(
                X,
                labels,
            )

            candidatos = []

            for destino in sorted(
                receptores
            ):

                centroide = centroides[
                    destino
                ]

                distancia = float(
                    np.linalg.norm(
                        X[idx] - centroide
                    )
                )

                candidatos.append(
                    (
                        distancia,
                        destino,
                    )
                )

            # Distancia + ID como desempate
            candidatos.sort(
                key=lambda x: (
                    x[0],
                    x[1],
                )
            )

            destino = candidatos[0][1]

            labels[idx] = destino

        # ---------------------------------------------------------------------
        # Volver a evaluar.
        # ---------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Renumeración consecutiva desde 1
    # -------------------------------------------------------------------------

    clusters = sorted(
        np.unique(labels)
    )

    mapa = {
        viejo: nuevo
        for nuevo, viejo
        in enumerate(
            clusters,
            start=1,
        )
    }

    return np.array(
        [
            mapa[int(x)]
            for x in labels
        ],
        dtype=int,
    )


# =============================================================================
# EVALUACIÓN DE K
# =============================================================================

def evaluar_k(
    X: np.ndarray,
    k: int,
) -> tuple[
    np.ndarray,
    float,
    int,
    int,
    int,
]:

    modelo = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=50,
        max_iter=500,
        algorithm="lloyd",
    )

    labels_originales = (
        modelo.fit_predict(X)
    )

    labels_finales = (
        reparar_clusters_pequenos(
            X,
            labels_originales,
            MIN_PROYECTOS_ESCENARIO,
        )
    )

    conteo = (
        pd.Series(labels_finales)
        .value_counts()
    )

    cantidad_clusters = int(
        len(
            np.unique(
                labels_finales
            )
        )
    )

    minimo = int(
        conteo.min()
    )

    maximo = int(
        conteo.max()
    )

    if cantidad_clusters < 2:

        silhouette = -1.0

    else:

        silhouette = float(
            silhouette_score(
                X,
                labels_finales,
                metric="euclidean",
            )
        )

    return (
        labels_finales,
        silhouette,
        minimo,
        maximo,
        cantidad_clusters,
    )


# =============================================================================
# SELECCIÓN DE K
# =============================================================================

def seleccionar_k(
    X: np.ndarray,
) -> tuple[
    int,
    np.ndarray,
    pd.DataFrame,
]:

    imprimir_subtitulo(
        "6",
        "SELECCIONANDO CANTIDAD DE ESCENARIOS",
    )

    n = len(X)

    k_max_real = min(
        K_MAX,
        n // MIN_PROYECTOS_ESCENARIO,
    )

    if k_max_real < K_MIN:

        raise ValueError(
            "No existe una cantidad de escenarios válida "
            "con el mínimo de proyectos definido."
        )

    resultados = []

    mejor_score = -np.inf
    mejor_k = None
    mejor_labels = None

    for k in range(
        K_MIN,
        k_max_real + 1,
    ):

        (
            labels,
            silhouette,
            minimo,
            maximo,
            k_final,
        ) = evaluar_k(
            X,
            k,
        )

        cumple = (
            minimo
            >= MIN_PROYECTOS_ESCENARIO
        )

        # ---------------------------------------------------------------------
        # Penalización pequeña por complejidad.
        # ---------------------------------------------------------------------

        penalizacion = (
            0.003
            * max(
                k_final - K_MIN,
                0,
            )
        )

        score = (
            silhouette
            - penalizacion
        )

        resultados.append(
            {
                "k_solicitado": k,
                "k_final": k_final,
                "silhouette": silhouette,
                "min_proyectos": minimo,
                "max_proyectos": maximo,
                "penalizacion_k": penalizacion,
                "score_seleccion": score,
                "cumple_minimo": cumple,
            }
        )

        print(
            f"K={k:2d} | "
            f"K final={k_final:2d} | "
            f"silhouette={silhouette: .4f} | "
            f"mín={minimo:3d} | "
            f"máx={maximo:3d} | "
            f"score={score: .4f}"
        )

        if not cumple:
            continue

        if score > mejor_score:

            mejor_score = score
            mejor_k = k
            mejor_labels = labels.copy()

    if mejor_k is None or mejor_labels is None:

        raise RuntimeError(
            "No fue posible obtener una solución válida "
            "que respete el tamaño mínimo de escenario."
        )

    evaluacion = pd.DataFrame(
        resultados
    )

    print()
    print(
        f"K seleccionado     : {mejor_k}"
    )

    print(
        f"K final             : "
        f"{len(np.unique(mejor_labels))}"
    )

    print(
        f"Score de selección  : "
        f"{mejor_score:.4f}"
    )

    print(
        f"Mínimo por escenario: "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )

    return (
        int(mejor_k),
        mejor_labels,
        evaluacion,
    )


# =============================================================================
# PREPARACIÓN DE INDICADORES GLOBALES
# =============================================================================

def preparar_indicadores_globales(
    cartera: gpd.GeoDataFrame,
    columnas: dict[str, str | None],
) -> gpd.GeoDataFrame:

    imprimir_subtitulo(
        "7",
        "NORMALIZANDO INDICADORES GLOBALES",
    )

    trabajo = cartera.copy()

    indicadores = [
        "impacto_potencial",
        "indice_demanda_estructural",
        "deficit_infraestructura",
        "indice_conectividad_estructural",
        "indice_intermodalidad_estructural",
        "indice_integracion_territorial",
        "indice_centralidad_estructural",
        "urgencia_intervencion",
        "score_cartera",
        "score_prioridad_territorial",
    ]

    for nombre in indicadores:

        columna = columnas.get(
            nombre
        )

        nombre_norm = (
            f"{nombre}_norm"
        )

        if columna is None:

            trabajo[nombre_norm] = 0.5

            print(
                f"  {nombre:<42} "
                "NO DISPONIBLE -> 0.5"
            )

        else:

            trabajo[nombre_norm] = (
                normalizar_global(
                    trabajo[columna]
                )
            )

            print(
                f"  {nombre:<42} "
                f"OK -> {columna}"
            )

    return trabajo


# =============================================================================
# CONSTRUCCIÓN DE ESCENARIOS
# =============================================================================

def construir_escenarios(
    cartera: gpd.GeoDataFrame,
    columnas: dict[str, str | None],
    labels: np.ndarray,
) -> pd.DataFrame:

    imprimir_subtitulo(
        "8",
        "CONSTRUYENDO ESCENARIOS TERRITORIALES",
    )

    trabajo = cartera.copy()

    trabajo[
        "cluster_territorial"
    ] = labels

    # -------------------------------------------------------------------------
    # Cobertura territorial
    #
    # Se utiliza como medida relativa de presencia dentro de la cartera.
    # No representa superficie geográfica.
    # -------------------------------------------------------------------------

    total_proyectos = len(
        trabajo
    )

    escenarios = []

    for cluster in sorted(
        trabajo[
            "cluster_territorial"
        ].unique()
    ):

        subset = trabajo[
            trabajo[
                "cluster_territorial"
            ] == cluster
        ]

        cantidad = len(
            subset
        )

        def promedio(
            nombre: str,
        ) -> float:

            columna_norm = (
                f"{nombre}_norm"
            )

            return float(
                subset[
                    columna_norm
                ].mean()
                * 100.0
            )

        impacto = promedio(
            "impacto_potencial"
        )

        demanda = promedio(
            "indice_demanda_estructural"
        )

        deficit = promedio(
            "deficit_infraestructura"
        )

        conectividad = promedio(
            "indice_conectividad_estructural"
        )

        intermodalidad = promedio(
            "indice_intermodalidad_estructural"
        )

        integracion = promedio(
            "indice_integracion_territorial"
        )

        centralidad = promedio(
            "indice_centralidad_estructural"
        )

        urgencia = promedio(
            "urgencia_intervencion"
        )

        score_cartera = promedio(
            "score_cartera"
        )

        score_prioridad = promedio(
            "score_prioridad_territorial"
        )

        cobertura = (
            cantidad
            / max(
                total_proyectos,
                1,
            )
            * 100.0
        )

        complementariedad = float(
            np.mean(
                [
                    conectividad,
                    intermodalidad,
                    integracion,
                    centralidad,
                ]
            )
        )

        # ---------------------------------------------------------------------
        # Score territorial
        #
        # Se evita utilizar cobertura como componente dominante del score,
        # porque el tamaño del cluster es consecuencia del algoritmo y no
        # necesariamente una medida de necesidad territorial.
        # ---------------------------------------------------------------------

        score = (
            impacto * 0.25
            + deficit * 0.20
            + demanda * 0.20
            + complementariedad * 0.15
            + urgencia * 0.10
            + score_prioridad * 0.10
        )

        # ---------------------------------------------------------------------
        # Diagnóstico
        # ---------------------------------------------------------------------

        if urgencia >= 70:

            diagnostico = (
                "ALTA_URGENCIA"
            )

        elif deficit >= 70:

            diagnostico = (
                "ALTO_DEFICIT_ATENDIDO"
            )

        elif impacto >= 70:

            diagnostico = (
                "ALTO_IMPACTO_POTENCIAL"
            )

        elif complementariedad >= 65:

            diagnostico = (
                "ALTA_COMPLEMENTARIEDAD"
            )

        elif demanda >= 65:

            diagnostico = (
                "ALTA_DEMANDA"
            )

        else:

            diagnostico = (
                "INTERVENCION_TERRITORIAL_MEDIA"
            )

        # ---------------------------------------------------------------------
        # Tipo de escenario
        # ---------------------------------------------------------------------

        valores_tipo = {
            "IMPACTO": impacto,
            "DEMANDA": demanda,
            "DEFICIT": deficit,
            "COMPLEMENTARIEDAD": complementariedad,
        }

        dominante = max(
            valores_tipo,
            key=valores_tipo.get,
        )

        if dominante == "IMPACTO":

            tipo = (
                "ESCENARIO_SELECTIVO"
            )

        elif dominante == "DEMANDA":

            tipo = (
                "ESCENARIO_METROPOLITANO"
            )

        elif dominante == "DEFICIT":

            tipo = (
                "ESCENARIO_DEFICIT"
            )

        else:

            tipo = (
                "ESCENARIO_INTEGRADO"
            )

        # ---------------------------------------------------------------------
        # Horizonte
        # ---------------------------------------------------------------------

        if (
            urgencia >= 70
            or deficit >= 70
        ):

            horizonte = (
                "CORTO_PLAZO"
            )

        elif (
            impacto >= 65
            or demanda >= 65
        ):

            horizonte = (
                "MEDIANO_PLAZO"
            )

        else:

            horizonte = (
                "LARGO_PLAZO"
            )

        # ---------------------------------------------------------------------
        # Prioridad
        # ---------------------------------------------------------------------

        if score >= 70:

            prioridad = (
                "PRIORIDAD_1_CRITICA"
            )

        elif score >= 58:

            prioridad = (
                "PRIORIDAD_2_ALTA"
            )

        elif score >= 45:

            prioridad = (
                "PRIORIDAD_3_MEDIA"
            )

        else:

            prioridad = (
                "PRIORIDAD_4_BAJA"
            )

        # ---------------------------------------------------------------------
        # Dimensiones prioritarias
        # ---------------------------------------------------------------------

        dimensiones = {
            "DEMANDA": demanda,
            "DEFICIT": deficit,
            "IMPACTO": impacto,
            "CONECTIVIDAD": conectividad,
            "INTERMODALIDAD": intermodalidad,
            "INTEGRACION": integracion,
            "CENTRALIDAD": centralidad,
            "URGENCIA": urgencia,
        }

        dimensiones_ordenadas = sorted(
            dimensiones.items(),
            key=lambda x: (
                -x[1],
                x[0],
            ),
        )

        dimensiones_prioritarias = ", ".join(
            nombre
            for nombre, _ in
            dimensiones_ordenadas[:3]
        )

        # ---------------------------------------------------------------------
        # Textos
        # ---------------------------------------------------------------------

        diagnostico_texto = (
            f"{diagnostico}. "
            f"El escenario agrupa {cantidad} proyectos. "
            f"Score territorial: {score:.2f}. "
            f"Déficit: {deficit:.2f}; "
            f"demanda: {demanda:.2f}; "
            f"impacto: {impacto:.2f}; "
            f"complementariedad: "
            f"{complementariedad:.2f}."
        )

        objetivo = (
            "Concentrar inversiones territoriales "
            f"con énfasis en "
            f"{dimensiones_prioritarias}."
        )

        justificacion = (
            f"El agrupamiento contiene {cantidad} proyectos "
            "y presenta una combinación territorial "
            "caracterizada por "
            f"{dimensiones_prioritarias}."
        )

        escenarios.append(
            {
                "cluster_territorial": int(
                    cluster
                ),
                "cantidad_proyectos": int(
                    cantidad
                ),
                "impacto_territorial": round(
                    impacto,
                    4,
                ),
                "cobertura_territorial": round(
                    cobertura,
                    4,
                ),
                "deficit_atendido": round(
                    deficit,
                    4,
                ),
                "demanda_cubierta": round(
                    demanda,
                    4,
                ),
                "complementariedad": round(
                    complementariedad,
                    4,
                ),
                "urgencia_promedio": round(
                    urgencia,
                    4,
                ),
                "score_cartera_promedio": round(
                    score_cartera,
                    4,
                ),
                "score_prioridad_territorial": round(
                    score_prioridad,
                    4,
                ),
                "conectividad_promedio": round(
                    conectividad,
                    4,
                ),
                "intermodalidad_promedio": round(
                    intermodalidad,
                    4,
                ),
                "integracion_promedio": round(
                    integracion,
                    4,
                ),
                "centralidad_promedio": round(
                    centralidad,
                    4,
                ),
                "score_escenario": round(
                    score,
                    4,
                ),
                "prioridad_escenario": prioridad,
                "tipo_escenario": tipo,
                "horizonte_escenario": horizonte,
                "diagnostico_escenario": diagnostico,
                "objetivo_escenario": objetivo,
                "justificacion_escenario": justificacion,
                "dimensiones_prioritarias":
                    dimensiones_prioritarias,
                "diagnostico_detallado":
                    diagnostico_texto,
            }
        )

    escenarios_df = pd.DataFrame(
        escenarios
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    escenarios_df = (
        escenarios_df
        .sort_values(
            [
                "score_escenario",
                "cantidad_proyectos",
                "cluster_territorial",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    escenarios_df[
        "ranking_escenario"
    ] = np.arange(
        1,
        len(escenarios_df) + 1,
    )

    escenarios_df[
        "escenario_id"
    ] = [
        f"AMBA-E{i:03d}"
        for i in escenarios_df[
            "ranking_escenario"
        ]
    ]

    escenarios_df[
        "escenario_nombre"
    ] = (
        escenarios_df[
            "escenario_id"
        ]
        + " - "
        + escenarios_df[
            "tipo_escenario"
        ]
        .str.replace(
            "ESCENARIO_",
            "",
            regex=False,
        )
    )

    print()
    print(
        "Distribución final:"
    )

    print(
        escenarios_df[
            [
                "escenario_id",
                "cantidad_proyectos",
                "score_escenario",
                "prioridad_escenario",
            ]
        ].to_string(
            index=False
        )
    )

    return escenarios_df


# =============================================================================
# ASIGNACIÓN
# =============================================================================

def asignar_escenarios(
    cartera: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    labels: np.ndarray,
) -> gpd.GeoDataFrame:

    imprimir_subtitulo(
        "9",
        "ASIGNANDO ESCENARIOS A PROYECTOS",
    )

    resultado = cartera.copy()

    resultado[
        "cluster_territorial"
    ] = labels

    mapa = (
        escenarios
        .set_index(
            "cluster_territorial"
        )
        .to_dict(
            orient="index"
        )
    )

    columnas_escenario = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "cantidad_proyectos",
        "impacto_territorial",
        "cobertura_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
        "urgencia_promedio",
        "score_cartera_promedio",
        "score_prioridad_territorial",
        "conectividad_promedio",
        "intermodalidad_promedio",
        "integracion_promedio",
        "centralidad_promedio",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "horizonte_escenario",
        "diagnostico_escenario",
        "objetivo_escenario",
        "justificacion_escenario",
        "dimensiones_prioritarias",
        "diagnostico_detallado",
    ]

    for columna in columnas_escenario:

        resultado[columna] = (
            resultado[
                "cluster_territorial"
            ]
            .map(
                lambda x:
                    mapa[
                        int(x)
                    ][columna]
            )
        )

    asignados = int(
        resultado[
            "escenario_id"
        ]
        .notna()
        .sum()
    )

    print(
        f"Proyectos asignados: "
        f"{asignados:,} / "
        f"{len(resultado):,}"
    )

    if asignados != len(resultado):

        raise RuntimeError(
            "No todos los proyectos fueron asignados "
            "a un escenario."
        )

    return resultado


# =============================================================================
# MÉTRICAS DE COHESIÓN
# =============================================================================

def calcular_metricas_cohesion(
    X: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:

    imprimir_subtitulo(
        "10",
        "CALCULANDO MÉTRICAS DE COHESIÓN",
    )

    filas = []

    for cluster in sorted(
        np.unique(labels)
    ):

        indices = np.where(
            labels == cluster
        )[0]

        puntos = X[
            indices
        ]

        centroide = puntos.mean(
            axis=0
        )

        distancias = np.linalg.norm(
            puntos - centroide,
            axis=1,
        )

        filas.append(
            {
                "cluster_territorial":
                    int(cluster),
                "cohesion_promedio":
                    float(
                        distancias.mean()
                    ),
                "cohesion_maxima":
                    float(
                        distancias.max()
                    ),
                "dispersion_std":
                    float(
                        distancias.std()
                    ),
            }
        )

    metricas = pd.DataFrame(
        filas
    )

    print(
        metricas.to_string(
            index=False
        )
    )

    return metricas


# =============================================================================
# GEOMETRÍAS
# =============================================================================

def construir_geometrias(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
) -> gpd.GeoDataFrame:

    imprimir_subtitulo(
        "11",
        "CONSTRUYENDO GEOMETRÍAS DE ESCENARIOS",
    )

    metric = proyectos.to_crs(
        CRS_METRICO
    )

    filas = []

    for escenario_id, grupo in metric.groupby(
        "escenario_id",
        sort=True,
    ):

        geometria = grupo.geometry

        try:

            union = geometria.union_all()

        except AttributeError:

            union = geometria.unary_union

        if union is None or union.is_empty:

            raise ValueError(
                f"El escenario {escenario_id} "
                "produjo una geometría vacía."
            )

        # ---------------------------------------------------------------------
        # Puntos
        # ---------------------------------------------------------------------

        if union.geom_type in (
            "Point",
            "MultiPoint",
        ):

            union = union.buffer(
                750
            )

        # ---------------------------------------------------------------------
        # Líneas
        # ---------------------------------------------------------------------

        elif union.geom_type in (
            "LineString",
            "MultiLineString",
        ):

            union = union.buffer(
                250
            )

        # ---------------------------------------------------------------------
        # Validación
        # ---------------------------------------------------------------------

        if not union.is_valid:

            union = union.buffer(
                0
            )

        if union.is_empty:

            raise ValueError(
                f"La geometría final de {escenario_id} "
                "está vacía."
            )

        filas.append(
            {
                "escenario_id":
                    escenario_id,
                "geometry":
                    union,
            }
        )

    geometrias = gpd.GeoDataFrame(
        filas,
        geometry="geometry",
        crs=CRS_METRICO,
    )

    # -------------------------------------------------------------------------
    # Merge explícito contra escenarios.
    #
    # Esto evita depender de columnas copiadas previamente a proyectos.
    # -------------------------------------------------------------------------

    columnas_resumen = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "cluster_territorial",
        "cantidad_proyectos",
        "impacto_territorial",
        "cobertura_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
        "urgencia_promedio",
        "score_cartera_promedio",
        "score_prioridad_territorial",
        "conectividad_promedio",
        "intermodalidad_promedio",
        "integracion_promedio",
        "centralidad_promedio",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "horizonte_escenario",
        "diagnostico_escenario",
        "objetivo_escenario",
        "justificacion_escenario",
        "dimensiones_prioritarias",
        "diagnostico_detallado",
    ]

    resumen = escenarios[
        columnas_resumen
    ].copy()

    geometrias = geometrias.merge(
        resumen,
        on="escenario_id",
        how="left",
        validate="one_to_one",
    )

    geometrias = geometrias.to_crs(
        CRS_GEOGRAFICO
    )

    print(
        f"Geometrías construidas: "
        f"{len(geometrias)}"
    )

    return geometrias


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

def validar_final(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    geometrias: gpd.GeoDataFrame,
    evaluacion_k: pd.DataFrame,
) -> None:

    imprimir_subtitulo(
        "12",
        "VALIDACIÓN FINAL",
    )

    columnas_escenario = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "cantidad_proyectos",
        "impacto_territorial",
        "cobertura_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "horizonte_escenario",
        "diagnostico_escenario",
        "objetivo_escenario",
        "justificacion_escenario",
        "dimensiones_prioritarias",
    ]

    # -------------------------------------------------------------------------
    # Columnas
    # -------------------------------------------------------------------------

    for columna in columnas_escenario:

        if columna not in escenarios.columns:

            raise ValueError(
                f"Falta columna obligatoria: {columna}"
            )

        nulos = int(
            escenarios[
                columna
            ].isna().sum()
        )

        print(
            f"{columna:<42} nulos={nulos}"
        )

        if nulos > 0:

            raise ValueError(
                f"La columna {columna} contiene nulos."
            )

    # -------------------------------------------------------------------------
    # IDs
    # -------------------------------------------------------------------------

    duplicados = int(
        escenarios[
            "escenario_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicados > 0:

        raise ValueError(
            "Existen escenarios duplicados."
        )

    # -------------------------------------------------------------------------
    # Asignaciones
    # -------------------------------------------------------------------------

    asignaciones = int(
        proyectos[
            "escenario_id"
        ]
        .notna()
        .sum()
    )

    print()
    print(
        f"Asignaciones: "
        f"{asignaciones:,}/"
        f"{len(proyectos):,}"
    )

    if asignaciones != len(proyectos):

        raise ValueError(
            "No todos los proyectos tienen escenario."
        )

    # -------------------------------------------------------------------------
    # Distribución
    # -------------------------------------------------------------------------

    minimo = int(
        escenarios[
            "cantidad_proyectos"
        ].min()
    )

    maximo = int(
        escenarios[
            "cantidad_proyectos"
        ].max()
    )

    print(
        f"Proyectos por escenario: "
        f"mín={minimo} | máx={maximo}"
    )

    if minimo < MIN_PROYECTOS_ESCENARIO:

        raise ValueError(
            "Existe un escenario por debajo del mínimo "
            f"de {MIN_PROYECTOS_ESCENARIO} proyectos."
        )

    # -------------------------------------------------------------------------
    # Consistencia de cantidades
    # -------------------------------------------------------------------------

    suma = int(
        escenarios[
            "cantidad_proyectos"
        ].sum()
    )

    if suma != len(proyectos):

        raise ValueError(
            "La suma de proyectos de los escenarios "
            "no coincide con la cartera."
        )

    # -------------------------------------------------------------------------
    # Geometrías
    # -------------------------------------------------------------------------

    if len(geometrias) != len(
        escenarios
    ):

        raise ValueError(
            "La cantidad de geometrías no coincide "
            "con la cantidad de escenarios."
        )

    if geometrias.geometry.isna().any():

        raise ValueError(
            "Existen geometrías nulas."
        )

    if geometrias.geometry.is_empty.any():

        raise ValueError(
            "Existen geometrías vacías."
        )

    if (
        ~geometrias.geometry.is_valid
    ).any():

        raise ValueError(
            "Existen geometrías inválidas."
        )

    # -------------------------------------------------------------------------
    # Evaluación K
    # -------------------------------------------------------------------------

    if evaluacion_k.empty:

        raise ValueError(
            "La evaluación de K está vacía."
        )

    if not evaluacion_k[
        "cumple_minimo"
    ].all():

        # No todas las K necesitan cumplir.
        # Debe existir al menos una.
        if not evaluacion_k[
            "cumple_minimo"
        ].any():

            raise ValueError(
                "Ninguna solución de K cumple el mínimo."
            )

    print()
    print(
        "VALIDACIÓN FINAL: OK"
    )


# =============================================================================
# RESUMEN
# =============================================================================

def construir_resumen(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
    metricas_cohesion: pd.DataFrame,
) -> dict[str, Any]:

    imprimir_subtitulo(
        "13",
        "CONSTRUYENDO RESUMEN JSON",
    )

    mejor_evaluacion = (
        evaluacion_k[
            evaluacion_k[
                "cumple_minimo"
            ]
        ]
        .sort_values(
            "score_seleccion",
            ascending=False,
        )
        .iloc[0]
    )

    resumen: dict[str, Any] = {

        "version": "V4_DEFINITIVA",

        "proyecto":
            "Construcción de escenarios territoriales AMBA",

        "fecha_ejecucion":
            pd.Timestamp.now().isoformat(),

        "entrada":
            str(INPUT_PATH),

        "salida":
            str(OUTPUT_DIR),

        "parametros": {

            "k_min":
                K_MIN,

            "k_max":
                K_MAX,

            "min_proyectos_escenario":
                MIN_PROYECTOS_ESCENARIO,

            "random_state":
                RANDOM_STATE,

            "crs":
                CRS_GEOGRAFICO,

            "crs_metrico":
                CRS_METRICO,
        },

        "modelo": {

            "metodo":
                "KMeans multicriterio + reparación restringida",

            "peso_espacial":
                PESO_ESPACIAL,

            "peso_territorial":
                PESO_TERRITORIAL,

            "normalizacion":
                "global",

            "evaluacion_k":
                "silhouette sobre solución final",

            "restriccion_tamano":
                MIN_PROYECTOS_ESCENARIO,
        },

        "pesos_territoriales": {

            "demanda":
                PESO_DEMANDA,

            "deficit":
                PESO_DEFICIT,

            "conectividad":
                PESO_CONECTIVIDAD,

            "intermodalidad":
                PESO_INTERMODALIDAD,

            "integracion":
                PESO_INTEGRACION,

            "centralidad":
                PESO_CENTRALIDAD,

            "impacto":
                PESO_IMPACTO,
        },

        "cantidad_proyectos":
            int(len(proyectos)),

        "cantidad_escenarios":
            int(len(escenarios)),

        "k_seleccionado":
            int(
                mejor_evaluacion[
                    "k_solicitado"
                ]
            ),

        "k_final":
            int(len(escenarios)),

        "silhouette_seleccionada":
            safe_float(
                mejor_evaluacion[
                    "silhouette"
                ]
            ),

        "distribucion_proyectos": {

            str(row["escenario_id"]):
                int(row["cantidad_proyectos"])

            for _, row
            in escenarios.iterrows()
        },

        "prioridades": {

            str(k): int(v)

            for k, v
            in escenarios[
                "prioridad_escenario"
            ]
            .value_counts()
            .items()
        },

        "horizontes": {

            str(k): int(v)

            for k, v
            in escenarios[
                "horizonte_escenario"
            ]
            .value_counts()
            .items()
        },

        "tipos": {

            str(k): int(v)

            for k, v
            in escenarios[
                "tipo_escenario"
            ]
            .value_counts()
            .items()
        },

        "diagnosticos": {

            str(k): int(v)

            for k, v
            in escenarios[
                "diagnostico_escenario"
            ]
            .value_counts()
            .items()
        },

        "cohesion": {

            str(
                row[
                    "cluster_territorial"
                ]
            ): {
                "cohesion_promedio":
                    safe_float(
                        row[
                            "cohesion_promedio"
                        ]
                    ),
                "cohesion_maxima":
                    safe_float(
                        row[
                            "cohesion_maxima"
                        ]
                    ),
                "dispersion_std":
                    safe_float(
                        row[
                            "dispersion_std"
                        ]
                    ),
            }

            for _, row
            in metricas_cohesion.iterrows()
        },

        "top_escenarios": [],

        "evaluacion_k":
            evaluacion_k.to_dict(
                orient="records"
            ),
    }

    for _, row in escenarios.head(
        10
    ).iterrows():

        resumen[
            "top_escenarios"
        ].append(
            {
                "ranking":
                    safe_int(
                        row[
                            "ranking_escenario"
                        ]
                    ),

                "escenario_id":
                    str(
                        row[
                            "escenario_id"
                        ]
                    ),

                "score":
                    safe_float(
                        row[
                            "score_escenario"
                        ]
                    ),

                "cantidad_proyectos":
                    safe_int(
                        row[
                            "cantidad_proyectos"
                        ]
                    ),

                "tipo":
                    str(
                        row[
                            "tipo_escenario"
                        ]
                    ),

                "horizonte":
                    str(
                        row[
                            "horizonte_escenario"
                        ]
                    ),

                "prioridad":
                    str(
                        row[
                            "prioridad_escenario"
                        ]
                    ),
            }
        )

    return resumen


# =============================================================================
# EXPORTACIÓN
# =============================================================================

def guardar_archivos(
    escenarios: gpd.GeoDataFrame,
    proyectos: gpd.GeoDataFrame,
    resumen: dict[str, Any],
    evaluacion_k: pd.DataFrame,
    metricas_cohesion: pd.DataFrame,
) -> None:

    imprimir_subtitulo(
        "14",
        "GUARDANDO ARCHIVOS",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 01 - Escenarios Parquet
    # -------------------------------------------------------------------------

    escenario_parquet = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.parquet"
    )

    escenarios.to_parquet(
        escenario_parquet,
        index=False,
    )

    print(
        f"Parquet escenarios:\n"
        f"{escenario_parquet}"
    )

    # -------------------------------------------------------------------------
    # 02 - Escenarios CSV
    # -------------------------------------------------------------------------

    escenario_csv = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.csv"
    )

    escenarios.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        escenario_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"CSV escenarios:\n"
        f"{escenario_csv}"
    )

    # -------------------------------------------------------------------------
    # 03 - GeoPackage
    # -------------------------------------------------------------------------

    gpkg = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.gpkg"
    )

    if gpkg.exists():

        try:
            gpkg.unlink()
        except Exception as exc:

            print(
                "ADVERTENCIA: no se pudo eliminar "
                f"el GPKG anterior: {exc}"
            )

    escenarios.to_file(
        gpkg,
        layer="escenarios",
        driver="GPKG",
    )

    print(
        f"GeoPackage:\n"
        f"{gpkg}"
    )

    # -------------------------------------------------------------------------
    # 04 - Proyectos Parquet
    # -------------------------------------------------------------------------

    proyectos_parquet = (
        OUTPUT_DIR
        / "proyectos_escenarios_territoriales_amba.parquet"
    )

    proyectos.to_parquet(
        proyectos_parquet,
        index=False,
    )

    print(
        f"Parquet proyectos:\n"
        f"{proyectos_parquet}"
    )

    # -------------------------------------------------------------------------
    # 05 - Proyectos CSV
    # -------------------------------------------------------------------------

    proyectos_csv = (
        OUTPUT_DIR
        / "proyectos_escenarios_territoriales_amba.csv"
    )

    proyectos.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        proyectos_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"CSV proyectos:\n"
        f"{proyectos_csv}"
    )

    # -------------------------------------------------------------------------
    # 06 - JSON
    # -------------------------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba_resumen.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"JSON:\n"
        f"{json_path}"
    )

    # -------------------------------------------------------------------------
    # 07 - Evaluación K
    # -------------------------------------------------------------------------

    evaluacion_path = (
        OUTPUT_DIR
        / "evaluacion_numero_escenarios.csv"
    )

    evaluacion_k.to_csv(
        evaluacion_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Evaluación K:\n"
        f"{evaluacion_path}"
    )

    # -------------------------------------------------------------------------
    # 08 - Diagnóstico
    # -------------------------------------------------------------------------

    diagnostico_path = (
        OUTPUT_DIR
        / "diagnostico_escenarios.csv"
    )

    escenarios.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        diagnostico_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Diagnóstico:\n"
        f"{diagnostico_path}"
    )

    # -------------------------------------------------------------------------
    # 09 - Cohesión
    # -------------------------------------------------------------------------

    cohesion_path = (
        OUTPUT_DIR
        / "metricas_cohesion_escenarios.csv"
    )

    metricas_cohesion.to_csv(
        cohesion_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Cohesión:\n"
        f"{cohesion_path}"
    )


# =============================================================================
# MATPLOTLIB
# =============================================================================

def importar_matplotlib():

    try:

        import matplotlib.pyplot as plt

        return plt

    except ImportError:

        print(
            "ADVERTENCIA: Matplotlib no disponible. "
            "Se omiten mapas y gráficos."
        )

        return None


# =============================================================================
# MAPAS
# =============================================================================

def generar_mapa(
    gdf: gpd.GeoDataFrame,
    columna: str,
    titulo: str,
    archivo: str,
) -> None:

    if gdf.empty:
        return

    plt = importar_matplotlib()

    if plt is None:
        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    try:

        gdf.plot(
            ax=ax,
            column=columna,
            legend=True,
            alpha=0.75,
        )

    except Exception:

        gdf.plot(
            ax=ax,
            alpha=0.75,
        )

    ax.set_title(
        titulo,
        fontsize=15,
    )

    ax.set_axis_off()

    path = (
        OUTPUT_DIR
        / archivo
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Mapa: {path}"
    )


def generar_mapa_proyectos(
    proyectos: gpd.GeoDataFrame,
) -> None:

    if proyectos.empty:
        return

    plt = importar_matplotlib()

    if plt is None:
        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    try:

        proyectos.plot(
            ax=ax,
            column="escenario_id",
            legend=True,
            markersize=12,
            alpha=0.75,
        )

    except Exception:

        proyectos.plot(
            ax=ax,
            markersize=12,
            alpha=0.75,
        )

    ax.set_title(
        "Proyectos asignados a escenarios territoriales AMBA",
        fontsize=15,
    )

    ax.set_axis_off()

    path = (
        OUTPUT_DIR
        / "05_mapa_proyectos_por_escenario.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Mapa: {path}"
    )


# =============================================================================
# GRÁFICOS
# =============================================================================

def generar_graficos(
    escenarios: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
) -> None:

    plt = importar_matplotlib()

    if plt is None:
        return

    # -------------------------------------------------------------------------
    # 06 - Demanda vs déficit
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.scatter(
        escenarios[
            "demanda_cubierta"
        ],
        escenarios[
            "deficit_atendido"
        ],
        s=100,
    )

    for _, row in escenarios.iterrows():

        ax.annotate(
            row[
                "escenario_id"
            ],
            (
                row[
                    "demanda_cubierta"
                ],
                row[
                    "deficit_atendido"
                ],
            ),
        )

    ax.set_xlabel(
        "Demanda cubierta"
    )

    ax.set_ylabel(
        "Déficit atendido"
    )

    ax.set_title(
        "Demanda vs déficit atendido"
    )

    ax.grid(
        alpha=0.25
    )

    path = (
        OUTPUT_DIR
        / "06_demanda_vs_deficit_atendido.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico: {path}"
    )

    # -------------------------------------------------------------------------
    # 07 - Prioridad
    # -------------------------------------------------------------------------

    conteo = (
        escenarios[
            "prioridad_escenario"
        ]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    conteo.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Escenarios por prioridad"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    path = (
        OUTPUT_DIR
        / "07_escenarios_por_prioridad.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico: {path}"
    )

    # -------------------------------------------------------------------------
    # 08 - Horizonte
    # -------------------------------------------------------------------------

    conteo = (
        escenarios[
            "horizonte_escenario"
        ]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    conteo.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Escenarios por horizonte"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    path = (
        OUTPUT_DIR
        / "08_escenarios_por_horizonte.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico: {path}"
    )

    # -------------------------------------------------------------------------
    # 09 - Score
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.bar(
        escenarios[
            "escenario_id"
        ],
        escenarios[
            "score_escenario"
        ],
    )

    ax.set_title(
        "Score de escenarios territoriales"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    path = (
        OUTPUT_DIR
        / "09_distribucion_score_escenarios.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico: {path}"
    )

    # -------------------------------------------------------------------------
    # 10 - Evaluación K
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.plot(
        evaluacion_k[
            "k_solicitado"
        ],
        evaluacion_k[
            "silhouette"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Cantidad de escenarios K"
    )

    ax.set_ylabel(
        "Silhouette"
    )

    ax.set_title(
        "Evaluación del número de escenarios"
    )

    ax.grid(
        alpha=0.25
    )

    path = (
        OUTPUT_DIR
        / "10_evaluacion_silhouette_k.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico: {path}"
    )


# =============================================================================
# RESUMEN FINAL
# =============================================================================

def mostrar_resumen_final(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
) -> None:

    imprimir_titulo(
        "27 - PROCESO FINALIZADO CORRECTAMENTE"
    )

    print(
        f"Proyectos analizados     : "
        f"{len(proyectos):,}"
    )

    print(
        f"Escenarios territoriales : "
        f"{len(escenarios):,}"
    )

    print()
    print(
        "DISTRIBUCIÓN DE PROYECTOS"
    )

    for _, row in escenarios.iterrows():

        print(
            f"  {row['escenario_id']}: "
            f"{int(row['cantidad_proyectos']):3d} proyectos | "
            f"{row['tipo_escenario']:<24} | "
            f"score={row['score_escenario']:.2f}"
        )

    print()
    print(
        "PRIORIDADES:"
    )

    print(
        escenarios[
            "prioridad_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "HORIZONTES:"
    )

    print(
        escenarios[
            "horizonte_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "TIPOS DE ESCENARIO:"
    )

    print(
        escenarios[
            "tipo_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "DIAGNÓSTICOS:"
    )

    print(
        escenarios[
            "diagnostico_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "ARCHIVOS GENERADOS:"
    )

    if OUTPUT_DIR.exists():

        for archivo in sorted(
            OUTPUT_DIR.iterdir()
        ):

            if archivo.is_file():

                print(
                    f"  {archivo.name}"
                )

    print()
    print(
        "SIGUIENTE ETAPA"
    )

    print(
        "Evaluar los escenarios territoriales mediante "
        "simulación de impactos, cobertura, demanda, "
        "déficit y selección de escenarios estratégicos."
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    imprimir_titulo(
        "27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA"
    )

    print(
        f"Proyecto    : {PROJECT_DIR}"
    )

    print(
        f"Entrada     : {INPUT_PATH}"
    )

    print(
        f"Salida      : {OUTPUT_DIR}"
    )

    print(
        f"CRS         : {CRS_GEOGRAFICO}"
    )

    print(
        f"CRS métrico : {CRS_METRICO}"
    )

    print()
    print(
        "CONFIGURACIÓN"
    )

    print(
        f"  K candidatos         : "
        f"{K_MIN} - {K_MAX}"
    )

    print(
        f"  Mínimo proyectos     : "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )

    print(
        "  Método               : "
        "KMeans multicriterio + reparación restringida"
    )

    print(
        f"  Random state         : "
        f"{RANDOM_STATE}"
    )

    print()
    print(
        "PESOS"
    )

    print(
        f"  Componente espacial  : "
        f"{PESO_ESPACIAL:.0%}"
    )

    print(
        f"  Componente territorial: "
        f"{PESO_TERRITORIAL:.0%}"
    )

    # =========================================================================
    # 1
    # =========================================================================

    imprimir_subtitulo(
        "1",
        "CARGANDO CARTERA DEL PROCESO 26",
    )

    cartera = cargar_cartera()

    print(
        f"Registros: {len(cartera):,}"
    )

    print(
        f"Columnas : {len(cartera.columns):,}"
    )

    print(
        f"CRS      : {cartera.crs}"
    )

    # =========================================================================
    # 2
    # =========================================================================

    columna_id = validar_entrada(
        cartera
    )

    # =========================================================================
    # 3
    # =========================================================================

    columnas = validar_componentes(
        cartera
    )

    # =========================================================================
    # 4
    # =========================================================================

    cartera = preparar_variables_espaciales(
        cartera
    )

    # =========================================================================
    # 5
    # =========================================================================

    (
        X,
        matriz_variables,
        pesos,
    ) = construir_matriz_multicriterio(
        cartera,
        columnas,
    )

    # =========================================================================
    # 6
    # =========================================================================

    (
        k_seleccionado,
        labels,
        evaluacion_k,
    ) = seleccionar_k(
        X
    )

    # =========================================================================
    # 7
    # =========================================================================

    cartera = preparar_indicadores_globales(
        cartera,
        columnas,
    )

    # =========================================================================
    # 8
    # =========================================================================

    escenarios = construir_escenarios(
        cartera,
        columnas,
        labels,
    )

    # =========================================================================
    # 9
    # =========================================================================

    proyectos = asignar_escenarios(
        cartera,
        escenarios,
        labels,
    )

    # =========================================================================
    # 10
    # =========================================================================

    metricas_cohesion = (
        calcular_metricas_cohesion(
            X,
            labels,
        )
    )

    escenarios = escenarios.merge(
        metricas_cohesion,
        on="cluster_territorial",
        how="left",
        validate="one_to_one",
    )

    # =========================================================================
    # 11
    # =========================================================================

    geometrias = construir_geometrias(
        proyectos,
        escenarios,
    )

    # =========================================================================
    # 12
    # =========================================================================

    validar_final(
        proyectos,
        escenarios,
        geometrias,
        evaluacion_k,
    )

    # =========================================================================
    # 13
    # =========================================================================

    imprimir_subtitulo(
        "13",
        "RANKING DE ESCENARIOS",
    )

    columnas_ranking = [
        "ranking_escenario",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "horizonte_escenario",
        "score_escenario",
        "prioridad_escenario",
        "impacto_territorial",
        "cobertura_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
    ]

    print(
        escenarios[
            columnas_ranking
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 14
    # =========================================================================

    resumen = construir_resumen(
        proyectos,
        escenarios,
        evaluacion_k,
        metricas_cohesion,
    )

    guardar_archivos(
        geometrias,
        proyectos,
        resumen,
        evaluacion_k,
        metricas_cohesion,
    )

    # =========================================================================
    # 15
    # =========================================================================

    imprimir_subtitulo(
        "15",
        "GENERANDO MAPAS",
    )

    generar_mapa(
        geometrias,
        "ranking_escenario",
        "Escenarios territoriales AMBA",
        "01_mapa_escenarios_territoriales.png",
    )

    generar_mapa(
        geometrias,
        "score_escenario",
        "Score de escenarios territoriales",
        "02_mapa_prioridad_escenarios.png",
    )

    generar_mapa(
        geometrias,
        "cobertura_territorial",
        "Cobertura territorial por escenario",
        "03_mapa_cobertura_metropolitana.png",
    )

    generar_mapa(
        geometrias,
        "impacto_territorial",
        "Impacto territorial por escenario",
        "04_mapa_impacto_territorial.png",
    )

    generar_mapa_proyectos(
        proyectos
    )

    # =========================================================================
    # 16
    # =========================================================================

    imprimir_subtitulo(
        "16",
        "GENERANDO GRÁFICOS",
    )

    generar_graficos(
        escenarios,
        evaluacion_k,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    mostrar_resumen_final(
        proyectos,
        escenarios,
    )


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Proceso interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "ERROR DURANTE EL PROCESO 27"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise