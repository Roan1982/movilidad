# -*- coding: utf-8 -*-

"""
27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - V7

Reconstrucción completa del proceso 27.

Objetivo
--------
Construir escenarios territoriales de proyectos de movilidad para AMBA
mediante clustering multicriterio espacial + territorial.

Principios V7
-------------
1. No utilizar valores ficticios 50 para indicadores faltantes.
2. Resolver nombres de columnas mediante normalización robusta.
3. Utilizar los indicadores reales presentes en la cartera.
4. Separar:
      - necesidad territorial
      - capacidad estratégica
5. Combinar componente espacial y territorial.
6. Evaluar K entre 6 y 12.
7. Garantizar un mínimo de proyectos por escenario.
8. Generar métricas de cohesión.
9. Generar geometrías agregadas.
10. Generar mapas, gráficos, CSV, Parquet, GeoPackage y JSON.

Entrada
-------
data/processed/cartera_proyectos_amba/cartera_proyectos_amba.parquet

Salida
------
data/processed/escenarios_territoriales_amba/
"""

from __future__ import annotations

import json
import math
import re
import shutil
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V7"

RANDOM_STATE = 42

K_MIN = 6
K_MAX = 12

MIN_PROYECTOS = 8

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

PESO_ESPACIAL = 0.55
PESO_TERRITORIAL = 0.45

PESO_NECESIDAD = 0.40
PESO_ESTRATEGICA = 0.60


# =============================================================================
# RUTAS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "cartera_proyectos_amba"
)

INPUT_FILE = INPUT_DIR / "cartera_proyectos_amba.parquet"

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)


# =============================================================================
# INDICADORES CANÓNICOS
# =============================================================================

INDICADORES = {
    "demanda": [
        "indice_demanda_estructural",
        "demanda",
        "demanda_estructural",
        "score_demanda",
    ],

    "deficit": [
        "deficit_infraestructura",
        "deficit",
        "deficit_estructural",
        "score_deficit",
    ],

    "conectividad": [
        "indice_conectividad_estructural",
        "conectividad",
        "conectividad_estructural",
        "score_conectividad",
    ],

    "intermodalidad": [
        "indice_intermodalidad_estructural",
        "intermodalidad",
        "intermodalidad_estructural",
        "score_intermodalidad",
    ],

    "integracion": [
        "indice_integracion_territorial",
        "integracion",
        "integracion_territorial",
        "score_integracion",
    ],

    "centralidad": [
        "indice_centralidad_estructural",
        "centralidad",
        "centralidad_estructural",
        "score_centralidad",
    ],

    "impacto": [
        "impacto_potencial",
        "impacto",
        "impacto_territorial",
        "score_impacto",
    ],

    "urgencia": [
        "urgencia_intervencion",
        "urgencia",
        "score_urgencia",
    ],

    "prioridad_territorial": [
        "score_prioridad_territorial",
        "prioridad_territorial",
        "prioridad",
        "score_prioridad",
    ],

    "score_cartera": [
        "score_cartera",
        "score_proyecto",
        "score_total",
    ],
}


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto: str):
    print()
    print("=" * 80)
    print(texto)
    print("=" * 80)


def normalizar_nombre(nombre: str) -> str:
    """
    Normaliza nombres de columnas para evitar problemas de:
    mayúsculas/minúsculas, tildes, espacios, guiones, etc.
    """
    texto = str(nombre)

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    ).encode(
        "ascii",
        "ignore",
    ).decode("ascii")

    texto = texto.lower().strip()

    texto = re.sub(r"[^a-z0-9]+", "_", texto)

    texto = re.sub(r"_+", "_", texto)

    return texto.strip("_")


def construir_indice_columnas(df: pd.DataFrame) -> dict:
    """
    Construye:

        nombre_normalizado -> nombre_real
    """

    indice = {}

    for columna in df.columns:

        normalizada = normalizar_nombre(columna)

        if normalizada not in indice:
            indice[normalizada] = columna

    return indice


def buscar_columna(
    indice: dict,
    candidatos: list[str],
) -> str | None:

    for candidato in candidatos:

        candidato_norm = normalizar_nombre(candidato)

        if candidato_norm in indice:
            return indice[candidato_norm]

    return None


def resolver_indicadores(df: pd.DataFrame) -> dict:
    """
    Resuelve todos los indicadores utilizando los nombres reales
    de la cartera.

    Importante:
    NO asigna valores artificiales.
    """

    indice = construir_indice_columnas(df)

    resultado = {}

    for canonico, candidatos in INDICADORES.items():

        encontrada = buscar_columna(
            indice,
            candidatos,
        )

        resultado[canonico] = encontrada

    return resultado


def convertir_numerico(
    serie: pd.Series,
    nombre: str,
) -> pd.Series:

    salida = pd.to_numeric(
        serie,
        errors="coerce",
    )

    if salida.notna().sum() == 0:

        raise ValueError(
            f"El indicador '{nombre}' no contiene valores numéricos."
        )

    return salida


def normalizar_0_100(
    serie: pd.Series,
    nombre: str,
) -> pd.Series:

    x = convertir_numerico(
        serie,
        nombre,
    )

    minimo = x.min()
    maximo = x.max()

    if pd.isna(minimo) or pd.isna(maximo):

        raise ValueError(
            f"No se pudo normalizar '{nombre}'."
        )

    if math.isclose(
        float(minimo),
        float(maximo),
    ):

        raise ValueError(
            f"El indicador '{nombre}' es constante "
            f"({minimo}). No puede utilizarse para discriminar "
            f"escenarios."
        )

    return (
        (x - minimo)
        / (maximo - minimo)
        * 100.0
    )


def minmax_array(X: np.ndarray) -> np.ndarray:

    X = np.asarray(
        X,
        dtype=float,
    )

    minimo = np.nanmin(
        X,
        axis=0,
    )

    maximo = np.nanmax(
        X,
        axis=0,
    )

    denominador = maximo - minimo

    denominador[
        np.isclose(denominador, 0)
    ] = 1.0

    return (
        (X - minimo)
        / denominador
    )


def limitar_0_100(valor):

    if pd.isna(valor):
        return 0.0

    return float(
        np.clip(
            valor,
            0,
            100,
        )
    )


# =============================================================================
# CARGA
# =============================================================================

def cargar_cartera() -> gpd.GeoDataFrame:

    titulo(
        "1. CARGANDO CARTERA DEL PROCESO 26"
    )

    print(
        f"Archivo de entrada:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"No existe la entrada:\n{INPUT_FILE}"
        )

    gdf = gpd.read_parquet(
        INPUT_FILE
    )

    print(
        f"Registros: {len(gdf):,}"
    )

    print(
        f"Columnas : {len(gdf.columns)}"
    )

    print(
        f"CRS      : {gdf.crs}"
    )

    return gdf


# =============================================================================
# VALIDACIÓN
# =============================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
):

    titulo(
        "2. VALIDANDO DATOS DE ENTRADA"
    )

    if len(gdf) == 0:

        raise ValueError(
            "La cartera está vacía."
        )

    if "proyecto_id" not in gdf.columns:

        raise ValueError(
            "La cartera no contiene 'proyecto_id'."
        )

    duplicados = int(
        gdf["proyecto_id"]
        .duplicated()
        .sum()
    )

    nulos_geom = int(
        gdf.geometry.isna().sum()
    )

    vacias_geom = int(
        gdf.geometry.is_empty.sum()
    )

    invalidas_geom = int(
        (~gdf.geometry.is_valid)
        .sum()
    )

    print(
        f"Registros              : {len(gdf)}"
    )

    print(
        f"Columnas               : {len(gdf.columns)}"
    )

    print(
        f"CRS                    : {gdf.crs}"
    )

    print(
        f"Geometrías nulas       : {nulos_geom}"
    )

    print(
        f"Geometrías vacías      : {vacias_geom}"
    )

    print(
        f"Geometrías inválidas   : {invalidas_geom}"
    )

    print(
        "ID utilizado           : proyecto_id"
    )

    print(
        f"IDs duplicados         : {duplicados}"
    )

    if duplicados:
        raise ValueError(
            "Existen proyecto_id duplicados."
        )

    if nulos_geom:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if vacias_geom:
        raise ValueError(
            "Existen geometrías vacías."
        )

    if invalidas_geom:

        print(
            "Intentando reparar geometrías..."
        )

        gdf.geometry = gdf.geometry.make_valid()

        invalidas_post = int(
            (~gdf.geometry.is_valid)
            .sum()
        )

        if invalidas_post:

            raise ValueError(
                "Persisten geometrías inválidas."
            )

    if gdf.crs is None:

        print(
            "CRS ausente. Se asume EPSG:4326."
        )

        gdf = gdf.set_crs(
            CRS_GEOGRAFICO
        )

    else:

        gdf = gdf.to_crs(
            CRS_GEOGRAFICO
        )

    print(
        f"Proyectos válidos       : {len(gdf)}"
    )

    print(
        "Validación de entrada: OK"
    )

    return gdf


# =============================================================================
# INDICADORES
# =============================================================================

def preparar_indicadores(
    gdf: gpd.GeoDataFrame,
):

    titulo(
        "3. RESOLVIENDO INDICADORES TERRITORIALES"
    )

    columnas = resolver_indicadores(
        gdf
    )

    print()

    for canonico, real in columnas.items():

        if real is None:

            print(
                f"  {canonico:<25} NO ENCONTRADO"
            )

        else:

            print(
                f"  {canonico:<25} OK -> {real}"
            )

    # -------------------------------------------------------------------------
    # Indicadores obligatorios para construir el modelo
    # -------------------------------------------------------------------------

    obligatorios = [
        "demanda",
        "deficit",
        "conectividad",
        "intermodalidad",
        "integracion",
        "centralidad",
        "impacto",
        "urgencia",
    ]

    faltantes = [
        x
        for x in obligatorios
        if columnas[x] is None
    ]

    if faltantes:

        raise ValueError(
            "\n\n"
            "NO SE PUEDE CONSTRUIR V7.\n\n"
            "Faltan indicadores territoriales obligatorios:\n"
            + "\n".join(
                f"  - {x}"
                for x in faltantes
            )
            + "\n\n"
            "La V7 no reemplaza indicadores faltantes "
            "por 50 porque eso destruye la capacidad "
            "de discriminación territorial."
        )

    # -------------------------------------------------------------------------
    # Crear columnas canónicas
    # -------------------------------------------------------------------------

    for canonico, real in columnas.items():

        if real is not None:

            gdf[f"_v7_{canonico}"] = convertir_numerico(
                gdf[real],
                canonico,
            )

    # -------------------------------------------------------------------------
    # Normalización global
    # -------------------------------------------------------------------------

    titulo(
        "4. NORMALIZACIÓN GLOBAL DE INDICADORES"
    )

    for canonico in obligatorios + ["score_cartera"]:

        columna = f"_v7_{canonico}"

        if columna not in gdf.columns:

            if canonico == "score_cartera":

                print(
                    "  score_cartera            NO DISPONIBLE"
                )

                continue

            raise ValueError(
                f"Falta columna canónica: {columna}"
            )

        gdf[
            f"norm_{canonico}"
        ] = normalizar_0_100(
            gdf[columna],
            canonico,
        )

        print(
            f"  {canonico:<25} OK"
        )

    print()

    print(
        "Los indicadores fueron normalizados "
        "globalmente sobre los 144 proyectos."
    )

    print(
        "No se utilizaron valores artificiales."
    )

    return gdf, columnas


# =============================================================================
# COMPONENTE ESPACIAL
# =============================================================================

def preparar_espacio(
    gdf: gpd.GeoDataFrame,
):

    titulo(
        "5. PREPARANDO COMPONENTE ESPACIAL"
    )

    metric = gdf.to_crs(
        CRS_METRICO
    )

    centroids = metric.geometry.centroid

    gdf["x_m"] = centroids.x
    gdf["y_m"] = centroids.y

    print(
        f"X mínimo: {gdf['x_m'].min():,.2f} m"
    )

    print(
        f"X máximo: {gdf['x_m'].max():,.2f} m"
    )

    print(
        f"Y mínimo: {gdf['y_m'].min():,.2f} m"
    )

    print(
        f"Y máximo: {gdf['y_m'].max():,.2f} m"
    )

    print(
        "Componente espacial: OK"
    )

    return gdf


# =============================================================================
# MATRIZ MULTICRITERIO
# =============================================================================

def construir_matriz(
    gdf: gpd.GeoDataFrame,
):

    titulo(
        "6. CONSTRUYENDO MATRIZ MULTICRITERIO V7"
    )

    indicadores = [
        "demanda",
        "deficit",
        "conectividad",
        "intermodalidad",
        "integracion",
        "centralidad",
        "impacto",
        "urgencia",
    ]

    # -------------------------------------------------------------------------
    # Peso territorial
    #
    # Los indicadores de necesidad tienen mayor peso dentro de la parte
    # territorial, mientras que conectividad/intermodalidad/integración/
    # centralidad/impacto representan la dimensión estratégica.
    # -------------------------------------------------------------------------

    pesos_territoriales = {
        "demanda": 0.20,
        "deficit": 0.20,
        "urgencia": 0.10,
        "conectividad": 0.125,
        "intermodalidad": 0.125,
        "integracion": 0.10,
        "centralidad": 0.075,
        "impacto": 0.075,
    }

    # -------------------------------------------------------------------------
    # Componente espacial
    # -------------------------------------------------------------------------

    X_espacial = gdf[
        [
            "x_m",
            "y_m",
        ]
    ].to_numpy()

    scaler_espacial = StandardScaler()

    X_espacial_std = scaler_espacial.fit_transform(
        X_espacial
    )

    # -------------------------------------------------------------------------
    # Componente territorial
    # -------------------------------------------------------------------------

    X_territorial = gdf[
        [
            f"norm_{x}"
            for x in indicadores
        ]
    ].to_numpy()

    scaler_territorial = StandardScaler()

    X_territorial_std = scaler_territorial.fit_transform(
        X_territorial
    )

    # -------------------------------------------------------------------------
    # Balancear componentes
    # -------------------------------------------------------------------------

    X_espacial_final = (
        X_espacial_std
        * math.sqrt(PESO_ESPACIAL / 2.0)
    )

    columnas_territoriales = []

    for i, indicador in enumerate(
        indicadores
    ):

        peso = (
            PESO_TERRITORIAL
            * pesos_territoriales[indicador]
        )

        columna = (
            X_territorial_std[:, i]
            * math.sqrt(peso)
        )

        columnas_territoriales.append(
            columna
        )

    X_territorial_final = np.column_stack(
        columnas_territoriales
    )

    X = np.column_stack(
        [
            X_espacial_final,
            X_territorial_final,
        ]
    )

    nombres_variables = [
        "x_m",
        "y_m",
    ] + indicadores

    print(
        f"Proyectos        : {len(gdf)}"
    )

    print(
        f"Variables        : {len(nombres_variables)}"
    )

    print(
        f"Peso espacial    : {PESO_ESPACIAL:.0%}"
    )

    print(
        f"Peso territorial : {PESO_TERRITORIAL:.0%}"
    )

    print()

    print(
        "Pesos territoriales efectivos:"
    )

    for indicador in indicadores:

        print(
            f"  {indicador:<20} "
            f"{PESO_TERRITORIAL * pesos_territoriales[indicador]:.4f}"
        )

    return X, indicadores


# =============================================================================
# SELECCIÓN DE K
# =============================================================================

def seleccionar_k(
    X: np.ndarray,
):

    titulo(
        "7. SELECCIONANDO CANTIDAD DE ESCENARIOS V7"
    )

    resultados = []

    n = len(X)

    for k in range(
        K_MIN,
        K_MAX + 1,
    ):

        if k * MIN_PROYECTOS > n:

            continue

        modelo = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=30,
        )

        labels = modelo.fit_predict(
            X
        )

        conteos = np.bincount(
            labels,
            minlength=k,
        )

        minimo = int(
            conteos.min()
        )

        maximo = int(
            conteos.max()
        )

        # ---------------------------------------------------------------------
        # Si algún escenario queda debajo del mínimo, descartamos ese K.
        # ---------------------------------------------------------------------

        if minimo < MIN_PROYECTOS:

            print(
                f"K={k:2d} | DESCARTADO | "
                f"mín={minimo:2d} < {MIN_PROYECTOS}"
            )

            continue

        sil = silhouette_score(
            X,
            labels,
        )

        db = davies_bouldin_score(
            X,
            labels,
        )

        equilibrio = (
            1
            - (
                np.std(conteos)
                / np.mean(conteos)
            )
        )

        equilibrio = float(
            np.clip(
                equilibrio,
                0,
                1,
            )
        )

        # DB menor es mejor.
        db_score = 1.0 / (
            1.0 + db
        )

        score = (
            0.50 * sil
            + 0.25 * db_score
            + 0.25 * equilibrio
        )

        resultados.append(
            {
                "k": k,
                "silhouette": sil,
                "davies_bouldin": db,
                "min_proyectos": minimo,
                "max_proyectos": maximo,
                "equilibrio": equilibrio,
                "score": score,
                "labels": labels,
                "modelo": modelo,
            }
        )

        print(
            f"K={k:2d} | "
            f"sil={sil:.4f} | "
            f"DB={db:.4f} | "
            f"mín={minimo:2d} | "
            f"máx={maximo:2d} | "
            f"eq={equilibrio:.3f} | "
            f"score={score:.4f}"
        )

    if not resultados:

        raise RuntimeError(
            "No existe ningún K válido respetando "
            f"MIN_PROYECTOS={MIN_PROYECTOS}."
        )

    mejor = max(
        resultados,
        key=lambda x: x["score"],
    )

    print()

    print(
        f"K seleccionado : {mejor['k']}"
    )

    print(
        f"Score selección : {mejor['score']:.4f}"
    )

    return mejor, resultados


# =============================================================================
# REPARACIÓN DE CLUSTERS
# =============================================================================

def reparar_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    min_proyectos: int,
) -> np.ndarray:

    """
    Repara clusters pequeños.

    La reparación mueve proyectos del cluster pequeño hacia el
    cluster válido más cercano según distancia al centroide.
    """

    labels = labels.copy()

    while True:

        valores, conteos = np.unique(
            labels,
            return_counts=True,
        )

        pequenos = [
            int(v)
            for v, c in zip(
                valores,
                conteos,
            )
            if c < min_proyectos
        ]

        if not pequenos:
            break

        if len(valores) <= 1:

            raise RuntimeError(
                "No es posible reparar los clusters."
            )

        centros = {}

        for cluster in valores:

            indices = np.where(
                labels == cluster
            )[0]

            centros[int(cluster)] = X[
                indices
            ].mean(axis=0)

        cluster_pequeno = pequenos[0]

        indices_pequeno = np.where(
            labels == cluster_pequeno
        )[0]

        # ---------------------------------------------------------------------
        # Para cada proyecto del cluster pequeño buscamos el cluster válido
        # más próximo.
        # ---------------------------------------------------------------------

        candidatos = [
            int(v)
            for v, c in zip(
                valores,
                conteos,
            )
            if (
                v != cluster_pequeno
                and c >= min_proyectos
            )
        ]

        if not candidatos:

            # Si todos son pequeños, usar el mayor.
            candidatos = [
                int(
                    valores[
                        np.argmax(conteos)
                    ]
                )
            ]

        mejor_movimiento = None

        for idx in indices_pequeno:

            for destino in candidatos:

                distancia = np.linalg.norm(
                    X[idx]
                    - centros[destino]
                )

                if (
                    mejor_movimiento is None
                    or distancia
                    < mejor_movimiento[0]
                ):

                    mejor_movimiento = (
                        distancia,
                        idx,
                        destino,
                    )

        _, idx, destino = (
            mejor_movimiento
        )

        labels[idx] = destino

    # -------------------------------------------------------------------------
    # Renumerar clusters 1..K
    # -------------------------------------------------------------------------

    valores = sorted(
        np.unique(labels)
    )

    mapa = {
        viejo: nuevo
        for nuevo, viejo in enumerate(
            valores,
            start=1,
        )
    }

    labels = np.array(
        [
            mapa[x]
            for x in labels
        ],
        dtype=int,
    )

    return labels


# =============================================================================
# MÉTRICAS DE PROYECTO
# =============================================================================

def calcular_scores_proyecto(
    gdf: gpd.GeoDataFrame,
):

    # -------------------------------------------------------------------------
    # Necesidad territorial
    # -------------------------------------------------------------------------

    gdf["necesidad_territorial_proyecto"] = (
        0.35 * gdf["norm_demanda"]
        + 0.35 * gdf["norm_deficit"]
        + 0.15 * gdf["norm_urgencia"]
        + 0.15 * gdf["norm_conectividad"]
    )

    # -------------------------------------------------------------------------
    # Capacidad estratégica
    # -------------------------------------------------------------------------

    gdf["capacidad_estrategica_proyecto"] = (
        0.20 * gdf["norm_conectividad"]
        + 0.20 * gdf["norm_intermodalidad"]
        + 0.20 * gdf["norm_integracion"]
        + 0.15 * gdf["norm_centralidad"]
        + 0.15 * gdf["norm_impacto"]
        + 0.10 * gdf["norm_urgencia"]
    )

    # -------------------------------------------------------------------------
    # Impacto
    # -------------------------------------------------------------------------

    gdf["impacto_territorial_proyecto"] = (
        0.50 * gdf["norm_impacto"]
        + 0.25 * gdf["norm_integracion"]
        + 0.25 * gdf["norm_centralidad"]
    )

    # -------------------------------------------------------------------------
    # Cobertura
    # -------------------------------------------------------------------------

    gdf["cobertura_territorial_proyecto"] = (
        0.40 * gdf["norm_conectividad"]
        + 0.30 * gdf["norm_integracion"]
        + 0.30 * gdf["norm_intermodalidad"]
    )

    # -------------------------------------------------------------------------
    # Déficit atendido
    # -------------------------------------------------------------------------

    gdf["deficit_atendido_proyecto"] = (
        0.60 * gdf["norm_deficit"]
        + 0.20 * gdf["norm_conectividad"]
        + 0.20 * gdf["norm_integracion"]
    )

    # -------------------------------------------------------------------------
    # Demanda cubierta
    # -------------------------------------------------------------------------

    gdf["demanda_cubierta_proyecto"] = (
        0.70 * gdf["norm_demanda"]
        + 0.15 * gdf["norm_conectividad"]
        + 0.15 * gdf["norm_intermodalidad"]
    )

    # -------------------------------------------------------------------------
    # Complementariedad
    # -------------------------------------------------------------------------

    gdf["complementariedad_proyecto"] = (
        0.35 * gdf["norm_intermodalidad"]
        + 0.30 * gdf["norm_integracion"]
        + 0.20 * gdf["norm_conectividad"]
        + 0.15 * gdf["norm_centralidad"]
    )

    # -------------------------------------------------------------------------
    # Score final de proyecto
    # -------------------------------------------------------------------------

    gdf["score_territorial_proyecto"] = (
        PESO_NECESIDAD
        * gdf["necesidad_territorial_proyecto"]
        + PESO_ESTRATEGICA
        * gdf["capacidad_estrategica_proyecto"]
    )

    return gdf


# =============================================================================
# CLASIFICACIÓN
# =============================================================================

def clasificar_prioridad(
    score: float,
) -> str:

    if score >= 70:
        return "PRIORIDAD_1_ALTA"

    if score >= 55:
        return "PRIORIDAD_2_MEDIA_ALTA"

    if score >= 40:
        return "PRIORIDAD_3_MEDIA"

    if score >= 25:
        return "PRIORIDAD_4_MEDIA_BAJA"

    return "PRIORIDAD_5_BAJA"


def clasificar_horizonte(
    score: float,
) -> str:

    if score >= 70:
        return "CORTO_PLAZO"

    if score >= 55:
        return "MEDIANO_PLAZO"

    return "LARGO_PLAZO"


def clasificar_tipo(
    necesidad: float,
    estrategia: float,
) -> str:

    diferencia = necesidad - estrategia

    if diferencia >= 15:

        return "ESCENARIO_DE_NECESIDAD"

    if diferencia <= -15:

        return "ESCENARIO_ESTRATEGICO"

    return "ESCENARIO_INTEGRADO"


def dimension_dominante(
    fila: pd.Series,
) -> str:

    dimensiones = {
        "DEMANDA": fila["demanda_cubierta"],
        "DEFICIT": fila["deficit_atendido"],
        "CONECTIVIDAD": fila["conectividad"],
        "INTERMODALIDAD": fila["intermodalidad"],
        "INTEGRACION": fila["integracion"],
        "CENTRALIDAD": fila["centralidad"],
        "IMPACTO": fila["impacto_territorial"],
        "URGENCIA": fila["urgencia"],
    }

    return max(
        dimensiones,
        key=dimensiones.get,
    )


# =============================================================================
# CONSTRUCCIÓN DE ESCENARIOS
# =============================================================================

def construir_escenarios(
    gdf: gpd.GeoDataFrame,
    labels: np.ndarray,
):

    titulo(
        "8. CONSTRUYENDO ESCENARIOS TERRITORIALES V7"
    )

    gdf = calcular_scores_proyecto(
        gdf
    )

    gdf["cluster_territorial"] = labels

    # -------------------------------------------------------------------------
    # Generar ID de escenario
    # -------------------------------------------------------------------------

    gdf["escenario_id"] = (
        "AMBA-E"
        + gdf["cluster_territorial"]
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )

    registros = []

    for cluster in sorted(
        gdf["cluster_territorial"].unique()
    ):

        grupo = gdf[
            gdf["cluster_territorial"]
            == cluster
        ]

        necesidad = grupo[
            "necesidad_territorial_proyecto"
        ].mean()

        estrategia = grupo[
            "capacidad_estrategica_proyecto"
        ].mean()

        impacto = grupo[
            "impacto_territorial_proyecto"
        ].mean()

        cobertura = grupo[
            "cobertura_territorial_proyecto"
        ].mean()

        deficit = grupo[
            "deficit_atendido_proyecto"
        ].mean()

        demanda = grupo[
            "demanda_cubierta_proyecto"
        ].mean()

        complementariedad = grupo[
            "complementariedad_proyecto"
        ].mean()

        conectividad = grupo[
            "norm_conectividad"
        ].mean()

        intermodalidad = grupo[
            "norm_intermodalidad"
        ].mean()

        integracion = grupo[
            "norm_integracion"
        ].mean()

        centralidad = grupo[
            "norm_centralidad"
        ].mean()

        urgencia = grupo[
            "norm_urgencia"
        ].mean()

        score = (
            PESO_NECESIDAD * necesidad
            + PESO_ESTRATEGICA * estrategia
        )

        prioridad = clasificar_prioridad(
            score
        )

        horizonte = clasificar_horizonte(
            score
        )

        tipo = clasificar_tipo(
            necesidad,
            estrategia,
        )

        temp = pd.Series(
            {
                "demanda_cubierta": demanda,
                "deficit_atendido": deficit,
                "conectividad": conectividad,
                "intermodalidad": intermodalidad,
                "integracion": integracion,
                "centralidad": centralidad,
                "impacto_territorial": impacto,
                "urgencia": urgencia,
            }
        )

        dominante = dimension_dominante(
            temp
        )

        if score >= 70:

            diagnostico = (
                "INTERVENCION_TERRITORIAL_ALTA"
            )

        elif score >= 55:

            diagnostico = (
                "INTERVENCION_TERRITORIAL_MEDIA_ALTA"
            )

        elif score >= 40:

            diagnostico = (
                "INTERVENCION_TERRITORIAL_MEDIA"
            )

        else:

            diagnostico = (
                "INTERVENCION_TERRITORIAL_BAJA"
            )

        registros.append(
            {
                "cluster_territorial": cluster,
                "escenario_id": (
                    f"AMBA-E{cluster:03d}"
                ),
                "cantidad_proyectos": len(
                    grupo
                ),
                "necesidad_territorial": necesidad,
                "capacidad_estrategica": estrategia,
                "impacto_territorial": impacto,
                "cobertura_territorial": cobertura,
                "deficit_atendido": deficit,
                "demanda_cubierta": demanda,
                "complementariedad": complementariedad,
                "conectividad": conectividad,
                "intermodalidad": intermodalidad,
                "integracion": integracion,
                "centralidad": centralidad,
                "urgencia": urgencia,
                "score_escenario": score,
                "prioridad_escenario": prioridad,
                "tipo_escenario": tipo,
                "dimension_dominante": dominante,
                "horizonte_escenario": horizonte,
                "diagnostico_escenario": diagnostico,
            }
        )

    escenarios = pd.DataFrame(
        registros
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    escenarios = escenarios.sort_values(
        [
            "score_escenario",
            "impacto_territorial",
            "cantidad_proyectos",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    escenarios[
        "ranking_escenario"
    ] = np.arange(
        1,
        len(escenarios) + 1,
    )

    # -------------------------------------------------------------------------
    # Objetivos y justificaciones
    # -------------------------------------------------------------------------

    escenarios[
        "objetivo_escenario"
    ] = escenarios.apply(
        lambda r: (
            f"Concentrar {int(r['cantidad_proyectos'])} "
            f"proyectos con prioridad en {r['dimension_dominante'].lower()}, "
            f"atendiendo una necesidad territorial de "
            f"{r['necesidad_territorial']:.1f}/100 "
            f"y una capacidad estratégica de "
            f"{r['capacidad_estrategica']:.1f}/100."
        ),
        axis=1,
    )

    escenarios[
        "justificacion_escenario"
    ] = escenarios.apply(
        lambda r: (
            f"Escenario {r['tipo_escenario'].lower()} "
            f"con impacto territorial {r['impacto_territorial']:.1f}/100, "
            f"cobertura {r['cobertura_territorial']:.1f}/100 "
            f"y complementariedad {r['complementariedad']:.1f}/100."
        ),
        axis=1,
    )

    escenarios[
        "dimensiones_prioritarias"
    ] = escenarios.apply(
        lambda r: ", ".join(
            sorted(
                [
                    (
                        "DEMANDA",
                        r["demanda_cubierta"],
                    ),
                    (
                        "DEFICIT",
                        r["deficit_atendido"],
                    ),
                    (
                        "CONECTIVIDAD",
                        r["conectividad"],
                    ),
                    (
                        "INTERMODALIDAD",
                        r["intermodalidad"],
                    ),
                    (
                        "INTEGRACION",
                        r["integracion"],
                    ),
                    (
                        "CENTRALIDAD",
                        r["centralidad"],
                    ),
                    (
                        "IMPACTO",
                        r["impacto_territorial"],
                    ),
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            and [
                x[0]
                for x in sorted(
                    [
                        (
                            "DEMANDA",
                            r["demanda_cubierta"],
                        ),
                        (
                            "DEFICIT",
                            r["deficit_atendido"],
                        ),
                        (
                            "CONECTIVIDAD",
                            r["conectividad"],
                        ),
                        (
                            "INTERMODALIDAD",
                            r["intermodalidad"],
                        ),
                        (
                            "INTEGRACION",
                            r["integracion"],
                        ),
                        (
                            "CENTRALIDAD",
                            r["centralidad"],
                        ),
                        (
                            "IMPACTO",
                            r["impacto_territorial"],
                        ),
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
            ]
        ),
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Nombre
    # -------------------------------------------------------------------------

    escenarios[
        "escenario_nombre"
    ] = escenarios.apply(
        lambda r: (
            f"Escenario Territorial AMBA "
            f"{int(r['ranking_escenario']):02d}"
        ),
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Mostrar distribución
    # -------------------------------------------------------------------------

    print()

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

    return gdf, escenarios


# =============================================================================
# ASIGNACIÓN
# =============================================================================

def asignar_escenarios(
    gdf: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
):

    titulo(
        "9. ASIGNANDO ESCENARIOS A PROYECTOS"
    )

    columnas = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "dimension_dominante",
        "horizonte_escenario",
    ]

    lookup = escenarios[
        [
            "cluster_territorial"
        ] + columnas
    ].copy()

    gdf = gdf.merge(
        lookup,
        on="cluster_territorial",
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_esc",
        ),
    )

    asignados = int(
        gdf["escenario_id"]
        .notna()
        .sum()
    )

    print(
        f"Proyectos asignados: "
        f"{asignados} / {len(gdf)}"
    )

    if asignados != len(gdf):

        raise RuntimeError(
            "No todos los proyectos fueron asignados."
        )

    return gdf


# =============================================================================
# COHESIÓN
# =============================================================================

def calcular_cohesion(
    X: np.ndarray,
    labels: np.ndarray,
):

    titulo(
        "10. CALCULANDO MÉTRICAS DE COHESIÓN"
    )

    sil_samples = silhouette_samples(
        X,
        labels,
    )

    centros = {}

    for cluster in sorted(
        np.unique(labels)
    ):

        indices = np.where(
            labels == cluster
        )[0]

        centros[cluster] = X[
            indices
        ].mean(axis=0)

    registros = []

    for cluster in sorted(
        np.unique(labels)
    ):

        indices = np.where(
            labels == cluster
        )[0]

        distancias = np.linalg.norm(
            X[indices]
            - centros[cluster],
            axis=1,
        )

        dispersion = (
            np.std(distancias)
        )

        registros.append(
            {
                "cluster_territorial": cluster,
                "cantidad_proyectos": len(
                    indices
                ),
                "silhouette_promedio": float(
                    np.mean(
                        sil_samples[indices]
                    )
                ),
                "silhouette_minima": float(
                    np.min(
                        sil_samples[indices]
                    )
                ),
                "silhouette_maxima": float(
                    np.max(
                        sil_samples[indices]
                    )
                ),
                "distancia_centroide_promedio": float(
                    np.mean(distancias)
                ),
                "distancia_centroide_maxima": float(
                    np.max(distancias)
                ),
                "dispersion_std": float(
                    dispersion
                ),
            }
        )

    cohesion = pd.DataFrame(
        registros
    )

    print(
        cohesion.to_string(
            index=False
        )
    )

    return cohesion


# =============================================================================
# GEOMETRÍAS
# =============================================================================

def construir_geometrias(
    gdf: gpd.GeoDataFrame,
):

    titulo(
        "11. CONSTRUYENDO GEOMETRÍAS DE ESCENARIOS"
    )

    metric = gdf.to_crs(
        CRS_METRICO
    )

    registros = []

    for escenario_id, grupo in metric.groupby(
        "escenario_id"
    ):

        geometries = list(
            grupo.geometry
        )

        try:

            union = unary_union(
                geometries
            )

            if union.is_empty:

                raise ValueError

            geom = union.convex_hull

        except Exception:

            puntos = [
                geom.centroid
                for geom in geometries
            ]

            geom = MultiPoint(
                puntos
            ).convex_hull

        registros.append(
            {
                "escenario_id": escenario_id,
                "geometry": geom,
            }
        )

    escenarios_geom = gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=CRS_METRICO,
    ).to_crs(
        CRS_GEOGRAFICO
    )

    print(
        f"Geometrías construidas: "
        f"{len(escenarios_geom)}"
    )

    return escenarios_geom


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

def validar_final(
    escenarios: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
):

    titulo(
        "12. VALIDACIÓN FINAL V7"
    )

    campos = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "cantidad_proyectos",
        "impacto_territorial",
        "cobertura_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "necesidad_territorial",
        "capacidad_estrategica",
        "complementariedad",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "dimension_dominante",
        "horizonte_escenario",
        "diagnostico_escenario",
        "objetivo_escenario",
        "justificacion_escenario",
        "dimensiones_prioritarias",
    ]

    for campo in campos:

        if campo not in escenarios.columns:

            raise ValueError(
                f"Falta campo obligatorio: {campo}"
            )

        nulos = int(
            escenarios[campo]
            .isna()
            .sum()
        )

        print(
            f"{campo:<40} nulos={nulos}"
        )

        if nulos:

            raise ValueError(
                f"El campo {campo} contiene nulos."
            )

    asignados = int(
        gdf["escenario_id"]
        .notna()
        .sum()
    )

    total = len(gdf)

    print()

    print(
        f"Asignaciones: {asignados}/{total}"
    )

    if asignados != total:

        raise ValueError(
            "No todos los proyectos tienen escenario."
        )

    minimo = int(
        escenarios["cantidad_proyectos"].min()
    )

    maximo = int(
        escenarios["cantidad_proyectos"].max()
    )

    print(
        f"Proyectos por escenario: "
        f"mín={minimo} | máx={maximo}"
    )

    if minimo < MIN_PROYECTOS:

        raise ValueError(
            "Existe un escenario por debajo del mínimo."
        )

    # -------------------------------------------------------------------------
    # Validar que no estén todos los escenarios en 50.
    # -------------------------------------------------------------------------

    for campo in [
        "necesidad_territorial",
        "capacidad_estrategica",
        "impacto_territorial",
        "deficit_atendido",
        "demanda_cubierta",
    ]:

        valores = escenarios[
            campo
        ].round(6).nunique()

        if valores <= 1:

            raise ValueError(
                f"El indicador '{campo}' "
                "no discrimina escenarios."
            )

    print()

    print(
        "VALIDACIÓN FINAL V7: OK"
    )


# =============================================================================
# RANKING
# =============================================================================

def mostrar_ranking(
    escenarios: pd.DataFrame,
):

    titulo(
        "13. RANKING DE ESCENARIOS"
    )

    columnas = [
        "ranking_escenario",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_dominante",
        "horizonte_escenario",
        "score_escenario",
        "prioridad_escenario",
        "necesidad_territorial",
        "capacidad_estrategica",
        "impacto_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
    ]

    print(
        escenarios[
            columnas
        ].round(4).to_string(
            index=False
        )
    )


# =============================================================================
# JSON
# =============================================================================

def construir_json(
    escenarios: pd.DataFrame,
    cohesion: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
):

    titulo(
        "14. CONSTRUYENDO RESUMEN JSON"
    )

    data = {
        "version": VERSION,
        "fecha_proceso": pd.Timestamp.now().isoformat(),
        "configuracion": {
            "k_min": K_MIN,
            "k_max": K_MAX,
            "min_proyectos": MIN_PROYECTOS,
            "peso_espacial": PESO_ESPACIAL,
            "peso_territorial": PESO_TERRITORIAL,
            "peso_necesidad": PESO_NECESIDAD,
            "peso_estrategica": PESO_ESTRATEGICA,
            "random_state": RANDOM_STATE,
        },
        "cantidad_proyectos": int(
            escenarios[
                "cantidad_proyectos"
            ].sum()
        ),
        "cantidad_escenarios": int(
            len(escenarios)
        ),
        "escenarios": (
            escenarios
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "cohesion": (
            cohesion
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "evaluacion_k": (
            evaluacion_k
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    return data


# =============================================================================
# MAPAS
# =============================================================================

def generar_mapa_escenarios(
    gdf: gpd.GeoDataFrame,
    nombre: str,
    columna: str,
    titulo_mapa: str,
    cmap: str = "tab10",
):

    fig, ax = plt.subplots(
        figsize=(13, 10)
    )

    gdf.plot(
        ax=ax,
        column=columna,
        cmap=cmap,
        legend=True,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_title(
        titulo_mapa,
        fontsize=16,
        fontweight="bold",
    )

    ax.set_axis_off()

    plt.tight_layout()

    ruta = OUTPUT_DIR / nombre

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Mapa: {ruta}"
    )


def generar_mapas(
    proyectos: gpd.GeoDataFrame,
    escenarios_geom: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
):

    titulo(
        "16. GENERANDO MAPAS"
    )

    generar_mapa_escenarios(
        proyectos,
        "01_mapa_escenarios_territoriales.png",
        "escenario_id",
        "Escenarios Territoriales AMBA - V7",
    )

    generar_mapa_escenarios(
        proyectos,
        "02_mapa_prioridad_escenarios.png",
        "prioridad_escenario",
        "Prioridad de Escenarios Territoriales",
        "RdYlGn",
    )

    generar_mapa_escenarios(
        proyectos,
        "03_mapa_cobertura_metropolitana.png",
        "escenario_id",
        "Cobertura Territorial de Escenarios",
    )

    generar_mapa_escenarios(
        proyectos,
        "04_mapa_impacto_territorial.png",
        "escenario_id",
        "Impacto Territorial por Escenario",
    )

    generar_mapa_escenarios(
        proyectos,
        "05_mapa_proyectos_por_escenario.png",
        "escenario_id",
        "Proyectos por Escenario",
    )


# =============================================================================
# GRÁFICOS
# =============================================================================

def generar_graficos(
    escenarios: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
):

    titulo(
        "17. GENERANDO GRÁFICOS"
    )

    # -------------------------------------------------------------------------
    # 06 - Necesidad vs capacidad
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.scatter(
        escenarios[
            "necesidad_territorial"
        ],
        escenarios[
            "capacidad_estrategica"
        ],
        s=120,
    )

    for _, row in escenarios.iterrows():

        ax.annotate(
            row["escenario_id"],
            (
                row["necesidad_territorial"],
                row["capacidad_estrategica"],
            ),
        )

    ax.set_xlabel(
        "Necesidad territorial"
    )

    ax.set_ylabel(
        "Capacidad estratégica"
    )

    ax.set_title(
        "Necesidad Territorial vs Capacidad Estratégica"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    ruta = (
        OUTPUT_DIR
        / "06_necesidad_vs_capacidad_estrategica.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {ruta}"
    )

    # -------------------------------------------------------------------------
    # 07 - Demanda vs déficit
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
        s=120,
    )

    for _, row in escenarios.iterrows():

        ax.annotate(
            row["escenario_id"],
            (
                row["demanda_cubierta"],
                row["deficit_atendido"],
            ),
        )

    ax.set_xlabel(
        "Demanda cubierta"
    )

    ax.set_ylabel(
        "Déficit atendido"
    )

    ax.set_title(
        "Demanda vs Déficit Atendido"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    ruta = (
        OUTPUT_DIR
        / "07_demanda_vs_deficit_atendido.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {ruta}"
    )

    # -------------------------------------------------------------------------
    # 08 - Prioridad
    # -------------------------------------------------------------------------

    prioridad = (
        escenarios[
            "prioridad_escenario"
        ]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    prioridad.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Escenarios por Prioridad"
    )

    ax.set_xlabel(
        "Prioridad"
    )

    ax.set_ylabel(
        "Cantidad de escenarios"
    )

    plt.tight_layout()

    ruta = (
        OUTPUT_DIR
        / "08_escenarios_por_prioridad.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {ruta}"
    )

    # -------------------------------------------------------------------------
    # 09 - Horizonte
    # -------------------------------------------------------------------------

    horizonte = (
        escenarios[
            "horizonte_escenario"
        ]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    horizonte.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Escenarios por Horizonte"
    )

    ax.set_xlabel(
        "Horizonte"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    plt.tight_layout()

    ruta = (
        OUTPUT_DIR
        / "09_escenarios_por_horizonte.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {ruta}"
    )

    # -------------------------------------------------------------------------
    # 10 - Distribución de score
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.hist(
        escenarios[
            "score_escenario"
        ],
        bins=10,
    )

    ax.set_title(
        "Distribución del Score de Escenarios"
    )

    ax.set_xlabel(
        "Score"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    plt.tight_layout()

    ruta = (
        OUTPUT_DIR
        / "10_distribucion_score_escenarios.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {ruta}"
    )

    # -------------------------------------------------------------------------
    # 11 - Silhouette
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.plot(
        evaluacion_k["k"],
        evaluacion_k["silhouette"],
        marker="o",
    )

    ax.set_title(
        "Evaluación Silhouette por K"
    )

    ax.set_xlabel(
        "K"
    )

    ax.set_ylabel(
        "Silhouette"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    ruta = (
        OUTPUT_DIR
        / "11_evaluacion_silhouette_k.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {ruta}"
    )

    # -------------------------------------------------------------------------
    # 12 - Davies Bouldin
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.plot(
        evaluacion_k["k"],
        evaluacion_k["davies_bouldin"],
        marker="o",
    )

    ax.set_title(
        "Evaluación Davies-Bouldin por K"
    )

    ax.set_xlabel(
        "K"
    )

    ax.set_ylabel(
        "Davies-Bouldin"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    ruta = (
        OUTPUT_DIR
        / "12_evaluacion_davies_bouldin_k.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {ruta}"
    )


# =============================================================================
# GUARDADO
# =============================================================================

def guardar_resultados(
    escenarios: pd.DataFrame,
    proyectos: gpd.GeoDataFrame,
    escenarios_geom: gpd.GeoDataFrame,
    cohesion: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
    resumen_json: dict,
):

    titulo(
        "15. GUARDANDO ARCHIVOS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Escenarios
    # -------------------------------------------------------------------------

    escenarios_salida = escenarios.copy()

    escenarios_salida.to_parquet(
        OUTPUT_DIR
        / "escenarios_territoriales_amba.parquet",
        index=False,
    )

    escenarios_salida.to_csv(
        OUTPUT_DIR
        / "escenarios_territoriales_amba.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Parquet escenarios:\n"
        f"{OUTPUT_DIR / 'escenarios_territoriales_amba.parquet'}"
    )

    print(
        f"CSV escenarios:\n"
        f"{OUTPUT_DIR / 'escenarios_territoriales_amba.csv'}"
    )

    # -------------------------------------------------------------------------
    # GeoPackage
    # -------------------------------------------------------------------------

    gpkg = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.gpkg"
    )

    if gpkg.exists():

        gpkg.unlink()

    escenarios_geom.to_file(
        gpkg,
        layer="escenarios",
        driver="GPKG",
    )

    print(
        f"GeoPackage:\n{gpkg}"
    )

    # -------------------------------------------------------------------------
    # Proyectos
    # -------------------------------------------------------------------------

    proyectos_salida = proyectos.copy()

    # Eliminar columnas internas de modelado.
    columnas_internas = [
        c
        for c in proyectos_salida.columns
        if c.startswith("_v7_")
        or c.startswith("norm_")
    ]

    columnas_internas += [
        "x_m",
        "y_m",
    ]

    columnas_internas = [
        c
        for c in columnas_internas
        if c in proyectos_salida.columns
    ]

    proyectos_salida = proyectos_salida.drop(
        columns=columnas_internas
    )

    proyectos_salida.to_parquet(
        OUTPUT_DIR
        / "proyectos_escenarios_territoriales_amba.parquet",
        index=False,
    )

    proyectos_salida.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        OUTPUT_DIR
        / "proyectos_escenarios_territoriales_amba.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Parquet proyectos:\n"
        f"{OUTPUT_DIR / 'proyectos_escenarios_territoriales_amba.parquet'}"
    )

    print(
        f"CSV proyectos:\n"
        f"{OUTPUT_DIR / 'proyectos_escenarios_territoriales_amba.csv'}"
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba_resumen.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            resumen_json,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(
        f"JSON:\n{json_path}"
    )

    # -------------------------------------------------------------------------
    # Evaluación K
    # -------------------------------------------------------------------------

    evaluacion_k.to_csv(
        OUTPUT_DIR
        / "evaluacion_numero_escenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Evaluación K:\n"
        f"{OUTPUT_DIR / 'evaluacion_numero_escenarios.csv'}"
    )

    # -------------------------------------------------------------------------
    # Diagnóstico
    # -------------------------------------------------------------------------

    diagnostico = escenarios[
        [
            "ranking_escenario",
            "escenario_id",
            "cantidad_proyectos",
            "score_escenario",
            "prioridad_escenario",
            "tipo_escenario",
            "dimension_dominante",
            "horizonte_escenario",
            "diagnostico_escenario",
            "necesidad_territorial",
            "capacidad_estrategica",
            "impacto_territorial",
            "cobertura_territorial",
            "deficit_atendido",
            "demanda_cubierta",
            "complementariedad",
        ]
    ].copy()

    diagnostico.to_csv(
        OUTPUT_DIR
        / "diagnostico_escenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Diagnóstico:\n"
        f"{OUTPUT_DIR / 'diagnostico_escenarios.csv'}"
    )

    # -------------------------------------------------------------------------
    # Cohesión
    # -------------------------------------------------------------------------

    cohesion.to_csv(
        OUTPUT_DIR
        / "metricas_cohesion_escenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Cohesión:\n"
        f"{OUTPUT_DIR / 'metricas_cohesion_escenarios.csv'}"
    )


# =============================================================================
# EVALUACIÓN K
# =============================================================================

def preparar_evaluacion_k(
    resultados_k: list[dict],
):

    registros = []

    for r in resultados_k:

        registros.append(
            {
                "k": r["k"],
                "silhouette": r["silhouette"],
                "davies_bouldin": r["davies_bouldin"],
                "min_proyectos": r["min_proyectos"],
                "max_proyectos": r["max_proyectos"],
                "equilibrio": r["equilibrio"],
                "score": r["score"],
            }
        )

    return pd.DataFrame(
        registros
    )


# =============================================================================
# RESUMEN FINAL
# =============================================================================

def resumen_final(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
):

    titulo(
        f"27 - PROCESO FINALIZADO CORRECTAMENTE - {VERSION}"
    )

    print(
        f"Proyectos analizados     : {len(proyectos)}"
    )

    print(
        f"Escenarios territoriales : {len(escenarios)}"
    )

    print()

    print(
        "DISTRIBUCIÓN DE PROYECTOS"
    )

    for _, row in escenarios.iterrows():

        print(
            f"  {row['escenario_id']}: "
            f"{int(row['cantidad_proyectos']):3d} proyectos | "
            f"{row['tipo_escenario']:<28} | "
            f"dominante={row['dimension_dominante']:<18} | "
            f"necesidad={row['necesidad_territorial']:6.2f} | "
            f"estratégico={row['capacidad_estrategica']:6.2f} | "
            f"score={row['score_escenario']:6.2f}"
        )

    print()

    print(
        "PRIORIDADES:"
    )

    print(
        escenarios[
            "prioridad_escenario"
        ].value_counts().to_string()
    )

    print()

    print(
        "HORIZONTES:"
    )

    print(
        escenarios[
            "horizonte_escenario"
        ].value_counts().to_string()
    )

    print()

    print(
        "TIPOS DE ESCENARIO:"
    )

    print(
        escenarios[
            "tipo_escenario"
        ].value_counts().to_string()
    )

    print()

    print(
        "DIMENSIONES DOMINANTES:"
    )

    print(
        escenarios[
            "dimension_dominante"
        ].value_counts().to_string()
    )

    print()

    print(
        "DIAGNÓSTICOS:"
    )

    print(
        escenarios[
            "diagnostico_escenario"
        ].value_counts().to_string()
    )

    print()

    print(
        "SIGUIENTE ETAPA"
    )

    print(
        "Evaluar los escenarios mediante simulación "
        "de impactos, cobertura, demanda, déficit, "
        "conectividad e interacción territorial para "
        "seleccionar escenarios estratégicos."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    titulo(
        f"27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - {VERSION}"
    )

    print(
        f"Proyecto    : {PROJECT_DIR}"
    )

    print(
        f"Entrada     : {INPUT_FILE}"
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
        f"  Versión              : {VERSION}"
    )

    print(
        f"  K candidatos         : {K_MIN} - {K_MAX}"
    )

    print(
        f"  Mínimo proyectos     : {MIN_PROYECTOS}"
    )

    print(
        "  Método               : "
        "KMeans espacial + territorial"
    )

    print(
        f"  Random state         : {RANDOM_STATE}"
    )

    print()

    print(
        "PESOS DEL CLUSTERING"
    )

    print(
        f"  Componente espacial   : {PESO_ESPACIAL:.0%}"
    )

    print(
        f"  Componente territorial: {PESO_TERRITORIAL:.0%}"
    )

    print()

    print(
        "MACRO-DIMENSIONES"
    )

    print(
        f"  Necesidad territorial : {PESO_NECESIDAD:.0%}"
    )

    print(
        f"  Capacidad estratégica  : {PESO_ESTRATEGICA:.0%}"
    )

    # -------------------------------------------------------------------------
    # 1
    # -------------------------------------------------------------------------

    gdf = cargar_cartera()

    # -------------------------------------------------------------------------
    # 2
    # -------------------------------------------------------------------------

    gdf = validar_entrada(
        gdf
    )

    # -------------------------------------------------------------------------
    # 3-4
    # -------------------------------------------------------------------------

    gdf, columnas_indicadores = preparar_indicadores(
        gdf
    )

    gdf = preparar_espacio(
        gdf
    )

    # -------------------------------------------------------------------------
    # 5
    # -------------------------------------------------------------------------

    X, indicadores = construir_matriz(
        gdf
    )

    # -------------------------------------------------------------------------
    # 6
    # -------------------------------------------------------------------------

    mejor, resultados_k = seleccionar_k(
        X
    )

    # -------------------------------------------------------------------------
    # Reparar clusters
    # -------------------------------------------------------------------------

    labels = reparar_clusters(
        X,
        mejor["labels"],
        MIN_PROYECTOS,
    )

    # -------------------------------------------------------------------------
    # Cohesión
    # -------------------------------------------------------------------------

    cohesion = calcular_cohesion(
        X,
        labels,
    )

    # -------------------------------------------------------------------------
    # Escenarios
    # -------------------------------------------------------------------------

    gdf, escenarios = construir_escenarios(
        gdf,
        labels,
    )

    # -------------------------------------------------------------------------
    # Asignaciones
    # -------------------------------------------------------------------------

    gdf = asignar_escenarios(
        gdf,
        escenarios,
    )

    # -------------------------------------------------------------------------
    # Geometrías
    # -------------------------------------------------------------------------

    escenarios_geom = construir_geometrias(
        gdf
    )

    # -------------------------------------------------------------------------
    # Validación
    # -------------------------------------------------------------------------

    validar_final(
        escenarios,
        gdf,
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    mostrar_ranking(
        escenarios
    )

    # -------------------------------------------------------------------------
    # Evaluación K
    # -------------------------------------------------------------------------

    evaluacion_k = preparar_evaluacion_k(
        resultados_k
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    resumen_json = construir_json(
        escenarios,
        cohesion,
        evaluacion_k,
    )

    # -------------------------------------------------------------------------
    # Guardado
    # -------------------------------------------------------------------------

    guardar_resultados(
        escenarios,
        gdf,
        escenarios_geom,
        cohesion,
        evaluacion_k,
        resumen_json,
    )

    # -------------------------------------------------------------------------
    # Mapas
    # -------------------------------------------------------------------------

    generar_mapas(
        gdf,
        escenarios_geom,
        escenarios,
    )

    # -------------------------------------------------------------------------
    # Gráficos
    # -------------------------------------------------------------------------

    generar_graficos(
        escenarios,
        evaluacion_k,
    )

    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------

    resumen_final(
        gdf,
        escenarios,
    )


if __name__ == "__main__":
    main()