# -*- coding: utf-8 -*-

"""
========================================================================================
28 - EVALUACIÓN DE ESCENARIOS TERRITORIALES AMBA
========================================================================================

Proceso 28 del pipeline de análisis de movilidad AMBA.

OBJETIVO
--------
Evaluar los escenarios territoriales construidos por el Proceso 27.

El proceso realiza:

1. Carga de escenarios.
2. Validación estructural.
3. Validación de cobertura.
4. Evaluación del tamaño de escenarios.
5. Evaluación de cohesión territorial.
6. Evaluación de concentración de indicadores.
7. Evaluación de balance entre escenarios.
8. Evaluación multicriterio.
9. Clasificación de escenarios.
10. Generación de recomendaciones.
11. Exportación de resultados.

IMPORTANTE
----------
Este proceso NO modifica los resultados del Proceso 27.

No genera valores artificiales.

No depende de archivos externos adicionales.

Compatible con ejecución desde:

    python src/28_evaluar_escenarios_territoriales_amba.py
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


# ======================================================================================
# CONFIGURACIÓN
# ======================================================================================

VERSION = "V1.0"

PROCESO = 28

NOMBRE = "EVALUACIÓN DE ESCENARIOS TERRITORIALES AMBA"

K_MIN = 6
K_MAX = 12

MIN_PROYECTOS_ESCENARIO = 8

# Pesos de evaluación
PESO_COBERTURA = 0.20
PESO_TAMANO = 0.15
PESO_COHESION = 0.25
PESO_INDICADORES = 0.20
PESO_BALANCE = 0.20


# ======================================================================================
# RUTAS
# ======================================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_PARQUET = (
    INPUT_DIR
    / "escenarios_territoriales_amba.parquet"
)

OUTPUT_DIR = INPUT_DIR

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "evaluacion_escenarios_territoriales_amba.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "evaluacion_escenarios_territoriales_amba.csv"
)

OUTPUT_ESCENARIOS_CSV = (
    OUTPUT_DIR
    / "evaluacion_detalle_escenarios_territoriales_amba.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "evaluacion_escenarios_territoriales_amba.json"
)

OUTPUT_RECOMENDACIONES = (
    OUTPUT_DIR
    / "recomendaciones_escenarios_territoriales_amba.csv"
)


# ======================================================================================
# UTILIDADES
# ======================================================================================

def encabezado(numero: str, titulo: str) -> None:
    print()
    print("=" * 88)
    print(f"{numero}. {titulo}")
    print("=" * 88)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", ".")

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


def normalizar_serie(serie: pd.Series) -> pd.Series:
    """
    Normalización min-max robusta.

    Si todos los valores son iguales devuelve 1.
    """

    s = pd.to_numeric(serie, errors="coerce")

    minimo = s.min()
    maximo = s.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(
            np.zeros(len(s)),
            index=s.index,
            dtype=float,
        )

    if maximo == minimo:
        return pd.Series(
            np.ones(len(s)),
            index=s.index,
            dtype=float,
        )

    return (s - minimo) / (maximo - minimo)


def porcentaje(valor: float) -> float:
    return round(float(valor) * 100.0, 2)


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: list[str],
) -> str | None:

    columnas = {str(c).lower(): c for c in df.columns}

    for candidato in candidatos:

        if candidato.lower() in columnas:
            return columnas[candidato.lower()]

    return None


def encontrar_columnas_parciales(
    df: pd.DataFrame,
    patrones: list[str],
) -> list[str]:

    resultado = []

    for columna in df.columns:

        nombre = str(columna).lower()

        for patron in patrones:

            if patron.lower() in nombre:

                resultado.append(columna)
                break

    return resultado


def convertir_numericas(
    df: pd.DataFrame,
    columnas: list[str],
) -> pd.DataFrame:

    for columna in columnas:

        if columna in df.columns:

            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

    return df


# ======================================================================================
# CARGA
# ======================================================================================

def cargar_escenarios() -> gpd.GeoDataFrame:

    encabezado(
        "1",
        "CARGANDO ESCENARIOS DEL PROCESO 27",
    )

    print(f"Entrada:")
    print(INPUT_PARQUET)

    if not INPUT_PARQUET.exists():

        raise FileNotFoundError(
            f"No existe la entrada del Proceso 27:\n"
            f"{INPUT_PARQUET}"
        )

    gdf = gpd.read_parquet(INPUT_PARQUET)

    print(f"Registros : {len(gdf):,}")
    print(f"Columnas  : {len(gdf.columns):,}")

    if hasattr(gdf, "crs"):
        print(f"CRS       : {gdf.crs}")

    return gdf


# ======================================================================================
# VALIDACIÓN
# ======================================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> dict[str, Any]:

    encabezado(
        "2",
        "VALIDANDO ESTRUCTURA DE ENTRADA",
    )

    resultados: dict[str, Any] = {}

    resultados["registros"] = len(gdf)

    if "geometry" in gdf.columns:

        nulas = int(gdf.geometry.isna().sum())
        vacias = int(gdf.geometry.is_empty.sum())

        try:
            invalidas = int(
                (~gdf.geometry.is_valid).sum()
            )
        except Exception:
            invalidas = 0

    else:

        nulas = 0
        vacias = 0
        invalidas = 0

    resultados["geometrias_nulas"] = nulas
    resultados["geometrias_vacias"] = vacias
    resultados["geometrias_invalidas"] = invalidas

    escenario_col = encontrar_columna(
        gdf,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    resultados["columna_escenario"] = escenario_col

    if escenario_col is None:

        raise ValueError(
            "No se encontró la columna de identificación "
            "del escenario."
        )

    cantidad_escenarios = int(
        gdf[escenario_col]
        .dropna()
        .nunique()
    )

    resultados["cantidad_escenarios"] = cantidad_escenarios

    duplicados = int(
        gdf[escenario_col]
        .duplicated()
        .sum()
    )

    resultados["identificadores_duplicados"] = duplicados

    print(
        f"Geometrías nulas      : {nulas}"
    )

    print(
        f"Geometrías vacías     : {vacias}"
    )

    print(
        f"Geometrías inválidas  : {invalidas}"
    )

    print(
        f"Escenarios detectados : {cantidad_escenarios}"
    )

    print(
        f"Duplicados escenario  : {duplicados}"
    )

    if nulas > 0:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if vacias > 0:
        raise ValueError(
            "Existen geometrías vacías."
        )

    if invalidas > 0:
        raise ValueError(
            "Existen geometrías inválidas."
        )

    if cantidad_escenarios < 1:

        raise ValueError(
            "No se encontraron escenarios."
        )

    print(
        "Validación estructural: OK"
    )

    return resultados


# ======================================================================================
# RESOLUCIÓN DE COLUMNAS
# ======================================================================================

def resolver_columnas(
    gdf: gpd.GeoDataFrame,
) -> dict[str, str | None]:

    encabezado(
        "3",
        "RESOLVIENDO VARIABLES DE EVALUACIÓN",
    )

    candidatos = {

        "escenario":
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],

        "score_escenario":
            [
                "score_escenario",
                "score",
            ],

        "tipo":
            [
                "tipo_escenario",
                "tipo",
            ],

        "dimension":
            [
                "dimension_dominante",
                "dimension",
            ],

        "prioridad":
            [
                "prioridad_escenario",
                "prioridad",
            ],

        "proyecto":
            [
                "proyecto_id",
                "id_proyecto",
                "codigo_proyecto",
                "id",
            ],

        "silhouette":
            [
                "silhouette",
                "silhouette_score",
            ],

        "distancia_centroide":
            [
                "distancia_centroide",
            ],

    }

    columnas = {}

    for clave, nombres in candidatos.items():

        columna = encontrar_columna(
            gdf,
            nombres,
        )

        columnas[clave] = columna

        if columna:

            print(
                f"  {clave:<24} OK -> {columna}"
            )

        else:

            print(
                f"  {clave:<24} -- no disponible"
            )

    return columnas


# ======================================================================================
# COBERTURA
# ======================================================================================

def evaluar_cobertura(
    gdf: pd.DataFrame,
    columnas: dict[str, str | None],
) -> tuple[float, dict[str, Any]]:

    encabezado(
        "4",
        "EVALUANDO COBERTURA DE PROYECTOS",
    )

    escenario_col = columnas["escenario"]

    if escenario_col is None:

        return 0.0, {
            "estado": "ERROR",
        }

    total = len(gdf)

    asignados = int(
        gdf[escenario_col]
        .notna()
        .sum()
    )

    sin_asignar = total - asignados

    cobertura = (
        asignados / total
        if total > 0
        else 0
    )

    escenarios = (
        gdf.loc[
            gdf[escenario_col].notna(),
            escenario_col,
        ]
        .nunique()
    )

    print(
        f"Proyectos totales       : {total:,}"
    )

    print(
        f"Proyectos asignados     : {asignados:,}"
    )

    print(
        f"Proyectos sin escenario : {sin_asignar:,}"
    )

    print(
        f"Cobertura               : {porcentaje(cobertura)}%"
    )

    print(
        f"Escenarios utilizados   : {escenarios}"
    )

    return cobertura, {

        "total_proyectos": total,

        "proyectos_asignados":
            asignados,

        "proyectos_sin_asignar":
            sin_asignar,

        "cobertura":
            cobertura,

        "escenarios_utilizados":
            int(escenarios),
    }


# ======================================================================================
# TAMAÑO
# ======================================================================================

def evaluar_tamano(
    gdf: pd.DataFrame,
    columnas: dict[str, str | None],
) -> tuple[float, pd.DataFrame]:

    encabezado(
        "5",
        "EVALUANDO TAMAÑO DE LOS ESCENARIOS",
    )

    escenario_col = columnas["escenario"]

    conteo = (
        gdf.groupby(
            escenario_col,
            dropna=False,
        )
        .size()
        .reset_index(
            name="cantidad_proyectos"
        )
    )

    conteo["cumple_minimo"] = (
        conteo["cantidad_proyectos"]
        >= MIN_PROYECTOS_ESCENARIO
    )

    conteo["score_tamano"] = np.where(
        conteo["cumple_minimo"],
        1.0,
        conteo["cantidad_proyectos"]
        / MIN_PROYECTOS_ESCENARIO,
    )

    minimo = float(
        conteo["cantidad_proyectos"].min()
    )

    maximo = float(
        conteo["cantidad_proyectos"].max()
    )

    promedio = float(
        conteo["cantidad_proyectos"].mean()
    )

    cantidad_validos = int(
        conteo["cumple_minimo"].sum()
    )

    total = len(conteo)

    proporcion_validos = (
        cantidad_validos / total
        if total > 0
        else 0
    )

    # Penalización moderada por desbalance.
    cv = (
        float(
            conteo["cantidad_proyectos"].std()
            /
            promedio
        )
        if promedio > 0 and total > 1
        else 0
    )

    balance = max(
        0.0,
        1.0 - min(cv, 1.0),
    )

    score = (
        0.70 * proporcion_validos
        +
        0.30 * balance
    )

    print(
        f"Escenarios              : {total}"
    )

    print(
        f"Mínimo proyectos       : {minimo:.0f}"
    )

    print(
        f"Máximo proyectos       : {maximo:.0f}"
    )

    print(
        f"Promedio proyectos     : {promedio:.2f}"
    )

    print(
        f"Cumplen mínimo         : "
        f"{cantidad_validos}/{total}"
    )

    print(
        f"Score tamaño            : "
        f"{score:.4f}"
    )

    return score, conteo


# ======================================================================================
# COHESIÓN
# ======================================================================================

def evaluar_cohesion(
    gdf: gpd.GeoDataFrame,
    columnas: dict[str, str | None],
) -> tuple[float, pd.DataFrame]:

    encabezado(
        "6",
        "EVALUANDO COHESIÓN TERRITORIAL",
    )

    escenario_col = columnas["escenario"]

    # ------------------------------------------------------------------
    # Caso 1: ya existe silhouette
    # ------------------------------------------------------------------

    silhouette_col = columnas["silhouette"]

    if silhouette_col is not None:

        valores = pd.to_numeric(
            gdf[silhouette_col],
            errors="coerce",
        )

        resumen = (
            gdf.assign(
                _silhouette=valores
            )
            .groupby(
                escenario_col
            )["_silhouette"]
            .agg(
                [
                    "count",
                    "mean",
                    "min",
                    "max",
                ]
            )
            .reset_index()
        )

        resumen = resumen.rename(
            columns={
                "count": "cantidad",
                "mean": "silhouette_promedio",
                "min": "silhouette_minima",
                "max": "silhouette_maxima",
            }
        )

        media = safe_float(
            resumen["silhouette_promedio"].mean()
        )

        # Silhouette teórico [-1, 1].
        # Transformación a [0, 1].
        score = max(
            0.0,
            min(
                1.0,
                (media + 1.0) / 2.0,
            ),
        )

        print(
            f"Silhouette promedio   : {media:.4f}"
        )

        print(
            f"Score cohesión        : {score:.4f}"
        )

        return score, resumen

    # ------------------------------------------------------------------
    # Caso 2: calcular cohesión geométrica
    # ------------------------------------------------------------------

    if "geometry" not in gdf.columns:

        print(
            "No existe geometría."
        )

        return 0.0, pd.DataFrame()

    centroides = gdf.geometry.centroid

    datos = gdf.copy()

    datos["_cx"] = centroides.x
    datos["_cy"] = centroides.y

    registros = []

    for escenario, grupo in datos.groupby(
        escenario_col,
        dropna=False,
    ):

        if len(grupo) <= 1:

            distancia_media = 0.0
            distancia_maxima = 0.0

        else:

            cx = grupo["_cx"].mean()
            cy = grupo["_cy"].mean()

            distancias = np.sqrt(
                (
                    grupo["_cx"] - cx
                ) ** 2
                +
                (
                    grupo["_cy"] - cy
                ) ** 2
            )

            distancia_media = float(
                distancias.mean()
            )

            distancia_maxima = float(
                distancias.max()
            )

        registros.append(
            {
                escenario_col: escenario,
                "cantidad_proyectos": len(grupo),
                "distancia_centroide_promedio":
                    distancia_media,
                "distancia_centroide_maxima":
                    distancia_maxima,
            }
        )

    resumen = pd.DataFrame(
        registros
    )

    if resumen.empty:

        return 0.0, resumen

    media = float(
        resumen[
            "distancia_centroide_promedio"
        ].mean()
    )

    maximo = float(
        resumen[
            "distancia_centroide_promedio"
        ].max()
    )

    if maximo <= 0:

        score = 1.0

    else:

        # Menor distancia = mayor cohesión.
        score = max(
            0.0,
            min(
                1.0,
                1.0 - media / maximo,
            ),
        )

    print(
        f"Distancia media        : {media:.6f}"
    )

    print(
        f"Score cohesión         : {score:.4f}"
    )

    return score, resumen


# ======================================================================================
# INDICADORES
# ======================================================================================

def evaluar_indicadores(
    gdf: pd.DataFrame,
    columnas: dict[str, str | None],
) -> tuple[float, pd.DataFrame]:

    encabezado(
        "7",
        "EVALUANDO CONCENTRACIÓN DE INDICADORES",
    )

    escenario_col = columnas["escenario"]

    patrones = [
        "demanda",
        "deficit",
        "conectividad",
        "intermodalidad",
        "integracion",
        "centralidad",
        "impacto",
        "urgencia",
        "score_cartera",
        "prioridad_territorial",
    ]

    columnas_indicadores = (
        encontrar_columnas_parciales(
            gdf,
            patrones,
        )
    )

    # Eliminar columnas no numéricas.
    columnas_validas = []

    for columna in columnas_indicadores:

        serie = pd.to_numeric(
            gdf[columna],
            errors="coerce",
        )

        if serie.notna().sum() > 0:

            columnas_validas.append(
                columna
            )

    if not columnas_validas:

        print(
            "No se encontraron indicadores numéricos."
        )

        return 0.0, pd.DataFrame()

    resumen = (
        gdf.groupby(
            escenario_col
        )[columnas_validas]
        .mean()
        .reset_index()
    )

    # Convertimos cada indicador a rango 0-1.
    scores = []

    for columna in columnas_validas:

        serie = pd.to_numeric(
            resumen[columna],
            errors="coerce",
        )

        normalizada = normalizar_serie(
            serie
        )

        scores.append(
            normalizada
        )

    matriz = pd.concat(
        scores,
        axis=1,
    )

    matriz.columns = columnas_validas

    resumen["score_indicadores"] = (
        matriz.mean(axis=1)
    )

    score = float(
        resumen[
            "score_indicadores"
        ].mean()
    )

    print(
        f"Indicadores detectados : "
        f"{len(columnas_validas)}"
    )

    for columna in columnas_validas:

        print(
            f"  - {columna}"
        )

    print(
        f"Score indicadores      : "
        f"{score:.4f}"
    )

    return score, resumen


# ======================================================================================
# BALANCE
# ======================================================================================

def evaluar_balance(
    gdf: pd.DataFrame,
    columnas: dict[str, str | None],
) -> tuple[float, pd.DataFrame]:

    encabezado(
        "8",
        "EVALUANDO BALANCE ENTRE ESCENARIOS",
    )

    escenario_col = columnas["escenario"]

    conteo = (
        gdf.groupby(
            escenario_col
        )
        .size()
        .reset_index(
            name="cantidad_proyectos"
        )
    )

    if conteo.empty:

        return 0.0, conteo

    media = float(
        conteo["cantidad_proyectos"].mean()
    )

    desviacion = float(
        conteo["cantidad_proyectos"].std()
    ) if len(conteo) > 1 else 0.0

    cv = (
        desviacion / media
        if media > 0
        else 0.0
    )

    score = max(
        0.0,
        min(
            1.0,
            1.0 - cv,
        ),
    )

    conteo["participacion"] = (
        conteo["cantidad_proyectos"]
        / conteo["cantidad_proyectos"].sum()
    )

    print(
        f"Promedio                : {media:.2f}"
    )

    print(
        f"Desvío estándar         : {desviacion:.2f}"
    )

    print(
        f"Coeficiente variación   : {cv:.4f}"
    )

    print(
        f"Score balance           : {score:.4f}"
    )

    return score, conteo


# ======================================================================================
# CLASIFICACIÓN
# ======================================================================================

def clasificar_score(
    score: float,
) -> str:

    if score >= 0.80:
        return "EXCELENTE"

    if score >= 0.65:
        return "BUENO"

    if score >= 0.50:
        return "ACEPTABLE"

    if score >= 0.35:
        return "DEBIL"

    return "CRITICO"


def clasificar_escenario(
    score: float,
) -> str:

    if score >= 0.80:
        return "ESCENARIO_PRIORITARIO"

    if score >= 0.65:
        return "ESCENARIO_FUERTE"

    if score >= 0.50:
        return "ESCENARIO_ADECUADO"

    if score >= 0.35:
        return "ESCENARIO_A_REVISAR"

    return "ESCENARIO_CRITICO"


# ======================================================================================
# EVALUACIÓN INTEGRADA
# ======================================================================================

def construir_evaluacion_integrada(
    gdf: gpd.GeoDataFrame,
    columnas: dict[str, str | None],
    detalle_tamano: pd.DataFrame,
    detalle_cohesion: pd.DataFrame,
    detalle_indicadores: pd.DataFrame,
    detalle_balance: pd.DataFrame,
) -> pd.DataFrame:

    encabezado(
        "9",
        "CONSTRUYENDO EVALUACIÓN INTEGRADA",
    )

    escenario_col = columnas["escenario"]

    escenarios = sorted(
        gdf[
            escenario_col
        ]
        .dropna()
        .unique()
        .tolist()
    )

    resultado = pd.DataFrame(
        {
            escenario_col: escenarios
        }
    )

    # ------------------------------------------------------------------
    # Cantidad
    # ------------------------------------------------------------------

    if not detalle_tamano.empty:

        resultado = resultado.merge(
            detalle_tamano[
                [
                    escenario_col,
                    "cantidad_proyectos",
                    "score_tamano",
                ]
            ],
            on=escenario_col,
            how="left",
        )

    else:

        resultado["cantidad_proyectos"] = (
            resultado[escenario_col]
            .map(
                gdf.groupby(
                    escenario_col
                ).size()
            )
        )

        resultado["score_tamano"] = 0.0

    # ------------------------------------------------------------------
    # Cohesión
    # ------------------------------------------------------------------

    if not detalle_cohesion.empty:

        columnas_cohesion = [
            c
            for c in [
                escenario_col,
                "silhouette_promedio",
                "silhouette_minima",
                "silhouette_maxima",
                "distancia_centroide_promedio",
                "distancia_centroide_maxima",
            ]
            if c in detalle_cohesion.columns
        ]

        resultado = resultado.merge(
            detalle_cohesion[columnas_cohesion],
            on=escenario_col,
            how="left",
        )

        if "silhouette_promedio" in resultado:

            resultado["score_cohesion"] = (
                (
                    pd.to_numeric(
                        resultado[
                            "silhouette_promedio"
                        ],
                        errors="coerce",
                    )
                    + 1
                )
                / 2
            ).clip(
                lower=0,
                upper=1,
            )

        else:

            resultado["score_cohesion"] = 0.0

    else:

        resultado["score_cohesion"] = 0.0

    # ------------------------------------------------------------------
    # Indicadores
    # ------------------------------------------------------------------

    if not detalle_indicadores.empty:

        resultado = resultado.merge(
            detalle_indicadores[
                [
                    escenario_col,
                    "score_indicadores",
                ]
            ],
            on=escenario_col,
            how="left",
        )

    else:

        resultado["score_indicadores"] = 0.0

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    if not detalle_balance.empty:

        resultado = resultado.merge(
            detalle_balance[
                [
                    escenario_col,
                    "participacion",
                ]
            ],
            on=escenario_col,
            how="left",
        )

    else:

        resultado["participacion"] = 0.0

    # ------------------------------------------------------------------
    # Score de escenario
    # ------------------------------------------------------------------

    resultado[
        "score_evaluacion"
    ] = (

        resultado["score_tamano"].fillna(0)
        * PESO_TAMANO

        +

        resultado["score_cohesion"].fillna(0)
        * PESO_COHESION

        +

        resultado["score_indicadores"].fillna(0)
        * PESO_INDICADORES

        +

        (
            1
            -
            (
                resultado["participacion"]
                -
                1
                /
                max(
                    len(resultado),
                    1,
                )
            ).abs()
            /
            max(
                1
                /
                max(
                    len(resultado),
                    1,
                ),
                1e-9,
            )
        ).clip(
            lower=0,
            upper=1,
        )
        * PESO_BALANCE
    )

    resultado[
        "clasificacion"
    ] = resultado[
        "score_evaluacion"
    ].apply(
        clasificar_escenario
    )

    resultado = resultado.sort_values(
        "score_evaluacion",
        ascending=False,
    ).reset_index(
        drop=True
    )

    resultado[
        "ranking_evaluacion"
    ] = (
        resultado.index + 1
    )

    print()

    print(
        resultado[
            [
                escenario_col,
                "cantidad_proyectos",
                "score_evaluacion",
                "clasificacion",
                "ranking_evaluacion",
            ]
        ].to_string(
            index=False
        )
    )

    return resultado


# ======================================================================================
# SCORE GLOBAL
# ======================================================================================

def calcular_score_global(
    cobertura_score: float,
    tamano_score: float,
    cohesion_score: float,
    indicadores_score: float,
    balance_score: float,
) -> float:

    return (

        cobertura_score
        * PESO_COBERTURA

        +

        tamano_score
        * PESO_TAMANO

        +

        cohesion_score
        * PESO_COHESION

        +

        indicadores_score
        * PESO_INDICADORES

        +

        balance_score
        * PESO_BALANCE
    )


# ======================================================================================
# RECOMENDACIONES
# ======================================================================================

def generar_recomendaciones(
    evaluacion: pd.DataFrame,
    columnas: dict[str, str | None],
    score_global: float,
) -> pd.DataFrame:

    encabezado(
        "10",
        "GENERANDO RECOMENDACIONES",
    )

    escenario_col = columnas["escenario"]

    recomendaciones = []

    for _, fila in evaluacion.iterrows():

        escenario = fila[
            escenario_col
        ]

        score = safe_float(
            fila.get(
                "score_evaluacion",
                0,
            )
        )

        cantidad = int(
            safe_float(
                fila.get(
                    "cantidad_proyectos",
                    0,
                )
            )
        )

        clasificacion = fila.get(
            "clasificacion",
            "",
        )

        if cantidad < MIN_PROYECTOS_ESCENARIO:

            recomendacion = (
                "REVISAR TAMAÑO: el escenario "
                "no alcanza el mínimo recomendado "
                "de proyectos."
            )

            nivel = "ALTA"

        elif score < 0.35:

            recomendacion = (
                "REVISIÓN CRÍTICA: revisar "
                "cohesión territorial y composición "
                "multicriterio."
            )

            nivel = "ALTA"

        elif score < 0.50:

            recomendacion = (
                "REVISAR: el escenario presenta "
                "desempeño inferior al conjunto."
            )

            nivel = "MEDIA"

        elif score < 0.65:

            recomendacion = (
                "ACEPTABLE: mantener como escenario "
                "válido y revisar oportunidades de "
                "mejora."
            )

            nivel = "MEDIA"

        else:

            recomendacion = (
                "CONSOLIDAR: escenario con buen "
                "desempeño relativo."
            )

            nivel = "BAJA"

        recomendaciones.append(
            {
                escenario_col: escenario,
                "score_evaluacion": score,
                "clasificacion": clasificacion,
                "cantidad_proyectos": cantidad,
                "nivel_recomendacion": nivel,
                "recomendacion": recomendacion,
            }
        )

    resultado = pd.DataFrame(
        recomendaciones
    )

    print(
        resultado.to_string(
            index=False
        )
    )

    return resultado


# ======================================================================================
# METADATA
# ======================================================================================

def construir_metadata(
    gdf: gpd.GeoDataFrame,
    validacion: dict[str, Any],
    columnas: dict[str, str | None],
    score_global: float,
    scores: dict[str, float],
    evaluacion: pd.DataFrame,
) -> dict[str, Any]:

    escenarios = int(
        gdf[
            columnas["escenario"]
        ].nunique()
    )

    return {

        "proceso": PROCESO,

        "nombre":
            NOMBRE,

        "version":
            VERSION,

        "fecha_evaluacion":
            pd.Timestamp.now(
                tz="America/Argentina/Buenos_Aires"
            ).isoformat(),

        "entrada":
            str(INPUT_PARQUET),

        "salida":
            str(OUTPUT_DIR),

        "proyectos":
            int(len(gdf)),

        "escenarios":
            escenarios,

        "score_global":
            round(
                float(score_global),
                6,
            ),

        "clasificacion_global":
            clasificar_score(
                score_global
            ),

        "scores":

            {
                clave:
                    round(
                        float(valor),
                        6,
                    )
                for clave, valor
                in scores.items()
            },

        "pesos":

            {
                "cobertura":
                    PESO_COBERTURA,

                "tamano":
                    PESO_TAMANO,

                "cohesion":
                    PESO_COHESION,

                "indicadores":
                    PESO_INDICADORES,

                "balance":
                    PESO_BALANCE,
            },

        "validacion":
            validacion,

        "columnas_resueltas":
            columnas,

        "ranking":

            evaluacion[
                [
                    columnas["escenario"],
                    "ranking_evaluacion",
                    "score_evaluacion",
                    "clasificacion",
                ]
            ].to_dict(
                orient="records"
            ),
    }


# ======================================================================================
# EXPORTACIÓN
# ======================================================================================

def exportar(
    evaluacion: pd.DataFrame,
    recomendaciones: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:

    encabezado(
        "11",
        "EXPORTANDO RESULTADOS",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Parquet
    # --------------------------------------------------------------

    evaluacion.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    print(
        f"Evaluación Parquet    : "
        f"{OUTPUT_PARQUET}"
    )

    # --------------------------------------------------------------
    # CSV principal
    # --------------------------------------------------------------

    evaluacion.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Evaluación CSV        : "
        f"{OUTPUT_CSV}"
    )

    # --------------------------------------------------------------
    # CSV detalle
    # --------------------------------------------------------------

    evaluacion.to_csv(
        OUTPUT_ESCENARIOS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Detalle escenarios    : "
        f"{OUTPUT_ESCENARIOS_CSV}"
    )

    # --------------------------------------------------------------
    # Recomendaciones
    # --------------------------------------------------------------

    recomendaciones.to_csv(
        OUTPUT_RECOMENDACIONES,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Recomendaciones       : "
        f"{OUTPUT_RECOMENDACIONES}"
    )

    # --------------------------------------------------------------
    # JSON
    # --------------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            metadata,
            archivo,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(
        f"Metadata              : "
        f"{OUTPUT_JSON}"
    )


# ======================================================================================
# MAIN
# ======================================================================================

def main() -> int:

    inicio = pd.Timestamp.now()

    print()
    print("=" * 88)
    print(
        f"{PROCESO:02d} - "
        f"{NOMBRE} - {VERSION}"
    )
    print("=" * 88)

    print(
        f"Proyecto : {BASE_DIR}"
    )

    print(
        f"Entrada  : {INPUT_PARQUET}"
    )

    print(
        f"Salida   : {OUTPUT_DIR}"
    )

    print()
    print("CONFIGURACIÓN")
    print(
        f"  Versión             : {VERSION}"
    )

    print(
        f"  K esperado          : "
        f"{K_MIN} - {K_MAX}"
    )

    print(
        f"  Mínimo proyectos    : "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )

    print(
        f"  Peso cobertura      : "
        f"{PESO_COBERTURA:.0%}"
    )

    print(
        f"  Peso tamaño         : "
        f"{PESO_TAMANO:.0%}"
    )

    print(
        f"  Peso cohesión       : "
        f"{PESO_COHESION:.0%}"
    )

    print(
        f"  Peso indicadores    : "
        f"{PESO_INDICADORES:.0%}"
    )

    print(
        f"  Peso balance        : "
        f"{PESO_BALANCE:.0%}"
    )

    # ------------------------------------------------------------------
    # 1
    # ------------------------------------------------------------------

    gdf = cargar_escenarios()

    # ------------------------------------------------------------------
    # 2
    # ------------------------------------------------------------------

    validacion = validar_entrada(
        gdf
    )

    # ------------------------------------------------------------------
    # 3
    # ------------------------------------------------------------------

    columnas = resolver_columnas(
        gdf
    )

    # ------------------------------------------------------------------
    # 4
    # ------------------------------------------------------------------

    cobertura_score, detalle_cobertura = (
        evaluar_cobertura(
            gdf,
            columnas,
        )
    )

    # ------------------------------------------------------------------
    # 5
    # ------------------------------------------------------------------

    tamano_score, detalle_tamano = (
        evaluar_tamano(
            gdf,
            columnas,
        )
    )

    # ------------------------------------------------------------------
    # 6
    # ------------------------------------------------------------------

    cohesion_score, detalle_cohesion = (
        evaluar_cohesion(
            gdf,
            columnas,
        )
    )

    # ------------------------------------------------------------------
    # 7
    # ------------------------------------------------------------------

    indicadores_score, detalle_indicadores = (
        evaluar_indicadores(
            gdf,
            columnas,
        )
    )

    # ------------------------------------------------------------------
    # 8
    # ------------------------------------------------------------------

    balance_score, detalle_balance = (
        evaluar_balance(
            gdf,
            columnas,
        )
    )

    # ------------------------------------------------------------------
    # 9
    # ------------------------------------------------------------------

    encabezado(
        "9",
        "CALCULANDO SCORE GLOBAL",
    )

    score_global = calcular_score_global(
        cobertura_score,
        tamano_score,
        cohesion_score,
        indicadores_score,
        balance_score,
    )

    print(
        f"Cobertura             : "
        f"{cobertura_score:.4f}"
    )

    print(
        f"Tamaño                : "
        f"{tamano_score:.4f}"
    )

    print(
        f"Cohesión              : "
        f"{cohesion_score:.4f}"
    )

    print(
        f"Indicadores           : "
        f"{indicadores_score:.4f}"
    )

    print(
        f"Balance               : "
        f"{balance_score:.4f}"
    )

    print()

    print(
        f"SCORE GLOBAL          : "
        f"{score_global:.4f}"
    )

    print(
        f"CLASIFICACIÓN         : "
        f"{clasificar_score(score_global)}"
    )

    # ------------------------------------------------------------------
    # 10
    # ------------------------------------------------------------------

    evaluacion = construir_evaluacion_integrada(
        gdf,
        columnas,
        detalle_tamano,
        detalle_cohesion,
        detalle_indicadores,
        detalle_balance,
    )

    # ------------------------------------------------------------------
    # 11
    # ------------------------------------------------------------------

    recomendaciones = generar_recomendaciones(
        evaluacion,
        columnas,
        score_global,
    )

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    scores = {

        "cobertura":
            cobertura_score,

        "tamano":
            tamano_score,

        "cohesion":
            cohesion_score,

        "indicadores":
            indicadores_score,

        "balance":
            balance_score,

    }

    metadata = construir_metadata(
        gdf,
        validacion,
        columnas,
        score_global,
        scores,
        evaluacion,
    )

    # ------------------------------------------------------------------
    # EXPORTAR
    # ------------------------------------------------------------------

    exportar(
        evaluacion,
        recomendaciones,
        metadata,
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    fin = pd.Timestamp.now()

    duracion = (
        fin - inicio
    ).total_seconds()

    encabezado(
        "12",
        "PROCESO 28 FINALIZADO CORRECTAMENTE",
    )

    print(
        f"Proyectos evaluados    : "
        f"{len(gdf):,}"
    )

    print(
        f"Escenarios evaluados   : "
        f"{gdf[columnas['escenario']].nunique()}"
    )

    print(
        f"Score global           : "
        f"{score_global:.4f}"
    )

    print(
        f"Clasificación          : "
        f"{clasificar_score(score_global)}"
    )

    print(
        f"Duración               : "
        f"{duracion:.2f} segundos"
    )

    print()

    print(
        "RANKING DE ESCENARIOS"
    )

    print(
        evaluacion[
            [
                columnas["escenario"],
                "ranking_evaluacion",
                "cantidad_proyectos",
                "score_evaluacion",
                "clasificacion",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Salida principal:"
    )

    print(
        OUTPUT_PARQUET
    )

    print()

    return 0


# ======================================================================================
# EJECUCIÓN
# ======================================================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "Proceso interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 88)
        print(
            "ERROR EN PROCESO 28"
        )
        print("=" * 88)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        traceback.print_exc()

        sys.exit(1)