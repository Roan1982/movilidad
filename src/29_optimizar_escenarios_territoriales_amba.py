# -*- coding: utf-8 -*-

"""
================================================================================
29 - OPTIMIZACIÓN DE ESCENARIOS TERRITORIALES AMBA - V4.0
================================================================================
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

VERSION = "V4.0"

# Restricciones
MIN_PROYECTOS = 8
MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

# Optimización
MAX_ITERACIONES = 100

# Un movimiento debe producir al menos esta mejora
MEJORA_MINIMA = 0.00001

# Penalización muy pequeña para evitar movimientos innecesarios
PENALIZACION_MOVIMIENTO = 0.000001

# Escala territorial de cohesión
RADIO_COHESION_METROS = 50000.0

# ------------------------------------------------------------------------------
# PESOS DE LA FUNCIÓN OBJETIVO
# ------------------------------------------------------------------------------

PESO_COHESION = 0.30
PESO_BALANCE = 0.20
PESO_INDICADORES = 0.20
PESO_SCORE_TERRITORIAL = 0.15
PESO_ESTABILIDAD = 0.15

SUMA_PESOS = (
    PESO_COHESION
    + PESO_BALANCE
    + PESO_INDICADORES
    + PESO_SCORE_TERRITORIAL
    + PESO_ESTABILIDAD
)

if not math.isclose(SUMA_PESOS, 1.0):
    raise ValueError(
        f"Los pesos de optimización deben sumar 1.0. "
        f"Actualmente suman {SUMA_PESOS}."
    )


# ==============================================================================
# RUTAS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_ESCENARIOS = (
    INPUT_DIR
    / "escenarios_territoriales_amba.parquet"
)

INPUT_EVALUACION = (
    INPUT_DIR
    / "evaluacion_escenarios_territoriales_amba.parquet"
)

INPUT_RECOMENDACIONES = (
    INPUT_DIR
    / "recomendaciones_escenarios_territoriales_amba.csv"
)

OUTPUT_OPTIMIZADO = (
    INPUT_DIR
    / "escenarios_territoriales_amba_optimizado.parquet"
)

OUTPUT_CSV = (
    INPUT_DIR
    / "escenarios_territoriales_amba_optimizado.csv"
)

OUTPUT_EVALUACION = (
    INPUT_DIR
    / "evaluacion_escenarios_optimizada.csv"
)

OUTPUT_MOVIMIENTOS = (
    INPUT_DIR
    / "movimientos_optimizacion_escenarios.csv"
)

OUTPUT_RESUMEN = (
    INPUT_DIR
    / "resumen_optimizacion_escenarios.csv"
)

OUTPUT_METADATA = (
    INPUT_DIR
    / "metadata_optimizacion_escenarios.json"
)


# ==============================================================================
# UTILIDADES
# ==============================================================================

def encabezado(titulo: str) -> None:
    print()
    print("=" * 96)
    print(titulo)
    print("=" * 96)


def subtitulo(titulo: str) -> None:
    print()
    print("-" * 80)
    print(titulo)
    print("-" * 80)


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: List[str],
    obligatoria: bool = True,
) -> Optional[str]:

    for columna in candidatos:
        if columna in df.columns:
            return columna

    if obligatoria:
        raise KeyError(
            f"No se encontró ninguna de las columnas: {candidatos}"
        )

    return None


def normalizar_serie(
    serie: pd.Series,
) -> pd.Series:

    valores = pd.to_numeric(
        serie,
        errors="coerce",
    )

    resultado = pd.Series(
        0.5,
        index=serie.index,
        dtype=float,
    )

    validos = valores.notna()

    if validos.sum() == 0:
        return resultado

    minimo = float(valores[validos].min())
    maximo = float(valores[validos].max())

    if math.isclose(minimo, maximo):
        resultado.loc[validos] = 0.5
        return resultado

    resultado.loc[validos] = (
        valores.loc[validos] - minimo
    ) / (
        maximo - minimo
    )

    return resultado


def safe_float(
    valor,
    default: float = 0.0,
) -> float:

    try:
        resultado = float(valor)

        if not math.isfinite(resultado):
            return default

        return resultado

    except Exception:
        return default


# ==============================================================================
# CARGA
# ==============================================================================

def cargar_escenarios() -> gpd.GeoDataFrame:

    encabezado(
        "1. CARGANDO ESCENARIOS DEL PROCESO 27"
    )

    if not INPUT_ESCENARIOS.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{INPUT_ESCENARIOS}"
        )

    gdf = gpd.read_parquet(
        INPUT_ESCENARIOS
    )

    print(f"Archivo     : {INPUT_ESCENARIOS}")
    print(f"Registros   : {len(gdf):,}")
    print(f"Columnas    : {len(gdf.columns)}")
    print(f"CRS         : {gdf.crs}")

    if gdf.empty:
        raise ValueError(
            "El archivo de escenarios está vacío."
        )

    return gdf


def cargar_evaluacion() -> Optional[pd.DataFrame]:

    subtitulo(
        "Cargando evaluación del proceso 28"
    )

    if not INPUT_EVALUACION.exists():
        print(
            "Evaluación del proceso 28: NO DISPONIBLE"
        )
        return None

    evaluacion = pd.read_parquet(
        INPUT_EVALUACION
    )

    print(
        f"Archivo     : {INPUT_EVALUACION}"
    )

    print(
        f"Registros   : {len(evaluacion):,}"
    )

    print(
        f"Columnas    : {len(evaluacion.columns)}"
    )

    return evaluacion


def cargar_recomendaciones() -> Optional[pd.DataFrame]:

    if not INPUT_RECOMENDACIONES.exists():
        print(
            "Recomendaciones proceso 28: NO DISPONIBLES"
        )
        return None

    recomendaciones = pd.read_csv(
        INPUT_RECOMENDACIONES,
        encoding="utf-8-sig",
    )

    print(
        f"Recomendaciones encontradas: "
        f"{len(recomendaciones):,}"
    )

    return recomendaciones


# ==============================================================================
# VALIDACIÓN
# ==============================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> Tuple[str, str]:

    encabezado(
        "2. VALIDANDO ENTRADA"
    )

    escenario_col = encontrar_columna(
        gdf,
        [
            "escenario_id",
            "id_escenario",
        ],
    )

    proyecto_col = encontrar_columna(
        gdf,
        [
            "proyecto_id",
            "id_proyecto",
        ],
    )

    geometria_nula = int(
        gdf.geometry.isna().sum()
    )

    geometria_vacia = int(
        gdf.geometry.is_empty.sum()
    )

    try:
        geometria_invalida = int(
            (~gdf.geometry.is_valid).sum()
        )
    except Exception:
        geometria_invalida = 0

    proyectos_nulos = int(
        gdf[proyecto_col].isna().sum()
    )

    escenarios_nulos = int(
        gdf[escenario_col].isna().sum()
    )

    duplicados = int(
        gdf[proyecto_col].duplicated().sum()
    )

    print(
        f"Geometrías nulas       : {geometria_nula}"
    )

    print(
        f"Geometrías vacías      : {geometria_vacia}"
    )

    print(
        f"Geometrías inválidas   : {geometria_invalida}"
    )

    print(
        f"Proyectos nulos        : {proyectos_nulos}"
    )

    print(
        f"Escenarios nulos       : {escenarios_nulos}"
    )

    print(
        f"Escenarios             : "
        f"{gdf[escenario_col].nunique()}"
    )

    print(
        f"Proyectos              : "
        f"{gdf[proyecto_col].nunique()}"
    )

    print(
        f"Duplicados proyecto    : {duplicados}"
    )

    if geometria_nula > 0:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if geometria_vacia > 0:
        raise ValueError(
            "Existen geometrías vacías."
        )

    if geometria_invalida > 0:
        raise ValueError(
            "Existen geometrías inválidas."
        )

    if proyectos_nulos > 0:
        raise ValueError(
            "Existen proyectos nulos."
        )

    if escenarios_nulos > 0:
        raise ValueError(
            "Existen escenarios nulos."
        )

    if duplicados > 0:
        raise ValueError(
            "Un proyecto aparece más de una vez."
        )

    print(
        "Validación de entrada: OK"
    )

    return escenario_col, proyecto_col


# ==============================================================================
# INDICADORES
# ==============================================================================

INDICADORES_PREFERIDOS = [
    "indice_demanda_estructural",
    "deficit_infraestructura",
    "indice_conectividad_estructural",
    "indice_intermodalidad_estructural",
    "indice_integracion_territorial",
    "indice_centralidad_estructural",
    "impacto_potencial",
    "urgencia_intervencion",
    "score_prioridad_territorial",
    "score_cartera",
]


def detectar_indicadores(
    gdf: pd.DataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> List[str]:

    encabezado(
        "3. PREPARANDO INDICADORES"
    )

    subtitulo(
        "Detectando indicadores territoriales"
    )

    excluir = {
        escenario_col,
        proyecto_col,
        "geometry",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
        "ranking_escenario",
        "ranking",
    }

    indicadores = []

    for columna in INDICADORES_PREFERIDOS:

        if columna not in gdf.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            gdf[columna]
        ):
            continue

        indicadores.append(columna)

    if len(indicadores) < 3:

        for columna in gdf.columns:

            if columna in excluir:
                continue

            if columna in indicadores:
                continue

            if columna.startswith("ranking_"):
                continue

            if columna.startswith("nivel_"):
                continue

            if not pd.api.types.is_numeric_dtype(
                gdf[columna]
            ):
                continue

            indicadores.append(columna)

    print(
        f"Indicadores seleccionados: "
        f"{len(indicadores)}"
    )

    for indicador in indicadores:
        print(
            f"  - {indicador}"
        )

    return indicadores


def detectar_score_territorial(
    gdf: pd.DataFrame,
) -> Optional[str]:

    candidatos = [
        "score_prioridad_territorial",
        "score_cartera",
        "score_territorial",
        "prioridad_score",
    ]

    return encontrar_columna(
        gdf,
        candidatos,
        obligatoria=False,
    )


# ==============================================================================
# GEOMETRÍA MÉTRICA
# ==============================================================================

def preparar_geometria_metrica(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    if gdf.crs is None:
        raise ValueError(
            "La capa no tiene CRS."
        )

    if gdf.crs.is_geographic:
        return gdf.to_crs(
            epsg=3857
        )

    return gdf.copy()


def union_geometrias(
    geometria,
):
    if hasattr(
        geometria,
        "union_all",
    ):
        return geometria.union_all()

    return geometria.unary_union


# ==============================================================================
# MATRIZ DE INDICADORES
# ==============================================================================

def construir_indicadores_normalizados(
    gdf: pd.DataFrame,
    indicadores: List[str],
) -> pd.DataFrame:

    resultado = pd.DataFrame(
        index=gdf.index
    )

    for indicador in indicadores:

        resultado[indicador] = (
            normalizar_serie(
                gdf[indicador]
            )
        )

    return resultado


# ==============================================================================
# MÉTRICAS
# ==============================================================================

def calcular_balance(
    cantidades: np.ndarray,
) -> float:

    if len(cantidades) == 0:
        return 0.0

    cantidades = np.asarray(
        cantidades,
        dtype=float,
    )

    promedio = float(
        cantidades.mean()
    )

    if promedio <= 0:
        return 0.0

    cv = float(
        cantidades.std(ddof=0)
        / promedio
    )

    return max(
        0.0,
        1.0 - cv,
    )


def calcular_cohesion(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> float:

    distancias_medias = []

    for escenario, grupo in gdf_metric.groupby(
        escenario_col
    ):

        if len(grupo) < 2:
            continue

        puntos = grupo.geometry

        centroide = (
            union_geometrias(
                puntos
            ).centroid
        )

        distancias = np.array(
            [
                safe_float(
                    punto.distance(
                        centroide
                    )
                )
                for punto in puntos
            ],
            dtype=float,
        )

        if len(distancias) == 0:
            continue

        distancias_medias.append(
            float(
                np.mean(
                    distancias
                )
            )
        )

    if not distancias_medias:
        return 0.0

    distancia_media = float(
        np.mean(
            distancias_medias
        )
    )

    score = (
        1.0
        - (
            distancia_media
            / RADIO_COHESION_METROS
        )
    )

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def calcular_indicadores(
    gdf: pd.DataFrame,
    escenario_col: str,
    indicadores: List[str],
) -> float:

    if not indicadores:
        return 0.0

    scores = []

    for indicador in indicadores:

        valores = pd.to_numeric(
            gdf[indicador],
            errors="coerce",
        )

        if valores.notna().sum() < 2:
            continue

        media_global = float(
            valores.mean()
        )

        if math.isclose(
            media_global,
            0.0,
        ):
            continue

        medias = (
            pd.DataFrame(
                {
                    "escenario": gdf[
                        escenario_col
                    ],
                    "valor": valores,
                }
            )
            .groupby(
                "escenario"
            )["valor"]
            .mean()
        )

        if len(medias) < 2:
            scores.append(1.0)
            continue

        dispersion = float(
            medias.std(
                ddof=0
            )
        )

        variacion_relativa = (
            dispersion
            / (
                abs(
                    media_global
                )
                + 1e-9
            )
        )

        score = (
            1.0
            / (
                1.0
                + variacion_relativa
            )
        )

        scores.append(
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


def calcular_score_territorial(
    gdf: pd.DataFrame,
    escenario_col: str,
    score_col: Optional[str],
) -> float:

    if score_col is None:
        return 0.5

    valores = normalizar_serie(
        gdf[score_col]
    )

    medias = (
        pd.DataFrame(
            {
                "escenario": gdf[
                    escenario_col
                ],
                "score": valores,
            }
        )
        .groupby(
            "escenario"
        )["score"]
        .mean()
    )

    if medias.empty:
        return 0.5

    resultado = float(
        medias.mean()
    )

    return max(
        0.0,
        min(
            1.0,
            resultado,
        ),
    )


def calcular_estabilidad(
    gdf: pd.DataFrame,
    escenario_col: str,
    escenario_original_col: str,
) -> float:

    if (
        escenario_original_col
        not in gdf.columns
    ):
        return 1.0

    validos = gdf[
        escenario_original_col
    ].notna()

    if validos.sum() == 0:
        return 1.0

    iguales = (
        gdf.loc[
            validos,
            escenario_col,
        ].astype(str)
        ==
        gdf.loc[
            validos,
            escenario_original_col,
        ].astype(str)
    )

    return float(
        iguales.mean()
    )


# ==============================================================================
# RESTRICCIONES
# ==============================================================================

def validar_restricciones(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> bool:

    if gdf[proyecto_col].duplicated().any():
        return False

    if gdf[proyecto_col].isna().any():
        return False

    if gdf[escenario_col].isna().any():
        return False

    cantidad_escenarios = int(
        gdf[
            escenario_col
        ].nunique()
    )

    if (
        cantidad_escenarios
        < MIN_ESCENARIOS
    ):
        return False

    if (
        cantidad_escenarios
        > MAX_ESCENARIOS
    ):
        return False

    conteos = (
        gdf.groupby(
            escenario_col
        )[proyecto_col]
        .count()
    )

    if conteos.empty:
        return False

    if (
        int(conteos.min())
        < MIN_PROYECTOS
    ):
        return False

    return True


# ==============================================================================
# FUNCIÓN OBJETIVO
# ==============================================================================

def evaluar_estado(
    gdf: gpd.GeoDataFrame,
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
    escenario_original_col: str,
) -> Dict[str, float]:

    conteos = (
        gdf.groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .to_numpy(
            dtype=float
        )
    )

    cobertura = 1.0

    estructura = (
        1.0
        if (
            MIN_ESCENARIOS
            <= len(conteos)
            <= MAX_ESCENARIOS
        )
        else 0.0
    )

    tamaño = float(
        np.mean(
            [
                min(
                    1.0,
                    n / (MIN_PROYECTOS if MIN_PROYECTOS > 0 else 1),
                )
                for n in conteos
            ]
        )
    ) if len(conteos) > 0 else 0.0

    cohesion = calcular_cohesion(
        gdf_metric,
        escenario_col,
    )

    balance = calcular_balance(
        conteos
    )

    indicadores_score = (
        calcular_indicadores(
            gdf,
            escenario_col,
            indicadores,
        )
    )

    score_territorial = (
        calcular_score_territorial(
            gdf,
            escenario_col,
            score_col,
        )
    )

    estabilidad = calcular_estabilidad(
        gdf,
        escenario_col,
        escenario_original_col,
    )

    score_objetivo = (
        PESO_COHESION
        * cohesion
        +
        PESO_BALANCE
        * balance
        +
        PESO_INDICADORES
        * indicadores_score
        +
        PESO_SCORE_TERRITORIAL
        * score_territorial
        +
        PESO_ESTABILIDAD
        * estabilidad
    )

    return {
        "cobertura": cobertura,
        "estructura_escenarios": estructura,
        "tamano": tamaño,
        "cohesion": cohesion,
        "balance": balance,
        "indicadores": indicadores_score,
        "score_territorial": score_territorial,
        "estabilidad": estabilidad,
        "score_objetivo": float(
            score_objetivo
        ),
    }


# ==============================================================================
# CENTROIDES
# ==============================================================================

def obtener_centroides(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> Dict:

    centroides = {}

    for escenario, grupo in gdf_metric.groupby(
        escenario_col
    ):

        centroides[escenario] = (
            union_geometrias(
                grupo.geometry
            ).centroid
        )

    return centroides


# ==============================================================================
# CANDIDATOS
# ==============================================================================

def generar_candidatos(
    gdf: gpd.GeoDataFrame,
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> List[Tuple]:

    escenarios = sorted(
        gdf[
            escenario_col
        ].dropna()
        .unique()
    )

    conteos = (
        gdf.groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .to_dict()
    )

    centroides = obtener_centroides(
        gdf_metric,
        escenario_col,
    )

    candidatos = []

    for idx in gdf.index:

        origen = gdf.loc[
            idx,
            escenario_col,
        ]

        cantidad_origen = int(
            conteos.get(
                origen,
                0,
            )
        )

        if (
            cantidad_origen
            <= MIN_PROYECTOS
        ):
            continue

        punto = gdf_metric.loc[
            idx,
            "geometry",
        ]

        centro_origen = centroides.get(
            origen
        )

        if centro_origen is None:
            continue

        distancia_origen = safe_float(
            punto.distance(
                centro_origen
            )
        )

        for destino in escenarios:

            if destino == origen:
                continue

            if destino not in centroides:
                continue

            centro_destino = centroides[
                destino
            ]

            distancia_destino = safe_float(
                punto.distance(
                    centro_destino
                )
            )

            mejora_espacial = (
                distancia_origen
                - distancia_destino
            )

            candidatos.append(
                (
                    idx,
                    origen,
                    destino,
                    mejora_espacial,
                )
            )

    return candidatos


# ==============================================================================
# OPTIMIZACIÓN MULTIOBJETIVO
# ==============================================================================

def optimizar(
    gdf_original: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
) -> Tuple[
    gpd.GeoDataFrame,
    List[Dict],
    Dict[str, float],
    Dict[str, float],
]:

    encabezado(
        "5. OPTIMIZACIÓN MULTIOBJETIVO V4"
    )

    gdf_opt = gdf_original.copy()

    escenario_original_col = (
        "__escenario_original_v4"
    )

    gdf_opt[
        escenario_original_col
    ] = gdf_opt[
        escenario_col
    ].astype(str)

    gdf_metric = preparar_geometria_metrica(
        gdf_opt
    )

    if not validar_restricciones(
        gdf_opt,
        escenario_col,
        proyecto_col,
    ):
        raise ValueError(
            "El estado inicial no cumple las restricciones."
        )

    metricas_estado_actual = evaluar_estado(
        gdf_opt,
        gdf_metric,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
        escenario_original_col,
    )

    metricas_iniciales = dict(metricas_estado_actual)

    print(
        f"Score objetivo inicial : "
        f"{metricas_iniciales['score_objetivo']:.6f}"
    )

    print(
        f"Escenarios             : "
        f"{gdf_opt[escenario_col].nunique()}"
    )

    print(
        f"Cohesión               : "
        f"{metricas_iniciales['cohesion']:.6f}"
    )

    print(
        f"Balance                : "
        f"{metricas_iniciales['balance']:.6f}"
    )

    print(
        f"Indicadores            : "
        f"{metricas_iniciales['indicadores']:.6f}"
    )

    print(
        f"Score territorial      : "
        f"{metricas_iniciales['score_territorial']:.6f}"
    )

    print(
        f"Estabilidad            : "
        f"{metricas_iniciales['estabilidad']:.6f}"
    )

    movimientos = []

    for iteracion in range(
        1,
        MAX_ITERACIONES + 1,
    ):

        candidatos = generar_candidatos(
            gdf_opt,
            gdf_metric,
            escenario_col,
            proyecto_col,
        )

        if not candidatos:
            print(
                f"Iteración {iteracion:03d} "
                f"| sin candidatos"
            )
            break

        mejor_movimiento = None
        mejor_score = (
            metricas_estado_actual[
                "score_objetivo"
            ]
        )
        mejor_metricas = None
        candidatos_evaluados = 0

        for (
            idx,
            origen,
            destino,
            mejora_espacial,
        ) in candidatos:

            estado_origen = gdf_opt.loc[
                idx,
                escenario_col,
            ]

            if estado_origen != origen:
                continue

            gdf_opt.loc[
                idx,
                escenario_col,
            ] = destino

            valido = validar_restricciones(
                gdf_opt,
                escenario_col,
                proyecto_col,
            )

            if valido:

                metricas = evaluar_estado(
                    gdf_opt,
                    gdf_metric,
                    escenario_col,
                    proyecto_col,
                    indicadores,
                    score_col,
                    escenario_original_col,
                )

                score = (
                    metricas[
                        "score_objetivo"
                    ]
                    - PENALIZACION_MOVIMIENTO
                )

                candidatos_evaluados += 1

                if score > (
                    mejor_score
                    + MEJORA_MINIMA
                ):

                    mejor_score = score
                    mejor_movimiento = (
                        idx,
                        origen,
                        destino,
                        mejora_espacial,
                    )
                    mejor_metricas = metricas

            gdf_opt.loc[
                idx,
                escenario_col,
            ] = origen

        if mejor_movimiento is None:
            print(
                f"Iteración {iteracion:03d} "
                f"| candidatos evaluados: "
                f"{candidatos_evaluados} "
                f"| sin mejora"
            )
            break

        (
            idx,
            origen,
            destino,
            mejora_espacial,
        ) = mejor_movimiento

        proyecto = gdf_opt.loc[
            idx,
            proyecto_col,
        ]

        score_antes = (
            metricas_estado_actual[
                "score_objetivo"
            ]
        )

        score_despues = (
            mejor_metricas[
                "score_objetivo"
            ]
        )

        gdf_opt.loc[
            idx,
            escenario_col,
        ] = destino

        movimiento = {
            "iteracion": iteracion,
            "proyecto_id": proyecto,
            "escenario_origen": origen,
            "escenario_destino": destino,
            "mejora_espacial_metros": mejora_espacial,
            "score_antes": score_antes,
            "score_despues": score_despues,
            "mejora_score": score_despues - score_antes,
            "cohesion_antes": metricas_estado_actual["cohesion"],
            "cohesion_despues": mejor_metricas["cohesion"],
            "balance_antes": metricas_estado_actual["balance"],
            "balance_despues": mejor_metricas["balance"],
            "indicadores_antes": metricas_estado_actual["indicadores"],
            "indicadores_despues": mejor_metricas["indicadores"],
            "score_territorial_antes": metricas_estado_actual["score_territorial"],
            "score_territorial_despues": mejor_metricas["score_territorial"],
            "estabilidad_antes": metricas_estado_actual["estabilidad"],
            "estabilidad_despues": mejor_metricas["estabilidad"],
        }

        movimientos.append(movimiento)

        # Actualización del estado actual
        metricas_estado_actual = mejor_metricas

        print(
            f"Iteración {iteracion:03d} "
            f"| candidatos: {candidatos_evaluados} "
            f"| movimiento: {proyecto} {origen} -> {destino} "
            f"| mejora: {score_despues - score_antes:+.8f}"
        )

    metricas_finales = evaluar_estado(
        gdf_opt,
        gdf_metric,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
        escenario_original_col,
    )

    print()
    print(
        f"Movimientos aceptados : "
        f"{len(movimientos)}"
    )

    print(
        f"Score final            : "
        f"{metricas_finales['score_objetivo']:.6f}"
    )

    mejora_global = metricas_finales['score_objetivo'] - metricas_iniciales['score_objetivo']
    print(
        f"Mejora total           : "
        f"{mejora_global:+.8f}"
    )

    return (
        gdf_opt,
        movimientos,
        metricas_iniciales,
        metricas_finales,
    )


# ==============================================================================
# EVALUACIÓN DETALLADA
# ==============================================================================

def evaluar_escenarios_detalladamente(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
    score_col: Optional[str],
    gdf_metric: gpd.GeoDataFrame,
) -> pd.DataFrame:

    encabezado(
        "6. EVALUACIÓN DETALLADA DE ESCENARIOS"
    )

    filas = []

    conteos = (
        gdf.groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .sort_values(
            ascending=False
        )
    )

    for escenario, cantidad in conteos.items():

        grupo = gdf[
            gdf[escenario_col]
            == escenario
        ]

        grupo_metric = gdf_metric[
            gdf_metric[escenario_col]
            == escenario
        ]

        if len(grupo_metric) >= 2:

            centro = (
                union_geometrias(
                    grupo_metric.geometry
                ).centroid
            )

            distancias = [
                safe_float(
                    geometria.distance(
                        centro
                    )
                )
                for geometria
                in grupo_metric.geometry
            ]

            distancia_media = float(
                np.mean(
                    distancias
                )
            )

            cohesion = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        distancia_media
                        / RADIO_COHESION_METROS
                    ),
                ),
            )

        else:

            distancia_media = 0.0
            cohesion = 1.0

        if score_col is not None:

            score = float(
                normalizar_serie(
                    gdf[
                        score_col
                    ]
                ).loc[
                    grupo.index
                ].mean()
            )

        else:

            score = 0.5

        indicadores_medios = {}

        for indicador in indicadores:

            valores = pd.to_numeric(
                grupo[indicador],
                errors="coerce",
            )

            if valores.notna().sum():

                indicadores_medios[
                    f"media_{indicador}"
                ] = float(
                    valores.mean()
                )

        fila = {
            "escenario_id": escenario,
            "cantidad_proyectos": int(
                cantidad
            ),
            "porcentaje_proyectos": (
                float(
                    cantidad
                    / len(gdf)
                    * 100.0
                )
            ),
            "distancia_media_centroide_m": (
                distancia_media
            ),
            "cohesion_escenario": cohesion,
            "score_territorial_escenario": score,
        }

        fila.update(
            indicadores_medios
        )

        filas.append(
            fila
        )

    resultado = pd.DataFrame(
        filas
    )

    if not resultado.empty:

        resultado[
            "ranking_escenario"
        ] = (
            resultado[
                "score_territorial_escenario"
            ]
            .rank(
                ascending=False,
                method="dense",
            )
            .astype(int)
        )

    return resultado


# ==============================================================================
# VALIDACIÓN FINAL
# ==============================================================================

def validar_resultado(
    gdf_original: gpd.GeoDataFrame,
    gdf_optimizado: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> Dict:

    encabezado(
        "7. VALIDACIÓN FINAL"
    )

    proyectos_originales = set(
        gdf_original[
            proyecto_col
        ]
        .astype(str)
    )

    proyectos_optimizados = set(
        gdf_optimizado[
            proyecto_col
        ]
        .astype(str)
    )

    escenarios_originales = set(
        gdf_original[
            escenario_col
        ]
        .astype(str)
    )

    escenarios_optimizados = set(
        gdf_optimizado[
            escenario_col
        ]
        .astype(str)
    )

    conteos = (
        gdf_optimizado.groupby(
            escenario_col
        )[proyecto_col]
        .count()
    )

    validacion = {
        "proyectos_originales": int(
            len(
                gdf_original
            )
        ),
        "proyectos_optimizados": int(
            len(
                gdf_optimizado
            )
        ),
        "mismos_proyectos": (
            proyectos_originales
            == proyectos_optimizados
        ),
        "mismos_escenarios": (
            escenarios_originales
            == escenarios_optimizados
        ),
        "duplicados_proyecto": int(
            gdf_optimizado[
                proyecto_col
            ].duplicated().sum()
        ),
        "proyectos_sin_escenario": int(
            gdf_optimizado[
                escenario_col
            ].isna().sum()
        ),
        "escenarios": int(
            gdf_optimizado[
                escenario_col
            ].nunique()
        ),
        "min_proyectos": int(
            conteos.min()
        ),
        "max_proyectos": int(
            conteos.max()
        ),
    }

    for clave, valor in validacion.items():
        print(
            f"{clave:<28}: {valor}"
        )

    if not validacion[
        "mismos_proyectos"
    ]:
        raise ValueError(
            "El conjunto de proyectos fue alterado."
        )

    if not validacion[
        "mismos_escenarios"
    ]:
        raise ValueError(
            "El conjunto de escenarios fue alterado."
        )

    if (
        validacion[
            "duplicados_proyecto"
        ]
        > 0
    ):
        raise ValueError(
            "Aparecieron proyectos duplicados."
        )

    if (
        validacion[
            "proyectos_sin_escenario"
        ]
        > 0
    ):
        raise ValueError(
            "Existen proyectos sin escenario."
        )

    if (
        validacion[
            "min_proyectos"
        ]
        < MIN_PROYECTOS
    ):
        raise ValueError(
            "Existe un escenario por debajo "
            "del mínimo permitido."
        )

    if not (
        MIN_ESCENARIOS
        <= validacion[
            "escenarios"
        ]
        <= MAX_ESCENARIOS
    ):
        raise ValueError(
            "Cantidad de escenarios fuera del rango permitido."
        )

    print()
    print(
        "Validación final: OK"
    )

    return validacion


# ==============================================================================
# EXPORTACIÓN
# ==============================================================================

def exportar_resultados(
    gdf_optimizado: gpd.GeoDataFrame,
    evaluacion_detallada: pd.DataFrame,
    movimientos: List[Dict],
    metricas_originales: Dict,
    metricas_finales: Dict,
    validacion: Dict,
    tiempo: float,
    escenario_col: str,
) -> None:

    encabezado(
        "8. EXPORTANDO RESULTADOS"
    )

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gdf_export = gdf_optimizado.copy()

    columna_temporal = (
        "__escenario_original_v4"
    )

    if columna_temporal in gdf_export.columns:
        gdf_export = gdf_export.drop(
            columns=[
                columna_temporal
            ]
        )

    gdf_export.to_parquet(
        OUTPUT_OPTIMIZADO,
        index=False,
    )

    csv_export = pd.DataFrame(
        gdf_export.drop(
            columns=["geometry"],
            errors="ignore",
        )
    )

    csv_export.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    evaluacion_detallada.to_csv(
        OUTPUT_EVALUACION,
        index=False,
        encoding="utf-8-sig",
    )

    movimientos_df = pd.DataFrame(
        movimientos
    )

    if movimientos_df.empty:

        movimientos_df = pd.DataFrame(
            columns=[
                "iteracion",
                "proyecto_id",
                "escenario_origen",
                "escenario_destino",
                "mejora_espacial_metros",
                "score_antes",
                "score_despues",
                "mejora_score",
            ]
        )

    movimientos_df.to_csv(
        OUTPUT_MOVIMIENTOS,
        index=False,
        encoding="utf-8-sig",
    )

    claves = [
        "cohesion",
        "balance",
        "indicadores",
        "score_territorial",
        "estabilidad",
        "score_objetivo",
    ]

    filas_resumen = []

    for clave in claves:

        filas_resumen.append(
            {
                "metrica": clave,
                "original": (
                    metricas_originales[
                        clave
                    ]
                ),
                "optimizado": (
                    metricas_finales[
                        clave
                    ]
                ),
                "cambio": (
                    metricas_finales[
                        clave
                    ]
                    -
                    metricas_originales[
                        clave
                    ]
                ),
            }
        )

    resumen = pd.DataFrame(
        filas_resumen
    )

    resumen.to_csv(
        OUTPUT_RESUMEN,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "proceso": 29,
        "nombre": (
            "Optimización de escenarios "
            "territoriales AMBA"
        ),
        "version": VERSION,
        "fecha_ejecucion": (
            pd.Timestamp.now(
                tz="America/Argentina/Buenos_Aires"
            ).isoformat()
        ),
        "proyectos": int(
            len(gdf_optimizado)
        ),
        "escenarios": int(
            gdf_optimizado[
                escenario_col
            ].nunique()
        ),
        "movimientos": int(
            len(movimientos)
        ),
        "configuracion": {
            "min_proyectos": MIN_PROYECTOS,
            "min_escenarios": MIN_ESCENARIOS,
            "max_escenarios": MAX_ESCENARIOS,
            "max_iteraciones": MAX_ITERACIONES,
            "mejora_minima": MEJORA_MINIMA,
            "penalizacion_movimiento": (
                PENALIZACION_MOVIMIENTO
            ),
            "radio_cohesion_metros": (
                RADIO_COHESION_METROS
            ),
        },
        "pesos": {
            "cohesion": PESO_COHESION,
            "balance": PESO_BALANCE,
            "indicadores": PESO_INDICADORES,
            "score_territorial": (
                PESO_SCORE_TERRITORIAL
            ),
            "estabilidad": PESO_ESTABILIDAD,
        },
        "metricas_originales": (
            metricas_originales
        ),
        "metricas_optimizadas": (
            metricas_finales
        ),
        "mejora_score_objetivo": (
            metricas_finales[
                "score_objetivo"
            ]
            -
            metricas_originales[
                "score_objetivo"
            ]
        ),
        "validacion": validacion,
        "duracion_segundos": tiempo,
    }

    OUTPUT_METADATA.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"GeoParquet : {OUTPUT_OPTIMIZADO}"
    )

    print(
        f"CSV        : {OUTPUT_CSV}"
    )

    print(
        f"Evaluación : {OUTPUT_EVALUACION}"
    )

    print(
        f"Movimientos: {OUTPUT_MOVIMIENTOS}"
    )

    print(
        f"Resumen    : {OUTPUT_RESUMEN}"
    )

    print(
        f"Metadata   : {OUTPUT_METADATA}"
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:

    inicio = time.perf_counter()

    encabezado(
        f"29 - OPTIMIZACIÓN DE ESCENARIOS "
        f"TERRITORIALES AMBA - {VERSION}"
    )

    print(
        f"Proyecto : {PROJECT_ROOT}"
    )

    print(
        f"Entrada  : {INPUT_ESCENARIOS}"
    )

    print(
        f"Salida   : {INPUT_DIR}"
    )

    subtitulo(
        "Configuración"
    )

    print(
        f"Mínimo proyectos          : "
        f"{MIN_PROYECTOS}"
    )

    print(
        f"Rango escenarios          : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
    )

    print(
        f"Máximo iteraciones        : "
        f"{MAX_ITERACIONES}"
    )

    print(
        f"Mejora mínima             : "
        f"{MEJORA_MINIMA}"
    )

    print(
        f"Penalización movimiento   : "
        f"{PENALIZACION_MOVIMIENTO}"
    )

    print(
        f"Radio cohesión            : "
        f"{RADIO_COHESION_METROS:,.0f} m"
    )

    print()
    print(
        "PESOS DE OPTIMIZACIÓN"
    )

    print(
        f"  Cohesión territorial    : "
        f"{PESO_COHESION:.0%}"
    )

    print(
        f"  Balance                 : "
        f"{PESO_BALANCE:.0%}"
    )

    print(
        f"  Indicadores             : "
        f"{PESO_INDICADORES:.0%}"
    )

    print(
        f"  Score territorial       : "
        f"{PESO_SCORE_TERRITORIAL:.0%}"
    )

    print(
        f"  Estabilidad             : "
        f"{PESO_ESTABILIDAD:.0%}"
    )

    print(
        f"  TOTAL                   : "
        f"{SUMA_PESOS:.0%}"
    )

    print()
    print(
        "RESTRICCIONES"
    )

    print(
        "  Cobertura               : 100%"
    )

    print(
        f"  Escenarios              : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
    )

    print(
        f"  Mínimo proyectos        : "
        f"{MIN_PROYECTOS}"
    )

    gdf_original = cargar_escenarios()

    evaluacion_proceso_28 = (
        cargar_evaluacion()
    )

    recomendaciones = (
        cargar_recomendaciones()
    )

    (
        escenario_col,
        proyecto_col,
    ) = validar_entrada(
        gdf_original
    )

    indicadores = detectar_indicadores(
        gdf_original,
        escenario_col,
        proyecto_col,
    )

    score_col = detectar_score_territorial(
        gdf_original
    )

    print()
    print(
        "Score territorial utilizado: "
        f"{score_col}"
    )

    print(
        "Evaluación del proceso 28: "
        + (
            "DISPONIBLE"
            if evaluacion_proceso_28 is not None
            else "NO DISPONIBLE"
        )
    )

    print(
        "Recomendaciones proceso 28: "
        + (
            "DISPONIBLES"
            if recomendaciones is not None
            else "NO DISPONIBLES"
        )
    )

    gdf_metric_original = (
        preparar_geometria_metrica(
            gdf_original
        )
    )

    encabezado(
        "4. EVALUACIÓN BASE"
    )

    gdf_base = gdf_original.copy()

    gdf_base[
        "__escenario_original_v4"
    ] = gdf_base[
        escenario_col
    ].astype(str)

    metricas_originales = evaluar_estado(
        gdf_base,
        gdf_metric_original,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
        "__escenario_original_v4",
    )

    for clave, valor in metricas_originales.items():

        print(
            f"{clave:<25}: "
            f"{valor:.6f}"
        )

    (
        gdf_optimizado,
        movimientos,
        _,
        metricas_finales,
    ) = optimizar(
        gdf_original,
        escenario_col,
        proyecto_col,
        indicadores,
        score_col,
    )

    gdf_metric_final = (
        preparar_geometria_metrica(
            gdf_optimizado
        )
    )

    evaluacion_detallada = (
        evaluar_escenarios_detalladamente(
            gdf_optimizado,
            escenario_col,
            proyecto_col,
            indicadores,
            score_col,
            gdf_metric_final,
        )
    )

    validacion = validar_resultado(
        gdf_original,
        gdf_optimizado,
        escenario_col,
        proyecto_col,
    )

    tiempo = (
        time.perf_counter()
        - inicio
    )

    exportar_resultados(
        gdf_optimizado,
        evaluacion_detallada,
        movimientos,
        metricas_originales,
        metricas_finales,
        validacion,
        tiempo,
        escenario_col,
    )

    encabezado(
        "9. PROCESO 29 FINALIZADO CORRECTAMENTE"
    )

    print(
        f"Proyectos procesados     : "
        f"{len(gdf_optimizado)}"
    )

    print(
        f"Escenarios               : "
        f"{gdf_optimizado[escenario_col].nunique()}"
    )

    print(
        f"Movimientos realizados   : "
        f"{len(movimientos)}"
    )

    print()
    print(
        "FUNCIÓN OBJETIVO"
    )

    print(
        f"Score original           : "
        f"{metricas_originales['score_objetivo']:.6f}"
    )

    print(
        f"Score optimizado         : "
        f"{metricas_finales['score_objetivo']:.6f}"
    )

    mejora = (
        metricas_finales[
            "score_objetivo"
        ]
        -
        metricas_originales[
            "score_objetivo"
        ]
    )

    print(
        f"Mejora global            : "
        f"{mejora:+.8f}"
    )

    print()
    print(
        "COMPONENTES"
    )

    componentes = [
        "cohesion",
        "balance",
        "indicadores",
        "score_territorial",
        "estabilidad",
    ]

    for componente in componentes:

        antes = metricas_originales[
            componente
        ]

        despues = metricas_finales[
            componente
        ]

        print(
            f"{componente:<25}: "
            f"{antes:.6f} -> "
            f"{despues:.6f} "
            f"({despues - antes:+.6f})"
        )

    print()
    print(
        "DISTRIBUCIÓN FINAL DE ESCENARIOS"
    )

    conteos = (
        gdf_optimizado.groupby(
            escenario_col
        )[proyecto_col]
        .count()
        .sort_values(
            ascending=False
        )
    )

    print(
        conteos.to_string()
    )

    print()
    print(
        "CONTROL DE OPTIMIZACIÓN"
    )

    if mejora > MEJORA_MINIMA:

        print(
            "Resultado: MEJORA REAL"
        )

    elif abs(mejora) <= MEJORA_MINIMA:

        print(
            "Resultado: SIN CAMBIO SIGNIFICATIVO"
        )

    else:

        print(
            "Resultado: ADVERTENCIA - "
            "EL SCORE EMPEORÓ"
        )

    print()
    print(
        "Salida principal:"
    )

    print(
        OUTPUT_OPTIMIZADO
    )

    print()
    print(
        f"Duración                 : "
        f"{tiempo:.2f} segundos"
    )

    print()
    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()