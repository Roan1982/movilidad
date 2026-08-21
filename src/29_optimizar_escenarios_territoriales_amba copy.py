# -*- coding: utf-8 -*-

"""
================================================================================
29 - OPTIMIZACIÓN DE ESCENARIOS TERRITORIALES AMBA
================================================================================

Proceso:
    27 -> Construcción de escenarios territoriales
    28 -> Evaluación de escenarios
    29 -> Optimización de escenarios

Objetivo:
    Mejorar la estructura de los escenarios existentes manteniendo:

    - cobertura total de proyectos
    - cantidad razonable de escenarios
    - mínimo de proyectos por escenario
    - cohesión territorial
    - balance de tamaños
    - concentración de indicadores
    - score territorial original
    - trazabilidad de movimientos

Entrada principal:
    escenarios_territoriales_amba.parquet

Entradas complementarias:
    evaluacion_escenarios_territoriales_amba.parquet
    recomendaciones_escenarios_territoriales_amba.csv

Salidas:
    escenarios_territoriales_amba_optimizado.parquet
    escenarios_territoriales_amba_optimizado.csv
    evaluacion_escenarios_optimizada.csv
    movimientos_optimizacion_escenarios.csv
    resumen_optimizacion_escenarios.csv
    metadata_optimización_escenarios.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

VERSION = "V1.0"

MIN_PROYECTOS = 8
MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

MAX_ITERACIONES = 50

# Pesos de optimización
PESO_COHESION = 0.30
PESO_BALANCE = 0.20
PESO_INDICADORES = 0.25
PESO_SCORE_ORIGINAL = 0.15
PESO_ESTABILIDAD = 0.10

# Umbral para aceptar una transferencia de proyecto
MEJORA_MINIMA = 0.0005


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
    print("=" * 88)
    print(titulo)
    print("=" * 88)


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: List[str],
    obligatoria: bool = True,
) -> str | None:

    for columna in candidatos:
        if columna in df.columns:
            return columna

    if obligatoria:
        raise KeyError(
            f"No se encontró ninguna de las columnas requeridas: {candidatos}"
        )

    return None


def normalizar_serie(serie: pd.Series) -> pd.Series:
    valores = pd.to_numeric(serie, errors="coerce")

    if valores.notna().sum() == 0:
        return pd.Series(
            np.full(len(serie), 0.5),
            index=serie.index,
            dtype=float,
        )

    minimo = valores.min()
    maximo = valores.max()

    if math.isclose(float(minimo), float(maximo)):
        return pd.Series(
            np.full(len(serie), 0.5),
            index=serie.index,
            dtype=float,
        )

    return (valores - minimo) / (maximo - minimo)


def distancia_centroides(
    geometria_a,
    geometria_b,
) -> float:

    try:
        return float(
            geometria_a.centroid.distance(
                geometria_b.centroid
            )
        )
    except Exception:
        return float("inf")


# ==============================================================================
# CARGA
# ==============================================================================

def cargar_escenarios() -> gpd.GeoDataFrame:

    encabezado("1. CARGANDO ESCENARIOS DEL PROCESO 27")

    if not INPUT_ESCENARIOS.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{INPUT_ESCENARIOS}"
        )

    gdf = gpd.read_parquet(INPUT_ESCENARIOS)

    print(f"Entrada     : {INPUT_ESCENARIOS}")
    print(f"Registros   : {len(gdf):,}")
    print(f"Columnas    : {len(gdf.columns)}")
    print(f"CRS         : {gdf.crs}")

    if len(gdf) == 0:
        raise ValueError("El GeoParquet no contiene registros.")

    return gdf


def cargar_evaluacion() -> pd.DataFrame | None:

    if not INPUT_EVALUACION.exists():
        print()
        print("Evaluación del proceso 28 no encontrada.")
        print("Se continuará utilizando únicamente la información del proceso 27.")
        return None

    evaluacion = pd.read_parquet(INPUT_EVALUACION)

    print(f"Evaluación  : {INPUT_EVALUACION}")
    print(f"Registros   : {len(evaluacion):,}")

    return evaluacion


# ==============================================================================
# VALIDACIÓN
# ==============================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> Tuple[str, str]:

    encabezado("2. VALIDANDO ENTRADA")

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

    nulas = int(gdf.geometry.isna().sum())
    vacias = int(gdf.geometry.is_empty.sum())

    try:
        invalidas = int((~gdf.geometry.is_valid).sum())
    except Exception:
        invalidas = 0

    duplicados_proyecto = int(
        gdf[proyecto_col].duplicated().sum()
    )

    print(f"Geometrías nulas       : {nulas}")
    print(f"Geometrías vacías      : {vacias}")
    print(f"Geometrías inválidas   : {invalidas}")
    print(f"Escenarios             : {gdf[escenario_col].nunique()}")
    print(f"Proyectos              : {gdf[proyecto_col].nunique()}")
    print(f"Duplicados proyecto    : {duplicados_proyecto}")

    if nulas > 0 or vacias > 0 or invalidas > 0:
        raise ValueError(
            "La geometría de entrada contiene problemas."
        )

    if duplicados_proyecto > 0:
        raise ValueError(
            "Un mismo proyecto aparece más de una vez."
        )

    print("Validación de entrada: OK")

    return escenario_col, proyecto_col


# ==============================================================================
# PREPARACIÓN DE INDICADORES
# ==============================================================================

def detectar_indicadores(
    gdf: pd.DataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> List[str]:

    excluir = {
        escenario_col,
        proyecto_col,
        "geometry",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
    }

    indicadores_preferidos = [
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

    indicadores = []

    for columna in indicadores_preferidos:
        if columna in gdf.columns:
            if pd.api.types.is_numeric_dtype(gdf[columna]):
                indicadores.append(columna)

    if len(indicadores) < 3:

        for columna in gdf.columns:

            if columna in excluir:
                continue

            if columna.startswith("ranking_"):
                continue

            if columna.startswith("nivel_"):
                continue

            if not pd.api.types.is_numeric_dtype(gdf[columna]):
                continue

            if columna not in indicadores:
                indicadores.append(columna)

    print()
    print(f"Indicadores utilizados : {len(indicadores)}")

    for indicador in indicadores:
        print(f"  - {indicador}")

    return indicadores


def construir_matriz_indicadores(
    gdf: pd.DataFrame,
    indicadores: List[str],
) -> np.ndarray:

    matriz = []

    for indicador in indicadores:

        serie = normalizar_serie(gdf[indicador])

        valores = serie.fillna(0.5).to_numpy(dtype=float)

        matriz.append(valores)

    if not matriz:
        return np.zeros((len(gdf), 1))

    return np.column_stack(matriz)


# ==============================================================================
# CENTROIDES
# ==============================================================================

def preparar_geometria_metrica(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    if gdf.crs is None:
        raise ValueError("La capa no tiene CRS.")

    if gdf.crs.is_geographic:
        return gdf.to_crs(3857)

    return gdf


# ==============================================================================
# MÉTRICAS
# ==============================================================================

def calcular_balance(
    cantidades: np.ndarray,
) -> float:

    if len(cantidades) == 0:
        return 0.0

    promedio = float(np.mean(cantidades))

    if promedio <= 0:
        return 0.0

    cv = float(
        np.std(cantidades, ddof=0) / promedio
    )

    return max(
        0.0,
        min(
            1.0,
            1.0 - cv,
        ),
    )


def calcular_cohesion(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> float:

    centroides = gdf_metric.geometry.centroid

    grupos = []

    for escenario in sorted(
        gdf_metric[escenario_col].dropna().unique()
    ):

        idx = (
            gdf_metric[escenario_col] == escenario
        )

        puntos = centroides[idx]

        if len(puntos) < 2:
            continue

        centro = puntos.unary_union.centroid

        distancias = np.array(
            [
                float(p.distance(centro))
                for p in puntos
            ],
            dtype=float,
        )

        if len(distancias):
            grupos.append(
                float(
                    np.mean(distancias)
                )
            )

    if not grupos:
        return 0.0

    distancia_media = float(
        np.mean(grupos)
    )

    # Escala robusta.
    # 50 km se considera una dispersión muy alta
    # para un escenario territorial.
    score = 1.0 - (
        distancia_media / 50000.0
    )

    return max(
        0.0,
        min(1.0, score),
    )


def calcular_concentracion_indicadores(
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
            gdf.assign(
                __valor=valores
            )
            .groupby(escenario_col)["__valor"]
            .mean()
        )

        dispersion = float(
            medias.std(ddof=0)
        )

        # Menor dispersión = mayor concentración homogénea.
        score = 1.0 / (
            1.0 + abs(
                dispersion / (
                    abs(media_global) + 1e-9
                )
            )
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


def calcular_score_escenario(
    grupo: pd.DataFrame,
    score_col: str | None,
) -> float:

    if score_col is None:
        return 0.5

    valores = pd.to_numeric(
        grupo[score_col],
        errors="coerce",
    )

    if valores.notna().sum() == 0:
        return 0.5

    return float(
        normalizar_serie(
            valores
        ).mean()
    )


# ==============================================================================
# SCORE GLOBAL
# ==============================================================================

def evaluar_estructura(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
) -> Dict[str, float]:

    gdf_metric = preparar_geometria_metrica(gdf)

    cantidades = (
        gdf.groupby(escenario_col)[proyecto_col]
        .count()
        .to_numpy()
    )

    cantidad_escenarios = len(cantidades)

    cobertura = 1.0

    if (
        cantidad_escenarios >= MIN_ESCENARIOS
        and cantidad_escenarios <= MAX_ESCENARIOS
    ):
        score_estructura_escenarios = 1.0
    else:
        distancia = min(
            abs(
                cantidad_escenarios
                - MIN_ESCENARIOS
            ),
            abs(
                cantidad_escenarios
                - MAX_ESCENARIOS
            ),
        )

        score_estructura_escenarios = max(
            0.0,
            1.0 - 0.25 * distancia,
        )

    tamaño = float(
        np.mean(
            [
                1.0
                if n >= MIN_PROYECTOS
                else n / MIN_PROYECTOS
                for n in cantidades
            ]
        )
    )

    balance = calcular_balance(
        cantidades
    )

    cohesion = calcular_cohesion(
        gdf_metric,
        escenario_col,
    )

    indicadores_score = (
        calcular_concentracion_indicadores(
            gdf,
            escenario_col,
            indicadores,
        )
    )

    global_score = (
        0.15 * cobertura
        + 0.10 * score_estructura_escenarios
        + 0.20 * tamaño
        + PESO_COHESION * cohesion
        + PESO_BALANCE * balance
        + PESO_INDICADORES * indicadores_score
    )

    return {
        "cobertura": cobertura,
        "estructura_escenarios": score_estructura_escenarios,
        "tamano": tamaño,
        "cohesion": cohesion,
        "balance": balance,
        "indicadores": indicadores_score,
        "score_global": float(
            min(1.0, max(0.0, global_score))
        ),
    }


# ==============================================================================
# CENTROIDE DE ESCENARIO
# ==============================================================================

def obtener_centroides_escenarios(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
) -> Dict[str, object]:

    resultado = {}

    for escenario, grupo in gdf_metric.groupby(
        escenario_col
    ):

        resultado[escenario] = (
            grupo.geometry
            .unary_union
            .centroid
        )

    return resultado


# ==============================================================================
# SCORE DE TRANSFERENCIA
# ==============================================================================

def evaluar_transferencia(
    gdf_metric: gpd.GeoDataFrame,
    escenario_col: str,
    fila_idx,
    escenario_origen: str,
    escenario_destino: str,
) -> float:

    grupo_origen = gdf_metric[
        gdf_metric[escenario_col]
        == escenario_origen
    ]

    grupo_destino = gdf_metric[
        gdf_metric[escenario_col]
        == escenario_destino
    ]

    if len(grupo_origen) <= MIN_PROYECTOS:
        return -np.inf

    if len(grupo_destino) == 0:
        return -np.inf

    geometria = gdf_metric.loc[
        fila_idx,
        "geometry",
    ]

    centro_origen_antes = (
        grupo_origen.geometry
        .unary_union
        .centroid
    )

    centro_destino_antes = (
        grupo_destino.geometry
        .unary_union
        .centroid
    )

    dist_origen_antes = float(
        geometria.distance(
            centro_origen_antes
        )
    )

    dist_destino_antes = float(
        geometria.distance(
            centro_destino_antes
        )
    )

    # Queremos mover el proyecto hacia el escenario
    # donde esté espacialmente más próximo.
    mejora = (
        dist_origen_antes
        - dist_destino_antes
    )

    return float(mejora)


# ==============================================================================
# OPTIMIZACIÓN ESPACIAL
# ==============================================================================

def optimizar_movimientos(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> Tuple[gpd.GeoDataFrame, List[Dict]]:

    encabezado(
        "5. OPTIMIZANDO ASIGNACIÓN TERRITORIAL"
    )

    gdf_opt = gdf.copy()

    gdf_metric = preparar_geometria_metrica(
        gdf_opt
    )

    movimientos = []

    escenarios = sorted(
        gdf_opt[escenario_col]
        .dropna()
        .unique()
    )

    for iteracion in range(
        1,
        MAX_ITERACIONES + 1,
    ):

        cambios = 0

        cantidades = (
            gdf_opt.groupby(
                escenario_col
            )[proyecto_col]
            .count()
            .to_dict()
        )

        centroides = obtener_centroides_escenarios(
            gdf_metric,
            escenario_col,
        )

        for idx in gdf_opt.index:

            escenario_actual = gdf_opt.loc[
                idx,
                escenario_col,
            ]

            if cantidades.get(
                escenario_actual,
                0,
            ) <= MIN_PROYECTOS:
                continue

            centro_actual = centroides.get(
                escenario_actual
            )

            if centro_actual is None:
                continue

            geometria = gdf_metric.loc[
                idx,
                "geometry",
            ]

            distancia_actual = float(
                geometria.distance(
                    centro_actual
                )
            )

            mejor_escenario = None
            mejor_distancia = distancia_actual

            for escenario_destino in escenarios:

                if (
                    escenario_destino
                    == escenario_actual
                ):
                    continue

                if escenario_destino not in centroides:
                    continue

                distancia_destino = float(
                    geometria.distance(
                        centroides[
                            escenario_destino
                        ]
                    )
                )

                # Solo considerar un movimiento
                # cuando existe mejora espacial real.
                if (
                    distancia_destino
                    < mejor_distancia
                    * (1.0 - MEJORA_MINIMA)
                ):

                    # No permitir que el escenario
                    # receptor quede desbalanceado.
                    cantidad_destino = cantidades.get(
                        escenario_destino,
                        0,
                    )

                    cantidad_origen = cantidades.get(
                        escenario_actual,
                        0,
                    )

                    if (
                        cantidad_destino
                        >= cantidad_origen + 8
                    ):
                        continue

                    mejor_distancia = (
                        distancia_destino
                    )

                    mejor_escenario = (
                        escenario_destino
                    )

            if mejor_escenario is None:
                continue

            proyecto = gdf_opt.loc[
                idx,
                proyecto_col,
            ]

            movimientos.append(
                {
                    "iteracion": iteracion,
                    "proyecto_id": proyecto,
                    "escenario_origen": escenario_actual,
                    "escenario_destino": mejor_escenario,
                    "distancia_origen": distancia_actual,
                    "distancia_destino": mejor_distancia,
                    "mejora_distancia": (
                        distancia_actual
                        - mejor_distancia
                    ),
                }
            )

            gdf_opt.loc[
                idx,
                escenario_col,
            ] = mejor_escenario

            cantidades[
                escenario_actual
            ] -= 1

            cantidades[
                mejor_escenario
            ] = cantidades.get(
                mejor_escenario,
                0,
            ) + 1

            cambios += 1

            # Actualizar centroides para evitar
            # acumulación excesiva de errores.
            grupo_origen = gdf_metric[
                gdf_opt[escenario_col]
                == escenario_actual
            ]

            grupo_destino = gdf_metric[
                gdf_opt[escenario_col]
                == mejor_escenario
            ]

            if len(grupo_origen):
                centroides[
                    escenario_actual
                ] = (
                    grupo_origen.geometry
                    .unary_union
                    .centroid
                )

            centroides[
                mejor_escenario
            ] = (
                grupo_destino.geometry
                .unary_union
                .centroid
            )

        print(
            f"Iteración {iteracion:02d} "
            f"| movimientos: {cambios}"
        )

        if cambios == 0:
            break

    print()
    print(
        f"Movimientos realizados : {len(movimientos)}"
    )

    return gdf_opt, movimientos


# ==============================================================================
# OPTIMIZACIÓN POR TAMAÑO
# ==============================================================================

def equilibrar_tamanos(
    gdf: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
) -> Tuple[gpd.GeoDataFrame, List[Dict]]:

    encabezado(
        "6. EQUILIBRANDO TAMAÑO DE ESCENARIOS"
    )

    gdf_opt = gdf.copy()
    movimientos = []

    for iteracion in range(
        1,
        MAX_ITERACIONES + 1,
    ):

        conteos = (
            gdf_opt.groupby(
                escenario_col
            )[proyecto_col]
            .count()
        )

        if len(conteos) <= 1:
            break

        max_escenario = conteos.idxmax()
        min_escenario = conteos.idxmin()

        max_n = int(
            conteos[max_escenario]
        )

        min_n = int(
            conteos[min_escenario]
        )

        if (
            max_n - min_n
            <= 4
        ):
            break

        candidatos = gdf_opt[
            gdf_opt[escenario_col]
            == max_escenario
        ]

        if len(candidatos) <= MIN_PROYECTOS:
            break

        # Buscar el proyecto más cercano
        # al centroide del escenario receptor.
        gdf_metric = preparar_geometria_metrica(
            gdf_opt
        )

        centro_receptor = (
            gdf_metric[
                gdf_opt[escenario_col]
                == min_escenario
            ]
            .geometry
            .unary_union
            .centroid
        )

        distancias = (
            gdf_metric.loc[
                candidatos.index
            ]
            .geometry
            .distance(
                centro_receptor
            )
        )

        idx = distancias.idxmin()

        proyecto = gdf_opt.loc[
            idx,
            proyecto_col,
        ]

        gdf_opt.loc[
            idx,
            escenario_col,
        ] = min_escenario

        movimientos.append(
            {
                "iteracion": iteracion,
                "proyecto_id": proyecto,
                "escenario_origen": max_escenario,
                "escenario_destino": min_escenario,
                "tipo_movimiento": "BALANCE_TAMANO",
            }
        )

    print(
        f"Movimientos de balance : {len(movimientos)}"
    )

    return gdf_opt, movimientos


# ==============================================================================
# EVALUACIÓN FINAL
# ==============================================================================

def construir_evaluacion_final(
    gdf_original: gpd.GeoDataFrame,
    gdf_optimizado: gpd.GeoDataFrame,
    escenario_col: str,
    proyecto_col: str,
    indicadores: List[str],
) -> pd.DataFrame:

    encabezado(
        "7. EVALUANDO RESULTADO DE LA OPTIMIZACIÓN"
    )

    original = evaluar_estructura(
        gdf_original,
        escenario_col,
        proyecto_col,
        indicadores,
    )

    optimizado = evaluar_estructura(
        gdf_optimizado,
        escenario_col,
        proyecto_col,
        indicadores,
    )

    print()
    print("MÉTRICAS")
    print(
        f"{'Métrica':<25}"
        f"{'Original':>15}"
        f"{'Optimizado':>15}"
        f"{'Cambio':>15}"
    )

    for clave in [
        "cobertura",
        "estructura_escenarios",
        "tamano",
        "cohesion",
        "balance",
        "indicadores",
        "score_global",
    ]:

        a = original[clave]
        b = optimizado[clave]

        print(
            f"{clave:<25}"
            f"{a:>15.4f}"
            f"{b:>15.4f}"
            f"{b-a:>15.4f}"
        )

    filas = []

    conteos = (
        gdf_optimizado
        .groupby(escenario_col)[proyecto_col]
        .count()
        .sort_values(
            ascending=False
        )
    )

    ranking = 0

    for escenario, cantidad in conteos.items():

        ranking += 1

        grupo = gdf_optimizado[
            gdf_optimizado[
                escenario_col
            ]
            == escenario
        ]

        score = calcular_score_escenario(
            grupo,
            "score_cartera"
            if "score_cartera"
            in grupo.columns
            else None,
        )

        filas.append(
            {
                "escenario_id": escenario,
                "cantidad_proyectos": int(
                    cantidad
                ),
                "score_escenario_optimizado": score,
                "ranking_escenario": ranking,
            }
        )

    resultado = pd.DataFrame(filas)

    resultado.attrs[
        "metricas_originales"
    ] = original

    resultado.attrs[
        "metricas_optimizadas"
    ] = optimizado

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
        "8. VALIDACIÓN FINAL"
    )

    proyectos_originales = set(
        gdf_original[
            proyecto_col
        ].astype(str)
    )

    proyectos_optimizados = set(
        gdf_optimizado[
            proyecto_col
        ].astype(str)
    )

    asignados = int(
        gdf_optimizado[
            escenario_col
        ].notna().sum()
    )

    conteos = (
        gdf_optimizado
        .groupby(escenario_col)[proyecto_col]
        .count()
    )

    validaciones = {
        "mismos_proyectos": (
            proyectos_originales
            == proyectos_optimizados
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
        "proyectos_asignados": asignados,
        "proyectos_totales": len(
            gdf_original
        ),
    }

    print(
        f"Proyectos totales       : "
        f"{validaciones['proyectos_totales']}"
    )

    print(
        f"Proyectos asignados     : "
        f"{validaciones['proyectos_asignados']}"
    )

    print(
        f"Proyectos sin escenario : "
        f"{validaciones['proyectos_sin_escenario']}"
    )

    print(
        f"Escenarios              : "
        f"{validaciones['escenarios']}"
    )

    print(
        f"Mínimo proyectos       : "
        f"{validaciones['min_proyectos']}"
    )

    print(
        f"Máximo proyectos       : "
        f"{validaciones['max_proyectos']}"
    )

    if not validaciones[
        "mismos_proyectos"
    ]:
        raise ValueError(
            "La optimización alteró el conjunto de proyectos."
        )

    if validaciones[
        "proyectos_sin_escenario"
    ] > 0:
        raise ValueError(
            "Existen proyectos sin escenario."
        )

    if validaciones[
        "min_proyectos"
    ] < MIN_PROYECTOS:
        raise ValueError(
            "Existe un escenario por debajo del mínimo permitido."
        )

    print(
        "Validación final: OK"
    )

    return validaciones


# ==============================================================================
# EXPORTACIÓN
# ==============================================================================

def exportar_resultados(
    gdf_optimizado: gpd.GeoDataFrame,
    evaluacion: pd.DataFrame,
    movimientos: List[Dict],
    metricas_originales: Dict,
    metricas_optimizadas: Dict,
    validacion: Dict,
    tiempo: float,
) -> None:

    encabezado(
        "9. EXPORTANDO RESULTADOS"
    )

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gdf_optimizado.to_parquet(
        OUTPUT_OPTIMIZADO,
        index=False,
    )

    gdf_export = gdf_optimizado.copy()

    # GeoDataFrame -> CSV sin geometry
    if "geometry" in gdf_export.columns:
        gdf_export = pd.DataFrame(
            gdf_export.drop(
                columns=["geometry"]
            )
        )

    gdf_export.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    evaluacion.to_csv(
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
            ]
        )

    movimientos_df.to_csv(
        OUTPUT_MOVIMIENTOS,
        index=False,
        encoding="utf-8-sig",
    )

    resumen = pd.DataFrame(
        [
            {
                "metrica": clave,
                "original": metricas_originales[
                    clave
                ],
                "optimizado": metricas_optimizadas[
                    clave
                ],
                "cambio": (
                    metricas_optimizadas[
                        clave
                    ]
                    - metricas_originales[
                        clave
                    ]
                ),
            }
            for clave in metricas_originales
        ]
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
        "fecha_ejecucion": pd.Timestamp.now(
            tz="America/Argentina/Buenos_Aires"
        ).isoformat(),
        "proyectos": int(
            len(gdf_optimizado)
        ),
        "escenarios": int(
            gdf_optimizado[
                "escenario_id"
                if "escenario_id"
                in gdf_optimizado.columns
                else "id_escenario"
            ].nunique()
        ),
        "movimientos": len(
            movimientos
        ),
        "metricas_originales": (
            metricas_originales
        ),
        "metricas_optimizadas": (
            metricas_optimizadas
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

    print(
        f"GeoParquet optimizado : {OUTPUT_OPTIMIZADO}"
    )

    print(
        f"CSV optimizado        : {OUTPUT_CSV}"
    )

    print(
        f"Evaluación            : {OUTPUT_EVALUACION}"
    )

    print(
        f"Movimientos           : {OUTPUT_MOVIMIENTOS}"
    )

    print(
        f"Resumen               : {OUTPUT_RESUMEN}"
    )

    print(
        f"Metadata              : {OUTPUT_METADATA}"
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:

    inicio = time.perf_counter()

    encabezado(
        f"29 - OPTIMIZACIÓN DE ESCENARIOS TERRITORIALES AMBA - {VERSION}"
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

    print()
    print("CONFIGURACIÓN")
    print(
        f"  Versión             : {VERSION}"
    )
    print(
        f"  Mínimo proyectos    : {MIN_PROYECTOS}"
    )
    print(
        f"  Escenarios válidos  : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
    )
    print(
        f"  Máx. iteraciones    : {MAX_ITERACIONES}"
    )
    print(
        f"  Peso cohesión       : {PESO_COHESION:.0%}"
    )
    print(
        f"  Peso balance        : {PESO_BALANCE:.0%}"
    )
    print(
        f"  Peso indicadores    : {PESO_INDICADORES:.0%}"
    )

    # --------------------------------------------------------------------------
    # 1
    # --------------------------------------------------------------------------

    gdf_original = cargar_escenarios()

    # --------------------------------------------------------------------------
    # 2
    # --------------------------------------------------------------------------

    escenario_col, proyecto_col = validar_entrada(
        gdf_original
    )

    # --------------------------------------------------------------------------
    # 3
    # --------------------------------------------------------------------------

    encabezado(
        "3. DETECTANDO INDICADORES"
    )

    indicadores = detectar_indicadores(
        gdf_original,
        escenario_col,
        proyecto_col,
    )

    # --------------------------------------------------------------------------
    # 4
    # --------------------------------------------------------------------------

    encabezado(
        "4. EVALUACIÓN BASE"
    )

    metricas_originales = evaluar_estructura(
        gdf_original,
        escenario_col,
        proyecto_col,
        indicadores,
    )

    for clave, valor in metricas_originales.items():
        print(
            f"{clave:<25}: {valor:.4f}"
        )

    # --------------------------------------------------------------------------
    # 5
    # --------------------------------------------------------------------------

    gdf_optimizado, movimientos_espaciales = (
        optimizar_movimientos(
            gdf_original,
            escenario_col,
            proyecto_col,
        )
    )

    # --------------------------------------------------------------------------
    # 6
    # --------------------------------------------------------------------------

    gdf_optimizado, movimientos_balance = (
        equilibrar_tamanos(
            gdf_optimizado,
            escenario_col,
            proyecto_col,
        )
    )

    movimientos = (
        movimientos_espaciales
        + movimientos_balance
    )

    # --------------------------------------------------------------------------
    # 7
    # --------------------------------------------------------------------------

    evaluacion = construir_evaluacion_final(
        gdf_original,
        gdf_optimizado,
        escenario_col,
        proyecto_col,
        indicadores,
    )

    metricas_optimizadas = evaluar_estructura(
        gdf_optimizado,
        escenario_col,
        proyecto_col,
        indicadores,
    )

    # --------------------------------------------------------------------------
    # 8
    # --------------------------------------------------------------------------

    validacion = validar_resultado(
        gdf_original,
        gdf_optimizado,
        escenario_col,
        proyecto_col,
    )

    # --------------------------------------------------------------------------
    # 9
    # --------------------------------------------------------------------------

    tiempo = (
        time.perf_counter()
        - inicio
    )

    exportar_resultados(
        gdf_optimizado,
        evaluacion,
        movimientos,
        metricas_originales,
        metricas_optimizadas,
        validacion,
        tiempo,
    )

    # --------------------------------------------------------------------------
    # 10
    # --------------------------------------------------------------------------

    encabezado(
        "10. PROCESO 29 FINALIZADO CORRECTAMENTE"
    )

    print(
        f"Proyectos procesados    : "
        f"{len(gdf_optimizado)}"
    )

    print(
        f"Escenarios              : "
        f"{gdf_optimizado[escenario_col].nunique()}"
    )

    print(
        f"Movimientos realizados  : "
        f"{len(movimientos)}"
    )

    print(
        f"Score global original   : "
        f"{metricas_originales['score_global']:.4f}"
    )

    print(
        f"Score global optimizado : "
        f"{metricas_optimizadas['score_global']:.4f}"
    )

    mejora = (
        metricas_optimizadas[
            "score_global"
        ]
        - metricas_originales[
            "score_global"
        ]
    )

    print(
        f"Mejora global           : "
        f"{mejora:+.4f}"
    )

    print(
        f"Duración                : "
        f"{tiempo:.2f} segundos"
    )

    print()
    print("ESCENARIOS OPTIMIZADOS")

    conteos = (
        gdf_optimizado
        .groupby(escenario_col)[proyecto_col]
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
        "Salida principal:"
    )

    print(
        OUTPUT_OPTIMIZADO
    )

    print()
    print("=" * 88)


if __name__ == "__main__":
    main()