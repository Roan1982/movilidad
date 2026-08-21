# -*- coding: utf-8 -*-

"""
30 - VALIDACIÓN FINAL DE ESCENARIOS TERRITORIALES AMBA - V3.0

Objetivo
--------
Auditoría final e independiente del resultado producido por el proceso 29.

El proceso 30 NO modifica escenarios.
NO reasigna proyectos.
NO optimiza.
NO corrige silenciosamente datos.

Su función es determinar si el resultado del proceso 29:

    1. mantiene integridad estructural,
    2. conserva la cobertura de proyectos,
    3. mantiene identificadores únicos,
    4. genera una cantidad válida de escenarios,
    5. mantiene escenarios con tamaño razonable,
    6. presenta cohesión territorial,
    7. mantiene indicadores estructurales completos,
    8. es consistente internamente,
    9. puede ser trazado contra procesos 27, 28 y 29,
    10. merece un dictamen final.

Entrada principal
-----------------
data/processed/escenarios_territoriales_amba/
    escenarios_territoriales_amba_optimizado.parquet

Salidas
-------
validacion_final_escenarios_territoriales_amba.parquet
validacion_final_escenarios_territoriales_amba.csv
detalle_validacion_escenarios_territoriales_amba.csv
comparacion_procesos_27_28_29_30.csv
alertas_validacion_escenarios_territoriales_amba.csv
resumen_validacion_escenarios_territoriales_amba.json
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V3.0"

MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

MIN_PROYECTOS_ESCENARIO = 8

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

DISTANCIA_COHESION_MAX_M = 50_000.0

SCORE_VALIDADO = 0.80
SCORE_OBSERVACIONES = 0.60

PESOS = {
    "cobertura": 0.25,
    "tamano": 0.15,
    "cohesion": 0.25,
    "indicadores": 0.15,
    "balance": 0.20,
}

INDICADORES_ESPERADOS = [
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


# =============================================================================
# RUTAS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_29 = BASE_DIR / "escenarios_territoriales_amba_optimizado.parquet"

INPUT_27 = BASE_DIR / "escenarios_territoriales_amba.parquet"

INPUT_28 = (
    BASE_DIR
    / "evaluacion_escenarios_territoriales_amba.parquet"
)

INPUT_29_EVALUACION = (
    BASE_DIR
    / "evaluacion_escenarios_optimizada.csv"
)

INPUT_29_RESUMEN = (
    BASE_DIR
    / "resumen_optimizacion_escenarios.csv"
)

OUTPUT_PARQUET = (
    BASE_DIR
    / "validacion_final_escenarios_territoriales_amba.parquet"
)

OUTPUT_CSV = (
    BASE_DIR
    / "validacion_final_escenarios_territoriales_amba.csv"
)

OUTPUT_DETALLE = (
    BASE_DIR
    / "detalle_validacion_escenarios_territoriales_amba.csv"
)

OUTPUT_COMPARACION = (
    BASE_DIR
    / "comparacion_procesos_27_28_29_30.csv"
)

OUTPUT_ALERTAS = (
    BASE_DIR
    / "alertas_validacion_escenarios_territoriales_amba.csv"
)

OUTPUT_JSON = (
    BASE_DIR
    / "resumen_validacion_escenarios_territoriales_amba.json"
)


# =============================================================================
# UTILIDADES
# =============================================================================

def print_header(title: str, char: str = "=") -> None:
    print()
    print(char * 88)
    print(title)
    print(char * 88)


def safe_float(value, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, safe_float(value)),
    )


def clean_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def first_existing(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for column in candidates:
        if column in df.columns:
            return column

    return None


def add_alert(
    alerts: list[dict],
    nivel: str,
    codigo: str,
    mensaje: str,
    escenario_id: str | None = None,
) -> None:

    alerts.append(
        {
            "nivel": nivel,
            "codigo": codigo,
            "escenario_id": escenario_id,
            "mensaje": mensaje,
        }
    )


def json_default(obj):

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, Path):
        return str(obj)

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)


# =============================================================================
# RESOLUCIÓN DE COLUMNAS
# =============================================================================

def resolve_columns(df: pd.DataFrame) -> dict:

    return {
        "proyecto": first_existing(
            df,
            [
                "proyecto_id",
                "id_proyecto",
                "project_id",
                "id",
            ],
        ),

        "escenario": first_existing(
            df,
            [
                "escenario_id",
                "cluster_territorial",
                "cluster_id",
            ],
        ),

        "tipo": first_existing(
            df,
            [
                "tipo_escenario",
                "tipo",
            ],
        ),

        "dimension": first_existing(
            df,
            [
                "dimension_dominante",
                "dimension",
            ],
        ),

        "prioridad": first_existing(
            df,
            [
                "prioridad_escenario",
                "prioridad",
            ],
        ),

        "score_escenario": first_existing(
            df,
            [
                "score_escenario",
                "score_cartera",
                "score_global",
            ],
        ),
    }


# =============================================================================
# CARGA
# =============================================================================

def load_input() -> gpd.GeoDataFrame:

    if not INPUT_29.exists():
        raise FileNotFoundError(
            "\n"
            "No existe el resultado del proceso 29.\n\n"
            f"Archivo esperado:\n{INPUT_29}\n"
        )

    print(f"Cargando: {INPUT_29}")

    gdf = gpd.read_parquet(INPUT_29)

    if gdf.empty:
        raise ValueError(
            "El archivo del proceso 29 existe pero no contiene registros."
        )

    if gdf.crs is None:
        warnings.warn(
            "La entrada no declara CRS. Se asumirá EPSG:4326.",
            RuntimeWarning,
        )

        gdf = gdf.set_crs(
            CRS_GEOGRAFICO,
            allow_override=True,
        )

    return gdf


# =============================================================================
# 1. ESTRUCTURA
# =============================================================================

def validate_structure(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("1. VALIDACIÓN DE ESTRUCTURA")

    geometry_null = int(gdf.geometry.isna().sum())

    geometry_empty = int(
        gdf.geometry.is_empty.sum()
    )

    geometry_invalid = int(
        (~gdf.geometry.is_valid).sum()
    )

    print(f"Registros              : {len(gdf)}")
    print(f"Columnas               : {len(gdf.columns)}")
    print(f"CRS                    : {gdf.crs}")
    print(f"Geometrías nulas       : {geometry_null}")
    print(f"Geometrías vacías      : {geometry_empty}")
    print(f"Geometrías inválidas   : {geometry_invalid}")

    if geometry_null:
        add_alert(
            alerts,
            "ERROR",
            "GEOM_NULL",
            f"Existen {geometry_null} geometrías nulas.",
        )

    if geometry_empty:
        add_alert(
            alerts,
            "ERROR",
            "GEOM_EMPTY",
            f"Existen {geometry_empty} geometrías vacías.",
        )

    if geometry_invalid:
        add_alert(
            alerts,
            "ERROR",
            "GEOM_INVALID",
            f"Existen {geometry_invalid} geometrías inválidas.",
        )

    if gdf.crs is None:
        add_alert(
            alerts,
            "ERROR",
            "CRS_MISSING",
            "La capa no tiene CRS.",
        )

    if cols["proyecto"] is None:
        add_alert(
            alerts,
            "ERROR",
            "PROJECT_ID_MISSING",
            "No se encontró identificador de proyecto.",
        )

    if cols["escenario"] is None:
        add_alert(
            alerts,
            "ERROR",
            "SCENARIO_ID_MISSING",
            "No se encontró identificador de escenario.",
        )

    return {
        "registros": len(gdf),
        "columnas": len(gdf.columns),
        "geometrias_nulas": geometry_null,
        "geometrias_vacias": geometry_empty,
        "geometrias_invalidas": geometry_invalid,
        "crs": str(gdf.crs),
    }


# =============================================================================
# 2. PROYECTOS
# =============================================================================

def validate_projects(
    gdf: pd.DataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("2. VALIDACIÓN DE PROYECTOS")

    project_col = cols["proyecto"]

    if project_col is None:

        return {
            "total": len(gdf),
            "sin_id": len(gdf),
            "duplicados": 0,
        }

    values = gdf[project_col].astype("string")

    missing = int(
        values.isna().sum()
        + (values.str.strip() == "").sum()
    )

    duplicated = int(
        values.dropna().duplicated().sum()
    )

    print(f"Campo proyecto         : {project_col}")
    print(f"Total registros        : {len(gdf)}")
    print(f"Sin ID                 : {missing}")
    print(f"IDs duplicados         : {duplicated}")

    if missing:
        add_alert(
            alerts,
            "ERROR",
            "PROJECT_ID_NULL",
            f"Existen {missing} proyectos sin identificador.",
        )

    if duplicated:
        add_alert(
            alerts,
            "ERROR",
            "PROJECT_ID_DUPLICATE",
            f"Existen {duplicated} IDs de proyecto duplicados.",
        )

    return {
        "total": len(gdf),
        "sin_id": missing,
        "duplicados": duplicated,
    }


# =============================================================================
# 3. ESCENARIOS
# =============================================================================

def validate_scenarios(
    gdf: pd.DataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("3. VALIDACIÓN DE ESCENARIOS")

    scenario_col = cols["escenario"]

    if scenario_col is None:

        return {
            "cantidad": 0,
            "nulos": len(gdf),
            "min_proyectos": 0,
            "max_proyectos": 0,
            "conteos": pd.Series(dtype=int),
        }

    scenario = (
        gdf[scenario_col]
        .astype("string")
        .str.strip()
    )

    nulls = int(
        scenario.isna().sum()
        + (scenario == "").sum()
    )

    valid = scenario[
        scenario.notna()
        & (scenario != "")
    ]

    counts = valid.value_counts().sort_index()

    n_scenarios = int(len(counts))

    minimum = int(counts.min()) if len(counts) else 0
    maximum = int(counts.max()) if len(counts) else 0

    print(f"Campo escenario        : {scenario_col}")
    print(f"Escenarios             : {n_scenarios}")
    print(f"Rango permitido        : {MIN_ESCENARIOS}-{MAX_ESCENARIOS}")
    print(f"Escenarios nulos       : {nulls}")
    print(f"Mínimo proyectos       : {minimum}")
    print(f"Máximo proyectos       : {maximum}")

    if not MIN_ESCENARIOS <= n_scenarios <= MAX_ESCENARIOS:

        add_alert(
            alerts,
            "ERROR",
            "SCENARIO_COUNT",
            (
                f"La cantidad de escenarios ({n_scenarios}) "
                f"está fuera del rango "
                f"{MIN_ESCENARIOS}-{MAX_ESCENARIOS}."
            ),
        )

    if nulls:

        add_alert(
            alerts,
            "ERROR",
            "SCENARIO_NULL",
            f"Existen {nulls} registros sin escenario.",
        )

    small = counts[
        counts < MIN_PROYECTOS_ESCENARIO
    ]

    for scenario_id, count in small.items():

        add_alert(
            alerts,
            "ERROR",
            "SCENARIO_TOO_SMALL",
            (
                f"El escenario contiene {count} proyectos; "
                f"mínimo requerido: "
                f"{MIN_PROYECTOS_ESCENARIO}."
            ),
            str(scenario_id),
        )

    return {
        "cantidad": n_scenarios,
        "nulos": nulls,
        "min_proyectos": minimum,
        "max_proyectos": maximum,
        "conteos": counts,
    }


# =============================================================================
# 4. COBERTURA
# =============================================================================

def validate_coverage(
    gdf: pd.DataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("4. VALIDACIÓN DE COBERTURA")

    project_col = cols["proyecto"]
    scenario_col = cols["escenario"]

    if project_col is None or scenario_col is None:

        add_alert(
            alerts,
            "ERROR",
            "COVERAGE_COLUMNS_MISSING",
            "No es posible calcular cobertura.",
        )

        return {
            "total": len(gdf),
            "asignados": 0,
            "sin_escenario": len(gdf),
            "cobertura": 0.0,
        }

    project_valid = (
        gdf[project_col]
        .astype("string")
        .str.strip()
        .notna()
        & (
            gdf[project_col]
            .astype("string")
            .str.strip()
            != ""
        )
    )

    scenario_valid = (
        gdf[scenario_col]
        .astype("string")
        .str.strip()
        .notna()
        & (
            gdf[scenario_col]
            .astype("string")
            .str.strip()
            != ""
        )
    )

    total = int(project_valid.sum())

    assigned = int(
        (project_valid & scenario_valid).sum()
    )

    missing = total - assigned

    coverage = (
        assigned / total
        if total > 0
        else 0.0
    )

    print(f"Proyectos válidos      : {total}")
    print(f"Proyectos asignados    : {assigned}")
    print(f"Sin escenario          : {missing}")
    print(f"Cobertura              : {coverage:.2%}")

    if missing:

        add_alert(
            alerts,
            "ERROR",
            "COVERAGE_INCOMPLETE",
            f"Existen {missing} proyectos sin escenario.",
        )

    return {
        "total": total,
        "asignados": assigned,
        "sin_escenario": missing,
        "cobertura": coverage,
    }


# =============================================================================
# 5. TAMAÑO
# =============================================================================

def calculate_size_score(
    counts: pd.Series,
) -> tuple[float, dict]:

    if counts.empty:
        return 0.0, {}

    mean = float(counts.mean())

    std = float(
        counts.std(ddof=0)
    )

    minimum_compliance = float(
        (
            counts >= MIN_PROYECTOS_ESCENARIO
        ).mean()
    )

    cv = (
        std / mean
        if mean > 0
        else 1.0
    )

    balance = clamp(
        1.0 - cv
    )

    score = clamp(
        0.70 * minimum_compliance
        + 0.30 * balance
    )

    return score, {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": mean,
        "std": std,
        "cv": cv,
        "cumplen_minimo": int(
            (
                counts >= MIN_PROYECTOS_ESCENARIO
            ).sum()
        ),
    }


def validate_size(
    counts: pd.Series,
) -> tuple[float, dict]:

    print_header("5. VALIDACIÓN DEL TAMAÑO")

    score, stats = calculate_size_score(counts)

    if not stats:

        print("No existen escenarios para evaluar.")

        return score, stats

    print(
        f"Mínimo                 : {stats['min']}"
    )

    print(
        f"Máximo                 : {stats['max']}"
    )

    print(
        f"Promedio               : {stats['mean']:.2f}"
    )

    print(
        f"Desvío                 : {stats['std']:.2f}"
    )

    print(
        f"CV                     : {stats['cv']:.4f}"
    )

    print(
        f"Cumplen mínimo         : "
        f"{stats['cumplen_minimo']}/{len(counts)}"
    )

    print(
        f"Score tamaño           : {score:.4f}"
    )

    return score, stats


# =============================================================================
# 6. COHESIÓN TERRITORIAL
# =============================================================================

def validate_cohesion(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> tuple[float, dict, pd.DataFrame]:

    print_header("6. VALIDACIÓN DE COHESIÓN TERRITORIAL")

    scenario_col = cols["escenario"]

    if scenario_col is None:
        return 0.0, {}, pd.DataFrame()

    if gdf.empty:
        return 0.0, {}, pd.DataFrame()

    work = gdf[
        [
            scenario_col,
            "geometry",
        ]
    ].copy()

    if work.crs is None:

        work = work.set_crs(
            CRS_GEOGRAFICO,
            allow_override=True,
        )

    try:

        metric = work.to_crs(
            CRS_METRICO
        )

    except Exception as exc:

        add_alert(
            alerts,
            "ERROR",
            "COHESION_CRS_ERROR",
            f"No fue posible reproyectar a {CRS_METRICO}: {exc}",
        )

        return 0.0, {}, pd.DataFrame()

    centroids = metric.geometry.centroid

    rows = []

    for scenario_id, indexes in (
        metric.groupby(
            scenario_col
        ).groups.items()
    ):

        points = centroids.loc[indexes]

        if len(points) <= 1:

            mean_distance = 0.0
            max_distance = 0.0

        else:

            xy = np.column_stack(
                [
                    points.x.to_numpy(),
                    points.y.to_numpy(),
                ]
            )

            center = xy.mean(axis=0)

            distances = np.sqrt(
                (
                    (xy - center) ** 2
                ).sum(axis=1)
            )

            mean_distance = float(
                distances.mean()
            )

            max_distance = float(
                distances.max()
            )

        score = clamp(
            1.0
            - (
                mean_distance
                / DISTANCIA_COHESION_MAX_M
            )
        )

        rows.append(
            {
                "escenario_id": str(
                    scenario_id
                ),
                "distancia_media_m": mean_distance,
                "distancia_maxima_m": max_distance,
                "score_cohesion": score,
            }
        )

    detail = pd.DataFrame(rows)

    if detail.empty:

        return 0.0, {}, detail

    mean_distance = float(
        detail["distancia_media_m"].mean()
    )

    maximum_distance = float(
        detail["distancia_maxima_m"].max()
    )

    score = float(
        detail["score_cohesion"].mean()
    )

    print(
        f"Distancia media       : "
        f"{mean_distance:.2f} m"
    )

    print(
        f"Distancia máxima      : "
        f"{maximum_distance:.2f} m"
    )

    print(
        f"Score cohesión        : "
        f"{score:.4f}"
    )

    if maximum_distance > DISTANCIA_COHESION_MAX_M:

        add_alert(
            alerts,
            "ADVERTENCIA",
            "COHESION_LOW",
            (
                "Al menos un escenario presenta "
                "dispersión territorial superior a "
                f"{DISTANCIA_COHESION_MAX_M / 1000:.0f} km."
            ),
        )

    return (
        score,
        {
            "distancia_media_m": mean_distance,
            "distancia_maxima_m": maximum_distance,
        },
        detail,
    )


# =============================================================================
# 7. INDICADORES
# =============================================================================

def validate_indicators(
    gdf: pd.DataFrame,
    alerts: list[dict],
) -> tuple[float, list[str], pd.DataFrame]:

    print_header("7. VALIDACIÓN DE INDICADORES")

    available = [
        column
        for column in INDICADORES_ESPERADOS
        if column in gdf.columns
    ]

    missing = [
        column
        for column in INDICADORES_ESPERADOS
        if column not in gdf.columns
    ]

    print(
        f"Indicadores disponibles : "
        f"{len(available)}/{len(INDICADORES_ESPERADOS)}"
    )

    for column in available:
        print(f"  - {column}")

    if missing:

        add_alert(
            alerts,
            "ADVERTENCIA",
            "INDICATORS_MISSING",
            (
                "No están disponibles algunos indicadores "
                "esperados: "
                + ", ".join(missing)
            ),
        )

    if not available:

        add_alert(
            alerts,
            "ERROR",
            "NO_INDICATORS",
            "No se encontró ningún indicador estructural.",
        )

        return (
            0.0,
            [],
            pd.DataFrame(),
        )

    rows = []

    for column in available:

        values = pd.to_numeric(
            gdf[column],
            errors="coerce",
        )

        valid = int(
            values.notna().sum()
        )

        finite = int(
            np.isfinite(
                values.dropna()
            ).sum()
        )

        non_finite = int(
            values.notna().sum()
            - finite
        )

        coverage = (
            valid / len(gdf)
            if len(gdf)
            else 0.0
        )

        rows.append(
            {
                "indicador": column,
                "registros": len(gdf),
                "validos": valid,
                "no_finitos": non_finite,
                "cobertura": coverage,
            }
        )

        if non_finite:

            add_alert(
                alerts,
                "ERROR",
                "INDICATOR_NON_FINITE",
                (
                    f"El indicador {column} "
                    f"contiene {non_finite} "
                    "valores no finitos."
                ),
            )

    detail = pd.DataFrame(rows)

    score = float(
        detail["cobertura"].mean()
    )

    print(
        f"Completitud indicadores : "
        f"{score:.4f}"
    )

    return score, available, detail


# =============================================================================
# 8. BALANCE
# =============================================================================

def validate_balance(
    counts: pd.Series,
) -> tuple[float, dict]:

    print_header("8. VALIDACIÓN DE BALANCE")

    if counts.empty:
        return 0.0, {}

    mean = float(counts.mean())

    std = float(
        counts.std(ddof=0)
    )

    cv = (
        std / mean
        if mean > 0
        else 1.0
    )

    score = clamp(
        1.0 - cv
    )

    print(
        f"Promedio              : {mean:.2f}"
    )

    print(
        f"Desvío                : {std:.2f}"
    )

    print(
        f"CV                    : {cv:.4f}"
    )

    print(
        f"Score balance          : {score:.4f}"
    )

    return score, {
        "mean": mean,
        "std": std,
        "cv": cv,
    }


# =============================================================================
# 9. CONSISTENCIA INTERNA
# =============================================================================

def validate_internal_consistency(
    gdf: pd.DataFrame,
    cols: dict,
    alerts: list[dict],
) -> tuple[int, pd.DataFrame]:

    print_header("9. CONSISTENCIA INTERNA")

    scenario_col = cols["escenario"]

    if scenario_col is None:
        return 0, pd.DataFrame()

    fields = [
        cols["tipo"],
        cols["dimension"],
        cols["prioridad"],
    ]

    fields = [
        field
        for field in fields
        if field is not None
    ]

    issues = []

    for scenario_id, group in gdf.groupby(
        scenario_col,
        dropna=False,
    ):

        scenario_text = clean_text(
            scenario_id
        )

        for field in fields:

            values = (
                group[field]
                .dropna()
                .astype(str)
                .str.strip()
            )

            values = values[
                values != ""
            ]

            unique = int(
                values.nunique()
            )

            if unique > 1:

                detail = (
                    f"El campo {field} contiene "
                    f"{unique} valores distintos "
                    "dentro del mismo escenario."
                )

                issues.append(
                    {
                        "escenario_id": scenario_text,
                        "campo": field,
                        "detalle": detail,
                    }
                )

                add_alert(
                    alerts,
                    "ADVERTENCIA",
                    "INTERNAL_CONSISTENCY",
                    detail,
                    scenario_text,
                )

    print(
        f"Inconsistencias detectadas : "
        f"{len(issues)}"
    )

    return (
        len(issues),
        pd.DataFrame(issues),
    )


# =============================================================================
# 10. COMPARACIÓN 27 → 29
# =============================================================================

def compare_27_to_29(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("10. TRAZABILIDAD PROCESO 27 → 29")

    result = {
        "disponible": INPUT_27.exists(),
        "proyectos_27": None,
        "escenarios_27": None,
        "proyectos_comunes": None,
        "proyectos_solo_27": None,
        "proyectos_solo_29": None,
        "asignaciones_modificadas": None,
    }

    if not INPUT_27.exists():

        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_27_MISSING",
            "No se encontró el archivo del proceso 27.",
        )

        return result

    try:

        old = gpd.read_parquet(
            INPUT_27
        )

        old_cols = resolve_columns(old)

        result["proyectos_27"] = len(old)

        if old_cols["escenario"]:

            result["escenarios_27"] = int(
                old[
                    old_cols["escenario"]
                ]
                .dropna()
                .nunique()
            )

        if not all(
            [
                cols["proyecto"],
                cols["escenario"],
                old_cols["proyecto"],
                old_cols["escenario"],
            ]
        ):

            add_alert(
                alerts,
                "ADVERTENCIA",
                "PROCESS_27_COLUMNS",
                (
                    "No fue posible realizar "
                    "la comparación completa "
                    "contra proceso 27."
                ),
            )

            return result

        current = gdf[
            [
                cols["proyecto"],
                cols["escenario"],
            ]
        ].copy()

        previous = old[
            [
                old_cols["proyecto"],
                old_cols["escenario"],
            ]
        ].copy()

        current.columns = [
            "proyecto_id",
            "escenario_29",
        ]

        previous.columns = [
            "proyecto_id",
            "escenario_27",
        ]

        current["proyecto_id"] = (
            current["proyecto_id"]
            .astype(str)
            .str.strip()
        )

        previous["proyecto_id"] = (
            previous["proyecto_id"]
            .astype(str)
            .str.strip()
        )

        current_ids = set(
            current["proyecto_id"]
        )

        previous_ids = set(
            previous["proyecto_id"]
        )

        only_27 = (
            previous_ids
            - current_ids
        )

        only_29 = (
            current_ids
            - previous_ids
        )

        result["proyectos_solo_27"] = len(
            only_27
        )

        result["proyectos_solo_29"] = len(
            only_29
        )

        merged = current.merge(
            previous,
            on="proyecto_id",
            how="inner",
        )

        result["proyectos_comunes"] = len(
            merged
        )

        modified = int(
            (
                merged["escenario_29"]
                .astype(str)
                !=
                merged["escenario_27"]
                .astype(str)
            ).sum()
        )

        result["asignaciones_modificadas"] = (
            modified
        )

        print(
            f"Proyectos proceso 27    : "
            f"{result['proyectos_27']}"
        )

        print(
            f"Proyectos proceso 29    : "
            f"{len(gdf)}"
        )

        print(
            f"Proyectos comunes       : "
            f"{result['proyectos_comunes']}"
        )

        print(
            f"Solo en 27              : "
            f"{result['proyectos_solo_27']}"
        )

        print(
            f"Solo en 29              : "
            f"{result['proyectos_solo_29']}"
        )

        print(
            f"Asignaciones modificadas: "
            f"{modified}"
        )

        if modified:

            add_alert(
                alerts,
                "INFORMACION",
                "EXPECTED_REALLOCATION",
                (
                    f"El proceso 29 modificó "
                    f"{modified} asignaciones respecto "
                    "del proceso 27. Este cambio se "
                    "considera esperado por tratarse "
                    "de una optimización."
                ),
            )

        if only_27:

            add_alert(
                alerts,
                "ERROR",
                "PROJECTS_LOST_27_29",
                (
                    f"Existen {len(only_27)} proyectos "
                    "presentes en 27 y ausentes en 29."
                ),
            )

        if only_29:

            add_alert(
                alerts,
                "ADVERTENCIA",
                "PROJECTS_NEW_27_29",
                (
                    f"Existen {len(only_29)} proyectos "
                    "en 29 que no estaban en 27."
                ),
            )

    except Exception as exc:

        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_27_READ_ERROR",
            f"No fue posible comparar 27 → 29: {exc}",
        )

    return result


# =============================================================================
# 11. COMPARACIÓN PROCESO 28
# =============================================================================

def compare_28(
    alerts: list[dict],
) -> dict:

    print_header("11. TRAZABILIDAD PROCESO 28")

    result = {
        "disponible": INPUT_28.exists(),
        "registros": None,
        "score_referencia": None,
    }

    if not INPUT_28.exists():

        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_28_MISSING",
            "No se encontró el archivo del proceso 28.",
        )

        return result

    try:

        df = pd.read_parquet(
            INPUT_28
        )

        result["registros"] = len(df)

        candidates = [
            "score_global",
            "score_evaluacion",
            "score_validacion",
        ]

        for column in candidates:

            if column not in df.columns:
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            if len(values):

                result["score_referencia"] = (
                    float(values.mean())
                )

                break

        print(
            f"Registros proceso 28    : "
            f"{result['registros']}"
        )

        print(
            f"Score referencia        : "
            f"{result['score_referencia']}"
        )

    except Exception as exc:

        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_28_READ_ERROR",
            f"No fue posible leer proceso 28: {exc}",
        )

    return result


# =============================================================================
# 12. COMPARACIÓN PROCESO 29
# =============================================================================

def compare_29(
    alerts: list[dict],
) -> dict:

    print_header("12. TRAZABILIDAD PROCESO 29")

    result = {
        "evaluacion_disponible":
            INPUT_29_EVALUACION.exists(),

        "resumen_disponible":
            INPUT_29_RESUMEN.exists(),

        "score_referencia": None,
    }

    if INPUT_29_EVALUACION.exists():

        try:

            df = pd.read_csv(
                INPUT_29_EVALUACION
            )

            candidates = [
                "score_global",
                "score_global_optimizado",
                "score_optimizado",
            ]

            for column in candidates:

                if column not in df.columns:
                    continue

                values = pd.to_numeric(
                    df[column],
                    errors="coerce",
                ).dropna()

                if len(values):

                    result["score_referencia"] = (
                        float(values.mean())
                    )

                    break

        except Exception as exc:

            add_alert(
                alerts,
                "ADVERTENCIA",
                "PROCESS_29_EVALUATION_ERROR",
                (
                    "No fue posible leer "
                    f"{INPUT_29_EVALUACION.name}: {exc}"
                ),
            )

    if (
        result["score_referencia"] is None
        and INPUT_29_RESUMEN.exists()
    ):

        try:

            df = pd.read_csv(
                INPUT_29_RESUMEN
            )

            candidates = [
                "score_global_optimizado",
                "score_optimizado",
                "score_global",
            ]

            for column in candidates:

                if column not in df.columns:
                    continue

                values = pd.to_numeric(
                    df[column],
                    errors="coerce",
                ).dropna()

                if len(values):

                    result["score_referencia"] = (
                        float(values.iloc[-1])
                    )

                    break

        except Exception as exc:

            add_alert(
                alerts,
                "ADVERTENCIA",
                "PROCESS_29_SUMMARY_ERROR",
                (
                    "No fue posible leer "
                    f"{INPUT_29_RESUMEN.name}: {exc}"
                ),
            )

    print(
        f"Evaluación disponible   : "
        f"{result['evaluacion_disponible']}"
    )

    print(
        f"Resumen disponible      : "
        f"{result['resumen_disponible']}"
    )

    print(
        f"Score proceso 29        : "
        f"{result['score_referencia']}"
    )

    return result


# =============================================================================
# 13. DETALLE POR ESCENARIO
# =============================================================================

def build_scenario_detail(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    cohesion_detail: pd.DataFrame,
) -> pd.DataFrame:

    scenario_col = cols["escenario"]

    if scenario_col is None:
        return pd.DataFrame()

    grouped = (
        gdf.groupby(
            scenario_col,
            dropna=False,
        )
        .size()
        .rename("cantidad_proyectos")
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            scenario_col: "escenario_id"
        }
    )

    mean_count = float(
        grouped["cantidad_proyectos"].mean()
    )

    if mean_count > 0:

        grouped["score_tamano"] = (
            1
            - (
                (
                    grouped["cantidad_proyectos"]
                    - mean_count
                ).abs()
                / mean_count
            )
        ).clip(0, 1)

    else:

        grouped["score_tamano"] = 0.0

    if not cohesion_detail.empty:

        grouped = grouped.merge(
            cohesion_detail,
            on="escenario_id",
            how="left",
        )

    else:

        grouped[
            "distancia_media_m"
        ] = np.nan

        grouped[
            "distancia_maxima_m"
        ] = np.nan

        grouped[
            "score_cohesion"
        ] = np.nan

    indicator_cols = [
        column
        for column in INDICADORES_ESPERADOS
        if column in gdf.columns
    ]

    if indicator_cols:

        temp = gdf.copy()

        for column in indicator_cols:

            temp[column] = pd.to_numeric(
                temp[column],
                errors="coerce",
            )

        grouped_indicators = (
            temp.groupby(
                scenario_col
            )[indicator_cols]
            .mean()
            .reset_index()
        )

        grouped_indicators = (
            grouped_indicators.rename(
                columns={
                    scenario_col: "escenario_id"
                }
            )
        )

        normalized_columns = []

        for column in indicator_cols:

            values = pd.to_numeric(
                gdf[column],
                errors="coerce",
            )

            minimum = values.min()
            maximum = values.max()

            normalized = (
                f"{column}__norm"
            )

            if (
                pd.isna(minimum)
                or pd.isna(maximum)
                or maximum == minimum
            ):

                grouped_indicators[
                    normalized
                ] = 1.0

            else:

                grouped_indicators[
                    normalized
                ] = (
                    (
                        grouped_indicators[column]
                        - minimum
                    )
                    / (
                        maximum
                        - minimum
                    )
                ).clip(0, 1)

            normalized_columns.append(
                normalized
            )

        grouped_indicators[
            "score_indicadores"
        ] = (
            grouped_indicators[
                normalized_columns
            ].mean(axis=1)
        )

        grouped = grouped.merge(
            grouped_indicators[
                [
                    "escenario_id",
                    "score_indicadores",
                ]
            ],
            on="escenario_id",
            how="left",
        )

    else:

        grouped[
            "score_indicadores"
        ] = 0.0

    grouped[
        "score_validacion_escenario"
    ] = (
        0.20
        * grouped[
            "score_tamano"
        ].fillna(0)
        +
        0.50
        * grouped[
            "score_cohesion"
        ].fillna(0)
        +
        0.30
        * grouped[
            "score_indicadores"
        ].fillna(0)
    )

    grouped = grouped.sort_values(
        "score_validacion_escenario",
        ascending=False,
    ).reset_index(drop=True)

    grouped[
        "ranking_validacion"
    ] = np.arange(
        1,
        len(grouped) + 1,
    )

    grouped[
        "clasificacion_validacion"
    ] = np.select(
        [
            grouped[
                "score_validacion_escenario"
            ] >= 0.75,

            grouped[
                "score_validacion_escenario"
            ] >= 0.55,
        ],
        [
            "VALIDADO",
            "VALIDADO_CON_OBSERVACIONES",
        ],
        default="REVISAR",
    )

    return grouped


# =============================================================================
# 14. SCORE GLOBAL
# =============================================================================

def calculate_global_score(
    metrics: dict,
) -> float:

    return clamp(
        PESOS["cobertura"]
        * metrics["cobertura"]

        + PESOS["tamano"]
        * metrics["tamano"]

        + PESOS["cohesion"]
        * metrics["cohesion"]

        + PESOS["indicadores"]
        * metrics["indicadores"]

        + PESOS["balance"]
        * metrics["balance"]
    )


# =============================================================================
# 15. DICTAMEN
# =============================================================================

def calculate_dictamen(
    score: float,
    blocking_errors: int,
    critical_warnings: int,
) -> tuple[str, str]:

    if blocking_errors > 0:

        return (
            "RECHAZADO",
            (
                "Se detectaron errores estructurales "
                "o de integridad que impiden validar "
                "el resultado del proceso 29."
            ),
        )

    if score < SCORE_OBSERVACIONES:

        return (
            "REVISAR",
            (
                "El resultado no presenta errores "
                "estructurales bloqueantes, pero su "
                "calidad global está por debajo del "
                "umbral mínimo."
            ),
        )

    if (
        score >= SCORE_VALIDADO
        and critical_warnings == 0
    ):

        return (
            "VALIDADO",
            (
                "El resultado cumple los controles "
                "estructurales, de cobertura y de "
                "calidad territorial definidos."
            ),
        )

    return (
        "VALIDADO_CON_OBSERVACIONES",
        (
            "El resultado supera el umbral mínimo "
            "de validación, pero presenta "
            "observaciones que deben ser consideradas."
        ),
    )


# =============================================================================
# 16. EXPORTACIÓN
# =============================================================================

def export_results(
    scenario_detail: pd.DataFrame,
    comparison: pd.DataFrame,
    alerts: pd.DataFrame,
    summary: dict,
) -> None:

    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario_detail.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    scenario_detail.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    scenario_detail.to_csv(
        OUTPUT_DETALLE,
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        OUTPUT_COMPARACION,
        index=False,
        encoding="utf-8-sig",
    )

    alerts.to_csv(
        OUTPUT_ALERTAS,
        index=False,
        encoding="utf-8-sig",
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    start = time.perf_counter()

    print_header(
        f"30 - VALIDACIÓN FINAL DE ESCENARIOS "
        f"TERRITORIALES AMBA - {VERSION}"
    )

    print(f"Proyecto : {PROJECT_ROOT}")
    print(f"Entrada  : {INPUT_29}")
    print(f"Salida   : {BASE_DIR}")

    print()
    print("CONFIGURACIÓN")
    print(
        f"  Escenarios válidos    : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
    )
    print(
        f"  Mínimo proyectos      : "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )
    print(
        f"  CRS geográfico        : "
        f"{CRS_GEOGRAFICO}"
    )
    print(
        f"  CRS métrico           : "
        f"{CRS_METRICO}"
    )

    print()
    print("PESOS")

    for name, weight in PESOS.items():

        print(
            f"  {name:<20}: "
            f"{weight:.0%}"
        )

    # =========================================================================
    # CARGA
    # =========================================================================

    print_header(
        "CARGANDO RESULTADO DEL PROCESO 29"
    )

    gdf = load_input()

    print(
        f"Registros              : {len(gdf)}"
    )

    print(
        f"Columnas               : {len(gdf.columns)}"
    )

    print(
        f"CRS                    : {gdf.crs}"
    )

    alerts: list[dict] = []

    cols = resolve_columns(gdf)

    print()
    print("COLUMNAS RESUELTAS")

    for key, value in cols.items():

        print(
            f"  {key:<20}: {value}"
        )

    # =========================================================================
    # VALIDACIONES
    # =========================================================================

    structure = validate_structure(
        gdf,
        cols,
        alerts,
    )

    projects = validate_projects(
        gdf,
        cols,
        alerts,
    )

    scenarios = validate_scenarios(
        gdf,
        cols,
        alerts,
    )

    counts = scenarios[
        "conteos"
    ]

    coverage = validate_coverage(
        gdf,
        cols,
        alerts,
    )

    size_score, size_stats = (
        validate_size(counts)
    )

    (
        cohesion_score,
        cohesion_stats,
        cohesion_detail,
    ) = validate_cohesion(
        gdf,
        cols,
        alerts,
    )

    (
        indicator_score,
        indicators,
        indicator_detail,
    ) = validate_indicators(
        gdf,
        alerts,
    )

    balance_score, balance_stats = (
        validate_balance(counts)
    )

    (
        inconsistencies,
        internal_detail,
    ) = validate_internal_consistency(
        gdf,
        cols,
        alerts,
    )

    # =========================================================================
    # TRAZABILIDAD
    # =========================================================================

    comparison_27 = compare_27_to_29(
        gdf,
        cols,
        alerts,
    )

    comparison_28 = compare_28(
        alerts,
    )

    comparison_29 = compare_29(
        alerts,
    )

    # =========================================================================
    # SCORE
    # =========================================================================

    print_header(
        "13. SCORE GLOBAL DE VALIDACIÓN"
    )

    metrics = {
        "cobertura": coverage[
            "cobertura"
        ],

        "tamano": size_score,

        "cohesion": cohesion_score,

        "indicadores": indicator_score,

        "balance": balance_score,
    }

    score_global = calculate_global_score(
        metrics
    )

    # -------------------------------------------------------------------------
    # ERRORES BLOQUEANTES
    # -------------------------------------------------------------------------

    blocking_codes = {
        "GEOM_NULL",
        "GEOM_EMPTY",
        "GEOM_INVALID",
        "CRS_MISSING",
        "PROJECT_ID_MISSING",
        "PROJECT_ID_NULL",
        "PROJECT_ID_DUPLICATE",
        "SCENARIO_ID_MISSING",
        "SCENARIO_COUNT",
        "SCENARIO_NULL",
        "SCENARIO_TOO_SMALL",
        "COVERAGE_INCOMPLETE",
        "PROJECTS_LOST_27_29",
        "NO_INDICATORS",
        "INDICATOR_NON_FINITE",
    }

    blocking_errors = sum(
        1
        for alert in alerts
        if (
            alert["nivel"] == "ERROR"
            and alert["codigo"]
            in blocking_codes
        )
    )

    # Advertencias que realmente afectan la
    # interpretación del resultado.
    critical_warning_codes = {
        "COHESION_LOW",
        "INDICATORS_MISSING",
        "INTERNAL_CONSISTENCY",
    }

    critical_warnings = sum(
        1
        for alert in alerts
        if (
            alert["nivel"]
            == "ADVERTENCIA"
            and alert["codigo"]
            in critical_warning_codes
        )
    )

    dictamen, fundamento = calculate_dictamen(
        score_global,
        blocking_errors,
        critical_warnings,
    )

    print(
        f"Cobertura              : "
        f"{metrics['cobertura']:.4f}"
    )

    print(
        f"Tamaño                 : "
        f"{metrics['tamano']:.4f}"
    )

    print(
        f"Cohesión               : "
        f"{metrics['cohesion']:.4f}"
    )

    print(
        f"Indicadores            : "
        f"{metrics['indicadores']:.4f}"
    )

    print(
        f"Balance                : "
        f"{metrics['balance']:.4f}"
    )

    print()
    print(
        f"SCORE VALIDACIÓN       : "
        f"{score_global:.4f}"
    )

    print(
        f"ERRORES BLOQUEANTES    : "
        f"{blocking_errors}"
    )

    print(
        f"ADVERTENCIAS CRÍTICAS  : "
        f"{critical_warnings}"
    )

    print(
        f"DICTAMEN               : "
        f"{dictamen}"
    )

    # =========================================================================
    # DETALLE
    # =========================================================================

    print_header(
        "14. DETALLE POR ESCENARIO"
    )

    scenario_detail = build_scenario_detail(
        gdf,
        cols,
        cohesion_detail,
    )

    if not scenario_detail.empty:

        display_columns = [
            "escenario_id",
            "cantidad_proyectos",
            "score_tamano",
            "score_cohesion",
            "score_indicadores",
            "score_validacion_escenario",
            "ranking_validacion",
            "clasificacion_validacion",
        ]

        existing_display = [
            column
            for column in display_columns
            if column in scenario_detail.columns
        ]

        print(
            scenario_detail[
                existing_display
            ].to_string(index=False)
        )

    # =========================================================================
    # RESUMEN DE ALERTAS
    # =========================================================================

    error_count = sum(
        1
        for alert in alerts
        if alert["nivel"] == "ERROR"
    )

    warning_count = sum(
        1
        for alert in alerts
        if alert["nivel"] == "ADVERTENCIA"
    )

    info_count = sum(
        1
        for alert in alerts
        if alert["nivel"] == "INFORMACION"
    )

    # =========================================================================
    # COMPARACIÓN
    # =========================================================================

    comparison_rows = [
        {
            "proceso": "27",
            "archivo_disponible":
                comparison_27["disponible"],
            "proyectos":
                comparison_27["proyectos_27"],
            "escenarios":
                comparison_27["escenarios_27"],
            "proyectos_comunes":
                comparison_27["proyectos_comunes"],
            "proyectos_solo_proceso":
                comparison_27["proyectos_solo_27"],
            "proyectos_solo_actual":
                comparison_27["proyectos_solo_29"],
            "asignaciones_modificadas":
                comparison_27[
                    "asignaciones_modificadas"
                ],
            "score":
                None,
        },

        {
            "proceso": "28",
            "archivo_disponible":
                comparison_28["disponible"],
            "proyectos": None,
            "escenarios": None,
            "proyectos_comunes": None,
            "proyectos_solo_proceso": None,
            "proyectos_solo_actual": None,
            "asignaciones_modificadas": None,
            "score":
                comparison_28[
                    "score_referencia"
                ],
        },

        {
            "proceso": "29",
            "archivo_disponible": True,
            "proyectos": len(gdf),
            "escenarios": len(counts),
            "proyectos_comunes": None,
            "proyectos_solo_proceso": None,
            "proyectos_solo_actual": None,
            "asignaciones_modificadas": None,
            "score":
                comparison_29[
                    "score_referencia"
                ],
        },

        {
            "proceso": "30",
            "archivo_disponible": True,
            "proyectos": len(gdf),
            "escenarios": len(counts),
            "proyectos_comunes": None,
            "proyectos_solo_proceso": None,
            "proyectos_solo_actual": None,
            "asignaciones_modificadas": None,
            "score": score_global,
        },
    ]

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # =========================================================================
    # ALERTAS
    # =========================================================================

    alerts_df = pd.DataFrame(
        alerts,
        columns=[
            "nivel",
            "codigo",
            "escenario_id",
            "mensaje",
        ],
    )

    # =========================================================================
    # RESUMEN JSON
    # =========================================================================

    elapsed = (
        time.perf_counter()
        - start
    )

    summary = {
        "proceso": 30,
        "version": VERSION,
        "fecha_ejecucion":
            pd.Timestamp.now().isoformat(),

        "proyecto":
            str(PROJECT_ROOT),

        "entrada":
            str(INPUT_29),

        "resultado": {
            "dictamen": dictamen,
            "fundamento": fundamento,
            "score_global": score_global,
            "score_validado":
                SCORE_VALIDADO,
            "score_observaciones":
                SCORE_OBSERVACIONES,
        },

        "estructura": structure,

        "proyectos": {
            "total":
                projects["total"],
            "sin_id":
                projects["sin_id"],
            "duplicados":
                projects["duplicados"],
            "asignados":
                coverage["asignados"],
            "sin_escenario":
                coverage["sin_escenario"],
            "cobertura":
                coverage["cobertura"],
        },

        "escenarios": {
            "cantidad":
                scenarios["cantidad"],
            "nulos":
                scenarios["nulos"],
            "min_proyectos":
                scenarios["min_proyectos"],
            "max_proyectos":
                scenarios["max_proyectos"],
        },

        "metricas":
            metrics,

        "pesos":
            PESOS,

        "cohesion":
            cohesion_stats,

        "balance":
            balance_stats,

        "tamano":
            size_stats,

        "indicadores":
            indicators,

        "consistencia": {
            "inconsistencias":
                inconsistencies,
        },

        "trazabilidad": {
            "proceso_27":
                comparison_27,
            "proceso_28":
                comparison_28,
            "proceso_29":
                comparison_29,
        },

        "alertas": {
            "errores":
                error_count,
            "errores_bloqueantes":
                blocking_errors,
            "advertencias":
                warning_count,
            "advertencias_criticas":
                critical_warnings,
            "informaciones":
                info_count,
        },

        "salidas": {
            "parquet":
                str(OUTPUT_PARQUET),
            "csv":
                str(OUTPUT_CSV),
            "detalle":
                str(OUTPUT_DETALLE),
            "comparacion":
                str(OUTPUT_COMPARACION),
            "alertas":
                str(OUTPUT_ALERTAS),
            "json":
                str(OUTPUT_JSON),
        },

        "duracion_segundos":
            elapsed,
    }

    # =========================================================================
    # EXPORTACIÓN
    # =========================================================================

    print_header(
        "15. EXPORTANDO RESULTADOS"
    )

    export_results(
        scenario_detail,
        comparison_df,
        alerts_df,
        summary,
    )

    print(
        f"Parquet                : "
        f"{OUTPUT_PARQUET}"
    )

    print(
        f"CSV                    : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Detalle                : "
        f"{OUTPUT_DETALLE}"
    )

    print(
        f"Comparación            : "
        f"{OUTPUT_COMPARACION}"
    )

    print(
        f"Alertas                : "
        f"{OUTPUT_ALERTAS}"
    )

    print(
        f"Resumen JSON           : "
        f"{OUTPUT_JSON}"
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print_header(
        "16. PROCESO 30 FINALIZADO"
    )

    print(
        f"Proyectos evaluados    : "
        f"{len(gdf)}"
    )

    print(
        f"Escenarios evaluados   : "
        f"{len(counts)}"
    )

    print(
        f"Cobertura              : "
        f"{coverage['cobertura']:.2%}"
    )

    print(
        f"Score validación       : "
        f"{score_global:.4f}"
    )

    print(
        f"Errores bloqueantes    : "
        f"{blocking_errors}"
    )

    print(
        f"Errores totales        : "
        f"{error_count}"
    )

    print(
        f"Advertencias           : "
        f"{warning_count}"
    )

    print(
        f"Informaciones          : "
        f"{info_count}"
    )

    print(
        f"Duración               : "
        f"{elapsed:.2f} segundos"
    )

    print()
    print("=" * 88)
    print("DICTAMEN FINAL")
    print("=" * 88)
    print()
    print(f"  {dictamen}")
    print()
    print(f"  {fundamento}")
    print()
    print("=" * 88)

    if not alerts_df.empty:

        print()
        print("ALERTAS")
        print()

        print(
            alerts_df.to_string(
                index=False
            )
        )

    print()
    print("=" * 88)

    return (
        0
        if dictamen != "RECHAZADO"
        else 2
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())