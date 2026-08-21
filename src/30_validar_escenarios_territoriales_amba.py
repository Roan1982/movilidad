from pathlib import Path

code = r'''# -*- coding: utf-8 -*-
"""
30 - VALIDACIÓN FINAL DE ESCENARIOS TERRITORIALES AMBA - V2.0

Valida el resultado del proceso 29 sin confundir:
- cambios esperables de asignación con errores,
- advertencias con errores bloqueantes,
- métricas de calidad con integridad estructural.

Entrada principal:
data/processed/escenarios_territoriales_amba/
    escenarios_territoriales_amba_optimizado.parquet

Salidas:
- validacion_final_escenarios_territoriales_amba.parquet
- validacion_final_escenarios_territoriales_amba.csv
- detalle_validacion_escenarios_territoriales_amba.csv
- comparacion_procesos_27_28_29_30.csv
- alertas_validacion_escenarios_territoriales_amba.csv
- resumen_validacion_escenarios_territoriales_amba.json
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V2.0"

MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12
MIN_PROYECTOS_ESCENARIO = 8

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

PESOS = {
    "cobertura": 0.20,
    "tamano": 0.15,
    "cohesion": 0.25,
    "indicadores": 0.20,
    "balance": 0.20,
}

# Umbrales de dictamen
SCORE_VALIDADO = 0.80
SCORE_OBSERVACIONES = 0.60

# Cohesión:
# 0 m -> 1.0
# 50 km o más -> 0.0
DISTANCIA_COHESION_MAX_M = 50_000.0

INDICADORES_PRIORITARIOS = [
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

INPUT_OPTIMIZADO = BASE_DIR / "escenarios_territoriales_amba_optimizado.parquet"
INPUT_PROCESO_27 = BASE_DIR / "escenarios_territoriales_amba.parquet"
INPUT_PROCESO_28 = BASE_DIR / "evaluacion_escenarios_territoriales_amba.parquet"
INPUT_PROCESO_29_EVAL = BASE_DIR / "evaluacion_escenarios_optimizada.csv"
INPUT_PROCESO_29_RESUMEN = BASE_DIR / "resumen_optimizacion_escenarios.csv"


OUTPUT_PARQUET = (
    BASE_DIR / "validacion_final_escenarios_territoriales_amba.parquet"
)
OUTPUT_CSV = (
    BASE_DIR / "validacion_final_escenarios_territoriales_amba.csv"
)
OUTPUT_DETALLE = (
    BASE_DIR / "detalle_validacion_escenarios_territoriales_amba.csv"
)
OUTPUT_COMPARACION = (
    BASE_DIR / "comparacion_procesos_27_28_29_30.csv"
)
OUTPUT_ALERTAS = (
    BASE_DIR / "alertas_validacion_escenarios_territoriales_amba.csv"
)
OUTPUT_JSON = (
    BASE_DIR / "resumen_validacion_escenarios_territoriales_amba.json"
)


# =============================================================================
# UTILIDADES
# =============================================================================

def print_header(title: str, char: str = "=") -> None:
    print("\n" + char * 88)
    print(title)
    print(char * 88)


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def safe_float(value, default=0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, safe_float(value)))


def score_minimum(n: int, minimum: int) -> float:
    if n >= minimum:
        return 1.0
    if minimum <= 0:
        return 1.0
    return clamp(n / minimum)


def normalize_score(value, minimum, maximum) -> float:
    value = safe_float(value)
    if maximum <= minimum:
        return 1.0
    return clamp((value - minimum) / (maximum - minimum))


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Advertencia: no se pudo leer {path.name}: {exc}")
        return None


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    return str(obj)


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


# =============================================================================
# CARGA
# =============================================================================

def load_input() -> gpd.GeoDataFrame:
    if not INPUT_OPTIMIZADO.exists():
        raise FileNotFoundError(
            f"No existe la entrada del proceso 29:\n{INPUT_OPTIMIZADO}"
        )

    gdf = gpd.read_parquet(INPUT_OPTIMIZADO)

    if gdf.crs is None:
        warnings.warn(
            "La entrada no declara CRS. Se asumirá EPSG:4326.",
            RuntimeWarning,
        )
        gdf = gdf.set_crs(CRS_GEOGRAFICO, allow_override=True)

    return gdf


# =============================================================================
# RESOLUCIÓN DE VARIABLES
# =============================================================================

def resolve_columns(gdf: pd.DataFrame) -> dict:
    return {
        "proyecto": first_existing(
            gdf,
            ["proyecto_id", "id_proyecto", "project_id", "id"],
        ),
        "escenario": first_existing(
            gdf,
            ["escenario_id", "cluster_territorial", "cluster_id"],
        ),
        "score_escenario": first_existing(
            gdf,
            ["score_escenario", "score_cartera"],
        ),
        "tipo": first_existing(
            gdf,
            ["tipo_escenario", "tipo"],
        ),
        "dimension": first_existing(
            gdf,
            ["dimension_dominante", "dimension"],
        ),
        "prioridad": first_existing(
            gdf,
            ["prioridad_escenario", "prioridad"],
        ),
    }


# =============================================================================
# VALIDACIÓN ESTRUCTURAL
# =============================================================================

def validate_structure(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("2. VALIDANDO ESTRUCTURA GEOGRÁFICA")

    null_geom = int(gdf.geometry.isna().sum())
    empty_geom = int(gdf.geometry.is_empty.sum())
    invalid_geom = int((~gdf.geometry.is_valid).sum())

    print(f"Geometrías nulas      : {null_geom}")
    print(f"Geometrías vacías     : {empty_geom}")
    print(f"Geometrías inválidas  : {invalid_geom}")
    print(f"CRS                   : {gdf.crs}")

    if null_geom:
        add_alert(
            alerts, "ERROR", "GEOM_NULL",
            f"Existen {null_geom} geometrías nulas."
        )

    if empty_geom:
        add_alert(
            alerts, "ERROR", "GEOM_EMPTY",
            f"Existen {empty_geom} geometrías vacías."
        )

    if invalid_geom:
        add_alert(
            alerts, "ERROR", "GEOM_INVALID",
            f"Existen {invalid_geom} geometrías inválidas."
        )

    if cols["proyecto"] is None:
        add_alert(
            alerts, "ERROR", "PROJECT_ID_MISSING",
            "No se encontró identificador de proyecto."
        )

    if cols["escenario"] is None:
        add_alert(
            alerts, "ERROR", "SCENARIO_ID_MISSING",
            "No se encontró identificador de escenario."
        )

    if gdf.crs is None:
        add_alert(
            alerts, "ERROR", "CRS_MISSING",
            "La capa no tiene CRS."
        )

    return {
        "geometrias_nulas": null_geom,
        "geometrias_vacias": empty_geom,
        "geometrias_invalidas": invalid_geom,
    }


# =============================================================================
# VALIDACIÓN DE PROYECTOS
# =============================================================================

def validate_projects(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("3. VALIDANDO PROYECTOS")

    project_col = cols["proyecto"]

    if project_col is None:
        return {
            "proyectos_totales": len(gdf),
            "proyectos_sin_id": len(gdf),
            "duplicados_proyecto": 0,
        }

    ids = gdf[project_col].astype("string")
    missing = int(ids.isna().sum() + (ids.str.strip() == "").sum())
    duplicated = int(ids.dropna().duplicated().sum())

    print(f"Identificador proyecto : {project_col}")
    print(f"Proyectos totales       : {len(gdf)}")
    print(f"Proyectos sin ID        : {missing}")
    print(f"Duplicados proyecto     : {duplicated}")

    if missing:
        add_alert(
            alerts, "ERROR", "PROJECT_ID_NULL",
            f"Existen {missing} proyectos sin identificador."
        )

    if duplicated:
        add_alert(
            alerts, "ERROR", "PROJECT_ID_DUPLICATE",
            f"Existen {duplicated} identificadores de proyecto duplicados."
        )

    return {
        "proyectos_totales": int(len(gdf)),
        "proyectos_sin_id": missing,
        "duplicados_proyecto": duplicated,
    }


# =============================================================================
# VALIDACIÓN DE ESCENARIOS
# =============================================================================

def validate_scenarios(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("4. VALIDANDO ESCENARIOS")

    scenario_col = cols["escenario"]

    if scenario_col is None:
        return {
            "escenarios": 0,
            "escenarios_nulos": len(gdf),
            "min_proyectos": 0,
            "max_proyectos": 0,
        }

    scenario = gdf[scenario_col].astype("string")
    nulls = int(scenario.isna().sum() + (scenario.str.strip() == "").sum())

    counts = (
        gdf.assign(_scenario=scenario)
        .dropna(subset=["_scenario"])
        .groupby("_scenario")
        .size()
    )

    n_scenarios = int(len(counts))
    min_projects = int(counts.min()) if len(counts) else 0
    max_projects = int(counts.max()) if len(counts) else 0

    print(f"Escenarios detectados   : {n_scenarios}")
    print(f"Rango válido            : {MIN_ESCENARIOS}-{MAX_ESCENARIOS}")
    print(f"Escenarios nulos        : {nulls}")

    if not MIN_ESCENARIOS <= n_scenarios <= MAX_ESCENARIOS:
        add_alert(
            alerts, "ERROR", "SCENARIO_COUNT",
            f"Cantidad de escenarios fuera del rango permitido: {n_scenarios}."
        )

    if nulls:
        add_alert(
            alerts, "ERROR", "SCENARIO_NULL",
            f"Existen {nulls} proyectos sin escenario."
        )

    too_small = counts[counts < MIN_PROYECTOS_ESCENARIO]
    if len(too_small):
        for sid, n in too_small.items():
            add_alert(
                alerts,
                "ERROR",
                "SCENARIO_TOO_SMALL",
                f"El escenario tiene {n} proyectos; mínimo requerido "
                f"{MIN_PROYECTOS_ESCENARIO}.",
                str(sid),
            )

    return {
        "escenarios": n_scenarios,
        "escenarios_nulos": nulls,
        "min_proyectos": min_projects,
        "max_proyectos": max_projects,
        "conteos": counts,
    }


# =============================================================================
# COBERTURA
# =============================================================================

def validate_coverage(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    print_header("5. VALIDANDO COBERTURA")

    project_col = cols["proyecto"]
    scenario_col = cols["escenario"]

    if project_col is None or scenario_col is None:
        return {
            "total": len(gdf),
            "asignados": 0,
            "sin_escenario": len(gdf),
            "cobertura": 0.0,
        }

    total = int(gdf[project_col].notna().sum())
    assigned = int(
        gdf[scenario_col].notna().sum()
        & gdf[project_col].notna().sum()
    ) if False else int(
        (
            gdf[project_col].notna()
            & gdf[scenario_col].notna()
            & (gdf[scenario_col].astype("string").str.strip() != "")
        ).sum()
    )

    missing = total - assigned
    coverage = assigned / total if total else 0.0

    print(f"Proyectos totales       : {total}")
    print(f"Proyectos asignados     : {assigned}")
    print(f"Proyectos sin escenario : {missing}")
    print(f"Cobertura               : {coverage:.2%}")

    if missing:
        add_alert(
            alerts, "ERROR", "COVERAGE_INCOMPLETE",
            f"Existen {missing} proyectos sin escenario."
        )

    return {
        "total": total,
        "asignados": assigned,
        "sin_escenario": missing,
        "cobertura": coverage,
    }


# =============================================================================
# TAMAÑO
# =============================================================================

def calculate_size_score(counts: pd.Series) -> tuple[float, dict]:
    if len(counts) == 0:
        return 0.0, {}

    mean = float(counts.mean())
    std = float(counts.std(ddof=0))

    minimum_score = float(
        (counts >= MIN_PROYECTOS_ESCENARIO).mean()
    )

    # Penalización suave por dispersión respecto del tamaño medio.
    cv = std / mean if mean > 0 else 1.0
    balance_component = clamp(1.0 - cv)

    score = clamp(
        0.70 * minimum_score +
        0.30 * balance_component
    )

    return score, {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": mean,
        "std": std,
        "cv": cv,
    }


def validate_size(
    counts: pd.Series,
    alerts: list[dict],
) -> tuple[float, dict]:

    print_header("6. VALIDANDO TAMAÑO DE ESCENARIOS")

    score, stats = calculate_size_score(counts)

    print(f"Escenarios              : {len(counts)}")
    print(f"Mínimo proyectos        : {stats.get('min', 0)}")
    print(f"Máximo proyectos        : {stats.get('max', 0)}")
    print(f"Promedio proyectos      : {stats.get('mean', 0):.4f}")
    print(f"Desvío                  : {stats.get('std', 0):.4f}")
    print(
        f"Cumplen mínimo          : "
        f"{int((counts >= MIN_PROYECTOS_ESCENARIO).sum())}/{len(counts)}"
    )
    print(f"Score tamaño            : {score:.4f}")

    return score, stats


# =============================================================================
# COHESIÓN TERRITORIAL
# =============================================================================

def validate_cohesion(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> tuple[float, dict, pd.DataFrame]:

    print_header("7. VALIDANDO COHESIÓN TERRITORIAL")

    scenario_col = cols["escenario"]

    if scenario_col is None or len(gdf) == 0:
        return 0.0, {}, pd.DataFrame()

    work = gdf[[scenario_col, "geometry"]].copy()

    # Nunca calcular distancias sobre grados.
    if work.crs is None:
        work = work.set_crs(CRS_GEOGRAFICO, allow_override=True)

    metric = work.to_crs(CRS_METRICO)

    centroids = metric.geometry.centroid

    rows = []

    for sid, idx in metric.groupby(scenario_col).groups.items():
        pts = centroids.loc[idx]

        if len(pts) <= 1:
            mean_dist = 0.0
            max_dist = 0.0
        else:
            xy = np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])
            center = xy.mean(axis=0)
            distances = np.sqrt(((xy - center) ** 2).sum(axis=1))
            mean_dist = float(distances.mean())
            max_dist = float(distances.max())

        score = clamp(
            1.0 - mean_dist / DISTANCIA_COHESION_MAX_M
        )

        rows.append(
            {
                "escenario_id": str(sid),
                "distancia_media_m": mean_dist,
                "distancia_maxima_m": max_dist,
                "score_cohesion": score,
            }
        )

    detail = pd.DataFrame(rows)

    if detail.empty:
        return 0.0, {}, detail

    mean_distance = float(detail["distancia_media_m"].mean())
    max_distance = float(detail["distancia_maxima_m"].max())
    score = float(detail["score_cohesion"].mean())

    print(f"Distancia media         : {mean_distance:.2f} m")
    print(f"Distancia máxima        : {max_distance:.2f} m")
    print(f"Score cohesión          : {score:.4f}")

    if max_distance > DISTANCIA_COHESION_MAX_M:
        add_alert(
            alerts,
            "ADVERTENCIA",
            "COHESION_LOW",
            f"Existe al menos un escenario con dispersión superior a "
            f"{DISTANCIA_COHESION_MAX_M / 1000:.0f} km.",
        )

    return score, {
        "distancia_media_m": mean_distance,
        "distancia_maxima_m": max_distance,
    }, detail


# =============================================================================
# INDICADORES
# =============================================================================

def validate_indicators(
    gdf: pd.DataFrame,
    alerts: list[dict],
) -> tuple[float, list[str], pd.DataFrame]:

    print_header("8. VALIDANDO INDICADORES ESTRUCTURALES")

    available = [
        col for col in INDICADORES_PRIORITARIOS
        if col in gdf.columns
    ]

    print(f"Indicadores detectados : {len(available)}")
    for col in available:
        print(f"  - {col}")

    missing = [
        col for col in INDICADORES_PRIORITARIOS
        if col not in gdf.columns
    ]

    if missing:
        add_alert(
            alerts,
            "ADVERTENCIA",
            "INDICATORS_MISSING",
            "No están disponibles algunos indicadores esperados: "
            + ", ".join(missing),
        )

    if not available:
        add_alert(
            alerts,
            "ERROR",
            "NO_INDICATORS",
            "No se encontró ningún indicador estructural.",
        )
        return 0.0, [], pd.DataFrame()

    indicator_rows = []

    for col in available:
        values = pd.to_numeric(gdf[col], errors="coerce")
        valid = int(values.notna().sum())
        finite = int(np.isfinite(values.dropna()).sum())
        invalid = len(values) - finite

        if invalid:
            add_alert(
                alerts,
                "ERROR",
                "INDICATOR_NON_NUMERIC",
                f"El indicador {col} contiene {invalid} valores no numéricos.",
            )

        coverage = valid / len(gdf) if len(gdf) else 0.0

        indicator_rows.append(
            {
                "indicador": col,
                "registros": len(gdf),
                "validos": valid,
                "cobertura": coverage,
            }
        )

    detail = pd.DataFrame(indicator_rows)
    score = float(detail["cobertura"].mean())

    # La presencia y completitud de los indicadores no se usa para
    # premiar artificialmente los valores; sólo se valida integridad.
    print(f"Score indicadores      : {score:.4f}")

    return score, available, detail


# =============================================================================
# BALANCE
# =============================================================================

def validate_balance(
    counts: pd.Series,
) -> tuple[float, dict]:

    print_header("9. VALIDANDO BALANCE ENTRE ESCENARIOS")

    if len(counts) == 0:
        return 0.0, {}

    mean = float(counts.mean())
    std = float(counts.std(ddof=0))
    cv = std / mean if mean > 0 else 1.0
    score = clamp(1.0 - cv)

    print(f"Promedio                : {mean:.4f}")
    print(f"Desvío estándar         : {std:.4f}")
    print(f"Coeficiente variación   : {cv:.4f}")
    print(f"Score balance           : {score:.4f}")

    return score, {
        "mean": mean,
        "std": std,
        "cv": cv,
    }


# =============================================================================
# CONSISTENCIA INTERNA
# =============================================================================

def validate_internal_consistency(
    gdf: pd.DataFrame,
    cols: dict,
    alerts: list[dict],
) -> tuple[int, int, pd.DataFrame]:

    print_header("10. VALIDANDO CONSISTENCIA INTERNA DE ESCENARIOS")

    scenario_col = cols["escenario"]

    if scenario_col is None:
        return 0, 0, pd.DataFrame()

    issues = []

    # Campos que deberían ser constantes dentro de cada escenario.
    categorical_fields = [
        cols["tipo"],
        cols["dimension"],
        cols["prioridad"],
    ]
    categorical_fields = [
        x for x in categorical_fields if x is not None
    ]

    for sid, group in gdf.groupby(scenario_col, dropna=False):

        sid_text = clean_text(sid)

        for col in categorical_fields:
            values = (
                group[col]
                .dropna()
                .astype(str)
                .str.strip()
            )
            unique = values[values != ""].nunique()

            if unique > 1:
                issues.append(
                    {
                        "escenario_id": sid_text,
                        "tipo": "INCONSISTENCIA_CAMPO",
                        "campo": col,
                        "detalle": (
                            f"El campo contiene {unique} valores distintos "
                            "dentro del mismo escenario."
                        ),
                    }
                )

        # El score de escenario puede variar por proyecto en algunos
        # procesos. Por eso NO se considera error si varía.
        # Se registra solamente como observación si hay valores extremos.
        if cols["score_escenario"]:
            values = pd.to_numeric(
                group[cols["score_escenario"]],
                errors="coerce"
            ).dropna()

            if len(values) and not np.isfinite(values).all():
                issues.append(
                    {
                        "escenario_id": sid_text,
                        "tipo": "SCORE_NO_FINITO",
                        "campo": cols["score_escenario"],
                        "detalle": "Existen valores no finitos.",
                    }
                )

    # Estas inconsistencias son informativas salvo que afecten la
    # asignación o la integridad del identificador.
    for issue in issues:
        add_alert(
            alerts,
            "ADVERTENCIA",
            "INTERNAL_CONSISTENCY",
            issue["detalle"],
            issue["escenario_id"],
        )

    print(f"Escenarios revisados    : {gdf[scenario_col].nunique()}")
    print(f"Inconsistencias         : {len(issues)}")

    return len(issues), len(gdf[scenario_col].unique()), pd.DataFrame(issues)


# =============================================================================
# COMPARACIÓN CON PROCESOS ANTERIORES
# =============================================================================

def compare_with_27(
    gdf: gpd.GeoDataFrame,
    cols: dict,
    alerts: list[dict],
) -> dict:

    result = {
        "archivo_27": INPUT_PROCESO_27.exists(),
        "proyectos_27": None,
        "escenarios_27": None,
        "asignaciones_modificadas": None,
    }

    if not INPUT_PROCESO_27.exists():
        print("Archivo proceso 27    : No disponible")
        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_27_MISSING",
            "No se encontró el archivo del proceso 27.",
        )
        return result

    try:
        old = gpd.read_parquet(INPUT_PROCESO_27)
        old_cols = resolve_columns(old)

        result["proyectos_27"] = len(old)

        if old_cols["escenario"]:
            result["escenarios_27"] = int(
                old[old_cols["escenario"]].nunique()
            )

        if (
            cols["proyecto"]
            and cols["escenario"]
            and old_cols["proyecto"]
            and old_cols["escenario"]
        ):
            a = gdf[[cols["proyecto"], cols["escenario"]]].copy()
            b = old[[old_cols["proyecto"], old_cols["escenario"]]].copy()

            a.columns = ["proyecto_id", "escenario_nuevo"]
            b.columns = ["proyecto_id", "escenario_original"]

            merged = a.merge(b, on="proyecto_id", how="inner")

            modified = int(
                (
                    merged["escenario_nuevo"].astype(str)
                    != merged["escenario_original"].astype(str)
                ).sum()
            )

            result["asignaciones_modificadas"] = modified

    except Exception as exc:
        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_27_READ_ERROR",
            f"No se pudo comparar contra proceso 27: {exc}",
        )

    return result


def compare_with_28(alerts: list[dict]) -> dict:
    result = {
        "archivo_28": INPUT_PROCESO_28.exists(),
        "score_referencia_28": None,
    }

    if not INPUT_PROCESO_28.exists():
        print("Archivo proceso 28    : No disponible")
        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_28_MISSING",
            "No se encontró la evaluación del proceso 28.",
        )
        return result

    try:
        df = pd.read_parquet(INPUT_PROCESO_28)

        candidates = [
            "score_global",
            "score_evaluacion",
            "score_validacion",
        ]

        for col in candidates:
            if col in df.columns:
                value = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(value):
                    result["score_referencia_28"] = float(value.mean())
                    break

    except Exception as exc:
        add_alert(
            alerts,
            "ADVERTENCIA",
            "PROCESS_28_READ_ERROR",
            f"No se pudo leer proceso 28: {exc}",
        )

    return result


def compare_with_29(alerts: list[dict]) -> dict:
    result = {
        "archivo_evaluacion_29": INPUT_PROCESO_29_EVAL.exists(),
        "archivo_resumen_29": INPUT_PROCESO_29_RESUMEN.exists(),
        "score_proceso_29": None,
    }

    if INPUT_PROCESO_29_EVAL.exists():
        try:
            df = pd.read_csv(INPUT_PROCESO_29_EVAL)
            for col in [
                "score_global",
                "score_global_optimizado",
                "score_optimizado",
            ]:
                if col in df.columns:
                    values = pd.to_numeric(
                        df[col], errors="coerce"
                    ).dropna()
                    if len(values):
                        result["score_proceso_29"] = float(values.mean())
                        break
        except Exception as exc:
            add_alert(
                alerts,
                "ADVERTENCIA",
                "PROCESS_29_READ_ERROR",
                f"No se pudo leer evaluación del proceso 29: {exc}",
            )

    if result["score_proceso_29"] is None and INPUT_PROCESO_29_RESUMEN.exists():
        try:
            df = pd.read_csv(INPUT_PROCESO_29_RESUMEN)

            for col in [
                "score_global_optimizado",
                "score_optimizado",
                "score_global",
            ]:
                if col in df.columns:
                    values = pd.to_numeric(
                        df[col], errors="coerce"
                    ).dropna()
                    if len(values):
                        result["score_proceso_29"] = float(values.iloc[-1])
                        break
        except Exception as exc:
            add_alert(
                alerts,
                "ADVERTENCIA",
                "PROCESS_29_SUMMARY_READ_ERROR",
                f"No se pudo leer resumen del proceso 29: {exc}",
            )

    return result


# =============================================================================
# SCORE GLOBAL
# =============================================================================

def calculate_global_score(metrics: dict) -> float:
    return clamp(
        PESOS["cobertura"] * metrics["cobertura"]
        + PESOS["tamano"] * metrics["tamano"]
        + PESOS["cohesion"] * metrics["cohesion"]
        + PESOS["indicadores"] * metrics["indicadores"]
        + PESOS["balance"] * metrics["balance"]
    )


def classify_score(score: float, blocking_errors: int) -> str:
    if blocking_errors > 0:
        return "RECHAZADO"

    if score >= SCORE_VALIDADO:
        return "VALIDADO"

    if score >= SCORE_OBSERVACIONES:
        return "VALIDADO_CON_OBSERVACIONES"

    return "REVISAR"


# =============================================================================
# DETALLE DE ESCENARIOS
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
        gdf.groupby(scenario_col, dropna=False)
        .size()
        .rename("cantidad_proyectos")
        .reset_index()
    )

    grouped = grouped.rename(columns={scenario_col: "escenario_id"})

    size_scores = []

    mean_count = grouped["cantidad_proyectos"].mean()

    for _, row in grouped.iterrows():
        n = row["cantidad_proyectos"]
        # Score de tamaño basado en cercanía al tamaño medio esperado.
        if mean_count > 0:
            deviation = abs(n - mean_count) / mean_count
            score = clamp(1.0 - deviation)
        else:
            score = 0.0

        size_scores.append(score)

    grouped["score_tamano"] = size_scores

    if not cohesion_detail.empty:
        grouped = grouped.merge(
            cohesion_detail,
            on="escenario_id",
            how="left",
        )
    else:
        grouped["distancia_media_m"] = np.nan
        grouped["distancia_maxima_m"] = np.nan
        grouped["score_cohesion"] = np.nan

    # Indicadores por escenario.
    indicator_cols = [
        c for c in INDICADORES_PRIORITARIOS
        if c in gdf.columns
    ]

    if indicator_cols:
        temp = gdf.copy()
        for c in indicator_cols:
            temp[c] = pd.to_numeric(temp[c], errors="coerce")

        ind = (
            temp.groupby(scenario_col)[indicator_cols]
            .mean(numeric_only=True)
            .reset_index()
        )

        ind = ind.rename(columns={scenario_col: "escenario_id"})

        # Cada indicador se normaliza globalmente antes de promediar.
        scores = []

        for c in indicator_cols:
            values = pd.to_numeric(gdf[c], errors="coerce")
            lo = values.min()
            hi = values.max()

            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                ind[c + "__norm"] = 1.0
            else:
                ind[c + "__norm"] = (
                    (ind[c] - lo) / (hi - lo)
                ).clip(0, 1)

            scores.append(c + "__norm")

        ind["score_indicadores"] = ind[scores].mean(axis=1)
        grouped = grouped.merge(
            ind[["escenario_id", "score_indicadores"]],
            on="escenario_id",
            how="left",
        )
    else:
        grouped["score_indicadores"] = 0.0

    grouped["score_validacion_escenario"] = (
        0.15 * grouped["score_tamano"].fillna(0)
        + 0.50 * grouped["score_cohesion"].fillna(0)
        + 0.35 * grouped["score_indicadores"].fillna(0)
    )

    grouped = grouped.sort_values(
        "score_validacion_escenario",
        ascending=False,
    ).reset_index(drop=True)

    grouped["ranking_validacion"] = np.arange(1, len(grouped) + 1)

    grouped["clasificacion_validacion"] = np.select(
        [
            grouped["score_validacion_escenario"] >= 0.75,
            grouped["score_validacion_escenario"] >= 0.55,
        ],
        [
            "VALIDADO",
            "VALIDADO_CON_OBSERVACIONES",
        ],
        default="REVISAR",
    )

    return grouped


# =============================================================================
# EXPORTACIÓN
# =============================================================================

def export_results(
    gdf: gpd.GeoDataFrame,
    scenario_detail: pd.DataFrame,
    comparison: pd.DataFrame,
    alerts_df: pd.DataFrame,
    summary: dict,
) -> None:

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Capa principal de resumen por escenario.
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

    alerts_df.to_csv(
        OUTPUT_ALERTAS,
        index=False,
        encoding="utf-8-sig",
    )

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
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
        f"30 - VALIDACIÓN FINAL DE ESCENARIOS TERRITORIALES AMBA - {VERSION}"
    )

    print(f"Proyecto : {PROJECT_ROOT}")
    print(f"Entrada  : {INPUT_OPTIMIZADO}")
    print(f"Salida   : {BASE_DIR}")

    print("\nCONFIGURACIÓN")
    print(f"  Versión                  : {VERSION}")
    print(f"  Escenarios válidos       : {MIN_ESCENARIOS} - {MAX_ESCENARIOS}")
    print(f"  Mínimo proyectos         : {MIN_PROYECTOS_ESCENARIO}")
    print(f"  CRS geográfico           : {CRS_GEOGRAFICO}")
    print(f"  CRS métrico              : {CRS_METRICO}")
    print("  Pesos")
    for key, value in PESOS.items():
        print(f"    {key:<20}: {value:.0%}")

    # -------------------------------------------------------------------------
    # 1. CARGA
    # -------------------------------------------------------------------------
    print_header("1. CARGANDO RESULTADO OPTIMIZADO DEL PROCESO 29")

    gdf = load_input()

    print(f"Entrada                  : {INPUT_OPTIMIZADO}")
    print(f"Registros                : {len(gdf)}")
    print(f"Columnas                 : {len(gdf.columns)}")
    print(f"CRS                      : {gdf.crs}")

    alerts: list[dict] = []

    cols = resolve_columns(gdf)

    # -------------------------------------------------------------------------
    # 2. ESTRUCTURA
    # -------------------------------------------------------------------------
    structure = validate_structure(gdf, cols, alerts)

    # -------------------------------------------------------------------------
    # 3. PROYECTOS
    # -------------------------------------------------------------------------
    projects = validate_projects(gdf, cols, alerts)

    # -------------------------------------------------------------------------
    # 4. ESCENARIOS
    # -------------------------------------------------------------------------
    scenarios = validate_scenarios(gdf, cols, alerts)
    counts = scenarios.get("conteos", pd.Series(dtype=int))

    # -------------------------------------------------------------------------
    # 5. COBERTURA
    # -------------------------------------------------------------------------
    coverage = validate_coverage(gdf, cols, alerts)

    # -------------------------------------------------------------------------
    # 6. TAMAÑO
    # -------------------------------------------------------------------------
    size_score, size_stats = validate_size(counts, alerts)

    # -------------------------------------------------------------------------
    # 7. COHESIÓN
    # -------------------------------------------------------------------------
    cohesion_score, cohesion_stats, cohesion_detail = validate_cohesion(
        gdf,
        cols,
        alerts,
    )

    # -------------------------------------------------------------------------
    # 8. INDICADORES
    # -------------------------------------------------------------------------
    indicator_score, indicators, indicator_detail = validate_indicators(
        gdf,
        alerts,
    )

    # -------------------------------------------------------------------------
    # 9. BALANCE
    # -------------------------------------------------------------------------
    balance_score, balance_stats = validate_balance(counts)

    # -------------------------------------------------------------------------
    # 10. CONSISTENCIA
    # -------------------------------------------------------------------------
    inconsistencies, reviewed, internal_detail = (
        validate_internal_consistency(
            gdf,
            cols,
            alerts,
        )
    )

    # -------------------------------------------------------------------------
    # 11. COMPARACIÓN 27
    # -------------------------------------------------------------------------
    print_header("11. COMPARANDO CONTRA PROCESO 27")

    comparison_27 = compare_with_27(
        gdf,
        cols,
        alerts,
    )

    print(f"Archivo disponible      : {comparison_27['archivo_27']}")
    print(f"Proyectos originales    : {comparison_27['proyectos_27']}")
    print(f"Escenarios originales   : {comparison_27['escenarios_27']}")
    print(
        "Asignaciones modificadas: "
        f"{comparison_27['asignaciones_modificadas']}"
    )

    # Los cambios de asignación son ESPERADOS del proceso 29.
    # No se convierten en error.
    if comparison_27["asignaciones_modificadas"] is not None:
        add_alert(
            alerts,
            "INFORMACION",
            "EXPECTED_REALLOCATION",
            (
                f"El proceso 29 modificó "
                f"{comparison_27['asignaciones_modificadas']} asignaciones "
                "respecto del proceso 27. Esto es un resultado esperado "
                "de la optimización y no constituye un error."
            ),
        )

    # -------------------------------------------------------------------------
    # 12. COMPARACIÓN 28
    # -------------------------------------------------------------------------
    print_header("12. COMPARANDO CONTRA PROCESO 28")

    comparison_28 = compare_with_28(alerts)

    print(f"Archivo disponible      : {comparison_28['archivo_28']}")
    print(
        f"Score referencia 28     : "
        f"{comparison_28['score_referencia_28']}"
    )

    # -------------------------------------------------------------------------
    # 13. COMPARACIÓN 29
    # -------------------------------------------------------------------------
    print_header("13. COMPARANDO CONTRA PROCESO 29")

    comparison_29 = compare_with_29(alerts)

    print(
        f"Evaluación disponible   : "
        f"{comparison_29['archivo_evaluacion_29']}"
    )
    print(
        f"Resumen disponible      : "
        f"{comparison_29['archivo_resumen_29']}"
    )
    print(
        f"Score proceso 29        : "
        f"{comparison_29['score_proceso_29']}"
    )

    # -------------------------------------------------------------------------
    # 14. SCORE FINAL
    # -------------------------------------------------------------------------
    print_header("14. CALCULANDO SCORE FINAL DE VALIDACIÓN")

    metrics = {
        "cobertura": coverage["cobertura"],
        "tamano": size_score,
        "cohesion": cohesion_score,
        "indicadores": indicator_score,
        "balance": balance_score,
    }

    score_global = calculate_global_score(metrics)

    # Sólo errores estructurales/bloqueantes rechazan automáticamente.
    blocking_codes = {
        "GEOM_NULL",
        "GEOM_EMPTY",
        "GEOM_INVALID",
        "PROJECT_ID_MISSING",
        "PROJECT_ID_NULL",
        "PROJECT_ID_DUPLICATE",
        "SCENARIO_ID_MISSING",
        "SCENARIO_COUNT",
        "SCENARIO_NULL",
        "SCENARIO_TOO_SMALL",
        "COVERAGE_INCOMPLETE",
        "CRS_MISSING",
        "NO_INDICATORS",
        "INDICATOR_NON_NUMERIC",
    }

    blocking_errors = sum(
        1
        for alert in alerts
        if alert["nivel"] == "ERROR"
        and alert["codigo"] in blocking_codes
    )

    classification = classify_score(
        score_global,
        blocking_errors,
    )

    print(f"Cobertura             : {metrics['cobertura']:.4f}")
    print(f"Tamaño                : {metrics['tamano']:.4f}")
    print(f"Cohesión              : {metrics['cohesion']:.4f}")
    print(f"Indicadores           : {metrics['indicadores']:.4f}")
    print(f"Balance               : {metrics['balance']:.4f}")
    print()
    print(f"SCORE VALIDACIÓN      : {score_global:.4f}")
    print(f"CLASIFICACIÓN SCORE    : {classification}")

    # -------------------------------------------------------------------------
    # 15. DETALLE
    # -------------------------------------------------------------------------
    print_header("15. CONSTRUYENDO DETALLE DE ESCENARIOS")

    scenario_detail = build_scenario_detail(
        gdf,
        cols,
        cohesion_detail,
    )

    if not scenario_detail.empty:
        print(
            scenario_detail[
                [
                    "escenario_id",
                    "cantidad_proyectos",
                    "score_tamano",
                    "score_cohesion",
                    "score_indicadores",
                    "score_validacion_escenario",
                    "ranking_validacion",
                    "clasificacion_validacion",
                ]
            ].to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # 16. DICTAMEN
    # -------------------------------------------------------------------------
    print_header("16. CONSTRUYENDO DICTAMEN FINAL")

    error_count = sum(
        1 for x in alerts if x["nivel"] == "ERROR"
    )
    warning_count = sum(
        1 for x in alerts if x["nivel"] == "ADVERTENCIA"
    )
    info_count = sum(
        1 for x in alerts if x["nivel"] == "INFORMACION"
    )

    if blocking_errors > 0:
        dictamen = "RECHAZADO"
        fundamento = (
            "Se detectaron errores estructurales o de integridad "
            "que impiden validar el resultado."
        )
    elif score_global >= SCORE_VALIDADO and warning_count == 0:
        dictamen = "VALIDADO"
        fundamento = (
            "El resultado optimizado cumple los controles estructurales, "
            "territoriales y de calidad definidos."
        )
    elif score_global >= SCORE_OBSERVACIONES:
        dictamen = "VALIDADO_CON_OBSERVACIONES"
        fundamento = (
            "El resultado cumple los controles estructurales y de cobertura, "
            "pero presenta observaciones de calidad que deben ser consideradas."
        )
    else:
        dictamen = "REVISAR"
        fundamento = (
            "El resultado no presenta errores estructurales bloqueantes, "
            "pero su calidad global se encuentra por debajo del umbral."
        )

    print(f"Dictamen                : {dictamen}")
    print(f"Fundamento              : {fundamento}")
    print(f"Errores bloqueantes     : {blocking_errors}")
    print(f"Errores totales         : {error_count}")
    print(f"Advertencias            : {warning_count}")
    print(f"Informaciones           : {info_count}")

    # -------------------------------------------------------------------------
    # 17. EXPORTACIÓN
    # -------------------------------------------------------------------------
    print_header("17. EXPORTANDO RESULTADOS")

    alerts_df = pd.DataFrame(
        alerts,
        columns=[
            "nivel",
            "codigo",
            "escenario_id",
            "mensaje",
        ],
    )

    comparison_rows = [
        {
            "proceso": "27",
            "archivo_disponible": comparison_27["archivo_27"],
            "proyectos": comparison_27["proyectos_27"],
            "escenarios": comparison_27["escenarios_27"],
            "asignaciones_modificadas": comparison_27[
                "asignaciones_modificadas"
            ],
            "score": None,
        },
        {
            "proceso": "28",
            "archivo_disponible": comparison_28["archivo_28"],
            "proyectos": None,
            "escenarios": None,
            "asignaciones_modificadas": None,
            "score": comparison_28["score_referencia_28"],
        },
        {
            "proceso": "29",
            "archivo_disponible": (
                comparison_29["archivo_evaluacion_29"]
                or comparison_29["archivo_resumen_29"]
            ),
            "proyectos": len(gdf),
            "escenarios": len(counts),
            "asignaciones_modificadas": None,
            "score": comparison_29["score_proceso_29"],
        },
        {
            "proceso": "30",
            "archivo_disponible": True,
            "proyectos": len(gdf),
            "escenarios": len(counts),
            "asignaciones_modificadas": None,
            "score": score_global,
        },
    ]

    comparison_df = pd.DataFrame(comparison_rows)

    summary = {
        "proceso": 30,
        "version": VERSION,
        "fecha_ejecucion": pd.Timestamp.now().isoformat(),
        "proyecto": str(PROJECT_ROOT),
        "entrada": str(INPUT_OPTIMIZADO),
        "salidas": {
            "parquet": str(OUTPUT_PARQUET),
            "csv": str(OUTPUT_CSV),
            "detalle": str(OUTPUT_DETALLE),
            "comparacion": str(OUTPUT_COMPARACION),
            "alertas": str(OUTPUT_ALERTAS),
        },
        "proyectos": {
            "total": projects["proyectos_totales"],
            "sin_id": projects["proyectos_sin_id"],
            "duplicados": projects["duplicados_proyecto"],
            "asignados": coverage["asignados"],
            "sin_escenario": coverage["sin_escenario"],
        },
        "escenarios": {
            "cantidad": scenarios["escenarios"],
            "min_proyectos": scenarios["min_proyectos"],
            "max_proyectos": scenarios["max_proyectos"],
        },
        "metricas": metrics,
        "score_global": score_global,
        "clasificacion_score": classification,
        "dictamen": dictamen,
        "fundamento": fundamento,
        "errores_bloqueantes": blocking_errors,
        "errores_totales": error_count,
        "advertencias": warning_count,
        "informaciones": info_count,
        "inconsistencias_internas": inconsistencies,
        "asignaciones_modificadas_vs_27": comparison_27[
            "asignaciones_modificadas"
        ],
        "indicadores_utilizados": indicators,
        "pesos": PESOS,
        "cohesion": cohesion_stats,
        "balance": balance_stats,
        "tamano": size_stats,
        "proceso_28": comparison_28,
        "proceso_29": comparison_29,
    }

    export_results(
        gdf,
        scenario_detail,
        comparison_df,
        alerts_df,
        summary,
    )

    print(f"Resumen Parquet         : {OUTPUT_PARQUET}")
    print(f"Resumen CSV             : {OUTPUT_CSV}")
    print(f"Detalle escenarios      : {OUTPUT_DETALLE}")
    print(f"Comparación procesos    : {OUTPUT_COMPARACION}")
    print(f"Alertas                 : {OUTPUT_ALERTAS}")
    print(f"Metadata                : {OUTPUT_JSON}")

    # -------------------------------------------------------------------------
    # 18. FINAL
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - start

    print_header("18. PROCESO 30 FINALIZADO")

    print(f"Proyectos evaluados      : {len(gdf)}")
    print(f"Escenarios evaluados     : {len(counts)}")
    print(f"Cobertura                : {coverage['cobertura']:.2%}")
    print(f"Score validación         : {score_global:.4f}")
    print(f"Clasificación            : {classification}")
    print(f"Dictamen final           : {dictamen}")
    print(f"Errores bloqueantes      : {blocking_errors}")
    print(f"Advertencias             : {warning_count}")
    print(f"Duración                 : {elapsed:.2f} segundos")

    print("\nDICTAMEN FINAL")
    print(f"  {dictamen}")
    print(f"  {fundamento}")

    if not alerts_df.empty:
        print("\nALERTAS")
        print(
            alerts_df[
                ["nivel", "codigo", "escenario_id", "mensaje"]
            ].to_string(index=False)
        )

    print("\n" + "=" * 88)

    return 0 if dictamen != "RECHAZADO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

out = Path("/mnt/data/30_validar_escenarios_territoriales_amba.py")
out.write_text(code, encoding="utf-8")

print(f"Archivo creado: {out}")
print(f"Líneas: {len(code.splitlines())}")
