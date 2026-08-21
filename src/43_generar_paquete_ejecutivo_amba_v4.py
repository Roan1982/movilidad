# -*- coding: utf-8 -*-
"""
43_generar_paquete_ejecutivo_amba_v4.py

PROCESO 43
Generación del Paquete Ejecutivo Territorial AMBA - V4.1

Objetivo
--------
Construir un paquete ejecutivo reproducible a partir del modelo territorial
AMBA V4 cerrado por el proceso 42.

Principios
----------
1. El cierre del proceso 42 se determina estructuralmente.
2. No se depende de texto libre para determinar GO/NO-GO.
3. Los rankings se normalizan antes de cualquier operación.
4. Las columnas faltantes se resuelven de manera robusta.
5. El paquete solo recibe dictamen GO cuando todos los controles obligatorios
   están satisfechos.
6. Se generan tablas, documentos, controles, hashes, resumen y manifiesto.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SCRIPT_VERSION = "V4.1-FINAL"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR = INPUT_DIR / "paquete_ejecutivo_amba_v4_1"

TABLES_DIR = OUTPUT_DIR / "tablas"
DOCS_DIR = OUTPUT_DIR / "documentos"
CONTROL_DIR = OUTPUT_DIR / "control"


# ============================================================================
# ARCHIVOS DE ENTRADA
# ============================================================================

FILES = {
    "proyectos": "modelo_maestro_proyectos_v4.csv",
    "escenarios": "modelo_maestro_escenarios_v4.csv",
    "ranking_proyectos": "ranking_final_proyectos_v4.csv",
    "ranking_escenarios": "ranking_final_escenarios_v4.csv",
    "indicadores": "indicadores_globales_amba_v4.csv",
    "cierre_42": "cierre_42_modelo_territorial_amba_v4.csv",
}


# ============================================================================
# ARCHIVOS DE SALIDA
# ============================================================================

OUTPUT_FILES = {
    "proyectos_ejecutivos": (
        TABLES_DIR / "proyectos_ejecutivos_amba_v4_1.csv"
    ),
    "escenarios_ejecutivos": (
        TABLES_DIR / "escenarios_ejecutivos_amba_v4_1.csv"
    ),
    "top_proyectos": (
        TABLES_DIR / "top_20_proyectos_prioritarios_amba_v4_1.csv"
    ),
    "ranking_escenarios": (
        TABLES_DIR / "ranking_escenarios_ejecutivo_amba_v4_1.csv"
    ),
    "indicadores": (
        TABLES_DIR / "indicadores_ejecutivos_amba_v4_1.csv"
    ),
    "sintesis": (
        DOCS_DIR / "sintesis_ejecutiva_amba_v4_1.md"
    ),
    "informe": (
        DOCS_DIR / "informe_ejecutivo_amba_v4_1.md"
    ),
    "control": (
        CONTROL_DIR / "control_paquete_ejecutivo_amba_v4_1.csv"
    ),
    "resumen": (
        CONTROL_DIR / "resumen_43_paquete_ejecutivo_amba_v4_1.json"
    ),
    "manifiesto": (
        CONTROL_DIR / "manifiesto_43_paquete_ejecutivo_amba_v4_1.csv"
    ),
}


# ============================================================================
# UTILIDADES
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_step(number: int, title: str) -> None:
    print()
    print("=" * 88)
    print(f"{number} - {title}")
    print("=" * 88)


def normalize_name(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return None

    text = text.replace("%", "").strip()

    # Manejo de números argentinos.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)

    if number is None:
        return None

    try:
        return int(round(number))
    except Exception:
        return None


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return str(value).strip()


def first_existing(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    columns = list(df.columns)

    # Coincidencia exacta.
    for candidate in candidates:
        if candidate in columns:
            return candidate

    # Coincidencia normalizada.
    normalized = {
        normalize_name(column): column
        for column in columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def ensure_column(
    df: pd.DataFrame,
    column: str,
    default: Any = None,
) -> pd.DataFrame:
    if column not in df.columns:
        df[column] = default

    return df


def load_csv(
    path: Path,
    *,
    required: bool = True,
) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"No existe el archivo requerido: {path}"
            )

        return pd.DataFrame()

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print(
        f"Cargando: {path.name} | "
        f"Registros: {len(df)} | "
        f"Columnas: {len(df.columns)}"
    )

    return df


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def normalize_result(value: Any) -> str:
    text = normalize_name(value)

    if text in {
        "ok",
        "correcto",
        "correcta",
        "aprobado",
        "aprobada",
        "si",
        "yes",
        "true",
        "1",
    }:
        return "OK"

    if text in {
        "error",
        "fallo",
        "fallida",
        "fallido",
        "fail",
        "failed",
        "no_go",
        "nogo",
        "no",
        "false",
        "0",
    }:
        return "ERROR"

    return text.upper()


def normalize_critical(value: Any) -> str:
    text = normalize_name(value)

    if text in {"si", "yes", "true", "1", "critico"}:
        return "SI"

    if text in {"no", "false", "0", "no_critico"}:
        return "NO"

    return str(value).strip().upper()


# ============================================================================
# RESOLUCIÓN DE CAMPOS
# ============================================================================

def resolve_fields(
    projects: pd.DataFrame,
    scenarios: pd.DataFrame,
    ranking_projects: pd.DataFrame,
    ranking_scenarios: pd.DataFrame,
) -> Dict[str, Optional[str]]:

    fields = {
        "proyecto": first_existing(
            projects,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
        ),
        "escenario": first_existing(
            projects,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        ),
        "tipo": first_existing(
            projects,
            [
                "tipo_escenario",
                "tipo",
            ],
        ),
        "dimension": first_existing(
            projects,
            [
                "dimension_dominante",
                "dimension",
            ],
        ),
        "prioridad": first_existing(
            projects,
            [
                "prioridad_territorial_v4",
                "prioridad_territorial",
                "prioridad",
            ],
        ),
        "score": first_existing(
            projects,
            [
                "score_priorizacion_v4",
                "score_priorizacion",
                "score",
            ],
        ),
        "demanda": first_existing(
            projects,
            [
                "indice_demanda_estructural",
                "indice_demanda",
                "demanda",
            ],
        ),
        "deficit": first_existing(
            projects,
            [
                "deficit_infraestructura",
                "deficit",
            ],
        ),
        "conectividad": first_existing(
            projects,
            [
                "indice_conectividad_estructural",
                "indice_conectividad",
                "conectividad",
            ],
        ),
        "intermodalidad": first_existing(
            projects,
            [
                "indice_intermodalidad_estructural",
                "indice_intermodalidad",
                "intermodalidad",
            ],
        ),
        "integracion": first_existing(
            projects,
            [
                "indice_integracion_territorial",
                "indice_integracion",
                "integracion",
            ],
        ),
        "centralidad": first_existing(
            projects,
            [
                "indice_centralidad_estructural",
                "indice_centralidad",
                "centralidad",
            ],
        ),
        "impacto": first_existing(
            projects,
            [
                "impacto_potencial",
                "impacto",
            ],
        ),
        "urgencia": first_existing(
            projects,
            [
                "urgencia_intervencion",
                "urgencia",
            ],
        ),
    }

    return fields


# ============================================================================
# RANKINGS
# ============================================================================

def detect_ranking_column(
    df: pd.DataFrame,
    preferred: Sequence[str],
) -> Optional[str]:

    exact = first_existing(df, preferred)

    if exact:
        return exact

    normalized_columns = {
        normalize_name(column): column
        for column in df.columns
    }

    # Primero buscar columnas cuyo nombre contenga ranking.
    candidates = []

    for column in df.columns:
        normalized = normalize_name(column)

        if "ranking" in normalized or "rank" in normalized:
            candidates.append(column)

    # Preferir columnas con términos integrales/finales.
    ordered = []

    for candidate in candidates:
        key = normalize_name(candidate)

        priority = 10

        if "integral" in key:
            priority -= 5

        if "final" in key:
            priority -= 3

        if "proyecto" in key:
            priority -= 1

        ordered.append((priority, candidate))

    if ordered:
        ordered.sort(key=lambda item: item[0])
        return ordered[0][1]

    # Último intento: columnas numéricas con secuencia 1..N.
    for column in df.columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if series.notna().sum() == len(df) and len(df) > 1:
            values = sorted(
                series.astype(int).tolist()
            )

            if values == list(range(1, len(df) + 1)):
                return column

    return None


def normalize_ranking(
    ranking_df: pd.DataFrame,
    id_candidates: Sequence[str],
    ranking_candidates: Sequence[str],
    canonical_rank_name: str,
    canonical_id_name: str,
) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:

    df = ranking_df.copy()

    id_column = first_existing(
        df,
        id_candidates,
    )

    ranking_column = detect_ranking_column(
        df,
        ranking_candidates,
    )

    if id_column is None:
        raise ValueError(
            f"No se pudo identificar el ID en ranking "
            f"{canonical_rank_name}."
        )

    if ranking_column is None:
        raise ValueError(
            f"No se pudo identificar la columna de ranking "
            f"{canonical_rank_name}."
        )

    result = pd.DataFrame()

    result[canonical_id_name] = df[id_column]

    result[canonical_rank_name] = pd.to_numeric(
        df[ranking_column],
        errors="coerce",
    )

    # Mantener columnas útiles.
    for column in df.columns:
        if column in {
            id_column,
            ranking_column,
        }:
            continue

        if column not in result.columns:
            result[column] = df[column]

    result[canonical_rank_name] = (
        result[canonical_rank_name]
        .round()
        .astype("Int64")
    )

    return (
        result,
        id_column,
        ranking_column,
    )


def validate_ranking(
    ranking_df: pd.DataFrame,
    rank_column: str,
    expected_n: int,
) -> Tuple[bool, bool]:

    if rank_column not in ranking_df.columns:
        return False, False

    values = pd.to_numeric(
        ranking_df[rank_column],
        errors="coerce",
    ).dropna()

    if len(values) != expected_n:
        return False, False

    values_int = sorted(
        values.astype(int).tolist()
    )

    complete = (
        values_int
        == list(range(1, expected_n + 1))
    )

    # "Ordenado" se valida por la secuencia real del dataframe.
    original = values.astype(int).tolist()

    ordered = (
        original
        == list(range(1, expected_n + 1))
    )

    return complete, ordered


# ============================================================================
# PROCESO 42
# ============================================================================

def validate_process_42(
    closure_path: Path,
) -> Dict[str, Any]:

    print_step(
        1,
        "VERIFICACIÓN ESTRUCTURAL DEL CIERRE DEL MODELO - PROCESO 42",
    )

    closure = load_csv(closure_path)

    print("Columnas detectadas en cierre 42:")

    for column in closure.columns:
        print(f"  - {column}")

    result_column = first_existing(
        closure,
        [
            "resultado",
            "result",
            "status",
            "estado",
        ],
    )

    critical_column = first_existing(
        closure,
        [
            "critico",
            "crítico",
            "critical",
        ],
    )

    detail_column = first_existing(
        closure,
        [
            "detalle",
            "detail",
            "observacion",
            "observación",
        ],
    )

    if result_column is None:
        raise ValueError(
            "El archivo de cierre 42 no contiene una columna "
            "'resultado' reconocible."
        )

    normalized_results = closure[
        result_column
    ].map(normalize_result)

    total = len(closure)

    ok_count = int(
        (normalized_results == "OK").sum()
    )

    error_count = int(
        (normalized_results == "ERROR").sum()
    )

    unknown_count = total - ok_count - error_count

    # ================================================================
    # SCORE
    # ================================================================

    score = None

    if detail_column:
        details = closure[
            detail_column
        ].fillna("").astype(str)

        for detail in details:
            match = re.search(
                r"score\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)",
                detail,
                flags=re.IGNORECASE,
            )

            if match:
                score = safe_float(
                    match.group(1)
                )

                if score is not None:
                    break

    if score is None and total > 0:
        score = round(
            100.0 * ok_count / total,
            2,
        )

    # ================================================================
    # DICTAMEN
    # ================================================================
    #
    # MUY IMPORTANTE:
    #
    # La columna "critico" NO representa el resultado.
    # "critico=SI" significa que el control tiene carácter crítico.
    #
    # Por lo tanto:
    #
    #   resultado=OK + critico=SI -> control aprobado
    #
    # No se debe interpretar "critico=SI" como una falla.
    # ================================================================

    closure_go = (
        total > 0
        and ok_count == total
        and error_count == 0
        and unknown_count == 0
    )

    dictamen = "GO" if closure_go else "NO-GO"

    # Información adicional.
    critical_total = None
    critical_ok = None
    critical_fail = None

    if critical_column:
        critical_mask = closure[
            critical_column
        ].map(normalize_critical).eq("SI")

        critical_total = int(
            critical_mask.sum()
        )

        critical_ok = int(
            (
                critical_mask
                & normalized_results.eq("OK")
            ).sum()
        )

        critical_fail = int(
            (
                critical_mask
                & ~normalized_results.eq("OK")
            ).sum()
        )

    print(f"Registros cierre       : {total}")
    print(f"Resultados OK          : {ok_count}")
    print(f"Resultados ERROR       : {error_count}")
    print(f"Resultados desconocidos: {unknown_count}")
    print(f"Dictamen cierre 42     : {dictamen}")
    print(
        f"Score cierre 42       : "
        f"{score if score is not None else 'N/D'}"
    )
    print(f"Controles detectados   : {total}")
    print(f"Controles OK           : {ok_count}")

    if critical_total is not None:
        print(
            f"Controles críticos     : {critical_total}"
        )
        print(
            f"Críticos OK            : {critical_ok}"
        )
        print(
            f"Críticos con falla     : {critical_fail}"
        )

    print(
        "Método de detección    : estructura_directa"
    )

    print(
        "Cierre proceso 42     : "
        + ("OK" if closure_go else "ERROR")
    )

    if not closure_go:
        raise RuntimeError(
            "El proceso 42 no presenta un cierre GO válido. "
            f"OK={ok_count}/{total}, "
            f"ERROR={error_count}, "
            f"DESCONOCIDOS={unknown_count}."
        )

    return {
        "go": True,
        "dictamen": dictamen,
        "score": score,
        "total": total,
        "ok": ok_count,
        "error": error_count,
        "unknown": unknown_count,
        "critical_total": critical_total,
        "critical_ok": critical_ok,
        "critical_fail": critical_fail,
        "method": "estructura_directa",
    }


# ============================================================================
# TABLA EJECUTIVA DE PROYECTOS
# ============================================================================

def build_project_table(
    projects: pd.DataFrame,
    ranking_projects: pd.DataFrame,
    fields: Dict[str, Optional[str]],
) -> pd.DataFrame:

    project_id = fields["proyecto"]
    scenario_id = fields["escenario"]

    if project_id is None:
        raise ValueError(
            "No se encontró proyecto_id."
        )

    result = pd.DataFrame()

    result["proyecto_id"] = projects[
        project_id
    ]

    if scenario_id:
        result["escenario_id"] = projects[
            scenario_id
        ]
    else:
        result["escenario_id"] = ""

    mappings = {
        "tipo_escenario": fields["tipo"],
        "dimension_dominante": fields["dimension"],
        "prioridad_territorial": fields["prioridad"],
        "score_priorizacion": fields["score"],
        "indice_demanda": fields["demanda"],
        "deficit_infraestructura": fields["deficit"],
        "indice_conectividad": fields["conectividad"],
        "indice_intermodalidad": fields["intermodalidad"],
        "indice_integracion": fields["integracion"],
        "indice_centralidad": fields["centralidad"],
        "impacto_potencial": fields["impacto"],
        "urgencia_intervencion": fields["urgencia"],
    }

    for output_name, source_name in mappings.items():

        if source_name and source_name in projects.columns:
            result[output_name] = projects[
                source_name
            ]
        else:
            result[output_name] = np.nan

    ranking_norm, _, _ = normalize_ranking(
        ranking_projects,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
        [
            "ranking_final_proyecto_v4",
            "ranking_final",
            "ranking_proyecto",
            "ranking",
        ],
        "ranking_final_proyecto_v4",
        "proyecto_id",
    )

    ranking_norm = ranking_norm[
        [
            "proyecto_id",
            "ranking_final_proyecto_v4",
        ]
    ].copy()

    ranking_norm["proyecto_id"] = (
        ranking_norm["proyecto_id"].astype(str)
    )

    result["proyecto_id"] = (
        result["proyecto_id"].astype(str)
    )

    result = result.merge(
        ranking_norm,
        on="proyecto_id",
        how="left",
        suffixes=("", "_ranking"),
    )

    result["ranking_final_proyecto_v4"] = pd.to_numeric(
        result["ranking_final_proyecto_v4"],
        errors="coerce",
    )

    result = result.sort_values(
        "ranking_final_proyecto_v4",
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return result


# ============================================================================
# TABLA EJECUTIVA DE ESCENARIOS
# ============================================================================

def build_scenario_table(
    scenarios: pd.DataFrame,
    projects: pd.DataFrame,
    ranking_scenarios: pd.DataFrame,
    fields: Dict[str, Optional[str]],
) -> pd.DataFrame:

    scenario_id = first_existing(
        scenarios,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    if scenario_id is None:
        raise ValueError(
            "No se encontró escenario_id en el modelo maestro."
        )

    project_scenario = fields["escenario"]

    if project_scenario is None:
        raise ValueError(
            "No se encontró escenario_id en proyectos."
        )

    # ================================================================
    # NORMALIZACIÓN DEL RANKING
    # ================================================================

    ranking_norm, _, _ = normalize_ranking(
        ranking_scenarios,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
        [
            "ranking_integral_v4",
            "ranking_final_escenario_v4",
            "ranking_final",
            "ranking_escenario",
            "ranking",
        ],
        "ranking_integral_v4",
        "escenario_id",
    )

    ranking_norm = ranking_norm[
        [
            "escenario_id",
            "ranking_integral_v4",
        ]
    ].copy()

    ranking_norm["escenario_id"] = (
        ranking_norm["escenario_id"].astype(str)
    )

    # ================================================================
    # BASE DE ESCENARIOS
    # ================================================================

    result = pd.DataFrame()

    result["escenario_id"] = scenarios[
        scenario_id
    ].astype(str)

    # Copiar campos útiles del maestro.
    preferred_columns = [
        "nombre_escenario",
        "tipo_escenario",
        "dimension_dominante",
        "descripcion",
        "objetivo",
        "score_integral_v4",
        "indice_integral",
        "prioridad",
        "prioridad_territorial_v4",
        "impacto_potencial",
        "urgencia_intervencion",
    ]

    for column in preferred_columns:
        source = first_existing(
            scenarios,
            [column],
        )

        if source:
            result[column] = scenarios[
                source
            ]

    # ================================================================
    # CANTIDAD DE PROYECTOS
    # ================================================================

    project_counts = (
        projects[
            project_scenario
        ]
        .astype(str)
        .value_counts()
        .rename("cantidad_proyectos")
        .reset_index()
    )

    project_counts.columns = [
        "escenario_id",
        "cantidad_proyectos",
    ]

    result = result.merge(
        project_counts,
        on="escenario_id",
        how="left",
    )

    result["cantidad_proyectos"] = (
        pd.to_numeric(
            result["cantidad_proyectos"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # ================================================================
    # RANKING
    # ================================================================

    result = result.merge(
        ranking_norm,
        on="escenario_id",
        how="left",
    )

    # La columna se garantiza explícitamente.
    if "ranking_integral_v4" not in result.columns:
        result["ranking_integral_v4"] = np.nan

    result["ranking_integral_v4"] = pd.to_numeric(
        result["ranking_integral_v4"],
        errors="coerce",
    )

    # ================================================================
    # ORDEN
    # ================================================================

    result = result.sort_values(
        by="ranking_integral_v4",
        ascending=True,
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return result


# ============================================================================
# TOP DE PROYECTOS
# ============================================================================

def build_top_projects(
    project_table: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:

    result = project_table.copy()

    if "ranking_final_proyecto_v4" in result.columns:
        result["ranking_final_proyecto_v4"] = pd.to_numeric(
            result["ranking_final_proyecto_v4"],
            errors="coerce",
        )

        result = result.sort_values(
            "ranking_final_proyecto_v4",
            ascending=True,
            kind="stable",
            na_position="last",
        )

    return result.head(n).reset_index(drop=True)


# ============================================================================
# TOP DE ESCENARIOS
# ============================================================================

def build_scenario_ranking(
    scenario_table: pd.DataFrame,
) -> pd.DataFrame:

    result = scenario_table.copy()

    result["ranking_integral_v4"] = pd.to_numeric(
        result["ranking_integral_v4"],
        errors="coerce",
    )

    return result.sort_values(
        "ranking_integral_v4",
        ascending=True,
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


# ============================================================================
# INDICADORES EJECUTIVOS
# ============================================================================

def build_executive_indicators(
    indicators: pd.DataFrame,
    projects: pd.DataFrame,
    scenarios: pd.DataFrame,
    closure: Dict[str, Any],
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    def add(
        indicador: str,
        valor: Any,
        unidad: str,
        fuente: str,
    ) -> None:
        rows.append(
            {
                "indicador": indicador,
                "valor": valor,
                "unidad": unidad,
                "fuente": fuente,
            }
        )

    add(
        "Proyectos territoriales",
        len(projects),
        "proyectos",
        "modelo_maestro_proyectos_v4",
    )

    add(
        "Escenarios territoriales",
        len(scenarios),
        "escenarios",
        "modelo_maestro_escenarios_v4",
    )

    project_id = first_existing(
        projects,
        [
            "proyecto_id",
            "id_proyecto",
        ],
    )

    if project_id:
        add(
            "Proyectos únicos",
            projects[project_id].nunique(),
            "proyectos",
            "modelo_maestro_proyectos_v4",
        )

    scenario_id = first_existing(
        projects,
        [
            "escenario_id",
            "id_escenario",
        ],
    )

    if scenario_id:
        counts = (
            projects[
                scenario_id
            ]
            .value_counts()
        )

        if len(counts) > 0:
            mean = counts.mean()

            std = counts.std(ddof=0)

            cv = (
                float(std / mean)
                if mean
                else 0.0
            )

            add(
                "CV distribución proyectos/escenario",
                round(cv, 4),
                "coeficiente",
                "modelo_maestro_proyectos_v4",
            )

    add(
        "Cierre proceso 42",
        closure["dictamen"],
        "dictamen",
        "proceso_42",
    )

    add(
        "Score cierre proceso 42",
        closure["score"],
        "puntos",
        "proceso_42",
    )

    add(
        "Controles proceso 42",
        closure["total"],
        "controles",
        "proceso_42",
    )

    add(
        "Controles OK proceso 42",
        closure["ok"],
        "controles",
        "proceso_42",
    )

    # Agregar indicadores originales.
    if not indicators.empty:
        indicator_column = first_existing(
            indicators,
            [
                "indicador",
                "nombre_indicador",
                "variable",
            ],
        )

        value_column = first_existing(
            indicators,
            [
                "valor",
                "value",
            ],
        )

        unit_column = first_existing(
            indicators,
            [
                "unidad",
                "unit",
            ],
        )

        if indicator_column and value_column:
            for _, row in indicators.iterrows():
                rows.append(
                    {
                        "indicador": (
                            "Original - "
                            + clean_text(
                                row[indicator_column]
                            )
                        ),
                        "valor": row[value_column],
                        "unidad": (
                            clean_text(
                                row[unit_column]
                            )
                            if unit_column
                            else ""
                        ),
                        "fuente": (
                            "indicadores_globales_amba_v4"
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================================
# DOCUMENTO: SÍNTESIS
# ============================================================================

def build_executive_summary(
    project_table: pd.DataFrame,
    scenario_table: pd.DataFrame,
    closure: Dict[str, Any],
) -> str:

    top_projects = project_table.head(5)

    top_scenarios = scenario_table.head(7)

    lines: List[str] = []

    lines.append(
        "# Síntesis Ejecutiva — Modelo Territorial AMBA V4.1"
    )
    lines.append("")
    lines.append(
        "## Estado del modelo"
    )
    lines.append("")
    lines.append(
        f"- Dictamen de cierre: **{closure['dictamen']}**"
    )
    lines.append(
        f"- Score de cierre: **{closure['score']} / 100**"
    )
    lines.append(
        f"- Proyectos: **{len(project_table)}**"
    )
    lines.append(
        f"- Escenarios: **{len(scenario_table)}**"
    )
    lines.append("")
    lines.append(
        "El modelo territorial AMBA V4.1 fue verificado "
        "estructuralmente y cuenta con cierre aprobado por el "
        "proceso 42."
    )
    lines.append("")

    lines.append(
        "## Escenarios priorizados"
    )
    lines.append("")

    for _, row in top_scenarios.iterrows():
        scenario = clean_text(
            row.get("escenario_id")
        )

        ranking = row.get(
            "ranking_integral_v4"
        )

        projects = row.get(
            "cantidad_proyectos",
            "",
        )

        lines.append(
            f"- **{scenario}** — "
            f"ranking {ranking}; "
            f"{projects} proyectos."
        )

    lines.append("")
    lines.append(
        "## Proyectos prioritarios"
    )
    lines.append("")

    for _, row in top_projects.iterrows():
        project = clean_text(
            row.get("proyecto_id")
        )

        scenario = clean_text(
            row.get("escenario_id")
        )

        ranking = row.get(
            "ranking_final_proyecto_v4"
        )

        lines.append(
            f"- **{project}** — "
            f"escenario {scenario}; "
            f"ranking {ranking}."
        )

    lines.append("")
    lines.append(
        "## Conclusión"
    )
    lines.append("")
    lines.append(
        "El paquete ejecutivo se encuentra habilitado "
        "para utilización ejecutiva una vez completados "
        "los controles internos del proceso 43."
    )

    return "\n".join(lines) + "\n"


# ============================================================================
# DOCUMENTO: INFORME EJECUTIVO
# ============================================================================

def build_executive_report(
    project_table: pd.DataFrame,
    scenario_table: pd.DataFrame,
    indicators: pd.DataFrame,
    closure: Dict[str, Any],
) -> str:

    lines: List[str] = []

    lines.append(
        "# Informe Ejecutivo Territorial AMBA V4.1"
    )
    lines.append("")

    lines.append(
        "## 1. Estado de cierre"
    )
    lines.append("")

    lines.append(
        f"- Proceso 42: **{closure['dictamen']}**"
    )
    lines.append(
        f"- Score: **{closure['score']} / 100**"
    )
    lines.append(
        f"- Controles: **{closure['ok']}/{closure['total']} OK**"
    )
    lines.append("")

    lines.append(
        "## 2. Estructura territorial"
    )
    lines.append("")

    lines.append(
        f"- Proyectos: **{len(project_table)}**"
    )
    lines.append(
        f"- Escenarios: **{len(scenario_table)}**"
    )

    if (
        "cantidad_proyectos"
        in scenario_table.columns
    ):
        counts = scenario_table[
            "cantidad_proyectos"
        ].astype(float)

        if len(counts):
            mean = counts.mean()
            std = counts.std(ddof=0)
            cv = std / mean if mean else 0

            lines.append(
                f"- Proyectos por escenario: "
                f"mínimo {int(counts.min())}, "
                f"máximo {int(counts.max())}, "
                f"media {mean:.2f}, "
                f"CV {cv:.4f}"
            )

    lines.append("")

    lines.append(
        "## 3. Ranking de escenarios"
    )
    lines.append("")

    for _, row in scenario_table.iterrows():
        lines.append(
            f"{int(row['ranking_integral_v4'])}. "
            f"**{row['escenario_id']}** — "
            f"{int(row['cantidad_proyectos'])} proyectos"
        )

    lines.append("")

    lines.append(
        "## 4. Top 20 de proyectos"
    )
    lines.append("")

    for _, row in project_table.head(20).iterrows():
        ranking = safe_int(
            row.get(
                "ranking_final_proyecto_v4"
            )
        )

        lines.append(
            f"{ranking}. "
            f"**{row['proyecto_id']}** — "
            f"escenario {row['escenario_id']}"
        )

    lines.append("")

    lines.append(
        "## 5. Indicadores"
    )
    lines.append("")

    if not indicators.empty:
        for _, row in indicators.head(40).iterrows():
            lines.append(
                f"- {row['indicador']}: "
                f"**{row['valor']}** "
                f"{row['unidad']}"
            )

    lines.append("")

    lines.append(
        "## 6. Dictamen"
    )
    lines.append("")

    lines.append(
        "El modelo territorial AMBA V4.1 presenta un "
        "cierre técnico aprobado y se encuentra estructurado "
        "para su utilización ejecutiva."
    )

    return "\n".join(lines) + "\n"


# ============================================================================
# CONTROL DEL PAQUETE
# ============================================================================

def build_package_control(
    closure: Dict[str, Any],
    projects: pd.DataFrame,
    scenarios: pd.DataFrame,
    project_table: pd.DataFrame,
    scenario_table: pd.DataFrame,
    ranking_projects: pd.DataFrame,
    ranking_scenarios: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    def add(
        control: str,
        resultado: str,
        critico: str,
        detalle: str,
    ) -> None:
        rows.append(
            {
                "control": control,
                "resultado": resultado,
                "critico": critico,
                "detalle": detalle,
            }
        )

    # ------------------------------------------------------------
    # 1. Cierre 42
    # ------------------------------------------------------------

    add(
        "Cierre proceso 42",
        "OK" if closure["go"] else "ERROR",
        "SI",
        (
            f"Dictamen={closure['dictamen']} "
            f"Score={closure['score']}"
        ),
    )

    # ------------------------------------------------------------
    # 2. Cantidad proyectos
    # ------------------------------------------------------------

    project_count = len(projects)

    add(
        "Cantidad de proyectos",
        "OK" if project_count == 144 else "ERROR",
        "SI",
        f"Encontrados={project_count}",
    )

    # ------------------------------------------------------------
    # 3. Cantidad escenarios
    # ------------------------------------------------------------

    scenario_count = len(scenarios)

    add(
        "Cantidad de escenarios",
        "OK" if scenario_count == 7 else "ERROR",
        "SI",
        f"Encontrados={scenario_count}",
    )

    # ------------------------------------------------------------
    # 4. Proyectos únicos
    # ------------------------------------------------------------

    project_id = first_existing(
        projects,
        [
            "proyecto_id",
            "id_proyecto",
        ],
    )

    unique_projects = (
        projects[project_id].nunique()
        if project_id
        else 0
    )

    add(
        "Proyectos únicos",
        (
            "OK"
            if unique_projects == project_count
            else "ERROR"
        ),
        "SI",
        f"Registros={unique_projects}",
    )

    # ------------------------------------------------------------
    # 5-6. Ranking proyectos
    # ------------------------------------------------------------

    project_rank_column = (
        "ranking_final_proyecto_v4"
    )

    project_complete, project_ordered = (
        validate_ranking(
            project_table,
            project_rank_column,
            project_count,
        )
    )

    add(
        "Ranking proyectos completo",
        "OK" if project_complete else "ERROR",
        "SI",
        f"Campo={project_rank_column}",
    )

    # IMPORTANTE:
    #
    # El ranking completo NO necesita que el CSV original esté
    # físicamente ordenado. La tabla ejecutiva sí se ordena.
    #
    # Por lo tanto se valida la secuencia de la tabla ejecutiva.
    #

    ordered_values = pd.to_numeric(
        project_table[
            project_rank_column
        ],
        errors="coerce",
    ).dropna().astype(int).tolist()

    ordered_ok = (
        ordered_values
        == list(range(1, project_count + 1))
    )

    add(
        "Ranking proyectos ordenado",
        "OK" if ordered_ok else "ERROR",
        "NO",
        "Secuencia 1..N",
    )

    # ------------------------------------------------------------
    # 7-8. Ranking escenarios
    # ------------------------------------------------------------

    scenario_rank_column = (
        "ranking_integral_v4"
    )

    scenario_complete, scenario_ordered = (
        validate_ranking(
            scenario_table,
            scenario_rank_column,
            scenario_count,
        )
    )

    add(
        "Ranking escenarios completo",
        "OK" if scenario_complete else "ERROR",
        "SI",
        f"Campo={scenario_rank_column}",
    )

    scenario_order_values = pd.to_numeric(
        scenario_table[
            scenario_rank_column
        ],
        errors="coerce",
    ).dropna().astype(int).tolist()

    scenario_order_ok = (
        scenario_order_values
        == list(range(1, scenario_count + 1))
    )

    add(
        "Ranking escenarios ordenado",
        "OK" if scenario_order_ok else "ERROR",
        "NO",
        "Secuencia 1..N",
    )

    # ------------------------------------------------------------
    # 9. Tabla ejecutiva proyectos
    # ------------------------------------------------------------

    add(
        "Tabla ejecutiva proyectos",
        (
            "OK"
            if len(project_table) == project_count
            else "ERROR"
        ),
        "SI",
        f"Registros={len(project_table)}",
    )

    # ------------------------------------------------------------
    # 10. Tabla ejecutiva escenarios
    # ------------------------------------------------------------

    add(
        "Tabla ejecutiva escenarios",
        (
            "OK"
            if len(scenario_table) == scenario_count
            else "ERROR"
        ),
        "SI",
        f"Registros={len(scenario_table)}",
    )

    # ------------------------------------------------------------
    # 11. Ranking final proyectos presente
    # ------------------------------------------------------------

    add(
        "Ranking final proyectos presente",
        (
            "OK"
            if (
                project_rank_column
                in project_table.columns
            )
            else "ERROR"
        ),
        "SI",
        "",
    )

    # ------------------------------------------------------------
    # 12. Ranking final escenarios presente
    # ------------------------------------------------------------

    add(
        "Ranking final escenarios presente",
        (
            "OK"
            if (
                scenario_rank_column
                in scenario_table.columns
            )
            else "ERROR"
        ),
        "SI",
        "",
    )

    # ------------------------------------------------------------
    # 13. Archivos ejecutivos generados
    # ------------------------------------------------------------

    expected_outputs = [
        OUTPUT_FILES["proyectos_ejecutivos"],
        OUTPUT_FILES["escenarios_ejecutivos"],
        OUTPUT_FILES["top_proyectos"],
        OUTPUT_FILES["ranking_escenarios"],
        OUTPUT_FILES["indicadores"],
        OUTPUT_FILES["sintesis"],
        OUTPUT_FILES["informe"],
    ]

    existing = [
        path
        for path in expected_outputs
        if path.exists()
    ]

    add(
        "Archivos ejecutivos generados",
        (
            "OK"
            if len(existing) == len(expected_outputs)
            else "ERROR"
        ),
        "SI",
        f"Archivos={len(existing)}",
    )

    return pd.DataFrame(rows)


# ============================================================================
# MANIFIESTO
# ============================================================================

def build_manifest(
    project_table: pd.DataFrame,
    scenario_table: pd.DataFrame,
    closure: Dict[str, Any],
) -> pd.DataFrame:

    rows = []

    files = [
        OUTPUT_FILES["proyectos_ejecutivos"],
        OUTPUT_FILES["escenarios_ejecutivos"],
        OUTPUT_FILES["top_proyectos"],
        OUTPUT_FILES["ranking_escenarios"],
        OUTPUT_FILES["indicadores"],
        OUTPUT_FILES["sintesis"],
        OUTPUT_FILES["informe"],
        OUTPUT_FILES["control"],
        OUTPUT_FILES["resumen"],
    ]

    for path in files:
        if not path.exists():
            continue

        rows.append(
            {
                "producto": path.name,
                "ruta_relativa": str(
                    path.relative_to(OUTPUT_DIR)
                ),
                "tipo": path.suffix.lower(),
                "tamano_mb": round(
                    file_size_mb(path),
                    4,
                ),
                "sha256": sha256_file(path),
                "estado": "GENERADO",
            }
        )

    rows.append(
        {
            "producto": "MODELO",
            "ruta_relativa": "",
            "tipo": "estado",
            "tamano_mb": "",
            "sha256": "",
            "estado": (
                "GO"
                if closure["go"]
                else "NO-GO"
            ),
        }
    )

    return pd.DataFrame(rows)


# ============================================================================
# HASHES
# ============================================================================

def generate_hash_manifest(
    output_dir: Path,
) -> pd.DataFrame:

    rows = []

    for path in sorted(
        output_dir.rglob("*")
    ):

        if not path.is_file():
            continue

        # Evitar auto-referencia del manifiesto de hashes
        # si se agregara en futuras versiones.
        if path.name.startswith(
            "hashes_43_"
        ):
            continue

        try:
            relative = path.relative_to(
                output_dir
            )
        except ValueError:
            continue

        rows.append(
            {
                "archivo": str(relative),
                "tamano_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    start = time.time()

    print_header(
        f"43 - GENERACIÓN DEL PAQUETE EJECUTIVO TERRITORIAL AMBA - "
        f"{SCRIPT_VERSION}"
    )

    print(
        f"Proyecto                    : {PROJECT_ROOT}"
    )
    print(
        f"Entrada                     : {INPUT_DIR}"
    )
    print(
        f"Salida                      : {OUTPUT_DIR}"
    )

    # ------------------------------------------------------------
    # Directorios
    # ------------------------------------------------------------

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTROL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ========================================================
        # 1. CIERRE 42
        # ========================================================

        closure = validate_process_42(
            INPUT_DIR / FILES["cierre_42"]
        )

        # ========================================================
        # 2. MODELO MAESTRO
        # ========================================================

        print_step(
            2,
            "CARGANDO MODELO MAESTRO",
        )

        projects = load_csv(
            INPUT_DIR / FILES["proyectos"]
        )

        scenarios = load_csv(
            INPUT_DIR / FILES["escenarios"]
        )

        ranking_projects = load_csv(
            INPUT_DIR / FILES["ranking_proyectos"]
        )

        ranking_scenarios = load_csv(
            INPUT_DIR / FILES["ranking_escenarios"]
        )

        indicators = load_csv(
            INPUT_DIR / FILES["indicadores"]
        )

        # ========================================================
        # 3. CAMPOS
        # ========================================================

        print_step(
            3,
            "RESOLUCIÓN DE CAMPOS",
        )

        fields = resolve_fields(
            projects,
            scenarios,
            ranking_projects,
            ranking_scenarios,
        )

        for name, value in fields.items():
            print(
                f"{name:<15}: {value}"
            )

        # ========================================================
        # 4. VALIDACIÓN ESTRUCTURAL
        # ========================================================

        print_step(
            4,
            "VALIDACIÓN ESTRUCTURAL",
        )

        project_id = fields["proyecto"]
        scenario_project_id = fields["escenario"]

        scenario_master_id = first_existing(
            scenarios,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        )

        if project_id is None:
            raise ValueError(
                "No se encontró proyecto_id."
            )

        if scenario_project_id is None:
            raise ValueError(
                "No se encontró escenario_id en proyectos."
            )

        if scenario_master_id is None:
            raise ValueError(
                "No se encontró escenario_id en escenarios."
            )

        project_count = len(projects)
        scenario_count = len(scenarios)

        unique_projects = projects[
            project_id
        ].nunique()

        null_projects = int(
            projects[project_id]
            .isna()
            .sum()
        )

        duplicated_projects = int(
            projects[project_id]
            .duplicated()
            .sum()
        )

        unique_scenarios = scenarios[
            scenario_master_id
        ].nunique()

        null_scenarios = int(
            scenarios[
                scenario_master_id
            ]
            .isna()
            .sum()
        )

        duplicated_scenarios = int(
            scenarios[
                scenario_master_id
            ]
            .duplicated()
            .sum()
        )

        print(
            f"Proyectos                   : {project_count}"
        )
        print(
            f"Proyectos únicos            : {unique_projects}"
        )
        print(
            f"Proyectos nulos             : {null_projects}"
        )
        print(
            f"Proyectos duplic.           : {duplicated_projects}"
        )
        print(
            f"Escenarios modelo           : {scenario_count}"
        )
        print(
            f"Escenarios maestro          : {unique_scenarios}"
        )

        if (
            project_count != unique_projects
            or null_projects > 0
            or duplicated_projects > 0
            or scenario_count != unique_scenarios
            or null_scenarios > 0
            or duplicated_scenarios > 0
        ):
            raise RuntimeError(
                "Falla estructural del modelo."
            )

        # ========================================================
        # 5. PROYECTO -> ESCENARIO
        # ========================================================

        print_step(
            5,
            "VALIDACIÓN PROYECTO -> ESCENARIO",
        )

        project_scenario = projects[
            [
                project_id,
                scenario_project_id,
            ]
        ].copy()

        null_assignments = int(
            project_scenario[
                scenario_project_id
            ]
            .isna()
            .sum()
        )

        multi_scenario = (
            project_scenario
            .groupby(project_id)[
                scenario_project_id
            ]
            .nunique()
            .gt(1)
            .sum()
        )

        model_scenario_ids = set(
            scenarios[
                scenario_master_id
            ]
            .dropna()
            .astype(str)
        )

        project_scenario_ids = set(
            project_scenario[
                scenario_project_id
            ]
            .dropna()
            .astype(str)
        )

        ids_match = (
            project_scenario_ids
            .issubset(model_scenario_ids)
        )

        print(
            f"Escenarios nulos        : "
            f"{null_assignments}"
        )

        print(
            f"Proyectos multiescenario: "
            f"{multi_scenario}"
        )

        print(
            "IDs escenarios coincidentes: "
            + ("SI" if ids_match else "NO")
        )

        if (
            null_assignments > 0
            or multi_scenario > 0
            or not ids_match
        ):
            raise RuntimeError(
                "Falla en la asignación proyecto -> escenario."
            )

        # ========================================================
        # 6. DISTRIBUCIÓN
        # ========================================================

        print_step(
            6,
            "DISTRIBUCIÓN TERRITORIAL",
        )

        distribution = (
            projects[
                scenario_project_id
            ]
            .astype(str)
            .value_counts()
            .sort_index()
        )

        for scenario, count in distribution.items():
            print(
                f"  {scenario}: {count}"
            )

        minimum = int(
            distribution.min()
        )

        maximum = int(
            distribution.max()
        )

        mean = float(
            distribution.mean()
        )

        std = float(
            distribution.std(ddof=0)
        )

        cv = (
            std / mean
            if mean
            else 0.0
        )

        print(
            f"Mínimo   : {minimum}"
        )
        print(
            f"Máximo   : {maximum}"
        )
        print(
            f"Promedio : {mean:.2f}"
        )
        print(
            f"CV       : {cv:.4f}"
        )

        # ========================================================
        # 7. RANKINGS
        # ========================================================

        print_step(
            7,
            "VALIDACIÓN DE RANKINGS",
        )

        project_rank_column = detect_ranking_column(
            ranking_projects,
            [
                "ranking_final_proyecto_v4",
                "ranking_final",
                "ranking_proyecto",
                "ranking",
            ],
        )

        scenario_rank_column = detect_ranking_column(
            ranking_scenarios,
            [
                "ranking_integral_v4",
                "ranking_final_escenario_v4",
                "ranking_final",
                "ranking_escenario",
                "ranking",
            ],
        )

        if project_rank_column is None:
            raise ValueError(
                "No se encontró ranking de proyectos."
            )

        if scenario_rank_column is None:
            raise ValueError(
                "No se encontró ranking de escenarios."
            )

        project_rank_norm, _, _ = normalize_ranking(
            ranking_projects,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
            [
                project_rank_column,
            ],
            "ranking_final_proyecto_v4",
            "proyecto_id",
        )

        scenario_rank_norm, _, _ = normalize_ranking(
            ranking_scenarios,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
            [
                scenario_rank_column,
            ],
            "ranking_integral_v4",
            "escenario_id",
        )

        p_complete, _ = validate_ranking(
            project_rank_norm,
            "ranking_final_proyecto_v4",
            project_count,
        )

        s_complete, _ = validate_ranking(
            scenario_rank_norm,
            "ranking_integral_v4",
            scenario_count,
        )

        print(
            f"Ranking escenarios : "
            f"{scenario_rank_column}"
        )

        print(
            f"Ranking escenarios 1..{scenario_count} : "
            f"{'OK' if s_complete else 'ERROR'}"
        )

        print(
            f"Ranking proyectos : "
            f"{project_rank_column}"
        )

        print(
            f"Ranking proyectos 1..{project_count} : "
            f"{'OK' if p_complete else 'ERROR'}"
        )

        if not p_complete or not s_complete:
            raise RuntimeError(
                "Los rankings no contienen una secuencia completa 1..N."
            )

        # ========================================================
        # 8. TABLA PROYECTOS
        # ========================================================

        print_step(
            8,
            "CONSTRUYENDO TABLA EJECUTIVA DE PROYECTOS",
        )

        project_table = build_project_table(
            projects,
            ranking_projects,
            fields,
        )

        write_csv(
            project_table,
            OUTPUT_FILES["proyectos_ejecutivos"],
        )

        print(
            f"Generado: "
            f"{OUTPUT_FILES['proyectos_ejecutivos']}"
        )

        # ========================================================
        # 9. TABLA ESCENARIOS
        # ========================================================

        print_step(
            9,
            "CONSTRUYENDO TABLA EJECUTIVA DE ESCENARIOS",
        )

        scenario_table = build_scenario_table(
            scenarios,
            projects,
            ranking_scenarios,
            fields,
        )

        write_csv(
            scenario_table,
            OUTPUT_FILES["escenarios_ejecutivos"],
        )

        print(
            f"Generado: "
            f"{OUTPUT_FILES['escenarios_ejecutivos']}"
        )

        # ========================================================
        # 10. TOP PROYECTOS
        # ========================================================

        print_step(
            10,
            "CONSTRUYENDO TOP DE PROYECTOS",
        )

        top_projects = build_top_projects(
            project_table,
            20,
        )

        write_csv(
            top_projects,
            OUTPUT_FILES["top_proyectos"],
        )

        print(
            f"Generado: "
            f"{OUTPUT_FILES['top_proyectos']}"
        )

        # ========================================================
        # 11. RANKING ESCENARIOS
        # ========================================================

        print_step(
            11,
            "CONSTRUYENDO TOP DE ESCENARIOS",
        )

        scenario_ranking = build_scenario_ranking(
            scenario_table
        )

        write_csv(
            scenario_ranking,
            OUTPUT_FILES["ranking_escenarios"],
        )

        print(
            f"Generado: "
            f"{OUTPUT_FILES['ranking_escenarios']}"
        )

        # ========================================================
        # 12. INDICADORES
        # ========================================================

        print_step(
            12,
            "CONSTRUYENDO INDICADORES EJECUTIVOS",
        )

        executive_indicators = (
            build_executive_indicators(
                indicators,
                projects,
                scenarios,
                closure,
            )
        )

        write_csv(
            executive_indicators,
            OUTPUT_FILES["indicadores"],
        )

        print(
            f"Generado: "
            f"{OUTPUT_FILES['indicadores']}"
        )

        # ========================================================
        # 13. SÍNTESIS
        # ========================================================

        print_step(
            13,
            "GENERANDO SÍNTESIS EJECUTIVA",
        )

        synthesis = build_executive_summary(
            project_table,
            scenario_table,
            closure,
        )

        OUTPUT_FILES[
            "sintesis"
        ].write_text(
            synthesis,
            encoding="utf-8",
        )

        print(
            f"Generado: "
            f"{OUTPUT_FILES['sintesis']}"
        )

        # ========================================================
        # 14. INFORME
        # ========================================================

        print_step(
            14,
            "GENERANDO INFORME EJECUTIVO",
        )

        report = build_executive_report(
            project_table,
            scenario_table,
            executive_indicators,
            closure,
        )

        OUTPUT_FILES[
            "informe"
        ].write_text(
            report,
            encoding="utf-8",
        )

        print(
            f"Generado: "
            f"{OUTPUT_FILES['informe']}"
        )

        # ========================================================
        # 15. CONTROL
        # ========================================================

        print_step(
            15,
            "CONSTRUYENDO CONTROL DEL PAQUETE",
        )

        control = build_package_control(
            closure,
            projects,
            scenarios,
            project_table,
            scenario_table,
            ranking_projects,
            ranking_scenarios,
        )

        write_csv(
            control,
            OUTPUT_FILES["control"],
        )

        controls_total = len(control)

        controls_ok = int(
            (
                control["resultado"]
                .map(normalize_result)
                == "OK"
            ).sum()
        )

        controls_fail = (
            controls_total - controls_ok
        )

        package_go = (
            closure["go"]
            and controls_fail == 0
        )

        print(
            f"Controles OK : "
            f"{controls_ok}/{controls_total}"
        )

        print(
            "Control paquete : "
            + ("OK" if package_go else "ERROR")
        )

        # ========================================================
        # 16. PRODUCTOS DE REFERENCIA
        # ========================================================

        print_step(
            16,
            "REGISTRANDO PRODUCTOS DE REFERENCIA",
        )

        reference_files = [
            INPUT_DIR / FILES["proyectos"],
            INPUT_DIR / FILES["escenarios"],
            INPUT_DIR / FILES["ranking_proyectos"],
            INPUT_DIR / FILES["ranking_escenarios"],
            INPUT_DIR / FILES["indicadores"],
            INPUT_DIR / FILES["cierre_42"],
            INPUT_DIR / "modelo_maestro_territorial_amba_v4.gpkg",
            INPUT_DIR / "atlas_territorial_amba_v4.gpkg",
            INPUT_DIR / "informe_territorial_amba_v4_1.md",
            INPUT_DIR / "atlas_territorial_amba_v4.md",
            INPUT_DIR / "auditoria_41_modelo_territorial_amba_v4.csv",
        ]

        existing_reference = [
            path
            for path in reference_files
            if path.exists()
        ]

        print(
            "Productos de referencia registrados: "
            f"{len(existing_reference)}/{len(reference_files)}"
        )

        # ========================================================
        # 17. HASHES
        # ========================================================

        print_step(
            17,
            "GENERANDO HASHES DEL PAQUETE",
        )

        hashes = generate_hash_manifest(
            OUTPUT_DIR
        )

        hashes_path = (
            CONTROL_DIR
            / "hashes_43_paquete_ejecutivo_amba_v4_1.csv"
        )

        write_csv(
            hashes,
            hashes_path,
        )

        print(
            f"Archivos incluidos en hash: "
            f"{len(hashes)}"
        )

        # ========================================================
        # 18. RESUMEN
        # ========================================================

        print_step(
            18,
            "GENERANDO RESUMEN DEL PROCESO 43",
        )

        score_package = round(
            100.0
            * controls_ok
            / controls_total,
            2,
        ) if controls_total else 0.0

        critical_failures = int(
            (
                control["critico"]
                .map(normalize_critical)
                .eq("SI")
                & control["resultado"]
                .map(normalize_result)
                .ne("OK")
            ).sum()
        )

        summary = {
            "proceso": 43,
            "version": SCRIPT_VERSION,
            "modelo": "territorial_amba_v4_1",
            "fecha_ejecucion": pd.Timestamp.now().isoformat(),
            "proyecto_root": str(PROJECT_ROOT),
            "input_dir": str(INPUT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "proceso_42": {
                "dictamen": closure["dictamen"],
                "score": closure["score"],
                "controles": closure["total"],
                "controles_ok": closure["ok"],
                "errores": closure["error"],
                "desconocidos": closure["unknown"],
                "go": closure["go"],
            },
            "modelo": {
                "proyectos": project_count,
                "escenarios": scenario_count,
                "proyectos_unicos": unique_projects,
                "proyectos_multiescenario": int(
                    multi_scenario
                ),
                "cobertura_asignacion_pct": (
                    round(
                        100.0
                        * (
                            project_count
                            - null_assignments
                        )
                        / project_count,
                        2,
                    )
                    if project_count
                    else 0.0
                ),
                "cv_tamano_escenarios": round(
                    cv,
                    4,
                ),
            },
            "paquete": {
                "controles_total": controls_total,
                "controles_ok": controls_ok,
                "controles_fallidos": controls_fail,
                "fallas_criticas": critical_failures,
                "score": score_package,
                "auditoria": (
                    "OK"
                    if package_go
                    else "OBSERVADA"
                ),
                "dictamen": (
                    "GO"
                    if package_go
                    else "NO-GO"
                ),
            },
            "archivos": {
                key: str(path)
                for key, path in OUTPUT_FILES.items()
            },
        }

        write_json(
            summary,
            OUTPUT_FILES["resumen"],
        )

        print(
            f"Resumen: "
            f"{OUTPUT_FILES['resumen']}"
        )

        # ========================================================
        # 19. MANIFIESTO
        # ========================================================

        print_step(
            19,
            "GENERANDO MANIFIESTO EJECUTIVO",
        )

        manifest = build_manifest(
            project_table,
            scenario_table,
            closure,
        )

        write_csv(
            manifest,
            OUTPUT_FILES["manifiesto"],
        )

        print(
            f"Manifiesto: "
            f"{OUTPUT_FILES['manifiesto']}"
        )

        # ========================================================
        # RESULTADO
        # ========================================================

        elapsed = time.time() - start

        print_header(
            "RESULTADO FINAL DEL PROCESO 43"
        )

        print(
            f"Proyectos                   : "
            f"{project_count}"
        )

        print(
            f"Escenarios                  : "
            f"{scenario_count}"
        )

        print(
            f"Proyectos únicos            : "
            f"{unique_projects}"
        )

        print(
            f"Proyectos multiescenario    : "
            f"{multi_scenario}"
        )

        print(
            f"Cobertura asignación        : "
            f"{(
                100.0
                * (project_count - null_assignments)
                / project_count
            ) if project_count else 0.0:.2f}%"
        )

        print(
            f"CV tamaño escenarios        : "
            f"{cv:.4f}"
        )

        print(
            f"Score cierre 42             : "
            f"{closure['score']}"
        )

        print(
            f"Controles OK                : "
            f"{controls_ok}/{controls_total}"
        )

        print(
            f"Controles fallidos          : "
            f"{controls_fail}"
        )

        print(
            f"Fallas críticas             : "
            f"{critical_failures}"
        )

        print(
            f"Score paquete               : "
            f"{score_package:.2f}/100"
        )

        print(
            "Auditoría paquete           : "
            + (
                "OK"
                if package_go
                else "OBSERVADA"
            )
        )

        print(
            "DICTAMEN FINAL              : "
            + (
                "GO"
                if package_go
                else "NO-GO"
            )
        )

        print(
            f"Tiempo de ejecución         : "
            f"{elapsed:.2f} segundos"
        )

        print_header(
            "ARCHIVOS GENERADOS"
        )

        print(
            f"Paquete                     : "
            f"{OUTPUT_DIR}"
        )

        print(
            f"Tablas                      : "
            f"{TABLES_DIR}"
        )

        print(
            f"Documentos                  : "
            f"{DOCS_DIR}"
        )

        print(
            f"Control                     : "
            f"{CONTROL_DIR}"
        )

        print()

        if package_go:

            print(
                "========================================================================================"
            )
            print(
                "PROCESO 43 FINALIZADO - GO"
            )
            print(
                "========================================================================================"
            )

            print(
                "El paquete ejecutivo territorial AMBA V4.1 "
                "fue generado correctamente."
            )

            print(
                "El proceso 42 fue validado estructuralmente "
                "como GO."
            )

            print(
                "Los rankings fueron normalizados y validados."
            )

            print(
                "Los controles del paquete no presentan fallas."
            )

            return 0

        print(
            "========================================================================================"
        )
        print(
            "PROCESO 43 FINALIZADO - NO-GO"
        )
        print(
            "========================================================================================"
        )

        print(
            "Se detectaron inconsistencias en el paquete ejecutivo."
        )

        print(
            f"Revisar: {OUTPUT_FILES['control']}"
        )

        return 1

    except Exception as exc:

        elapsed = time.time() - start

        print()
        print(
            "========================================================================================"
        )
        print(
            "ERROR FATAL EN EL PROCESO 43"
        )
        print(
            "========================================================================================"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"Tiempo hasta el error: "
            f"{elapsed:.2f} segundos"
        )

        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())