# -*- coding: utf-8 -*-
"""
44 - AUDITORÍA FINAL DEL PAQUETE TERRITORIAL AMBA - V4.1

Auditoría integral del paquete generado por el proceso 43.

Correcciones principales V4.1-FINAL-AUDIT:
- No genera ni sobrescribe este propio archivo.
- Valida el cierre del proceso 42 mediante estructura directa.
- Valida el control del proceso 43.
- Valida modelos maestros y rankings.
- Valida tablas ejecutivas.
- Valida manifiesto aceptando referencias lógicas como MODELO.
- Valida SHA-256 solamente sobre archivos físicos verificables.
- No interpreta campos categóricos como prioridad_territorial_v4 como numéricos.
- Valida categorías de prioridad territorial.
- Valida coherencia numérica únicamente sobre indicadores cuantitativos.
- Genera auditoría, inventario, hashes, resumen e informe.
- Dictamen GO solamente cuando no existen fallas.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.1-FINAL-AUDIT"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

PACKAGE_DIR = (
    INPUT_DIR
    / "paquete_ejecutivo_amba_v4_1"
)

AUDIT_FILE = (
    INPUT_DIR
    / "auditoria_44_paquete_final_amba_v4.csv"
)

INVENTORY_FILE = (
    INPUT_DIR
    / "inventario_44_paquete_final_amba_v4.csv"
)

HASH_FILE = (
    INPUT_DIR
    / "hashes_44_paquete_final_amba_v4.csv"
)

SUMMARY_FILE = (
    INPUT_DIR
    / "resumen_44_auditoria_paquete_final_amba_v4.json"
)

REPORT_FILE = (
    INPUT_DIR
    / "informe_44_auditoria_paquete_final_amba_v4.md"
)


# =============================================================================
# ARCHIVOS DE ENTRADA
# =============================================================================

CLOSURE_42 = (
    INPUT_DIR
    / "cierre_42_modelo_territorial_amba_v4.csv"
)

CONTROL_43 = (
    PACKAGE_DIR
    / "control"
    / "control_paquete_ejecutivo_amba_v4_1.csv"
)

MANIFEST_43 = (
    PACKAGE_DIR
    / "control"
    / "manifiesto_43_paquete_ejecutivo_amba_v4_1.csv"
)

MASTER_PROJECTS = (
    INPUT_DIR
    / "modelo_maestro_proyectos_v4.csv"
)

MASTER_SCENARIOS = (
    INPUT_DIR
    / "modelo_maestro_escenarios_v4.csv"
)

RANK_PROJECTS = (
    INPUT_DIR
    / "ranking_final_proyectos_v4.csv"
)

RANK_SCENARIOS = (
    INPUT_DIR
    / "ranking_final_escenarios_v4.csv"
)

GLOBAL_INDICATORS = (
    INPUT_DIR
    / "indicadores_globales_amba_v4.csv"
)


# =============================================================================
# ARCHIVOS EJECUTIVOS
# =============================================================================

EXEC_PROJECTS = (
    PACKAGE_DIR
    / "tablas"
    / "proyectos_ejecutivos_amba_v4_1.csv"
)

EXEC_SCENARIOS = (
    PACKAGE_DIR
    / "tablas"
    / "escenarios_ejecutivos_amba_v4_1.csv"
)

TOP_PROJECTS = (
    PACKAGE_DIR
    / "tablas"
    / "top_20_proyectos_prioritarios_amba_v4_1.csv"
)

EXEC_SCENARIO_RANKING = (
    PACKAGE_DIR
    / "tablas"
    / "ranking_escenarios_ejecutivo_amba_v4_1.csv"
)

EXEC_INDICATORS = (
    PACKAGE_DIR
    / "tablas"
    / "indicadores_ejecutivos_amba_v4_1.csv"
)

EXEC_SUMMARY = (
    PACKAGE_DIR
    / "documentos"
    / "sintesis_ejecutiva_amba_v4_1.md"
)

EXEC_REPORT = (
    PACKAGE_DIR
    / "documentos"
    / "informe_ejecutivo_amba_v4_1.md"
)


# =============================================================================
# UTILIDADES
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_kv(label: str, value: Any) -> None:
    print(f"{label:<30}: {value}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def norm_upper(value: Any) -> str:
    return normalize_text(value).upper()


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Lectura robusta de CSV.

    Se intenta UTF-8 primero y luego latin-1.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:

    columns = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in columns:
            return columns[key]

    return None


def sequence_ok(
    series: pd.Series,
    expected_count: Optional[int] = None,
) -> bool:

    values = pd.to_numeric(series, errors="coerce")

    if values.isna().any():
        return False

    values = values.astype(int).tolist()

    if expected_count is None:
        expected_count = len(values)

    return sorted(values) == list(range(1, expected_count + 1))


def unique_non_null_count(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def safe_numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    return pd.to_numeric(df[column], errors="coerce")


def is_finite_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")

    return bool(
        numeric.notna().all()
        and numeric.map(math.isfinite).all()
    )


def add_control(
    controls: list[dict[str, Any]],
    control: str,
    resultado: str,
    critico: str,
    detalle: str,
) -> None:

    controls.append(
        {
            "control": control,
            "resultado": resultado,
            "critico": critico,
            "detalle": detalle,
        }
    )


# =============================================================================
# CIERRE PROCESO 42
# =============================================================================

def validate_process_42(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "1 - VALIDACIÓN DEL CIERRE FORMAL DEL PROCESO 42"
    )

    if not CLOSURE_42.exists():
        add_control(
            controls,
            "Cierre proceso 42",
            "ERROR",
            "SI",
            f"No existe: {CLOSURE_42.name}",
        )

        return {
            "ok": False,
            "score": 0.0,
            "dictamen": "NO-GO",
        }

    df = safe_read_csv(CLOSURE_42)

    print(
        f"Cargando: {CLOSURE_42.name} | "
        f"Registros: {len(df)} | "
        f"Columnas: {len(df.columns)}"
    )

    print("Columnas detectadas:")

    for col in df.columns:
        print(f"  - {col}")

    resultado_col = find_column(
        df,
        ["resultado", "status", "estado"],
    )

    critico_col = find_column(
        df,
        ["critico", "crítico", "critical"],
    )

    if resultado_col is None:
        add_control(
            controls,
            "Cierre proceso 42",
            "ERROR",
            "SI",
            "No se detectó columna resultado.",
        )

        return {
            "ok": False,
            "score": 0.0,
            "dictamen": "NO-GO",
        }

    resultado = (
        df[resultado_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    ok_count = int(
        resultado.isin(
            ["OK", "GO", "PASS", "APROBADO"]
        ).sum()
    )

    error_count = int(
        resultado.isin(
            ["ERROR", "FAIL", "FALLA", "NO-GO"]
        ).sum()
    )

    unknown_count = len(df) - ok_count - error_count

    dictamen = "GO" if (
        error_count == 0
        and unknown_count == 0
        and len(df) > 0
    ) else "NO-GO"

    score = (
        round(ok_count / len(df) * 100.0, 2)
        if len(df)
        else 0.0
    )

    critical_count = 0
    critical_failures = 0

    if critico_col is not None:

        critico = (
            df[critico_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        critical_mask = critico.isin(
            ["SI", "SÍ", "YES", "TRUE", "1"]
        )

        critical_count = int(critical_mask.sum())

        critical_failures = int(
            (
                critical_mask
                & ~resultado.isin(
                    ["OK", "GO", "PASS", "APROBADO"]
                )
            ).sum()
        )

    print_kv("Registros cierre", len(df))
    print_kv("Resultados OK", ok_count)
    print_kv("Resultados ERROR", error_count)
    print_kv("Resultados desconocidos", unknown_count)
    print_kv("Dictamen cierre 42", dictamen)
    print_kv("Score cierre 42", score)
    print_kv("Controles detectados", len(df))
    print_kv("Controles OK", ok_count)
    print_kv("Controles críticos", critical_count)
    print_kv("Críticos OK", critical_count - critical_failures)
    print_kv("Críticos con falla", critical_failures)
    print_kv("Método de detección", "estructura_directa")
    print_kv(
        "Cierre proceso 42",
        "OK" if dictamen == "GO" else "ERROR",
    )

    ok = dictamen == "GO"

    add_control(
        controls,
        "Cierre proceso 42",
        "OK" if ok else "ERROR",
        "SI",
        (
            f"Dictamen={dictamen} "
            f"Score={score} "
            f"OK={ok_count} "
            f"ERROR={error_count} "
            f"DESCONOCIDOS={unknown_count}"
        ),
    )

    return {
        "ok": ok,
        "score": score,
        "dictamen": dictamen,
        "ok_count": ok_count,
        "error_count": error_count,
    }


# =============================================================================
# PROCESO 43
# =============================================================================

def validate_process_43(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "2 - VALIDACIÓN DEL PAQUETE EJECUTIVO DEL PROCESO 43"
    )

    if not CONTROL_43.exists():

        add_control(
            controls,
            "Paquete proceso 43",
            "ERROR",
            "SI",
            f"No existe: {CONTROL_43.name}",
        )

        return {
            "ok": False,
            "score": 0.0,
            "dictamen": "NO-GO",
        }

    df = safe_read_csv(CONTROL_43)

    resultado_col = find_column(
        df,
        ["resultado", "status", "estado"],
    )

    if resultado_col is None:

        add_control(
            controls,
            "Paquete proceso 43",
            "ERROR",
            "SI",
            "No se detectó columna resultado.",
        )

        return {
            "ok": False,
            "score": 0.0,
            "dictamen": "NO-GO",
        }

    result = (
        df[resultado_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    ok_count = int(
        result.isin(
            ["OK", "GO", "PASS", "APROBADO"]
        ).sum()
    )

    error_count = int(
        result.isin(
            ["ERROR", "FAIL", "FALLA", "NO-GO"]
        ).sum()
    )

    unknown_count = len(df) - ok_count - error_count

    score = (
        round(ok_count / len(df) * 100.0, 2)
        if len(df)
        else 0.0
    )

    dictamen = (
        "GO"
        if len(df) > 0
        and error_count == 0
        and unknown_count == 0
        else "NO-GO"
    )

    print(
        f"Cargando: {CONTROL_43.name} | "
        f"Registros: {len(df)} | "
        f"Columnas: {len(df.columns)}"
    )

    print_kv("Controles proceso 43", len(df))
    print_kv("Controles OK", ok_count)
    print_kv("Controles ERROR", error_count)
    print_kv("Desconocidos", unknown_count)
    print_kv("Score proceso 43", score)
    print_kv("Dictamen proceso 43", dictamen)

    ok = dictamen == "GO"

    add_control(
        controls,
        "Paquete proceso 43",
        "OK" if ok else "ERROR",
        "SI",
        (
            f"Dictamen={dictamen} "
            f"Score={score} "
            f"OK={ok_count} "
            f"ERROR={error_count}"
        ),
    )

    return {
        "ok": ok,
        "score": score,
        "dictamen": dictamen,
    }


# =============================================================================
# CARGA DE MAESTROS
# =============================================================================

def load_master_data() -> dict[str, pd.DataFrame]:

    print_header(
        "3 - CARGANDO PRODUCTOS MAESTROS Y PRODUCTOS DE CONTROL"
    )

    files = {
        "projects": MASTER_PROJECTS,
        "scenarios": MASTER_SCENARIOS,
        "ranking_projects": RANK_PROJECTS,
        "ranking_scenarios": RANK_SCENARIOS,
        "global": GLOBAL_INDICATORS,
    }

    data: dict[str, pd.DataFrame] = {}

    for key, path in files.items():

        df = safe_read_csv(path)
        data[key] = df

        print(
            f"Cargando: {path.name} | "
            f"Registros: {len(df)} | "
            f"Columnas: {len(df.columns)}"
        )

    return data


# =============================================================================
# ESTRUCTURA MAESTRA
# =============================================================================

def validate_master_structure(
    data: dict[str, pd.DataFrame],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "4 - VALIDACIÓN ESTRUCTURAL DEL MODELO MAESTRO"
    )

    projects = data["projects"]
    scenarios = data["scenarios"]

    project_id = find_column(
        projects,
        ["proyecto_id"],
    )

    scenario_id_projects = find_column(
        projects,
        ["escenario_id"],
    )

    scenario_id_master = find_column(
        scenarios,
        ["escenario_id"],
    )

    if project_id is None:
        raise ValueError(
            "No existe proyecto_id en modelo_maestro_proyectos_v4.csv"
        )

    if scenario_id_projects is None:
        raise ValueError(
            "No existe escenario_id en modelo_maestro_proyectos_v4.csv"
        )

    if scenario_id_master is None:
        raise ValueError(
            "No existe escenario_id en modelo_maestro_escenarios_v4.csv"
        )

    project_nulls = int(projects[project_id].isna().sum())
    project_unique = unique_non_null_count(
        projects[project_id]
    )

    scenario_nulls = int(
        scenarios[scenario_id_master].isna().sum()
    )

    scenario_unique = unique_non_null_count(
        scenarios[scenario_id_master]
    )

    project_duplicates = len(projects) - project_unique
    scenario_duplicates = len(scenarios) - scenario_unique

    print_kv("Proyecto ID", project_id)
    print_kv("Escenario ID", scenario_id_projects)
    print_kv("Escenario maestro", scenario_id_master)
    print_kv("Proyectos", len(projects))
    print_kv("Proyectos únicos", project_unique)
    print_kv("Proyectos nulos", project_nulls)
    print_kv("Proyectos duplic.", project_duplicates)
    print_kv("Escenarios", len(scenarios))
    print_kv("Escenarios únicos", scenario_unique)
    print_kv("Escenarios nulos", scenario_nulls)
    print_kv("Escenarios duplic.", scenario_duplicates)

    ok = (
        len(projects) == 144
        and project_unique == len(projects)
        and project_nulls == 0
        and project_duplicates == 0
        and len(scenarios) == 7
        and scenario_unique == len(scenarios)
        and scenario_nulls == 0
        and scenario_duplicates == 0
    )

    add_control(
        controls,
        "Estructura modelo maestro",
        "OK" if ok else "ERROR",
        "SI",
        (
            f"Proyectos={len(projects)} "
            f"Escenarios={len(scenarios)} "
            f"IDs únicos y sin nulos"
        ),
    )

    return {
        "project_id": project_id,
        "scenario_id_projects": scenario_id_projects,
        "scenario_id_master": scenario_id_master,
        "ok": ok,
    }


# =============================================================================
# ASIGNACIÓN PROYECTO -> ESCENARIO
# =============================================================================

def validate_project_scenario(
    data: dict[str, pd.DataFrame],
    fields: dict[str, Any],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "5 - VALIDACIÓN PROYECTO → ESCENARIO"
    )

    projects = data["projects"]
    scenarios = data["scenarios"]

    project_id = fields["project_id"]
    scenario_project_col = fields["scenario_id_projects"]
    scenario_master_col = fields["scenario_id_master"]

    project_scenarios = set(
        projects[scenario_project_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    master_scenarios = set(
        scenarios[scenario_master_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    nulls = int(
        projects[scenario_project_col].isna().sum()
    )

    counts = (
        projects
        .groupby(project_id)[scenario_project_col]
        .nunique(dropna=True)
    )

    multiscenario = int((counts > 1).sum())

    extra_scenarios = project_scenarios - master_scenarios

    print_kv("Escenarios nulos", nulls)
    print_kv("Proyectos multiescenario", multiscenario)
    print_kv(
        "IDs escenarios coincidentes",
        "SI" if not extra_scenarios else "NO",
    )

    ok = (
        nulls == 0
        and multiscenario == 0
        and not extra_scenarios
    )

    add_control(
        controls,
        "Asignación proyecto-escenario",
        "OK" if ok else "ERROR",
        "SI",
        (
            f"Nulos={nulls} "
            f"Multiescenario={multiscenario} "
            f"Escenarios extra={len(extra_scenarios)}"
        ),
    )

    return {
        "ok": ok,
        "multiscenario": multiscenario,
    }


# =============================================================================
# RANKINGS
# =============================================================================

def validate_rankings(
    data: dict[str, pd.DataFrame],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "6 - VALIDACIÓN DE RANKINGS MAESTROS"
    )

    rp = data["ranking_projects"]
    rs = data["ranking_scenarios"]

    project_rank_col = find_column(
        rp,
        ["ranking_final_proyecto_v4"],
    )

    scenario_rank_col = find_column(
        rs,
        ["ranking_integral_v4"],
    )

    if project_rank_col is None:
        raise ValueError(
            "No existe ranking_final_proyecto_v4 "
            "en ranking_final_proyectos_v4.csv"
        )

    if scenario_rank_col is None:
        raise ValueError(
            "No existe ranking_integral_v4 "
            "en ranking_final_escenarios_v4.csv"
        )

    project_ok = sequence_ok(
        rp[project_rank_col],
        len(rp),
    )

    scenario_ok = sequence_ok(
        rs[scenario_rank_col],
        len(rs),
    )

    print_kv(
        "Ranking proyectos",
        project_rank_col,
    )

    print_kv(
        "Ranking escenarios",
        scenario_rank_col,
    )

    print(
        f"Secuencia proyectos: "
        f"{'OK' if project_ok else 'ERROR'}"
    )

    print(
        f"Secuencia escenarios: "
        f"{'OK' if scenario_ok else 'ERROR'}"
    )

    add_control(
        controls,
        "Ranking proyectos completo",
        "OK" if len(rp) == 144 else "ERROR",
        "SI",
        (
            f"Campo={project_rank_col} "
            f"Registros={len(rp)}"
        ),
    )

    add_control(
        controls,
        "Ranking proyectos ordenado",
        "OK" if project_ok else "ERROR",
        "SI",
        f"Secuencia 1..{len(rp)}",
    )

    add_control(
        controls,
        "Ranking escenarios completo",
        "OK" if len(rs) == 7 else "ERROR",
        "SI",
        (
            f"Campo={scenario_rank_col} "
            f"Registros={len(rs)}"
        ),
    )

    add_control(
        controls,
        "Ranking escenarios ordenado",
        "OK" if scenario_ok else "ERROR",
        "SI",
        f"Secuencia 1..{len(rs)}",
    )

    return {
        "project_rank_col": project_rank_col,
        "scenario_rank_col": scenario_rank_col,
        "project_ok": project_ok and len(rp) == 144,
        "scenario_ok": scenario_ok and len(rs) == 7,
    }


# =============================================================================
# TABLAS EJECUTIVAS
# =============================================================================

def validate_executive_tables(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "7 - VALIDACIÓN DE TABLAS EJECUTIVAS"
    )

    definitions = [
        (
            "proyectos",
            EXEC_PROJECTS,
            144,
            "proyecto_id",
        ),
        (
            "escenarios",
            EXEC_SCENARIOS,
            7,
            "escenario_id",
        ),
        (
            "top_proyectos",
            TOP_PROJECTS,
            20,
            "proyecto_id",
        ),
        (
            "ranking_escenarios",
            EXEC_SCENARIO_RANKING,
            7,
            "escenario_id",
        ),
        (
            "indicadores",
            EXEC_INDICATORS,
            35,
            None,
        ),
    ]

    result: dict[str, Any] = {}

    for name, path, expected, id_column in definitions:

        if not path.exists():

            add_control(
                controls,
                (
                    "Tabla ejecutiva proyectos"
                    if name == "proyectos"
                    else "Tabla ejecutiva escenarios"
                    if name == "escenarios"
                    else "Top proyectos"
                    if name == "top_proyectos"
                    else "Ranking ejecutivo escenarios"
                    if name == "ranking_escenarios"
                    else "Indicadores ejecutivos"
                ),
                "ERROR",
                "SI",
                f"No existe: {path.name}",
            )

            result[name] = None
            continue

        df = safe_read_csv(path)

        id_ok = (
            True
            if id_column is None
            else id_column in df.columns
        )

        count_ok = len(df) == expected

        ok = count_ok and id_ok

        if id_column is not None:
            detail = (
                f"Registros={len(df)} "
                f"Esperados={expected} "
                f"ID={id_column}"
            )
        else:
            detail = (
                f"Registros={len(df)} "
                f"Esperados={expected}"
            )

        print(
            f"Cargando: {path.name} | "
            f"Registros: {len(df)} | "
            f"Columnas: {len(df.columns)}"
        )

        print(
            f"{name}: "
            f"{'OK' if ok else 'ERROR'} | "
            f"{detail}"
        )

        control_name = {
            "proyectos": "Tabla ejecutiva proyectos",
            "escenarios": "Tabla ejecutiva escenarios",
            "top_proyectos": "Top proyectos",
            "ranking_escenarios": "Ranking ejecutivo escenarios",
            "indicadores": "Indicadores ejecutivos",
        }[name]

        add_control(
            controls,
            control_name,
            "OK" if ok else "ERROR",
            "SI",
            detail,
        )

        result[name] = df

    return result


# =============================================================================
# CRUCES
# =============================================================================

def validate_cross_projects(
    data: dict[str, pd.DataFrame],
    executive: dict[str, Any],
    fields: dict[str, Any],
    controls: list[dict[str, Any]],
) -> None:

    print_header(
        "8 - CONTROL CRUZADO MODELO MAESTRO ↔ TABLA EJECUTIVA DE PROYECTOS"
    )

    master_ids = set(
        data["projects"][fields["project_id"]]
        .dropna()
        .astype(str)
        .str.strip()
    )

    exec_df = executive["proyectos"]

    if exec_df is None or fields["project_id"] not in exec_df.columns:

        missing = len(master_ids)
        extra = 0

    else:

        exec_ids = set(
            exec_df[fields["project_id"]]
            .dropna()
            .astype(str)
            .str.strip()
        )

        missing = len(master_ids - exec_ids)
        extra = len(exec_ids - master_ids)

    print_kv("IDs maestro no presentes", missing)
    print_kv("IDs ejecutivos extra", extra)

    ok = missing == 0 and extra == 0

    add_control(
        controls,
        "Cruce maestro-proyectos ejecutivos",
        "OK" if ok else "ERROR",
        "SI",
        (
            "Todos los IDs coinciden."
            if ok
            else f"Faltantes={missing} Extra={extra}"
        ),
    )


def validate_cross_scenarios(
    data: dict[str, pd.DataFrame],
    executive: dict[str, Any],
    fields: dict[str, Any],
    controls: list[dict[str, Any]],
) -> None:

    print_header(
        "9 - CONTROL CRUZADO MODELO MAESTRO ↔ TABLA EJECUTIVA DE ESCENARIOS"
    )

    master_ids = set(
        data["scenarios"][fields["scenario_id_master"]]
        .dropna()
        .astype(str)
        .str.strip()
    )

    exec_df = executive["escenarios"]

    if exec_df is None or fields["scenario_id_master"] not in exec_df.columns:

        missing = len(master_ids)
        extra = 0

    else:

        exec_ids = set(
            exec_df[fields["scenario_id_master"]]
            .dropna()
            .astype(str)
            .str.strip()
        )

        missing = len(master_ids - exec_ids)
        extra = len(exec_ids - master_ids)

    print_kv("IDs maestro no presentes", missing)
    print_kv("IDs ejecutivos extra", extra)

    ok = missing == 0 and extra == 0

    add_control(
        controls,
        "Cruce maestro-escenarios ejecutivos",
        "OK" if ok else "ERROR",
        "SI",
        (
            "Todos los IDs coinciden."
            if ok
            else f"Faltantes={missing} Extra={extra}"
        ),
    )


# =============================================================================
# DOCUMENTOS
# =============================================================================

def validate_documents(
    controls: list[dict[str, Any]],
) -> None:

    print_header(
        "10 - VALIDACIÓN DE DOCUMENTOS EJECUTIVOS"
    )

    definitions = [
        (
            "Síntesis ejecutiva",
            EXEC_SUMMARY,
            500,
        ),
        (
            "Informe ejecutivo",
            EXEC_REPORT,
            1000,
        ),
    ]

    for label, path, minimum_chars in definitions:

        if not path.exists():

            add_control(
                controls,
                label,
                "ERROR",
                "SI",
                f"No existe: {path.name}",
            )

            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        chars = len(text)

        ok = chars >= minimum_chars

        print(
            f"{label}: "
            f"{'OK' if ok else 'ERROR'} | "
            f"Caracteres={chars}"
        )

        add_control(
            controls,
            label,
            "OK" if ok else "ERROR",
            "SI",
            f"Caracteres={chars}",
        )


# =============================================================================
# MANIFIESTO
# =============================================================================

def resolve_manifest_reference(
    reference: str,
) -> Optional[Path]:

    """
    Resuelve una referencia del manifiesto.

    Reglas:

    1. Referencias lógicas:
       MODELO
       MODELOS
       MODEL
       MAESTRO
       MAESTROS
       CONTROL
       PRODUCTOS
       etc.

       No se consideran archivos físicos.

    2. Si es ruta relativa existente:
       PACKAGE_DIR / reference

    3. Si contiene separadores:
       se prueba contra PACKAGE_DIR e INPUT_DIR.

    4. Si es nombre de archivo:
       se busca dentro del paquete.

    La referencia lógica MODELO es válida pero no representa
    un archivo físico individual.
    """

    ref = normalize_text(reference)

    if not ref:
        return None

    logical_tokens = {
        "MODELO",
        "MODELOS",
        "MODEL",
        "MAESTRO",
        "MAESTROS",
        "PRODUCTOS",
        "PRODUCTO",
        "CONTROL",
        "CONTROLES",
        "PAQUETE",
        "PACKAGE",
        "EJECUTIVO",
        "EJECUTIVOS",
    }

    if ref.upper() in logical_tokens:
        return None

    candidate = Path(ref)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidates = [
        PACKAGE_DIR / candidate,
        INPUT_DIR / candidate,
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    name = candidate.name

    matches = list(
        PACKAGE_DIR.rglob(name)
    )

    if len(matches) == 1:
        return matches[0]

    return None


def validate_manifest(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "11 - VALIDACIÓN DEL MANIFIESTO DEL PAQUETE"
    )

    if not MANIFEST_43.exists():

        add_control(
            controls,
            "Manifiesto paquete 43",
            "ERROR",
            "SI",
            f"No existe: {MANIFEST_43.name}",
        )

        return {
            "ok": False,
            "df": None,
            "physical": [],
        }

    df = safe_read_csv(MANIFEST_43)

    print(
        f"Cargando: {MANIFEST_43.name} | "
        f"Registros: {len(df)} | "
        f"Columnas: {len(df.columns)}"
    )

    file_col = find_column(
        df,
        [
            "producto",
            "archivo",
            "file",
            "nombre_archivo",
            "nombre",
        ],
    )

    hash_col = find_column(
        df,
        [
            "sha256",
            "hash",
            "hash_sha256",
        ],
    )

    print_kv(
        "Columna archivo detectada",
        file_col if file_col else "N/D",
    )

    print_kv(
        "Columna hash detectada",
        hash_col if hash_col else "N/D",
    )

    if file_col is None:

        add_control(
            controls,
            "Manifiesto paquete 43",
            "ERROR",
            "SI",
            "No se detectó columna de archivo.",
        )

        return {
            "ok": False,
            "df": df,
            "physical": [],
        }

    physical = []
    logical = []
    missing = []

    for _, row in df.iterrows():

        ref = normalize_text(row[file_col])

        resolved = resolve_manifest_reference(ref)

        if resolved is None:

            if ref.upper() in {
                "MODELO",
                "MODELOS",
                "MODEL",
                "MAESTRO",
                "MAESTROS",
                "PRODUCTOS",
                "PRODUCTO",
                "CONTROL",
                "CONTROLES",
                "PAQUETE",
                "PACKAGE",
                "EJECUTIVO",
                "EJECUTIVOS",
            }:

                logical.append(ref)

            else:

                missing.append(ref)

        else:

            physical.append(
                {
                    "reference": ref,
                    "path": resolved,
                    "manifest_hash": (
                        normalize_text(row[hash_col])
                        if hash_col is not None
                        else ""
                    ),
                }
            )

    print_kv(
        "Referencias lógicas",
        len(logical),
    )

    print_kv(
        "Archivos físicos",
        len(physical),
    )

    print_kv(
        "Archivos no encontrados",
        len(missing),
    )

    if missing:
        print(
            "Ejemplos faltantes:",
            missing[:5],
        )

    ok = (
        file_col is not None
        and len(missing) == 0
    )

    detail = (
        f"Archivos no encontrados={len(missing)}"
    )

    if missing:
        detail += f" Ejemplos={missing[:5]}"

    elif logical:
        detail += (
            f" Referencias_lógicas={len(logical)}"
        )

    add_control(
        controls,
        "Manifiesto paquete 43",
        "OK" if ok else "ERROR",
        "SI",
        detail,
    )

    return {
        "ok": ok,
        "df": df,
        "physical": physical,
        "logical": logical,
        "missing": missing,
        "file_col": file_col,
        "hash_col": hash_col,
    }


# =============================================================================
# SHA-256
# =============================================================================

def validate_hashes(
    manifest: dict[str, Any],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "12 - VALIDACIÓN DE INTEGRIDAD SHA-256"
    )

    if not manifest.get("ok", False):

        add_control(
            controls,
            "Integridad SHA-256 paquete",
            "ERROR",
            "SI",
            "No fue posible validar el manifiesto.",
        )

        return {
            "ok": False,
            "validated": [],
            "errors": [],
        }

    physical = manifest.get("physical", [])
    hash_col = manifest.get("hash_col")

    validated = []
    errors = []

    if hash_col is None:

        add_control(
            controls,
            "Integridad SHA-256 paquete",
            "ERROR",
            "SI",
            "No se detectó columna hash.",
        )

        return {
            "ok": False,
            "validated": [],
            "errors": [
                "No existe columna SHA-256"
            ],
        }

    for item in physical:

        path = item["path"]
        expected = normalize_text(
            item["manifest_hash"]
        ).lower()

        if not expected:
            errors.append(
                f"{item['reference']}: hash vacío"
            )
            continue

        try:
            actual = sha256_file(path).lower()
        except Exception as exc:
            errors.append(
                f"{item['reference']}: {exc}"
            )
            continue

        if actual != expected:
            errors.append(
                (
                    f"{item['reference']}: "
                    f"esperado={expected} "
                    f"actual={actual}"
                )
            )
        else:
            validated.append(
                {
                    "reference": item["reference"],
                    "path": path,
                    "sha256": actual,
                }
            )

    print_kv(
        "Archivos físicos validados",
        len(validated),
    )

    print_kv(
        "Errores hash",
        len(errors),
    )

    ok = len(errors) == 0

    print(
        f"Integridad SHA-256: "
        f"{'OK' if ok else 'ERROR'}"
    )

    detail = (
        f"Archivos validados={len(validated)} "
        f"Errores={len(errors)}"
    )

    add_control(
        controls,
        "Integridad SHA-256 paquete",
        "OK" if ok else "ERROR",
        "SI",
        detail,
    )

    return {
        "ok": ok,
        "validated": validated,
        "errors": errors,
    }


# =============================================================================
# INVENTARIO OBLIGATORIO
# =============================================================================

def validate_required_inventory(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "13 - INVENTARIO DE PRODUCTOS OBLIGATORIOS"
    )

    required = [
        CLOSURE_42,
        CONTROL_43,
        MANIFEST_43,
        MASTER_PROJECTS,
        MASTER_SCENARIOS,
        RANK_PROJECTS,
        RANK_SCENARIOS,
        GLOBAL_INDICATORS,
        EXEC_PROJECTS,
        EXEC_SCENARIOS,
        TOP_PROJECTS,
        EXEC_SCENARIO_RANKING,
        EXEC_INDICATORS,
        EXEC_SUMMARY,
        EXEC_REPORT,
    ]

    # MODELO / maestro son archivos del modelo base fuera del paquete.
    # Todos deben existir para que la auditoría sea reproducible.

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    print_kv(
        "Productos obligatorios",
        len(required),
    )

    print_kv(
        "Productos faltantes",
        len(missing),
    )

    if missing:
        for path in missing[:10]:
            print(f"  - {path}")

    ok = len(missing) == 0

    add_control(
        controls,
        "Inventario productos obligatorios",
        "OK" if ok else "ERROR",
        "SI",
        (
            f"Productos={len(required)} "
            f"Faltantes={len(missing)}"
        ),
    )

    return {
        "ok": ok,
        "required": required,
        "missing": missing,
    }


# =============================================================================
# COMPLETITUD
# =============================================================================

def validate_package_completeness(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "14 - CONTROL DE COMPLETITUD DEL PAQUETE FINAL"
    )

    directories = [
        PACKAGE_DIR,
        PACKAGE_DIR / "tablas",
        PACKAGE_DIR / "documentos",
        PACKAGE_DIR / "control",
    ]

    required_files = [
        EXEC_PROJECTS,
        EXEC_SCENARIOS,
        TOP_PROJECTS,
        EXEC_SCENARIO_RANKING,
        EXEC_INDICATORS,
        EXEC_SUMMARY,
        EXEC_REPORT,
        CONTROL_43,
        MANIFEST_43,
    ]

    missing_dirs = [
        path
        for path in directories
        if not path.exists()
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    print_kv(
        "Directorios faltantes",
        len(missing_dirs),
    )

    print_kv(
        "Archivos faltantes",
        len(missing_files),
    )

    ok = (
        len(missing_dirs) == 0
        and len(missing_files) == 0
    )

    add_control(
        controls,
        "Completitud paquete final",
        "OK" if ok else "ERROR",
        "SI",
        (
            "Todos los directorios y archivos "
            "obligatorios existen."
            if ok
            else (
                f"Directorios={len(missing_dirs)} "
                f"Archivos={len(missing_files)}"
            )
        ),
    )

    return {
        "ok": ok,
        "missing_dirs": missing_dirs,
        "missing_files": missing_files,
    }


# =============================================================================
# INDICADORES GLOBALES
# =============================================================================

def validate_global_indicators(
    data: dict[str, pd.DataFrame],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "15 - VALIDACIÓN DE INDICADORES GLOBALES"
    )

    df = data["global"]

    indicator_col = find_column(
        df,
        ["indicador", "indicator", "nombre"],
    )

    value_col = find_column(
        df,
        ["valor", "value"],
    )

    print_kv("Indicadores", len(df))
    print_kv(
        "Campo indicador",
        indicator_col if indicator_col else "N/D",
    )
    print_kv(
        "Campo valor",
        value_col if value_col else "N/D",
    )

    ok = (
        len(df) == 27
        and indicator_col is not None
        and value_col is not None
    )

    add_control(
        controls,
        "Indicadores globales",
        "OK" if ok else "ERROR",
        "SI",
        (
            f"Indicadores={len(df)} "
            f"Campo indicador={indicator_col} "
            f"Campo valor={value_col}"
        ),
    )

    return {
        "ok": ok,
        "indicator_col": indicator_col,
        "value_col": value_col,
    }


# =============================================================================
# COHERENCIA NUMÉRICA
# =============================================================================

def validate_numeric_coherence(
    data: dict[str, pd.DataFrame],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "16 - VALIDACIÓN DE COHERENCIA NUMÉRICA DEL MODELO"
    )

    projects = data["projects"]
    scenarios = data["scenarios"]
    rank_projects = data["ranking_projects"]
    rank_scenarios = data["ranking_scenarios"]

    """
    IMPORTANTE:

    prioridad_territorial_v4 es categórica.

    Ejemplos reales:
        PRIORIDAD_2_ALTA
        PRIORIDAD_3_MEDIA
        PRIORIDAD_4_MEDIA_BAJA

    Por eso NO se incluye en numeric_fields.

    El ranking de proyectos se valida desde
    ranking_final_proyectos_v4.csv, no desde el maestro.
    """

    numeric_fields_projects = [
        "score_priorizacion_v4",
        "indice_demanda_estructural",
        "deficit_infraestructura",
        "indice_conectividad_estructural",
        "indice_intermodalidad_estructural",
        "indice_integracion_territorial",
        "indice_centralidad_estructural",
        "impacto_potencial",
        "urgencia_intervencion",
    ]

    numeric_fields_scenarios = [
        "ranking_integral_v4",
    ]

    numeric_fields_rank_projects = [
        "ranking_final_proyecto_v4",
        "score_final_proyecto_v4",
    ]

    evaluated = 0
    anomalies: list[str] = []

    for field in numeric_fields_projects:

        if field not in projects.columns:
            continue

        evaluated += 1

        series = safe_numeric_series(
            projects,
            field,
        )

        nulls = int(series.isna().sum())

        if nulls > 0:
            anomalies.append(
                f"{field}: NULOS={nulls}"
            )
            continue

        if not is_finite_series(series):
            anomalies.append(
                f"{field}: VALORES_NO_FINITOS"
            )

    for field in numeric_fields_scenarios:

        if field not in rank_scenarios.columns:
            continue

        evaluated += 1

        series = safe_numeric_series(
            rank_scenarios,
            field,
        )

        nulls = int(series.isna().sum())

        if nulls > 0:
            anomalies.append(
                f"{field}: NULOS={nulls}"
            )
            continue

        if not is_finite_series(series):
            anomalies.append(
                f"{field}: VALORES_NO_FINITOS"
            )

    for field in numeric_fields_rank_projects:

        if field not in rank_projects.columns:
            continue

        evaluated += 1

        series = safe_numeric_series(
            rank_projects,
            field,
        )

        nulls = int(series.isna().sum())

        if nulls > 0:
            anomalies.append(
                f"{field}: NULOS={nulls}"
            )
            continue

        if not is_finite_series(series):
            anomalies.append(
                f"{field}: VALORES_NO_FINITOS"
            )

    # -------------------------------------------------------------------------
    # VALIDACIÓN CATEGÓRICA DE PRIORIDAD TERRITORIAL
    # -------------------------------------------------------------------------

    priority_field = "prioridad_territorial_v4"

    allowed_priorities = {
        "PRIORIDAD_2_ALTA",
        "PRIORIDAD_3_MEDIA",
        "PRIORIDAD_4_MEDIA_BAJA",
    }

    if priority_field in projects.columns:

        values = (
            projects[priority_field]
            .dropna()
            .astype(str)
            .str.strip()
        )

        invalid_priorities = sorted(
            set(values) - allowed_priorities
        )

        null_priorities = int(
            projects[priority_field].isna().sum()
        )

        if null_priorities > 0:
            anomalies.append(
                f"{priority_field}: NULOS={null_priorities}"
            )

        if invalid_priorities:
            anomalies.append(
                (
                    f"{priority_field}: "
                    f"CATEGORIAS_INVALIDAS="
                    f"{invalid_priorities}"
                )
            )

    print_kv(
        "Indicadores numéricos evaluados",
        evaluated,
    )

    print_kv(
        "Anomalías numéricas",
        len(anomalies),
    )

    for anomaly in anomalies:
        print(f"  - {anomaly}")

    ok = len(anomalies) == 0

    add_control(
        controls,
        "Coherencia numérica",
        "OK" if ok else "ERROR",
        "NO",
        (
            f"Campos evaluados={evaluated} "
            f"Anomalías={len(anomalies)}"
        ),
    )

    return {
        "ok": ok,
        "evaluated": evaluated,
        "anomalies": anomalies,
    }


# =============================================================================
# INVENTARIO DE ARCHIVOS
# =============================================================================

def build_inventory() -> pd.DataFrame:

    rows = []

    if PACKAGE_DIR.exists():

        for path in sorted(
            PACKAGE_DIR.rglob("*")
        ):

            if not path.is_file():
                continue

            rel = path.relative_to(
                PACKAGE_DIR
            )

            stat = path.stat()

            rows.append(
                {
                    "archivo": str(rel).replace("\\", "/"),
                    "ruta_absoluta": str(path),
                    "tamano_bytes": stat.st_size,
                    "sha256": sha256_file(path),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# HASHES DEL PAQUETE AUDITADO
# =============================================================================

def generate_package_hashes() -> pd.DataFrame:

    print_header(
        "17 - GENERACIÓN DE HASHES DEL PAQUETE AUDITADO"
    )

    rows = []

    if PACKAGE_DIR.exists():

        for path in sorted(
            PACKAGE_DIR.rglob("*")
        ):

            if not path.is_file():
                continue

            relative = path.relative_to(
                PACKAGE_DIR
            )

            rows.append(
                {
                    "archivo": str(relative).replace(
                        "\\",
                        "/",
                    ),
                    "sha256": sha256_file(path),
                    "tamano_bytes": path.stat().st_size,
                }
            )

    df = pd.DataFrame(rows)

    print_kv(
        "Archivos incluidos en hash",
        len(df),
    )

    return df


# =============================================================================
# SCORE Y DICTAMEN
# =============================================================================

def determine_final_result(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:

    print_header(
        "18 - DETERMINACIÓN DEL DICTAMEN FINAL"
    )

    total = len(controls)

    ok_count = sum(
        1
        for c in controls
        if c["resultado"] == "OK"
    )

    failed = total - ok_count

    critical_failures = sum(
        1
        for c in controls
        if (
            c["resultado"] != "OK"
            and norm_upper(c["critico"]) in {
                "SI",
                "SÍ",
                "YES",
                "TRUE",
                "1",
            }
        )
    )

    important_failures = sum(
        1
        for c in controls
        if (
            c["resultado"] != "OK"
            and norm_upper(c["critico"]) not in {
                "SI",
                "SÍ",
                "YES",
                "TRUE",
                "1",
            }
        )
    )

    score = (
        round(
            ok_count / total * 100.0,
            2,
        )
        if total
        else 0.0
    )

    """
    Regla final estricta:

    GO únicamente si:
      - todos los controles están OK
      - no hay fallas críticas
      - no hay fallas importantes
    """

    final_ok = (
        total > 0
        and failed == 0
        and critical_failures == 0
        and important_failures == 0
    )

    audit_status = (
        "OK"
        if final_ok
        else "OBSERVADA"
    )

    dictamen = (
        "GO"
        if final_ok
        else "NO-GO"
    )

    print_kv(
        "Controles OK",
        f"{ok_count}/{total}",
    )

    print_kv(
        "Controles fallidos",
        failed,
    )

    print_kv(
        "Fallas críticas",
        critical_failures,
    )

    print_kv(
        "Fallas importantes",
        important_failures,
    )

    print_kv(
        "Score auditoría",
        f"{score:.2f}/100",
    )

    print_kv(
        "Auditoría",
        audit_status,
    )

    print_kv(
        "DICTAMEN FINAL",
        dictamen,
    )

    return {
        "total": total,
        "ok": ok_count,
        "failed": failed,
        "critical_failures": critical_failures,
        "important_failures": important_failures,
        "score": score,
        "audit_status": audit_status,
        "dictamen": dictamen,
    }


# =============================================================================
# EXPORTAR AUDITORÍA
# =============================================================================

def export_audit(
    controls: list[dict[str, Any]],
) -> None:

    df = pd.DataFrame(
        controls,
        columns=[
            "control",
            "resultado",
            "critico",
            "detalle",
        ],
    )

    df.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )


# =============================================================================
# RESUMEN JSON
# =============================================================================

def generate_summary(
    controls: list[dict[str, Any]],
    closure: dict[str, Any],
    process43: dict[str, Any],
    result: dict[str, Any],
) -> None:

    payload = {
        "proceso": 44,
        "version": VERSION,
        "proyecto": str(PROJECT_ROOT),
        "entrada": str(INPUT_DIR),
        "paquete": str(PACKAGE_DIR),
        "proceso_42": {
            "dictamen": closure.get("dictamen"),
            "score": closure.get("score"),
            "ok": closure.get("ok"),
        },
        "proceso_43": {
            "dictamen": process43.get("dictamen"),
            "score": process43.get("score"),
            "ok": process43.get("ok"),
        },
        "auditoria": {
            "controles": result["total"],
            "controles_ok": result["ok"],
            "controles_fallidos": result["failed"],
            "fallas_criticas": result["critical_failures"],
            "fallas_importantes": result["important_failures"],
            "score": result["score"],
            "estado": result["audit_status"],
            "dictamen": result["dictamen"],
        },
        "controles": controls,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# INFORME MARKDOWN
# =============================================================================

def generate_report(
    controls: list[dict[str, Any]],
    closure: dict[str, Any],
    process43: dict[str, Any],
    result: dict[str, Any],
) -> None:

    lines = []

    lines.append(
        "# Auditoría final del paquete territorial AMBA V4.1"
    )
    lines.append("")
    lines.append(
        "**Proceso:** 44"
    )
    lines.append(
        f"**Versión auditor:** {VERSION}"
    )
    lines.append("")

    lines.append("## Resultado")
    lines.append("")
    lines.append(
        f"- Controles: {result['total']}"
    )
    lines.append(
        f"- Controles OK: {result['ok']}"
    )
    lines.append(
        f"- Fallas: {result['failed']}"
    )
    lines.append(
        f"- Fallas críticas: {result['critical_failures']}"
    )
    lines.append(
        f"- Fallas importantes: {result['important_failures']}"
    )
    lines.append(
        f"- Score: {result['score']}/100"
    )
    lines.append(
        f"- Auditoría: {result['audit_status']}"
    )
    lines.append(
        f"- Dictamen: **{result['dictamen']}**"
    )
    lines.append("")

    lines.append("## Controles")
    lines.append("")
    lines.append(
        "| Control | Resultado | Crítico | Detalle |"
    )
    lines.append(
        "|---|---|---|---|"
    )

    for control in controls:

        detail = (
            normalize_text(
                control["detalle"]
            )
            .replace("|", "/")
            .replace("\n", " ")
        )

        lines.append(
            f"| {control['control']} "
            f"| {control['resultado']} "
            f"| {control['critico']} "
            f"| {detail} |"
        )

    failures = [
        c
        for c in controls
        if c["resultado"] != "OK"
    ]

    lines.append("")
    lines.append("## Observaciones")
    lines.append("")

    if failures:

        for failure in failures:
            lines.append(
                f"- **{failure['control']}**: "
                f"{failure['detalle']}"
            )

    else:

        lines.append(
            "No se detectaron observaciones."
        )

    lines.append("")

    lines.append("## Criterio de auditoría")
    lines.append("")
    lines.append(
        "El dictamen final es GO únicamente cuando "
        "todos los controles obligatorios resultan OK. "
        "Las referencias lógicas del manifiesto, como "
        "`MODELO`, no se consideran archivos físicos faltantes. "
        "El campo `prioridad_territorial_v4` se valida como "
        "variable categórica y no como indicador numérico."
    )
    lines.append("")

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    start = time.perf_counter()

    print("=" * 88)
    print(
        "44 - AUDITORÍA FINAL DEL PAQUETE TERRITORIAL AMBA - V4.1"
    )
    print("=" * 88)

    print_kv(
        "Proyecto",
        PROJECT_ROOT,
    )

    print_kv(
        "Entrada",
        INPUT_DIR,
    )

    print_kv(
        "Paquete",
        PACKAGE_DIR,
    )

    controls: list[dict[str, Any]] = []

    try:

        # ---------------------------------------------------------------------
        # 1
        # ---------------------------------------------------------------------

        closure = validate_process_42(
            controls
        )

        # ---------------------------------------------------------------------
        # 2
        # ---------------------------------------------------------------------

        process43 = validate_process_43(
            controls
        )

        # ---------------------------------------------------------------------
        # 3
        # ---------------------------------------------------------------------

        data = load_master_data()

        # ---------------------------------------------------------------------
        # 4
        # ---------------------------------------------------------------------

        fields = validate_master_structure(
            data,
            controls,
        )

        # ---------------------------------------------------------------------
        # 5
        # ---------------------------------------------------------------------

        project_scenario = validate_project_scenario(
            data,
            fields,
            controls,
        )

        # ---------------------------------------------------------------------
        # 6
        # ---------------------------------------------------------------------

        rankings = validate_rankings(
            data,
            controls,
        )

        # ---------------------------------------------------------------------
        # DISTRIBUCIÓN TERRITORIAL
        # ---------------------------------------------------------------------

        print_header(
            "6B - DISTRIBUCIÓN TERRITORIAL"
        )

        projects = data["projects"]

        scenario_col = fields[
            "scenario_id_projects"
        ]

        distribution = (
            projects[scenario_col]
            .value_counts()
            .sort_index()
        )

        for scenario_id, count in distribution.items():
            print(
                f"  {scenario_id}: {count}"
            )

        minimum = int(distribution.min())
        maximum = int(distribution.max())
        average = float(distribution.mean())

        cv = (
            float(distribution.std(ddof=0) / average)
            if average
            else 0.0
        )

        print_kv("Mínimo", minimum)
        print_kv("Máximo", maximum)
        print_kv("Promedio", f"{average:.2f}")
        print_kv("CV", f"{cv:.4f}")

        # ---------------------------------------------------------------------
        # 7
        # ---------------------------------------------------------------------

        executive = validate_executive_tables(
            controls
        )

        # ---------------------------------------------------------------------
        # 8
        # ---------------------------------------------------------------------

        validate_cross_projects(
            data,
            executive,
            fields,
            controls,
        )

        # ---------------------------------------------------------------------
        # 9
        # ---------------------------------------------------------------------

        validate_cross_scenarios(
            data,
            executive,
            fields,
            controls,
        )

        # ---------------------------------------------------------------------
        # 10
        # ---------------------------------------------------------------------

        validate_documents(
            controls
        )

        # ---------------------------------------------------------------------
        # 11
        # ---------------------------------------------------------------------

        manifest = validate_manifest(
            controls
        )

        # ---------------------------------------------------------------------
        # 12
        # ---------------------------------------------------------------------

        hashes = validate_hashes(
            manifest,
            controls,
        )

        # ---------------------------------------------------------------------
        # 13
        # ---------------------------------------------------------------------

        inventory = validate_required_inventory(
            controls
        )

        # ---------------------------------------------------------------------
        # 14
        # ---------------------------------------------------------------------

        completeness = validate_package_completeness(
            controls
        )

        # ---------------------------------------------------------------------
        # 15
        # ---------------------------------------------------------------------

        global_indicators = validate_global_indicators(
            data,
            controls,
        )

        # ---------------------------------------------------------------------
        # 16
        # ---------------------------------------------------------------------

        numeric = validate_numeric_coherence(
            data,
            controls,
        )

        # ---------------------------------------------------------------------
        # 17
        # ---------------------------------------------------------------------

        package_hashes = generate_package_hashes()

        # ---------------------------------------------------------------------
        # 18
        # ---------------------------------------------------------------------

        final_result = determine_final_result(
            controls
        )

        # ---------------------------------------------------------------------
        # 19
        # ---------------------------------------------------------------------

        print_header(
            "19 - EXPORTANDO RESULTADOS DE LA AUDITORÍA FINAL"
        )

        AUDIT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        export_audit(
            controls
        )

        inventory_df = build_inventory()

        inventory_df.to_csv(
            INVENTORY_FILE,
            index=False,
            encoding="utf-8",
        )

        package_hashes.to_csv(
            HASH_FILE,
            index=False,
            encoding="utf-8",
        )

        generate_summary(
            controls,
            closure,
            process43,
            final_result,
        )

        generate_report(
            controls,
            closure,
            process43,
            final_result,
        )

        print_kv(
            "Auditoría",
            AUDIT_FILE,
        )

        print_kv(
            "Inventario",
            INVENTORY_FILE,
        )

        print_kv(
            "Hashes",
            HASH_FILE,
        )

        print_kv(
            "Resumen",
            SUMMARY_FILE,
        )

        print_kv(
            "Informe",
            REPORT_FILE,
        )

        # ---------------------------------------------------------------------
        # RESULTADO FINAL
        # ---------------------------------------------------------------------

        elapsed = time.perf_counter() - start

        print()
        print("=" * 88)
        print(
            "RESULTADO FINAL DEL PROCESO 44"
        )
        print("=" * 88)

        print_kv(
            "Proyectos",
            len(data["projects"]),
        )

        print_kv(
            "Escenarios",
            len(data["scenarios"]),
        )

        print_kv(
            "Proceso 42",
            closure.get("dictamen"),
        )

        print_kv(
            "Proceso 43",
            process43.get("dictamen"),
        )

        print_kv(
            "Controles OK",
            f"{final_result['ok']}/{final_result['total']}",
        )

        print_kv(
            "Fallas críticas",
            final_result["critical_failures"],
        )

        print_kv(
            "Fallas importantes",
            final_result["important_failures"],
        )

        print_kv(
            "Score auditoría",
            f"{final_result['score']:.2f}/100",
        )

        print_kv(
            "Auditoría",
            final_result["audit_status"],
        )

        print_kv(
            "DICTAMEN FINAL",
            final_result["dictamen"],
        )

        print_kv(
            "Tiempo de ejecución",
            f"{elapsed:.2f} segundos",
        )

        print()
        print("=" * 88)
        print(
            "ARCHIVOS GENERADOS"
        )
        print("=" * 88)

        print_kv(
            "Auditoría",
            AUDIT_FILE,
        )

        print_kv(
            "Inventario",
            INVENTORY_FILE,
        )

        print_kv(
            "Hashes",
            HASH_FILE,
        )

        print_kv(
            "Resumen",
            SUMMARY_FILE,
        )

        print_kv(
            "Informe",
            REPORT_FILE,
        )

        print()
        print("=" * 88)

        if final_result["dictamen"] == "GO":

            print(
                "PROCESO 44 FINALIZADO - GO"
            )

            print(
                "El paquete ejecutivo territorial AMBA V4.1 "
                "superó la auditoría final."
            )

            print(
                "No se detectaron fallas críticas ni importantes."
            )

            print("=" * 88)

            return 0

        print(
            "PROCESO 44 FINALIZADO - NO-GO"
        )

        print(
            "[ERROR] Se detectaron inconsistencias "
            "en la auditoría final."
        )

        print(
            "[ERROR] Revisar el informe y los controles generados."
        )

        print("=" * 88)

        return 1

    except Exception as exc:

        elapsed = time.perf_counter() - start

        print()
        print("=" * 88)
        print(
            "ERROR FATAL EN EL PROCESO 44"
        )
        print("=" * 88)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"Tiempo transcurrido: {elapsed:.2f} segundos"
        )

        # Intentamos dejar registrado el error.
        try:

            add_control(
                controls,
                "Error fatal proceso 44",
                "ERROR",
                "SI",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            export_audit(
                controls
            )

            generate_report(
                controls,
                {
                    "dictamen": "N/D",
                    "score": 0,
                    "ok": False,
                },
                {
                    "dictamen": "N/D",
                    "score": 0,
                    "ok": False,
                },
                {
                    "total": len(controls),
                    "ok": sum(
                        c["resultado"] == "OK"
                        for c in controls
                    ),
                    "failed": sum(
                        c["resultado"] != "OK"
                        for c in controls
                    ),
                    "critical_failures": sum(
                        (
                            c["resultado"] != "OK"
                            and norm_upper(c["critico"])
                            in {
                                "SI",
                                "SÍ",
                                "YES",
                                "TRUE",
                                "1",
                            }
                        )
                        for c in controls
                    ),
                    "important_failures": 0,
                    "score": 0,
                    "audit_status": "ERROR",
                    "dictamen": "NO-GO",
                },
            )

        except Exception:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())