# -*- coding: utf-8 -*-

"""
45_auditoria_cierre_amba_v4.py

AUDITORÍA INDEPENDIENTE DE CIERRE
MODELO TERRITORIAL AMBA - V4.1

Objetivo
--------
Realizar una auditoría independiente del paquete ejecutivo generado
por el proceso 43 y validado por el proceso 44.

El proceso 45 NO modifica los productos maestros ni el paquete ejecutivo.
Solamente lee, valida y genera productos de auditoría.

Controles principales
---------------------
1.  Proceso 42
2.  Proceso 43
3.  Proceso 44
4.  Estructura física del paquete
5.  Productos maestros
6.  Estructura de proyectos
7.  Estructura de escenarios
8.  Relación proyecto -> escenario
9.  Rankings
10. Tablas ejecutivas
11. Identificadores cruzados
12. Documentos
13. Manifiesto
14. SHA-256
15. Productos obligatorios
16. Indicadores globales
17. Coherencia numérica
18. Correspondencia de productos
19. Integridad final
20. Dictamen independiente

Salida
------
data/processed/escenarios_territoriales_amba/
    auditoria_45_cierre_amba_v4/
        auditoria_45_cierre_amba_v4.csv
        inventario_45_cierre_amba_v4.csv
        hashes_45_cierre_amba_v4.csv
        resumen_45_cierre_amba_v4.json
        informe_45_cierre_amba_v4.md

IMPORTANTE
----------
El script está diseñado para ejecutarse desde cualquier directorio.
No depende del directorio de trabajo actual.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

VERSION = "V4.1-FINAL"

SCRIPT_NAME = "45_auditoria_cierre_amba_v4.py"

# Detectamos el proyecto a partir de la ubicación del script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

PACKAGE_DIR = DATA_ROOT / "paquete_ejecutivo_amba_v4_1"

OUTPUT_DIR = DATA_ROOT / "auditoria_45_cierre_amba_v4"


# ============================================================================
# NOMBRES DE ARCHIVOS
# ============================================================================

FILES = {
    # Proceso 42
    "cierre_42": "cierre_42_modelo_territorial_amba_v4.csv",

    # Proceso 43
    "control_43": "control_paquete_ejecutivo_amba_v4_1.csv",

    # Proceso 43 - manifiesto
    "manifiesto_43": "manifiesto_43_paquete_ejecutivo_amba_v4_1.csv",

    # Proceso 44
    "auditoria_44": "auditoria_44_paquete_final_amba_v4.csv",
    "resumen_44": "resumen_44_auditoria_paquete_final_amba_v4.json",

    # Maestros
    "proyectos": "modelo_maestro_proyectos_v4.csv",
    "escenarios": "modelo_maestro_escenarios_v4.csv",
    "ranking_proyectos": "ranking_final_proyectos_v4.csv",
    "ranking_escenarios": "ranking_final_escenarios_v4.csv",
    "indicadores": "indicadores_globales_amba_v4.csv",

    # Paquete ejecutivo
    "ejecutivos_proyectos": (
        "tablas/proyectos_ejecutivos_amba_v4_1.csv"
    ),
    "ejecutivos_escenarios": (
        "tablas/escenarios_ejecutivos_amba_v4_1.csv"
    ),
    "top_proyectos": (
        "tablas/top_20_proyectos_prioritarios_amba_v4_1.csv"
    ),
    "ranking_ejecutivo_escenarios": (
        "tablas/ranking_escenarios_ejecutivo_amba_v4_1.csv"
    ),
    "indicadores_ejecutivos": (
        "tablas/indicadores_ejecutivos_amba_v4_1.csv"
    ),

    # Documentos
    "sintesis": (
        "documentos/sintesis_ejecutiva_amba_v4_1.md"
    ),
    "informe": (
        "documentos/informe_ejecutivo_amba_v4_1.md"
    ),

    # Control del paquete
    "control_paquete": (
        "control/control_paquete_ejecutivo_amba_v4_1.csv"
    ),
    "resumen_43": (
        "control/resumen_43_paquete_ejecutivo_amba_v4_1.json"
    ),
    "manifiesto_paquete": (
        "control/manifiesto_43_paquete_ejecutivo_amba_v4_1.csv"
    ),
}


# ============================================================================
# CONFIGURACIÓN DE INVENTARIO
# ============================================================================

REQUIRED_PACKAGE_DIRS = [
    PACKAGE_DIR / "tablas",
    PACKAGE_DIR / "documentos",
    PACKAGE_DIR / "control",
]

REQUIRED_PACKAGE_FILES = [
    PACKAGE_DIR / FILES["ejecutivos_proyectos"],
    PACKAGE_DIR / FILES["ejecutivos_escenarios"],
    PACKAGE_DIR / FILES["top_proyectos"],
    PACKAGE_DIR / FILES["ranking_ejecutivo_escenarios"],
    PACKAGE_DIR / FILES["indicadores_ejecutivos"],
    PACKAGE_DIR / FILES["sintesis"],
    PACKAGE_DIR / FILES["informe"],
    PACKAGE_DIR / FILES["control_paquete"],
    PACKAGE_DIR / FILES["resumen_43"],
    PACKAGE_DIR / FILES["manifiesto_paquete"],
]


# ============================================================================
# UTILIDADES GENERALES
# ============================================================================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_kv(label: str, value: Any, width: int = 30) -> None:
    print(f"{label:<{width}}: {value}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def upper_text(value: Any) -> str:
    return normalize_text(value).upper()


def clean_column_name(value: Any) -> str:
    return (
        normalize_text(value)
        .replace("\ufeff", "")
        .strip()
        .lower()
    )


def unique_columns(df: pd.DataFrame) -> List[str]:
    """
    Devuelve nombres de columnas únicos.

    Importante:
    pandas permite columnas duplicadas. En ese caso df["x"] puede
    devolver un DataFrame en lugar de Series.

    Esta función evita esa ambigüedad.
    """
    result = []

    seen = {}

    for col in df.columns:
        base = normalize_text(col)

        if base not in seen:
            seen[base] = 0
            result.append(base)
        else:
            seen[base] += 1
            result.append(f"{base}__DUP{seen[base]}")

    return result


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copia un DataFrame y garantiza nombres de columnas únicos.
    """
    out = df.copy()
    out.columns = unique_columns(out)
    return out


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    """
    Busca una columna por nombre exacto ignorando mayúsculas,
    espacios y BOM.
    """
    mapping = {}

    for col in df.columns:
        normalized = clean_column_name(col)

        if normalized not in mapping:
            mapping[normalized] = col

    for candidate in candidates:
        key = clean_column_name(candidate)

        if key in mapping:
            return mapping[key]

    return None


def find_columns_containing(
    df: pd.DataFrame,
    tokens: Sequence[str],
) -> List[str]:
    result = []

    for col in df.columns:
        normalized = clean_column_name(col)

        if all(
            clean_column_name(token) in normalized
            for token in tokens
        ):
            result.append(col)

    return result


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Lectura robusta de CSV.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )

            return deduplicate_columns(df)

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"No se pudo leer CSV {path}: {last_error}"
    )


def safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        return json.load(fh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def relative_package_path(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PACKAGE_DIR.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def is_empty_file(path: Path) -> bool:
    try:
        return path.stat().st_size == 0
    except OSError:
        return True


def is_missing_or_empty(path: Path) -> bool:
    return (not path.exists()) or is_empty_file(path)


def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Convierte una columna a numérica.

    Si por alguna razón existe una columna duplicada y pandas
    devuelve un DataFrame, seleccionamos explícitamente la primera.
    """
    obj = df[column]

    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]

    return pd.to_numeric(
        obj,
        errors="coerce",
    )


# ============================================================================
# SISTEMA DE CONTROLES
# ============================================================================

controls: List[Dict[str, Any]] = []


def add_control(
    name: str,
    result: str,
    critical: str,
    detail: str,
) -> None:
    result = upper_text(result)

    if result not in {"OK", "ERROR"}:
        result = "ERROR"

    critical = (
        "SI"
        if upper_text(critical) in {"SI", "YES", "TRUE", "1"}
        else "NO"
    )

    controls.append(
        {
            "control": name,
            "resultado": result,
            "critico": critical,
            "detalle": detail,
        }
    )


def controls_ok() -> int:
    return sum(
        1
        for c in controls
        if c["resultado"] == "OK"
    )


def controls_error() -> int:
    return sum(
        1
        for c in controls
        if c["resultado"] == "ERROR"
    )


def critical_errors() -> int:
    return sum(
        1
        for c in controls
        if (
            c["resultado"] == "ERROR"
            and c["critico"] == "SI"
        )
    )


def important_errors() -> int:
    return sum(
        1
        for c in controls
        if (
            c["resultado"] == "ERROR"
            and c["critico"] == "NO"
        )
    )


# ============================================================================
# RESOLUCIÓN DE ESTADOS
# ============================================================================

def result_counts(
    df: pd.DataFrame,
) -> Tuple[int, int, int]:
    result_col = find_column(
        df,
        [
            "resultado",
            "estado",
            "status",
            "dictamen",
        ],
    )

    if result_col is None:
        return 0, 0, len(df)

    values = (
        df[result_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    ok = int(
        values.isin(
            [
                "OK",
                "GO",
                "TRUE",
                "1",
                "PASS",
                "PASSED",
            ]
        ).sum()
    )

    error = int(
        values.isin(
            [
                "ERROR",
                "FAIL",
                "FAILED",
                "NO-GO",
                "NOGO",
                "FALLA",
            ]
        ).sum()
    )

    unknown = len(df) - ok - error

    return ok, error, unknown


def extract_score_from_dataframe(
    df: pd.DataFrame,
) -> Optional[float]:
    """
    Busca score de forma flexible.
    """
    for col in df.columns:
        normalized = clean_column_name(col)

        if (
            "score" in normalized
            or "puntaje" in normalized
        ):
            series = numeric_series(df, col)

            values = series.dropna()

            if not values.empty:
                return float(values.iloc[0])

    # Segunda alternativa: buscar en texto.
    text = " ".join(
        df.astype(str).fillna("").values.flatten().tolist()
    )

    match = re.search(
        r"SCORE\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        text.upper(),
    )

    if match:
        return float(match.group(1))

    return None


def extract_dictamen_from_dataframe(
    df: pd.DataFrame,
) -> Optional[str]:
    for col in df.columns:
        normalized = clean_column_name(col)

        if normalized in {
            "dictamen",
            "resultado_final",
            "auditoria",
            "estado_final",
        }:
            values = (
                df[col]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            for value in values:
                if value in {
                    "GO",
                    "NO-GO",
                    "NOGO",
                    "OBSERVADA",
                }:
                    return value

    text = " ".join(
        df.astype(str).fillna("").values.flatten().tolist()
    ).upper()

    if "NO-GO" in text:
        return "NO-GO"

    if re.search(r"\bGO\b", text):
        return "GO"

    return None


# ============================================================================
# PROCESO 42
# ============================================================================

def validate_process_42() -> Dict[str, Any]:
    path = DATA_ROOT / FILES["cierre_42"]

    if not path.exists():
        detail = f"No existe {path.name}"

        add_control(
            "Cierre proceso 42",
            "ERROR",
            "SI",
            detail,
        )

        return {
            "dictamen": None,
            "score": None,
            "ok": False,
        }

    try:
        df = safe_read_csv(path)

        ok, error, unknown = result_counts(df)

        dictamen = extract_dictamen_from_dataframe(df)
        score = extract_score_from_dataframe(df)

        valid = (
            error == 0
            and unknown == 0
            and dictamen == "GO"
            and (
                score is None
                or score >= 100.0
            )
        )

        detail = (
            f"Registros={len(df)} "
            f"OK={ok} "
            f"ERROR={error} "
            f"DESCONOCIDOS={unknown} "
            f"Dictamen={dictamen} "
            f"Score={score}"
        )

        add_control(
            "Cierre proceso 42",
            "OK" if valid else "ERROR",
            "SI",
            detail,
        )

        return {
            "dictamen": dictamen,
            "score": score,
            "ok": valid,
        }

    except Exception as exc:
        add_control(
            "Cierre proceso 42",
            "ERROR",
            "SI",
            f"Error de lectura: {exc}",
        )

        return {
            "dictamen": None,
            "score": None,
            "ok": False,
        }


# ============================================================================
# PROCESO 43
# ============================================================================

def validate_process_43() -> Dict[str, Any]:
    path = PACKAGE_DIR / FILES["control_paquete"]

    if not path.exists():
        path = DATA_ROOT / FILES["control_43"]

    if not path.exists():
        add_control(
            "Paquete proceso 43",
            "ERROR",
            "SI",
            "No existe el control del proceso 43.",
        )

        return {
            "dictamen": None,
            "score": None,
            "ok": False,
        }

    try:
        df = safe_read_csv(path)

        ok, error, unknown = result_counts(df)

        dictamen = extract_dictamen_from_dataframe(df)
        score = extract_score_from_dataframe(df)

        valid = (
            error == 0
            and unknown == 0
            and dictamen == "GO"
            and (
                score is None
                or score >= 100.0
            )
        )

        detail = (
            f"Controles={len(df)} "
            f"OK={ok} "
            f"ERROR={error} "
            f"DESCONOCIDOS={unknown} "
            f"Dictamen={dictamen} "
            f"Score={score}"
        )

        add_control(
            "Paquete proceso 43",
            "OK" if valid else "ERROR",
            "SI",
            detail,
        )

        return {
            "dictamen": dictamen,
            "score": score,
            "ok": valid,
        }

    except Exception as exc:
        add_control(
            "Paquete proceso 43",
            "ERROR",
            "SI",
            f"Error: {exc}",
        )

        return {
            "dictamen": None,
            "score": None,
            "ok": False,
        }


# ============================================================================
# PROCESO 44
# ============================================================================

def validate_process_44() -> Dict[str, Any]:
    path = DATA_ROOT / FILES["auditoria_44"]

    if not path.exists():
        add_control(
            "Auditoría proceso 44",
            "ERROR",
            "SI",
            f"No existe {path.name}",
        )

        return {
            "dictamen": None,
            "score": None,
            "ok": False,
        }

    try:
        df = safe_read_csv(path)

        result_col = find_column(
            df,
            [
                "resultado",
                "estado",
            ],
        )

        if result_col:
            values = (
                df[result_col]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            ok = int(
                values.isin(
                    ["OK", "GO", "PASS", "PASSED"]
                ).sum()
            )

            error = int(
                values.isin(
                    ["ERROR", "FAIL", "FAILED"]
                ).sum()
            )

            unknown = len(df) - ok - error

        else:
            ok, error, unknown = result_counts(df)

        dictamen = extract_dictamen_from_dataframe(df)
        score = extract_score_from_dataframe(df)

        # El 44 también puede tener el dictamen en el JSON.
        if dictamen is None:
            json_path = DATA_ROOT / FILES["resumen_44"]

            if json_path.exists():
                try:
                    data = safe_read_json(json_path)

                    for key in [
                        "dictamen",
                        "dictamen_final",
                        "auditoria",
                        "resultado",
                    ]:
                        if key in data:
                            candidate = upper_text(data[key])

                            if candidate in {
                                "GO",
                                "NO-GO",
                                "NOGO",
                            }:
                                dictamen = candidate
                                break

                except Exception:
                    pass

        valid = (
            error == 0
            and unknown == 0
            and dictamen == "GO"
            and (
                score is None
                or score >= 100.0
            )
        )

        detail = (
            f"Controles={len(df)} "
            f"OK={ok} "
            f"ERROR={error} "
            f"DESCONOCIDOS={unknown} "
            f"Dictamen={dictamen} "
            f"Score={score}"
        )

        add_control(
            "Auditoría proceso 44",
            "OK" if valid else "ERROR",
            "SI",
            detail,
        )

        return {
            "dictamen": dictamen,
            "score": score,
            "ok": valid,
        }

    except Exception as exc:
        add_control(
            "Auditoría proceso 44",
            "ERROR",
            "SI",
            f"Error: {exc}",
        )

        return {
            "dictamen": None,
            "score": None,
            "ok": False,
        }


# ============================================================================
# INVENTARIO FÍSICO
# ============================================================================

def inventory_package() -> Dict[str, Any]:
    missing_dirs = [
        str(path)
        for path in REQUIRED_PACKAGE_DIRS
        if not path.exists()
    ]

    missing_files = [
        str(path)
        for path in REQUIRED_PACKAGE_FILES
        if not path.exists()
    ]

    empty_files = [
        str(path)
        for path in REQUIRED_PACKAGE_FILES
        if path.exists() and is_empty_file(path)
    ]

    physical_files = []

    if PACKAGE_DIR.exists():
        for path in PACKAGE_DIR.rglob("*"):
            if path.is_file():
                physical_files.append(path)

    valid = (
        len(missing_dirs) == 0
        and len(missing_files) == 0
        and len(empty_files) == 0
    )

    detail = (
        f"Directorios faltantes={len(missing_dirs)} "
        f"Archivos obligatorios faltantes={len(missing_files)} "
        f"Archivos obligatorios vacíos={len(empty_files)} "
        f"Archivos físicos={len(physical_files)}"
    )

    add_control(
        "Inventario y estructura física",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )

    return {
        "physical_files": physical_files,
        "missing_dirs": missing_dirs,
        "missing_files": missing_files,
        "empty_files": empty_files,
        "ok": valid,
    }


# ============================================================================
# CARGA DE PRODUCTOS
# ============================================================================

def load_products() -> Dict[str, Any]:
    paths = {
        "proyectos": DATA_ROOT / FILES["proyectos"],
        "escenarios": DATA_ROOT / FILES["escenarios"],
        "ranking_proyectos": DATA_ROOT / FILES["ranking_proyectos"],
        "ranking_escenarios": DATA_ROOT / FILES["ranking_escenarios"],
        "indicadores": DATA_ROOT / FILES["indicadores"],
        "ejecutivos_proyectos": PACKAGE_DIR / FILES["ejecutivos_proyectos"],
        "ejecutivos_escenarios": PACKAGE_DIR / FILES["ejecutivos_escenarios"],
        "top_proyectos": PACKAGE_DIR / FILES["top_proyectos"],
        "ranking_ejecutivo_escenarios": (
            PACKAGE_DIR / FILES["ranking_ejecutivo_escenarios"]
        ),
        "indicadores_ejecutivos": (
            PACKAGE_DIR / FILES["indicadores_ejecutivos"]
        ),
    }

    products = {}
    errors = []

    for key, path in paths.items():
        try:
            products[key] = safe_read_csv(path)

        except Exception as exc:
            products[key] = None
            errors.append(
                f"{key}: {exc}"
            )

    valid = len(errors) == 0

    detail = (
        f"Productos cargados={sum(v is not None for v in products.values())}/"
        f"{len(products)}"
    )

    if errors:
        detail += " | " + " ; ".join(errors)

    add_control(
        "Carga independiente de productos maestros",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )

    return products


# ============================================================================
# ESTRUCTURA MAESTRA
# ============================================================================

def validate_master_structure(
    products: Dict[str, Any],
) -> Dict[str, Any]:
    proyectos = products.get("proyectos")
    escenarios = products.get("escenarios")

    if proyectos is None or escenarios is None:
        add_control(
            "Estructura modelo maestro",
            "ERROR",
            "SI",
            "No se pudieron cargar proyectos y/o escenarios.",
        )

        return {}

    proyecto_col = find_column(
        proyectos,
        [
            "proyecto_id",
            "id_proyecto",
        ],
    )

    escenario_col_proyecto = find_column(
        proyectos,
        [
            "escenario_id",
        ],
    )

    escenario_col = find_column(
        escenarios,
        [
            "escenario_id",
            "id_escenario",
        ],
    )

    if not proyecto_col or not escenario_col:
        add_control(
            "Estructura modelo maestro",
            "ERROR",
            "SI",
            "No se encontraron identificadores obligatorios.",
        )

        return {}

    proyecto_series = proyectos[proyecto_col]

    escenario_series = escenarios[escenario_col]

    project_nulls = int(proyecto_series.isna().sum())
    project_duplicates = int(
        proyecto_series.duplicated().sum()
    )

    scenario_nulls = int(escenario_series.isna().sum())
    scenario_duplicates = int(
        escenario_series.duplicated().sum()
    )

    valid = (
        project_nulls == 0
        and project_duplicates == 0
        and scenario_nulls == 0
        and scenario_duplicates == 0
    )

    detail = (
        f"Proyectos={len(proyectos)} "
        f"Proyectos únicos={proyecto_series.nunique(dropna=True)} "
        f"Proyectos nulos={project_nulls} "
        f"Proyectos duplicados={project_duplicates} "
        f"Escenarios={len(escenarios)} "
        f"Escenarios únicos={escenario_series.nunique(dropna=True)} "
        f"Escenarios nulos={scenario_nulls} "
        f"Escenarios duplicados={scenario_duplicates}"
    )

    add_control(
        "Estructura modelo maestro",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )

    return {
        "proyecto_col": proyecto_col,
        "escenario_col_proyecto": escenario_col_proyecto,
        "escenario_col": escenario_col,
    }


# ============================================================================
# PROYECTO -> ESCENARIO
# ============================================================================

def validate_project_scenario(
    products: Dict[str, Any],
    columns: Dict[str, Any],
) -> Dict[str, Any]:
    proyectos = products["proyectos"]
    escenarios = products["escenarios"]

    proyecto_col = columns["proyecto_col"]
    escenario_col_proyecto = columns["escenario_col_proyecto"]
    escenario_col = columns["escenario_col"]

    if escenario_col_proyecto is None:
        add_control(
            "Asignación proyecto-escenario",
            "ERROR",
            "SI",
            "El maestro de proyectos no contiene escenario_id.",
        )

        return {
            "project_ids": set(),
            "scenario_ids": set(),
        }

    project_scenarios = proyectos[
        escenario_col_proyecto
    ]

    nulls = int(
        project_scenarios.isna().sum()
    )

    multi = (
        proyectos.groupby(
            proyecto_col,
            dropna=False,
        )[escenario_col_proyecto]
        .nunique(dropna=True)
    )

    multiscenario = int(
        (multi > 1).sum()
    )

    master_scenarios = set(
        escenarios[escenario_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    project_scenarios_set = set(
        project_scenarios
        .dropna()
        .astype(str)
        .str.strip()
    )

    extra = project_scenarios_set - master_scenarios

    valid = (
        nulls == 0
        and multiscenario == 0
        and len(extra) == 0
    )

    detail = (
        f"Escenarios nulos={nulls} "
        f"Proyectos multiescenario={multiscenario} "
        f"Escenarios extra={len(extra)}"
    )

    add_control(
        "Asignación proyecto-escenario",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )

    return {
        "project_ids": set(
            proyectos[proyecto_col]
            .dropna()
            .astype(str)
            .str.strip()
        ),
        "scenario_ids": master_scenarios,
    }


# ============================================================================
# RANKINGS
# ============================================================================

def validate_ranking(
    df: Optional[pd.DataFrame],
    id_candidates: Sequence[str],
    ranking_candidates: Sequence[str],
    expected_count: int,
    label: str,
) -> bool:
    if df is None:
        add_control(
            f"Ranking {label} completo",
            "ERROR",
            "SI",
            "No se pudo cargar el ranking.",
        )

        add_control(
            f"Ranking {label} ordenado",
            "ERROR",
            "SI",
            "No se pudo validar el ranking.",
        )

        return False

    id_col = find_column(df, id_candidates)
    ranking_col = find_column(df, ranking_candidates)

    complete = (
        id_col is not None
        and ranking_col is not None
        and len(df) == expected_count
        and df[id_col].notna().all()
        and df[id_col].nunique() == expected_count
    )

    detail = (
        f"Campo ranking={ranking_col} "
        f"Registros={len(df)} "
        f"Esperados={expected_count}"
    )

    add_control(
        f"Ranking {label} completo",
        "OK" if complete else "ERROR",
        "SI",
        detail,
    )

    ordered = False

    if ranking_col is not None:
        ranking_series = numeric_series(
            df,
            ranking_col,
        )

        values = ranking_series.dropna().tolist()

        ordered = (
            len(values) == expected_count
            and set(values) == set(range(1, expected_count + 1))
        )

    add_control(
        f"Ranking {label} ordenado",
        "OK" if ordered else "ERROR",
        "SI",
        (
            f"Secuencia 1..{expected_count}"
            if ordered
            else "Secuencia de ranking inválida."
        ),
    )

    return complete and ordered


# ============================================================================
# DISTRIBUCIÓN TERRITORIAL
# ============================================================================

def validate_territorial_distribution(
    products: Dict[str, Any],
    columns: Dict[str, Any],
) -> Dict[str, Any]:
    proyectos = products["proyectos"]

    escenario_col = columns["escenario_col_proyecto"]

    if escenario_col is None:
        add_control(
            "Distribución territorial",
            "ERROR",
            "SI",
            "No existe escenario_id.",
        )

        return {}

    counts = (
        proyectos[escenario_col]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    if counts.empty:
        add_control(
            "Distribución territorial",
            "ERROR",
            "SI",
            "No existen asignaciones territoriales.",
        )

        return {}

    minimum = int(counts.min())
    maximum = int(counts.max())
    mean = float(counts.mean())

    cv = (
        float(counts.std(ddof=0) / mean)
        if mean != 0
        else 0.0
    )

    # No exigimos una distribución exactamente uniforme.
    # Solamente verificamos ausencia de escenarios vacíos y una
    # distribución razonablemente consistente con el modelo actual.
    valid = (
        len(counts) == 7
        and minimum > 0
        and maximum > 0
        and math.isfinite(cv)
    )

    detail = (
        f"Escenarios={len(counts)} "
        f"Mínimo={minimum} "
        f"Máximo={maximum} "
        f"Promedio={mean:.2f} "
        f"CV={cv:.4f}"
    )

    add_control(
        "Distribución territorial",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )

    return {
        "counts": {
            str(k): int(v)
            for k, v in counts.items()
        },
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "cv": cv,
    }


# ============================================================================
# TABLAS EJECUTIVAS
# ============================================================================

def validate_executive_tables(
    products: Dict[str, Any],
) -> Dict[str, Any]:
    expected = {
        "ejecutivos_proyectos": (
            144,
            "proyecto_id",
            "Tabla ejecutiva proyectos",
        ),
        "ejecutivos_escenarios": (
            7,
            "escenario_id",
            "Tabla ejecutiva escenarios",
        ),
        "top_proyectos": (
            20,
            "proyecto_id",
            "Top proyectos",
        ),
        "ranking_ejecutivo_escenarios": (
            7,
            "escenario_id",
            "Ranking ejecutivo escenarios",
        ),
        "indicadores_ejecutivos": (
            35,
            None,
            "Indicadores ejecutivos",
        ),
    }

    results = {}

    for key, (
        expected_count,
        expected_id,
        label,
    ) in expected.items():

        df = products.get(key)

        if df is None:
            valid = False
            detail = "Producto no disponible."

        else:
            id_ok = True

            if expected_id:
                id_col = find_column(
                    df,
                    [expected_id],
                )

                id_ok = (
                    id_col is not None
                    and df[id_col].notna().all()
                    and df[id_col].nunique() == expected_count
                )

            valid = (
                len(df) == expected_count
                and id_ok
            )

            detail = (
                f"Registros={len(df)} "
                f"Esperados={expected_count}"
            )

            if expected_id:
                detail += (
                    f" ID={expected_id}"
                )

        add_control(
            label,
            "OK" if valid else "ERROR",
            "SI",
            detail,
        )

        results[key] = valid

    return results


# ============================================================================
# CRUCE DE IDENTIFICADORES
# ============================================================================

def validate_id_crosswalk(
    products: Dict[str, Any],
    columns: Dict[str, Any],
) -> None:
    proyectos = products["proyectos"]
    escenarios = products["escenarios"]

    proyecto_col = columns["proyecto_col"]
    escenario_col = columns["escenario_col"]

    executive_projects = products[
        "ejecutivos_proyectos"
    ]

    executive_scenarios = products[
        "ejecutivos_escenarios"
    ]

    executive_project_col = find_column(
        executive_projects,
        ["proyecto_id"],
    )

    executive_scenario_col = find_column(
        executive_scenarios,
        ["escenario_id"],
    )

    master_projects = set(
        proyectos[proyecto_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    exec_projects = set(
        executive_projects[executive_project_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    missing_projects = master_projects - exec_projects
    extra_projects = exec_projects - master_projects

    valid_projects = (
        len(missing_projects) == 0
        and len(extra_projects) == 0
    )

    add_control(
        "Cruce maestro-proyectos ejecutivos",
        "OK" if valid_projects else "ERROR",
        "SI",
        (
            f"Faltantes={len(missing_projects)} "
            f"Extras={len(extra_projects)}"
        ),
    )

    master_scenarios = set(
        escenarios[escenario_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    exec_scenarios = set(
        executive_scenarios[executive_scenario_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    missing_scenarios = (
        master_scenarios - exec_scenarios
    )

    extra_scenarios = (
        exec_scenarios - master_scenarios
    )

    valid_scenarios = (
        len(missing_scenarios) == 0
        and len(extra_scenarios) == 0
    )

    add_control(
        "Cruce maestro-escenarios ejecutivos",
        "OK" if valid_scenarios else "ERROR",
        "SI",
        (
            f"Faltantes={len(missing_scenarios)} "
            f"Extras={len(extra_scenarios)}"
        ),
    )


# ============================================================================
# DOCUMENTOS
# ============================================================================

def validate_documents() -> None:
    documents = [
        (
            "Síntesis ejecutiva",
            PACKAGE_DIR / FILES["sintesis"],
            500,
        ),
        (
            "Informe ejecutivo",
            PACKAGE_DIR / FILES["informe"],
            1000,
        ),
    ]

    for label, path, minimum_chars in documents:

        if not path.exists():
            add_control(
                label,
                "ERROR",
                "SI",
                "Archivo inexistente.",
            )

            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            length = len(text)

            valid = (
                length >= minimum_chars
                and not text.isspace()
            )

            add_control(
                label,
                "OK" if valid else "ERROR",
                "SI",
                f"Caracteres={length}",
            )

        except Exception as exc:
            add_control(
                label,
                "ERROR",
                "SI",
                f"Error de lectura: {exc}",
            )


# ============================================================================
# MANIFIESTO
# ============================================================================

def normalize_manifest_reference(
    value: Any,
) -> str:
    """
    Normaliza una referencia proveniente del manifiesto.

    Puede recibir:
        producto
        ruta
        archivo
        MODELO
        tablas/foo.csv
        documentos/foo.md
        control/foo.csv

    Devuelve una referencia lógica normalizada.
    """
    text = normalize_text(value)

    if not text:
        return ""

    text = text.strip()
    text = text.replace("\\", "/")

    # Quitamos prefijos comunes.
    text = re.sub(
        r"^\./+",
        "",
        text,
    )

    return text


def resolve_manifest_reference(
    reference: str,
) -> Optional[Path]:
    """
    Resuelve una referencia del manifiesto contra el paquete.

    MUY IMPORTANTE:
    referencias lógicas como MODELO no se consideran archivos físicos.
    """

    ref = normalize_manifest_reference(reference)

    if not ref:
        return None

    # Referencias puramente lógicas.
    logical_tokens = {
        "MODELO",
        "MODELO_MAESTRO",
        "MODELO TERRITORIAL",
        "PAQUETE",
        "PRODUCTO",
        "CONTROL",
        "TABLAS",
        "DOCUMENTOS",
    }

    if ref.upper() in logical_tokens:
        return None

    # Si contiene una ruta absoluta, solamente la aceptamos
    # si está dentro del paquete.
    candidate = Path(ref)

    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()

            if (
                PACKAGE_DIR.resolve()
                in resolved.parents
                or resolved == PACKAGE_DIR.resolve()
            ):
                return resolved

        except Exception:
            return None

        return None

    # Normalización de posibles prefijos.
    candidates = [
        PACKAGE_DIR / ref,
        PACKAGE_DIR / ref.lstrip("/"),
    ]

    # Si el manifiesto utiliza solo el nombre del archivo,
    # buscamos dentro del paquete.
    if "/" not in ref:
        for path in PACKAGE_DIR.rglob(ref):
            candidates.append(path)

    # Algunos manifiestos pueden incluir prefijos como:
    # paquete_ejecutivo_amba_v4_1/archivo.csv
    prefix = PACKAGE_DIR.name + "/"

    if ref.startswith(prefix):
        candidates.append(
            PACKAGE_DIR / ref[len(prefix):]
        )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()

            if not resolved.exists():
                continue

            if (
                resolved == PACKAGE_DIR.resolve()
                or PACKAGE_DIR.resolve() in resolved.parents
            ):
                return resolved

        except Exception:
            continue

    return None


def validate_manifest() -> Dict[str, Any]:
    manifest_path = (
        PACKAGE_DIR / FILES["manifiesto_paquete"]
    )

    if not manifest_path.exists():
        # Fallback al manifiesto que podría existir en DATA_ROOT.
        manifest_path = (
            DATA_ROOT / FILES["manifiesto_43"]
        )

    if not manifest_path.exists():
        add_control(
            "Manifiesto paquete 43",
            "ERROR",
            "SI",
            "No existe el manifiesto.",
        )

        add_control(
            "Integridad SHA-256 paquete",
            "ERROR",
            "SI",
            "No existe el manifiesto.",
        )

        return {
            "path": None,
            "references": [],
            "physical": [],
            "missing": [],
            "hash_column": None,
            "file_column": None,
            "ok": False,
        }

    try:
        df = safe_read_csv(manifest_path)

        file_col = find_column(
            df,
            [
                "producto",
                "archivo",
                "file",
                "filename",
                "ruta",
                "path",
                "archivo_producto",
            ],
        )

        hash_col = find_column(
            df,
            [
                "sha256",
                "hash",
                "sha_256",
                "checksum",
            ],
        )

        if file_col is None:
            add_control(
                "Manifiesto paquete 43",
                "ERROR",
                "SI",
                "No se detectó columna de producto/archivo.",
            )

            add_control(
                "Integridad SHA-256 paquete",
                "ERROR",
                "SI",
                "No se detectó columna de producto/archivo.",
            )

            return {
                "path": manifest_path,
                "references": [],
                "physical": [],
                "missing": [],
                "hash_column": hash_col,
                "file_column": file_col,
                "ok": False,
            }

        references = []

        for _, row in df.iterrows():
            reference = normalize_manifest_reference(
                row[file_col]
            )

            if reference:
                references.append(
                    {
                        "reference": reference,
                        "hash": (
                            normalize_text(row[hash_col])
                            if hash_col
                            else ""
                        ),
                    }
                )

        physical = []
        missing = []
        logical = []

        for item in references:
            path = resolve_manifest_reference(
                item["reference"]
            )

            if path is None:
                # Si no existe como archivo, puede ser una
                # referencia lógica.
                token = item["reference"].upper()

                if token in {
                    "MODELO",
                    "MODELO_MAESTRO",
                    "MODELO TERRITORIAL",
                    "PAQUETE",
                    "PRODUCTO",
                }:
                    logical.append(item)
                else:
                    missing.append(item)

                continue

            physical.append(
                {
                    "reference": item["reference"],
                    "path": path,
                    "hash": item["hash"],
                }
            )

        valid = len(missing) == 0

        detail = (
            f"Registros={len(df)} "
            f"Referencias físicas={len(physical)} "
            f"Referencias lógicas={len(logical)} "
            f"Archivos no encontrados={len(missing)}"
        )

        add_control(
            "Manifiesto paquete 43",
            "OK" if valid else "ERROR",
            "SI",
            detail,
        )

        return {
            "path": manifest_path,
            "references": references,
            "physical": physical,
            "logical": logical,
            "missing": missing,
            "hash_column": hash_col,
            "file_column": file_col,
            "ok": valid,
        }

    except Exception as exc:
        add_control(
            "Manifiesto paquete 43",
            "ERROR",
            "SI",
            f"Error de lectura: {exc}",
        )

        return {
            "path": manifest_path,
            "references": [],
            "physical": [],
            "missing": [],
            "hash_column": None,
            "file_column": None,
            "ok": False,
        }


# ============================================================================
# SHA-256
# ============================================================================

def validate_hashes(
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    physical = manifest.get(
        "physical",
        [],
    )

    if not physical:
        # Si el manifiesto no contiene hashes físicos,
        # no declaramos automáticamente que hay corrupción.
        # Pero el cierre exige evidencia de integridad.
        add_control(
            "Integridad SHA-256 paquete",
            "ERROR",
            "SI",
            "No se encontraron referencias físicas con SHA-256.",
        )

        return []

    records = []
    errors = 0

    for item in physical:
        path = item["path"]

        expected = normalize_text(
            item.get("hash", "")
        ).lower()

        try:
            actual = sha256_file(path)

            if expected:
                valid = actual == expected
            else:
                # Si no hay hash esperado, registramos el hash
                # para auditoría, pero no lo consideramos mismatch.
                valid = True

            if not valid:
                errors += 1

            records.append(
                {
                    "archivo": relative_package_path(path),
                    "sha256_esperado": expected,
                    "sha256_actual": actual,
                    "resultado": (
                        "OK"
                        if valid
                        else "ERROR"
                    ),
                }
            )

        except Exception as exc:
            errors += 1

            records.append(
                {
                    "archivo": relative_package_path(path),
                    "sha256_esperado": expected,
                    "sha256_actual": "",
                    "resultado": "ERROR",
                    "detalle": str(exc),
                }
            )

    valid = (
        len(physical) > 0
        and errors == 0
    )

    detail = (
        f"Archivos SHA-256 validados={len(physical)} "
        f"Errores SHA-256={errors}"
    )

    add_control(
        "Integridad SHA-256 paquete",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )

    return records


# ============================================================================
# PRODUCTOS OBLIGATORIOS
# ============================================================================

def validate_required_products() -> None:
    required = [
        DATA_ROOT / FILES["proyectos"],
        DATA_ROOT / FILES["escenarios"],
        DATA_ROOT / FILES["ranking_proyectos"],
        DATA_ROOT / FILES["ranking_escenarios"],
        DATA_ROOT / FILES["indicadores"],
        PACKAGE_DIR / FILES["ejecutivos_proyectos"],
        PACKAGE_DIR / FILES["ejecutivos_escenarios"],
        PACKAGE_DIR / FILES["top_proyectos"],
        PACKAGE_DIR / FILES["ranking_ejecutivo_escenarios"],
        PACKAGE_DIR / FILES["indicadores_ejecutivos"],
        PACKAGE_DIR / FILES["sintesis"],
        PACKAGE_DIR / FILES["informe"],
        PACKAGE_DIR / FILES["control_paquete"],
        PACKAGE_DIR / FILES["resumen_43"],
        PACKAGE_DIR / FILES["manifiesto_paquete"],
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    empty = [
        str(path)
        for path in required
        if path.exists() and is_empty_file(path)
    ]

    valid = (
        len(missing) == 0
        and len(empty) == 0
    )

    detail = (
        f"Productos obligatorios={len(required)} "
        f"Faltantes={len(missing)} "
        f"Vacíos={len(empty)}"
    )

    add_control(
        "Inventario productos obligatorios",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )


# ============================================================================
# INDICADORES GLOBALES
# ============================================================================

def validate_global_indicators(
    products: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    df = products.get("indicadores")

    if df is None:
        add_control(
            "Indicadores globales",
            "ERROR",
            "SI",
            "No se pudo cargar indicadores_globales_amba_v4.csv",
        )

        return None

    indicator_col = find_column(
        df,
        [
            "indicador",
            "nombre_indicador",
            "metric",
        ],
    )

    value_col = find_column(
        df,
        [
            "valor",
            "value",
        ],
    )

    valid = (
        indicator_col is not None
        and value_col is not None
        and len(df) > 0
    )

    detail = (
        f"Indicadores={len(df)} "
        f"Campo indicador={indicator_col} "
        f"Campo valor={value_col}"
    )

    add_control(
        "Indicadores globales",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )

    return df


# ============================================================================
# COHERENCIA NUMÉRICA
# ============================================================================

NUMERIC_FIELDS = [
    "score_priorizacion_v4",
    "indice_demanda_estructural",
    "deficit_infraestructura",
    "indice_conectividad_estructural",
    "indice_intermodalidad_estructural",
    "indice_integracion_territorial",
    "indice_centralidad_estructural",
    "impacto_potencial",
    "urgencia_intervencion",
    "score_final_proyecto_v4",
    "prioridad_territorial_v4",
]


def validate_numeric_coherence(
    products: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Control numérico robusto.

    CORRECCIÓN IMPORTANTE:
    nunca hacemos int(series) ni asumimos que df[col] devuelve Series.
    numeric_series() resuelve columnas duplicadas.

    Además:
    - Campos categóricos no se fuerzan a numéricos.
    - NULOS se controlan solamente cuando el campo debe ser numérico.
    - Valores infinitos se detectan correctamente.
    - prioridad_territorial_v4 se reconoce como categórica.
    """

    proyectos = products.get("proyectos")

    if proyectos is None:
        add_control(
            "Coherencia numérica",
            "ERROR",
            "NO",
            "No se pudo cargar el modelo de proyectos.",
        )

        return {
            "evaluated": 0,
            "anomalies": 1,
            "details": [],
        }

    anomalies = []
    evaluated = 0

    for field in NUMERIC_FIELDS:

        col = find_column(
            proyectos,
            [field],
        )

        if col is None:
            continue

        # ------------------------------------------------------------
        # PRIORIDAD TERRITORIAL ES CATEGÓRICA, NO NUMÉRICA
        # ------------------------------------------------------------
        if field == "prioridad_territorial_v4":

            series = proyectos[col]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            null_count = int(
                series.isna().sum()
            )

            evaluated += 1

            # Los valores permitidos observados en V4.
            allowed = {
                "PRIORIDAD_1_MUY_ALTA",
                "PRIORIDAD_2_ALTA",
                "PRIORIDAD_3_MEDIA",
                "PRIORIDAD_4_MEDIA_BAJA",
                "PRIORIDAD_5_BAJA",
            }

            values = (
                series
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
            )

            invalid_values = sorted(
                set(values) - allowed
            )

            if null_count > 0:
                anomalies.append(
                    f"{field}: NULOS={null_count}"
                )

            if invalid_values:
                anomalies.append(
                    f"{field}: VALORES_NO_RECONOCIDOS="
                    f"{invalid_values[:10]}"
                )

            continue

        # ------------------------------------------------------------
        # CAMPOS NUMÉRICOS
        # ------------------------------------------------------------

        series = numeric_series(
            proyectos,
            col,
        )

        evaluated += 1

        null_count = int(
            series.isna().sum()
        )

        finite_mask = series.notna()

        if finite_mask.any():
            finite_values = series[finite_mask]

            infinite_count = int(
                (~finite_values.apply(math.isfinite)).sum()
            )
        else:
            infinite_count = 0

        if null_count > 0:
            anomalies.append(
                f"{field}: NULOS={null_count}"
            )

        if infinite_count > 0:
            anomalies.append(
                f"{field}: INFINITOS={infinite_count}"
            )

    # ------------------------------------------------------------
    # RANGOS ESPERADOS
    # ------------------------------------------------------------

    range_rules = {
        "score_priorizacion_v4": (0, 100),
        "indice_demanda_estructural": (0, 100),
        "deficit_infraestructura": (0, 100),
        "indice_conectividad_estructural": (0, 100),
        "indice_intermodalidad_estructural": (0, 100),
        "indice_integracion_territorial": (0, 100),
        "indice_centralidad_estructural": (0, 100),
        "impacto_potencial": (0, 100),
        "urgencia_intervencion": (0, 100),
        "score_final_proyecto_v4": (0, 100),
    }

    for field, (lower, upper) in range_rules.items():

        col = find_column(
            proyectos,
            [field],
        )

        if col is None:
            continue

        series = numeric_series(
            proyectos,
            col,
        )

        valid_values = series.dropna()

        if valid_values.empty:
            continue

        below = int(
            (valid_values < lower).sum()
        )

        above = int(
            (valid_values > upper).sum()
        )

        if below > 0:
            anomalies.append(
                f"{field}: BAJO_MIN={below}"
            )

        if above > 0:
            anomalies.append(
                f"{field}: SOBRE_MAX={above}"
            )

    valid = len(anomalies) == 0

    detail = (
        f"Campos evaluados={evaluated} "
        f"Anomalías={len(anomalies)}"
    )

    if anomalies:
        detail += " | " + " ; ".join(
            anomalies[:10]
        )

    add_control(
        "Coherencia numérica",
        "OK" if valid else "ERROR",
        "NO",
        detail,
    )

    return {
        "evaluated": evaluated,
        "anomalies": len(anomalies),
        "details": anomalies,
    }


# ============================================================================
# CORRESPONDENCIA DE PRODUCTOS
# ============================================================================

def validate_product_correspondence(
    products: Dict[str, Any],
) -> None:
    """
    Verifica que los principales productos ejecutivos tengan los mismos
    identificadores y no introduzcan proyectos/escenarios inexistentes.
    """

    proyectos = products["proyectos"]
    escenarios = products["escenarios"]

    proyecto_col = find_column(
        proyectos,
        ["proyecto_id"],
    )

    escenario_col = find_column(
        escenarios,
        ["escenario_id"],
    )

    master_projects = set(
        proyectos[proyecto_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    master_scenarios = set(
        escenarios[escenario_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Top proyectos
    top = products["top_proyectos"]

    top_col = find_column(
        top,
        ["proyecto_id"],
    )

    if top_col is None:
        add_control(
            "Correspondencia top proyectos",
            "ERROR",
            "SI",
            "No existe proyecto_id.",
        )
    else:
        top_ids = set(
            top[top_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        extra = top_ids - master_projects

        valid = (
            len(top) == 20
            and len(extra) == 0
        )

        add_control(
            "Correspondencia top proyectos",
            "OK" if valid else "ERROR",
            "SI",
            (
                f"Registros={len(top)} "
                f"IDs extra={len(extra)}"
            ),
        )

    # Ranking escenarios
    ranking = products[
        "ranking_ejecutivo_escenarios"
    ]

    ranking_col = find_column(
        ranking,
        ["escenario_id"],
    )

    if ranking_col is None:
        add_control(
            "Correspondencia ranking escenarios",
            "ERROR",
            "SI",
            "No existe escenario_id.",
        )

    else:
        ranking_ids = set(
            ranking[ranking_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        extra = ranking_ids - master_scenarios

        valid = (
            len(ranking) == 7
            and len(extra) == 0
        )

        add_control(
            "Correspondencia ranking escenarios",
            "OK" if valid else "ERROR",
            "SI",
            (
                f"Registros={len(ranking)} "
                f"IDs extra={len(extra)}"
            ),
        )


# ============================================================================
# COMPLETITUD FINAL
# ============================================================================

def validate_final_completeness() -> None:
    dirs = REQUIRED_PACKAGE_DIRS

    files = REQUIRED_PACKAGE_FILES

    missing_dirs = [
        path
        for path in dirs
        if not path.exists()
    ]

    missing_files = [
        path
        for path in files
        if not path.exists()
    ]

    valid = (
        len(missing_dirs) == 0
        and len(missing_files) == 0
    )

    detail = (
        f"Directorios faltantes={len(missing_dirs)} "
        f"Archivos faltantes={len(missing_files)}"
    )

    add_control(
        "Completitud paquete final",
        "OK" if valid else "ERROR",
        "SI",
        detail,
    )


# ============================================================================
# INVENTARIO DE ARCHIVOS AUDITADOS
# ============================================================================

def generate_inventory(
    physical_files: List[Path],
) -> List[Dict[str, Any]]:
    records = []

    for path in sorted(
        physical_files,
        key=lambda p: str(p).lower(),
    ):
        try:
            stat = path.stat()

            records.append(
                {
                    "archivo": relative_package_path(path),
                    "tipo": path.suffix.lower().lstrip("."),
                    "bytes": int(stat.st_size),
                    "vacio": (
                        "SI"
                        if stat.st_size == 0
                        else "NO"
                    ),
                    "sha256": sha256_file(path),
                }
            )

        except Exception as exc:
            records.append(
                {
                    "archivo": relative_package_path(path),
                    "tipo": path.suffix.lower().lstrip("."),
                    "bytes": "",
                    "vacio": "SI",
                    "sha256": "",
                    "error": str(exc),
                }
            )

    return records


# ============================================================================
# EXPORTACIÓN DE RESULTADOS
# ============================================================================

def export_results(
    inventory_records: List[Dict[str, Any]],
    hash_records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Path]:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        OUTPUT_DIR
        / "auditoria_45_cierre_amba_v4.csv"
    )

    inventory_path = (
        OUTPUT_DIR
        / "inventario_45_cierre_amba_v4.csv"
    )

    hashes_path = (
        OUTPUT_DIR
        / "hashes_45_cierre_amba_v4.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_45_cierre_amba_v4.json"
    )

    report_path = (
        OUTPUT_DIR
        / "informe_45_cierre_amba_v4.md"
    )

    pd.DataFrame(
        controls,
        columns=[
            "control",
            "resultado",
            "critico",
            "detalle",
        ],
    ).to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        inventory_records,
    ).to_csv(
        inventory_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        hash_records,
    ).to_csv(
        hashes_path,
        index=False,
        encoding="utf-8-sig",
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            summary,
            fh,
            indent=2,
            ensure_ascii=False,
        )

    report = build_report(
        summary,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return {
        "audit": audit_path,
        "inventory": inventory_path,
        "hashes": hashes_path,
        "summary": summary_path,
        "report": report_path,
    }


# ============================================================================
# INFORME
# ============================================================================

def build_report(
    summary: Dict[str, Any],
) -> str:

    result = summary["resultado"]

    lines = []

    lines.append(
        "# Auditoría independiente de cierre "
        "del modelo territorial AMBA V4.1"
    )
    lines.append("")
    lines.append(
        f"**Proceso:** 45"
    )
    lines.append(
        f"**Versión:** {VERSION}"
    )
    lines.append(
        f"**Fecha:** {summary['fecha']}"
    )
    lines.append("")

    lines.append("## Resultado")
    lines.append("")
    lines.append(
        f"- Controles: {result['controles_totales']}"
    )
    lines.append(
        f"- Controles OK: {result['controles_ok']}"
    )
    lines.append(
        f"- Fallas: {result['controles_error']}"
    )
    lines.append(
        f"- Fallas críticas: {result['fallas_criticas']}"
    )
    lines.append(
        f"- Fallas importantes: {result['fallas_importantes']}"
    )
    lines.append(
        f"- Score: {result['score']:.2f}/100"
    )
    lines.append(
        f"- Auditoría: **{result['auditoria']}**"
    )
    lines.append(
        f"- Dictamen: **{result['dictamen']}**"
    )
    lines.append("")

    lines.append("## Resumen estructural")
    lines.append("")
    lines.append(
        f"- Proyectos: {summary['proyectos']}"
    )
    lines.append(
        f"- Escenarios: {summary['escenarios']}"
    )
    lines.append(
        f"- Proceso 42: {summary['procesos']['42']['dictamen']}"
    )
    lines.append(
        f"- Proceso 43: {summary['procesos']['43']['dictamen']}"
    )
    lines.append(
        f"- Proceso 44: {summary['procesos']['44']['dictamen']}"
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
            ).replace("|", "/")
        )

        lines.append(
            f"| {control['control']} "
            f"| {control['resultado']} "
            f"| {control['critico']} "
            f"| {detail} |"
        )

    lines.append("")

    if result["controles_error"] == 0:
        lines.append("## Conclusión")
        lines.append("")
        lines.append(
            "La auditoría independiente de cierre "
            "no detectó inconsistencias."
        )
        lines.append("")
        lines.append(
            "El modelo territorial AMBA V4.1 y su paquete "
            "ejecutivo superaron los controles estructurales, "
            "documentales, de integridad y coherencia definidos "
            "para el proceso 45."
        )
        lines.append("")
        lines.append(
            "**DICTAMEN FINAL: GO**"
        )

    else:
        lines.append("## Observaciones")
        lines.append("")

        for control in controls:
            if control["resultado"] == "ERROR":
                lines.append(
                    f"- **{control['control']}**: "
                    f"{control['detalle']}"
                )

        lines.append("")
        lines.append(
            "**DICTAMEN FINAL: NO-GO**"
        )

    return "\n".join(lines) + "\n"


# ============================================================================
# RESUMEN
# ============================================================================

def build_summary(
    products: Dict[str, Any],
    process_42: Dict[str, Any],
    process_43: Dict[str, Any],
    process_44: Dict[str, Any],
    inventory: Dict[str, Any],
    numeric: Dict[str, Any],
    hash_records: List[Dict[str, Any]],
) -> Dict[str, Any]:

    total = len(controls)
    ok = controls_ok()
    error = controls_error()
    critical = critical_errors()
    important = important_errors()

    score = (
        100.0 * ok / total
        if total
        else 0.0
    )

    # Regla estricta de cierre:
    # GO solamente si no existe ningún error.
    final_go = (
        total > 0
        and error == 0
        and critical == 0
        and important == 0
        and process_42["ok"]
        and process_43["ok"]
        and process_44["ok"]
        and inventory["ok"]
        and numeric["anomalies"] == 0
    )

    auditoria = (
        "OK"
        if final_go
        else "OBSERVADA"
    )

    dictamen = (
        "GO"
        if final_go
        else "NO-GO"
    )

    proyectos = products.get("proyectos")

    escenarios = products.get("escenarios")

    project_count = (
        len(proyectos)
        if proyectos is not None
        else 0
    )

    scenario_count = (
        len(escenarios)
        if escenarios is not None
        else 0
    )

    return {
        "proceso": 45,
        "version": VERSION,
        "script": SCRIPT_NAME,
        "fecha": now_iso(),
        "proyecto_root": str(PROJECT_ROOT),
        "entrada": str(DATA_ROOT),
        "paquete": str(PACKAGE_DIR),
        "salida": str(OUTPUT_DIR),
        "proyectos": project_count,
        "escenarios": scenario_count,
        "procesos": {
            "42": {
                "dictamen": process_42["dictamen"],
                "score": process_42["score"],
                "ok": process_42["ok"],
            },
            "43": {
                "dictamen": process_43["dictamen"],
                "score": process_43["score"],
                "ok": process_43["ok"],
            },
            "44": {
                "dictamen": process_44["dictamen"],
                "score": process_44["score"],
                "ok": process_44["ok"],
            },
        },
        "inventario": {
            "archivos_fisicos": len(
                inventory["physical_files"]
            ),
            "directorios_faltantes": len(
                inventory["missing_dirs"]
            ),
            "archivos_faltantes": len(
                inventory["missing_files"]
            ),
            "archivos_vacios": len(
                inventory["empty_files"]
            ),
        },
        "sha256": {
            "archivos_validados": len(hash_records),
            "errores": sum(
                1
                for row in hash_records
                if row.get("resultado") == "ERROR"
            ),
        },
        "coherencia_numerica": numeric,
        "resultado": {
            "controles_totales": total,
            "controles_ok": ok,
            "controles_error": error,
            "fallas_criticas": critical,
            "fallas_importantes": important,
            "score": score,
            "auditoria": auditoria,
            "dictamen": dictamen,
        },
    }


# ============================================================================
# CONSOLA
# ============================================================================

def print_final_result(
    summary: Dict[str, Any],
    outputs: Dict[str, Path],
    elapsed: float,
) -> None:

    result = summary["resultado"]

    print_section(
        "18 - DETERMINACIÓN DEL DICTAMEN FINAL"
    )

    print_kv(
        "Controles OK",
        f"{result['controles_ok']}/{result['controles_totales']}",
    )

    print_kv(
        "Controles fallidos",
        result["controles_error"],
    )

    print_kv(
        "Fallas críticas",
        result["fallas_criticas"],
    )

    print_kv(
        "Fallas importantes",
        result["fallas_importantes"],
    )

    print_kv(
        "Score auditoría",
        f"{result['score']:.2f}/100",
    )

    print_kv(
        "Auditoría",
        result["auditoria"],
    )

    print_kv(
        "DICTAMEN FINAL",
        result["dictamen"],
    )

    print_section(
        "19 - EXPORTANDO RESULTADOS DE LA AUDITORÍA FINAL"
    )

    print_kv(
        "Auditoría",
        outputs["audit"],
    )

    print_kv(
        "Inventario",
        outputs["inventory"],
    )

    print_kv(
        "Hashes",
        outputs["hashes"],
    )

    print_kv(
        "Resumen",
        outputs["summary"],
    )

    print_kv(
        "Informe",
        outputs["report"],
    )

    print_section(
        "RESULTADO FINAL DEL PROCESO 45"
    )

    print_kv(
        "Proyectos",
        summary["proyectos"],
    )

    print_kv(
        "Escenarios",
        summary["escenarios"],
    )

    print_kv(
        "Proceso 42",
        summary["procesos"]["42"]["dictamen"],
    )

    print_kv(
        "Proceso 43",
        summary["procesos"]["43"]["dictamen"],
    )

    print_kv(
        "Proceso 44",
        summary["procesos"]["44"]["dictamen"],
    )

    print_kv(
        "Controles OK",
        f"{result['controles_ok']}/{result['controles_totales']}",
    )

    print_kv(
        "Fallas críticas",
        result["fallas_criticas"],
    )

    print_kv(
        "Fallas importantes",
        result["fallas_importantes"],
    )

    print_kv(
        "Score auditoría",
        f"{result['score']:.2f}/100",
    )

    print_kv(
        "Auditoría",
        result["auditoria"],
    )

    print_kv(
        "DICTAMEN FINAL",
        result["dictamen"],
    )

    print_kv(
        "Tiempo de ejecución",
        f"{elapsed:.2f} segundos",
    )

    print_section(
        "ARCHIVOS GENERADOS"
    )

    print_kv(
        "Directorio",
        OUTPUT_DIR,
    )

    for label, path in [
        ("Auditoría", outputs["audit"]),
        ("Inventario", outputs["inventory"]),
        ("Hashes", outputs["hashes"]),
        ("Resumen", outputs["summary"]),
        ("Informe", outputs["report"]),
    ]:
        print_kv(
            label,
            path,
        )

    print()

    if result["dictamen"] == "GO":
        print(
            "========================================================================================"
        )
        print(
            "45 - AUDITORÍA INDEPENDIENTE DE CIERRE - GO"
        )
        print(
            "========================================================================================"
        )
        print(
            "El modelo territorial AMBA V4.1 superó la auditoría independiente de cierre."
        )
        print(
            "El paquete ejecutivo fue validado estructuralmente."
        )
        print(
            "La integridad de archivos fue verificada."
        )
        print(
            "Los hashes SHA-256 fueron validados."
        )
        print(
            "La coherencia numérica fue validada."
        )
        print(
            "No se detectaron fallas críticas ni importantes."
        )
        print()
        print(
            "DICTAMEN FINAL: GO"
        )
        print(
            "========================================================================================"
        )

    else:
        print(
            "========================================================================================"
        )
        print(
            "45 - AUDITORÍA INDEPENDIENTE DE CIERRE - NO-GO"
        )
        print(
            "========================================================================================"
        )
        print(
            "Se detectaron inconsistencias en la auditoría independiente."
        )
        print(
            "Revisar auditoria_45_cierre_amba_v4.csv"
        )
        print(
            "Revisar informe_45_cierre_amba_v4.md"
        )
        print(
            "========================================================================================"
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    started = time.perf_counter()

    # ------------------------------------------------------------
    # CABECERA
    # ------------------------------------------------------------

    print("=" * 88)
    print(
        "45 - AUDITORÍA INDEPENDIENTE DE CIERRE "
        "DEL MODELO TERRITORIAL AMBA - V4.1"
    )
    print("=" * 88)

    print_kv(
        "Proyecto",
        PROJECT_ROOT,
    )

    print_kv(
        "Entrada",
        DATA_ROOT,
    )

    print_kv(
        "Paquete",
        PACKAGE_DIR,
    )

    print_kv(
        "Salida",
        OUTPUT_DIR,
    )

    # ------------------------------------------------------------
    # 1
    # ------------------------------------------------------------

    print_section(
        "1 - VALIDACIÓN INDEPENDIENTE DEL PROCESO 42"
    )

    process_42 = validate_process_42()

    # Mostrar datos si es posible.
    path42 = DATA_ROOT / FILES["cierre_42"]

    if path42.exists():
        try:
            df42 = safe_read_csv(path42)
            ok42, err42, unk42 = result_counts(df42)

            print_kv(
                "Registros cierre",
                len(df42),
            )
            print_kv(
                "Resultados OK",
                ok42,
            )
            print_kv(
                "Resultados ERROR",
                err42,
            )
            print_kv(
                "Desconocidos",
                unk42,
            )
            print_kv(
                "Dictamen proceso 42",
                process_42["dictamen"],
            )
            print_kv(
                "Score proceso 42",
                process_42["score"],
            )

        except Exception:
            pass

    # ------------------------------------------------------------
    # 2
    # ------------------------------------------------------------

    print_section(
        "2 - VALIDACIÓN INDEPENDIENTE DEL PROCESO 43"
    )

    process_43 = validate_process_43()

    path43 = PACKAGE_DIR / FILES["control_paquete"]

    if path43.exists():
        try:
            df43 = safe_read_csv(path43)
            ok43, err43, unk43 = result_counts(df43)

            print_kv(
                "Controles proceso 43",
                len(df43),
            )
            print_kv(
                "Controles OK",
                ok43,
            )
            print_kv(
                "Controles ERROR",
                err43,
            )
            print_kv(
                "Desconocidos",
                unk43,
            )
            print_kv(
                "Score proceso 43",
                process_43["score"],
            )
            print_kv(
                "Dictamen proceso 43",
                process_43["dictamen"],
            )

        except Exception:
            pass

    # ------------------------------------------------------------
    # 3
    # ------------------------------------------------------------

    print_section(
        "3 - VALIDACIÓN INDEPENDIENTE DEL PROCESO 44"
    )

    process_44 = validate_process_44()

    path44 = DATA_ROOT / FILES["auditoria_44"]

    if path44.exists():
        try:
            df44 = safe_read_csv(path44)
            ok44, err44, unk44 = result_counts(df44)

            print_kv(
                "Controles proceso 44",
                len(df44),
            )
            print_kv(
                "Controles OK",
                ok44,
            )
            print_kv(
                "Controles ERROR",
                err44,
            )
            print_kv(
                "Desconocidos",
                unk44,
            )
            print_kv(
                "Score proceso 44",
                process_44["score"],
            )
            print_kv(
                "Dictamen proceso 44",
                process_44["dictamen"],
            )

        except Exception:
            pass

    # ------------------------------------------------------------
    # 4
    # ------------------------------------------------------------

    print_section(
        "4 - INVENTARIO Y ESTRUCTURA FÍSICA DEL PAQUETE"
    )

    inventory = inventory_package()

    print_kv(
        "Directorios faltantes",
        len(inventory["missing_dirs"]),
    )

    print_kv(
        "Archivos obligatorios faltantes",
        len(inventory["missing_files"]),
    )

    print_kv(
        "Archivos obligatorios vacíos",
        len(inventory["empty_files"]),
    )

    print_kv(
        "Archivos físicos inventariados",
        len(inventory["physical_files"]),
    )

    # ------------------------------------------------------------
    # 5
    # ------------------------------------------------------------

    print_section(
        "5 - CARGA INDEPENDIENTE DE PRODUCTOS MAESTROS"
    )

    products = load_products()

    for key, df in products.items():
        if df is not None:
            print(
                f"{key:<32}: "
                f"{len(df)} registros | "
                f"{len(df.columns)} columnas"
            )

    # ------------------------------------------------------------
    # 6
    # ------------------------------------------------------------

    print_section(
        "6 - VALIDACIÓN ESTRUCTURAL DE PROYECTOS Y ESCENARIOS"
    )

    columns = validate_master_structure(
        products,
    )

    if products.get("proyectos") is not None:
        proyectos = products["proyectos"]

        proyecto_col = columns.get(
            "proyecto_col"
        )

        escenario_col_proyecto = columns.get(
            "escenario_col_proyecto"
        )

        print_kv(
            "Proyectos",
            len(proyectos),
        )

        if proyecto_col:
            print_kv(
                "Proyectos únicos",
                proyectos[proyecto_col].nunique(
                    dropna=True
                ),
            )

            print_kv(
                "Proyectos nulos",
                int(
                    proyectos[proyecto_col].isna().sum()
                ),
            )

            print_kv(
                "Proyectos duplicados",
                int(
                    proyectos[proyecto_col]
                    .duplicated()
                    .sum()
                ),
            )

        if products.get("escenarios") is not None:
            escenarios = products["escenarios"]

            escenario_col = columns.get(
                "escenario_col"
            )

            print_kv(
                "Escenarios",
                len(escenarios),
            )

            if escenario_col:
                print_kv(
                    "Escenarios únicos",
                    escenarios[escenario_col].nunique(
                        dropna=True
                    ),
                )

                print_kv(
                    "Escenarios nulos",
                    int(
                        escenarios[escenario_col]
                        .isna()
                        .sum()
                    ),
                )

                print_kv(
                    "Escenarios duplicados",
                    int(
                        escenarios[escenario_col]
                        .duplicated()
                        .sum()
                    ),
                )

    # ------------------------------------------------------------
    # 7
    # ------------------------------------------------------------

    print_section(
        "7 - VALIDACIÓN PROYECTO → ESCENARIO"
    )

    crosswalk = validate_project_scenario(
        products,
        columns,
    )

    # ------------------------------------------------------------
    # 8
    # ------------------------------------------------------------

    print_section(
        "8 - VALIDACIÓN INDEPENDIENTE DE RANKINGS"
    )

    proyectos = products.get("proyectos")
    escenarios = products.get("escenarios")

    expected_projects = (
        len(proyectos)
        if proyectos is not None
        else 144
    )

    expected_scenarios = (
        len(escenarios)
        if escenarios is not None
        else 7
    )

    validate_ranking(
        products.get("ranking_proyectos"),
        ["proyecto_id"],
        [
            "ranking_final_proyecto_v4",
            "ranking_final_proyecto",
        ],
        expected_projects,
        "proyectos",
    )

    validate_ranking(
        products.get("ranking_escenarios"),
        ["escenario_id"],
        [
            "ranking_integral_v4",
            "ranking_integral",
        ],
        expected_scenarios,
        "escenarios",
    )

    # ------------------------------------------------------------
    # 9 - DISTRIBUCIÓN
    # ------------------------------------------------------------

    print_section(
        "9 - DISTRIBUCIÓN TERRITORIAL"
    )

    distribution = validate_territorial_distribution(
        products,
        columns,
    )

    for key, value in distribution.get(
        "counts",
        {},
    ).items():
        print(
            f"  {key}: {value}"
        )

    if distribution:
        print_kv(
            "Mínimo",
            distribution.get("minimum"),
        )

        print_kv(
            "Máximo",
            distribution.get("maximum"),
        )

        print_kv(
            "Promedio",
            f"{distribution.get('mean', 0):.2f}",
        )

        print_kv(
            "CV",
            f"{distribution.get('cv', 0):.4f}",
        )

    # ------------------------------------------------------------
    # 10
    # ------------------------------------------------------------

    print_section(
        "10 - VALIDACIÓN DE TABLAS EJECUTIVAS"
    )

    validate_executive_tables(
        products,
    )

    # ------------------------------------------------------------
    # 11
    # ------------------------------------------------------------

    print_section(
        "11 - CONTROL CRUZADO DE IDENTIFICADORES"
    )

    if (
        products.get("proyectos") is not None
        and products.get("escenarios") is not None
        and products.get("ejecutivos_proyectos") is not None
        and products.get("ejecutivos_escenarios") is not None
    ):
        validate_id_crosswalk(
            products,
            columns,
        )

    # ------------------------------------------------------------
    # 12
    # ------------------------------------------------------------

    print_section(
        "12 - VALIDACIÓN DE DOCUMENTOS EJECUTIVOS"
    )

    validate_documents()

    # ------------------------------------------------------------
    # 13
    # ------------------------------------------------------------

    print_section(
        "13 - VALIDACIÓN INDEPENDIENTE DEL MANIFIESTO"
    )

    manifest = validate_manifest()

    if manifest["path"]:
        print_kv(
            "Manifiesto",
            manifest["path"].name,
        )

        print_kv(
            "Columna archivo/producto",
            manifest["file_column"],
        )

        print_kv(
            "Columna hash",
            manifest["hash_column"],
        )

        print_kv(
            "Referencias físicas",
            len(manifest.get("physical", [])),
        )

        print_kv(
            "Referencias lógicas",
            len(manifest.get("logical", [])),
        )

        print_kv(
            "Archivos no encontrados",
            len(manifest.get("missing", [])),
        )

    # ------------------------------------------------------------
    # 14
    # ------------------------------------------------------------

    print_section(
        "14 - VALIDACIÓN SHA-256 DEL PAQUETE"
    )

    hash_records = validate_hashes(
        manifest,
    )

    print_kv(
        "Archivos SHA-256 validados",
        len(hash_records),
    )

    print_kv(
        "Errores SHA-256",
        sum(
            1
            for row in hash_records
            if row.get("resultado") == "ERROR"
        ),
    )

    # ------------------------------------------------------------
    # 15
    # ------------------------------------------------------------

    print_section(
        "15 - VALIDACIÓN DE PRODUCTOS OBLIGATORIOS"
    )

    validate_required_products()

    # ------------------------------------------------------------
    # 16
    # ------------------------------------------------------------

    print_section(
        "16 - VALIDACIÓN DE INDICADORES GLOBALES"
    )

    validate_global_indicators(
        products,
    )

    # ------------------------------------------------------------
    # 17
    # ------------------------------------------------------------

    print_section(
        "17 - CONTROL DE COHERENCIA NUMÉRICA"
    )

    numeric = validate_numeric_coherence(
        products,
    )

    print_kv(
        "Indicadores/campos evaluados",
        numeric["evaluated"],
    )

    print_kv(
        "Anomalías numéricas",
        numeric["anomalies"],
    )

    for anomaly in numeric["details"]:
        print(
            f"  - {anomaly}"
        )

    # ------------------------------------------------------------
    # 18
    # ------------------------------------------------------------

    print_section(
        "18 - CONTROL DE CORRESPONDENCIA DE PRODUCTOS"
    )

    if (
        products.get("proyectos") is not None
        and products.get("escenarios") is not None
        and products.get("top_proyectos") is not None
        and products.get(
            "ranking_ejecutivo_escenarios"
        ) is not None
    ):
        validate_product_correspondence(
            products,
        )
    else:
        add_control(
            "Correspondencia de productos",
            "ERROR",
            "SI",
            "No se pudieron cargar productos requeridos.",
        )

    # ------------------------------------------------------------
    # 19
    # ------------------------------------------------------------

    print_section(
        "19 - CONTROL DE COMPLETITUD FINAL"
    )

    validate_final_completeness()

    # ------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------

    summary = build_summary(
        products,
        process_42,
        process_43,
        process_44,
        inventory,
        numeric,
        hash_records,
    )

    # Inventario físico independiente.
    inventory_records = generate_inventory(
        inventory["physical_files"],
    )

    outputs = export_results(
        inventory_records,
        hash_records,
        summary,
    )

    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------
    # 20 - RESULTADO
    # ------------------------------------------------------------

    print_section(
        "20 - DETERMINACIÓN DEL DICTAMEN FINAL"
    )

    print_kv(
        "Controles OK",
        f"{summary['resultado']['controles_ok']}/"
        f"{summary['resultado']['controles_totales']}",
    )

    print_kv(
        "Controles fallidos",
        summary["resultado"]["controles_error"],
    )

    print_kv(
        "Fallas críticas",
        summary["resultado"]["fallas_criticas"],
    )

    print_kv(
        "Fallas importantes",
        summary["resultado"]["fallas_importantes"],
    )

    print_kv(
        "Score auditoría",
        f"{summary['resultado']['score']:.2f}/100",
    )

    print_kv(
        "Auditoría",
        summary["resultado"]["auditoria"],
    )

    print_kv(
        "DICTAMEN FINAL",
        summary["resultado"]["dictamen"],
    )

    print_final_result(
        summary,
        outputs,
        elapsed,
    )

    return (
        0
        if summary["resultado"]["dictamen"] == "GO"
        else 1
    )


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:
        print()
        print(
            "Proceso 45 interrumpido por el usuario."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 88)
        print(
            "ERROR FATAL EN PROCESO 45"
        )
        print("=" * 88)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "El auditor no pudo completar la ejecución."
        )
        sys.exit(1)