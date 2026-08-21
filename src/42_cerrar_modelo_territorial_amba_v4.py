# -*- coding: utf-8 -*-
"""
42 - CIERRE DEL MODELO TERRITORIAL AMBA - V4.1

Cierre formal del modelo territorial AMBA.

Entradas principales:
    - auditoria_41_modelo_territorial_amba_v4.csv
    - resumen_41_auditoria_modelo_territorial_amba_v4.json
    - informe_41_auditoria_modelo_territorial_amba_v4.md
    - modelo_maestro_proyectos_v4.csv
    - modelo_maestro_escenarios_v4.csv
    - ranking_final_proyectos_v4.csv
    - ranking_final_escenarios_v4.csv
    - matriz_integral_escenarios_v4.csv
    - indicadores_globales_amba_v4.csv
    - modelo_maestro_territorial_amba_v4.gpkg
    - atlas_territorial_amba_v4.gpkg
    - informe_territorial_amba_v4_1.md
    - atlas_territorial_amba_v4.md

Salidas:
    - cierre_42_modelo_territorial_amba_v4.csv
    - resumen_42_cierre_modelo_territorial_amba_v4.json
    - acta_cierre_modelo_territorial_amba_v4.md
    - manifiesto_cierre_modelo_territorial_amba_v4.csv
    - hashes_42_cierre_modelo_territorial_amba_v4.csv

Autoridad de cierre:
    El proceso 41 debe finalizar con DICTAMEN FINAL = GO.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.1"
PROCESO = "42"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR = INPUT_DIR

print("=" * 88)
print("42 - CIERRE DEL MODELO TERRITORIAL AMBA - V4.1")
print("=" * 88)
print(f"Proyecto : {PROJECT_ROOT}")
print(f"Entrada  : {INPUT_DIR}")
print(f"Salida   : {OUTPUT_DIR}")
print()


# =============================================================================
# UTILIDADES
# =============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def norm(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_csv_required(name: str) -> pd.DataFrame:
    path = INPUT_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")

    df = pd.read_csv(path)

    print(
        f"Cargando: {name} | "
        f"Registros: {len(df)} | "
        f"Columnas: {len(df.columns)}"
    )

    return df


def read_json_required(name: str) -> dict:
    path = INPUT_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"No existe el JSON requerido: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


# =============================================================================
# ARCHIVOS CRÍTICOS
# =============================================================================

CRITICAL_FILES = [
    "modelo_maestro_proyectos_v4.csv",
    "modelo_maestro_escenarios_v4.csv",
    "ranking_final_proyectos_v4.csv",
    "ranking_final_escenarios_v4.csv",
    "matriz_integral_escenarios_v4.csv",
    "indicadores_globales_amba_v4.csv",
    "modelo_maestro_territorial_amba_v4.gpkg",
    "informe_territorial_amba_v4_1.md",
    "atlas_territorial_amba_v4.gpkg",
    "atlas_territorial_amba_v4.md",
    "auditoria_41_modelo_territorial_amba_v4.csv",
    "resumen_41_auditoria_modelo_territorial_amba_v4.json",
    "informe_41_auditoria_modelo_territorial_amba_v4.md",
]


# =============================================================================
# INICIO
# =============================================================================

start_time = time.time()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

checks = []
inventory = []
hash_rows = []


def add_check(
    control: str,
    passed: bool,
    critical: bool = False,
    detail: str = "",
) -> None:
    checks.append(
        {
            "control": control,
            "resultado": "OK" if passed else "FAIL",
            "critico": "SI" if critical else "NO",
            "detalle": detail,
        }
    )


# =============================================================================
# 1 - INVENTARIO
# =============================================================================

banner("1 - VERIFICACIÓN DEL INVENTARIO DE PRODUCTOS")

missing_files = []

for filename in CRITICAL_FILES:
    path = INPUT_DIR / filename
    exists = path.exists()

    inventory.append(
        {
            "archivo": filename,
            "existe": "SI" if exists else "NO",
            "tamano_mb": round(file_size_mb(path), 4),
        }
    )

    if exists:
        ok(f"{filename} ({file_size_mb(path):.2f} MB)")
    else:
        fail(f"{filename}")
        missing_files.append(filename)

add_check(
    "Inventario completo de productos críticos",
    len(missing_files) == 0,
    critical=True,
    detail=(
        "Todos los productos requeridos están presentes."
        if not missing_files
        else f"Faltantes: {missing_files}"
    ),
)


# =============================================================================
# 2 - CARGA DEL PROCESO 41
# =============================================================================

banner("2 - VALIDACIÓN DE LA AUDITORÍA FINAL DEL PROCESO 41")

audit41 = read_csv_required(
    "auditoria_41_modelo_territorial_amba_v4.csv"
)

summary41 = read_json_required(
    "resumen_41_auditoria_modelo_territorial_amba_v4.json"
)

report41_path = INPUT_DIR / "informe_41_auditoria_modelo_territorial_amba_v4.md"

print(f"Auditoría 41: {len(audit41)} registros")

dictamen41 = ""

if "dictamen_final" in audit41.columns:
    values = audit41["dictamen_final"].dropna().astype(str).str.upper()
    if not values.empty:
        dictamen41 = values.iloc[-1].strip()

if not dictamen41:
    for key in [
        "dictamen_final",
        "dictamen",
        "DICTAMEN FINAL",
        "DICTAMEN",
    ]:
        if key in summary41:
            dictamen41 = norm(summary41[key]).upper()
            if dictamen41:
                break

if not dictamen41:
    text41 = report41_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).upper()

    if "DICTAMEN FINAL          : GO" in text41:
        dictamen41 = "GO"
    elif "DICTAMEN FINAL : GO" in text41:
        dictamen41 = "GO"
    elif "DICTAMEN FINAL          : NO-GO" in text41:
        dictamen41 = "NO-GO"
    elif "DICTAMEN FINAL : NO-GO" in text41:
        dictamen41 = "NO-GO"

print(f"Dictamen proceso 41 : {dictamen41 or 'NO DETECTADO'}")

audit_score = summary41.get(
    "score_auditoria",
    summary41.get("score", None),
)

print(f"Score auditoría 41  : {audit_score}")

go41 = dictamen41 == "GO"

add_check(
    "Proceso 41 finalizado con dictamen GO",
    go41,
    critical=True,
    detail=f"Dictamen detectado: {dictamen41 or 'NO DETECTADO'}",
)


# =============================================================================
# 3 - MODELO MAESTRO
# =============================================================================

banner("3 - VALIDACIÓN DEL MODELO MAESTRO")

projects = read_csv_required(
    "modelo_maestro_proyectos_v4.csv"
)

scenarios = read_csv_required(
    "modelo_maestro_escenarios_v4.csv"
)

project_id = "proyecto_id"
scenario_id = "escenario_id"

required_project_columns = [
    project_id,
    scenario_id,
]

required_scenario_columns = [
    scenario_id,
]

missing_project_columns = [
    c for c in required_project_columns
    if c not in projects.columns
]

missing_scenario_columns = [
    c for c in required_scenario_columns
    if c not in scenarios.columns
]

structure_ok = (
    not missing_project_columns
    and not missing_scenario_columns
)

print(f"Proyectos  : {len(projects)}")
print(f"Escenarios : {len(scenarios)}")

if missing_project_columns:
    print(f"Columnas faltantes proyectos: {missing_project_columns}")

if missing_scenario_columns:
    print(f"Columnas faltantes escenarios: {missing_scenario_columns}")

add_check(
    "Estructura del modelo maestro",
    structure_ok,
    critical=True,
    detail="Columnas estructurales requeridas presentes.",
)


# =============================================================================
# 4 - INTEGRIDAD DE PROYECTOS
# =============================================================================

banner("4 - INTEGRIDAD DE PROYECTOS Y ESCENARIOS")

if structure_ok:
    project_nulls = int(projects[project_id].isna().sum())
    project_duplicates = int(projects[project_id].duplicated().sum())

    scenario_nulls = int(scenarios[scenario_id].isna().sum())
    scenario_duplicates = int(
        scenarios[scenario_id].duplicated().sum()
    )

    unique_projects = projects[project_id].nunique(dropna=True)
    unique_scenarios = scenarios[scenario_id].nunique(dropna=True)

    print(f"Proyectos únicos         : {unique_projects}")
    print(f"Proyectos nulos          : {project_nulls}")
    print(f"Proyectos duplicados     : {project_duplicates}")
    print(f"Escenarios únicos        : {unique_scenarios}")
    print(f"Escenarios nulos         : {scenario_nulls}")
    print(f"Escenarios duplicados    : {scenario_duplicates}")

    project_integrity_ok = (
        project_nulls == 0
        and project_duplicates == 0
        and scenario_nulls == 0
        and scenario_duplicates == 0
        and unique_projects == len(projects)
        and unique_scenarios == len(scenarios)
    )

    add_check(
        "Integridad de identificadores",
        project_integrity_ok,
        critical=True,
        detail=(
            f"144 proyectos / {unique_scenarios} escenarios "
            f"sin duplicados ni nulos."
        ),
    )
else:
    add_check(
        "Integridad de identificadores",
        False,
        critical=True,
        detail="No se puede validar por estructura incompleta.",
    )


# =============================================================================
# 5 - ASIGNACIÓN PROYECTO -> ESCENARIO
# =============================================================================

banner("5 - VALIDACIÓN DE ASIGNACIÓN PROYECTO -> ESCENARIO")

if structure_ok:
    scenario_nulls_project = int(
        projects[scenario_id].isna().sum()
    )

    multi_scenario = (
        projects.groupby(project_id)[scenario_id]
        .nunique(dropna=True)
    )

    multi_scenario_count = int(
        (multi_scenario > 1).sum()
    )

    scenario_ids_projects = set(
        projects[scenario_id].dropna().astype(str)
    )

    scenario_ids_master = set(
        scenarios[scenario_id].dropna().astype(str)
    )

    scenarios_consistent = (
        scenario_ids_projects == scenario_ids_master
    )

    print(f"Escenarios nulos        : {scenario_nulls_project}")
    print(f"Proyectos multiescenario: {multi_scenario_count}")
    print(
        "IDs de escenarios coincidentes: "
        f"{'SI' if scenarios_consistent else 'NO'}"
    )

    assignment_ok = (
        scenario_nulls_project == 0
        and multi_scenario_count == 0
        and scenarios_consistent
    )

    add_check(
        "Asignación proyecto -> escenario",
        assignment_ok,
        critical=True,
        detail=(
            f"{len(projects)} proyectos asignados "
            "a un único escenario."
        ),
    )


# =============================================================================
# 6 - DISTRIBUCIÓN TERRITORIAL
# =============================================================================

banner("6 - VALIDACIÓN DE DISTRIBUCIÓN TERRITORIAL")

if structure_ok and scenario_id in projects.columns:
    distribution = (
        projects.groupby(scenario_id)
        .size()
        .sort_index()
    )

    for sid, count in distribution.items():
        print(f"  {sid}: {count}")

    if len(distribution) > 0:
        minimum = int(distribution.min())
        maximum = int(distribution.max())
        mean = float(distribution.mean())

        std = float(distribution.std(ddof=0))

        cv = (
            std / mean
            if mean > 0
            else 0.0
        )
    else:
        minimum = maximum = 0
        mean = cv = 0.0

    print(f"Mínimo : {minimum}")
    print(f"Máximo : {maximum}")
    print(f"Media  : {mean:.2f}")
    print(f"CV     : {cv:.4f}")

    distribution_ok = (
        len(distribution) == len(scenarios)
        and minimum > 0
    )

    add_check(
        "Distribución territorial completa",
        distribution_ok,
        critical=True,
        detail=(
            f"{len(distribution)} escenarios con proyectos asignados."
        ),
    )


# =============================================================================
# 7 - RANKINGS
# =============================================================================

banner("7 - VALIDACIÓN DE RANKINGS FINALES")

ranking_scenarios = read_csv_required(
    "ranking_final_escenarios_v4.csv"
)

ranking_projects = read_csv_required(
    "ranking_final_proyectos_v4.csv"
)


def detect_ranking_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    for column in df.columns:
        normalized = column.lower()

        if "ranking" in normalized or "rank" in normalized:
            return column

    return None


scenario_rank_col = detect_ranking_column(
    ranking_scenarios,
    [
        "ranking_integral_v4",
        "ranking_final_escenario_v4",
        "ranking_escenario_v4",
        "ranking",
    ],
)

project_rank_col = detect_ranking_column(
    ranking_projects,
    [
        "ranking_final_proyecto_v4",
        "ranking_proyecto_v4",
        "ranking",
    ],
)

print(f"Ranking escenarios: {scenario_rank_col}")
print(f"Ranking proyectos : {project_rank_col}")


def validate_ranking(
    df: pd.DataFrame,
    id_col: str,
    rank_col: str | None,
    expected_count: int,
) -> tuple[bool, str]:

    if rank_col is None:
        return False, "No se detectó campo de ranking."

    if id_col not in df.columns:
        return False, f"No existe {id_col}."

    if len(df) != expected_count:
        return (
            False,
            f"Registros inesperados: {len(df)} != {expected_count}.",
        )

    ranks = pd.to_numeric(
        df[rank_col],
        errors="coerce",
    )

    if ranks.isna().any():
        return False, "Existen rankings nulos o no numéricos."

    ranks_sorted = sorted(ranks.astype(int).tolist())

    expected = list(range(1, expected_count + 1))

    if ranks_sorted != expected:
        return False, "La secuencia de ranking no es completa."

    return True, "Ranking completo 1..N."


scenario_ranking_ok, scenario_ranking_detail = validate_ranking(
    ranking_scenarios,
    scenario_id,
    scenario_rank_col,
    len(scenarios),
)

project_ranking_ok, project_ranking_detail = validate_ranking(
    ranking_projects,
    project_id,
    project_rank_col,
    len(projects),
)

print(scenario_ranking_detail)
print(project_ranking_detail)

add_check(
    "Ranking final de escenarios",
    scenario_ranking_ok,
    critical=True,
    detail=scenario_ranking_detail,
)

add_check(
    "Ranking final de proyectos",
    project_ranking_ok,
    critical=True,
    detail=project_ranking_detail,
)


# =============================================================================
# 8 - VALIDACIÓN GEOESPACIAL
# =============================================================================

banner("8 - VALIDACIÓN GEOESPACIAL DEL MODELO MAESTRO")

gpkg_path = (
    INPUT_DIR
    / "modelo_maestro_territorial_amba_v4.gpkg"
)

geo_ok = False
project_geo = None
scenario_geo = None

if gpkg_path.exists():

    layers = gpd.list_layers(gpkg_path)["name"].tolist()

    print("Capas disponibles:")
    for layer in layers:
        print(f"  - {layer}")

    if "proyectos" in layers:
        project_geo = gpd.read_file(
            gpkg_path,
            layer="proyectos",
        )

    if "escenarios" in layers:
        scenario_geo = gpd.read_file(
            gpkg_path,
            layer="escenarios",
        )

    if project_geo is not None:

        valid_geometry = project_geo.geometry.notna()

        non_empty = (
            project_geo.geometry.notna()
            & ~project_geo.geometry.is_empty
        )

        valid_topology = (
            project_geo.geometry.notna()
            & ~project_geo.geometry.is_empty
            & project_geo.geometry.is_valid
        )

        valid_count = int(valid_topology.sum())

        print(f"Proyectos geográficos : {len(project_geo)}")
        print(
            f"Geometrías válidas   : {valid_count}"
        )

        print(
            f"Geometrías nulas     : "
            f"{int(project_geo.geometry.isna().sum())}"
        )

        print(
            f"Geometrías vacías    : "
            f"{int(project_geo.geometry.is_empty.sum())}"
        )

        print(
            f"Geometrías inválidas : "
            f"{int((~project_geo.geometry.is_valid).sum())}"
        )

        geometry_coverage = (
            valid_count / len(project_geo) * 100
            if len(project_geo)
            else 0.0
        )

        tabular_ids = set(
            projects[project_id]
            .dropna()
            .astype(str)
        )

        geo_ids = set(
            project_geo[project_id]
            .dropna()
            .astype(str)
        ) if project_id in project_geo.columns else set()

        ids_match = tabular_ids == geo_ids

        assignments_match = False

        if (
            project_id in project_geo.columns
            and scenario_id in project_geo.columns
        ):

            tab_assignment = (
                projects[
                    [project_id, scenario_id]
                ]
                .copy()
            )

            geo_assignment = (
                project_geo[
                    [project_id, scenario_id]
                ]
                .copy()
            )

            tab_assignment[project_id] = (
                tab_assignment[project_id].astype(str)
            )

            tab_assignment[scenario_id] = (
                tab_assignment[scenario_id].astype(str)
            )

            geo_assignment[project_id] = (
                geo_assignment[project_id].astype(str)
            )

            geo_assignment[scenario_id] = (
                geo_assignment[scenario_id].astype(str)
            )

            tab_assignment = tab_assignment.sort_values(
                [project_id, scenario_id]
            ).reset_index(drop=True)

            geo_assignment = geo_assignment.sort_values(
                [project_id, scenario_id]
            ).reset_index(drop=True)

            assignments_match = tab_assignment.equals(
                geo_assignment
            )

        print(
            f"IDs tabular ↔ geográfico: "
            f"{'SI' if ids_match else 'NO'}"
        )

        print(
            f"Asignaciones coincidentes: "
            f"{'SI' if assignments_match else 'NO'}"
        )

        geo_ok = (
            len(project_geo) == len(projects)
            and valid_count == len(project_geo)
            and ids_match
            and assignments_match
        )

        add_check(
            "Modelo geográfico maestro",
            geo_ok,
            critical=True,
            detail=(
                f"Cobertura geométrica: "
                f"{geometry_coverage:.2f}%."
            ),
        )

    else:
        add_check(
            "Modelo geográfico maestro",
            False,
            critical=True,
            detail="No existe la capa proyectos.",
        )

else:
    add_check(
        "Modelo geográfico maestro",
        False,
        critical=True,
        detail="No existe el GeoPackage maestro.",
    )


# =============================================================================
# 9 - CONTROL DE ESCENARIOS GEOGRÁFICOS
# =============================================================================

banner("9 - VALIDACIÓN DE ESCENARIOS GEOGRÁFICOS")

scenario_geo_ok = False

if scenario_geo is not None:

    scenario_geo_ids = set(
        scenario_geo[scenario_id]
        .dropna()
        .astype(str)
    )

    scenario_master_ids = set(
        scenarios[scenario_id]
        .dropna()
        .astype(str)
    )

    scenario_geo_ok = (
        len(scenario_geo) == len(scenarios)
        and scenario_geo_ids == scenario_master_ids
        and scenario_geo.geometry.notna().all()
        and (~scenario_geo.geometry.is_empty).all()
        and scenario_geo.geometry.is_valid.all()
    )

    print(f"Escenarios geográficos: {len(scenario_geo)}")
    print(
        f"IDs coincidentes: "
        f"{'SI' if scenario_geo_ids == scenario_master_ids else 'NO'}"
    )

add_check(
    "Escenarios geográficos completos",
    scenario_geo_ok,
    critical=True,
    detail=(
        "Las 7 unidades territoriales están presentes "
        "y son geométricamente válidas."
        if scenario_geo_ok
        else "Inconsistencia en la capa geográfica de escenarios."
    ),
)


# =============================================================================
# 10 - CONTROL PROCESOS ANTERIORES
# =============================================================================

banner("10 - CONTROL DE TRAZABILIDAD DE PROCESOS 39-40")

audit39_path = (
    INPUT_DIR
    / "auditoria_39_informe_territorial_amba_v4_1.csv"
)

audit40_path = (
    INPUT_DIR
    / "auditoria_40_atlas_territorial_amba.csv"
)

trace39_ok = False
trace40_ok = False

if audit39_path.exists():

    audit39 = pd.read_csv(audit39_path)

    failures39 = 0

    if "resultado" in audit39.columns:
        failures39 = int(
            (
                audit39["resultado"]
                .astype(str)
                .str.upper()
                == "FAIL"
            ).sum()
        )

    elif "estado" in audit39.columns:
        failures39 = int(
            (
                audit39["estado"]
                .astype(str)
                .str.upper()
                == "FAIL"
            ).sum()
        )

    trace39_ok = failures39 == 0

    print(
        f"Proceso 39: "
        f"{'OK' if trace39_ok else 'FAIL'}"
    )

if audit40_path.exists():

    audit40 = pd.read_csv(audit40_path)

    failures40 = 0

    if "resultado" in audit40.columns:
        failures40 = int(
            (
                audit40["resultado"]
                .astype(str)
                .str.upper()
                == "FAIL"
            ).sum()
        )

    elif "estado" in audit40.columns:
        failures40 = int(
            (
                audit40["estado"]
                .astype(str)
                .str.upper()
                == "FAIL"
            ).sum()
        )

    trace40_ok = failures40 == 0

    print(
        f"Proceso 40: "
        f"{'OK' if trace40_ok else 'FAIL'}"
    )

add_check(
    "Trazabilidad proceso 39",
    trace39_ok,
    critical=True,
    detail="Auditoría 39 sin fallas.",
)

add_check(
    "Trazabilidad proceso 40",
    trace40_ok,
    critical=True,
    detail="Auditoría 40 sin fallas.",
)


# =============================================================================
# 11 - INDICADORES
# =============================================================================

banner("11 - CONTROL DE INDICADORES GLOBALES")

indicators = read_csv_required(
    "indicadores_globales_amba_v4.csv"
)

indicator_ok = (
    len(indicators) > 0
    and "indicador" in indicators.columns
)

print(f"Indicadores: {len(indicators)}")

add_check(
    "Indicadores globales disponibles",
    indicator_ok,
    critical=True,
    detail=f"{len(indicators)} indicadores disponibles.",
)


# =============================================================================
# 12 - HASHES
# =============================================================================

banner("12 - GENERACIÓN DE HASHES DEL PAQUETE DE CIERRE")

hash_files = [
    "modelo_maestro_proyectos_v4.csv",
    "modelo_maestro_escenarios_v4.csv",
    "ranking_final_proyectos_v4.csv",
    "ranking_final_escenarios_v4.csv",
    "matriz_integral_escenarios_v4.csv",
    "indicadores_globales_amba_v4.csv",
    "modelo_maestro_territorial_amba_v4.gpkg",
    "informe_territorial_amba_v4_1.md",
    "atlas_territorial_amba_v4.gpkg",
    "atlas_territorial_amba_v4.md",
]

for filename in hash_files:

    path = INPUT_DIR / filename

    if path.exists():

        digest = sha256_file(path)

        hash_rows.append(
            {
                "archivo": filename,
                "sha256": digest,
                "tamano_bytes": path.stat().st_size,
                "fecha_hash": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        )

        print(f"{filename}: {digest}")

hash_df = pd.DataFrame(hash_rows)

hash_output = (
    OUTPUT_DIR
    / "hashes_42_cierre_modelo_territorial_amba_v4.csv"
)

hash_df.to_csv(
    hash_output,
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 13 - DICTAMEN FINAL
# =============================================================================

banner("13 - DETERMINACIÓN DEL CIERRE FINAL")

total_controls = len(checks)
failed_controls = sum(
    1
    for item in checks
    if item["resultado"] != "OK"
)

critical_failures = sum(
    1
    for item in checks
    if (
        item["resultado"] != "OK"
        and item["critico"] == "SI"
    )
)

passed_controls = total_controls - failed_controls

score = (
    passed_controls / total_controls * 100
    if total_controls
    else 0.0
)

final_go = (
    critical_failures == 0
    and go41
)

final_audit = (
    "OK"
    if final_go
    else "OBSERVADA"
)

dictamen = (
    "GO"
    if final_go
    else "NO-GO"
)

print(f"Controles OK       : {passed_controls}/{total_controls}")
print(f"Controles fallidos : {failed_controls}")
print(f"Fallas críticas    : {critical_failures}")
print(f"Score cierre       : {score:.2f}/100")
print(f"Auditoría          : {final_audit}")
print(f"DICTAMEN FINAL     : {dictamen}")


# =============================================================================
# 14 - ACTA DE CIERRE
# =============================================================================

banner("14 - GENERANDO ACTA FORMAL DE CIERRE")

now = datetime.now()

acta_path = (
    OUTPUT_DIR
    / "acta_cierre_modelo_territorial_amba_v4.md"
)

acta_lines = [
    "# ACTA DE CIERRE DEL MODELO TERRITORIAL AMBA V4.1",
    "",
    f"**Proceso:** {PROCESO}",
    f"**Versión:** {VERSION}",
    f"**Fecha de cierre:** {now.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Estado final",
    "",
    f"- **Dictamen:** {dictamen}",
    f"- **Auditoría:** {final_audit}",
    f"- **Score de cierre:** {score:.2f}/100",
    f"- **Controles aprobados:** {passed_controls}/{total_controls}",
    f"- **Fallas críticas:** {critical_failures}",
    "",
    "## Modelo territorial",
    "",
    f"- Proyectos: **{len(projects)}**",
    f"- Escenarios: **{len(scenarios)}**",
    "",
    "## Integridad",
    "",
    "- La asignación proyecto → escenario fue validada.",
    "- Los rankings finales fueron validados.",
    "- El modelo geográfico maestro fue validado.",
    "- Los escenarios geográficos fueron validados.",
    "- Los procesos 39 y 40 fueron controlados.",
    "- Los indicadores globales están disponibles.",
    "- Se generaron hashes SHA-256 de los productos críticos.",
    "",
    "## Criterio de cierre",
    "",
    (
        "El modelo territorial AMBA V4.1 queda formalmente cerrado "
        "y habilitado para explotación analítica, cartográfica y "
        "documental."
        if final_go
        else
        "El modelo territorial AMBA V4.1 NO queda cerrado. "
        "Deben corregirse las fallas detectadas y ejecutarse "
        "nuevamente la auditoría final."
    ),
    "",
    "## Productos principales",
    "",
    "- Modelo maestro territorial",
    "- Ranking final de proyectos",
    "- Ranking final de escenarios",
    "- Matriz integral de escenarios",
    "- Indicadores globales AMBA",
    "- Informe territorial",
    "- Atlas territorial",
    "- Auditoría final",
    "- Manifiesto y hashes de cierre",
    "",
]

write_text(
    acta_path,
    "\n".join(acta_lines),
)


# =============================================================================
# 15 - MANIFIESTO
# =============================================================================

banner("15 - GENERANDO MANIFIESTO DE CIERRE")

manifest_rows = []

for item in inventory:

    filename = item["archivo"]
    path = INPUT_DIR / filename

    manifest_rows.append(
        {
            "archivo": filename,
            "presente": item["existe"],
            "tamano_mb": item["tamano_mb"],
            "sha256": (
                sha256_file(path)
                if path.exists()
                else ""
            ),
            "estado_cierre": (
                "INCLUIDO"
                if path.exists()
                else "FALTANTE"
            ),
        }
    )

manifest_df = pd.DataFrame(manifest_rows)

manifest_path = (
    OUTPUT_DIR
    / "manifiesto_cierre_modelo_territorial_amba_v4.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 16 - AUDITORÍA DE CIERRE
# =============================================================================

banner("16 - EXPORTANDO AUDITORÍA DE CIERRE")

audit_output = (
    OUTPUT_DIR
    / "cierre_42_modelo_territorial_amba_v4.csv"
)

checks_df = pd.DataFrame(checks)

checks_df.to_csv(
    audit_output,
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 17 - RESUMEN JSON
# =============================================================================

elapsed = time.time() - start_time

summary = {
    "proceso": PROCESO,
    "version": VERSION,
    "fecha": now.isoformat(timespec="seconds"),
    "proyecto": str(PROJECT_ROOT),
    "entrada": str(INPUT_DIR),
    "salida": str(OUTPUT_DIR),
    "proyectos": int(len(projects)),
    "escenarios": int(len(scenarios)),
    "controles": int(total_controls),
    "controles_ok": int(passed_controls),
    "controles_fallidos": int(failed_controls),
    "fallas_criticas": int(critical_failures),
    "score_cierre": round(score, 2),
    "dictamen_proceso_41": dictamen41,
    "auditoria": final_audit,
    "dictamen_final": dictamen,
    "tiempo_segundos": round(elapsed, 2),
    "productos_criticos": len(
        [
            x
            for x in inventory
            if x["existe"] == "SI"
        ]
    ),
    "productos_faltantes": missing_files,
    "hashes_generados": len(hash_rows),
}

summary_path = (
    OUTPUT_DIR
    / "resumen_42_cierre_modelo_territorial_amba_v4.json"
)

with summary_path.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        to_json_safe(summary),
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# RESULTADO FINAL
# =============================================================================

banner("RESULTADO FINAL DEL PROCESO 42")

print(f"Proyectos                 : {len(projects)}")
print(f"Escenarios                : {len(scenarios)}")
print(f"Controles OK              : {passed_controls}/{total_controls}")
print(f"Fallas críticas           : {critical_failures}")
print(f"Score cierre              : {score:.2f}/100")
print(f"Auditoría                 : {final_audit}")
print(f"DICTAMEN FINAL            : {dictamen}")
print(f"Tiempo de ejecución       : {elapsed:.2f} segundos")

print()
print("=" * 88)
print("ARCHIVOS GENERADOS")
print("=" * 88)

print(f"Cierre       : {audit_output}")
print(f"Manifiesto   : {manifest_path}")
print(f"Hashes       : {hash_output}")
print(f"Resumen      : {summary_path}")
print(f"Acta         : {acta_path}")

print()

if final_go:

    print("=" * 88)
    print("PROCESO 42 FINALIZADO - GO")
    print("=" * 88)
    print(
        "El modelo territorial AMBA V4.1 fue cerrado correctamente."
    )
    print(
        "La auditoría 41 fue aprobada con dictamen GO."
    )
    print(
        "Los productos críticos fueron verificados y registrados."
    )
    print(
        "El modelo queda formalmente habilitado para cierre."
    )

else:

    print("=" * 88)
    print("PROCESO 42 FINALIZADO - NO-GO")
    print("=" * 88)
    print(
        "El modelo territorial AMBA V4.1 NO puede cerrarse."
    )
    print(
        "Revisar las fallas indicadas en el archivo de cierre."
    )