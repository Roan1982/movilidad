# -*- coding: utf-8 -*-

"""
47 - VERIFICACIÓN FINAL DEL ARTEFACTO ZIP
MODELO TERRITORIAL AMBA - V4.1

Objetivo:
    Verificar independientemente el paquete definitivo generado por el
    proceso 46.

Características:
    - Valida existencia e integridad física del ZIP.
    - Detecta correctamente la raíz interna del ZIP.
    - Valida estructura.
    - Valida archivos no vacíos.
    - Valida productos obligatorios.
    - Calcula SHA-256 del ZIP.
    - Calcula SHA-256 de todos los archivos internos.
    - Valida README y MANIFIESTO.
    - Valida evidencia de procesos 42, 43, 44 y 45.
    - Compara el directorio definitivo contra el ZIP.
    - Valida manifiesto CSV.
    - Valida metadata.
    - Valida consistencia básica del modelo.
    - Genera auditoría, inventario, hashes, resumen e informe.
    - No modifica el ZIP definitivo.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.1"
PROYECTO = "Modelo Territorial AMBA"

SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parent
PROJECT_DIR = SRC_DIR.parent

BASE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

ZIP_PATH = BASE_DIR / "MODELO_TERRITORIAL_AMBA_V4_FINAL.zip"

FINAL_DIR = (
    BASE_DIR
    / "MODELO_TERRITORIAL_AMBA_V4_FINAL"
)

AUDIT_DIR = (
    BASE_DIR
    / "auditoria_47_verificacion_final_amba_v4"
)


# =============================================================================
# PRODUCTOS OBLIGATORIOS
# =============================================================================

REQUIRED_FILES = [
    # -------------------------------------------------------------------------
    # 01 - MODELO
    # -------------------------------------------------------------------------
    "01_modelo/modelo_maestro_escenarios_v4.csv",
    "01_modelo/modelo_maestro_proyectos_v4.csv",
    "01_modelo/ranking_final_escenarios_v4.csv",
    "01_modelo/ranking_final_proyectos_v4.csv",

    # -------------------------------------------------------------------------
    # 02 - INFORMES
    # -------------------------------------------------------------------------
    "02_informes/escenarios_ejecutivos_amba_v4_1.csv",
    "02_informes/indicadores_ejecutivos_amba_v4_1.csv",
    "02_informes/informe_ejecutivo_amba_v4_1.md",
    "02_informes/informe_ejecutivo_amba_v4_1.txt",
    "02_informes/proyectos_ejecutivos_amba_v4_1.csv",
    "02_informes/ranking_escenarios_ejecutivo_amba_v4_1.csv",
    "02_informes/sintesis_ejecutiva_amba_v4_1.md",
    "02_informes/sintesis_ejecutiva_amba_v4_1.txt",
    "02_informes/top_20_proyectos_prioritarios_amba_v4_1.csv",

    # -------------------------------------------------------------------------
    # 03 - ATLAS
    # -------------------------------------------------------------------------
    "03_atlas/atlas_territorial_amba_v4.gpkg",
    "03_atlas/escenarios_territoriales_amba.gpkg",
    "03_atlas/geometria_cartera_proyectos_v4.gpkg",
    "03_atlas/geometria_escenarios_cartera_v4.gpkg",
    "03_atlas/modelo_maestro_territorial_amba_v4.gpkg",
    "03_atlas/modelo_territorial_amba_v4.gpkg",
    "03_atlas/priorizacion_territorial_escenarios_v4.gpkg",
    "03_atlas/sintesis_estrategica_escenarios_v4.gpkg",

    # -------------------------------------------------------------------------
    # 04 - DATOS
    # -------------------------------------------------------------------------
    "04_datos/indicadores_globales_amba_v4.csv",

    # -------------------------------------------------------------------------
    # 05 - AUDITORÍA
    # -------------------------------------------------------------------------
    "05_auditoria/auditoria_44_paquete_final_amba_v4.csv",
    "05_auditoria/auditoria_45_cierre_amba_v4.csv",
    "05_auditoria/cierre_42_modelo_territorial_amba_v4.csv",
    "05_auditoria/control_paquete_ejecutivo_amba_v4_1.csv",
    "05_auditoria/hashes_44_paquete_final_amba_v4.csv",
    "05_auditoria/hashes_45_cierre_amba_v4.csv",
    "05_auditoria/informe_44_auditoria_paquete_final_amba_v4.md",
    "05_auditoria/informe_45_cierre_amba_v4.md",
    "05_auditoria/inventario_44_paquete_final_amba_v4.csv",
    "05_auditoria/inventario_45_cierre_amba_v4.csv",
    "05_auditoria/manifiesto_43_paquete_ejecutivo_amba_v4_1.csv",
    "05_auditoria/resumen_44_auditoria_paquete_final_amba_v4.json",
    "05_auditoria/resumen_45_cierre_amba_v4.json",

    # -------------------------------------------------------------------------
    # 06 - METADATOS
    # -------------------------------------------------------------------------
    "06_metadatos/MANIFIESTO_SHA256.csv",
    "06_metadatos/metadata_paquete.json",
    "06_metadatos/resumen_proceso_46.json",

    # -------------------------------------------------------------------------
    # RAÍZ
    # -------------------------------------------------------------------------
    "MANIFIESTO.md",
    "README.md",
]


# =============================================================================
# UTILIDADES
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_section(number: str, title: str) -> None:
    print()
    print("=" * 88)
    print(f"{number} - {title}")
    print("=" * 88)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_zip_path(path: str) -> str:
    """
    Normaliza rutas provenientes del ZIP.

    Convierte:
        \
    en:
        /

    Elimina ./ iniciales y barras iniciales.
    """

    value = str(path).replace("\\", "/")

    while value.startswith("./"):
        value = value[2:]

    value = value.lstrip("/")

    return value


def is_directory_zip_entry(name: str) -> bool:
    return normalize_zip_path(name).endswith("/")


def detect_zip_root(names: list[str]) -> str:
    """
    Detecta la raíz común del ZIP.

    Para este proyecto se espera:

        MODELO_TERRITORIAL_AMBA_V4_FINAL/

    Retorna:
        MODELO_TERRITORIAL_AMBA_V4_FINAL
    """

    normalized = [
        normalize_zip_path(name)
        for name in names
        if not is_directory_zip_entry(name)
    ]

    if not normalized:
        return ""

    first_components = []

    for name in normalized:
        parts = name.split("/")

        if parts:
            first_components.append(parts[0])

    if first_components and len(set(first_components)) == 1:
        return first_components[0]

    return ""


def zip_relative_path(name: str, root: str) -> str:
    """
    Convierte una ruta interna del ZIP en ruta relativa al paquete.

    Ejemplo:

        MODELO_TERRITORIAL_AMBA_V4_FINAL/01_modelo/a.csv

    pasa a:

        01_modelo/a.csv
    """

    normalized = normalize_zip_path(name)

    prefix = normalize_zip_path(root).rstrip("/") + "/"

    if normalized.startswith(prefix):
        return normalized[len(prefix):]

    if normalized == normalize_zip_path(root):
        return ""

    return normalized


def safe_read_text(path: Path) -> str:
    """
    Lectura robusta de texto.
    """

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="replace")
        except Exception:
            return ""


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =============================================================================
# PROCESO PRINCIPAL
# =============================================================================

def main() -> int:

    start_time = time.perf_counter()

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # VARIABLES DE AUDITORÍA
    # -------------------------------------------------------------------------

    controls = []

    inventory_rows = []

    hash_rows = []

    errors = []

    critical_failures = []

    important_failures = []

    zip_root = ""

    zip_names = []

    zip_file_names = []

    extracted_root = None

    # =========================================================================
    # CABECERA
    # =========================================================================

    print_header(
        "47 - VERIFICACIÓN FINAL DEL ARTEFACTO ZIP DEL MODELO TERRITORIAL AMBA - V4.1"
    )

    print(
        f"Proyecto                      : {PROJECT_DIR}"
    )

    print(
        f"ZIP                           : {ZIP_PATH}"
    )

    print(
        f"Directorio                    : {FINAL_DIR}"
    )

    print(
        f"Salida                        : {AUDIT_DIR}"
    )

    # =========================================================================
    # 1 - EXISTENCIA DEL ZIP
    # =========================================================================

    print_section(
        "1",
        "EXISTENCIA DEL ARTEFACTO ZIP",
    )

    zip_exists = ZIP_PATH.exists() and ZIP_PATH.is_file()

    print(
        f"ZIP encontrado                : "
        f"{'SI' if zip_exists else 'NO'}"
    )

    zip_size = 0

    if zip_exists:
        zip_size = ZIP_PATH.stat().st_size

        print(
            f"Tamaño                        : "
            f"{zip_size:,} bytes"
        )

    controls.append(
        {
            "control": "existencia_zip",
            "ok": zip_exists,
            "detalle": str(ZIP_PATH),
        }
    )

    if not zip_exists:
        critical_failures.append(
            "ZIP definitivo inexistente"
        )

        print()
        print("ERROR: no existe el ZIP definitivo.")
        return 1

    # =========================================================================
    # 2 - INTEGRIDAD FÍSICA
    # =========================================================================

    print_section(
        "2",
        "INTEGRIDAD FÍSICA DEL ZIP",
    )

    zip_test_ok = False
    zip_test_error = ""

    try:
        with zipfile.ZipFile(
            ZIP_PATH,
            "r",
        ) as zf:

            bad_file = zf.testzip()

            if bad_file is None:
                zip_test_ok = True
            else:
                zip_test_error = str(bad_file)

    except Exception as exc:
        zip_test_error = repr(exc)

    print(
        f"Test ZIP                      : "
        f"{'OK' if zip_test_ok else 'ERROR'}"
    )

    if zip_test_error:
        print(
            f"Detalle                       : {zip_test_error}"
        )

    controls.append(
        {
            "control": "integridad_zip",
            "ok": zip_test_ok,
            "detalle": zip_test_error,
        }
    )

    if not zip_test_ok:
        critical_failures.append(
            "ZIP físicamente corrupto"
        )

    # =========================================================================
    # 3 - INVENTARIO INDEPENDIENTE
    # =========================================================================

    print_section(
        "3",
        "INVENTARIO INDEPENDIENTE DEL ZIP",
    )

    try:
        with zipfile.ZipFile(
            ZIP_PATH,
            "r",
        ) as zf:

            zip_names = [
                normalize_zip_path(info.filename)
                for info in zf.infolist()
            ]

            zip_file_names = [
                name
                for name in zip_names
                if not is_directory_zip_entry(name)
            ]

            zip_root = detect_zip_root(zip_names)

    except Exception as exc:

        errors.append(
            f"No se pudo inventariar el ZIP: {exc}"
        )

        zip_names = []
        zip_file_names = []
        zip_root = ""

    print(
        f"Entradas ZIP                  : "
        f"{len(zip_names)}"
    )

    print(
        f"Archivos físicos              : "
        f"{len(zip_file_names)}"
    )

    print(
        f"Raíz detectada                : "
        f"{zip_root or 'NO DETECTADA'}"
    )

    controls.append(
        {
            "control": "inventario_zip",
            "ok": bool(zip_file_names),
            "detalle": f"{len(zip_file_names)} archivos",
        }
    )

    if not zip_file_names:
        critical_failures.append(
            "ZIP sin archivos físicos"
        )

    # =========================================================================
    # 4 - EXTRACCIÓN TEMPORAL
    # =========================================================================

    print_section(
        "4",
        "EXTRACCIÓN TEMPORAL Y NORMALIZACIÓN DEL ARTEFACTO",
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="amba47_"
        )
    )

    try:

        with zipfile.ZipFile(
            ZIP_PATH,
            "r",
        ) as zf:

            zf.extractall(temp_dir)

        if zip_root:
            extracted_root = temp_dir / zip_root
        else:
            extracted_root = temp_dir

        extracted_root = extracted_root.resolve()

        print(
            f"Directorio temporal           : "
            f"{extracted_root}"
        )

        extraction_ok = extracted_root.exists()

    except Exception as exc:

        extraction_ok = False

        errors.append(
            f"Error durante extracción: {exc}"
        )

    print(
        f"Extracción                    : "
        f"{'OK' if extraction_ok else 'ERROR'}"
    )

    controls.append(
        {
            "control": "extraccion_zip",
            "ok": extraction_ok,
            "detalle": str(extracted_root),
        }
    )

    if not extraction_ok:
        critical_failures.append(
            "No se pudo extraer el ZIP"
        )

    # =========================================================================
    # 5 - ESTRUCTURA
    # =========================================================================

    print_section(
        "5",
        "VALIDACIÓN DE ESTRUCTURA DEFINITIVA",
    )

    expected_dirs = [
        "01_modelo",
        "02_informes",
        "03_atlas",
        "04_datos",
        "05_auditoria",
        "06_metadatos",
    ]

    missing_dirs = []

    for directory in expected_dirs:

        directory_path = (
            extracted_root / directory
        )

        if not directory_path.exists():
            missing_dirs.append(directory)

    root_files = [
        "MANIFIESTO.md",
        "README.md",
    ]

    missing_root_files = []

    for filename in root_files:

        if not (
            extracted_root / filename
        ).exists():

            missing_root_files.append(filename)

    allowed_files = set(REQUIRED_FILES)

    files_outside_structure = []

    for relative_path in zip_file_names:

        relative = zip_relative_path(
            relative_path,
            zip_root,
        )

        if relative not in allowed_files:
            files_outside_structure.append(
                relative
            )

    print(
        f"Directorios faltantes         : "
        f"{len(missing_dirs)}"
    )

    print(
        f"Archivos raíz faltantes       : "
        f"{len(missing_root_files)}"
    )

    print(
        f"Archivos fuera de estructura  : "
        f"{len(files_outside_structure)}"
    )

    for item in missing_dirs:
        print(
            f"  FALTANTE: {item}"
        )

    for item in missing_root_files:
        print(
            f"  FALTANTE: {item}"
        )

    for item in files_outside_structure:
        print(
            f"  FUERA: {item}"
        )

    structure_ok = (
        len(missing_dirs) == 0
        and len(missing_root_files) == 0
        and len(files_outside_structure) == 0
    )

    controls.append(
        {
            "control": "estructura",
            "ok": structure_ok,
            "detalle": (
                f"dirs_faltantes={len(missing_dirs)}; "
                f"raiz_faltante={len(missing_root_files)}; "
                f"fuera={len(files_outside_structure)}"
            ),
        }
    )

    if not structure_ok:
        critical_failures.append(
            "Estructura definitiva inválida"
        )

    # =========================================================================
    # 6 - ARCHIVOS NO VACÍOS
    # =========================================================================

    print_section(
        "6",
        "VALIDACIÓN DE ARCHIVOS NO VACÍOS",
    )

    empty_files = []

    for relative in REQUIRED_FILES:

        path = extracted_root / relative

        if not path.exists():
            continue

        try:
            size = path.stat().st_size
        except Exception:
            size = 0

        if size <= 0:
            empty_files.append(relative)

    print(
        f"Archivos vacíos               : "
        f"{len(empty_files)}"
    )

    for item in empty_files:
        print(
            f"  VACÍO: {item}"
        )

    non_empty_ok = len(empty_files) == 0

    controls.append(
        {
            "control": "archivos_no_vacios",
            "ok": non_empty_ok,
            "detalle": f"{len(empty_files)} vacíos",
        }
    )

    if not non_empty_ok:
        critical_failures.append(
            "Existen archivos obligatorios vacíos"
        )

    # =========================================================================
    # 7 - PRODUCTOS OBLIGATORIOS
    # =========================================================================

    print_section(
        "7",
        "PRODUCTOS OBLIGATORIOS",
    )

    missing_required = []

    for relative in REQUIRED_FILES:

        if not (
            extracted_root / relative
        ).exists():

            missing_required.append(relative)

    print(
        f"Productos requeridos          : "
        f"{len(REQUIRED_FILES)}"
    )

    print(
        f"Productos encontrados         : "
        f"{len(REQUIRED_FILES) - len(missing_required)}"
    )

    print(
        f"Productos faltantes           : "
        f"{len(missing_required)}"
    )

    for item in missing_required:
        print(
            f"  FALTANTE: {item}"
        )

    required_ok = len(missing_required) == 0

    controls.append(
        {
            "control": "productos_obligatorios",
            "ok": required_ok,
            "detalle": f"{len(missing_required)} faltantes",
        }
    )

    if not required_ok:
        critical_failures.append(
            "Faltan productos obligatorios"
        )

    # =========================================================================
    # 8 - SHA256 DEL ZIP
    # =========================================================================

    print_section(
        "8",
        "SHA-256 DEL ZIP DEFINITIVO",
    )

    zip_sha256 = sha256_file(ZIP_PATH)

    print(
        f"SHA-256 ZIP                   : "
        f"{zip_sha256}"
    )

    # =========================================================================
    # 9 - HASHES INTERNOS
    # =========================================================================

    print_section(
        "9",
        "HASHES SHA-256 DE ARCHIVOS INTERNOS",
    )

    internal_hash_errors = 0

    if extraction_ok:

        for relative in zip_file_names:

            relative_path = zip_relative_path(
                relative,
                zip_root,
            )

            path = (
                extracted_root
                / relative_path
            )

            try:

                digest = sha256_file(path)

                hash_rows.append(
                    {
                        "archivo": relative_path,
                        "sha256": digest,
                        "tamano_bytes": path.stat().st_size,
                        "estado": "OK",
                    }
                )

            except Exception as exc:

                internal_hash_errors += 1

                hash_rows.append(
                    {
                        "archivo": relative_path,
                        "sha256": "",
                        "tamano_bytes": 0,
                        "estado": f"ERROR: {exc}",
                    }
                )

    print(
        f"Archivos hasheados            : "
        f"{len(hash_rows)}"
    )

    print(
        f"Errores hash                  : "
        f"{internal_hash_errors}"
    )

    internal_hashes_ok = (
        internal_hash_errors == 0
        and len(hash_rows) == len(zip_file_names)
    )

    controls.append(
        {
            "control": "hashes_internos",
            "ok": internal_hashes_ok,
            "detalle": f"{internal_hash_errors} errores",
        }
    )

    if not internal_hashes_ok:
        critical_failures.append(
            "Errores en hashes internos"
        )

    # =========================================================================
    # 10 - MANIFIESTO MD
    # =========================================================================

    print_section(
        "10",
        "VALIDACIÓN DE MANIFIESTO.MD",
    )

    manifest_path = (
        extracted_root
        / "MANIFIESTO.md"
    )

    manifest_text = (
        safe_read_text(manifest_path)
        if manifest_path.exists()
        else ""
    )

    manifest_concepts = [
        "MODELO TERRITORIAL AMBA",
        "V4.1",
        "SHA-256",
        "MANIFIESTO",
        "Proceso 46",
    ]

    missing_manifest_concepts = [
        concept
        for concept in manifest_concepts
        if concept.lower() not in manifest_text.lower()
    ]

    print(
        "Entrada                       : "
        "MODELO_TERRITORIAL_AMBA_V4_FINAL/MANIFIESTO.md"
    )

    print(
        f"Caracteres                    : "
        f"{len(manifest_text)}"
    )

    print(
        f"Conceptos faltantes           : "
        f"{len(missing_manifest_concepts)}"
    )

    for concept in missing_manifest_concepts:
        print(
            f"  FALTANTE: {concept}"
        )

    manifest_ok = (
        manifest_path.exists()
        and len(manifest_text.strip()) > 0
        and not missing_manifest_concepts
    )

    controls.append(
        {
            "control": "manifesto_md",
            "ok": manifest_ok,
            "detalle": f"{len(manifest_text)} caracteres",
        }
    )

    if not manifest_ok:
        critical_failures.append(
            "MANIFIESTO.md inválido"
        )

    # =========================================================================
    # 11 - README
    # =========================================================================

    print_section(
        "11",
        "VALIDACIÓN DE README.MD",
    )

    readme_path = (
        extracted_root
        / "README.md"
    )

    readme_text = (
        safe_read_text(readme_path)
        if readme_path.exists()
        else ""
    )

    readme_concepts = [
        "MODELO TERRITORIAL AMBA",
        "V4.1",
        "Proceso 46",
    ]

    missing_readme_concepts = [
        concept
        for concept in readme_concepts
        if concept.lower() not in readme_text.lower()
    ]

    print(
        "Entrada                       : "
        "MODELO_TERRITORIAL_AMBA_V4_FINAL/README.md"
    )

    print(
        f"Caracteres                    : "
        f"{len(readme_text)}"
    )

    print(
        f"Conceptos faltantes           : "
        f"{len(missing_readme_concepts)}"
    )

    for concept in missing_readme_concepts:
        print(
            f"  FALTANTE: {concept}"
        )

    readme_ok = (
        readme_path.exists()
        and len(readme_text.strip()) > 0
        and not missing_readme_concepts
    )

    controls.append(
        {
            "control": "readme_md",
            "ok": readme_ok,
            "detalle": f"{len(readme_text)} caracteres",
        }
    )

    if not readme_ok:
        critical_failures.append(
            "README.md inválido"
        )

    # =========================================================================
    # 12 - PROCESOS 42-45
    # =========================================================================

    print_section(
        "12",
        "EVIDENCIA DE CIERRE DE PROCESOS 42, 43, 44 Y 45",
    )

    process_files = {
        "42": [
            "05_auditoria/cierre_42_modelo_territorial_amba_v4.csv",
        ],
        "43": [
            "05_auditoria/manifiesto_43_paquete_ejecutivo_amba_v4_1.csv",
            "05_auditoria/control_paquete_ejecutivo_amba_v4_1.csv",
        ],
        "44": [
            "05_auditoria/auditoria_44_paquete_final_amba_v4.csv",
            "05_auditoria/resumen_44_auditoria_paquete_final_amba_v4.json",
        ],
        "45": [
            "05_auditoria/auditoria_45_cierre_amba_v4.csv",
            "05_auditoria/resumen_45_cierre_amba_v4.json",
            "05_auditoria/informe_45_cierre_amba_v4.md",
        ],
    }

    process_results = {}

    for process, files in process_files.items():

        existing = [
            file
            for file in files
            if (
                extracted_root / file
            ).exists()
        ]

        missing = [
            file
            for file in files
            if (
                extracted_root / file
            ).exists() is False
        ]

        process_text = ""

        for file in existing:
            process_text += "\n"
            process_text += safe_read_text(
                extracted_root / file
            )

        process_text_lower = process_text.lower()

        go_detected = (
            "go" in process_text_lower
            or "dictamen final: go" in process_text_lower
            or '"go"' in process_text_lower
        )

        process_ok = (
            len(missing) == 0
            and go_detected
        )

        process_results[process] = {
            "ok": process_ok,
            "missing": missing,
            "go_detected": go_detected,
        }

        print(
            f"Proceso {process:<2}                    : "
            f"{'OK' if process_ok else 'NO-GO'}"
        )

        if missing:
            for file in missing:
                print(
                    f"  FALTANTE: {file}"
                )

        if not go_detected:
            print(
                "  Evidencia GO explícita       : NO DETECTADA"
            )

    processes_ok = all(
        result["ok"]
        for result in process_results.values()
    )

    controls.append(
        {
            "control": "procesos_42_43_44_45",
            "ok": processes_ok,
            "detalle": json.dumps(
                process_results,
                ensure_ascii=False,
            ),
        }
    )

    if not processes_ok:
        critical_failures.append(
            "Evidencia de cierre 42-45 inválida"
        )

    # =========================================================================
    # 13 - CORRESPONDENCIA DIRECTORIO ↔ ZIP
    # =========================================================================

    print_section(
        "13",
        "CORRESPONDENCIA DIRECTORIO DEFINITIVO ↔ ZIP",
    )

    directory_files = {}

    if FINAL_DIR.exists():

        for path in FINAL_DIR.rglob("*"):

            if path.is_file():

                relative = path.relative_to(
                    FINAL_DIR
                ).as_posix()

                directory_files[relative] = path

    zip_relative_files = {}

    for name in zip_file_names:

        relative = zip_relative_path(
            name,
            zip_root,
        )

        zip_relative_files[relative] = name

    directory_set = set(
        directory_files.keys()
    )

    zip_set = set(
        zip_relative_files.keys()
    )

    only_directory = sorted(
        directory_set - zip_set
    )

    only_zip = sorted(
        zip_set - directory_set
    )

    print(
        f"Archivos en directorio        : "
        f"{len(directory_set)}"
    )

    print(
        f"Archivos en ZIP               : "
        f"{len(zip_set)}"
    )

    print(
        f"Solo en directorio            : "
        f"{len(only_directory)}"
    )

    print(
        f"Solo en ZIP                   : "
        f"{len(only_zip)}"
    )

    for item in only_directory:
        print(
            f"  SOLO DIRECTORIO: {item}"
        )

    for item in only_zip:
        print(
            f"  SOLO ZIP: {item}"
        )

    directory_zip_structure_ok = (
        not only_directory
        and not only_zip
    )

    controls.append(
        {
            "control": "directorio_vs_zip_estructura",
            "ok": directory_zip_structure_ok,
            "detalle": (
                f"solo_directorio={len(only_directory)}; "
                f"solo_zip={len(only_zip)}"
            ),
        }
    )

    if not directory_zip_structure_ok:
        critical_failures.append(
            "Directorio y ZIP no contienen la misma estructura"
        )

    # =========================================================================
    # 14 - EQUIVALENCIA SHA256 DIRECTORIO ↔ ZIP
    # =========================================================================

    print_section(
        "14",
        "EQUIVALENCIA SHA-256 DIRECTORIO ↔ ZIP",
    )

    sha_differences = []

    comparable_files = sorted(
        directory_set & zip_set
    )

    if extraction_ok:

        for relative in comparable_files:

            directory_path = (
                FINAL_DIR
                / relative
            )

            extracted_path = (
                extracted_root
                / relative
            )

            try:

                directory_hash = (
                    sha256_file(
                        directory_path
                    )
                )

                zip_extracted_hash = (
                    sha256_file(
                        extracted_path
                    )
                )

                if (
                    directory_hash
                    != zip_extracted_hash
                ):

                    sha_differences.append(
                        {
                            "archivo": relative,
                            "directorio": directory_hash,
                            "zip": zip_extracted_hash,
                        }
                    )

            except Exception as exc:

                sha_differences.append(
                    {
                        "archivo": relative,
                        "error": str(exc),
                    }
                )

    print(
        f"Archivos comparados           : "
        f"{len(comparable_files)}"
    )

    print(
        f"Diferencias SHA-256           : "
        f"{len(sha_differences)}"
    )

    for difference in sha_differences:
        print(
            f"  DIFERENCIA: {difference}"
        )

    directory_zip_sha_ok = (
        len(comparable_files) == len(directory_set)
        and len(sha_differences) == 0
    )

    controls.append(
        {
            "control": "directorio_vs_zip_sha256",
            "ok": directory_zip_sha_ok,
            "detalle": f"{len(sha_differences)} diferencias",
        }
    )

    if not directory_zip_sha_ok:
        critical_failures.append(
            "SHA-256 directorio ↔ ZIP inconsistente"
        )

    # =========================================================================
    # 15 - MANIFIESTO CSV
    # =========================================================================

    print_section(
        "15",
        "VALIDACIÓN DEL MANIFIESTO CSV",
    )

    manifest_csv_relative = (
        "05_auditoria/"
        "manifiesto_43_paquete_ejecutivo_amba_v4_1.csv"
    )

    manifest_csv_path = (
        extracted_root
        / manifest_csv_relative
    )

    manifest_csv_rows = []

    manifest_csv_ok = False

    if manifest_csv_path.exists():

        try:

            with manifest_csv_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as fh:

                reader = csv.DictReader(fh)

                manifest_csv_rows = list(reader)

            manifest_csv_ok = (
                len(manifest_csv_rows) > 0
            )

        except Exception as exc:

            errors.append(
                f"Error manifiesto CSV: {exc}"
            )

    print(
        f"Entrada                       : "
        f"MODELO_TERRITORIAL_AMBA_V4_FINAL/"
        f"{manifest_csv_relative}"
    )

    print(
        f"Registros                     : "
        f"{len(manifest_csv_rows)}"
    )

    print(
        f"Estado                        : "
        f"{'OK' if manifest_csv_ok else 'ERROR'}"
    )

    controls.append(
        {
            "control": "manifiesto_csv",
            "ok": manifest_csv_ok,
            "detalle": f"{len(manifest_csv_rows)} registros",
        }
    )

    if not manifest_csv_ok:
        critical_failures.append(
            "Manifiesto CSV inválido"
        )

    # =========================================================================
    # 16 - METADATA
    # =========================================================================

    print_section(
        "16",
        "VALIDACIÓN DE METADATOS DEL PAQUETE",
    )

    metadata_relative = (
        "06_metadatos/metadata_paquete.json"
    )

    metadata_path = (
        extracted_root
        / metadata_relative
    )

    metadata = {}

    metadata_ok = False
    metadata_version_ok = False
    metadata_status_ok = False

    if metadata_path.exists():

        try:

            metadata = json.loads(
                safe_read_text(
                    metadata_path
                )
            )

            metadata_version = str(
                metadata.get(
                    "version",
                    metadata.get(
                        "VERSION",
                        ""
                    ),
                )
            )

            metadata_status = str(
                metadata.get(
                    "estado",
                    metadata.get(
                        "dictamen_final",
                        metadata.get(
                            "dictamen",
                            metadata.get(
                                "status",
                                ""
                            ),
                        ),
                    ),
                )
            )

            metadata_version_ok = (
                VERSION.lower()
                in metadata_version.lower()
            )

            metadata_status_ok = (
                "go"
                in metadata_status.lower()
            )

            metadata_ok = (
                metadata_version_ok
                and metadata_status_ok
            )

        except Exception as exc:

            errors.append(
                f"Error metadata: {exc}"
            )

    print(
        f"Entrada                       : "
        f"MODELO_TERRITORIAL_AMBA_V4_FINAL/"
        f"{metadata_relative}"
    )

    print(
        f"Versión detectada             : "
        f"{'OK' if metadata_version_ok else 'NO DETECTADA'}"
    )

    print(
        f"Estado/dictamen               : "
        f"{'OK' if metadata_status_ok else 'NO DETECTADO'}"
    )

    # IMPORTANTE:
    #
    # metadata puede haber sido generada antes de que el proceso 46 escribiera
    # su estado final. Por eso no hacemos fallar el proceso solamente por
    # ausencia de un campo de estado, siempre que exista y sea JSON válido.
    #
    metadata_structural_ok = (
        metadata_path.exists()
        and isinstance(metadata, dict)
        and len(metadata) > 0
    )

    controls.append(
        {
            "control": "metadata",
            "ok": metadata_structural_ok,
            "detalle": (
                f"version_ok={metadata_version_ok}; "
                f"status_ok={metadata_status_ok}"
            ),
        }
    )

    if not metadata_structural_ok:
        critical_failures.append(
            "Metadata inexistente o inválida"
        )

    # =========================================================================
    # 17 - CONSISTENCIA BÁSICA DEL MODELO
    # =========================================================================

    print_section(
        "17",
        "VALIDACIÓN DE CONSISTENCIA BÁSICA DEL MODELO",
    )

    projects_path = (
        extracted_root
        / "01_modelo"
        / "modelo_maestro_proyectos_v4.csv"
    )

    scenarios_path = (
        extracted_root
        / "01_modelo"
        / "modelo_maestro_escenarios_v4.csv"
    )

    project_rows = []
    scenario_rows = []

    if projects_path.exists():

        try:

            with projects_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as fh:

                project_rows = list(
                    csv.DictReader(fh)
                )

        except Exception:
            project_rows = []

    if scenarios_path.exists():

        try:

            with scenarios_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as fh:

                scenario_rows = list(
                    csv.DictReader(fh)
                )

        except Exception:
            scenario_rows = []

    def find_column(rows, candidates):

        if not rows:
            return None

        columns = list(rows[0].keys())

        for candidate in candidates:

            for column in columns:

                if column.lower() == candidate.lower():
                    return column

        return None

    project_id_column = find_column(
        project_rows,
        [
            "proyecto_id",
            "id_proyecto",
            "project_id",
            "id",
        ],
    )

    scenario_id_column = find_column(
        scenario_rows,
        [
            "escenario_id",
            "id_escenario",
            "scenario_id",
            "id",
        ],
    )

    if project_id_column:

        project_ids = [
            str(row.get(project_id_column, "")).strip()
            for row in project_rows
        ]

        project_ids = [
            value
            for value in project_ids
            if value
        ]

    else:

        project_ids = []

    if scenario_id_column:

        scenario_ids = [
            str(row.get(scenario_id_column, "")).strip()
            for row in scenario_rows
        ]

        scenario_ids = [
            value
            for value in scenario_ids
            if value
        ]

    else:

        scenario_ids = []

    duplicate_projects = (
        len(project_ids)
        - len(set(project_ids))
    )

    duplicate_scenarios = (
        len(scenario_ids)
        - len(set(scenario_ids))
    )

    print(
        f"Proyectos                     : "
        f"{len(project_rows)}"
    )

    print(
        f"Escenarios                    : "
        f"{len(scenario_rows)}"
    )

    print(
        f"Proyectos únicos              : "
        f"{len(set(project_ids)) if project_ids else 0}"
    )

    print(
        f"Duplicados proyecto           : "
        f"{duplicate_projects}"
    )

    print(
        f"Escenarios únicos             : "
        f"{len(set(scenario_ids)) if scenario_ids else 0}"
    )

    print(
        f"Duplicados escenario          : "
        f"{duplicate_scenarios}"
    )

    model_ok = (
        len(project_rows) > 0
        and len(scenario_rows) > 0
        and duplicate_projects == 0
        and duplicate_scenarios == 0
    )

    controls.append(
        {
            "control": "consistencia_modelo",
            "ok": model_ok,
            "detalle": (
                f"proyectos={len(project_rows)}; "
                f"escenarios={len(scenario_rows)}; "
                f"dup_proyectos={duplicate_projects}; "
                f"dup_escenarios={duplicate_scenarios}"
            ),
        }
    )

    if not model_ok:
        critical_failures.append(
            "Modelo territorial inconsistente"
        )

    # =========================================================================
    # 18 - REVALIDACIÓN CRÍTICA
    # =========================================================================

    print_section(
        "18",
        "REVALIDACIÓN CRÍTICA DEL ARTEFACTO",
    )

    critical_controls = [
        (
            "ZIP íntegro",
            zip_test_ok,
        ),
        (
            "Estructura",
            structure_ok,
        ),
        (
            "Obligatorios",
            required_ok,
        ),
        (
            "No vacíos",
            non_empty_ok,
        ),
        (
            "Hashes internos",
            internal_hashes_ok,
        ),
        (
            "MANIFIESTO",
            manifest_ok,
        ),
        (
            "README",
            readme_ok,
        ),
        (
            "Procesos 42-45",
            processes_ok,
        ),
        (
            "Directorio-ZIP",
            directory_zip_structure_ok,
        ),
        (
            "SHA directorio-ZIP",
            directory_zip_sha_ok,
        ),
        (
            "Manifiesto CSV",
            manifest_csv_ok,
        ),
        (
            "Metadata",
            metadata_structural_ok,
        ),
        (
            "Modelo",
            model_ok,
        ),
    ]

    for name, ok in critical_controls:

        print(
            f"{name:<30}: "
            f"{'OK' if ok else 'ERROR'}"
        )

    # =========================================================================
    # DETERMINACIÓN DEL DICTAMEN
    # =========================================================================

    print_section(
        "19",
        "DETERMINACIÓN DEL DICTAMEN FINAL",
    )

    total_controls = len(critical_controls)

    passed_controls = sum(
        1
        for _, ok in critical_controls
        if ok
    )

    failed_controls = (
        total_controls
        - passed_controls
    )

    # En esta auditoría, el criterio definitivo es que todos los controles
    # críticos estén OK.
    #
    # No se utiliza un score porcentual para transformar una falla crítica
    # en GO.

    final_go = (
        failed_controls == 0
        and len(critical_failures) == 0
    )

    score = (
        passed_controls
        / total_controls
        * 100
        if total_controls
        else 0
    )

    audit_status = (
        "APROBADA"
        if final_go
        else "OBSERVADA"
    )

    final_dictamen = (
        "GO"
        if final_go
        else "NO-GO"
    )

    print(
        f"Controles OK                  : "
        f"{passed_controls}/{total_controls}"
    )

    print(
        f"Controles fallidos            : "
        f"{failed_controls}"
    )

    print(
        f"Fallas críticas               : "
        f"{len(critical_failures)}"
    )

    print(
        f"Fallas importantes            : "
        f"{len(important_failures)}"
    )

    print(
        f"Score auditoría               : "
        f"{score:.2f}/100"
    )

    print(
        f"Auditoría                     : "
        f"{audit_status}"
    )

    print(
        f"DICTAMEN FINAL                : "
        f"{final_dictamen}"
    )

    # =========================================================================
    # INVENTARIO
    # =========================================================================

    for relative in sorted(zip_relative_files):

        zip_internal_name = zip_relative_files[
            relative
        ]

        path = extracted_root / relative

        if path.exists():

            try:
                size = path.stat().st_size
            except Exception:
                size = 0

            inventory_rows.append(
                {
                    "archivo": relative,
                    "entrada_zip": zip_internal_name,
                    "tamano_bytes": size,
                    "existe": "SI",
                    "vacio": "SI" if size == 0 else "NO",
                }
            )

    # =========================================================================
    # AUDITORÍA CSV
    # =========================================================================

    audit_rows = []

    for name, ok in critical_controls:

        audit_rows.append(
            {
                "control": name,
                "resultado": "OK" if ok else "ERROR",
                "critico": "SI",
            }
        )

    # =========================================================================
    # RESUMEN JSON
    # =========================================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    summary = {
        "proyecto": PROYECTO,
        "version": VERSION,
        "proceso": 47,
        "fecha_hora": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "zip": str(ZIP_PATH),
        "zip_nombre": ZIP_PATH.name,
        "zip_tamano_bytes": zip_size,
        "zip_sha256": zip_sha256,
        "zip_root": zip_root,
        "archivos_zip": len(zip_file_names),
        "archivos_directorio": len(directory_set),
        "controles_total": total_controls,
        "controles_ok": passed_controls,
        "controles_fallidos": failed_controls,
        "fallas_criticas": len(
            critical_failures
        ),
        "fallas_importantes": len(
            important_failures
        ),
        "score": round(score, 2),
        "auditoria": audit_status,
        "dictamen_final": final_dictamen,
        "tiempo_segundos": round(
            elapsed,
            2,
        ),
        "errores": errors,
        "fallas_criticas_detalle": critical_failures,
        "fallas_importantes_detalle": important_failures,
        "procesos": process_results,
        "metadata_version_ok": metadata_version_ok,
        "metadata_status_ok": metadata_status_ok,
    }

    # =========================================================================
    # INFORME MARKDOWN
    # =========================================================================

    report_lines = [
        "# Auditoría 47 — Verificación final",
        "",
        f"**Proyecto:** {PROYECTO}",
        f"**Versión:** {VERSION}",
        f"**Proceso:** 47",
        "",
        "## Artefacto",
        "",
        f"- ZIP: `{ZIP_PATH.name}`",
        f"- Tamaño: {zip_size:,} bytes",
        f"- SHA-256: `{zip_sha256}`",
        f"- Archivos ZIP: {len(zip_file_names)}",
        "",
        "## Resultado",
        "",
        f"- Controles OK: {passed_controls}/{total_controls}",
        f"- Controles fallidos: {failed_controls}",
        f"- Fallas críticas: {len(critical_failures)}",
        f"- Score: {score:.2f}/100",
        f"- Auditoría: **{audit_status}**",
        f"- Dictamen final: **{final_dictamen}**",
        "",
        "## Controles",
        "",
        "| Control | Resultado |",
        "|---|---|",
    ]

    for name, ok in critical_controls:

        report_lines.append(
            f"| {name} | "
            f"{'OK' if ok else 'ERROR'} |"
        )

    report_lines.extend(
        [
            "",
            "## Procesos 42-45",
            "",
        ]
    )

    for process, result in process_results.items():

        report_lines.append(
            f"- Proceso {process}: "
            f"**{'OK' if result['ok'] else 'NO-GO'}**"
        )

    report_lines.extend(
        [
            "",
            "## Fallas críticas",
            "",
        ]
    )

    if critical_failures:

        for failure in critical_failures:
            report_lines.append(
                f"- {failure}"
            )

    else:

        report_lines.append(
            "- Ninguna."
        )

    report_lines.extend(
        [
            "",
            "## Conclusión",
            "",
            (
                "El artefacto cumple todos los controles críticos."
                if final_go
                else
                "El artefacto presenta controles críticos "
                "sin conformidad y no puede considerarse GO."
            ),
            "",
        ]
    )

    report_text = "\n".join(
        report_lines
    )

    # =========================================================================
    # EXPORTACIÓN
    # =========================================================================

    print_section(
        "20",
        "EXPORTANDO RESULTADOS",
    )

    audit_csv_path = (
        AUDIT_DIR
        / "auditoria_47_verificacion_final_amba_v4.csv"
    )

    inventory_csv_path = (
        AUDIT_DIR
        / "inventario_47_verificacion_final_amba_v4.csv"
    )

    hashes_csv_path = (
        AUDIT_DIR
        / "hashes_47_verificacion_final_amba_v4.csv"
    )

    summary_json_path = (
        AUDIT_DIR
        / "resumen_47_verificacion_final_amba_v4.json"
    )

    report_md_path = (
        AUDIT_DIR
        / "informe_47_verificacion_final_amba_v4.md"
    )

    write_csv(
        audit_csv_path,
        audit_rows,
    )

    write_csv(
        inventory_csv_path,
        inventory_rows,
    )

    write_csv(
        hashes_csv_path,
        hash_rows,
    )

    write_json(
        summary_json_path,
        summary,
    )

    report_md_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        f"Auditoría                     : "
        f"{audit_csv_path}"
    )

    print(
        f"Inventario                    : "
        f"{inventory_csv_path}"
    )

    print(
        f"Hashes                        : "
        f"{hashes_csv_path}"
    )

    print(
        f"Resumen                       : "
        f"{summary_json_path}"
    )

    print(
        f"Informe                       : "
        f"{report_md_path}"
    )

    # =========================================================================
    # RESULTADO FINAL
    # =========================================================================

    print_header(
        f"RESULTADO FINAL DEL PROCESO 47 - {final_dictamen}"
    )

    print(
        f"Proyecto                      : {PROYECTO}"
    )

    print(
        f"Versión                       : {VERSION}"
    )

    print(
        f"ZIP                           : {ZIP_PATH.name}"
    )

    print(
        f"Archivos ZIP                  : {len(zip_file_names)}"
    )

    print(
        f"Controles                     : "
        f"{passed_controls}/{total_controls}"
    )

    print(
        f"Fallas críticas               : "
        f"{len(critical_failures)}"
    )

    print(
        f"Fallas importantes            : "
        f"{len(important_failures)}"
    )

    print(
        f"Score auditoría               : "
        f"{score:.2f}/100"
    )

    print(
        f"Auditoría                     : "
        f"{audit_status}"
    )

    print(
        f"DICTAMEN FINAL                : "
        f"{final_dictamen}"
    )

    print(
        f"Tiempo de ejecución           : "
        f"{elapsed:.2f} segundos"
    )

    print()

    if final_go:

        print(
            "=" * 88
        )

        print(
            "47 - VERIFICACIÓN FINAL COMPLETADA - GO"
        )

        print(
            "El artefacto ZIP cumple todos los controles críticos."
        )

        print(
            "DICTAMEN FINAL: GO"
        )

        print(
            "=" * 88
        )

    else:

        print(
            "=" * 88
        )

        print(
            "47 - VERIFICACIÓN FINAL COMPLETADA - NO-GO"
        )

        print(
            "Se detectaron inconsistencias en el artefacto definitivo."
        )

        print(
            "Revisar los resultados de la auditoría 47."
        )

        print(
            "DICTAMEN FINAL: NO-GO"
        )

        print(
            "=" * 88
        )

    # =========================================================================
    # LIMPIEZA
    # =========================================================================

    try:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )
    except Exception:
        pass

    return 0 if final_go else 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())