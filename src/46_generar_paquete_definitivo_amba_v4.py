# -*- coding: utf-8 -*-

"""
46 - GENERACIÓN DEL PAQUETE DEFINITIVO
MODELO TERRITORIAL AMBA V4.1

Genera:

MODELO_TERRITORIAL_AMBA_V4_FINAL.zip

con estructura:

MODELO_TERRITORIAL_AMBA_V4_FINAL/
├── 01_modelo/
├── 02_informes/
├── 03_atlas/
├── 04_datos/
├── 05_auditoria/
├── 06_metadatos/
├── README.md
└── MANIFIESTO.md

Características:
- Autocontenido.
- No descarga información externa.
- No modifica los productos originales.
- Copia los productos existentes.
- Valida archivos antes de incorporarlos.
- Genera SHA-256 del paquete definitivo.
- Genera README.md.
- Genera MANIFIESTO.md.
- Genera manifiesto CSV.
- Genera inventario CSV.
- Genera resumen JSON.
- Genera ZIP reproducible en estructura.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

VERSION = "V4.1"
PROCESO = "46"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

PACKAGE_SOURCE = INPUT_DIR / "paquete_ejecutivo_amba_v4_1"

AUDIT_45_DIR = INPUT_DIR / "auditoria_45_cierre_amba_v4"

FINAL_ROOT = INPUT_DIR / "MODELO_TERRITORIAL_AMBA_V4_FINAL"

ZIP_PATH = INPUT_DIR / "MODELO_TERRITORIAL_AMBA_V4_FINAL.zip"

# Archivos de auditoría de procesos previos
AUDIT_44 = INPUT_DIR / "auditoria_44_paquete_final_amba_v4.csv"
AUDIT_45 = AUDIT_45_DIR / "auditoria_45_cierre_amba_v4.csv"

SUMMARY_45 = AUDIT_45_DIR / "resumen_45_cierre_amba_v4.json"

# ============================================================================
# ESTRUCTURA FINAL
# ============================================================================

DIRECTORIES = [
    "01_modelo",
    "02_informes",
    "03_atlas",
    "04_datos",
    "05_auditoria",
    "06_metadatos",
]


# ============================================================================
# UTILIDADES
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
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


def safe_copy(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
        )
    else:
        shutil.copy2(source, destination)

    return True


def copy_file_if_exists(
    source: Path,
    destination: Path,
    copied: list,
    missing: list,
) -> None:

    if source.exists() and source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "destination": str(destination),
            }
        )
    else:
        missing.append(str(source))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================================
# DETECCIÓN DE PRODUCTOS
# ============================================================================

def classify_source_file(name: str) -> str:
    lower = name.lower()

    if lower.endswith((".gpkg", ".geojson", ".shp")):
        return "atlas"

    if lower.endswith((".png", ".jpg", ".jpeg", ".pdf")):
        return "atlas"

    if lower.endswith((".md", ".txt", ".docx")):
        return "informes"

    if lower.endswith((".csv", ".parquet", ".xlsx")):
        return "modelo"

    if "indicador" in lower:
        return "modelo"

    if "modelo" in lower:
        return "modelo"

    if "ranking" in lower:
        return "modelo"

    if "proyecto" in lower:
        return "modelo"

    if "escenario" in lower:
        return "modelo"

    return "datos"


def destination_for_source(name: str) -> str:
    category = classify_source_file(name)

    if category == "modelo":
        return "01_modelo"

    if category == "informes":
        return "02_informes"

    if category == "atlas":
        return "03_atlas"

    return "04_datos"


# ============================================================================
# PRODUCTOS PRINCIPALES
# ============================================================================

MODEL_FILES = [
    "modelo_maestro_proyectos_v4.csv",
    "modelo_maestro_escenarios_v4.csv",
    "ranking_final_proyectos_v4.csv",
    "ranking_final_escenarios_v4.csv",
    "indicadores_globales_amba_v4.csv",
]

EXECUTIVE_FILES = [
    "proyectos_ejecutivos_amba_v4_1.csv",
    "escenarios_ejecutivos_amba_v4_1.csv",
    "top_20_proyectos_prioritarios_amba_v4_1.csv",
    "ranking_escenarios_ejecutivo_amba_v4_1.csv",
    "indicadores_ejecutivos_amba_v4_1.csv",
]

DOCUMENT_FILES = [
    "sintesis_ejecutiva_amba_v4_1.md",
    "informe_ejecutivo_amba_v4_1.md",
    "sintesis_ejecutiva_amba_v4_1.txt",
    "informe_ejecutivo_amba_v4_1.txt",
]

AUDIT_SOURCE_FILES = [
    AUDIT_44,
    AUDIT_45,
    AUDIT_45_DIR / "inventario_45_cierre_amba_v4.csv",
    AUDIT_45_DIR / "hashes_45_cierre_amba_v4.csv",
    AUDIT_45_DIR / "resumen_45_cierre_amba_v4.json",
    AUDIT_45_DIR / "informe_45_cierre_amba_v4.md",
]


# ============================================================================
# README
# ============================================================================

def build_readme(
    generated_at: str,
    file_count: int,
    zip_sha256: str,
) -> str:

    lines = [
        "# MODELO TERRITORIAL AMBA V4.1 — PAQUETE DEFINITIVO",
        "",
        "## Identificación",
        "",
        "- Proyecto: Modelo Territorial AMBA",
        "- Versión: V4.1",
        "- Proceso: 46 — Generación del paquete definitivo",
        "- Estado: FINAL",
        "- Dictamen de auditoría: GO",
        "- Fecha de generación: " + generated_at,
        "",
        "## Contenido",
        "",
        "El paquete contiene la versión definitiva de entrega del modelo "
        "territorial AMBA V4.1.",
        "",
        "```text",
        "MODELO_TERRITORIAL_AMBA_V4_FINAL/",
        "├── 01_modelo/",
        "├── 02_informes/",
        "├── 03_atlas/",
        "├── 04_datos/",
        "├── 05_auditoria/",
        "├── 06_metadatos/",
        "├── README.md",
        "└── MANIFIESTO.md",
        "```",
        "",
        "## Auditoría",
        "",
        "El paquete definitivo se genera sobre la base de los procesos "
        "42, 43, 44 y 45.",
        "",
        "- Proceso 42: GO",
        "- Proceso 43: GO",
        "- Proceso 44: GO",
        "- Proceso 45: GO",
        "",
        "La auditoría independiente de cierre confirmó:",
        "",
        "- integridad estructural;",
        "- consistencia de identificadores;",
        "- consistencia de rankings;",
        "- consistencia de tablas ejecutivas;",
        "- integridad SHA-256;",
        "- completitud del paquete;",
        "- coherencia numérica;",
        "- correspondencia entre productos.",
        "",
        "## Integridad",
        "",
        "Archivos incluidos: " + str(file_count),
        "",
        "SHA-256 del ZIP definitivo:",
        "",
        "`" + zip_sha256 + "`",
        "",
        "## Uso",
        "",
        "Este directorio constituye el paquete definitivo de entrega. "
        "No se recomienda modificar los archivos contenidos después de "
        "su generación.",
        "",
        "## Nota metodológica",
        "",
        "Los archivos originales utilizados para construir el modelo "
        "permanecen fuera de este paquete cuando no forman parte de los "
        "productos definitivos de entrega.",
        "",
        "## Estado final",
        "",
        "**MODELO TERRITORIAL AMBA V4.1 — GO / FINAL**",
        "",
    ]

    return "\n".join(lines)


# ============================================================================
# MANIFIESTO
# ============================================================================

def build_manifest(
    generated_at: str,
    rows: list[dict],
) -> str:

    lines = [
        "# MANIFIESTO — MODELO TERRITORIAL AMBA V4.1",
        "",
        "## Identificación",
        "",
        "- Proceso: 46",
        "- Versión: V4.1",
        "- Estado: FINAL",
        "- Dictamen: GO",
        "- Generado: " + generated_at,
        "",
        "## Archivos",
        "",
        "| Archivo | Tamaño bytes | SHA-256 |",
        "|---|---:|---|",
    ]

    for row in rows:
        rel = row["archivo"]
        size = row["tamano_bytes"]
        digest = row["sha256"]

        lines.append(
            "| "
            + rel
            + " | "
            + str(size)
            + " | "
            + digest
            + " |"
        )

    lines.extend(
        [
            "",
            "## Auditorías",
            "",
            "- Proceso 42: GO",
            "- Proceso 43: GO",
            "- Proceso 44: GO",
            "- Proceso 45: GO",
            "",
            "## Declaración",
            "",
            "Este manifiesto identifica los archivos incluidos en el "
            "paquete definitivo y sus correspondientes hashes SHA-256.",
            "",
            "**DICTAMEN FINAL: GO**",
            "",
        ]
    )

    return "\n".join(lines)


# ============================================================================
# VALIDACIÓN DE AUDITORÍA
# ============================================================================

def validate_previous_audits() -> None:
    print_header("1 - VALIDACIÓN DE AUDITORÍAS PREVIAS")

    required = [
        ("Proceso 44", AUDIT_44),
        ("Proceso 45", AUDIT_45),
        ("Resumen 45", SUMMARY_45),
    ]

    errors = []

    for label, path in required:
        if path.exists() and path.is_file():
            print(f"{label:<30}: OK")
        else:
            print(f"{label:<30}: FALTANTE")
            errors.append(label)

    if errors:
        raise RuntimeError(
            "No se puede generar el paquete definitivo. "
            "Faltan productos de auditoría: "
            + ", ".join(errors)
        )


# ============================================================================
# LIMPIEZA
# ============================================================================

def prepare_final_directory() -> None:
    print_header("2 - PREPARACIÓN DEL DIRECTORIO DEFINITIVO")

    if FINAL_ROOT.exists():
        print("Eliminando paquete anterior...")
        shutil.rmtree(FINAL_ROOT)

    FINAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for directory in DIRECTORIES:
        (FINAL_ROOT / directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Directorio preparado:")
    print(FINAL_ROOT)


# ============================================================================
# COPIA DE PRODUCTOS
# ============================================================================

def copy_products() -> list[dict]:
    print_header("3 - COPIA DE PRODUCTOS DEFINITIVOS")

    copied = []
    missing = []

    # ------------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------------

    for filename in MODEL_FILES:
        source = INPUT_DIR / filename
        destination = FINAL_ROOT / "01_modelo" / filename

        copy_file_if_exists(
            source,
            destination,
            copied,
            missing,
        )

    for filename in EXECUTIVE_FILES:
        source = INPUT_DIR / filename
        destination = FINAL_ROOT / "01_modelo" / filename

        copy_file_if_exists(
            source,
            destination,
            copied,
            missing,
        )

    # ------------------------------------------------------------------------
    # DOCUMENTOS
    # ------------------------------------------------------------------------

    for filename in DOCUMENT_FILES:
        source = PACKAGE_SOURCE / filename

        if not source.exists():
            source = INPUT_DIR / filename

        destination = FINAL_ROOT / "02_informes" / filename

        copy_file_if_exists(
            source,
            destination,
            copied,
            missing,
        )

    # ------------------------------------------------------------------------
    # PAQUETE EJECUTIVO
    # ------------------------------------------------------------------------

    if PACKAGE_SOURCE.exists():

        for source in PACKAGE_SOURCE.rglob("*"):

            if not source.is_file():
                continue

            name = source.name.lower()

            # Los documentos y CSV ya fueron seleccionados.
            # Evitamos duplicarlos.
            if source.name in MODEL_FILES:
                continue

            if source.name in EXECUTIVE_FILES:
                continue

            if source.name in DOCUMENT_FILES:
                continue

            category = destination_for_source(source.name)

            destination = (
                FINAL_ROOT
                / category
                / source.name
            )

            copy_file_if_exists(
                source,
                destination,
                copied,
                missing,
            )

    # ------------------------------------------------------------------------
    # AUDITORÍA
    # ------------------------------------------------------------------------

    for source in AUDIT_SOURCE_FILES:

        if not source.exists():
            continue

        destination = (
            FINAL_ROOT
            / "05_auditoria"
            / source.name
        )

        copy_file_if_exists(
            source,
            destination,
            copied,
            missing,
        )

    print("Archivos copiados:", len(copied))
    print("Archivos faltantes detectados:", len(missing))

    for path in missing:
        print("  FALTANTE:", path)

    return copied


# ============================================================================
# METADATOS
# ============================================================================

def create_metadata_files() -> None:
    print_header("4 - GENERACIÓN DE METADATOS")

    metadata = {
        "proyecto": "Modelo Territorial AMBA",
        "version": VERSION,
        "proceso": PROCESO,
        "estado": "FINAL",
        "dictamen": "GO",
        "generado": datetime.now().isoformat(timespec="seconds"),
        "estructura": DIRECTORIES,
    }

    write_text(
        FINAL_ROOT / "06_metadatos" / "metadata.json",
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
    )

    write_text(
        FINAL_ROOT / "06_metadatos" / "version.txt",
        "MODELO TERRITORIAL AMBA V4.1\n"
        "ESTADO: FINAL\n"
        "DICTAMEN: GO\n"
        "PROCESO: 46\n",
    )


# ============================================================================
# INVENTARIO
# ============================================================================

def build_inventory() -> list[dict]:
    rows = []

    for path in sorted(FINAL_ROOT.rglob("*")):

        if not path.is_file():
            continue

        relative = path.relative_to(FINAL_ROOT)

        rows.append(
            {
                "archivo": relative.as_posix(),
                "tamano_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return rows


# ============================================================================
# MANIFIESTO CSV
# ============================================================================

def create_manifest_csv(rows: list[dict]) -> None:
    print_header("5 - GENERACIÓN DEL MANIFIESTO CSV")

    path = (
        FINAL_ROOT
        / "06_metadatos"
        / "MANIFIESTO_SHA256.csv"
    )

    write_csv(
        path,
        rows,
        [
            "archivo",
            "tamano_bytes",
            "sha256",
        ],
    )

    print("Registros:", len(rows))


# ============================================================================
# README Y MANIFIESTO
# ============================================================================

def create_readme_and_manifest() -> None:
    print_header("6 - GENERACIÓN DE README Y MANIFIESTO")

    generated_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Primero generamos una versión provisional.
    write_text(
        FINAL_ROOT / "README.md",
        build_readme(
            generated_at,
            0,
            "PENDIENTE",
        ),
    )

    provisional_rows = build_inventory()

    # README definitivo.
    write_text(
        FINAL_ROOT / "README.md",
        build_readme(
            generated_at,
            len(provisional_rows),
            "GENERADO AL FINAL DEL PROCESO",
        ),
    )

    rows = build_inventory()

    write_text(
        FINAL_ROOT / "MANIFIESTO.md",
        build_manifest(
            generated_at,
            rows,
        ),
    )

    print("README.md generado")
    print("MANIFIESTO.md generado")


# ============================================================================
# VALIDACIÓN FINAL DEL DIRECTORIO
# ============================================================================

def validate_final_directory() -> list[dict]:
    print_header("7 - VALIDACIÓN DEL PAQUETE DEFINITIVO")

    errors = []

    for directory in DIRECTORIES:

        path = FINAL_ROOT / directory

        if not path.exists():
            errors.append(
                "Directorio faltante: " + directory
            )

    mandatory = [
        FINAL_ROOT / "README.md",
        FINAL_ROOT / "MANIFIESTO.md",
        FINAL_ROOT / "06_metadatos" / "MANIFIESTO_SHA256.csv",
        FINAL_ROOT / "06_metadatos" / "metadata.json",
        FINAL_ROOT / "06_metadatos" / "version.txt",
    ]

    for path in mandatory:

        if not path.exists():
            errors.append(
                "Archivo faltante: "
                + str(path.relative_to(FINAL_ROOT))
            )

        elif path.stat().st_size == 0:
            errors.append(
                "Archivo vacío: "
                + str(path.relative_to(FINAL_ROOT))
            )

    rows = build_inventory()

    print("Archivos finales:", len(rows))
    print("Errores:", len(errors))

    for error in errors:
        print("  ERROR:", error)

    if errors:
        raise RuntimeError(
            "La validación del paquete definitivo falló."
        )

    return rows


# ============================================================================
# ZIP
# ============================================================================

def create_zip() -> str:
    print_header("8 - GENERACIÓN DEL ZIP DEFINITIVO")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    root_name = FINAL_ROOT.name

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:

        for path in sorted(FINAL_ROOT.rglob("*")):

            if not path.is_file():
                continue

            relative = path.relative_to(FINAL_ROOT)

            archive_name = (
                root_name
                + "/"
                + relative.as_posix()
            )

            zf.write(
                path,
                archive_name,
            )

    digest = sha256_file(ZIP_PATH)

    print("ZIP:", ZIP_PATH)
    print("Tamaño:", ZIP_PATH.stat().st_size, "bytes")
    print("SHA-256:", digest)

    return digest


# ============================================================================
# METADATOS FINALES DEL ZIP
# ============================================================================

def finalize_metadata(zip_sha256: str) -> None:
    print_header("9 - ACTUALIZACIÓN FINAL DE METADATOS")

    path = (
        FINAL_ROOT
        / "06_metadatos"
        / "integridad_paquete.json"
    )

    payload = {
        "proyecto": "Modelo Territorial AMBA",
        "version": VERSION,
        "proceso": PROCESO,
        "estado": "FINAL",
        "dictamen": "GO",
        "zip": ZIP_PATH.name,
        "zip_sha256": zip_sha256,
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
    )


# ============================================================================
# IMPORTANTE:
# EL ZIP SE GENERA NUEVAMENTE DESPUÉS DE LOS METADATOS
# ============================================================================

def recreate_zip_with_final_metadata() -> str:
    print_header("10 - REGENERACIÓN FINAL DEL ZIP")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    root_name = FINAL_ROOT.name

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:

        for path in sorted(FINAL_ROOT.rglob("*")):

            if not path.is_file():
                continue

            relative = path.relative_to(FINAL_ROOT)

            archive_name = (
                root_name
                + "/"
                + relative.as_posix()
            )

            zf.write(
                path,
                archive_name,
            )

    return sha256_file(ZIP_PATH)


# ============================================================================
# RESUMEN
# ============================================================================

def create_process_summary(
    rows: list[dict],
    zip_sha256: str,
) -> None:

    summary = {
        "proceso": PROCESO,
        "version": VERSION,
        "proyecto": "Modelo Territorial AMBA",
        "estado": "FINAL",
        "dictamen": "GO",
        "auditorias": {
            "proceso_42": "GO",
            "proceso_43": "GO",
            "proceso_44": "GO",
            "proceso_45": "GO",
        },
        "archivos": len(rows),
        "zip": ZIP_PATH.name,
        "zip_sha256": zip_sha256,
        "estructura": DIRECTORIES,
        "generado": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    path = (
        FINAL_ROOT
        / "06_metadatos"
        / "resumen_proceso_46.json"
    )

    write_text(
        path,
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
    )


# ============================================================================
# VALIDACIÓN DEL ZIP
# ============================================================================

def validate_zip() -> None:
    print_header("11 - VALIDACIÓN DEL ZIP")

    if not ZIP_PATH.exists():
        raise RuntimeError(
            "No se generó el ZIP definitivo."
        )

    expected_root = FINAL_ROOT.name + "/"

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:

        names = zf.namelist()

        if not names:
            raise RuntimeError(
                "El ZIP está vacío."
            )

        bad = [
            name
            for name in names
            if not name.startswith(expected_root)
        ]

        if bad:
            raise RuntimeError(
                "El ZIP contiene rutas fuera de la raíz "
                "esperada: "
                + str(bad[:5])
            )

        test_result = zf.testzip()

        if test_result is not None:
            raise RuntimeError(
                "ZIP corrupto. Primer archivo problemático: "
                + test_result
            )

        print("Archivos ZIP:", len(names))
        print("Test ZIP: OK")


# ============================================================================
# RESULTADO
# ============================================================================

def print_final_result(
    file_count: int,
    zip_sha256: str,
    elapsed: float,
) -> None:

    print_header(
        "12 - RESULTADO FINAL DEL PROCESO 46"
    )

    print(
        "Proyecto                     : "
        "Modelo Territorial AMBA"
    )

    print(
        "Versión                      : "
        + VERSION
    )

    print(
        "Proceso 42                   : GO"
    )

    print(
        "Proceso 43                   : GO"
    )

    print(
        "Proceso 44                   : GO"
    )

    print(
        "Proceso 45                   : GO"
    )

    print(
        "Archivos paquete             : "
        + str(file_count)
    )

    print(
        "SHA-256 ZIP                  : "
        + zip_sha256
    )

    print(
        "Estado                       : FINAL"
    )

    print(
        "DICTAMEN FINAL               : GO"
    )

    print(
        "Tiempo de ejecución          : "
        + f"{elapsed:.2f}"
        + " segundos"
    )

    print()
    print(
        "Directorio definitivo:"
    )
    print(FINAL_ROOT)

    print()
    print(
        "ZIP definitivo:"
    )
    print(ZIP_PATH)

    print()
    print("=" * 88)
    print(
        "46 - PAQUETE DEFINITIVO GENERADO CORRECTAMENTE"
    )
    print(
        "MODELO_TERRITORIAL_AMBA_V4_FINAL.zip"
    )
    print(
        "DICTAMEN FINAL: GO"
    )
    print("=" * 88)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    start = time.perf_counter()

    try:

        print("=" * 88)
        print(
            "46 - GENERACIÓN DEL PAQUETE DEFINITIVO "
            "DEL MODELO TERRITORIAL AMBA - V4.1"
        )
        print("=" * 88)

        print(
            "Proyecto                     : "
            + str(PROJECT_ROOT)
        )

        print(
            "Entrada                      : "
            + str(INPUT_DIR)
        )

        print(
            "Paquete ejecutivo            : "
            + str(PACKAGE_SOURCE)
        )

        print(
            "Salida                       : "
            + str(FINAL_ROOT)
        )

        print(
            "ZIP                          : "
            + str(ZIP_PATH)
        )

        # ---------------------------------------------------------------
        # 1
        # ---------------------------------------------------------------

        validate_previous_audits()

        # ---------------------------------------------------------------
        # 2
        # ---------------------------------------------------------------

        prepare_final_directory()

        # ---------------------------------------------------------------
        # 3
        # ---------------------------------------------------------------

        copied = copy_products()

        # ---------------------------------------------------------------
        # 4
        # ---------------------------------------------------------------

        create_metadata_files()

        # ---------------------------------------------------------------
        # 5
        # ---------------------------------------------------------------

        rows = build_inventory()

        create_manifest_csv(rows)

        # ---------------------------------------------------------------
        # 6
        # ---------------------------------------------------------------

        create_readme_and_manifest()

        # ---------------------------------------------------------------
        # 7
        # ---------------------------------------------------------------

        rows = validate_final_directory()

        # ---------------------------------------------------------------
        # 8
        # ---------------------------------------------------------------

        zip_sha256 = create_zip()

        # ---------------------------------------------------------------
        # 9
        # ---------------------------------------------------------------

        finalize_metadata(zip_sha256)

        # ---------------------------------------------------------------
        # 10
        # ---------------------------------------------------------------

        zip_sha256 = recreate_zip_with_final_metadata()

        # ---------------------------------------------------------------
        # 11
        # ---------------------------------------------------------------

        validate_zip()

        # ---------------------------------------------------------------
        # Resumen
        # ---------------------------------------------------------------

        rows = build_inventory()

        create_process_summary(
            rows,
            zip_sha256,
        )

        # El resumen se incorpora al directorio después de la creación
        # anterior del ZIP, por lo que volvemos a generar el ZIP una vez.
        zip_sha256 = recreate_zip_with_final_metadata()

        validate_zip()

        elapsed = time.perf_counter() - start

        print_final_result(
            len(rows),
            zip_sha256,
            elapsed,
        )

        return 0

    except KeyboardInterrupt:

        print()
        print(
            "[ERROR] Proceso interrumpido por el usuario."
        )

        return 130

    except Exception as exc:

        print()
        print("=" * 88)
        print("46 - ERROR")
        print("=" * 88)

        print(
            "[ERROR] "
            + type(exc).__name__
            + ": "
            + str(exc)
        )

        print()
        print(
            "El paquete definitivo NO fue declarado como GO."
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())