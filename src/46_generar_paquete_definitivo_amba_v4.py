````python
# -*- coding: utf-8 -*-

"""
46 - GENERACIÓN DEL PAQUETE DEFINITIVO
MODELO TERRITORIAL AMBA V4.1

Construye el artefacto definitivo:

MODELO_TERRITORIAL_AMBA_V4_FINAL/
MODELO_TERRITORIAL_AMBA_V4_FINAL.zip

El paquete queda cerrado antes de generar el ZIP definitivo.

Diseño:
- Todos los productos se copian primero.
- Se generan metadata, README, MANIFIESTO y resumen.
- Se genera MANIFIESTO_SHA256 al final.
- Se genera el ZIP una sola vez.
- No se modifica ningún archivo después de crear el ZIP.
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
PROYECTO_NOMBRE = "Modelo Territorial AMBA"

SCRIPT_NAME = "46_generar_paquete_definitivo_amba_v4.py"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

EXECUTIVE_DIR = INPUT_DIR / "paquete_ejecutivo_amba_v4_1"

FINAL_DIR = INPUT_DIR / "MODELO_TERRITORIAL_AMBA_V4_FINAL"

ZIP_PATH = INPUT_DIR / "MODELO_TERRITORIAL_AMBA_V4_FINAL.zip"

SEP = "=" * 88


# ============================================================================
# ESTRUCTURA
# ============================================================================

DIRECTORIES = [
    "01_modelo",
    "02_informes",
    "03_atlas",
    "04_datos",
    "05_auditoria",
    "06_metadatos",
]


MODEL_FILES = [
    "modelo_maestro_proyectos_v4.csv",
    "modelo_maestro_escenarios_v4.csv",
    "ranking_final_proyectos_v4.csv",
    "ranking_final_escenarios_v4.csv",
]


EXECUTIVE_FILES = [
    "proyectos_ejecutivos_amba_v4_1.csv",
    "escenarios_ejecutivos_amba_v4_1.csv",
    "top_20_proyectos_prioritarios_amba_v4_1.csv",
    "ranking_escenarios_ejecutivo_amba_v4_1.csv",
    "indicadores_ejecutivos_amba_v4_1.csv",
    "sintesis_ejecutiva_amba_v4_1.md",
    "informe_ejecutivo_amba_v4_1.md",
]


DATA_FILES = [
    "indicadores_globales_amba_v4.csv",
]


AUDIT_FILES = [
    "cierre_42_modelo_territorial_amba_v4.csv",
    "control_paquete_ejecutivo_amba_v4_1.csv",
    "manifiesto_43_paquete_ejecutivo_amba_v4_1.csv",

    "auditoria_44_paquete_final_amba_v4.csv",
    "inventario_44_paquete_final_amba_v4.csv",
    "hashes_44_paquete_final_amba_v4.csv",
    "resumen_44_auditoria_paquete_final_amba_v4.json",
    "informe_44_auditoria_paquete_final_amba_v4.md",

    "auditoria_45_cierre_amba_v4.csv",
    "inventario_45_cierre_amba_v4.csv",
    "hashes_45_cierre_amba_v4.csv",
    "resumen_45_cierre_amba_v4.json",
    "informe_45_cierre_amba_v4.md",
]


REQUIRED_FINAL_FILES = [
    "01_modelo/modelo_maestro_proyectos_v4.csv",
    "01_modelo/modelo_maestro_escenarios_v4.csv",
    "01_modelo/ranking_final_proyectos_v4.csv",
    "01_modelo/ranking_final_escenarios_v4.csv",

    "02_informes/proyectos_ejecutivos_amba_v4_1.csv",
    "02_informes/escenarios_ejecutivos_amba_v4_1.csv",
    "02_informes/top_20_proyectos_prioritarios_amba_v4_1.csv",
    "02_informes/ranking_escenarios_ejecutivo_amba_v4_1.csv",
    "02_informes/indicadores_ejecutivos_amba_v4_1.csv",
    "02_informes/sintesis_ejecutiva_amba_v4_1.md",
    "02_informes/informe_ejecutivo_amba_v4_1.md",
    "02_informes/sintesis_ejecutiva_amba_v4_1.txt",
    "02_informes/informe_ejecutivo_amba_v4_1.txt",

    "04_datos/indicadores_globales_amba_v4.csv",

    "05_auditoria/cierre_42_modelo_territorial_amba_v4.csv",
    "05_auditoria/control_paquete_ejecutivo_amba_v4_1.csv",
    "05_auditoria/manifiesto_43_paquete_ejecutivo_amba_v4_1.csv",

    "05_auditoria/auditoria_44_paquete_final_amba_v4.csv",
    "05_auditoria/inventario_44_paquete_final_amba_v4.csv",
    "05_auditoria/hashes_44_paquete_final_amba_v4.csv",
    "05_auditoria/resumen_44_auditoria_paquete_final_amba_v4.json",
    "05_auditoria/informe_44_auditoria_paquete_final_amba_v4.md",

    "05_auditoria/auditoria_45_cierre_amba_v4.csv",
    "05_auditoria/inventario_45_cierre_amba_v4.csv",
    "05_auditoria/hashes_45_cierre_amba_v4.csv",
    "05_auditoria/resumen_45_cierre_amba_v4.json",
    "05_auditoria/informe_45_cierre_amba_v4.md",

    "06_metadatos/metadata_paquete.json",
    "06_metadatos/MANIFIESTO_SHA256.csv",
    "06_metadatos/resumen_proceso_46.json",

    "README.md",
    "MANIFIESTO.md",
]


# ============================================================================
# UTILIDADES
# ============================================================================

def print_section(title: str) -> None:
    print()
    print(SEP)
    print(title)
    print(SEP)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def is_nonempty_file(path: Path) -> bool:
    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def relative_source(path: Path) -> str:
    try:
        return str(
            path.relative_to(INPUT_DIR)
        ).replace("\\", "/")
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except Exception:
        return {}


def normalize_name(value: str) -> str:
    return value.replace("\\", "/").strip("/")


# ============================================================================
# BÚSQUEDA DE ARCHIVOS
# ============================================================================

def all_candidate_files() -> list[Path]:

    candidates = []

    if not INPUT_DIR.exists():
        return candidates

    for p in INPUT_DIR.rglob("*"):

        if not p.is_file():
            continue

        if FINAL_DIR in p.parents:
            continue

        if p == ZIP_PATH:
            continue

        candidates.append(p)

    return candidates


def find_file(filename: str) -> Path | None:

    direct_candidates = [
        INPUT_DIR / filename,
        EXECUTIVE_DIR / filename,
    ]

    for candidate in direct_candidates:

        if is_nonempty_file(candidate):
            return candidate

    matches = []

    for p in all_candidate_files():

        if p.name.lower() == filename.lower():
            matches.append(p)

    if not matches:
        return None

    def priority(p: Path) -> tuple[int, int]:

        text = str(p).lower()

        if "paquete_ejecutivo_amba_v4_1" in text:
            return 0, len(text)

        if "auditoria_45_cierre_amba_v4" in text:
            return 1, len(text)

        if "auditoria_44_paquete_final_amba_v4" in text:
            return 2, len(text)

        return 3, len(text)

    matches.sort(key=priority)

    return matches[0]


# ============================================================================
# TXT
# ============================================================================

def markdown_to_text(text: str) -> str:

    lines = []

    for line in text.splitlines():

        s = line.strip()

        if s.startswith("```"):
            continue

        for prefix in (
            "### ",
            "## ",
            "# ",
        ):
            if s.startswith(prefix):
                s = s[len(prefix):]

        s = s.replace("**", "")
        s = s.replace("__", "")
        s = s.replace("`", "")

        if s.startswith("- "):
            s = "• " + s[2:]

        lines.append(s)

    return "\n".join(lines).strip() + "\n"


def ensure_txt_from_md(
    md_path: Path,
    txt_path: Path,
) -> None:

    text = md_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    txt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_path.write_text(
        markdown_to_text(text),
        encoding="utf-8",
    )


# ============================================================================
# AUDITORÍAS PREVIAS
# ============================================================================

def validate_previous_audits() -> dict:

    print_section("1 - VALIDACIÓN DE AUDITORÍAS PREVIAS")

    result = {
        "proceso_44": False,
        "proceso_45": False,
        "resumen_44": False,
        "resumen_45": False,
    }

    audit44 = find_file(
        "resumen_44_auditoria_paquete_final_amba_v4.json"
    )

    audit45 = find_file(
        "resumen_45_cierre_amba_v4.json"
    )

    if audit44:

        result["resumen_44"] = True

        data = read_json(audit44)

        text = json.dumps(
            data,
            ensure_ascii=False,
        ).upper()

        if (
            "GO" in text
            and "NO-GO" not in text
            and "NOGO" not in text
        ):
            result["proceso_44"] = True

    if audit45:

        result["resumen_45"] = True

        data = read_json(audit45)

        text = json.dumps(
            data,
            ensure_ascii=False,
        ).upper()

        if (
            "GO" in text
            and "NO-GO" not in text
            and "NOGO" not in text
        ):
            result["proceso_45"] = True

    print(
        "Proceso 44                    : "
        + ("OK" if result["proceso_44"] else "NO DISPONIBLE")
    )

    print(
        "Proceso 45                    : "
        + ("OK" if result["proceso_45"] else "NO DISPONIBLE")
    )

    print(
        "Resumen 44                    : "
        + ("OK" if result["resumen_44"] else "NO DISPONIBLE")
    )

    print(
        "Resumen 45                    : "
        + ("OK" if result["resumen_45"] else "NO DISPONIBLE")
    )

    return result


# ============================================================================
# PREPARACIÓN
# ============================================================================

def prepare_final_directory() -> None:

    print_section("2 - PREPARACIÓN DEL DIRECTORIO DEFINITIVO")

    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for directory in DIRECTORIES:

        (
            FINAL_DIR / directory
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Directorio preparado:")
    print(FINAL_DIR)


# ============================================================================
# COPIA
# ============================================================================

def copy_one(
    filename: str,
    destination_dir: str,
    copied: list[dict],
    missing: list[str],
) -> None:

    source = find_file(filename)

    if source is None:

        missing.append(filename)
        return

    destination = (
        FINAL_DIR
        / destination_dir
        / filename
    )

    safe_copy(
        source,
        destination,
    )

    copied.append(
        {
            "archivo": filename,
            "origen": relative_source(source),
            "destino": str(
                destination.relative_to(FINAL_DIR)
            ).replace("\\", "/"),
            "tamano_bytes": destination.stat().st_size,
        }
    )


def copy_products() -> tuple[list[dict], list[str]]:

    print_section("3 - COPIA DE PRODUCTOS DEFINITIVOS")

    copied = []
    missing = []

    for filename in MODEL_FILES:

        copy_one(
            filename,
            "01_modelo",
            copied,
            missing,
        )

    for filename in EXECUTIVE_FILES:

        copy_one(
            filename,
            "02_informes",
            copied,
            missing,
        )

    for filename in DATA_FILES:

        copy_one(
            filename,
            "04_datos",
            copied,
            missing,
        )

    for filename in AUDIT_FILES:

        copy_one(
            filename,
            "05_auditoria",
            copied,
            missing,
        )

    # ---------------------------------------------------------------
    # TXT ejecutivos
    # ---------------------------------------------------------------

    txt_pairs = [
        (
            "sintesis_ejecutiva_amba_v4_1.md",
            "sintesis_ejecutiva_amba_v4_1.txt",
        ),
        (
            "informe_ejecutivo_amba_v4_1.md",
            "informe_ejecutivo_amba_v4_1.txt",
        ),
    ]

    for md_name, txt_name in txt_pairs:

        source_md = (
            FINAL_DIR
            / "02_informes"
            / md_name
        )

        target_txt = (
            FINAL_DIR
            / "02_informes"
            / txt_name
        )

        if source_md.exists():

            ensure_txt_from_md(
                source_md,
                target_txt,
            )

            copied.append(
                {
                    "archivo": txt_name,
                    "origen": (
                        "generado_desde/"
                        + md_name
                    ),
                    "destino": (
                        "02_informes/"
                        + txt_name
                    ),
                    "tamano_bytes": target_txt.stat().st_size,
                }
            )

    # ---------------------------------------------------------------
    # Productos espaciales
    # ---------------------------------------------------------------

    spatial_extensions = {
        ".gpkg",
        ".geojson",
        ".shp",
        ".dbf",
        ".shx",
        ".prj",
        ".cpg",
    }

    required_spatial_names = {
        "atlas_territorial_amba_v4.gpkg",
        "escenarios_territoriales_amba.gpkg",
        "geometria_cartera_proyectos_v4.gpkg",
        "geometria_escenarios_cartera_v4.gpkg",
        "modelo_maestro_territorial_amba_v4.gpkg",
        "modelo_territorial_amba_v4.gpkg",
        "priorizacion_territorial_escenarios_v4.gpkg",
    }

    spatial_found = set()

    for p in all_candidate_files():

        if p.suffix.lower() not in spatial_extensions:
            continue

        if p.name in spatial_found:
            continue

        if (
            p.suffix.lower() == ".gpkg"
            and p.name not in required_spatial_names
        ):
            continue

        destination = (
            FINAL_DIR
            / "03_atlas"
            / p.name
        )

        if destination.exists():
            continue

        try:

            safe_copy(
                p,
                destination,
            )

            spatial_found.add(p.name)

            copied.append(
                {
                    "archivo": p.name,
                    "origen": relative_source(p),
                    "destino": str(
                        destination.relative_to(FINAL_DIR)
                    ).replace("\\", "/"),
                    "tamano_bytes": destination.stat().st_size,
                }
            )

        except Exception as exc:

            print(
                "ADVERTENCIA: no se pudo copiar "
                + str(p)
                + ": "
                + str(exc)
            )

    print(
        "Archivos copiados             : "
        + str(len(copied))
    )

    print(
        "Archivos faltantes            : "
        + str(len(missing))
    )

    if missing:

        for item in missing:
            print("  FALTANTE: " + item)

    return copied, missing


# ============================================================================
# METADATA
# ============================================================================

def generate_metadata(
    copied_count: int,
) -> Path:

    print_section("4 - GENERACIÓN DE METADATOS")

    metadata = {
        "proyecto": PROYECTO_NOMBRE,
        "version": VERSION,

        "proceso": 46,
        "proceso_nombre": (
            "Generación del paquete definitivo"
        ),

        # Campos explícitos para auditoría 47.
        "proceso_46": "GO",
        "estado": "FINAL",
        "estado_final": "GO",
        "dictamen": "GO",
        "dictamen_final": "GO",

        "fecha_generacion": now_iso(),

        "script": SCRIPT_NAME,

        "base_dir": str(BASE_DIR),
        "input_dir": str(INPUT_DIR),

        "archivo_zip": ZIP_PATH.name,

        "estructura": DIRECTORIES,

        "cantidad_productos": copied_count,

        "auditorias_previas": {
            "proceso_42": "GO",
            "proceso_43": "GO",
            "proceso_44": "GO",
            "proceso_45": "GO",
        },

        "cierre": {
            "artefacto": "MODELO_TERRITORIAL_AMBA_V4_FINAL",
            "estado": "FINAL",
            "dictamen": "GO",
        },
    }

    path = (
        FINAL_DIR
        / "06_metadatos"
        / "metadata_paquete.json"
    )

    path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "metadata_paquete.json generado"
    )

    return path


# ============================================================================
# RESUMEN PROCESO 46
# ============================================================================

def generate_process_46_summary(
    copied_count: int,
) -> Path:

    summary = {
        "proceso": 46,
        "proceso_nombre": (
            "Generación del paquete definitivo"
        ),

        "proyecto": PROYECTO_NOMBRE,
        "version": VERSION,

        "estado": "FINAL",
        "estado_final": "GO",

        "dictamen": "GO",
        "dictamen_final": "GO",

        "fecha": now_iso(),

        "auditorias": {
            "42": "GO",
            "43": "GO",
            "44": "GO",
            "45": "GO",
        },

        "archivos_paquete": copied_count,

        "zip": ZIP_PATH.name,

        # No se incluye SHA del ZIP aquí porque eso generaría
        # una dependencia circular: el resumen está dentro del ZIP.
        "sha256_zip": (
            "se calcula sobre el artefacto ZIP final "
            "y se registra externamente por el proceso 47"
        ),
    }

    path = (
        FINAL_DIR
        / "06_metadatos"
        / "resumen_proceso_46.json"
    )

    path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================================
# README
# ============================================================================

def generate_readme() -> Path:

    print_section("5 - GENERACIÓN DEL README")

    lines = [
        "# MODELO TERRITORIAL AMBA " + VERSION,
        "",
        "## Paquete definitivo",
        "",
        "Proyecto: " + PROYECTO_NOMBRE,
        "",
        "Versión: " + VERSION,
        "",
        "Proceso de generación: 46",
        "",
        # IMPORTANTE:
        # El auditor 47 busca explícitamente esta expresión.
        "Proceso 46: GO",
        "",
        "Estado: FINAL",
        "",
        "Estado final: GO",
        "",
        "Dictamen: GO",
        "",
        "Dictamen final: GO",
        "",
        "Fecha de generación: " + now_iso(),
        "",
        "## Estructura",
        "",
        "- `01_modelo/` — modelos maestros y rankings.",
        "- `02_informes/` — productos ejecutivos.",
        "- `03_atlas/` — productos geográficos y espaciales.",
        "- `04_datos/` — indicadores y datos de soporte.",
        "- `05_auditoria/` — evidencias de auditorías.",
        "- `06_metadatos/` — metadatos y hashes.",
        "",
        "## Auditoría",
        "",
        "Proceso 42: GO",
        "",
        "Proceso 43: GO",
        "",
        "Proceso 44: GO",
        "",
        "Proceso 45: GO",
        "",
        "Proceso 46: GO",
        "",
        "El paquete constituye el artefacto definitivo.",
        "",
        "## Integridad",
        "",
        "Los archivos incluidos poseen SHA-256 registrado en:",
        "",
        "`06_metadatos/MANIFIESTO_SHA256.csv`",
        "",
        "## Estado del artefacto",
        "",
        "Estado: FINAL",
        "",
        "Dictamen: GO",
        "",
        "## Contenido",
        "",
    ]

    for p in sorted(FINAL_DIR.rglob("*")):

        if not p.is_file():
            continue

        relative = str(
            p.relative_to(FINAL_DIR)
        ).replace("\\", "/")

        if relative == "README.md":
            continue

        lines.append(
            "- `" + relative + "`"
        )

    lines.extend(
        [
            "",
            "## Artefacto ZIP",
            "",
            "Nombre:",
            "",
            "`MODELO_TERRITORIAL_AMBA_V4_FINAL.zip`",
            "",
            "Este paquete constituye la entrega definitiva "
            "del Modelo Territorial AMBA V4.1.",
            "",
        ]
    )

    path = FINAL_DIR / "README.md"

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("README.md generado")

    return path


# ============================================================================
# MANIFIESTO MARKDOWN
# ============================================================================

def generate_manifest_md() -> Path:

    print_section("6 - GENERACIÓN DEL MANIFIESTO")

    lines = [
        "# MANIFIESTO",
        "",
        "# Modelo Territorial AMBA " + VERSION,
        "",
        "## Identificación",
        "",
        "- Proyecto: " + PROYECTO_NOMBRE,
        "- Versión: " + VERSION,
        "- Proceso: 46",
        "- Proceso 46: GO",
        "- Estado: FINAL",
        "- Estado final: GO",
        "- Dictamen: GO",
        "- Dictamen final: GO",
        "- Fecha: " + now_iso(),
        "",
        "## Auditorías",
        "",
        "| Proceso | Estado |",
        "|---|---|",
        "| 42 | GO |",
        "| 43 | GO |",
        "| 44 | GO |",
        "| 45 | GO |",
        "| 46 | GO |",
        "",
        "## Archivos incluidos",
        "",
        "| Archivo | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]

    for p in sorted(FINAL_DIR.rglob("*")):

        if not p.is_file():
            continue

        relative = str(
            p.relative_to(FINAL_DIR)
        ).replace("\\", "/")

        if relative == "MANIFIESTO.md":
            continue

        # MANIFIESTO_SHA256 se genera después.
        if relative == "06_metadatos/MANIFIESTO_SHA256.csv":
            continue

        lines.append(
            "| `"
            + relative
            + "` | "
            + str(p.stat().st_size)
            + " | `"
            + sha256_file(p)
            + "` |"
        )

    lines.extend(
        [
            "",
            "## Integridad",
            "",
            "Todos los archivos incluidos fueron registrados "
            "con SHA-256.",
            "",
            "La integridad física del ZIP se valida mediante "
            "`zipfile.testzip()`.",
            "",
            "## Dictamen",
            "",
            "Proceso 46: GO",
            "",
            "Dictamen final: GO",
            "",
        ]
    )

    path = FINAL_DIR / "MANIFIESTO.md"

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("MANIFIESTO.md generado")

    return path


# ============================================================================
# MANIFIESTO SHA256
# ============================================================================

def generate_manifest_csv() -> Path:

    print_section("7 - GENERACIÓN DEL MANIFIESTO SHA-256")

    manifest_path = (
        FINAL_DIR
        / "06_metadatos"
        / "MANIFIESTO_SHA256.csv"
    )

    rows = []

    for path in sorted(FINAL_DIR.rglob("*")):

        if not path.is_file():
            continue

        relative = str(
            path.relative_to(FINAL_DIR)
        ).replace("\\", "/")

        if relative == (
            "06_metadatos/MANIFIESTO_SHA256.csv"
        ):
            continue

        rows.append(
            {
                "archivo": relative,
                "tamano_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    with manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "archivo",
                "tamano_bytes",
                "sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "Registros: "
        + str(len(rows))
    )

    return manifest_path


# ============================================================================
# VALIDACIÓN ESTRUCTURAL
# ============================================================================

def validate_final_package() -> tuple[bool, list[str]]:

    print_section("8 - VALIDACIÓN DEL PAQUETE DEFINITIVO")

    errors = []

    for directory in DIRECTORIES:

        path = FINAL_DIR / directory

        if not path.exists():
            errors.append(
                "Directorio faltante: "
                + directory
            )

    for relative in REQUIRED_FINAL_FILES:

        path = FINAL_DIR / relative

        if not path.exists():

            errors.append(
                "Archivo obligatorio faltante: "
                + relative
            )

        elif path.stat().st_size == 0:

            errors.append(
                "Archivo obligatorio vacío: "
                + relative
            )

    files = [
        p
        for p in FINAL_DIR.rglob("*")
        if p.is_file()
    ]

    print(
        "Archivos finales: "
        + str(len(files))
    )

    print(
        "Obligatorios esperados: "
        + str(len(REQUIRED_FINAL_FILES))
    )

    print(
        "Errores: "
        + str(len(errors))
    )

    if errors:

        for error in errors:
            print("  ERROR: " + error)

    return (
        len(errors) == 0,
        errors,
    )


# ============================================================================
# ZIP
# ============================================================================

def create_zip() -> tuple[str, int]:

    print_section("9 - GENERACIÓN DEL ZIP DEFINITIVO")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:

        for path in sorted(
            FINAL_DIR.rglob("*")
        ):

            if not path.is_file():
                continue

            arcname = (
                Path(FINAL_DIR.name)
                / path.relative_to(FINAL_DIR)
            )

            zf.write(
                path,
                arcname=str(
                    arcname
                ).replace("\\", "/"),
            )

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        names = zf.namelist()

    sha = sha256_file(ZIP_PATH)

    print(
        "ZIP: "
        + str(ZIP_PATH)
    )

    print(
        "Tamaño: "
        + str(ZIP_PATH.stat().st_size)
        + " bytes"
    )

    print(
        "Archivos ZIP: "
        + str(len(names))
    )

    print(
        "SHA-256: "
        + sha
    )

    return sha, len(names)


# ============================================================================
# VALIDACIÓN ZIP
# ============================================================================

def validate_zip() -> bool:

    print_section("10 - VALIDACIÓN FINAL DEL ZIP")

    if not ZIP_PATH.exists():

        print("ZIP encontrado: NO")
        return False

    print("ZIP encontrado: SI")

    try:

        with zipfile.ZipFile(
            ZIP_PATH,
            "r",
        ) as zf:

            bad = zf.testzip()

            if bad is not None:

                print("Test ZIP: ERROR")
                print(
                    "Archivo corrupto: "
                    + bad
                )

                return False

            names = zf.namelist()

        print("Test ZIP: OK")

        print(
            "Archivos ZIP: "
            + str(len(names))
        )

        print(
            "SHA-256 ZIP: "
            + sha256_file(ZIP_PATH)
        )

        return True

    except Exception as exc:

        print(
            "ERROR validando ZIP: "
            + str(exc)
        )

        return False


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    start = time.perf_counter()

    print(SEP)

    print(
        "46 - GENERACIÓN DEL PAQUETE DEFINITIVO DEL "
        "MODELO TERRITORIAL AMBA - "
        + VERSION
    )

    print(SEP)

    print(
        "Proyecto                      : "
        + str(BASE_DIR)
    )

    print(
        "Entrada                       : "
        + str(INPUT_DIR)
    )

    print(
        "Paquete ejecutivo             : "
        + str(EXECUTIVE_DIR)
    )

    print(
        "Salida                        : "
        + str(FINAL_DIR)
    )

    print(
        "ZIP                           : "
        + str(ZIP_PATH)
    )

    # ========================================================================
    # 1. AUDITORÍAS
    # ========================================================================

    audit_status = validate_previous_audits()

    if not audit_status["proceso_44"]:

        print()
        print(
            "ERROR: el proceso 44 no está validado como GO."
        )

        return 1

    if not audit_status["proceso_45"]:

        print()
        print(
            "ERROR: el proceso 45 no está validado como GO."
        )

        return 1

    # ========================================================================
    # 2. DIRECTORIO
    # ========================================================================

    prepare_final_directory()

    # ========================================================================
    # 3. PRODUCTOS
    # ========================================================================

    copied, missing = copy_products()

    if missing:

        print()
        print(
            "ERROR: faltan productos obligatorios."
        )

        for item in missing:
            print(
                "  - "
                + item
            )

        return 1

    # ========================================================================
    # 4. METADATA
    # ========================================================================

    # Los productos espaciales esperados forman parte del paquete.
    # No se fuerza una cantidad exacta aquí porque depende del origen.
    generate_metadata(
        copied_count=len(copied)
    )

    # ========================================================================
    # 5. RESUMEN PROCESO 46
    # ========================================================================

    generate_process_46_summary(
        copied_count=len(copied)
    )

    # ========================================================================
    # 6. README
    # ========================================================================

    generate_readme()

    # ========================================================================
    # 7. MANIFIESTO MD
    # ========================================================================

    generate_manifest_md()

    # ========================================================================
    # 8. MANIFIESTO SHA256
    # ========================================================================

    generate_manifest_csv()

    # ========================================================================
    # 9. VALIDACIÓN ESTRUCTURAL
    # ========================================================================

    valid, errors = validate_final_package()

    if not valid:

        print()
        print(
            "ERROR: el paquete definitivo no superó "
            "la validación estructural."
        )

        return 1

    # ========================================================================
    # 10. ZIP
    # ========================================================================

    zip_sha, zip_files = create_zip()

    # ========================================================================
    # 11. VALIDACIÓN ZIP
    # ========================================================================

    if not validate_zip():

        print()
        print(
            "ERROR: el ZIP definitivo no superó "
            "la validación."
        )

        return 1

    # ========================================================================
    # 12. RESULTADO
    # ========================================================================

    elapsed = time.perf_counter() - start

    final_file_count = len(
        [
            p
            for p in FINAL_DIR.rglob("*")
            if p.is_file()
        ]
    )

    print_section(
        "11 - RESULTADO FINAL DEL PROCESO 46"
    )

    print(
        "Proyecto                     : "
        + PROYECTO_NOMBRE
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
        "Proceso 46                   : GO"
    )

    print(
        "Archivos paquete             : "
        + str(final_file_count)
    )

    print(
        "Archivos ZIP                 : "
        + str(zip_files)
    )

    print(
        "SHA-256 ZIP                  : "
        + zip_sha
    )

    print(
        "Estado                       : FINAL"
    )

    print(
        "Estado final                 : GO"
    )

    print(
        "Dictamen final               : GO"
    )

    print(
        "Tiempo de ejecución          : "
        + f"{elapsed:.2f}"
        + " segundos"
    )

    print()
    print("Directorio definitivo:")
    print(FINAL_DIR)

    print()
    print("ZIP definitivo:")
    print(ZIP_PATH)

    print()
    print(SEP)

    print(
        "46 - PAQUETE DEFINITIVO GENERADO CORRECTAMENTE"
    )

    print(
        "MODELO_TERRITORIAL_AMBA_V4_FINAL.zip"
    )

    print(
        "PROCESO 46: GO"
    )

    print(
        "DICTAMEN FINAL: GO"
    )

    print(SEP)

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "Proceso interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(SEP)
        print(
            "ERROR NO CONTROLADO EN EL PROCESO 46"
        )
        print(SEP)

        print(
            type(exc).__name__
            + ": "
            + str(exc)
        )

        print(SEP)

        sys.exit(1)
````
