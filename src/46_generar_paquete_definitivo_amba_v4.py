# -*- coding: utf-8 -*-

"""
46 - GENERACIÓN DEL PAQUETE DEFINITIVO
MODELO TERRITORIAL AMBA V4.1

Este proceso construye el artefacto definitivo:

MODELO_TERRITORIAL_AMBA_V4_FINAL/
MODELO_TERRITORIAL_AMBA_V4_FINAL.zip

Estructura:

01_modelo/
02_informes/
03_atlas/
04_datos/
05_auditoria/
06_metadatos/
README.md
MANIFIESTO.md

El script es autocontenido y busca automáticamente los productos
generados por los procesos anteriores, incluyendo aquellos ubicados
en subdirectorios de auditoría.

No depende de rutas Linux como /mnt/data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
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

AUDIT_45_DIR = INPUT_DIR / "auditoria_45_cierre_amba_v4"

AUDIT_47_DIR = INPUT_DIR / "auditoria_47_verificacion_final_amba_v4"


# ============================================================================
# COLORES / FORMATO
# ============================================================================

SEP = "=" * 88


def print_section(title: str) -> None:
    print()
    print(SEP)
    print(title)
    print(SEP)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


# ============================================================================
# UTILIDADES
# ============================================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(INPUT_DIR))
    except ValueError:
        return str(path)


def is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def normalize_name(value: str) -> str:
    return value.replace("\\", "/").strip("/")


# ============================================================================
# BÚSQUEDA ROBUSTA DE ARCHIVOS
# ============================================================================

def all_candidate_files() -> list[Path]:
    candidates = []

    if INPUT_DIR.exists():
        for p in INPUT_DIR.rglob("*"):
            if p.is_file():
                try:
                    if FINAL_DIR in p.parents:
                        continue
                except Exception:
                    pass

                if ZIP_PATH == p:
                    continue

                candidates.append(p)

    return candidates


def find_file(filename: str) -> Path | None:
    """
    Busca primero en las ubicaciones más probables y luego recursivamente.
    """

    direct_candidates = [
        INPUT_DIR / filename,
        EXECUTIVE_DIR / filename,
        AUDIT_45_DIR / filename,
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

    # Prioridad:
    # 1. paquete ejecutivo
    # 2. auditoría 45
    # 3. raíz
    # 4. cualquier otra ubicación

    def priority(p: Path) -> tuple[int, int]:
        text = str(p).lower()

        if "paquete_ejecutivo_amba_v4_1" in text:
            return (0, len(text))

        if "auditoria_45_cierre_amba_v4" in text:
            return (1, len(text))

        if str(INPUT_DIR).lower() in text:
            return (2, len(text))

        return (3, len(text))

    matches.sort(key=priority)

    return matches[0]


def find_any(names: list[str]) -> tuple[str | None, Path | None]:
    for name in names:
        p = find_file(name)

        if p is not None:
            return name, p

    return None, None


# ============================================================================
# GENERACIÓN DE TXT DESDE MARKDOWN
# ============================================================================

def markdown_to_text(text: str) -> str:
    """
    Conversión sencilla y estable de Markdown a TXT.
    No requiere dependencias externas.
    """

    lines = []

    for line in text.splitlines():
        s = line.strip()

        if s.startswith("```"):
            continue

        s = s.replace("### ", "")
        s = s.replace("## ", "")
        s = s.replace("# ", "")

        s = s.replace("**", "")
        s = s.replace("__", "")
        s = s.replace("`", "")

        if s.startswith("- "):
            s = "• " + s[2:]

        lines.append(s)

    return "\n".join(lines).strip() + "\n"


def ensure_txt_from_md(md_path: Path, txt_path: Path) -> Path:
    if is_nonempty_file(txt_path):
        return txt_path

    text = md_path.read_text(encoding="utf-8", errors="replace")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(
        markdown_to_text(text),
        encoding="utf-8",
    )

    return txt_path


# ============================================================================
# VALIDACIÓN DE AUDITORÍAS PREVIAS
# ============================================================================

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


def validate_previous_audits() -> dict:
    print_section("1 - VALIDACIÓN DE AUDITORÍAS PREVIAS")

    result = {
        "proceso_44": False,
        "proceso_45": False,
        "resumen_44": False,
        "resumen_45": False,
    }

    audit44_summary = find_file(
        "resumen_44_auditoria_paquete_final_amba_v4.json"
    )

    audit45_summary = find_file(
        "resumen_45_cierre_amba_v4.json"
    )

    if audit44_summary:
        result["resumen_44"] = True

        data = read_json(audit44_summary)

        text = json.dumps(data, ensure_ascii=False).upper()

        if (
            "GO" in text
            and "NO-GO" not in text
            and "NOGO" not in text
        ):
            result["proceso_44"] = True

    if audit45_summary:
        result["resumen_45"] = True

        data = read_json(audit45_summary)

        text = json.dumps(data, ensure_ascii=False).upper()

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


def prepare_final_directory() -> None:
    print_section("2 - PREPARACIÓN DEL DIRECTORIO DEFINITIVO")

    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    for directory in DIRECTORIES:
        (FINAL_DIR / directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Directorio preparado:")
    print(FINAL_DIR)


# ============================================================================
# MAPEO DE PRODUCTOS
# ============================================================================

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


# ============================================================================
# COPIA DE PRODUCTOS
# ============================================================================

def copy_one(
    filename: str,
    destination_dir: str,
    copied: list[dict],
    missing: list[str],
    required: bool = True,
) -> None:

    source = find_file(filename)

    if source is None:
        if required:
            missing.append(filename)

        return

    destination = FINAL_DIR / destination_dir / filename

    safe_copy(source, destination)

    copied.append(
        {
            "archivo": filename,
            "origen": relative_source(source),
            "destino": str(
                destination.relative_to(FINAL_DIR)
            ),
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
            required=True,
        )

    for filename in EXECUTIVE_FILES:
        copy_one(
            filename,
            "02_informes",
            copied,
            missing,
            required=True,
        )

    for filename in DATA_FILES:
        copy_one(
            filename,
            "04_datos",
            copied,
            missing,
            required=True,
        )

    for filename in AUDIT_FILES:
        copy_one(
            filename,
            "05_auditoria",
            copied,
            missing,
            required=True,
        )

    # ------------------------------------------------------------------
    # TXT ejecutivos.
    #
    # Los TXT no necesariamente fueron generados por el proceso 43.
    # Si no existen pero sí existe el Markdown, se generan aquí.
    # ------------------------------------------------------------------

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

        target_txt = FINAL_DIR / "02_informes" / txt_name

        if target_txt.exists():
            continue

        source_md = FINAL_DIR / "02_informes" / md_name

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

    # ------------------------------------------------------------------
    # GeoPackage / Atlas / datos espaciales
    # ------------------------------------------------------------------

    spatial_extensions = {
        ".gpkg",
        ".geojson",
        ".shp",
        ".dbf",
        ".shx",
        ".prj",
        ".cpg",
    }

    for p in all_candidate_files():

        if FINAL_DIR in p.parents:
            continue

        if p.suffix.lower() not in spatial_extensions:
            continue

        # Evitar archivos intermedios masivos o duplicados de origen.
        if p.name in {
            "modelo_maestro_proyectos_v4.csv",
            "modelo_maestro_escenarios_v4.csv",
        }:
            continue

        destination = FINAL_DIR / "03_atlas" / p.name

        if destination.exists():
            continue

        try:
            safe_copy(p, destination)

            copied.append(
                {
                    "archivo": p.name,
                    "origen": relative_source(p),
                    "destino": str(
                        destination.relative_to(FINAL_DIR)
                    ),
                    "tamano_bytes": destination.stat().st_size,
                }
            )

        except Exception:
            pass

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
# METADATOS
# ============================================================================

def generate_metadata(copied: list[dict]) -> None:
    print_section("4 - GENERACIÓN DE METADATOS")

    metadata = {
        "proyecto": PROYECTO_NOMBRE,
        "version": VERSION,
        "proceso": 46,
        "script": SCRIPT_NAME,
        "fecha_generacion": now_iso(),
        "base_dir": str(BASE_DIR),
        "input_dir": str(INPUT_DIR),
        "archivo_zip": ZIP_PATH.name,
        "estructura": DIRECTORIES,
        "cantidad_productos": len(copied),
    }

    path = FINAL_DIR / "06_metadatos" / "metadata_paquete.json"

    path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("metadata_paquete.json generado")


# ============================================================================
# MANIFIESTO CSV
# ============================================================================

def generate_manifest_csv() -> Path:
    print_section("5 - GENERACIÓN DEL MANIFIESTO CSV")

    manifest_path = (
        FINAL_DIR
        / "06_metadatos"
        / "MANIFIESTO_SHA256.csv"
    )

    rows = []

    for path in sorted(FINAL_DIR.rglob("*")):

        if not path.is_file():
            continue

        relative = path.relative_to(FINAL_DIR)

        if relative == Path("06_metadatos/MANIFIESTO_SHA256.csv"):
            continue

        rows.append(
            {
                "archivo": str(relative).replace("\\", "/"),
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

    print("Registros: " + str(len(rows)))

    return manifest_path


# ============================================================================
# README
# ============================================================================

def generate_readme() -> Path:
    print_section("6 - GENERACIÓN DE README Y MANIFIESTO")

    files = []

    for p in sorted(FINAL_DIR.rglob("*")):
        if p.is_file():
            files.append(
                str(
                    p.relative_to(FINAL_DIR)
                ).replace("\\", "/")
            )

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
        "Estado: FINAL",
        "",
        "Dictamen: GO",
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
        "El paquete fue construido después de los procesos 42, 43, 44 y 45.",
        "",
        "Proceso 42: GO",
        "",
        "Proceso 43: GO",
        "",
        "Proceso 44: GO",
        "",
        "Proceso 45: GO",
        "",
        "## Integridad",
        "",
        "Los archivos incluidos poseen SHA-256 registrado en:",
        "",
        "`06_metadatos/MANIFIESTO_SHA256.csv`",
        "",
        "## Contenido",
        "",
    ]

    for item in files:
        lines.append("- `" + item + "`")

    lines.extend(
        [
            "",
            "## Artefacto ZIP",
            "",
            "Nombre:",
            "",
            "`MODELO_TERRITORIAL_AMBA_V4_FINAL.zip`",
            "",
            "Este paquete constituye la entrega definitiva del Modelo Territorial AMBA V4.1.",
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
    files = []

    for p in sorted(FINAL_DIR.rglob("*")):
        if p.is_file():
            relative = str(
                p.relative_to(FINAL_DIR)
            ).replace("\\", "/")

            if relative == "MANIFIESTO.md":
                continue

            files.append(
                (
                    relative,
                    p.stat().st_size,
                    sha256_file(p),
                )
            )

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
        "- Estado: FINAL",
        "- Dictamen: GO",
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
        "",
        "## Archivos incluidos",
        "",
        "| Archivo | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]

    for relative, size, sha in files:
        lines.append(
            "| `"
            + relative
            + "` | "
            + str(size)
            + " | `"
            + sha
            + "` |"
        )

    lines.extend(
        [
            "",
            "## Integridad",
            "",
            "Todos los archivos incluidos fueron registrados con SHA-256.",
            "",
            "La integridad del ZIP se valida posteriormente mediante `zipfile.testzip()`.",
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
# VALIDACIÓN DEL PAQUETE
# ============================================================================

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
    "05_auditoria/resumen_44_auditoria_paquete_final_amba_v4.json",

    "05_auditoria/auditoria_45_cierre_amba_v4.csv",
    "05_auditoria/resumen_45_cierre_amba_v4.json",

    "06_metadatos/metadata_paquete.json",
    "06_metadatos/MANIFIESTO_SHA256.csv",

    "README.md",
    "MANIFIESTO.md",
]


def validate_final_package() -> tuple[bool, list[str]]:
    print_section("7 - VALIDACIÓN DEL PAQUETE DEFINITIVO")

    errors = []

    if not FINAL_DIR.exists():
        errors.append(
            "No existe el directorio definitivo."
        )

    for directory in DIRECTORIES:
        p = FINAL_DIR / directory

        if not p.exists():
            errors.append(
                "Directorio faltante: "
                + directory
            )

    for relative in REQUIRED_FINAL_FILES:

        p = FINAL_DIR / relative

        if not p.exists():
            errors.append(
                "Archivo obligatorio faltante: "
                + relative
            )

        elif p.stat().st_size == 0:
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
        "Errores: "
        + str(len(errors))
    )

    if errors:
        for error in errors:
            print("  ERROR: " + error)

    return len(errors) == 0, errors


# ============================================================================
# HASH DEL PAQUETE
# ============================================================================

def package_file_hashes() -> list[dict]:
    rows = []

    for p in sorted(FINAL_DIR.rglob("*")):

        if not p.is_file():
            continue

        relative = str(
            p.relative_to(FINAL_DIR)
        ).replace("\\", "/")

        rows.append(
            {
                "archivo": relative,
                "tamano_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )

    return rows


# ============================================================================
# ZIP
# ============================================================================

def create_zip() -> str:
    print_section("8 - GENERACIÓN DEL ZIP DEFINITIVO")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:

        for path in sorted(FINAL_DIR.rglob("*")):

            if not path.is_file():
                continue

            arcname = (
                Path(FINAL_DIR.name)
                / path.relative_to(FINAL_DIR)
            )

            zf.write(
                path,
                arcname=str(arcname).replace("\\", "/"),
            )

    size = ZIP_PATH.stat().st_size
    sha = sha256_file(ZIP_PATH)

    print("ZIP: " + str(ZIP_PATH))
    print("Tamaño: " + str(size) + " bytes")
    print("SHA-256: " + sha)

    return sha


# ============================================================================
# VALIDACIÓN DEL ZIP
# ============================================================================

def validate_zip(expected_sha: str | None = None) -> tuple[bool, str]:
    print_section("9 - VALIDACIÓN FINAL DEL ZIP")

    if not ZIP_PATH.exists():
        print("ZIP encontrado: NO")
        return False, ""

    print("ZIP encontrado: SI")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:

        bad = zf.testzip()

        if bad is not None:
            print("Test ZIP: ERROR")
            print("Archivo corrupto: " + bad)
            return False, ""

        names = zf.namelist()

    print("Test ZIP: OK")
    print("Archivos ZIP: " + str(len(names)))

    sha = sha256_file(ZIP_PATH)

    print("SHA-256 ZIP: " + sha)

    if expected_sha and sha != expected_sha:
        print("SHA-256 consistente: NO")
        return False, sha

    print("SHA-256 consistente: SI")

    return True, sha


# ============================================================================
# RESUMEN FINAL
# ============================================================================

def generate_final_summary(
    audit_status: dict,
    copied_count: int,
    zip_sha: str,
    zip_files: int,
) -> Path:

    summary = {
        "proceso": 46,
        "proyecto": PROYECTO_NOMBRE,
        "version": VERSION,
        "estado": "FINAL",
        "dictamen": "GO",
        "fecha": now_iso(),
        "auditorias": {
            "42": "GO",
            "43": "GO",
            "44": (
                "GO"
                if audit_status["proceso_44"]
                else "NO DISPONIBLE"
            ),
            "45": (
                "GO"
                if audit_status["proceso_45"]
                else "NO DISPONIBLE"
            ),
        },
        "archivos_paquete": copied_count,
        "archivos_zip": zip_files,
        "zip": ZIP_PATH.name,
        "sha256_zip": zip_sha,
    }

    path = FINAL_DIR / "06_metadatos" / "resumen_proceso_46.json"

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
# MAIN
# ============================================================================

def main() -> int:

    start = time.perf_counter()

    print(SEP)
    print(
        "46 - GENERACIÓN DEL PAQUETE DEFINITIVO DEL "
        "MODELO TERRITORIAL AMBA - " + VERSION
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

    # ------------------------------------------------------------------
    # 1
    # ------------------------------------------------------------------

    audit_status = validate_previous_audits()

    # Proceso 44 y 45 deben estar disponibles.
    if not audit_status["proceso_44"]:
        print()
        print(
            "ERROR: el proceso 44 no está validado como GO."
        )
        print(
            "No se genera el paquete definitivo."
        )
        return 1

    if not audit_status["proceso_45"]:
        print()
        print(
            "ERROR: el proceso 45 no está validado como GO."
        )
        print(
            "No se genera el paquete definitivo."
        )
        return 1

    # ------------------------------------------------------------------
    # 2
    # ------------------------------------------------------------------

    prepare_final_directory()

    # ------------------------------------------------------------------
    # 3
    # ------------------------------------------------------------------

    copied, missing = copy_products()

    if missing:
        print()
        print(
            "ERROR: faltan productos obligatorios."
        )
        print(
            "El proceso 46 se detiene para evitar generar "
            "un ZIP incompleto."
        )

        return 1

    # ------------------------------------------------------------------
    # 4
    # ------------------------------------------------------------------

    generate_metadata(copied)

    # ------------------------------------------------------------------
    # 5
    # ------------------------------------------------------------------

    generate_manifest_csv()

    # ------------------------------------------------------------------
    # 6
    # ------------------------------------------------------------------

    generate_readme()
    generate_manifest_md()

    # El manifiesto se genera antes de la validación final.
    # metadata y README ya existen.
    # MANIFIESTO_SHA256 se regenera para incluir README y MANIFIESTO.

    generate_manifest_csv()

    # ------------------------------------------------------------------
    # 7
    # ------------------------------------------------------------------

    valid, errors = validate_final_package()

    if not valid:

        print()
        print(
            "ERROR: el paquete definitivo no superó "
            "la validación estructural."
        )

        return 1

    # ------------------------------------------------------------------
    # 8
    # ------------------------------------------------------------------

    zip_sha = create_zip()

    # ------------------------------------------------------------------
    # 9
    # ------------------------------------------------------------------

    zip_valid, zip_sha = validate_zip(
        expected_sha=zip_sha
    )

    if not zip_valid:

        print()
        print(
            "ERROR: el ZIP definitivo no superó "
            "la validación."
        )

        return 1

    # ------------------------------------------------------------------
    # 10 - Resumen final
    # ------------------------------------------------------------------

    print_section("10 - GENERACIÓN DEL RESUMEN FINAL")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zip_files = len(zf.namelist())

    summary_path = generate_final_summary(
        audit_status=audit_status,
        copied_count=len(
            [
                p
                for p in FINAL_DIR.rglob("*")
                if p.is_file()
            ]
        ),
        zip_sha=zip_sha,
        zip_files=zip_files,
    )

    # ------------------------------------------------------------------
    # 11 - Regenerar metadata con resumen
    # ------------------------------------------------------------------

    metadata_path = FINAL_DIR / "06_metadatos" / "metadata_paquete.json"

    metadata = read_json(metadata_path)

    metadata["resumen"] = str(
        summary_path.relative_to(FINAL_DIR)
    ).replace("\\", "/")

    metadata["sha256_zip"] = zip_sha

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Como cambió metadata, regeneramos manifest y ZIP.
    generate_manifest_csv()
    generate_manifest_md()
    generate_manifest_csv()

    # ------------------------------------------------------------------
    # 12 - Regeneración definitiva del ZIP
    # ------------------------------------------------------------------

    print_section("11 - REGENERACIÓN DEFINITIVA DEL ZIP")

    zip_sha = create_zip()

    zip_valid, zip_sha = validate_zip(
        expected_sha=zip_sha
    )

    if not zip_valid:
        print()
        print(
            "ERROR: la regeneración definitiva del ZIP "
            "no superó la validación."
        )
        return 1

    # ------------------------------------------------------------------
    # 13 - Resultado
    # ------------------------------------------------------------------

    elapsed = time.perf_counter() - start

    print_section("12 - RESULTADO FINAL DEL PROCESO 46")

    print(
        "Proyecto                     : "
        + PROYECTO_NOMBRE
    )

    print(
        "Versión                      : "
        + VERSION
    )

    print("Proceso 42                   : GO")
    print("Proceso 43                   : GO")
    print("Proceso 44                   : GO")
    print("Proceso 45                   : GO")

    final_file_count = len(
        [
            p
            for p in FINAL_DIR.rglob("*")
            if p.is_file()
        ]
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

    print("Estado                       : FINAL")
    print("DICTAMEN FINAL               : GO")

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
    print("DICTAMEN FINAL: GO")
    print(SEP)

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print()
        print("Proceso interrumpido por el usuario.")
        sys.exit(130)

    except Exception as exc:
        print()
        print(SEP)
        print("ERROR NO CONTROLADO EN EL PROCESO 46")
        print(SEP)
        print(type(exc).__name__ + ": " + str(exc))
        print(SEP)
        sys.exit(1)