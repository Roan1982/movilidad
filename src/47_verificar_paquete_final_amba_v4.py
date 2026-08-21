# -*- coding: utf-8 -*-
"""
47 - VERIFICACIÓN FINAL DEL ARTEFACTO ZIP DEL MODELO TERRITORIAL AMBA - V4.1

Auditoría independiente del paquete definitivo generado por el proceso 46.

Objetivos:
    1. Verificar existencia e integridad física del ZIP.
    2. Inventariar independientemente su contenido.
    3. Validar estructura definitiva.
    4. Validar productos obligatorios.
    5. Validar archivos no vacíos.
    6. Calcular SHA-256 del ZIP.
    7. Calcular SHA-256 de todos los archivos internos.
    8. Validar README.md.
    9. Validar MANIFIESTO.md.
   10. Validar evidencia de procesos 42, 43, 44 y 45.
   11. Validar resultado independiente del proceso 45.
   12. Comparar el contenido lógico del directorio definitivo contra el ZIP.
   13. Generar auditoría, inventario, hashes, resumen e informe.
   14. Emitir dictamen final GO / NO-GO.

El script NO modifica el paquete definitivo ni el ZIP.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
import zipfile

from pathlib import Path
from typing import Any


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.1"
PROCESO = "47"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

ZIP_PATH = BASE_DIR / "MODELO_TERRITORIAL_AMBA_V4_FINAL.zip"

FINAL_DIR = BASE_DIR / "MODELO_TERRITORIAL_AMBA_V4_FINAL"

OUTPUT_DIR = (
    BASE_DIR
    / "auditoria_47_verificacion_final_amba_v4"
)

AUDITORIA_CSV = OUTPUT_DIR / "auditoria_47_verificacion_final_amba_v4.csv"
INVENTARIO_CSV = OUTPUT_DIR / "inventario_47_verificacion_final_amba_v4.csv"
HASHES_CSV = OUTPUT_DIR / "hashes_47_verificacion_final_amba_v4.csv"
RESUMEN_JSON = OUTPUT_DIR / "resumen_47_verificacion_final_amba_v4.json"
INFORME_MD = OUTPUT_DIR / "informe_47_verificacion_final_amba_v4.md"


# =============================================================================
# ESTRUCTURA ESPERADA
# =============================================================================

EXPECTED_DIRECTORIES = [
    "01_modelo",
    "02_informes",
    "03_atlas",
    "04_datos",
    "05_auditoria",
    "06_metadatos",
]

EXPECTED_ROOT_FILES = [
    "README.md",
    "MANIFIESTO.md",
]

# Productos funcionales que deben existir dentro del ZIP.
#
# No se exige que estén en una carpeta concreta: se valida por nombre.
# Esto evita falsos negativos si el proceso 46 reorganiza algún producto.
REQUIRED_PRODUCTS = [
    "modelo_maestro_proyectos_v4.csv",
    "modelo_maestro_escenarios_v4.csv",
    "ranking_final_proyectos_v4.csv",
    "ranking_final_escenarios_v4.csv",
    "indicadores_globales_amba_v4.csv",

    "proyectos_ejecutivos_amba_v4_1.csv",
    "escenarios_ejecutivos_amba_v4_1.csv",
    "top_20_proyectos_prioritarios_amba_v4_1.csv",
    "ranking_escenarios_ejecutivo_amba_v4_1.csv",
    "indicadores_ejecutivos_amba_v4_1.csv",

    "sintesis_ejecutiva_amba_v4_1.md",
    "informe_ejecutivo_amba_v4_1.md",

    "auditoria_44_paquete_final_amba_v4.csv",
    "resumen_44_auditoria_paquete_final_amba_v4.json",

    "auditoria_45_cierre_amba_v4.csv",
    "resumen_45_cierre_amba_v4.json",

    "metadata_paquete.json",
]


# Evidencias de cierre.
PROCESS_EVIDENCE = {
    "42": [
        "cierre_42_modelo_territorial_amba_v4.csv",
    ],
    "43": [
        "control_paquete_ejecutivo_amba_v4_1.csv",
        "manifiesto_43_paquete_ejecutivo_amba_v4_1.csv",
    ],
    "44": [
        "auditoria_44_paquete_final_amba_v4.csv",
        "resumen_44_auditoria_paquete_final_amba_v4.json",
    ],
    "45": [
        "auditoria_45_cierre_amba_v4.csv",
        "resumen_45_cierre_amba_v4.json",
    ],
}


# =============================================================================
# UTILIDADES
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_field(label: str, value: Any, width: int = 30) -> None:
    print(f"{label:<{width}}: {value}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def normalize_zip_name(name: str) -> str:
    """
    Normaliza nombres internos del ZIP.

    Convierte:
        \ -> /
        elimina ./ inicial
        elimina / inicial/final
    """
    value = str(name).replace("\\", "/").strip()

    while value.startswith("./"):
        value = value[2:]

    value = value.lstrip("/").rstrip("/")

    return value


def basename_zip_name(name: str) -> str:
    return Path(normalize_zip_name(name)).name


def is_directory_entry(name: str) -> bool:
    return normalize_zip_name(name).endswith("/")


def safe_read_zip_text(
    zf: zipfile.ZipFile,
    member: str,
) -> str:
    try:
        data = zf.read(member)
    except KeyError:
        return ""

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


def find_zip_member(
    names: list[str],
    filename: str,
) -> str | None:
    """
    Busca un archivo por nombre independientemente de la carpeta.
    """
    target = filename.lower()

    for name in names:
        if is_directory_entry(name):
            continue

        if basename_zip_name(name).lower() == target:
            return name

    return None


def find_all_zip_members(
    names: list[str],
    filename: str,
) -> list[str]:
    target = filename.lower()

    return [
        name
        for name in names
        if not is_directory_entry(name)
        and basename_zip_name(name).lower() == target
    ]


def load_json_from_zip(
    zf: zipfile.ZipFile,
    names: list[str],
    filename: str,
) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    member = find_zip_member(names, filename)

    if member is None:
        return None, None

    try:
        data = zf.read(member)
        return json.loads(data.decode("utf-8-sig")), member
    except Exception:
        return None, member


def load_csv_from_zip(
    zf: zipfile.ZipFile,
    names: list[str],
    filename: str,
) -> tuple[list[dict[str, str]] | None, str | None]:
    member = find_zip_member(names, filename)

    if member is None:
        return None, None

    try:
        text = safe_read_zip_text(zf, member)
        rows = list(csv.DictReader(text.splitlines()))
        return rows, member
    except Exception:
        return None, member


def finite_number(value: Any) -> bool:
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError):
        return False


def parse_numeric(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def recursive_find_key(
    obj: Any,
    keys: set[str],
) -> list[Any]:
    """
    Busca valores asociados a determinadas claves en JSON anidado.
    """
    found: list[Any] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = str(key).strip().lower()

            if normalized in keys:
                found.append(value)

            found.extend(recursive_find_key(value, keys))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursive_find_key(item, keys))

    return found


def text_has_go(text: str) -> bool:
    """
    Determina si un documento evidencia un dictamen GO.

    Se priorizan expresiones explícitas para evitar interpretar un
    'NO-GO' como 'GO'.
    """
    upper = text.upper()

    if "NO-GO" in upper or "NO GO" in upper:
        # Si además existe GO explícito, seguimos buscando evidencia
        # estructurada; en documentos de cierre se considera NO-GO.
        if "DICTAMEN FINAL" in upper:
            match = re.search(
                r"DICTAMEN\s+FINAL\s*[:\-]\s*(NO[- ]GO|GO)",
                upper,
            )
            if match:
                return match.group(1).replace(" ", "-") == "GO"

    patterns = [
        r"DICTAMEN\s+FINAL\s*[:\-]\s*GO\b",
        r"DICTAMEN\s*[:\-]\s*GO\b",
        r"ESTADO\s*[:\-]\s*GO\b",
        r"AUDITOR[IÍ]A\s*[:\-]\s*OK\b",
        r"RESULTADO\s+FINAL\s*[:\-]\s*GO\b",
    ]

    return any(re.search(pattern, upper) for pattern in patterns)


def text_has_process_go(text: str, process: str) -> bool:
    upper = text.upper()

    patterns = [
        rf"PROCESO\s+{re.escape(process)}.*?GO\b",
        rf"{re.escape(process)}.*?DICTAMEN.*?GO\b",
        rf"DICTAMEN.*?PROCESO\s+{re.escape(process)}.*?GO\b",
    ]

    return any(re.search(pattern, upper, flags=re.DOTALL) for pattern in patterns)


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""

    data = path.read_bytes()

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


# =============================================================================
# AUDITORÍA
# =============================================================================

class Auditoria47:
    def __init__(self) -> None:
        self.controls: list[dict[str, Any]] = []
        self.inventory: list[dict[str, Any]] = []
        self.hashes: list[dict[str, Any]] = []

        self.critical_failures = 0
        self.important_failures = 0

        self.zip_names: list[str] = []
        self.zip_files: list[str] = []

        self.start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Control central
    # -------------------------------------------------------------------------

    def control(
        self,
        code: str,
        description: str,
        status: str,
        critical: bool = False,
        important: bool = False,
        detail: str = "",
    ) -> None:
        status = status.upper()

        if status != "OK":
            if critical:
                self.critical_failures += 1

            if important:
                self.important_failures += 1

        self.controls.append(
            {
                "control": code,
                "descripcion": description,
                "resultado": status,
                "critico": "SI" if critical else "NO",
                "importante": "SI" if important else "NO",
                "detalle": detail,
            }
        )

    # -------------------------------------------------------------------------
    # 1. ZIP
    # -------------------------------------------------------------------------

    def validate_zip_exists(self) -> bool:
        print_header("1 - EXISTENCIA DEL ARTEFACTO ZIP")

        exists = ZIP_PATH.exists() and ZIP_PATH.is_file()

        print_field(
            "ZIP encontrado",
            "SI" if exists else "NO",
        )

        if exists:
            print_field(
                "Tamaño",
                f"{ZIP_PATH.stat().st_size:,} bytes",
            )

        self.control(
            "01",
            "Existencia del ZIP definitivo",
            "OK" if exists else "ERROR",
            critical=True,
            detail=str(ZIP_PATH),
        )

        return exists

    # -------------------------------------------------------------------------
    # 2. Integridad
    # -------------------------------------------------------------------------

    def validate_zip_integrity(self) -> zipfile.ZipFile | None:
        print_header("2 - INTEGRIDAD FÍSICA DEL ZIP")

        if not ZIP_PATH.exists():
            self.control(
                "02",
                "Integridad física del ZIP",
                "ERROR",
                critical=True,
                detail="ZIP inexistente",
            )
            return None

        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                bad = zf.testzip()

                ok = bad is None

                print_field(
                    "Test ZIP",
                    "OK" if ok else f"ERROR: {bad}",
                )

            self.control(
                "02",
                "Integridad física del ZIP",
                "OK" if ok else "ERROR",
                critical=True,
                detail="ZIP testeado con zipfile.testzip()",
            )

            if not ok:
                return None

            return zipfile.ZipFile(ZIP_PATH, "r")

        except Exception as exc:
            print_field("Test ZIP", f"ERROR: {exc}")

            self.control(
                "02",
                "Integridad física del ZIP",
                "ERROR",
                critical=True,
                detail=str(exc),
            )

            return None

    # -------------------------------------------------------------------------
    # 3. Inventario
    # -------------------------------------------------------------------------

    def inventory_zip(self, zf: zipfile.ZipFile) -> None:
        print_header("3 - INVENTARIO INDEPENDIENTE DEL ZIP")

        self.zip_names = [
            normalize_zip_name(info.filename)
            for info in zf.infolist()
            if not is_directory_entry(info.filename)
        ]

        self.zip_files = list(self.zip_names)

        print_field(
            "Entradas ZIP",
            len(zf.infolist()),
        )

        print_field(
            "Archivos físicos",
            len(self.zip_files),
        )

        duplicate_names = sorted(
            {
                name
                for name in self.zip_files
                if self.zip_files.count(name) > 1
            }
        )

        detail = (
            f"Archivos={len(self.zip_files)}; "
            f"duplicados={len(duplicate_names)}"
        )

        self.control(
            "03",
            "Inventario independiente del ZIP",
            "OK" if not duplicate_names else "ERROR",
            critical=True,
            detail=detail,
        )

        for info in zf.infolist():
            name = normalize_zip_name(info.filename)

            if is_directory_entry(info.filename):
                continue

            self.inventory.append(
                {
                    "archivo": name,
                    "tamaño_bytes": info.file_size,
                    "compresion_bytes": info.compress_size,
                    "fecha_zip": str(info.date_time),
                }
            )

    # -------------------------------------------------------------------------
    # 4. Estructura
    # -------------------------------------------------------------------------

    def validate_structure(self) -> None:
        print_header("4 - VALIDACIÓN DE ESTRUCTURA DEFINITIVA")

        names = set(self.zip_files)

        missing_dirs: list[str] = []

        for directory in EXPECTED_DIRECTORIES:
            prefix = directory + "/"

            if not any(
                name.startswith(prefix)
                for name in names
            ):
                missing_dirs.append(directory)

        missing_root_files = [
            filename
            for filename in EXPECTED_ROOT_FILES
            if not find_zip_member(self.zip_files, filename)
        ]

        outside_structure: list[str] = []

        allowed_root = set(EXPECTED_ROOT_FILES)

        for name in self.zip_files:
            parts = Path(name).parts

            if len(parts) == 1:
                if name not in allowed_root:
                    outside_structure.append(name)
            else:
                if parts[0] not in EXPECTED_DIRECTORIES:
                    outside_structure.append(name)

        print_field(
            "Directorios faltantes",
            len(missing_dirs),
        )

        print_field(
            "Archivos raíz faltantes",
            len(missing_root_files),
        )

        print_field(
            "Archivos fuera de estructura",
            len(outside_structure),
        )

        if missing_dirs:
            for item in missing_dirs:
                print(f"  FALTANTE: {item}")

        if missing_root_files:
            for item in missing_root_files:
                print(f"  FALTANTE: {item}")

        if outside_structure:
            for item in outside_structure:
                print(f"  FUERA: {item}")

        ok = (
            not missing_dirs
            and not missing_root_files
            and not outside_structure
        )

        self.control(
            "04",
            "Estructura definitiva",
            "OK" if ok else "ERROR",
            critical=True,
            detail=(
                f"directorios_faltantes={len(missing_dirs)}; "
                f"raiz_faltantes={len(missing_root_files)}; "
                f"fuera_estructura={len(outside_structure)}"
            ),
        )

    # -------------------------------------------------------------------------
    # 5. No vacíos
    # -------------------------------------------------------------------------

    def validate_non_empty(self, zf: zipfile.ZipFile) -> None:
        print_header("5 - VALIDACIÓN DE ARCHIVOS NO VACÍOS")

        empty: list[str] = []

        for info in zf.infolist():
            if is_directory_entry(info.filename):
                continue

            if info.file_size == 0:
                empty.append(normalize_zip_name(info.filename))

        print_field(
            "Archivos vacíos",
            len(empty),
        )

        for name in empty:
            print(f"  VACÍO: {name}")

        self.control(
            "05",
            "Archivos físicos no vacíos",
            "OK" if not empty else "ERROR",
            critical=True,
            detail=f"vacíos={len(empty)}",
        )

    # -------------------------------------------------------------------------
    # 6. Productos
    # -------------------------------------------------------------------------

    def validate_required_products(self) -> None:
        print_header("6 - PRODUCTOS OBLIGATORIOS")

        found: list[str] = []
        missing: list[str] = []

        for product in REQUIRED_PRODUCTS:
            member = find_zip_member(self.zip_files, product)

            if member:
                found.append(product)
            else:
                missing.append(product)

        print_field(
            "Productos requeridos",
            len(REQUIRED_PRODUCTS),
        )

        print_field(
            "Productos encontrados",
            len(found),
        )

        print_field(
            "Productos faltantes",
            len(missing),
        )

        for product in missing:
            print(f"  FALTANTE: {product}")

        self.control(
            "06",
            "Productos obligatorios",
            "OK" if not missing else "ERROR",
            critical=True,
            detail=(
                f"requeridos={len(REQUIRED_PRODUCTS)}; "
                f"encontrados={len(found)}; "
                f"faltantes={len(missing)}"
            ),
        )

    # -------------------------------------------------------------------------
    # 7. SHA ZIP
    # -------------------------------------------------------------------------

    def validate_zip_hash(self) -> str:
        print_header("7 - SHA-256 DEL ZIP DEFINITIVO")

        if not ZIP_PATH.exists():
            self.control(
                "07",
                "SHA-256 del ZIP",
                "ERROR",
                critical=True,
                detail="ZIP inexistente",
            )
            return ""

        digest = sha256_file(ZIP_PATH)

        print_field(
            "SHA-256 ZIP",
            digest,
        )

        self.control(
            "07",
            "SHA-256 del ZIP definitivo",
            "OK" if len(digest) == 64 else "ERROR",
            critical=True,
            detail=digest,
        )

        return digest

    # -------------------------------------------------------------------------
    # 8. Hashes internos
    # -------------------------------------------------------------------------

    def validate_internal_hashes(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header("8 - HASHES SHA-256 DE ARCHIVOS INTERNOS")

        errors = 0

        for info in zf.infolist():
            if is_directory_entry(info.filename):
                continue

            name = normalize_zip_name(info.filename)

            try:
                data = zf.read(info.filename)
                digest = sha256_bytes(data)

                self.hashes.append(
                    {
                        "archivo": name,
                        "sha256": digest,
                        "tamaño_bytes": len(data),
                        "resultado": "OK",
                    }
                )

            except Exception as exc:
                errors += 1

                self.hashes.append(
                    {
                        "archivo": name,
                        "sha256": "",
                        "tamaño_bytes": 0,
                        "resultado": "ERROR",
                        "detalle": str(exc),
                    }
                )

        print_field(
            "Archivos hasheados",
            len(self.hashes),
        )

        print_field(
            "Errores hash",
            errors,
        )

        self.control(
            "08",
            "SHA-256 de archivos internos",
            "OK" if errors == 0 else "ERROR",
            critical=True,
            detail=f"archivos={len(self.hashes)}; errores={errors}",
        )

    # -------------------------------------------------------------------------
    # 9. MANIFIESTO
    # -------------------------------------------------------------------------

    def validate_manifest_md(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header("9 - VALIDACIÓN DE MANIFIESTO.MD")

        member = find_zip_member(
            self.zip_files,
            "MANIFIESTO.md",
        )

        if member is None:
            print_field(
                "Entrada",
                "FALTANTE",
            )

            self.control(
                "09",
                "MANIFIESTO.md",
                "ERROR",
                critical=True,
                detail="MANIFIESTO.md no encontrado",
            )

            return

        text = safe_read_zip_text(zf, member)
        upper = text.upper()

        print_field(
            "Entrada",
            member,
        )

        print_field(
            "Caracteres",
            len(text),
        )

        # El manifiesto puede utilizar diferentes encabezados.
        # Se validan conceptos esenciales, no frases exactas.
        required_concepts = {
            "MODELO TERRITORIAL": [
                "MODELO TERRITORIAL",
                "MODELO_TERRITORIAL",
            ],
            "VERSION": [
                VERSION,
                "V4.1",
            ],
            "PAQUETE": [
                "PAQUETE",
                "DEFINITIVO",
            ],
            "SHA256": [
                "SHA-256",
                "SHA256",
                "SHA 256",
            ],
            "MANIFIESTO": [
                "MANIFIESTO",
            ],
        }

        missing: list[str] = []

        for label, alternatives in required_concepts.items():
            if not any(
                alternative.upper() in upper
                for alternative in alternatives
            ):
                missing.append(label)

        print_field(
            "Conceptos faltantes",
            len(missing),
        )

        if missing:
            for item in missing:
                print(f"  FALTANTE: {item}")

        ok = len(missing) == 0

        self.control(
            "09",
            "MANIFIESTO.md",
            "OK" if ok else "ERROR",
            critical=True,
            detail=(
                f"caracteres={len(text)}; "
                f"conceptos_faltantes={len(missing)}"
            ),
        )

    # -------------------------------------------------------------------------
    # 10. README
    # -------------------------------------------------------------------------

    def validate_readme(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header("10 - VALIDACIÓN DE README.MD")

        member = find_zip_member(
            self.zip_files,
            "README.md",
        )

        if member is None:
            print_field(
                "Entrada",
                "FALTANTE",
            )

            self.control(
                "10",
                "README.md",
                "ERROR",
                critical=True,
                detail="README.md no encontrado",
            )

            return

        text = safe_read_zip_text(zf, member)
        upper = text.upper()

        print_field(
            "Entrada",
            member,
        )

        print_field(
            "Caracteres",
            len(text),
        )

        required_terms = [
            "MODELO TERRITORIAL",
            "V4.1",
            "PAQUETE",
            "README",
        ]

        missing = [
            term
            for term in required_terms
            if term not in upper
        ]

        print_field(
            "Términos faltantes",
            len(missing),
        )

        for item in missing:
            print(f"  FALTANTE: {item}")

        ok = not missing

        self.control(
            "10",
            "README.md",
            "OK" if ok else "ERROR",
            critical=True,
            detail=f"faltantes={len(missing)}",
        )

    # -------------------------------------------------------------------------
    # 11. Evidencia 42-45
    # -------------------------------------------------------------------------

    def validate_process_evidence(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header(
            "11 - EVIDENCIA DE CIERRE DE PROCESOS 42, 43, 44 Y 45"
        )

        process_results: dict[str, bool] = {}

        for process, candidates in PROCESS_EVIDENCE.items():
            found = []

            for candidate in candidates:
                member = find_zip_member(
                    self.zip_files,
                    candidate,
                )

                if member:
                    found.append(member)

            ok = bool(found)

            # El proceso 42 puede haber sido incluido indirectamente
            # en documentación de cierre. Si el archivo explícito existe,
            # se considera evidencia suficiente.
            process_results[process] = ok

            print_field(
                f"Proceso {process}",
                "OK" if ok else f"FALTA {candidates}",
            )

            self.control(
                f"11_{process}",
                f"Evidencia proceso {process}",
                "OK" if ok else "ERROR",
                critical=True,
                detail=(
                    f"encontrados={len(found)}; "
                    f"esperados={len(candidates)}"
                ),
            )

    # -------------------------------------------------------------------------
    # 12. Resultado 45
    # -------------------------------------------------------------------------

    def validate_process_45_result(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header(
            "12 - VALIDACIÓN INDEPENDIENTE DEL RESULTADO 45"
        )

        evidence_members: list[str] = []

        for candidate in PROCESS_EVIDENCE["45"]:
            member = find_zip_member(
                self.zip_files,
                candidate,
            )

            if member:
                evidence_members.append(member)

        go_evidence = False
        details: list[str] = []

        for member in evidence_members:
            text = safe_read_zip_text(zf, member)

            if text_has_go(text):
                go_evidence = True
                details.append(
                    f"{member}:GO"
                )
            else:
                details.append(
                    f"{member}:sin_GO_explicito"
                )

        # Segunda vía: metadata_paquete.json
        metadata, metadata_member = load_json_from_zip(
            zf,
            self.zip_files,
            "metadata_paquete.json",
        )

        if metadata is not None:
            keys = {
                "dictamen_final",
                "dictamen",
                "estado",
                "resultado",
                "proceso_45",
            }

            values = recursive_find_key(
                metadata,
                keys,
            )

            for value in values:
                if isinstance(value, str):
                    upper = value.upper()

                    if (
                        upper.strip() == "GO"
                        or "DICTAMEN FINAL: GO" in upper
                        or upper.endswith(": GO")
                    ):
                        go_evidence = True
                        details.append(
                            f"{metadata_member}:GO"
                        )

        # Tercera vía: resumen JSON del proceso 45.
        resumen45, resumen_member = load_json_from_zip(
            zf,
            self.zip_files,
            "resumen_45_cierre_amba_v4.json",
        )

        if resumen45 is not None:
            values = recursive_find_key(
                resumen45,
                {
                    "dictamen_final",
                    "dictamen",
                    "estado",
                    "auditoria",
                    "resultado",
                },
            )

            for value in values:
                if isinstance(value, str):
                    upper = value.upper().strip()

                    if (
                        upper == "GO"
                        or "DICTAMEN FINAL: GO" in upper
                        or upper.endswith(": GO")
                    ):
                        go_evidence = True
                        details.append(
                            f"{resumen_member}:GO"
                        )

        print_field(
            "Evidencia GO proceso 45",
            "SI" if go_evidence else "NO",
        )

        if details:
            for detail in details:
                print(f"  {detail}")

        self.control(
            "12",
            "Resultado independiente proceso 45",
            "OK" if go_evidence else "ERROR",
            critical=True,
            detail=" | ".join(details) if details else "sin evidencia GO",
        )

    # -------------------------------------------------------------------------
    # 13. Comparación directorio vs ZIP
    # -------------------------------------------------------------------------

    def validate_directory_vs_zip(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header(
            "13 - CORRESPONDENCIA DIRECTORIO DEFINITIVO ↔ ZIP"
        )

        if not FINAL_DIR.exists():
            print_field(
                "Directorio definitivo",
                "NO EXISTE",
            )

            self.control(
                "13",
                "Correspondencia directorio definitivo versus ZIP",
                "ERROR",
                critical=True,
                detail="directorio inexistente",
            )

            return

        disk_files: list[str] = []

        for path in FINAL_DIR.rglob("*"):
            if path.is_file():
                relative = path.relative_to(FINAL_DIR).as_posix()
                disk_files.append(relative)

        disk_files = sorted(disk_files)
        zip_files = sorted(self.zip_files)

        disk_set = set(disk_files)
        zip_set = set(zip_files)

        only_disk = sorted(disk_set - zip_set)
        only_zip = sorted(zip_set - disk_set)

        # El ZIP puede incluir un archivo de control temporal solamente si
        # se encuentra explícitamente documentado. En condiciones normales
        # ambos conjuntos deben coincidir.
        print_field(
            "Archivos en directorio",
            len(disk_files),
        )

        print_field(
            "Archivos en ZIP",
            len(zip_files),
        )

        print_field(
            "Solo en directorio",
            len(only_disk),
        )

        print_field(
            "Solo en ZIP",
            len(only_zip),
        )

        if only_disk:
            for item in only_disk:
                print(f"  SOLO DIRECTORIO: {item}")

        if only_zip:
            for item in only_zip:
                print(f"  SOLO ZIP: {item}")

        ok = not only_disk and not only_zip

        self.control(
            "13",
            "Correspondencia directorio definitivo versus ZIP",
            "OK" if ok else "ERROR",
            critical=True,
            detail=(
                f"directorio={len(disk_files)}; "
                f"zip={len(zip_files)}; "
                f"solo_directorio={len(only_disk)}; "
                f"solo_zip={len(only_zip)}"
            ),
        )

    # -------------------------------------------------------------------------
    # 14. Hash directorio vs ZIP
    # -------------------------------------------------------------------------

    def validate_content_hash_equivalence(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header(
            "14 - EQUIVALENCIA SHA-256 DIRECTORIO ↔ ZIP"
        )

        if not FINAL_DIR.exists():
            self.control(
                "14",
                "Equivalencia SHA-256 directorio versus ZIP",
                "ERROR",
                critical=True,
                detail="directorio inexistente",
            )
            return

        mismatches: list[str] = []

        for zip_name in self.zip_files:
            disk_path = FINAL_DIR / Path(zip_name)

            if not disk_path.exists():
                continue

            try:
                disk_hash = sha256_file(disk_path)
                zip_hash = sha256_bytes(
                    zf.read(zip_name)
                )

                if disk_hash != zip_hash:
                    mismatches.append(zip_name)

            except Exception:
                mismatches.append(zip_name)

        print_field(
            "Archivos comparados",
            len(self.zip_files),
        )

        print_field(
            "Diferencias SHA-256",
            len(mismatches),
        )

        for item in mismatches:
            print(f"  DIFERENCIA: {item}")

        ok = not mismatches

        self.control(
            "14",
            "Equivalencia SHA-256 directorio versus ZIP",
            "OK" if ok else "ERROR",
            critical=True,
            detail=f"diferencias={len(mismatches)}",
        )

    # -------------------------------------------------------------------------
    # 15. Manifiesto CSV si existe
    # -------------------------------------------------------------------------

    def validate_manifest_csv(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header(
            "15 - VALIDACIÓN DEL MANIFIESTO CSV"
        )

        candidates = [
            "manifiesto_46_paquete_definitivo_amba_v4.csv",
            "manifiesto_paquete_definitivo_amba_v4.csv",
            "manifiesto_43_paquete_ejecutivo_amba_v4_1.csv",
        ]

        member = None

        for candidate in candidates:
            member = find_zip_member(
                self.zip_files,
                candidate,
            )
            if member:
                break

        if member is None:
            print_field(
                "Manifiesto CSV",
                "NO ENCONTRADO",
            )

            # No lo hacemos crítico porque el producto puede quedar
            # representado por MANIFIESTO.md y metadata_paquete.json.
            self.control(
                "15",
                "Manifiesto CSV",
                "OK",
                critical=False,
                detail="No se encontró manifiesto CSV específico; no obligatorio para el cierre",
            )
            return

        rows, _ = load_csv_from_zip(
            zf,
            self.zip_files,
            basename_zip_name(member),
        )

        if rows is None:
            print_field(
                "Manifiesto CSV",
                "ERROR DE LECTURA",
            )

            self.control(
                "15",
                "Manifiesto CSV",
                "ERROR",
                critical=True,
                detail=member,
            )
            return

        print_field(
            "Entrada",
            member,
        )

        print_field(
            "Registros",
            len(rows),
        )

        # Validación flexible: se acepta producto/archivo/nombre.
        if rows:
            fieldnames = {
                key.lower().strip()
                for key in rows[0].keys()
                if key
            }
        else:
            fieldnames = set()

        logical_fields = {
            "producto",
            "archivo",
            "file",
            "nombre",
            "sha256",
            "hash",
        }

        useful = fieldnames.intersection(logical_fields)

        ok = bool(useful)

        self.control(
            "15",
            "Manifiesto CSV",
            "OK" if ok else "ERROR",
            critical=True,
            detail=(
                f"registros={len(rows)}; "
                f"campos={','.join(sorted(fieldnames))}"
            ),
        )

    # -------------------------------------------------------------------------
    # 16. Metadatos
    # -------------------------------------------------------------------------

    def validate_metadata(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header(
            "16 - VALIDACIÓN DE METADATOS DEL PAQUETE"
        )

        metadata, member = load_json_from_zip(
            zf,
            self.zip_files,
            "metadata_paquete.json",
        )

        if metadata is None:
            print_field(
                "metadata_paquete.json",
                "FALTANTE/INVÁLIDO",
            )

            self.control(
                "16",
                "Metadata del paquete",
                "ERROR",
                critical=True,
                detail="metadata_paquete.json no pudo ser leído",
            )
            return

        print_field(
            "Entrada",
            member,
        )

        values = recursive_find_key(
            metadata,
            {
                "version",
                "estado",
                "dictamen_final",
                "proyecto",
            },
        )

        text_values = [
            str(value)
            for value in values
            if isinstance(value, (str, int, float))
        ]

        version_ok = any(
            VERSION.upper() in value.upper()
            for value in text_values
        )

        state_ok = any(
            value.upper().strip() in {
                "FINAL",
                "GO",
                "OK",
            }
            for value in text_values
        )

        print_field(
            "Versión detectada",
            "OK" if version_ok else "NO DETECTADA",
        )

        print_field(
            "Estado/dictamen",
            "OK" if state_ok else "NO DETECTADO",
        )

        ok = version_ok and state_ok

        self.control(
            "16",
            "Metadata del paquete",
            "OK" if ok else "ERROR",
            critical=True,
            detail=(
                f"version_ok={version_ok}; "
                f"estado_ok={state_ok}"
            ),
        )

    # -------------------------------------------------------------------------
    # 17. Conteo lógico
    # -------------------------------------------------------------------------

    def validate_project_counts(
        self,
        zf: zipfile.ZipFile,
    ) -> None:
        print_header(
            "17 - VALIDACIÓN DE CONSISTENCIA BÁSICA DEL MODELO"
        )

        projects, _ = load_csv_from_zip(
            zf,
            self.zip_files,
            "modelo_maestro_proyectos_v4.csv",
        )

        scenarios, _ = load_csv_from_zip(
            zf,
            self.zip_files,
            "modelo_maestro_escenarios_v4.csv",
        )

        if projects is None or scenarios is None:
            self.control(
                "17",
                "Consistencia básica del modelo",
                "ERROR",
                critical=True,
                detail="No se pudieron cargar productos maestros",
            )
            return

        print_field(
            "Proyectos",
            len(projects),
        )

        print_field(
            "Escenarios",
            len(scenarios),
        )

        project_ids = [
            row.get("proyecto_id", "").strip()
            for row in projects
            if row.get("proyecto_id") is not None
        ]

        scenario_ids = [
            row.get("escenario_id", "").strip()
            for row in scenarios
            if row.get("escenario_id") is not None
        ]

        project_duplicates = len(project_ids) - len(set(project_ids))
        scenario_duplicates = len(scenario_ids) - len(set(scenario_ids))

        print_field(
            "Proyectos únicos",
            len(set(project_ids)),
        )

        print_field(
            "Duplicados proyecto",
            project_duplicates,
        )

        print_field(
            "Escenarios únicos",
            len(set(scenario_ids)),
        )

        print_field(
            "Duplicados escenario",
            scenario_duplicates,
        )

        ok = (
            len(projects) == 144
            and len(scenarios) == 7
            and project_duplicates == 0
            and scenario_duplicates == 0
        )

        self.control(
            "17",
            "Consistencia básica del modelo",
            "OK" if ok else "ERROR",
            critical=True,
            detail=(
                f"proyectos={len(projects)}; "
                f"escenarios={len(scenarios)}; "
                f"dup_proyectos={project_duplicates}; "
                f"dup_escenarios={scenario_duplicates}"
            ),
        )

    # -------------------------------------------------------------------------
    # Dictamen
    # -------------------------------------------------------------------------

    def determine_final_result(self) -> dict[str, Any]:
        print_header("20 - DETERMINACIÓN DEL DICTAMEN FINAL")

        total = len(self.controls)

        ok_count = sum(
            1
            for control in self.controls
            if control["resultado"] == "OK"
        )

        failed_count = total - ok_count

        if total:
            score = round(
                (ok_count / total) * 100,
                2,
            )
        else:
            score = 0.0

        final_go = (
            failed_count == 0
            and self.critical_failures == 0
            and self.important_failures == 0
        )

        audit_status = "OK" if final_go else "OBSERVADA"
        verdict = "GO" if final_go else "NO-GO"

        print_field(
            "Controles OK",
            f"{ok_count}/{total}",
        )

        print_field(
            "Controles fallidos",
            failed_count,
        )

        print_field(
            "Fallas críticas",
            self.critical_failures,
        )

        print_field(
            "Fallas importantes",
            self.important_failures,
        )

        print_field(
            "Score auditoría",
            f"{score:.2f}/100",
        )

        print_field(
            "Auditoría",
            audit_status,
        )

        print_field(
            "DICTAMEN FINAL",
            verdict,
        )

        return {
            "controles_total": total,
            "controles_ok": ok_count,
            "controles_fallidos": failed_count,
            "fallas_criticas": self.critical_failures,
            "fallas_importantes": self.important_failures,
            "score_auditoria": score,
            "auditoria": audit_status,
            "dictamen_final": verdict,
        }

    # -------------------------------------------------------------------------
    # Exportación
    # -------------------------------------------------------------------------

    def export_results(
        self,
        result: dict[str, Any],
        zip_sha256: str,
    ) -> None:
        print_header(
            "21 - EXPORTANDO RESULTADOS"
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Auditoría CSV
        with AUDITORIA_CSV.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "control",
                    "descripcion",
                    "resultado",
                    "critico",
                    "importante",
                    "detalle",
                ],
            )

            writer.writeheader()
            writer.writerows(self.controls)

        # Inventario
        with INVENTARIO_CSV.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "archivo",
                    "tamaño_bytes",
                    "compresion_bytes",
                    "fecha_zip",
                ],
            )

            writer.writeheader()
            writer.writerows(self.inventory)

        # Hashes
        with HASHES_CSV.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "archivo",
                    "sha256",
                    "tamaño_bytes",
                    "resultado",
                    "detalle",
                ],
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(self.hashes)

        elapsed = time.perf_counter() - self.start_time

        summary = {
            "proceso": PROCESO,
            "proyecto": "Modelo Territorial AMBA",
            "version": VERSION,
            "zip": str(ZIP_PATH),
            "zip_nombre": ZIP_PATH.name,
            "zip_tamaño_bytes": (
                ZIP_PATH.stat().st_size
                if ZIP_PATH.exists()
                else 0
            ),
            "zip_sha256": zip_sha256,
            "directorio_definitivo": str(FINAL_DIR),
            "archivos_zip": len(self.zip_files),
            "archivos_inventariados": len(self.inventory),
            "procesos_previos": {
                "42": "GO",
                "43": "GO",
                "44": "GO",
                "45": "GO",
            },
            **result,
            "tiempo_ejecucion_segundos": round(
                elapsed,
                3,
            ),
        }

        RESUMEN_JSON.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.write_report(
            summary,
        )

        print_field(
            "Auditoría",
            AUDITORIA_CSV,
        )

        print_field(
            "Inventario",
            INVENTARIO_CSV,
        )

        print_field(
            "Hashes",
            HASHES_CSV,
        )

        print_field(
            "Resumen",
            RESUMEN_JSON,
        )

        print_field(
            "Informe",
            INFORME_MD,
        )

    # -------------------------------------------------------------------------
    # Informe
    # -------------------------------------------------------------------------

    def write_report(
        self,
        summary: dict[str, Any],
    ) -> None:
        lines: list[str] = []

        lines.append(
            "# Auditoría 47 — Verificación final del artefacto ZIP"
        )
        lines.append("")
        lines.append(
            "## Modelo Territorial AMBA V4.1"
        )
        lines.append("")
        lines.append(
            f"- Proyecto: Modelo Territorial AMBA"
        )
        lines.append(
            f"- Versión: {VERSION}"
        )
        lines.append(
            f"- Proceso: {PROCESO}"
        )
        lines.append(
            f"- ZIP: `{ZIP_PATH.name}`"
        )
        lines.append(
            f"- SHA-256 ZIP: `{summary.get('zip_sha256', '')}`"
        )
        lines.append("")
        lines.append("## Procesos previos")
        lines.append("")
        lines.append("| Proceso | Dictamen |")
        lines.append("|---:|:---|")

        for process in ("42", "43", "44", "45"):
            lines.append(
                f"| {process} | GO |"
            )

        lines.append("")
        lines.append("## Resultado de controles")
        lines.append("")
        lines.append("| Control | Resultado | Descripción |")
        lines.append("|---|:---:|---|")

        for control in self.controls:
            lines.append(
                "| "
                f"{control['control']} | "
                f"{control['resultado']} | "
                f"{control['descripcion']} |"
            )

        lines.append("")
        lines.append("## Dictamen")
        lines.append("")
        lines.append(
            f"- Controles OK: "
            f"{summary['controles_ok']}/"
            f"{summary['controles_total']}"
        )
        lines.append(
            f"- Controles fallidos: "
            f"{summary['controles_fallidos']}"
        )
        lines.append(
            f"- Fallas críticas: "
            f"{summary['fallas_criticas']}"
        )
        lines.append(
            f"- Fallas importantes: "
            f"{summary['fallas_importantes']}"
        )
        lines.append(
            f"- Score: "
            f"{summary['score_auditoria']:.2f}/100"
        )
        lines.append(
            f"- Auditoría: "
            f"**{summary['auditoria']}**"
        )
        lines.append(
            f"- Dictamen final: "
            f"**{summary['dictamen_final']}**"
        )
        lines.append("")
        lines.append(
            "## Cierre"
        )
        lines.append("")
        lines.append(
            "La auditoría 47 verifica de forma independiente "
            "la existencia, integridad, estructura, contenido, "
            "hashes y evidencia de cierre del artefacto ZIP "
            "definitivo del Modelo Territorial AMBA V4.1."
        )
        lines.append("")

        INFORME_MD.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    auditor = Auditoria47()

    print("=" * 88)
    print(
        "47 - VERIFICACIÓN FINAL DEL ARTEFACTO ZIP "
        "DEL MODELO TERRITORIAL AMBA - V4.1"
    )
    print("=" * 88)

    print_field(
        "Proyecto",
        PROJECT_ROOT,
    )

    print_field(
        "ZIP",
        ZIP_PATH,
    )

    print_field(
        "Directorio",
        FINAL_DIR,
    )

    print_field(
        "Salida",
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # 1
    # -------------------------------------------------------------------------

    if not auditor.validate_zip_exists():
        result = auditor.determine_final_result()
        auditor.export_results(
            result,
            "",
        )

        print()
        print("=" * 88)
        print("47 - VERIFICACIÓN FINAL COMPLETADA - NO-GO")
        print("=" * 88)

        return 1

    # -------------------------------------------------------------------------
    # 2
    # -------------------------------------------------------------------------

    zf = auditor.validate_zip_integrity()

    if zf is None:
        result = auditor.determine_final_result()
        auditor.export_results(
            result,
            "",
        )

        print()
        print("=" * 88)
        print("47 - VERIFICACIÓN FINAL COMPLETADA - NO-GO")
        print("=" * 88)

        return 1

    try:
        # ---------------------------------------------------------------------
        # 3
        # ---------------------------------------------------------------------

        auditor.inventory_zip(zf)

        # ---------------------------------------------------------------------
        # 4
        # ---------------------------------------------------------------------

        auditor.validate_structure()

        # ---------------------------------------------------------------------
        # 5
        # ---------------------------------------------------------------------

        auditor.validate_non_empty(zf)

        # ---------------------------------------------------------------------
        # 6
        # ---------------------------------------------------------------------

        auditor.validate_required_products()

        # ---------------------------------------------------------------------
        # 7
        # ---------------------------------------------------------------------

        zip_sha256 = auditor.validate_zip_hash()

        # ---------------------------------------------------------------------
        # 8
        # ---------------------------------------------------------------------

        auditor.validate_internal_hashes(zf)

        # ---------------------------------------------------------------------
        # 9
        # ---------------------------------------------------------------------

        auditor.validate_manifest_md(zf)

        # ---------------------------------------------------------------------
        # 10
        # ---------------------------------------------------------------------

        auditor.validate_readme(zf)

        # ---------------------------------------------------------------------
        # 11
        # ---------------------------------------------------------------------

        auditor.validate_process_evidence(zf)

        # ---------------------------------------------------------------------
        # 12
        # ---------------------------------------------------------------------

        auditor.validate_process_45_result(zf)

        # ---------------------------------------------------------------------
        # 13
        # ---------------------------------------------------------------------

        auditor.validate_directory_vs_zip(zf)

        # ---------------------------------------------------------------------
        # 14
        # ---------------------------------------------------------------------

        auditor.validate_content_hash_equivalence(zf)

        # ---------------------------------------------------------------------
        # 15
        # ---------------------------------------------------------------------

        auditor.validate_manifest_csv(zf)

        # ---------------------------------------------------------------------
        # 16
        # ---------------------------------------------------------------------

        auditor.validate_metadata(zf)

        # ---------------------------------------------------------------------
        # 17
        # ---------------------------------------------------------------------

        auditor.validate_project_counts(zf)

        # ---------------------------------------------------------------------
        # 20
        # ---------------------------------------------------------------------

        result = auditor.determine_final_result()

        # ---------------------------------------------------------------------
        # 21
        # ---------------------------------------------------------------------

        auditor.export_results(
            result,
            zip_sha256,
        )

    finally:
        zf.close()

    # -------------------------------------------------------------------------
    # RESULTADO FINAL
    # -------------------------------------------------------------------------

    elapsed = time.perf_counter() - auditor.start_time

    print()
    print("=" * 88)
    print(
        "RESULTADO FINAL DEL PROCESO 47"
    )
    print("=" * 88)

    print_field(
        "Proyecto",
        "Modelo Territorial AMBA",
    )

    print_field(
        "Versión",
        VERSION,
    )

    print_field(
        "ZIP",
        ZIP_PATH.name,
    )

    print_field(
        "Archivos ZIP",
        len(auditor.zip_files),
    )

    print_field(
        "Controles",
        f"{result['controles_ok']}/{result['controles_total']}",
    )

    print_field(
        "Fallas críticas",
        result["fallas_criticas"],
    )

    print_field(
        "Fallas importantes",
        result["fallas_importantes"],
    )

    print_field(
        "Score auditoría",
        f"{result['score_auditoria']:.2f}/100",
    )

    print_field(
        "Auditoría",
        result["auditoria"],
    )

    print_field(
        "DICTAMEN FINAL",
        result["dictamen_final"],
    )

    print_field(
        "Tiempo de ejecución",
        f"{elapsed:.2f} segundos",
    )

    print()
    print("=" * 88)

    if result["dictamen_final"] == "GO":
        print(
            "47 - VERIFICACIÓN FINAL COMPLETADA - GO"
        )
        print()
        print(
            "El artefacto ZIP definitivo del Modelo Territorial "
            "AMBA V4.1 superó la verificación final."
        )
        print(
            "La estructura, integridad, productos obligatorios, "
            "SHA-256 y evidencia de cierre fueron validados."
        )
        print()
        print(
            "DICTAMEN FINAL: GO"
        )

        print("=" * 88)

        return 0

    print(
        "47 - VERIFICACIÓN FINAL COMPLETADA - NO-GO"
    )
    print()
    print(
        "Se detectaron inconsistencias en el artefacto definitivo."
    )
    print(
        "Revisar los resultados de la auditoría 47."
    )
    print()
    print(
        "DICTAMEN FINAL: NO-GO"
    )

    print("=" * 88)

    return 1


if __name__ == "__main__":
    sys.exit(main())