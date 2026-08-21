# -*- coding: utf-8 -*-
"""
41_auditar_modelo_territorial_amba_v4.py

AUDITORÍA FINAL DEL MODELO TERRITORIAL AMBA V4

Proceso 41
----------
Auditoría integral y certificación final del modelo territorial AMBA V4.

Características
---------------
- No modifica ninguna fuente.
- Audita procesos 38, 39 y 40.
- Verifica integridad tabular.
- Verifica integridad geoespacial.
- Verifica consistencia proyecto -> escenario.
- Verifica rankings.
- Verifica cobertura geométrica.
- Verifica GeoPackages.
- Verifica productos esperados.
- Calcula hashes SHA-256 de productos críticos.
- Genera auditoría detallada.
- Genera inventario de archivos.
- Genera resumen JSON.
- Genera informe Markdown.
- Emite dictamen GO / NO-GO.

Salida principal
----------------
auditoria_41_modelo_territorial_amba_v4.csv
resumen_41_auditoria_modelo_territorial_amba_v4.json
informe_41_auditoria_modelo_territorial_amba_v4.md
inventario_41_productos_amba_v4.csv
hashes_41_productos_amba_v4.csv
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.0"
PROCESO = "41"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

SALIDA = DATA_DIR

print("=" * 88)
print("41 - AUDITORÍA FINAL DEL MODELO TERRITORIAL AMBA - V4")
print("=" * 88)
print(f"Proyecto : {PROJECT_ROOT}")
print(f"Entrada  : {DATA_DIR}")
print(f"Salida   : {SALIDA}")
print()


# =============================================================================
# ARCHIVOS ESPERADOS
# =============================================================================

ARCHIVOS_ESPERADOS = {
    # Proceso 38
    "modelo_maestro_proyectos": (
        "modelo_maestro_proyectos_v4.csv",
        "Proceso 38 - modelo maestro de proyectos",
        True,
    ),
    "modelo_maestro_escenarios": (
        "modelo_maestro_escenarios_v4.csv",
        "Proceso 38 - modelo maestro de escenarios",
        True,
    ),
    "ranking_final_escenarios": (
        "ranking_final_escenarios_v4.csv",
        "Proceso 38 - ranking final de escenarios",
        True,
    ),
    "ranking_final_proyectos": (
        "ranking_final_proyectos_v4.csv",
        "Proceso 38 - ranking final de proyectos",
        True,
    ),
    "matriz_integral": (
        "matriz_integral_escenarios_v4.csv",
        "Proceso 38 - matriz integral",
        True,
    ),
    "indicadores_globales": (
        "indicadores_globales_amba_v4.csv",
        "Proceso 38 - indicadores globales",
        True,
    ),
    "auditoria_38": (
        "auditoria_38_consolidacion_territorial_amba.csv",
        "Proceso 38 - auditoría",
        True,
    ),
    "gpkg_maestro": (
        "modelo_maestro_territorial_amba_v4.gpkg",
        "Proceso 38 - GeoPackage maestro",
        True,
    ),

    # Proceso 39
    "informe_39": (
        "informe_territorial_amba_v4_1.md",
        "Proceso 39 - informe territorial V4.1",
        True,
    ),
    "resumen_ejecutivo_39": (
        "resumen_ejecutivo_amba_v4_1.md",
        "Proceso 39 - resumen ejecutivo",
        True,
    ),
    "auditoria_39": (
        "auditoria_39_informe_territorial_amba_v4_1.csv",
        "Proceso 39 - auditoría",
        True,
    ),
    "resumen_39": (
        "resumen_39_informe_territorial_amba_v4_1.json",
        "Proceso 39 - resumen JSON",
        True,
    ),
    "anexo_proyectos_39": (
        "anexo_proyectos_amba_v4_1.csv",
        "Proceso 39 - anexo proyectos",
        True,
    ),
    "anexo_escenarios_39": (
        "anexo_escenarios_amba_v4_1.csv",
        "Proceso 39 - anexo escenarios",
        True,
    ),
    "anexo_indicadores_39": (
        "anexo_indicadores_globales_amba_v4_1.csv",
        "Proceso 39 - anexo indicadores",
        True,
    ),

    # Proceso 40
    "atlas_gpkg": (
        "atlas_territorial_amba_v4.gpkg",
        "Proceso 40 - Atlas GeoPackage",
        True,
    ),
    "control_cartografico_40": (
        "control_cartografico_amba_v4.csv",
        "Proceso 40 - control cartográfico",
        True,
    ),
    "auditoria_40": (
        "auditoria_40_atlas_territorial_amba.csv",
        "Proceso 40 - auditoría",
        True,
    ),
    "resumen_40": (
        "resumen_40_atlas_territorial_amba.json",
        "Proceso 40 - resumen JSON",
        True,
    ),
    "atlas_html": (
        "atlas_territorial_amba_v4/atlas_territorial_amba_v4.html",
        "Proceso 40 - Atlas HTML",
        True,
    ),
    "atlas_md": (
        "atlas_territorial_amba_v4.md",
        "Proceso 40 - documentación Markdown",
        True,
    ),
}


# =============================================================================
# UTILIDADES
# =============================================================================

def separador(titulo: str) -> None:
    print()
    print("=" * 88)
    print(titulo)
    print("=" * 88)


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def cargar_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )


def columna_existente(
    df: pd.DataFrame,
    candidatos: list[str],
) -> str | None:
    for col in candidatos:
        if col in df.columns:
            return col
    return None


def normalizar_id_serie(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NaN": pd.NA,
            }
        )
    )


def registrar(
    auditoria: list[dict[str, Any]],
    control: str,
    categoria: str,
    resultado: bool,
    valor: Any,
    esperado: Any,
    detalle: str = "",
    severidad: str = "CRITICO",
) -> None:

    auditoria.append(
        {
            "proceso": PROCESO,
            "control": control,
            "categoria": categoria,
            "resultado": "OK" if resultado else "FALLA",
            "valor": valor,
            "esperado": esperado,
            "severidad": severidad,
            "detalle": detalle,
        }
    )


def contar_ok(auditoria: list[dict[str, Any]]) -> tuple[int, int]:
    total = len(auditoria)
    ok = sum(1 for x in auditoria if x["resultado"] == "OK")
    return ok, total


# =============================================================================
# RESOLUCIÓN DE CAMPOS
# =============================================================================

def resolver_campos_proyectos(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "proyecto": columna_existente(
            df,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
        ),
        "escenario": columna_existente(
            df,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        ),
    }


def resolver_campos_escenarios(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "escenario": columna_existente(
            df,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    inicio = time.time()

    SALIDA.mkdir(parents=True, exist_ok=True)

    auditoria: list[dict[str, Any]] = []
    inventario: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []

    # =========================================================================
    # 1. INVENTARIO DE PRODUCTOS
    # =========================================================================

    separador("1 - INVENTARIO DE PRODUCTOS DE LOS PROCESOS 38-40")

    archivos_ok = 0
    archivos_faltantes = 0

    for clave, (nombre, descripcion, obligatorio) in ARCHIVOS_ESPERADOS.items():

        path = DATA_DIR / nombre
        existe = path.exists()

        if existe:
            stat = path.stat()

            inventario.append(
                {
                    "clave": clave,
                    "archivo": nombre,
                    "descripcion": descripcion,
                    "obligatorio": obligatorio,
                    "existe": True,
                    "tamano_bytes": stat.st_size,
                    "tamano_mb": round(stat.st_size / 1024 / 1024, 4),
                    "fecha_modificacion": pd.Timestamp.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(),
                }
            )

            archivos_ok += 1

            print(
                f"[OK] {nombre} "
                f"({stat.st_size / 1024 / 1024:.2f} MB)"
            )

        else:

            inventario.append(
                {
                    "clave": clave,
                    "archivo": nombre,
                    "descripcion": descripcion,
                    "obligatorio": obligatorio,
                    "existe": False,
                    "tamano_bytes": 0,
                    "tamano_mb": 0,
                    "fecha_modificacion": None,
                }
            )

            if obligatorio:
                archivos_faltantes += 1

            print(
                f"[{'FALLA' if obligatorio else 'OBSERVADO'}] "
                f"{nombre}"
            )

    registrar(
        auditoria,
        "productos_esperados",
        "integridad_archivos",
        archivos_faltantes == 0,
        archivos_ok,
        len(ARCHIVOS_ESPERADOS),
        f"Archivos encontrados: {archivos_ok}; faltantes obligatorios: "
        f"{archivos_faltantes}",
    )

    # =========================================================================
    # 2. CARGA DEL MODELO MAESTRO
    # =========================================================================

    separador("2 - CARGANDO MODELO MAESTRO DEL PROCESO 38")

    proyectos_path = DATA_DIR / "modelo_maestro_proyectos_v4.csv"
    escenarios_path = DATA_DIR / "modelo_maestro_escenarios_v4.csv"

    if not proyectos_path.exists() or not escenarios_path.exists():
        raise FileNotFoundError(
            "No se encontraron los archivos maestros del proceso 38."
        )

    proyectos = cargar_csv(proyectos_path)
    escenarios = cargar_csv(escenarios_path)

    print(
        f"Proyectos : {len(proyectos)} registros | "
        f"{len(proyectos.columns)} columnas"
    )

    print(
        f"Escenarios: {len(escenarios)} registros | "
        f"{len(escenarios.columns)} columnas"
    )

    campos_p = resolver_campos_proyectos(proyectos)
    campos_e = resolver_campos_escenarios(escenarios)

    print()
    print("Campos resueltos:")
    print(f"  proyecto  : {campos_p['proyecto']}")
    print(f"  escenario : {campos_p['escenario']}")
    print(f"  escenario maestro: {campos_e['escenario']}")

    proyecto_col = campos_p["proyecto"]
    escenario_proyecto_col = campos_p["escenario"]
    escenario_col = campos_e["escenario"]

    registrar(
        auditoria,
        "campo_proyecto",
        "estructura",
        proyecto_col is not None,
        proyecto_col or "NO DISPONIBLE",
        "proyecto_id",
    )

    registrar(
        auditoria,
        "campo_escenario_proyecto",
        "estructura",
        escenario_proyecto_col is not None,
        escenario_proyecto_col or "NO DISPONIBLE",
        "escenario_id",
    )

    registrar(
        auditoria,
        "campo_escenario_maestro",
        "estructura",
        escenario_col is not None,
        escenario_col or "NO DISPONIBLE",
        "escenario_id",
    )

    if proyecto_col is None:
        raise KeyError("No se encontró proyecto_id en modelo maestro.")

    if escenario_col is None:
        raise KeyError("No se encontró escenario_id en modelo maestro.")

    # =========================================================================
    # 3. VALIDACIÓN ESTRUCTURAL
    # =========================================================================

    separador("3 - VALIDACIÓN ESTRUCTURAL DEL MODELO")

    proyectos_ids = normalizar_id_serie(proyectos[proyecto_col])
    escenarios_ids = normalizar_id_serie(escenarios[escenario_col])

    n_proyectos = len(proyectos)
    n_proyectos_unicos = proyectos_ids.nunique(dropna=True)
    n_proyectos_nulos = int(proyectos_ids.isna().sum())
    n_proyectos_dup = int(proyectos_ids.duplicated().sum())

    n_escenarios = len(escenarios)
    n_escenarios_unicos = escenarios_ids.nunique(dropna=True)
    n_escenarios_nulos = int(escenarios_ids.isna().sum())
    n_escenarios_dup = int(escenarios_ids.duplicated().sum())

    print(f"Proyectos                 : {n_proyectos}")
    print(f"Proyectos únicos          : {n_proyectos_unicos}")
    print(f"Proyectos nulos           : {n_proyectos_nulos}")
    print(f"Proyectos duplicados      : {n_proyectos_dup}")
    print(f"Escenarios                : {n_escenarios}")
    print(f"Escenarios únicos         : {n_escenarios_unicos}")
    print(f"Escenarios nulos          : {n_escenarios_nulos}")
    print(f"Escenarios duplicados     : {n_escenarios_dup}")

    registrar(
        auditoria,
        "cantidad_proyectos",
        "estructura",
        n_proyectos == 144,
        n_proyectos,
        144,
    )

    registrar(
        auditoria,
        "proyectos_unicos",
        "estructura",
        n_proyectos_unicos == n_proyectos,
        n_proyectos_unicos,
        n_proyectos,
    )

    registrar(
        auditoria,
        "proyectos_nulos",
        "estructura",
        n_proyectos_nulos == 0,
        n_proyectos_nulos,
        0,
    )

    registrar(
        auditoria,
        "proyectos_duplicados",
        "estructura",
        n_proyectos_dup == 0,
        n_proyectos_dup,
        0,
    )

    registrar(
        auditoria,
        "cantidad_escenarios",
        "estructura",
        n_escenarios == 7,
        n_escenarios,
        7,
    )

    registrar(
        auditoria,
        "escenarios_unicos",
        "estructura",
        n_escenarios_unicos == n_escenarios,
        n_escenarios_unicos,
        n_escenarios,
    )

    registrar(
        auditoria,
        "escenarios_nulos",
        "estructura",
        n_escenarios_nulos == 0,
        n_escenarios_nulos,
        0,
    )

    registrar(
        auditoria,
        "escenarios_duplicados",
        "estructura",
        n_escenarios_dup == 0,
        n_escenarios_dup,
        0,
    )

    # =========================================================================
    # 4. CONSISTENCIA PROYECTO -> ESCENARIO
    # =========================================================================

    separador("4 - VALIDACIÓN DE ASIGNACIÓN PROYECTO -> ESCENARIO")

    if escenario_proyecto_col is not None:

        escenario_proyecto = normalizar_id_serie(
            proyectos[escenario_proyecto_col]
        )

        n_escenario_nulos = int(escenario_proyecto.isna().sum())

        conteo_escenario = escenario_proyecto.value_counts(
            dropna=True
        )

        proyectos_multiescenario = 0

        # El modelo maestro ya tiene una fila por proyecto.
        # Se verifica además que no existan proyectos repetidos con
        # escenarios diferentes.
        tmp = pd.DataFrame(
            {
                "proyecto": proyectos_ids,
                "escenario": escenario_proyecto,
            }
        )

        multi = (
            tmp.dropna(subset=["proyecto"])
            .groupby("proyecto")["escenario"]
            .nunique()
        )

        proyectos_multiescenario = int((multi > 1).sum())

        print(f"Escenarios nulos           : {n_escenario_nulos}")
        print(
            f"Proyectos multiescenario   : "
            f"{proyectos_multiescenario}"
        )

        print()
        print("Distribución de proyectos por escenario:")

        for esc, cantidad in conteo_escenario.sort_index().items():
            print(f"  {esc}: {cantidad}")

        registrar(
            auditoria,
            "escenarios_proyecto_nulos",
            "consistencia_territorial",
            n_escenario_nulos == 0,
            n_escenario_nulos,
            0,
        )

        registrar(
            auditoria,
            "proyectos_multiescenario",
            "consistencia_territorial",
            proyectos_multiescenario == 0,
            proyectos_multiescenario,
            0,
        )

        registrar(
            auditoria,
            "cantidad_escenarios_asignados",
            "consistencia_territorial",
            escenario_proyecto.nunique(dropna=True) == 7,
            int(escenario_proyecto.nunique(dropna=True)),
            7,
        )

    else:
        registrar(
            auditoria,
            "asignacion_proyecto_escenario",
            "consistencia_territorial",
            False,
            "NO DISPONIBLE",
            "escenario_id",
        )

    # =========================================================================
    # 5. VALIDACIÓN DE TAMAÑO DE ESCENARIOS
    # =========================================================================

    separador("5 - VALIDACIÓN DE DISTRIBUCIÓN TERRITORIAL")

    if escenario_proyecto_col is not None:

        conteos = escenario_proyecto.value_counts(
            dropna=True
        )

        minimo = int(conteos.min())
        maximo = int(conteos.max())
        promedio = float(conteos.mean())

        if promedio != 0:
            cv = float(conteos.std(ddof=0) / promedio)
        else:
            cv = 0.0

        print(f"Mínimo proyectos/escenario : {minimo}")
        print(f"Máximo proyectos/escenario : {maximo}")
        print(f"Promedio                   : {promedio:.2f}")
        print(f"CV                         : {cv:.4f}")

        registrar(
            auditoria,
            "minimo_proyectos_escenario",
            "distribucion",
            minimo >= 20,
            minimo,
            ">= 20",
            severidad="IMPORTANTE",
        )

        registrar(
            auditoria,
            "maximo_proyectos_escenario",
            "distribucion",
            maximo <= 21,
            maximo,
            "<= 21",
            severidad="IMPORTANTE",
        )

        registrar(
            auditoria,
            "cv_escenarios",
            "distribucion",
            cv <= 0.10,
            round(cv, 6),
            "<= 0.10",
            severidad="IMPORTANTE",
        )

    else:
        minimo = maximo = 0
        promedio = 0.0
        cv = 999.0

    # =========================================================================
    # 6. RANKING DE ESCENARIOS
    # =========================================================================

    separador("6 - AUDITORÍA DEL RANKING DE ESCENARIOS")

    ranking_e_path = DATA_DIR / "ranking_final_escenarios_v4.csv"

    if ranking_e_path.exists():

        ranking_e = cargar_csv(ranking_e_path)

        rank_esc_col = columna_existente(
            ranking_e,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        )

        rank_col = columna_existente(
            ranking_e,
            [
                "ranking",
                "rank",
                "posicion",
                "ranking_final",
            ],
        )

        print(f"Registros ranking: {len(ranking_e)}")
        print(f"Campo escenario  : {rank_esc_col}")
        print(f"Campo ranking    : {rank_col}")

        ranking_ids_ok = False
        ranking_pos_ok = False

        if rank_esc_col:
            ranking_ids = normalizar_id_serie(
                ranking_e[rank_esc_col]
            )

            ranking_ids_ok = (
                ranking_ids.nunique(dropna=True) == 7
                and set(ranking_ids.dropna())
                == set(escenarios_ids.dropna())
            )

        if rank_col:
            ranking_num = pd.to_numeric(
                ranking_e[rank_col],
                errors="coerce",
            )

            valores = sorted(
                ranking_num.dropna().astype(int).unique().tolist()
            )

            ranking_pos_ok = valores == list(range(1, 8))

        print(
            f"IDs escenarios completos : "
            f"{'SI' if ranking_ids_ok else 'NO'}"
        )

        print(
            f"Ranking 1..7 completo     : "
            f"{'SI' if ranking_pos_ok else 'NO'}"
        )

        registrar(
            auditoria,
            "ranking_escenarios_ids",
            "ranking",
            ranking_ids_ok,
            ranking_e[rank_esc_col].nunique()
            if rank_esc_col else 0,
            7,
        )

        registrar(
            auditoria,
            "ranking_escenarios_posiciones",
            "ranking",
            ranking_pos_ok,
            valores if rank_col else [],
            list(range(1, 8)),
        )

    else:

        registrar(
            auditoria,
            "ranking_escenarios_archivo",
            "ranking",
            False,
            "NO EXISTE",
            "ranking_final_escenarios_v4.csv",
        )

    # =========================================================================
    # 7. RANKING DE PROYECTOS
    # =========================================================================

    separador("7 - AUDITORÍA DEL RANKING DE PROYECTOS")

    ranking_p_path = DATA_DIR / "ranking_final_proyectos_v4.csv"

    if ranking_p_path.exists():

        ranking_p = cargar_csv(ranking_p_path)

        rank_p_id = columna_existente(
            ranking_p,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
        )

        rank_p_rank = columna_existente(
            ranking_p,
            [
                "ranking",
                "rank",
                "posicion",
                "ranking_final",
            ],
        )

        print(f"Registros ranking: {len(ranking_p)}")
        print(f"Campo proyecto   : {rank_p_id}")
        print(f"Campo ranking    : {rank_p_rank}")

        ranking_p_ids_ok = False
        ranking_p_pos_ok = False

        if rank_p_id:
            ranking_ids = normalizar_id_serie(
                ranking_p[rank_p_id]
            )

            ranking_p_ids_ok = (
                ranking_ids.nunique(dropna=True) == n_proyectos
                and set(ranking_ids.dropna())
                == set(proyectos_ids.dropna())
            )

        if rank_p_rank:

            ranking_num = pd.to_numeric(
                ranking_p[rank_p_rank],
                errors="coerce",
            )

            ranking_p_pos_ok = (
                ranking_num.notna().sum() == n_proyectos
                and ranking_num.nunique() == n_proyectos
                and ranking_num.min() == 1
                and ranking_num.max() == n_proyectos
            )

        print(
            f"IDs de proyectos completos : "
            f"{'SI' if ranking_p_ids_ok else 'NO'}"
        )

        print(
            f"Ranking 1..{n_proyectos} completo : "
            f"{'SI' if ranking_p_pos_ok else 'NO'}"
        )

        registrar(
            auditoria,
            "ranking_proyectos_ids",
            "ranking",
            ranking_p_ids_ok,
            ranking_p[rank_p_id].nunique()
            if rank_p_id else 0,
            n_proyectos,
        )

        registrar(
            auditoria,
            "ranking_proyectos_posiciones",
            "ranking",
            ranking_p_pos_ok,
            (
                f"1..{n_proyectos}"
                if ranking_p_pos_ok
                else "INCORRECTO"
            ),
            f"1..{n_proyectos}",
        )

    else:

        registrar(
            auditoria,
            "ranking_proyectos_archivo",
            "ranking",
            False,
            "NO EXISTE",
            "ranking_final_proyectos_v4.csv",
        )

    # =========================================================================
    # 8. GEO PACKAGE MAESTRO
    # =========================================================================

    separador("8 - AUDITORÍA GEOESPACIAL DEL MODELO MAESTRO")

    gpkg_path = DATA_DIR / "modelo_maestro_territorial_amba_v4.gpkg"

    if not gpkg_path.exists():
        registrar(
            auditoria,
            "gpkg_maestro_existe",
            "geoespacial",
            False,
            "NO EXISTE",
            gpkg_path.name,
        )
        raise FileNotFoundError(
            f"No existe el GeoPackage maestro: {gpkg_path}"
        )

    try:
        capas = gpd.list_layers(gpkg_path)

        nombres_capas = capas["name"].astype(str).tolist()

        print("Capas disponibles:")
        for capa in nombres_capas:
            print(f"  - {capa}")

        registrar(
            auditoria,
            "gpkg_capas",
            "geoespacial",
            "proyectos" in nombres_capas
            and "escenarios" in nombres_capas,
            nombres_capas,
            ["proyectos", "escenarios"],
        )

        gdf_proyectos = gpd.read_file(
            gpkg_path,
            layer="proyectos",
        )

        gdf_escenarios = gpd.read_file(
            gpkg_path,
            layer="escenarios",
        )

        print()
        print(f"Proyectos geográficos : {len(gdf_proyectos)}")
        print(f"Escenarios geográficos: {len(gdf_escenarios)}")
        print(f"CRS proyectos         : {gdf_proyectos.crs}")
        print(f"CRS escenarios        : {gdf_escenarios.crs}")

        # ---------------------------------------------------------------------
        # Geometrías
        # ---------------------------------------------------------------------

        geom_validas = int(
            gdf_proyectos.geometry.notna().sum()
            - gdf_proyectos.geometry.is_empty.sum()
            - (~gdf_proyectos.geometry.is_valid).sum()
        )

        geom_nulas = int(
            gdf_proyectos.geometry.isna().sum()
        )

        geom_vacias = int(
            gdf_proyectos.geometry.is_empty.sum()
        )

        geom_invalidas = int(
            (
                gdf_proyectos.geometry.notna()
                & ~gdf_proyectos.geometry.is_empty
                & ~gdf_proyectos.geometry.is_valid
            ).sum()
        )

        cobertura = (
            geom_validas / len(gdf_proyectos) * 100
            if len(gdf_proyectos)
            else 0
        )

        print()
        print(f"Geometrías válidas      : {geom_validas}")
        print(f"Geometrías nulas        : {geom_nulas}")
        print(f"Geometrías vacías       : {geom_vacias}")
        print(f"Geometrías inválidas    : {geom_invalidas}")
        print(f"Cobertura geométrica    : {cobertura:.2f}%")

        registrar(
            auditoria,
            "gpkg_proyectos_cantidad",
            "geoespacial",
            len(gdf_proyectos) == n_proyectos,
            len(gdf_proyectos),
            n_proyectos,
        )

        registrar(
            auditoria,
            "geometrias_validas",
            "geoespacial",
            geom_validas == n_proyectos,
            geom_validas,
            n_proyectos,
        )

        registrar(
            auditoria,
            "geometrias_nulas",
            "geoespacial",
            geom_nulas == 0,
            geom_nulas,
            0,
        )

        registrar(
            auditoria,
            "geometrias_vacias",
            "geoespacial",
            geom_vacias == 0,
            geom_vacias,
            0,
        )

        registrar(
            auditoria,
            "geometrias_invalidas",
            "geoespacial",
            geom_invalidas == 0,
            geom_invalidas,
            0,
        )

        registrar(
            auditoria,
            "cobertura_geometrica",
            "geoespacial",
            cobertura == 100.0,
            round(cobertura, 4),
            100.0,
        )

        # ---------------------------------------------------------------------
        # IDs geográficos
        # ---------------------------------------------------------------------

        geo_proyecto_col = columna_existente(
            gdf_proyectos,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
        )

        geo_escenario_col = columna_existente(
            gdf_proyectos,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        )

        geo_escenario_master_col = columna_existente(
            gdf_escenarios,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        )

        geo_ids_ok = False

        if geo_proyecto_col:

            geo_ids = normalizar_id_serie(
                gdf_proyectos[geo_proyecto_col]
            )

            geo_ids_ok = (
                geo_ids.nunique(dropna=True) == n_proyectos
                and set(geo_ids.dropna())
                == set(proyectos_ids.dropna())
            )

        registrar(
            auditoria,
            "gpkg_ids_proyectos",
            "geoespacial",
            geo_ids_ok,
            (
                gdf_proyectos[geo_proyecto_col].nunique()
                if geo_proyecto_col
                else 0
            ),
            n_proyectos,
        )

        escenarios_geo_ok = False

        if geo_escenario_master_col:

            escenarios_geo_ids = normalizar_id_serie(
                gdf_escenarios[geo_escenario_master_col]
            )

            escenarios_geo_ok = (
                escenarios_geo_ids.nunique(dropna=True) == 7
                and set(escenarios_geo_ids.dropna())
                == set(escenarios_ids.dropna())
            )

        registrar(
            auditoria,
            "gpkg_ids_escenarios",
            "geoespacial",
            escenarios_geo_ok,
            (
                gdf_escenarios[geo_escenario_master_col].nunique()
                if geo_escenario_master_col
                else 0
            ),
            7,
        )

        # ---------------------------------------------------------------------
        # Consistencia proyecto -> escenario geográfico
        # ---------------------------------------------------------------------

        if geo_proyecto_col and geo_escenario_col:

            geo_tmp = pd.DataFrame(
                {
                    "proyecto": normalizar_id_serie(
                        gdf_proyectos[geo_proyecto_col]
                    ),
                    "escenario": normalizar_id_serie(
                        gdf_proyectos[geo_escenario_col]
                    ),
                }
            )

            merged = tmp.merge(
                geo_tmp,
                on="proyecto",
                how="outer",
                suffixes=("_modelo", "_geo"),
                indicator=True,
            )

            correspondencia = (
                (merged["_merge"] == "both")
                & (
                    merged["escenario_modelo"]
                    == merged["escenario_geo"]
                )
            )

            coincidencias = int(correspondencia.sum())

            print()
            print(
                f"Asignaciones proyecto -> escenario coincidentes: "
                f"{coincidencias}/{n_proyectos}"
            )

            registrar(
                auditoria,
                "correspondencia_proyecto_escenario_geografica",
                "geoespacial",
                coincidencias == n_proyectos,
                coincidencias,
                n_proyectos,
            )

    except Exception as exc:

        registrar(
            auditoria,
            "lectura_gpkg",
            "geoespacial",
            False,
            str(exc),
            "lectura correcta",
        )

        raise

    # =========================================================================
    # 9. PROCESO 39
    # =========================================================================

    separador("9 - AUDITORÍA DEL PROCESO 39")

    auditoria_39_path = DATA_DIR / (
        "auditoria_39_informe_territorial_amba_v4_1.csv"
    )

    if auditoria_39_path.exists():

        a39 = cargar_csv(auditoria_39_path)

        print(f"Registros auditoría 39: {len(a39)}")

        if "resultado" in a39.columns:

            fallas_39 = int(
                (
                    a39["resultado"]
                    .astype(str)
                    .str.upper()
                    == "FALLA"
                ).sum()
            )

            print(f"Fallas reportadas 39: {fallas_39}")

            registrar(
                auditoria,
                "auditoria_proceso_39",
                "procesos_previos",
                fallas_39 == 0,
                fallas_39,
                0,
            )

        else:

            registrar(
                auditoria,
                "auditoria_proceso_39_estructura",
                "procesos_previos",
                False,
                list(a39.columns),
                "columna resultado",
            )

    else:

        registrar(
            auditoria,
            "auditoria_proceso_39",
            "procesos_previos",
            False,
            "NO EXISTE",
            auditoria_39_path.name,
        )

    # =========================================================================
    # 10. PROCESO 40
    # =========================================================================

    separador("10 - AUDITORÍA DEL PROCESO 40")

    auditoria_40_path = DATA_DIR / (
        "auditoria_40_atlas_territorial_amba.csv"
    )

    if auditoria_40_path.exists():

        a40 = cargar_csv(auditoria_40_path)

        print(f"Registros auditoría 40: {len(a40)}")

        if "resultado" in a40.columns:

            fallas_40 = int(
                (
                    a40["resultado"]
                    .astype(str)
                    .str.upper()
                    == "FALLA"
                ).sum()
            )

            print(f"Fallas reportadas 40: {fallas_40}")

            registrar(
                auditoria,
                "auditoria_proceso_40",
                "procesos_previos",
                fallas_40 == 0,
                fallas_40,
                0,
            )

        else:

            registrar(
                auditoria,
                "auditoria_proceso_40_estructura",
                "procesos_previos",
                False,
                list(a40.columns),
                "columna resultado",
            )

    else:

        registrar(
            auditoria,
            "auditoria_proceso_40",
            "procesos_previos",
            False,
            "NO EXISTE",
            auditoria_40_path.name,
        )

    # =========================================================================
    # 11. CONTROL DE INDICADORES
    # =========================================================================

    separador("11 - CONTROL DE INDICADORES ORIGINALES")

    indicadores_path = DATA_DIR / (
        "indicadores_globales_amba_v4.csv"
    )

    indicadores_ok = False

    if indicadores_path.exists():

        indicadores = cargar_csv(indicadores_path)

        print(
            f"Indicadores globales: "
            f"{len(indicadores)}"
        )

        print(
            f"Columnas: "
            f"{list(indicadores.columns)}"
        )

        indicadores_ok = len(indicadores) > 0

        registrar(
            auditoria,
            "indicadores_globales_existentes",
            "indicadores",
            indicadores_ok,
            len(indicadores),
            "> 0",
        )

        # Verificar que no haya columnas completamente vacías.
        columnas_vacias = [
            c
            for c in indicadores.columns
            if indicadores[c].notna().sum() == 0
        ]

        registrar(
            auditoria,
            "indicadores_columnas_vacias",
            "indicadores",
            len(columnas_vacias) == 0,
            columnas_vacias,
            [],
            severidad="IMPORTANTE",
        )

    else:

        registrar(
            auditoria,
            "indicadores_globales_existentes",
            "indicadores",
            False,
            "NO EXISTE",
            indicadores_path.name,
        )

    # =========================================================================
    # 12. HASHES SHA-256
    # =========================================================================

    separador("12 - GENERANDO HASHES SHA-256 DE PRODUCTOS CRÍTICOS")

    archivos_hash = [
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

    for nombre in archivos_hash:

        path = DATA_DIR / nombre

        if path.exists():

            sha = sha256_file(path)

            hashes.append(
                {
                    "archivo": nombre,
                    "sha256": sha,
                    "tamano_bytes": path.stat().st_size,
                }
            )

            print(
                f"{nombre}: {sha}"
            )

        else:

            hashes.append(
                {
                    "archivo": nombre,
                    "sha256": None,
                    "tamano_bytes": 0,
                }
            )

    registrar(
        auditoria,
        "hashes_productos_criticos",
        "reproducibilidad",
        all(
            x["sha256"] is not None
            for x in hashes
        ),
        len(
            [
                x
                for x in hashes
                if x["sha256"] is not None
            ]
        ),
        len(hashes),
        severidad="IMPORTANTE",
    )

    # =========================================================================
    # 13. CONTROL CRUZADO MODELO ↔ GPKG
    # =========================================================================

    separador("13 - CONTROL CRUZADO MODELO TABULAR ↔ MODELO GEOGRÁFICO")

    if (
        len(gdf_proyectos) == len(proyectos)
        and geo_proyecto_col is not None
    ):

        ids_tabulares = set(
            proyectos_ids.dropna()
        )

        ids_geograficos = set(
            normalizar_id_serie(
                gdf_proyectos[geo_proyecto_col]
            ).dropna()
        )

        faltantes_geo = ids_tabulares - ids_geograficos
        extras_geo = ids_geograficos - ids_tabulares

        print(
            f"IDs tabulares no presentes en GeoPackage: "
            f"{len(faltantes_geo)}"
        )

        print(
            f"IDs geográficos no presentes en modelo: "
            f"{len(extras_geo)}"
        )

        registrar(
            auditoria,
            "modelo_vs_geopackage",
            "integracion",
            len(faltantes_geo) == 0
            and len(extras_geo) == 0,
            {
                "faltantes_geo": len(faltantes_geo),
                "extras_geo": len(extras_geo),
            },
            {
                "faltantes_geo": 0,
                "extras_geo": 0,
            },
        )

    # =========================================================================
    # 14. DETERMINACIÓN DEL DICTAMEN
    # =========================================================================

    separador("14 - DETERMINACIÓN DEL DICTAMEN FINAL")

    ok, total = contar_ok(auditoria)

    fallas_criticas = [
        x
        for x in auditoria
        if x["resultado"] == "FALLA"
        and x["severidad"] == "CRITICO"
    ]

    fallas_importantes = [
        x
        for x in auditoria
        if x["resultado"] == "FALLA"
        and x["severidad"] == "IMPORTANTE"
    ]

    dictamen = (
        "GO"
        if len(fallas_criticas) == 0
        else "NO-GO"
    )

    auditoria_estado = (
        "OK"
        if len(fallas_criticas) == 0
        else "OBSERVADA"
    )

    score = (
        ok / total * 100
        if total
        else 0
    )

    print(f"Controles OK             : {ok}/{total}")
    print(f"Controles fallidos      : {total - ok}")
    print(f"Fallas críticas         : {len(fallas_criticas)}")
    print(f"Fallas importantes      : {len(fallas_importantes)}")
    print(f"Score auditoría         : {score:.2f}/100")
    print(f"Auditoría               : {auditoria_estado}")
    print(f"DICTAMEN FINAL          : {dictamen}")

    # =========================================================================
    # 15. EXPORTAR AUDITORÍA
    # =========================================================================

    separador("15 - EXPORTANDO RESULTADOS DE AUDITORÍA")

    auditoria_df = pd.DataFrame(auditoria)

    auditoria_path = SALIDA / (
        "auditoria_41_modelo_territorial_amba_v4.csv"
    )

    auditoria_df.to_csv(
        auditoria_path,
        index=False,
        encoding="utf-8-sig",
    )

    inventario_df = pd.DataFrame(inventario)

    inventario_path = SALIDA / (
        "inventario_41_productos_amba_v4.csv"
    )

    inventario_df.to_csv(
        inventario_path,
        index=False,
        encoding="utf-8-sig",
    )

    hashes_df = pd.DataFrame(hashes)

    hashes_path = SALIDA / (
        "hashes_41_productos_amba_v4.csv"
    )

    hashes_df.to_csv(
        hashes_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Auditoría  : {auditoria_path}")
    print(f"Inventario : {inventario_path}")
    print(f"Hashes     : {hashes_path}")

    # =========================================================================
    # 16. RESUMEN JSON
    # =========================================================================

    resumen = {
        "proceso": PROCESO,
        "version": VERSION,
        "fecha": pd.Timestamp.now().isoformat(),
        "proyecto": str(PROJECT_ROOT),
        "directorio_datos": str(DATA_DIR),
        "proyectos": int(n_proyectos),
        "proyectos_unicos": int(n_proyectos_unicos),
        "escenarios": int(n_escenarios),
        "escenarios_unicos": int(n_escenarios_unicos),
        "proyectos_nulos": int(n_proyectos_nulos),
        "proyectos_duplicados": int(n_proyectos_dup),
        "escenarios_nulos": int(n_escenarios_nulos),
        "escenarios_duplicados": int(n_escenarios_dup),
        "proyectos_multiescenario": int(
            proyectos_multiescenario
            if escenario_proyecto_col is not None
            else -1
        ),
        "minimo_proyectos_escenario": int(minimo),
        "maximo_proyectos_escenario": int(maximo),
        "promedio_proyectos_escenario": round(
            promedio,
            6,
        ),
        "cv_tamano_escenarios": round(
            cv,
            6,
        ),
        "geometrias_validas": int(
            geom_validas
            if "geom_validas" in locals()
            else 0
        ),
        "geometrias_nulas": int(
            geom_nulas
            if "geom_nulas" in locals()
            else 0
        ),
        "geometrias_vacias": int(
            geom_vacias
            if "geom_vacias" in locals()
            else 0
        ),
        "geometrias_invalidas": int(
            geom_invalidas
            if "geom_invalidas" in locals()
            else 0
        ),
        "cobertura_geometrica": round(
            cobertura
            if "cobertura" in locals()
            else 0,
            4,
        ),
        "controles_ok": int(ok),
        "controles_total": int(total),
        "score_auditoria": round(
            score,
            4,
        ),
        "fallas_criticas": int(
            len(fallas_criticas)
        ),
        "fallas_importantes": int(
            len(fallas_importantes)
        ),
        "auditoria": auditoria_estado,
        "dictamen": dictamen,
        "tiempo_segundos": round(
            time.time() - inicio,
            3,
        ),
    }

    resumen_path = SALIDA / (
        "resumen_41_auditoria_modelo_territorial_amba_v4.json"
    )

    with resumen_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            resumen,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================================
    # 17. INFORME MARKDOWN
    # =========================================================================

    informe_path = SALIDA / (
        "informe_41_auditoria_modelo_territorial_amba_v4.md"
    )

    lineas = []

    lineas.append(
        "# Auditoría Final del Modelo Territorial AMBA V4"
    )
    lineas.append("")
    lineas.append(
        f"**Proceso:** {PROCESO}"
    )
    lineas.append(
        f"**Versión:** {VERSION}"
    )
    lineas.append(
        f"**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lineas.append("")
    lineas.append("---")
    lineas.append("")
    lineas.append("## Dictamen")
    lineas.append("")
    lineas.append(
        f"**DICTAMEN FINAL: {dictamen}**"
    )
    lineas.append("")
    lineas.append(
        f"Auditoría: **{auditoria_estado}**"
    )
    lineas.append("")
    lineas.append(
        f"Score de auditoría: **{score:.2f}/100**"
    )
    lineas.append("")
    lineas.append("---")
    lineas.append("")
    lineas.append("## Resumen estructural")
    lineas.append("")
    lineas.append(
        f"- Proyectos: **{n_proyectos}**"
    )
    lineas.append(
        f"- Proyectos únicos: **{n_proyectos_unicos}**"
    )
    lineas.append(
        f"- Escenarios: **{n_escenarios}**"
    )
    lineas.append(
        f"- Proyectos nulos: **{n_proyectos_nulos}**"
    )
    lineas.append(
        f"- Proyectos duplicados: **{n_proyectos_dup}**"
    )
    lineas.append(
        f"- Proyectos multiescenario: "
        f"**{proyectos_multiescenario}**"
    )
    lineas.append("")
    lineas.append("## Validación geoespacial")
    lineas.append("")
    lineas.append(
        f"- Geometrías válidas: **{geom_validas}**"
    )
    lineas.append(
        f"- Geometrías nulas: **{geom_nulas}**"
    )
    lineas.append(
        f"- Geometrías vacías: **{geom_vacias}**"
    )
    lineas.append(
        f"- Geometrías inválidas: **{geom_invalidas}**"
    )
    lineas.append(
        f"- Cobertura geométrica: **{cobertura:.2f}%**"
    )
    lineas.append("")
    lineas.append("## Distribución territorial")
    lineas.append("")
    lineas.append(
        f"- Mínimo proyectos/escenario: **{minimo}**"
    )
    lineas.append(
        f"- Máximo proyectos/escenario: **{maximo}**"
    )
    lineas.append(
        f"- Promedio: **{promedio:.2f}**"
    )
    lineas.append(
        f"- CV: **{cv:.4f}**"
    )
    lineas.append("")
    lineas.append("## Controles")
    lineas.append("")
    lineas.append(
        f"- Controles OK: **{ok}/{total}**"
    )
    lineas.append(
        f"- Fallas críticas: **{len(fallas_criticas)}**"
    )
    lineas.append(
        f"- Fallas importantes: **{len(fallas_importantes)}**"
    )
    lineas.append("")

    if fallas_criticas:

        lineas.append("## Fallas críticas")
        lineas.append("")

        for falla in fallas_criticas:
            lineas.append(
                f"- **{falla['control']}**: "
                f"{falla['detalle']}"
            )

        lineas.append("")

    if fallas_importantes:

        lineas.append("## Observaciones importantes")
        lineas.append("")

        for falla in fallas_importantes:
            lineas.append(
                f"- **{falla['control']}**: "
                f"{falla['detalle']}"
            )

        lineas.append("")

    lineas.append("## Productos auditados")
    lineas.append("")

    for item in inventario:

        estado = (
            "OK"
            if item["existe"]
            else "FALTANTE"
        )

        lineas.append(
            f"- `{item['archivo']}` — **{estado}**"
        )

    lineas.append("")
    lineas.append("## Hashes SHA-256")
    lineas.append("")

    for item in hashes:

        sha = item["sha256"] or "NO DISPONIBLE"

        lineas.append(
            f"- `{item['archivo']}`"
        )
        lineas.append(
            f"  - SHA-256: `{sha}`"
        )

    lineas.append("")
    lineas.append("---")
    lineas.append("")
    lineas.append(
        "## Conclusión"
    )
    lineas.append("")

    if dictamen == "GO":

        lineas.append(
            "El modelo territorial AMBA V4 supera la auditoría "
            "integral de cierre. Los controles estructurales, "
            "territoriales, geoespaciales, cartográficos y de "
            "integridad de productos resultan satisfactorios."
        )

        lineas.append("")
        lineas.append(
            "El modelo queda certificado para su utilización "
            "como producto territorial consolidado y para la "
            "elaboración de entregables finales."
        )

    else:

        lineas.append(
            "El modelo territorial AMBA V4 presenta controles "
            "fallidos que impiden emitir un dictamen GO."
        )

        lineas.append("")
        lineas.append(
            "Las fallas críticas deben ser corregidas y el "
            "Proceso 41 debe ejecutarse nuevamente."
        )

    informe_path.write_text(
        "\n".join(lineas),
        encoding="utf-8",
    )

    # =========================================================================
    # 18. RESULTADO FINAL
    # =========================================================================

    separador("RESULTADO FINAL DEL PROCESO 41")

    print(f"Proyectos                 : {n_proyectos}")
    print(f"Proyectos únicos          : {n_proyectos_unicos}")
    print(f"Escenarios                : {n_escenarios}")
    print(f"Proyectos multiescenario  : {proyectos_multiescenario}")
    print(
        f"Cobertura geométrica      : "
        f"{cobertura:.2f}%"
    )
    print(
        f"Geometrías válidas        : "
        f"{geom_validas}"
    )
    print(
        f"Geometrías nulas          : "
        f"{geom_nulas}"
    )
    print(
        f"Geometrías inválidas      : "
        f"{geom_invalidas}"
    )
    print(
        f"CV tamaño escenarios      : "
        f"{cv:.4f}"
    )
    print(
        f"Controles OK              : "
        f"{ok}/{total}"
    )
    print(
        f"Fallas críticas           : "
        f"{len(fallas_criticas)}"
    )
    print(
        f"Fallas importantes        : "
        f"{len(fallas_importantes)}"
    )
    print(
        f"Score auditoría           : "
        f"{score:.2f}/100"
    )
    print(
        f"Auditoría                 : "
        f"{auditoria_estado}"
    )
    print(
        f"DICTAMEN FINAL            : "
        f"{dictamen}"
    )
    print(
        f"Tiempo de ejecución       : "
        f"{time.time() - inicio:.2f} segundos"
    )

    print()
    print("=" * 88)
    print("ARCHIVOS GENERADOS")
    print("=" * 88)
    print(f"Auditoría  : {auditoria_path}")
    print(f"Inventario : {inventario_path}")
    print(f"Hashes     : {hashes_path}")
    print(f"Resumen    : {resumen_path}")
    print(f"Informe    : {informe_path}")

    print()
    print("=" * 88)

    if dictamen == "GO":

        print(
            "PROCESO 41 FINALIZADO - MODELO CERTIFICADO"
        )
        print()
        print(
            "El modelo territorial AMBA V4 superó la "
            "auditoría integral."
        )
        print(
            "Los procesos 38, 39 y 40 presentan "
            "consistencia estructural y geoespacial."
        )
        print(
            "El modelo queda en condición GO para cierre."
        )

        return 0

    else:

        print(
            "PROCESO 41 FINALIZADO - NO-GO"
        )
        print()
        print(
            "Se detectaron fallas críticas."
        )
        print(
            "Revisar el informe y ejecutar nuevamente "
            "el proceso después de corregirlas."
        )

        return 1


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    try:

        sys.exit(main())

    except Exception as exc:

        print()
        print("=" * 88)
        print("ERROR FATAL EN EL PROCESO 41")
        print("=" * 88)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise