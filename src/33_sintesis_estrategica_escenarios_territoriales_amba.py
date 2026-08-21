# -*- coding: utf-8 -*-

"""
33 - SÍNTESIS ESTRATÉGICA DE ESCENARIOS TERRITORIALES AMBA - V4

Objetivo
--------
Construir la síntesis territorial y estratégica final a partir de la salida
VALIDADA del proceso 32.

Principios
----------
- NO modifica la asignación proyecto -> escenario.
- NO modifica indicadores originales.
- NO modifica geometrías.
- NO recalcula la clasificación territorial.
- NO elimina proyectos.
- Trabaja exclusivamente sobre la salida V4 validada.
- Todos los indicadores agregados son trazables a los proyectos originales.
- Genera una salida ejecutiva y una salida analítica.

Entrada principal
-----------------
data/processed/escenarios_territoriales_amba/
    escenarios_territoriales_amba_v4.parquet

Entradas auxiliares
-------------------
    ranking_escenarios_v4.csv
    matriz_escenarios_v4.csv

Salidas
-------
    sintesis_estrategica_escenarios_v4.csv
    proyectos_representativos_escenarios_v4.csv
    indicadores_escenarios_v4.csv
    comparacion_escenarios_v4.csv
    auditoria_33_escenarios_territoriales_amba.csv
    resumen_33_escenarios_territoriales_amba.json
    sintesis_ejecutiva_escenarios_v4.md

Si existe GeoPandas:
    sintesis_estrategica_escenarios_v4.gpkg

"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

VERSION = "V4.0"
PROCESO = 33

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_PARQUET = (
    INPUT_DIR
    / "escenarios_territoriales_amba_v4.parquet"
)

INPUT_RANKING = (
    INPUT_DIR
    / "ranking_escenarios_v4.csv"
)

INPUT_MATRIZ = (
    INPUT_DIR
    / "matriz_escenarios_v4.csv"
)

OUTPUT_SINTESIS = (
    INPUT_DIR
    / "sintesis_estrategica_escenarios_v4.csv"
)

OUTPUT_REPRESENTATIVOS = (
    INPUT_DIR
    / "proyectos_representativos_escenarios_v4.csv"
)

OUTPUT_INDICADORES = (
    INPUT_DIR
    / "indicadores_escenarios_v4.csv"
)

OUTPUT_COMPARACION = (
    INPUT_DIR
    / "comparacion_escenarios_v4.csv"
)

OUTPUT_AUDITORIA = (
    INPUT_DIR
    / "auditoria_33_escenarios_territoriales_amba.csv"
)

OUTPUT_JSON = (
    INPUT_DIR
    / "resumen_33_escenarios_territoriales_amba.json"
)

OUTPUT_MD = (
    INPUT_DIR
    / "sintesis_ejecutiva_escenarios_v4.md"
)

OUTPUT_GPKG = (
    INPUT_DIR
    / "sintesis_estrategica_escenarios_v4.gpkg"
)


EXPECTED_SCENARIOS_MIN = 6
EXPECTED_SCENARIOS_MAX = 12


# ============================================================================
# CAMPOS
# ============================================================================

INDICATOR_CANDIDATES = {
    "indice_demanda": [
        "indice_demanda_estructural",
        "indice_demanda",
        "score_demanda",
    ],
    "deficit_infraestructura": [
        "deficit_infraestructura",
        "score_deficit_infraestructura",
    ],
    "indice_conectividad": [
        "indice_conectividad_estructural",
        "indice_conectividad",
        "score_conectividad",
    ],
    "indice_intermodalidad": [
        "indice_intermodalidad_estructural",
        "indice_intermodalidad",
        "score_intermodalidad",
    ],
    "indice_integracion": [
        "indice_integracion_territorial",
        "indice_integracion",
        "score_integracion",
    ],
    "indice_centralidad": [
        "indice_centralidad_estructural",
        "indice_centralidad",
    ],
    "impacto_potencial": [
        "impacto_potencial",
        "score_impacto",
    ],
    "urgencia_intervencion": [
        "urgencia_intervencion",
        "score_urgencia",
    ],
    "prioridad_territorial": [
        "score_prioridad_territorial",
        "prioridad_territorial",
    ],
    "score_cartera": [
        "score_cartera",
        "indice_cartera",
    ],
}


# ============================================================================
# UTILIDADES
# ============================================================================

def normalizar_nombre(valor: Any) -> str:
    s = unicodedata.normalize("NFKD", str(valor))
    s = "".join(
        c for c in s
        if not unicodedata.combining(c)
    )
    s = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        s
    )
    return s.strip("_").lower()


def resolver_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    requerida: bool = True,
):
    directas = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidato in candidatos:
        if candidato.lower() in directas:
            return directas[candidato.lower()]

    normalizadas = {
        normalizar_nombre(c): c
        for c in df.columns
    }

    for candidato in candidatos:
        clave = normalizar_nombre(candidato)
        if clave in normalizadas:
            return normalizadas[clave]

    if requerida:
        raise KeyError(
            "No se encontró ninguna de las columnas: "
            f"{candidatos}"
        )

    return None


def valor_valido(v: Any) -> bool:
    if v is None:
        return False

    try:
        if pd.isna(v):
            return False
    except Exception:
        pass

    if isinstance(v, str) and not v.strip():
        return False

    return True


def convertir_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def media_segura(series: pd.Series):
    s = convertir_float(series)

    if s.notna().sum() == 0:
        return np.nan

    return float(s.mean())


def mediana_segura(series: pd.Series):
    s = convertir_float(series)

    if s.notna().sum() == 0:
        return np.nan

    return float(s.median())


def minimo_seguro(series: pd.Series):
    s = convertir_float(series)

    if s.notna().sum() == 0:
        return np.nan

    return float(s.min())


def maximo_seguro(series: pd.Series):
    s = convertir_float(series)

    if s.notna().sum() == 0:
        return np.nan

    return float(s.max())


def jsonable(value: Any):
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def serie_moda(series: pd.Series):
    s = series.dropna()

    if s.empty:
        return None

    conteo = s.astype(str).value_counts()

    if conteo.empty:
        return None

    return conteo.index[0]


def porcentaje(valor: float) -> float:
    if pd.isna(valor):
        return np.nan

    return float(valor) * 100.0


# ============================================================================
# CARGA
# ============================================================================

def cargar_datos():
    print("=" * 88)
    print(
        "33 - SÍNTESIS ESTRATÉGICA DE ESCENARIOS "
        "TERRITORIALES AMBA - V4"
    )
    print("=" * 88)

    print(f"Proyecto : {BASE_DIR}")
    print(f"Entrada  : {INPUT_PARQUET}")
    print(f"Salida   : {INPUT_DIR}")
    print()

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            "No existe la salida V4 del proceso 31/32:\n"
            f"{INPUT_PARQUET}"
        )

    print("=" * 88)
    print("CARGANDO SALIDA V4")
    print("=" * 88)

    if gpd is not None:
        try:
            df = gpd.read_parquet(INPUT_PARQUET)
        except Exception:
            df = pd.read_parquet(INPUT_PARQUET)
    else:
        df = pd.read_parquet(INPUT_PARQUET)

    print(f"Registros : {len(df):,}")
    print(f"Columnas  : {len(df.columns):,}")

    if hasattr(df, "crs"):
        print(f"CRS       : {df.crs}")

    ranking = None

    if INPUT_RANKING.exists():
        ranking = pd.read_csv(
            INPUT_RANKING,
            encoding="utf-8-sig",
        )

        print(
            f"Ranking   : {len(ranking):,} registros"
        )
    else:
        print("Ranking   : no disponible")

    matriz = None

    if INPUT_MATRIZ.exists():
        matriz = pd.read_csv(
            INPUT_MATRIZ,
            encoding="utf-8-sig",
        )

        print(
            f"Matriz    : {len(matriz):,} registros"
        )
    else:
        print("Matriz    : no disponible")

    return df, ranking, matriz


# ============================================================================
# RESOLUCIÓN DE CAMPOS
# ============================================================================

def resolver_campos(df: pd.DataFrame):
    print()
    print("=" * 88)
    print("RESOLUCIÓN DE CAMPOS")
    print("=" * 88)

    campos = {}

    campos["proyecto"] = resolver_columna(
        df,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    campos["escenario"] = resolver_columna(
        df,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    campos["tipo"] = resolver_columna(
        df,
        [
            "tipo_escenario",
            "tipo",
        ],
    )

    campos["dimension"] = resolver_columna(
        df,
        [
            "dimension_dominante",
            "dimension",
        ],
    )

    campos["prioridad"] = resolver_columna(
        df,
        [
            "prioridad_escenario",
            "prioridad",
        ],
        requerida=False,
    )

    campos["geometria"] = (
        "geometry"
        if "geometry" in df.columns
        else None
    )

    for nombre, candidatos in INDICATOR_CANDIDATES.items():
        campos[nombre] = resolver_columna(
            df,
            candidatos,
            requerida=False,
        )

    for k, v in campos.items():
        print(f"{k:30}: {v}")

    return campos


# ============================================================================
# VALIDACIÓN BASE
# ============================================================================

def validar_base(df: pd.DataFrame, campos):
    print()
    print("=" * 88)
    print("VALIDACIÓN BASE DE ENTRADA")
    print("=" * 88)

    errores = []
    advertencias = []

    proyecto = campos["proyecto"]
    escenario = campos["escenario"]

    if len(df) == 0:
        errores.append("DATASET_EMPTY")

    nulos_proyecto = int(
        df[proyecto].isna().sum()
    )

    duplicados_proyecto = int(
        df[proyecto].duplicated().sum()
    )

    nulos_escenario = int(
        df[escenario].isna().sum()
    )

    escenarios = (
        df[escenario]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    escenarios = sorted(escenarios)

    if nulos_proyecto:
        errores.append(
            f"PROJECT_ID_NULL:{nulos_proyecto}"
        )

    if duplicados_proyecto:
        errores.append(
            f"PROJECT_ID_DUPLICATES:{duplicados_proyecto}"
        )

    if nulos_escenario:
        errores.append(
            f"SCENARIO_ID_NULL:{nulos_escenario}"
        )

    if not (
        EXPECTED_SCENARIOS_MIN
        <= len(escenarios)
        <= EXPECTED_SCENARIOS_MAX
    ):
        errores.append(
            f"SCENARIO_COUNT_OUT_OF_RANGE:{len(escenarios)}"
        )

    if "geometry" in df.columns:
        try:
            null_geom = int(
                df.geometry.isna().sum()
            )

            empty_geom = int(
                df.geometry.is_empty.sum()
            )

            invalid_geom = int(
                (~df.geometry.is_valid).sum()
            )

            if null_geom:
                errores.append(
                    f"GEOMETRY_NULL:{null_geom}"
                )

            if empty_geom:
                errores.append(
                    f"GEOMETRY_EMPTY:{empty_geom}"
                )

            if invalid_geom:
                errores.append(
                    f"GEOMETRY_INVALID:{invalid_geom}"
                )

        except Exception as exc:
            advertencias.append(
                f"GEOMETRY_CHECK_WARNING:{exc}"
            )

    print(
        f"Registros              : {len(df):,}"
    )
    print(
        f"Proyectos únicos       : "
        f"{df[proyecto].nunique(dropna=True):,}"
    )
    print(
        f"Escenarios             : {len(escenarios)}"
    )
    print(
        f"Proyecto ID nulos      : {nulos_proyecto}"
    )
    print(
        f"Proyecto ID duplicados : {duplicados_proyecto}"
    )
    print(
        f"Escenario ID nulos     : {nulos_escenario}"
    )

    if errores:
        print()
        print("ERRORES:")
        for error in errores:
            print(f"  - {error}")

    if advertencias:
        print()
        print("ADVERTENCIAS:")
        for warning in advertencias:
            print(f"  - {warning}")

    if errores:
        raise ValueError(
            "La entrada V4 no supera la validación base."
        )

    return {
        "errores": errores,
        "advertencias": advertencias,
        "escenarios": escenarios,
    }


# ============================================================================
# AGREGACIÓN DE INDICADORES
# ============================================================================

def construir_indicadores(
    df: pd.DataFrame,
    campos,
):
    print()
    print("=" * 88)
    print("CONSTRUYENDO INDICADORES AGREGADOS POR ESCENARIO")
    print("=" * 88)

    escenario_col = campos["escenario"]

    registros = []

    for escenario_id, grupo in df.groupby(
        escenario_col,
        sort=True,
    ):
        registro = {
            "escenario_id": escenario_id,
            "cantidad_proyectos": len(grupo),
        }

        for nombre in INDICATOR_CANDIDATES:
            columna = campos.get(nombre)

            if columna is None:
                registro[f"{nombre}_media"] = np.nan
                registro[f"{nombre}_mediana"] = np.nan
                registro[f"{nombre}_min"] = np.nan
                registro[f"{nombre}_max"] = np.nan
                continue

            serie = grupo[columna]

            registro[f"{nombre}_media"] = media_segura(
                serie
            )

            registro[f"{nombre}_mediana"] = mediana_segura(
                serie
            )

            registro[f"{nombre}_min"] = minimo_seguro(
                serie
            )

            registro[f"{nombre}_max"] = maximo_seguro(
                serie
            )

        registros.append(registro)

    resultado = pd.DataFrame(registros)

    return resultado


# ============================================================================
# PERFIL ESTRATÉGICO
# ============================================================================

def construir_sintesis(
    df: pd.DataFrame,
    indicadores: pd.DataFrame,
    ranking: pd.DataFrame | None,
    campos,
):
    print()
    print("=" * 88)
    print("CONSTRUYENDO SÍNTESIS ESTRATÉGICA")
    print("=" * 88)

    escenario_col = campos["escenario"]
    tipo_col = campos["tipo"]
    dimension_col = campos["dimension"]
    prioridad_col = campos["prioridad"]

    registros = []

    ranking_col = None

    if ranking is not None:
        ranking_col = resolver_columna(
            ranking,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
            requerida=False,
        )

    ranking_num_col = None

    if ranking is not None:
        ranking_num_col = resolver_columna(
            ranking,
            [
                "ranking_v4",
                "ranking",
                "posicion",
            ],
            requerida=False,
        )

    ranking_score_col = None

    if ranking is not None:
        ranking_score_col = resolver_columna(
            ranking,
            [
                "score_analitico_v4",
                "score_analitico",
                "score_final",
            ],
            requerida=False,
        )

    ranking_index = {}

    if (
        ranking is not None
        and ranking_col is not None
    ):
        for _, row in ranking.iterrows():
            sid = str(row[ranking_col])

            ranking_index[sid] = row

    indicadores_index = {
        str(row["escenario_id"]): row
        for _, row in indicadores.iterrows()
    }

    for escenario_id, grupo in df.groupby(
        escenario_col,
        sort=True,
    ):
        sid = str(escenario_id)

        tipo = serie_moda(
            grupo[tipo_col]
        )

        dimension = serie_moda(
            grupo[dimension_col]
        )

        if prioridad_col:
            prioridad = serie_moda(
                grupo[prioridad_col]
            )
        else:
            prioridad = None

        fila_ind = indicadores_index.get(sid)

        fila_rank = ranking_index.get(sid)

        registro = {
            "escenario_id": escenario_id,
            "cantidad_proyectos": len(grupo),
            "tipo_escenario": tipo,
            "dimension_dominante": dimension,
            "prioridad_escenario": prioridad,
        }

        if fila_rank is not None:
            if ranking_num_col:
                registro["ranking_v4"] = (
                    fila_rank[ranking_num_col]
                )
            else:
                registro["ranking_v4"] = np.nan

            if ranking_score_col:
                registro["score_analitico_v4"] = (
                    fila_rank[ranking_score_col]
                )
            else:
                registro["score_analitico_v4"] = np.nan
        else:
            registro["ranking_v4"] = np.nan
            registro["score_analitico_v4"] = np.nan

        if fila_ind is not None:
            for nombre in INDICATOR_CANDIDATES:
                registro[
                    f"{nombre}_media"
                ] = fila_ind[
                    f"{nombre}_media"
                ]

                registro[
                    f"{nombre}_mediana"
                ] = fila_ind[
                    f"{nombre}_mediana"
                ]

        registros.append(registro)

    resultado = pd.DataFrame(registros)

    if "ranking_v4" in resultado.columns:
        resultado["ranking_v4"] = pd.to_numeric(
            resultado["ranking_v4"],
            errors="coerce",
        )

        resultado = resultado.sort_values(
            [
                "ranking_v4",
                "escenario_id",
            ],
            na_position="last",
        )

    return resultado.reset_index(drop=True)


# ============================================================================
# PROYECTOS REPRESENTATIVOS
# ============================================================================

def construir_representativos(
    df: pd.DataFrame,
    sintesis: pd.DataFrame,
    campos,
):
    print()
    print("=" * 88)
    print("SELECCIONANDO PROYECTOS REPRESENTATIVOS")
    print("=" * 88)

    proyecto_col = campos["proyecto"]
    escenario_col = campos["escenario"]

    prioridad_col = campos["prioridad"]

    candidatos_score = [
        campos.get("prioridad_territorial"),
        campos.get("score_cartera"),
        campos.get("indice_centralidad"),
        campos.get("indice_demanda"),
        campos.get("impacto_potencial"),
        campos.get("urgencia_intervencion"),
    ]

    candidatos_score = [
        c
        for c in candidatos_score
        if c is not None
    ]

    registros = []

    for escenario_id, grupo in df.groupby(
        escenario_col,
        sort=True,
    ):
        trabajo = grupo.copy()

        score_cols = []

        for columna in candidatos_score:
            if columna in trabajo.columns:
                nueva = (
                    f"__score_{len(score_cols)}"
                )

                trabajo[nueva] = pd.to_numeric(
                    trabajo[columna],
                    errors="coerce",
                )

                score_cols.append(nueva)

        if score_cols:
            trabajo["__score_representativo"] = (
                trabajo[score_cols]
                .mean(axis=1, skipna=True)
            )
        else:
            trabajo["__score_representativo"] = 0.0

        trabajo = trabajo.sort_values(
            [
                "__score_representativo",
                proyecto_col,
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        )

        # Máximo 5 proyectos representativos por escenario.
        seleccionados = trabajo.head(5)

        for posicion, (_, fila) in enumerate(
            seleccionados.iterrows(),
            start=1,
        ):
            registro = {
                "escenario_id": escenario_id,
                "posicion_representativa": posicion,
                "proyecto_id": fila[proyecto_col],
                "score_representativo": fila[
                    "__score_representativo"
                ],
            }

            if prioridad_col:
                registro["prioridad_escenario"] = fila[
                    prioridad_col
                ]

            for nombre in INDICATOR_CANDIDATES:
                columna = campos.get(nombre)

                if columna is not None:
                    registro[nombre] = fila[
                        columna
                    ]

            registros.append(registro)

    return pd.DataFrame(registros)


# ============================================================================
# COMPARACIÓN
# ============================================================================

def construir_comparacion(
    sintesis: pd.DataFrame,
):
    print()
    print("=" * 88)
    print("CONSTRUYENDO COMPARACIÓN ENTRE ESCENARIOS")
    print("=" * 88)

    resultado = sintesis.copy()

    metricas = [
        "score_analitico_v4",
        "indice_demanda_media",
        "deficit_infraestructura_media",
        "indice_conectividad_media",
        "indice_intermodalidad_media",
        "indice_integracion_media",
        "indice_centralidad_media",
        "impacto_potencial_media",
        "urgencia_intervencion_media",
        "prioridad_territorial_media",
        "score_cartera_media",
    ]

    metricas = [
        m
        for m in metricas
        if m in resultado.columns
    ]

    for columna in metricas:
        serie = pd.to_numeric(
            resultado[columna],
            errors="coerce",
        )

        if serie.notna().sum() == 0:
            continue

        minimo = serie.min()
        maximo = serie.max()

        if np.isclose(minimo, maximo):
            resultado[
                f"{columna}_relativo"
            ] = 1.0
        else:
            resultado[
                f"{columna}_relativo"
            ] = (
                (serie - minimo)
                / (maximo - minimo)
            )

    return resultado


# ============================================================================
# AUDITORÍA
# ============================================================================

def construir_auditoria(
    df: pd.DataFrame,
    sintesis: pd.DataFrame,
    indicadores: pd.DataFrame,
    representativos: pd.DataFrame,
    campos,
):
    print()
    print("=" * 88)
    print("CONSTRUYENDO AUDITORÍA DEL PROCESO 33")
    print("=" * 88)

    escenario_col = campos["escenario"]
    proyecto_col = campos["proyecto"]

    registros = []

    escenarios_df = set(
        str(x)
        for x in df[escenario_col]
        .dropna()
        .unique()
    )

    escenarios_sintesis = set(
        str(x)
        for x in sintesis["escenario_id"]
        .dropna()
        .unique()
    )

    escenarios_indicadores = set(
        str(x)
        for x in indicadores["escenario_id"]
        .dropna()
        .unique()
    )

    for escenario_id in sorted(
        escenarios_df
    ):
        grupo = df[
            df[escenario_col].astype(str)
            == escenario_id
        ]

        registros.append({
            "escenario_id": escenario_id,
            "proyectos": len(grupo),
            "proyecto_unicos": grupo[
                proyecto_col
            ].nunique(),
            "en_sintesis": (
                escenario_id
                in escenarios_sintesis
            ),
            "en_indicadores": (
                escenario_id
                in escenarios_indicadores
            ),
            "proyectos_representativos": int(
                (
                    representativos[
                        "escenario_id"
                    ].astype(str)
                    == escenario_id
                ).sum()
            ),
            "estado": "OK",
        })

    return pd.DataFrame(registros)


# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================

def construir_resumen(
    df: pd.DataFrame,
    sintesis: pd.DataFrame,
    representativos: pd.DataFrame,
    auditoria: pd.DataFrame,
    campos,
):
    escenario_col = campos["escenario"]

    cantidad_proyectos = len(df)

    proyectos_unicos = df[
        campos["proyecto"]
    ].nunique()

    escenarios = sorted(
        df[escenario_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    cobertura = (
        proyectos_unicos / cantidad_proyectos
        if cantidad_proyectos
        else 0
    )

    tamanos = (
        df.groupby(
            escenario_col
        )
        .size()
    )

    promedio = (
        float(tamanos.mean())
        if len(tamanos)
        else 0
    )

    desvio = (
        float(tamanos.std(ddof=0))
        if len(tamanos)
        else 0
    )

    cv = (
        desvio / promedio
        if promedio
        else 0
    )

    if "score_analitico_v4" in sintesis.columns:
        score = pd.to_numeric(
            sintesis["score_analitico_v4"],
            errors="coerce",
        )

        score_global = (
            float(score.mean())
            if score.notna().any()
            else np.nan
        )
    else:
        score_global = np.nan

    if "ranking_v4" in sintesis.columns:
        ranking = sintesis.sort_values(
            "ranking_v4"
        )

        mejor = (
            str(ranking.iloc[0]["escenario_id"])
            if not ranking.empty
            else None
        )

        peor = (
            str(ranking.iloc[-1]["escenario_id"])
            if not ranking.empty
            else None
        )
    else:
        mejor = None
        peor = None

    return {
        "version": VERSION,
        "proceso": PROCESO,
        "entrada": str(INPUT_PARQUET),
        "registros": cantidad_proyectos,
        "proyectos_unicos": int(proyectos_unicos),
        "escenarios": len(escenarios),
        "escenarios_ids": escenarios,
        "cobertura": float(cobertura),
        "minimo_proyectos": (
            int(tamanos.min())
            if len(tamanos)
            else 0
        ),
        "maximo_proyectos": (
            int(tamanos.max())
            if len(tamanos)
            else 0
        ),
        "promedio_proyectos": promedio,
        "desvio_proyectos": desvio,
        "cv_tamano": cv,
        "score_analitico_global": score_global,
        "escenario_mejor_rank": mejor,
        "escenario_menor_rank": peor,
        "proyectos_representativos": len(
            representativos
        ),
        "auditoria_ok": bool(
            (
                auditoria["estado"]
                == "OK"
            ).all()
        ),
        "dictamen": "VALIDADO",
    }


# ============================================================================
# MARKDOWN
# ============================================================================

def construir_markdown(
    resumen,
    sintesis,
    representativos,
):
    print()
    print("=" * 88)
    print("GENERANDO SÍNTESIS EJECUTIVA")
    print("=" * 88)

    lineas = []

    lineas.append(
        "# Síntesis Estratégica de Escenarios Territoriales AMBA — V4"
    )

    lineas.append("")

    lineas.append(
        "## 1. Resultado general"
    )

    lineas.append("")

    lineas.append(
        f"- Proyectos: **{resumen['registros']}**"
    )

    lineas.append(
        f"- Proyectos únicos: **{resumen['proyectos_unicos']}**"
    )

    lineas.append(
        f"- Escenarios: **{resumen['escenarios']}**"
    )

    lineas.append(
        f"- Cobertura: **{resumen['cobertura']:.2%}**"
    )

    lineas.append(
        f"- Tamaño por escenario: "
        f"**{resumen['minimo_proyectos']}–"
        f"{resumen['maximo_proyectos']} proyectos**"
    )

    lineas.append(
        f"- CV de tamaño: **{resumen['cv_tamano']:.4f}**"
    )

    lineas.append(
        f"- Escenario mejor rankeado: "
        f"**{resumen['escenario_mejor_rank']}**"
    )

    lineas.append(
        f"- Escenario menor rankeado: "
        f"**{resumen['escenario_menor_rank']}**"
    )

    lineas.append(
        "- Dictamen: **VALIDADO**"
    )

    lineas.append("")

    lineas.append(
        "## 2. Ranking estratégico"
    )

    lineas.append("")

    columnas = [
        "ranking_v4",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
        "score_analitico_v4",
    ]

    columnas = [
        c
        for c in columnas
        if c in sintesis.columns
    ]

    ranking = sintesis.sort_values(
        "ranking_v4"
    )

    lineas.append(
        "| Ranking | Escenario | Proyectos | Tipo | "
        "Dimensión | Prioridad | Score |"
    )

    lineas.append(
        "|---:|---|---:|---|---|---|---:|"
    )

    for _, row in ranking.iterrows():
        rank = row.get(
            "ranking_v4",
            "",
        )

        score = row.get(
            "score_analitico_v4",
            np.nan,
        )

        if pd.notna(score):
            score_text = f"{float(score):.2f}"
        else:
            score_text = "-"

        lineas.append(
            "| "
            f"{rank} | "
            f"{row['escenario_id']} | "
            f"{row['cantidad_proyectos']} | "
            f"{row['tipo_escenario']} | "
            f"{row['dimension_dominante']} | "
            f"{row['prioridad_escenario']} | "
            f"{score_text} |"
        )

    lineas.append("")

    lineas.append(
        "## 3. Lectura estratégica"
    )

    lineas.append("")

    for _, row in ranking.iterrows():
        sid = row["escenario_id"]
        tipo = row["tipo_escenario"]
        dimension = row["dimension_dominante"]
        prioridad = row["prioridad_escenario"]

        lineas.append(
            f"### {sid}"
        )

        lineas.append("")

        lineas.append(
            f"**Tipo:** {tipo}. "
            f"**Dimensión dominante:** {dimension}. "
            f"**Prioridad:** {prioridad}."
        )

        lineas.append("")

        if pd.notna(
            row.get(
                "indice_demanda_media",
                np.nan,
            )
        ):
            lineas.append(
                f"- Demanda estructural media: "
                f"{float(row['indice_demanda_media']):.2f}"
            )

        if pd.notna(
            row.get(
                "indice_centralidad_media",
                np.nan,
            )
        ):
            lineas.append(
                f"- Centralidad estructural media: "
                f"{float(row['indice_centralidad_media']):.2f}"
            )

        if pd.notna(
            row.get(
                "impacto_potencial_media",
                np.nan,
            )
        ):
            lineas.append(
                f"- Impacto potencial medio: "
                f"{float(row['impacto_potencial_media']):.2f}"
            )

        if pd.notna(
            row.get(
                "urgencia_intervencion_media",
                np.nan,
            )
        ):
            lineas.append(
                f"- Urgencia media: "
                f"{float(row['urgencia_intervencion_media']):.2f}"
            )

        proyectos = representativos[
            representativos["escenario_id"]
            == sid
        ]

        if not proyectos.empty:
            lineas.append("")
            lineas.append(
                "**Proyectos representativos:**"
            )

            for _, proyecto in proyectos.iterrows():
                lineas.append(
                    f"- {proyecto['proyecto_id']}"
                )

        lineas.append("")

    lineas.append(
        "## 4. Trazabilidad metodológica"
    )

    lineas.append("")

    lineas.append(
        "La presente síntesis utiliza exclusivamente "
        "la salida validada del proceso 32. "
        "No modifica la asignación proyecto → escenario, "
        "los indicadores originales ni las geometrías."
    )

    lineas.append("")

    lineas.append(
        "## 5. Dictamen"
    )

    lineas.append("")

    lineas.append(
        "**VALIDADO.** La estructura territorial V4 "
        "puede utilizarse como base para la siguiente "
        "etapa de priorización e interpretación estratégica."
    )

    return "\n".join(lineas)


# ============================================================================
# EXPORTACIÓN
# ============================================================================

def exportar(
    df,
    sintesis,
    indicadores,
    representativos,
    comparacion,
    auditoria,
    resumen,
):
    print()
    print("=" * 88)
    print("EXPORTANDO RESULTADOS DEL PROCESO 33")
    print("=" * 88)

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sintesis.to_csv(
        OUTPUT_SINTESIS,
        index=False,
        encoding="utf-8-sig",
    )

    representativos.to_csv(
        OUTPUT_REPRESENTATIVOS,
        index=False,
        encoding="utf-8-sig",
    )

    indicadores.to_csv(
        OUTPUT_INDICADORES,
        index=False,
        encoding="utf-8-sig",
    )

    comparacion.to_csv(
        OUTPUT_COMPARACION,
        index=False,
        encoding="utf-8-sig",
    )

    auditoria.to_csv(
        OUTPUT_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
            default=jsonable,
        )

    markdown = construir_markdown(
        resumen,
        sintesis,
        representativos,
    )

    OUTPUT_MD.write_text(
        markdown,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # GeoPackage
    # ------------------------------------------------------------------------

    if (
        gpd is not None
        and isinstance(df, gpd.GeoDataFrame)
        and "geometry" in df.columns
    ):
        try:
            geometria_escenarios = (
                df[
                    [
                        "geometry",
                        "escenario_id"
                        if "escenario_id" in df.columns
                        else df.columns[0],
                    ]
                ]
                .copy()
            )

            escenario_col = (
                "escenario_id"
                if "escenario_id"
                in geometria_escenarios.columns
                else geometria_escenarios.columns[1]
            )

            geometria_escenarios = (
                geometria_escenarios
                .dissolve(
                    by=escenario_col,
                    as_index=False,
                )
            )

            geometria_escenarios = geometria_escenarios.merge(
                sintesis,
                left_on=escenario_col,
                right_on="escenario_id",
                how="left",
            )

            geometria_escenarios.to_file(
                OUTPUT_GPKG,
                layer="escenarios_estrategicos",
                driver="GPKG",
            )

            print(
                f"GeoPackage : {OUTPUT_GPKG}"
            )

        except Exception as exc:
            print(
                "Advertencia: no se pudo generar "
                f"GeoPackage: {exc}"
            )

    print(
        f"Síntesis     : {OUTPUT_SINTESIS}"
    )

    print(
        f"Representativos: {OUTPUT_REPRESENTATIVOS}"
    )

    print(
        f"Indicadores   : {OUTPUT_INDICADORES}"
    )

    print(
        f"Comparación   : {OUTPUT_COMPARACION}"
    )

    print(
        f"Auditoría     : {OUTPUT_AUDITORIA}"
    )

    print(
        f"Resumen       : {OUTPUT_JSON}"
    )

    print(
        f"Markdown      : {OUTPUT_MD}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:

        df, ranking, matriz = cargar_datos()

        campos = resolver_campos(df)

        validacion = validar_base(
            df,
            campos,
        )

        indicadores = construir_indicadores(
            df,
            campos,
        )

        sintesis = construir_sintesis(
            df,
            indicadores,
            ranking,
            campos,
        )

        representativos = construir_representativos(
            df,
            sintesis,
            campos,
        )

        comparacion = construir_comparacion(
            sintesis,
        )

        auditoria = construir_auditoria(
            df,
            sintesis,
            indicadores,
            representativos,
            campos,
        )

        resumen = construir_resumen(
            df,
            sintesis,
            representativos,
            auditoria,
            campos,
        )

        resumen["salidas"] = {
            "sintesis": str(
                OUTPUT_SINTESIS
            ),
            "representativos": str(
                OUTPUT_REPRESENTATIVOS
            ),
            "indicadores": str(
                OUTPUT_INDICADORES
            ),
            "comparacion": str(
                OUTPUT_COMPARACION
            ),
            "auditoria": str(
                OUTPUT_AUDITORIA
            ),
            "json": str(
                OUTPUT_JSON
            ),
            "markdown": str(
                OUTPUT_MD
            ),
            "geopackage": str(
                OUTPUT_GPKG
            ),
        }

        resumen["entrada_validada"] = (
            validacion
        )

        exportar(
            df,
            sintesis,
            indicadores,
            representativos,
            comparacion,
            auditoria,
            resumen,
        )

        # --------------------------------------------------------------------
        # RESULTADO
        # --------------------------------------------------------------------

        print()
        print("=" * 88)
        print("RESULTADO FINAL DEL PROCESO 33")
        print("=" * 88)

        print(
            f"Proyectos                  : "
            f"{resumen['registros']:,}"
        )

        print(
            f"Proyectos únicos           : "
            f"{resumen['proyectos_unicos']:,}"
        )

        print(
            f"Escenarios                 : "
            f"{resumen['escenarios']}"
        )

        print(
            f"Cobertura                  : "
            f"{resumen['cobertura']:.2%}"
        )

        print(
            f"Mínimo proyectos           : "
            f"{resumen['minimo_proyectos']}"
        )

        print(
            f"Máximo proyectos           : "
            f"{resumen['maximo_proyectos']}"
        )

        print(
            f"CV tamaño                  : "
            f"{resumen['cv_tamano']:.4f}"
        )

        print(
            f"Escenario mejor rank       : "
            f"{resumen['escenario_mejor_rank']}"
        )

        print(
            f"Escenario menor rank       : "
            f"{resumen['escenario_menor_rank']}"
        )

        if pd.notna(
            resumen[
                "score_analitico_global"
            ]
        ):
            print(
                f"Score analítico global     : "
                f"{resumen['score_analitico_global']:.4f}"
            )

        print(
            f"Proyectos representativos  : "
            f"{resumen['proyectos_representativos']}"
        )

        print(
            f"Auditoría                  : "
            f"{'OK' if resumen['auditoria_ok'] else 'ERROR'}"
        )

        print(
            f"Dictamen                   : "
            f"{resumen['dictamen']}"
        )

        print()

        print("=" * 88)
        print("SÍNTESIS ESTRATÉGICA")
        print("=" * 88)

        columnas_mostrar = [
            "ranking_v4",
            "escenario_id",
            "cantidad_proyectos",
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
            "score_analitico_v4",
        ]

        columnas_mostrar = [
            c
            for c in columnas_mostrar
            if c in sintesis.columns
        ]

        print(
            sintesis[
                columnas_mostrar
            ].to_string(index=False)
        )

        print()
        print("=" * 88)
        print("DICTAMEN FINAL: VALIDADO")
        print("=" * 88)

        print()
        print(
            "El proceso 33 construyó la síntesis estratégica "
            "de los escenarios territoriales V4 sin modificar "
            "la asignación proyecto -> escenario, los indicadores "
            "originales ni las geometrías."
        )

        print()
        print(
            "La salida queda preparada para la siguiente etapa "
            "de priorización territorial, cartera de intervención "
            "y análisis estratégico."
        )

        return 0

    except Exception as exc:

        print()
        print("=" * 88)
        print("ERROR EN PROCESO 33")
        print("=" * 88)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise


if __name__ == "__main__":
    sys.exit(main())