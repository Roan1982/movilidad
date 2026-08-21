# -*- coding: utf-8 -*-

"""
34 - PRIORIZACIÓN TERRITORIAL DE ESCENARIOS AMBA - V4

Objetivo
--------
Construir la priorización territorial final de los escenarios AMBA a partir
de la síntesis estratégica generada por el proceso 33.

Entrada principal
-----------------
data/processed/escenarios_territoriales_amba/
    escenarios_territoriales_amba_v4.parquet

Entradas auxiliares
-------------------
data/processed/escenarios_territoriales_amba/
    sintesis_estrategica_escenarios_v4.csv
    indicadores_escenarios_v4.csv
    comparacion_escenarios_v4.csv
    proyectos_representativos_escenarios_v4.csv

Principios
----------
- No modifica la asignación proyecto -> escenario.
- No modifica indicadores originales.
- No modifica geometrías.
- No elimina proyectos.
- La priorización se realiza a nivel escenario.
- Mantiene trazabilidad completa.
- Usa reglas determinísticas.
- Normaliza los indicadores antes de combinarlos.
- Produce una clasificación territorial interpretable.
- Separa prioridad analítica de prioridad operativa.
- Genera auditoría y resumen ejecutivo.

Salidas
-------
data/processed/escenarios_territoriales_amba/

    priorizacion_territorial_escenarios_v4.csv
    cartera_priorizada_escenarios_v4.csv
    matriz_priorizacion_escenarios_v4.csv
    proyectos_priorizados_v4.csv
    auditoria_34_escenarios_territoriales_amba.csv
    resumen_34_escenarios_territoriales_amba.json
    priorizacion_territorial_escenarios_v4.gpkg
    sintesis_priorizacion_territorial_v4.md

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
PROCESO = 34

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

INPUT_SINTESIS = (
    INPUT_DIR
    / "sintesis_estrategica_escenarios_v4.csv"
)

INPUT_INDICADORES = (
    INPUT_DIR
    / "indicadores_escenarios_v4.csv"
)

INPUT_COMPARACION = (
    INPUT_DIR
    / "comparacion_escenarios_v4.csv"
)

INPUT_REPRESENTATIVOS = (
    INPUT_DIR
    / "proyectos_representativos_escenarios_v4.csv"
)


OUTPUT_PRIORIZACION = (
    INPUT_DIR
    / "priorizacion_territorial_escenarios_v4.csv"
)

OUTPUT_CARTERA = (
    INPUT_DIR
    / "cartera_priorizada_escenarios_v4.csv"
)

OUTPUT_MATRIZ = (
    INPUT_DIR
    / "matriz_priorizacion_escenarios_v4.csv"
)

OUTPUT_PROYECTOS = (
    INPUT_DIR
    / "proyectos_priorizados_v4.csv"
)

OUTPUT_AUDITORIA = (
    INPUT_DIR
    / "auditoria_34_escenarios_territoriales_amba.csv"
)

OUTPUT_JSON = (
    INPUT_DIR
    / "resumen_34_escenarios_territoriales_amba.json"
)

OUTPUT_GPKG = (
    INPUT_DIR
    / "priorizacion_territorial_escenarios_v4.gpkg"
)

OUTPUT_MD = (
    INPUT_DIR
    / "sintesis_priorizacion_territorial_v4.md"
)


# Rango esperado según la estructura consolidada.
EXPECTED_SCENARIOS_MIN = 6
EXPECTED_SCENARIOS_MAX = 12

MIN_PROJECTS = 8


# ============================================================================
# PESOS DE PRIORIZACIÓN
# ============================================================================

"""
La prioridad territorial se construye con cinco dimensiones:

1. Demanda estructural              25 %
2. Déficit de infraestructura      20 %
3. Conectividad / intermodalidad   20 %
4. Integración / centralidad       20 %
5. Impacto / urgencia              15 %

Los pesos suman 100 %.

La prioridad no reemplaza los indicadores originales.
Es un indicador derivado del proceso 34.
"""

WEIGHTS = {
    "demanda": 0.25,
    "deficit": 0.20,
    "conectividad": 0.20,
    "integracion": 0.20,
    "impacto": 0.15,
}


# ============================================================================
# UMBRALES
# ============================================================================

PRIORITY_THRESHOLDS = {
    "PRIORIDAD_1_CRITICA": 80.0,
    "PRIORIDAD_2_ALTA": 65.0,
    "PRIORIDAD_3_MEDIA": 45.0,
    "PRIORIDAD_4_MEDIA_BAJA": 30.0,
}


# ============================================================================
# UTILIDADES
# ============================================================================

def normalizar_nombre(nombre: Any) -> str:
    """
    Normaliza nombres de columnas para tolerar pequeñas diferencias.
    """

    s = unicodedata.normalize("NFKD", str(nombre))

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
    """
    Resuelve una columna mediante coincidencia exacta o normalizada.
    """

    direct = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidato in candidatos:
        if candidato.lower() in direct:
            return direct[candidato.lower()]

    normalizadas = {
        normalizar_nombre(c): c
        for c in df.columns
    }

    for candidato in candidatos:
        n = normalizar_nombre(candidato)

        if n in normalizadas:
            return normalizadas[n]

    if requerida:
        raise KeyError(
            "No se encontró ninguna de las columnas esperadas: "
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

    if isinstance(v, str):
        return bool(v.strip())

    return True


def convertir_float(v: Any) -> float:
    try:
        x = pd.to_numeric(
            pd.Series([v]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(x):
            return np.nan

        return float(x)

    except Exception:
        return np.nan


def promedio_seguro(
    serie: pd.Series,
) -> float:

    x = pd.to_numeric(
        serie,
        errors="coerce",
    )

    if x.notna().sum() == 0:
        return np.nan

    return float(x.mean())


def mediana_segura(
    serie: pd.Series,
) -> float:

    x = pd.to_numeric(
        serie,
        errors="coerce",
    )

    if x.notna().sum() == 0:
        return np.nan

    return float(x.median())


def normalizar_minmax(
    serie: pd.Series,
) -> pd.Series:
    """
    Normalización 0-100.

    Si todos los valores son iguales, devuelve 50 para evitar
    artificialmente una prioridad 0 o 100.
    """

    x = pd.to_numeric(
        serie,
        errors="coerce",
    )

    validos = x.dropna()

    if validos.empty:
        return pd.Series(
            np.nan,
            index=serie.index,
        )

    minimo = float(validos.min())
    maximo = float(validos.max())

    if math.isclose(minimo, maximo):
        return pd.Series(
            np.where(
                x.notna(),
                50.0,
                np.nan,
            ),
            index=serie.index,
        )

    return (
        (x - minimo)
        / (maximo - minimo)
        * 100.0
    )


def normalizar_rank(
    serie: pd.Series,
) -> pd.Series:
    """
    Rank percentil 0-100.

    Se utiliza como respaldo cuando la escala de un indicador es
    extremadamente heterogénea.
    """

    x = pd.to_numeric(
        serie,
        errors="coerce",
    )

    n = int(x.notna().sum())

    if n <= 1:
        return pd.Series(
            np.where(
                x.notna(),
                100.0,
                np.nan,
            ),
            index=serie.index,
        )

    return (
        x.rank(
            method="average",
            pct=True,
        )
        * 100.0
    )


def elegir_indicador(
    df: pd.DataFrame,
    candidatos: list[str],
):
    return resolver_columna(
        df,
        candidatos,
        requerida=False,
    )


def jsonable(v: Any):
    """
    Convierte objetos NumPy/Pandas a tipos JSON.
    """

    if isinstance(v, (np.integer,)):
        return int(v)

    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None

        return float(v)

    if isinstance(v, np.ndarray):
        return v.tolist()

    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    if isinstance(v, (list, tuple)):
        return [
            jsonable(x)
            for x in v
        ]

    if isinstance(v, dict):
        return {
            str(k): jsonable(val)
            for k, val in v.items()
        }

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    return v


# ============================================================================
# CARGA
# ============================================================================

def cargar_datos():
    print("=" * 88)
    print(
        "34 - PRIORIZACIÓN TERRITORIAL DE ESCENARIOS AMBA - "
        f"{VERSION}"
    )
    print("=" * 88)

    print(f"Proyecto : {BASE_DIR}")
    print(f"Entrada  : {INPUT_PARQUET}")
    print(f"Salida   : {INPUT_DIR}")
    print()

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            "No existe la entrada principal del proceso 34:\n"
            f"{INPUT_PARQUET}"
        )

    print("=" * 88)
    print("CARGANDO SALIDA V4")
    print("=" * 88)

    if gpd is not None:
        try:
            df = gpd.read_parquet(
                INPUT_PARQUET
            )
        except Exception:
            df = pd.read_parquet(
                INPUT_PARQUET
            )
    else:
        df = pd.read_parquet(
            INPUT_PARQUET
        )

    print(f"Registros : {len(df):,}")
    print(f"Columnas  : {len(df.columns):,}")

    if hasattr(df, "crs"):
        print(f"CRS       : {df.crs}")

    print()

    return df


def cargar_auxiliar(
    path: Path,
    nombre: str,
):
    """
    Carga una salida auxiliar del proceso 33 si existe.

    Las auxiliares no son obligatorias porque el parquet V4 contiene
    los indicadores necesarios para la priorización.
    """

    if not path.exists():
        print(
            f"Advertencia: no existe {nombre}: {path}"
        )
        return None

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        print(
            f"{nombre:20}: "
            f"{len(df):,} registros"
        )

        return df

    except Exception as exc:
        print(
            f"Advertencia al cargar {nombre}: {exc}"
        )

        return None


# ============================================================================
# RESOLUCIÓN DE CAMPOS
# ============================================================================

def resolver_campos(
    df: pd.DataFrame,
):
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
        requerida=False,
    )

    campos["dimension"] = resolver_columna(
        df,
        [
            "dimension_dominante",
            "dimension",
        ],
        requerida=False,
    )

    campos["prioridad_original"] = resolver_columna(
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

    # Indicadores.
    campos["demanda"] = elegir_indicador(
        df,
        [
            "indice_demanda_estructural",
            "indice_demanda",
            "score_demanda",
            "demanda",
        ],
    )

    campos["deficit"] = elegir_indicador(
        df,
        [
            "deficit_infraestructura",
            "score_deficit",
            "deficit",
        ],
    )

    campos["conectividad"] = elegir_indicador(
        df,
        [
            "indice_conectividad_estructural",
            "indice_conectividad",
            "score_conectividad",
            "score_alcance",
            "conectividad",
        ],
    )

    campos["intermodalidad"] = elegir_indicador(
        df,
        [
            "indice_intermodalidad_estructural",
            "indice_intermodalidad",
            "score_intermodalidad",
            "intermodalidad",
        ],
    )

    campos["integracion"] = elegir_indicador(
        df,
        [
            "indice_integracion_territorial",
            "indice_integracion",
            "score_integracion",
            "integracion",
        ],
    )

    campos["centralidad"] = elegir_indicador(
        df,
        [
            "indice_centralidad_estructural",
            "indice_centralidad",
            "centralidad",
        ],
    )

    campos["impacto"] = elegir_indicador(
        df,
        [
            "impacto_potencial",
            "score_impacto",
            "impacto",
        ],
    )

    campos["urgencia"] = elegir_indicador(
        df,
        [
            "urgencia_intervencion",
            "score_urgencia",
            "urgencia",
        ],
    )

    campos["prioridad_territorial_original"] = elegir_indicador(
        df,
        [
            "score_prioridad_territorial",
            "prioridad_territorial",
        ],
    )

    campos["score_cartera"] = elegir_indicador(
        df,
        [
            "score_cartera",
        ],
    )

    print("=" * 88)
    print("RESOLUCIÓN DE CAMPOS")
    print("=" * 88)

    for key, value in campos.items():
        print(
            f"{key:35}: {value}"
        )

    print()

    return campos


# ============================================================================
# VALIDACIÓN BASE
# ============================================================================

def validar_base(
    df: pd.DataFrame,
    campos: dict,
):
    proyecto = campos["proyecto"]
    escenario = campos["escenario"]

    errores = []
    advertencias = []

    n = len(df)

    if n == 0:
        errores.append(
            "DATASET_EMPTY"
        )

    null_proyectos = int(
        df[proyecto].isna().sum()
    )

    duplicados = int(
        df[proyecto].duplicated().sum()
    )

    null_escenarios = int(
        df[escenario].isna().sum()
    )

    escenarios = (
        df[escenario]
        .dropna()
        .unique()
        .tolist()
    )

    if not (
        EXPECTED_SCENARIOS_MIN
        <= len(escenarios)
        <= EXPECTED_SCENARIOS_MAX
    ):
        errores.append(
            "SCENARIO_COUNT_OUT_OF_RANGE:"
            f"{len(escenarios)}"
        )

    if null_proyectos:
        errores.append(
            f"PROJECT_ID_NULL:{null_proyectos}"
        )

    if duplicados:
        errores.append(
            f"PROJECT_ID_DUPLICATES:{duplicados}"
        )

    if null_escenarios:
        errores.append(
            f"SCENARIO_ID_NULL:{null_escenarios}"
        )

    counts = (
        df.groupby(escenario)
        .size()
    )

    if not counts.empty:
        if (counts < MIN_PROJECTS).any():
            errores.append(
                "SCENARIO_MIN_PROJECTS:"
                f"{counts[counts < MIN_PROJECTS].to_dict()}"
            )

    print("=" * 88)
    print("VALIDACIÓN BASE DE ENTRADA")
    print("=" * 88)

    print(
        f"Registros              : {n:,}"
    )

    print(
        f"Proyectos únicos       : "
        f"{df[proyecto].nunique():,}"
    )

    print(
        f"Escenarios             : "
        f"{len(escenarios)}"
    )

    print(
        f"Proyecto ID nulos      : "
        f"{null_proyectos}"
    )

    print(
        f"Proyecto ID duplicados : "
        f"{duplicados}"
    )

    print(
        f"Escenario ID nulos     : "
        f"{null_escenarios}"
    )

    print()

    if errores:
        print(
            "Errores detectados:"
        )

        for error in errores:
            print(
                f"  - {error}"
            )

    return {
        "registros": n,
        "proyectos_unicos": int(
            df[proyecto].nunique()
        ),
        "escenarios": len(escenarios),
        "escenarios_ids": [
            str(x)
            for x in sorted(
                escenarios,
                key=str,
            )
        ],
        "errores": errores,
        "advertencias": advertencias,
    }


# ============================================================================
# AGREGACIÓN POR ESCENARIO
# ============================================================================

def agregar_escenarios(
    df: pd.DataFrame,
    campos: dict,
):
    """
    Reduce el dataset proyecto -> escenario a una ficha por escenario.

    Para indicadores cuantitativos se utiliza la mediana, que evita que
    un único proyecto extremo domine la caracterización del escenario.
    """

    escenario = campos["escenario"]

    print("=" * 88)
    print(
        "CONSTRUYENDO INDICADORES AGREGADOS POR ESCENARIO"
    )
    print("=" * 88)

    registros = []

    for escenario_id, grupo in df.groupby(
        escenario,
        sort=True,
        dropna=False,
    ):

        if pd.isna(escenario_id):
            continue

        registro = {
            "escenario_id": escenario_id,
            "cantidad_proyectos": int(
                len(grupo)
            ),
        }

        # ---------------------------------------------------------------
        # Campos conceptuales
        # ---------------------------------------------------------------

        for nombre in [
            "tipo",
            "dimension",
            "prioridad_original",
        ]:

            columna = campos.get(nombre)

            if not columna:
                continue

            valores = (
                grupo[columna]
                .dropna()
                .astype(str)
            )

            if valores.empty:
                registro[
                    f"{nombre}_escenario"
                ] = None

            else:
                registro[
                    f"{nombre}_escenario"
                ] = valores.mode().iloc[0]

        # ---------------------------------------------------------------
        # Indicadores
        # ---------------------------------------------------------------

        indicador_map = {
            "demanda": campos.get("demanda"),
            "deficit": campos.get("deficit"),
            "conectividad": campos.get("conectividad"),
            "intermodalidad": campos.get("intermodalidad"),
            "integracion": campos.get("integracion"),
            "centralidad": campos.get("centralidad"),
            "impacto": campos.get("impacto"),
            "urgencia": campos.get("urgencia"),
            "prioridad_territorial_original":
                campos.get(
                    "prioridad_territorial_original"
                ),
            "score_cartera":
                campos.get("score_cartera"),
        }

        for nombre, columna in indicador_map.items():

            if not columna:
                registro[nombre] = np.nan
                continue

            registro[nombre] = mediana_segura(
                grupo[columna]
            )

        registros.append(
            registro
        )

    resumen = pd.DataFrame(
        registros
    )

    return resumen


# ============================================================================
# CONSTRUCCIÓN DE COMPONENTES
# ============================================================================

def construir_componentes(
    escenarios: pd.DataFrame,
):
    """
    Construye los cinco componentes de la prioridad territorial.

    Cada componente se expresa en escala 0-100.
    """

    out = escenarios.copy()

    # ------------------------------------------------------------------------
    # DEMANDA
    # ------------------------------------------------------------------------

    if out["demanda"].notna().any():

        out["score_demanda_v4"] = (
            normalizar_minmax(
                out["demanda"]
            )
        )

    else:
        out["score_demanda_v4"] = np.nan

    # ------------------------------------------------------------------------
    # DÉFICIT
    # ------------------------------------------------------------------------

    if out["deficit"].notna().any():

        out["score_deficit_v4"] = (
            normalizar_minmax(
                out["deficit"]
            )
        )

    else:
        out["score_deficit_v4"] = np.nan

    # ------------------------------------------------------------------------
    # CONECTIVIDAD + INTERMODALIDAD
    # ------------------------------------------------------------------------

    componentes = []

    if out["conectividad"].notna().any():

        componentes.append(
            normalizar_minmax(
                out["conectividad"]
            )
        )

    if out["intermodalidad"].notna().any():

        componentes.append(
            normalizar_minmax(
                out["intermodalidad"]
            )
        )

    if componentes:

        out["score_conectividad_v4"] = (
            pd.concat(
                componentes,
                axis=1,
            )
            .mean(axis=1)
        )

    else:

        out["score_conectividad_v4"] = np.nan

    # ------------------------------------------------------------------------
    # INTEGRACIÓN + CENTRALIDAD
    # ------------------------------------------------------------------------

    componentes = []

    if out["integracion"].notna().any():

        componentes.append(
            normalizar_minmax(
                out["integracion"]
            )
        )

    if out["centralidad"].notna().any():

        componentes.append(
            normalizar_minmax(
                out["centralidad"]
            )
        )

    if componentes:

        out["score_integracion_v4"] = (
            pd.concat(
                componentes,
                axis=1,
            )
            .mean(axis=1)
        )

    else:

        out["score_integracion_v4"] = np.nan

    # ------------------------------------------------------------------------
    # IMPACTO + URGENCIA
    # ------------------------------------------------------------------------

    componentes = []

    if out["impacto"].notna().any():

        componentes.append(
            normalizar_minmax(
                out["impacto"]
            )
        )

    if out["urgencia"].notna().any():

        componentes.append(
            normalizar_minmax(
                out["urgencia"]
            )
        )

    if componentes:

        out["score_impacto_v4"] = (
            pd.concat(
                componentes,
                axis=1,
            )
            .mean(axis=1)
        )

    else:

        out["score_impacto_v4"] = np.nan

    return out


# ============================================================================
# SCORE FINAL
# ============================================================================

def calcular_score_final(
    df: pd.DataFrame,
):
    """
    Calcula el score analítico territorial V4.

    Si un componente no está disponible, redistribuye proporcionalmente
    el peso entre los componentes disponibles.

    Esto evita introducir ceros artificiales por ausencia de información.
    """

    componentes = {
        "demanda": (
            "score_demanda_v4",
            WEIGHTS["demanda"],
        ),
        "deficit": (
            "score_deficit_v4",
            WEIGHTS["deficit"],
        ),
        "conectividad": (
            "score_conectividad_v4",
            WEIGHTS["conectividad"],
        ),
        "integracion": (
            "score_integracion_v4",
            WEIGHTS["integracion"],
        ),
        "impacto": (
            "score_impacto_v4",
            WEIGHTS["impacto"],
        ),
    }

    scores = []
    pesos_usados = []

    for _, row in df.iterrows():

        suma = 0.0
        peso_total = 0.0

        for _, (
            columna,
            peso,
        ) in componentes.items():

            valor = row[columna]

            if pd.notna(valor):

                suma += (
                    float(valor)
                    * peso
                )

                peso_total += peso

        if peso_total > 0:

            score = (
                suma
                / peso_total
            )

        else:

            score = np.nan

        scores.append(score)
        pesos_usados.append(
            peso_total
        )

    df["score_priorizacion_v4"] = scores
    df["peso_informacion_v4"] = pesos_usados

    return df


# ============================================================================
# CLASIFICACIÓN
# ============================================================================

def clasificar_prioridad(
    score: Any,
):
    if pd.isna(score):
        return "SIN_PRIORIZACION"

    score = float(score)

    if score >= PRIORITY_THRESHOLDS[
        "PRIORIDAD_1_CRITICA"
    ]:
        return "PRIORIDAD_1_CRITICA"

    if score >= PRIORITY_THRESHOLDS[
        "PRIORIDAD_2_ALTA"
    ]:
        return "PRIORIDAD_2_ALTA"

    if score >= PRIORITY_THRESHOLDS[
        "PRIORIDAD_3_MEDIA"
    ]:
        return "PRIORIDAD_3_MEDIA"

    return "PRIORIDAD_4_MEDIA_BAJA"


def prioridad_operativa(
    score: Any,
    ranking: int,
):
    """
    Traduce el score analítico a una categoría operativa.

    Se mantiene separada de la prioridad conceptual original del escenario.
    """

    if pd.isna(score):
        return "SIN_PRIORIZACION"

    if float(score) >= 80:
        return "INTERVENCION_INMEDIATA"

    if float(score) >= 65:
        return "INTERVENCION_PRIORITARIA"

    if float(score) >= 45:
        return "INTERVENCION_PROGRAMADA"

    return "SEGUIMIENTO_TERRITORIAL"


def construir_ranking(
    df: pd.DataFrame,
):
    out = df.copy()

    out = out.sort_values(
        [
            "score_priorizacion_v4",
            "score_impacto_v4",
            "score_demanda_v4",
            "cantidad_proyectos",
            "escenario_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    out["ranking_territorial_v4"] = (
        np.arange(len(out))
        + 1
    )

    out["prioridad_territorial_v4"] = (
        out[
            "score_priorizacion_v4"
        ].apply(
            clasificar_prioridad
        )
    )

    out["prioridad_operativa_v4"] = (
        out.apply(
            lambda row:
            prioridad_operativa(
                row[
                    "score_priorizacion_v4"
                ],
                int(
                    row[
                        "ranking_territorial_v4"
                    ]
                ),
            ),
            axis=1,
        )
    )

    return out


# ============================================================================
# INDICADOR DE COHERENCIA
# ============================================================================

def calcular_coherencia(
    df: pd.DataFrame,
):
    """
    Mide cuán equilibrado es el perfil de cada escenario.

    Un escenario con valores altos en una sola dimensión y muy bajos
    en las restantes recibe una coherencia menor.

    No afecta directamente al score de prioridad.
    Se utiliza para diagnóstico y planificación.
    """

    columnas = [
        "score_demanda_v4",
        "score_deficit_v4",
        "score_conectividad_v4",
        "score_integracion_v4",
        "score_impacto_v4",
    ]

    valores = []

    for _, row in df.iterrows():

        x = pd.to_numeric(
            pd.Series(
                [
                    row[c]
                    for c in columnas
                ]
            ),
            errors="coerce",
        ).dropna()

        if len(x) <= 1:
            valores.append(
                100.0
            )
            continue

        media = float(
            x.mean()
        )

        if math.isclose(
            media,
            0.0,
        ):
            valores.append(
                0.0
            )
            continue

        cv = float(
            x.std(
                ddof=0
            )
            / media
        )

        score = max(
            0.0,
            min(
                100.0,
                100.0
                * (1.0 - cv),
            ),
        )

        valores.append(
            score
        )

    df[
        "coherencia_perfil_v4"
    ] = valores

    return df


# ============================================================================
# CONSTRUCCIÓN DE MATRIZ
# ============================================================================

def construir_matriz(
    df: pd.DataFrame,
):
    columnas = [
        "ranking_territorial_v4",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_escenario",
        "prioridad_original",
        "score_demanda_v4",
        "score_deficit_v4",
        "score_conectividad_v4",
        "score_integracion_v4",
        "score_impacto_v4",
        "score_priorizacion_v4",
        "coherencia_perfil_v4",
        "prioridad_territorial_v4",
        "prioridad_operativa_v4",
    ]

    disponibles = [
        c
        for c in columnas
        if c in df.columns
    ]

    return df[
        disponibles
    ].copy()


# ============================================================================
# PROYECTOS PRIORIZADOS
# ============================================================================

def construir_proyectos_priorizados(
    df_original: pd.DataFrame,
    escenarios: pd.DataFrame,
    campos: dict,
):
    """
    Devuelve nuevamente el nivel proyecto, incorporando exclusivamente
    atributos derivados del escenario.

    No altera los valores originales.
    """

    escenario_col = campos["escenario"]

    columnas_escenario = [
        "escenario_id",
        "ranking_territorial_v4",
        "score_priorizacion_v4",
        "prioridad_territorial_v4",
        "prioridad_operativa_v4",
        "coherencia_perfil_v4",
    ]

    lookup = escenarios[
        columnas_escenario
    ].copy()

    lookup = lookup.rename(
        columns={
            "escenario_id":
                escenario_col
        }
    )

    out = df_original.merge(
        lookup,
        on=escenario_col,
        how="left",
        validate="many_to_one",
    )

    return out


# ============================================================================
# CARTERA
# ============================================================================

def construir_cartera(
    df: pd.DataFrame,
):
    """
    Construye una cartera territorial compacta.

    Categorías:

    P1 - intervención inmediata
    P2 - intervención prioritaria
    P3 - intervención programada
    P4 - seguimiento
    """

    out = df.copy()

    def categoria(score):

        if pd.isna(score):
            return "SIN_PRIORIZACION"

        score = float(score)

        if score >= 80:
            return "P1_INMEDIATA"

        if score >= 65:
            return "P2_PRIORITARIA"

        if score >= 45:
            return "P3_PROGRAMADA"

        return "P4_SEGUIMIENTO"

    out[
        "categoria_cartera_v4"
    ] = out[
        "score_priorizacion_v4"
    ].apply(
        categoria
    )

    def horizonte(categoria):

        if categoria == "P1_INMEDIATA":
            return "0_2_ANIOS"

        if categoria == "P2_PRIORITARIA":
            return "2_4_ANIOS"

        if categoria == "P3_PROGRAMADA":
            return "4_8_ANIOS"

        if categoria == "P4_SEGUIMIENTO":
            return "8_ANIOS_MAS"

        return "SIN_HORIZONTE"

    out[
        "horizonte_intervencion_v4"
    ] = out[
        "categoria_cartera_v4"
    ].apply(
        horizonte
    )

    def accion(categoria):

        if categoria == "P1_INMEDIATA":
            return "EJECUTAR"

        if categoria == "P2_PRIORITARIA":
            return "PROGRAMAR"

        if categoria == "P3_PROGRAMADA":
            return "PREPARAR"

        if categoria == "P4_SEGUIMIENTO":
            return "MONITOREAR"

        return "SIN_ACCION"

    out[
        "accion_recomendada_v4"
    ] = out[
        "categoria_cartera_v4"
    ].apply(
        accion
    )

    return out


# ============================================================================
# AUDITORÍA
# ============================================================================

def construir_auditoria(
    df_original: pd.DataFrame,
    escenarios: pd.DataFrame,
    campos: dict,
    validacion: dict,
):
    registros = []

    proyecto = campos["proyecto"]
    escenario = campos["escenario"]

    # ------------------------------------------------------------------------
    # Integridad proyecto
    # ------------------------------------------------------------------------

    n_original = len(
        df_original
    )

    n_proyectos = int(
        df_original[
            proyecto
        ].nunique()
    )

    n_salida = int(
        escenarios[
            "cantidad_proyectos"
        ].sum()
    )

    integridad = (
        n_original == n_salida
        and n_original == n_proyectos
    )

    registros.append(
        {
            "proceso": PROCESO,
            "control": "INTEGRIDAD_PROYECTOS",
            "resultado": (
                "OK"
                if integridad
                else "ERROR"
            ),
            "valor": n_original,
            "detalle": (
                "La cantidad de proyectos "
                "se conserva."
                if integridad
                else
                "Diferencia en cantidad "
                "de proyectos."
            ),
        }
    )

    # ------------------------------------------------------------------------
    # Integridad escenarios
    # ------------------------------------------------------------------------

    n_escenarios = len(
        escenarios
    )

    cobertura = (
        n_salida
        / n_original
        if n_original
        else 0
    )

    ok_cobertura = (
        math.isclose(
            cobertura,
            1.0,
        )
    )

    registros.append(
        {
            "proceso": PROCESO,
            "control": "COBERTURA_ESCENARIOS",
            "resultado": (
                "OK"
                if ok_cobertura
                else "ERROR"
            ),
            "valor": cobertura,
            "detalle": (
                "Cobertura completa."
                if ok_cobertura
                else
                "Existe pérdida de cobertura."
            ),
        }
    )

    # ------------------------------------------------------------------------
    # Indicadores
    # ------------------------------------------------------------------------

    componentes = [
        "score_demanda_v4",
        "score_deficit_v4",
        "score_conectividad_v4",
        "score_integracion_v4",
        "score_impacto_v4",
    ]

    for columna in componentes:

        disponibles = int(
            escenarios[
                columna
            ].notna().sum()
        )

        registros.append(
            {
                "proceso": PROCESO,
                "control":
                    f"COMPONENTE_{columna}",
                "resultado": (
                    "OK"
                    if disponibles > 0
                    else "ADVERTENCIA"
                ),
                "valor":
                    disponibles,
                "detalle":
                    (
                        "Componente disponible."
                        if disponibles > 0
                        else
                        "No existe información "
                        "para el componente."
                    ),
            }
        )

    # ------------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------------

    scores = escenarios[
        "score_priorizacion_v4"
    ]

    scores_validos = int(
        scores.notna().sum()
    )

    registros.append(
        {
            "proceso": PROCESO,
            "control":
                "SCORE_PRIORIZACION",
            "resultado": (
                "OK"
                if scores_validos
                == n_escenarios
                else "ADVERTENCIA"
            ),
            "valor":
                scores_validos,
            "detalle":
                (
                    "Todos los escenarios "
                    "poseen score."
                    if scores_validos
                    == n_escenarios
                    else
                    "Existen escenarios "
                    "sin score."
                ),
        }
    )

    # ------------------------------------------------------------------------
    # Dictamen
    # ------------------------------------------------------------------------

    errores = [
        r
        for r in registros
        if r["resultado"] == "ERROR"
    ]

    registros.append(
        {
            "proceso": PROCESO,
            "control": "DICTAMEN",
            "resultado": (
                "OK"
                if not errores
                else "ERROR"
            ),
            "valor":
                len(errores),
            "detalle":
                (
                    "Proceso 34 validado."
                    if not errores
                    else
                    "El proceso presenta "
                    "errores estructurales."
                ),
        }
    )

    return pd.DataFrame(
        registros
    )


# ============================================================================
# RESUMEN JSON
# ============================================================================

def construir_resumen(
    df_original: pd.DataFrame,
    escenarios: pd.DataFrame,
    auditoria: pd.DataFrame,
    validacion: dict,
):
    n = len(
        df_original
    )

    proyectos = int(
        df_original[
            "proyecto_id"
        ].nunique()
        if "proyecto_id"
        in df_original.columns
        else n
    )

    counts = escenarios[
        "cantidad_proyectos"
    ]

    if len(counts):

        minimo = int(
            counts.min()
        )

        maximo = int(
            counts.max()
        )

        promedio = float(
            counts.mean()
        )

        desvio = float(
            counts.std(
                ddof=0
            )
        )

        cv = (
            desvio
            / promedio
            if promedio
            else 0
        )

    else:

        minimo = 0
        maximo = 0
        promedio = 0
        desvio = 0
        cv = 0

    score = escenarios[
        "score_priorizacion_v4"
    ]

    score_global = (
        float(
            score.mean()
        )
        if score.notna().any()
        else 0
    )

    mejor = None
    menor = None

    if not escenarios.empty:

        mejor_row = (
            escenarios
            .sort_values(
                "score_priorizacion_v4",
                ascending=False,
            )
            .iloc[0]
        )

        menor_row = (
            escenarios
            .sort_values(
                "score_priorizacion_v4",
                ascending=True,
            )
            .iloc[0]
        )

        mejor = str(
            mejor_row[
                "escenario_id"
            ]
        )

        menor = str(
            menor_row[
                "escenario_id"
            ]
        )

    cartera_counts = (
        escenarios[
            "categoria_cartera_v4"
        ]
        .value_counts()
        .to_dict()
    )

    auditoria_ok = not (
        (
            auditoria[
                "resultado"
            ] == "ERROR"
        ).any()
    )

    return {
        "version": VERSION,
        "proceso": PROCESO,
        "entrada": str(
            INPUT_PARQUET
        ),
        "salidas": {
            "priorizacion":
                str(
                    OUTPUT_PRIORIZACION
                ),
            "cartera":
                str(
                    OUTPUT_CARTERA
                ),
            "matriz":
                str(
                    OUTPUT_MATRIZ
                ),
            "proyectos":
                str(
                    OUTPUT_PROYECTOS
                ),
            "auditoria":
                str(
                    OUTPUT_AUDITORIA
                ),
            "json":
                str(
                    OUTPUT_JSON
                ),
            "gpkg":
                str(
                    OUTPUT_GPKG
                ),
            "markdown":
                str(
                    OUTPUT_MD
                ),
        },
        "registros": n,
        "proyectos_unicos": proyectos,
        "escenarios": len(
            escenarios
        ),
        "cobertura": (
            1.0
            if n
            else 0.0
        ),
        "minimo_proyectos": minimo,
        "maximo_proyectos": maximo,
        "promedio_proyectos": promedio,
        "desvio_proyectos": desvio,
        "cv_tamano": cv,
        "score_analitico_global":
            score_global,
        "mejor_escenario":
            mejor,
        "menor_escenario":
            menor,
        "cartera":
            cartera_counts,
        "auditoria":
            "OK"
            if auditoria_ok
            else "ERROR",
        "dictamen":
            "VALIDADO"
            if auditoria_ok
            else "NO_VALIDADO",
        "validacion_base":
            validacion,
    }


# ============================================================================
# MARKDOWN EJECUTIVO
# ============================================================================

def generar_markdown(
    escenarios: pd.DataFrame,
    resumen: dict,
):
    lineas = []

    lineas.append(
        "# Síntesis de Priorización Territorial AMBA - V4"
    )

    lineas.append("")
    lineas.append(
        "## Proceso 34"
    )

    lineas.append("")

    lineas.append(
        "La presente síntesis constituye la priorización territorial "
        "de los escenarios consolidados en el proceso 33."
    )

    lineas.append("")

    lineas.append(
        "La priorización se construye a partir de cinco dimensiones "
        "analíticas: demanda, déficit de infraestructura, "
        "conectividad/intermodalidad, integración/centralidad e "
        "impacto/urgencia."
    )

    lineas.append("")

    lineas.append(
        "## Resultado general"
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
        f"- Score analítico global: "
        f"**{resumen['score_analitico_global']:.2f}**"
    )

    lineas.append(
        f"- Escenario prioritario: "
        f"**{resumen['mejor_escenario']}**"
    )

    lineas.append(
        f"- Escenario de menor prioridad: "
        f"**{resumen['menor_escenario']}**"
    )

    lineas.append("")

    lineas.append(
        "## Ranking territorial"
    )

    lineas.append("")

    lineas.append(
        "| Rank | Escenario | Proyectos | Score | Prioridad | Cartera |"
    )

    lineas.append(
        "|---:|---|---:|---:|---|---|"
    )

    for _, row in escenarios.iterrows():

        lineas.append(
            "| "
            f"{int(row['ranking_territorial_v4'])} | "
            f"{row['escenario_id']} | "
            f"{int(row['cantidad_proyectos'])} | "
            f"{float(row['score_priorizacion_v4']):.2f} | "
            f"{row['prioridad_territorial_v4']} | "
            f"{row['categoria_cartera_v4']} |"
        )

    lineas.append("")

    lineas.append(
        "## Criterio de priorización"
    )

    lineas.append("")

    lineas.append(
        "- Demanda estructural: **25 %**"
    )

    lineas.append(
        "- Déficit de infraestructura: **20 %**"
    )

    lineas.append(
        "- Conectividad e intermodalidad: **20 %**"
    )

    lineas.append(
        "- Integración y centralidad: **20 %**"
    )

    lineas.append(
        "- Impacto y urgencia: **15 %**"
    )

    lineas.append("")

    lineas.append(
        "## Interpretación"
    )

    lineas.append("")

    if not escenarios.empty:

        top = escenarios.iloc[0]

        lineas.append(
            f"El escenario **{top['escenario_id']}** presenta "
            "la mayor prioridad territorial relativa dentro del "
            "conjunto analizado."
        )

        lineas.append("")

        lineas.append(
            f"Su score de priorización es "
            f"**{float(top['score_priorizacion_v4']):.2f}/100**."
        )

    lineas.append("")

    lineas.append(
        "La clasificación debe utilizarse como herramienta de "
        "ordenamiento territorial y programación de intervenciones, "
        "no como sustituto de la evaluación técnica individual "
        "de cada proyecto."
    )

    lineas.append("")

    lineas.append(
        "## Trazabilidad"
    )

    lineas.append("")

    lineas.append(
        "El proceso 34 no modifica la asignación "
        "proyecto → escenario, los indicadores originales "
        "ni las geometrías."
    )

    lineas.append("")

    lineas.append(
        f"**Dictamen:** {resumen['dictamen']}"
    )

    OUTPUT_MD.write_text(
        "\n".join(lineas),
        encoding="utf-8",
    )


# ============================================================================
# EXPORTACIÓN
# ============================================================================

def exportar(
    df_original: pd.DataFrame,
    escenarios: pd.DataFrame,
    matriz: pd.DataFrame,
    proyectos: pd.DataFrame,
    auditoria: pd.DataFrame,
    resumen: dict,
):
    print("=" * 88)
    print(
        "EXPORTANDO RESULTADOS DEL PROCESO 34"
    )
    print("=" * 88)

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # CSV principal
    # ------------------------------------------------------------------------

    escenarios.to_csv(
        OUTPUT_PRIORIZACION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Priorización : "
        f"{OUTPUT_PRIORIZACION}"
    )

    # ------------------------------------------------------------------------
    # Cartera
    # ------------------------------------------------------------------------

    cartera_columnas = [
        "ranking_territorial_v4",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_escenario",
        "prioridad_original",
        "score_demanda_v4",
        "score_deficit_v4",
        "score_conectividad_v4",
        "score_integracion_v4",
        "score_impacto_v4",
        "score_priorizacion_v4",
        "coherencia_perfil_v4",
        "prioridad_territorial_v4",
        "prioridad_operativa_v4",
        "categoria_cartera_v4",
        "horizonte_intervencion_v4",
        "accion_recomendada_v4",
    ]

    cartera_columnas = [
        c
        for c in cartera_columnas
        if c in escenarios.columns
    ]

    escenarios[
        cartera_columnas
    ].to_csv(
        OUTPUT_CARTERA,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Cartera      : "
        f"{OUTPUT_CARTERA}"
    )

    # ------------------------------------------------------------------------
    # Matriz
    # ------------------------------------------------------------------------

    matriz.to_csv(
        OUTPUT_MATRIZ,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Matriz       : "
        f"{OUTPUT_MATRIZ}"
    )

    # ------------------------------------------------------------------------
    # Proyectos
    # ------------------------------------------------------------------------

    proyectos.to_csv(
        OUTPUT_PROYECTOS,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Proyectos    : "
        f"{OUTPUT_PROYECTOS}"
    )

    # ------------------------------------------------------------------------
    # Auditoría
    # ------------------------------------------------------------------------

    auditoria.to_csv(
        OUTPUT_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Auditoría    : "
        f"{OUTPUT_AUDITORIA}"
    )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            resumen,
            f,
            ensure_ascii=False,
            indent=2,
            default=jsonable,
        )

    print(
        f"Resumen      : "
        f"{OUTPUT_JSON}"
    )

    # ------------------------------------------------------------------------
    # GeoPackage
    # ------------------------------------------------------------------------

    if (
        gpd is not None
        and isinstance(
            df_original,
            gpd.GeoDataFrame,
        )
        and "geometry"
        in df_original.columns
    ):

        try:

            geo = df_original[
                [
                    "geometry"
                ]
            ].copy()

            lookup = escenarios[
                [
                    "escenario_id",
                    "ranking_territorial_v4",
                    "score_priorizacion_v4",
                    "prioridad_territorial_v4",
                    "prioridad_operativa_v4",
                    "categoria_cartera_v4",
                ]
            ].copy()

            proyecto_col = (
                "escenario_id"
                if "escenario_id"
                in df_original.columns
                else None
            )

            if proyecto_col:

                geo = df_original.merge(
                    lookup,
                    on=proyecto_col,
                    how="left",
                    validate="many_to_one",
                )

                geo = gpd.GeoDataFrame(
                    geo,
                    geometry="geometry",
                    crs=df_original.crs,
                )

                # GeoPackage no admite cualquier tipo de objeto
                # arbitrario en columnas; conservamos sólo campos
                # relevantes.
                keep = [
                    c
                    for c in [
                        "proyecto_id",
                        "escenario_id",
                        "ranking_territorial_v4",
                        "score_priorizacion_v4",
                        "prioridad_territorial_v4",
                        "prioridad_operativa_v4",
                        "categoria_cartera_v4",
                        "geometry",
                    ]
                    if c in geo.columns
                ]

                geo[
                    keep
                ].to_file(
                    OUTPUT_GPKG,
                    layer="priorizacion_proyectos",
                    driver="GPKG",
                )

                print(
                    f"GeoPackage   : "
                    f"{OUTPUT_GPKG}"
                )

        except Exception as exc:

            print(
                "Advertencia: no se pudo generar "
                f"GeoPackage: {exc}"
            )

    # ------------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------------

    generar_markdown(
        escenarios,
        resumen,
    )

    print(
        f"Markdown     : "
        f"{OUTPUT_MD}"
    )


# ============================================================================
# VALIDACIÓN FINAL
# ============================================================================

def validar_final(
    df_original: pd.DataFrame,
    escenarios: pd.DataFrame,
    proyectos: pd.DataFrame,
    campos: dict,
):
    errores = []
    advertencias = []

    proyecto = campos["proyecto"]
    escenario = campos["escenario"]

    # ------------------------------------------------------------------------
    # Proyectos
    # ------------------------------------------------------------------------

    if len(
        df_original
    ) != len(
        proyectos
    ):
        errores.append(
            "PROJECT_ROW_COUNT_CHANGED"
        )

    originales = set(
        df_original[
            proyecto
        ]
        .dropna()
        .astype(str)
    )

    finales = set(
        proyectos[
            proyecto
        ]
        .dropna()
        .astype(str)
    )

    if originales != finales:
        errores.append(
            "PROJECT_ID_SET_CHANGED"
        )

    # ------------------------------------------------------------------------
    # Escenarios
    # ------------------------------------------------------------------------

    escenarios_originales = set(
        df_original[
            escenario
        ]
        .dropna()
        .astype(str)
    )

    escenarios_finales = set(
        escenarios[
            "escenario_id"
        ]
        .dropna()
        .astype(str)
    )

    if (
        escenarios_originales
        != escenarios_finales
    ):
        errores.append(
            "SCENARIO_ID_SET_CHANGED"
        )

    # ------------------------------------------------------------------------
    # Cobertura
    # ------------------------------------------------------------------------

    if proyectos[
        "score_priorizacion_v4"
    ].isna().any():

        cantidad = int(
            proyectos[
                "score_priorizacion_v4"
            ].isna().sum()
        )

        advertencias.append(
            "PROJECTS_WITHOUT_PRIORITY:"
            f"{cantidad}"
        )

    # ------------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------------

    ranks = escenarios[
        "ranking_territorial_v4"
    ]

    esperados = list(
        range(
            1,
            len(escenarios) + 1,
        )
    )

    if sorted(
        ranks.astype(int).tolist()
    ) != esperados:

        errores.append(
            "RANKING_NOT_CONTIGUOUS"
        )

    # ------------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------------

    score = escenarios[
        "score_priorizacion_v4"
    ]

    fuera = (
        (
            score < 0
        )
        | (
            score > 100
        )
    )

    if fuera.any():
        errores.append(
            "SCORE_OUT_OF_RANGE"
        )

    # ------------------------------------------------------------------------
    # Resultado
    # ------------------------------------------------------------------------

    dictamen = (
        "VALIDADO"
        if not errores
        else "NO_VALIDADO"
    )

    return {
        "errores": errores,
        "advertencias": advertencias,
        "dictamen": dictamen,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    try:

        # ====================================================================
        # CARGA
        # ====================================================================

        df = cargar_datos()

        # ====================================================================
        # CAMPOS
        # ====================================================================

        campos = resolver_campos(
            df
        )

        # ====================================================================
        # VALIDACIÓN BASE
        # ====================================================================

        validacion_base = validar_base(
            df,
            campos,
        )

        if validacion_base[
            "errores"
        ]:

            print()
            print(
                "=" * 88
            )

            print(
                "ERROR: LA ENTRADA NO SUPERA "
                "LA VALIDACIÓN BASE"
            )

            for error in validacion_base[
                "errores"
            ]:

                print(
                    f"  - {error}"
                )

            return 1

        # ====================================================================
        # AGREGACIÓN
        # ====================================================================

        escenarios = agregar_escenarios(
            df,
            campos,
        )

        # ====================================================================
        # COMPONENTES
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "CONSTRUYENDO COMPONENTES "
            "DE PRIORIZACIÓN TERRITORIAL"
        )

        print(
            "=" * 88
        )

        escenarios = construir_componentes(
            escenarios
        )

        # ====================================================================
        # SCORE
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "CALCULANDO SCORE DE "
            "PRIORIZACIÓN TERRITORIAL"
        )

        print(
            "=" * 88
        )

        escenarios = calcular_score_final(
            escenarios
        )

        # ====================================================================
        # COHERENCIA
        # ====================================================================

        escenarios = calcular_coherencia(
            escenarios
        )

        # ====================================================================
        # RANKING
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "CONSTRUYENDO RANKING "
            "TERRITORIAL"
        )

        print(
            "=" * 88
        )

        escenarios = construir_ranking(
            escenarios
        )

        # ====================================================================
        # CARTERA
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "CONSTRUYENDO CARTERA "
            "TERRITORIAL"
        )

        print(
            "=" * 88
        )

        escenarios = construir_cartera(
            escenarios
        )

        # ====================================================================
        # MATRIZ
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "CONSTRUYENDO MATRIZ "
            "DE PRIORIZACIÓN"
        )

        print(
            "=" * 88
        )

        matriz = construir_matriz(
            escenarios
        )

        # ====================================================================
        # NIVEL PROYECTO
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "PROPAGANDO PRIORIDAD "
            "A NIVEL PROYECTO"
        )

        print(
            "=" * 88
        )

        proyectos = construir_proyectos_priorizados(
            df,
            escenarios,
            campos,
        )

        # ====================================================================
        # AUDITORÍA
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "CONSTRUYENDO AUDITORÍA "
            "DEL PROCESO 34"
        )

        print(
            "=" * 88
        )

        auditoria = construir_auditoria(
            df,
            escenarios,
            campos,
            validacion_base,
        )

        # ====================================================================
        # VALIDACIÓN FINAL
        # ====================================================================

        validacion_final = validar_final(
            df,
            escenarios,
            proyectos,
            campos,
        )

        # ====================================================================
        # RESUMEN
        # ====================================================================

        resumen = construir_resumen(
            df,
            escenarios,
            auditoria,
            validacion_base,
        )

        resumen[
            "validacion_final"
        ] = validacion_final

        resumen[
            "dictamen"
        ] = validacion_final[
            "dictamen"
        ]

        # ====================================================================
        # EXPORTACIÓN
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        exportar(
            df,
            escenarios,
            matriz,
            proyectos,
            auditoria,
            resumen,
        )

        # ====================================================================
        # RESULTADO
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "RESULTADO FINAL "
            "DEL PROCESO 34"
        )

        print(
            "=" * 88
        )

        print(
            f"Proyectos                 : "
            f"{len(df):,}"
        )

        print(
            f"Proyectos únicos          : "
            f"{df[campos['proyecto']].nunique():,}"
        )

        print(
            f"Escenarios                : "
            f"{len(escenarios)}"
        )

        print(
            f"Cobertura                 : "
            f"{resumen['cobertura']:.2%}"
        )

        print(
            f"Mínimo proyectos         : "
            f"{resumen['minimo_proyectos']}"
        )

        print(
            f"Máximo proyectos         : "
            f"{resumen['maximo_proyectos']}"
        )

        print(
            f"CV tamaño                : "
            f"{resumen['cv_tamano']:.4f}"
        )

        print(
            f"Score analítico global    : "
            f"{resumen['score_analitico_global']:.4f}"
        )

        print(
            f"Escenario prioritario     : "
            f"{resumen['mejor_escenario']}"
        )

        print(
            f"Escenario menor prioridad : "
            f"{resumen['menor_escenario']}"
        )

        print(
            f"Auditoría                 : "
            f"{resumen['auditoria']}"
        )

        print(
            f"Dictamen                  : "
            f"{resumen['dictamen']}"
        )

        # ====================================================================
        # RANKING
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        print(
            "RANKING TERRITORIAL"
        )

        print(
            "=" * 88
        )

        mostrar = [
            "ranking_territorial_v4",
            "escenario_id",
            "cantidad_proyectos",
            "tipo_escenario",
            "dimension_escenario",
            "score_demanda_v4",
            "score_deficit_v4",
            "score_conectividad_v4",
            "score_integracion_v4",
            "score_impacto_v4",
            "score_priorizacion_v4",
            "prioridad_territorial_v4",
            "prioridad_operativa_v4",
            "categoria_cartera_v4",
        ]

        mostrar = [
            c
            for c in mostrar
            if c in escenarios.columns
        ]

        print(
            escenarios[
                mostrar
            ].to_string(
                index=False
            )
        )

        # ====================================================================
        # DICTAMEN
        # ====================================================================

        print()
        print(
            "=" * 88
        )

        if (
            resumen[
                "dictamen"
            ]
            == "VALIDADO"
        ):

            print(
                "DICTAMEN FINAL: VALIDADO"
            )

            print()
            print(
                "El proceso 34 construyó la "
                "priorización territorial de los "
                "escenarios AMBA V4."
            )

            print(
                "La asignación proyecto -> escenario "
                "se mantiene íntegra."
            )

            print(
                "Los indicadores originales y las "
                "geometrías no fueron modificados."
            )

            print(
                "La salida queda preparada para "
                "la siguiente etapa de formulación "
                "de cartera e intervención territorial."
            )

            return 0

        print(
            "DICTAMEN FINAL: NO_VALIDADO"
        )

        print()

        print(
            "Errores:"
        )

        for error in (
            validacion_final[
                "errores"
            ]
        ):

            print(
                f"  - {error}"
            )

        return 1

    except Exception as exc:

        print()
        print(
            "=" * 88
        )

        print(
            "ERROR EN PROCESO 34"
        )

        print(
            "=" * 88
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(
        main()
    )