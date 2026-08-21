# -*- coding: utf-8 -*-
"""
35_construir_cartera_territorial_amba_v4.py

PROCESO 35
Construcción de cartera territorial de intervención AMBA - V4

Entrada principal:
    data/processed/escenarios_territoriales_amba/proyectos_priorizados_v4.csv

Entrada geográfica:
    data/processed/escenarios_territoriales_amba/
        priorizacion_territorial_escenarios_v4.gpkg

Salidas:
    cartera_territorial_amba_v4.csv
    cartera_escenarios_v4.csv
    cartera_proyectos_v4.csv
    matriz_cartera_territorial_v4.csv
    auditoria_35_cartera_territorial_amba.csv
    resumen_35_cartera_territorial_amba.json
    cartera_territorial_amba_v4.gpkg
    sintesis_cartera_territorial_v4.md

IMPORTANTE:
    Este archivo es el script ejecutable.
    NO genera otro archivo .py.
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
import geopandas as gpd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.0"
PROCESO = "35"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR = INPUT_DIR

PROYECTOS_INPUT = INPUT_DIR / "proyectos_priorizados_v4.csv"

GPKG_INPUT = (
    INPUT_DIR
    / "priorizacion_territorial_escenarios_v4.gpkg"
)

OUTPUT_PROYECTOS = (
    OUTPUT_DIR
    / "cartera_proyectos_v4.csv"
)

OUTPUT_ESCENARIOS = (
    OUTPUT_DIR
    / "cartera_escenarios_v4.csv"
)

OUTPUT_GENERAL = (
    OUTPUT_DIR
    / "cartera_territorial_amba_v4.csv"
)

OUTPUT_MATRIZ = (
    OUTPUT_DIR
    / "matriz_cartera_territorial_v4.csv"
)

OUTPUT_AUDITORIA = (
    OUTPUT_DIR
    / "auditoria_35_cartera_territorial_amba.csv"
)

OUTPUT_RESUMEN = (
    OUTPUT_DIR
    / "resumen_35_cartera_territorial_amba.json"
)

OUTPUT_GPKG = (
    OUTPUT_DIR
    / "cartera_territorial_amba_v4.gpkg"
)

OUTPUT_MARKDOWN = (
    OUTPUT_DIR
    / "sintesis_cartera_territorial_v4.md"
)


# =============================================================================
# UTILIDADES
# =============================================================================

def encabezado(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def normalizar_nombre_columna(nombre: Any) -> str:
    """
    Normaliza un nombre de columna para poder resolver variantes.
    """
    texto = str(nombre).strip().lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        c
        for c in texto
        if not unicodedata.combining(c)
    )

    texto = re.sub(
        r"[^a-z0-9]+",
        "_",
        texto,
    )

    return texto.strip("_")


def resolver_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = True,
) -> str | None:
    """
    Resuelve una columna permitiendo variantes de nombres.
    """

    mapa = {
        normalizar_nombre_columna(c): c
        for c in df.columns
    }

    for candidato in candidatos:
        clave = normalizar_nombre_columna(candidato)

        if clave in mapa:
            return mapa[clave]

    if obligatoria:
        raise KeyError(
            "No se pudo resolver la columna. "
            f"Candidatos: {candidatos}. "
            f"Disponibles: {list(df.columns)}"
        )

    return None


def serie_numerica(
    df: pd.DataFrame,
    columna: str | None,
) -> pd.Series:
    """
    Convierte una columna a numérica.
    Si no existe, devuelve una serie de ceros.
    """

    if columna is None:
        return pd.Series(
            0.0,
            index=df.index,
        )

    return pd.to_numeric(
        df[columna],
        errors="coerce",
    ).fillna(0.0)


def promedio_seguro(
    serie: pd.Series,
) -> float:
    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    return float(valores.mean())


def suma_segura(
    serie: pd.Series,
) -> float:
    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).fillna(0.0)

    return float(valores.sum())


def normalizar_0_100(
    serie: pd.Series,
) -> pd.Series:
    """
    Normalización min-max 0-100.
    """

    x = pd.to_numeric(
        serie,
        errors="coerce",
    ).fillna(0.0)

    minimo = float(x.min())
    maximo = float(x.max())

    if math.isclose(minimo, maximo):
        return pd.Series(
            100.0,
            index=x.index,
        )

    return (
        (x - minimo)
        / (maximo - minimo)
        * 100.0
    )


def porcentaje(
    valor: float,
    total: float,
) -> float:
    if total == 0:
        return 0.0

    return float(valor / total * 100.0)


def texto_seguro(valor: Any) -> str:
    if pd.isna(valor):
        return ""

    return str(valor)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    return str(obj)


# =============================================================================
# RESOLUCIÓN DE CAMPOS
# =============================================================================

def resolver_campos(
    df: pd.DataFrame,
) -> dict[str, str | None]:

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
        obligatoria=False,
    )

    campos["dimension"] = resolver_columna(
        df,
        [
            "dimension_dominante",
            "dimension_escenario",
            "dimension",
        ],
        obligatoria=False,
    )

    campos["prioridad"] = resolver_columna(
        df,
        [
            "prioridad_territorial_v4",
            "prioridad_territorial",
            "prioridad_escenario",
        ],
        obligatoria=False,
    )

    campos["score"] = resolver_columna(
        df,
        [
            "score_priorizacion_v4",
            "score_priorizacion",
            "score_analitico_v4",
        ],
        obligatoria=False,
    )

    campos["demanda"] = resolver_columna(
        df,
        [
            "indice_demanda_estructural",
            "indice_demanda",
            "score_demanda_v4",
        ],
        obligatoria=False,
    )

    campos["deficit"] = resolver_columna(
        df,
        [
            "deficit_infraestructura",
            "score_deficit_v4",
            "deficit",
        ],
        obligatoria=False,
    )

    campos["conectividad"] = resolver_columna(
        df,
        [
            "indice_conectividad_estructural",
            "indice_conectividad",
            "score_conectividad_v4",
        ],
        obligatoria=False,
    )

    campos["intermodalidad"] = resolver_columna(
        df,
        [
            "indice_intermodalidad_estructural",
            "indice_intermodalidad",
            "score_intermodalidad_v4",
        ],
        obligatoria=False,
    )

    campos["integracion"] = resolver_columna(
        df,
        [
            "indice_integracion_territorial",
            "indice_integracion",
            "score_integracion_v4",
        ],
        obligatoria=False,
    )

    campos["centralidad"] = resolver_columna(
        df,
        [
            "indice_centralidad_estructural",
            "indice_centralidad",
        ],
        obligatoria=False,
    )

    campos["impacto"] = resolver_columna(
        df,
        [
            "impacto_potencial",
            "score_impacto_v4",
            "impacto",
        ],
        obligatoria=False,
    )

    campos["urgencia"] = resolver_columna(
        df,
        [
            "urgencia_intervencion",
            "urgencia",
        ],
        obligatoria=False,
    )

    campos["score_cartera"] = resolver_columna(
        df,
        [
            "score_cartera",
        ],
        obligatoria=False,
    )

    campos["geometria"] = resolver_columna(
        df,
        [
            "geometry",
        ],
        obligatoria=False,
    )

    return campos


# =============================================================================
# NORMALIZACIÓN DE ENTRADA
# =============================================================================

def normalizar_entrada(
    df: pd.DataFrame,
    campos: dict[str, str | None],
) -> pd.DataFrame:

    salida = df.copy()

    salida["_proyecto"] = (
        salida[campos["proyecto"]]
        .astype(str)
        .str.strip()
    )

    salida["_escenario"] = (
        salida[campos["escenario"]]
        .astype(str)
        .str.strip()
    )

    if campos["tipo"]:
        salida["_tipo"] = (
            salida[campos["tipo"]]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        salida["_tipo"] = ""

    if campos["dimension"]:
        salida["_dimension"] = (
            salida[campos["dimension"]]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        salida["_dimension"] = ""

    if campos["prioridad"]:
        salida["_prioridad"] = (
            salida[campos["prioridad"]]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        salida["_prioridad"] = ""

    salida["_score"] = serie_numerica(
        salida,
        campos["score"],
    )

    salida["_demanda"] = serie_numerica(
        salida,
        campos["demanda"],
    )

    salida["_deficit"] = serie_numerica(
        salida,
        campos["deficit"],
    )

    salida["_conectividad"] = serie_numerica(
        salida,
        campos["conectividad"],
    )

    salida["_intermodalidad"] = serie_numerica(
        salida,
        campos["intermodalidad"],
    )

    salida["_integracion"] = serie_numerica(
        salida,
        campos["integracion"],
    )

    salida["_centralidad"] = serie_numerica(
        salida,
        campos["centralidad"],
    )

    salida["_impacto"] = serie_numerica(
        salida,
        campos["impacto"],
    )

    salida["_urgencia"] = serie_numerica(
        salida,
        campos["urgencia"],
    )

    salida["_score_cartera_original"] = serie_numerica(
        salida,
        campos["score_cartera"],
    )

    return salida


# =============================================================================
# VALIDACIÓN BASE
# =============================================================================

def validar_base(
    df: pd.DataFrame,
) -> dict[str, Any]:

    total = len(df)

    proyectos_unicos = (
        df["_proyecto"]
        .nunique()
    )

    escenarios = (
        df["_escenario"]
        .nunique()
    )

    nulos_proyecto = int(
        df["_proyecto"]
        .isin(["", "nan", "None"])
        .sum()
    )

    duplicados_proyecto = int(
        df["_proyecto"]
        .duplicated()
        .sum()
    )

    nulos_escenario = int(
        df["_escenario"]
        .isin(["", "nan", "None"])
        .sum()
    )

    return {
        "registros": total,
        "proyectos_unicos": proyectos_unicos,
        "escenarios": escenarios,
        "proyecto_nulos": nulos_proyecto,
        "proyecto_duplicados": duplicados_proyecto,
        "escenario_nulos": nulos_escenario,
    }


# =============================================================================
# CARTERA POR ESCENARIO
# =============================================================================

def construir_cartera_escenarios(
    df: pd.DataFrame,
) -> pd.DataFrame:

    registros = []

    for escenario_id, grupo in df.groupby(
        "_escenario",
        sort=False,
    ):

        cantidad = len(grupo)

        tipo = (
            grupo["_tipo"]
            .mode()
            .iloc[0]
            if not grupo["_tipo"].mode().empty
            else ""
        )

        dimension = (
            grupo["_dimension"]
            .mode()
            .iloc[0]
            if not grupo["_dimension"].mode().empty
            else ""
        )

        prioridad = (
            grupo["_prioridad"]
            .mode()
            .iloc[0]
            if not grupo["_prioridad"].mode().empty
            else ""
        )

        score = promedio_seguro(
            grupo["_score"]
        )

        demanda = promedio_seguro(
            grupo["_demanda"]
        )

        deficit = promedio_seguro(
            grupo["_deficit"]
        )

        conectividad = promedio_seguro(
            grupo["_conectividad"]
        )

        intermodalidad = promedio_seguro(
            grupo["_intermodalidad"]
        )

        integracion = promedio_seguro(
            grupo["_integracion"]
        )

        centralidad = promedio_seguro(
            grupo["_centralidad"]
        )

        impacto = promedio_seguro(
            grupo["_impacto"]
        )

        urgencia = promedio_seguro(
            grupo["_urgencia"]
        )

        score_cartera = promedio_seguro(
            grupo["_score_cartera_original"]
        )

        registros.append(
            {
                "escenario_id": escenario_id,
                "cantidad_proyectos": cantidad,
                "tipo_escenario": tipo,
                "dimension_escenario": dimension,
                "prioridad_original": prioridad,
                "score_analitico": score,
                "demanda_media": demanda,
                "deficit_medio": deficit,
                "conectividad_media": conectividad,
                "intermodalidad_media": intermodalidad,
                "integracion_media": integracion,
                "centralidad_media": centralidad,
                "impacto_medio": impacto,
                "urgencia_media": urgencia,
                "score_cartera_original": score_cartera,
            }
        )

    return pd.DataFrame(registros)


# =============================================================================
# CLASIFICACIÓN TERRITORIAL
# =============================================================================

def clasificar_cartera(
    row: pd.Series,
) -> str:

    prioridad = texto_seguro(
        row["prioridad_original"]
    )

    score = float(
        row["score_analitico"]
    )

    if "PRIORIDAD_1" in prioridad:
        return "P1_ESTRUCTURANTE"

    if "PRIORIDAD_2" in prioridad:
        return "P2_PRIORITARIA"

    if "PRIORIDAD_4" in prioridad:
        return "P4_SEGUIMIENTO"

    if score >= 65:
        return "P2_PRIORITARIA"

    if score >= 40:
        return "P3_PROGRAMADA"

    return "P4_SEGUIMIENTO"


def asignar_linea_estrategica(
    row: pd.Series,
) -> str:

    dimension = texto_seguro(
        row["dimension_escenario"]
    ).upper()

    tipo = texto_seguro(
        row["tipo_escenario"]
    ).upper()

    if "INTEGRACION" in dimension:
        return "INTEGRACION_TERRITORIAL"

    if "CONECTIVIDAD" in dimension:
        return "CONECTIVIDAD_METROPOLITANA"

    if "DEFICIT" in dimension:
        return "REDUCCION_DE_BRECHAS"

    if "IMPACTO" in dimension:
        return "IMPACTO_TERRITORIAL"

    if "ESTRATEGICO" in tipo:
        return "DESARROLLO_ESTRATEGICO"

    return "FORTALECIMIENTO_TERRITORIAL"


def asignar_horizonte(
    categoria: str,
) -> str:

    if categoria == "P1_ESTRUCTURANTE":
        return "CORTO_PLAZO_PRIORITARIO"

    if categoria == "P2_PRIORITARIA":
        return "CORTO_MEDIANO_PLAZO"

    if categoria == "P3_PROGRAMADA":
        return "MEDIANO_PLAZO"

    return "MEDIANO_LARGO_PLAZO"


def asignar_programa(
    linea: str,
) -> str:

    programas = {
        "INTEGRACION_TERRITORIAL":
            "PROGRAMA_INTEGRACION_INTERMODAL",

        "CONECTIVIDAD_METROPOLITANA":
            "PROGRAMA_CONECTIVIDAD_METROPOLITANA",

        "REDUCCION_DE_BRECHAS":
            "PROGRAMA_REDUCCION_DEFICITS",

        "IMPACTO_TERRITORIAL":
            "PROGRAMA_IMPACTO_TERRITORIAL",

        "DESARROLLO_ESTRATEGICO":
            "PROGRAMA_DESARROLLO_ESTRATEGICO",

        "FORTALECIMIENTO_TERRITORIAL":
            "PROGRAMA_FORTALECIMIENTO_TERRITORIAL",
    }

    return programas.get(
        linea,
        "PROGRAMA_FORTALECIMIENTO_TERRITORIAL",
    )


# =============================================================================
# SCORE DE CARTERA
# =============================================================================

def construir_scores(
    cartera: pd.DataFrame,
) -> pd.DataFrame:

    df = cartera.copy()

    df["score_demanda_cartera"] = normalizar_0_100(
        df["demanda_media"]
    )

    df["score_deficit_cartera"] = normalizar_0_100(
        df["deficit_medio"]
    )

    df["score_conectividad_cartera"] = normalizar_0_100(
        df["conectividad_media"]
    )

    df["score_integracion_cartera"] = normalizar_0_100(
        df["integracion_media"]
    )

    df["score_impacto_cartera"] = normalizar_0_100(
        df["impacto_medio"]
    )

    # Ponderación estratégica.
    #
    # Demanda        25%
    # Déficit        20%
    # Conectividad   20%
    # Integración    15%
    # Impacto        20%

    df["score_cartera_v4"] = (
        df["score_demanda_cartera"] * 0.25
        + df["score_deficit_cartera"] * 0.20
        + df["score_conectividad_cartera"] * 0.20
        + df["score_integracion_cartera"] * 0.15
        + df["score_impacto_cartera"] * 0.20
    )

    return df


# =============================================================================
# RANKING
# =============================================================================

def construir_ranking(
    cartera: pd.DataFrame,
) -> pd.DataFrame:

    df = cartera.copy()

    df = df.sort_values(
        [
            "score_cartera_v4",
            "score_analitico",
            "cantidad_proyectos",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    df["ranking_cartera_v4"] = (
        np.arange(len(df)) + 1
    )

    df["categoria_cartera_v4"] = (
        df.apply(
            lambda row:
            clasificar_cartera(row),
            axis=1,
        )
    )

    df["linea_estrategica_v4"] = (
        df.apply(
            lambda row:
            asignar_linea_estrategica(row),
            axis=1,
        )
    )

    df["horizonte_intervencion_v4"] = (
        df["categoria_cartera_v4"]
        .apply(asignar_horizonte)
    )

    df["programa_estrategico_v4"] = (
        df["linea_estrategica_v4"]
        .apply(asignar_programa)
    )

    return df


# =============================================================================
# MATRIZ
# =============================================================================

def construir_matriz(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    columnas = [
        "ranking_cartera_v4",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_escenario",
        "prioridad_original",
        "score_cartera_v4",
        "categoria_cartera_v4",
        "linea_estrategica_v4",
        "horizonte_intervencion_v4",
        "programa_estrategico_v4",
        "score_demanda_cartera",
        "score_deficit_cartera",
        "score_conectividad_cartera",
        "score_integracion_cartera",
        "score_impacto_cartera",
    ]

    columnas = [
        c for c in columnas
        if c in ranking.columns
    ]

    return ranking[columnas].copy()


# =============================================================================
# PROPAGACIÓN A PROYECTOS
# =============================================================================

def propagar_cartera_proyectos(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    columnas_cartera = [
        "escenario_id",
        "ranking_cartera_v4",
        "score_cartera_v4",
        "categoria_cartera_v4",
        "linea_estrategica_v4",
        "horizonte_intervencion_v4",
        "programa_estrategico_v4",
    ]

    cartera_map = ranking[
        columnas_cartera
    ].copy()

    salida = df.merge(
        cartera_map,
        how="left",
        left_on="_escenario",
        right_on="escenario_id",
        validate="many_to_one",
    )

    salida = salida.drop(
        columns=[
            "escenario_id",
        ],
        errors="ignore",
    )

    return salida


# =============================================================================
# AUDITORÍA
# =============================================================================

def construir_auditoria(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    base: dict[str, Any],
) -> pd.DataFrame:

    total = len(df)

    escenarios = len(ranking)

    proyectos_carterizados = int(
        df["score_cartera_v4"]
        .notna()
        .sum()
    )

    escenarios_carterizados = int(
        ranking["score_cartera_v4"]
        .notna()
        .sum()
    )

    duplicados = int(
        df["_proyecto"]
        .duplicated()
        .sum()
    )

    registros = [
        (
            "REGISTROS_ENTRADA",
            total,
            144,
            total == 144,
        ),
        (
            "PROYECTOS_UNICOS",
            base["proyectos_unicos"],
            144,
            base["proyectos_unicos"] == 144,
        ),
        (
            "ESCENARIOS",
            escenarios,
            7,
            escenarios == 7,
        ),
        (
            "PROYECTOS_CARTERIZADOS",
            proyectos_carterizados,
            total,
            proyectos_carterizados == total,
        ),
        (
            "ESCENARIOS_CARTERIZADOS",
            escenarios_carterizados,
            escenarios,
            escenarios_carterizados == escenarios,
        ),
        (
            "DUPLICADOS_PROYECTO",
            duplicados,
            0,
            duplicados == 0,
        ),
    ]

    return pd.DataFrame(
        registros,
        columns=[
            "control",
            "valor",
            "esperado",
            "cumple",
        ],
    )


# =============================================================================
# MARKDOWN
# =============================================================================

def generar_markdown(
    ranking: pd.DataFrame,
    base: dict[str, Any],
    auditoria: pd.DataFrame,
) -> str:

    lineas = []

    lineas.append(
        "# Síntesis de Cartera Territorial AMBA V4"
    )

    lineas.append("")

    lineas.append(
        "## Proceso 35"
    )

    lineas.append("")

    lineas.append(
        "Construcción de cartera territorial de "
        "intervención a partir de los escenarios "
        "territoriales validados V4."
    )

    lineas.append("")

    lineas.append("## Resumen")

    lineas.append("")

    lineas.append(
        f"- Proyectos: {base['registros']}"
    )

    lineas.append(
        f"- Proyectos únicos: {base['proyectos_unicos']}"
    )

    lineas.append(
        f"- Escenarios: {base['escenarios']}"
    )

    lineas.append("")

    lineas.append(
        "## Ranking territorial"
    )

    lineas.append("")

    lineas.append(
        "| Ranking | Escenario | Proyectos | "
        "Score cartera | Categoría | Línea estratégica | "
        "Horizonte | Programa |"
    )

    lineas.append(
        "|---:|---|---:|---:|---|---|---|---|"
    )

    for _, row in ranking.iterrows():

        # IMPORTANTE:
        # El DataFrame de ranking siempre contiene
        # escenario_id.
        #
        # Esto corrige el KeyError que tenía la versión
        # anterior al intentar buscar escenario_id en
        # una tabla que no lo contenía.

        escenario_id = texto_seguro(
            row.get("escenario_id", "")
        )

        ranking_id = texto_seguro(
            row.get("ranking_cartera_v4", "")
        )

        cantidad = int(
            row.get(
                "cantidad_proyectos",
                0,
            )
        )

        score = float(
            row.get(
                "score_cartera_v4",
                0.0,
            )
        )

        categoria = texto_seguro(
            row.get(
                "categoria_cartera_v4",
                "",
            )
        )

        linea = texto_seguro(
            row.get(
                "linea_estrategica_v4",
                "",
            )
        )

        horizonte = texto_seguro(
            row.get(
                "horizonte_intervencion_v4",
                "",
            )
        )

        programa = texto_seguro(
            row.get(
                "programa_estrategico_v4",
                "",
            )
        )

        lineas.append(
            f"| {ranking_id} | "
            f"{escenario_id} | "
            f"{cantidad} | "
            f"{score:.2f} | "
            f"{categoria} | "
            f"{linea} | "
            f"{horizonte} | "
            f"{programa} |"
        )

    lineas.append("")

    lineas.append(
        "## Controles de auditoría"
    )

    lineas.append("")

    for _, row in auditoria.iterrows():

        estado = (
            "OK"
            if bool(row["cumple"])
            else "ERROR"
        )

        lineas.append(
            f"- {row['control']}: "
            f"{row['valor']} / "
            f"esperado {row['esperado']} "
            f"-> {estado}"
        )

    lineas.append("")

    lineas.append(
        "## Dictamen"
    )

    lineas.append("")

    if bool(auditoria["cumple"].all()):
        lineas.append(
            "**VALIDADO**"
        )
    else:
        lineas.append(
            "**OBSERVADO**"
        )

    lineas.append("")

    lineas.append(
        "La cartera se construye sin modificar "
        "la asignación proyecto -> escenario "
        "ni los indicadores originales."
    )

    return "\n".join(lineas)


# =============================================================================
# RESUMEN JSON
# =============================================================================

def construir_resumen(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    auditoria: pd.DataFrame,
) -> dict[str, Any]:

    total = len(df)

    proyectos_unicos = (
        df["_proyecto"]
        .nunique()
    )

    escenarios = len(ranking)

    cobertura = porcentaje(
        int(
            df["score_cartera_v4"]
            .notna()
            .sum()
        ),
        total,
    )

    score_global = promedio_seguro(
        ranking["score_cartera_v4"]
    )

    if not ranking.empty:
        mejor = texto_seguro(
            ranking.iloc[0]["escenario_id"]
        )

        menor = texto_seguro(
            ranking.iloc[-1]["escenario_id"]
        )
    else:
        mejor = ""
        menor = ""

    auditoria_ok = bool(
        auditoria["cumple"].all()
    )

    return {
        "proceso": PROCESO,
        "version": VERSION,
        "proyecto": str(BASE_DIR),
        "registros": total,
        "proyectos_unicos": proyectos_unicos,
        "escenarios": escenarios,
        "cobertura_cartera": cobertura,
        "score_cartera_global": score_global,
        "escenario_prioritario": mejor,
        "escenario_menor_prioridad": menor,
        "auditoria": (
            "OK"
            if auditoria_ok
            else "ERROR"
        ),
        "dictamen": (
            "VALIDADO"
            if auditoria_ok
            else "OBSERVADO"
        ),
        "categorias": (
            ranking[
                "categoria_cartera_v4"
            ]
            .value_counts()
            .to_dict()
            if not ranking.empty
            else {}
        ),
    }


# =============================================================================
# EXPORTACIÓN
# =============================================================================

def exportar_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    salida = df.copy()

    columnas_auxiliares = [
        c
        for c in salida.columns
        if c.startswith("_")
    ]

    salida = salida.drop(
        columns=columnas_auxiliares,
        errors="ignore",
    )

    if "geometry" in salida.columns:
        salida["geometry"] = (
            salida["geometry"]
            .apply(
                lambda g:
                g.wkt
                if hasattr(g, "wkt")
                else ""
            )
        )

    salida.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def exportar_gpkg(
    df: pd.DataFrame,
) -> None:

    if "geometry" not in df.columns:
        return

    geo = df.copy()

    geo = gpd.GeoDataFrame(
        geo,
        geometry="geometry",
        crs="EPSG:4326",
    )

    columnas_auxiliares = [
        c
        for c in geo.columns
        if c.startswith("_")
    ]

    geo = geo.drop(
        columns=columnas_auxiliares,
        errors="ignore",
    )

    # Eliminar objetos que pueden causar problemas
    # de escritura en GPKG.
    for columna in geo.columns:

        if columna == "geometry":
            continue

        if geo[columna].dtype == "object":
            geo[columna] = (
                geo[columna]
                .astype(str)
            )

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    geo.to_file(
        OUTPUT_GPKG,
        layer="cartera_territorial",
        driver="GPKG",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    encabezado(
        "35 - CONSTRUCCIÓN DE CARTERA TERRITORIAL "
        "DE INTERVENCIÓN AMBA - V4"
    )

    print(
        f"Proyecto : {BASE_DIR}"
    )

    print(
        f"Entrada  : {INPUT_DIR}"
    )

    print(
        f"Salida   : {OUTPUT_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # CARGA
    # -------------------------------------------------------------------------

    encabezado(
        "CARGANDO RESULTADOS DEL PROCESO 34"
    )

    if not PROYECTOS_INPUT.exists():
        raise FileNotFoundError(
            "No existe el archivo de entrada:\n"
            f"{PROYECTOS_INPUT}"
        )

    print(
        f"Cargando proyectos: {PROYECTOS_INPUT}"
    )

    df = pd.read_csv(
        PROYECTOS_INPUT,
        encoding="utf-8-sig",
    )

    print(
        f"Registros : {len(df)}"
    )

    print(
        f"Columnas  : {len(df.columns)}"
    )

    # -------------------------------------------------------------------------
    # GEOMETRÍA
    # -------------------------------------------------------------------------

    if GPKG_INPUT.exists():

        print(
            f"GeoPackage: {GPKG_INPUT}"
        )

        try:

            gdf = gpd.read_file(
                GPKG_INPUT,
                layer="priorizacion_territorial",
            )

            print(
                f"Geometrías: {len(gdf)}"
            )

            print(
                f"CRS       : {gdf.crs}"
            )

            campo_geo_proyecto = resolver_columna(
                gdf,
                [
                    "proyecto_id",
                    "id_proyecto",
                    "proyecto",
                ],
                obligatoria=False,
            )

            if (
                campo_geo_proyecto
                and campo_geo_proyecto in gdf.columns
                and "geometry" in gdf.columns
            ):

                columnas_geo = [
                    campo_geo_proyecto,
                    "geometry",
                ]

                geo_aux = (
                    gdf[columnas_geo]
                    .copy()
                    .rename(
                        columns={
                            campo_geo_proyecto:
                            "_geo_proyecto"
                        }
                    )
                )

                campo_proyecto_csv = resolver_columna(
                    df,
                    [
                        "proyecto_id",
                        "id_proyecto",
                        "proyecto",
                    ],
                )

                geo_aux["_geo_proyecto"] = (
                    geo_aux["_geo_proyecto"]
                    .astype(str)
                    .str.strip()
                )

                df[campo_proyecto_csv] = (
                    df[campo_proyecto_csv]
                    .astype(str)
                    .str.strip()
                )

                df = df.merge(
                    geo_aux,
                    how="left",
                    left_on=campo_proyecto_csv,
                    right_on="_geo_proyecto",
                    validate="one_to_one",
                )

                df = gpd.GeoDataFrame(
                    df,
                    geometry="geometry",
                    crs=gdf.crs,
                )

                df = df.drop(
                    columns=[
                        "_geo_proyecto",
                    ],
                    errors="ignore",
                )

        except Exception as exc:

            print(
                "ADVERTENCIA: no se pudo integrar "
                f"la geometría del GeoPackage: {exc}"
            )

    # -------------------------------------------------------------------------
    # RESOLUCIÓN
    # -------------------------------------------------------------------------

    encabezado(
        "RESOLUCIÓN DE CAMPOS"
    )

    campos = resolver_campos(
        df
    )

    etiquetas = {
        "proyecto": "proyecto",
        "escenario": "escenario",
        "tipo": "tipo",
        "dimension": "dimension",
        "prioridad": "prioridad",
        "score": "score",
        "demanda": "demanda",
        "deficit": "deficit",
        "conectividad": "conectividad",
        "intermodalidad": "intermodalidad",
        "integracion": "integracion",
        "centralidad": "centralidad",
        "impacto": "impacto",
        "urgencia": "urgencia",
        "score_cartera": "score_cartera",
        "geometria": "geometria",
    }

    for clave, etiqueta in etiquetas.items():

        print(
            f"{etiqueta:<32}: "
            f"{campos.get(clave)}"
        )

    # -------------------------------------------------------------------------
    # NORMALIZACIÓN
    # -------------------------------------------------------------------------

    encabezado(
        "NORMALIZANDO DATOS DE ENTRADA"
    )

    df = normalizar_entrada(
        df,
        campos,
    )

    # -------------------------------------------------------------------------
    # VALIDACIÓN BASE
    # -------------------------------------------------------------------------

    encabezado(
        "VALIDACIÓN BASE DE ENTRADA"
    )

    base = validar_base(
        df
    )

    print(
        f"Registros              : "
        f"{base['registros']}"
    )

    print(
        f"Proyectos únicos       : "
        f"{base['proyectos_unicos']}"
    )

    print(
        f"Proyectos nulos       : "
        f"{base['proyecto_nulos']}"
    )

    print(
        f"Proyectos duplicados  : "
        f"{base['proyecto_duplicados']}"
    )

    print(
        f"Escenarios             : "
        f"{base['escenarios']}"
    )

    print(
        f"Escenarios nulos       : "
        f"{base['escenario_nulos']}"
    )

    if (
        base["proyectos_unicos"]
        != base["registros"]
    ):
        raise ValueError(
            "La entrada contiene proyectos duplicados."
        )

    # -------------------------------------------------------------------------
    # CARTERA ESCENARIOS
    # -------------------------------------------------------------------------

    encabezado(
        "CONSTRUYENDO CARTERA A NIVEL ESCENARIO"
    )

    cartera = construir_cartera_escenarios(
        df
    )

    # -------------------------------------------------------------------------
    # SCORES
    # -------------------------------------------------------------------------

    encabezado(
        "CALCULANDO SCORE DE CARTERA TERRITORIAL"
    )

    cartera = construir_scores(
        cartera
    )

    # -------------------------------------------------------------------------
    # RANKING
    # -------------------------------------------------------------------------

    encabezado(
        "CONSTRUYENDO RANKING DE CARTERA"
    )

    ranking = construir_ranking(
        cartera
    )

    # -------------------------------------------------------------------------
    # MATRIZ
    # -------------------------------------------------------------------------

    encabezado(
        "CONSTRUYENDO MATRIZ DE CARTERA"
    )

    matriz = construir_matriz(
        ranking
    )

    # -------------------------------------------------------------------------
    # PROPAGACIÓN
    # -------------------------------------------------------------------------

    encabezado(
        "PROPAGANDO CARTERA A NIVEL PROYECTO"
    )

    proyectos = propagar_cartera_proyectos(
        df,
        ranking,
    )

    # -------------------------------------------------------------------------
    # AUDITORÍA
    # -------------------------------------------------------------------------

    encabezado(
        "CONSTRUYENDO AUDITORÍA DEL PROCESO 35"
    )

    auditoria = construir_auditoria(
        proyectos,
        ranking,
        base,
    )

    # -------------------------------------------------------------------------
    # EXPORTACIÓN
    # -------------------------------------------------------------------------

    encabezado(
        "EXPORTANDO RESULTADOS DEL PROCESO 35"
    )

    exportar_csv(
        ranking,
        OUTPUT_ESCENARIOS,
    )

    exportar_csv(
        proyectos,
        OUTPUT_PROYECTOS,
    )

    exportar_csv(
        proyectos,
        OUTPUT_GENERAL,
    )

    exportar_csv(
        matriz,
        OUTPUT_MATRIZ,
    )

    auditoria.to_csv(
        OUTPUT_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    resumen = construir_resumen(
        proyectos,
        ranking,
        auditoria,
    )

    OUTPUT_RESUMEN.write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    markdown = generar_markdown(
        ranking,
        base,
        auditoria,
    )

    OUTPUT_MARKDOWN.write_text(
        markdown,
        encoding="utf-8",
    )

    try:
        exportar_gpkg(
            proyectos
        )

    except Exception as exc:

        print(
            "ADVERTENCIA GPKG: "
            f"{exc}"
        )

    print(
        f"Priorización : {OUTPUT_ESCENARIOS}"
    )

    print(
        f"Cartera      : {OUTPUT_GENERAL}"
    )

    print(
        f"Proyectos    : {OUTPUT_PROYECTOS}"
    )

    print(
        f"Matriz       : {OUTPUT_MATRIZ}"
    )

    print(
        f"Auditoría    : {OUTPUT_AUDITORIA}"
    )

    print(
        f"Resumen      : {OUTPUT_RESUMEN}"
    )

    print(
        f"GeoPackage   : {OUTPUT_GPKG}"
    )

    print(
        f"Markdown     : {OUTPUT_MARKDOWN}"
    )

    # -------------------------------------------------------------------------
    # RESULTADO
    # -------------------------------------------------------------------------

    cobertura = porcentaje(
        int(
            proyectos["score_cartera_v4"]
            .notna()
            .sum()
        ),
        len(proyectos),
    )

    score_global = promedio_seguro(
        ranking["score_cartera_v4"]
    )

    if not ranking.empty:

        mejor = ranking.iloc[0]

        menor = ranking.iloc[-1]

        mejor_id = texto_seguro(
            mejor["escenario_id"]
        )

        menor_id = texto_seguro(
            menor["escenario_id"]
        )

    else:

        mejor_id = ""
        menor_id = ""

    auditoria_ok = bool(
        auditoria["cumple"].all()
    )

    dictamen = (
        "VALIDADO"
        if auditoria_ok
        else "OBSERVADO"
    )

    encabezado(
        "RESULTADO FINAL DEL PROCESO 35"
    )

    print(
        f"Proyectos                 : "
        f"{len(proyectos)}"
    )

    print(
        f"Proyectos únicos          : "
        f"{proyectos['_proyecto'].nunique()}"
    )

    print(
        f"Escenarios                : "
        f"{len(ranking)}"
    )

    print(
        f"Cobertura                 : "
        f"{cobertura:.2f}%"
    )

    print(
        f"Score cartera global      : "
        f"{score_global:.4f}"
    )

    print(
        f"Escenario prioritario     : "
        f"{mejor_id}"
    )

    print(
        f"Escenario menor prioridad : "
        f"{menor_id}"
    )

    print(
        f"Auditoría                 : "
        f"{'OK' if auditoria_ok else 'ERROR'}"
    )

    print(
        f"Dictamen                  : "
        f"{dictamen}"
    )

    encabezado(
        "RANKING DE CARTERA TERRITORIAL"
    )

    columnas_print = [
        "ranking_cartera_v4",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_escenario",
        "score_cartera_v4",
        "categoria_cartera_v4",
        "linea_estrategica_v4",
        "horizonte_intervencion_v4",
    ]

    columnas_print = [
        c
        for c in columnas_print
        if c in ranking.columns
    ]

    print(
        ranking[
            columnas_print
        ].to_string(
            index=False
        )
    )

    encabezado(
        f"DICTAMEN FINAL: {dictamen}"
    )

    if dictamen == "VALIDADO":

        print(
            "El proceso 35 construyó la cartera "
            "territorial de intervención AMBA V4."
        )

        print(
            "La asignación proyecto -> escenario "
            "se mantiene íntegra."
        )

        print(
            "Los indicadores originales "
            "no fueron modificados."
        )

        print(
            "La cartera queda preparada para "
            "la siguiente etapa de programación "
            "de inversiones, cronograma y "
            "formulación del informe final."
        )

    else:

        print(
            "La salida presenta observaciones "
            "en la auditoría del proceso 35."
        )

        print(
            "Revisar el archivo de auditoría "
            "antes de continuar."
        )


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()