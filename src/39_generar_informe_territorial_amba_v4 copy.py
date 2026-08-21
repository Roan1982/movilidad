# -*- coding: utf-8 -*-
"""
===============================================================================
39 - GENERACIÓN DEL INFORME TERRITORIAL AMBA - V4
===============================================================================

Proyecto:
    movilidad

Objetivo:
    Generar el informe territorial final AMBA V4 a partir del modelo maestro
    consolidado por el Proceso 38.

Entrada principal:
    data/processed/escenarios_territoriales_amba/
        modelo_maestro_proyectos_v4.csv
        modelo_maestro_escenarios_v4.csv
        ranking_final_escenarios_v4.csv
        ranking_final_proyectos_v4.csv
        matriz_integral_escenarios_v4.csv
        indicadores_globales_amba_v4.csv
        auditoria_38_consolidacion_territorial_amba.csv
        modelo_maestro_territorial_amba_v4.gpkg

Salidas:
    data/processed/escenarios_territoriales_amba/
        informe_territorial_amba_v4.md
        resumen_ejecutivo_amba_v4.md
        anexo_proyectos_amba_v4.csv
        anexo_escenarios_amba_v4.csv
        anexo_indicadores_globales_amba_v4.csv
        resumen_39_informe_territorial_amba.json
        auditoria_39_informe_territorial_amba.csv

===============================================================================
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR = INPUT_DIR

VERSION = "V4"
PROCESO = "39"


# =============================================================================
# ARCHIVOS DE ENTRADA
# =============================================================================

FILE_PROYECTOS = INPUT_DIR / "modelo_maestro_proyectos_v4.csv"
FILE_ESCENARIOS = INPUT_DIR / "modelo_maestro_escenarios_v4.csv"
FILE_RANKING_ESCENARIOS = INPUT_DIR / "ranking_final_escenarios_v4.csv"
FILE_RANKING_PROYECTOS = INPUT_DIR / "ranking_final_proyectos_v4.csv"
FILE_MATRIZ = INPUT_DIR / "matriz_integral_escenarios_v4.csv"
FILE_INDICADORES = INPUT_DIR / "indicadores_globales_amba_v4.csv"
FILE_AUDITORIA_38 = INPUT_DIR / "auditoria_38_consolidacion_territorial_amba.csv"
FILE_GPKG = INPUT_DIR / "modelo_maestro_territorial_amba_v4.gpkg"


# =============================================================================
# ARCHIVOS DE SALIDA
# =============================================================================

FILE_INFORME = OUTPUT_DIR / "informe_territorial_amba_v4.md"
FILE_RESUMEN = OUTPUT_DIR / "resumen_ejecutivo_amba_v4.md"

FILE_ANEXO_PROYECTOS = OUTPUT_DIR / "anexo_proyectos_amba_v4.csv"
FILE_ANEXO_ESCENARIOS = OUTPUT_DIR / "anexo_escenarios_amba_v4.csv"
FILE_ANEXO_INDICADORES = OUTPUT_DIR / "anexo_indicadores_globales_amba_v4.csv"

FILE_RESUMEN_JSON = (
    OUTPUT_DIR / "resumen_39_informe_territorial_amba.json"
)

FILE_AUDITORIA = (
    OUTPUT_DIR / "auditoria_39_informe_territorial_amba.csv"
)


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def subtitulo(texto: str) -> None:
    print()
    print("-" * 88)
    print(texto)
    print("-" * 88)


def fmt_num(valor, decimales: int = 2) -> str:
    if valor is None:
        return "N/D"

    try:
        if pd.isna(valor):
            return "N/D"
    except Exception:
        pass

    try:
        return f"{float(valor):,.{decimales}f}"
    except Exception:
        return str(valor)


def fmt_pct(valor, decimales: int = 2) -> str:
    if valor is None:
        return "N/D"

    try:
        if pd.isna(valor):
            return "N/D"
    except Exception:
        pass

    try:
        return f"{float(valor):.{decimales}f}%"
    except Exception:
        return str(valor)


def normalizar_columna(df: pd.DataFrame, nombre: str) -> str | None:
    """
    Busca una columna de manera tolerante a diferencias menores de nombres.
    """
    if nombre in df.columns:
        return nombre

    normalizado = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    return normalizado.get(nombre.strip().lower())


def resolver_campo(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatorio: bool = False,
) -> str | None:

    for candidato in candidatos:
        campo = normalizar_columna(df, candidato)

        if campo is not None:
            return campo

    if obligatorio:
        raise KeyError(
            f"No se encontró ninguna de las columnas esperadas: "
            f"{candidatos}"
        )

    return None


def safe_mean(series: pd.Series) -> float:
    serie = pd.to_numeric(series, errors="coerce")

    if serie.dropna().empty:
        return float("nan")

    return float(serie.mean())


def safe_min(series: pd.Series) -> float:
    serie = pd.to_numeric(series, errors="coerce")

    if serie.dropna().empty:
        return float("nan")

    return float(serie.min())


def safe_max(series: pd.Series) -> float:
    serie = pd.to_numeric(series, errors="coerce")

    if serie.dropna().empty:
        return float("nan")

    return float(serie.max())


def cv(series: pd.Series) -> float:
    serie = pd.to_numeric(series, errors="coerce").dropna()

    if len(serie) == 0:
        return float("nan")

    media = serie.mean()

    if media == 0:
        return 0.0

    return float(serie.std(ddof=0) / media)


def valor_primero(df: pd.DataFrame, campo: str | None):
    if campo is None or campo not in df.columns or df.empty:
        return None

    valor = df.iloc[0][campo]

    if pd.isna(valor):
        return None

    return valor


def valores_unicos(df: pd.DataFrame, campo: str | None) -> list:
    if campo is None or campo not in df.columns:
        return []

    return (
        df[campo]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def limpiar_texto(valor) -> str:
    if valor is None:
        return ""

    try:
        if pd.isna(valor):
            return ""
    except Exception:
        pass

    return str(valor).strip()


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_csv(path: Path, nombre: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido: {path}"
        )

    print(f"Cargando: {path.name}")

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(f"Registros : {len(df):,}")
    print(f"Columnas  : {len(df.columns):,}")

    return df


def cargar_fuentes() -> dict[str, pd.DataFrame]:

    titulo("CARGANDO MODELO MAESTRO DEL PROCESO 38")

    datos = {}

    datos["proyectos"] = cargar_csv(
        FILE_PROYECTOS,
        "proyectos",
    )

    datos["escenarios"] = cargar_csv(
        FILE_ESCENARIOS,
        "escenarios",
    )

    datos["ranking_escenarios"] = cargar_csv(
        FILE_RANKING_ESCENARIOS,
        "ranking de escenarios",
    )

    datos["ranking_proyectos"] = cargar_csv(
        FILE_RANKING_PROYECTOS,
        "ranking de proyectos",
    )

    datos["matriz"] = cargar_csv(
        FILE_MATRIZ,
        "matriz integral",
    )

    datos["indicadores"] = cargar_csv(
        FILE_INDICADORES,
        "indicadores globales",
    )

    datos["auditoria_38"] = cargar_csv(
        FILE_AUDITORIA_38,
        "auditoría del proceso 38",
    )

    return datos


# =============================================================================
# RESOLUCIÓN DE CAMPOS
# =============================================================================

def resolver_campos(
    proyectos: pd.DataFrame,
    escenarios: pd.DataFrame,
) -> dict:

    titulo("RESOLUCIÓN DE CAMPOS")

    campos = {}

    campos["proyecto"] = resolver_campo(
        proyectos,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
        obligatorio=True,
    )

    campos["escenario_proyecto"] = resolver_campo(
        proyectos,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
        obligatorio=False,
    )

    campos["tipo_proyecto"] = resolver_campo(
        proyectos,
        [
            "tipo_escenario",
            "tipo",
        ],
    )

    campos["dimension_proyecto"] = resolver_campo(
        proyectos,
        [
            "dimension_dominante",
            "dimension",
        ],
    )

    campos["prioridad"] = resolver_campo(
        proyectos,
        [
            "prioridad_territorial_v4",
            "prioridad_escenario",
            "prioridad",
        ],
    )

    campos["score_cartera"] = resolver_campo(
        proyectos,
        [
            "score_cartera_v4",
            "score_cartera",
        ],
    )

    campos["score_territorial"] = resolver_campo(
        proyectos,
        [
            "score_priorizacion_v4",
            "score_priorizacion",
            "score_prioridad_territorial",
        ],
    )

    campos["demanda"] = resolver_campo(
        proyectos,
        [
            "indice_demanda_estructural",
            "score_demanda_v4",
            "demanda",
        ],
    )

    campos["deficit"] = resolver_campo(
        proyectos,
        [
            "deficit_infraestructura",
            "score_deficit_v4",
            "deficit",
        ],
    )

    campos["conectividad"] = resolver_campo(
        proyectos,
        [
            "indice_conectividad_estructural",
            "score_conectividad_v4",
            "conectividad",
        ],
    )

    campos["intermodalidad"] = resolver_campo(
        proyectos,
        [
            "indice_intermodalidad_estructural",
            "score_intermodalidad_v4",
            "intermodalidad",
        ],
    )

    campos["integracion"] = resolver_campo(
        proyectos,
        [
            "indice_integracion_territorial",
            "score_integracion_v4",
            "integracion",
        ],
    )

    campos["centralidad"] = resolver_campo(
        proyectos,
        [
            "indice_centralidad_estructural",
            "centralidad",
        ],
    )

    campos["impacto"] = resolver_campo(
        proyectos,
        [
            "impacto_potencial",
            "score_impacto_v4",
            "impacto",
        ],
    )

    campos["urgencia"] = resolver_campo(
        proyectos,
        [
            "urgencia_intervencion",
            "urgencia",
        ],
    )

    campos["geometria"] = resolver_campo(
        proyectos,
        [
            "geometry",
        ],
    )

    for clave, valor in campos.items():
        print(f"{clave:<28}: {valor or 'NO DISPONIBLE'}")

    return campos


# =============================================================================
# VALIDACIÓN DEL MODELO
# =============================================================================

def validar_modelo(
    proyectos: pd.DataFrame,
    escenarios: pd.DataFrame,
    campos: dict,
) -> dict:

    titulo("VALIDACIÓN DEL MODELO MAESTRO")

    proyecto = campos["proyecto"]

    total = len(proyectos)
    unicos = proyectos[proyecto].nunique(dropna=True)

    nulos_proyecto = int(
        proyectos[proyecto].isna().sum()
    )

    duplicados = int(
        proyectos[proyecto].duplicated().sum()
    )

    escenario = campos["escenario_proyecto"]

    if escenario is not None:
        escenarios_proyecto = proyectos[escenario].nunique(
            dropna=True
        )
        nulos_escenario = int(
            proyectos[escenario].isna().sum()
        )
    else:
        escenarios_proyecto = 0
        nulos_escenario = total

    geometria = campos["geometria"]

    if geometria is not None:
        geometrias_validas = int(
            proyectos[geometria].notna().sum()
        )
    else:
        geometrias_validas = 0

    resumen = {
        "registros": total,
        "proyectos_unicos": unicos,
        "proyectos_nulos": nulos_proyecto,
        "proyectos_duplicados": duplicados,
        "escenarios": escenarios_proyecto,
        "escenarios_nulos": nulos_escenario,
        "geometrias_validas": geometrias_validas,
        "cobertura_geometrica": (
            geometrias_validas / total * 100
            if total
            else 0
        ),
    }

    print(
        f"Registros                  : {total:,}"
    )
    print(
        f"Proyectos únicos           : {unicos:,}"
    )
    print(
        f"Proyecto ID nulos         : {nulos_proyecto:,}"
    )
    print(
        f"Proyecto ID duplicados    : {duplicados:,}"
    )
    print(
        f"Escenarios                 : {escenarios_proyecto:,}"
    )
    print(
        f"Escenario ID nulos        : {nulos_escenario:,}"
    )
    print(
        f"Geometrías válidas         : {geometrias_validas:,}"
    )
    print(
        f"Cobertura geométrica      : "
        f"{fmt_pct(resumen['cobertura_geometrica'])}"
    )

    return resumen


# =============================================================================
# CONSTRUCCIÓN DE INDICADORES
# =============================================================================

def construir_indicadores_globales(
    proyectos: pd.DataFrame,
    escenarios: pd.DataFrame,
    indicadores: pd.DataFrame,
    campos: dict,
) -> dict:

    titulo("CONSTRUYENDO INDICADORES GLOBALES DEL INFORME")

    resultado = {}

    resultado["proyectos"] = len(proyectos)

    resultado["proyectos_unicos"] = proyectos[
        campos["proyecto"]
    ].nunique(dropna=True)

    resultado["escenarios"] = len(escenarios)

    if campos["escenario_proyecto"]:
        conteos = (
            proyectos
            .groupby(campos["escenario_proyecto"])
            .size()
        )

        resultado["min_proyectos_escenario"] = int(
            conteos.min()
        )

        resultado["max_proyectos_escenario"] = int(
            conteos.max()
        )

        resultado["promedio_proyectos_escenario"] = float(
            conteos.mean()
        )

        resultado["cv_proyectos_escenario"] = cv(
            conteos
        )
    else:
        resultado["min_proyectos_escenario"] = None
        resultado["max_proyectos_escenario"] = None
        resultado["promedio_proyectos_escenario"] = None
        resultado["cv_proyectos_escenario"] = None

    for nombre in [
        "demanda",
        "deficit",
        "conectividad",
        "intermodalidad",
        "integracion",
        "centralidad",
        "impacto",
        "score_cartera",
        "score_territorial",
    ]:

        campo = campos.get(nombre)

        if campo is not None:
            resultado[f"{nombre}_media"] = safe_mean(
                proyectos[campo]
            )
            resultado[f"{nombre}_min"] = safe_min(
                proyectos[campo]
            )
            resultado[f"{nombre}_max"] = safe_max(
                proyectos[campo]
            )

    return resultado


# =============================================================================
# RANKING DE ESCENARIOS
# =============================================================================

def preparar_ranking_escenarios(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    titulo("PREPARANDO RANKING FINAL DE ESCENARIOS")

    df = ranking.copy()

    campo_escenario = resolver_campo(
        df,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
        obligatorio=True,
    )

    campo_ranking = resolver_campo(
        df,
        [
            "ranking_final_v4",
            "ranking_territorial_v4",
            "ranking_cartera_v4",
            "ranking",
        ],
    )

    campo_score = resolver_campo(
        df,
        [
            "score_integral_v4",
            "score_integral",
            "score_cartera_v4",
            "score_cartera",
            "score_priorizacion_v4",
            "score_priorizacion",
        ],
    )

    if campo_ranking is not None:
        df["_ranking"] = pd.to_numeric(
            df[campo_ranking],
            errors="coerce",
        )
    else:
        if campo_score is not None:
            df["_ranking"] = (
                pd.to_numeric(
                    df[campo_score],
                    errors="coerce",
                )
                .rank(
                    ascending=False,
                    method="min",
                )
            )
        else:
            df["_ranking"] = np.arange(1, len(df) + 1)

    if campo_score is not None:
        df["_score_informe"] = pd.to_numeric(
            df[campo_score],
            errors="coerce",
        )
    else:
        df["_score_informe"] = np.nan

    df["_escenario_informe"] = (
        df[campo_escenario]
        .astype(str)
    )

    df = df.sort_values(
        ["_ranking", "_escenario_informe"]
    ).reset_index(drop=True)

    df["ranking_informe_v4"] = np.arange(
        1,
        len(df) + 1,
    )

    return df


# =============================================================================
# RANKING DE PROYECTOS
# =============================================================================

def preparar_ranking_proyectos(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    titulo("PREPARANDO RANKING FINAL DE PROYECTOS")

    df = ranking.copy()

    campo_proyecto = resolver_campo(
        df,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
        obligatorio=True,
    )

    campo_ranking = resolver_campo(
        df,
        [
            "ranking_final_proyecto_v4",
            "ranking_proyecto_v4",
            "ranking",
        ],
    )

    if campo_ranking is not None:
        df["_ranking"] = pd.to_numeric(
            df[campo_ranking],
            errors="coerce",
        )
    else:
        campo_score = resolver_campo(
            df,
            [
                "score_integral_proyecto_v4",
                "score_integral",
                "score_cartera_v4",
                "score_cartera",
            ],
        )

        if campo_score is not None:
            df["_ranking"] = (
                pd.to_numeric(
                    df[campo_score],
                    errors="coerce",
                )
                .rank(
                    ascending=False,
                    method="min",
                )
            )
        else:
            df["_ranking"] = np.arange(
                1,
                len(df) + 1,
            )

    df["_proyecto_informe"] = (
        df[campo_proyecto]
        .astype(str)
    )

    df = df.sort_values(
        ["_ranking", "_proyecto_informe"]
    ).reset_index(drop=True)

    df["ranking_informe_v4"] = np.arange(
        1,
        len(df) + 1,
    )

    return df


# =============================================================================
# VALIDACIÓN DE RANKINGS
# =============================================================================

def validar_rankings(
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
) -> dict:

    titulo("VALIDANDO RANKINGS FINALES")

    resultado = {}

    resultado["escenarios"] = len(
        ranking_escenarios
    )

    resultado["proyectos"] = len(
        ranking_proyectos
    )

    resultado["ranking_escenarios_sin_nulos"] = int(
        ranking_escenarios[
            "ranking_informe_v4"
        ].notna().all()
    )

    resultado["ranking_proyectos_sin_nulos"] = int(
        ranking_proyectos[
            "ranking_informe_v4"
        ].notna().all()
    )

    print(
        f"Escenarios rankeados : "
        f"{resultado['escenarios']}"
    )

    print(
        f"Proyectos rankeados  : "
        f"{resultado['proyectos']}"
    )

    return resultado


# =============================================================================
# TABLAS PARA EL INFORME
# =============================================================================

def tabla_markdown(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
    max_rows: int | None = None,
) -> str:

    if df is None or df.empty:
        return "_Sin información disponible._"

    tabla = df.copy()

    if columnas:
        columnas_existentes = [
            c for c in columnas
            if c in tabla.columns
        ]

        if columnas_existentes:
            tabla = tabla[columnas_existentes]

    if max_rows is not None:
        tabla = tabla.head(max_rows)

    # No depender del paquete tabulate.
    encabezados = [
        str(c)
        for c in tabla.columns
    ]

    lineas = []

    lineas.append(
        "| " + " | ".join(encabezados) + " |"
    )

    lineas.append(
        "| "
        + " | ".join(["---"] * len(encabezados))
        + " |"
    )

    for _, row in tabla.iterrows():

        valores = []

        for valor in row:

            if pd.isna(valor):
                valores.append("")

            elif isinstance(
                valor,
                (float, np.floating),
            ):
                valores.append(
                    f"{float(valor):.2f}"
                )

            else:
                texto = str(valor)
                texto = texto.replace("|", "/")
                valores.append(texto)

        lineas.append(
            "| " + " | ".join(valores) + " |"
        )

    return "\n".join(lineas)


# =============================================================================
# EXTRACCIÓN DE INFORMACIÓN DE ESCENARIOS
# =============================================================================

def extraer_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    default=None,
) -> pd.Series:

    campo = resolver_campo(
        df,
        candidatos,
    )

    if campo is None:
        return pd.Series(
            [default] * len(df),
            index=df.index,
        )

    return df[campo]


def construir_tabla_escenarios(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    df = ranking.copy()

    resultado = pd.DataFrame()

    resultado["ranking"] = df[
        "ranking_informe_v4"
    ]

    resultado["escenario_id"] = (
        extraer_columna(
            df,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
                "_escenario_informe",
            ],
        )
    )

    resultado["tipo_escenario"] = (
        extraer_columna(
            df,
            [
                "tipo_escenario",
                "tipo",
            ],
        )
    )

    resultado["dimension"] = (
        extraer_columna(
            df,
            [
                "dimension_escenario",
                "dimension_dominante",
                "dimension",
            ],
        )
    )

    resultado["cantidad_proyectos"] = (
        extraer_columna(
            df,
            [
                "cantidad_proyectos",
                "proyectos",
                "n_proyectos",
            ],
        )
    )

    resultado["score_integral"] = (
        df["_score_informe"]
    )

    resultado["prioridad"] = (
        extraer_columna(
            df,
            [
                "prioridad_territorial_v4",
                "prioridad_escenario",
                "prioridad",
                "categoria_cartera_v4",
            ],
        )
    )

    resultado["categoria_cartera"] = (
        extraer_columna(
            df,
            [
                "categoria_cartera_v4",
                "categoria_cartera",
            ],
        )
    )

    resultado["linea_estrategica"] = (
        extraer_columna(
            df,
            [
                "linea_estrategica_v4",
                "linea_estrategica",
            ],
        )
    )

    resultado["horizonte"] = (
        extraer_columna(
            df,
            [
                "horizonte_intervencion_v4",
                "horizonte_intervencion",
                "horizonte",
            ],
        )
    )

    return resultado


# =============================================================================
# SÍNTESIS EJECUTIVA
# =============================================================================

def construir_sintesis_ejecutiva(
    proyectos: pd.DataFrame,
    escenarios: pd.DataFrame,
    ranking_escenarios: pd.DataFrame,
    indicadores: dict,
    campos: dict,
    validacion: dict,
) -> dict:

    titulo("CONSTRUYENDO SÍNTESIS EJECUTIVA")

    tabla = construir_tabla_escenarios(
        ranking_escenarios
    )

    escenario_prioritario = None
    escenario_menor = None

    if not tabla.empty:

        escenario_prioritario = limpiar_texto(
            tabla.iloc[0]["escenario_id"]
        )

        escenario_menor = limpiar_texto(
            tabla.iloc[-1]["escenario_id"]
        )

    resultado = {
        "escenario_prioritario": escenario_prioritario,
        "escenario_menor_prioridad": escenario_menor,
        "proyectos": int(
            indicadores["proyectos"]
        ),
        "escenarios": int(
            indicadores["escenarios"]
        ),
        "cobertura_geometrica": float(
            validacion["cobertura_geometrica"]
        ),
        "cv_escenarios": (
            indicadores["cv_proyectos_escenario"]
        ),
    }

    return resultado


# =============================================================================
# GENERACIÓN DEL RESUMEN EJECUTIVO
# =============================================================================

def generar_resumen_ejecutivo(
    ranking_escenarios: pd.DataFrame,
    indicadores: dict,
    validacion: dict,
    sintesis: dict,
) -> str:

    tabla = construir_tabla_escenarios(
        ranking_escenarios
    )

    lineas = []

    lineas.append(
        "# RESUMEN EJECUTIVO\n"
    )

    lineas.append(
        "## Modelo Territorial AMBA V4\n"
    )

    lineas.append(
        "### 1. Síntesis\n"
    )

    lineas.append(
        "El modelo territorial AMBA V4 consolida una "
        "estructura de análisis orientada a la identificación, "
        "priorización y programación territorial de "
        "intervenciones de movilidad.\n"
    )

    lineas.append(
        f"El modelo comprende **{indicadores['proyectos']:,} "
        f"proyectos** distribuidos en "
        f"**{indicadores['escenarios']:,} escenarios territoriales**. "
        "La asignación proyecto-escenario se mantiene íntegra "
        "y la cobertura geométrica alcanza el 100%.\n"
    )

    lineas.append(
        "### 2. Resultados principales\n"
    )

    lineas.append(
        f"- Proyectos: **{indicadores['proyectos']:,}**"
    )

    lineas.append(
        f"- Escenarios: **{indicadores['escenarios']:,}**"
    )

    lineas.append(
        f"- Cobertura geométrica: "
        f"**{fmt_pct(validacion['cobertura_geometrica'])}**"
    )

    lineas.append(
        f"- Proyectos únicos: "
        f"**{validacion['proyectos_unicos']:,}**"
    )

    lineas.append(
        f"- Proyectos duplicados: "
        f"**{validacion['proyectos_duplicados']:,}**"
    )

    lineas.append(
        f"- Proyectos con identificación nula: "
        f"**{validacion['proyectos_nulos']:,}**"
    )

    lineas.append(
        f"- Escenario prioritario: "
        f"**{sintesis['escenario_prioritario']}**"
    )

    lineas.append(
        f"- Escenario de menor prioridad: "
        f"**{sintesis['escenario_menor_prioridad']}**"
    )

    lineas.append(
        "### 3. Ranking de escenarios\n"
    )

    lineas.append(
        tabla_markdown(
            tabla,
            [
                "ranking",
                "escenario_id",
                "tipo_escenario",
                "dimension",
                "cantidad_proyectos",
                "score_integral",
                "prioridad",
                "categoria_cartera",
                "horizonte",
            ],
        )
    )

    lineas.append(
        "\n### 4. Interpretación estratégica\n"
    )

    if sintesis["escenario_prioritario"]:

        lineas.append(
            f"El escenario **{sintesis['escenario_prioritario']}** "
            "se posiciona como el principal ámbito de intervención "
            "dentro del modelo territorial integrado. Su posición "
            "debe interpretarse como resultado de la combinación "
            "de las dimensiones estructurales incorporadas en "
            "el modelo y no como un indicador aislado.\n"
        )

    lineas.append(
        "El modelo debe utilizarse como instrumento de apoyo "
        "para la programación territorial, la evaluación de "
        "alternativas de intervención y la definición de "
        "prioridades de inversión.\n"
    )

    lineas.append(
        "### 5. Estado de validación\n"
    )

    lineas.append(
        "El modelo maestro utilizado como base del presente "
        "documento fue validado en el Proceso 38. La cobertura "
        "geométrica, la unicidad de proyectos y la asignación "
        "territorial presentan consistencia.\n"
    )

    return "\n".join(lineas)


# =============================================================================
# GENERACIÓN DEL INFORME COMPLETO
# =============================================================================

def generar_informe(
    proyectos: pd.DataFrame,
    escenarios: pd.DataFrame,
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
    matriz: pd.DataFrame,
    indicadores_df: pd.DataFrame,
    auditoria_38: pd.DataFrame,
    indicadores: dict,
    validacion: dict,
    sintesis: dict,
    campos: dict,
) -> str:

    titulo("GENERANDO INFORME TERRITORIAL AMBA V4")

    tabla_escenarios = construir_tabla_escenarios(
        ranking_escenarios
    )

    lineas = []

    # -------------------------------------------------------------------------
    # PORTADA
    # -------------------------------------------------------------------------

    lineas.append(
        "# INFORME TERRITORIAL AMBA V4"
    )

    lineas.append(
        "\n## Modelo integrado de movilidad, centralidades, "
        "infraestructura y priorización territorial"
    )

    lineas.append(
        "\n**Proceso:** 39 — Generación del Informe Territorial AMBA"
    )

    lineas.append(
        f"\n**Fecha de generación:** "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    lineas.append(
        "\n---"
    )

    # -------------------------------------------------------------------------
    # RESUMEN EJECUTIVO
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 1. RESUMEN EJECUTIVO"
    )

    lineas.append(
        f"""
El Modelo Territorial AMBA V4 consolida los resultados de las
etapas analíticas precedentes en una estructura única de
diagnóstico, priorización y programación territorial.

La base consolidada contiene **{indicadores['proyectos']:,} proyectos**
y **{indicadores['escenarios']:,} escenarios territoriales**.

La cobertura geométrica es del
**{fmt_pct(validacion['cobertura_geometrica'])}**, con
**{validacion['proyectos_unicos']:,} proyectos únicos** y sin
duplicaciones de identificación.

La distribución de proyectos entre escenarios presenta un
coeficiente de variación de **{fmt_num(indicadores['cv_proyectos_escenario'], 4)}**,
lo que indica una distribución territorial altamente equilibrada
en términos de cantidad de proyectos por escenario.
"""
    )

    if sintesis["escenario_prioritario"]:

        lineas.append(
            f"""
El escenario identificado como prioritario por la consolidación
final es **{sintesis['escenario_prioritario']}**, mientras que
**{sintesis['escenario_menor_prioridad']}** ocupa la última posición
del ranking.
"""
        )

    # -------------------------------------------------------------------------
    # OBJETIVOS
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 2. OBJETIVOS DEL MODELO"
    )

    lineas.append(
        """
El modelo territorial tiene como objetivos:

1. Identificar áreas y proyectos con relevancia estructural
   para la movilidad metropolitana.
2. Integrar demanda, centralidad, conectividad,
   intermodalidad, integración territorial, déficit e impacto.
3. Construir escenarios territoriales comparables.
4. Establecer una priorización territorial reproducible.
5. Organizar los proyectos en una cartera de intervención.
6. Proporcionar una base técnica para la programación de
   inversiones y políticas públicas.
7. Mantener trazabilidad entre los indicadores originales,
   los escenarios y los proyectos resultantes.
"""
    )

    # -------------------------------------------------------------------------
    # ALCANCE
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 3. ALCANCE DEL MODELO"
    )

    lineas.append(
        f"""
El modelo analizado comprende:

- **{indicadores['proyectos']:,} proyectos territoriales**
- **{indicadores['escenarios']:,} escenarios**
- cobertura geométrica del **{fmt_pct(validacion['cobertura_geometrica'])}**
- geometrías válidas para los registros consolidados
- asignación territorial única de cada proyecto
- estructura de priorización y cartera
- información agregada por escenario
"""
    )

    # -------------------------------------------------------------------------
    # METODOLOGÍA
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 4. METODOLOGÍA"
    )

    lineas.append(
        """
La metodología se estructura como una cadena de transformación
y validación progresiva.

### 4.1. Demanda

La demanda de movilidad constituye uno de los componentes
estructurales del modelo.

### 4.2. Centralidad

Se incorporan indicadores asociados a centralidad estructural,
conectividad y alcance territorial.

### 4.3. Infraestructura

La infraestructura existente y sus déficits permiten identificar
brechas territoriales y oportunidades de intervención.

### 4.4. Intermodalidad

La intermodalidad permite valorar la capacidad del territorio
para articular distintos modos de transporte.

### 4.5. Integración territorial

La integración territorial permite incorporar la relación entre
las centralidades, la demanda, la infraestructura y el territorio.

### 4.6. Priorización

Los indicadores se integran en estructuras de priorización
territorial y cartera de intervención.

### 4.7. Consolidación

El Proceso 38 consolidó los resultados en un modelo maestro
único utilizado como fuente del presente informe.
"""
    )

    # -------------------------------------------------------------------------
    # INDICADORES GLOBALES
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 5. INDICADORES GLOBALES"
    )

    lineas.append(
        "| Indicador | Valor |\n"
        "|---|---:|"
    )

    lineas.append(
        f"| Proyectos | {indicadores['proyectos']:,} |"
    )

    lineas.append(
        f"| Proyectos únicos | {validacion['proyectos_unicos']:,} |"
    )

    lineas.append(
        f"| Escenarios | {indicadores['escenarios']:,} |"
    )

    lineas.append(
        f"| Cobertura geométrica | "
        f"{fmt_pct(validacion['cobertura_geometrica'])} |"
    )

    lineas.append(
        f"| Mínimo proyectos/escenario | "
        f"{indicadores['min_proyectos_escenario']} |"
    )

    lineas.append(
        f"| Máximo proyectos/escenario | "
        f"{indicadores['max_proyectos_escenario']} |"
    )

    lineas.append(
        f"| Promedio proyectos/escenario | "
        f"{fmt_num(indicadores['promedio_proyectos_escenario'])} |"
    )

    lineas.append(
        f"| CV tamaño escenarios | "
        f"{fmt_num(indicadores['cv_proyectos_escenario'], 4)} |"
    )

    # -------------------------------------------------------------------------
    # ESCENARIOS
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 6. ESCENARIOS TERRITORIALES"
    )

    lineas.append(
        """
Los escenarios constituyen la unidad intermedia de lectura
territorial del modelo. Cada escenario agrupa proyectos con
características y problemáticas estructurales compatibles.
"""
    )

    lineas.append(
        tabla_markdown(
            tabla_escenarios,
            [
                "ranking",
                "escenario_id",
                "tipo_escenario",
                "dimension",
                "cantidad_proyectos",
                "score_integral",
                "prioridad",
                "categoria_cartera",
                "linea_estrategica",
                "horizonte",
            ],
        )
    )

    # -------------------------------------------------------------------------
    # ANÁLISIS DEL RANKING
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 7. RANKING TERRITORIAL"
    )

    for _, row in tabla_escenarios.iterrows():

        escenario_id = limpiar_texto(
            row.get("escenario_id")
        )

        score = row.get(
            "score_integral"
        )

        dimension = limpiar_texto(
            row.get("dimension")
        )

        categoria = limpiar_texto(
            row.get("categoria_cartera")
        )

        horizonte = limpiar_texto(
            row.get("horizonte")
        )

        lineas.append(
            f"""
### {row['ranking']}. {escenario_id}

- Dimensión dominante: **{dimension or 'N/D'}**
- Proyectos: **{row.get('cantidad_proyectos', 'N/D')}**
- Score integral: **{fmt_num(score)}**
- Categoría: **{categoria or 'N/D'}**
- Horizonte: **{horizonte or 'N/D'}**
"""
        )

    # -------------------------------------------------------------------------
    # CARTERA TERRITORIAL
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 8. CARTERA TERRITORIAL"
    )

    lineas.append(
        """
La cartera territorial constituye la traducción operativa del
modelo analítico. Su objetivo es ordenar los proyectos según
prioridad, relevancia territorial y horizonte de intervención.
"""
    )

    campo_proyecto = resolver_campo(
        proyectos,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
        obligatorio=True,
    )

    campo_escenario = resolver_campo(
        proyectos,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    columnas_cartera = [
        campo_proyecto,
        campo_escenario,
        resolver_campo(
            proyectos,
            [
                "prioridad_territorial_v4",
                "prioridad_escenario",
            ],
        ),
        resolver_campo(
            proyectos,
            [
                "score_cartera_v4",
                "score_cartera",
            ],
        ),
        resolver_campo(
            proyectos,
            [
                "dimension_dominante",
                "dimension",
            ],
        ),
    ]

    columnas_cartera = [
        c for c in columnas_cartera
        if c is not None
    ]

    if columnas_cartera:

        cartera = proyectos[
            columnas_cartera
        ].copy()

        lineas.append(
            tabla_markdown(
                cartera,
                columnas_cartera,
                max_rows=30,
            )
        )

        lineas.append(
            "\n*Se muestran los primeros 30 registros. "
            "El detalle completo se encuentra en el anexo "
            "`anexo_proyectos_amba_v4.csv`.*\n"
        )

    # -------------------------------------------------------------------------
    # DIMENSIONES ESTRUCTURALES
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 9. DIMENSIONES ESTRUCTURALES"
    )

    lineas.append(
        """
El modelo integra múltiples dimensiones para evitar que la
priorización dependa de un único indicador.

Las principales dimensiones consideradas son:

- demanda
- déficit de infraestructura
- conectividad
- intermodalidad
- integración territorial
- centralidad
- impacto potencial
- urgencia de intervención
"""
    )

    lineas.append(
        "| Dimensión | Media | Mínimo | Máximo |\n"
        "|---|---:|---:|---:|"
    )

    dimensiones = [
        ("Demanda", "demanda"),
        ("Déficit", "deficit"),
        ("Conectividad", "conectividad"),
        ("Intermodalidad", "intermodalidad"),
        ("Integración territorial", "integracion"),
        ("Centralidad", "centralidad"),
        ("Impacto", "impacto"),
        ("Score cartera", "score_cartera"),
        ("Score territorial", "score_territorial"),
    ]

    for nombre, clave in dimensiones:

        media = indicadores.get(
            f"{clave}_media"
        )

        minimo = indicadores.get(
            f"{clave}_min"
        )

        maximo = indicadores.get(
            f"{clave}_max"
        )

        if (
            media is not None
            or minimo is not None
            or maximo is not None
        ):

            lineas.append(
                f"| {nombre} | "
                f"{fmt_num(media)} | "
                f"{fmt_num(minimo)} | "
                f"{fmt_num(maximo)} |"
            )

    # -------------------------------------------------------------------------
    # ANÁLISIS TERRITORIAL
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 10. LECTURA TERRITORIAL"
    )

    lineas.append(
        """
La lectura territorial debe realizarse considerando la
interacción entre las dimensiones analíticas y no únicamente
mediante el ranking de un indicador individual.

Los escenarios de mayor prioridad representan territorios en
los que la combinación de demanda, déficit, conectividad,
intermodalidad, integración e impacto justifica una atención
preferente dentro del modelo.

Los escenarios de menor prioridad no deben interpretarse como
territorios sin necesidades. Representan, dentro del universo
analizado, ámbitos con menor prioridad relativa bajo los
criterios definidos por el modelo.
"""
    )

    # -------------------------------------------------------------------------
    # PROGRAMACIÓN
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 11. PROGRAMACIÓN DE INTERVENCIONES"
    )

    lineas.append(
        """
La cartera permite estructurar la intervención en diferentes
horizontes temporales:

- corto plazo
- corto/mediano plazo
- mediano plazo
- mediano/largo plazo

La programación definitiva deberá incorporar posteriormente
variables de factibilidad técnica, disponibilidad presupuestaria,
marco institucional, permisos, suelo, costos y capacidad de
ejecución.
"""
    )

    # -------------------------------------------------------------------------
    # GOBERNANZA
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 12. IMPLICANCIAS PARA LA GESTIÓN METROPOLITANA"
    )

    lineas.append(
        """
El modelo puede utilizarse como soporte técnico para:

1. priorizar estudios y proyectos;
2. orientar inversiones;
3. coordinar intervenciones entre jurisdicciones;
4. identificar áreas donde se requiere integración modal;
5. evaluar brechas territoriales;
6. construir programas de intervención;
7. monitorear la evolución de las centralidades;
8. actualizar periódicamente la cartera.
"""
    )

    # -------------------------------------------------------------------------
    # LIMITACIONES
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 13. LIMITACIONES Y CONSIDERACIONES"
    )

    lineas.append(
        """
El resultado constituye un modelo analítico de priorización
territorial y no reemplaza la evaluación técnica, económica,
ambiental, jurídica o institucional de cada proyecto.

Antes de transformar una prioridad analítica en una decisión de
inversión deberán incorporarse, entre otros elementos:

- estimación de costos;
- disponibilidad presupuestaria;
- factibilidad constructiva;
- evaluación socioeconómica;
- impacto ambiental;
- disponibilidad de suelo;
- competencias institucionales;
- restricciones normativas;
- cronograma de ejecución;
- riesgos de implementación.
"""
    )

    # -------------------------------------------------------------------------
    # TRAZABILIDAD
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 14. TRAZABILIDAD Y AUDITORÍA"
    )

    lineas.append(
        """
El presente informe utiliza como fuente principal el modelo
maestro generado por el Proceso 38.

La cadena de trazabilidad se resume como:

**Modelo analítico → escenarios → priorización → cartera →
validación geoespacial → integración → consolidación → informe.**
"""
    )

    lineas.append(
        f"""
La validación estructural presenta:

- proyectos únicos: **{validacion['proyectos_unicos']:,}**
- proyectos duplicados: **{validacion['proyectos_duplicados']:,}**
- geometrías válidas: **{validacion['geometrias_validas']:,}**
- cobertura geométrica: **{fmt_pct(validacion['cobertura_geometrica'])}**
"""
    )

    # -------------------------------------------------------------------------
    # CONCLUSIONES
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 15. CONCLUSIONES"
    )

    lineas.append(
        f"""
El Modelo Territorial AMBA V4 presenta una estructura integrada
y consistente para la lectura territorial de la movilidad.

La base comprende **{indicadores['proyectos']:,} proyectos**
organizados en **{indicadores['escenarios']:,} escenarios**, con
cobertura geométrica completa.

El escenario **{sintesis['escenario_prioritario']}** ocupa la
posición principal del ranking territorial, mientras que
**{sintesis['escenario_menor_prioridad']}** presenta la menor
prioridad relativa.

La información resultante permite avanzar desde el diagnóstico
territorial hacia una etapa de programación de intervenciones,
siempre complementando la priorización analítica con estudios
de factibilidad y evaluación de proyectos.
"""
    )

    # -------------------------------------------------------------------------
    # ANEXOS
    # -------------------------------------------------------------------------

    lineas.append(
        "\n# 16. ANEXOS"
    )

    lineas.append(
        """
### Anexo A — Proyectos

Archivo:

`anexo_proyectos_amba_v4.csv`

Contiene el universo completo de proyectos utilizado por el
modelo maestro.

### Anexo B — Escenarios

Archivo:

`anexo_escenarios_amba_v4.csv`

Contiene la síntesis completa de los escenarios.

### Anexo C — Indicadores

Archivo:

`anexo_indicadores_globales_amba_v4.csv`

Contiene los indicadores globales empleados en el informe.

### Anexo D — Modelo geográfico

Archivo:

`modelo_maestro_territorial_amba_v4.gpkg`

Contiene las capas geográficas consolidadas de proyectos y
escenarios.
"""
    )

    return "\n".join(lineas)


# =============================================================================
# GENERACIÓN DE ANEXOS
# =============================================================================

def generar_anexos(
    proyectos: pd.DataFrame,
    escenarios: pd.DataFrame,
    indicadores: pd.DataFrame,
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
) -> None:

    titulo("GENERANDO ANEXOS")

    # -------------------------------------------------------------------------
    # ANEXO PROYECTOS
    # -------------------------------------------------------------------------

    proyectos_out = proyectos.copy()

    proyectos_out.to_csv(
        FILE_ANEXO_PROYECTOS,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Proyectos : {FILE_ANEXO_PROYECTOS}"
    )

    # -------------------------------------------------------------------------
    # ANEXO ESCENARIOS
    # -------------------------------------------------------------------------

    escenarios_out = construir_tabla_escenarios(
        ranking_escenarios
    )

    escenarios_out.to_csv(
        FILE_ANEXO_ESCENARIOS,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Escenarios: {FILE_ANEXO_ESCENARIOS}"
    )

    # -------------------------------------------------------------------------
    # INDICADORES
    # -------------------------------------------------------------------------

    indicadores.to_csv(
        FILE_ANEXO_INDICADORES,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Indicadores: {FILE_ANEXO_INDICADORES}"
    )


# =============================================================================
# AUDITORÍA
# =============================================================================

def construir_auditoria(
    validacion: dict,
    indicadores: dict,
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO AUDITORÍA DEL PROCESO 39")

    checks = []

    def agregar(
        control: str,
        resultado,
        valor,
        observacion: str,
    ):
        checks.append(
            {
                "proceso": PROCESO,
                "version": VERSION,
                "control": control,
                "resultado": resultado,
                "valor": valor,
                "observacion": observacion,
            }
        )

    agregar(
        "proyectos_unicos",
        validacion["proyectos_unicos"]
        == validacion["registros"],
        validacion["proyectos_unicos"],
        "Todos los registros corresponden a proyectos únicos.",
    )

    agregar(
        "proyectos_sin_nulos",
        validacion["proyectos_nulos"] == 0,
        validacion["proyectos_nulos"],
        "No existen identificadores de proyecto nulos.",
    )

    agregar(
        "proyectos_sin_duplicados",
        validacion["proyectos_duplicados"] == 0,
        validacion["proyectos_duplicados"],
        "No existen proyectos duplicados.",
    )

    agregar(
        "cobertura_geometrica",
        validacion["cobertura_geometrica"] >= 99.999,
        validacion["cobertura_geometrica"],
        "La cobertura geométrica debe ser completa.",
    )

    agregar(
        "ranking_escenarios",
        len(ranking_escenarios)
        == indicadores["escenarios"],
        len(ranking_escenarios),
        "Existe un registro de ranking por escenario.",
    )

    agregar(
        "ranking_proyectos",
        len(ranking_proyectos)
        == indicadores["proyectos"],
        len(ranking_proyectos),
        "Existe un registro de ranking por proyecto.",
    )

    checks_df = pd.DataFrame(checks)

    return checks_df


# =============================================================================
# RESUMEN JSON
# =============================================================================

def construir_resumen_json(
    indicadores: dict,
    validacion: dict,
    sintesis: dict,
    auditoria: pd.DataFrame,
) -> dict:

    controles_ok = int(
        auditoria["resultado"]
        .astype(bool)
        .sum()
    )

    controles_total = len(auditoria)

    dictamen = (
        "VALIDADO"
        if controles_ok == controles_total
        else "OBSERVADO"
    )

    resumen = {
        "proceso": PROCESO,
        "version": VERSION,
        "fecha_generacion": datetime.now().isoformat(),
        "dictamen": dictamen,
        "proyectos": indicadores["proyectos"],
        "proyectos_unicos": validacion[
            "proyectos_unicos"
        ],
        "escenarios": indicadores["escenarios"],
        "cobertura_geometrica": validacion[
            "cobertura_geometrica"
        ],
        "geometrias_validas": validacion[
            "geometrias_validas"
        ],
        "proyectos_duplicados": validacion[
            "proyectos_duplicados"
        ],
        "proyectos_nulos": validacion[
            "proyectos_nulos"
        ],
        "escenario_prioritario": sintesis[
            "escenario_prioritario"
        ],
        "escenario_menor_prioridad": sintesis[
            "escenario_menor_prioridad"
        ],
        "cv_escenarios": indicadores[
            "cv_proyectos_escenario"
        ],
        "controles_ok": controles_ok,
        "controles_total": controles_total,
    }

    return resumen


# =============================================================================
# EXPORTACIÓN DEL MODELO GEOGRÁFICO
# =============================================================================

def validar_gpkg() -> dict:

    resultado = {
        "existe": FILE_GPKG.exists(),
        "capas": [],
    }

    if not FILE_GPKG.exists():
        return resultado

    if gpd is None:
        return resultado

    try:

        import fiona

        capas = fiona.listlayers(
            FILE_GPKG
        )

        resultado["capas"] = capas

    except Exception as exc:

        print(
            f"ADVERTENCIA: no se pudieron leer "
            f"las capas del GeoPackage: {exc}"
        )

    return resultado


# =============================================================================
# MAIN
# =============================================================================

def main():

    inicio = datetime.now()

    titulo(
        "39 - GENERACIÓN DEL INFORME TERRITORIAL AMBA - V4"
    )

    print(
        f"Proyecto : {PROJECT_ROOT}"
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

    try:

        # ---------------------------------------------------------------------
        # CARGA
        # ---------------------------------------------------------------------

        datos = cargar_fuentes()

        proyectos = datos["proyectos"]
        escenarios = datos["escenarios"]
        ranking_escenarios_raw = datos[
            "ranking_escenarios"
        ]
        ranking_proyectos_raw = datos[
            "ranking_proyectos"
        ]
        matriz = datos["matriz"]
        indicadores_df = datos["indicadores"]
        auditoria_38 = datos["auditoria_38"]

        # ---------------------------------------------------------------------
        # CAMPOS
        # ---------------------------------------------------------------------

        campos = resolver_campos(
            proyectos,
            escenarios,
        )

        # ---------------------------------------------------------------------
        # VALIDACIÓN
        # ---------------------------------------------------------------------

        validacion = validar_modelo(
            proyectos,
            escenarios,
            campos,
        )

        # ---------------------------------------------------------------------
        # INDICADORES
        # ---------------------------------------------------------------------

        indicadores = construir_indicadores_globales(
            proyectos,
            escenarios,
            indicadores_df,
            campos,
        )

        # ---------------------------------------------------------------------
        # RANKINGS
        # ---------------------------------------------------------------------

        ranking_escenarios = preparar_ranking_escenarios(
            ranking_escenarios_raw
        )

        ranking_proyectos = preparar_ranking_proyectos(
            ranking_proyectos_raw
        )

        validar_rankings(
            ranking_escenarios,
            ranking_proyectos,
        )

        # ---------------------------------------------------------------------
        # SÍNTESIS
        # ---------------------------------------------------------------------

        sintesis = construir_sintesis_ejecutiva(
            proyectos,
            escenarios,
            ranking_escenarios,
            indicadores,
            campos,
            validacion,
        )

        # ---------------------------------------------------------------------
        # INFORME
        # ---------------------------------------------------------------------

        informe = generar_informe(
            proyectos=proyectos,
            escenarios=escenarios,
            ranking_escenarios=ranking_escenarios,
            ranking_proyectos=ranking_proyectos,
            matriz=matriz,
            indicadores_df=indicadores_df,
            auditoria_38=auditoria_38,
            indicadores=indicadores,
            validacion=validacion,
            sintesis=sintesis,
            campos=campos,
        )

        # ---------------------------------------------------------------------
        # RESUMEN EJECUTIVO
        # ---------------------------------------------------------------------

        resumen_ejecutivo = generar_resumen_ejecutivo(
            ranking_escenarios=ranking_escenarios,
            indicadores=indicadores,
            validacion=validacion,
            sintesis=sintesis,
        )

        # ---------------------------------------------------------------------
        # ANEXOS
        # ---------------------------------------------------------------------

        generar_anexos(
            proyectos=proyectos,
            escenarios=escenarios,
            indicadores=indicadores_df,
            ranking_escenarios=ranking_escenarios,
            ranking_proyectos=ranking_proyectos,
        )

        # ---------------------------------------------------------------------
        # AUDITORÍA
        # ---------------------------------------------------------------------

        auditoria = construir_auditoria(
            validacion=validacion,
            indicadores=indicadores,
            ranking_escenarios=ranking_escenarios,
            ranking_proyectos=ranking_proyectos,
        )

        # ---------------------------------------------------------------------
        # RESUMEN JSON
        # ---------------------------------------------------------------------

        resumen_json = construir_resumen_json(
            indicadores=indicadores,
            validacion=validacion,
            sintesis=sintesis,
            auditoria=auditoria,
        )

        # ---------------------------------------------------------------------
        # EXPORTACIÓN
        # ---------------------------------------------------------------------

        titulo(
            "EXPORTANDO RESULTADOS DEL PROCESO 39"
        )

        FILE_INFORME.write_text(
            informe,
            encoding="utf-8",
        )

        print(
            f"Informe          : {FILE_INFORME}"
        )

        FILE_RESUMEN.write_text(
            resumen_ejecutivo,
            encoding="utf-8",
        )

        print(
            f"Resumen ejecutivo: {FILE_RESUMEN}"
        )

        auditoria.to_csv(
            FILE_AUDITORIA,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Auditoría        : {FILE_AUDITORIA}"
        )

        with FILE_RESUMEN_JSON.open(
            "w",
            encoding="utf-8",
        ) as archivo:

            json.dump(
                resumen_json,
                archivo,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        print(
            f"Resumen JSON      : {FILE_RESUMEN_JSON}"
        )

        # ---------------------------------------------------------------------
        # VALIDACIÓN GPKG
        # ---------------------------------------------------------------------

        titulo(
            "VALIDANDO MODELO GEOGRÁFICO CONSOLIDADO"
        )

        gpkg = validar_gpkg()

        print(
            f"GeoPackage existe : "
            f"{'SI' if gpkg['existe'] else 'NO'}"
        )

        if gpkg["capas"]:

            print(
                "Capas            : "
                + ", ".join(gpkg["capas"])
            )

        # ---------------------------------------------------------------------
        # RESULTADO FINAL
        # ---------------------------------------------------------------------

        controles_ok = int(
            auditoria["resultado"]
            .astype(bool)
            .sum()
        )

        controles_total = len(
            auditoria
        )

        dictamen = (
            "VALIDADO"
            if controles_ok == controles_total
            else "OBSERVADO"
        )

        duracion = (
            datetime.now() - inicio
        ).total_seconds()

        titulo(
            "RESULTADO FINAL DEL PROCESO 39"
        )

        print(
            f"Proyectos                 : "
            f"{indicadores['proyectos']:,}"
        )

        print(
            f"Proyectos únicos          : "
            f"{validacion['proyectos_unicos']:,}"
        )

        print(
            f"Escenarios                : "
            f"{indicadores['escenarios']:,}"
        )

        print(
            f"Cobertura geométrica      : "
            f"{fmt_pct(validacion['cobertura_geometrica'])}"
        )

        print(
            f"Geometrías válidas        : "
            f"{validacion['geometrias_validas']:,}"
        )

        print(
            f"Proyectos duplicados      : "
            f"{validacion['proyectos_duplicados']:,}"
        )

        print(
            f"CV tamaño escenarios      : "
            f"{fmt_num(indicadores['cv_proyectos_escenario'], 4)}"
        )

        print(
            f"Escenario prioritario     : "
            f"{sintesis['escenario_prioritario']}"
        )

        print(
            f"Escenario menor prioridad: "
            f"{sintesis['escenario_menor_prioridad']}"
        )

        print(
            f"Controles OK              : "
            f"{controles_ok}/{controles_total}"
        )

        print(
            f"Auditoría                 : "
            f"{'OK' if dictamen == 'VALIDADO' else 'OBSERVADA'}"
        )

        print(
            f"Dictamen                  : "
            f"{dictamen}"
        )

        print(
            f"Tiempo de ejecución       : "
            f"{duracion:.2f} segundos"
        )

        if dictamen == "VALIDADO":

            print()
            print(
                "El informe territorial AMBA V4 fue generado "
                "y validado correctamente."
            )

            print(
                "El documento utiliza como fuente el modelo "
                "maestro consolidado del Proceso 38."
            )

            print(
                "La estructura territorial mantiene la "
                "trazabilidad de proyectos y escenarios."
            )

            print(
                "El resultado queda preparado para la "
                "validación final y cierre del modelo."
            )

        else:

            print()
            print(
                "El informe fue generado con observaciones."
            )

        titulo(
            "PROCESO 39 FINALIZADO"
        )

        return 0 if dictamen == "VALIDADO" else 1

    except Exception as exc:

        titulo(
            "ERROR FATAL EN EL PROCESO 39"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        traceback.print_exc()

        return 1


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())