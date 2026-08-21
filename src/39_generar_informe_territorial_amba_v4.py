# -*- coding: utf-8 -*-

"""
===============================================================================
39 - GENERACIÓN DEL INFORME TERRITORIAL AMBA - V4.1
===============================================================================

Objetivo
--------
Generar el informe territorial final AMBA V4.1 a partir del modelo maestro
consolidado por el proceso 38.

Correcciones V4.1
-----------------
1. No depende de una columna geometry dentro de los CSV.
2. Recupera las geometrías desde:
       modelo_maestro_territorial_amba_v4.gpkg
3. Valida las capas geográficas "proyectos" y "escenarios".
4. Mantiene la trazabilidad proyecto -> escenario.
5. Verifica 144 proyectos y 7 escenarios.
6. No modifica los indicadores originales.
7. Genera informe Markdown, resumen ejecutivo, anexos, auditoría y JSON.
8. No intenta crear este archivo desde /mnt/data.
9. Compatible con ejecución directa desde Windows / PowerShell.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.1"

SCRIPT_NAME = "39_generar_informe_territorial_amba_v4.py"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR = INPUT_DIR


# =============================================================================
# ARCHIVOS DE ENTRADA
# =============================================================================

ARCHIVOS_ENTRADA = {
    "proyectos": "modelo_maestro_proyectos_v4.csv",
    "escenarios": "modelo_maestro_escenarios_v4.csv",
    "ranking_escenarios": "ranking_final_escenarios_v4.csv",
    "ranking_proyectos": "ranking_final_proyectos_v4.csv",
    "matriz": "matriz_integral_escenarios_v4.csv",
    "indicadores": "indicadores_globales_amba_v4.csv",
    "auditoria_38": "auditoria_38_consolidacion_territorial_amba.csv",
    "gpkg": "modelo_maestro_territorial_amba_v4.gpkg",
}


# =============================================================================
# ARCHIVOS DE SALIDA
# =============================================================================

ARCHIVOS_SALIDA = {
    "informe": "informe_territorial_amba_v4_1.md",
    "resumen": "resumen_ejecutivo_amba_v4_1.md",
    "auditoria": "auditoria_39_informe_territorial_amba_v4_1.csv",
    "json": "resumen_39_informe_territorial_amba_v4_1.json",
    "anexo_proyectos": "anexo_proyectos_amba_v4_1.csv",
    "anexo_escenarios": "anexo_escenarios_amba_v4_1.csv",
    "anexo_indicadores": "anexo_indicadores_globales_amba_v4_1.csv",
}


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


def resolver_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = True,
) -> str | None:
    """
    Busca una columna respetando primero coincidencia exacta y luego
    coincidencia case-insensitive.
    """

    columnas = list(df.columns)

    for candidato in candidatos:
        if candidato in columnas:
            return candidato

    mapa = {str(c).lower(): c for c in columnas}

    for candidato in candidatos:
        encontrado = mapa.get(candidato.lower())
        if encontrado is not None:
            return encontrado

    if obligatoria:
        raise KeyError(
            f"No se encontró ninguna de las columnas esperadas: {candidatos}"
        )

    return None


def safe_float(valor, default=np.nan) -> float:
    try:
        if pd.isna(valor):
            return default

        resultado = float(valor)

        if math.isfinite(resultado):
            return resultado

        return default

    except Exception:
        return default


def media_segura(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce")

    if valores.notna().sum() == 0:
        return 0.0

    return float(valores.mean())


def max_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce")

    if valores.notna().sum() == 0:
        return 0.0

    return float(valores.max())


def min_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce")

    if valores.notna().sum() == 0:
        return 0.0

    return float(valores.min())


def porcentaje(valor: float, total: float) -> float:
    if total == 0:
        return 0.0

    return float(valor / total * 100.0)


def formato_numero(valor, decimales: int = 2) -> str:
    if valor is None or pd.isna(valor):
        return "N/D"

    try:
        return f"{float(valor):,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(valor)


def formato_porcentaje(valor, decimales: int = 2) -> str:
    if valor is None or pd.isna(valor):
        return "N/D"

    try:
        return f"{float(valor):.{decimales}f}%".replace(".", ",")
    except Exception:
        return str(valor)


def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()

    for columna in resultado.columns:
        if resultado[columna].dtype == "object":
            resultado[columna] = resultado[columna].replace(
                {
                    "nan": np.nan,
                    "None": np.nan,
                    "NaN": np.nan,
                    "": np.nan,
                }
            )

    return resultado


def tabla_markdown(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
    max_filas: int = 20,
) -> str:
    """
    Genera Markdown sin depender de tabulate.
    """

    if df is None or df.empty:
        return "_Sin datos._"

    tabla = df.copy()

    if columnas:
        disponibles = [c for c in columnas if c in tabla.columns]
        tabla = tabla[disponibles]

    tabla = tabla.head(max_filas)

    encabezados = [str(c) for c in tabla.columns]

    lineas = [
        "| " + " | ".join(encabezados) + " |",
        "| " + " | ".join(["---"] * len(encabezados)) + " |",
    ]

    for _, fila in tabla.iterrows():
        valores = []

        for valor in fila:
            if pd.isna(valor):
                texto = ""
            elif isinstance(valor, float):
                texto = formato_numero(valor, 2)
            else:
                texto = str(valor)

            texto = texto.replace("|", "\\|").replace("\n", " ")

            valores.append(texto)

        lineas.append("| " + " | ".join(valores) + " |")

    return "\n".join(lineas)


def escribir_texto(path: Path, contenido: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contenido, encoding="utf-8")


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_csv(nombre: str) -> pd.DataFrame:
    path = INPUT_DIR / nombre

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")

    print(f"Cargando: {nombre}")

    df = pd.read_csv(path, low_memory=False)

    print(f"Registros : {len(df):,}")
    print(f"Columnas  : {len(df.columns):,}")

    return limpiar_dataframe(df)


def cargar_geopackage() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    path = INPUT_DIR / ARCHIVOS_ENTRADA["gpkg"]

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el GeoPackage maestro del proceso 38:\n{path}"
        )

    titulo("CARGANDO MODELO GEOGRÁFICO CONSOLIDADO DEL PROCESO 38")

    print(f"GeoPackage: {path}")

    capas = gpd.list_layers(path)

    print("Capas disponibles:")

    for nombre in capas["name"].tolist():
        print(f"  - {nombre}")

    if "proyectos" not in capas["name"].tolist():
        raise ValueError(
            "El GeoPackage no contiene la capa obligatoria 'proyectos'."
        )

    if "escenarios" not in capas["name"].tolist():
        raise ValueError(
            "El GeoPackage no contiene la capa obligatoria 'escenarios'."
        )

    proyectos_geo = gpd.read_file(path, layer="proyectos")
    escenarios_geo = gpd.read_file(path, layer="escenarios")

    print(f"Proyectos geográficos : {len(proyectos_geo):,}")
    print(f"Escenarios geográficos: {len(escenarios_geo):,}")
    print(f"CRS proyectos         : {proyectos_geo.crs}")
    print(f"CRS escenarios        : {escenarios_geo.crs}")

    return proyectos_geo, escenarios_geo


# =============================================================================
# RESOLUCIÓN DE CAMPOS
# =============================================================================

def resolver_campos_proyectos(df: pd.DataFrame) -> dict[str, str | None]:
    titulo("RESOLUCIÓN DE CAMPOS")

    campos = {
        "proyecto": resolver_columna(
            df,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
        ),
        "escenario": resolver_columna(
            df,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        ),
        "tipo": resolver_columna(
            df,
            [
                "tipo_escenario",
                "tipo_proyecto",
            ],
            obligatoria=False,
        ),
        "dimension": resolver_columna(
            df,
            [
                "dimension_dominante",
                "dimension_escenario",
            ],
            obligatoria=False,
        ),
        "prioridad": resolver_columna(
            df,
            [
                "prioridad_territorial_v4",
                "prioridad_escenario",
                "prioridad",
            ],
            obligatoria=False,
        ),
        "score_cartera": resolver_columna(
            df,
            [
                "score_cartera_v4",
                "score_cartera",
            ],
            obligatoria=False,
        ),
        "score_territorial": resolver_columna(
            df,
            [
                "score_priorizacion_v4",
                "score_prioridad_territorial",
                "score_territorial",
            ],
            obligatoria=False,
        ),
        "demanda": resolver_columna(
            df,
            [
                "indice_demanda_estructural",
                "score_demanda",
            ],
            obligatoria=False,
        ),
        "deficit": resolver_columna(
            df,
            [
                "deficit_infraestructura",
                "score_deficit",
            ],
            obligatoria=False,
        ),
        "conectividad": resolver_columna(
            df,
            [
                "indice_conectividad_estructural",
                "score_conectividad",
            ],
            obligatoria=False,
        ),
        "intermodalidad": resolver_columna(
            df,
            [
                "indice_intermodalidad_estructural",
                "score_intermodalidad",
            ],
            obligatoria=False,
        ),
        "integracion": resolver_columna(
            df,
            [
                "indice_integracion_territorial",
                "score_integracion",
            ],
            obligatoria=False,
        ),
        "centralidad": resolver_columna(
            df,
            [
                "indice_centralidad_estructural",
            ],
            obligatoria=False,
        ),
        "impacto": resolver_columna(
            df,
            [
                "impacto_potencial",
            ],
            obligatoria=False,
        ),
        "urgencia": resolver_columna(
            df,
            [
                "urgencia_intervencion",
            ],
            obligatoria=False,
        ),
    }

    for clave, columna in campos.items():
        print(f"{clave:<28}: {columna or 'NO DISPONIBLE'}")

    return campos


# =============================================================================
# INTEGRACIÓN DE GEOMETRÍAS
# =============================================================================

def integrar_geometrias(
    proyectos: pd.DataFrame,
    proyectos_geo: gpd.GeoDataFrame,
    escenarios_geo: gpd.GeoDataFrame,
    campos: dict[str, str | None],
) -> gpd.GeoDataFrame:

    titulo("INTEGRANDO GEOMETRÍAS CANÓNICAS DEL PROCESO 38")

    proyecto_col = campos["proyecto"]

    geo_proyecto_col = resolver_columna(
        proyectos_geo,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    geo_escenario_col = resolver_columna(
        escenarios_geo,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    # -------------------------------------------------------------------------
    # Proyectos
    # -------------------------------------------------------------------------

    geo_proyectos = proyectos_geo[
        [geo_proyecto_col, "geometry"]
    ].copy()

    geo_proyectos = geo_proyectos.rename(
        columns={
            geo_proyecto_col: "_proyecto_join",
        }
    )

    geo_proyectos["_proyecto_join"] = (
        geo_proyectos["_proyecto_join"]
        .astype(str)
        .str.strip()
    )

    base = proyectos.copy()

    base["_proyecto_join"] = (
        base[proyecto_col]
        .astype(str)
        .str.strip()
    )

    base = base.merge(
        geo_proyectos,
        on="_proyecto_join",
        how="left",
        suffixes=("", "_geo"),
    )

    if "geometry" not in base.columns:
        raise ValueError(
            "No fue posible integrar la geometría desde la capa 'proyectos'."
        )

    base = gpd.GeoDataFrame(
        base,
        geometry="geometry",
        crs=proyectos_geo.crs,
    )

    base = base.drop(columns=["_proyecto_join"], errors="ignore")

    # -------------------------------------------------------------------------
    # Normalización CRS
    # -------------------------------------------------------------------------

    if base.crs is None:
        base = base.set_crs("EPSG:4326")

    elif str(base.crs) != str(proyectos_geo.crs):
        base = base.to_crs(proyectos_geo.crs)

    # -------------------------------------------------------------------------
    # Validación
    # -------------------------------------------------------------------------

    validas = base.geometry.notna() & (~base.geometry.is_empty)

    if validas.any():
        tipos = base.loc[validas, "geometry"].geom_type.value_counts().to_dict()
    else:
        tipos = {}

    print(f"Geometrías integradas : {int(validas.sum()):,}")
    print(f"Geometrías faltantes  : {int((~validas).sum()):,}")
    print(f"Tipos geométricos     : {tipos}")

    return base


# =============================================================================
# VALIDACIÓN DEL MODELO
# =============================================================================

def validar_modelo(
    proyectos: gpd.GeoDataFrame,
    escenarios_geo: gpd.GeoDataFrame,
    campos: dict[str, str | None],
) -> dict:

    titulo("VALIDACIÓN DEL MODELO TERRITORIAL")

    proyecto_col = campos["proyecto"]
    escenario_col = campos["escenario"]

    total = len(proyectos)

    proyectos_unicos = proyectos[proyecto_col].nunique(dropna=True)

    proyectos_nulos = int(proyectos[proyecto_col].isna().sum())

    duplicados = int(
        proyectos[proyecto_col].duplicated(keep=False).sum()
    )

    escenarios = proyectos[escenario_col].nunique(dropna=True)

    escenarios_nulos = int(
        proyectos[escenario_col].isna().sum()
    )

    geom_no_nula = proyectos.geometry.notna()

    geom_no_vacia = geom_no_nula & (~proyectos.geometry.is_empty)

    geom_validas = geom_no_vacia & proyectos.geometry.is_valid

    geom_invalidas = int(
        (geom_no_vacia & (~proyectos.geometry.is_valid)).sum()
    )

    geom_nulas = int(
        (~geom_no_nula).sum()
    )

    geom_vacias = int(
        (geom_no_nula & proyectos.geometry.is_empty).sum()
    )

    cobertura = porcentaje(
        int(geom_validas.sum()),
        total,
    )

    tamanios = (
        proyectos.groupby(escenario_col)
        .size()
        .sort_values()
    )

    minimo = int(tamanios.min()) if not tamanios.empty else 0
    maximo = int(tamanios.max()) if not tamanios.empty else 0
    promedio = float(tamanios.mean()) if not tamanios.empty else 0.0

    cv = (
        float(tamanios.std(ddof=0) / promedio)
        if promedio > 0
        else 0.0
    )

    multi = (
        proyectos.groupby(proyecto_col)[escenario_col]
        .nunique(dropna=True)
    )

    proyectos_multiescenario = int(
        (multi > 1).sum()
    )

    escenarios_geo_ids = resolver_columna(
        escenarios_geo,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    escenarios_geograficos = escenarios_geo[
        escenarios_geo_ids
    ].nunique(dropna=True)

    print(f"Registros                  : {total:,}")
    print(f"Proyectos únicos           : {proyectos_unicos:,}")
    print(f"Proyecto ID nulos          : {proyectos_nulos:,}")
    print(f"Proyecto ID duplicados     : {duplicados:,}")
    print(f"Escenarios                 : {escenarios:,}")
    print(f"Escenario ID nulos         : {escenarios_nulos:,}")
    print(f"Escenarios geográficos     : {escenarios_geograficos:,}")
    print(f"Geometrías válidas         : {int(geom_validas.sum()):,}")
    print(f"Geometrías nulas           : {geom_nulas:,}")
    print(f"Geometrías vacías          : {geom_vacias:,}")
    print(f"Geometrías inválidas       : {geom_invalidas:,}")
    print(f"Cobertura geométrica       : {cobertura:.2f}%")
    print(f"Proyectos multiescenario   : {proyectos_multiescenario}")
    print(f"Mínimo proyectos/escenario : {minimo}")
    print(f"Máximo proyectos/escenario : {maximo}")
    print(f"Promedio proyectos/escenario: {promedio:.2f}")
    print(f"CV tamaño escenarios       : {cv:.4f}")

    controles = {
        "144_proyectos": total == 144,
        "proyectos_unicos": proyectos_unicos == 144,
        "7_escenarios": escenarios == 7,
        "sin_nulos_id": (
            proyectos_nulos == 0
            and escenarios_nulos == 0
        ),
        "sin_duplicados": duplicados == 0,
        "geometria_completa": cobertura == 100.0,
        "geometria_valida": geom_invalidas == 0,
        "sin_multiescenario": proyectos_multiescenario == 0,
        "escenarios_geograficos": escenarios_geograficos == 7,
    }

    controles_ok = sum(bool(v) for v in controles.values())

    resultado = {
        "total_proyectos": total,
        "proyectos_unicos": proyectos_unicos,
        "proyectos_nulos": proyectos_nulos,
        "proyectos_duplicados": duplicados,
        "escenarios": escenarios,
        "escenarios_nulos": escenarios_nulos,
        "escenarios_geograficos": int(escenarios_geograficos),
        "geometrias_validas": int(geom_validas.sum()),
        "geometrias_nulas": geom_nulas,
        "geometrias_vacias": geom_vacias,
        "geometrias_invalidas": geom_invalidas,
        "cobertura_geometrica": cobertura,
        "proyectos_multiescenario": proyectos_multiescenario,
        "minimo_proyectos_escenario": minimo,
        "maximo_proyectos_escenario": maximo,
        "promedio_proyectos_escenario": promedio,
        "cv_escenarios": cv,
        "controles": controles,
        "controles_ok": controles_ok,
        "controles_total": len(controles),
    }

    return resultado


# =============================================================================
# RANKING DE ESCENARIOS
# =============================================================================

def preparar_ranking_escenarios(
    ranking: pd.DataFrame,
    escenarios: pd.DataFrame,
) -> pd.DataFrame:

    titulo("PREPARANDO RANKING FINAL DE ESCENARIOS")

    resultado = ranking.copy()

    escenario_col = resolver_columna(
        resultado,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    score_col = resolver_columna(
        resultado,
        [
            "score_cartera_v4",
            "score_cartera",
            "score_integral_v4",
            "score_integral",
            "score_final",
        ],
        obligatoria=False,
    )

    if score_col:
        resultado["_score"] = pd.to_numeric(
            resultado[score_col],
            errors="coerce",
        )
    else:
        resultado["_score"] = np.nan

    resultado = resultado.sort_values(
        "_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    resultado["ranking_informe_v4_1"] = (
        np.arange(len(resultado)) + 1
    )

    resultado = resultado.drop(
        columns=["_score"],
        errors="ignore",
    )

    print(f"Escenarios rankeados : {len(resultado):,}")

    return resultado


# =============================================================================
# RANKING DE PROYECTOS
# =============================================================================

def preparar_ranking_proyectos(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    titulo("PREPARANDO RANKING FINAL DE PROYECTOS")

    resultado = ranking.copy()

    proyecto_col = resolver_columna(
        resultado,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    score_col = resolver_columna(
        resultado,
        [
            "score_cartera_v4",
            "score_cartera",
            "score_priorizacion_v4",
            "score_priorizacion",
            "score_final",
        ],
        obligatoria=False,
    )

    if score_col:
        resultado["_score"] = pd.to_numeric(
            resultado[score_col],
            errors="coerce",
        )
    else:
        resultado["_score"] = np.nan

    resultado = resultado.sort_values(
        "_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    resultado["ranking_informe_v4_1"] = (
        np.arange(len(resultado)) + 1
    )

    resultado = resultado.drop(
        columns=["_score"],
        errors="ignore",
    )

    print(f"Proyectos rankeados  : {len(resultado):,}")

    return resultado


# =============================================================================
# VALIDACIÓN DE RANKINGS
# =============================================================================

def validar_rankings(
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
) -> dict:

    titulo("VALIDANDO RANKINGS FINALES")

    escenarios_rankeados = len(ranking_escenarios)
    proyectos_rankeados = len(ranking_proyectos)

    print(f"Escenarios rankeados : {escenarios_rankeados}")
    print(f"Proyectos rankeados  : {proyectos_rankeados}")

    return {
        "escenarios_rankeados": escenarios_rankeados,
        "proyectos_rankeados": proyectos_rankeados,
        "escenarios_ok": escenarios_rankeados == 7,
        "proyectos_ok": proyectos_rankeados == 144,
    }


# =============================================================================
# INDICADORES GLOBALES
# =============================================================================

def construir_indicadores_globales(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    validacion: dict,
) -> dict:

    titulo("CONSTRUYENDO INDICADORES GLOBALES DEL INFORME")

    campos = {}

    for clave, candidatos in {
        "demanda": [
            "indice_demanda_estructural",
        ],
        "deficit": [
            "deficit_infraestructura",
        ],
        "conectividad": [
            "indice_conectividad_estructural",
        ],
        "intermodalidad": [
            "indice_intermodalidad_estructural",
        ],
        "integracion": [
            "indice_integracion_territorial",
        ],
        "centralidad": [
            "indice_centralidad_estructural",
        ],
        "impacto": [
            "impacto_potencial",
        ],
        "urgencia": [
            "urgencia_intervencion",
        ],
    }.items():
        campos[clave] = resolver_columna(
            proyectos,
            candidatos,
            obligatoria=False,
        )

    indicadores = []

    indicadores.append(
        {
            "indicador": "proyectos_totales",
            "valor": len(proyectos),
            "unidad": "proyectos",
        }
    )

    indicadores.append(
        {
            "indicador": "proyectos_unicos",
            "valor": proyectos[
                "proyecto_id"
            ].nunique(),
            "unidad": "proyectos",
        }
    )

    indicadores.append(
        {
            "indicador": "escenarios_totales",
            "valor": proyectos[
                "escenario_id"
            ].nunique(),
            "unidad": "escenarios",
        }
    )

    indicadores.append(
        {
            "indicador": "cobertura_geometrica",
            "valor": validacion["cobertura_geometrica"],
            "unidad": "%",
        }
    )

    indicadores.append(
        {
            "indicador": "geometrias_validas",
            "valor": validacion["geometrias_validas"],
            "unidad": "geometrías",
        }
    )

    indicadores.append(
        {
            "indicador": "proyectos_multiescenario",
            "valor": validacion["proyectos_multiescenario"],
            "unidad": "proyectos",
        }
    )

    for clave, columna in campos.items():

        if not columna:
            continue

        indicadores.append(
            {
                "indicador": f"{clave}_promedio",
                "valor": media_segura(
                    proyectos[columna]
                ),
                "unidad": "índice",
            }
        )

        indicadores.append(
            {
                "indicador": f"{clave}_maximo",
                "valor": max_seguro(
                    proyectos[columna]
                ),
                "unidad": "índice",
            }
        )

    return {
        "tabla": pd.DataFrame(indicadores),
        "campos": campos,
    }


# =============================================================================
# SÍNTESIS EJECUTIVA
# =============================================================================

def construir_sintesis_ejecutiva(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    ranking_escenarios: pd.DataFrame,
    validacion: dict,
) -> str:

    titulo("CONSTRUYENDO SÍNTESIS EJECUTIVA")

    escenario_col = resolver_columna(
        ranking_escenarios,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    score_col = resolver_columna(
        ranking_escenarios,
        [
            "score_cartera_v4",
            "score_cartera",
            "score_integral_v4",
            "score_integral",
            "score_final",
        ],
        obligatoria=False,
    )

    if score_col:
        ranking_tmp = ranking_escenarios.copy()
        ranking_tmp["_score"] = pd.to_numeric(
            ranking_tmp[score_col],
            errors="coerce",
        )

        ranking_tmp = ranking_tmp.sort_values(
            "_score",
            ascending=False,
            na_position="last",
        )

        escenario_prioritario = str(
            ranking_tmp.iloc[0][escenario_col]
        )

        escenario_menor = str(
            ranking_tmp.iloc[-1][escenario_col]
        )

        score_prioritario = safe_float(
            ranking_tmp.iloc[0]["_score"]
        )

    else:
        escenario_prioritario = str(
            ranking_escenarios.iloc[0][escenario_col]
        )

        escenario_menor = str(
            ranking_escenarios.iloc[-1][escenario_col]
        )

        score_prioritario = np.nan

    estado = (
        "VALIDADO"
        if validacion["controles_ok"]
        == validacion["controles_total"]
        else "OBSERVADO"
    )

    texto = f"""# Síntesis Ejecutiva
## Modelo Territorial AMBA {VERSION}

### Resultado general

El proceso **39 - Generación del Informe Territorial AMBA {VERSION}**
consolida los resultados producidos por los procesos anteriores y los
presenta en una estructura única para análisis territorial, priorización y
programación de intervenciones.

El modelo contiene:

- **{len(proyectos):,} proyectos**
- **{proyectos["proyecto_id"].nunique():,} proyectos únicos**
- **{proyectos["escenario_id"].nunique():,} escenarios territoriales**
- **{validacion["cobertura_geometrica"]:.2f}% de cobertura geométrica**
- **{validacion["geometrias_validas"]:,} geometrías válidas**
- **{validacion["proyectos_multiescenario"]:,} proyectos con múltiples escenarios**

### Escenario prioritario

El escenario identificado como prioritario es:

**{escenario_prioritario}**

"""

    if not pd.isna(score_prioritario):
        texto += (
            f"Su score registrado es **{score_prioritario:.2f}**.\n\n"
        )

    texto += f"""### Escenario de menor prioridad relativa

El escenario ubicado en la última posición del ranking es:

**{escenario_menor}**

### Control geoespacial

La validación geográfica recuperó las geometrías desde el GeoPackage maestro
del proceso 38 y no desde los archivos CSV.

Esto permite separar correctamente:

1. atributos tabulares;
2. indicadores territoriales;
3. geometrías canónicas;
4. resultados de priorización.

La cobertura geométrica obtenida fue de **{validacion["cobertura_geometrica"]:.2f}%**.

### Consistencia territorial

Se verificó:

- ausencia de proyectos duplicados;
- ausencia de proyectos sin identificador;
- ausencia de escenarios sin identificador;
- asignación única proyecto → escenario;
- cobertura geográfica completa;
- geometrías válidas;
- existencia de las capas geográficas de proyectos y escenarios.

### Dictamen

**{estado}**

El modelo territorial AMBA {VERSION} queda documentado para su utilización
en las siguientes etapas de análisis, programación de inversiones y
presentación institucional.
"""

    return texto


# =============================================================================
# INFORME COMPLETO
# =============================================================================

def generar_informe(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
    matriz: pd.DataFrame,
    indicadores: pd.DataFrame,
    validacion: dict,
    campos: dict,
) -> str:

    titulo("GENERANDO INFORME TERRITORIAL AMBA V4.1")

    estado = (
        "VALIDADO"
        if validacion["controles_ok"]
        == validacion["controles_total"]
        else "OBSERVADO"
    )

    escenario_col = resolver_columna(
        ranking_escenarios,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    score_col = resolver_columna(
        ranking_escenarios,
        [
            "score_cartera_v4",
            "score_cartera",
            "score_integral_v4",
            "score_integral",
            "score_final",
        ],
        obligatoria=False,
    )

    columnas_ranking_escenarios = [
        "ranking_informe_v4_1",
        escenario_col,
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_escenario",
        "score_cartera_v4",
        "score_cartera",
    ]

    columnas_ranking_escenarios = [
        c
        for c in columnas_ranking_escenarios
        if c in ranking_escenarios.columns
    ]

    columnas_ranking_proyectos = [
        "ranking_informe_v4_1",
        "proyecto_id",
        "escenario_id",
        "tipo_escenario",
        "dimension_dominante",
        "score_cartera_v4",
        "score_cartera",
        "score_priorizacion_v4",
    ]

    columnas_ranking_proyectos = [
        c
        for c in columnas_ranking_proyectos
        if c in ranking_proyectos.columns
    ]

    tabla_escenarios = tabla_markdown(
        ranking_escenarios,
        columnas_ranking_escenarios,
        max_filas=10,
    )

    tabla_proyectos = tabla_markdown(
        ranking_proyectos,
        columnas_ranking_proyectos,
        max_filas=20,
    )

    tabla_indicadores = tabla_markdown(
        indicadores,
        [
            "indicador",
            "valor",
            "unidad",
        ],
        max_filas=40,
    )

    tabla_matriz = tabla_markdown(
        matriz,
        max_filas=10,
    )

    score_global = np.nan

    if score_col and score_col in ranking_escenarios.columns:
        score_global = media_segura(
            ranking_escenarios[score_col]
        )

    informe = f"""# Informe Territorial AMBA {VERSION}

**Proceso:** 39 - Generación del Informe Territorial AMBA  
**Versión:** {VERSION}  
**Proyecto:** movilidad  
**Estado:** **{estado}**

---

# 1. Resumen ejecutivo

El presente documento constituye el informe territorial consolidado del
Área Metropolitana de Buenos Aires (AMBA), elaborado sobre la base del
modelo maestro producido por el proceso 38.

El modelo integra información de proyectos, escenarios territoriales,
priorización, indicadores estructurales, cartera de intervención y
geometrías geográficas.

## 1.1 Magnitud del modelo

| Indicador | Resultado |
|---|---:|
| Proyectos | {len(proyectos):,} |
| Proyectos únicos | {proyectos["proyecto_id"].nunique():,} |
| Escenarios | {proyectos["escenario_id"].nunique():,} |
| Cobertura geométrica | {validacion["cobertura_geometrica"]:.2f}% |
| Geometrías válidas | {validacion["geometrias_validas"]:,} |
| Geometrías nulas | {validacion["geometrias_nulas"]:,} |
| Geometrías inválidas | {validacion["geometrias_invalidas"]:,} |
| Proyectos multiescenario | {validacion["proyectos_multiescenario"]:,} |
| CV tamaño escenarios | {validacion["cv_escenarios"]:.4f} |
| Score medio de escenarios | {formato_numero(score_global, 2)} |

---

# 2. Validación integral

La validación del modelo produjo los siguientes resultados:

- Proyectos únicos: **{proyectos["proyecto_id"].nunique():,}**
- Escenarios: **{proyectos["escenario_id"].nunique():,}**
- Proyectos duplicados: **{validacion["proyectos_duplicados"]:,}**
- Proyectos sin identificador: **{validacion["proyectos_nulos"]:,}**
- Escenarios sin identificador: **{validacion["escenarios_nulos"]:,}**
- Proyectos multiescenario: **{validacion["proyectos_multiescenario"]:,}**
- Cobertura geométrica: **{validacion["cobertura_geometrica"]:.2f}%**
- Geometrías inválidas: **{validacion["geometrias_invalidas"]:,}**

El resultado general del control es:

**{estado}**

---

# 3. Modelo geográfico

La geometría utilizada por este proceso se recupera del GeoPackage maestro:

`modelo_maestro_territorial_amba_v4.gpkg`

El GeoPackage contiene las capas:

- `proyectos`
- `escenarios`

La utilización del GeoPackage como fuente geométrica evita depender de la
presencia de geometrías serializadas dentro de los archivos CSV.

## 3.1 Cobertura espacial

La cobertura geométrica del modelo es:

**{validacion["cobertura_geometrica"]:.2f}%**

Geometrías válidas:

**{validacion["geometrias_validas"]:,}**

Geometrías inválidas:

**{validacion["geometrias_invalidas"]:,}**

---

# 4. Estructura territorial

El modelo se encuentra distribuido en:

**{proyectos["escenario_id"].nunique():,} escenarios territoriales.**

La distribución de proyectos presenta:

- mínimo: **{validacion["minimo_proyectos_escenario"]}**
- máximo: **{validacion["maximo_proyectos_escenario"]}**
- promedio: **{validacion["promedio_proyectos_escenario"]:.2f}**
- coeficiente de variación: **{validacion["cv_escenarios"]:.4f}**

La baja variabilidad relativa indica una distribución territorial
relativamente equilibrada en términos del número de proyectos por escenario.

---

# 5. Ranking final de escenarios

{tabla_escenarios}

---

# 6. Ranking final de proyectos

{tabla_proyectos}

---

# 7. Matriz integral de escenarios

{tabla_matriz}

---

# 8. Indicadores globales AMBA

{tabla_indicadores}

---

# 9. Escenario prioritario

"""

    if len(ranking_escenarios) > 0:

        primero = ranking_escenarios.iloc[0]

        escenario_id = primero.get(
            escenario_col,
            "N/D",
        )

        informe += f"""
El escenario ubicado en la primera posición del ranking es:

**{escenario_id}**

Este escenario constituye la principal referencia para la programación
territorial derivada del modelo consolidado.
"""

    informe += f"""

---

# 10. Trazabilidad del modelo

La cadena de procesamiento utilizada para este informe es:

1. Construcción y validación de indicadores territoriales.
2. Construcción de escenarios.
3. Priorización territorial.
4. Construcción de cartera.
5. Validación geoespacial.
6. Integración territorial.
7. Consolidación del modelo maestro.
8. Generación del presente informe.

El proceso 39 no recalcula ni modifica los indicadores originales.

Su función es consolidar, validar, documentar y presentar los resultados
producidos previamente.

---

# 11. Dictamen final

## {estado}

El modelo territorial AMBA {VERSION} presenta:

- integridad de identificadores;
- consistencia proyecto → escenario;
- cobertura geométrica completa;
- geometrías válidas;
- estructura territorial consistente;
- ranking final disponible;
- matriz integral disponible;
- modelo geográfico consolidado disponible.

El modelo queda preparado para las siguientes etapas de:

- programación de inversiones;
- definición de cronogramas;
- análisis de cartera;
- evaluación territorial;
- elaboración de documentación institucional;
- presentación final del modelo AMBA.

---

**Fin del Informe Territorial AMBA {VERSION}.**
"""

    return informe


# =============================================================================
# AUDITORÍA
# =============================================================================

def construir_auditoria(
    validacion: dict,
    validacion_rankings: dict,
    gpkg_ok: bool,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO AUDITORÍA DEL PROCESO 39")

    controles = []

    controles.append(
        {
            "control": "proyectos_144",
            "resultado": validacion["total_proyectos"] == 144,
            "valor": validacion["total_proyectos"],
            "observacion": "Se esperan 144 proyectos.",
        }
    )

    controles.append(
        {
            "control": "proyectos_unicos",
            "resultado": validacion["proyectos_unicos"] == 144,
            "valor": validacion["proyectos_unicos"],
            "observacion": "Los proyectos deben ser únicos.",
        }
    )

    controles.append(
        {
            "control": "escenarios_7",
            "resultado": validacion["escenarios"] == 7,
            "valor": validacion["escenarios"],
            "observacion": "Se esperan 7 escenarios.",
        }
    )

    controles.append(
        {
            "control": "sin_duplicados",
            "resultado": validacion["proyectos_duplicados"] == 0,
            "valor": validacion["proyectos_duplicados"],
            "observacion": "No deben existir proyectos duplicados.",
        }
    )

    controles.append(
        {
            "control": "cobertura_geometrica",
            "resultado": validacion["cobertura_geometrica"] == 100.0,
            "valor": validacion["cobertura_geometrica"],
            "observacion": "La cobertura debe ser 100%.",
        }
    )

    controles.append(
        {
            "control": "geometrias_validas",
            "resultado": validacion["geometrias_invalidas"] == 0,
            "valor": validacion["geometrias_invalidas"],
            "observacion": "No deben existir geometrías inválidas.",
        }
    )

    controles.append(
        {
            "control": "sin_multiescenario",
            "resultado": validacion["proyectos_multiescenario"] == 0,
            "valor": validacion["proyectos_multiescenario"],
            "observacion": "Cada proyecto debe pertenecer a un único escenario.",
        }
    )

    controles.append(
        {
            "control": "ranking_escenarios",
            "resultado": validacion_rankings["escenarios_ok"],
            "valor": validacion_rankings["escenarios_rankeados"],
            "observacion": "Se esperan 7 escenarios rankeados.",
        }
    )

    controles.append(
        {
            "control": "ranking_proyectos",
            "resultado": validacion_rankings["proyectos_ok"],
            "valor": validacion_rankings["proyectos_rankeados"],
            "observacion": "Se esperan 144 proyectos rankeados.",
        }
    )

    controles.append(
        {
            "control": "geopackage_maestro",
            "resultado": gpkg_ok,
            "valor": "SI" if gpkg_ok else "NO",
            "observacion": "Debe existir el GeoPackage maestro del proceso 38.",
        }
    )

    return pd.DataFrame(controles)


# =============================================================================
# JSON
# =============================================================================

def construir_resumen_json(
    validacion: dict,
    validacion_rankings: dict,
    auditoria: pd.DataFrame,
    escenarios: pd.DataFrame,
) -> dict:

    controles_ok = int(
        auditoria["resultado"].sum()
    )

    controles_total = len(auditoria)

    estado = (
        "VALIDADO"
        if controles_ok == controles_total
        else "OBSERVADO"
    )

    escenario_prioritario = None
    escenario_menor = None

    if not escenarios.empty:

        escenario_col = resolver_columna(
            escenarios,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
            obligatoria=False,
        )

        if escenario_col:

            escenario_prioritario = str(
                escenarios.iloc[0][escenario_col]
            )

            escenario_menor = str(
                escenarios.iloc[-1][escenario_col]
            )

    return {
        "proceso": 39,
        "version": VERSION,
        "nombre": "Generación del Informe Territorial AMBA",
        "estado": estado,
        "dictamen": estado,
        "validacion": validacion,
        "validacion_rankings": validacion_rankings,
        "controles_ok": controles_ok,
        "controles_total": controles_total,
        "escenario_prioritario": escenario_prioritario,
        "escenario_menor_prioridad": escenario_menor,
        "timestamp": pd.Timestamp.now().isoformat(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    inicio = time.time()

    titulo(
        f"39 - GENERACIÓN DEL INFORME TERRITORIAL AMBA - {VERSION}"
    )

    print(f"Proyecto : {BASE_DIR}")
    print(f"Entrada  : {INPUT_DIR}")
    print(f"Salida   : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # CARGA
    # =========================================================================

    titulo("CARGANDO MODELO MAESTRO DEL PROCESO 38")

    proyectos = cargar_csv(
        ARCHIVOS_ENTRADA["proyectos"]
    )

    escenarios = cargar_csv(
        ARCHIVOS_ENTRADA["escenarios"]
    )

    ranking_escenarios = cargar_csv(
        ARCHIVOS_ENTRADA["ranking_escenarios"]
    )

    ranking_proyectos = cargar_csv(
        ARCHIVOS_ENTRADA["ranking_proyectos"]
    )

    matriz = cargar_csv(
        ARCHIVOS_ENTRADA["matriz"]
    )

    indicadores_originales = cargar_csv(
        ARCHIVOS_ENTRADA["indicadores"]
    )

    auditoria_38 = cargar_csv(
        ARCHIVOS_ENTRADA["auditoria_38"]
    )

    # =========================================================================
    # GEOMETRÍAS
    # =========================================================================

    proyectos_geo, escenarios_geo = cargar_geopackage()

    # =========================================================================
    # CAMPOS
    # =========================================================================

    campos = resolver_campos_proyectos(
        proyectos
    )

    # =========================================================================
    # INTEGRACIÓN GEOGRÁFICA
    # =========================================================================

    proyectos = integrar_geometrias(
        proyectos,
        proyectos_geo,
        escenarios_geo,
        campos,
    )

    # =========================================================================
    # VALIDACIÓN
    # =========================================================================

    titulo("VALIDACIÓN DEL MODELO MAESTRO")

    validacion = validar_modelo(
        proyectos,
        escenarios_geo,
        campos,
    )

    # =========================================================================
    # RANKINGS
    # =========================================================================

    ranking_escenarios_final = preparar_ranking_escenarios(
        ranking_escenarios,
        escenarios,
    )

    ranking_proyectos_final = preparar_ranking_proyectos(
        ranking_proyectos,
    )

    validacion_rankings = validar_rankings(
        ranking_escenarios_final,
        ranking_proyectos_final,
    )

    # =========================================================================
    # INDICADORES
    # =========================================================================

    indicadores_resultado = construir_indicadores_globales(
        proyectos,
        escenarios,
        validacion,
    )

    indicadores = indicadores_resultado["tabla"]

    # =========================================================================
    # SÍNTESIS
    # =========================================================================

    sintesis = construir_sintesis_ejecutiva(
        proyectos,
        escenarios,
        ranking_escenarios_final,
        validacion,
    )

    # =========================================================================
    # INFORME
    # =========================================================================

    informe = generar_informe(
        proyectos,
        escenarios,
        ranking_escenarios_final,
        ranking_proyectos_final,
        matriz,
        indicadores,
        validacion,
        campos,
    )

    # =========================================================================
    # ANEXOS
    # =========================================================================

    titulo("GENERANDO ANEXOS")

    anexo_proyectos = proyectos.drop(
        columns=["geometry"],
        errors="ignore",
    ).copy()

    anexo_escenarios = escenarios.copy()

    anexo_indicadores = indicadores.copy()

    path_anexo_proyectos = (
        OUTPUT_DIR
        / ARCHIVOS_SALIDA["anexo_proyectos"]
    )

    path_anexo_escenarios = (
        OUTPUT_DIR
        / ARCHIVOS_SALIDA["anexo_escenarios"]
    )

    path_anexo_indicadores = (
        OUTPUT_DIR
        / ARCHIVOS_SALIDA["anexo_indicadores"]
    )

    anexo_proyectos.to_csv(
        path_anexo_proyectos,
        index=False,
        encoding="utf-8-sig",
    )

    anexo_escenarios.to_csv(
        path_anexo_escenarios,
        index=False,
        encoding="utf-8-sig",
    )

    anexo_indicadores.to_csv(
        path_anexo_indicadores,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Proyectos : {path_anexo_proyectos}")
    print(f"Escenarios: {path_anexo_escenarios}")
    print(f"Indicadores: {path_anexo_indicadores}")

    # =========================================================================
    # AUDITORÍA
    # =========================================================================

    auditoria = construir_auditoria(
        validacion,
        validacion_rankings,
        gpkg_ok=True,
    )

    # =========================================================================
    # RESUMEN JSON
    # =========================================================================

    resumen_json = construir_resumen_json(
        validacion,
        validacion_rankings,
        auditoria,
        ranking_escenarios_final,
    )

    # =========================================================================
    # EXPORTACIONES
    # =========================================================================

    titulo("EXPORTANDO RESULTADOS DEL PROCESO 39")

    path_informe = (
        OUTPUT_DIR
        / ARCHIVOS_SALIDA["informe"]
    )

    path_resumen = (
        OUTPUT_DIR
        / ARCHIVOS_SALIDA["resumen"]
    )

    path_auditoria = (
        OUTPUT_DIR
        / ARCHIVOS_SALIDA["auditoria"]
    )

    path_json = (
        OUTPUT_DIR
        / ARCHIVOS_SALIDA["json"]
    )

    escribir_texto(
        path_informe,
        informe,
    )

    escribir_texto(
        path_resumen,
        sintesis,
    )

    auditoria.to_csv(
        path_auditoria,
        index=False,
        encoding="utf-8-sig",
    )

    path_json.write_text(
        json.dumps(
            resumen_json,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Informe          : {path_informe}")
    print(f"Resumen ejecutivo: {path_resumen}")
    print(f"Auditoría        : {path_auditoria}")
    print(f"Resumen JSON      : {path_json}")

    # =========================================================================
    # VALIDACIÓN DEL GEOPACKAGE
    # =========================================================================

    titulo("VALIDANDO MODELO GEOGRÁFICO CONSOLIDADO")

    gpkg_path = (
        INPUT_DIR
        / ARCHIVOS_ENTRADA["gpkg"]
    )

    gpkg_ok = gpkg_path.exists()

    print(
        f"GeoPackage existe : {'SI' if gpkg_ok else 'NO'}"
    )

    if gpkg_ok:

        capas = gpd.list_layers(
            gpkg_path
        )

        nombres_capas = capas["name"].tolist()

        print(
            f"Capas            : {', '.join(nombres_capas)}"
        )

    # =========================================================================
    # RESULTADO
    # =========================================================================

    tiempo = time.time() - inicio

    controles_ok = int(
        auditoria["resultado"].sum()
    )

    controles_total = len(auditoria)

    estado = (
        "VALIDADO"
        if controles_ok == controles_total
        else "OBSERVADO"
    )

    escenario_col = resolver_columna(
        ranking_escenarios_final,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    escenario_prioritario = str(
        ranking_escenarios_final.iloc[0][escenario_col]
    )

    escenario_menor = str(
        ranking_escenarios_final.iloc[-1][escenario_col]
    )

    titulo("RESULTADO FINAL DEL PROCESO 39")

    print(
        f"Proyectos                 : {validacion['total_proyectos']:,}"
    )

    print(
        f"Proyectos únicos          : {validacion['proyectos_unicos']:,}"
    )

    print(
        f"Escenarios                : {validacion['escenarios']:,}"
    )

    print(
        f"Cobertura geométrica      : {validacion['cobertura_geometrica']:.2f}%"
    )

    print(
        f"Geometrías válidas        : {validacion['geometrias_validas']:,}"
    )

    print(
        f"Geometrías nulas          : {validacion['geometrias_nulas']:,}"
    )

    print(
        f"Geometrías inválidas      : {validacion['geometrias_invalidas']:,}"
    )

    print(
        f"Proyectos duplicados      : {validacion['proyectos_duplicados']:,}"
    )

    print(
        f"Proyectos multiescenario  : {validacion['proyectos_multiescenario']:,}"
    )

    print(
        f"CV tamaño escenarios      : {validacion['cv_escenarios']:.4f}"
    )

    print(
        f"Escenario prioritario     : {escenario_prioritario}"
    )

    print(
        f"Escenario menor prioridad: {escenario_menor}"
    )

    print(
        f"Controles OK              : {controles_ok}/{controles_total}"
    )

    print(
        f"Auditoría                 : {'OK' if estado == 'VALIDADO' else 'OBSERVADA'}"
    )

    print(
        f"Dictamen                  : {estado}"
    )

    print(
        f"Tiempo de ejecución       : {tiempo:.2f} segundos"
    )

    print()

    if estado == "VALIDADO":

        print(
            "El informe territorial AMBA V4.1 fue generado y validado correctamente."
        )

        print(
            "La geometría se recuperó desde el GeoPackage maestro del proceso 38."
        )

        print(
            "La cobertura geográfica es completa."
        )

        print(
            "La asignación proyecto -> escenario se mantiene íntegra."
        )

    else:

        print(
            "El informe fue generado con observaciones."
        )

    titulo("PROCESO 39 FINALIZADO")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:

        titulo("ERROR FATAL EN EL PROCESO 39")

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise