# -*- coding: utf-8 -*-

"""
========================================================================================
38 - CONSOLIDACIÓN DEL MODELO TERRITORIAL AMBA - V4
========================================================================================

Objetivo
--------
Consolidar los resultados validados de los procesos 31 a 37 en un único modelo
territorial maestro, preservando:

    proyecto -> escenario -> indicadores -> prioridad -> cartera -> geometría

El proceso 38 NO recalcula los indicadores estructurales originales.
Su función es integrar, auditar y consolidar los resultados existentes.

Entradas principales
--------------------
    escenarios_territoriales_amba_v4.parquet
    ranking_escenarios_v4.csv
    comparacion_escenarios_v4.csv
    proyectos_priorizados_v4.csv
    cartera_escenarios_v4.csv
    cartera_territorial_amba_v4.csv
    cartera_proyectos_v4.csv
    validacion_geoespacial_cartera_v4.csv
    modelo_territorial_amba_v4.gpkg

Salidas
-------
    modelo_maestro_territorial_amba_v4.gpkg
    modelo_maestro_proyectos_v4.csv
    modelo_maestro_escenarios_v4.csv
    ranking_final_escenarios_v4.csv
    ranking_final_proyectos_v4.csv
    matriz_integral_escenarios_v4.csv
    indicadores_globales_amba_v4.csv
    auditoria_38_consolidacion_territorial_amba.csv
    resumen_38_consolidacion_territorial_amba.json
    modelo_territorial_amba_v4.md

========================================================================================
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd


# ======================================================================================
# CONFIGURACIÓN
# ======================================================================================

VERSION = "V4"
PROCESO = "38"

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

SALIDA_DIR = DATA_DIR
SALIDA_DIR.mkdir(parents=True, exist_ok=True)

FUENTE_CANONICA = DATA_DIR / "escenarios_territoriales_amba_v4.parquet"

RANKING_32 = DATA_DIR / "ranking_escenarios_v4.csv"
COMPARACION_33 = DATA_DIR / "comparacion_escenarios_v4.csv"

PRIORIZACION_34 = DATA_DIR / "priorizacion_territorial_escenarios_v4.csv"
PROYECTOS_34 = DATA_DIR / "proyectos_priorizados_v4.csv"

CARTERA_35 = DATA_DIR / "cartera_territorial_amba_v4.csv"
CARTERA_ESCENARIOS_35 = DATA_DIR / "cartera_escenarios_v4.csv"
CARTERA_PROYECTOS_35 = DATA_DIR / "cartera_proyectos_v4.csv"

VALIDACION_36 = DATA_DIR / "validacion_geoespacial_cartera_v4.csv"

MODELO_37_GPKG = DATA_DIR / "modelo_territorial_amba_v4.gpkg"


# Salidas del proceso 38

SALIDA_PROYECTOS = (
    SALIDA_DIR / "modelo_maestro_proyectos_v4.csv"
)

SALIDA_ESCENARIOS = (
    SALIDA_DIR / "modelo_maestro_escenarios_v4.csv"
)

SALIDA_RANKING_ESCENARIOS = (
    SALIDA_DIR / "ranking_final_escenarios_v4.csv"
)

SALIDA_RANKING_PROYECTOS = (
    SALIDA_DIR / "ranking_final_proyectos_v4.csv"
)

SALIDA_MATRIZ = (
    SALIDA_DIR / "matriz_integral_escenarios_v4.csv"
)

SALIDA_INDICADORES = (
    SALIDA_DIR / "indicadores_globales_amba_v4.csv"
)

SALIDA_AUDITORIA = (
    SALIDA_DIR
    / "auditoria_38_consolidacion_territorial_amba.csv"
)

SALIDA_RESUMEN = (
    SALIDA_DIR
    / "resumen_38_consolidacion_territorial_amba.json"
)

SALIDA_GPKG = (
    SALIDA_DIR
    / "modelo_maestro_territorial_amba_v4.gpkg"
)

SALIDA_MARKDOWN = (
    SALIDA_DIR
    / "modelo_territorial_amba_v4.md"
)


# ======================================================================================
# UTILIDADES
# ======================================================================================

def titulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def limpiar_valor(valor):
    if pd.isna(valor):
        return None

    if isinstance(valor, (np.integer,)):
        return int(valor)

    if isinstance(valor, (np.floating,)):
        if not np.isfinite(valor):
            return None
        return float(valor)

    if isinstance(valor, (np.bool_,)):
        return bool(valor)

    return valor


def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte tipos problemáticos para exportaciones CSV/JSON.
    """
    result = df.copy()

    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].astype(str)

    return result


def resolver_columna(
    df: pd.DataFrame,
    candidatos: Iterable[str],
    requerida: bool = True,
) -> Optional[str]:
    """
    Busca una columna por nombre exacto, ignorando mayúsculas/minúsculas.
    """
    mapa = {str(c).lower(): c for c in df.columns}

    for candidato in candidatos:
        if candidato.lower() in mapa:
            return mapa[candidato.lower()]

    if requerida:
        raise KeyError(
            "No se encontró ninguna de las columnas esperadas: "
            + str(list(candidatos))
        )

    return None


def leer_csv_seguro(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"ADVERTENCIA: archivo no encontrado: {path}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"Cargando: {path.name}")
        print(f"Registros : {len(df)}")
        print(f"Columnas  : {len(df.columns)}")
        return df
    except Exception as exc:
        print(
            f"ADVERTENCIA: no se pudo cargar {path.name}: {exc}"
        )
        return None


def normalizar_id_serie(serie: pd.Series) -> pd.Series:
    """
    Normaliza identificadores sin alterar su contenido conceptual.
    """
    return (
        serie.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def normalizar_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in [
        "proyecto_id",
        "escenario_id",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
        "prioridad_territorial_v4",
        "prioridad_operativa_v4",
        "categoria_cartera_v4",
        "linea_estrategica_v4",
        "horizonte_intervencion_v4",
        "categoria_cartera",
    ]:
        if col in result.columns:
            result[col] = normalizar_id_serie(result[col])

    return result


def promedio_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce").dropna()

    if valores.empty:
        return 0.0

    return float(valores.mean())


def suma_segura(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce").dropna()

    if valores.empty:
        return 0.0

    return float(valores.sum())


def max_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce").dropna()

    if valores.empty:
        return 0.0

    return float(valores.max())


def min_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce").dropna()

    if valores.empty:
        return 0.0

    return float(valores.min())


def coef_variacion(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce").dropna()

    if len(valores) == 0:
        return 0.0

    media = valores.mean()

    if media == 0:
        return 0.0

    return float(valores.std(ddof=0) / media)


def unir_unicos(valores: Iterable) -> str:
    encontrados = []

    for valor in valores:
        if pd.isna(valor):
            continue

        texto = str(valor).strip()

        if not texto:
            continue

        if texto not in encontrados:
            encontrados.append(texto)

    return " | ".join(encontrados)


def score_normalizado_0_100(serie: pd.Series) -> pd.Series:
    valores = pd.to_numeric(serie, errors="coerce")

    if valores.notna().sum() == 0:
        return pd.Series(
            0.0,
            index=serie.index,
        )

    minimo = valores.min()
    maximo = valores.max()

    if maximo == minimo:
        return pd.Series(
            100.0,
            index=serie.index,
        )

    return ((valores - minimo) / (maximo - minimo) * 100).fillna(0.0)


def dataframe_to_markdown(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
) -> str:
    """
    Genera Markdown manualmente.
    No requiere tabulate.
    """
    if df is None or df.empty:
        return "_Sin datos._"

    data = df.copy()

    if max_rows is not None:
        data = data.head(max_rows)

    headers = [str(c) for c in data.columns]

    lines = []

    lines.append(
        "| "
        + " | ".join(headers)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(["---"] * len(headers))
        + " |"
    )

    for _, row in data.iterrows():
        valores = []

        for value in row:
            if pd.isna(value):
                texto = ""
            else:
                texto = str(value)

            texto = texto.replace("|", "\\|")
            texto = texto.replace("\n", " ")

            valores.append(texto)

        lines.append(
            "| "
            + " | ".join(valores)
            + " |"
        )

    return "\n".join(lines)


# ======================================================================================
# CARGA DE FUENTE CANÓNICA
# ======================================================================================

def cargar_fuente_canonica() -> gpd.GeoDataFrame:
    titulo("CARGANDO FUENTE CANÓNICA V4")

    if not FUENTE_CANONICA.exists():
        raise FileNotFoundError(
            f"No existe la fuente canónica:\n{FUENTE_CANONICA}"
        )

    gdf = gpd.read_parquet(FUENTE_CANONICA)

    print(f"Registros : {len(gdf)}")
    print(f"Columnas  : {len(gdf.columns)}")
    print(f"CRS       : {gdf.crs}")

    return gdf


# ======================================================================================
# RESOLUCIÓN DE CAMPOS
# ======================================================================================

def resolver_campos(gdf: gpd.GeoDataFrame) -> dict:
    titulo("RESOLUCIÓN DE CAMPOS")

    campos = {}

    campos["proyecto"] = resolver_columna(
        gdf,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    campos["escenario"] = resolver_columna(
        gdf,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    campos["tipo"] = resolver_columna(
        gdf,
        [
            "tipo_escenario",
            "tipo",
        ],
        requerida=False,
    )

    campos["dimension"] = resolver_columna(
        gdf,
        [
            "dimension_dominante",
            "dimension_escenario",
            "dimension",
        ],
        requerida=False,
    )

    campos["prioridad"] = resolver_columna(
        gdf,
        [
            "prioridad_escenario",
            "prioridad_territorial_v4",
            "prioridad_territorial",
            "prioridad",
        ],
        requerida=False,
    )

    campos["score_cartera"] = resolver_columna(
        gdf,
        [
            "score_cartera",
            "score_cartera_v4",
        ],
        requerida=False,
    )

    campos["demanda"] = resolver_columna(
        gdf,
        [
            "indice_demanda_estructural",
            "score_demanda_v4",
        ],
        requerida=False,
    )

    campos["deficit"] = resolver_columna(
        gdf,
        [
            "deficit_infraestructura",
            "score_deficit_v4",
        ],
        requerida=False,
    )

    campos["conectividad"] = resolver_columna(
        gdf,
        [
            "indice_conectividad_estructural",
            "score_conectividad_v4",
        ],
        requerida=False,
    )

    campos["intermodalidad"] = resolver_columna(
        gdf,
        [
            "indice_intermodalidad_estructural",
            "score_intermodalidad_v4",
        ],
        requerida=False,
    )

    campos["integracion"] = resolver_columna(
        gdf,
        [
            "indice_integracion_territorial",
            "score_integracion_v4",
        ],
        requerida=False,
    )

    campos["centralidad"] = resolver_columna(
        gdf,
        [
            "indice_centralidad_estructural",
            "score_centralidad_v4",
        ],
        requerida=False,
    )

    campos["impacto"] = resolver_columna(
        gdf,
        [
            "impacto_potencial",
            "score_impacto_v4",
        ],
        requerida=False,
    )

    campos["urgencia"] = resolver_columna(
        gdf,
        [
            "urgencia_intervencion",
            "score_urgencia_v4",
        ],
        requerida=False,
    )

    campos["score_territorial"] = resolver_columna(
        gdf,
        [
            "score_prioridad_territorial",
            "score_priorizacion_v4",
            "score_territorial_v4",
        ],
        requerida=False,
    )

    campos["geometry"] = "geometry"

    for nombre, columna in campos.items():
        if columna is None:
            print(f"{nombre:24}: NO DISPONIBLE")
        else:
            print(f"{nombre:24}: {columna}")

    return campos


# ======================================================================================
# CARGA DE RESULTADOS COMPLEMENTARIOS
# ======================================================================================

def cargar_resultados_complementarios() -> dict:
    titulo("CARGANDO RESULTADOS COMPLEMENTARIOS")

    archivos = {
        "ranking_32": RANKING_32,
        "comparacion_33": COMPARACION_33,
        "priorizacion_34": PRIORIZACION_34,
        "proyectos_34": PROYECTOS_34,
        "cartera_35": CARTERA_35,
        "cartera_escenarios_35": CARTERA_ESCENARIOS_35,
        "cartera_proyectos_35": CARTERA_PROYECTOS_35,
        "validacion_36": VALIDACION_36,
    }

    resultados = {}

    for nombre, path in archivos.items():
        resultados[nombre] = leer_csv_seguro(path)

    return resultados


# ======================================================================================
# INTEGRACIÓN DE RESULTADOS
# ======================================================================================

def integrar_resultados(
    gdf: gpd.GeoDataFrame,
    campos: dict,
    resultados: dict,
) -> gpd.GeoDataFrame:

    titulo("INTEGRANDO RESULTADOS DE LOS PROCESOS 31-37")

    proyecto_col = campos["proyecto"]
    escenario_col = campos["escenario"]

    maestro = gdf.copy()

    maestro["proyecto_id"] = normalizar_id_serie(
        maestro[proyecto_col]
    )

    maestro["escenario_id"] = normalizar_id_serie(
        maestro[escenario_col]
    )

    if campos["tipo"]:
        maestro["tipo_escenario"] = maestro[
            campos["tipo"]
        ]

    if campos["dimension"]:
        maestro["dimension_dominante"] = maestro[
            campos["dimension"]
        ]

    if campos["prioridad"]:
        maestro["prioridad_escenario"] = maestro[
            campos["prioridad"]
        ]

    # ------------------------------------------------------------------
    # Integración de proceso 34
    # ------------------------------------------------------------------

    p34 = resultados.get("proyectos_34")

    if p34 is not None and not p34.empty:

        p34_proyecto = resolver_columna(
            p34,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
            requerida=False,
        )

        if p34_proyecto:

            p34 = p34.copy()

            p34[p34_proyecto] = normalizar_id_serie(
                p34[p34_proyecto]
            )

            columnas_integrar = [
                c
                for c in [
                    "prioridad_territorial_v4",
                    "prioridad_operativa_v4",
                    "categoria_cartera_v4",
                    "score_priorizacion_v4",
                    "score_demanda_v4",
                    "score_deficit_v4",
                    "score_conectividad_v4",
                    "score_integracion_v4",
                    "score_impacto_v4",
                ]
                if c in p34.columns
            ]

            if columnas_integrar:

                aux = p34[
                    [p34_proyecto] + columnas_integrar
                ].drop_duplicates(
                    subset=[p34_proyecto]
                )

                aux = aux.rename(
                    columns={
                        p34_proyecto: "proyecto_id"
                    }
                )

                maestro = maestro.merge(
                    aux,
                    on="proyecto_id",
                    how="left",
                    suffixes=("", "_p34"),
                )

    # ------------------------------------------------------------------
    # Integración de proceso 35
    # ------------------------------------------------------------------

    p35 = resultados.get("cartera_proyectos_35")

    if p35 is not None and not p35.empty:

        p35_proyecto = resolver_columna(
            p35,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
            requerida=False,
        )

        if p35_proyecto:

            p35 = p35.copy()

            p35[p35_proyecto] = normalizar_id_serie(
                p35[p35_proyecto]
            )

            columnas_integrar = [
                c
                for c in [
                    "categoria_cartera_v4",
                    "linea_estrategica_v4",
                    "horizonte_intervencion_v4",
                    "score_cartera_v4",
                    "prioridad_operativa_v4",
                ]
                if c in p35.columns
            ]

            if columnas_integrar:

                aux = p35[
                    [p35_proyecto] + columnas_integrar
                ].drop_duplicates(
                    subset=[p35_proyecto]
                )

                aux = aux.rename(
                    columns={
                        p35_proyecto: "proyecto_id"
                    }
                )

                maestro = maestro.merge(
                    aux,
                    on="proyecto_id",
                    how="left",
                    suffixes=("", "_p35"),
                )

    # ------------------------------------------------------------------
    # Limpieza de columnas duplicadas
    # ------------------------------------------------------------------

    duplicadas = [
        c for c in maestro.columns
        if c.endswith("_p34") or c.endswith("_p35")
    ]

    for col in duplicadas:

        base = col.rsplit("_", 1)[0]

        if base in maestro.columns:

            maestro[base] = maestro[base].where(
                maestro[base].notna(),
                maestro[col],
            )

            maestro.drop(
                columns=[col],
                inplace=True,
            )

    maestro = normalizar_ids(maestro)

    print(
        f"Registros integrados : {len(maestro)}"
    )

    print(
        f"Proyectos únicos      : "
        f"{maestro['proyecto_id'].nunique()}"
    )

    print(
        f"Escenarios            : "
        f"{maestro['escenario_id'].nunique()}"
    )

    return maestro


# ======================================================================================
# VALIDACIÓN DEL MODELO MAESTRO
# ======================================================================================

def validar_modelo(
    maestro: gpd.GeoDataFrame,
) -> dict:

    titulo("VALIDACIÓN INTEGRAL DEL MODELO MAESTRO")

    resultados = {}

    n = len(maestro)

    proyectos_unicos = maestro[
        "proyecto_id"
    ].nunique()

    escenarios_unicos = maestro[
        "escenario_id"
    ].nunique()

    nulos_proyecto = int(
        maestro["proyecto_id"].isna().sum()
    )

    duplicados_proyecto = int(
        maestro["proyecto_id"].duplicated().sum()
    )

    nulos_escenario = int(
        maestro["escenario_id"].isna().sum()
    )

    geometria_nula = int(
        maestro.geometry.isna().sum()
    )

    geometria_vacia = int(
        maestro.geometry.is_empty.sum()
    )

    geometria_valida = int(
        maestro.geometry.is_valid.sum()
    )

    geometria_invalida = int(
        (~maestro.geometry.is_valid).sum()
    )

    cobertura_geom = (
        geometria_valida / n * 100
        if n
        else 0.0
    )

    conteo_escenarios = (
        maestro.groupby("escenario_id")
        .size()
        .sort_values()
    )

    minimo = (
        int(conteo_escenarios.min())
        if not conteo_escenarios.empty
        else 0
    )

    maximo = (
        int(conteo_escenarios.max())
        if not conteo_escenarios.empty
        else 0
    )

    promedio = (
        float(conteo_escenarios.mean())
        if not conteo_escenarios.empty
        else 0.0
    )

    cv = coef_variacion(conteo_escenarios)

    # Un proyecto sólo puede pertenecer a un escenario.
    multi_escenario = (
        maestro.groupby("proyecto_id")[
            "escenario_id"
        ]
        .nunique()
    )

    proyectos_multiescenario = int(
        (multi_escenario > 1).sum()
    )

    resultados.update(
        {
            "registros": n,
            "proyectos_unicos": proyectos_unicos,
            "escenarios": escenarios_unicos,
            "proyecto_id_nulos": nulos_proyecto,
            "proyecto_id_duplicados": duplicados_proyecto,
            "escenario_id_nulos": nulos_escenario,
            "geometrias_nulas": geometria_nula,
            "geometrias_vacias": geometria_vacia,
            "geometrias_validas": geometria_valida,
            "geometrias_invalidas": geometria_invalida,
            "cobertura_geometrica_pct": cobertura_geom,
            "proyectos_multiescenario": proyectos_multiescenario,
            "min_proyectos_escenario": minimo,
            "max_proyectos_escenario": maximo,
            "promedio_proyectos_escenario": promedio,
            "cv_tamano_escenarios": cv,
        }
    )

    print(
        f"Registros                  : {n}"
    )
    print(
        f"Proyectos únicos           : {proyectos_unicos}"
    )
    print(
        f"Escenarios                 : {escenarios_unicos}"
    )
    print(
        f"Proyecto ID nulos          : {nulos_proyecto}"
    )
    print(
        f"Proyecto ID duplicados     : {duplicados_proyecto}"
    )
    print(
        f"Escenario ID nulos         : {nulos_escenario}"
    )
    print(
        f"Geometrías válidas         : {geometria_valida}"
    )
    print(
        f"Geometrías nulas           : {geometria_nula}"
    )
    print(
        f"Geometrías vacías          : {geometria_vacia}"
    )
    print(
        f"Geometrías inválidas       : {geometria_invalida}"
    )
    print(
        f"Cobertura geométrica      : {cobertura_geom:.2f}%"
    )
    print(
        f"Proyectos multiescenario   : {proyectos_multiescenario}"
    )
    print(
        f"Mínimo proyectos/escenario : {minimo}"
    )
    print(
        f"Máximo proyectos/escenario : {maximo}"
    )
    print(
        f"Promedio proyectos/escenario: {promedio:.2f}"
    )
    print(
        f"CV tamaño escenarios       : {cv:.4f}"
    )

    return resultados


# ======================================================================================
# MODELO AGREGADO POR ESCENARIO
# ======================================================================================

def construir_modelo_escenarios(
    maestro: gpd.GeoDataFrame,
    campos: dict,
) -> gpd.GeoDataFrame:

    titulo("CONSTRUYENDO MODELO MAESTRO POR ESCENARIO")

    registros = []

    for escenario_id, grupo in maestro.groupby(
        "escenario_id",
        dropna=False,
    ):

        fila = {
            "escenario_id": escenario_id,
            "cantidad_proyectos": int(len(grupo)),
        }

        # --------------------------------------------------------------
        # Campos conceptuales
        # --------------------------------------------------------------

        for col in [
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
            "prioridad_territorial_v4",
            "prioridad_operativa_v4",
            "categoria_cartera_v4",
            "linea_estrategica_v4",
            "horizonte_intervencion_v4",
        ]:

            if col in grupo.columns:

                fila[col] = unir_unicos(
                    grupo[col]
                )

        # --------------------------------------------------------------
        # Indicadores
        # --------------------------------------------------------------

        indicadores = [
            "indice_demanda_estructural",
            "deficit_infraestructura",
            "indice_conectividad_estructural",
            "indice_intermodalidad_estructural",
            "indice_integracion_territorial",
            "indice_centralidad_estructural",
            "impacto_potencial",
            "urgencia_intervencion",
            "score_prioridad_territorial",
            "score_priorizacion_v4",
            "score_cartera",
            "score_cartera_v4",
        ]

        for indicador in indicadores:

            if indicador in grupo.columns:

                valores = pd.to_numeric(
                    grupo[indicador],
                    errors="coerce",
                )

                fila[f"{indicador}_media"] = (
                    float(valores.mean())
                    if valores.notna().any()
                    else 0.0
                )

                fila[f"{indicador}_max"] = (
                    float(valores.max())
                    if valores.notna().any()
                    else 0.0
                )

        # --------------------------------------------------------------
        # Geometría
        # --------------------------------------------------------------

        geometria = None

        try:

            geometria_valida = (
                grupo.geometry
                .dropna()
            )

            if not geometria_valida.empty:

                if hasattr(
                    geometria_valida,
                    "union_all",
                ):
                    geometria = (
                        geometria_valida
                        .union_all()
                    )
                else:
                    geometria = (
                        geometria_valida
                        .unary_union
                    )

        except Exception as exc:

            warnings.warn(
                f"No se pudo construir geometría "
                f"para {escenario_id}: {exc}"
            )

        fila["geometry"] = geometria

        registros.append(fila)

    escenarios = gpd.GeoDataFrame(
        registros,
        geometry="geometry",
        crs=maestro.crs,
    )

    print(
        f"Escenarios construidos : {len(escenarios)}"
    )

    return escenarios


# ======================================================================================
# SCORE INTEGRAL
# ======================================================================================

def calcular_score_integral(
    escenarios: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    titulo("CALCULANDO SCORE INTEGRAL DE ESCENARIOS")

    df = escenarios.copy()

    componentes = {}

    candidatos = {
        "demanda": [
            "indice_demanda_estructural_media",
            "score_demanda_v4_media",
        ],
        "deficit": [
            "deficit_infraestructura_media",
            "score_deficit_v4_media",
        ],
        "conectividad": [
            "indice_conectividad_estructural_media",
            "score_conectividad_v4_media",
        ],
        "intermodalidad": [
            "indice_intermodalidad_estructural_media",
            "score_intermodalidad_v4_media",
        ],
        "integracion": [
            "indice_integracion_territorial_media",
            "score_integracion_v4_media",
        ],
        "centralidad": [
            "indice_centralidad_estructural_media",
            "score_centralidad_v4_media",
        ],
        "impacto": [
            "impacto_potencial_media",
            "score_impacto_v4_media",
        ],
        "urgencia": [
            "urgencia_intervencion_media",
            "score_urgencia_v4_media",
        ],
        "territorial": [
            "score_prioridad_territorial_media",
            "score_priorizacion_v4_media",
        ],
        "cartera": [
            "score_cartera_media",
            "score_cartera_v4_media",
        ],
    }

    for nombre, opciones in candidatos.items():

        columna = None

        for opcion in opciones:

            if opcion in df.columns:

                columna = opcion
                break

        if columna:

            componentes[nombre] = score_normalizado_0_100(
                df[columna]
            )

        else:

            componentes[nombre] = pd.Series(
                0.0,
                index=df.index,
            )

    # ------------------------------------------------------------------
    # Ponderación integral
    # ------------------------------------------------------------------

    pesos = {
        "demanda": 0.15,
        "deficit": 0.10,
        "conectividad": 0.10,
        "intermodalidad": 0.10,
        "integracion": 0.15,
        "centralidad": 0.10,
        "impacto": 0.10,
        "urgencia": 0.05,
        "territorial": 0.10,
        "cartera": 0.05,
    }

    score = pd.Series(
        0.0,
        index=df.index,
    )

    peso_disponible = 0.0

    for nombre, peso in pesos.items():

        serie = componentes[nombre]

        if (serie != 0).any():

            score += serie * peso
            peso_disponible += peso

    if peso_disponible > 0:

        score = score / peso_disponible

    df["score_integral_v4"] = score.round(6)

    df["ranking_integral_v4"] = (
        df["score_integral_v4"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return df


# ======================================================================================
# RANKING DE ESCENARIOS
# ======================================================================================

def construir_ranking_escenarios(
    escenarios: gpd.GeoDataFrame,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO RANKING FINAL DE ESCENARIOS")

    columnas = [
        "ranking_integral_v4",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
        "prioridad_territorial_v4",
        "prioridad_operativa_v4",
        "categoria_cartera_v4",
        "linea_estrategica_v4",
        "horizonte_intervencion_v4",
        "score_integral_v4",
    ]

    disponibles = [
        c for c in columnas
        if c in escenarios.columns
    ]

    ranking = escenarios[disponibles].copy()

    ranking = ranking.sort_values(
        [
            "ranking_integral_v4",
            "score_integral_v4",
        ],
        ascending=[
            True,
            False,
        ],
    )

    ranking = ranking.reset_index(drop=True)

    return ranking


# ======================================================================================
# RANKING DE PROYECTOS
# ======================================================================================

def construir_ranking_proyectos(
    maestro: gpd.GeoDataFrame,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO RANKING FINAL DE PROYECTOS")

    df = maestro.copy()

    candidatos = [
        "score_cartera_v4",
        "score_cartera",
        "score_priorizacion_v4",
        "score_prioridad_territorial",
        "score_prioridad_territorial",
    ]

    score_col = None

    for col in candidatos:

        if col in df.columns:

            valores = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if valores.notna().any():

                score_col = col
                break

    if score_col:

        df["score_final_proyecto_v4"] = (
            pd.to_numeric(
                df[score_col],
                errors="coerce",
            )
            .fillna(0.0)
        )

    else:

        # Fallback: promedio de indicadores disponibles.
        indicadores = [
            c for c in [
                "indice_demanda_estructural",
                "deficit_infraestructura",
                "indice_conectividad_estructural",
                "indice_intermodalidad_estructural",
                "indice_integracion_territorial",
                "indice_centralidad_estructural",
                "impacto_potencial",
                "urgencia_intervencion",
            ]
            if c in df.columns
        ]

        if indicadores:

            normalizados = pd.DataFrame(
                {
                    c: score_normalizado_0_100(
                        df[c]
                    )
                    for c in indicadores
                }
            )

            df["score_final_proyecto_v4"] = (
                normalizados.mean(axis=1)
            )

        else:

            df["score_final_proyecto_v4"] = 0.0

    df["ranking_final_proyecto_v4"] = (
        df["score_final_proyecto_v4"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    columnas = [
        "ranking_final_proyecto_v4",
        "proyecto_id",
        "escenario_id",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
        "prioridad_territorial_v4",
        "prioridad_operativa_v4",
        "categoria_cartera_v4",
        "linea_estrategica_v4",
        "horizonte_intervencion_v4",
        "score_final_proyecto_v4",
    ]

    disponibles = [
        c for c in columnas
        if c in df.columns
    ]

    ranking = (
        df[disponibles]
        .sort_values(
            "ranking_final_proyecto_v4"
        )
        .reset_index(drop=True)
    )

    return ranking


# ======================================================================================
# MATRIZ INTEGRAL
# ======================================================================================

def construir_matriz_integral(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO MATRIZ INTEGRAL DE ESCENARIOS")

    columnas = [
        c for c in [
            "ranking_integral_v4",
            "escenario_id",
            "cantidad_proyectos",
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
            "prioridad_territorial_v4",
            "categoria_cartera_v4",
            "linea_estrategica_v4",
            "horizonte_intervencion_v4",
            "score_integral_v4",
        ]
        if c in ranking.columns
    ]

    return ranking[columnas].copy()


# ======================================================================================
# INDICADORES GLOBALES
# ======================================================================================

def construir_indicadores_globales(
    maestro: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    validacion: dict,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO INDICADORES GLOBALES AMBA")

    indicadores = []

    indicadores.append(
        {
            "indicador": "proyectos_totales",
            "valor": len(maestro),
            "unidad": "proyectos",
        }
    )

    indicadores.append(
        {
            "indicador": "proyectos_unicos",
            "valor": maestro[
                "proyecto_id"
            ].nunique(),
            "unidad": "proyectos",
        }
    )

    indicadores.append(
        {
            "indicador": "escenarios_totales",
            "valor": maestro[
                "escenario_id"
            ].nunique(),
            "unidad": "escenarios",
        }
    )

    indicadores.append(
        {
            "indicador": "cobertura_geometrica",
            "valor": validacion[
                "cobertura_geometrica_pct"
            ],
            "unidad": "porcentaje",
        }
    )

    indicadores.append(
        {
            "indicador": "proyectos_multiescenario",
            "valor": validacion[
                "proyectos_multiescenario"
            ],
            "unidad": "proyectos",
        }
    )

    indicadores.append(
        {
            "indicador": "promedio_proyectos_escenario",
            "valor": validacion[
                "promedio_proyectos_escenario"
            ],
            "unidad": "proyectos",
        }
    )

    indicadores.append(
        {
            "indicador": "cv_tamano_escenarios",
            "valor": validacion[
                "cv_tamano_escenarios"
            ],
            "unidad": "coeficiente",
        }
    )

    # --------------------------------------------------------------
    # Indicadores disponibles
    # --------------------------------------------------------------

    for columna in [
        "indice_demanda_estructural",
        "deficit_infraestructura",
        "indice_conectividad_estructural",
        "indice_intermodalidad_estructural",
        "indice_integracion_territorial",
        "indice_centralidad_estructural",
        "impacto_potencial",
        "urgencia_intervencion",
        "score_prioridad_territorial",
        "score_cartera",
    ]:

        if columna not in maestro.columns:
            continue

        indicadores.append(
            {
                "indicador": f"{columna}_media_amba",
                "valor": promedio_seguro(
                    maestro[columna]
                ),
                "unidad": "score",
            }
        )

        indicadores.append(
            {
                "indicador": f"{columna}_max_amba",
                "valor": max_seguro(
                    maestro[columna]
                ),
                "unidad": "score",
            }
        )

    return pd.DataFrame(indicadores)


# ======================================================================================
# AUDITORÍA
# ======================================================================================

def construir_auditoria(
    maestro: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    validacion: dict,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO AUDITORÍA DEL PROCESO 38")

    auditoria = []

    def agregar(
        control: str,
        valor,
        esperado,
        estado: str,
        observacion: str,
    ):
        auditoria.append(
            {
                "control": control,
                "valor": valor,
                "esperado": esperado,
                "estado": estado,
                "observacion": observacion,
            }
        )

    agregar(
        "proyectos_totales",
        len(maestro),
        144,
        "OK" if len(maestro) == 144 else "OBSERVADO",
        "Cantidad total de proyectos consolidada.",
    )

    agregar(
        "proyectos_unicos",
        maestro["proyecto_id"].nunique(),
        len(maestro),
        "OK"
        if maestro["proyecto_id"].nunique() == len(maestro)
        else "OBSERVADO",
        "Cada proyecto debe aparecer una sola vez.",
    )

    agregar(
        "escenarios",
        maestro["escenario_id"].nunique(),
        7,
        "OK"
        if maestro["escenario_id"].nunique() == 7
        else "OBSERVADO",
        "Cantidad esperada de escenarios.",
    )

    agregar(
        "proyecto_id_nulos",
        validacion["proyecto_id_nulos"],
        0,
        "OK"
        if validacion["proyecto_id_nulos"] == 0
        else "OBSERVADO",
        "No debe haber proyectos sin identificador.",
    )

    agregar(
        "escenario_id_nulos",
        validacion["escenario_id_nulos"],
        0,
        "OK"
        if validacion["escenario_id_nulos"] == 0
        else "OBSERVADO",
        "No debe haber proyectos sin escenario.",
    )

    agregar(
        "proyecto_id_duplicados",
        validacion["proyecto_id_duplicados"],
        0,
        "OK"
        if validacion["proyecto_id_duplicados"] == 0
        else "OBSERVADO",
        "No debe haber proyectos duplicados.",
    )

    agregar(
        "proyectos_multiescenario",
        validacion["proyectos_multiescenario"],
        0,
        "OK"
        if validacion["proyectos_multiescenario"] == 0
        else "OBSERVADO",
        "Cada proyecto debe pertenecer a un único escenario.",
    )

    agregar(
        "geometrias_nulas",
        validacion["geometrias_nulas"],
        0,
        "OK"
        if validacion["geometrias_nulas"] == 0
        else "OBSERVADO",
        "Todas las geometrías deben estar presentes.",
    )

    agregar(
        "geometrias_invalidas",
        validacion["geometrias_invalidas"],
        0,
        "OK"
        if validacion["geometrias_invalidas"] == 0
        else "OBSERVADO",
        "Todas las geometrías deben ser válidas.",
    )

    agregar(
        "cobertura_geometrica_pct",
        round(
            validacion["cobertura_geometrica_pct"],
            4,
        ),
        100.0,
        "OK"
        if validacion["cobertura_geometrica_pct"] == 100
        else "OBSERVADO",
        "Cobertura espacial completa.",
    )

    score_integracion = 0.0

    controles = auditoria

    if controles:

        total = len(controles)
        ok = sum(
            1
            for item in controles
            if item["estado"] == "OK"
        )

        score_integracion = (
            ok / total * 100
        )

    agregar(
        "score_integracion",
        round(score_integracion, 4),
        100.0,
        "OK"
        if score_integracion == 100
        else "OBSERVADO",
        "Score integral de auditoría.",
    )

    return pd.DataFrame(auditoria)


# ======================================================================================
# EXPORTACIÓN GEOGRÁFICA
# ======================================================================================

def exportar_geopackage(
    maestro: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
) -> None:

    titulo("EXPORTANDO MODELO MAESTRO GEOGRÁFICO")

    if SALIDA_GPKG.exists():
        try:
            SALIDA_GPKG.unlink()
        except Exception:
            pass

    maestro_export = maestro.copy()

    # GeoPackage no necesita columnas problemáticas de tipo object
    # que contengan listas/dicts.
    for col in maestro_export.columns:

        if col == "geometry":
            continue

        if maestro_export[col].dtype == "object":

            maestro_export[col] = maestro_export[
                col
            ].apply(
                lambda x: (
                    json.dumps(
                        x,
                        ensure_ascii=False,
                    )
                    if isinstance(
                        x,
                        (dict, list, tuple),
                    )
                    else x
                )
            )

    maestro_export.to_file(
        SALIDA_GPKG,
        layer="proyectos",
        driver="GPKG",
    )

    escenarios_export = escenarios.copy()

    for col in escenarios_export.columns:

        if col == "geometry":
            continue

        if escenarios_export[col].dtype == "object":

            escenarios_export[col] = escenarios_export[
                col
            ].apply(
                lambda x: (
                    json.dumps(
                        x,
                        ensure_ascii=False,
                    )
                    if isinstance(
                        x,
                        (dict, list, tuple),
                    )
                    else x
                )
            )

    escenarios_export.to_file(
        SALIDA_GPKG,
        layer="escenarios",
        driver="GPKG",
    )

    print(
        f"GeoPackage : {SALIDA_GPKG}"
    )
    print(
        "Capas      : proyectos, escenarios"
    )


# ======================================================================================
# MARKDOWN MAESTRO
# ======================================================================================

def generar_markdown(
    maestro: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
    indicadores: pd.DataFrame,
    auditoria: pd.DataFrame,
    validacion: dict,
) -> str:

    titulo("GENERANDO DOCUMENTO MAESTRO DE CONSOLIDACIÓN")

    score_auditoria = (
        auditoria.loc[
            auditoria["control"] == "score_integracion",
            "valor",
        ]
    )

    if score_auditoria.empty:
        score_auditoria_valor = 0.0
    else:
        score_auditoria_valor = float(
            score_auditoria.iloc[0]
        )

    dictamen = (
        "VALIDADO"
        if score_auditoria_valor == 100
        else "OBSERVADO"
    )

    mejor = None

    if not ranking_escenarios.empty:
        mejor = ranking_escenarios.iloc[0]

    peor = None

    if not ranking_escenarios.empty:
        peor = ranking_escenarios.iloc[-1]

    partes = []

    partes.append(
        "# MODELO TERRITORIAL AMBA V4\n"
    )

    partes.append(
        "## Proceso 38 — Consolidación integral\n"
    )

    partes.append(
        "Este documento consolida los resultados validados "
        "de los procesos territoriales 31 a 37. "
        "La consolidación no modifica los indicadores "
        "originales ni la asignación proyecto → escenario.\n"
    )

    partes.append(
        "## 1. Resultado general\n"
    )

    partes.append(
        f"- Proyectos: **{len(maestro)}**\n"
        f"- Proyectos únicos: **{maestro['proyecto_id'].nunique()}**\n"
        f"- Escenarios: **{maestro['escenario_id'].nunique()}**\n"
        f"- Cobertura geométrica: **{validacion['cobertura_geometrica_pct']:.2f}%**\n"
        f"- Geometrías válidas: **{validacion['geometrias_validas']}**\n"
        f"- Geometrías inválidas: **{validacion['geometrias_invalidas']}**\n"
        f"- Proyectos multiescenario: **{validacion['proyectos_multiescenario']}**\n"
        f"- CV de tamaño de escenarios: **{validacion['cv_tamano_escenarios']:.4f}**\n"
        f"- Auditoría: **{dictamen}**\n"
    )

    partes.append(
        "## 2. Escenario prioritario\n"
    )

    if mejor is not None:

        partes.append(
            f"El escenario con mayor score integral es "
            f"**{mejor.get('escenario_id', '')}**, "
            f"con un score de "
            f"**{float(mejor.get('score_integral_v4', 0)):.2f}/100**.\n"
        )

    if peor is not None:

        partes.append(
            f"El escenario con menor score integral es "
            f"**{peor.get('escenario_id', '')}**, "
            f"con un score de "
            f"**{float(peor.get('score_integral_v4', 0)):.2f}/100**.\n"
        )

    partes.append(
        "## 3. Ranking final de escenarios\n"
    )

    ranking_md = ranking_escenarios.copy()

    if not ranking_md.empty:

        ranking_md = ranking_md[
            [
                c
                for c in [
                    "ranking_integral_v4",
                    "escenario_id",
                    "cantidad_proyectos",
                    "tipo_escenario",
                    "dimension_dominante",
                    "prioridad_escenario",
                    "categoria_cartera_v4",
                    "linea_estrategica_v4",
                    "horizonte_intervencion_v4",
                    "score_integral_v4",
                ]
                if c in ranking_md.columns
            ]
        ]

        partes.append(
            dataframe_to_markdown(
                ranking_md
            )
        )

    partes.append(
        "\n## 4. Indicadores globales\n"
    )

    partes.append(
        dataframe_to_markdown(
            indicadores
        )
    )

    partes.append(
        "\n## 5. Auditoría de consolidación\n"
    )

    partes.append(
        dataframe_to_markdown(
            auditoria
        )
    )

    partes.append(
        "\n## 6. Trazabilidad del modelo\n"
    )

    partes.append(
        "La estructura consolidada mantiene la siguiente "
        "cadena de trazabilidad:\n\n"
        "**Proyecto → Escenario → Indicadores → "
        "Priorización → Cartera → Geometría**\n"
    )

    partes.append(
        "## 7. Estado de validación\n"
    )

    if dictamen == "VALIDADO":

        partes.append(
            "El modelo territorial AMBA V4 supera los "
            "controles de consistencia estructural, "
            "territorial y geoespacial definidos para "
            "el proceso 38.\n"
        )

    else:

        partes.append(
            "El modelo presenta observaciones que deben "
            "ser revisadas antes de considerar finalizada "
            "la consolidación.\n"
        )

    partes.append(
        "## 8. Próxima etapa\n"
    )

    partes.append(
        "La salida del proceso 38 queda preparada para "
        "la generación del informe territorial integral "
        "AMBA, incluyendo metodología, diagnóstico, "
        "escenarios, priorización, cartera territorial, "
        "resultados espaciales y conclusiones estratégicas.\n"
    )

    return "\n".join(partes)


# ======================================================================================
# EXPORTACIÓN CSV
# ======================================================================================

def exportar_csv(
    maestro: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    ranking_escenarios: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
    matriz: pd.DataFrame,
    indicadores: pd.DataFrame,
    auditoria: pd.DataFrame,
) -> None:

    titulo("EXPORTANDO RESULTADOS DEL PROCESO 38")

    # --------------------------------------------------------------
    # Proyecto
    # --------------------------------------------------------------

    proyecto_csv = maestro.drop(
        columns=["geometry"],
        errors="ignore",
    ).copy()

    proyecto_csv = limpiar_dataframe(
        proyecto_csv
    )

    proyecto_csv.to_csv(
        SALIDA_PROYECTOS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Escenarios
    # --------------------------------------------------------------

    escenario_csv = escenarios.drop(
        columns=["geometry"],
        errors="ignore",
    ).copy()

    escenario_csv = limpiar_dataframe(
        escenario_csv
    )

    escenario_csv.to_csv(
        SALIDA_ESCENARIOS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Ranking escenarios
    # --------------------------------------------------------------

    ranking_escenarios.to_csv(
        SALIDA_RANKING_ESCENARIOS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Ranking proyectos
    # --------------------------------------------------------------

    ranking_proyectos.to_csv(
        SALIDA_RANKING_PROYECTOS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Matriz
    # --------------------------------------------------------------

    matriz.to_csv(
        SALIDA_MATRIZ,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Indicadores
    # --------------------------------------------------------------

    indicadores.to_csv(
        SALIDA_INDICADORES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Auditoría
    # --------------------------------------------------------------

    auditoria.to_csv(
        SALIDA_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Proyectos     : {SALIDA_PROYECTOS}"
    )
    print(
        f"Escenarios    : {SALIDA_ESCENARIOS}"
    )
    print(
        f"Ranking       : {SALIDA_RANKING_ESCENARIOS}"
    )
    print(
        f"Ranking proj. : {SALIDA_RANKING_PROYECTOS}"
    )
    print(
        f"Matriz        : {SALIDA_MATRIZ}"
    )
    print(
        f"Indicadores   : {SALIDA_INDICADORES}"
    )
    print(
        f"Auditoría     : {SALIDA_AUDITORIA}"
    )


# ======================================================================================
# RESUMEN JSON
# ======================================================================================

def generar_resumen_json(
    maestro: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    ranking_escenarios: pd.DataFrame,
    validacion: dict,
    auditoria: pd.DataFrame,
) -> dict:

    score_auditoria = (
        auditoria.loc[
            auditoria["control"] == "score_integracion",
            "valor",
        ]
    )

    score = (
        float(score_auditoria.iloc[0])
        if not score_auditoria.empty
        else 0.0
    )

    dictamen = (
        "VALIDADO"
        if score == 100
        else "OBSERVADO"
    )

    ranking = []

    for _, row in ranking_escenarios.iterrows():

        item = {}

        for col in ranking_escenarios.columns:

            item[col] = limpiar_valor(
                row[col]
            )

        ranking.append(item)

    mejor = (
        ranking[0]["escenario_id"]
        if ranking
        else None
    )

    menor = (
        ranking[-1]["escenario_id"]
        if ranking
        else None
    )

    resumen = {
        "proceso": PROCESO,
        "version": VERSION,
        "fecha_generacion": pd.Timestamp.now().isoformat(),
        "proyecto": str(BASE_DIR),
        "entrada": str(FUENTE_CANONICA),
        "salida": str(SALIDA_DIR),
        "estado": dictamen,
        "validacion": validacion,
        "score_integracion": score,
        "escenario_prioritario": mejor,
        "escenario_menor_prioridad": menor,
        "cantidad_proyectos": int(len(maestro)),
        "cantidad_proyectos_unicos": int(
            maestro["proyecto_id"].nunique()
        ),
        "cantidad_escenarios": int(
            maestro["escenario_id"].nunique()
        ),
        "ranking_escenarios": ranking,
        "salidas": {
            "proyectos": str(
                SALIDA_PROYECTOS
            ),
            "escenarios": str(
                SALIDA_ESCENARIOS
            ),
            "ranking_escenarios": str(
                SALIDA_RANKING_ESCENARIOS
            ),
            "ranking_proyectos": str(
                SALIDA_RANKING_PROYECTOS
            ),
            "matriz": str(
                SALIDA_MATRIZ
            ),
            "indicadores": str(
                SALIDA_INDICADORES
            ),
            "auditoria": str(
                SALIDA_AUDITORIA
            ),
            "geopackage": str(
                SALIDA_GPKG
            ),
            "markdown": str(
                SALIDA_MARKDOWN
            ),
        },
    }

    return resumen


# ======================================================================================
# FUNCIÓN PRINCIPAL
# ======================================================================================

def main() -> None:

    print()
    print("=" * 88)
    print(
        "38 - CONSOLIDACIÓN INTEGRAL DEL MODELO TERRITORIAL AMBA - V4"
    )
    print("=" * 88)

    print(
        f"Proyecto : {BASE_DIR}"
    )

    print(
        f"Entrada  : {FUENTE_CANONICA}"
    )

    print(
        f"Salida   : {SALIDA_DIR}"
    )

    # ------------------------------------------------------------------
    # 1. Fuente canónica
    # ------------------------------------------------------------------

    gdf = cargar_fuente_canonica()

    # ------------------------------------------------------------------
    # 2. Campos
    # ------------------------------------------------------------------

    campos = resolver_campos(gdf)

    # ------------------------------------------------------------------
    # 3. Resultados complementarios
    # ------------------------------------------------------------------

    resultados = cargar_resultados_complementarios()

    # ------------------------------------------------------------------
    # 4. Integración
    # ------------------------------------------------------------------

    maestro = integrar_resultados(
        gdf,
        campos,
        resultados,
    )

    # ------------------------------------------------------------------
    # 5. Validación
    # ------------------------------------------------------------------

    validacion = validar_modelo(
        maestro
    )

    # ------------------------------------------------------------------
    # 6. Modelo por escenario
    # ------------------------------------------------------------------

    escenarios = construir_modelo_escenarios(
        maestro,
        campos,
    )

    # ------------------------------------------------------------------
    # 7. Score integral
    # ------------------------------------------------------------------

    escenarios = calcular_score_integral(
        escenarios
    )

    # ------------------------------------------------------------------
    # 8. Ranking
    # ------------------------------------------------------------------

    ranking_escenarios = (
        construir_ranking_escenarios(
            escenarios
        )
    )

    # ------------------------------------------------------------------
    # 9. Ranking de proyectos
    # ------------------------------------------------------------------

    ranking_proyectos = (
        construir_ranking_proyectos(
            maestro
        )
    )

    # ------------------------------------------------------------------
    # 10. Matriz
    # ------------------------------------------------------------------

    matriz = construir_matriz_integral(
        ranking_escenarios
    )

    # ------------------------------------------------------------------
    # 11. Indicadores globales
    # ------------------------------------------------------------------

    indicadores = construir_indicadores_globales(
        maestro,
        escenarios,
        validacion,
    )

    # ------------------------------------------------------------------
    # 12. Auditoría
    # ------------------------------------------------------------------

    auditoria = construir_auditoria(
        maestro,
        escenarios,
        validacion,
    )

    # ------------------------------------------------------------------
    # 13. Exportación CSV
    # ------------------------------------------------------------------

    exportar_csv(
        maestro,
        escenarios,
        ranking_escenarios,
        ranking_proyectos,
        matriz,
        indicadores,
        auditoria,
    )

    # ------------------------------------------------------------------
    # 14. GeoPackage
    # ------------------------------------------------------------------

    exportar_geopackage(
        maestro,
        escenarios,
    )

    # ------------------------------------------------------------------
    # 15. Markdown
    # ------------------------------------------------------------------

    markdown = generar_markdown(
        maestro,
        escenarios,
        ranking_escenarios,
        ranking_proyectos,
        indicadores,
        auditoria,
        validacion,
    )

    SALIDA_MARKDOWN.write_text(
        markdown,
        encoding="utf-8",
    )

    print(
        f"Markdown      : {SALIDA_MARKDOWN}"
    )

    # ------------------------------------------------------------------
    # 16. JSON
    # ------------------------------------------------------------------

    resumen = generar_resumen_json(
        maestro,
        escenarios,
        ranking_escenarios,
        validacion,
        auditoria,
    )

    with SALIDA_RESUMEN.open(
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(
        f"Resumen       : {SALIDA_RESUMEN}"
    )

    # ------------------------------------------------------------------
    # 17. Resultado final
    # ------------------------------------------------------------------

    score_series = auditoria.loc[
        auditoria["control"] == "score_integracion",
        "valor",
    ]

    score_integracion = (
        float(score_series.iloc[0])
        if not score_series.empty
        else 0.0
    )

    dictamen = (
        "VALIDADO"
        if score_integracion == 100
        else "OBSERVADO"
    )

    escenario_prioritario = (
        ranking_escenarios.iloc[0][
            "escenario_id"
        ]
        if not ranking_escenarios.empty
        else "NO DISPONIBLE"
    )

    escenario_menor = (
        ranking_escenarios.iloc[-1][
            "escenario_id"
        ]
        if not ranking_escenarios.empty
        else "NO DISPONIBLE"
    )

    titulo("RESULTADO FINAL DEL PROCESO 38")

    print(
        f"Proyectos                 : {len(maestro)}"
    )

    print(
        f"Proyectos únicos          : "
        f"{maestro['proyecto_id'].nunique()}"
    )

    print(
        f"Escenarios                : "
        f"{maestro['escenario_id'].nunique()}"
    )

    print(
        f"Cobertura geométrica      : "
        f"{validacion['cobertura_geometrica_pct']:.2f}%"
    )

    print(
        f"Geometrías válidas        : "
        f"{validacion['geometrias_validas']}"
    )

    print(
        f"Geometrías nulas          : "
        f"{validacion['geometrias_nulas']}"
    )

    print(
        f"Geometrías inválidas      : "
        f"{validacion['geometrias_invalidas']}"
    )

    print(
        f"Proyectos multiescenario  : "
        f"{validacion['proyectos_multiescenario']}"
    )

    print(
        f"CV tamaño escenarios      : "
        f"{validacion['cv_tamano_escenarios']:.4f}"
    )

    print(
        f"Score integración         : "
        f"{score_integracion:.2f}/100"
    )

    print(
        f"Escenario prioritario     : "
        f"{escenario_prioritario}"
    )

    print(
        f"Escenario menor prioridad: "
        f"{escenario_menor}"
    )

    print(
        f"Auditoría                 : "
        f"{'OK' if dictamen == 'VALIDADO' else 'OBSERVADO'}"
    )

    print(
        f"Dictamen                  : {dictamen}"
    )

    print()

    if dictamen == "VALIDADO":

        print(
            "La consolidación integral del modelo territorial "
            "AMBA V4 fue validada correctamente."
        )

        print(
            "La asignación proyecto -> escenario se mantiene íntegra."
        )

        print(
            "Los indicadores originales no fueron modificados."
        )

        print(
            "Las geometrías presentan cobertura completa y "
            "consistencia espacial."
        )

        print(
            "El modelo maestro queda preparado para la "
            "elaboración del informe territorial AMBA."
        )

    else:

        print(
            "La consolidación presenta observaciones que "
            "deben ser revisadas antes de la etapa final."
        )

    print()
    print("=" * 88)
    print(
        "PROCESO 38 FINALIZADO"
    )
    print("=" * 88)


# ======================================================================================
# EJECUCIÓN
# ======================================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Proceso interrumpido por el usuario."
        )
        raise

    except Exception as exc:

        print()
        print("=" * 88)
        print(
            "ERROR FATAL EN EL PROCESO 38"
        )
        print("=" * 88)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise