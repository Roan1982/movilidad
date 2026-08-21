# -*- coding: utf-8 -*-

"""
========================================================================================
40 - GENERACIÓN DEL ATLAS TERRITORIAL AMBA - V4
========================================================================================

Objetivo
--------
Generar el atlas cartográfico final del modelo territorial AMBA V4 a partir
de los resultados validados de los procesos 38 y 39.

Principios
----------
1. No recalcular indicadores originales.
2. No modificar prioridades ni rankings.
3. Utilizar el GeoPackage maestro del proceso 38 como fuente geométrica canónica.
4. Validar 144 proyectos y 7 escenarios.
5. Generar cartografía reproducible.
6. Evitar dependencias opcionales innecesarias.
7. Generar auditoría completa del proceso.

Entradas principales
--------------------
modelo_maestro_territorial_amba_v4.gpkg
modelo_maestro_proyectos_v4.csv
modelo_maestro_escenarios_v4.csv
ranking_final_escenarios_v4.csv
ranking_final_proyectos_v4.csv
indicadores_globales_amba_v4.csv
informe_territorial_amba_v4_1.md
auditoria_39_informe_territorial_amba_v4_1.csv

Salidas principales
-------------------
atlas_territorial_amba_v4/
    mapas/
    escenarios/
    resumen/
    atlas_territorial_amba_v4.html

atlas_territorial_amba_v4.gpkg
control_cartografico_amba_v4.csv
auditoria_40_atlas_territorial_amba.csv
resumen_40_atlas_territorial_amba.json
atlas_territorial_amba_v4.md

========================================================================================
"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# ======================================================================================
# CONFIGURACIÓN
# ======================================================================================

SCRIPT_VERSION = "4.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR = INPUT_DIR

ATLAS_DIR = OUTPUT_DIR / "atlas_territorial_amba_v4"
MAP_DIR = ATLAS_DIR / "mapas"
SCENARIO_DIR = ATLAS_DIR / "escenarios"
SUMMARY_DIR = ATLAS_DIR / "resumen"

GPKG_INPUT = INPUT_DIR / "modelo_maestro_territorial_amba_v4.gpkg"

CSV_PROYECTOS = INPUT_DIR / "modelo_maestro_proyectos_v4.csv"
CSV_ESCENARIOS = INPUT_DIR / "modelo_maestro_escenarios_v4.csv"
CSV_RANK_ESCENARIOS = INPUT_DIR / "ranking_final_escenarios_v4.csv"
CSV_RANK_PROYECTOS = INPUT_DIR / "ranking_final_proyectos_v4.csv"
CSV_INDICADORES = INPUT_DIR / "indicadores_globales_amba_v4.csv"

INFORME_39 = INPUT_DIR / "informe_territorial_amba_v4_1.md"
RESUMEN_39 = INPUT_DIR / "resumen_ejecutivo_amba_v4_1.md"
AUDITORIA_39 = INPUT_DIR / "auditoria_39_informe_territorial_amba_v4_1.csv"


# ======================================================================================
# SALIDAS
# ======================================================================================

OUT_GPKG = OUTPUT_DIR / "atlas_territorial_amba_v4.gpkg"

OUT_CONTROL = OUTPUT_DIR / "control_cartografico_amba_v4.csv"

OUT_AUDIT = OUTPUT_DIR / "auditoria_40_atlas_territorial_amba.csv"

OUT_JSON = OUTPUT_DIR / "resumen_40_atlas_territorial_amba.json"

OUT_MD = OUTPUT_DIR / "atlas_territorial_amba_v4.md"

OUT_HTML = ATLAS_DIR / "atlas_territorial_amba_v4.html"


# ======================================================================================
# CAMPOS CANÓNICOS
# ======================================================================================

FIELD_CANDIDATES = {
    "proyecto": [
        "proyecto_id",
        "id_proyecto",
        "proyecto",
    ],
    "escenario": [
        "escenario_id",
        "id_escenario",
        "escenario",
    ],
    "tipo": [
        "tipo_escenario",
        "tipo_proyecto",
        "tipo",
    ],
    "dimension": [
        "dimension_dominante",
        "dimension_escenario",
        "dimension",
    ],
    "prioridad": [
        "prioridad_territorial_v4",
        "prioridad_escenario",
        "prioridad",
    ],
    "score_cartera": [
        "score_cartera_v4",
        "score_cartera",
    ],
    "score_territorial": [
        "score_priorizacion_v4",
        "score_prioridad_territorial",
        "score_territorial",
        "score_priorizacion",
    ],
    "demanda": [
        "indice_demanda_estructural",
        "score_demanda",
        "demanda",
    ],
    "deficit": [
        "deficit_infraestructura",
        "deficit",
    ],
    "conectividad": [
        "indice_conectividad_estructural",
        "score_conectividad",
        "conectividad",
    ],
    "intermodalidad": [
        "indice_intermodalidad_estructural",
        "score_intermodalidad",
        "intermodalidad",
    ],
    "integracion": [
        "indice_integracion_territorial",
        "score_integracion",
        "integracion",
    ],
    "centralidad": [
        "indice_centralidad_estructural",
        "centralidad",
    ],
    "impacto": [
        "impacto_potencial",
        "impacto",
    ],
    "urgencia": [
        "urgencia_intervencion",
        "urgencia",
    ],
    "ranking": [
        "ranking_final",
        "ranking_proyecto",
        "ranking_cartera_v4",
        "ranking",
    ],
}


# ======================================================================================
# UTILIDADES
# ======================================================================================

def titulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def log(texto: str) -> None:
    print(texto)


def asegurar_directorios() -> None:
    for path in [
        ATLAS_DIR,
        MAP_DIR,
        SCENARIO_DIR,
        SUMMARY_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolver_campo(
    df: pd.DataFrame,
    candidatos: List[str],
    requerido: bool = False,
) -> Optional[str]:
    for campo in candidatos:
        if campo in df.columns:
            return campo

    if requerido:
        raise KeyError(
            f"No se encontró ninguna de las columnas esperadas: {candidatos}"
        )

    return None


def resolver_campos(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    resultado = {}

    for nombre, candidatos in FIELD_CANDIDATES.items():
        resultado[nombre] = resolver_campo(
            df,
            candidatos,
            requerido=nombre in ["proyecto", "escenario"],
        )

    return resultado


def convertir_numerico(
    df: pd.DataFrame,
    campo: Optional[str],
) -> None:
    if campo and campo in df.columns:
        df[campo] = pd.to_numeric(
            df[campo],
            errors="coerce",
        )


def normalizar_id(serie: pd.Series) -> pd.Series:
    return (
        serie
        .astype(str)
        .str.strip()
        .replace(
            {
                "nan": np.nan,
                "None": np.nan,
                "NaN": np.nan,
            }
        )
    )


def rango_normalizado(serie: pd.Series) -> pd.Series:
    x = pd.to_numeric(serie, errors="coerce")

    if x.notna().sum() == 0:
        return pd.Series(
            np.nan,
            index=serie.index,
        )

    minimo = x.min()
    maximo = x.max()

    if math.isclose(
        float(minimo),
        float(maximo),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return pd.Series(
            100.0,
            index=serie.index,
        )

    return ((x - minimo) / (maximo - minimo)) * 100.0


def safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def geometry_union(series):
    """
    Compatible con versiones nuevas y anteriores de Shapely.
    """
    geoms = [g for g in series if g is not None and not g.is_empty]

    if not geoms:
        return None

    try:
        from shapely.ops import unary_union as _unary_union
        return _unary_union(geoms)
    except Exception:
        return None


def limpiar_geometrias(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    invalidas = ~gdf.geometry.is_valid

    if invalidas.any():
        try:
            gdf.loc[invalidas, "geometry"] = (
                gdf.loc[invalidas, "geometry"].make_valid()
            )
        except Exception:
            pass

    return gdf


def bbox_expandida(gdf: gpd.GeoDataFrame, margen: float = 0.03):
    bounds = gdf.total_bounds

    xmin, ymin, xmax, ymax = bounds

    dx = xmax - xmin
    dy = ymax - ymin

    if dx == 0:
        dx = 0.01

    if dy == 0:
        dy = 0.01

    return (
        xmin - dx * margen,
        xmax + dx * margen,
        ymin - dy * margen,
        ymax + dy * margen,
    )


# ======================================================================================
# CARGA DEL MODELO GEOGRÁFICO
# ======================================================================================

def cargar_modelo_geografico() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    titulo("CARGANDO MODELO GEOGRÁFICO MAESTRO DEL PROCESO 38")

    if not GPKG_INPUT.exists():
        raise FileNotFoundError(
            f"No existe el GeoPackage maestro:\n{GPKG_INPUT}"
        )

    capas = gpd.list_layers(GPKG_INPUT)

    log("Capas disponibles:")

    for _, fila in capas.iterrows():
        log(f"  - {fila['name']}")

    if "proyectos" not in capas["name"].tolist():
        raise ValueError(
            "El GeoPackage no contiene la capa 'proyectos'."
        )

    if "escenarios" not in capas["name"].tolist():
        raise ValueError(
            "El GeoPackage no contiene la capa 'escenarios'."
        )

    proyectos = gpd.read_file(
        GPKG_INPUT,
        layer="proyectos",
    )

    escenarios = gpd.read_file(
        GPKG_INPUT,
        layer="escenarios",
    )

    log(f"Proyectos geográficos : {len(proyectos)}")
    log(f"Escenarios geográficos: {len(escenarios)}")
    log(f"CRS proyectos         : {proyectos.crs}")
    log(f"CRS escenarios        : {escenarios.crs}")

    return proyectos, escenarios


# ======================================================================================
# CARGA DE CSV
# ======================================================================================

def cargar_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    log(
        f"Cargando: {path.name} | "
        f"Registros: {len(df)} | "
        f"Columnas: {len(df.columns)}"
    )

    return df


# ======================================================================================
# INTEGRACIÓN DE DATOS
# ======================================================================================

def integrar_datos(
    proyectos_geo: gpd.GeoDataFrame,
    escenarios_geo: gpd.GeoDataFrame,
    proyectos_csv: pd.DataFrame,
    escenarios_csv: pd.DataFrame,
    ranking_proyectos: pd.DataFrame,
    ranking_escenarios: pd.DataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    titulo("INTEGRANDO INFORMACIÓN TABULAR Y GEOGRÁFICA")

    campos = resolver_campos(proyectos_csv)

    log("Resolución de campos:")

    for nombre, campo in campos.items():
        if campo:
            log(f"{nombre:24}: {campo}")

    proyecto_geo_field = resolver_campo(
        proyectos_geo,
        FIELD_CANDIDATES["proyecto"],
        requerido=True,
    )

    escenario_geo_field = resolver_campo(
        escenarios_geo,
        FIELD_CANDIDATES["escenario"],
        requerido=True,
    )

    proyecto_csv_field = campos["proyecto"]
    escenario_csv_field = campos["escenario"]

    proyectos_csv = proyectos_csv.copy()
    escenarios_csv = escenarios_csv.copy()

    proyectos_csv[proyecto_csv_field] = normalizar_id(
        proyectos_csv[proyecto_csv_field]
    )

    escenarios_csv[escenario_csv_field] = normalizar_id(
        escenarios_csv[escenario_csv_field]
    )

    proyectos_geo = proyectos_geo.copy()
    escenarios_geo = escenarios_geo.copy()

    proyectos_geo[proyecto_geo_field] = normalizar_id(
        proyectos_geo[proyecto_geo_field]
    )

    escenarios_geo[escenario_geo_field] = normalizar_id(
        escenarios_geo[escenario_geo_field]
    )

    # ------------------------------------------------------------------
    # TABLA DE PROYECTOS
    # ------------------------------------------------------------------

    base = proyectos_csv.copy()

    columnas_rank = [
        c for c in ranking_proyectos.columns
        if c not in base.columns
        and c != "geometry"
    ]

    if columnas_rank:
        base = base.merge(
            ranking_proyectos[
                [proyecto_csv_field] + columnas_rank
            ],
            on=proyecto_csv_field,
            how="left",
        )

    base = base.merge(
        proyectos_geo[
            [
                proyecto_geo_field,
                "geometry",
            ]
        ],
        left_on=proyecto_csv_field,
        right_on=proyecto_geo_field,
        how="left",
        suffixes=("", "_geo"),
    )

    if proyecto_geo_field != proyecto_csv_field:
        base.drop(
            columns=[proyecto_geo_field],
            inplace=True,
            errors="ignore",
        )

    if "geometry_geo" in base.columns:
        base["geometry"] = base["geometry_geo"]
        base.drop(
            columns=["geometry_geo"],
            inplace=True,
        )

    gdf_proyectos = gpd.GeoDataFrame(
        base,
        geometry="geometry",
        crs=proyectos_geo.crs,
    )

    # ------------------------------------------------------------------
    # TABLA DE ESCENARIOS
    # ------------------------------------------------------------------

    escenario_base = escenarios_csv.copy()

    escenario_rank_field = resolver_campo(
        ranking_escenarios,
        FIELD_CANDIDATES["escenario"],
        requerido=True,
    )

    ranking_escenarios = ranking_escenarios.copy()

    ranking_escenarios[escenario_rank_field] = normalizar_id(
        ranking_escenarios[escenario_rank_field]
    )

    columnas_rank_esc = [
        c for c in ranking_escenarios.columns
        if c not in escenario_base.columns
    ]

    if columnas_rank_esc:
        escenario_base = escenario_base.merge(
            ranking_escenarios[
                [escenario_rank_field] + columnas_rank_esc
            ],
            left_on=escenario_csv_field,
            right_on=escenario_rank_field,
            how="left",
            suffixes=("", "_rank"),
        )

        if escenario_rank_field != escenario_csv_field:
            escenario_base.drop(
                columns=[escenario_rank_field],
                inplace=True,
                errors="ignore",
            )

    escenario_base = escenario_base.merge(
        escenarios_geo[
            [
                escenario_geo_field,
                "geometry",
            ]
        ],
        left_on=escenario_csv_field,
        right_on=escenario_geo_field,
        how="left",
        suffixes=("", "_geo"),
    )

    if escenario_geo_field != escenario_csv_field:
        escenario_base.drop(
            columns=[escenario_geo_field],
            inplace=True,
            errors="ignore",
        )

    if "geometry_geo" in escenario_base.columns:
        escenario_base["geometry"] = escenario_base["geometry_geo"]
        escenario_base.drop(
            columns=["geometry_geo"],
            inplace=True,
        )

    gdf_escenarios = gpd.GeoDataFrame(
        escenario_base,
        geometry="geometry",
        crs=escenarios_geo.crs,
    )

    return gdf_proyectos, gdf_escenarios


# ======================================================================================
# VALIDACIÓN
# ======================================================================================

def validar_modelo(
    proyectos: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
) -> Dict:

    titulo("VALIDACIÓN DEL MODELO CARTOGRÁFICO")

    proyecto_field = resolver_campo(
        proyectos,
        FIELD_CANDIDATES["proyecto"],
        requerido=True,
    )

    escenario_field = resolver_campo(
        proyectos,
        FIELD_CANDIDATES["escenario"],
        requerido=True,
    )

    escenario_geo_field = resolver_campo(
        escenarios,
        FIELD_CANDIDATES["escenario"],
        requerido=True,
    )

    proyectos[proyecto_field] = normalizar_id(
        proyectos[proyecto_field]
    )

    proyectos[escenario_field] = normalizar_id(
        proyectos[escenario_field]
    )

    escenarios[escenario_geo_field] = normalizar_id(
        escenarios[escenario_geo_field]
    )

    proyecto_nulos = int(
        proyectos[proyecto_field].isna().sum()
    )

    proyecto_duplicados = int(
        proyectos[proyecto_field].duplicated().sum()
    )

    escenario_nulos = int(
        proyectos[escenario_field].isna().sum()
    )

    geometria_valida = (
        proyectos.geometry.notna()
        & ~proyectos.geometry.is_empty
        & proyectos.geometry.is_valid
    )

    geometria_nula = int(
        proyectos.geometry.isna().sum()
    )

    geometria_vacia = int(
        proyectos.geometry.notna()
        .groupby(proyectos.geometry.is_empty)
        .sum()
        .get(True, 0)
    )

    geometria_invalida = int(
        (
            proyectos.geometry.notna()
            & ~proyectos.geometry.is_empty
            & ~proyectos.geometry.is_valid
        ).sum()
    )

    cobertura = (
        geometria_valida.mean() * 100
        if len(proyectos) > 0
        else 0
    )

    escenarios_count = int(
        proyectos[escenario_field].nunique(
            dropna=True
        )
    )

    escenario_geo_count = int(
        escenarios[escenario_geo_field].nunique(
            dropna=True
        )
    )

    conteos = (
        proyectos
        .groupby(escenario_field)
        .size()
    )

    minimo = int(conteos.min()) if len(conteos) else 0
    maximo = int(conteos.max()) if len(conteos) else 0
    promedio = float(conteos.mean()) if len(conteos) else 0.0

    cv = (
        float(conteos.std(ddof=0) / promedio)
        if promedio > 0
        else 0.0
    )

    multi_escenario = (
        proyectos
        .groupby(proyecto_field)[escenario_field]
        .nunique()
    )

    proyectos_multiescenario = int(
        (multi_escenario > 1).sum()
    )

    log(f"Proyectos                 : {len(proyectos)}")
    log(f"Proyectos únicos          : {proyectos[proyecto_field].nunique()}")
    log(f"Proyecto ID nulos         : {proyecto_nulos}")
    log(f"Proyecto ID duplicados    : {proyecto_duplicados}")
    log(f"Escenarios                : {escenarios_count}")
    log(f"Escenarios geográficos    : {escenario_geo_count}")
    log(f"Escenario ID nulos        : {escenario_nulos}")
    log(f"Geometrías válidas        : {int(geometria_valida.sum())}")
    log(f"Geometrías nulas          : {geometria_nula}")
    log(f"Geometrías vacías         : {geometria_vacia}")
    log(f"Geometrías inválidas      : {geometria_invalida}")
    log(f"Cobertura geométrica      : {cobertura:.2f}%")
    log(f"Proyectos multiescenario   : {proyectos_multiescenario}")
    log(f"Mínimo proyectos/escenario : {minimo}")
    log(f"Máximo proyectos/escenario : {maximo}")
    log(f"Promedio proyectos/escenario: {promedio:.2f}")
    log(f"CV tamaño escenarios       : {cv:.4f}")

    controles = {
        "proyectos_144": len(proyectos) == 144,
        "proyectos_unicos_144": (
            proyectos[proyecto_field].nunique() == 144
        ),
        "escenarios_7": escenarios_count == 7,
        "proyectos_nulos_0": proyecto_nulos == 0,
        "proyectos_duplicados_0": proyecto_duplicados == 0,
        "escenarios_nulos_0": escenario_nulos == 0,
        "escenarios_geograficos_7": escenario_geo_count == 7,
        "geometrias_validas_144": (
            int(geometria_valida.sum()) == 144
        ),
        "geometrias_nulas_0": geometria_nula == 0,
        "geometrias_invalidas_0": geometria_invalida == 0,
        "multiescenario_0": proyectos_multiescenario == 0,
    }

    ok = all(controles.values())

    return {
        "proyectos": len(proyectos),
        "proyectos_unicos": int(
            proyectos[proyecto_field].nunique()
        ),
        "escenarios": escenarios_count,
        "escenarios_geograficos": escenario_geo_count,
        "proyecto_nulos": proyecto_nulos,
        "proyecto_duplicados": proyecto_duplicados,
        "escenario_nulos": escenario_nulos,
        "geometrias_validas": int(
            geometria_valida.sum()
        ),
        "geometrias_nulas": geometria_nula,
        "geometrias_vacias": geometria_vacia,
        "geometrias_invalidas": geometria_invalida,
        "cobertura_geometrica": float(cobertura),
        "proyectos_multiescenario": proyectos_multiescenario,
        "min_proyectos_escenario": minimo,
        "max_proyectos_escenario": maximo,
        "promedio_proyectos_escenario": promedio,
        "cv_tamano_escenarios": cv,
        "controles": controles,
        "ok": ok,
    }


# ======================================================================================
# MAPAS
# ======================================================================================

def crear_mapa_variable(
    gdf: gpd.GeoDataFrame,
    campo: Optional[str],
    titulo_mapa: str,
    nombre_archivo: str,
    etiqueta: str,
) -> Optional[Path]:

    if campo is None or campo not in gdf.columns:
        return None

    valores = pd.to_numeric(
        gdf[campo],
        errors="coerce",
    )

    if valores.notna().sum() == 0:
        return None

    fig, ax = plt.subplots(
        figsize=(13, 10)
    )

    try:
        gdf.plot(
            ax=ax,
            column=campo,
            legend=True,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.15,
            alpha=0.82,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "hatch": "///",
            },
        )

        ax.set_title(
            titulo_mapa,
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel(
            "Longitud"
        )

        ax.set_ylabel(
            "Latitud"
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.3,
            alpha=0.4,
        )

        texto = (
            f"Indicador: {etiqueta}\n"
            f"Proyectos: {len(gdf)}\n"
            f"Cobertura: "
            f"{valores.notna().mean() * 100:.1f}%"
        )

        ax.text(
            0.01,
            0.01,
            texto,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.85,
            ),
        )

        fig.tight_layout()

        salida = MAP_DIR / nombre_archivo

        fig.savefig(
            salida,
            dpi=180,
            bbox_inches="tight",
        )

        return salida

    finally:
        plt.close(fig)


def crear_mapa_prioridad(
    gdf: gpd.GeoDataFrame,
    campo: str,
) -> Optional[Path]:

    if campo not in gdf.columns:
        return None

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    try:
        gdf.plot(
            ax=ax,
            column=campo,
            cmap="RdYlGn_r",
            legend=True,
            edgecolor="black",
            linewidth=0.2,
            alpha=0.85,
        )

        ax.set_title(
            "AMBA - Prioridad Territorial de Intervención",
            fontsize=17,
            fontweight="bold",
        )

        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.3,
            alpha=0.4,
        )

        fig.tight_layout()

        salida = MAP_DIR / "01_prioridad_territorial.png"

        fig.savefig(
            salida,
            dpi=200,
            bbox_inches="tight",
        )

        return salida

    finally:
        plt.close(fig)


def crear_mapa_escenarios(
    proyectos: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    escenario_field_proy: str,
    escenario_field_geo: str,
) -> Path:

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    try:
        escenarios.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Índices categóricos estables
        escenarios_plot = escenarios.copy()

        escenarios_plot["_cat"] = pd.factorize(
            escenarios_plot[escenario_field_geo]
        )[0]

        escenarios_plot.plot(
            ax=ax,
            column="_cat",
            cmap="tab10",
            alpha=0.16,
            edgecolor="black",
            linewidth=1.5,
        )

        proyectos.plot(
            ax=ax,
            column=escenario_field_proy,
            categorical=True,
            cmap="tab10",
            markersize=18,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.85,
        )

        ax.set_title(
            "AMBA - Escenarios Territoriales y Proyectos",
            fontsize=17,
            fontweight="bold",
        )

        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.3,
            alpha=0.4,
        )

        fig.tight_layout()

        salida = MAP_DIR / "00_escenarios_y_proyectos.png"

        fig.savefig(
            salida,
            dpi=200,
            bbox_inches="tight",
        )

        return salida

    finally:
        plt.close(fig)


def crear_mapa_proyectos(
    gdf: gpd.GeoDataFrame,
) -> Path:

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    try:
        gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.55,
        )

        centroides = gdf.geometry.centroid

        ax.scatter(
            centroides.x,
            centroides.y,
            s=10,
            alpha=0.75,
        )

        ax.set_title(
            "AMBA - Cartera Territorial de Proyectos",
            fontsize=17,
            fontweight="bold",
        )

        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.3,
            alpha=0.4,
        )

        fig.tight_layout()

        salida = MAP_DIR / "00_cartera_proyectos.png"

        fig.savefig(
            salida,
            dpi=200,
            bbox_inches="tight",
        )

        return salida

    finally:
        plt.close(fig)


# ======================================================================================
# MAPAS POR ESCENARIO
# ======================================================================================

def generar_mapas_por_escenario(
    proyectos: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    escenario_field: str,
    escenario_geo_field: str,
) -> List[Path]:

    titulo("GENERANDO MAPAS INDIVIDUALES POR ESCENARIO")

    archivos = []

    valores = sorted(
        proyectos[escenario_field]
        .dropna()
        .astype(str)
        .unique()
    )

    for escenario_id in valores:

        subset = proyectos[
            proyectos[escenario_field].astype(str)
            == escenario_id
        ].copy()

        if subset.empty:
            continue

        escenario_geom = escenarios[
            escenarios[escenario_geo_field].astype(str)
            == escenario_id
        ].copy()

        fig, ax = plt.subplots(
            figsize=(13, 10)
        )

        try:
            if not escenario_geom.empty:
                escenario_geom.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=2.0,
                )

            subset.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=0.25,
                alpha=0.55,
            )

            cent = subset.geometry.centroid

            ax.scatter(
                cent.x,
                cent.y,
                s=30,
                alpha=0.85,
            )

            ax.set_title(
                f"AMBA - {escenario_id}",
                fontsize=17,
                fontweight="bold",
            )

            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.3,
                alpha=0.4,
            )

            ax.text(
                0.01,
                0.01,
                f"Proyectos: {len(subset)}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85,
                ),
            )

            fig.tight_layout()

            salida = (
                SCENARIO_DIR
                / f"{escenario_id}_territorial.png"
            )

            fig.savefig(
                salida,
                dpi=180,
                bbox_inches="tight",
            )

            archivos.append(salida)

        finally:
            plt.close(fig)

    return archivos


# ======================================================================================
# GENERACIÓN DEL CONTROL CARTOGRÁFICO
# ======================================================================================

def construir_control_cartografico(
    proyectos: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    campos: Dict[str, Optional[str]],
) -> pd.DataFrame:

    registros = []

    proyecto_field = campos["proyecto"]
    escenario_field = campos["escenario"]

    for _, row in proyectos.iterrows():

        geom = row.geometry

        if geom is None:
            tipo = None
            valido = False
            vacio = False
        else:
            tipo = geom.geom_type
            valido = bool(geom.is_valid)
            vacio = bool(geom.is_empty)

        registros.append(
            {
                "proyecto_id": row.get(proyecto_field),
                "escenario_id": row.get(escenario_field),
                "geometry_type": tipo,
                "geometry_valid": valido,
                "geometry_empty": vacio,
                "geometry_null": geom is None,
                "crs": str(proyectos.crs),
            }
        )

    return pd.DataFrame(registros)


# ======================================================================================
# EXPORTACIÓN GEOGRÁFICA
# ======================================================================================

def exportar_geopackage(
    proyectos: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
) -> None:

    titulo("EXPORTANDO ATLAS GEOGRÁFICO")

    if OUT_GPKG.exists():
        try:
            OUT_GPKG.unlink()
        except Exception:
            pass

    proyectos.to_file(
        OUT_GPKG,
        layer="proyectos",
        driver="GPKG",
    )

    escenarios.to_file(
        OUT_GPKG,
        layer="escenarios",
        driver="GPKG",
    )

    log(f"GeoPackage: {OUT_GPKG}")
    log("Capas     : proyectos, escenarios")


# ======================================================================================
# MARKDOWN
# ======================================================================================

def generar_markdown(
    validacion: Dict,
    proyectos: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
    archivos_mapas: List[Path],
) -> str:

    proyecto_field = resolver_campo(
        proyectos,
        FIELD_CANDIDATES["proyecto"],
        requerido=True,
    )

    escenario_field = resolver_campo(
        proyectos,
        FIELD_CANDIDATES["escenario"],
        requerido=True,
    )

    prioridad_field = resolver_campo(
        proyectos,
        FIELD_CANDIDATES["prioridad"]
    )

    score_field = resolver_campo(
        proyectos,
        FIELD_CANDIDATES["score_territorial"]
    )

    lineas = []

    lineas.append(
        "# Atlas Territorial AMBA V4\n"
    )

    lineas.append(
        "## Proceso 40 - Generación del Atlas Territorial\n"
    )

    lineas.append(
        "Este documento presenta la salida cartográfica final "
        "del modelo territorial AMBA V4.\n"
    )

    lineas.append(
        "La cartografía utiliza como fuente geométrica canónica "
        "el GeoPackage maestro generado en el proceso 38.\n"
    )

    lineas.append("## Validación\n")

    lineas.append(
        f"- Proyectos: **{validacion['proyectos']}**\n"
    )

    lineas.append(
        f"- Proyectos únicos: **{validacion['proyectos_unicos']}**\n"
    )

    lineas.append(
        f"- Escenarios: **{validacion['escenarios']}**\n"
    )

    lineas.append(
        f"- Cobertura geométrica: "
        f"**{validacion['cobertura_geometrica']:.2f}%**\n"
    )

    lineas.append(
        f"- Geometrías válidas: "
        f"**{validacion['geometrias_validas']}**\n"
    )

    lineas.append(
        f"- Geometrías inválidas: "
        f"**{validacion['geometrias_invalidas']}**\n"
    )

    lineas.append(
        f"- Proyectos multiescenario: "
        f"**{validacion['proyectos_multiescenario']}**\n"
    )

    lineas.append("\n## Mapas generales\n")

    for archivo in archivos_mapas:
        rel = archivo.relative_to(ATLAS_DIR).as_posix()
        lineas.append(
            f"- [{archivo.name}]({rel})\n"
        )

    lineas.append("\n## Distribución por escenario\n")

    conteos = (
        proyectos
        .groupby(escenario_field)
        .size()
        .sort_values(
            ascending=False
        )
    )

    lineas.append(
        "| Escenario | Proyectos |\n"
    )

    lineas.append(
        "|---|---:|\n"
    )

    for escenario, cantidad in conteos.items():
        lineas.append(
            f"| {escenario} | {cantidad} |\n"
        )

    if prioridad_field:
        lineas.append(
            "\n## Prioridades territoriales\n"
        )

        tabla = (
            proyectos[
                [
                    proyecto_field,
                    escenario_field,
                    prioridad_field,
                ]
            ]
            .copy()
            .head(20)
        )

        lineas.append(
            "| Proyecto | Escenario | Prioridad |\n"
        )

        lineas.append(
            "|---|---|---|\n"
        )

        for _, row in tabla.iterrows():
            lineas.append(
                f"| {row[proyecto_field]} "
                f"| {row[escenario_field]} "
                f"| {row[prioridad_field]} |\n"
            )

    if score_field:
        valores = pd.to_numeric(
            proyectos[score_field],
            errors="coerce",
        )

        if valores.notna().any():
            lineas.append(
                "\n## Score territorial\n"
            )

            lineas.append(
                f"- Mínimo: **{valores.min():.2f}**\n"
            )

            lineas.append(
                f"- Máximo: **{valores.max():.2f}**\n"
            )

            lineas.append(
                f"- Promedio: **{valores.mean():.2f}**\n"
            )

    lineas.append(
        "\n## Estado del proceso\n"
    )

    lineas.append(
        "**DICTAMEN: VALIDADO**\n"
    )

    return "".join(lineas)


# ======================================================================================
# HTML
# ======================================================================================

def generar_html(
    validacion: Dict,
    archivos_mapas: List[Path],
    archivos_escenarios: List[Path],
) -> str:

    html = []

    html.append(
        """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Atlas Territorial AMBA V4</title>
<style>
body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 40px;
    background: #f5f5f5;
    color: #222;
}
h1 {
    margin-bottom: 5px;
}
h2 {
    margin-top: 40px;
}
.card {
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 5px rgba(0,0,0,.12);
}
.metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}
.metric {
    background: #fafafa;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 6px;
}
.metric .value {
    font-size: 24px;
    font-weight: bold;
}
.map {
    width: 100%;
    max-width: 1200px;
    border: 1px solid #ddd;
    margin: 10px 0 30px 0;
}
.scenario-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}
.scenario-grid img {
    width: 100%;
    border: 1px solid #ddd;
}
.ok {
    color: #146b2e;
    font-weight: bold;
}
</style>
</head>
<body>
"""
    )

    html.append(
        "<h1>Atlas Territorial AMBA V4</h1>"
    )

    html.append(
        "<p>Proceso 40 - Generación del Atlas Territorial AMBA.</p>"
    )

    html.append(
        '<div class="card">'
    )

    html.append(
        '<div class="metrics">'
    )

    metricas = [
        ("Proyectos", validacion["proyectos"]),
        ("Proyectos únicos", validacion["proyectos_unicos"]),
        ("Escenarios", validacion["escenarios"]),
        (
            "Cobertura",
            f"{validacion['cobertura_geometrica']:.2f}%",
        ),
        (
            "Geometrías válidas",
            validacion["geometrias_validas"],
        ),
        (
            "Multiescenario",
            validacion["proyectos_multiescenario"],
        ),
    ]

    for nombre, valor in metricas:
        html.append(
            f"""
<div class="metric">
<div>{nombre}</div>
<div class="value">{valor}</div>
</div>
"""
        )

    html.append("</div>")

    html.append(
        '<p class="ok">DICTAMEN: VALIDADO</p>'
    )

    html.append("</div>")

    html.append(
        '<div class="card"><h2>Cartografía general</h2>'
    )

    for archivo in archivos_mapas:
        rel = archivo.relative_to(ATLAS_DIR).as_posix()

        html.append(
            f"""
<h3>{archivo.stem.replace("_", " ")}</h3>
<img class="map" src="{rel}" alt="{archivo.name}">
"""
        )

    html.append("</div>")

    html.append(
        '<div class="card"><h2>Escenarios territoriales</h2>'
    )

    html.append(
        '<div class="scenario-grid">'
    )

    for archivo in archivos_escenarios:
        rel = archivo.relative_to(ATLAS_DIR).as_posix()

        html.append(
            f"""
<div>
<h3>{archivo.stem.replace("_", " ")}</h3>
<img src="{rel}" alt="{archivo.name}">
</div>
"""
        )

    html.append(
        "</div></div>"
    )

    html.append(
        """
<div class="card">
<h2>Fuente</h2>
<p>
Modelo maestro territorial AMBA V4,
procesos 38 y 39.
</p>
</div>
</body>
</html>
"""
    )

    return "".join(html)


# ======================================================================================
# AUDITORÍA
# ======================================================================================

def construir_auditoria(
    validacion: Dict,
    cantidad_mapas: int,
    cantidad_escenarios_mapas: int,
) -> pd.DataFrame:

    controles = validacion["controles"].copy()

    controles["mapas_generales_generados"] = (
        cantidad_mapas >= 3
    )

    controles["mapas_escenarios_generados"] = (
        cantidad_escenarios_mapas == 7
    )

    controles["atlas_html_generado"] = OUT_HTML.exists()

    controles["geopackage_generado"] = OUT_GPKG.exists()

    filas = []

    for nombre, resultado in controles.items():

        filas.append(
            {
                "proceso": 40,
                "control": nombre,
                "resultado": bool(resultado),
                "estado": (
                    "OK"
                    if resultado
                    else "OBSERVADO"
                ),
            }
        )

    return pd.DataFrame(filas)


# ======================================================================================
# JSON
# ======================================================================================

def construir_json(
    validacion: Dict,
    cantidad_mapas: int,
    cantidad_escenarios_mapas: int,
    controles: pd.DataFrame,
) -> Dict:

    return {
        "proceso": 40,
        "version": SCRIPT_VERSION,
        "nombre": (
            "Generación del Atlas Territorial AMBA"
        ),
        "fecha_generacion": (
            pd.Timestamp.now()
            .isoformat()
        ),
        "proyecto_root": str(PROJECT_ROOT),
        "input": {
            "geopackage": str(GPKG_INPUT),
            "proyectos": str(CSV_PROYECTOS),
            "escenarios": str(CSV_ESCENARIOS),
            "ranking_proyectos": str(CSV_RANK_PROYECTOS),
            "ranking_escenarios": str(CSV_RANK_ESCENARIOS),
        },
        "validacion": validacion,
        "cartografia": {
            "mapas_generales": cantidad_mapas,
            "mapas_por_escenario": cantidad_escenarios_mapas,
        },
        "controles_totales": int(
            len(controles)
        ),
        "controles_ok": int(
            (controles["resultado"] == True).sum()
        ),
        "dictamen": (
            "VALIDADO"
            if controles["resultado"].all()
            else "OBSERVADO"
        ),
    }


# ======================================================================================
# MAIN
# ======================================================================================

def main():

    inicio = time.time()

    titulo(
        "40 - GENERACIÓN DEL ATLAS TERRITORIAL AMBA - V4"
    )

    log(f"Proyecto : {PROJECT_ROOT}")
    log(f"Entrada  : {INPUT_DIR}")
    log(f"Salida   : {OUTPUT_DIR}")

    asegurar_directorios()

    # ------------------------------------------------------------------
    # CARGA
    # ------------------------------------------------------------------

    titulo(
        "CARGANDO MODELO GEOGRÁFICO MAESTRO DEL PROCESO 38"
    )

    proyectos_geo, escenarios_geo = (
        cargar_modelo_geografico()
    )

    titulo(
        "CARGANDO INFORMACIÓN TABULAR DEL MODELO MAESTRO"
    )

    proyectos_csv = cargar_csv(
        CSV_PROYECTOS
    )

    escenarios_csv = cargar_csv(
        CSV_ESCENARIOS
    )

    ranking_proyectos = cargar_csv(
        CSV_RANK_PROYECTOS
    )

    ranking_escenarios = cargar_csv(
        CSV_RANK_ESCENARIOS
    )

    # Estos archivos se validan por existencia,
    # pero no son necesarios para recalcular nada.
    if CSV_INDICADORES.exists():
        log(
            f"Indicadores globales disponibles: "
            f"{CSV_INDICADORES.name}"
        )

    if INFORME_39.exists():
        log(
            f"Informe 39 disponible: "
            f"{INFORME_39.name}"
        )

    if AUDITORIA_39.exists():
        log(
            f"Auditoría 39 disponible: "
            f"{AUDITORIA_39.name}"
        )

    # ------------------------------------------------------------------
    # INTEGRACIÓN
    # ------------------------------------------------------------------

    proyectos, escenarios = integrar_datos(
        proyectos_geo=proyectos_geo,
        escenarios_geo=escenarios_geo,
        proyectos_csv=proyectos_csv,
        escenarios_csv=escenarios_csv,
        ranking_proyectos=ranking_proyectos,
        ranking_escenarios=ranking_escenarios,
    )

    # ------------------------------------------------------------------
    # LIMPIEZA
    # ------------------------------------------------------------------

    titulo(
        "VALIDANDO Y NORMALIZANDO GEOMETRÍAS"
    )

    proyectos = limpiar_geometrias(
        proyectos
    )

    escenarios = limpiar_geometrias(
        escenarios
    )

    # ------------------------------------------------------------------
    # VALIDACIÓN
    # ------------------------------------------------------------------

    validacion = validar_modelo(
        proyectos,
        escenarios,
    )

    if not validacion["ok"]:
        raise RuntimeError(
            "La validación del modelo cartográfico "
            "no fue satisfactoria. "
            "No se generará el atlas final."
        )

    # ------------------------------------------------------------------
    # CAMPOS
    # ------------------------------------------------------------------

    campos = resolver_campos(
        proyectos
    )

    # ------------------------------------------------------------------
    # MAPAS GENERALES
    # ------------------------------------------------------------------

    titulo(
        "GENERANDO CARTOGRAFÍA GENERAL"
    )

    archivos_mapas = []

    archivo = crear_mapa_proyectos(
        proyectos
    )

    archivos_mapas.append(
        archivo
    )

    escenario_field = campos["escenario"]

    escenario_geo_field = resolver_campo(
        escenarios,
        FIELD_CANDIDATES["escenario"],
        requerido=True,
    )

    archivo = crear_mapa_escenarios(
        proyectos,
        escenarios,
        escenario_field,
        escenario_geo_field,
    )

    archivos_mapas.append(
        archivo
    )

    if campos["prioridad"]:
        archivo = crear_mapa_prioridad(
            proyectos,
            campos["prioridad"],
        )

        if archivo:
            archivos_mapas.append(
                archivo
            )

    # ------------------------------------------------------------------
    # MAPAS DE INDICADORES
    # ------------------------------------------------------------------

    titulo(
        "GENERANDO MAPAS DE INDICADORES TERRITORIALES"
    )

    mapas_indicadores = [
        (
            "demanda",
            "02_demanda.png",
            "Demanda estructural",
        ),
        (
            "deficit",
            "03_deficit_infraestructura.png",
            "Déficit de infraestructura",
        ),
        (
            "conectividad",
            "04_conectividad.png",
            "Conectividad estructural",
        ),
        (
            "intermodalidad",
            "05_intermodalidad.png",
            "Intermodalidad estructural",
        ),
        (
            "integracion",
            "06_integracion_territorial.png",
            "Integración territorial",
        ),
        (
            "centralidad",
            "07_centralidad_estructural.png",
            "Centralidad estructural",
        ),
        (
            "impacto",
            "08_impacto_potencial.png",
            "Impacto potencial",
        ),
        (
            "urgencia",
            "09_urgencia_intervencion.png",
            "Urgencia de intervención",
        ),
        (
            "score_territorial",
            "10_score_territorial.png",
            "Score de priorización territorial",
        ),
        (
            "score_cartera",
            "11_score_cartera.png",
            "Score de cartera",
        ),
    ]

    for nombre_campo, archivo, etiqueta in mapas_indicadores:

        campo = campos.get(
            nombre_campo
        )

        if campo is None:
            log(
                f"Indicador no disponible: "
                f"{nombre_campo}"
            )
            continue

        resultado = crear_mapa_variable(
            proyectos,
            campo,
            f"AMBA - {etiqueta}",
            archivo,
            etiqueta,
        )

        if resultado:
            archivos_mapas.append(
                resultado
            )

    # ------------------------------------------------------------------
    # MAPAS POR ESCENARIO
    # ------------------------------------------------------------------

    archivos_escenarios = (
        generar_mapas_por_escenario(
            proyectos,
            escenarios,
            escenario_field,
            escenario_geo_field,
        )
    )

    # ------------------------------------------------------------------
    # CONTROL CARTOGRÁFICO
    # ------------------------------------------------------------------

    titulo(
        "CONSTRUYENDO CONTROL CARTOGRÁFICO"
    )

    control = construir_control_cartografico(
        proyectos,
        escenarios,
        campos,
    )

    control.to_csv(
        OUT_CONTROL,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # GEOPACKAGE
    # ------------------------------------------------------------------

    exportar_geopackage(
        proyectos,
        escenarios,
    )

    # ------------------------------------------------------------------
    # MARKDOWN
    # ------------------------------------------------------------------

    titulo(
        "GENERANDO DOCUMENTACIÓN DEL ATLAS"
    )

    markdown = generar_markdown(
        validacion,
        proyectos,
        escenarios,
        archivos_mapas,
    )

    OUT_MD.write_text(
        markdown,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    html = generar_html(
        validacion,
        archivos_mapas,
        archivos_escenarios,
    )

    OUT_HTML.write_text(
        html,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # AUDITORÍA
    # ------------------------------------------------------------------

    titulo(
        "CONSTRUYENDO AUDITORÍA DEL PROCESO 40"
    )

    auditoria = construir_auditoria(
        validacion,
        len(archivos_mapas),
        len(archivos_escenarios),
    )

    auditoria.to_csv(
        OUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    resumen = construir_json(
        validacion,
        len(archivos_mapas),
        len(archivos_escenarios),
        auditoria,
    )

    OUT_JSON.write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------------------

    controles_ok = int(
        auditoria["resultado"].sum()
    )

    controles_total = len(
        auditoria
    )

    dictamen = (
        "VALIDADO"
        if auditoria["resultado"].all()
        else "OBSERVADO"
    )

    tiempo = time.time() - inicio

    titulo(
        "RESULTADO FINAL DEL PROCESO 40"
    )

    log(
        f"Proyectos                 : "
        f"{validacion['proyectos']}"
    )

    log(
        f"Proyectos únicos          : "
        f"{validacion['proyectos_unicos']}"
    )

    log(
        f"Escenarios                : "
        f"{validacion['escenarios']}"
    )

    log(
        f"Cobertura geométrica      : "
        f"{validacion['cobertura_geometrica']:.2f}%"
    )

    log(
        f"Geometrías válidas        : "
        f"{validacion['geometrias_validas']}"
    )

    log(
        f"Geometrías nulas          : "
        f"{validacion['geometrias_nulas']}"
    )

    log(
        f"Geometrías inválidas      : "
        f"{validacion['geometrias_invalidas']}"
    )

    log(
        f"Proyectos multiescenario   : "
        f"{validacion['proyectos_multiescenario']}"
    )

    log(
        f"Mapas generales            : "
        f"{len(archivos_mapas)}"
    )

    log(
        f"Mapas por escenario        : "
        f"{len(archivos_escenarios)}"
    )

    log(
        f"Controles OK               : "
        f"{controles_ok}/{controles_total}"
    )

    log(
        f"Auditoría                  : "
        f"{'OK' if dictamen == 'VALIDADO' else 'OBSERVADA'}"
    )

    log(
        f"Dictamen                   : "
        f"{dictamen}"
    )

    log(
        f"Tiempo de ejecución        : "
        f"{tiempo:.2f} segundos"
    )

    titulo(
        "ARCHIVOS GENERADOS"
    )

    log(
        f"Atlas HTML : {OUT_HTML}"
    )

    log(
        f"Atlas MD   : {OUT_MD}"
    )

    log(
        f"GeoPackage : {OUT_GPKG}"
    )

    log(
        f"Control     : {OUT_CONTROL}"
    )

    log(
        f"Auditoría   : {OUT_AUDIT}"
    )

    log(
        f"Resumen     : {OUT_JSON}"
    )

    log(
        f"Mapas       : {MAP_DIR}"
    )

    log(
        f"Escenarios  : {SCENARIO_DIR}"
    )

    if dictamen == "VALIDADO":

        titulo(
            "PROCESO 40 FINALIZADO - VALIDADO"
        )

        log(
            "El Atlas Territorial AMBA V4 fue generado "
            "y validado correctamente."
        )

        log(
            "La cartografía utiliza las geometrías "
            "canónicas del modelo maestro del proceso 38."
        )

        log(
            "La cobertura geométrica es completa."
        )

        log(
            "Los 144 proyectos mantienen su asignación "
            "territorial a los 7 escenarios."
        )

        log(
            "Los indicadores originales no fueron modificados."
        )

    else:

        titulo(
            "PROCESO 40 FINALIZADO - OBSERVADO"
        )

        log(
            "El atlas fue generado con observaciones."
        )

        log(
            "Revisar el archivo de auditoría:"
        )

        log(
            str(OUT_AUDIT)
        )


# ======================================================================================
# EJECUCIÓN
# ======================================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        titulo(
            "ERROR FATAL EN EL PROCESO 40"
        )

        log(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        raise