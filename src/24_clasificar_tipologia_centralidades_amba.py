# -*- coding: utf-8 -*-

"""
==============================================================================
24 - CLASIFICACIÓN DE TIPOLOGÍA DE CENTRALIDADES AMBA
==============================================================================

Construye una tipología multidimensional de las 144 centralidades analizadas
en los procesos anteriores.

Entrada:
    data/processed/indice_centralidad_estructural_amba/
        indice_centralidad_estructural_amba.parquet

Salida:
    data/processed/tipologia_centralidades_amba/

Productos:
    - tipologia_centralidades_amba.parquet
    - tipologia_centralidades_amba.csv
    - tipologia_centralidades_amba.gpkg
    - tipologia_centralidades_amba_resumen.json
    - mapas
    - gráficos

Dimensiones utilizadas:
    - demanda
    - infraestructura
    - intermodalidad
    - conectividad
    - integración territorial
    - déficit infraestructura
    - prioridad de intervención

La clasificación distingue:
    1. CENTRALIDAD_METROPOLITANA
    2. CENTRALIDAD_ESTRATEGICA
    3. CENTRALIDAD_INTERMODAL
    4. CENTRALIDAD_DEMANDA
    5. CENTRALIDAD_DEFICITARIA
    6. CENTRALIDAD_CONECTIVA
    7. CENTRALIDAD_TERRITORIAL
    8. CENTRALIDAD_EMERGENTE
    9. CENTRALIDAD_LOCAL

Además construye un perfil multidimensional independiente de la tipología
principal.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SCRIPT_NAME = "24 - CLASIFICACIÓN DE TIPOLOGÍA DE CENTRALIDADES AMBA - V1"

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

RADIO_MAPA = 2500


# =============================================================================
# RUTAS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "indice_centralidad_estructural_amba"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "tipologia_centralidades_amba"
)

INPUT_FILE = INPUT_DIR / "indice_centralidad_estructural_amba.parquet"

OUTPUT_PARQUET = OUTPUT_DIR / "tipologia_centralidades_amba.parquet"
OUTPUT_CSV = OUTPUT_DIR / "tipologia_centralidades_amba.csv"
OUTPUT_GPKG = OUTPUT_DIR / "tipologia_centralidades_amba.gpkg"
OUTPUT_JSON = OUTPUT_DIR / "tipologia_centralidades_amba_resumen.json"


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto: str) -> None:
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def subtitulo(texto: str) -> None:
    print()
    print("-" * 78)
    print(texto)
    print("-" * 78)


def asegurar_directorio() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validar_columna(df: pd.DataFrame, columna: str) -> None:
    if columna not in df.columns:
        raise ValueError(
            f"No se encontró la columna obligatoria: {columna}"
        )


def minmax_0_100(serie: pd.Series) -> pd.Series:
    """
    Normalización robusta 0-100.
    """
    s = pd.to_numeric(serie, errors="coerce")

    minimo = s.min()
    maximo = s.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(50.0, index=s.index)

    if np.isclose(maximo, minimo):
        return pd.Series(50.0, index=s.index)

    resultado = (s - minimo) / (maximo - minimo) * 100.0

    return resultado.clip(0, 100)


def promedio_seguro(*series: pd.Series) -> pd.Series:
    frame = pd.concat(series, axis=1)
    return frame.mean(axis=1).fillna(0)


# =============================================================================
# CLASIFICACIONES
# =============================================================================

def nivel(valor: float) -> str:
    if pd.isna(valor):
        return "SIN_DATOS"

    if valor >= 80:
        return "MUY_ALTO"

    if valor >= 60:
        return "ALTO"

    if valor >= 40:
        return "MEDIO"

    if valor >= 20:
        return "BAJO"

    return "MUY_BAJO"


def clasificar_tipologia(row: pd.Series) -> str:

    demanda = row["indice_demanda_estructural"]
    infraestructura = row["indice_infraestructura_estructural"]
    intermodalidad = row["indice_intermodalidad_estructural"]
    conectividad = row["indice_conectividad_estructural"]
    integracion = row["indice_integracion_territorial"]
    deficit = row["deficit_infraestructura"]
    prioridad = row["prioridad_intervencion"]
    estructural = row["indice_centralidad_estructural_robusto"]

    # -------------------------------------------------------------------------
    # 1. CENTRALIDAD METROPOLITANA
    # -------------------------------------------------------------------------

    if (
        demanda >= 80
        and infraestructura >= 75
        and intermodalidad >= 70
        and conectividad >= 60
    ):
        return "CENTRALIDAD_METROPOLITANA"

    # -------------------------------------------------------------------------
    # 2. CENTRALIDAD ESTRATÉGICA
    # -------------------------------------------------------------------------

    if (
        demanda >= 80
        and infraestructura >= 50
        and conectividad >= 60
        and estructural >= 75
    ):
        return "CENTRALIDAD_ESTRATEGICA"

    # -------------------------------------------------------------------------
    # 3. CENTRALIDAD INTERMODAL
    # -------------------------------------------------------------------------

    if (
        intermodalidad >= 75
        and infraestructura >= 45
    ):
        return "CENTRALIDAD_INTERMODAL"

    # -------------------------------------------------------------------------
    # 4. CENTRALIDAD DE DEMANDA
    # -------------------------------------------------------------------------

    if (
        demanda >= 80
        and infraestructura < 50
        and deficit >= 35
    ):
        return "CENTRALIDAD_DEMANDA"

    # -------------------------------------------------------------------------
    # 5. CENTRALIDAD DEFICITARIA
    # -------------------------------------------------------------------------

    if (
        deficit >= 45
        and demanda >= 65
        and prioridad >= 60
    ):
        return "CENTRALIDAD_DEFICITARIA"

    # -------------------------------------------------------------------------
    # 6. CENTRALIDAD CONECTIVA
    # -------------------------------------------------------------------------

    if (
        conectividad >= 75
        and demanda >= 50
    ):
        return "CENTRALIDAD_CONECTIVA"

    # -------------------------------------------------------------------------
    # 7. CENTRALIDAD TERRITORIAL
    # -------------------------------------------------------------------------

    if (
        integracion >= 75
        and demanda >= 45
    ):
        return "CENTRALIDAD_TERRITORIAL"

    # -------------------------------------------------------------------------
    # 8. CENTRALIDAD EMERGENTE
    # -------------------------------------------------------------------------

    if (
        infraestructura >= 50
        and intermodalidad >= 50
        and demanda < 65
    ):
        return "CENTRALIDAD_EMERGENTE"

    # -------------------------------------------------------------------------
    # 9. CENTRALIDAD LOCAL
    # -------------------------------------------------------------------------

    return "CENTRALIDAD_LOCAL"


def clasificar_prioridad(row: pd.Series) -> str:

    prioridad = row["prioridad_intervencion"]
    deficit = row["deficit_infraestructura"]

    if prioridad >= 75 and deficit >= 50:
        return "PRIORIDAD_CRITICA"

    if prioridad >= 60 and deficit >= 35:
        return "PRIORIDAD_ALTA"

    if prioridad >= 40:
        return "PRIORIDAD_MEDIA"

    return "PRIORIDAD_BAJA"


def construir_perfil(row: pd.Series) -> str:

    dimensiones = {
        "DEMANDA": row["indice_demanda_estructural"],
        "INFRAESTRUCTURA": row["indice_infraestructura_estructural"],
        "INTERMODALIDAD": row["indice_intermodalidad_estructural"],
        "CONECTIVIDAD": row["indice_conectividad_estructural"],
        "TERRITORIAL": row["indice_integracion_territorial"],
    }

    ordenadas = sorted(
        dimensiones.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    principales = [
        nombre
        for nombre, valor in ordenadas
        if valor >= 70
    ]

    if not principales:
        principales = [ordenadas[0][0]]

    return "|".join(principales[:3])


def construir_dimension_dominante(row: pd.Series) -> str:

    dimensiones = {
        "DEMANDA": row["indice_demanda_estructural"],
        "INFRAESTRUCTURA": row["indice_infraestructura_estructural"],
        "INTERMODALIDAD": row["indice_intermodalidad_estructural"],
        "CONECTIVIDAD": row["indice_conectividad_estructural"],
        "INTEGRACION_TERRITORIAL": row[
            "indice_integracion_territorial"
        ],
    }

    return max(dimensiones, key=dimensiones.get)


def construir_diagnostico(row: pd.Series) -> str:

    demanda = row["indice_demanda_estructural"]
    infraestructura = row["indice_infraestructura_estructural"]
    intermodalidad = row["indice_intermodalidad_estructural"]
    conectividad = row["indice_conectividad_estructural"]
    integracion = row["indice_integracion_territorial"]

    if demanda >= 80 and infraestructura < 50:
        return "ALTA_DEMANDA_BAJO_SOPORTE"

    if demanda >= 80 and infraestructura >= 70:
        return "ALTA_DEMANDA_ALTO_SOPORTE"

    if intermodalidad >= 75:
        return "FUERTE_INTERMODALIDAD"

    if conectividad >= 75:
        return "FUERTE_CONECTIVIDAD"

    if integracion >= 75:
        return "FUERTE_INTEGRACION_TERRITORIAL"

    if (
        infraestructura >= 60
        and demanda < 60
    ):
        return "SOPORTE_SUPERIOR_A_DEMANDA"

    return "PERFIL_EQUILIBRADO"


# =============================================================================
# VALIDACIÓN
# =============================================================================

def validar_geometria(gdf: gpd.GeoDataFrame) -> None:

    subtitulo("VALIDACIÓN GEOMÉTRICA")

    print(f"Registros: {len(gdf):,}")
    print(f"Geometrías nulas: {gdf.geometry.isna().sum():,}")
    print(f"Geometrías vacías: {gdf.geometry.is_empty.sum():,}")
    print(f"Geometrías inválidas: {(~gdf.geometry.is_valid).sum():,}")

    if gdf.geometry.isna().any():
        raise ValueError("Existen geometrías nulas.")

    if gdf.geometry.is_empty.any():
        raise ValueError("Existen geometrías vacías.")

    if (~gdf.geometry.is_valid).any():
        raise ValueError("Existen geometrías inválidas.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    asegurar_directorio()

    titulo(
        "24 - CLASIFICACIÓN DE TIPOLOGÍA DE CENTRALIDADES AMBA - V1"
    )

    print(f"Proyecto : {PROJECT_DIR}")
    print(f"Entrada  : {INPUT_FILE}")
    print(f"Salida   : {OUTPUT_DIR}")
    print(f"CRS      : {CRS_GEOGRAFICO}")
    print(f"CRS métrico: {CRS_METRICO}")

    # =========================================================================
    # 1. CARGA
    # =========================================================================

    titulo("1. CARGANDO RESULTADOS DEL PROCESO 23")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada:\n{INPUT_FILE}"
        )

    print(f"Archivo: {INPUT_FILE}")

    gdf = gpd.read_parquet(INPUT_FILE)

    print(f"Registros: {len(gdf):,}")
    print(f"Columnas: {len(gdf.columns):,}")
    print(f"CRS: {gdf.crs}")

    # =========================================================================
    # 2. VALIDACIÓN
    # =========================================================================

    titulo("2. VALIDANDO DATOS DE ENTRADA")

    validar_geometria(gdf)

    validar_columna(gdf, "nodo_id")

    if gdf["nodo_id"].duplicated().any():
        raise ValueError(
            "Existen nodo_id duplicados."
        )

    print("Columna identificadora: nodo_id")
    print(f"IDs duplicados: {gdf['nodo_id'].duplicated().sum()}")
    print(f"Centralidades esperadas: {len(gdf):,}")

    columnas_obligatorias = [
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
        "indice_centralidad_estructural",
        "indice_centralidad_estructural_robusto",
        "deficit_infraestructura",
        "prioridad_intervencion",
    ]

    subtitulo("COLUMNAS ESTRUCTURALES")

    for columna in columnas_obligatorias:

        if columna in gdf.columns:
            print(f"  OK  {columna}")

        else:
            raise ValueError(
                f"Falta columna estructural obligatoria: {columna}"
            )

    # =========================================================================
    # 3. NORMALIZACIÓN DE DIMENSIONES
    # =========================================================================

    titulo("3. NORMALIZANDO DIMENSIONES")

    dimensiones = [
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
    ]

    for columna in dimensiones:

        nueva = columna.replace(
            "indice_",
            "nivel_",
            1,
        )

        gdf[nueva] = gdf[columna].clip(0, 100)

        print(
            f"{columna}: "
            f"{gdf[nueva].min():.2f} - "
            f"{gdf[nueva].max():.2f}"
        )

    # =========================================================================
    # 4. NIVELES CUALITATIVOS
    # =========================================================================

    titulo("4. CLASIFICANDO NIVELES DIMENSIONALES")

    gdf["nivel_demanda"] = gdf[
        "indice_demanda_estructural"
    ].apply(nivel)

    gdf["nivel_infraestructura"] = gdf[
        "indice_infraestructura_estructural"
    ].apply(nivel)

    gdf["nivel_intermodalidad"] = gdf[
        "indice_intermodalidad_estructural"
    ].apply(nivel)

    gdf["nivel_conectividad"] = gdf[
        "indice_conectividad_estructural"
    ].apply(nivel)

    gdf["nivel_integracion_territorial"] = gdf[
        "indice_integracion_territorial"
    ].apply(nivel)

    # =========================================================================
    # 5. DÉFICIT ESTRUCTURAL
    # =========================================================================

    titulo("5. ANALIZANDO DÉFICIT ESTRUCTURAL")

    gdf["deficit_infraestructura"] = pd.to_numeric(
        gdf["deficit_infraestructura"],
        errors="coerce",
    ).fillna(0).clip(0, 100)

    gdf["nivel_deficit_infraestructura"] = (
        gdf["deficit_infraestructura"]
        .apply(nivel)
    )

    print(
        "Déficit infraestructura: "
        f"{gdf['deficit_infraestructura'].min():.2f} - "
        f"{gdf['deficit_infraestructura'].max():.2f}"
    )

    # =========================================================================
    # 6. PRIORIDAD
    # =========================================================================

    titulo("6. CLASIFICANDO PRIORIDAD DE INTERVENCIÓN")

    gdf["prioridad_intervencion"] = pd.to_numeric(
        gdf["prioridad_intervencion"],
        errors="coerce",
    ).fillna(0).clip(0, 100)

    gdf["categoria_prioridad_intervencion"] = gdf.apply(
        clasificar_prioridad,
        axis=1,
    )

    print(
        gdf["categoria_prioridad_intervencion"]
        .value_counts()
        .to_string()
    )

    # =========================================================================
    # 7. TIPOLOGÍA PRINCIPAL
    # =========================================================================

    titulo("7. CONSTRUYENDO TIPOLOGÍA PRINCIPAL")

    gdf["tipologia_centralidad"] = gdf.apply(
        clasificar_tipologia,
        axis=1,
    )

    print()
    print(
        gdf["tipologia_centralidad"]
        .value_counts()
        .to_string()
    )

    # =========================================================================
    # 8. DIMENSIÓN DOMINANTE
    # =========================================================================

    titulo("8. IDENTIFICANDO DIMENSIÓN DOMINANTE")

    gdf["dimension_dominante"] = gdf.apply(
        construir_dimension_dominante,
        axis=1,
    )

    print()
    print(
        gdf["dimension_dominante"]
        .value_counts()
        .to_string()
    )

    # =========================================================================
    # 9. PERFIL MULTIDIMENSIONAL
    # =========================================================================

    titulo("9. CONSTRUYENDO PERFIL MULTIDIMENSIONAL")

    gdf["perfil_multidimensional"] = gdf.apply(
        construir_perfil,
        axis=1,
    )

    print(
        "Perfiles multidimensionales construidos: "
        f"{gdf['perfil_multidimensional'].nunique()}"
    )

    # =========================================================================
    # 10. DIAGNÓSTICO
    # =========================================================================

    titulo("10. CONSTRUYENDO DIAGNÓSTICO TERRITORIAL")

    gdf["diagnostico_tipologico"] = gdf.apply(
        construir_diagnostico,
        axis=1,
    )

    print()
    print(
        gdf["diagnostico_tipologico"]
        .value_counts()
        .to_string()
    )

    # =========================================================================
    # 11. RANKINGS
    # =========================================================================

    titulo("11. CONSTRUYENDO RANKINGS")

    gdf = gdf.sort_values(
        "indice_centralidad_estructural_robusto",
        ascending=False,
    ).reset_index(drop=True)

    gdf["ranking_tipologia_estructural"] = (
        np.arange(len(gdf)) + 1
    )

    gdf = gdf.sort_values(
        "prioridad_intervencion",
        ascending=False,
    ).reset_index(drop=True)

    gdf["ranking_tipologia_prioridad"] = (
        np.arange(len(gdf)) + 1
    )

    # =========================================================================
    # 12. PERFIL DE PRIORIDAD
    # =========================================================================

    titulo("12. CONSTRUYENDO PERFIL DE INTERVENCIÓN")

    def perfil_intervencion(row: pd.Series) -> str:

        demanda = row["indice_demanda_estructural"]
        infraestructura = row["indice_infraestructura_estructural"]
        intermodalidad = row["indice_intermodalidad_estructural"]
        conectividad = row["indice_conectividad_estructural"]

        if demanda >= 80 and infraestructura < 45:
            return "AMPLIAR_INFRAESTRUCTURA"

        if (
            demanda >= 70
            and intermodalidad < 50
        ):
            return "MEJORAR_INTERMODALIDAD"

        if (
            demanda >= 70
            and conectividad < 50
        ):
            return "MEJORAR_CONECTIVIDAD"

        if (
            infraestructura >= 60
            and intermodalidad < 50
        ):
            return "FORTALECER_INTERCAMBIO_MODAL"

        if (
            demanda < 60
            and infraestructura >= 60
        ):
            return "MONITOREAR_DEMANDA"

        return "MANTENER_Y_OPTIMIZAR"

    gdf["perfil_intervencion"] = gdf.apply(
        perfil_intervencion,
        axis=1,
    )

    print(
        gdf["perfil_intervencion"]
        .value_counts()
        .to_string()
    )

    # =========================================================================
    # 13. VALIDACIÓN FINAL
    # =========================================================================

    titulo("13. VALIDACIÓN FINAL")

    columnas_nuevas = [
        "tipologia_centralidad",
        "dimension_dominante",
        "perfil_multidimensional",
        "diagnostico_tipologico",
        "perfil_intervencion",
        "categoria_prioridad_intervencion",
    ]

    for columna in columnas_nuevas:

        nulos = gdf[columna].isna().sum()

        print(
            f"{columna}: {nulos} nulos"
        )

        if nulos > 0:
            raise ValueError(
                f"La columna {columna} contiene nulos."
            )

    if len(gdf) != 144:
        print(
            f"ADVERTENCIA: se esperaban 144 centralidades "
            f"y se encontraron {len(gdf)}."
        )

    # =========================================================================
    # 14. TOP 20
    # =========================================================================

    titulo("TOP 20 CENTRALIDADES POR ÍNDICE ESTRUCTURAL")

    columnas_top = [
        "nodo_id",
        "tipologia_centralidad",
        "indice_centralidad_estructural",
        "indice_centralidad_estructural_robusto",
        "ranking_centralidad_estructural",
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
        "deficit_infraestructura",
        "prioridad_intervencion",
    ]

    columnas_top = [
        c for c in columnas_top
        if c in gdf.columns
    ]

    print(
        gdf.sort_values(
            "indice_centralidad_estructural_robusto",
            ascending=False,
        )[columnas_top]
        .head(20)
        .to_string(index=False)
    )

    # =========================================================================
    # 15. TOP PRIORIDADES
    # =========================================================================

    titulo("TOP 20 CENTRALIDADES POR PRIORIDAD DE INTERVENCIÓN")

    columnas_prioridad = [
        "nodo_id",
        "tipologia_centralidad",
        "prioridad_intervencion",
        "ranking_tipologia_prioridad",
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "deficit_infraestructura",
        "perfil_intervencion",
    ]

    print(
        gdf.sort_values(
            "prioridad_intervencion",
            ascending=False,
        )[columnas_prioridad]
        .head(20)
        .to_string(index=False)
    )

    # =========================================================================
    # 16. RESUMEN JSON
    # =========================================================================

    titulo("16. CONSTRUYENDO RESUMEN JSON")

    resumen = {
        "proceso": 24,
        "nombre": "Clasificación de tipología de centralidades AMBA",
        "version": "V1",
        "fecha_ejecucion": pd.Timestamp.now().isoformat(),
        "centralidades_analizadas": int(len(gdf)),
        "crs": CRS_GEOGRAFICO,
        "crs_metrico": CRS_METRICO,
        "tipologias": (
            gdf["tipologia_centralidad"]
            .value_counts()
            .to_dict()
        ),
        "prioridades": (
            gdf["categoria_prioridad_intervencion"]
            .value_counts()
            .to_dict()
        ),
        "dimensiones_dominantes": (
            gdf["dimension_dominante"]
            .value_counts()
            .to_dict()
        ),
        "diagnosticos": (
            gdf["diagnostico_tipologico"]
            .value_counts()
            .to_dict()
        ),
        "perfiles_intervencion": (
            gdf["perfil_intervencion"]
            .value_counts()
            .to_dict()
        ),
        "perfiles_multidimensionales": (
            gdf["perfil_multidimensional"]
            .value_counts()
            .to_dict()
        ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================================
    # 17. GUARDAR PARQUET
    # =========================================================================

    titulo("17. GUARDANDO ARCHIVOS")

    gdf.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    print(f"Parquet:\n{OUTPUT_PARQUET}")

    # =========================================================================
    # 18. GUARDAR CSV
    # =========================================================================

    df_csv = pd.DataFrame(gdf.drop(columns="geometry"))

    df_csv.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"CSV:\n{OUTPUT_CSV}")

    # =========================================================================
    # 19. GUARDAR GEOPACKAGE
    # =========================================================================

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    gdf.to_file(
        OUTPUT_GPKG,
        layer="tipologia_centralidades",
        driver="GPKG",
    )

    print(f"GeoPackage:\n{OUTPUT_GPKG}")

    print(f"JSON:\n{OUTPUT_JSON}")

    # =========================================================================
    # 20. MAPA TIPOLOGÍAS
    # =========================================================================

    titulo("18. GENERANDO MAPAS Y GRÁFICOS")

    mapa_tipologia = OUTPUT_DIR / (
        "01_mapa_tipologia_centralidades.png"
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    gdf.plot(
        ax=ax,
        column="tipologia_centralidad",
        categorical=True,
        legend=True,
        markersize=45,
        alpha=0.85,
    )

    ax.set_title(
        "Tipología de Centralidades de Movilidad - AMBA",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(
        mapa_tipologia,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Mapa: {mapa_tipologia}")

    # =========================================================================
    # 21. MAPA PRIORIDAD
    # =========================================================================

    mapa_prioridad = OUTPUT_DIR / (
        "02_mapa_prioridad_intervencion.png"
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    gdf.plot(
        ax=ax,
        column="prioridad_intervencion",
        cmap="RdYlGn_r",
        legend=True,
        markersize=50,
        alpha=0.9,
        vmin=0,
        vmax=100,
    )

    ax.set_title(
        "Prioridad de Intervención - Centralidades AMBA",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(
        mapa_prioridad,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Mapa: {mapa_prioridad}")

    # =========================================================================
    # 22. MAPA DÉFICIT
    # =========================================================================

    mapa_deficit = OUTPUT_DIR / (
        "03_mapa_deficit_tipologico.png"
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    gdf.plot(
        ax=ax,
        column="deficit_infraestructura",
        cmap="OrRd",
        legend=True,
        markersize=50,
        alpha=0.9,
        vmin=0,
        vmax=100,
    )

    ax.set_title(
        "Déficit de Infraestructura por Centralidad - AMBA",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(
        mapa_deficit,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Mapa: {mapa_deficit}")

    # =========================================================================
    # 23. GRÁFICO DE TIPOLOGÍAS
    # =========================================================================

    grafico_tipologias = OUTPUT_DIR / (
        "04_centralidades_por_tipologia.png"
    )

    conteo = (
        gdf["tipologia_centralidad"]
        .value_counts()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    conteo.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Centralidades por Tipología",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Cantidad de centralidades"
    )

    ax.set_ylabel(
        "Tipología"
    )

    plt.tight_layout()
    plt.savefig(
        grafico_tipologias,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Gráfico: {grafico_tipologias}")

    # =========================================================================
    # 24. GRÁFICO PRIORIDAD
    # =========================================================================

    grafico_prioridad = OUTPUT_DIR / (
        "05_centralidades_por_prioridad.png"
    )

    conteo_prioridad = (
        gdf["categoria_prioridad_intervencion"]
        .value_counts()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    conteo_prioridad.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Centralidades por Prioridad de Intervención",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Cantidad de centralidades"
    )

    ax.set_ylabel(
        "Prioridad"
    )

    plt.tight_layout()
    plt.savefig(
        grafico_prioridad,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Gráfico: {grafico_prioridad}")

    # =========================================================================
    # 25. DEMANDA VS INFRAESTRUCTURA
    # =========================================================================

    grafico_demanda = OUTPUT_DIR / (
        "06_demanda_vs_infraestructura_tipologia.png"
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    scatter = ax.scatter(
        gdf["indice_demanda_estructural"],
        gdf["indice_infraestructura_estructural"],
        s=55,
        alpha=0.75,
        c=gdf["prioridad_intervencion"],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
    )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Índice de demanda estructural"
    )

    ax.set_ylabel(
        "Índice de infraestructura estructural"
    )

    ax.set_title(
        "Demanda vs. Infraestructura por Centralidad",
        fontsize=15,
        fontweight="bold",
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Prioridad de intervención",
    )

    plt.tight_layout()
    plt.savefig(
        grafico_demanda,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Gráfico: {grafico_demanda}")

    # =========================================================================
    # 26. PERFIL MULTIDIMENSIONAL
    # =========================================================================

    grafico_perfiles = OUTPUT_DIR / (
        "07_perfiles_multidimensionales.png"
    )

    conteo_perfiles = (
        gdf["perfil_multidimensional"]
        .value_counts()
        .head(15)
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    conteo_perfiles.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Principales Perfiles Multidimensionales",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Cantidad de centralidades"
    )

    plt.tight_layout()
    plt.savefig(
        grafico_perfiles,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Gráfico: {grafico_perfiles}")

    # =========================================================================
    # 27. FINAL
    # =========================================================================

    titulo(
        "24 - PROCESO FINALIZADO"
    )

    print(
        f"Centralidades analizadas: {len(gdf):,}"
    )

    print()
    print("ARCHIVOS GENERADOS")

    archivos = [
        mapa_tipologia,
        mapa_prioridad,
        mapa_deficit,
        grafico_tipologias,
        grafico_prioridad,
        grafico_demanda,
        grafico_perfiles,
        OUTPUT_CSV,
        OUTPUT_GPKG,
        OUTPUT_PARQUET,
        OUTPUT_JSON,
    ]

    for archivo in archivos:
        print(f"  {archivo.name}")

    print()
    print("SIGUIENTE ETAPA")
    print(
        "Construir la priorización territorial de intervenciones "
        "integrando tipología, demanda, déficit, intermodalidad, "
        "conectividad y prioridad."
    )


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Proceso interrumpido por el usuario.")
        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("24 - ERROR")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "El proceso fue detenido para evitar "
            "generar resultados incompletos."
        )

        traceback.print_exc()

        sys.exit(1)