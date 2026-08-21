# -*- coding: utf-8 -*-

"""
25 - PRIORIZACIÓN DE INTERVENCIONES TERRITORIALES AMBA

Construye una priorización territorial de intervenciones a partir de:

    - Demanda
    - Infraestructura
    - Intermodalidad
    - Conectividad
    - Integración territorial
    - Déficit estructural
    - Índice de centralidad estructural
    - Tipología de centralidad
    - Prioridad de intervención

Entrada:
    data/processed/tipologia_centralidades_amba/
        tipologia_centralidades_amba.parquet

Salidas:
    data/processed/priorizacion_intervenciones_territoriales_amba/

    - priorizacion_intervenciones_territoriales_amba.parquet
    - priorizacion_intervenciones_territoriales_amba.csv
    - priorizacion_intervenciones_territoriales_amba.gpkg
    - priorizacion_intervenciones_territoriales_amba_resumen.json

    - 01_mapa_prioridad_territorial.png
    - 02_mapa_intervenciones.png
    - 03_mapa_urgencia.png
    - 04_mapa_impacto_potencial.png
    - 05_mapa_deficit_estructural.png
    - 06_demanda_vs_infraestructura.png
    - 07_prioridades_por_intervencion.png
    - 08_prioridades_por_tipologia.png
    - 09_distribucion_prioridad.png
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SCRIPT_VERSION = "V1"

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "tipologia_centralidades_amba"
)

INPUT_FILE = INPUT_DIR / "tipologia_centralidades_amba.parquet"

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "priorizacion_intervenciones_territoriales_amba"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "priorizacion_intervenciones_territoriales_amba.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "priorizacion_intervenciones_territoriales_amba.csv"
)

OUTPUT_GPKG = (
    OUTPUT_DIR
    / "priorizacion_intervenciones_territoriales_amba.gpkg"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "priorizacion_intervenciones_territoriales_amba_resumen.json"
)

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"


# ============================================================================
# PESOS
# ============================================================================

# Los pesos representan el aporte de cada dimensión a la prioridad territorial.

PESO_DEMANDA = 0.25
PESO_DEFICIT_INFRAESTRUCTURA = 0.20
PESO_DEFICIT_INTERMODALIDAD = 0.15
PESO_DEFICIT_CONECTIVIDAD = 0.10
PESO_DEFICIT_INTEGRACION = 0.05
PESO_CENTRALIDAD = 0.15
PESO_IMPACTO = 0.10


# ============================================================================
# UTILIDADES
# ============================================================================

def encabezado(titulo: str) -> None:
    print()
    print("=" * 78)
    print(titulo)
    print("=" * 78)


def subtitulo(titulo: str) -> None:
    print()
    print("-" * 78)
    print(titulo)
    print("-" * 78)


def normalizar_0_100(
    serie: pd.Series,
    invertido: bool = False,
) -> pd.Series:
    """
    Normalización min-max a escala 0-100.

    Si invertido=True:
        valor alto original -> score bajo.

    Esto resulta útil para transformar indicadores positivos
    en déficits cuando corresponda.
    """

    valores = pd.to_numeric(serie, errors="coerce")

    minimo = valores.min()
    maximo = valores.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(
            np.zeros(len(serie)),
            index=serie.index,
            dtype=float,
        )

    if math.isclose(float(maximo), float(minimo)):
        resultado = pd.Series(
            np.full(len(serie), 50.0),
            index=serie.index,
            dtype=float,
        )
    else:
        resultado = (
            (valores - minimo)
            / (maximo - minimo)
            * 100.0
        )

    if invertido:
        resultado = 100.0 - resultado

    return resultado.clip(0, 100)


def clasificar_prioridad(score: float) -> str:
    if score >= 75:
        return "PRIORIDAD_CRITICA"

    if score >= 60:
        return "PRIORIDAD_ALTA"

    if score >= 40:
        return "PRIORIDAD_MEDIA"

    return "PRIORIDAD_BAJA"


def nivel_urgencia(score: float) -> str:
    if score >= 75:
        return "URGENTE"

    if score >= 60:
        return "ALTA"

    if score >= 40:
        return "MEDIA"

    return "BAJA"


def nombre_intervencion(
    fila: pd.Series,
) -> str:
    """
    Determina la intervención principal.

    La lógica prioriza primero situaciones críticas de demanda
    e infraestructura, luego déficits específicos.
    """

    demanda = float(fila["indice_demanda_estructural"])
    infraestructura = float(
        fila["indice_infraestructura_estructural"]
    )
    intermodalidad = float(
        fila["indice_intermodalidad_estructural"]
    )
    conectividad = float(
        fila["indice_conectividad_estructural"]
    )
    integracion = float(
        fila["indice_integracion_territorial"]
    )

    deficit_infra = float(
        fila["deficit_infraestructura"]
    )

    deficit_intermodal = float(
        fila["deficit_intermodalidad"]
    )

    deficit_conectividad = float(
        fila["deficit_conectividad"]
    )

    deficit_integracion = float(
        fila["deficit_integracion"]
    )

    prioridad = float(
        fila["prioridad_intervencion"]
    )

    tipologia = str(
        fila.get(
            "tipologia_centralidad",
            "",
        )
    )

    # ------------------------------------------------------------
    # Intervención integral
    # ------------------------------------------------------------

    if (
        demanda >= 80
        and deficit_infra >= 50
        and prioridad >= 75
    ):
        return "INTERVENCION_INTEGRAL"

    # ------------------------------------------------------------
    # Ampliación de infraestructura
    # ------------------------------------------------------------

    if (
        deficit_infra >= 45
        and demanda >= 70
    ):
        return "AMPLIAR_INFRAESTRUCTURA"

    # ------------------------------------------------------------
    # Intermodalidad
    # ------------------------------------------------------------

    if (
        deficit_intermodal >= 45
        and (
            demanda >= 60
            or intermodalidad >= 60
        )
    ):
        return "MEJORAR_INTERMODALIDAD"

    # ------------------------------------------------------------
    # Conectividad
    # ------------------------------------------------------------

    if deficit_conectividad >= 45:
        return "MEJORAR_CONECTIVIDAD"

    # ------------------------------------------------------------
    # Integración territorial
    # ------------------------------------------------------------

    if deficit_integracion >= 45:
        return "MEJORAR_INTEGRACION_TERRITORIAL"

    # ------------------------------------------------------------
    # Consolidación
    # ------------------------------------------------------------

    if (
        demanda >= 70
        and infraestructura >= 60
        and intermodalidad >= 60
    ):
        return "CONSOLIDAR_CENTRALIDAD"

    # ------------------------------------------------------------
    # Centralidades estratégicas/metropolitanas
    # ------------------------------------------------------------

    if tipologia in {
        "CENTRALIDAD_METROPOLITANA",
        "CENTRALIDAD_ESTRATEGICA",
    }:
        return "CONSOLIDAR_CENTRALIDAD"

    # ------------------------------------------------------------
    # Seguimiento
    # ------------------------------------------------------------

    return "MONITOREAR"


def justificacion_intervencion(
    fila: pd.Series,
) -> str:

    intervencion = fila["tipo_intervencion_recomendada"]

    demanda = float(
        fila["indice_demanda_estructural"]
    )

    infraestructura = float(
        fila["indice_infraestructura_estructural"]
    )

    deficit = float(
        fila["deficit_infraestructura"]
    )

    prioridad = float(
        fila["score_prioridad_territorial"]
    )

    if intervencion == "INTERVENCION_INTEGRAL":
        return (
            "Demanda elevada combinada con déficit "
            "infraestructural significativo y alta prioridad "
            "territorial."
        )

    if intervencion == "AMPLIAR_INFRAESTRUCTURA":
        return (
            f"Demanda {demanda:.1f}/100 frente a infraestructura "
            f"{infraestructura:.1f}/100, con déficit "
            f"infrastructural de {deficit:.1f} puntos."
        )

    if intervencion == "MEJORAR_INTERMODALIDAD":
        return (
            "Existe un déficit relevante de intermodalidad "
            "respecto del potencial estructural de la centralidad."
        )

    if intervencion == "MEJORAR_CONECTIVIDAD":
        return (
            "La conectividad constituye una de las principales "
            "restricciones estructurales de la centralidad."
        )

    if intervencion == "MEJORAR_INTEGRACION_TERRITORIAL":
        return (
            "La integración territorial presenta un déficit "
            "relevante respecto de las demás dimensiones."
        )

    if intervencion == "CONSOLIDAR_CENTRALIDAD":
        return (
            "Centralidad con condiciones estructurales suficientes "
            "para consolidar y optimizar su funcionamiento."
        )

    return (
        f"Prioridad territorial {prioridad:.1f}/100 sin un "
        "déficit dominante que justifique una intervención mayor."
    )


def calcular_categoria_impacto(score: float) -> str:

    if score >= 75:
        return "IMPACTO_MUY_ALTO"

    if score >= 60:
        return "IMPACTO_ALTO"

    if score >= 40:
        return "IMPACTO_MEDIO"

    return "IMPACTO_BAJO"


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    encabezado(
        "25 - PRIORIZACIÓN DE INTERVENCIONES TERRITORIALES AMBA - "
        f"{SCRIPT_VERSION}"
    )

    print(f"Proyecto : {PROJECT_DIR}")
    print(f"Entrada  : {INPUT_FILE}")
    print(f"Salida   : {OUTPUT_DIR}")
    print(f"CRS      : {CRS_GEOGRAFICO}")
    print(f"CRS métrico: {CRS_METRICO}")

    print()
    print("PESOS DEL MODELO")
    print(f"  Demanda:                       {PESO_DEMANDA:.0%}")
    print(
        f"  Déficit infraestructura:      "
        f"{PESO_DEFICIT_INFRAESTRUCTURA:.0%}"
    )
    print(
        f"  Déficit intermodalidad:       "
        f"{PESO_DEFICIT_INTERMODALIDAD:.0%}"
    )
    print(
        f"  Déficit conectividad:         "
        f"{PESO_DEFICIT_CONECTIVIDAD:.0%}"
    )
    print(
        f"  Déficit integración:          "
        f"{PESO_DEFICIT_INTEGRACION:.0%}"
    )
    print(
        f"  Centralidad estructural:      "
        f"{PESO_CENTRALIDAD:.0%}"
    )
    print(
        f"  Impacto potencial:            "
        f"{PESO_IMPACTO:.0%}"
    )

    # ========================================================================
    # 1
    # ========================================================================

    encabezado(
        "1. CARGANDO RESULTADOS DEL PROCESO 24"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada:\n{INPUT_FILE}"
        )

    print(f"Archivo:\n{INPUT_FILE}")

    gdf = gpd.read_parquet(INPUT_FILE)

    print(f"Registros: {len(gdf):,}")
    print(f"Columnas: {len(gdf.columns)}")
    print(f"CRS: {gdf.crs}")

    # ========================================================================
    # 2
    # ========================================================================

    encabezado("2. VALIDANDO DATOS DE ENTRADA")

    if "geometry" not in gdf.columns:
        raise ValueError(
            "El archivo no contiene columna geometry."
        )

    if gdf.crs is None:
        print(
            "CRS inexistente. Se asignará EPSG:22185 "
            "porque el proceso 24 trabaja en ese sistema."
        )
        gdf = gdf.set_crs(CRS_METRICO)

    if "nodo_id" not in gdf.columns:
        raise ValueError(
            "No se encontró la columna nodo_id."
        )

    nulos_geom = int(gdf.geometry.isna().sum())
    vacias_geom = int(gdf.geometry.is_empty.sum())
    invalidas_geom = int(
        (~gdf.geometry.is_valid).sum()
    )

    duplicados = int(
        gdf["nodo_id"].duplicated().sum()
    )

    print(f"Geometrías nulas: {nulos_geom}")
    print(f"Geometrías vacías: {vacias_geom}")
    print(f"Geometrías inválidas: {invalidas_geom}")
    print(f"IDs duplicados: {duplicados}")

    if nulos_geom > 0:
        raise ValueError(
            "Existen geometrías nulas."
        )

    if vacias_geom > 0:
        raise ValueError(
            "Existen geometrías vacías."
        )

    if invalidas_geom > 0:
        raise ValueError(
            "Existen geometrías inválidas."
        )

    if duplicados > 0:
        raise ValueError(
            "Existen nodo_id duplicados."
        )

    # ========================================================================
    # 3
    # ========================================================================

    encabezado("3. VALIDANDO COMPONENTES ESTRUCTURALES")

    columnas_requeridas = [
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

    for columna in columnas_requeridas:

        if columna not in gdf.columns:
            raise ValueError(
                f"No se encontró la columna requerida: "
                f"{columna}"
            )

        nulos = int(gdf[columna].isna().sum())

        print(
            f"  {'OK' if nulos == 0 else 'ERROR'} "
            f"{columna}: {nulos} nulos"
        )

        if nulos > 0:
            raise ValueError(
                f"La columna {columna} contiene nulos."
            )

    print()
    print(f"Centralidades validadas: {len(gdf):,}")

    # ========================================================================
    # 4
    # ========================================================================

    encabezado("4. NORMALIZANDO DIMENSIONES")

    dimensiones = [
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "indice_intermodalidad_estructural",
        "indice_conectividad_estructural",
        "indice_integracion_territorial",
    ]

    for columna in dimensiones:

        gdf[columna] = pd.to_numeric(
            gdf[columna],
            errors="coerce",
        )

        print(
            f"{columna}: "
            f"{gdf[columna].min():.2f} - "
            f"{gdf[columna].max():.2f}"
        )

    # ========================================================================
    # 5
    # ========================================================================

    encabezado("5. CALCULANDO DÉFICITS DIMENSIONALES")

    # Los índices estructurales representan capacidad/fortaleza.
    # Por lo tanto:
    #
    # déficit = 100 - índice

    gdf["deficit_demanda"] = (
        100
        - gdf["indice_demanda_estructural"]
    ).clip(0, 100)

    gdf["deficit_infraestructura"] = (
        100
        - gdf["indice_infraestructura_estructural"]
    ).clip(0, 100)

    gdf["deficit_intermodalidad"] = (
        100
        - gdf["indice_intermodalidad_estructural"]
    ).clip(0, 100)

    gdf["deficit_conectividad"] = (
        100
        - gdf["indice_conectividad_estructural"]
    ).clip(0, 100)

    gdf["deficit_integracion"] = (
        100
        - gdf["indice_integracion_territorial"]
    ).clip(0, 100)

    # Para infraestructura usamos también el déficit ya producido
    # en el proceso 23/24 cuando existe.
    if "deficit_infraestructura" in gdf.columns:

        deficit_preexistente = pd.to_numeric(
            gdf["deficit_infraestructura"],
            errors="coerce",
        )

        deficit_calculado = (
            100
            - gdf["indice_infraestructura_estructural"]
        ).clip(0, 100)

        gdf["deficit_infraestructura"] = (
            deficit_preexistente
            .fillna(deficit_calculado)
            .clip(0, 100)
        )

    print(
        f"Déficit infraestructura: "
        f"{gdf['deficit_infraestructura'].min():.2f} - "
        f"{gdf['deficit_infraestructura'].max():.2f}"
    )

    print(
        f"Déficit intermodalidad: "
        f"{gdf['deficit_intermodalidad'].min():.2f} - "
        f"{gdf['deficit_intermodalidad'].max():.2f}"
    )

    print(
        f"Déficit conectividad: "
        f"{gdf['deficit_conectividad'].min():.2f} - "
        f"{gdf['deficit_conectividad'].max():.2f}"
    )

    print(
        f"Déficit integración: "
        f"{gdf['deficit_integracion'].min():.2f} - "
        f"{gdf['deficit_integracion'].max():.2f}"
    )

    # ========================================================================
    # 6
    # ========================================================================

    encabezado("6. CALCULANDO POTENCIAL DE IMPACTO")

    # El impacto potencial representa la combinación de:
    #
    #   demanda
    #   +
    #   centralidad
    #   +
    #   déficit
    #
    # Una centralidad con demanda alta y déficit alto
    # presenta mayor potencial de mejora.

    demanda = gdf[
        "indice_demanda_estructural"
    ]

    centralidad = gdf[
        "indice_centralidad_estructural_robusto"
    ]

    deficit_promedio = (
        gdf["deficit_infraestructura"]
        + gdf["deficit_intermodalidad"]
        + gdf["deficit_conectividad"]
        + gdf["deficit_integracion"]
    ) / 4.0

    gdf["deficit_estructural_promedio"] = (
        deficit_promedio.clip(0, 100)
    )

    gdf["impacto_potencial"] = (
        0.50 * demanda
        + 0.30 * centralidad
        + 0.20 * deficit_promedio
    ).clip(0, 100)

    gdf["categoria_impacto"] = (
        gdf["impacto_potencial"]
        .apply(calcular_categoria_impacto)
    )

    print(
        "Impacto potencial: "
        f"{gdf['impacto_potencial'].min():.2f} - "
        f"{gdf['impacto_potencial'].max():.2f}"
    )

    # ========================================================================
    # 7
    # ========================================================================

    encabezado("7. CALCULANDO URGENCIA")

    # La urgencia se concentra en el déficit:
    #
    # infraestructura 40%
    # intermodalidad 25%
    # conectividad 20%
    # integración 15%

    gdf["urgencia_intervencion"] = (
        0.40 * gdf["deficit_infraestructura"]
        + 0.25 * gdf["deficit_intermodalidad"]
        + 0.20 * gdf["deficit_conectividad"]
        + 0.15 * gdf["deficit_integracion"]
    ).clip(0, 100)

    gdf["nivel_urgencia"] = (
        gdf["urgencia_intervencion"]
        .apply(nivel_urgencia)
    )

    print(
        f"Urgencia: "
        f"{gdf['urgencia_intervencion'].min():.2f} - "
        f"{gdf['urgencia_intervencion'].max():.2f}"
    )

    # ========================================================================
    # 8
    # ========================================================================

    encabezado(
        "8. CONSTRUYENDO SCORE DE PRIORIDAD TERRITORIAL"
    )

    # El score final combina:
    #
    # demanda                    25%
    # déficit infraestructura    20%
    # déficit intermodalidad     15%
    # déficit conectividad       10%
    # déficit integración         5%
    # centralidad estructural    15%
    # impacto potencial          10%
    #
    # Total                      100%

    gdf["score_prioridad_territorial"] = (
        PESO_DEMANDA
        * gdf["indice_demanda_estructural"]

        + PESO_DEFICIT_INFRAESTRUCTURA
        * gdf["deficit_infraestructura"]

        + PESO_DEFICIT_INTERMODALIDAD
        * gdf["deficit_intermodalidad"]

        + PESO_DEFICIT_CONECTIVIDAD
        * gdf["deficit_conectividad"]

        + PESO_DEFICIT_INTEGRACION
        * gdf["deficit_integracion"]

        + PESO_CENTRALIDAD
        * gdf["indice_centralidad_estructural_robusto"]

        + PESO_IMPACTO
        * gdf["impacto_potencial"]
    ).clip(0, 100)

    gdf["nivel_prioridad_territorial"] = (
        gdf["score_prioridad_territorial"]
        .apply(clasificar_prioridad)
    )

    # ========================================================================
    # 9
    # ========================================================================

    encabezado(
        "9. CLASIFICANDO TIPO DE INTERVENCIÓN"
    )

    gdf["tipo_intervencion_recomendada"] = (
        gdf.apply(
            nombre_intervencion,
            axis=1,
        )
    )

    gdf["justificacion_intervencion"] = (
        gdf.apply(
            justificacion_intervencion,
            axis=1,
        )
    )

    print()
    print(
        gdf["tipo_intervencion_recomendada"]
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # 10
    # ========================================================================

    encabezado(
        "10. CONSTRUYENDO RANKINGS TERRITORIALES"
    )

    gdf = gdf.sort_values(
        [
            "score_prioridad_territorial",
            "impacto_potencial",
            "urgencia_intervencion",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    gdf["ranking_prioridad_territorial"] = (
        np.arange(1, len(gdf) + 1)
    )

    # Ranking por tipo de intervención

    gdf["ranking_intervencion"] = (
        gdf.groupby(
            "tipo_intervencion_recomendada"
        )["score_prioridad_territorial"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    # Ranking por prioridad

    gdf["ranking_nivel_prioridad"] = (
        gdf.groupby(
            "nivel_prioridad_territorial"
        )["score_prioridad_territorial"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    # ========================================================================
    # 11
    # ========================================================================

    encabezado(
        "11. CONSTRUYENDO MATRIZ DE INTERVENCIÓN"
    )

    def construir_matriz(fila: pd.Series) -> str:

        componentes = {
            "INFRAESTRUCTURA": fila["deficit_infraestructura"],
            "INTERMODALIDAD": fila["deficit_intermodalidad"],
            "CONECTIVIDAD": fila["deficit_conectividad"],
            "INTEGRACION": fila["deficit_integracion"],
        }

        ordenados = sorted(
            componentes.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        principales = [
            nombre
            for nombre, valor in ordenados
            if valor >= 40
        ]

        if not principales:
            return "SIN_DEFICIT_DOMINANTE"

        return "|".join(principales)

    gdf["dimensiones_prioritarias"] = (
        gdf.apply(
            construir_matriz,
            axis=1,
        )
    )

    # ========================================================================
    # 12
    # ========================================================================

    encabezado(
        "12. CONSTRUYENDO DIAGNÓSTICO TERRITORIAL"
    )

    def diagnostico(fila: pd.Series) -> str:

        demanda = float(
            fila["indice_demanda_estructural"]
        )

        infraestructura = float(
            fila["indice_infraestructura_estructural"]
        )

        intermodalidad = float(
            fila["indice_intermodalidad_estructural"]
        )

        conectividad = float(
            fila["indice_conectividad_estructural"]
        )

        if (
            demanda >= 80
            and infraestructura < 50
        ):
            return "ALTA_DEMANDA_BAJO_SOPORTE"

        if (
            demanda >= 80
            and infraestructura >= 50
            and intermodalidad >= 60
        ):
            return "CENTRALIDAD_CONSOLIDADA"

        if (
            conectividad >= 70
            and demanda < 70
        ):
            return "CENTRALIDAD_CONECTIVA"

        if (
            infraestructura >= 65
            and demanda < 70
        ):
            return "SOPORTE_INFRAESTRUCTURAL"

        if (
            demanda < 40
            and infraestructura < 40
        ):
            return "BAJO_DESEMPEÑO_ESTRUCTURAL"

        return "INTERVENCION_SELECTIVA"

    gdf["diagnostico_territorial"] = (
        gdf.apply(
            diagnostico,
            axis=1,
        )
    )

    print(
        gdf["diagnostico_territorial"]
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # 13
    # ========================================================================

    encabezado("13. VALIDACIÓN FINAL")

    columnas_finales = [
        "score_prioridad_territorial",
        "nivel_prioridad_territorial",
        "tipo_intervencion_recomendada",
        "justificacion_intervencion",
        "impacto_potencial",
        "urgencia_intervencion",
        "nivel_urgencia",
        "deficit_estructural_promedio",
        "dimensiones_prioritarias",
        "diagnostico_territorial",
        "ranking_prioridad_territorial",
        "ranking_intervencion",
    ]

    errores = []

    for columna in columnas_finales:

        nulos = int(
            gdf[columna].isna().sum()
        )

        print(
            f"{columna}: {nulos} nulos"
        )

        if nulos > 0:
            errores.append(columna)

    if errores:
        raise ValueError(
            "Se encontraron columnas incompletas: "
            + ", ".join(errores)
        )

    print()
    print("Validación final: OK")

    # ========================================================================
    # 14
    # ========================================================================

    encabezado(
        "14. TOP 20 PRIORIDADES TERRITORIALES"
    )

    columnas_top = [
        "nodo_id",
        "ranking_prioridad_territorial",
        "score_prioridad_territorial",
        "nivel_prioridad_territorial",
        "tipo_intervencion_recomendada",
        "indice_demanda_estructural",
        "indice_infraestructura_estructural",
        "deficit_infraestructura",
        "impacto_potencial",
        "urgencia_intervencion",
    ]

    print(
        gdf[
            columnas_top
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # ========================================================================
    # 15
    # ========================================================================

    encabezado(
        "15. TOP 20 POR INTERVENCIÓN"
    )

    for intervencion, grupo in (
        gdf.groupby(
            "tipo_intervencion_recomendada"
        )
    ):

        print()
        print(
            f"--- {intervencion} ---"
        )

        print(
            grupo[
                [
                    "nodo_id",
                    "score_prioridad_territorial",
                    "nivel_prioridad_territorial",
                    "ranking_intervencion",
                    "impacto_potencial",
                    "urgencia_intervencion",
                ]
            ]
            .head(20)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    # ========================================================================
    # 16
    # ========================================================================

    encabezado(
        "16. RESUMEN DE PRIORIDADES"
    )

    print()
    print(
        gdf[
            "nivel_prioridad_territorial"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        gdf[
            "tipo_intervencion_recomendada"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # 17
    # ========================================================================

    encabezado(
        "17. CONSTRUYENDO RESUMEN JSON"
    )

    resumen = {
        "proceso": 25,
        "version": SCRIPT_VERSION,
        "descripcion": (
            "Priorización de intervenciones territoriales "
            "de centralidades del AMBA."
        ),
        "fecha_proceso": pd.Timestamp.now().isoformat(),
        "centralidades_analizadas": int(len(gdf)),
        "pesos": {
            "demanda": PESO_DEMANDA,
            "deficit_infraestructura": PESO_DEFICIT_INFRAESTRUCTURA,
            "deficit_intermodalidad": PESO_DEFICIT_INTERMODALIDAD,
            "deficit_conectividad": PESO_DEFICIT_CONECTIVIDAD,
            "deficit_integracion": PESO_DEFICIT_INTEGRACION,
            "centralidad": PESO_CENTRALIDAD,
            "impacto": PESO_IMPACTO,
        },
        "rangos": {
            "score_prioridad_territorial": {
                "min": float(
                    gdf["score_prioridad_territorial"].min()
                ),
                "max": float(
                    gdf["score_prioridad_territorial"].max()
                ),
            },
            "impacto_potencial": {
                "min": float(
                    gdf["impacto_potencial"].min()
                ),
                "max": float(
                    gdf["impacto_potencial"].max()
                ),
            },
            "urgencia_intervencion": {
                "min": float(
                    gdf["urgencia_intervencion"].min()
                ),
                "max": float(
                    gdf["urgencia_intervencion"].max()
                ),
            },
        },
        "prioridades": {
            str(k): int(v)
            for k, v in (
                gdf[
                    "nivel_prioridad_territorial"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },
        "intervenciones": {
            str(k): int(v)
            for k, v in (
                gdf[
                    "tipo_intervencion_recomendada"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },
        "diagnosticos": {
            str(k): int(v)
            for k, v in (
                gdf[
                    "diagnostico_territorial"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },
        "top_20_prioridades": [
            {
                "nodo_id": int(row["nodo_id"]),
                "ranking": int(
                    row["ranking_prioridad_territorial"]
                ),
                "score": float(
                    row["score_prioridad_territorial"]
                ),
                "prioridad": str(
                    row["nivel_prioridad_territorial"]
                ),
                "intervencion": str(
                    row["tipo_intervencion_recomendada"]
                ),
            }
            for _, row in gdf.head(20).iterrows()
        ],
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            resumen,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================================
    # 18
    # ========================================================================

    encabezado("18. GUARDANDO ARCHIVOS")

    # ------------------------------------------------------------------------
    # PARQUET
    # ------------------------------------------------------------------------

    gdf.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    print(
        f"Parquet:\n{OUTPUT_PARQUET}"
    )

    # ------------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------------

    gdf_csv = gdf.copy()

    gdf_csv["geometry"] = (
        gdf_csv.geometry
        .to_wkt()
    )

    gdf_csv.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"CSV:\n{OUTPUT_CSV}"
    )

    # ------------------------------------------------------------------------
    # GEOPACKAGE
    # ------------------------------------------------------------------------

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    gdf.to_file(
        OUTPUT_GPKG,
        layer="priorizacion_territorial",
        driver="GPKG",
    )

    print(
        f"GeoPackage:\n{OUTPUT_GPKG}"
    )

    print(
        f"JSON:\n{OUTPUT_JSON}"
    )

    # ========================================================================
    # 19
    # ========================================================================

    encabezado(
        "19. GENERANDO MAPAS Y GRÁFICOS"
    )

    # ------------------------------------------------------------------------
    # Preparación gráfica
    # ------------------------------------------------------------------------

    gdf_plot = gdf.to_crs(
        CRS_GEOGRAFICO
    )

    # ------------------------------------------------------------------------
    # 01 MAPA PRIORIDAD
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    gdf_plot.plot(
        ax=ax,
        column="score_prioridad_territorial",
        cmap="RdYlGn_r",
        legend=True,
        markersize=45,
    )

    ax.set_title(
        "Priorización Territorial de Intervenciones - AMBA"
    )

    ax.set_axis_off()

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "01_mapa_prioridad_territorial.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Mapa: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 02 MAPA INTERVENCIONES
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    gdf_plot.plot(
        ax=ax,
        column="tipo_intervencion_recomendada",
        categorical=True,
        legend=True,
        markersize=45,
    )

    ax.set_title(
        "Tipo de Intervención Recomendada - AMBA"
    )

    ax.set_axis_off()

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "02_mapa_intervenciones.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Mapa: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 03 MAPA URGENCIA
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    gdf_plot.plot(
        ax=ax,
        column="urgencia_intervencion",
        cmap="OrRd",
        legend=True,
        markersize=45,
    )

    ax.set_title(
        "Urgencia de Intervención - AMBA"
    )

    ax.set_axis_off()

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "03_mapa_urgencia.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Mapa: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 04 MAPA IMPACTO
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    gdf_plot.plot(
        ax=ax,
        column="impacto_potencial",
        cmap="YlGn",
        legend=True,
        markersize=45,
    )

    ax.set_title(
        "Impacto Potencial de Intervenciones - AMBA"
    )

    ax.set_axis_off()

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "04_mapa_impacto_potencial.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Mapa: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 05 MAPA DÉFICIT
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    gdf_plot.plot(
        ax=ax,
        column="deficit_estructural_promedio",
        cmap="Reds",
        legend=True,
        markersize=45,
    )

    ax.set_title(
        "Déficit Estructural Promedio - AMBA"
    )

    ax.set_axis_off()

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "05_mapa_deficit_estructural.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Mapa: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 06 DEMANDA VS INFRAESTRUCTURA
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.scatter(
        gdf["indice_infraestructura_estructural"],
        gdf["indice_demanda_estructural"],
        s=45,
        alpha=0.75,
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
        "Infraestructura estructural"
    )

    ax.set_ylabel(
        "Demanda estructural"
    )

    ax.set_title(
        "Demanda vs Infraestructura - AMBA"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "06_demanda_vs_infraestructura.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 07 INTERVENCIONES
    # ------------------------------------------------------------------------

    conteo_intervenciones = (
        gdf[
            "tipo_intervencion_recomendada"
        ]
        .value_counts()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    conteo_intervenciones.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_xlabel(
        "Cantidad de centralidades"
    )

    ax.set_ylabel(
        "Intervención"
    )

    ax.set_title(
        "Centralidades por Tipo de Intervención"
    )

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "07_prioridades_por_intervencion.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 08 TIPOLOGÍAS
    # ------------------------------------------------------------------------

    conteo_tipologias = (
        gdf[
            "tipologia_centralidad"
        ]
        .value_counts()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    conteo_tipologias.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_xlabel(
        "Cantidad de centralidades"
    )

    ax.set_ylabel(
        "Tipología"
    )

    ax.set_title(
        "Centralidades por Tipología"
    )

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "08_prioridades_por_tipologia.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {archivo}"
    )

    # ------------------------------------------------------------------------
    # 09 DISTRIBUCIÓN PRIORIDAD
    # ------------------------------------------------------------------------

    conteo_prioridades = (
        gdf[
            "nivel_prioridad_territorial"
        ]
        .value_counts()
        .reindex(
            [
                "PRIORIDAD_CRITICA",
                "PRIORIDAD_ALTA",
                "PRIORIDAD_MEDIA",
                "PRIORIDAD_BAJA",
            ],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    conteo_prioridades.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_xlabel(
        "Nivel de prioridad"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    ax.set_title(
        "Distribución de Prioridad Territorial"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    plt.tight_layout()

    archivo = (
        OUTPUT_DIR
        / "09_distribucion_prioridad.png"
    )

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {archivo}"
    )

    # ========================================================================
    # 20
    # ========================================================================

    encabezado(
        "25 - PROCESO FINALIZADO"
    )

    print(
        f"Centralidades analizadas: {len(gdf):,}"
    )

    print()
    print("ARCHIVOS GENERADOS")

    archivos = [
        "01_mapa_prioridad_territorial.png",
        "02_mapa_intervenciones.png",
        "03_mapa_urgencia.png",
        "04_mapa_impacto_potencial.png",
        "05_mapa_deficit_estructural.png",
        "06_demanda_vs_infraestructura.png",
        "07_prioridades_por_intervencion.png",
        "08_prioridades_por_tipologia.png",
        "09_distribucion_prioridad.png",
        "priorizacion_intervenciones_territoriales_amba.csv",
        "priorizacion_intervenciones_territoriales_amba.gpkg",
        "priorizacion_intervenciones_territoriales_amba.parquet",
        "priorizacion_intervenciones_territoriales_amba_resumen.json",
    ]

    for archivo in archivos:
        print(f"  {archivo}")

    print()
    print(
        "SIGUIENTE ETAPA"
    )

    print(
        "Construir la cartera territorial de proyectos "
        "y escenarios de intervención AMBA a partir de "
        "las centralidades priorizadas."
    )


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("25 - ERROR")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "El proceso fue detenido para evitar "
            "generar resultados incompletos."
        )

        raise