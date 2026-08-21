# -*- coding: utf-8 -*-

"""
26 - CONSTRUCCIÓN DE CARTERA DE PROYECTOS AMBA - V1

Transforma la priorización territorial del proceso 25 en una
cartera estructurada de intervenciones para las centralidades AMBA.

Entrada:
    data/processed/priorizacion_intervenciones_territoriales_amba/
    priorizacion_intervenciones_territoriales_amba.parquet

Salida:
    data/processed/cartera_proyectos_amba/

Autor:
    Proyecto Movilidad AMBA
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


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V1"

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "priorizacion_intervenciones_territoriales_amba"
)

INPUT_FILE = (
    INPUT_DIR
    / "priorizacion_intervenciones_territoriales_amba.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartera_proyectos_amba"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILIDADES
# =============================================================================

def encabezado(titulo: str) -> None:
    print()
    print("=" * 78)
    print(titulo)
    print("=" * 78)


def subencabezado(titulo: str) -> None:
    print()
    print("-" * 78)
    print(titulo)
    print("-" * 78)


def normalizar_serie(serie: pd.Series) -> pd.Series:
    """
    Normaliza una serie al rango 0-100.
    """
    valores = pd.to_numeric(serie, errors="coerce")

    minimo = valores.min()
    maximo = valores.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(0.0, index=serie.index)

    if math.isclose(float(maximo), float(minimo)):
        return pd.Series(50.0, index=serie.index)

    return ((valores - minimo) / (maximo - minimo) * 100.0).clip(0, 100)


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value, default="") -> str:
    if pd.isna(value):
        return default
    return str(value)


def pct(value: float) -> str:
    return f"{safe_float(value):.2f}"


# =============================================================================
# REGLAS DE CARTERA
# =============================================================================

def clasificar_escenario(row: pd.Series) -> str:
    """
    Determina el horizonte temporal de intervención.

    La clasificación utiliza prioridad territorial, urgencia e impacto.
    """

    prioridad = safe_float(row["score_prioridad_territorial"])
    urgencia = safe_float(row["urgencia_intervencion"])
    impacto = safe_float(row["impacto_potencial"])

    if prioridad >= 70 and urgencia >= 55:
        return "ESCENARIO_URGENTE"

    if prioridad >= 65:
        return "ESCENARIO_CORTO_PLAZO"

    if prioridad >= 50 or urgencia >= 60:
        return "ESCENARIO_MEDIANO_PLAZO"

    return "ESCENARIO_ESTRUCTURAL"


def clasificar_horizonte(escenario: str) -> str:
    mapping = {
        "ESCENARIO_URGENTE": "0-2 AÑOS",
        "ESCENARIO_CORTO_PLAZO": "2-5 AÑOS",
        "ESCENARIO_MEDIANO_PLAZO": "5-10 AÑOS",
        "ESCENARIO_ESTRUCTURAL": "10+ AÑOS",
    }

    return mapping.get(escenario, "SIN_DEFINIR")


def clasificar_proyecto(row: pd.Series) -> str:
    """
    Convierte la intervención recomendada del proceso 25
    en una categoría de proyecto de cartera.
    """

    intervencion = safe_str(row["tipo_intervencion_recomendada"])

    if intervencion == "INTERVENCION_INTEGRAL":
        return "PROYECTO_INTEGRAL_CENTRALIDAD"

    if intervencion == "AMPLIAR_INFRAESTRUCTURA":
        return "AMPLIACION_INFRAESTRUCTURA"

    if intervencion == "MEJORAR_INTERMODALIDAD":
        return "MEJORA_INTERMODAL"

    if intervencion == "MEJORAR_CONECTIVIDAD":
        return "MEJORA_CONECTIVIDAD"

    if intervencion == "MEJORAR_INTEGRACION_TERRITORIAL":
        return "INTEGRACION_TERRITORIAL"

    if intervencion == "CONSOLIDAR_CENTRALIDAD":
        return "CONSOLIDACION_CENTRALIDAD"

    if intervencion == "MONITOREAR":
        return "MONITOREO"

    return "INTERVENCION_GENERAL"


def determinar_prioridad_cartera(row: pd.Series) -> str:
    score = safe_float(row["score_prioridad_territorial"])

    if score >= 70:
        return "PRIORIDAD_1_MUY_ALTA"

    if score >= 60:
        return "PRIORIDAD_2_ALTA"

    if score >= 50:
        return "PRIORIDAD_3_MEDIA"

    return "PRIORIDAD_4_BAJA"


def determinar_nivel_intervencion(row: pd.Series) -> str:
    deficit = safe_float(row["deficit_estructural_promedio"])
    demanda = safe_float(row["indice_demanda_estructural"])
    infraestructura = safe_float(
        row["indice_infraestructura_estructural"]
    )

    if demanda >= 80 and infraestructura < 40:
        return "INTERVENCION_ESTRUCTURAL_ALTA"

    if deficit >= 50:
        return "INTERVENCION_ESTRUCTURAL_MEDIA"

    if deficit >= 30:
        return "INTERVENCION_SELECTIVA"

    return "INTERVENCION_OPTIMIZACION"


def construir_dimension_prioritaria(row: pd.Series) -> str:
    """
    Determina las dimensiones que presentan mayor necesidad relativa.
    """

    dimensiones = {
        "DEMANDA": safe_float(
            row["indice_demanda_estructural"]
        ),
        "INFRAESTRUCTURA": safe_float(
            row["indice_infraestructura_estructural"]
        ),
        "INTERMODALIDAD": safe_float(
            row["indice_intermodalidad_estructural"]
        ),
        "CONECTIVIDAD": safe_float(
            row["indice_conectividad_estructural"]
        ),
        "INTEGRACION_TERRITORIAL": safe_float(
            row["indice_integracion_territorial"]
        ),
    }

    deficits = {
        nombre: 100.0 - valor
        for nombre, valor in dimensiones.items()
    }

    ordenadas = sorted(
        deficits.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    principales = [
        nombre
        for nombre, deficit in ordenadas
        if deficit >= 30
    ][:3]

    if not principales:
        principales = [ordenadas[0][0]]

    return " | ".join(principales)


def construir_objetivo_proyecto(row: pd.Series) -> str:
    proyecto = safe_str(row["tipo_proyecto"])

    objetivos = {
        "PROYECTO_INTEGRAL_CENTRALIDAD": (
            "Reducir simultáneamente los principales déficits "
            "estructurales de la centralidad mediante una intervención "
            "integral sobre infraestructura, accesibilidad e integración."
        ),
        "AMPLIACION_INFRAESTRUCTURA": (
            "Incrementar la capacidad y disponibilidad de infraestructura "
            "de movilidad en una centralidad con elevada demanda y déficit "
            "de soporte físico."
        ),
        "MEJORA_INTERMODAL": (
            "Mejorar las condiciones de intercambio entre modos de "
            "transporte y fortalecer la articulación intermodal."
        ),
        "MEJORA_CONECTIVIDAD": (
            "Mejorar la conectividad territorial y la accesibilidad "
            "estructural de la centralidad."
        ),
        "INTEGRACION_TERRITORIAL": (
            "Reducir déficits de integración territorial y mejorar la "
            "articulación de la centralidad con su entorno."
        ),
        "CONSOLIDACION_CENTRALIDAD": (
            "Consolidar una centralidad de alto desempeño mediante "
            "optimización y mantenimiento de sus condiciones actuales."
        ),
        "MONITOREO": (
            "Realizar seguimiento de la evolución territorial antes "
            "de promover una intervención física de mayor escala."
        ),
    }

    return objetivos.get(
        proyecto,
        "Mejorar las condiciones estructurales de movilidad "
        "de la centralidad."
    )


def construir_justificacion(row: pd.Series) -> str:
    tipologia = safe_str(row["tipologia_centralidad"])
    diagnostico = safe_str(row["diagnostico_territorial"])
    intervencion = safe_str(row["tipo_intervencion_recomendada"])

    demanda = safe_float(row["indice_demanda_estructural"])
    infraestructura = safe_float(
        row["indice_infraestructura_estructural"]
    )
    deficit = safe_float(row["deficit_estructural_promedio"])
    prioridad = safe_float(row["score_prioridad_territorial"])

    return (
        f"Centralidad clasificada como {tipologia}, con diagnóstico "
        f"{diagnostico}. Presenta demanda estructural de {demanda:.2f}, "
        f"soporte de infraestructura de {infraestructura:.2f}, déficit "
        f"estructural promedio de {deficit:.2f} y score de prioridad "
        f"territorial de {prioridad:.2f}. La intervención recomendada es "
        f"{intervencion}."
    )


def construir_fase(row: pd.Series) -> str:
    escenario = safe_str(row["escenario_intervencion"])

    if escenario == "ESCENARIO_URGENTE":
        return "FASE_1"

    if escenario == "ESCENARIO_CORTO_PLAZO":
        return "FASE_2"

    if escenario == "ESCENARIO_MEDIANO_PLAZO":
        return "FASE_3"

    return "FASE_4"


# =============================================================================
# 1. CARGA
# =============================================================================

encabezado(
    "26 - CONSTRUCCIÓN DE CARTERA DE PROYECTOS AMBA - V1"
)

print(f"Proyecto : {PROJECT_ROOT}")
print(f"Entrada  : {INPUT_FILE}")
print(f"Salida   : {OUTPUT_DIR}")
print(f"CRS      : {CRS_GEOGRAFICO}")
print(f"CRS métrico: {CRS_METRICO}")


encabezado("1. CARGANDO RESULTADOS DEL PROCESO 25")

print(f"Archivo:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"No existe el archivo de entrada:\n{INPUT_FILE}"
    )

gdf = gpd.read_parquet(INPUT_FILE)

print(f"Registros: {len(gdf)}")
print(f"Columnas: {len(gdf.columns)}")
print(f"CRS: {gdf.crs}")


# =============================================================================
# 2. VALIDACIÓN
# =============================================================================

encabezado("2. VALIDANDO DATOS DE ENTRADA")

print(f"Geometrías nulas: {gdf.geometry.isna().sum()}")
print(f"Geometrías vacías: {gdf.geometry.is_empty.sum()}")
print(f"Geometrías inválidas: {(~gdf.geometry.is_valid).sum()}")

if "nodo_id" not in gdf.columns:
    raise ValueError(
        "No existe la columna obligatoria nodo_id."
    )

duplicados = gdf["nodo_id"].duplicated().sum()

print(f"IDs duplicados: {duplicados}")

if duplicados > 0:
    raise ValueError(
        f"Se encontraron {duplicados} nodo_id duplicados."
    )


# =============================================================================
# 3. VALIDACIÓN DE COMPONENTES
# =============================================================================

encabezado("3. VALIDANDO COMPONENTES DE PRIORIZACIÓN")

COLUMNAS_REQUERIDAS = [
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
    "indice_demanda_estructural",
    "indice_infraestructura_estructural",
    "indice_intermodalidad_estructural",
    "indice_conectividad_estructural",
    "indice_integracion_territorial",
    "indice_centralidad_estructural",
    "indice_centralidad_estructural_robusto",
    "deficit_infraestructura",
    "tipologia_centralidad",
]

faltantes = []

for columna in COLUMNAS_REQUERIDAS:
    if columna not in gdf.columns:
        faltantes.append(columna)
        print(f"  FALTA {columna}")
    else:
        nulos = gdf[columna].isna().sum()
        print(f"  OK {columna}: {nulos} nulos")

if faltantes:
    raise ValueError(
        "Faltan columnas obligatorias:\n"
        + "\n".join(f"- {x}" for x in faltantes)
    )


# =============================================================================
# 4. NORMALIZACIÓN DE DATOS
# =============================================================================

encabezado("4. NORMALIZANDO VARIABLES DE CARTERA")

variables_numericas = [
    "score_prioridad_territorial",
    "impacto_potencial",
    "urgencia_intervencion",
    "deficit_estructural_promedio",
    "indice_demanda_estructural",
    "indice_infraestructura_estructural",
    "indice_intermodalidad_estructural",
    "indice_conectividad_estructural",
    "indice_integracion_territorial",
    "indice_centralidad_estructural",
    "indice_centralidad_estructural_robusto",
    "deficit_infraestructura",
]

for columna in variables_numericas:
    gdf[columna] = pd.to_numeric(
        gdf[columna],
        errors="coerce",
    )

    if gdf[columna].isna().any():
        gdf[columna] = gdf[columna].fillna(
            gdf[columna].median()
        )

    print(
        f"{columna}: "
        f"{gdf[columna].min():.2f} - "
        f"{gdf[columna].max():.2f}"
    )


# =============================================================================
# 5. CONSTRUCCIÓN DE CATEGORÍAS DE CARTERA
# =============================================================================

encabezado("5. CONSTRUYENDO CATEGORÍAS DE PROYECTOS")

gdf["tipo_proyecto"] = gdf.apply(
    clasificar_proyecto,
    axis=1,
)

print()
print(gdf["tipo_proyecto"].value_counts().to_string())


# =============================================================================
# 6. ESCENARIOS
# =============================================================================

encabezado("6. CONSTRUYENDO ESCENARIOS DE INTERVENCIÓN")

gdf["escenario_intervencion"] = gdf.apply(
    clasificar_escenario,
    axis=1,
)

gdf["horizonte_intervencion"] = gdf[
    "escenario_intervencion"
].map(clasificar_horizonte)

gdf["fase_intervencion"] = gdf.apply(
    construir_fase,
    axis=1,
)

print()
print(
    gdf["escenario_intervencion"]
    .value_counts()
    .to_string()
)


# =============================================================================
# 7. PRIORIDAD DE CARTERA
# =============================================================================

encabezado("7. CLASIFICANDO PRIORIDAD DE CARTERA")

gdf["prioridad_cartera"] = gdf.apply(
    determinar_prioridad_cartera,
    axis=1,
)

gdf["nivel_intervencion"] = gdf.apply(
    determinar_nivel_intervencion,
    axis=1,
)

print()
print(
    gdf["prioridad_cartera"]
    .value_counts()
    .to_string()
)


# =============================================================================
# 8. DIMENSIONES PRIORITARIAS
# =============================================================================

encabezado("8. IDENTIFICANDO DIMENSIONES PRIORITARIAS")

gdf["dimensiones_prioritarias_cartera"] = gdf.apply(
    construir_dimension_prioritaria,
    axis=1,
)

print(
    "Dimensiones prioritarias construidas."
)


# =============================================================================
# 9. OBJETIVO DEL PROYECTO
# =============================================================================

encabezado("9. CONSTRUYENDO OBJETIVOS DE PROYECTO")

gdf["objetivo_proyecto"] = gdf.apply(
    construir_objetivo_proyecto,
    axis=1,
)

gdf["justificacion_cartera"] = gdf.apply(
    construir_justificacion,
    axis=1,
)


# =============================================================================
# 10. SCORE DE CARTERA
# =============================================================================

encabezado("10. CALCULANDO SCORE DE CARTERA")

"""
El score de cartera mantiene como base el score territorial del proceso 25,
pero incorpora una pequeña corrección por urgencia e impacto.

Pesos:

    Prioridad territorial: 70%
    Urgencia:              15%
    Impacto potencial:     15%

No reemplaza el score del proceso 25.
Es un score específico para ordenar la cartera.
"""

gdf["score_cartera"] = (
    gdf["score_prioridad_territorial"] * 0.70
    + gdf["urgencia_intervencion"] * 0.15
    + gdf["impacto_potencial"] * 0.15
)

gdf["score_cartera"] = gdf[
    "score_cartera"
].clip(0, 100)

print(
    "Score cartera: "
    f"{gdf['score_cartera'].min():.2f} - "
    f"{gdf['score_cartera'].max():.2f}"
)


# =============================================================================
# 11. RANKING GENERAL
# =============================================================================

encabezado("11. CONSTRUYENDO RANKING GENERAL DE CARTERA")

gdf = gdf.sort_values(
    [
        "score_cartera",
        "score_prioridad_territorial",
        "impacto_potencial",
    ],
    ascending=False,
).reset_index(drop=True)

gdf["ranking_cartera"] = (
    np.arange(len(gdf)) + 1
)

# Ranking dentro del tipo de proyecto
gdf["ranking_tipo_proyecto"] = (
    gdf.groupby("tipo_proyecto")[
        "score_cartera"
    ]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)

# Ranking dentro del escenario
gdf["ranking_escenario"] = (
    gdf.groupby("escenario_intervencion")[
        "score_cartera"
    ]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)


# =============================================================================
# 12. IDENTIFICADOR DE PROYECTO
# =============================================================================

encabezado("12. GENERANDO IDENTIFICADORES DE CARTERA")

gdf["proyecto_id"] = [
    f"AMBA-P{ranking:03d}"
    for ranking in gdf["ranking_cartera"]
]

gdf["centralidad_id"] = gdf["nodo_id"].apply(
    lambda x: f"AMBAC-{int(x):03d}"
)


# =============================================================================
# 13. MATRIZ DE INTERVENCIÓN
# =============================================================================

encabezado("13. CONSTRUYENDO MATRIZ DE INTERVENCIÓN")

gdf["prioridad_demanda"] = np.where(
    gdf["indice_demanda_estructural"] >= 70,
    "ALTA",
    np.where(
        gdf["indice_demanda_estructural"] >= 40,
        "MEDIA",
        "BAJA",
    ),
)

gdf["prioridad_infraestructura"] = np.where(
    gdf["deficit_infraestructura"] >= 60,
    "ALTA",
    np.where(
        gdf["deficit_infraestructura"] >= 30,
        "MEDIA",
        "BAJA",
    ),
)

gdf["prioridad_intermodalidad"] = np.where(
    (100 - gdf["indice_intermodalidad_estructural"]) >= 60,
    "ALTA",
    np.where(
        (100 - gdf["indice_intermodalidad_estructural"]) >= 30,
        "MEDIA",
        "BAJA",
    ),
)

gdf["prioridad_conectividad"] = np.where(
    (100 - gdf["indice_conectividad_estructural"]) >= 60,
    "ALTA",
    np.where(
        (100 - gdf["indice_conectividad_estructural"]) >= 30,
        "MEDIA",
        "BAJA",
    ),
)

gdf["prioridad_integracion"] = np.where(
    (100 - gdf["indice_integracion_territorial"]) >= 60,
    "ALTA",
    np.where(
        (100 - gdf["indice_integracion_territorial"]) >= 30,
        "MEDIA",
        "BAJA",
    ),
)


# =============================================================================
# 14. DIAGNÓSTICO DE CARTERA
# =============================================================================

encabezado("14. CONSTRUYENDO DIAGNÓSTICO DE CARTERA")


def diagnostico_cartera(row: pd.Series) -> str:

    demanda = safe_float(
        row["indice_demanda_estructural"]
    )

    infraestructura = safe_float(
        row["indice_infraestructura_estructural"]
    )

    deficit = safe_float(
        row["deficit_estructural_promedio"]
    )

    impacto = safe_float(
        row["impacto_potencial"]
    )

    if demanda >= 80 and infraestructura < 40:
        return "ALTA_DEMANDA_BAJO_SOPORTE"

    if deficit >= 60 and impacto >= 60:
        return "DEFICIT_ESTRUCTURAL_ALTO"

    if impacto >= 75:
        return "ALTO_IMPACTO_POTENCIAL"

    if deficit >= 40:
        return "DEFICIT_ESTRUCTURAL_MEDIO"

    if infraestructura >= 70 and demanda >= 70:
        return "CENTRALIDAD_CONSOLIDADA"

    return "INTERVENCION_SELECTIVA"


gdf["diagnostico_cartera"] = gdf.apply(
    diagnostico_cartera,
    axis=1,
)

print(
    gdf["diagnostico_cartera"]
    .value_counts()
    .to_string()
)


# =============================================================================
# 15. VALIDACIÓN FINAL
# =============================================================================

encabezado("15. VALIDACIÓN FINAL")

COLUMNAS_FINALES = [
    "proyecto_id",
    "centralidad_id",
    "nodo_id",
    "tipo_proyecto",
    "escenario_intervencion",
    "horizonte_intervencion",
    "fase_intervencion",
    "prioridad_cartera",
    "nivel_intervencion",
    "score_cartera",
    "ranking_cartera",
    "ranking_tipo_proyecto",
    "ranking_escenario",
    "objetivo_proyecto",
    "justificacion_cartera",
    "dimensiones_prioritarias_cartera",
    "diagnostico_cartera",
]

errores = []

for columna in COLUMNAS_FINALES:

    if columna not in gdf.columns:
        errores.append(
            f"Falta columna: {columna}"
        )
        continue

    nulos = gdf[columna].isna().sum()

    print(
        f"{columna}: {nulos} nulos"
    )

    if nulos > 0:
        errores.append(
            f"{columna}: {nulos} nulos"
        )

if gdf["proyecto_id"].duplicated().sum() > 0:
    errores.append(
        "proyecto_id contiene duplicados"
    )

if gdf["ranking_cartera"].duplicated().sum() > 0:
    errores.append(
        "ranking_cartera contiene duplicados"
    )

if errores:
    print()
    print("ERRORES DE VALIDACIÓN:")
    for error in errores:
        print(f"  - {error}")

    raise ValueError(
        "La validación final falló."
    )

print()
print("Validación final: OK")


# =============================================================================
# 16. TOP CARTERA
# =============================================================================

encabezado("16. TOP 30 PROYECTOS DE LA CARTERA")

columnas_top = [
    "ranking_cartera",
    "proyecto_id",
    "nodo_id",
    "tipologia_centralidad",
    "tipo_proyecto",
    "escenario_intervencion",
    "score_cartera",
    "score_prioridad_territorial",
    "indice_demanda_estructural",
    "indice_infraestructura_estructural",
    "deficit_infraestructura",
    "impacto_potencial",
    "urgencia_intervencion",
]

print(
    gdf[columnas_top]
    .head(30)
    .to_string(index=False)
)


# =============================================================================
# 17. TOP POR TIPO DE PROYECTO
# =============================================================================

encabezado("17. TOP 10 POR TIPO DE PROYECTO")

tipos = sorted(
    gdf["tipo_proyecto"].dropna().unique()
)

for tipo in tipos:

    subencabezado(tipo)

    datos = gdf[
        gdf["tipo_proyecto"] == tipo
    ].head(10)

    print(
        datos[
            [
                "proyecto_id",
                "nodo_id",
                "score_cartera",
                "ranking_tipo_proyecto",
                "escenario_intervencion",
                "impacto_potencial",
                "urgencia_intervencion",
            ]
        ].to_string(index=False)
    )


# =============================================================================
# 18. RESUMEN ESTADÍSTICO
# =============================================================================

encabezado("18. RESUMEN DE CARTERA")

print()
print("Por prioridad:")
print(
    gdf["prioridad_cartera"]
    .value_counts()
    .to_string()
)

print()
print("Por escenario:")
print(
    gdf["escenario_intervencion"]
    .value_counts()
    .to_string()
)

print()
print("Por tipo de proyecto:")
print(
    gdf["tipo_proyecto"]
    .value_counts()
    .to_string()
)

print()
print("Por diagnóstico:")
print(
    gdf["diagnostico_cartera"]
    .value_counts()
    .to_string()
)


# =============================================================================
# 19. RESUMEN JSON
# =============================================================================

encabezado("19. CONSTRUYENDO RESUMEN JSON")

resumen = {
    "proceso": "26_construir_cartera_proyectos_amba",
    "version": VERSION,
    "fecha_generacion": pd.Timestamp.now().isoformat(),
    "registros": int(len(gdf)),
    "crs": CRS_GEOGRAFICO,
    "crs_metrico": CRS_METRICO,
    "modelo": {
        "base": "score_prioridad_territorial",
        "peso_prioridad": 0.70,
        "peso_urgencia": 0.15,
        "peso_impacto": 0.15,
    },
    "resumen": {
        "total_proyectos": int(len(gdf)),
        "score_cartera_min": safe_float(
            gdf["score_cartera"].min()
        ),
        "score_cartera_max": safe_float(
            gdf["score_cartera"].max()
        ),
        "score_cartera_promedio": safe_float(
            gdf["score_cartera"].mean()
        ),
    },
    "por_tipo_proyecto": {
        str(k): int(v)
        for k, v in gdf[
            "tipo_proyecto"
        ].value_counts().items()
    },
    "por_escenario": {
        str(k): int(v)
        for k, v in gdf[
            "escenario_intervencion"
        ].value_counts().items()
    },
    "por_prioridad": {
        str(k): int(v)
        for k, v in gdf[
            "prioridad_cartera"
        ].value_counts().items()
    },
    "por_diagnostico": {
        str(k): int(v)
        for k, v in gdf[
            "diagnostico_cartera"
        ].value_counts().items()
    },
    "top_20": [],
}


for _, row in gdf.head(20).iterrows():

    resumen["top_20"].append(
        {
            "ranking_cartera": int(
                row["ranking_cartera"]
            ),
            "proyecto_id": safe_str(
                row["proyecto_id"]
            ),
            "nodo_id": int(
                row["nodo_id"]
            ),
            "tipo_proyecto": safe_str(
                row["tipo_proyecto"]
            ),
            "escenario": safe_str(
                row["escenario_intervencion"]
            ),
            "score_cartera": safe_float(
                row["score_cartera"]
            ),
            "prioridad": safe_str(
                row["prioridad_cartera"]
            ),
            "impacto": safe_float(
                row["impacto_potencial"]
            ),
            "urgencia": safe_float(
                row["urgencia_intervencion"]
            ),
        }
    )


JSON_FILE = (
    OUTPUT_DIR
    / "cartera_proyectos_amba_resumen.json"
)

with open(
    JSON_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        resumen,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"JSON: {JSON_FILE}")


# =============================================================================
# 20. GUARDANDO ARCHIVOS
# =============================================================================

encabezado("20. GUARDANDO ARCHIVOS")

PARQUET_FILE = (
    OUTPUT_DIR
    / "cartera_proyectos_amba.parquet"
)

CSV_FILE = (
    OUTPUT_DIR
    / "cartera_proyectos_amba.csv"
)

GPKG_FILE = (
    OUTPUT_DIR
    / "cartera_proyectos_amba.gpkg"
)


# Aseguramos CRS geográfico para salida
if gdf.crs is None:
    gdf = gdf.set_crs(CRS_GEOGRAFICO)

gdf = gdf.to_crs(CRS_GEOGRAFICO)


# Parquet
gdf.to_parquet(
    PARQUET_FILE,
    index=False,
)

print(f"Parquet:")
print(PARQUET_FILE)


# CSV
df_csv = pd.DataFrame(gdf.drop(columns="geometry"))

df_csv.to_csv(
    CSV_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"CSV:")
print(CSV_FILE)


# GeoPackage
if GPKG_FILE.exists():
    GPKG_FILE.unlink()

gdf.to_file(
    GPKG_FILE,
    layer="cartera_proyectos_amba",
    driver="GPKG",
)

print(f"GeoPackage:")
print(GPKG_FILE)


print(f"JSON:")
print(JSON_FILE)


# =============================================================================
# 21. GENERACIÓN DE MAPAS
# =============================================================================

encabezado("21. GENERANDO MAPAS Y GRÁFICOS")


def guardar_mapa(
    data: gpd.GeoDataFrame,
    columna: str,
    archivo: Path,
    titulo: str,
    cmap: str = "viridis",
    legend: bool = True,
) -> None:

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    data.plot(
        ax=ax,
        column=columna,
        cmap=cmap,
        legend=legend,
        markersize=25,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.25,
    )

    ax.set_title(
        titulo,
        fontsize=15,
        fontweight="bold",
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        archivo,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Mapa: {archivo}")


# 01 - Score cartera
guardar_mapa(
    gdf,
    "score_cartera",
    OUTPUT_DIR / "01_mapa_cartera_proyectos.png",
    "Cartera de Proyectos AMBA - Score de Cartera",
    cmap="viridis",
)


# 02 - Escenario
escenarios = {
    "ESCENARIO_URGENTE": 4,
    "ESCENARIO_CORTO_PLAZO": 3,
    "ESCENARIO_MEDIANO_PLAZO": 2,
    "ESCENARIO_ESTRUCTURAL": 1,
}

gdf["_escenario_num"] = gdf[
    "escenario_intervencion"
].map(escenarios).fillna(0)

guardar_mapa(
    gdf,
    "_escenario_num",
    OUTPUT_DIR / "02_mapa_escenarios.png",
    "Cartera de Proyectos AMBA - Escenarios",
    cmap="plasma",
)


# 03 - Tipo de proyecto
tipos_unicos = sorted(
    gdf["tipo_proyecto"].unique()
)

tipo_codigo = {
    tipo: i + 1
    for i, tipo in enumerate(tipos_unicos)
}

gdf["_tipo_proyecto_num"] = gdf[
    "tipo_proyecto"
].map(tipo_codigo)

guardar_mapa(
    gdf,
    "_tipo_proyecto_num",
    OUTPUT_DIR / "03_mapa_tipo_intervencion.png",
    "Cartera de Proyectos AMBA - Tipo de Intervención",
    cmap="tab20",
)


# 04 - Prioridad
prioridades = {
    "PRIORIDAD_1_MUY_ALTA": 4,
    "PRIORIDAD_2_ALTA": 3,
    "PRIORIDAD_3_MEDIA": 2,
    "PRIORIDAD_4_BAJA": 1,
}

gdf["_prioridad_num"] = gdf[
    "prioridad_cartera"
].map(prioridades).fillna(0)

guardar_mapa(
    gdf,
    "_prioridad_num",
    OUTPUT_DIR / "04_mapa_prioridad.png",
    "Cartera de Proyectos AMBA - Prioridad",
    cmap="RdYlGn",
)


# 05 - Déficit estructural
guardar_mapa(
    gdf,
    "deficit_estructural_promedio",
    OUTPUT_DIR / "05_mapa_deficit_estructural.png",
    "Cartera de Proyectos AMBA - Déficit Estructural",
    cmap="magma",
)


# =============================================================================
# 22. GRÁFICOS
# =============================================================================

subencabezado("GRÁFICOS DE CARTERA")


# -------------------------------------------------------------------------
# 06 - Demanda vs déficit
# -------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.scatter(
    gdf["indice_demanda_estructural"],
    gdf["deficit_estructural_promedio"],
    s=45,
    alpha=0.75,
)

ax.set_xlabel(
    "Índice de demanda estructural"
)

ax.set_ylabel(
    "Déficit estructural promedio"
)

ax.set_title(
    "Demanda vs Déficit Estructural",
    fontsize=15,
    fontweight="bold",
)

ax.grid(
    True,
    alpha=0.25,
)

plt.tight_layout()

archivo = (
    OUTPUT_DIR
    / "06_demanda_vs_deficit.png"
)

plt.savefig(
    archivo,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Gráfico: {archivo}")


# -------------------------------------------------------------------------
# 07 - Proyectos por intervención
# -------------------------------------------------------------------------

conteo = (
    gdf["tipo_proyecto"]
    .value_counts()
    .sort_values()
)

fig, ax = plt.subplots(
    figsize=(11, 8)
)

conteo.plot(
    kind="barh",
    ax=ax,
)

ax.set_title(
    "Cartera por Tipo de Proyecto",
    fontsize=15,
    fontweight="bold",
)

ax.set_xlabel(
    "Cantidad de centralidades"
)

ax.set_ylabel(
    "Tipo de proyecto"
)

plt.tight_layout()

archivo = (
    OUTPUT_DIR
    / "07_cartera_por_intervencion.png"
)

plt.savefig(
    archivo,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Gráfico: {archivo}")


# -------------------------------------------------------------------------
# 08 - Escenarios
# -------------------------------------------------------------------------

conteo = (
    gdf["escenario_intervencion"]
    .value_counts()
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

conteo.plot(
    kind="bar",
    ax=ax,
)

ax.set_title(
    "Cartera por Escenario de Intervención",
    fontsize=15,
    fontweight="bold",
)

ax.set_xlabel(
    "Escenario"
)

ax.set_ylabel(
    "Cantidad de centralidades"
)

ax.tick_params(
    axis="x",
    rotation=35,
)

plt.tight_layout()

archivo = (
    OUTPUT_DIR
    / "08_cartera_por_escenario.png"
)

plt.savefig(
    archivo,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Gráfico: {archivo}")


# -------------------------------------------------------------------------
# 09 - Distribución score
# -------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.hist(
    gdf["score_cartera"],
    bins=15,
    alpha=0.8,
)

ax.set_title(
    "Distribución del Score de Cartera",
    fontsize=15,
    fontweight="bold",
)

ax.set_xlabel(
    "Score de cartera"
)

ax.set_ylabel(
    "Cantidad de centralidades"
)

ax.grid(
    True,
    alpha=0.20,
)

plt.tight_layout()

archivo = (
    OUTPUT_DIR
    / "09_distribucion_score_cartera.png"
)

plt.savefig(
    archivo,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Gráfico: {archivo}")


# =============================================================================
# 23. LIMPIEZA DE COLUMNAS AUXILIARES
# =============================================================================

gdf = gdf.drop(
    columns=[
        "_escenario_num",
        "_tipo_proyecto_num",
        "_prioridad_num",
    ],
    errors="ignore",
)


# =============================================================================
# 24. REESCRIBIR ARCHIVOS CON DATASET FINAL LIMPIO
# =============================================================================

encabezado("24. CONSOLIDANDO DATASET FINAL")

gdf.to_parquet(
    PARQUET_FILE,
    index=False,
)

df_csv = pd.DataFrame(
    gdf.drop(columns="geometry")
)

df_csv.to_csv(
    CSV_FILE,
    index=False,
    encoding="utf-8-sig",
)

if GPKG_FILE.exists():
    GPKG_FILE.unlink()

gdf.to_file(
    GPKG_FILE,
    layer="cartera_proyectos_amba",
    driver="GPKG",
)

print("Dataset final consolidado.")


# =============================================================================
# 25. RESULTADO FINAL
# =============================================================================

encabezado(
    "26 - PROCESO FINALIZADO"
)

print(
    f"Centralidades analizadas: {len(gdf)}"
)

print()
print("PROYECTOS DE CARTERA:")

for tipo, cantidad in (
    gdf["tipo_proyecto"]
    .value_counts()
    .items()
):
    print(
        f"  {tipo}: {cantidad}"
    )

print()
print("ESCENARIOS:")

for escenario, cantidad in (
    gdf["escenario_intervencion"]
    .value_counts()
    .items()
):
    print(
        f"  {escenario}: {cantidad}"
    )

print()
print("PRIORIDADES:")

for prioridad, cantidad in (
    gdf["prioridad_cartera"]
    .value_counts()
    .items()
):
    print(
        f"  {prioridad}: {cantidad}"
    )


print()
print("ARCHIVOS GENERADOS")

archivos = [
    "01_mapa_cartera_proyectos.png",
    "02_mapa_escenarios.png",
    "03_mapa_tipo_intervencion.png",
    "04_mapa_prioridad.png",
    "05_mapa_deficit_estructural.png",
    "06_demanda_vs_deficit.png",
    "07_cartera_por_intervencion.png",
    "08_cartera_por_escenario.png",
    "09_distribucion_score_cartera.png",
    "cartera_proyectos_amba.csv",
    "cartera_proyectos_amba.gpkg",
    "cartera_proyectos_amba.parquet",
    "cartera_proyectos_amba_resumen.json",
]

for archivo in archivos:
    print(f"  {archivo}")


print()
print("SIGUIENTE ETAPA")
print(
    "Construir los escenarios territoriales de intervención "
    "y evaluar la cartera mediante agregación espacial, "
    "impacto potencial y cobertura metropolitana."
)

print()