# -*- coding: utf-8 -*-

"""
27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA

Construye escenarios territoriales de intervención a partir de la cartera
de proyectos generada por el proceso 26.

Entrada:
    data/processed/cartera_proyectos_amba/
        cartera_proyectos_amba.parquet

Salida:
    data/processed/escenarios_territoriales_amba/

Objetivos:
    - Agrupar proyectos territorialmente.
    - Identificar concentraciones espaciales de intervención.
    - Construir escenarios territoriales.
    - Evaluar cobertura metropolitana.
    - Evaluar impacto potencial.
    - Evaluar déficit atendido.
    - Medir complementariedad de intervenciones.
    - Construir ranking de escenarios.
    - Generar productos GIS y gráficos.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V1"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "cartera_proyectos_amba"
)

INPUT_FILE = INPUT_DIR / "cartera_proyectos_amba.parquet"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CRS_WGS84 = "EPSG:4326"
CRS_METRICO = "EPSG:22185"


# Distancia máxima para considerar que dos proyectos pertenecen
# al mismo sistema territorial.
DISTANCIA_AGRUPACION_M = 5000


# Pesos del score territorial
PESOS = {
    "impacto": 0.25,
    "cobertura": 0.20,
    "deficit": 0.20,
    "demanda": 0.15,
    "complementariedad": 0.10,
    "urgencia": 0.10,
}


# =============================================================================
# UTILIDADES
# =============================================================================

def encabezado(titulo):
    print()
    print("=" * 78)
    print(titulo)
    print("=" * 78)


def normalizar_serie(serie):
    """
    Normalización min-max 0-100.
    """
    serie = pd.to_numeric(serie, errors="coerce")

    minimo = serie.min()
    maximo = serie.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(0.0, index=serie.index)

    if maximo == minimo:
        return pd.Series(100.0, index=serie.index)

    return ((serie - minimo) / (maximo - minimo)) * 100.0


def safe_numeric(df, column, default=0.0):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def porcentaje(valor):
    return round(float(valor), 2)


# =============================================================================
# 1. CARGA
# =============================================================================

encabezado("27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - V1")

print(f"Proyecto : {BASE_DIR}")
print(f"Entrada  : {INPUT_FILE}")
print(f"Salida   : {OUTPUT_DIR}")
print(f"CRS      : {CRS_WGS84}")
print(f"CRS métrico: {CRS_METRICO}")

print()
print("PESOS DEL MODELO")
print(f"  Impacto potencial:        {PESOS['impacto']:.0%}")
print(f"  Cobertura territorial:    {PESOS['cobertura']:.0%}")
print(f"  Déficit atendido:         {PESOS['deficit']:.0%}")
print(f"  Demanda cubierta:         {PESOS['demanda']:.0%}")
print(f"  Complementariedad:        {PESOS['complementariedad']:.0%}")
print(f"  Urgencia:                 {PESOS['urgencia']:.0%}")


# =============================================================================
# 2. VALIDAR EXISTENCIA
# =============================================================================

encabezado("1. CARGANDO RESULTADOS DEL PROCESO 26")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"No existe el archivo de entrada:\n{INPUT_FILE}"
    )

gdf = gpd.read_parquet(INPUT_FILE)

print(f"Archivo:\n{INPUT_FILE}")
print(f"Registros: {len(gdf)}")
print(f"Columnas: {len(gdf.columns)}")
print(f"CRS: {gdf.crs}")


# =============================================================================
# 3. VALIDACIÓN GEOMÉTRICA
# =============================================================================

encabezado("2. VALIDANDO DATOS DE ENTRADA")

if gdf.empty:
    raise ValueError("El dataset de entrada está vacío.")

if gdf.crs is None:
    print("CRS inexistente. Se asigna EPSG:4326.")
    gdf = gdf.set_crs(CRS_WGS84)

null_geom = int(gdf.geometry.isna().sum())
empty_geom = int(gdf.geometry.is_empty.sum())
invalid_geom = int((~gdf.geometry.is_valid).sum())

print(f"Geometrías nulas: {null_geom}")
print(f"Geometrías vacías: {empty_geom}")
print(f"Geometrías inválidas: {invalid_geom}")

if null_geom > 0 or empty_geom > 0:
    raise ValueError("Existen geometrías nulas o vacías.")

if invalid_geom > 0:
    print("Corrigiendo geometrías inválidas...")
    gdf["geometry"] = gdf.geometry.make_valid()

    invalid_after = int((~gdf.geometry.is_valid).sum())

    if invalid_after > 0:
        raise ValueError(
            "Persisten geometrías inválidas luego de make_valid()."
        )

if "proyecto_id" in gdf.columns:
    duplicados = int(gdf["proyecto_id"].duplicated().sum())
    print(f"Proyectos duplicados: {duplicados}")

    if duplicados > 0:
        raise ValueError("Existen proyecto_id duplicados.")

print("Validación geométrica: OK")


# =============================================================================
# 4. VALIDAR COLUMNAS
# =============================================================================

encabezado("3. VALIDANDO COMPONENTES DE CARTERA")

COLUMNAS_REQUERIDAS = [
    "proyecto_id",
    "centralidad_id",
    "nodo_id",
    "tipo_proyecto",
    "escenario_intervencion",
    "prioridad_cartera",
    "score_cartera",
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
    "deficit_infraestructura",
]

faltantes = [
    c for c in COLUMNAS_REQUERIDAS
    if c not in gdf.columns
]

if faltantes:
    raise ValueError(
        "Faltan columnas requeridas:\n"
        + "\n".join(f"  - {c}" for c in faltantes)
    )

for col in COLUMNAS_REQUERIDAS:
    nulos = int(gdf[col].isna().sum())

    if nulos:
        print(f"  ERROR {col}: {nulos} nulos")
        raise ValueError(
            f"La columna {col} contiene valores nulos."
        )

    print(f"  OK {col}: 0 nulos")

print()
print(f"Proyectos validados: {len(gdf)}")


# =============================================================================
# 5. PREPARACIÓN ESPACIAL
# =============================================================================

encabezado("4. PREPARANDO INFORMACIÓN ESPACIAL")

if gdf.crs.to_string() != CRS_WGS84:
    gdf_wgs84 = gdf.to_crs(CRS_WGS84)
else:
    gdf_wgs84 = gdf.copy()

gdf_metric = gdf.to_crs(CRS_METRICO)

# Centroides para clustering espacial
gdf_metric["centroide"] = gdf_metric.geometry.centroid

centroides = gpd.GeoDataFrame(
    gdf_metric.drop(columns="geometry"),
    geometry="centroide",
    crs=CRS_METRICO,
)

print(f"Proyectos espaciales: {len(centroides)}")
print(f"Distancia de agrupación: {DISTANCIA_AGRUPACION_M:,} m")


# =============================================================================
# 6. VARIABLES BASE
# =============================================================================

encabezado("5. CONSTRUYENDO VARIABLES TERRITORIALES")

df = gdf_metric.copy()

df["valor_impacto"] = safe_numeric(
    df,
    "impacto_potencial"
)

df["valor_urgencia"] = safe_numeric(
    df,
    "urgencia_intervencion"
)

df["valor_deficit"] = safe_numeric(
    df,
    "deficit_estructural_promedio"
)

df["valor_demanda"] = safe_numeric(
    df,
    "indice_demanda_estructural"
)

df["valor_infraestructura"] = safe_numeric(
    df,
    "indice_infraestructura_estructural"
)

df["valor_intermodalidad"] = safe_numeric(
    df,
    "indice_intermodalidad_estructural"
)

df["valor_conectividad"] = safe_numeric(
    df,
    "indice_conectividad_estructural"
)

df["valor_integracion"] = safe_numeric(
    df,
    "indice_integracion_territorial"
)

df["valor_centralidad"] = safe_numeric(
    df,
    "indice_centralidad_estructural"
)

df["valor_prioridad"] = safe_numeric(
    df,
    "score_prioridad_territorial"
)

print("Variables territoriales construidas.")


# =============================================================================
# 7. AGRUPAMIENTO ESPACIAL
# =============================================================================

encabezado("6. CONSTRUYENDO AGRUPAMIENTOS TERRITORIALES")

"""
Se utiliza un clustering espacial simple basado en componentes conexas:

Dos proyectos pertenecen al mismo agrupamiento si la distancia entre
sus centroides es <= DISTANCIA_AGRUPACION_M.

Esto evita depender de scikit-learn y mantiene el proceso reproducible.
"""

coords = np.array([
    [geom.x, geom.y]
    for geom in centroides.geometry
])

n = len(coords)

parent = list(range(n))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra = find(a)
    rb = find(b)

    if ra != rb:
        parent[rb] = ra


for i in range(n):
    for j in range(i + 1, n):

        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]

        distancia = np.sqrt(
            dx * dx + dy * dy
        )

        if distancia <= DISTANCIA_AGRUPACION_M:
            union(i, j)


roots = [find(i) for i in range(n)]

root_to_cluster = {}

cluster_ids = []

for root in roots:

    if root not in root_to_cluster:
        root_to_cluster[root] = len(root_to_cluster) + 1

    cluster_ids.append(
        root_to_cluster[root]
    )

df["cluster_territorial"] = cluster_ids

cantidad_clusters = df["cluster_territorial"].nunique()

print(f"Escenarios territoriales espaciales: {cantidad_clusters}")

distribucion_clusters = (
    df["cluster_territorial"]
    .value_counts()
    .sort_index()
)

print()
print("Proyectos por agrupamiento:")
print(distribucion_clusters.to_string())


# =============================================================================
# 8. CARACTERIZACIÓN DE AGRUPAMIENTOS
# =============================================================================

encabezado("7. CARACTERIZANDO ESCENARIOS TERRITORIALES")

escenarios = []

for cluster_id, grupo in df.groupby(
    "cluster_territorial"
):

    cantidad_proyectos = len(grupo)

    tipos = grupo["tipo_proyecto"].astype(str).unique()

    escenarios.append({
        "cluster_territorial": cluster_id,
        "cantidad_proyectos": cantidad_proyectos,
        "cantidad_tipos_proyecto": len(tipos),
        "tipos_proyecto": "|".join(sorted(tipos)),
        "score_cartera_promedio": grupo["score_cartera"].mean(),
        "score_prioridad_promedio": grupo[
            "score_prioridad_territorial"
        ].mean(),
        "impacto_potencial_promedio": grupo[
            "valor_impacto"
        ].mean(),
        "urgencia_promedio": grupo[
            "valor_urgencia"
        ].mean(),
        "deficit_promedio": grupo[
            "valor_deficit"
        ].mean(),
        "demanda_promedio": grupo[
            "valor_demanda"
        ].mean(),
        "infraestructura_promedio": grupo[
            "valor_infraestructura"
        ].mean(),
        "intermodalidad_promedio": grupo[
            "valor_intermodalidad"
        ].mean(),
        "conectividad_promedio": grupo[
            "valor_conectividad"
        ].mean(),
        "integracion_promedio": grupo[
            "valor_integracion"
        ].mean(),
        "centralidad_promedio": grupo[
            "valor_centralidad"
        ].mean(),
        "demanda_total": grupo[
            "valor_demanda"
        ].sum(),
    })

escenarios_df = pd.DataFrame(escenarios)

print(
    f"Escenarios caracterizados: "
    f"{len(escenarios_df)}"
)


# =============================================================================
# 9. COBERTURA TERRITORIAL
# =============================================================================

encabezado("8. CALCULANDO COBERTURA TERRITORIAL")

max_proyectos = max(
    escenarios_df["cantidad_proyectos"].max(),
    1
)

max_demanda = max(
    escenarios_df["demanda_total"].max(),
    1
)

escenarios_df["cobertura_proyectos"] = (
    escenarios_df["cantidad_proyectos"]
    / max_proyectos
) * 100

escenarios_df["cobertura_demanda"] = (
    escenarios_df["demanda_total"]
    / max_demanda
) * 100

escenarios_df["cobertura_territorial"] = (
    escenarios_df["cobertura_proyectos"] * 0.50
    + escenarios_df["cobertura_demanda"] * 0.50
)

print(
    "Cobertura territorial: "
    f"{escenarios_df['cobertura_territorial'].min():.2f} - "
    f"{escenarios_df['cobertura_territorial'].max():.2f}"
)


# =============================================================================
# 10. DÉFICIT ATENDIDO
# =============================================================================

encabezado("9. CALCULANDO DÉFICIT TERRITORIAL ATENDIDO")

escenarios_df["deficit_atendido"] = (
    escenarios_df["deficit_promedio"]
)

escenarios_df["deficit_atendido_normalizado"] = normalizar_serie(
    escenarios_df["deficit_atendido"]
)

print(
    "Déficit atendido: "
    f"{escenarios_df['deficit_atendido'].min():.2f} - "
    f"{escenarios_df['deficit_atendido'].max():.2f}"
)


# =============================================================================
# 11. COMPLEMENTARIEDAD
# =============================================================================

encabezado("10. CALCULANDO COMPLEMENTARIEDAD DE INTERVENCIONES")

TIPOS_COMPLEMENTARIOS = {
    frozenset([
        "AMPLIACION_INFRAESTRUCTURA",
        "MEJORA_INTERMODAL",
    ]),
    frozenset([
        "AMPLIACION_INFRAESTRUCTURA",
        "MEJORA_CONECTIVIDAD",
    ]),
    frozenset([
        "MEJORA_CONECTIVIDAD",
        "INTEGRACION_TERRITORIAL",
    ]),
    frozenset([
        "MEJORA_INTERMODAL",
        "INTEGRACION_TERRITORIAL",
    ]),
    frozenset([
        "PROYECTO_INTEGRAL_CENTRALIDAD",
        "AMPLIACION_INFRAESTRUCTURA",
    ]),
    frozenset([
        "PROYECTO_INTEGRAL_CENTRALIDAD",
        "MEJORA_INTERMODAL",
    ]),
}

complementariedades = []

for cluster_id, grupo in df.groupby(
    "cluster_territorial"
):

    tipos = set(
        grupo["tipo_proyecto"]
        .astype(str)
        .tolist()
    )

    pares = set()

    tipos_lista = sorted(tipos)

    for i in range(len(tipos_lista)):
        for j in range(i + 1, len(tipos_lista)):

            par = frozenset([
                tipos_lista[i],
                tipos_lista[j]
            ])

            if par in TIPOS_COMPLEMENTARIOS:
                pares.add(par)

    cantidad_tipos = len(tipos)

    diversidad = min(
        cantidad_tipos / 5,
        1
    ) * 100

    sinergia = min(
        len(pares) / 3,
        1
    ) * 100

    complementariedad = (
        diversidad * 0.60
        + sinergia * 0.40
    )

    complementariedades.append({
        "cluster_territorial": cluster_id,
        "diversidad_intervenciones": diversidad,
        "sinergias_intervencion": sinergia,
        "complementariedad": complementariedad,
    })

comp_df = pd.DataFrame(complementariedades)

escenarios_df = escenarios_df.merge(
    comp_df,
    on="cluster_territorial",
    how="left"
)

print(
    "Complementariedad: "
    f"{escenarios_df['complementariedad'].min():.2f} - "
    f"{escenarios_df['complementariedad'].max():.2f}"
)


# =============================================================================
# 12. IMPACTO TERRITORIAL
# =============================================================================

encabezado("11. CALCULANDO IMPACTO TERRITORIAL")

escenarios_df["impacto_territorial"] = (
    escenarios_df["impacto_potencial_promedio"]
)

escenarios_df["impacto_territorial_normalizado"] = normalizar_serie(
    escenarios_df["impacto_territorial"]
)

print(
    "Impacto territorial: "
    f"{escenarios_df['impacto_territorial'].min():.2f} - "
    f"{escenarios_df['impacto_territorial'].max():.2f}"
)


# =============================================================================
# 13. DEMANDA CUBIERTA
# =============================================================================

encabezado("12. CALCULANDO DEMANDA CUBIERTA")

escenarios_df["demanda_cubierta"] = (
    escenarios_df["demanda_promedio"]
)

escenarios_df["demanda_cubierta_normalizada"] = normalizar_serie(
    escenarios_df["demanda_cubierta"]
)

print(
    "Demanda cubierta: "
    f"{escenarios_df['demanda_cubierta'].min():.2f} - "
    f"{escenarios_df['demanda_cubierta'].max():.2f}"
)


# =============================================================================
# 14. SCORE DE ESCENARIO
# =============================================================================

encabezado("13. CONSTRUYENDO SCORE DE ESCENARIO")

escenarios_df["score_escenario"] = (
    escenarios_df["impacto_territorial_normalizado"]
    * PESOS["impacto"]
    +
    escenarios_df["cobertura_territorial"]
    * PESOS["cobertura"]
    +
    escenarios_df["deficit_atendido_normalizado"]
    * PESOS["deficit"]
    +
    escenarios_df["demanda_cubierta_normalizada"]
    * PESOS["demanda"]
    +
    escenarios_df["complementariedad"]
    * PESOS["complementariedad"]
    +
    escenarios_df["urgencia_promedio"]
    * PESOS["urgencia"]
)

print(
    "Score escenario: "
    f"{escenarios_df['score_escenario'].min():.2f} - "
    f"{escenarios_df['score_escenario'].max():.2f}"
)


# =============================================================================
# 15. CLASIFICACIÓN DE PRIORIDAD
# =============================================================================

encabezado("14. CLASIFICANDO PRIORIDAD DE ESCENARIOS")

def clasificar_prioridad(score):

    if score >= 70:
        return "PRIORIDAD_1_MUY_ALTA"

    if score >= 60:
        return "PRIORIDAD_2_ALTA"

    if score >= 50:
        return "PRIORIDAD_3_MEDIA"

    return "PRIORIDAD_4_BAJA"


escenarios_df["prioridad_escenario"] = (
    escenarios_df["score_escenario"]
    .apply(clasificar_prioridad)
)

print(
    escenarios_df[
        "prioridad_escenario"
    ].value_counts()
)


# =============================================================================
# 16. HORIZONTE
# =============================================================================

encabezado("15. DETERMINANDO HORIZONTE DE INTERVENCIÓN")

def horizonte_escenario(row):

    urgencia = row["urgencia_promedio"]
    score = row["score_escenario"]

    if urgencia >= 75:
        return "INMEDIATO"

    if score >= 70:
        return "CORTO_PLAZO"

    if score >= 55:
        return "MEDIANO_PLAZO"

    return "LARGO_PLAZO"


escenarios_df["horizonte_escenario"] = (
    escenarios_df.apply(
        horizonte_escenario,
        axis=1
    )
)

print(
    escenarios_df[
        "horizonte_escenario"
    ].value_counts()
)


# =============================================================================
# 17. TIPO DE ESCENARIO
# =============================================================================

encabezado("16. CLASIFICANDO TIPO DE ESCENARIO")

def tipo_escenario(row):

    tipos = str(row["tipos_proyecto"])

    if row["cantidad_proyectos"] >= 8:
        return "ESCENARIO_METROPOLITANO"

    if "PROYECTO_INTEGRAL_CENTRALIDAD" in tipos:
        return "ESCENARIO_CENTRALIDAD"

    if (
        "AMPLIACION_INFRAESTRUCTURA" in tipos
        and "MEJORA_INTERMODAL" in tipos
    ):
        return "ESCENARIO_INTERMODAL"

    if (
        "MEJORA_CONECTIVIDAD" in tipos
        and "INTEGRACION_TERRITORIAL" in tipos
    ):
        return "ESCENARIO_CONECTIVIDAD_TERRITORIAL"

    if row["deficit_promedio"] >= 65:
        return "ESCENARIO_DEFICIT"

    return "ESCENARIO_SELECTIVO"


escenarios_df["tipo_escenario"] = (
    escenarios_df.apply(
        tipo_escenario,
        axis=1
    )
)

print(
    escenarios_df[
        "tipo_escenario"
    ].value_counts()
)


# =============================================================================
# 18. RANKING
# =============================================================================

encabezado("17. CONSTRUYENDO RANKING DE ESCENARIOS")

escenarios_df = escenarios_df.sort_values(
    [
        "score_escenario",
        "impacto_territorial",
        "cobertura_territorial",
    ],
    ascending=False
).reset_index(drop=True)

escenarios_df["ranking_escenario"] = (
    np.arange(len(escenarios_df)) + 1
)

print(
    escenarios_df[
        [
            "cluster_territorial",
            "ranking_escenario",
            "score_escenario",
            "prioridad_escenario",
            "tipo_escenario",
            "horizonte_escenario",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# 19. IDENTIFICADORES
# =============================================================================

encabezado("18. GENERANDO IDENTIFICADORES")

escenarios_df["escenario_id"] = [
    f"AMBA-E{i:03d}"
    for i in range(1, len(escenarios_df) + 1)
]

escenarios_df["escenario_nombre"] = (
    "Escenario Territorial AMBA "
    + escenarios_df["ranking_escenario"].astype(str)
)

print(
    f"Escenarios identificados: {len(escenarios_df)}"
)


# =============================================================================
# 20. ASIGNAR ESCENARIO A PROYECTOS
# =============================================================================

encabezado("19. ASIGNANDO ESCENARIOS A PROYECTOS")

mapping = escenarios_df[
    [
        "cluster_territorial",
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "horizonte_escenario",
    ]
].copy()

df = df.merge(
    mapping,
    on="cluster_territorial",
    how="left",
)

print(
    "Proyectos asignados: "
    f"{df['escenario_id'].notna().sum()} / {len(df)}"
)


# =============================================================================
# 21. DIAGNÓSTICO
# =============================================================================

encabezado("20. CONSTRUYENDO DIAGNÓSTICO TERRITORIAL")

def diagnostico(row):

    if row["deficit_promedio"] >= 70:
        return "DEFICIT_ESTRUCTURAL_CRITICO"

    if (
        row["demanda_promedio"] >= 75
        and row["infraestructura_promedio"] <= 40
    ):
        return "ALTA_DEMANDA_BAJO_SOPORTE"

    if row["impacto_territorial"] >= 75:
        return "ALTO_IMPACTO_POTENCIAL"

    if row["complementariedad"] >= 70:
        return "ALTA_COMPLEMENTARIEDAD"

    if row["urgencia_promedio"] >= 70:
        return "ALTA_URGENCIA"

    return "INTERVENCION_TERRITORIAL_MEDIA"


escenarios_df["diagnostico_escenario"] = (
    escenarios_df.apply(
        diagnostico,
        axis=1
    )
)

print(
    escenarios_df[
        "diagnostico_escenario"
    ].value_counts()
)


# =============================================================================
# 22. OBJETIVO
# =============================================================================

encabezado("21. CONSTRUYENDO OBJETIVOS DE ESCENARIO")

def objetivo(row):

    diag = row["diagnostico_escenario"]

    if diag == "DEFICIT_ESTRUCTURAL_CRITICO":
        return (
            "Reducir déficits estructurales concentrados mediante "
            "una intervención territorial integrada."
        )

    if diag == "ALTA_DEMANDA_BAJO_SOPORTE":
        return (
            "Incrementar la capacidad de infraestructura y soporte "
            "en áreas con elevada demanda de movilidad."
        )

    if diag == "ALTO_IMPACTO_POTENCIAL":
        return (
            "Capturar el elevado impacto potencial mediante "
            "intervenciones coordinadas sobre centralidades estratégicas."
        )

    if diag == "ALTA_COMPLEMENTARIEDAD":
        return (
            "Integrar proyectos complementarios para generar "
            "sinergias territoriales y metropolitanas."
        )

    if diag == "ALTA_URGENCIA":
        return (
            "Atender déficits y restricciones territoriales "
            "con elevada urgencia de intervención."
        )

    return (
        "Mejorar progresivamente las condiciones territoriales "
        "de movilidad mediante intervenciones coordinadas."
    )


escenarios_df["objetivo_escenario"] = (
    escenarios_df.apply(
        objetivo,
        axis=1
    )
)


# =============================================================================
# 23. JUSTIFICACIÓN
# =============================================================================

encabezado("22. CONSTRUYENDO JUSTIFICACIÓN")

def justificacion(row):

    return (
        f"Escenario {row['escenario_id']} compuesto por "
        f"{int(row['cantidad_proyectos'])} proyectos. "
        f"Presenta un score territorial de "
        f"{row['score_escenario']:.2f}, "
        f"impacto potencial de "
        f"{row['impacto_territorial']:.2f}, "
        f"cobertura territorial de "
        f"{row['cobertura_territorial']:.2f} y "
        f"déficit atendido de "
        f"{row['deficit_atendido']:.2f}."
    )


escenarios_df["justificacion_escenario"] = (
    escenarios_df.apply(
        justificacion,
        axis=1
    )
)


# =============================================================================
# 24. DIMENSIONES PRIORITARIAS
# =============================================================================

encabezado("23. IDENTIFICANDO DIMENSIONES PRIORITARIAS")

DIMENSIONES = {
    "infraestructura": "infraestructura_promedio",
    "intermodalidad": "intermodalidad_promedio",
    "conectividad": "conectividad_promedio",
    "integracion": "integracion_promedio",
    "demanda": "demanda_promedio",
    "centralidad": "centralidad_promedio",
}

def dimensiones_prioritarias(row):

    valores = {
        nombre: row[col]
        for nombre, col in DIMENSIONES.items()
    }

    ordenadas = sorted(
        valores.items(),
        key=lambda x: x[1]
    )

    # En este contexto, los valores bajos representan mayor necesidad.
    prioritarias = [
        nombre
        for nombre, valor in ordenadas[:3]
    ]

    return "|".join(prioritarias)


escenarios_df["dimensiones_prioritarias"] = (
    escenarios_df.apply(
        dimensiones_prioritarias,
        axis=1
    )
)


# =============================================================================
# 25. GEOMETRÍA DE ESCENARIOS
# =============================================================================

encabezado("24. CONSTRUYENDO GEOMETRÍAS DE ESCENARIOS")

"""
La geometría territorial del escenario es la envolvente convexa de los
proyectos que lo integran.

Para escenarios de un solo proyecto se conserva la geometría original.
"""

geometrias = []

for cluster_id, grupo in df.groupby(
    "cluster_territorial"
):

    geometria = unary_union(
        grupo.geometry.tolist()
    )

    if geometria.is_empty:
        geometria = grupo.geometry.iloc[0]

    if geometria.geom_type in [
        "Polygon",
        "MultiPolygon",
        "LineString",
        "MultiLineString",
        "Point",
        "MultiPoint",
    ]:
        pass
    else:
        geometria = grupo.geometry.iloc[0]

    geometrias.append({
        "cluster_territorial": cluster_id,
        "geometry": geometria,
    })

escenarios_geo = gpd.GeoDataFrame(
    geometrias,
    geometry="geometry",
    crs=CRS_METRICO,
)

escenarios_geo = escenarios_geo.merge(
    escenarios_df,
    on="cluster_territorial",
    how="left",
)

print(
    f"Geometrías de escenarios: "
    f"{len(escenarios_geo)}"
)


# =============================================================================
# 26. VALIDACIÓN FINAL
# =============================================================================

encabezado("25. VALIDACIÓN FINAL")

COLUMNAS_FINALES = [
    "escenario_id",
    "escenario_nombre",
    "ranking_escenario",
    "score_escenario",
    "prioridad_escenario",
    "tipo_escenario",
    "horizonte_escenario",
    "cantidad_proyectos",
    "impacto_territorial",
    "cobertura_territorial",
    "deficit_atendido",
    "demanda_cubierta",
    "complementariedad",
    "diagnostico_escenario",
    "objetivo_escenario",
    "justificacion_escenario",
    "dimensiones_prioritarias",
]

for col in COLUMNAS_FINALES:

    nulos = int(
        escenarios_df[col].isna().sum()
    )

    print(
        f"{col}: {nulos} nulos"
    )

    if nulos > 0:
        raise ValueError(
            f"La columna {col} contiene nulos."
        )

# Verificación de ranking
if (
    escenarios_df["ranking_escenario"].duplicated().any()
):
    raise ValueError(
        "Ranking de escenarios duplicado."
    )

# Verificación de IDs
if (
    escenarios_df["escenario_id"].duplicated().any()
):
    raise ValueError(
        "escenario_id duplicado."
    )

# Verificación score
if (
    (escenarios_df["score_escenario"] < 0).any()
    or
    (escenarios_df["score_escenario"] > 100).any()
):
    raise ValueError(
        "Score de escenario fuera del rango 0-100."
    )

print()
print("Validación final: OK")


# =============================================================================
# 27. TOP ESCENARIOS
# =============================================================================

encabezado("26. TOP 20 ESCENARIOS TERRITORIALES")

columnas_top = [
    "ranking_escenario",
    "escenario_id",
    "cantidad_proyectos",
    "tipo_escenario",
    "horizonte_escenario",
    "score_escenario",
    "prioridad_escenario",
    "impacto_territorial",
    "cobertura_territorial",
    "deficit_atendido",
    "demanda_cubierta",
    "complementariedad",
]

print(
    escenarios_df[
        columnas_top
    ]
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# 28. RESUMEN POR PRIORIDAD
# =============================================================================

encabezado("27. RESUMEN DE ESCENARIOS")

print()
print("Por prioridad:")
print(
    escenarios_df[
        "prioridad_escenario"
    ].value_counts()
)

print()
print("Por horizonte:")
print(
    escenarios_df[
        "horizonte_escenario"
    ].value_counts()
)

print()
print("Por tipo:")
print(
    escenarios_df[
        "tipo_escenario"
    ].value_counts()
)

print()
print("Por diagnóstico:")
print(
    escenarios_df[
        "diagnostico_escenario"
    ].value_counts()
)


# =============================================================================
# 29. CONSOLIDADO DE PROYECTOS
# =============================================================================

encabezado("28. CONSOLIDANDO PROYECTOS CON ESCENARIOS")

proyectos_final = gpd.GeoDataFrame(
    df,
    geometry="geometry",
    crs=CRS_METRICO,
)

proyectos_final = proyectos_final.to_crs(
    CRS_WGS84
)

print(
    f"Proyectos consolidados: "
    f"{len(proyectos_final)}"
)


# =============================================================================
# 30. GUARDAR ESCENARIOS
# =============================================================================

encabezado("29. GUARDANDO ESCENARIOS")

escenarios_geo_wgs84 = escenarios_geo.to_crs(
    CRS_WGS84
)

escenarios_parquet = (
    OUTPUT_DIR
    / "escenarios_territoriales_amba.parquet"
)

escenarios_csv = (
    OUTPUT_DIR
    / "escenarios_territoriales_amba.csv"
)

escenarios_gpkg = (
    OUTPUT_DIR
    / "escenarios_territoriales_amba.gpkg"
)

proyectos_escenarios_parquet = (
    OUTPUT_DIR
    / "proyectos_escenarios_territoriales_amba.parquet"
)

escenarios_geo_wgs84.to_parquet(
    escenarios_parquet,
    index=False
)

escenarios_df.drop(
    columns="geometry",
    errors="ignore"
).to_csv(
    escenarios_csv,
    index=False,
    encoding="utf-8-sig"
)

escenarios_geo_wgs84.to_file(
    escenarios_gpkg,
    layer="escenarios_territoriales",
    driver="GPKG",
)

proyectos_final.to_parquet(
    proyectos_escenarios_parquet,
    index=False
)

print(f"Parquet escenarios:\n{escenarios_parquet}")
print(f"CSV escenarios:\n{escenarios_csv}")
print(f"GeoPackage:\n{escenarios_gpkg}")
print(
    f"Parquet proyectos:\n"
    f"{proyectos_escenarios_parquet}"
)


# =============================================================================
# 31. RESUMEN JSON
# =============================================================================

encabezado("30. CONSTRUYENDO RESUMEN JSON")

resumen = {
    "proceso": 27,
    "nombre": "Construcción de escenarios territoriales AMBA",
    "version": VERSION,
    "fecha_proceso": pd.Timestamp.now().isoformat(),

    "parametros": {
        "distancia_agrupacion_m": DISTANCIA_AGRUPACION_M,
        "crs": CRS_WGS84,
        "crs_metrico": CRS_METRICO,
        "pesos": PESOS,
    },

    "entrada": {
        "archivo": str(INPUT_FILE),
        "proyectos": int(len(gdf)),
    },

    "resultado": {
        "escenarios": int(len(escenarios_df)),
        "proyectos": int(len(proyectos_final)),
    },

    "prioridades": {
        str(k): int(v)
        for k, v in escenarios_df[
            "prioridad_escenario"
        ].value_counts().items()
    },

    "horizontes": {
        str(k): int(v)
        for k, v in escenarios_df[
            "horizonte_escenario"
        ].value_counts().items()
    },

    "tipos": {
        str(k): int(v)
        for k, v in escenarios_df[
            "tipo_escenario"
        ].value_counts().items()
    },

    "diagnosticos": {
        str(k): int(v)
        for k, v in escenarios_df[
            "diagnostico_escenario"
        ].value_counts().items()
    },

    "top_10": (
        escenarios_df[
            [
                "ranking_escenario",
                "escenario_id",
                "score_escenario",
                "prioridad_escenario",
                "tipo_escenario",
                "horizonte_escenario",
                "cantidad_proyectos",
                "impacto_territorial",
                "cobertura_territorial",
            ]
        ]
        .head(10)
        .round(4)
        .to_dict(orient="records")
    ),
}

json_file = (
    OUTPUT_DIR
    / "escenarios_territoriales_amba_resumen.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        resumen,
        f,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print(f"JSON:\n{json_file}")


# =============================================================================
# 32. MAPA DE ESCENARIOS
# =============================================================================

encabezado("31. GENERANDO MAPAS Y GRÁFICOS")

def guardar_mapa(
    geo,
    columna,
    titulo,
    archivo,
    categorical=False,
):

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    if categorical:
        geo.plot(
            ax=ax,
            column=columna,
            legend=True,
            categorical=True,
            edgecolor="black",
            linewidth=0.3,
        )
    else:
        geo.plot(
            ax=ax,
            column=columna,
            legend=True,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_title(
        titulo,
        fontsize=14,
        fontweight="bold",
    )

    ax.set_axis_off()

    plt.tight_layout()

    path = OUTPUT_DIR / archivo

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Mapa: {path}")


guardar_mapa(
    escenarios_geo_wgs84,
    "score_escenario",
    "Escenarios territoriales AMBA - Score",
    "01_mapa_escenarios_territoriales.png",
)

guardar_mapa(
    escenarios_geo_wgs84,
    "prioridad_escenario",
    "Escenarios territoriales AMBA - Prioridad",
    "02_mapa_prioridad_escenarios.png",
    categorical=True,
)

guardar_mapa(
    escenarios_geo_wgs84,
    "cobertura_territorial",
    "Escenarios territoriales AMBA - Cobertura",
    "03_mapa_cobertura_metropolitana.png",
)

guardar_mapa(
    escenarios_geo_wgs84,
    "impacto_territorial",
    "Escenarios territoriales AMBA - Impacto",
    "04_mapa_impacto_territorial.png",
)

guardar_mapa(
    escenarios_geo_wgs84,
    "deficit_atendido",
    "Escenarios territoriales AMBA - Déficit atendido",
    "05_mapa_deficit_atendido.png",
)


# =============================================================================
# 33. GRÁFICO DEMANDA VS DÉFICIT
# =============================================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.scatter(
    escenarios_df["demanda_cubierta"],
    escenarios_df["deficit_atendido"],
    s=70,
    alpha=0.75,
)

ax.set_xlabel(
    "Demanda cubierta"
)

ax.set_ylabel(
    "Déficit atendido"
)

ax.set_title(
    "Demanda cubierta vs déficit atendido"
)

ax.grid(
    alpha=0.25
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "06_demanda_vs_deficit_atendido.png"
)

plt.savefig(
    path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print(f"Gráfico: {path}")


# =============================================================================
# 34. ESCENARIOS POR PRIORIDAD
# =============================================================================

conteo_prioridad = (
    escenarios_df[
        "prioridad_escenario"
    ]
    .value_counts()
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

conteo_prioridad.plot(
    kind="bar",
    ax=ax,
)

ax.set_title(
    "Escenarios por prioridad"
)

ax.set_xlabel(
    "Prioridad"
)

ax.set_ylabel(
    "Cantidad"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "07_escenarios_por_prioridad.png"
)

plt.savefig(
    path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print(f"Gráfico: {path}")


# =============================================================================
# 35. ESCENARIOS POR HORIZONTE
# =============================================================================

conteo_horizonte = (
    escenarios_df[
        "horizonte_escenario"
    ]
    .value_counts()
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

conteo_horizonte.plot(
    kind="bar",
    ax=ax,
)

ax.set_title(
    "Escenarios por horizonte de intervención"
)

ax.set_xlabel(
    "Horizonte"
)

ax.set_ylabel(
    "Cantidad"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "08_escenarios_por_horizonte.png"
)

plt.savefig(
    path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print(f"Gráfico: {path}")


# =============================================================================
# 36. DISTRIBUCIÓN DEL SCORE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.hist(
    escenarios_df["score_escenario"],
    bins=12,
    edgecolor="black",
)

ax.set_title(
    "Distribución del score de escenarios"
)

ax.set_xlabel(
    "Score escenario"
)

ax.set_ylabel(
    "Cantidad"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "09_distribucion_score_escenarios.png"
)

plt.savefig(
    path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print(f"Gráfico: {path}")


# =============================================================================
# 37. CONSOLIDADO FINAL
# =============================================================================

encabezado("32. CONSOLIDANDO DATASET FINAL")

escenarios_final = escenarios_geo_wgs84.copy()

# Orden lógico
columnas_orden = [
    "escenario_id",
    "escenario_nombre",
    "ranking_escenario",
    "prioridad_escenario",
    "tipo_escenario",
    "horizonte_escenario",
    "score_escenario",
    "cantidad_proyectos",
    "cantidad_tipos_proyecto",
    "tipos_proyecto",
    "impacto_territorial",
    "cobertura_territorial",
    "cobertura_proyectos",
    "cobertura_demanda",
    "deficit_atendido",
    "demanda_cubierta",
    "complementariedad",
    "diversidad_intervenciones",
    "sinergias_intervencion",
    "urgencia_promedio",
    "score_cartera_promedio",
    "score_prioridad_promedio",
    "diagnostico_escenario",
    "objetivo_escenario",
    "justificacion_escenario",
    "dimensiones_prioritarias",
    "geometry",
]

columnas_existentes = [
    c for c in columnas_orden
    if c in escenarios_final.columns
]

escenarios_final = escenarios_final[
    columnas_existentes
]

escenarios_final.to_parquet(
    escenarios_parquet,
    index=False
)

escenarios_final.to_file(
    escenarios_gpkg,
    layer="escenarios_territoriales",
    driver="GPKG",
)

print("Dataset final consolidado.")


# =============================================================================
# 38. FINAL
# =============================================================================

encabezado("27 - PROCESO FINALIZADO")

print(
    f"Proyectos analizados: {len(proyectos_final)}"
)

print(
    f"Escenarios territoriales: {len(escenarios_final)}"
)

print()
print("PRIORIDADES:")

for k, v in (
    escenarios_final[
        "prioridad_escenario"
    ]
    .value_counts()
    .items()
):

    print(f"  {k}: {v}")

print()
print("HORIZONTES:")

for k, v in (
    escenarios_final[
        "horizonte_escenario"
    ]
    .value_counts()
    .items()
):

    print(f"  {k}: {v}")

print()
print("TIPOS DE ESCENARIO:")

for k, v in (
    escenarios_final[
        "tipo_escenario"
    ]
    .value_counts()
    .items()
):

    print(f"  {k}: {v}")

print()
print("ARCHIVOS GENERADOS")

for archivo in sorted(
    OUTPUT_DIR.iterdir()
):

    if archivo.is_file():
        print(f"  {archivo.name}")

print()
print(
    "SIGUIENTE ETAPA"
)

print(
    "Evaluar escenarios metropolitanos, "
    "cobertura territorial y cartera de inversión "
    "mediante simulación de impactos y selección "
    "de escenarios estratégicos."
)