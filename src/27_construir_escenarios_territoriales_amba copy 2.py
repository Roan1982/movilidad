# -*- coding: utf-8 -*-

"""
===============================================================================
27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - V3
===============================================================================

Construye escenarios territoriales estratégicos a partir de la cartera de
proyectos generada por el proceso 26.

ENTRADA
-------
data/processed/cartera_proyectos_amba/
    cartera_proyectos_amba.parquet

SALIDA
------
data/processed/escenarios_territoriales_amba/

Archivos principales:

    escenarios_territoriales_amba.parquet
    escenarios_territoriales_amba.csv
    escenarios_territoriales_amba.gpkg
    proyectos_escenarios_territoriales_amba.parquet
    escenarios_territoriales_amba_resumen.json
    evaluacion_numero_escenarios.csv

Mapas:

    01_mapa_escenarios_territoriales.png
    02_mapa_prioridad_escenarios.png
    03_mapa_cobertura_metropolitana.png
    04_mapa_impacto_territorial.png
    05_mapa_deficit_atendido.png

Gráficos:

    06_demanda_vs_deficit_atendido.png
    07_escenarios_por_prioridad.png
    08_escenarios_por_horizonte.png
    09_distribucion_score_escenarios.png

===============================================================================
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

from shapely.ops import unary_union

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

warnings.filterwarnings("ignore")

VERSION = "V3"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartera_proyectos_amba"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_FILE = (
    INPUT_DIR
    / "cartera_proyectos_amba.parquet"
)

CRS_WGS84 = "EPSG:4326"

# Posiblemente ya utilizado en procesos anteriores del proyecto.
# Se mantiene como CRS métrico para AMBA.
CRS_METRICO = "EPSG:22185"

RANDOM_STATE = 42

MIN_ESCENARIOS = 6
MAX_ESCENARIOS = 12

MIN_PROYECTOS_ESCENARIO = 8

N_INIT = 50
MAX_ITER = 1000


# =============================================================================
# PESOS DEL SCORE DEL ESCENARIO
# =============================================================================

PESOS_SCORE = {
    "impacto_potencial": 0.25,
    "cobertura_territorial": 0.20,
    "deficit_atendido": 0.20,
    "demanda_cubierta": 0.15,
    "complementariedad": 0.10,
    "urgencia": 0.10,
}


# =============================================================================
# VARIABLES TERRITORIALES
# =============================================================================

VARIABLES_TERRITORIALES = [
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


# =============================================================================
# UTILIDADES
# =============================================================================

def imprimir_titulo(texto: str) -> None:
    print()
    print("=" * 80)
    print(texto)
    print("=" * 80)


def imprimir_seccion(numero: int, texto: str) -> None:
    print()
    print("=" * 80)
    print(f"{numero}. {texto}")
    print("=" * 80)


def normalizar_serie(
    serie: pd.Series,
) -> pd.Series:
    """
    Normaliza una serie a escala 0-100.

    Si todos los valores son iguales se asigna 50.
    """

    valores = pd.to_numeric(
        serie,
        errors="coerce",
    )

    minimo = valores.min()
    maximo = valores.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(
            50.0,
            index=serie.index,
            dtype=float,
        )

    if math.isclose(
        float(minimo),
        float(maximo),
    ):
        return pd.Series(
            50.0,
            index=serie.index,
            dtype=float,
        )

    resultado = (
        (valores - minimo)
        / (maximo - minimo)
        * 100.0
    )

    return resultado.fillna(0.0).astype(float)


def safe_mean(
    serie: pd.Series,
) -> float:

    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    return float(valores.mean())


def safe_sum(
    serie: pd.Series,
) -> float:

    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    return float(valores.sum())


def porcentaje(
    valor: float,
    total: float,
) -> float:

    if total <= 0:
        return 0.0

    return float(
        valor / total * 100.0
    )


def asegurar_numerica(
    gdf: gpd.GeoDataFrame,
    columna: str,
) -> None:

    gdf[columna] = pd.to_numeric(
        gdf[columna],
        errors="coerce",
    ).fillna(0.0)


# =============================================================================
# CLASIFICACIONES
# =============================================================================

def clasificar_prioridad(
    score: float,
) -> str:

    if score >= 75:
        return "PRIORIDAD_1_MUY_ALTA"

    if score >= 60:
        return "PRIORIDAD_2_ALTA"

    if score >= 45:
        return "PRIORIDAD_3_MEDIA"

    return "PRIORIDAD_4_BAJA"


def clasificar_horizonte(
    score: float,
    urgencia: float,
) -> str:

    if (
        urgencia >= 70
        or score >= 70
    ):
        return "CORTO_PLAZO"

    if (
        score >= 55
        or urgencia >= 50
    ):
        return "MEDIANO_PLAZO"

    return "LARGO_PLAZO"


def clasificar_tipo_escenario(
    demanda: float,
    deficit: float,
    impacto: float,
    complementariedad: float,
    centralidad: float,
    cantidad: int,
) -> str:

    if (
        cantidad >= 15
        and demanda >= 60
        and complementariedad >= 55
    ):
        return "ESCENARIO_METROPOLITANO"

    if (
        centralidad >= 70
        and impacto >= 65
    ):
        return "ESCENARIO_CENTRALIDAD"

    if deficit >= 60:
        return "ESCENARIO_DEFICIT"

    if complementariedad >= 55:
        return "ESCENARIO_INTEGRADO"

    return "ESCENARIO_SELECTIVO"


def construir_diagnostico(
    demanda: float,
    deficit: float,
    impacto: float,
    complementariedad: float,
    urgencia: float,
) -> str:

    if (
        demanda >= 70
        and deficit >= 60
    ):
        return "ALTA_DEMANDA_BAJO_SOPORTE"

    if impacto >= 75:
        return "ALTO_IMPACTO_POTENCIAL"

    if complementariedad >= 70:
        return "ALTA_COMPLEMENTARIEDAD"

    if urgencia >= 70:
        return "ALTA_URGENCIA"

    if deficit >= 70:
        return "DEFICIT_ESTRUCTURAL_ALTO"

    return "INTERVENCION_TERRITORIAL_MEDIA"


def construir_objetivo(
    tipo: str,
) -> str:

    objetivos = {

        "ESCENARIO_METROPOLITANO":
            (
                "Integrar intervenciones estratégicas de alcance "
                "metropolitano para mejorar simultáneamente demanda, "
                "conectividad, intermodalidad y cobertura territorial."
            ),

        "ESCENARIO_CENTRALIDAD":
            (
                "Fortalecer centralidades estratégicas mediante "
                "intervenciones orientadas a mejorar capacidad, "
                "accesibilidad e integración de los nodos de movilidad."
            ),

        "ESCENARIO_DEFICIT":
            (
                "Reducir déficits estructurales de infraestructura "
                "mediante intervenciones concentradas en territorios "
                "con bajo soporte frente a la demanda existente."
            ),

        "ESCENARIO_INTEGRADO":
            (
                "Coordinar proyectos complementarios para generar "
                "efectos sinérgicos sobre conectividad, "
                "intermodalidad e integración territorial."
            ),

        "ESCENARIO_SELECTIVO":
            (
                "Implementar intervenciones focalizadas sobre "
                "territorios prioritarios con impacto potencial "
                "y necesidades específicas."
            ),
    }

    return objetivos.get(
        tipo,
        objetivos["ESCENARIO_SELECTIVO"],
    )


def construir_justificacion(
    tipo: str,
    cantidad: int,
    demanda: float,
    deficit: float,
    impacto: float,
    urgencia: float,
) -> str:

    return (
        f"Escenario de tipo {tipo} compuesto por "
        f"{cantidad} proyectos. Presenta demanda territorial "
        f"promedio de {demanda:.1f}/100, déficit atendible "
        f"de {deficit:.1f}/100, impacto potencial de "
        f"{impacto:.1f}/100 y urgencia de {urgencia:.1f}/100."
    )


# =============================================================================
# DIRECTORIOS
# =============================================================================

def preparar_directorio() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# CARGA
# =============================================================================

def cargar_datos() -> gpd.GeoDataFrame:

    imprimir_seccion(
        1,
        "CARGANDO CARTERA DEL PROCESO 26",
    )

    print(
        f"Archivo de entrada:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nNo existe el archivo de entrada.\n\n"
            f"Esperado:\n{INPUT_FILE}\n\n"
            "Verificá que el proceso 26 haya generado "
            "cartera_proyectos_amba.parquet."
        )

    gdf = gpd.read_parquet(
        INPUT_FILE
    )

    if not isinstance(
        gdf,
        gpd.GeoDataFrame,
    ):
        raise TypeError(
            "El archivo no contiene una GeoDataFrame válida."
        )

    print(
        f"Registros : {len(gdf):,}"
    )

    print(
        f"Columnas  : {len(gdf.columns):,}"
    )

    print(
        f"CRS       : {gdf.crs}"
    )

    return gdf


# =============================================================================
# VALIDACIÓN DE ENTRADA
# =============================================================================

def validar_entrada(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    imprimir_seccion(
        2,
        "VALIDANDO DATOS DE ENTRADA",
    )

    gdf = gdf.copy()

    # -------------------------------------------------------------------------
    # CRS
    # -------------------------------------------------------------------------

    if gdf.crs is None:

        print(
            "CRS ausente. Se asignará EPSG:4326."
        )

        gdf = gdf.set_crs(
            CRS_WGS84
        )

    else:

        print(
            f"CRS detectado: {gdf.crs}"
        )

    # -------------------------------------------------------------------------
    # Geometrías
    # -------------------------------------------------------------------------

    nulas = int(
        gdf.geometry.isna().sum()
    )

    vacias = int(
        gdf.geometry.is_empty.sum()
    )

    invalidas = int(
        (~gdf.geometry.is_valid).sum()
    )

    print(
        f"Geometrías nulas    : {nulas}"
    )

    print(
        f"Geometrías vacías   : {vacias}"
    )

    print(
        f"Geometrías inválidas: {invalidas}"
    )

    if nulas > 0 or vacias > 0:

        raise ValueError(
            "La cartera contiene geometrías nulas o vacías."
        )

    if invalidas > 0:

        print(
            "Reparando geometrías inválidas..."
        )

        gdf["geometry"] = (
            gdf.geometry.make_valid()
        )

        invalidas_despues = int(
            (~gdf.geometry.is_valid).sum()
        )

        if invalidas_despues > 0:

            raise ValueError(
                "Persisten geometrías inválidas después "
                "de make_valid()."
            )

    # -------------------------------------------------------------------------
    # proyecto_id
    # -------------------------------------------------------------------------

    if "proyecto_id" not in gdf.columns:

        raise ValueError(
            "Falta la columna obligatoria 'proyecto_id'."
        )

    duplicados = int(
        gdf["proyecto_id"]
        .duplicated()
        .sum()
    )

    print(
        f"proyecto_id duplicados: {duplicados}"
    )

    if duplicados > 0:

        raise ValueError(
            "Existen proyecto_id duplicados."
        )

    # -------------------------------------------------------------------------
    # Registros
    # -------------------------------------------------------------------------

    if len(gdf) == 0:

        raise ValueError(
            "La cartera no contiene proyectos."
        )

    print(
        f"Proyectos válidos: {len(gdf):,}"
    )

    print(
        "Validación de entrada: OK"
    )

    return gdf


# =============================================================================
# VALIDACIÓN DE COMPONENTES
# =============================================================================

def validar_componentes(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    imprimir_seccion(
        3,
        "VALIDANDO COMPONENTES DE LA CARTERA",
    )

    gdf = gdf.copy()

    obligatorias = [
        "proyecto_id",
        "centralidad_id",
        "nodo_id",
        "tipo_proyecto",
        "escenario_intervencion",
        "prioridad_cartera",
        "score_cartera",
    ]

    # -------------------------------------------------------------------------
    # Obligatorias
    # -------------------------------------------------------------------------

    for columna in obligatorias:

        if columna not in gdf.columns:

            raise ValueError(
                f"Falta columna obligatoria: {columna}"
            )

    # -------------------------------------------------------------------------
    # Variables territoriales
    # -------------------------------------------------------------------------

    for columna in VARIABLES_TERRITORIALES:

        if columna not in gdf.columns:

            print(
                f"  AVISO: {columna} no existe. "
                "Se crea con valor 0."
            )

            gdf[columna] = 0.0

        else:

            asegurar_numerica(
                gdf,
                columna,
            )

            nulos = int(
                gdf[columna].isna().sum()
            )

            print(
                f"  {columna}: OK"
            )

    # -------------------------------------------------------------------------
    # Variables obligatorias numéricas
    # -------------------------------------------------------------------------

    asegurar_numerica(
        gdf,
        "score_cartera",
    )

    print()
    print(
        f"Proyectos validados: {len(gdf):,}"
    )

    return gdf


# =============================================================================
# PREPARACIÓN DE VARIABLES
# =============================================================================

def preparar_variables(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    imprimir_seccion(
        4,
        "PREPARANDO VARIABLES TERRITORIALES",
    )

    gdf = gdf.copy()

    # -------------------------------------------------------------------------
    # WGS84
    # -------------------------------------------------------------------------

    if gdf.crs is None:

        gdf = gdf.set_crs(
            CRS_WGS84
        )

    gdf = gdf.to_crs(
        CRS_WGS84
    )

    # -------------------------------------------------------------------------
    # Geometría métrica
    # -------------------------------------------------------------------------

    try:

        gdf_metric = gdf.to_crs(
            CRS_METRICO
        )

    except Exception as exc:

        raise RuntimeError(
            "No fue posible transformar la geometría "
            f"a {CRS_METRICO}."
        ) from exc

    centroides = (
        gdf_metric.geometry.centroid
    )

    gdf["x_m"] = centroides.x
    gdf["y_m"] = centroides.y

    # -------------------------------------------------------------------------
    # Normalización
    # -------------------------------------------------------------------------

    for columna in VARIABLES_TERRITORIALES:

        gdf[
            f"{columna}_norm"
        ] = normalizar_serie(
            gdf[columna]
        )

    # -------------------------------------------------------------------------
    # Presión de demanda
    # -------------------------------------------------------------------------

    gdf["presion_demanda"] = (
        0.60
        * gdf[
            "indice_demanda_estructural_norm"
        ]
        +
        0.40
        * gdf[
            "score_prioridad_territorial_norm"
        ]
    )

    # -------------------------------------------------------------------------
    # Presión de déficit
    # -------------------------------------------------------------------------

    gdf["presion_deficit"] = (
        0.60
        * gdf[
            "deficit_infraestructura_norm"
        ]
        +
        0.40
        * gdf[
            "deficit_estructural_promedio_norm"
        ]
    )

    # -------------------------------------------------------------------------
    # Capacidad de integración
    # -------------------------------------------------------------------------

    gdf["capacidad_integracion"] = (
        0.35
        * gdf[
            "indice_conectividad_estructural_norm"
        ]
        +
        0.35
        * gdf[
            "indice_intermodalidad_estructural_norm"
        ]
        +
        0.30
        * gdf[
            "indice_integracion_territorial_norm"
        ]
    )

    # -------------------------------------------------------------------------
    # Peso estratégico
    # -------------------------------------------------------------------------

    gdf["peso_estrategico"] = (
        0.35
        * gdf[
            "score_cartera_norm"
        ]
        +
        0.25
        * gdf[
            "impacto_potencial_norm"
        ]
        +
        0.20
        * gdf[
            "presion_demanda"
        ]
        +
        0.20
        * gdf[
            "presion_deficit"
        ]
    )

    print(
        "Variables territoriales construidas:"
    )

    print(
        "  x_m"
    )

    print(
        "  y_m"
    )

    print(
        "  presion_demanda"
    )

    print(
        "  presion_deficit"
    )

    print(
        "  capacidad_integracion"
    )

    print(
        "  peso_estrategico"
    )

    return gdf


# =============================================================================
# MATRIZ DE CLUSTERING
# =============================================================================

def construir_matriz_clustering(
    gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, list[str]]:

    imprimir_seccion(
        5,
        "CONSTRUYENDO MATRIZ MULTICRITERIO",
    )

    variables = [
        "x_m",
        "y_m",
        "indice_demanda_estructural_norm",
        "deficit_infraestructura_norm",
        "indice_conectividad_estructural_norm",
        "indice_intermodalidad_estructural_norm",
        "indice_integracion_territorial_norm",
        "indice_centralidad_estructural_norm",
        "impacto_potencial_norm",
        "urgencia_intervencion_norm",
        "score_cartera_norm",
    ]

    matriz_df = (
        gdf[variables]
        .copy()
    )

    matriz_df = (
        matriz_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    scaler = StandardScaler()

    matriz = scaler.fit_transform(
        matriz_df
    )

    # -------------------------------------------------------------------------
    # Peso espacial
    # -------------------------------------------------------------------------

    matriz[:, 0] *= 0.55
    matriz[:, 1] *= 0.55

    # -------------------------------------------------------------------------
    # Refuerzo territorial
    # -------------------------------------------------------------------------

    for nombre in [
        "indice_demanda_estructural_norm",
        "deficit_infraestructura_norm",
        "impacto_potencial_norm",
    ]:

        indice = variables.index(
            nombre
        )

        matriz[:, indice] *= 1.20

    # -------------------------------------------------------------------------
    # Refuerzo de integración
    # -------------------------------------------------------------------------

    for nombre in [
        "indice_intermodalidad_estructural_norm",
        "indice_integracion_territorial_norm",
    ]:

        indice = variables.index(
            nombre
        )

        matriz[:, indice] *= 1.10

    print(
        f"Proyectos : {matriz.shape[0]:,}"
    )

    print(
        f"Variables : {matriz.shape[1]:,}"
    )

    print(
        "Posición espacial ponderada: 0.55"
    )

    print(
        "Demanda / déficit / impacto: 1.20"
    )

    return matriz, variables


# =============================================================================
# EVALUACIÓN DE K
# =============================================================================

def evaluar_kmeans(
    matriz: np.ndarray,
    k: int,
) -> dict:

    modelo = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )

    labels = modelo.fit_predict(
        matriz
    )

    conteos = (
        pd.Series(labels)
        .value_counts()
    )

    min_size = int(
        conteos.min()
    )

    max_size = int(
        conteos.max()
    )

    if (
        len(np.unique(labels)) > 1
        and len(labels) > k
    ):

        silhouette = float(
            silhouette_score(
                matriz,
                labels,
            )
        )

    else:

        silhouette = -1.0

    penalizacion = 0.0

    # -------------------------------------------------------------------------
    # Penalización por clusters pequeños
    # -------------------------------------------------------------------------

    if min_size < MIN_PROYECTOS_ESCENARIO:

        deficit = (
            MIN_PROYECTOS_ESCENARIO
            - min_size
        )

        penalizacion += (
            deficit
            * 0.08
        )

    # -------------------------------------------------------------------------
    # Penalización por desbalance
    # -------------------------------------------------------------------------

    media = (
        len(labels)
        / k
    )

    if media > 0:

        relacion = (
            max_size
            / media
        )

        if relacion > 2.5:

            penalizacion += (
                relacion
                - 2.5
            ) * 0.10

    score = (
        silhouette
        - penalizacion
    )

    return {
        "k": k,
        "modelo": modelo,
        "labels": labels,
        "silhouette": silhouette,
        "min_size": min_size,
        "max_size": max_size,
        "score": score,
    }


# =============================================================================
# SELECCIÓN DE K
# =============================================================================

def seleccionar_numero_escenarios(
    matriz: np.ndarray,
) -> tuple[KMeans, np.ndarray, pd.DataFrame]:

    imprimir_seccion(
        6,
        "SELECCIONANDO CANTIDAD DE ESCENARIOS",
    )

    cantidad_proyectos = (
        matriz.shape[0]
    )

    k_min = max(
        2,
        MIN_ESCENARIOS,
    )

    k_max = min(
        MAX_ESCENARIOS,
        cantidad_proyectos - 1,
    )

    if k_max < 2:

        raise ValueError(
            "No existen suficientes proyectos para "
            "realizar clustering."
        )

    if k_min > k_max:

        k_min = 2

    resultados = []

    for k in range(
        k_min,
        k_max + 1,
    ):

        resultado = evaluar_kmeans(
            matriz,
            k,
        )

        resultados.append(
            {
                "k": resultado["k"],
                "silhouette": resultado[
                    "silhouette"
                ],
                "min_size": resultado[
                    "min_size"
                ],
                "max_size": resultado[
                    "max_size"
                ],
                "score_modelo": resultado[
                    "score"
                ],
            }
        )

        print(
            f"K={k:2d} | "
            f"silhouette="
            f"{resultado['silhouette']:.4f} | "
            f"mín="
            f"{resultado['min_size']:3d} | "
            f"máx="
            f"{resultado['max_size']:3d} | "
            f"score="
            f"{resultado['score']:.4f}"
        )

    evaluacion = (
        pd.DataFrame(
            resultados
        )
    )

    # -------------------------------------------------------------------------
    # Soluciones que cumplen tamaño mínimo
    # -------------------------------------------------------------------------

    validas = evaluacion[
        evaluacion["min_size"]
        >= MIN_PROYECTOS_ESCENARIO
    ]

    if not validas.empty:

        mejor_fila = (
            validas
            .sort_values(
                [
                    "score_modelo",
                    "silhouette",
                ],
                ascending=False,
            )
            .iloc[0]
        )

    else:

        print()
        print(
            "AVISO: ninguna solución cumple "
            f"el mínimo de {MIN_PROYECTOS_ESCENARIO} "
            "proyectos por escenario."
        )

        mejor_fila = (
            evaluacion
            .sort_values(
                "score_modelo",
                ascending=False,
            )
            .iloc[0]
        )

    mejor_k = int(
        mejor_fila["k"]
    )

    resultado_final = evaluar_kmeans(
        matriz,
        mejor_k,
    )

    print()
    print(
        f"K seleccionado: {mejor_k}"
    )

    print(
        f"Silhouette: "
        f"{resultado_final['silhouette']:.4f}"
    )

    print(
        f"Mínimo cluster: "
        f"{resultado_final['min_size']}"
    )

    print(
        f"Máximo cluster: "
        f"{resultado_final['max_size']}"
    )

    return (
        resultado_final["modelo"],
        resultado_final["labels"],
        evaluacion,
    )


# =============================================================================
# CORRECCIÓN DE CLUSTERS PEQUEÑOS
# =============================================================================

def corregir_clusters_pequenos(
    gdf: gpd.GeoDataFrame,
    labels: np.ndarray,
) -> np.ndarray:

    """
    Corrige clusters con menos de MIN_PROYECTOS_ESCENARIO.

    Los proyectos de un cluster pequeño se asignan al cluster válido
    más cercano utilizando simultáneamente:

        - posición espacial
        - demanda
        - déficit
        - impacto
        - conectividad
        - intermodalidad
        - integración
        - centralidad
        - urgencia
        - score cartera
    """

    labels = np.asarray(
        labels,
        dtype=int,
    ).copy()

    if len(labels) != len(gdf):

        raise ValueError(
            "La cantidad de labels no coincide "
            "con la cantidad de proyectos."
        )

    # -------------------------------------------------------------------------
    # Matriz secundaria para reasignación
    # -------------------------------------------------------------------------

    variables = [
        "x_m",
        "y_m",
        "indice_demanda_estructural_norm",
        "deficit_infraestructura_norm",
        "indice_conectividad_estructural_norm",
        "indice_intermodalidad_estructural_norm",
        "indice_integracion_territorial_norm",
        "indice_centralidad_estructural_norm",
        "impacto_potencial_norm",
        "urgencia_intervencion_norm",
        "score_cartera_norm",
    ]

    matriz_df = (
        gdf[variables]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    scaler = StandardScaler()

    matriz = scaler.fit_transform(
        matriz_df
    )

    matriz[:, 0] *= 0.55
    matriz[:, 1] *= 0.55

    # -------------------------------------------------------------------------
    # Iteraciones
    # -------------------------------------------------------------------------

    max_iteraciones = 100

    for _ in range(
        max_iteraciones
    ):

        conteos = (
            pd.Series(labels)
            .value_counts()
        )

        pequenos = conteos[
            conteos
            < MIN_PROYECTOS_ESCENARIO
        ]

        if pequenos.empty:
            break

        cluster_pequeno = int(
            pequenos
            .sort_values()
            .index[0]
        )

        indices_pequeno = np.where(
            labels
            == cluster_pequeno
        )[0]

        clusters_validos = [
            int(cluster)
            for cluster, cantidad
            in conteos.items()
            if (
                cluster
                != cluster_pequeno
                and cantidad
                >= MIN_PROYECTOS_ESCENARIO
            )
        ]

        if not clusters_validos:

            # Si no existe ningún cluster válido,
            # usamos el cluster de mayor tamaño.
            clusters_validos = [
                int(
                    conteos
                    .drop(
                        index=cluster_pequeno
                    )
                    .idxmax()
                )
            ]

        centros = {}

        for cluster in clusters_validos:

            indices_cluster = np.where(
                labels == cluster
            )[0]

            centros[cluster] = (
                matriz[
                    indices_cluster
                ]
                .mean(
                    axis=0
                )
            )

        # ---------------------------------------------------------------------
        # Reasignación
        # ---------------------------------------------------------------------

        for indice in indices_pequeno:

            punto = matriz[
                indice
            ]

            distancias = {}

            for cluster, centro in centros.items():

                distancias[
                    cluster
                ] = float(
                    np.linalg.norm(
                        punto
                        - centro
                    )
                )

            destino = min(
                distancias,
                key=distancias.get,
            )

            labels[
                indice
            ] = destino

    return labels


# =============================================================================
# RENORMALIZACIÓN DE CLUSTERS
# =============================================================================

def renumerar_clusters(
    labels: np.ndarray,
) -> np.ndarray:

    """
    Convierte los labels originales de KMeans en una secuencia continua:

        0, 1, 2, 3...

    Esto evita IDs faltantes después de fusionar clusters pequeños.
    """

    labels = np.asarray(
        labels,
        dtype=int,
    )

    clusters = sorted(
        np.unique(labels)
    )

    mapping = {
        cluster: indice
        for indice, cluster
        in enumerate(clusters)
    }

    return np.array(
        [
            mapping[int(label)]
            for label in labels
        ],
        dtype=int,
    )


# =============================================================================
# COMPLEMENTARIEDAD
# =============================================================================

def calcular_complementariedad(
    sub: pd.DataFrame,
) -> float:

    tipos = (
        sub[
            "tipo_proyecto"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    tipos = tipos[
        tipos != ""
    ]

    cantidad_tipos = (
        tipos.nunique()
    )

    if cantidad_tipos == 0:
        return 0.0

    # Escala estable:
    # 1 tipo = baja
    # 7 o más = máxima
    complementariedad = (
        cantidad_tipos
        / 7.0
        * 100.0
    )

    return float(
        min(
            100.0,
            complementariedad,
        )
    )


# =============================================================================
# CONSTRUCCIÓN DE ESCENARIOS
# =============================================================================

def construir_escenarios(
    gdf: gpd.GeoDataFrame,
    labels: np.ndarray,
) -> tuple[
    gpd.GeoDataFrame,
    pd.DataFrame,
]:

    imprimir_seccion(
        7,
        "CONSTRUYENDO ESCENARIOS TERRITORIALES",
    )

    gdf = gdf.copy()

    labels = renumerar_clusters(
        labels
    )

    gdf[
        "cluster_territorial"
    ] = labels + 1

    conteos = (
        gdf[
            "cluster_territorial"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Distribución inicial/final:"
    )

    print(
        conteos.to_string()
    )

    escenarios = []

    total_proyectos = len(gdf)

    for cluster_id in sorted(
        gdf[
            "cluster_territorial"
        ].unique()
    ):

        sub = gdf[
            gdf[
                "cluster_territorial"
            ]
            == cluster_id
        ].copy()

        cantidad = len(sub)

        demanda = safe_mean(
            sub[
                "indice_demanda_estructural"
            ]
        )

        deficit = (
            0.60
            * safe_mean(
                sub[
                    "deficit_infraestructura"
                ]
            )
            +
            0.40
            * safe_mean(
                sub[
                    "deficit_estructural_promedio"
                ]
            )
        )

        impacto = safe_mean(
            sub[
                "impacto_potencial"
            ]
        )

        urgencia = safe_mean(
            sub[
                "urgencia_intervencion"
            ]
        )

        score = safe_mean(
            sub[
                "score_cartera"
            ]
        )

        centralidad = safe_mean(
            sub[
                "indice_centralidad_estructural"
            ]
        )

        conectividad = safe_mean(
            sub[
                "indice_conectividad_estructural"
            ]
        )

        intermodalidad = safe_mean(
            sub[
                "indice_intermodalidad_estructural"
            ]
        )

        integracion = safe_mean(
            sub[
                "indice_integracion_territorial"
            ]
        )

        complementariedad = (
            calcular_complementariedad(
                sub
            )
        )

        cobertura = porcentaje(
            cantidad,
            total_proyectos,
        )

        # ---------------------------------------------------------------------
        # Demanda cubierta
        # ---------------------------------------------------------------------

        demanda_cubierta = (
            0.60 * demanda
            +
            0.20 * centralidad
            +
            0.20 * impacto
        )

        # ---------------------------------------------------------------------
        # Score
        # ---------------------------------------------------------------------

        score_escenario = (
            PESOS_SCORE[
                "impacto_potencial"
            ]
            * impacto
            +
            PESOS_SCORE[
                "cobertura_territorial"
            ]
            * cobertura
            +
            PESOS_SCORE[
                "deficit_atendido"
            ]
            * deficit
            +
            PESOS_SCORE[
                "demanda_cubierta"
            ]
            * demanda_cubierta
            +
            PESOS_SCORE[
                "complementariedad"
            ]
            * complementariedad
            +
            PESOS_SCORE[
                "urgencia"
            ]
            * urgencia
        )

        tipo = (
            clasificar_tipo_escenario(
                demanda=demanda,
                deficit=deficit,
                impacto=impacto,
                complementariedad=complementariedad,
                centralidad=centralidad,
                cantidad=cantidad,
            )
        )

        prioridad = (
            clasificar_prioridad(
                score_escenario
            )
        )

        horizonte = (
            clasificar_horizonte(
                score_escenario,
                urgencia,
            )
        )

        diagnostico = (
            construir_diagnostico(
                demanda=demanda,
                deficit=deficit,
                impacto=impacto,
                complementariedad=complementariedad,
                urgencia=urgencia,
            )
        )

        escenarios.append(
            {
                "cluster_territorial": int(
                    cluster_id
                ),

                "cantidad_proyectos": int(
                    cantidad
                ),

                "cobertura_territorial": float(
                    cobertura
                ),

                "deficit_atendido": float(
                    deficit
                ),

                "demanda_cubierta": float(
                    demanda_cubierta
                ),

                "complementariedad": float(
                    complementariedad
                ),

                "impacto_territorial": float(
                    impacto
                ),

                "urgencia_territorial": float(
                    urgencia
                ),

                "score_cartera_promedio": float(
                    score
                ),

                "demanda_promedio": float(
                    demanda
                ),

                "centralidad_promedio": float(
                    centralidad
                ),

                "conectividad_promedio": float(
                    conectividad
                ),

                "intermodalidad_promedio": float(
                    intermodalidad
                ),

                "integracion_promedio": float(
                    integracion
                ),

                "score_escenario": float(
                    score_escenario
                ),

                "tipo_escenario": tipo,

                "prioridad_escenario": prioridad,

                "horizonte_escenario": horizonte,

                "diagnostico_escenario": diagnostico,
            }
        )

    escenarios_df = (
        pd.DataFrame(
            escenarios
        )
    )

    print()
    print(
        f"Escenarios construidos: "
        f"{len(escenarios_df)}"
    )

    return (
        gdf,
        escenarios_df,
    )


# =============================================================================
# RANKING
# =============================================================================

def construir_ranking(
    escenarios: pd.DataFrame,
) -> pd.DataFrame:

    imprimir_seccion(
        8,
        "CONSTRUYENDO RANKING DE ESCENARIOS",
    )

    escenarios = (
        escenarios
        .copy()
    )

    escenarios = (
        escenarios
        .sort_values(
            [
                "score_escenario",
                "impacto_territorial",
                "deficit_atendido",
                "demanda_cubierta",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    escenarios[
        "ranking_escenario"
    ] = (
        np.arange(
            len(escenarios)
        )
        + 1
    )

    escenarios[
        "escenario_id"
    ] = escenarios[
        "ranking_escenario"
    ].map(
        lambda x:
        f"AMBA-E{x:03d}"
    )

    escenarios[
        "escenario_nombre"
    ] = escenarios.apply(
        lambda row:
        (
            f"{row['escenario_id']} - "
            f"{row['tipo_escenario']}"
        ),
        axis=1,
    )

    escenarios[
        "objetivo_escenario"
    ] = (
        escenarios[
            "tipo_escenario"
        ]
        .map(
            construir_objetivo
        )
    )

    escenarios[
        "justificacion_escenario"
    ] = escenarios.apply(
        lambda row:
        construir_justificacion(
            tipo=row[
                "tipo_escenario"
            ],
            cantidad=int(
                row[
                    "cantidad_proyectos"
                ]
            ),
            demanda=float(
                row[
                    "demanda_promedio"
                ]
            ),
            deficit=float(
                row[
                    "deficit_atendido"
                ]
            ),
            impacto=float(
                row[
                    "impacto_territorial"
                ]
            ),
            urgencia=float(
                row[
                    "urgencia_territorial"
                ]
            ),
        ),
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Dimensiones prioritarias
    # -------------------------------------------------------------------------

    def dimensiones(
        row: pd.Series,
    ) -> str:

        valores = {
            "DEMANDA":
                row[
                    "demanda_promedio"
                ],

            "DEFICIT":
                row[
                    "deficit_atendido"
                ],

            "IMPACTO":
                row[
                    "impacto_territorial"
                ],

            "CENTRALIDAD":
                row[
                    "centralidad_promedio"
                ],

            "CONECTIVIDAD":
                row[
                    "conectividad_promedio"
                ],

            "INTERMODALIDAD":
                row[
                    "intermodalidad_promedio"
                ],

            "INTEGRACION_TERRITORIAL":
                row[
                    "integracion_promedio"
                ],

            "URGENCIA":
                row[
                    "urgencia_territorial"
                ],
        }

        orden = sorted(
            valores.items(),
            key=lambda item:
            item[1],
            reverse=True,
        )

        return " | ".join(
            nombre
            for nombre, _
            in orden[:3]
        )

    escenarios[
        "dimensiones_prioritarias"
    ] = escenarios.apply(
        dimensiones,
        axis=1,
    )

    return escenarios


# =============================================================================
# ASIGNAR ESCENARIOS A PROYECTOS
# =============================================================================

def asignar_escenarios_a_proyectos(
    gdf: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
) -> gpd.GeoDataFrame:

    imprimir_seccion(
        9,
        "ASIGNANDO ESCENARIOS A PROYECTOS",
    )

    gdf = gdf.copy()

    columnas_mapping = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "horizonte_escenario",
        "diagnostico_escenario",
    ]

    mapping = (
        escenarios
        .set_index(
            "cluster_territorial"
        )[
            columnas_mapping
        ]
    )

    # Evitamos conflictos de nombres si el archivo de entrada ya contiene
    # alguna de estas columnas.
    for columna in columnas_mapping:

        if columna in gdf.columns:

            gdf = gdf.drop(
                columns=[
                    columna
                ]
            )

    gdf = gdf.join(
        mapping,
        on="cluster_territorial",
        how="left",
    )

    asignados = int(
        gdf[
            "escenario_id"
        ]
        .notna()
        .sum()
    )

    print(
        f"Proyectos asignados: "
        f"{asignados:,} / {len(gdf):,}"
    )

    if asignados != len(gdf):

        faltantes = gdf[
            gdf[
                "escenario_id"
            ].isna()
        ]

        raise ValueError(
            "Existen proyectos sin escenario asignado. "
            f"Cantidad: {len(faltantes)}"
        )

    return gdf


# =============================================================================
# GEOMETRÍAS DE ESCENARIOS
# =============================================================================

def construir_geometrias(
    gdf: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
) -> gpd.GeoDataFrame:

    imprimir_seccion(
        10,
        "CONSTRUYENDO GEOMETRÍAS DE ESCENARIOS",
    )

    gdf_metric = (
        gdf
        .to_crs(
            CRS_METRICO
        )
    )

    geometrias = []

    for cluster_id in sorted(
        gdf[
            "cluster_territorial"
        ]
        .unique()
    ):

        sub = gdf_metric[
            gdf_metric[
                "cluster_territorial"
            ]
            == cluster_id
        ]

        if sub.empty:
            continue

        # ---------------------------------------------------------------------
        # Centroides
        # ---------------------------------------------------------------------

        puntos = list(
            sub.geometry.centroid
        )

        buffers = [
            punto.buffer(
                1500
            )
            for punto in puntos
        ]

        if len(buffers) == 1:

            geometry = buffers[0]

        else:

            geometry = unary_union(
                buffers
            )

        # ---------------------------------------------------------------------
        # Reparar geometría
        # ---------------------------------------------------------------------

        if geometry is None:
            continue

        if geometry.is_empty:
            continue

        if not geometry.is_valid:

            geometry = (
                geometry.make_valid()
            )

        geometrias.append(
            {
                "cluster_territorial":
                    int(cluster_id),

                "geometry":
                    geometry,
            }
        )

    geometria_gdf = (
        gpd.GeoDataFrame(
            geometrias,
            geometry="geometry",
            crs=CRS_METRICO,
        )
        .to_crs(
            CRS_WGS84
        )
    )

    escenarios = escenarios.merge(
        geometria_gdf,
        on="cluster_territorial",
        how="left",
    )

    if escenarios[
        "geometry"
    ].isna().any():

        faltantes = int(
            escenarios[
                "geometry"
            ].isna().sum()
        )

        raise ValueError(
            "No fue posible construir geometría "
            f"para {faltantes} escenarios."
        )

    print(
        f"Geometrías construidas: "
        f"{len(escenarios)}"
    )

    return gpd.GeoDataFrame(
        escenarios,
        geometry="geometry",
        crs=CRS_WGS84,
    )


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

def validar_final(
    escenarios: gpd.GeoDataFrame,
    proyectos: gpd.GeoDataFrame,
) -> None:

    imprimir_seccion(
        11,
        "VALIDACIÓN FINAL",
    )

    columnas = [
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

    errores = []

    for columna in columnas:

        if columna not in escenarios.columns:

            print(
                f"{columna}: AUSENTE"
            )

            errores.append(
                columna
            )

            continue

        nulos = int(
            escenarios[
                columna
            ]
            .isna()
            .sum()
        )

        print(
            f"{columna}: "
            f"{nulos} nulos"
        )

        if nulos > 0:

            errores.append(
                columna
            )

    # -------------------------------------------------------------------------
    # Asignaciones
    # -------------------------------------------------------------------------

    asignaciones = int(
        proyectos[
            "escenario_id"
        ]
        .notna()
        .sum()
    )

    if asignaciones != len(
        proyectos
    ):

        print(
            f"Asignaciones: ERROR "
            f"{asignaciones}/"
            f"{len(proyectos)}"
        )

        errores.append(
            "asignacion_proyectos"
        )

    else:

        print(
            f"Asignaciones: OK "
            f"{asignaciones}/"
            f"{len(proyectos)}"
        )

    # -------------------------------------------------------------------------
    # Cantidad de proyectos
    # -------------------------------------------------------------------------

    suma_proyectos = int(
        escenarios[
            "cantidad_proyectos"
        ]
        .sum()
    )

    if suma_proyectos != len(
        proyectos
    ):

        errores.append(
            "cantidad_proyectos"
        )

        print(
            "Cantidad de proyectos: ERROR"
        )

    else:

        print(
            "Cantidad de proyectos: OK"
        )

    # -------------------------------------------------------------------------
    # Geometrías
    # -------------------------------------------------------------------------

    geometria_invalida = int(
        (
            ~escenarios
            .geometry
            .is_valid
        )
        .sum()
    )

    if geometria_invalida > 0:

        errores.append(
            "geometrias"
        )

        print(
            "Geometrías: ERROR"
        )

    else:

        print(
            "Geometrías: OK"
        )

    if errores:

        raise ValueError(
            "Validación final fallida: "
            + ", ".join(
                errores
            )
        )

    print()
    print(
        "VALIDACIÓN FINAL: OK"
    )


# =============================================================================
# RESUMEN JSON
# =============================================================================

def construir_resumen(
    escenarios: gpd.GeoDataFrame,
    proyectos: gpd.GeoDataFrame,
) -> dict:

    imprimir_seccion(
        12,
        "CONSTRUYENDO RESUMEN JSON",
    )

    resumen = {

        "proceso": 27,

        "version": VERSION,

        "descripcion":
            (
                "Construcción de escenarios territoriales "
                "estratégicos AMBA."
            ),

        "fecha_proceso":
            pd.Timestamp
            .now()
            .isoformat(),

        "proyectos_analizados":
            int(
                len(proyectos)
            ),

        "escenarios_construidos":
            int(
                len(escenarios)
            ),

        "min_proyectos_escenario":
            MIN_PROYECTOS_ESCENARIO,

        "rango_escenarios_evaluado":
            [
                MIN_ESCENARIOS,
                MAX_ESCENARIOS,
            ],

        "metodo":
            "KMeans multicriterio",

        "random_state":
            RANDOM_STATE,

        "pesos_modelo":
            PESOS_SCORE,

        "por_prioridad":
            (
                escenarios[
                    "prioridad_escenario"
                ]
                .value_counts()
                .to_dict()
            ),

        "por_tipo":
            (
                escenarios[
                    "tipo_escenario"
                ]
                .value_counts()
                .to_dict()
            ),

        "por_horizonte":
            (
                escenarios[
                    "horizonte_escenario"
                ]
                .value_counts()
                .to_dict()
            ),

        "por_diagnostico":
            (
                escenarios[
                    "diagnostico_escenario"
                ]
                .value_counts()
                .to_dict()
            ),

        "escenarios": [],
    }

    for _, row in (
        escenarios.iterrows()
    ):

        resumen[
            "escenarios"
        ].append(
            {
                "escenario_id":
                    row[
                        "escenario_id"
                    ],

                "escenario_nombre":
                    row[
                        "escenario_nombre"
                    ],

                "ranking":
                    int(
                        row[
                            "ranking_escenario"
                        ]
                    ),

                "cantidad_proyectos":
                    int(
                        row[
                            "cantidad_proyectos"
                        ]
                    ),

                "score":
                    float(
                        row[
                            "score_escenario"
                        ]
                    ),

                "prioridad":
                    row[
                        "prioridad_escenario"
                    ],

                "tipo":
                    row[
                        "tipo_escenario"
                    ],

                "horizonte":
                    row[
                        "horizonte_escenario"
                    ],

                "impacto":
                    float(
                        row[
                            "impacto_territorial"
                        ]
                    ),

                "cobertura":
                    float(
                        row[
                            "cobertura_territorial"
                        ]
                    ),

                "deficit_atendido":
                    float(
                        row[
                            "deficit_atendido"
                        ]
                    ),

                "demanda_cubierta":
                    float(
                        row[
                            "demanda_cubierta"
                        ]
                    ),

                "complementariedad":
                    float(
                        row[
                            "complementariedad"
                        ]
                    ),

                "diagnostico":
                    row[
                        "diagnostico_escenario"
                    ],
            }
        )

    return resumen


# =============================================================================
# GUARDAR ARCHIVOS
# =============================================================================

def guardar_archivos(
    escenarios: gpd.GeoDataFrame,
    proyectos: gpd.GeoDataFrame,
    resumen: dict,
    evaluacion_k: pd.DataFrame,
) -> None:

    imprimir_seccion(
        13,
        "GUARDANDO ARCHIVOS",
    )

    preparar_directorio()

    parquet_escenarios = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.parquet"
    )

    csv_escenarios = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.csv"
    )

    gpkg_escenarios = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.gpkg"
    )

    parquet_proyectos = (
        OUTPUT_DIR
        / "proyectos_escenarios_territoriales_amba.parquet"
    )

    json_resumen = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba_resumen.json"
    )

    csv_k = (
        OUTPUT_DIR
        / "evaluacion_numero_escenarios.csv"
    )

    # -------------------------------------------------------------------------
    # Parquet escenarios
    # -------------------------------------------------------------------------

    escenarios.to_parquet(
        parquet_escenarios,
        index=False,
    )

    # -------------------------------------------------------------------------
    # CSV escenarios
    # -------------------------------------------------------------------------

    escenarios_sin_geom = (
        escenarios.drop(
            columns=[
                "geometry"
            ],
            errors="ignore",
        )
    )

    escenarios_sin_geom.to_csv(
        csv_escenarios,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # GeoPackage
    # -------------------------------------------------------------------------

    if gpkg_escenarios.exists():

        try:
            gpkg_escenarios.unlink()
        except PermissionError as exc:

            raise PermissionError(
                "No se puede reemplazar el GeoPackage. "
                "Probablemente está abierto en QGIS u otro programa."
            ) from exc

    escenarios.to_file(
        gpkg_escenarios,
        layer="escenarios_territoriales",
        driver="GPKG",
    )

    # -------------------------------------------------------------------------
    # Proyectos
    # -------------------------------------------------------------------------

    proyectos.to_parquet(
        parquet_proyectos,
        index=False,
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    with open(
        json_resumen,
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

    # -------------------------------------------------------------------------
    # Evaluación K
    # -------------------------------------------------------------------------

    evaluacion_k.to_csv(
        csv_k,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Parquet escenarios:\n"
        f"{parquet_escenarios}"
    )

    print(
        f"CSV escenarios:\n"
        f"{csv_escenarios}"
    )

    print(
        f"GeoPackage:\n"
        f"{gpkg_escenarios}"
    )

    print(
        f"Parquet proyectos:\n"
        f"{parquet_proyectos}"
    )

    print(
        f"JSON:\n"
        f"{json_resumen}"
    )

    print(
        f"Evaluación K:\n"
        f"{csv_k}"
    )


# =============================================================================
# MAPAS
# =============================================================================

def guardar_mapa(
    gdf: gpd.GeoDataFrame,
    columna: str,
    titulo: str,
    archivo: Path,
    cmap: str,
    proyectos: gpd.GeoDataFrame | None = None,
) -> None:

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    gdf.plot(
        ax=ax,
        column=columna,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.70,
    )

    if (
        proyectos is not None
        and not proyectos.empty
    ):

        proyectos.plot(
            ax=ax,
            markersize=10,
            color="black",
            alpha=0.60,
        )

    ax.set_title(
        titulo,
        fontsize=16,
        fontweight="bold",
    )

    ax.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        archivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def generar_mapas(
    escenarios: gpd.GeoDataFrame,
    proyectos: gpd.GeoDataFrame,
) -> None:

    imprimir_seccion(
        14,
        "GENERANDO MAPAS",
    )

    # -------------------------------------------------------------------------
    # Mapa 1
    # -------------------------------------------------------------------------

    salida = (
        OUTPUT_DIR
        / "01_mapa_escenarios_territoriales.png"
    )

    guardar_mapa(
        escenarios,
        "ranking_escenario",
        "Escenarios Territoriales Estratégicos AMBA",
        salida,
        "viridis",
        proyectos,
    )

    print(
        f"Mapa: {salida}"
    )

    # -------------------------------------------------------------------------
    # Mapa 2
    # -------------------------------------------------------------------------

    salida = (
        OUTPUT_DIR
        / "02_mapa_prioridad_escenarios.png"
    )

    guardar_mapa(
        escenarios,
        "score_escenario",
        "Prioridad de Escenarios Territoriales",
        salida,
        "plasma",
    )

    print(
        f"Mapa: {salida}"
    )

    # -------------------------------------------------------------------------
    # Mapa 3
    # -------------------------------------------------------------------------

    salida = (
        OUTPUT_DIR
        / "03_mapa_cobertura_metropolitana.png"
    )

    guardar_mapa(
        escenarios,
        "cobertura_territorial",
        "Cobertura Relativa de la Cartera por Escenario",
        salida,
        "YlGn",
    )

    print(
        f"Mapa: {salida}"
    )

    # -------------------------------------------------------------------------
    # Mapa 4
    # -------------------------------------------------------------------------

    salida = (
        OUTPUT_DIR
        / "04_mapa_impacto_territorial.png"
    )

    guardar_mapa(
        escenarios,
        "impacto_territorial",
        "Impacto Territorial de los Escenarios",
        salida,
        "OrRd",
    )

    print(
        f"Mapa: {salida}"
    )

    # -------------------------------------------------------------------------
    # Mapa 5
    # -------------------------------------------------------------------------

    salida = (
        OUTPUT_DIR
        / "05_mapa_deficit_atendido.png"
    )

    guardar_mapa(
        escenarios,
        "deficit_atendido",
        "Déficit Territorial Atendido",
        salida,
        "Reds",
    )

    print(
        f"Mapa: {salida}"
    )


# =============================================================================
# GRÁFICOS
# =============================================================================

def generar_graficos(
    escenarios: gpd.GeoDataFrame,
) -> None:

    imprimir_seccion(
        15,
        "GENERANDO GRÁFICOS",
    )

    # -------------------------------------------------------------------------
    # 06 - Demanda vs déficit
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    tamaños = (
        pd.to_numeric(
            escenarios[
                "cantidad_proyectos"
            ],
            errors="coerce",
        )
        .fillna(1.0)
        * 18
        + 30
    )

    ax.scatter(
        escenarios[
            "demanda_cubierta"
        ],
        escenarios[
            "deficit_atendido"
        ],
        s=tamaños,
        alpha=0.75,
    )

    for _, row in (
        escenarios.iterrows()
    ):

        ax.annotate(
            row[
                "escenario_id"
            ],
            (
                row[
                    "demanda_cubierta"
                ],
                row[
                    "deficit_atendido"
                ],
            ),
            fontsize=8,
        )

    ax.set_xlabel(
        "Demanda cubierta"
    )

    ax.set_ylabel(
        "Déficit atendido"
    )

    ax.set_title(
        "Demanda vs Déficit Atendido",
        fontweight="bold",
    )

    ax.grid(
        alpha=0.25
    )

    salida = (
        OUTPUT_DIR
        / "06_demanda_vs_deficit_atendido.png"
    )

    plt.tight_layout()

    plt.savefig(
        salida,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {salida}"
    )

    # -------------------------------------------------------------------------
    # 07 - Prioridad
    # -------------------------------------------------------------------------

    conteo = (
        escenarios[
            "prioridad_escenario"
        ]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    conteo.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Escenarios por Prioridad",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Prioridad"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    salida = (
        OUTPUT_DIR
        / "07_escenarios_por_prioridad.png"
    )

    plt.tight_layout()

    plt.savefig(
        salida,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {salida}"
    )

    # -------------------------------------------------------------------------
    # 08 - Horizonte
    # -------------------------------------------------------------------------

    conteo = (
        escenarios[
            "horizonte_escenario"
        ]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    conteo.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Escenarios por Horizonte",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Horizonte"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    salida = (
        OUTPUT_DIR
        / "08_escenarios_por_horizonte.png"
    )

    plt.tight_layout()

    plt.savefig(
        salida,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {salida}"
    )

    # -------------------------------------------------------------------------
    # 09 - Distribución score
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.hist(
        escenarios[
            "score_escenario"
        ],
        bins=min(
            10,
            max(
                1,
                len(escenarios),
            ),
        ),
        alpha=0.80,
    )

    ax.set_title(
        "Distribución del Score de Escenarios",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Score escenario"
    )

    ax.set_ylabel(
        "Cantidad"
    )

    ax.grid(
        alpha=0.25
    )

    salida = (
        OUTPUT_DIR
        / "09_distribucion_score_escenarios.png"
    )

    plt.tight_layout()

    plt.savefig(
        salida,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfico: {salida}"
    )


# =============================================================================
# TOP ESCENARIOS
# =============================================================================

def imprimir_top(
    escenarios: pd.DataFrame,
) -> None:

    imprimir_seccion(
        16,
        "TOP ESCENARIOS TERRITORIALES",
    )

    columnas = [
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

    disponibles = [
        columna
        for columna in columnas
        if columna in escenarios.columns
    ]

    print(
        escenarios[
            disponibles
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# =============================================================================
# RESUMEN FINAL
# =============================================================================

def imprimir_resumen(
    escenarios: pd.DataFrame,
) -> None:

    imprimir_seccion(
        17,
        "RESUMEN DE ESCENARIOS",
    )

    print()
    print(
        "Por prioridad:"
    )

    print(
        escenarios[
            "prioridad_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Por horizonte:"
    )

    print(
        escenarios[
            "horizonte_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Por tipo:"
    )

    print(
        escenarios[
            "tipo_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Por diagnóstico:"
    )

    print(
        escenarios[
            "diagnostico_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Distribución de proyectos:"
    )

    print(
        escenarios[
            [
                "escenario_id",
                "cantidad_proyectos",
            ]
        ]
        .sort_values(
            "cantidad_proyectos",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    imprimir_titulo(
        f"27 - CONSTRUCCIÓN DE ESCENARIOS "
        f"TERRITORIALES AMBA - {VERSION}"
    )

    print(
        f"Proyecto       : {PROJECT_ROOT}"
    )

    print(
        f"Entrada        : {INPUT_FILE}"
    )

    print(
        f"Salida         : {OUTPUT_DIR}"
    )

    print(
        f"CRS             : {CRS_WGS84}"
    )

    print(
        f"CRS métrico     : {CRS_METRICO}"
    )

    print()
    print(
        "CONFIGURACIÓN"
    )

    print(
        f"  Escenarios candidatos : "
        f"{MIN_ESCENARIOS} - {MAX_ESCENARIOS}"
    )

    print(
        f"  Mínimo proyectos     : "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )

    print(
        "  Método                : "
        "KMeans multicriterio"
    )

    print(
        f"  Random state          : "
        f"{RANDOM_STATE}"
    )

    print()
    print(
        "PESOS DEL MODELO"
    )

    print(
        f"  Impacto potencial     : "
        f"{PESOS_SCORE['impacto_potencial']:.0%}"
    )

    print(
        f"  Cobertura territorial : "
        f"{PESOS_SCORE['cobertura_territorial']:.0%}"
    )

    print(
        f"  Déficit atendido      : "
        f"{PESOS_SCORE['deficit_atendido']:.0%}"
    )

    print(
        f"  Demanda cubierta      : "
        f"{PESOS_SCORE['demanda_cubierta']:.0%}"
    )

    print(
        f"  Complementariedad     : "
        f"{PESOS_SCORE['complementariedad']:.0%}"
    )

    print(
        f"  Urgencia              : "
        f"{PESOS_SCORE['urgencia']:.0%}"
    )

    preparar_directorio()

    # =========================================================================
    # 1. CARGA
    # =========================================================================

    gdf = cargar_datos()

    # =========================================================================
    # 2. VALIDACIÓN DE ENTRADA
    # =========================================================================

    gdf = validar_entrada(
        gdf
    )

    # =========================================================================
    # 3. VALIDACIÓN DE COMPONENTES
    # =========================================================================

    gdf = validar_componentes(
        gdf
    )

    # =========================================================================
    # 4. VARIABLES TERRITORIALES
    # =========================================================================

    gdf = preparar_variables(
        gdf
    )

    # =========================================================================
    # 5. MATRIZ MULTICRITERIO
    # =========================================================================

    matriz, variables = (
        construir_matriz_clustering(
            gdf
        )
    )

    print()
    print(
        "Variables utilizadas:"
    )

    for variable in variables:

        print(
            f"  - {variable}"
        )

    # =========================================================================
    # 6. SELECCIÓN DE K
    # =========================================================================

    modelo, labels, evaluacion_k = (
        seleccionar_numero_escenarios(
            matriz
        )
    )

    # =========================================================================
    # 7. CORRECCIÓN DE CLUSTERS
    # =========================================================================

    imprimir_seccion(
        7,
        "CORRIGIENDO CLUSTERS PEQUEÑOS",
    )

    labels = (
        corregir_clusters_pequenos(
            gdf,
            labels,
        )
    )

    labels = (
        renumerar_clusters(
            labels
        )
    )

    conteos = (
        pd.Series(
            labels + 1
        )
        .value_counts()
        .sort_index()
    )

    print(
        "Distribución final:"
    )

    print(
        conteos.to_string()
    )

    # =========================================================================
    # 8. CONSTRUCCIÓN DE ESCENARIOS
    # =========================================================================

    proyectos, escenarios = (
        construir_escenarios(
            gdf,
            labels,
        )
    )

    # =========================================================================
    # 9. RANKING
    # =========================================================================

    escenarios = (
        construir_ranking(
            escenarios
        )
    )

    # =========================================================================
    # 10. ASIGNACIÓN A PROYECTOS
    # =========================================================================

    proyectos = (
        asignar_escenarios_a_proyectos(
            proyectos,
            escenarios,
        )
    )

    # =========================================================================
    # 11. GEOMETRÍAS
    # =========================================================================

    escenarios_geo = (
        construir_geometrias(
            proyectos,
            escenarios,
        )
    )

    # =========================================================================
    # 12. VALIDACIÓN FINAL
    # =========================================================================

    validar_final(
        escenarios_geo,
        proyectos,
    )

    # =========================================================================
    # 13. TOP
    # =========================================================================

    imprimir_top(
        escenarios_geo
    )

    # =========================================================================
    # 14. RESUMEN
    # =========================================================================

    imprimir_resumen(
        escenarios_geo
    )

    # =========================================================================
    # 15. RESUMEN JSON
    # =========================================================================

    resumen = (
        construir_resumen(
            escenarios_geo,
            proyectos,
        )
    )

    # =========================================================================
    # 16. GUARDADO
    # =========================================================================

    guardar_archivos(
        escenarios_geo,
        proyectos,
        resumen,
        evaluacion_k,
    )

    # =========================================================================
    # 17. MAPAS
    # =========================================================================

    generar_mapas(
        escenarios_geo,
        proyectos,
    )

    # =========================================================================
    # 18. GRÁFICOS
    # =========================================================================

    generar_graficos(
        escenarios_geo
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    imprimir_titulo(
        "27 - PROCESO FINALIZADO CORRECTAMENTE"
    )

    print(
        f"Proyectos analizados      : "
        f"{len(proyectos):,}"
    )

    print(
        f"Escenarios territoriales  : "
        f"{len(escenarios_geo):,}"
    )

    print()
    print(
        "DISTRIBUCIÓN DE PROYECTOS"
    )

    for _, row in (
        escenarios_geo
        .sort_values(
            "ranking_escenario"
        )
        .iterrows()
    ):

        print(
            f"  {row['escenario_id']}: "
            f"{int(row['cantidad_proyectos']):3d} "
            f"proyectos | "
            f"{row['tipo_escenario']} | "
            f"score="
            f"{row['score_escenario']:.2f}"
        )

    print()
    print(
        "PRIORIDADES:"
    )

    print(
        escenarios_geo[
            "prioridad_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "HORIZONTES:"
    )

    print(
        escenarios_geo[
            "horizonte_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "TIPOS DE ESCENARIO:"
    )

    print(
        escenarios_geo[
            "tipo_escenario"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "ARCHIVOS GENERADOS:"
    )

    archivos = [

        "01_mapa_escenarios_territoriales.png",

        "02_mapa_prioridad_escenarios.png",

        "03_mapa_cobertura_metropolitana.png",

        "04_mapa_impacto_territorial.png",

        "05_mapa_deficit_atendido.png",

        "06_demanda_vs_deficit_atendido.png",

        "07_escenarios_por_prioridad.png",

        "08_escenarios_por_horizonte.png",

        "09_distribucion_score_escenarios.png",

        "evaluacion_numero_escenarios.csv",

        "escenarios_territoriales_amba.csv",

        "escenarios_territoriales_amba.gpkg",

        "escenarios_territoriales_amba.parquet",

        "escenarios_territoriales_amba_resumen.json",

        "proyectos_escenarios_territoriales_amba.parquet",
    ]

    for archivo in archivos:

        print(
            f"  {archivo}"
        )

    print()
    print(
        "SIGUIENTE ETAPA"
    )

    print(
        "Evaluar escenarios metropolitanos, cobertura "
        "territorial y cartera de inversión mediante "
        "simulación de impactos y selección de escenarios "
        "estratégicos."
    )


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Proceso interrumpido por el usuario."
        )

        raise SystemExit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "ERROR EN EL PROCESO 27"
        )
        print("=" * 80)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "El archivo Python NO será modificado."
        )

        raise