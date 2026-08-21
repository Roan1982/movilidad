# -*- coding: utf-8 -*-
"""
===============================================================================
27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - V5
===============================================================================

Objetivo
--------
Construir escenarios territoriales de inversión a partir de la cartera
consolidada del proceso 26.

Principios de V5
----------------
1. El clustering define grupos territoriales; no define por sí solo la
   prioridad de inversión.
2. La cantidad de proyectos NO participa del score de prioridad.
3. El componente espacial y el componente territorial tienen pesos explícitos.
4. La dimensión territorial se pondera de forma transparente.
5. Se conserva la reparación determinística de clusters pequeños.
6. La selección de K se realiza sobre la solución FINAL reparada.
7. Se calculan silhouette, cohesión y dispersión.
8. La tipología de escenario se determina por el perfil territorial dominante,
   evitando que déficit domine artificialmente todos los escenarios.
9. Se genera un diagnóstico explicable para cada escenario.
10. Se mantienen las salidas principales del proceso anterior y se agregan
    productos de auditoría.

Entrada
-------
data/processed/cartera_proyectos_amba/cartera_proyectos_amba.parquet

Salida
------
data/processed/escenarios_territoriales_amba/

Autor
------
Pipeline AMBA - Movilidad
===============================================================================
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import RobustScaler


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "cartera_proyectos_amba"
    / "cartera_proyectos_amba.parquet"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

CRS_GEOGRAFICO = "EPSG:4326"
CRS_METRICO = "EPSG:22185"

RANDOM_STATE = 42

K_MIN = 6
K_MAX = 12
MIN_PROYECTOS_ESCENARIO = 8

# -------------------------------------------------------------------------
# Pesos del clustering
# -------------------------------------------------------------------------
PESO_ESPACIAL = 0.55
PESO_TERRITORIAL = 0.45

PESO_DEMANDA = 0.20
PESO_DEFICIT = 0.20
PESO_CONECTIVIDAD = 0.15
PESO_INTERMODALIDAD = 0.15
PESO_INTEGRACION = 0.10
PESO_CENTRALIDAD = 0.10
PESO_IMPACTO = 0.10

# -------------------------------------------------------------------------
# Pesos del score de escenario.
# Importante: el tamaño del cluster NO participa.
# -------------------------------------------------------------------------
PESO_SCORE_DEMANDA = 0.20
PESO_SCORE_DEFICIT = 0.20
PESO_SCORE_IMPACTO = 0.20
PESO_SCORE_CONECTIVIDAD = 0.10
PESO_SCORE_INTERMODALIDAD = 0.10
PESO_SCORE_INTEGRACION = 0.07
PESO_SCORE_CENTRALIDAD = 0.05
PESO_SCORE_URGENCIA = 0.05
PESO_SCORE_CARTERA = 0.03

# Si la diferencia entre las dos dimensiones principales es menor a este
# umbral, se considera un perfil integrado.
MARGEN_PERFIL_INTEGRADO = 7.0


# =============================================================================
# COLUMNAS
# =============================================================================

COLUMNAS_ID = [
    "proyecto_id",
    "id_proyecto",
    "codigo_proyecto",
]

COLUMNAS_SCORE = ["score_cartera"]
COLUMNAS_PRIORIDAD = ["score_prioridad_territorial"]
COLUMNAS_IMPACTO = ["impacto_potencial"]
COLUMNAS_URGENCIA = ["urgencia_intervencion"]

COLUMNAS_DEMANDA = [
    "indice_demanda_estructural",
    "indice_demanda",
]

COLUMNAS_INFRA = [
    "indice_infraestructura_estructural",
]

COLUMNAS_DEFICIT = [
    "deficit_infraestructura",
    "deficit_estructural_promedio",
]

COLUMNAS_CONECTIVIDAD = [
    "indice_conectividad_estructural",
]

COLUMNAS_INTERMODALIDAD = [
    "indice_intermodalidad_estructural",
]

COLUMNAS_INTEGRACION = [
    "indice_integracion_territorial",
]

COLUMNAS_CENTRALIDAD = [
    "indice_centralidad_estructural",
]


# =============================================================================
# UTILIDADES
# =============================================================================

warnings.filterwarnings("ignore")


def titulo(texto: str) -> None:
    print()
    print("=" * 80)
    print(texto)
    print("=" * 80)


def subtitulo(numero: str, texto: str) -> None:
    print()
    print("=" * 80)
    print(f"{numero}. {texto}")
    print("=" * 80)


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = False,
) -> str | None:
    for columna in candidatos:
        if columna in df.columns:
            return columna

    if obligatoria:
        raise ValueError(
            "No se encontró ninguna columna requerida:\n"
            + "\n".join(f"  - {x}" for x in candidatos)
        )

    return None


def convertir_numerico(
    serie: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    resultado = pd.to_numeric(serie, errors="coerce")
    resultado = resultado.replace([np.inf, -np.inf], np.nan)
    return resultado.fillna(default).astype(float)


def normalizar_minmax(serie: pd.Series) -> pd.Series:
    x = convertir_numerico(serie)

    minimo = float(x.min())
    maximo = float(x.max())

    if not np.isfinite(minimo) or not np.isfinite(maximo):
        return pd.Series(0.5, index=x.index, dtype=float)

    if math.isclose(minimo, maximo):
        return pd.Series(0.5, index=x.index, dtype=float)

    return (x - minimo) / (maximo - minimo)


def safe_float(valor: Any) -> float:
    try:
        x = float(valor)
        return x if np.isfinite(x) else 0.0
    except Exception:
        return 0.0


def safe_int(valor: Any) -> int:
    try:
        return int(round(float(valor)))
    except Exception:
        return 0


def porcentaje(valor: float) -> float:
    return round(float(valor) * 100.0, 4)


def media_normalizada(
    serie: pd.Series,
    indices: pd.Index,
) -> float:
    if len(indices) == 0:
        return 50.0

    valores = convertir_numerico(serie.loc[indices])
    return porcentaje(float(normalizar_minmax(valores).mean()))


# =============================================================================
# CARGA Y VALIDACIÓN
# =============================================================================

def cargar_cartera() -> gpd.GeoDataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "No existe la cartera del proceso 26:\n"
            f"{INPUT_PATH}"
        )

    print("Archivo de entrada:")
    print(INPUT_PATH)

    cartera = gpd.read_parquet(INPUT_PATH)

    if cartera.empty:
        raise ValueError("La cartera está vacía.")

    if "geometry" not in cartera.columns:
        raise ValueError("La cartera no contiene geometry.")

    if cartera.crs is None:
        cartera = cartera.set_crs(
            CRS_GEOGRAFICO,
            allow_override=True,
        )

    return cartera


def validar_entrada(cartera: gpd.GeoDataFrame) -> str:
    subtitulo("2", "VALIDANDO DATOS DE ENTRADA")

    print(f"Registros : {len(cartera):,}")
    print(f"Columnas  : {len(cartera.columns):,}")
    print(f"CRS       : {cartera.crs}")

    nulos = int(cartera.geometry.isna().sum())
    vacias = int(cartera.geometry.is_empty.sum())
    invalidas = int((~cartera.geometry.is_valid).sum())

    print(f"Geometrías nulas    : {nulos}")
    print(f"Geometrías vacías   : {vacias}")
    print(f"Geometrías inválidas: {invalidas}")

    if nulos:
        raise ValueError("Existen geometrías nulas.")

    if vacias:
        raise ValueError("Existen geometrías vacías.")

    if invalidas:
        print("ADVERTENCIA: corrigiendo geometrías inválidas.")
        cartera["geometry"] = cartera.geometry.make_valid()

        if (~cartera.geometry.is_valid).any():
            raise ValueError(
                "No fue posible corregir todas las geometrías."
            )

    columna_id = encontrar_columna(
        cartera,
        COLUMNAS_ID,
        obligatoria=True,
    )

    duplicados = int(cartera[columna_id].duplicated().sum())

    print(f"ID utilizado       : {columna_id}")
    print(f"IDs duplicados     : {duplicados}")

    if duplicados:
        raise ValueError(
            f"La columna {columna_id} contiene duplicados."
        )

    minimo_requerido = K_MIN * MIN_PROYECTOS_ESCENARIO

    if len(cartera) < minimo_requerido:
        raise ValueError(
            f"Se requieren al menos {minimo_requerido} proyectos "
            f"para K_MIN={K_MIN} y mínimo={MIN_PROYECTOS_ESCENARIO}."
        )

    print(f"Proyectos válidos   : {len(cartera):,}")
    print("Validación de entrada: OK")

    return columna_id


def validar_componentes(
    cartera: gpd.GeoDataFrame,
) -> dict[str, str]:
    subtitulo("3", "VALIDANDO COMPONENTES TERRITORIALES")

    grupos = {
        "score_cartera": COLUMNAS_SCORE,
        "score_prioridad_territorial": COLUMNAS_PRIORIDAD,
        "impacto_potencial": COLUMNAS_IMPACTO,
        "urgencia_intervencion": COLUMNAS_URGENCIA,
        "indice_demanda_estructural": COLUMNAS_DEMANDA,
        "indice_infraestructura_estructural": COLUMNAS_INFRA,
        "deficit_infraestructura": COLUMNAS_DEFICIT,
        "indice_conectividad_estructural": COLUMNAS_CONECTIVIDAD,
        "indice_intermodalidad_estructural": COLUMNAS_INTERMODALIDAD,
        "indice_integracion_territorial": COLUMNAS_INTEGRACION,
        "indice_centralidad_estructural": COLUMNAS_CENTRALIDAD,
    }

    columnas: dict[str, str] = {}

    for nombre, candidatos in grupos.items():
        columna = encontrar_columna(cartera, candidatos)

        if columna is None:
            print(
                f"  {nombre:<40} NO DISPONIBLE -> se usará 0"
            )
        else:
            cartera[columna] = convertir_numerico(cartera[columna])
            columnas[nombre] = columna
            print(f"  {nombre:<40} OK -> {columna}")

    print()
    print(
        f"Componentes disponibles: "
        f"{len(columnas)} / {len(grupos)}"
    )

    return columnas


# =============================================================================
# COMPONENTE ESPACIAL
# =============================================================================

def preparar_componente_espacial(
    cartera: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    subtitulo("4", "PREPARANDO COMPONENTE ESPACIAL")

    trabajo = cartera.copy()

    metric = trabajo.to_crs(CRS_METRICO)

    # Para geometrías puntuales, centroid coincide con la ubicación.
    # Para polígonos/lineas se utiliza centroid en CRS métrico.
    trabajo["x_m"] = metric.geometry.centroid.x
    trabajo["y_m"] = metric.geometry.centroid.y

    print(f"X mínimo: {trabajo['x_m'].min():.2f} m")
    print(f"X máximo: {trabajo['x_m'].max():.2f} m")
    print(f"Y mínimo: {trabajo['y_m'].min():.2f} m")
    print(f"Y máximo: {trabajo['y_m'].max():.2f} m")
    print("Componente espacial: OK")

    return trabajo


# =============================================================================
# MATRIZ MULTICRITERIO
# =============================================================================

def construir_matriz_multicriterio(
    cartera: gpd.GeoDataFrame,
    columnas: dict[str, str],
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    subtitulo("5", "CONSTRUYENDO MATRIZ MULTICRITERIO V5")

    def obtener_norm(nombre: str) -> np.ndarray:
        if nombre in columnas:
            return normalizar_minmax(
                cartera[columnas[nombre]]
            ).to_numpy(dtype=float)

        return np.full(len(cartera), 0.5, dtype=float)

    # -------------------------------------------------------------------------
    # Espacio
    # -------------------------------------------------------------------------
    scaler_xy = RobustScaler()
    xy = scaler_xy.fit_transform(
        cartera[["x_m", "y_m"]]
    )
    xy = np.nan_to_num(xy)

    # -------------------------------------------------------------------------
    # Variables territoriales
    # -------------------------------------------------------------------------
    nombres = [
        "x_m",
        "y_m",
        "demanda",
        "deficit",
        "conectividad",
        "intermodalidad",
        "integracion",
        "centralidad",
        "impacto",
    ]

    matriz = pd.DataFrame(
        {
            "x_m": xy[:, 0],
            "y_m": xy[:, 1],
            "demanda": obtener_norm(
                "indice_demanda_estructural"
            ),
            "deficit": obtener_norm(
                "deficit_infraestructura"
            ),
            "conectividad": obtener_norm(
                "indice_conectividad_estructural"
            ),
            "intermodalidad": obtener_norm(
                "indice_intermodalidad_estructural"
            ),
            "integracion": obtener_norm(
                "indice_integracion_territorial"
            ),
            "centralidad": obtener_norm(
                "indice_centralidad_estructural"
            ),
            "impacto": obtener_norm(
                "impacto_potencial"
            ),
        }
    )

    # Cada dimensión espacial recibe la mitad del peso espacial.
    pesos = {
        "x_m": PESO_ESPACIAL / 2.0,
        "y_m": PESO_ESPACIAL / 2.0,
        "demanda": PESO_TERRITORIAL * PESO_DEMANDA,
        "deficit": PESO_TERRITORIAL * PESO_DEFICIT,
        "conectividad": PESO_TERRITORIAL * PESO_CONECTIVIDAD,
        "intermodalidad": PESO_TERRITORIAL * PESO_INTERMODALIDAD,
        "integracion": PESO_TERRITORIAL * PESO_INTEGRACION,
        "centralidad": PESO_TERRITORIAL * PESO_CENTRALIDAD,
        "impacto": PESO_TERRITORIAL * PESO_IMPACTO,
    }

    suma = sum(pesos.values())

    if not math.isclose(suma, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"La suma de pesos debe ser 1.0. Actual: {suma}"
        )

    # Estandarización robusta por dimensión.
    # Se realiza después de min-max para evitar que escalas originales
    # contaminen el clustering.
    scaler = RobustScaler()
    X = scaler.fit_transform(matriz)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # La distancia euclídea debe reflejar los pesos mediante sqrt(p).
    for i, nombre in enumerate(nombres):
        X[:, i] *= math.sqrt(pesos[nombre])

    print(f"Proyectos            : {len(cartera)}")
    print(f"Variables            : {len(nombres)}")
    print(f"Peso espacial        : {PESO_ESPACIAL:.0%}")
    print(f"Peso territorial     : {PESO_TERRITORIAL:.0%}")
    print(f"Suma de pesos        : {suma:.4f}")
    print()
    print("Pesos:")
    for nombre in nombres:
        print(f"  {nombre:<18}: {pesos[nombre]:.4f}")

    return X, nombres, pesos


# =============================================================================
# CLUSTERS
# =============================================================================

def calcular_centroides(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    centroides: dict[int, np.ndarray] = {}

    for cluster in sorted(np.unique(labels)):
        indices = np.where(labels == cluster)[0]

        if len(indices):
            centroides[int(cluster)] = X[indices].mean(axis=0)

    return centroides


def reparar_clusters_pequenos(
    X: np.ndarray,
    labels: np.ndarray,
    minimo: int,
) -> np.ndarray:
    """
    Repara clusters pequeños de forma determinística.

    Regla:
    - toma el cluster más pequeño;
    - reasigna sus proyectos, uno por uno, al cluster válido más cercano;
    - actualiza centroides después de cada reasignación;
    - nunca utiliza como receptor un cluster que quede por debajo del mínimo.
    """
    labels = labels.copy()

    while True:
        conteo = pd.Series(labels).value_counts()
        pequenos = (
            conteo[conteo < minimo]
            .sort_values()
        )

        if pequenos.empty:
            break

        cluster_pequeno = int(pequenos.index[0])

        indices = np.where(
            labels == cluster_pequeno
        )[0]

        centroides = calcular_centroides(X, labels)

        for idx in indices:
            conteo = pd.Series(labels).value_counts()

            receptores = [
                int(cluster)
                for cluster, cantidad in conteo.items()
                if int(cluster) != cluster_pequeno
                and int(cantidad) >= minimo
            ]

            if not receptores:
                raise RuntimeError(
                    "No existe cluster receptor válido para reparar "
                    "un cluster pequeño."
                )

            candidatos = []

            for destino in sorted(receptores):
                distancia = float(
                    np.linalg.norm(
                        X[idx] - centroides[destino]
                    )
                )
                candidatos.append(
                    (distancia, destino)
                )

            candidatos.sort(
                key=lambda item: (item[0], item[1])
            )

            labels[idx] = candidatos[0][1]
            centroides = calcular_centroides(X, labels)

        if not (
            pd.Series(labels).value_counts() < minimo
        ).any():
            break

    # Renumeración estable.
    mapa = {
        viejo: nuevo
        for nuevo, viejo in enumerate(
            sorted(np.unique(labels)),
            start=1,
        )
    }

    return np.array(
        [mapa[int(x)] for x in labels],
        dtype=int,
    )


def evaluar_k(
    X: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float, int, int, float]:
    modelo = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=50,
        max_iter=500,
        algorithm="lloyd",
    )

    labels_originales = modelo.fit_predict(X)

    labels_finales = reparar_clusters_pequenos(
        X,
        labels_originales,
        MIN_PROYECTOS_ESCENARIO,
    )

    conteo = pd.Series(labels_finales).value_counts()

    minimo = int(conteo.min())
    maximo = int(conteo.max())
    k_final = len(np.unique(labels_finales))

    if k_final < 2:
        silhouette = -1.0
    else:
        silhouette = float(
            silhouette_score(
                X,
                labels_finales,
            )
        )

    # Penalización pequeña por pedir muchos escenarios.
    penalizacion = 0.003 * max(k_final - K_MIN, 0)
    score = silhouette - penalizacion

    return (
        labels_finales,
        silhouette,
        minimo,
        maximo,
        score,
    )


def seleccionar_k(
    X: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    subtitulo("6", "SELECCIONANDO CANTIDAD DE ESCENARIOS")

    n = len(X)
    k_max_real = min(
        K_MAX,
        n // MIN_PROYECTOS_ESCENARIO,
    )

    resultados = []
    mejor: tuple[float, int, np.ndarray] | None = None

    for k in range(K_MIN, k_max_real + 1):
        (
            labels,
            silhouette,
            minimo,
            maximo,
            score,
        ) = evaluar_k(X, k)

        k_final = len(np.unique(labels))

        resultados.append(
            {
                "k_solicitado": k,
                "k_final": k_final,
                "silhouette": silhouette,
                "min_proyectos": minimo,
                "max_proyectos": maximo,
                "score_seleccion": score,
                "cumple_minimo": minimo >= MIN_PROYECTOS_ESCENARIO,
            }
        )

        print(
            f"K={k:2d} | "
            f"K final={k_final:2d} | "
            f"silhouette={silhouette:.4f} | "
            f"mín={minimo:3d} | "
            f"máx={maximo:3d} | "
            f"score={score:.4f}"
        )

        candidato = (score, k, labels)

        if mejor is None or candidato[0] > mejor[0]:
            mejor = candidato

    if mejor is None:
        raise RuntimeError(
            "No se pudo seleccionar una solución."
        )

    evaluacion = pd.DataFrame(resultados)

    print()
    print(f"K seleccionado     : {mejor[1]}")
    print(f"K final             : {len(np.unique(mejor[2]))}")
    print(f"Score de selección  : {mejor[0]:.4f}")
    print(
        f"Mínimo por escenario: "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )

    return (
        int(mejor[1]),
        mejor[2],
        evaluacion,
    )


# =============================================================================
# INDICADORES DE ESCENARIO
# =============================================================================

def obtener_indicador(
    cartera: pd.DataFrame,
    columnas: dict[str, str],
    nombre: str,
    indices: pd.Index,
) -> float:
    columna = columnas.get(nombre)

    if columna is None:
        return 50.0

    return media_normalizada(
        cartera[columna],
        indices,
    )


def construir_perfil(
    indicadores: dict[str, float],
) -> tuple[str, str, str]:
    """
    Devuelve:
    - tipo de escenario
    - dimensión dominante
    - dimensiones prioritarias
    """
    dimensiones = {
        "DEMANDA": indicadores["demanda"],
        "DEFICIT": indicadores["deficit"],
        "IMPACTO": indicadores["impacto"],
        "CONECTIVIDAD": indicadores["conectividad"],
        "INTERMODALIDAD": indicadores["intermodalidad"],
        "INTEGRACION": indicadores["integracion"],
        "CENTRALIDAD": indicadores["centralidad"],
    }

    ordenadas = sorted(
        dimensiones.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    primera, segunda = ordenadas[0], ordenadas[1]

    if (
        primera[1] - segunda[1]
        <= MARGEN_PERFIL_INTEGRADO
    ):
        tipo = "ESCENARIO_INTEGRADO"
    else:
        mapa_tipo = {
            "DEMANDA": "ESCENARIO_DEMANDA",
            "DEFICIT": "ESCENARIO_DEFICIT",
            "IMPACTO": "ESCENARIO_IMPACTO",
            "CONECTIVIDAD": "ESCENARIO_CONECTIVIDAD",
            "INTERMODALIDAD": "ESCENARIO_INTERMODAL",
            "INTEGRACION": "ESCENARIO_INTEGRADO",
            "CENTRALIDAD": "ESCENARIO_CENTRALIDAD",
        }
        tipo = mapa_tipo[primera[0]]

    dimensiones_prioritarias = ", ".join(
        nombre for nombre, _ in ordenadas[:3]
    )

    return (
        tipo,
        primera[0],
        dimensiones_prioritarias,
    )


def construir_escenarios(
    cartera: gpd.GeoDataFrame,
    columnas: dict[str, str],
    labels: np.ndarray,
) -> pd.DataFrame:
    subtitulo("7", "CONSTRUYENDO ESCENARIOS TERRITORIALES")

    trabajo = cartera.copy()
    trabajo["cluster_territorial"] = labels

    filas = []

    for cluster in sorted(trabajo["cluster_territorial"].unique()):
        subset = trabajo[
            trabajo["cluster_territorial"] == cluster
        ]

        indices = subset.index
        cantidad = len(subset)

        indicadores = {
            "impacto": obtener_indicador(
                trabajo, columnas, "impacto_potencial", indices
            ),
            "demanda": obtener_indicador(
                trabajo, columnas, "indice_demanda_estructural", indices
            ),
            "deficit": obtener_indicador(
                trabajo, columnas, "deficit_infraestructura", indices
            ),
            "conectividad": obtener_indicador(
                trabajo, columnas, "indice_conectividad_estructural", indices
            ),
            "intermodalidad": obtener_indicador(
                trabajo, columnas, "indice_intermodalidad_estructural", indices
            ),
            "integracion": obtener_indicador(
                trabajo, columnas, "indice_integracion_territorial", indices
            ),
            "centralidad": obtener_indicador(
                trabajo, columnas, "indice_centralidad_estructural", indices
            ),
            "urgencia": obtener_indicador(
                trabajo, columnas, "urgencia_intervencion", indices
            ),
            "score_cartera": obtener_indicador(
                trabajo, columnas, "score_cartera", indices
            ),
            "prioridad_territorial": obtener_indicador(
                trabajo, columnas, "score_prioridad_territorial", indices
            ),
        }

        # ---------------------------------------------------------------------
        # Complementariedad
        # ---------------------------------------------------------------------
        complementariedad = float(
            np.mean(
                [
                    indicadores["conectividad"],
                    indicadores["intermodalidad"],
                    indicadores["integracion"],
                    indicadores["centralidad"],
                ]
            )
        )

        # ---------------------------------------------------------------------
        # Score de escenario.
        # No depende de cantidad de proyectos.
        # ---------------------------------------------------------------------
        score = (
            indicadores["demanda"] * PESO_SCORE_DEMANDA
            + indicadores["deficit"] * PESO_SCORE_DEFICIT
            + indicadores["impacto"] * PESO_SCORE_IMPACTO
            + indicadores["conectividad"] * PESO_SCORE_CONECTIVIDAD
            + indicadores["intermodalidad"] * PESO_SCORE_INTERMODALIDAD
            + indicadores["integracion"] * PESO_SCORE_INTEGRACION
            + indicadores["centralidad"] * PESO_SCORE_CENTRALIDAD
            + indicadores["urgencia"] * PESO_SCORE_URGENCIA
            + indicadores["score_cartera"] * PESO_SCORE_CARTERA
        )

        tipo, dominante, dimensiones_prioritarias = construir_perfil(
            indicadores
        )

        # ---------------------------------------------------------------------
        # Diagnóstico
        # ---------------------------------------------------------------------
        if indicadores["urgencia"] >= 70:
            diagnostico = "ALTA_URGENCIA"
        elif indicadores["impacto"] >= 70:
            diagnostico = "ALTO_IMPACTO_POTENCIAL"
        elif indicadores["demanda"] >= 70:
            diagnostico = "ALTA_DEMANDA_ESTRUCTURAL"
        elif indicadores["deficit"] >= 70:
            diagnostico = "ALTO_DEFICIT_ATENDIDO"
        elif complementariedad >= 65:
            diagnostico = "ALTA_COMPLEMENTARIEDAD"
        else:
            diagnostico = "INTERVENCION_TERRITORIAL_MEDIA"

        # ---------------------------------------------------------------------
        # Horizonte
        # ---------------------------------------------------------------------
        if (
            indicadores["urgencia"] >= 70
            or indicadores["deficit"] >= 70
        ):
            horizonte = "CORTO_PLAZO"
        elif (
            indicadores["impacto"] >= 65
            or indicadores["demanda"] >= 65
        ):
            horizonte = "MEDIANO_PLAZO"
        else:
            horizonte = "LARGO_PLAZO"

        # ---------------------------------------------------------------------
        # Prioridad
        # ---------------------------------------------------------------------
        if score >= 65:
            prioridad = "PRIORIDAD_1_CRITICA"
        elif score >= 58:
            prioridad = "PRIORIDAD_2_ALTA"
        elif score >= 45:
            prioridad = "PRIORIDAD_3_MEDIA"
        else:
            prioridad = "PRIORIDAD_4_BAJA"

        diagnostico_texto = (
            f"{diagnostico}. El escenario agrupa {cantidad} proyectos. "
            f"Score territorial: {score:.2f}. "
            f"Dimensión dominante: {dominante}. "
            f"Demanda: {indicadores['demanda']:.2f}; "
            f"déficit: {indicadores['deficit']:.2f}; "
            f"impacto: {indicadores['impacto']:.2f}; "
            f"complementariedad: {complementariedad:.2f}."
        )

        objetivo = (
            f"Orientar inversiones hacia {dominante.lower()}, "
            f"articulando las dimensiones "
            f"{dimensiones_prioritarias.lower()}."
        )

        justificacion = (
            f"El escenario presenta un perfil {tipo.replace('ESCENARIO_', '').lower()} "
            f"con predominio de {dominante.lower()}. "
            f"La evaluación combina demanda, déficit, impacto y "
            f"capacidades de conectividad e integración."
        )

        filas.append(
            {
                "cluster_territorial": int(cluster),
                "cantidad_proyectos": cantidad,
                "impacto_territorial": indicadores["impacto"],
                "deficit_atendido": indicadores["deficit"],
                "demanda_cubierta": indicadores["demanda"],
                "complementariedad": complementariedad,
                "urgencia_promedio": indicadores["urgencia"],
                "score_cartera_promedio": indicadores["score_cartera"],
                "prioridad_territorial_promedio": indicadores[
                    "prioridad_territorial"
                ],
                "conectividad_promedio": indicadores["conectividad"],
                "intermodalidad_promedio": indicadores["intermodalidad"],
                "integracion_promedio": indicadores["integracion"],
                "centralidad_promedio": indicadores["centralidad"],
                "score_escenario": round(score, 4),
                "prioridad_escenario": prioridad,
                "tipo_escenario": tipo,
                "dimension_dominante": dominante,
                "horizonte_escenario": horizonte,
                "diagnostico_escenario": diagnostico,
                "objetivo_escenario": objetivo,
                "justificacion_escenario": justificacion,
                "dimensiones_prioritarias": dimensiones_prioritarias,
                "diagnostico_detallado": diagnostico_texto,
            }
        )

    escenarios = pd.DataFrame(filas)

    escenarios = escenarios.sort_values(
        ["score_escenario", "cantidad_proyectos"],
        ascending=[False, False],
    ).reset_index(drop=True)

    escenarios["ranking_escenario"] = np.arange(
        1,
        len(escenarios) + 1,
    )

    escenarios["escenario_id"] = [
        f"AMBA-E{i:03d}"
        for i in escenarios["ranking_escenario"]
    ]

    escenarios["escenario_nombre"] = (
        escenarios["escenario_id"]
        + " - "
        + escenarios["tipo_escenario"].str.replace(
            "ESCENARIO_",
            "",
            regex=False,
        )
    )

    # Cobertura se conserva como indicador descriptivo,
    # pero NO entra en score_escenario.
    escenarios["cobertura_territorial"] = (
        escenarios["cantidad_proyectos"]
        / max(len(trabajo), 1)
        * 100.0
    ).round(4)

    print()
    print("Distribución final:")
    print(
        escenarios[
            [
                "escenario_id",
                "cantidad_proyectos",
                "score_escenario",
                "tipo_escenario",
                "dimension_dominante",
                "prioridad_escenario",
            ]
        ].to_string(index=False)
    )

    return escenarios


# =============================================================================
# ASIGNACIÓN
# =============================================================================

def asignar_escenarios(
    cartera: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    labels: np.ndarray,
) -> gpd.GeoDataFrame:
    subtitulo("8", "ASIGNANDO ESCENARIOS A PROYECTOS")

    resultado = cartera.copy()
    resultado["cluster_territorial"] = labels

    mapa = (
        escenarios
        .set_index("cluster_territorial")
        .to_dict(orient="index")
    )

    columnas = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "dimension_dominante",
        "horizonte_escenario",
        "diagnostico_escenario",
        "objetivo_escenario",
        "justificacion_escenario",
        "dimensiones_prioritarias",
    ]

    for columna in columnas:
        resultado[columna] = resultado[
            "cluster_territorial"
        ].map(
            lambda cluster: mapa[int(cluster)][columna]
        )

    asignados = int(
        resultado["escenario_id"].notna().sum()
    )

    print(
        f"Proyectos asignados: {asignados} / {len(resultado)}"
    )

    if asignados != len(resultado):
        raise RuntimeError(
            "No todos los proyectos fueron asignados."
        )

    return resultado


# =============================================================================
# COHESIÓN
# =============================================================================

def calcular_metricas_cohesion(
    X: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    subtitulo("9", "CALCULANDO MÉTRICAS DE COHESIÓN")

    muestras = silhouette_samples(
        X,
        labels,
    )

    filas = []

    for cluster in sorted(np.unique(labels)):
        indices = np.where(labels == cluster)[0]

        valores = muestras[indices]

        # Distancia media al centroide del propio cluster.
        centroide = X[indices].mean(axis=0)

        distancias = np.linalg.norm(
            X[indices] - centroide,
            axis=1,
        )

        filas.append(
            {
                "cluster_territorial": int(cluster),
                "cantidad_proyectos": int(len(indices)),
                "silhouette_promedio": float(np.mean(valores)),
                "silhouette_minima": float(np.min(valores)),
                "silhouette_maxima": float(np.max(valores)),
                "distancia_centroide_promedio": float(
                    np.mean(distancias)
                ),
                "dispersion_std": float(
                    np.std(distancias)
                ),
            }
        )

    metricas = pd.DataFrame(filas)

    print(
        metricas.to_string(index=False)
    )

    return metricas


# =============================================================================
# GEOMETRÍAS
# =============================================================================

def construir_geometrias(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
) -> gpd.GeoDataFrame:
    subtitulo("10", "CONSTRUYENDO GEOMETRÍAS DE ESCENARIOS")

    metric = proyectos.to_crs(CRS_METRICO)

    filas = []

    for escenario_id, grupo in metric.groupby("escenario_id"):
        union = grupo.geometry.union_all()

        if union.geom_type in ("Point", "MultiPoint"):
            union = union.buffer(750)
        elif union.geom_type in ("LineString", "MultiLineString"):
            union = union.buffer(250)

        filas.append(
            {
                "escenario_id": escenario_id,
                "geometry": union,
            }
        )

    geometrias = gpd.GeoDataFrame(
        filas,
        geometry="geometry",
        crs=CRS_METRICO,
    )

    columnas = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "dimension_dominante",
        "horizonte_escenario",
        "cantidad_proyectos",
        "impacto_territorial",
        "cobertura_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
        "urgencia_promedio",
        "score_cartera_promedio",
        "prioridad_territorial_promedio",
        "conectividad_promedio",
        "intermodalidad_promedio",
        "integracion_promedio",
        "centralidad_promedio",
        "diagnostico_escenario",
        "objetivo_escenario",
        "justificacion_escenario",
        "dimensiones_prioritarias",
    ]

    geometrias = geometrias.merge(
        escenarios[columnas],
        on="escenario_id",
        how="left",
    )

    geometrias = geometrias.to_crs(CRS_GEOGRAFICO)

    print(f"Geometrías construidas: {len(geometrias)}")

    return geometrias


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

def validar_final(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    geometrias: gpd.GeoDataFrame,
    metricas: pd.DataFrame,
) -> None:
    subtitulo("11", "VALIDACIÓN FINAL")

    columnas_obligatorias = [
        "escenario_id",
        "escenario_nombre",
        "ranking_escenario",
        "cantidad_proyectos",
        "impacto_territorial",
        "cobertura_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
        "score_escenario",
        "prioridad_escenario",
        "tipo_escenario",
        "dimension_dominante",
        "horizonte_escenario",
        "diagnostico_escenario",
        "objetivo_escenario",
        "justificacion_escenario",
        "dimensiones_prioritarias",
    ]

    for columna in columnas_obligatorias:
        if columna not in escenarios.columns:
            raise ValueError(
                f"Falta columna obligatoria: {columna}"
            )

        nulos = int(escenarios[columna].isna().sum())
        print(f"{columna:<38} nulos={nulos}")

        if nulos:
            raise ValueError(
                f"La columna {columna} contiene nulos."
            )

    if len(proyectos) != int(
        escenarios["cantidad_proyectos"].sum()
    ):
        raise ValueError(
            "La suma de proyectos por escenario no coincide "
            "con la cartera."
        )

    asignados = int(
        proyectos["escenario_id"].notna().sum()
    )

    print()
    print(
        f"Asignaciones: {asignados}/{len(proyectos)}"
    )

    if asignados != len(proyectos):
        raise ValueError(
            "No todos los proyectos tienen escenario."
        )

    minimo = int(
        escenarios["cantidad_proyectos"].min()
    )
    maximo = int(
        escenarios["cantidad_proyectos"].max()
    )

    print(
        f"Proyectos por escenario: "
        f"mín={minimo} | máx={maximo}"
    )

    if minimo < MIN_PROYECTOS_ESCENARIO:
        raise ValueError(
            f"Existe un escenario por debajo de "
            f"{MIN_PROYECTOS_ESCENARIO} proyectos."
        )

    if len(geometrias) != len(escenarios):
        raise ValueError(
            "Cantidad de geometrías != cantidad de escenarios."
        )

    if geometrias.geometry.isna().any():
        raise ValueError("Existen geometrías nulas.")

    if geometrias.geometry.is_empty.any():
        raise ValueError("Existen geometrías vacías.")

    if len(metricas) != len(escenarios):
        raise ValueError(
            "Cantidad de métricas != cantidad de escenarios."
        )

    if not np.isfinite(
        escenarios["score_escenario"].to_numpy()
    ).all():
        raise ValueError(
            "Existen scores de escenario no finitos."
        )

    print()
    print("VALIDACIÓN FINAL: OK")


# =============================================================================
# RANKING
# =============================================================================

def mostrar_ranking(
    escenarios: pd.DataFrame,
) -> None:
    subtitulo("12", "RANKING DE ESCENARIOS")

    columnas = [
        "ranking_escenario",
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_dominante",
        "horizonte_escenario",
        "score_escenario",
        "prioridad_escenario",
        "impacto_territorial",
        "deficit_atendido",
        "demanda_cubierta",
        "complementariedad",
    ]

    print(
        escenarios[columnas].to_string(index=False)
    )


# =============================================================================
# RESUMEN JSON
# =============================================================================

def construir_resumen(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
    metricas: pd.DataFrame,
) -> dict[str, Any]:
    subtitulo("13", "CONSTRUYENDO RESUMEN JSON")

    resumen: dict[str, Any] = {
        "version": "V5",
        "proyecto": "Construcción de escenarios territoriales AMBA",
        "fecha_ejecucion": pd.Timestamp.now().isoformat(),
        "parametros": {
            "k_min": K_MIN,
            "k_max": K_MAX,
            "min_proyectos_escenario": MIN_PROYECTOS_ESCENARIO,
            "random_state": RANDOM_STATE,
            "crs": CRS_GEOGRAFICO,
            "crs_metrico": CRS_METRICO,
            "margen_perfil_integrado": MARGEN_PERFIL_INTEGRADO,
        },
        "pesos_clustering": {
            "espacial": PESO_ESPACIAL,
            "territorial": PESO_TERRITORIAL,
            "demanda": PESO_DEMANDA,
            "deficit": PESO_DEFICIT,
            "conectividad": PESO_CONECTIVIDAD,
            "intermodalidad": PESO_INTERMODALIDAD,
            "integracion": PESO_INTEGRACION,
            "centralidad": PESO_CENTRALIDAD,
            "impacto": PESO_IMPACTO,
        },
        "pesos_score_escenario": {
            "demanda": PESO_SCORE_DEMANDA,
            "deficit": PESO_SCORE_DEFICIT,
            "impacto": PESO_SCORE_IMPACTO,
            "conectividad": PESO_SCORE_CONECTIVIDAD,
            "intermodalidad": PESO_SCORE_INTERMODALIDAD,
            "integracion": PESO_SCORE_INTEGRACION,
            "centralidad": PESO_SCORE_CENTRALIDAD,
            "urgencia": PESO_SCORE_URGENCIA,
            "score_cartera": PESO_SCORE_CARTERA,
        },
        "cantidad_proyectos": int(len(proyectos)),
        "cantidad_escenarios": int(len(escenarios)),
        "distribucion_proyectos": {
            str(row["escenario_id"]): int(row["cantidad_proyectos"])
            for _, row in escenarios.iterrows()
        },
        "prioridades": {
            str(k): int(v)
            for k, v in escenarios[
                "prioridad_escenario"
            ].value_counts().items()
        },
        "horizontes": {
            str(k): int(v)
            for k, v in escenarios[
                "horizonte_escenario"
            ].value_counts().items()
        },
        "tipos": {
            str(k): int(v)
            for k, v in escenarios[
                "tipo_escenario"
            ].value_counts().items()
        },
        "dimensiones_dominantes": {
            str(k): int(v)
            for k, v in escenarios[
                "dimension_dominante"
            ].value_counts().items()
        },
        "diagnosticos": {
            str(k): int(v)
            for k, v in escenarios[
                "diagnostico_escenario"
            ].value_counts().items()
        },
        "evaluacion_k": evaluacion_k.to_dict(
            orient="records"
        ),
        "metricas_cohesion": metricas.to_dict(
            orient="records"
        ),
        "top_escenarios": [],
    }

    for _, row in escenarios.head(10).iterrows():
        resumen["top_escenarios"].append(
            {
                "ranking": safe_int(
                    row["ranking_escenario"]
                ),
                "escenario_id": str(
                    row["escenario_id"]
                ),
                "score": safe_float(
                    row["score_escenario"]
                ),
                "cantidad_proyectos": safe_int(
                    row["cantidad_proyectos"]
                ),
                "tipo": str(
                    row["tipo_escenario"]
                ),
                "dimension_dominante": str(
                    row["dimension_dominante"]
                ),
                "horizonte": str(
                    row["horizonte_escenario"]
                ),
                "prioridad": str(
                    row["prioridad_escenario"]
                ),
            }
        )

    return resumen


# =============================================================================
# EXPORTACIÓN
# =============================================================================

def guardar_archivos(
    escenarios: gpd.GeoDataFrame,
    proyectos: gpd.GeoDataFrame,
    resumen: dict[str, Any],
    evaluacion_k: pd.DataFrame,
    metricas: pd.DataFrame,
) -> None:
    subtitulo("14", "GUARDANDO ARCHIVOS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Escenarios
    escenarios_parquet = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.parquet"
    )
    escenarios.to_parquet(
        escenarios_parquet,
        index=False,
    )
    print(f"Parquet escenarios:\n{escenarios_parquet}")

    escenarios_csv = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.csv"
    )
    escenarios.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        escenarios_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"CSV escenarios:\n{escenarios_csv}")

    # GeoPackage
    gpkg = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba.gpkg"
    )

    if gpkg.exists():
        try:
            gpkg.unlink()
        except Exception:
            pass

    escenarios.to_file(
        gpkg,
        layer="escenarios",
        driver="GPKG",
    )
    print(f"GeoPackage:\n{gpkg}")

    # Proyectos
    proyectos_parquet = (
        OUTPUT_DIR
        / "proyectos_escenarios_territoriales_amba.parquet"
    )
    proyectos.to_parquet(
        proyectos_parquet,
        index=False,
    )
    print(f"Parquet proyectos:\n{proyectos_parquet}")

    proyectos_csv = (
        OUTPUT_DIR
        / "proyectos_escenarios_territoriales_amba.csv"
    )
    proyectos.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        proyectos_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"CSV proyectos:\n{proyectos_csv}")

    # JSON
    json_path = (
        OUTPUT_DIR
        / "escenarios_territoriales_amba_resumen.json"
    )
    with json_path.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
        )
    print(f"JSON:\n{json_path}")

    # K
    evaluacion_path = (
        OUTPUT_DIR
        / "evaluacion_numero_escenarios.csv"
    )
    evaluacion_k.to_csv(
        evaluacion_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Evaluación K:\n{evaluacion_path}")

    # Diagnóstico
    diagnostico_path = (
        OUTPUT_DIR
        / "diagnostico_escenarios.csv"
    )
    escenarios.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        diagnostico_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Diagnóstico:\n{diagnostico_path}")

    # Cohesión
    cohesion_path = (
        OUTPUT_DIR
        / "metricas_cohesion_escenarios.csv"
    )
    metricas.to_csv(
        cohesion_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Cohesión:\n{cohesion_path}")


# =============================================================================
# MAPAS
# =============================================================================

def importar_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print(
            "ADVERTENCIA: Matplotlib no disponible. "
            "Se omiten mapas y gráficos."
        )
        return None


def generar_mapa(
    gdf: gpd.GeoDataFrame,
    columna: str,
    titulo_mapa: str,
    archivo: str,
) -> None:
    if gdf.empty:
        return

    plt = importar_matplotlib()

    if plt is None:
        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    try:
        gdf.plot(
            ax=ax,
            column=columna,
            legend=True,
            alpha=0.75,
        )
    except Exception:
        gdf.plot(
            ax=ax,
            alpha=0.75,
        )

    ax.set_title(
        titulo_mapa,
        fontsize=15,
    )
    ax.set_axis_off()

    path = OUTPUT_DIR / archivo

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Mapa: {path}")


def generar_mapa_proyectos(
    proyectos: gpd.GeoDataFrame,
) -> None:
    plt = importar_matplotlib()

    if plt is None:
        return

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    proyectos.plot(
        ax=ax,
        column="escenario_id",
        legend=True,
        markersize=12,
        alpha=0.75,
    )

    ax.set_title(
        "Proyectos asignados a escenarios territoriales AMBA",
        fontsize=15,
    )

    ax.set_axis_off()

    path = (
        OUTPUT_DIR
        / "05_mapa_proyectos_por_escenario.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Mapa: {path}")


# =============================================================================
# GRÁFICOS
# =============================================================================

def generar_graficos(
    escenarios: pd.DataFrame,
    evaluacion_k: pd.DataFrame,
) -> None:
    plt = importar_matplotlib()

    if plt is None:
        return

    # 06 - Demanda vs déficit
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.scatter(
        escenarios["demanda_cubierta"],
        escenarios["deficit_atendido"],
        s=100,
    )

    for _, row in escenarios.iterrows():
        ax.annotate(
            row["escenario_id"],
            (
                row["demanda_cubierta"],
                row["deficit_atendido"],
            ),
        )

    ax.set_xlabel("Demanda cubierta")
    ax.set_ylabel("Déficit atendido")
    ax.set_title("Demanda vs déficit atendido")
    ax.grid(alpha=0.25)

    path = (
        OUTPUT_DIR
        / "06_demanda_vs_deficit_atendido.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Gráfico: {path}")

    # 07 - Prioridad
    conteo = (
        escenarios["prioridad_escenario"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    conteo.plot(kind="bar", ax=ax)

    ax.set_title("Escenarios por prioridad")
    ax.set_ylabel("Cantidad")
    ax.tick_params(axis="x", rotation=30)

    path = (
        OUTPUT_DIR
        / "07_escenarios_por_prioridad.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Gráfico: {path}")

    # 08 - Horizonte
    conteo = (
        escenarios["horizonte_escenario"]
        .value_counts()
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    conteo.plot(kind="bar", ax=ax)

    ax.set_title("Escenarios por horizonte")
    ax.set_ylabel("Cantidad")
    ax.tick_params(axis="x", rotation=30)

    path = (
        OUTPUT_DIR
        / "08_escenarios_por_horizonte.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Gráfico: {path}")

    # 09 - Score
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.bar(
        escenarios["escenario_id"],
        escenarios["score_escenario"],
    )

    ax.set_title("Score de escenarios territoriales")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)

    path = (
        OUTPUT_DIR
        / "09_distribucion_score_escenarios.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Gráfico: {path}")

    # 10 - Silhouette
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        evaluacion_k["k_solicitado"],
        evaluacion_k["silhouette"],
        marker="o",
    )

    ax.set_xlabel("Cantidad de escenarios K")
    ax.set_ylabel("Silhouette")
    ax.set_title("Evaluación del número de escenarios")
    ax.grid(alpha=0.25)

    path = (
        OUTPUT_DIR
        / "10_evaluacion_silhouette_k.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Gráfico: {path}")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

def mostrar_resumen_final(
    proyectos: gpd.GeoDataFrame,
    escenarios: pd.DataFrame,
) -> None:
    titulo("27 - PROCESO FINALIZADO CORRECTAMENTE")

    print(
        f"Proyectos analizados     : {len(proyectos)}"
    )
    print(
        f"Escenarios territoriales : {len(escenarios)}"
    )

    print()
    print("DISTRIBUCIÓN DE PROYECTOS")

    for _, row in escenarios.iterrows():
        print(
            f"  {row['escenario_id']}: "
            f"{int(row['cantidad_proyectos']):3d} proyectos | "
            f"{row['tipo_escenario']:<25} | "
            f"dominante={row['dimension_dominante']:<15} | "
            f"score={row['score_escenario']:.2f}"
        )

    print()
    print("PRIORIDADES:")
    print(
        escenarios["prioridad_escenario"]
        .value_counts()
        .to_string()
    )

    print()
    print("HORIZONTES:")
    print(
        escenarios["horizonte_escenario"]
        .value_counts()
        .to_string()
    )

    print()
    print("TIPOS DE ESCENARIO:")
    print(
        escenarios["tipo_escenario"]
        .value_counts()
        .to_string()
    )

    print()
    print("DIMENSIONES DOMINANTES:")
    print(
        escenarios["dimension_dominante"]
        .value_counts()
        .to_string()
    )

    print()
    print("DIAGNÓSTICOS:")
    print(
        escenarios["diagnostico_escenario"]
        .value_counts()
        .to_string()
    )

    print()
    print("ARCHIVOS GENERADOS:")

    if OUTPUT_DIR.exists():
        for archivo in sorted(OUTPUT_DIR.iterdir()):
            if archivo.is_file():
                print(f"  {archivo.name}")

    print()
    print("SIGUIENTE ETAPA")
    print(
        "Evaluar los escenarios mediante simulación de impactos, "
        "cobertura, demanda, déficit, conectividad e interacción "
        "territorial para seleccionar escenarios estratégicos."
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    titulo(
        "27 - CONSTRUCCIÓN DE ESCENARIOS TERRITORIALES AMBA - V5"
    )

    print(f"Proyecto    : {PROJECT_DIR}")
    print(f"Entrada     : {INPUT_PATH}")
    print(f"Salida      : {OUTPUT_DIR}")
    print(f"CRS         : {CRS_GEOGRAFICO}")
    print(f"CRS métrico : {CRS_METRICO}")

    print()
    print("CONFIGURACIÓN")
    print(f"  K candidatos         : {K_MIN} - {K_MAX}")
    print(
        f"  Mínimo proyectos     : "
        f"{MIN_PROYECTOS_ESCENARIO}"
    )
    print(
        "  Método               : "
        "KMeans multicriterio + reparación restringida"
    )
    print(f"  Random state         : {RANDOM_STATE}")

    print()
    print("PESOS DEL CLUSTERING")
    print(f"  Componente espacial  : {PESO_ESPACIAL:.0%}")
    print(f"  Componente territorial: {PESO_TERRITORIAL:.0%}")

    # 1
    subtitulo(
        "1",
        "CARGANDO CARTERA DEL PROCESO 26",
    )
    cartera = cargar_cartera()
    print(f"Registros: {len(cartera)}")
    print(f"Columnas : {len(cartera.columns)}")
    print(f"CRS      : {cartera.crs}")

    # 2
    validar_entrada(cartera)

    # 3
    columnas = validar_componentes(cartera)

    # 4
    cartera = preparar_componente_espacial(cartera)

    # 5
    X, variables, pesos = construir_matriz_multicriterio(
        cartera,
        columnas,
    )

    # 6
    (
        k_seleccionado,
        labels,
        evaluacion_k,
    ) = seleccionar_k(X)

    # 7
    escenarios = construir_escenarios(
        cartera,
        columnas,
        labels,
    )

    # 8
    proyectos = asignar_escenarios(
        cartera,
        escenarios,
        labels,
    )

    # 9
    metricas = calcular_metricas_cohesion(
        X,
        labels,
    )

    # 10
    geometrias = construir_geometrias(
        proyectos,
        escenarios,
    )

    # 11
    validar_final(
        proyectos,
        escenarios,
        geometrias,
        metricas,
    )

    # 12
    mostrar_ranking(escenarios)

    # 13
    resumen = construir_resumen(
        proyectos,
        escenarios,
        evaluacion_k,
        metricas,
    )

    # 14
    guardar_archivos(
        geometrias,
        proyectos,
        resumen,
        evaluacion_k,
        metricas,
    )

    # 15
    subtitulo(
        "15",
        "GENERANDO MAPAS",
    )

    generar_mapa(
        geometrias,
        "ranking_escenario",
        "Escenarios territoriales AMBA",
        "01_mapa_escenarios_territoriales.png",
    )

    generar_mapa(
        geometrias,
        "score_escenario",
        "Score de escenarios territoriales",
        "02_mapa_prioridad_escenarios.png",
    )

    generar_mapa(
        geometrias,
        "cobertura_territorial",
        "Cobertura territorial por escenario",
        "03_mapa_cobertura_metropolitana.png",
    )

    generar_mapa(
        geometrias,
        "impacto_territorial",
        "Impacto territorial por escenario",
        "04_mapa_impacto_territorial.png",
    )

    generar_mapa_proyectos(proyectos)

    # 16
    subtitulo(
        "16",
        "GENERANDO GRÁFICOS",
    )

    generar_graficos(
        escenarios,
        evaluacion_k,
    )

    mostrar_resumen_final(
        proyectos,
        escenarios,
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
        print("=" * 80)
        print("ERROR DURANTE EL PROCESO 27 V5")
        print("=" * 80)
        print(f"{type(exc).__name__}: {exc}")
        raise
