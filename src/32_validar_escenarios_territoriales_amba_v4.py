# -*- coding: utf-8 -*-

"""
32 - VALIDACIÓN ANALÍTICA FINAL DE ESCENARIOS TERRITORIALES AMBA - V4

Este proceso toma como entrada la salida VALIDADA del proceso 31:

    data/processed/escenarios_territoriales_amba/
        escenarios_territoriales_amba_v4.parquet

Objetivo
--------
Realizar la validación analítica final de los escenarios territoriales
normalizados en el proceso 31, sin modificar la asignación proyecto ->
escenario, los indicadores originales ni las geometrías.

Principios
----------
- No modifica el parquet V4 de entrada.
- No modifica proyecto_id.
- No modifica escenario_id.
- No modifica indicadores originales.
- No modifica geometrías.
- Verifica unicidad de proyectos.
- Verifica cobertura territorial.
- Verifica consistencia interna de atributos de escenario.
- Verifica geometrías.
- Construye métricas agregadas por escenario.
- Construye ranking de escenarios.
- Construye matriz comparativa.
- Genera auditoría completa.
- Genera resumen JSON.
- Emite dictamen final.

Entrada
-------
    escenarios_territoriales_amba_v4.parquet

Salidas
-------
    ficha_escenarios_v4.csv
    ranking_escenarios_v4.csv
    matriz_escenarios_v4.csv
    auditoria_32_escenarios_territoriales_amba.csv
    resumen_32_escenarios_territoriales_amba.json
"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

VERSION = "V4.0"
PROCESO = 32

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_FILE = (
    INPUT_DIR
    / "escenarios_territoriales_amba_v4.parquet"
)

OUTPUT_FICHA = (
    INPUT_DIR
    / "ficha_escenarios_v4.csv"
)

OUTPUT_RANKING = (
    INPUT_DIR
    / "ranking_escenarios_v4.csv"
)

OUTPUT_MATRIZ = (
    INPUT_DIR
    / "matriz_escenarios_v4.csv"
)

OUTPUT_AUDITORIA = (
    INPUT_DIR
    / "auditoria_32_escenarios_territoriales_amba.csv"
)

OUTPUT_JSON = (
    INPUT_DIR
    / "resumen_32_escenarios_territoriales_amba.json"
)

EXPECTED_SCENARIOS_MIN = 6
EXPECTED_SCENARIOS_MAX = 12

MIN_PROJECTS_PER_SCENARIO = 8

EXPECTED_PROJECTS = 144

EXPECTED_CRS = "EPSG:4326"


# ============================================================================
# CAMPOS ESPERADOS
# ============================================================================

INDICATOR_CANDIDATES = {
    "indice_demanda": [
        "indice_demanda_estructural",
        "indice_demanda",
        "score_demanda",
    ],
    "deficit_infraestructura": [
        "deficit_infraestructura",
        "indice_deficit_infraestructura",
    ],
    "indice_conectividad": [
        "indice_conectividad_estructural",
        "indice_conectividad",
        "score_conectividad",
    ],
    "indice_intermodalidad": [
        "indice_intermodalidad_estructural",
        "indice_intermodalidad",
        "score_intermodalidad",
    ],
    "indice_integracion": [
        "indice_integracion_territorial",
        "indice_integracion",
        "score_integracion",
    ],
    "indice_centralidad": [
        "indice_centralidad_estructural",
        "indice_centralidad",
    ],
    "impacto_potencial": [
        "impacto_potencial",
        "score_impacto",
        "indice_impacto",
    ],
    "urgencia_intervencion": [
        "urgencia_intervencion",
        "score_urgencia",
        "indice_urgencia",
    ],
    "prioridad_territorial": [
        "score_prioridad_territorial",
        "prioridad_territorial",
    ],
    "score_cartera": [
        "score_cartera",
        "indice_cartera",
    ],
}


# ============================================================================
# UTILIDADES
# ============================================================================

def normalizar_nombre(nombre: str) -> str:
    """
    Normaliza nombres de columnas para resolver diferencias menores.
    """
    s = unicodedata.normalize("NFKD", str(nombre))
    s = "".join(
        c for c in s
        if not unicodedata.combining(c)
    )

    s = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        s
    )

    return s.strip("_").lower()


def resolver_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    requerida: bool = False,
):
    """
    Busca una columna por nombre exacto o normalizado.
    """

    directas = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidato in candidatos:
        if candidato.lower() in directas:
            return directas[candidato.lower()]

    normalizadas = {
        normalizar_nombre(c): c
        for c in df.columns
    }

    for candidato in candidatos:
        nombre = normalizar_nombre(candidato)

        if nombre in normalizadas:
            return normalizadas[nombre]

    if requerida:
        raise KeyError(
            "No se encontró ninguna columna de: "
            f"{candidatos}"
        )

    return None


def valor_valido(valor) -> bool:
    """
    Determina si un valor puede considerarse válido.
    """

    if valor is None:
        return False

    try:
        if pd.isna(valor):
            return False
    except Exception:
        pass

    if isinstance(valor, str):
        if not valor.strip():
            return False

    return True


def valores_unicos(series: pd.Series) -> list:
    """
    Devuelve valores únicos válidos preservando orden.
    """

    salida = []
    vistos = set()

    for valor in series:

        if not valor_valido(valor):
            continue

        clave = str(valor).strip()

        if clave not in vistos:
            vistos.add(clave)
            salida.append(valor)

    return salida


def moda_deterministica(series: pd.Series):
    """
    Moda determinística.

    Criterios:
        1. Mayor frecuencia.
        2. Primer valor observado como desempate.
    """

    valores = [
        valor
        for valor in series
        if valor_valido(valor)
    ]

    if not valores:
        return None, 0

    frecuencias = Counter(
        str(valor).strip()
        for valor in valores
    )

    primera_posicion = {}

    for posicion, valor in enumerate(valores):
        clave = str(valor).strip()

        if clave not in primera_posicion:
            primera_posicion[clave] = posicion

    ganador = sorted(
        frecuencias.keys(),
        key=lambda clave: (
            -frecuencias[clave],
            primera_posicion[clave],
        ),
    )[0]

    valor_final = next(
        valor
        for valor in valores
        if str(valor).strip() == ganador
    )

    return valor_final, frecuencias[ganador]


def convertir_numerico(series: pd.Series) -> pd.Series:
    """
    Conversión numérica robusta.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def estadisticas_numericas(
    grupo: pd.DataFrame,
    columna: str,
) -> dict:
    """
    Calcula estadísticas descriptivas de una columna.
    """

    if columna not in grupo.columns:
        return {
            "n": 0,
            "media": np.nan,
            "mediana": np.nan,
            "minimo": np.nan,
            "maximo": np.nan,
            "desvio": np.nan,
        }

    serie = convertir_numerico(
        grupo[columna]
    ).dropna()

    if serie.empty:
        return {
            "n": 0,
            "media": np.nan,
            "mediana": np.nan,
            "minimo": np.nan,
            "maximo": np.nan,
            "desvio": np.nan,
        }

    return {
        "n": int(len(serie)),
        "media": float(serie.mean()),
        "mediana": float(serie.median()),
        "minimo": float(serie.min()),
        "maximo": float(serie.max()),
        "desvio": float(
            serie.std(ddof=0)
        ),
    }


def normalizar_score_0_100(
    serie: pd.Series,
) -> pd.Series:
    """
    Normaliza una serie a 0-100 mediante min-max.

    Si todos los valores son iguales, asigna 50.
    """

    valores = convertir_numerico(
        serie
    )

    validos = valores.dropna()

    if validos.empty:
        return pd.Series(
            np.nan,
            index=serie.index,
        )

    minimo = float(validos.min())
    maximo = float(validos.max())

    if math.isclose(
        minimo,
        maximo,
    ):
        resultado = pd.Series(
            np.nan,
            index=serie.index,
        )

        resultado.loc[
            valores.notna()
        ] = 50.0

        return resultado

    return (
        (valores - minimo)
        / (maximo - minimo)
        * 100.0
    )


def convertir_jsonable(valor):
    """
    Convierte tipos NumPy/Pandas a tipos JSON.
    """

    if isinstance(valor, np.integer):
        return int(valor)

    if isinstance(valor, np.floating):
        if np.isnan(valor):
            return None
        return float(valor)

    if isinstance(valor, np.ndarray):
        return valor.tolist()

    if isinstance(valor, (list, tuple)):
        return [
            convertir_jsonable(v)
            for v in valor
        ]

    if isinstance(valor, dict):
        return {
            str(k): convertir_jsonable(v)
            for k, v in valor.items()
        }

    try:
        if pd.isna(valor):
            return None
    except Exception:
        pass

    return valor


# ============================================================================
# CARGA
# ============================================================================

def cargar_datos():
    print("=" * 88)
    print(
        "32 - VALIDACIÓN ANALÍTICA FINAL DE "
        "ESCENARIOS TERRITORIALES AMBA - V4"
    )
    print("=" * 88)

    print(
        f"Proyecto : {BASE_DIR}"
    )

    print(
        f"Entrada  : {INPUT_FILE}"
    )

    print(
        f"Salida   : {INPUT_DIR}"
    )

    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nNo existe la entrada del proceso 32:\n"
            f"{INPUT_FILE}\n\n"
            "Ejecutá primero el proceso 31."
        )

    print("=" * 88)
    print("CARGANDO SALIDA VALIDADA DEL PROCESO 31")
    print("=" * 88)

    print(
        f"Cargando: {INPUT_FILE}"
    )

    if gpd is not None:

        try:
            df = gpd.read_parquet(
                INPUT_FILE
            )

        except Exception:

            df = pd.read_parquet(
                INPUT_FILE
            )

    else:

        df = pd.read_parquet(
            INPUT_FILE
        )

    print(
        f"Registros : {len(df):,}"
    )

    print(
        f"Columnas  : {len(df.columns):,}"
    )

    if hasattr(df, "crs"):

        print(
            f"CRS       : {df.crs}"
        )

    return df


# ============================================================================
# RESOLUCIÓN DE CAMPOS
# ============================================================================

def resolver_campos(df: pd.DataFrame):

    campos = {}

    campos["proyecto"] = resolver_columna(
        df,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
        requerida=True,
    )

    campos["escenario"] = resolver_columna(
        df,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
        requerida=True,
    )

    campos["tipo"] = resolver_columna(
        df,
        [
            "tipo_escenario",
            "tipo",
        ],
        requerida=True,
    )

    campos["dimension"] = resolver_columna(
        df,
        [
            "dimension_dominante",
            "dimension",
        ],
        requerida=True,
    )

    campos["prioridad"] = resolver_columna(
        df,
        [
            "prioridad_escenario",
            "prioridad",
        ],
        requerida=False,
    )

    campos["geometria"] = None

    if (
        hasattr(df, "geometry")
        and "geometry" in df.columns
    ):
        campos["geometria"] = "geometry"

    campos["indicadores"] = {}

    for nombre, candidatos in INDICATOR_CANDIDATES.items():

        columna = resolver_columna(
            df,
            candidatos,
            requerida=False,
        )

        if columna is not None:
            campos["indicadores"][nombre] = columna

    print()
    print("=" * 88)
    print("CAMPOS RESUELTOS")
    print("=" * 88)

    for nombre in [
        "proyecto",
        "escenario",
        "tipo",
        "dimension",
        "prioridad",
        "geometria",
    ]:

        print(
            f"{nombre:20}: "
            f"{campos[nombre]}"
        )

    print()
    print("INDICADORES DISPONIBLES")

    if campos["indicadores"]:

        for nombre, columna in (
            campos["indicadores"].items()
        ):

            print(
                f"  {nombre:28}: "
                f"{columna}"
            )

    else:

        print(
            "  No se encontraron indicadores "
            "numéricos reconocibles."
        )

    return campos


# ============================================================================
# VALIDACIÓN ESTRUCTURAL
# ============================================================================

def validar_estructura(
    df: pd.DataFrame,
    campos: dict,
):

    errores = []
    advertencias = []

    proyecto = campos["proyecto"]
    escenario = campos["escenario"]

    n = len(df)

    print()
    print("=" * 88)
    print("VALIDACIÓN ESTRUCTURAL")
    print("=" * 88)

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    if n == 0:

        errores.append(
            "DATASET_EMPTY"
        )

    print(
        f"Registros                  : {n:,}"
    )

    if n != EXPECTED_PROJECTS:

        advertencias.append(
            "PROJECT_COUNT_EXPECTED_"
            f"{EXPECTED_PROJECTS}_FOUND_{n}"
        )

    # ------------------------------------------------------------------------
    # Proyecto
    # ------------------------------------------------------------------------

    null_proyectos = int(
        df[proyecto].isna().sum()
    )

    duplicados = int(
        df[proyecto].duplicated().sum()
    )

    print(
        f"Proyecto ID nulos         : "
        f"{null_proyectos}"
    )

    print(
        f"Proyecto ID duplicados    : "
        f"{duplicados}"
    )

    if null_proyectos:
        errores.append(
            f"PROJECT_ID_NULL:{null_proyectos}"
        )

    if duplicados:
        errores.append(
            f"PROJECT_ID_DUPLICATES:{duplicados}"
        )

    # ------------------------------------------------------------------------
    # Escenarios
    # ------------------------------------------------------------------------

    escenarios = [
        valor
        for valor in df[escenario].dropna().unique()
    ]

    escenarios = sorted(
        escenarios,
        key=str,
    )

    cantidad_escenarios = len(
        escenarios
    )

    print(
        f"Escenarios                : "
        f"{cantidad_escenarios}"
    )

    print(
        "Escenarios IDs            : "
        f"{', '.join(map(str, escenarios))}"
    )

    if not (
        EXPECTED_SCENARIOS_MIN
        <= cantidad_escenarios
        <= EXPECTED_SCENARIOS_MAX
    ):

        errores.append(
            "SCENARIO_COUNT_OUT_OF_RANGE:"
            f"{cantidad_escenarios}"
        )

    # ------------------------------------------------------------------------
    # Cobertura
    # ------------------------------------------------------------------------

    escenarios_nulos = int(
        df[escenario].isna().sum()
    )

    cobertura = (
        (n - escenarios_nulos) / n
        if n
        else 0.0
    )

    print(
        f"Cobertura escenarios      : "
        f"{cobertura:.2%}"
    )

    if escenarios_nulos:

        errores.append(
            f"SCENARIO_ID_NULL:{escenarios_nulos}"
        )

    # ------------------------------------------------------------------------
    # Tamaño de escenarios
    # ------------------------------------------------------------------------

    counts = (
        df
        .groupby(escenario)
        .size()
        .sort_index()
    )

    if not counts.empty:

        minimo = int(
            counts.min()
        )

        maximo = int(
            counts.max()
        )

        promedio = float(
            counts.mean()
        )

        desvio = float(
            counts.std(ddof=0)
        )

        cv = (
            desvio / promedio
            if promedio
            else 0.0
        )

    else:

        minimo = 0
        maximo = 0
        promedio = 0.0
        desvio = 0.0
        cv = 0.0

    print(
        f"Mínimo proyectos/escenario: "
        f"{minimo}"
    )

    print(
        f"Máximo proyectos/escenario: "
        f"{maximo}"
    )

    print(
        f"Promedio proyectos/escenario: "
        f"{promedio:.2f}"
    )

    print(
        f"CV tamaño escenarios       : "
        f"{cv:.4f}"
    )

    if not counts.empty:

        escenarios_pequenos = (
            counts[
                counts
                < MIN_PROJECTS_PER_SCENARIO
            ]
        )

        if not escenarios_pequenos.empty:

            errores.append(
                "SCENARIO_MIN_PROJECTS:"
                f"{escenarios_pequenos.to_dict()}"
            )

    # ------------------------------------------------------------------------
    # Geometrías
    # ------------------------------------------------------------------------

    geometria = campos["geometria"]

    geometria_info = {
        "disponible": False,
        "nulas": None,
        "vacias": None,
        "invalidas": None,
        "crs": None,
    }

    if geometria is not None:

        geometria_info["disponible"] = True

        try:

            serie_geom = df[
                geometria
            ]

            nulas = int(
                serie_geom.isna().sum()
            )

            vacias = int(
                serie_geom.is_empty.sum()
            )

            invalidas = int(
                (~serie_geom.is_valid).sum()
            )

            geometria_info["nulas"] = nulas
            geometria_info["vacias"] = vacias
            geometria_info["invalidas"] = invalidas

            if hasattr(df, "crs"):

                geometria_info["crs"] = (
                    str(df.crs)
                    if df.crs is not None
                    else None
                )

            print(
                f"Geometrías nulas         : "
                f"{nulas}"
            )

            print(
                f"Geometrías vacías        : "
                f"{vacias}"
            )

            print(
                f"Geometrías inválidas     : "
                f"{invalidas}"
            )

            print(
                f"CRS                       : "
                f"{geometria_info['crs']}"
            )

            if nulas:
                errores.append(
                    f"GEOMETRY_NULL:{nulas}"
                )

            if vacias:
                errores.append(
                    f"GEOMETRY_EMPTY:{vacias}"
                )

            if invalidas:
                errores.append(
                    f"GEOMETRY_INVALID:{invalidas}"
                )

        except Exception as exc:

            advertencias.append(
                "GEOMETRY_VALIDATION_WARNING:"
                f"{exc}"
            )

    else:

        advertencias.append(
            "GEOMETRY_COLUMN_NOT_AVAILABLE"
        )

    return {
        "errores": errores,
        "advertencias": advertencias,
        "escenarios": escenarios,
        "counts": counts,
        "cobertura": cobertura,
        "minimo": minimo,
        "maximo": maximo,
        "promedio": promedio,
        "desvio": desvio,
        "cv": cv,
        "geometria": geometria_info,
    }


# ============================================================================
# VALIDACIÓN DE CONSISTENCIA
# ============================================================================

def validar_consistencia(
    df: pd.DataFrame,
    campos: dict,
):

    escenario = campos["escenario"]

    columnas = [
        (
            "tipo_escenario",
            campos["tipo"],
        ),
        (
            "dimension_dominante",
            campos["dimension"],
        ),
    ]

    if campos["prioridad"] is not None:

        columnas.append(
            (
                "prioridad_escenario",
                campos["prioridad"],
            )
        )

    inconsistencias = []

    print()
    print("=" * 88)
    print("VALIDACIÓN DE CONSISTENCIA INTERNA")
    print("=" * 88)

    for escenario_id, grupo in (
        df.groupby(
            escenario,
            sort=True,
        )
    ):

        for nombre, columna in columnas:

            unicos = valores_unicos(
                grupo[columna]
            )

            if len(unicos) > 1:

                inconsistencias.append(
                    {
                        "escenario_id": escenario_id,
                        "campo": nombre,
                        "valores": " | ".join(
                            map(str, unicos)
                        ),
                        "cantidad_valores": len(
                            unicos
                        ),
                    }
                )

    if inconsistencias:

        print(
            f"Inconsistencias encontradas: "
            f"{len(inconsistencias)}"
        )

    else:

        print(
            "Inconsistencias encontradas: 0"
        )

    return inconsistencias


# ============================================================================
# CONSTRUCCIÓN DE FICHAS
# ============================================================================

def construir_fichas(
    df: pd.DataFrame,
    campos: dict,
):

    escenario = campos["escenario"]

    print()
    print("=" * 88)
    print("CONSTRUYENDO FICHAS DE ESCENARIOS")
    print("=" * 88)

    registros = []

    for escenario_id, grupo in (
        df.groupby(
            escenario,
            sort=True,
        )
    ):

        registro = {
            "escenario_id": escenario_id,
            "cantidad_proyectos": int(
                len(grupo)
            ),
        }

        tipo, _ = moda_deterministica(
            grupo[campos["tipo"]]
        )

        dimension, _ = moda_deterministica(
            grupo[campos["dimension"]]
        )

        registro[
            "tipo_escenario"
        ] = tipo

        registro[
            "dimension_dominante"
        ] = dimension

        if campos["prioridad"] is not None:

            prioridad, _ = (
                moda_deterministica(
                    grupo[
                        campos["prioridad"]
                    ]
                )
            )

            registro[
                "prioridad_escenario"
            ] = prioridad

        # ------------------------------------------------------------
        # Indicadores
        # ------------------------------------------------------------

        for nombre, columna in (
            campos["indicadores"].items()
        ):

            stats = estadisticas_numericas(
                grupo,
                columna,
            )

            registro[
                f"{nombre}_media"
            ] = stats["media"]

            registro[
                f"{nombre}_mediana"
            ] = stats["mediana"]

            registro[
                f"{nombre}_minimo"
            ] = stats["minimo"]

            registro[
                f"{nombre}_maximo"
            ] = stats["maximo"]

        registros.append(
            registro
        )

    fichas = pd.DataFrame(
        registros
    )

    return fichas


# ============================================================================
# RANKING
# ============================================================================

def construir_ranking(
    fichas: pd.DataFrame,
):

    print()
    print("=" * 88)
    print("CONSTRUYENDO RANKING DE ESCENARIOS")
    print("=" * 88)

    ranking = fichas.copy()

    componentes = []

    columnas_prioritarias = [
        "indice_demanda_media",
        "deficit_infraestructura_media",
        "indice_conectividad_media",
        "indice_intermodalidad_media",
        "indice_integracion_media",
        "indice_centralidad_media",
        "impacto_potencial_media",
        "urgencia_intervencion_media",
        "prioridad_territorial_media",
        "score_cartera_media",
    ]

    for columna in columnas_prioritarias:

        if columna in ranking.columns:

            serie = convertir_numerico(
                ranking[columna]
            )

            if serie.notna().any():

                normalizada = (
                    normalizar_score_0_100(
                        serie
                    )
                )

                componentes.append(
                    normalizada
                )

                nombre_score = (
                    f"_score_{columna}"
                )

                ranking[
                    nombre_score
                ] = normalizada

    if componentes:

        matriz = pd.concat(
            componentes,
            axis=1,
        )

        ranking[
            "score_analitico_v4"
        ] = matriz.mean(
            axis=1,
            skipna=True,
        )

    else:

        ranking[
            "score_analitico_v4"
        ] = np.nan

    ranking = ranking.sort_values(
        [
            "score_analitico_v4",
            "cantidad_proyectos",
            "escenario_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    ranking[
        "ranking_v4"
    ] = np.arange(
        1,
        len(ranking) + 1,
    )

    return ranking


# ============================================================================
# MATRIZ COMPARATIVA
# ============================================================================

def construir_matriz(
    ranking: pd.DataFrame,
):

    print()
    print("=" * 88)
    print("CONSTRUYENDO MATRIZ COMPARATIVA")
    print("=" * 88)

    columnas = [
        "escenario_id",
        "ranking_v4",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_escenario",
        "score_analitico_v4",
    ]

    columnas = [
        columna
        for columna in columnas
        if columna in ranking.columns
    ]

    indicadores = [
        "indice_demanda_media",
        "deficit_infraestructura_media",
        "indice_conectividad_media",
        "indice_intermodalidad_media",
        "indice_integracion_media",
        "indice_centralidad_media",
        "impacto_potencial_media",
        "urgencia_intervencion_media",
        "prioridad_territorial_media",
        "score_cartera_media",
    ]

    columnas.extend(
        columna
        for columna in indicadores
        if columna in ranking.columns
    )

    matriz = ranking[
        columnas
    ].copy()

    return matriz


# ============================================================================
# AUDITORÍA
# ============================================================================

def construir_auditoria(
    df: pd.DataFrame,
    campos: dict,
    estructura: dict,
    inconsistencias: list,
):

    escenario = campos["escenario"]

    auditoria = []

    inconsistencias_set = {
        (
            str(item["escenario_id"]),
            item["campo"],
        )
        for item in inconsistencias
    }

    for escenario_id, grupo in (
        df.groupby(
            escenario,
            sort=True,
        )
    ):

        campos_estado = {}

        for nombre, columna in [
            (
                "tipo_escenario",
                campos["tipo"],
            ),
            (
                "dimension_dominante",
                campos["dimension"],
            ),
        ]:

            cantidad = len(
                valores_unicos(
                    grupo[columna]
                )
            )

            campos_estado[
                nombre
            ] = cantidad

        if campos["prioridad"] is not None:

            campos_estado[
                "prioridad_escenario"
            ] = len(
                valores_unicos(
                    grupo[
                        campos["prioridad"]
                    ]
                )
            )

        consistencia = all(
            (
                str(escenario_id),
                campo,
            )
            not in inconsistencias_set
            for campo in campos_estado
        )

        auditoria.append(
            {
                "escenario_id": escenario_id,
                "cantidad_proyectos": len(
                    grupo
                ),
                "tipo_valores_distintos":
                    campos_estado.get(
                        "tipo_escenario",
                        0,
                    ),
                "dimension_valores_distintos":
                    campos_estado.get(
                        "dimension_dominante",
                        0,
                    ),
                "prioridad_valores_distintos":
                    campos_estado.get(
                        "prioridad_escenario",
                        0,
                    ),
                "consistente": consistencia,
                "cobertura": 1.0,
                "estado": (
                    "VALIDADO"
                    if consistencia
                    else "NO_VALIDADO"
                ),
            }
        )

    return pd.DataFrame(
        auditoria
    )


# ============================================================================
# SCORE FINAL
# ============================================================================

def calcular_score_final(
    estructura: dict,
    inconsistencias: list,
    auditoria: pd.DataFrame,
):

    cobertura = float(
        estructura["cobertura"]
    )

    cv = float(
        estructura["cv"]
    )

    score_tamano = max(
        0.0,
        min(
            1.0,
            1.0 - cv,
        ),
    )

    score_consistencia = (
        1.0
        if not inconsistencias
        else 0.0
    )

    if auditoria.empty:

        score_auditoria = 0.0

    else:

        score_auditoria = float(
            auditoria[
                "consistente"
            ].mean()
        )

    # Integridad estructural.
    #
    # Cobertura       30%
    # Tamaño          20%
    # Consistencia    30%
    # Auditoría       20%

    score = (
        0.30 * cobertura
        + 0.20 * score_tamano
        + 0.30 * score_consistencia
        + 0.20 * score_auditoria
    )

    return {
        "score_cobertura": cobertura,
        "score_tamano": score_tamano,
        "score_consistencia": score_consistencia,
        "score_auditoria": score_auditoria,
        "score_final": float(score),
    }


# ============================================================================
# RESUMEN
# ============================================================================

def construir_resumen(
    df: pd.DataFrame,
    campos: dict,
    estructura: dict,
    inconsistencias: list,
    ranking: pd.DataFrame,
    auditoria: pd.DataFrame,
    score: dict,
):

    escenarios = estructura[
        "escenarios"
    ]

    errores = list(
        estructura["errores"]
    )

    advertencias = list(
        estructura["advertencias"]
    )

    if inconsistencias:

        errores.append(
            "INTERNAL_CONSISTENCY_REMAINING:"
            f"{len(inconsistencias)}"
        )

    dictamen = (
        "VALIDADO"
        if not errores
        else "NO_VALIDADO"
    )

    ranking_top = []

    if not ranking.empty:

        for _, fila in (
            ranking.head(10).iterrows()
        ):

            ranking_top.append(
                {
                    "ranking": int(
                        fila["ranking_v4"]
                    ),
                    "escenario_id": str(
                        fila["escenario_id"]
                    ),
                    "score": convertir_jsonable(
                        fila[
                            "score_analitico_v4"
                        ]
                    ),
                }
            )

    resumen = {
        "version": VERSION,
        "proceso": PROCESO,
        "entrada": str(INPUT_FILE),

        "registros": int(
            len(df)
        ),

        "proyectos_unicos": int(
            df[
                campos["proyecto"]
            ].nunique()
        ),

        "escenarios": int(
            len(escenarios)
        ),

        "escenarios_ids": [
            str(x)
            for x in escenarios
        ],

        "cobertura": float(
            estructura["cobertura"]
        ),

        "minimo_proyectos": int(
            estructura["minimo"]
        ),

        "maximo_proyectos": int(
            estructura["maximo"]
        ),

        "promedio_proyectos": float(
            estructura["promedio"]
        ),

        "desvio_proyectos": float(
            estructura["desvio"]
        ),

        "cv_proyectos": float(
            estructura["cv"]
        ),

        "inconsistencias": int(
            len(inconsistencias)
        ),

        "score_cobertura": score[
            "score_cobertura"
        ],

        "score_tamano": score[
            "score_tamano"
        ],

        "score_consistencia": score[
            "score_consistencia"
        ],

        "score_auditoria": score[
            "score_auditoria"
        ],

        "score_final": score[
            "score_final"
        ],

        "ranking_top": ranking_top,

        "geometria": estructura[
            "geometria"
        ],

        "errores": errores,

        "advertencias": advertencias,

        "dictamen": dictamen,

        "salidas": {
            "ficha": str(
                OUTPUT_FICHA
            ),
            "ranking": str(
                OUTPUT_RANKING
            ),
            "matriz": str(
                OUTPUT_MATRIZ
            ),
            "auditoria": str(
                OUTPUT_AUDITORIA
            ),
            "json": str(
                OUTPUT_JSON
            ),
        },
    }

    return resumen


# ============================================================================
# EXPORTACIÓN
# ============================================================================

def exportar(
    fichas: pd.DataFrame,
    ranking: pd.DataFrame,
    matriz: pd.DataFrame,
    auditoria: pd.DataFrame,
    resumen: dict,
):

    print()
    print("=" * 88)
    print("EXPORTANDO RESULTADOS DEL PROCESO 32")
    print("=" * 88)

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fichas.to_csv(
        OUTPUT_FICHA,
        index=False,
        encoding="utf-8-sig",
    )

    ranking.to_csv(
        OUTPUT_RANKING,
        index=False,
        encoding="utf-8-sig",
    )

    matriz.to_csv(
        OUTPUT_MATRIZ,
        index=False,
        encoding="utf-8-sig",
    )

    auditoria.to_csv(
        OUTPUT_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
            default=convertir_jsonable,
        )

    print(
        f"Ficha       : {OUTPUT_FICHA}"
    )

    print(
        f"Ranking     : {OUTPUT_RANKING}"
    )

    print(
        f"Matriz      : {OUTPUT_MATRIZ}"
    )

    print(
        f"Auditoría   : {OUTPUT_AUDITORIA}"
    )

    print(
        f"Resumen     : {OUTPUT_JSON}"
    )


# ============================================================================
# REPORTE FINAL
# ============================================================================

def imprimir_resultado(
    resumen: dict,
    fichas: pd.DataFrame,
    ranking: pd.DataFrame,
):

    print()
    print("=" * 88)
    print("RESULTADO FINAL DEL PROCESO 32")
    print("=" * 88)

    print(
        f"Proyectos                 : "
        f"{resumen['registros']:,}"
    )

    print(
        f"Proyectos únicos          : "
        f"{resumen['proyectos_unicos']:,}"
    )

    print(
        f"Escenarios                : "
        f"{resumen['escenarios']}"
    )

    print(
        f"Cobertura                 : "
        f"{resumen['cobertura']:.2%}"
    )

    print(
        f"Mínimo proyectos          : "
        f"{resumen['minimo_proyectos']}"
    )

    print(
        f"Máximo proyectos          : "
        f"{resumen['maximo_proyectos']}"
    )

    print(
        f"CV tamaño                 : "
        f"{resumen['cv_proyectos']:.4f}"
    )

    print(
        f"Inconsistencias           : "
        f"{resumen['inconsistencias']}"
    )

    print(
        f"Score cobertura           : "
        f"{resumen['score_cobertura']:.4f}"
    )

    print(
        f"Score tamaño              : "
        f"{resumen['score_tamano']:.4f}"
    )

    print(
        f"Score consistencia        : "
        f"{resumen['score_consistencia']:.4f}"
    )

    print(
        f"Score auditoría           : "
        f"{resumen['score_auditoria']:.4f}"
    )

    print(
        f"Score final               : "
        f"{resumen['score_final']:.4f}"
    )

    print(
        f"Dictamen                  : "
        f"{resumen['dictamen']}"
    )

    print()
    print("=" * 88)
    print("RANKING DE ESCENARIOS")
    print("=" * 88)

    if not ranking.empty:

        columnas = [
            columna
            for columna in [
                "ranking_v4",
                "escenario_id",
                "cantidad_proyectos",
                "tipo_escenario",
                "dimension_dominante",
                "prioridad_escenario",
                "score_analitico_v4",
            ]
            if columna in ranking.columns
        ]

        print(
            ranking[
                columnas
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 88)

    if resumen["dictamen"] == "VALIDADO":

        print(
            "DICTAMEN FINAL: VALIDADO"
        )

        print()

        print(
            "La salida V4 supera la validación "
            "analítica final del proceso 32."
        )

        print(
            "La asignación proyecto -> escenario "
            "se mantiene íntegra."
        )

        print(
            "No se detectan inconsistencias internas "
            "en los atributos conceptuales de escenario."
        )

        print(
            "Los indicadores originales y las "
            "geometrías no son modificados."
        )

    else:

        print(
            "DICTAMEN FINAL: NO_VALIDADO"
        )

        print()

        print("Errores:")

        for error in resumen["errores"]:

            print(
                f"  - {error}"
            )

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():

    try:

        # --------------------------------------------------------------------
        # 1. CARGA
        # --------------------------------------------------------------------

        df = cargar_datos()

        # --------------------------------------------------------------------
        # 2. CAMPOS
        # --------------------------------------------------------------------

        campos = resolver_campos(
            df
        )

        # --------------------------------------------------------------------
        # 3. VALIDACIÓN ESTRUCTURAL
        # --------------------------------------------------------------------

        estructura = validar_estructura(
            df,
            campos,
        )

        # --------------------------------------------------------------------
        # 4. CONSISTENCIA
        # --------------------------------------------------------------------

        inconsistencias = (
            validar_consistencia(
                df,
                campos,
            )
        )

        # --------------------------------------------------------------------
        # 5. FICHAS
        # --------------------------------------------------------------------

        fichas = construir_fichas(
            df,
            campos,
        )

        # --------------------------------------------------------------------
        # 6. RANKING
        # --------------------------------------------------------------------

        ranking = construir_ranking(
            fichas
        )

        # --------------------------------------------------------------------
        # 7. MATRIZ
        # --------------------------------------------------------------------

        matriz = construir_matriz(
            ranking
        )

        # --------------------------------------------------------------------
        # 8. AUDITORÍA
        # --------------------------------------------------------------------

        auditoria = construir_auditoria(
            df,
            campos,
            estructura,
            inconsistencias,
        )

        # --------------------------------------------------------------------
        # 9. SCORE
        # --------------------------------------------------------------------

        score = calcular_score_final(
            estructura,
            inconsistencias,
            auditoria,
        )

        # --------------------------------------------------------------------
        # 10. RESUMEN
        # --------------------------------------------------------------------

        resumen = construir_resumen(
            df,
            campos,
            estructura,
            inconsistencias,
            ranking,
            auditoria,
            score,
        )

        # --------------------------------------------------------------------
        # 11. EXPORTACIÓN
        # --------------------------------------------------------------------

        exportar(
            fichas,
            ranking,
            matriz,
            auditoria,
            resumen,
        )

        # --------------------------------------------------------------------
        # 12. REPORTE
        # --------------------------------------------------------------------

        imprimir_resultado(
            resumen,
            fichas,
            ranking,
        )

        if (
            resumen["dictamen"]
            == "VALIDADO"
        ):

            return 0

        return 1

    except Exception as exc:

        print()
        print("=" * 88)
        print("ERROR EN PROCESO 32")
        print("=" * 88)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(
        main()
    )