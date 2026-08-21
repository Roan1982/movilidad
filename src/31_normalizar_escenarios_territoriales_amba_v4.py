# -*- coding: utf-8 -*-

"""
31 - NORMALIZACIÓN Y VALIDACIÓN DE ESCENARIOS TERRITORIALES AMBA - V4

Objetivo
--------
Normalizar los atributos conceptuales de cada escenario territorial
a partir de la salida optimizada del proceso 29.

Principios V4
-------------
- No modifica la asignación proyecto -> escenario.
- No elimina proyectos.
- No modifica geometrías.
- No modifica indicadores originales.
- Normaliza atributos conceptuales a nivel escenario.
- Conserva los valores originales para auditoría.
- Utiliza reglas determinísticas y trazables.
- Valida nuevamente la consistencia interna.
- Produce Parquet, CSV, auditoría, detalle y resumen JSON.

Entrada
-------
data/processed/escenarios_territoriales_amba/
    escenarios_territoriales_amba_optimizado.parquet

Salidas
-------
data/processed/escenarios_territoriales_amba/
    escenarios_territoriales_amba_v4.parquet
    escenarios_territoriales_amba_v4.csv
    detalle_v4_escenarios_territoriales_amba.csv
    auditoria_v4_escenarios_territoriales_amba.csv
    resumen_v4_escenarios_territoriales_amba.json
"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

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
PROCESO = 31

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

INPUT_FILE = (
    INPUT_DIR
    / "escenarios_territoriales_amba_optimizado.parquet"
)

OUTPUT_PARQUET = (
    INPUT_DIR
    / "escenarios_territoriales_amba_v4.parquet"
)

OUTPUT_CSV = (
    INPUT_DIR
    / "escenarios_territoriales_amba_v4.csv"
)

OUTPUT_DETAIL = (
    INPUT_DIR
    / "detalle_v4_escenarios_territoriales_amba.csv"
)

OUTPUT_AUDIT = (
    INPUT_DIR
    / "auditoria_v4_escenarios_territoriales_amba.csv"
)

OUTPUT_JSON = (
    INPUT_DIR
    / "resumen_v4_escenarios_territoriales_amba.json"
)

EXPECTED_SCENARIOS_MIN = 6
EXPECTED_SCENARIOS_MAX = 12

MIN_PROJECTS = 8

CRS_GEOGRAPHIC = "EPSG:4326"
CRS_METRIC = "EPSG:22185"


# ============================================================================
# CAMPOS CONCEPTUALES
# ============================================================================

SCENARIO_FIELDS = [
    "tipo_escenario",
    "dimension_dominante",
    "prioridad_escenario",
]


INDICATOR_FIELDS = [
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
]


# ============================================================================
# UTILIDADES
# ============================================================================

def normalizar_nombre(nombre: str) -> str:
    """
    Normaliza nombres de columnas para permitir pequeñas diferencias
    de nomenclatura.
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
    requerida: bool = True,
):
    """
    Busca una columna por nombre exacto o normalizado.
    """

    direct = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidato in candidatos:

        if candidato.lower() in direct:
            return direct[candidato.lower()]

    normalizadas = {
        normalizar_nombre(c): c
        for c in df.columns
    }

    for candidato in candidatos:

        normalizado = normalizar_nombre(candidato)

        if normalizado in normalizadas:
            return normalizadas[normalizado]

    if requerida:
        raise KeyError(
            "No se encontró ninguna de las columnas esperadas: "
            f"{candidatos}"
        )

    return None


def valor_valido(valor) -> bool:
    """
    Determina si un valor puede utilizarse.
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


def clave_valor(valor):
    """
    Construye una clave estable para comparar valores heterogéneos.
    """

    if isinstance(valor, (np.integer, int)):

        return (
            "num",
            float(valor),
        )

    if isinstance(valor, (np.floating, float)):

        if math.isnan(float(valor)):
            return (
                "null",
                "",
            )

        return (
            "num",
            float(valor),
        )

    return (
        "str",
        str(valor).strip(),
    )


def valores_unicos_validos(
    serie: pd.Series,
) -> list:

    salida = []
    vistos = set()

    for valor in serie:

        if not valor_valido(valor):
            continue

        clave = clave_valor(valor)

        if clave not in vistos:

            vistos.add(clave)
            salida.append(valor)

    return salida


def frecuencia_valores(
    serie: pd.Series,
) -> Counter:

    contador = Counter()

    for valor in serie:

        if valor_valido(valor):

            contador[
                clave_valor(valor)
            ] += 1

    return contador


def moda_deterministica(
    serie: pd.Series,
):
    """
    Moda determinística.

    Criterio:
    1. mayor frecuencia
    2. primer valor observado como desempate
    """

    frecuencias = frecuencia_valores(serie)

    if not frecuencias:
        return None, 0, 0

    valores = [
        valor
        for valor in serie
        if valor_valido(valor)
    ]

    primera_posicion = {}

    for i, valor in enumerate(valores):

        clave = clave_valor(valor)

        if clave not in primera_posicion:
            primera_posicion[clave] = i

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
        if clave_valor(valor) == ganador
    )

    return (
        valor_final,
        frecuencias[ganador],
        len(valores),
    )


def resolver_prioridad(
    grupo: pd.DataFrame,
    columna: str,
):
    """
    Normalización de prioridad.

    Si es numérica:
        mediana.

    Si es categórica:
        moda determinística.
    """

    serie = grupo[columna]

    validos = serie[
        serie.apply(valor_valido)
    ]

    if validos.empty:

        return (
            None,
            "SIN_DATO",
        )

    numerica = pd.to_numeric(
        validos,
        errors="coerce",
    )

    if numerica.notna().all():

        mediana = float(
            numerica.median()
        )

        valores = numerica.to_numpy(
            dtype=float
        )

        todos_enteros = np.all(
            np.isclose(
                valores,
                np.round(valores),
            )
        )

        if todos_enteros:

            return (
                int(round(mediana)),
                "MEDIANA",
            )

        return (
            mediana,
            "MEDIANA",
        )

    valor, _, _ = moda_deterministica(
        validos
    )

    return (
        valor,
        "MODA",
    )


def score_numerico_grupo(
    grupo: pd.DataFrame,
    columnas: list[str],
) -> float:
    """
    Score auxiliar exclusivamente para diagnóstico.

    No modifica indicadores.
    """

    scores = []

    for columna in columnas:

        if columna not in grupo.columns:
            continue

        serie = pd.to_numeric(
            grupo[columna],
            errors="coerce",
        )

        if serie.notna().any():

            scores.append(
                float(serie.mean())
            )

    if not scores:
        return np.nan

    return float(
        np.mean(scores)
    )


def convertir_jsonable(valor):
    """
    Convierte valores NumPy/Pandas a tipos JSON.
    """

    if isinstance(valor, np.integer):
        return int(valor)

    if isinstance(valor, np.floating):

        if np.isnan(valor):
            return None

        return float(valor)

    if isinstance(valor, np.ndarray):
        return valor.tolist()

    if isinstance(valor, (list, dict, tuple)):
        return valor

    try:

        if pd.isna(valor):
            return None

    except Exception:
        pass

    return valor


# ============================================================================
# CARGA
# ============================================================================

def cargar():

    print("=" * 88)

    print(
        "31 - NORMALIZACIÓN Y VALIDACIÓN DE "
        f"ESCENARIOS TERRITORIALES AMBA - {VERSION}"
    )

    print("=" * 88)

    print(f"Proyecto : {BASE_DIR}")
    print(f"Entrada  : {INPUT_FILE}")
    print(f"Salida   : {INPUT_DIR}")
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nNo existe la entrada del proceso 29:\n"
            f"{INPUT_FILE}\n\n"
            "Ejecute primero el proceso que genera "
            "escenarios_territoriales_amba_optimizado.parquet."
        )

    print("=" * 88)
    print("CARGANDO RESULTADO DEL PROCESO 29")
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
# RESOLUCIÓN DE COLUMNAS
# ============================================================================

def resolver_columnas(df):

    proyecto = resolver_columna(
        df,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    escenario = resolver_columna(
        df,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    tipo = resolver_columna(
        df,
        [
            "tipo_escenario",
            "tipo",
        ],
    )

    dimension = resolver_columna(
        df,
        [
            "dimension_dominante",
            "dimension",
        ],
    )

    prioridad = resolver_columna(
        df,
        [
            "prioridad_escenario",
            "prioridad",
        ],
        requerida=False,
    )

    geometria = None

    if (
        hasattr(df, "geometry")
        and "geometry" in df.columns
    ):
        geometria = "geometry"

    resultado = {
        "proyecto": proyecto,
        "escenario": escenario,
        "tipo": tipo,
        "dimension": dimension,
        "prioridad": prioridad,
        "geometria": geometria,
    }

    print()
    print("COLUMNAS RESUELTAS")

    for clave, valor in resultado.items():

        print(
            f"  {clave:20}: {valor}"
        )

    return resultado


# ============================================================================
# NORMALIZACIÓN
# ============================================================================

def normalizar(
    df: pd.DataFrame,
    cols: dict,
):

    escenario = cols["escenario"]
    tipo = cols["tipo"]
    dimension = cols["dimension"]
    prioridad = cols["prioridad"]

    out = df.copy()

    # ------------------------------------------------------------------------
    # AUDITORÍA ORIGINAL
    # ------------------------------------------------------------------------

    out[
        "v4_original_tipo_escenario"
    ] = out[tipo]

    out[
        "v4_original_dimension_dominante"
    ] = out[dimension]

    if prioridad:

        out[
            "v4_original_prioridad_escenario"
        ] = out[prioridad]

    auditoria = []
    detalle = []

    print()
    print("=" * 88)
    print("NORMALIZACIÓN V4")
    print("=" * 88)

    grupos = out.groupby(
        escenario,
        sort=True,
        dropna=False,
    )

    for escenario_id, grupo in grupos:

        if pd.isna(escenario_id):
            continue

        # --------------------------------------------------------------------
        # TIPO
        # --------------------------------------------------------------------

        tipo_originales = (
            valores_unicos_validos(
                grupo[tipo]
            )
        )

        (
            tipo_final,
            tipo_frecuencia,
            tipo_total,
        ) = moda_deterministica(
            grupo[tipo]
        )

        # --------------------------------------------------------------------
        # DIMENSIÓN
        # --------------------------------------------------------------------

        dimension_originales = (
            valores_unicos_validos(
                grupo[dimension]
            )
        )

        (
            dimension_final,
            dimension_frecuencia,
            dimension_total,
        ) = moda_deterministica(
            grupo[dimension]
        )

        # --------------------------------------------------------------------
        # PRIORIDAD
        # --------------------------------------------------------------------

        if prioridad:

            (
                prioridad_final,
                prioridad_metodo,
            ) = resolver_prioridad(
                grupo,
                prioridad,
            )

            prioridad_originales = (
                valores_unicos_validos(
                    grupo[prioridad]
                )
            )

        else:

            prioridad_final = None
            prioridad_metodo = "NO_DISPONIBLE"
            prioridad_originales = []

        # --------------------------------------------------------------------
        # SCORE AUXILIAR
        # --------------------------------------------------------------------

        indicadores_disponibles = [
            columna
            for columna in INDICATOR_FIELDS
            if columna in grupo.columns
        ]

        score_auxiliar = (
            score_numerico_grupo(
                grupo,
                indicadores_disponibles,
            )
        )

        # --------------------------------------------------------------------
        # ASIGNACIÓN
        # --------------------------------------------------------------------

        mascara = (
            out[escenario]
            == escenario_id
        )

        out.loc[
            mascara,
            tipo,
        ] = tipo_final

        out.loc[
            mascara,
            dimension,
        ] = dimension_final

        if prioridad:

            out.loc[
                mascara,
                prioridad,
            ] = prioridad_final

        # --------------------------------------------------------------------
        # CAMBIOS
        # --------------------------------------------------------------------

        cambio_tipo = (
            len(tipo_originales) > 1
        )

        cambio_dimension = (
            len(dimension_originales) > 1
        )

        cambio_prioridad = (
            len(prioridad_originales) > 1
        )

        cantidad = len(grupo)

        hubo_cambio = (
            cambio_tipo
            or cambio_dimension
            or cambio_prioridad
        )

        # --------------------------------------------------------------------
        # AUDITORÍA
        # --------------------------------------------------------------------

        auditoria.append(
            {
                "escenario_id": escenario_id,
                "cantidad_proyectos": cantidad,

                "tipo_original_distintos":
                    len(tipo_originales),

                "tipo_original_valores":
                    " | ".join(
                        map(
                            str,
                            tipo_originales,
                        )
                    ),

                "tipo_final": tipo_final,

                "tipo_frecuencia":
                    tipo_frecuencia,

                "tipo_total_validos":
                    tipo_total,

                "tipo_metodo":
                    "MODA_DETERMINISTICA",

                "dimension_original_distintos":
                    len(dimension_originales),

                "dimension_original_valores":
                    " | ".join(
                        map(
                            str,
                            dimension_originales,
                        )
                    ),

                "dimension_final":
                    dimension_final,

                "dimension_frecuencia":
                    dimension_frecuencia,

                "dimension_total_validos":
                    dimension_total,

                "dimension_metodo":
                    "MODA_DETERMINISTICA",

                "prioridad_original_distintos":
                    len(prioridad_originales),

                "prioridad_original_valores":
                    " | ".join(
                        map(
                            str,
                            prioridad_originales,
                        )
                    ),

                "prioridad_final":
                    prioridad_final,

                "prioridad_metodo":
                    prioridad_metodo,

                "score_auxiliar_indicadores":
                    score_auxiliar,

                "corregido_tipo":
                    cambio_tipo,

                "corregido_dimension":
                    cambio_dimension,

                "corregido_prioridad":
                    cambio_prioridad,

                "escenario_modificado":
                    hubo_cambio,
            }
        )

        # --------------------------------------------------------------------
        # DETALLE
        # --------------------------------------------------------------------

        detalle.append(
            {
                "escenario_id":
                    escenario_id,

                "cantidad_proyectos":
                    cantidad,

                "tipo_escenario":
                    tipo_final,

                "dimension_dominante":
                    dimension_final,

                "prioridad_escenario":
                    prioridad_final,

                "score_auxiliar_indicadores":
                    score_auxiliar,

                "corregido_tipo":
                    cambio_tipo,

                "corregido_dimension":
                    cambio_dimension,

                "corregido_prioridad":
                    cambio_prioridad,

                "estado_v4":
                    "NORMALIZADO",
            }
        )

    auditoria_df = pd.DataFrame(
        auditoria
    )

    detalle_df = pd.DataFrame(
        detalle
    )

    return (
        out,
        auditoria_df,
        detalle_df,
    )


# ============================================================================
# VALIDACIÓN
# ============================================================================

def validar_v4(
    df: pd.DataFrame,
    cols: dict,
    auditoria: pd.DataFrame,
):

    proyecto = cols["proyecto"]
    escenario = cols["escenario"]
    tipo = cols["tipo"]
    dimension = cols["dimension"]
    prioridad = cols["prioridad"]

    errores = []
    advertencias = []

    n = len(df)

    # ========================================================================
    # ESTRUCTURA
    # ========================================================================

    if n == 0:
        errores.append(
            "DATASET_EMPTY"
        )

    if df[proyecto].isna().any():

        errores.append(
            "PROJECT_ID_NULL"
        )

    duplicados = int(
        df[proyecto].duplicated().sum()
    )

    if duplicados:

        errores.append(
            f"PROJECT_ID_DUPLICATES:{duplicados}"
        )

    # ========================================================================
    # GEOMETRÍA
    # ========================================================================

    if (
        gpd is not None
        and isinstance(
            df,
            gpd.GeoDataFrame,
        )
        and df.geometry is not None
    ):

        try:

            null_geom = int(
                df.geometry.isna().sum()
            )

            empty_geom = int(
                df.geometry.is_empty.sum()
            )

            invalid_geom = int(
                (~df.geometry.is_valid).sum()
            )

            if null_geom:

                errores.append(
                    f"GEOMETRY_NULL:{null_geom}"
                )

            if empty_geom:

                errores.append(
                    f"GEOMETRY_EMPTY:{empty_geom}"
                )

            if invalid_geom:

                errores.append(
                    f"GEOMETRY_INVALID:{invalid_geom}"
                )

        except Exception as exc:

            advertencias.append(
                "GEOMETRY_VALIDATION_WARNING:"
                f"{exc}"
            )

    # ========================================================================
    # ESCENARIOS
    # ========================================================================

    escenarios = [
        valor
        for valor
        in df[escenario]
        .dropna()
        .unique()
        .tolist()
    ]

    escenarios = sorted(
        escenarios,
        key=str,
    )

    cantidad_escenarios = len(
        escenarios
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

    counts = (
        df.groupby(
            escenario,
            dropna=True,
        )
        .size()
    )

    if not counts.empty:

        escenarios_pequenos = (
            counts[
                counts < MIN_PROJECTS
            ]
        )

        if not escenarios_pequenos.empty:

            errores.append(
                "SCENARIO_MIN_PROJECTS:"
                f"{escenarios_pequenos.to_dict()}"
            )

    # ========================================================================
    # CONSISTENCIA INTERNA
    # ========================================================================

    campos_consistencia = [
        (
            "tipo_escenario",
            tipo,
        ),
        (
            "dimension_dominante",
            dimension,
        ),
    ]

    if prioridad:

        campos_consistencia.append(
            (
                "prioridad_escenario",
                prioridad,
            )
        )

    inconsistencias = []

    for escenario_id, grupo in df.groupby(
        escenario,
        sort=True,
        dropna=True,
    ):

        for nombre, columna in campos_consistencia:

            unicos = (
                valores_unicos_validos(
                    grupo[columna]
                )
            )

            if len(unicos) > 1:

                inconsistencias.append(
                    {
                        "escenario_id":
                            escenario_id,

                        "campo":
                            nombre,

                        "valores":
                            " | ".join(
                                map(
                                    str,
                                    unicos,
                                )
                            ),

                        "cantidad_valores":
                            len(unicos),
                    }
                )

    if inconsistencias:

        errores.append(
            "INTERNAL_CONSISTENCY_REMAINING:"
            f"{len(inconsistencias)}"
        )

    # ========================================================================
    # COBERTURA
    # ========================================================================

    sin_escenario = int(
        df[escenario].isna().sum()
    )

    cobertura = (
        (n - sin_escenario) / n
        if n
        else 0.0
    )

    if cobertura < 1.0:

        errores.append(
            f"COVERAGE:{cobertura:.6f}"
        )

    # ========================================================================
    # TAMAÑO
    # ========================================================================

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
            counts.std(
                ddof=0
            )
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

    score_tamano = max(
        0.0,
        min(
            1.0,
            1.0 - cv,
        ),
    )

    # ========================================================================
    # AUDITORÍA
    # ========================================================================

    cambios = 0

    if not auditoria.empty:

        columnas_cambio = [
            "corregido_tipo",
            "corregido_dimension",
            "corregido_prioridad",
        ]

        columnas_existentes = [
            columna
            for columna in columnas_cambio
            if columna in auditoria.columns
        ]

        if columnas_existentes:

            cambios = int(
                auditoria[
                    columnas_existentes
                ]
                .fillna(False)
                .any(axis=1)
                .sum()
            )

    # ========================================================================
    # SCORES
    # ========================================================================

    score_consistencia = (
        1.0
        if not inconsistencias
        else 0.0
    )

    score_cobertura = cobertura

    score_v4 = (
        0.40 * score_cobertura
        + 0.20 * score_tamano
        + 0.40 * score_consistencia
    )

    # ========================================================================
    # DICTAMEN
    # ========================================================================

    if errores:

        dictamen = "NO_VALIDADO"

    else:

        dictamen = "VALIDADO"

    return {
        "registros":
            n,

        "escenarios":
            cantidad_escenarios,

        "escenarios_ids":
            [
                str(valor)
                for valor in escenarios
            ],

        "cobertura":
            cobertura,

        "sin_escenario":
            sin_escenario,

        "minimo_proyectos":
            minimo,

        "maximo_proyectos":
            maximo,

        "promedio_proyectos":
            promedio,

        "desvio_proyectos":
            desvio,

        "cv_proyectos":
            cv,

        "score_tamano":
            score_tamano,

        "inconsistencias_restantes":
            len(inconsistencias),

        "cambios_escenario":
            cambios,

        "score_v4":
            score_v4,

        "errores":
            errores,

        "advertencias":
            advertencias,

        "dictamen":
            dictamen,

        "detalle_inconsistencias":
            inconsistencias,
    }


# ============================================================================
# EXPORTACIÓN
# ============================================================================

def exportar(
    df: pd.DataFrame,
    auditoria: pd.DataFrame,
    detalle: pd.DataFrame,
    resumen: dict,
):

    print()
    print("=" * 88)
    print("EXPORTANDO RESULTADOS V4")
    print("=" * 88)

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # PARQUET
    # ========================================================================

    df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    # ========================================================================
    # CSV
    # ========================================================================

    csv_df = df.copy()

    if "geometry" in csv_df.columns:

        try:

            csv_df["geometry"] = (
                csv_df["geometry"]
                .apply(
                    lambda geom:
                    geom.wkt
                    if geom is not None
                    and not pd.isna(geom)
                    else None
                )
            )

        except Exception:

            csv_df["geometry"] = (
                csv_df["geometry"]
                .astype(str)
            )

    csv_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================================
    # DETALLE
    # ========================================================================

    detalle.to_csv(
        OUTPUT_DETAIL,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================================
    # AUDITORÍA
    # ========================================================================

    auditoria.to_csv(
        OUTPUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================================
    # JSON
    # ========================================================================

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
        f"Parquet    : {OUTPUT_PARQUET}"
    )

    print(
        f"CSV        : {OUTPUT_CSV}"
    )

    print(
        f"Detalle    : {OUTPUT_DETAIL}"
    )

    print(
        f"Auditoría  : {OUTPUT_AUDIT}"
    )

    print(
        f"Resumen    : {OUTPUT_JSON}"
    )


# ============================================================================
# REPORTE
# ============================================================================

def imprimir_resultado(
    resumen: dict,
    detalle: pd.DataFrame,
):

    print()
    print("=" * 88)
    print("RESULTADO V4")
    print("=" * 88)

    print(
        f"Proyectos                  : "
        f"{resumen['registros']:,}"
    )

    print(
        f"Escenarios                 : "
        f"{resumen['escenarios']}"
    )

    print(
        f"Cobertura                  : "
        f"{resumen['cobertura']:.2%}"
    )

    print(
        f"Mínimo proyectos           : "
        f"{resumen['minimo_proyectos']}"
    )

    print(
        f"Máximo proyectos           : "
        f"{resumen['maximo_proyectos']}"
    )

    print(
        f"Score tamaño               : "
        f"{resumen['score_tamano']:.4f}"
    )

    print(
        f"Inconsistencias restantes  : "
        f"{resumen['inconsistencias_restantes']}"
    )

    print(
        f"Escenarios normalizados    : "
        f"{resumen['cambios_escenario']}"
    )

    print(
        f"Score V4                   : "
        f"{resumen['score_v4']:.4f}"
    )

    print(
        f"Dictamen                   : "
        f"{resumen['dictamen']}"
    )

    # ========================================================================
    # DETALLE
    # ========================================================================

    print()
    print("=" * 88)
    print("DETALLE DE ESCENARIOS V4")
    print("=" * 88)

    if not detalle.empty:

        columnas_mostrar = [
            "escenario_id",
            "cantidad_proyectos",
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_escenario",
            "score_auxiliar_indicadores",
            "estado_v4",
        ]

        columnas_mostrar = [
            columna
            for columna in columnas_mostrar
            if columna in detalle.columns
        ]

        print(
            detalle[
                columnas_mostrar
            ].to_string(
                index=False
            )
        )

    # ========================================================================
    # INCONSISTENCIAS
    # ========================================================================

    if resumen[
        "detalle_inconsistencias"
    ]:

        print()
        print(
            "INCONSISTENCIAS RESTANTES"
        )

        print(
            pd.DataFrame(
                resumen[
                    "detalle_inconsistencias"
                ]
            ).to_string(
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
            "La salida V4 conserva la asignación "
            "proyecto -> escenario del proceso 29, "
            "mantiene los indicadores y geometrías "
            "originales y elimina las inconsistencias "
            "internas de los atributos conceptuales "
            "del escenario."
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


# ============================================================================
# MAIN
# ============================================================================

def main():

    try:

        # --------------------------------------------------------------------
        # CARGA
        # --------------------------------------------------------------------

        df = cargar()

        # --------------------------------------------------------------------
        # RESOLVER COLUMNAS
        # --------------------------------------------------------------------

        cols = resolver_columnas(
            df
        )

        # --------------------------------------------------------------------
        # NORMALIZACIÓN
        # --------------------------------------------------------------------

        print()
        print("=" * 88)
        print(
            "APLICANDO NORMALIZACIÓN "
            "DETERMINÍSTICA"
        )
        print("=" * 88)

        (
            df_v4,
            auditoria,
            detalle,
        ) = normalizar(
            df,
            cols,
        )

        # --------------------------------------------------------------------
        # VALIDACIÓN
        # --------------------------------------------------------------------

        resumen = validar_v4(
            df_v4,
            cols,
            auditoria,
        )

        # --------------------------------------------------------------------
        # METADATOS
        # --------------------------------------------------------------------

        resumen["version"] = VERSION

        resumen["proceso"] = PROCESO

        resumen["entrada"] = str(
            INPUT_FILE
        )

        resumen["salidas"] = {
            "parquet":
                str(OUTPUT_PARQUET),

            "csv":
                str(OUTPUT_CSV),

            "detalle":
                str(OUTPUT_DETAIL),

            "auditoria":
                str(OUTPUT_AUDIT),

            "json":
                str(OUTPUT_JSON),
        }

        resumen[
            "principios_v4"
        ] = {
            "asignacion_proyecto_escenario":
                "NO_MODIFICADA",

            "proyectos_eliminados":
                0,

            "geometrias_modificadas":
                False,

            "indicadores_originales_modificados":
                False,

            "atributos_escenario_normalizados":
                True,

            "auditoria_original_conservada":
                True,

            "metodo_tipo":
                "MODA_DETERMINISTICA",

            "metodo_dimension":
                "MODA_DETERMINISTICA",

            "metodo_prioridad":
                (
                    "MEDIANA_O_MODA"
                ),
        }

        # --------------------------------------------------------------------
        # EXPORTAR
        # --------------------------------------------------------------------

        exportar(
            df_v4,
            auditoria,
            detalle,
            resumen,
        )

        # --------------------------------------------------------------------
        # REPORTE
        # --------------------------------------------------------------------

        imprimir_resultado(
            resumen,
            detalle,
        )

        # --------------------------------------------------------------------
        # CÓDIGO DE SALIDA
        # --------------------------------------------------------------------

        if resumen[
            "dictamen"
        ] == "VALIDADO":

            return 0

        return 1

    except Exception as exc:

        print()
        print("=" * 88)
        print(
            "ERROR EN PROCESO 31 V4"
        )
        print("=" * 88)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())