# -*- coding: utf-8 -*-

"""
37 - INTEGRACIÓN DEL MODELO TERRITORIAL AMBA - V4

Integra los resultados consolidados de los procesos 31 a 36:

31 - Normalización de escenarios territoriales
32 - Validación analítica final
33 - Síntesis estratégica
34 - Priorización territorial
35 - Construcción de cartera territorial
36 - Validación geoespacial

Objetivo:
    Construir una salida territorial integrada, auditable y consistente,
    preservando la asignación proyecto -> escenario, indicadores y geometrías.

Entradas principales:
    escenarios_territoriales_amba_v4.parquet
    ranking_escenarios_v4.csv
    priorizacion_territorial_escenarios_v4.csv
    cartera_territorial_amba_v4.csv
    cartera_proyectos_v4.csv
    validacion_geoespacial_cartera_v4.csv
    geometria_cartera_proyectos_v4.gpkg
    geometria_escenarios_cartera_v4.gpkg

Salidas:
    modelo_territorial_amba_v4.parquet
    modelo_territorial_amba_v4.gpkg
    resumen_37_modelo_territorial_amba.json
    auditoria_37_modelo_territorial_amba.csv
    indicadores_modelo_territorial_amba_v4.csv
    escenarios_modelo_territorial_amba_v4.csv
    proyectos_modelo_territorial_amba_v4.csv
    sintesis_modelo_territorial_amba_v4.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4"
PROCESO = "37"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "escenarios_territoriales_amba"

INPUT_CANONICAL = DATA_DIR / "escenarios_territoriales_amba_v4.parquet"

INPUT_RANKING = DATA_DIR / "ranking_escenarios_v4.csv"

INPUT_PRIORIZACION = (
    DATA_DIR / "priorizacion_territorial_escenarios_v4.csv"
)

INPUT_CARTERA = DATA_DIR / "cartera_territorial_amba_v4.csv"

INPUT_PROYECTOS = DATA_DIR / "cartera_proyectos_v4.csv"

INPUT_VALIDACION_GEO = (
    DATA_DIR / "validacion_geoespacial_cartera_v4.csv"
)

INPUT_GPKG_PROYECTOS = (
    DATA_DIR / "geometria_cartera_proyectos_v4.gpkg"
)

INPUT_GPKG_ESCENARIOS = (
    DATA_DIR / "geometria_escenarios_cartera_v4.gpkg"
)

OUTPUT_PARQUET = (
    DATA_DIR / "modelo_territorial_amba_v4.parquet"
)

OUTPUT_GPKG = (
    DATA_DIR / "modelo_territorial_amba_v4.gpkg"
)

OUTPUT_RESUMEN = (
    DATA_DIR / "resumen_37_modelo_territorial_amba.json"
)

OUTPUT_AUDITORIA = (
    DATA_DIR / "auditoria_37_modelo_territorial_amba.csv"
)

OUTPUT_INDICADORES = (
    DATA_DIR / "indicadores_modelo_territorial_amba_v4.csv"
)

OUTPUT_ESCENARIOS = (
    DATA_DIR / "escenarios_modelo_territorial_amba_v4.csv"
)

OUTPUT_PROYECTOS = (
    DATA_DIR / "proyectos_modelo_territorial_amba_v4.csv"
)

OUTPUT_MARKDOWN = (
    DATA_DIR / "sintesis_modelo_territorial_amba_v4.md"
)


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def log(texto: str) -> None:
    print(texto)


def normalizar_texto(valor) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def resolver_columna(
    df: pd.DataFrame,
    candidatos: list[str],
    obligatoria: bool = True,
) -> Optional[str]:
    """
    Resuelve una columna tolerando diferencias menores de nomenclatura.
    """
    columnas = {str(c).strip().lower(): c for c in df.columns}

    for candidato in candidatos:
        clave = candidato.strip().lower()
        if clave in columnas:
            return columnas[clave]

    # segunda pasada: comparación sin guiones/espacios
    def compactar(x: str) -> str:
        return (
            x.lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

    columnas_compactadas = {
        compactar(str(c)): c for c in df.columns
    }

    for candidato in candidatos:
        clave = compactar(candidato)
        if clave in columnas_compactadas:
            return columnas_compactadas[clave]

    if obligatoria:
        raise KeyError(
            f"No se encontró ninguna de las columnas esperadas: "
            f"{candidatos}"
        )

    return None


def serie_numerica(
    df: pd.DataFrame,
    columna: Optional[str],
) -> pd.Series:
    if columna is None:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

    return pd.to_numeric(
        df[columna],
        errors="coerce",
    )


def media_segura(serie: pd.Series) -> float:
    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    return float(valores.mean())


def suma_segura(serie: pd.Series) -> float:
    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    return float(valores.sum())


def minimo_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    return float(valores.min())


def maximo_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    return float(valores.max())


def cv_seguro(serie: pd.Series) -> float:
    valores = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if valores.empty:
        return 0.0

    media = float(valores.mean())

    if media == 0:
        return 0.0

    return float(valores.std(ddof=0) / media)


def porcentaje(valor: float) -> float:
    return round(float(valor) * 100.0, 4)


def redondear_dataframe(
    df: pd.DataFrame,
    decimales: int = 4,
) -> pd.DataFrame:

    salida = df.copy()

    for col in salida.columns:
        if pd.api.types.is_numeric_dtype(salida[col]):
            salida[col] = salida[col].round(decimales)

    return salida


def geometria_valida(geom) -> bool:
    if geom is None:
        return False

    try:
        if geom.is_empty:
            return False

        return bool(geom.is_valid)

    except Exception:
        return False


# =============================================================================
# CARGA
# =============================================================================

def cargar_canonical() -> gpd.GeoDataFrame:

    titulo("CARGANDO FUENTE CANÓNICA V4")

    if not INPUT_CANONICAL.exists():
        raise FileNotFoundError(
            f"No existe la entrada canónica:\n{INPUT_CANONICAL}"
        )

    gdf = gpd.read_parquet(INPUT_CANONICAL)

    log(f"Registros : {len(gdf):,}")
    log(f"Columnas  : {len(gdf.columns)}")
    log(f"CRS       : {gdf.crs}")

    return gdf


def cargar_csv(
    path: Path,
    obligatorio: bool = False,
) -> Optional[pd.DataFrame]:

    if not path.exists():

        if obligatorio:
            raise FileNotFoundError(
                f"No existe el archivo requerido:\n{path}"
            )

        log(f"ADVERTENCIA: no existe {path.name}")
        return None

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    log(f"{path.name}: {len(df):,} registros")

    return df


def cargar_geometrias_proyectos() -> Optional[gpd.GeoDataFrame]:

    titulo("CARGANDO GEOMETRÍAS DE CARTERA")

    if not INPUT_GPKG_PROYECTOS.exists():
        log(
            "ADVERTENCIA: no existe el GeoPackage "
            f"{INPUT_GPKG_PROYECTOS.name}"
        )
        return None

    try:

        gdf = gpd.read_file(
            INPUT_GPKG_PROYECTOS,
        )

        log(f"Registros : {len(gdf):,}")
        log(f"CRS       : {gdf.crs}")

        return gdf

    except Exception as exc:

        log(
            "ADVERTENCIA: no se pudo cargar geometría de proyectos: "
            f"{exc}"
        )

        return None


# =============================================================================
# RESOLUCIÓN DE CAMPOS
# =============================================================================

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
        obligatoria=False,
    )

    campos["dimension"] = resolver_columna(
        gdf,
        [
            "dimension_dominante",
            "dimension_escenario",
            "dimension",
        ],
        obligatoria=False,
    )

    campos["prioridad"] = resolver_columna(
        gdf,
        [
            "prioridad_territorial_v4",
            "prioridad_territorial",
            "prioridad_escenario",
        ],
        obligatoria=False,
    )

    campos["score_priorizacion"] = resolver_columna(
        gdf,
        [
            "score_priorizacion_v4",
            "score_priorizacion",
        ],
        obligatoria=False,
    )

    campos["score_cartera"] = resolver_columna(
        gdf,
        [
            "score_cartera_v4",
            "score_cartera",
        ],
        obligatoria=False,
    )

    campos["demanda"] = resolver_columna(
        gdf,
        [
            "indice_demanda_estructural",
            "indice_demanda",
        ],
        obligatoria=False,
    )

    campos["deficit"] = resolver_columna(
        gdf,
        [
            "deficit_infraestructura",
            "deficit",
        ],
        obligatoria=False,
    )

    campos["conectividad"] = resolver_columna(
        gdf,
        [
            "indice_conectividad_estructural",
            "indice_conectividad",
        ],
        obligatoria=False,
    )

    campos["intermodalidad"] = resolver_columna(
        gdf,
        [
            "indice_intermodalidad_estructural",
            "indice_intermodalidad",
        ],
        obligatoria=False,
    )

    campos["integracion"] = resolver_columna(
        gdf,
        [
            "indice_integracion_territorial",
            "indice_integracion",
        ],
        obligatoria=False,
    )

    campos["centralidad"] = resolver_columna(
        gdf,
        [
            "indice_centralidad_estructural",
            "indice_centralidad",
        ],
        obligatoria=False,
    )

    campos["impacto"] = resolver_columna(
        gdf,
        [
            "impacto_potencial",
            "impacto",
        ],
        obligatoria=False,
    )

    campos["urgencia"] = resolver_columna(
        gdf,
        [
            "urgencia_intervencion",
            "urgencia",
        ],
        obligatoria=False,
    )

    campos["score_territorial"] = resolver_columna(
        gdf,
        [
            "score_prioridad_territorial",
            "prioridad_territorial_original",
        ],
        obligatoria=False,
    )

    campos["geometria"] = "geometry"

    for clave, valor in campos.items():
        log(
            f"{clave:<28}: "
            f"{valor if valor is not None else 'NO DISPONIBLE'}"
        )

    return campos


# =============================================================================
# INTEGRACIÓN DE GEOMETRÍAS
# =============================================================================

def integrar_geometrias(
    gdf: gpd.GeoDataFrame,
    campos: dict,
    geometria_proyectos: Optional[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:

    titulo("INTEGRANDO GEOMETRÍAS CANÓNICAS")

    salida = gdf.copy()

    if "geometry" in salida.columns:

        validas = salida.geometry.apply(
            geometria_valida
        )

        if int(validas.sum()) == len(salida):

            log(
                "Las geometrías de la fuente canónica "
                "son válidas."
            )

            return salida

    if geometria_proyectos is None:

        log(
            "ADVERTENCIA: no se dispone de una fuente "
            "geoespacial alternativa."
        )

        return salida

    campo_g = resolver_columna(
        geometria_proyectos,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
        obligatoria=False,
    )

    if campo_g is None:

        log(
            "ADVERTENCIA: no se encontró proyecto_id "
            "en el GeoPackage."
        )

        return salida

    campo_p = campos["proyecto"]

    mapa = (
        geometria_proyectos[
            [campo_g, "geometry"]
        ]
        .dropna(subset=[campo_g])
        .drop_duplicates(subset=[campo_g])
        .set_index(campo_g)["geometry"]
    )

    salida["geometry"] = (
        salida[campo_p]
        .map(mapa)
    )

    recuperadas = int(
        salida.geometry.apply(
            geometria_valida
        ).sum()
    )

    log(
        f"Geometrías recuperadas : {recuperadas:,}"
    )

    return salida


# =============================================================================
# VALIDACIÓN
# =============================================================================

def validar_base(
    gdf: gpd.GeoDataFrame,
    campos: dict,
) -> dict:

    titulo("VALIDACIÓN BASE DEL MODELO")

    proyecto = gdf[campos["proyecto"]]
    escenario = gdf[campos["escenario"]]

    metricas = {}

    metricas["registros"] = int(len(gdf))

    metricas["proyectos_unicos"] = int(
        proyecto.nunique(dropna=True)
    )

    metricas["proyectos_nulos"] = int(
        proyecto.isna().sum()
    )

    metricas["proyectos_duplicados"] = int(
        proyecto.duplicated().sum()
    )

    metricas["escenarios"] = int(
        escenario.nunique(dropna=True)
    )

    metricas["escenarios_nulos"] = int(
        escenario.isna().sum()
    )

    metricas["asignaciones_completas"] = int(
        (
            proyecto.notna()
            & escenario.notna()
        ).sum()
    )

    log(
        f"Registros              : "
        f"{metricas['registros']:,}"
    )

    log(
        f"Proyectos únicos       : "
        f"{metricas['proyectos_unicos']:,}"
    )

    log(
        f"Proyectos nulos        : "
        f"{metricas['proyectos_nulos']:,}"
    )

    log(
        f"Proyectos duplicados   : "
        f"{metricas['proyectos_duplicados']:,}"
    )

    log(
        f"Escenarios             : "
        f"{metricas['escenarios']:,}"
    )

    log(
        f"Escenarios nulos       : "
        f"{metricas['escenarios_nulos']:,}"
    )

    return metricas


def validar_geometrias(
    gdf: gpd.GeoDataFrame,
) -> dict:

    titulo("VALIDACIÓN GEOESPACIAL")

    if "geometry" not in gdf.columns:

        metricas = {
            "geometrias_validas": 0,
            "geometrias_nulas": len(gdf),
            "geometrias_vacias": 0,
            "geometrias_invalidas": 0,
            "cobertura_geometrica": 0.0,
        }

        log(
            "No existe columna geometry."
        )

        return metricas

    nulas = gdf.geometry.isna()

    vacias = (
        ~nulas
        & gdf.geometry.apply(
            lambda g: bool(g.is_empty)
        )
    )

    invalidas = (
        ~nulas
        & ~vacias
        & gdf.geometry.apply(
            lambda g: not bool(g.is_valid)
        )
    )

    validas = (
        ~nulas
        & ~vacias
        & ~invalidas
    )

    total = len(gdf)

    cobertura = (
        float(validas.sum()) / total
        if total > 0
        else 0.0
    )

    tipos = (
        gdf.loc[validas, "geometry"]
        .geom_type
        .value_counts()
        .to_dict()
    )

    metricas = {
        "geometrias_validas": int(validas.sum()),
        "geometrias_nulas": int(nulas.sum()),
        "geometrias_vacias": int(vacias.sum()),
        "geometrias_invalidas": int(invalidas.sum()),
        "cobertura_geometrica": cobertura,
        "tipos_geometricos": tipos,
    }

    log(
        f"Geometrías válidas      : "
        f"{metricas['geometrias_validas']:,}"
    )

    log(
        f"Geometrías nulas        : "
        f"{metricas['geometrias_nulas']:,}"
    )

    log(
        f"Geometrías vacías       : "
        f"{metricas['geometrias_vacias']:,}"
    )

    log(
        f"Geometrías inválidas    : "
        f"{metricas['geometrias_invalidas']:,}"
    )

    log(
        f"Cobertura geométrica    : "
        f"{cobertura * 100:.2f}%"
    )

    log(
        f"Tipos geométricos       : "
        f"{tipos}"
    )

    return metricas


def validar_consistencia(
    gdf: gpd.GeoDataFrame,
    campos: dict,
) -> dict:

    titulo("VALIDACIÓN DE CONSISTENCIA TERRITORIAL")

    proyecto = campos["proyecto"]
    escenario = campos["escenario"]

    pares = gdf[
        [proyecto, escenario]
    ].dropna()

    escenarios_por_proyecto = (
        pares
        .groupby(proyecto)[escenario]
        .nunique()
    )

    proyectos_multiescenario = int(
        (escenarios_por_proyecto > 1).sum()
    )

    conteos = (
        gdf[escenario]
        .value_counts()
        .sort_index()
    )

    minimo = (
        int(conteos.min())
        if not conteos.empty
        else 0
    )

    maximo = (
        int(conteos.max())
        if not conteos.empty
        else 0
    )

    promedio = (
        float(conteos.mean())
        if not conteos.empty
        else 0.0
    )

    cv = cv_seguro(conteos)

    metricas = {
        "escenarios_detectados": int(
            gdf[escenario].nunique()
        ),
        "proyectos_multiescenario": (
            proyectos_multiescenario
        ),
        "minimo_proyectos_escenario": minimo,
        "maximo_proyectos_escenario": maximo,
        "promedio_proyectos_escenario": promedio,
        "cv_tamano_escenarios": cv,
    }

    log(
        f"Escenarios detectados              : "
        f"{metricas['escenarios_detectados']}"
    )

    log(
        f"Proyectos con múltiples escenarios : "
        f"{proyectos_multiescenario}"
    )

    log(
        f"Mínimo proyectos/escenario        : "
        f"{minimo}"
    )

    log(
        f"Máximo proyectos/escenario        : "
        f"{maximo}"
    )

    log(
        f"Promedio proyectos/escenario      : "
        f"{promedio:.2f}"
    )

    log(
        f"CV tamaño escenarios              : "
        f"{cv:.4f}"
    )

    return metricas


# =============================================================================
# ESCENARIOS
# =============================================================================

def construir_escenarios(
    gdf: gpd.GeoDataFrame,
    campos: dict,
) -> gpd.GeoDataFrame:

    titulo("CONSTRUYENDO MODELO AGREGADO POR ESCENARIO")

    escenario = campos["escenario"]
    proyecto = campos["proyecto"]

    agregaciones = {
        "cantidad_proyectos": (
            proyecto,
            "nunique",
        ),
    }

    for clave in [
        "demanda",
        "deficit",
        "conectividad",
        "intermodalidad",
        "integracion",
        "centralidad",
        "impacto",
        "urgencia",
        "score_priorizacion",
        "score_cartera",
        "score_territorial",
    ]:

        columna = campos.get(clave)

        if columna is not None:
            agregaciones[
                clave
            ] = (
                columna,
                "mean",
            )

    tabla = (
        gdf
        .groupby(
            escenario,
            dropna=False,
        )
        .agg(
            **{
                nombre: pd.NamedAgg(
                    column=columna,
                    aggfunc=funcion,
                )
                for nombre, (
                    columna,
                    funcion,
                ) in agregaciones.items()
            }
        )
        .reset_index()
    )

    # Tipos y dimensiones dominantes
    if campos.get("tipo"):

        tipos = (
            gdf.groupby(escenario)[
                campos["tipo"]
            ]
            .agg(
                lambda x: (
                    x.dropna()
                    .astype(str)
                    .value_counts()
                    .index[0]
                    if not x.dropna().empty
                    else ""
                )
            )
            .reset_index(
                name="tipo_escenario"
            )
        )

        tabla = tabla.merge(
            tipos,
            on=escenario,
            how="left",
        )

    if campos.get("dimension"):

        dimensiones = (
            gdf.groupby(escenario)[
                campos["dimension"]
            ]
            .agg(
                lambda x: (
                    x.dropna()
                    .astype(str)
                    .value_counts()
                    .index[0]
                    if not x.dropna().empty
                    else ""
                )
            )
            .reset_index(
                name="dimension_dominante"
            )
        )

        tabla = tabla.merge(
            dimensiones,
            on=escenario,
            how="left",
        )

    # Prioridad dominante
    if campos.get("prioridad"):

        prioridades = (
            gdf.groupby(escenario)[
                campos["prioridad"]
            ]
            .agg(
                lambda x: (
                    x.dropna()
                    .astype(str)
                    .value_counts()
                    .index[0]
                    if not x.dropna().empty
                    else ""
                )
            )
            .reset_index(
                name="prioridad_territorial"
            )
        )

        tabla = tabla.merge(
            prioridades,
            on=escenario,
            how="left",
        )

    # Geometría
    geometria = (
        gdf
        .groupby(escenario)["geometry"]
        .apply(
            lambda x: (
                x.dropna().unary_union
                if not x.dropna().empty
                else None
            )
        )
        .reset_index()
    )

    escenarios = gpd.GeoDataFrame(
        tabla.merge(
            geometria,
            on=escenario,
            how="left",
        ),
        geometry="geometry",
        crs=gdf.crs,
    )

    return escenarios


# =============================================================================
# MODELO INTEGRADO
# =============================================================================

def construir_modelo_proyectos(
    gdf: gpd.GeoDataFrame,
    campos: dict,
) -> gpd.GeoDataFrame:

    titulo("CONSTRUYENDO MODELO TERRITORIAL INTEGRADO")

    salida = gdf.copy()

    # Nombres canónicos para facilitar integración posterior.
    renombres = {
        campos["proyecto"]: "proyecto_id",
        campos["escenario"]: "escenario_id",
    }

    if campos.get("tipo"):
        renombres[campos["tipo"]] = "tipo_escenario"

    if campos.get("dimension"):
        renombres[campos["dimension"]] = (
            "dimension_dominante"
        )

    # Solo renombrar si no existe ya el nombre destino
    for origen, destino in list(renombres.items()):

        if (
            origen in salida.columns
            and origen != destino
            and destino not in salida.columns
        ):

            salida = salida.rename(
                columns={
                    origen: destino
                }
            )

    # Asegurar identificadores canónicos.
    if "proyecto_id" not in salida.columns:
        salida["proyecto_id"] = (
            gdf[campos["proyecto"]]
        )

    if "escenario_id" not in salida.columns:
        salida["escenario_id"] = (
            gdf[campos["escenario"]]
        )

    # Orden lógico de identificación.
    columnas_inicio = [
        "proyecto_id",
        "escenario_id",
        "tipo_escenario",
        "dimension_dominante",
    ]

    columnas_inicio = [
        c for c in columnas_inicio
        if c in salida.columns
    ]

    restantes = [
        c for c in salida.columns
        if c not in columnas_inicio
        and c != "geometry"
    ]

    salida = salida[
        columnas_inicio
        + restantes
        + ["geometry"]
    ]

    return salida


# =============================================================================
# INDICADORES
# =============================================================================

def construir_indicadores(
    escenarios: gpd.GeoDataFrame,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO INDICADORES INTEGRADOS")

    columnas = [
        "escenario_id",
        "cantidad_proyectos",
        "tipo_escenario",
        "dimension_dominante",
        "prioridad_territorial",
        "demanda",
        "deficit",
        "conectividad",
        "intermodalidad",
        "integracion",
        "centralidad",
        "impacto",
        "urgencia",
        "score_priorizacion",
        "score_cartera",
        "score_territorial",
    ]

    disponibles = [
        c for c in columnas
        if c in escenarios.columns
    ]

    return redondear_dataframe(
        pd.DataFrame(
            escenarios[disponibles]
        )
    )


# =============================================================================
# AUDITORÍA
# =============================================================================

def construir_auditoria(
    base: dict,
    geo: dict,
    consistencia: dict,
) -> pd.DataFrame:

    titulo("CONSTRUYENDO AUDITORÍA DEL PROCESO 37")

    filas = []

    def agregar(
        control: str,
        valor,
        resultado: str,
        detalle: str,
    ):

        filas.append(
            {
                "proceso": PROCESO,
                "control": control,
                "valor": valor,
                "resultado": resultado,
                "detalle": detalle,
            }
        )

    agregar(
        "registros",
        base["registros"],
        "OK",
        "Cantidad total de registros.",
    )

    agregar(
        "proyectos_unicos",
        base["proyectos_unicos"],
        (
            "OK"
            if base["proyectos_unicos"]
            == base["registros"]
            else "OBSERVADO"
        ),
        "Control de unicidad de proyectos.",
    )

    agregar(
        "proyectos_nulos",
        base["proyectos_nulos"],
        (
            "OK"
            if base["proyectos_nulos"] == 0
            else "OBSERVADO"
        ),
        "No debe haber proyectos nulos.",
    )

    agregar(
        "escenarios_nulos",
        base["escenarios_nulos"],
        (
            "OK"
            if base["escenarios_nulos"] == 0
            else "OBSERVADO"
        ),
        "No debe haber escenarios nulos.",
    )

    agregar(
        "geometrias_validas",
        geo["geometrias_validas"],
        (
            "OK"
            if geo["geometrias_validas"]
            == base["registros"]
            else "OBSERVADO"
        ),
        "Todas las geometrías deben ser válidas.",
    )

    agregar(
        "geometrias_nulas",
        geo["geometrias_nulas"],
        (
            "OK"
            if geo["geometrias_nulas"] == 0
            else "OBSERVADO"
        ),
        "No deben existir geometrías nulas.",
    )

    agregar(
        "geometrias_invalidas",
        geo["geometrias_invalidas"],
        (
            "OK"
            if geo["geometrias_invalidas"] == 0
            else "OBSERVADO"
        ),
        "No deben existir geometrías inválidas.",
    )

    agregar(
        "cobertura_geometrica",
        geo["cobertura_geometrica"],
        (
            "OK"
            if geo["cobertura_geometrica"] >= 1.0
            else "OBSERVADO"
        ),
        "Cobertura geoespacial del modelo.",
    )

    agregar(
        "proyectos_multiescenario",
        consistencia["proyectos_multiescenario"],
        (
            "OK"
            if consistencia[
                "proyectos_multiescenario"
            ] == 0
            else "OBSERVADO"
        ),
        "Cada proyecto debe pertenecer a un único escenario.",
    )

    return pd.DataFrame(filas)


# =============================================================================
# SCORE GLOBAL
# =============================================================================

def calcular_score_integracion(
    base: dict,
    geo: dict,
    consistencia: dict,
) -> float:

    score_unicidad = (
        1.0
        if (
            base["proyectos_unicos"]
            == base["registros"]
        )
        else 0.0
    )

    score_nulos = (
        1.0
        if (
            base["proyectos_nulos"] == 0
            and base["escenarios_nulos"] == 0
        )
        else 0.0
    )

    score_geo = float(
        geo["cobertura_geometrica"]
    )

    score_consistencia = (
        1.0
        if consistencia[
            "proyectos_multiescenario"
        ] == 0
        else 0.0
    )

    score = (
        score_unicidad * 0.25
        + score_nulos * 0.20
        + score_geo * 0.30
        + score_consistencia * 0.25
    )

    return float(score)


# =============================================================================
# MARKDOWN
# =============================================================================

def generar_markdown(
    base: dict,
    geo: dict,
    consistencia: dict,
    escenarios: pd.DataFrame,
    score_global: float,
) -> str:

    lineas = []

    lineas.append(
        "# Modelo Territorial Integrado AMBA V4"
    )
    lineas.append("")
    lineas.append(
        "## Proceso 37 — Integración del modelo territorial"
    )
    lineas.append("")

    lineas.append(
        "### Resultado general"
    )
    lineas.append("")

    lineas.append(
        f"- Proyectos: **{base['registros']}**"
    )

    lineas.append(
        f"- Proyectos únicos: **{base['proyectos_unicos']}**"
    )

    lineas.append(
        f"- Escenarios: **{base['escenarios']}**"
    )

    cobertura = geo["cobertura_geometrica"] * 100.0

    lineas.append(
        f"- Cobertura geométrica: **{cobertura:.2f}%**"
    )

    lineas.append(
        f"- Proyectos multiescenario: "
        f"**{consistencia['proyectos_multiescenario']}**"
    )

    lineas.append(
        f"- Score de integración: "
        f"**{score_global * 100:.2f}/100**"
    )

    lineas.append("")

    dictamen = (
        "VALIDADO"
        if score_global >= 0.95
        else "OBSERVADO"
    )

    lineas.append(
        f"### Dictamen: **{dictamen}**"
    )

    lineas.append("")

    lineas.append(
        "## Ranking de escenarios"
    )
    lineas.append("")

    if not escenarios.empty:

        tabla = escenarios.copy()

        columnas = [
            "escenario_id",
            "cantidad_proyectos",
            "tipo_escenario",
            "dimension_dominante",
            "prioridad_territorial",
            "score_priorizacion",
            "score_cartera",
        ]

        columnas = [
            c for c in columnas
            if c in tabla.columns
        ]

        tabla = tabla[columnas]

        lineas.append(
            tabla.to_markdown(
                index=False,
            )
        )

        lineas.append("")

    lineas.append(
        "## Interpretación"
    )
    lineas.append("")

    lineas.append(
        "El proceso 37 integra los resultados de la "
        "normalización, validación analítica, síntesis "
        "estratégica, priorización territorial, construcción "
        "de cartera y validación geoespacial."
    )

    lineas.append("")

    lineas.append(
        "La asignación proyecto → escenario se conserva "
        "sin modificaciones."
    )

    lineas.append("")

    lineas.append(
        "Las geometrías se conservan desde la fuente "
        "territorial canónica y se controlan mediante "
        "validación geoespacial."
    )

    lineas.append("")

    lineas.append(
        "Esta salida constituye la base integrada para "
        "la etapa final de análisis y formulación del "
        "informe territorial AMBA."
    )

    return "\n".join(lineas)


# =============================================================================
# EXPORTACIÓN
# =============================================================================

def exportar_gpkg(
    gdf: gpd.GeoDataFrame,
    escenarios: gpd.GeoDataFrame,
) -> None:

    titulo("EXPORTANDO MODELO GEOGRÁFICO")

    if OUTPUT_GPKG.exists():
        try:
            OUTPUT_GPKG.unlink()
        except Exception:
            pass

    try:

        gdf.to_file(
            OUTPUT_GPKG,
            layer="proyectos",
            driver="GPKG",
        )

        escenarios.to_file(
            OUTPUT_GPKG,
            layer="escenarios",
            driver="GPKG",
        )

        log(
            f"GeoPackage : {OUTPUT_GPKG}"
        )

    except Exception as exc:

        log(
            "ADVERTENCIA GPKG: "
            f"{exc}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    titulo(
        "37 - INTEGRACIÓN DEL MODELO TERRITORIAL AMBA - V4"
    )

    log(
        f"Proyecto : {PROJECT_ROOT}"
    )

    log(
        f"Entrada  : {INPUT_CANONICAL}"
    )

    log(
        f"Salida   : {DATA_DIR}"
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. Carga
    # -------------------------------------------------------------------------

    gdf = cargar_canonical()

    # -------------------------------------------------------------------------
    # 2. Resolución
    # -------------------------------------------------------------------------

    campos = resolver_campos(gdf)

    # -------------------------------------------------------------------------
    # 3. Geometrías
    # -------------------------------------------------------------------------

    geometria_proyectos = (
        cargar_geometrias_proyectos()
    )

    gdf = integrar_geometrias(
        gdf,
        campos,
        geometria_proyectos,
    )

    # -------------------------------------------------------------------------
    # 4. Validaciones
    # -------------------------------------------------------------------------

    base = validar_base(
        gdf,
        campos,
    )

    geo = validar_geometrias(
        gdf,
    )

    consistencia = validar_consistencia(
        gdf,
        campos,
    )

    # -------------------------------------------------------------------------
    # 5. Modelo de proyectos
    # -------------------------------------------------------------------------

    modelo = construir_modelo_proyectos(
        gdf,
        campos,
    )

    # -------------------------------------------------------------------------
    # 6. Modelo de escenarios
    # -------------------------------------------------------------------------

    escenarios = construir_escenarios(
        gdf,
        campos,
    )

    # -------------------------------------------------------------------------
    # 7. Indicadores
    # -------------------------------------------------------------------------

    indicadores = construir_indicadores(
        escenarios,
    )

    # -------------------------------------------------------------------------
    # 8. Auditoría
    # -------------------------------------------------------------------------

    auditoria = construir_auditoria(
        base,
        geo,
        consistencia,
    )

    # -------------------------------------------------------------------------
    # 9. Score
    # -------------------------------------------------------------------------

    score_global = calcular_score_integracion(
        base,
        geo,
        consistencia,
    )

    dictamen = (
        "VALIDADO"
        if score_global >= 0.95
        else "OBSERVADO"
    )

    # -------------------------------------------------------------------------
    # 10. Ranking
    # -------------------------------------------------------------------------

    if "score_priorizacion" in escenarios.columns:

        escenarios["ranking_integrado_v4"] = (
            escenarios[
                "score_priorizacion"
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        escenarios = escenarios.sort_values(
            "ranking_integrado_v4"
        )

    # -------------------------------------------------------------------------
    # 11. Exportaciones tabulares
    # -------------------------------------------------------------------------

    titulo(
        "EXPORTANDO RESULTADOS DEL PROCESO 37"
    )

    modelo.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    escenarios_export = pd.DataFrame(
        escenarios.drop(
            columns=["geometry"],
            errors="ignore",
        )
    )

    escenarios_export = redondear_dataframe(
        escenarios_export
    )

    indicadores.to_csv(
        OUTPUT_INDICADORES,
        index=False,
        encoding="utf-8-sig",
    )

    escenarios_export.to_csv(
        OUTPUT_ESCENARIOS,
        index=False,
        encoding="utf-8-sig",
    )

    modelo_export = pd.DataFrame(
        modelo.drop(
            columns=["geometry"],
            errors="ignore",
        )
    )

    modelo_export.to_csv(
        OUTPUT_PROYECTOS,
        index=False,
        encoding="utf-8-sig",
    )

    auditoria.to_csv(
        OUTPUT_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 12. GeoPackage
    # -------------------------------------------------------------------------

    exportar_gpkg(
        modelo,
        escenarios,
    )

    # -------------------------------------------------------------------------
    # 13. Resumen JSON
    # -------------------------------------------------------------------------

    resumen = {
        "proceso": PROCESO,
        "version": VERSION,
        "proyecto": str(PROJECT_ROOT),
        "entrada": str(INPUT_CANONICAL),
        "salida": str(DATA_DIR),
        "registros": base["registros"],
        "proyectos_unicos": base[
            "proyectos_unicos"
        ],
        "proyectos_nulos": base[
            "proyectos_nulos"
        ],
        "proyectos_duplicados": base[
            "proyectos_duplicados"
        ],
        "escenarios": base["escenarios"],
        "escenarios_nulos": base[
            "escenarios_nulos"
        ],
        "geometrias_validas": geo[
            "geometrias_validas"
        ],
        "geometrias_nulas": geo[
            "geometrias_nulas"
        ],
        "geometrias_vacias": geo[
            "geometrias_vacias"
        ],
        "geometrias_invalidas": geo[
            "geometrias_invalidas"
        ],
        "cobertura_geometrica": geo[
            "cobertura_geometrica"
        ],
        "proyectos_multiescenario": consistencia[
            "proyectos_multiescenario"
        ],
        "minimo_proyectos_escenario": (
            consistencia[
                "minimo_proyectos_escenario"
            ]
        ),
        "maximo_proyectos_escenario": (
            consistencia[
                "maximo_proyectos_escenario"
            ]
        ),
        "promedio_proyectos_escenario": (
            consistencia[
                "promedio_proyectos_escenario"
            ]
        ),
        "cv_tamano_escenarios": (
            consistencia[
                "cv_tamano_escenarios"
            ]
        ),
        "score_integracion": score_global,
        "score_integracion_100": (
            score_global * 100.0
        ),
        "dictamen": dictamen,
        "salidas": {
            "parquet": str(OUTPUT_PARQUET),
            "gpkg": str(OUTPUT_GPKG),
            "indicadores": str(OUTPUT_INDICADORES),
            "escenarios": str(OUTPUT_ESCENARIOS),
            "proyectos": str(OUTPUT_PROYECTOS),
            "auditoria": str(OUTPUT_AUDITORIA),
            "markdown": str(OUTPUT_MARKDOWN),
        },
    }

    with OUTPUT_RESUMEN.open(
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
    # 14. Markdown
    # -------------------------------------------------------------------------

    markdown = generar_markdown(
        base,
        geo,
        consistencia,
        escenarios_export,
        score_global,
    )

    OUTPUT_MARKDOWN.write_text(
        markdown,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 15. Resultado
    # -------------------------------------------------------------------------

    titulo(
        "RESULTADO FINAL DEL PROCESO 37"
    )

    log(
        f"Proyectos                 : "
        f"{base['registros']:,}"
    )

    log(
        f"Proyectos únicos          : "
        f"{base['proyectos_unicos']:,}"
    )

    log(
        f"Escenarios                : "
        f"{base['escenarios']:,}"
    )

    log(
        f"Cobertura geométrica      : "
        f"{geo['cobertura_geometrica'] * 100:.2f}%"
    )

    log(
        f"Geometrías válidas        : "
        f"{geo['geometrias_validas']:,}"
    )

    log(
        f"Geometrías nulas          : "
        f"{geo['geometrias_nulas']:,}"
    )

    log(
        f"Geometrías inválidas      : "
        f"{geo['geometrias_invalidas']:,}"
    )

    log(
        f"Proyectos multiescenario  : "
        f"{consistencia['proyectos_multiescenario']:,}"
    )

    log(
        f"CV tamaño escenarios      : "
        f"{consistencia['cv_tamano_escenarios']:.4f}"
    )

    log(
        f"Score integración         : "
        f"{score_global * 100:.2f}/100"
    )

    log(
        f"Auditoría                 : "
        f"{'OK' if dictamen == 'VALIDADO' else 'OBSERVADO'}"
    )

    log(
        f"Dictamen                  : "
        f"{dictamen}"
    )

    print()

    if dictamen == "VALIDADO":

        log(
            "La integración territorial AMBA V4 fue "
            "validada correctamente."
        )

        log(
            "La asignación proyecto -> escenario se "
            "mantiene íntegra."
        )

        log(
            "Las geometrías presentan cobertura "
            "completa y consistencia espacial."
        )

        log(
            "La salida queda preparada para la "
            "etapa final de integración y elaboración "
            "del informe territorial AMBA."
        )

    else:

        log(
            "La integración presenta observaciones "
            "que deben revisarse antes de continuar."
        )

    titulo(
        "PROCESO 37 FINALIZADO"
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nProceso interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        titulo(
            "ERROR FATAL EN EL PROCESO 37"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise