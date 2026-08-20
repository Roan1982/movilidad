from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import h3

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import wkt, wkb


warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CENTRALIDADES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_centralidades_movilidad.parquet"
)

H3_CENTRALIDADES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_h3_centralidades.parquet"
)

RESUMEN_CENTRALIDADES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_centralidades_resumen.json"
)

OUTPUT_VALIDACION = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_validacion_centralidades.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR
    / "data"
    / "processed"
    / "sube_2025_validacion_centralidades_resumen.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "validacion_centralidades"
)

OUTPUT_MAPA_CENTRALIDADES = (
    OUTPUT_DIR
    / "01_mapa_centralidades.png"
)

OUTPUT_MAPA_DEMANDA = (
    OUTPUT_DIR
    / "02_demanda_vs_centralidad.png"
)

OUTPUT_MAPA_H3 = (
    OUTPUT_DIR
    / "03_mapa_h3_centralidad.png"
)

OUTPUT_RANKING = (
    OUTPUT_DIR
    / "04_ranking_operaciones_vs_centralidad.png"
)

OUTPUT_CONCENTRACION = (
    OUTPUT_DIR
    / "05_concentracion_operaciones.png"
)

OUTPUT_COMPONENTES = (
    OUTPUT_DIR
    / "06_componentes_indice.png"
)

OUTPUT_DISCREPANCIAS = (
    OUTPUT_DIR
    / "07_discrepancias_ranking.png"
)

CRS_GEOGRAFICO = "EPSG:4326"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def validar_archivo(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido:\n{path}"
        )


def convertir_numerico(df, columnas):
    for columna in columnas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )


def safe_float(valor):
    if valor is None:
        return None

    try:
        if pd.isna(valor):
            return None
    except Exception:
        pass

    try:
        return float(valor)
    except Exception:
        return None


def safe_int(valor):
    if valor is None:
        return None

    try:
        if pd.isna(valor):
            return None
    except Exception:
        pass

    try:
        return int(valor)
    except Exception:
        return None


def porcentaje(valor, total):
    if total == 0:
        return 0.0

    return float(
        valor / total * 100
    )


def correlacion_segura(serie_a, serie_b):
    datos = pd.DataFrame(
        {
            "a": serie_a,
            "b": serie_b,
        }
    ).dropna()

    if len(datos) < 2:
        return None

    if datos["a"].nunique() <= 1:
        return None

    if datos["b"].nunique() <= 1:
        return None

    valor = datos["a"].corr(
        datos["b"],
        method="pearson"
    )

    if pd.isna(valor):
        return None

    return float(valor)


def spearman_segura(serie_a, serie_b):
    datos = pd.DataFrame(
        {
            "a": serie_a,
            "b": serie_b,
        }
    ).dropna()

    if len(datos) < 2:
        return None

    if datos["a"].nunique() <= 1:
        return None

    if datos["b"].nunique() <= 1:
        return None

    valor = datos["a"].corr(
        datos["b"],
        method="spearman"
    )

    if pd.isna(valor):
        return None

    return float(valor)


def guardar_figura(path):
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


def geometria_es_valida(geom):
    """
    Valida una geometría Shapely individual.
    """

    if geom is None:
        return False

    try:

        if geom.is_empty:
            return False

        if not geom.is_valid:

            geom = geom.buffer(0)

            if geom is None:
                return False

            if geom.is_empty:
                return False

        return True

    except Exception:
        return False


# ============================================================
# GEOMETRÍA
# ============================================================

def convertir_geometry(valor):
    """
    Convierte distintos formatos a geometría Shapely.

    Soporta:
    - Shapely geometry
    - WKT
    - WKB
    - None / NaN
    """

    if valor is None:
        return None

    try:
        if pd.isna(valor):
            return None
    except Exception:
        pass

    if hasattr(valor, "geom_type"):
        return valor

    if isinstance(
        valor,
        (bytes, bytearray, memoryview)
    ):
        try:
            return wkb.loads(
                bytes(valor)
            )
        except Exception:
            return None

    if isinstance(valor, str):

        texto = valor.strip()

        if not texto:
            return None

        try:
            return wkt.loads(
                texto
            )
        except Exception:
            return None

    return None


def h3_es_valido(h3_id):
    """
    Valida un identificador H3.

    Compatible con versiones actuales de h3.
    """

    if h3_id is None:
        return False

    try:

        h3_id = str(h3_id).strip()

        if not h3_id:
            return False

        return bool(
            h3.is_valid_cell(h3_id)
        )

    except Exception:
        return False


def construir_poligono_h3(h3_id):
    """
    Convierte una celda H3 en Polygon Shapely.

    h3.cell_to_boundary() devuelve:
        [(lat, lon), ...]

    Shapely utiliza:
        [(lon, lat), ...]
    """

    try:

        h3_id = str(h3_id).strip()

        if not h3_es_valido(h3_id):
            return None

        boundary = h3.cell_to_boundary(
            h3_id
        )

        if not boundary:
            return None

        coordenadas = [
            (
                float(lon),
                float(lat)
            )
            for lat, lon in boundary
        ]

        if len(coordenadas) < 3:
            return None

        poligono = Polygon(
            coordenadas
        )

        if poligono.is_empty:
            return None

        if not poligono.is_valid:

            poligono = poligono.buffer(0)

            if poligono is None:
                return None

        if poligono.is_empty:
            return None

        return poligono

    except Exception:
        return None


# ============================================================
# RECONSTRUIR GEOMETRÍA DESDE H3
# ============================================================

def reconstruir_geometria_desde_h3(
    centralidades,
    h3_centralidades
):
    """
    Reconstruye la geometría espacial de cada nodo
    mediante la unión de los H3 asociados.

    centralidades:
        una fila por nodo.

    h3_centralidades:
        una fila por H3.

    Relación:
        nodo_id
    """

    print()
    print(
        "Reconstruyendo geometrías de nodos "
        "desde los H3 reales..."
    )

    # ========================================================
    # VALIDACIONES
    # ========================================================

    if not isinstance(
        centralidades,
        (pd.DataFrame, gpd.GeoDataFrame)
    ):
        raise TypeError(
            "centralidades debe ser DataFrame "
            "o GeoDataFrame."
        )

    if not isinstance(
        h3_centralidades,
        (pd.DataFrame, gpd.GeoDataFrame)
    ):
        raise TypeError(
            "h3_centralidades debe ser DataFrame "
            "o GeoDataFrame."
        )

    if "nodo_id" not in centralidades.columns:
        raise ValueError(
            "centralidades no contiene nodo_id."
        )

    if "nodo_id" not in h3_centralidades.columns:
        raise ValueError(
            "h3_centralidades no contiene nodo_id."
        )

    if "id_h3" not in h3_centralidades.columns:
        raise ValueError(
            "h3_centralidades no contiene id_h3."
        )

    # ========================================================
    # COPIAS
    # ========================================================

    nodos = centralidades.copy()
    h3_datos = h3_centralidades.copy()

    # La geometría H3 se reconstruye desde id_h3.
    # Si viene una geometría previa, la descartamos.
    if "geometry" in h3_datos.columns:

        h3_datos = h3_datos.drop(
            columns=["geometry"],
            errors="ignore"
        )

    # ========================================================
    # NORMALIZAR NODO_ID
    # ========================================================

    nodos["_nodo_id"] = pd.to_numeric(
        nodos["nodo_id"],
        errors="coerce"
    )

    h3_datos["_nodo_id"] = pd.to_numeric(
        h3_datos["nodo_id"],
        errors="coerce"
    )

    nodos_sin_nodo = int(
        nodos["_nodo_id"].isna().sum()
    )

    h3_sin_nodo = int(
        h3_datos["_nodo_id"].isna().sum()
    )

    if nodos_sin_nodo > 0:

        print(
            f"ADVERTENCIA: nodos sin nodo_id válido: "
            f"{nodos_sin_nodo:,}"
        )

    if h3_sin_nodo > 0:

        print(
            f"ADVERTENCIA: H3 sin nodo_id válido: "
            f"{h3_sin_nodo:,}"
        )

    nodos = nodos[
        nodos["_nodo_id"].notna()
    ].copy()

    h3_datos = h3_datos[
        h3_datos["_nodo_id"].notna()
    ].copy()

    if nodos.empty:
        raise ValueError(
            "No existen nodos con nodo_id válido."
        )

    if h3_datos.empty:
        raise ValueError(
            "No existen H3 con nodo_id válido."
        )

    # ========================================================
    # NORMALIZAR H3
    # ========================================================

    h3_datos["id_h3"] = (
        h3_datos["id_h3"]
        .astype("string")
        .str.strip()
    )

    h3_datos = h3_datos[
        h3_datos["id_h3"].notna()
        &
        (h3_datos["id_h3"] != "")
    ].copy()

    print(
        f"H3 disponibles: "
        f"{len(h3_datos):,}"
    )

    if h3_datos.empty:

        raise ValueError(
            "No existen identificadores H3."
        )

    # ========================================================
    # VALIDAR H3
    # ========================================================

    print(
        "Validando identificadores H3..."
    )

    h3_datos["_h3_valido"] = (
        h3_datos["id_h3"]
        .apply(h3_es_valido)
    )

    h3_invalidos = int(
        (~h3_datos["_h3_valido"]).sum()
    )

    print(
        f"H3 inválidos: "
        f"{h3_invalidos:,}"
    )

    if h3_invalidos > 0:

        print(
            "ADVERTENCIA: se excluirán "
            "los H3 inválidos."
        )

    h3_datos = h3_datos[
        h3_datos["_h3_valido"]
    ].copy()

    if h3_datos.empty:

        raise ValueError(
            "No quedaron H3 válidos."
        )

    # ========================================================
    # DUPLICADOS H3
    # ========================================================

    duplicados_h3 = int(
        h3_datos["id_h3"]
        .duplicated()
        .sum()
    )

    print(
        f"H3 duplicados: "
        f"{duplicados_h3:,}"
    )

    if duplicados_h3 > 0:

        print(
            "\nPrimeros H3 duplicados:"
        )

        print(
            h3_datos[
                h3_datos["id_h3"].duplicated(
                    keep=False
                )
            ]
            [
                [
                    "id_h3",
                    "nodo_id"
                ]
            ]
            .sort_values(
                "id_h3"
            )
            .head(20)
            .to_string(
                index=False
            )
        )

        raise ValueError(
            "Se encontraron H3 duplicados."
        )

    # ========================================================
    # DIAGNÓSTICO DE COBERTURA
    # ========================================================

    nodos_con_h3 = set(
        h3_datos["_nodo_id"]
        .dropna()
        .tolist()
    )

    nodos_sin_h3 = nodos[
        ~nodos["_nodo_id"].isin(
            nodos_con_h3
        )
    ]

    print()
    print(
        "COBERTURA H3 → NODOS"
    )

    print(
        f"  Nodos totales: "
        f"{len(nodos):,}"
    )

    print(
        f"  Nodos con al menos un H3: "
        f"{len(nodos_con_h3):,}"
    )

    print(
        f"  Nodos sin H3: "
        f"{len(nodos_sin_h3):,}"
    )

    print(
        f"  H3 utilizados: "
        f"{len(h3_datos):,}"
    )

    if not nodos_sin_h3.empty:

        print(
            "\nPrimeros nodos sin H3:"
        )

        print(
            nodos_sin_h3[
                ["nodo_id"]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # CONSTRUIR POLÍGONOS
    # ========================================================

    print()
    print(
        "Construyendo polígonos H3..."
    )

    h3_datos["geometry"] = (
        h3_datos["id_h3"]
        .apply(
            construir_poligono_h3
        )
    )

    geometria_valida = (
        h3_datos["geometry"]
        .apply(
            geometria_es_valida
        )
    )

    cantidad_invalidas = int(
        (~geometria_valida).sum()
    )

    print(
        f"Geometrías H3 inválidas: "
        f"{cantidad_invalidas:,}"
    )

    h3_datos = h3_datos[
        geometria_valida
    ].copy()

    if h3_datos.empty:

        raise ValueError(
            "No fue posible construir "
            "ninguna geometría H3."
        )

    print(
        f"Polígonos H3 construidos: "
        f"{len(h3_datos):,}"
    )

    # ========================================================
    # GEODATAFRAME H3
    # ========================================================

    h3_geo = gpd.GeoDataFrame(
        h3_datos,
        geometry="geometry",
        crs=CRS_GEOGRAFICO
    )

    # ========================================================
    # CANTIDAD H3 POR NODO
    # ========================================================

    cantidad_h3_por_nodo = (
        h3_geo
        .groupby("_nodo_id")
        .size()
        .rename(
            "cantidad_h3_calculada"
        )
    )

    print(
        f"Nodos con H3 geométrico: "
        f"{len(cantidad_h3_por_nodo):,}"
    )

    # ========================================================
    # UNIÓN GEOMÉTRICA
    # ========================================================

    print()
    print(
        "Uniendo geometrías H3 por nodo..."
    )

    geometrias_nodos = []

    errores_union = 0

    for nodo_id, grupo in h3_geo.groupby(
        "_nodo_id"
    ):

        try:

            geometrias = [
                geom
                for geom in grupo.geometry
                if geom is not None
                and not geom.is_empty
            ]

            if not geometrias:
                continue

            geometria = unary_union(
                geometrias
            )

            if geometria is None:
                continue

            if geometria.is_empty:
                continue

            if not geometria.is_valid:

                geometria = (
                    geometria
                    .buffer(0)
                )

            if geometria is None:
                continue

            if geometria.is_empty:
                continue

            geometrias_nodos.append(
                {
                    "_nodo_id":
                        nodo_id,

                    "geometry":
                        geometria,
                }
            )

        except Exception as error:

            errores_union += 1

            print(
                f"ADVERTENCIA: no se pudo "
                f"unir el nodo {nodo_id}: "
                f"{error}"
            )

    print(
        f"Errores de unión geométrica: "
        f"{errores_union:,}"
    )

    if not geometrias_nodos:

        raise ValueError(
            "No fue posible construir "
            "geometrías de nodos."
        )

    # ========================================================
    # GEODATAFRAME NODOS
    # ========================================================

    geometria_nodos = gpd.GeoDataFrame(
        geometrias_nodos,
        geometry="geometry",
        crs=CRS_GEOGRAFICO
    )

    print(
        f"Geometrías de nodos reconstruidas: "
        f"{len(geometria_nodos):,}"
    )

    # ========================================================
    # MERGE
    # ========================================================

    resultado = nodos.merge(
        geometria_nodos[
            [
                "_nodo_id",
                "geometry"
            ]
        ],
        on="_nodo_id",
        how="left",
        validate="one_to_one"
    )

    resultado = resultado.drop(
        columns=[
            "_nodo_id"
        ],
        errors="ignore"
    )

    resultado = gpd.GeoDataFrame(
        resultado,
        geometry="geometry",
        crs=CRS_GEOGRAFICO
    )

    # ========================================================
    # VALIDACIÓN FINAL GEOMETRÍA
    # ========================================================

    geometria_final_valida = (
        resultado["geometry"]
        .apply(
            geometria_es_valida
        )
    )

    nodos_con_geometria = int(
        geometria_final_valida.sum()
    )

    nodos_sin_geometria = int(
        (~geometria_final_valida).sum()
    )

    print()
    print(
        "VALIDACIÓN GEOMÉTRICA FINAL"
    )

    print(
        f"  Nodos totales: "
        f"{len(resultado):,}"
    )

    print(
        f"  Nodos con geometría: "
        f"{nodos_con_geometria:,}"
    )

    print(
        f"  Nodos sin geometría: "
        f"{nodos_sin_geometria:,}"
    )

    if nodos_con_geometria == 0:

        raise ValueError(
            "No se reconstruyó ninguna "
            "geometría de nodo."
        )

    # ========================================================
    # COMPARAR CANTIDAD H3
    # ========================================================

    if "h3" in resultado.columns:

        resultado[
            "_cantidad_h3_calculada"
        ] = (
            resultado["nodo_id"]
            .map(
                cantidad_h3_por_nodo
            )
            .fillna(0)
        )

        resultado[
            "_h3_original"
        ] = pd.to_numeric(
            resultado["h3"],
            errors="coerce"
        )

        resultado[
            "_diferencia_h3"
        ] = (
            resultado[
                "_h3_original"
            ]
            -
            resultado[
                "_cantidad_h3_calculada"
            ]
        )

        comparables = (
            resultado[
                "_h3_original"
            ].notna()
        )

        coincidencias = int(
            (
                resultado.loc[
                    comparables,
                    "_diferencia_h3"
                ] == 0
            ).sum()
        )

        discrepancias = int(
            (
                resultado.loc[
                    comparables,
                    "_diferencia_h3"
                ] != 0
            ).sum()
        )

        print()
        print(
            "VALIDACIÓN CANTIDAD H3 POR NODO"
        )

        print(
            f"  Nodos comparables: "
            f"{int(comparables.sum()):,}"
        )

        print(
            f"  Coincidencias: "
            f"{coincidencias:,}"
        )

        print(
            f"  Discrepancias: "
            f"{discrepancias:,}"
        )

        if discrepancias > 0:

            print(
                "\nPrimeras discrepancias:"
            )

            print(
                resultado.loc[
                    comparables
                    &
                    (
                        resultado[
                            "_diferencia_h3"
                        ] != 0
                    ),
                    [
                        "nodo_id",
                        "_h3_original",
                        "_cantidad_h3_calculada",
                        "_diferencia_h3"
                    ]
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

        resultado = resultado.drop(
            columns=[
                "_cantidad_h3_calculada",
                "_h3_original",
                "_diferencia_h3"
            ],
            errors="ignore"
        )

    return resultado


# ============================================================
# CARGAR CENTRALIDADES
# ============================================================

def cargar_centralidades(
    path,
    h3_centralidades
):
    """
    Carga centralidades.

    Si existe geometría GeoParquet válida,
    se utiliza directamente.

    Si el Parquet no posee metadata GeoParquet,
    se reconstruye la geometría desde los H3.
    """

    print(
        "Intentando cargar como GeoParquet..."
    )

    try:

        datos = gpd.read_parquet(
            path
        )

        print(
            "Archivo leído como GeoParquet."
        )

        if (
            "geometry" in datos.columns
            and datos.geometry.notna().any()
        ):

            if datos.crs is None:

                datos = datos.set_crs(
                    CRS_GEOGRAFICO
                )

            geometria_valida = (
                datos.geometry.notna()
                &
                ~datos.geometry.is_empty
            )

            if geometria_valida.any():

                print(
                    "Geometría GeoParquet válida encontrada."
                )

                return datos

    except Exception as error:

        print(
            "No se pudo utilizar directamente "
            "como GeoParquet."
        )

        print(
            f"Motivo: {error}"
        )

    print(
        "Cargando centralidades mediante pandas..."
    )

    datos = pd.read_parquet(
        path
    )

    print(
        f"Registros cargados: "
        f"{len(datos):,}"
    )

    return reconstruir_geometria_desde_h3(
        datos,
        h3_centralidades
    )


# ============================================================
# CUARTILES SEGUROS
# ============================================================

def calcular_cuartiles(serie):

    resultado = pd.Series(
        pd.NA,
        index=serie.index,
        dtype="string"
    )

    datos = pd.to_numeric(
        serie,
        errors="coerce"
    )

    validos = datos.notna()

    cantidad = int(
        validos.sum()
    )

    if cantidad == 0:
        return resultado

    if cantidad < 4:

        resultado.loc[
            validos
        ] = "Q4_ALTO"

        return resultado

    ranking = (
        datos[validos]
        .rank(
            method="first"
        )
    )

    try:

        resultado.loc[
            validos
        ] = pd.qcut(
            ranking,
            4,
            labels=[
                "Q1_BAJO",
                "Q2_MEDIO_BAJO",
                "Q3_MEDIO_ALTO",
                "Q4_ALTO",
            ]
        ).astype("string")

    except Exception:

        resultado.loc[
            validos
        ] = "Q4_ALTO"

    return resultado


# ============================================================
# CLASIFICAR DISCREPANCIA
# ============================================================

def clasificar_discrepancia(row):

    diferencia = safe_float(
        row[
            "diferencia_ranking"
        ]
    )

    if diferencia is None:
        return "SIN_DATOS"

    if diferencia >= 20:

        return (
            "CENTRALIDAD_MUCHO_MAYOR_QUE_DEMANDA"
        )

    if diferencia >= 10:

        return (
            "CENTRALIDAD_MAYOR_QUE_DEMANDA"
        )

    if diferencia <= -20:

        return (
            "DEMANDA_MUCHO_MAYOR_QUE_CENTRALIDAD"
        )

    if diferencia <= -10:

        return (
            "DEMANDA_MAYOR_QUE_CENTRALIDAD"
        )

    return "ALINEADO"


# ============================================================
# MATRIZ DEMANDA VS CENTRALIDAD
# ============================================================

def clasificar_matriz(row):

    demanda = row[
        "cuartil_operaciones"
    ]

    centralidad = row[
        "cuartil_centralidad"
    ]

    demanda_alta = demanda in [
        "Q3_MEDIO_ALTO",
        "Q4_ALTO",
    ]

    centralidad_alta = centralidad in [
        "Q3_MEDIO_ALTO",
        "Q4_ALTO",
    ]

    if demanda_alta and centralidad_alta:

        return (
            "ALTA_DEMANDA_ALTA_CENTRALIDAD"
        )

    if demanda_alta and not centralidad_alta:

        return (
            "ALTA_DEMANDA_BAJA_CENTRALIDAD"
        )

    if not demanda_alta and centralidad_alta:

        return (
            "BAJA_DEMANDA_ALTA_CENTRALIDAD"
        )

    return (
        "BAJA_DEMANDA_BAJA_CENTRALIDAD"
    )


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print(
    "VALIDACIÓN DE CENTRALIDADES "
    "DE MOVILIDAD SUBE 2025"
)
print("=" * 70)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ARCHIVOS
# ============================================================

print(
    "\nValidando archivos de entrada..."
)

validar_archivo(
    CENTRALIDADES_PATH
)

validar_archivo(
    H3_CENTRALIDADES_PATH
)

validar_archivo(
    RESUMEN_CENTRALIDADES_PATH
)

print(
    "Archivos de entrada encontrados correctamente."
)


# ============================================================
# CARGAR H3
# ============================================================

print(
    "\nCargando H3 → centralidad..."
)

h3_centralidades = pd.read_parquet(
    H3_CENTRALIDADES_PATH
)

print(
    f"H3 cargados: "
    f"{len(h3_centralidades):,}"
)

print(
    "Columnas:"
)

print(
    h3_centralidades.columns.tolist()
)


columnas_h3_necesarias = [
    "id_h3",
    "nodo_id",
    "indice_centralidad",
    "categoria_centralidad",
]

faltantes_h3 = [
    columna
    for columna in columnas_h3_necesarias
    if columna not in h3_centralidades.columns
]

if faltantes_h3:

    raise ValueError(
        "Faltan columnas en "
        "sube_2025_h3_centralidades.parquet:\n"
        +
        "\n".join(
            f" - {columna}"
            for columna in faltantes_h3
        )
    )


# ============================================================
# CARGAR CENTRALIDADES
# ============================================================

print(
    "\nCargando centralidades..."
)

centralidades = cargar_centralidades(
    CENTRALIDADES_PATH,
    h3_centralidades,
)

print(
    f"Centralidades cargadas: "
    f"{len(centralidades):,}"
)

print(
    "Columnas:"
)

print(
    centralidades.columns.tolist()
)


# ============================================================
# COLUMNAS OBLIGATORIAS
# ============================================================

columnas_obligatorias = [
    "nodo_id",
    "operaciones",
    "indice_centralidad",
    "categoria_centralidad",
    "tipo_centralidad",
    "ranking_centralidad",
    "pct_operaciones",
    "pct_operaciones_acumulado",
]

faltantes = [
    columna
    for columna in columnas_obligatorias
    if columna not in centralidades.columns
]

if faltantes:

    raise ValueError(
        "Faltan columnas obligatorias:\n"
        +
        "\n".join(
            f" - {columna}"
            for columna in faltantes
        )
    )


# ============================================================
# GEOMETRÍA
# ============================================================

print(
    "\nValidando geometrías..."
)

if "geometry" not in centralidades.columns:

    raise ValueError(
        "No existe columna geometry."
    )

if centralidades.crs is None:

    centralidades = centralidades.set_crs(
        CRS_GEOGRAFICO
    )

print(
    f"CRS: {centralidades.crs}"
)

geometrias_validas = (
    centralidades["geometry"]
    .apply(
        geometria_es_valida
    )
)

cantidad_geometrias_validas = int(
    geometrias_validas.sum()
)

print(
    f"Geometrías válidas: "
    f"{cantidad_geometrias_validas:,}"
)

if cantidad_geometrias_validas == 0:

    raise ValueError(
        "No existen geometrías válidas."
    )


# ============================================================
# CONVERSIONES NUMÉRICAS
# ============================================================

columnas_numericas = [
    "nodo_id",
    "h3",
    "operaciones",
    "superficie_km2",
    "operaciones_por_km2",
    "cantidad_corredores",
    "cantidad_clusters",
    "cantidad_jurisdicciones",
    "cantidad_provincias",
    "score_demanda",
    "score_densidad",
    "score_conectividad",
    "score_intermodalidad",
    "score_alcance",
    "score_integracion",
    "indice_centralidad",
    "ranking_centralidad",
    "pct_operaciones",
    "pct_operaciones_acumulado",
]

convertir_numerico(
    centralidades,
    columnas_numericas
)


# ============================================================
# RESUMEN ORIGINAL
# ============================================================

print(
    "\nCargando resumen original..."
)

with open(
    RESUMEN_CENTRALIDADES_PATH,
    "r",
    encoding="utf-8"
) as archivo:

    resumen_original = json.load(
        archivo
    )


# ============================================================
# IDENTIFICADORES
# ============================================================

print(
    "\nValidando identificadores..."
)

duplicados_nodo = int(
    centralidades[
        "nodo_id"
    ]
    .duplicated()
    .sum()
)

print(
    f"Nodos duplicados: "
    f"{duplicados_nodo:,}"
)

if duplicados_nodo > 0:

    raise ValueError(
        "Existen nodos duplicados."
    )


h3_ids = (
    h3_centralidades[
        "id_h3"
    ]
    .astype("string")
    .str.strip()
)

duplicados_h3 = int(
    h3_ids.duplicated().sum()
)

print(
    f"H3 duplicados: "
    f"{duplicados_h3:,}"
)

if duplicados_h3 > 0:

    raise ValueError(
        "Existen H3 duplicados."
    )


# ============================================================
# COPIA DE VALIDACIÓN
# ============================================================

validacion = centralidades.copy()


# ============================================================
# RANKING OPERACIONES
# ============================================================

print(
    "\nCalculando ranking por operaciones..."
)

ranking_operaciones = (
    validacion
    .sort_values(
        [
            "operaciones",
            "indice_centralidad",
            "nodo_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)

ranking_operaciones[
    "ranking_operaciones_validacion"
] = np.arange(
    1,
    len(ranking_operaciones) + 1
)


# ============================================================
# RANKING CENTRALIDAD
# ============================================================

print(
    "Calculando ranking por centralidad..."
)

ranking_centralidad = (
    validacion
    .sort_values(
        [
            "indice_centralidad",
            "operaciones",
            "nodo_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)

ranking_centralidad[
    "ranking_centralidad_validacion"
] = np.arange(
    1,
    len(ranking_centralidad) + 1
)


# ============================================================
# UNIFICAR
# ============================================================

validacion = (
    validacion
    .merge(
        ranking_operaciones[
            [
                "nodo_id",
                "ranking_operaciones_validacion",
            ]
        ],
        on="nodo_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        ranking_centralidad[
            [
                "nodo_id",
                "ranking_centralidad_validacion",
            ]
        ],
        on="nodo_id",
        how="left",
        validate="one_to_one"
    )
)


# ============================================================
# DIFERENCIA RANKING
# ============================================================

validacion[
    "diferencia_ranking"
] = (
    validacion[
        "ranking_operaciones_validacion"
    ]
    -
    validacion[
        "ranking_centralidad_validacion"
    ]
)

validacion[
    "diferencia_ranking_abs"
] = (
    validacion[
        "diferencia_ranking"
    ]
    .abs()
)


# ============================================================
# PERCENTILES
# ============================================================

cantidad_nodos = len(
    validacion
)

if cantidad_nodos > 0:

    validacion[
        "percentil_operaciones"
    ] = (
        100
        *
        (
            cantidad_nodos
            -
            validacion[
                "ranking_operaciones_validacion"
            ]
            + 1
        )
        /
        cantidad_nodos
    )

    validacion[
        "percentil_centralidad"
    ] = (
        100
        *
        (
            cantidad_nodos
            -
            validacion[
                "ranking_centralidad_validacion"
            ]
            + 1
        )
        /
        cantidad_nodos
    )


# ============================================================
# CLASIFICACIÓN DISCREPANCIAS
# ============================================================

validacion[
    "tipo_discrepancia"
] = validacion.apply(
    clasificar_discrepancia,
    axis=1
)


# ============================================================
# CUARTILES
# ============================================================

validacion[
    "cuartil_operaciones"
] = calcular_cuartiles(
    validacion[
        "operaciones"
    ]
)

validacion[
    "cuartil_centralidad"
] = calcular_cuartiles(
    validacion[
        "indice_centralidad"
    ]
)


# ============================================================
# MATRIZ DEMANDA VS CENTRALIDAD
# ============================================================

validacion[
    "matriz_demanda_centralidad"
] = validacion.apply(
    clasificar_matriz,
    axis=1
)


# ============================================================
# CORRELACIONES
# ============================================================

print(
    "\nCalculando correlaciones..."
)

correlaciones = {}

variables_correlacion = [
    (
        "operaciones",
        "Operaciones"
    ),
    (
        "operaciones_por_km2",
        "Operaciones por km²"
    ),
    (
        "cantidad_corredores",
        "Cantidad de corredores"
    ),
    (
        "cantidad_clusters",
        "Cantidad de clusters"
    ),
    (
        "cantidad_jurisdicciones",
        "Cantidad de jurisdicciones"
    ),
    (
        "score_intermodalidad",
        "Score de intermodalidad"
    ),
    (
        "score_demanda",
        "Score de demanda"
    ),
    (
        "score_densidad",
        "Score de densidad"
    ),
    (
        "score_conectividad",
        "Score de conectividad"
    ),
    (
        "score_alcance",
        "Score de alcance"
    ),
    (
        "score_integracion",
        "Score de integración"
    ),
]

for columna, nombre in variables_correlacion:

    if columna not in validacion.columns:
        continue

    correlaciones[columna] = {

        "nombre":
            nombre,

        "pearson":
            correlacion_segura(
                validacion[columna],
                validacion[
                    "indice_centralidad"
                ]
            ),

        "spearman":
            spearman_segura(
                validacion[columna],
                validacion[
                    "indice_centralidad"
                ]
            ),
    }


# ============================================================
# CONCENTRACIÓN
# ============================================================

print(
    "\nAnalizando concentración..."
)

orden_demanda = (
    validacion
    .sort_values(
        [
            "operaciones",
            "nodo_id",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)

operaciones_total = float(
    orden_demanda[
        "operaciones"
    ]
    .fillna(0)
    .sum()
)

if operaciones_total > 0:

    orden_demanda[
        "pct_operaciones_validacion"
    ] = (
        orden_demanda[
            "operaciones"
        ]
        .fillna(0)
        /
        operaciones_total
        *
        100
    )

else:

    orden_demanda[
        "pct_operaciones_validacion"
    ] = 0.0


orden_demanda[
    "pct_acumulado_validacion"
] = (
    orden_demanda[
        "pct_operaciones_validacion"
    ]
    .cumsum()
)


def concentracion_top(n):

    if operaciones_total == 0:
        return 0.0

    return porcentaje(
        orden_demanda[
            "operaciones"
        ]
        .fillna(0)
        .head(n)
        .sum(),
        operaciones_total
    )


concentracion = {

    "top_1":
        concentracion_top(1),

    "top_5":
        concentracion_top(5),

    "top_10":
        concentracion_top(10),

    "top_20":
        concentracion_top(20),

    "top_30":
        concentracion_top(30),

    "top_50":
        concentracion_top(50),
}


# ============================================================
# NODOS PRINCIPALES
# ============================================================

validacion_con_datos = validacion[
    validacion[
        "indice_centralidad"
    ].notna()
].copy()

if validacion_con_datos.empty:

    raise ValueError(
        "No existen nodos con índice de centralidad válido."
    )


principal = (
    validacion_con_datos
    .sort_values(
        [
            "indice_centralidad",
            "operaciones",
            "nodo_id",
        ],
        ascending=[
            False,
            False,
            True,
        ]
    )
    .iloc[0]
)


nodo_mayor_demanda = (
    validacion
    .sort_values(
        [
            "operaciones",
            "indice_centralidad",
            "nodo_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last"
    )
    .iloc[0]
)


nodo_mayor_discrepancia = (
    validacion
    .sort_values(
        [
            "diferencia_ranking_abs",
            "nodo_id",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last"
    )
    .iloc[0]
)


# ============================================================
# NODOS 1 Y 6
# ============================================================

print()
print("=" * 70)
print(
    "VALIDACIÓN DE NODOS DESTACADOS"
)
print("=" * 70)

for nodo_id in [1, 6]:

    datos = validacion[
        validacion["nodo_id"] == nodo_id
    ]

    if datos.empty:

        print(
            f"\nNodo {nodo_id}: no encontrado."
        )

        continue

    fila = datos.iloc[0]

    print(
        f"\nNodo {nodo_id}"
    )

    print(
        f"  Ranking operaciones: "
        f"{safe_int(fila['ranking_operaciones_validacion'])}"
    )

    print(
        f"  Ranking centralidad: "
        f"{safe_int(fila['ranking_centralidad_validacion'])}"
    )

    print(
        f"  Operaciones: "
        f"{safe_float(fila['operaciones']) or 0:,.0f}"
    )

    print(
        f"  Índice: "
        f"{safe_float(fila['indice_centralidad']) or 0:.2f}"
    )

    print(
        f"  Diferencia ranking: "
        f"{safe_int(fila['diferencia_ranking'])}"
    )

    print(
        f"  Tipo centralidad: "
        f"{fila['tipo_centralidad']}"
    )

    print(
        f"  Matriz: "
        f"{fila['matriz_demanda_centralidad']}"
    )


# ============================================================
# TOP DISCREPANCIAS
# ============================================================

print()
print("=" * 70)
print(
    "MAYORES DISCREPANCIAS ENTRE DEMANDA Y CENTRALIDAD"
)
print("=" * 70)

columnas_discrepancias = [
    "nodo_id",
    "operaciones",
    "ranking_operaciones_validacion",
    "ranking_centralidad_validacion",
    "diferencia_ranking",
    "indice_centralidad",
    "categoria_centralidad",
    "tipo_centralidad",
    "matriz_demanda_centralidad",
    "tipo_discrepancia",
]

print(
    validacion
    .sort_values(
        "diferencia_ranking_abs",
        ascending=False,
        na_position="last"
    )
    [
        columnas_discrepancias
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# MAPA 1 — CENTRALIDADES
# ============================================================

print(
    "\nGenerando mapa de centralidades..."
)

mapa = validacion[
    validacion.geometry.notna()
].copy()

mapa = mapa[
    mapa.geometry.apply(
        geometria_es_valida
    )
].copy()

fig, ax = plt.subplots(
    figsize=(14, 12)
)

categorias = [
    "CENTRALIDAD_BAJA",
    "CENTRALIDAD_MEDIA",
    "CENTRALIDAD_ALTA",
    "CENTRALIDAD_CRITICA",
]

for categoria in categorias:

    grupo = mapa[
        mapa[
            "categoria_centralidad"
        ] == categoria
    ]

    if grupo.empty:
        continue

    grupo.plot(
        ax=ax,
        alpha=0.60,
        label=categoria,
        edgecolor="none"
    )


# ------------------------------------------------------------
# TOP 15
# ------------------------------------------------------------

top15 = (
    mapa
    .sort_values(
        [
            "indice_centralidad",
            "operaciones",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .head(15)
)

for _, fila in top15.iterrows():

    try:

        punto = fila.geometry.representative_point()

        nodo = safe_int(
            fila["nodo_id"]
        )

        if nodo is None:
            continue

        ax.annotate(
            str(nodo),
            (
                punto.x,
                punto.y
            ),
            xytext=(
                4,
                4
            ),
            textcoords="offset points",
            fontsize=8
        )

    except Exception:
        pass


ax.set_title(
    "Centralidades de movilidad SUBE 2025",
    fontsize=15
)

ax.set_axis_off()

if ax.get_legend_handles_labels()[0]:

    ax.legend(
        title="Categoría",
        loc="best"
    )

guardar_figura(
    OUTPUT_MAPA_CENTRALIDADES
)


# ============================================================
# GRÁFICO 2 — DEMANDA VS CENTRALIDAD
# ============================================================

print(
    "Generando gráfico demanda vs centralidad..."
)

fig, ax = plt.subplots(
    figsize=(12, 9)
)

datos_scatter = validacion[
    [
        "operaciones",
        "indice_centralidad",
        "categoria_centralidad",
        "cantidad_corredores",
    ]
].copy()

datos_scatter = datos_scatter[
    datos_scatter["operaciones"] > 0
].copy()

datos_scatter = datos_scatter[
    datos_scatter["indice_centralidad"].notna()
].copy()

for categoria in categorias:

    grupo = datos_scatter[
        datos_scatter[
            "categoria_centralidad"
        ] == categoria
    ]

    if grupo.empty:
        continue

    tamanio = (
        grupo[
            "cantidad_corredores"
        ]
        .fillna(0)
        + 1
    ) * 20

    ax.scatter(
        grupo[
            "operaciones"
        ],
        grupo[
            "indice_centralidad"
        ],
        s=tamanio,
        alpha=0.70,
        label=categoria
    )


if not datos_scatter.empty:

    ax.set_xscale(
        "log"
    )


ax.set_xlabel(
    "Operaciones (escala logarítmica)"
)

ax.set_ylabel(
    "Índice de centralidad"
)

ax.set_title(
    "Demanda vs índice de centralidad"
)

ax.grid(
    alpha=0.25
)

if ax.get_legend_handles_labels()[0]:

    ax.legend()


for _, fila in (
    validacion
    .sort_values(
        "indice_centralidad",
        ascending=False
    )
    .head(10)
    .iterrows()
):

    operaciones = safe_float(
        fila["operaciones"]
    )

    indice = safe_float(
        fila["indice_centralidad"]
    )

    nodo = safe_int(
        fila["nodo_id"]
    )

    if (
        operaciones is None
        or operaciones <= 0
        or indice is None
        or nodo is None
    ):
        continue

    ax.annotate(
        str(nodo),
        (
            operaciones,
            indice
        ),
        xytext=(
            5,
            5
        ),
        textcoords="offset points",
        fontsize=8
    )


guardar_figura(
    OUTPUT_MAPA_DEMANDA
)


# ============================================================
# MAPA 3 — H3
# ============================================================

print(
    "Generando mapa H3..."
)

try:

    h3_mapa = h3_centralidades.copy()

    h3_mapa["geometry"] = (
        h3_mapa[
            "id_h3"
        ]
        .astype(str)
        .apply(
            construir_poligono_h3
        )
    )

    h3_mapa = gpd.GeoDataFrame(
        h3_mapa,
        geometry="geometry",
        crs=CRS_GEOGRAFICO
    )

    h3_mapa = h3_mapa[
        h3_mapa.geometry.apply(
            geometria_es_valida
        )
    ].copy()

    if not h3_mapa.empty:

        fig, ax = plt.subplots(
            figsize=(15, 12)
        )

        h3_mapa.plot(
            ax=ax,
            column="indice_centralidad",
            legend=True,
            alpha=0.70,
            edgecolor="none"
        )

        ax.set_title(
            "Distribución H3 según índice de centralidad",
            fontsize=15
        )

        ax.set_axis_off()

        guardar_figura(
            OUTPUT_MAPA_H3
        )

    else:

        print(
            "ADVERTENCIA: no existen geometrías H3."
        )

except Exception as error:

    print(
        "ADVERTENCIA: no se pudo generar "
        "el mapa H3."
    )

    print(
        f"Motivo: {error}"
    )


# ============================================================
# GRÁFICO 4 — RANKING
# ============================================================

print(
    "Generando comparación de rankings..."
)

ranking_plot = (
    validacion
    .sort_values(
        "ranking_centralidad_validacion"
    )
    .head(30)
)

fig, ax = plt.subplots(
    figsize=(14, 9)
)

ax.scatter(
    ranking_plot[
        "ranking_operaciones_validacion"
    ],
    ranking_plot[
        "ranking_centralidad_validacion"
    ],
    alpha=0.75
)

max_ranking = max(
    cantidad_nodos,
    1
)

ax.plot(
    [
        1,
        max_ranking
    ],
    [
        1,
        max_ranking
    ],
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Ranking por operaciones"
)

ax.set_ylabel(
    "Ranking por centralidad"
)

ax.set_title(
    "Ranking por demanda vs ranking por centralidad"
)

ax.grid(
    alpha=0.25
)

for _, fila in ranking_plot.iterrows():

    nodo = safe_int(
        fila["nodo_id"]
    )

    ranking_op = safe_float(
        fila[
            "ranking_operaciones_validacion"
        ]
    )

    ranking_cent = safe_float(
        fila[
            "ranking_centralidad_validacion"
        ]
    )

    if (
        nodo is None
        or ranking_op is None
        or ranking_cent is None
    ):
        continue

    ax.annotate(
        str(nodo),
        (
            ranking_op,
            ranking_cent
        ),
        xytext=(
            4,
            4
        ),
        textcoords="offset points",
        fontsize=7
    )

guardar_figura(
    OUTPUT_RANKING
)


# ============================================================
# GRÁFICO 5 — CONCENTRACIÓN
# ============================================================

print(
    "Generando curva de concentración..."
)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

x = np.arange(
    1,
    len(orden_demanda) + 1
)

y = orden_demanda[
    "pct_acumulado_validacion"
]

ax.plot(
    x,
    y,
    linewidth=2
)

for n in [
    1,
    5,
    10,
    20,
    30,
    50,
]:

    if n <= len(orden_demanda):

        valor = (
            orden_demanda[
                "pct_acumulado_validacion"
            ]
            .iloc[n - 1]
        )

        ax.scatter(
            [n],
            [valor]
        )

        ax.annotate(
            f"Top {n}: {valor:.1f}%",
            (
                n,
                valor
            ),
            xytext=(
                5,
                5
            ),
            textcoords="offset points",
            fontsize=8
        )


ax.set_xlabel(
    "Cantidad acumulada de nodos"
)

ax.set_ylabel(
    "% acumulado de operaciones"
)

ax.set_title(
    "Concentración de operaciones por nodo"
)

ax.grid(
    alpha=0.25
)

guardar_figura(
    OUTPUT_CONCENTRACION
)


# ============================================================
# GRÁFICO 6 — COMPONENTES
# ============================================================

print(
    "Generando análisis de componentes..."
)

componentes = [
    (
        "score_demanda",
        "Demanda"
    ),
    (
        "score_densidad",
        "Densidad"
    ),
    (
        "score_conectividad",
        "Conectividad"
    ),
    (
        "score_intermodalidad",
        "Intermodalidad"
    ),
    (
        "score_alcance",
        "Alcance"
    ),
    (
        "score_integracion",
        "Integración"
    ),
]

componentes_disponibles = [
    item
    for item in componentes
    if item[0] in validacion.columns
]

if componentes_disponibles:

    promedios_componentes = [
        pd.to_numeric(
            validacion[
                columna
            ],
            errors="coerce"
        ).mean()
        for columna, _ in componentes_disponibles
    ]

    nombres_componentes = [
        nombre
        for _, nombre in componentes_disponibles
    ]

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.bar(
        nombres_componentes,
        promedios_componentes
    )

    ax.set_ylabel(
        "Promedio del score"
    )

    ax.set_title(
        "Componentes utilizados en el índice de centralidad"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    guardar_figura(
        OUTPUT_COMPONENTES
    )

else:

    print(
        "ADVERTENCIA: no hay componentes disponibles."
    )


# ============================================================
# GRÁFICO 7 — DISCREPANCIAS
# ============================================================

print(
    "Generando gráfico de discrepancias..."
)

discrepancias_plot = (
    validacion
    .sort_values(
        "diferencia_ranking_abs",
        ascending=False,
        na_position="last"
    )
    .head(20)
    .sort_values(
        "diferencia_ranking"
    )
)

fig, ax = plt.subplots(
    figsize=(13, 9)
)

ax.barh(
    discrepancias_plot[
        "nodo_id"
    ].astype(str),
    discrepancias_plot[
        "diferencia_ranking"
    ]
)

ax.axvline(
    0,
    linewidth=1
)

ax.set_xlabel(
    "Ranking operaciones - ranking centralidad"
)

ax.set_ylabel(
    "Nodo"
)

ax.set_title(
    "Discrepancia entre demanda y centralidad"
)

ax.grid(
    axis="x",
    alpha=0.25
)

guardar_figura(
    OUTPUT_DISCREPANCIAS
)


# ============================================================
# DISTRIBUCIONES
# ============================================================

distribucion_categorias = {
    str(k): int(v)
    for k, v
    in validacion[
        "categoria_centralidad"
    ]
    .value_counts(dropna=False)
    .items()
}

distribucion_tipos = {
    str(k): int(v)
    for k, v
    in validacion[
        "tipo_centralidad"
    ]
    .value_counts(dropna=False)
    .items()
}

distribucion_matriz = {
    str(k): int(v)
    for k, v
    in validacion[
        "matriz_demanda_centralidad"
    ]
    .value_counts(dropna=False)
    .items()
}

distribucion_discrepancias = {
    str(k): int(v)
    for k, v
    in validacion[
        "tipo_discrepancia"
    ]
    .value_counts(dropna=False)
    .items()
}


# ============================================================
# TOP 10 CENTRALIDADES
# ============================================================

top10_centralidad = []

for _, fila in (
    validacion
    .sort_values(
        [
            "indice_centralidad",
            "operaciones",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last"
    )
    .head(10)
    .iterrows()
):

    top10_centralidad.append({

        "nodo_id":
            safe_int(
                fila["nodo_id"]
            ),

        "ranking":
            safe_int(
                fila[
                    "ranking_centralidad_validacion"
                ]
            ),

        "operaciones":
            safe_float(
                fila["operaciones"]
            ),

        "indice_centralidad":
            safe_float(
                fila[
                    "indice_centralidad"
                ]
            ),

        "categoria":
            str(
                fila[
                    "categoria_centralidad"
                ]
            ),

        "tipo":
            str(
                fila[
                    "tipo_centralidad"
                ]
            ),

        "ranking_operaciones":
            safe_int(
                fila[
                    "ranking_operaciones_validacion"
                ]
            ),

        "diferencia_ranking":
            safe_int(
                fila[
                    "diferencia_ranking"
                ]
            ),
    })


# ============================================================
# TOP 10 DEMANDA
# ============================================================

top10_demanda = []

for _, fila in (
    validacion
    .sort_values(
        [
            "operaciones",
            "indice_centralidad",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last"
    )
    .head(10)
    .iterrows()
):

    top10_demanda.append({

        "nodo_id":
            safe_int(
                fila["nodo_id"]
            ),

        "ranking_operaciones":
            safe_int(
                fila[
                    "ranking_operaciones_validacion"
                ]
            ),

        "operaciones":
            safe_float(
                fila["operaciones"]
            ),

        "pct_operaciones":
            safe_float(
                fila["pct_operaciones"]
            ),

        "indice_centralidad":
            safe_float(
                fila[
                    "indice_centralidad"
                ]
            ),

        "ranking_centralidad":
            safe_int(
                fila[
                    "ranking_centralidad_validacion"
                ]
            ),

        "categoria":
            str(
                fila[
                    "categoria_centralidad"
                ]
            ),
    })


# ============================================================
# BAJA DEMANDA / ALTA CENTRALIDAD
# ============================================================

estructurales = (
    validacion[
        validacion[
            "matriz_demanda_centralidad"
        ]
        ==
        "BAJA_DEMANDA_ALTA_CENTRALIDAD"
    ]
    .sort_values(
        [
            "indice_centralidad",
            "operaciones",
        ],
        ascending=[
            False,
            False,
        ]
    )
)

estructurales_lista = []

for _, fila in (
    estructurales
    .head(20)
    .iterrows()
):

    estructurales_lista.append({

        "nodo_id":
            safe_int(
                fila["nodo_id"]
            ),

        "operaciones":
            safe_float(
                fila["operaciones"]
            ),

        "indice_centralidad":
            safe_float(
                fila[
                    "indice_centralidad"
                ]
            ),

        "ranking_operaciones":
            safe_int(
                fila[
                    "ranking_operaciones_validacion"
                ]
            ),

        "ranking_centralidad":
            safe_int(
                fila[
                    "ranking_centralidad_validacion"
                ]
            ),

        "tipo":
            str(
                fila[
                    "tipo_centralidad"
                ]
            ),

        "categoria":
            str(
                fila[
                    "categoria_centralidad"
                ]
            ),
    })


# ============================================================
# ALTA DEMANDA / BAJA CENTRALIDAD
# ============================================================

demanda_alta = (
    validacion[
        validacion[
            "matriz_demanda_centralidad"
        ]
        ==
        "ALTA_DEMANDA_BAJA_CENTRALIDAD"
    ]
    .sort_values(
        [
            "operaciones",
            "indice_centralidad",
        ],
        ascending=[
            False,
            False,
        ]
    )
)

demanda_alta_lista = []

for _, fila in (
    demanda_alta
    .head(20)
    .iterrows()
):

    demanda_alta_lista.append({

        "nodo_id":
            safe_int(
                fila["nodo_id"]
            ),

        "operaciones":
            safe_float(
                fila["operaciones"]
            ),

        "indice_centralidad":
            safe_float(
                fila[
                    "indice_centralidad"
                ]
            ),

        "ranking_operaciones":
            safe_int(
                fila[
                    "ranking_operaciones_validacion"
                ]
            ),

        "ranking_centralidad":
            safe_int(
                fila[
                    "ranking_centralidad_validacion"
                ]
            ),

        "tipo":
            str(
                fila[
                    "tipo_centralidad"
                ]
            ),

        "categoria":
            str(
                fila[
                    "categoria_centralidad"
                ]
            ),
    })


# ============================================================
# RESUMEN EJECUTIVO
# ============================================================

print()
print("=" * 70)
print(
    "RESUMEN DE VALIDACIÓN"
)
print("=" * 70)

print(
    f"\nNodos analizados: "
    f"{cantidad_nodos:,}"
)

print(
    f"Operaciones totales: "
    f"{operaciones_total:,.0f}"
)

principal_id = safe_int(
    principal["nodo_id"]
)

principal_indice = safe_float(
    principal["indice_centralidad"]
)

print(
    f"\nNodo principal por centralidad: "
    f"{principal_id}"
)

print(
    f"Índice principal: "
    f"{principal_indice:.2f}"
)

print(
    f"Nodo principal por demanda: "
    f"{safe_int(nodo_mayor_demanda['nodo_id'])}"
)

print(
    f"Operaciones del nodo de mayor demanda: "
    f"{safe_float(nodo_mayor_demanda['operaciones']) or 0:,.0f}"
)

print(
    f"\nNodo con mayor discrepancia: "
    f"{safe_int(nodo_mayor_discrepancia['nodo_id'])}"
)

print(
    f"Diferencia absoluta: "
    f"{safe_int(nodo_mayor_discrepancia['diferencia_ranking'])}"
)


print(
    "\nCorrelaciones:"
)

for columna, datos in correlaciones.items():

    print(
        f"  {datos['nombre']}: "
        f"Pearson={datos['pearson']}, "
        f"Spearman={datos['spearman']}"
    )


print(
    "\nConcentración:"
)

for clave, valor in concentracion.items():

    print(
        f"  {clave}: "
        f"{valor:.2f}%"
    )


print(
    "\nMatriz demanda-centralidad:"
)

for clave, valor in distribucion_matriz.items():

    print(
        f"  {clave}: "
        f"{valor:,}"
    )


# ============================================================
# INTERPRETACIÓN NODO PRINCIPAL
# ============================================================

print()
print("=" * 70)
print(
    "INTERPRETACIÓN DEL NODO PRINCIPAL"
)
print("=" * 70)

print(
    f"\nNodo {principal_id}:"
)

print(
    f"  Índice de centralidad: "
    f"{principal_indice:.2f}/100"
)

print(
    f"  Ranking centralidad: "
    f"{safe_int(principal['ranking_centralidad_validacion'])}"
)

print(
    f"  Ranking operaciones: "
    f"{safe_int(principal['ranking_operaciones_validacion'])}"
)

print(
    f"  Operaciones: "
    f"{safe_float(principal['operaciones']) or 0:,.0f}"
)

print(
    f"  Diferencia ranking: "
    f"{safe_int(principal['diferencia_ranking'])}"
)

print(
    f"  Matriz: "
    f"{principal['matriz_demanda_centralidad']}"
)

print(
    f"  Tipo: "
    f"{principal['tipo_centralidad']}"
)


# ============================================================
# RESUMEN JSON
# ============================================================

resumen = {

    "fuente": {

        "centralidades":
            str(
                CENTRALIDADES_PATH
            ),

        "h3_centralidades":
            str(
                H3_CENTRALIDADES_PATH
            ),

        "resumen_original":
            str(
                RESUMEN_CENTRALIDADES_PATH
            ),
    },

    "analisis": {

        "nodos":
            int(
                cantidad_nodos
            ),

        "operaciones":
            float(
                operaciones_total
            ),

        "h3":
            int(
                len(
                    h3_centralidades
                )
            ),
    },

    "correlaciones":
        correlaciones,

    "concentracion":
        concentracion,

    "distribucion_categorias":
        distribucion_categorias,

    "distribucion_tipos":
        distribucion_tipos,

    "distribucion_matriz":
        distribucion_matriz,

    "distribucion_discrepancias":
        distribucion_discrepancias,

    "nodo_principal_centralidad": {

        "nodo_id":
            safe_int(
                principal[
                    "nodo_id"
                ]
            ),

        "indice":
            safe_float(
                principal[
                    "indice_centralidad"
                ]
            ),

        "ranking_centralidad":
            safe_int(
                principal[
                    "ranking_centralidad_validacion"
                ]
            ),

        "ranking_operaciones":
            safe_int(
                principal[
                    "ranking_operaciones_validacion"
                ]
            ),

        "operaciones":
            safe_float(
                principal[
                    "operaciones"
                ]
            ),

        "diferencia_ranking":
            safe_int(
                principal[
                    "diferencia_ranking"
                ]
            ),

        "categoria":
            str(
                principal[
                    "categoria_centralidad"
                ]
            ),

        "tipo":
            str(
                principal[
                    "tipo_centralidad"
                ]
            ),
    },

    "nodo_mayor_demanda": {

        "nodo_id":
            safe_int(
                nodo_mayor_demanda[
                    "nodo_id"
                ]
            ),

        "operaciones":
            safe_float(
                nodo_mayor_demanda[
                    "operaciones"
                ]
            ),

        "ranking_operaciones":
            safe_int(
                nodo_mayor_demanda[
                    "ranking_operaciones_validacion"
                ]
            ),

        "ranking_centralidad":
            safe_int(
                nodo_mayor_demanda[
                    "ranking_centralidad_validacion"
                ]
            ),

        "indice_centralidad":
            safe_float(
                nodo_mayor_demanda[
                    "indice_centralidad"
                ]
            ),
    },

    "mayor_discrepancia": {

        "nodo_id":
            safe_int(
                nodo_mayor_discrepancia[
                    "nodo_id"
                ]
            ),

        "diferencia_ranking":
            safe_int(
                nodo_mayor_discrepancia[
                    "diferencia_ranking"
                ]
            ),

        "diferencia_abs":
            safe_int(
                nodo_mayor_discrepancia[
                    "diferencia_ranking_abs"
                ]
            ),

        "ranking_operaciones":
            safe_int(
                nodo_mayor_discrepancia[
                    "ranking_operaciones_validacion"
                ]
            ),

        "ranking_centralidad":
            safe_int(
                nodo_mayor_discrepancia[
                    "ranking_centralidad_validacion"
                ]
            ),
    },

    "nodos_baja_demanda_alta_centralidad":
        estructurales_lista,

    "nodos_alta_demanda_baja_centralidad":
        demanda_alta_lista,

    "top10_centralidad":
        top10_centralidad,

    "top10_demanda":
        top10_demanda,

    "graficos": {

        "mapa_centralidades":
            str(
                OUTPUT_MAPA_CENTRALIDADES
            ),

        "demanda_vs_centralidad":
            str(
                OUTPUT_MAPA_DEMANDA
            ),

        "mapa_h3":
            str(
                OUTPUT_MAPA_H3
            ),

        "ranking":
            str(
                OUTPUT_RANKING
            ),

        "concentracion":
            str(
                OUTPUT_CONCENTRACION
            ),

        "componentes":
            str(
                OUTPUT_COMPONENTES
            ),

        "discrepancias":
            str(
                OUTPUT_DISCREPANCIAS
            ),
    },
}


# ============================================================
# GUARDAR GEOPARQUET
# ============================================================

print(
    "\nGuardando validación..."
)

validacion = gpd.GeoDataFrame(
    validacion,
    geometry="geometry",
    crs=CRS_GEOGRAFICO
)

validacion.to_parquet(
    OUTPUT_VALIDACION,
    index=False
)

print(
    f"Validación guardada:\n"
    f"{OUTPUT_VALIDACION}"
)


# ============================================================
# GUARDAR JSON
# ============================================================

print(
    "Guardando resumen..."
)

with open(
    OUTPUT_RESUMEN,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        resumen,
        archivo,
        ensure_ascii=False,
        indent=2,
        allow_nan=False
    )

print(
    f"Resumen guardado:\n"
    f"{OUTPUT_RESUMEN}"
)


# ============================================================
# ARCHIVOS GENERADOS
# ============================================================

print()
print("=" * 70)
print(
    "ARCHIVOS GENERADOS"
)
print("=" * 70)

print(
    "\nValidación:"
)

print(
    OUTPUT_VALIDACION
)

print(
    "\nResumen:"
)

print(
    OUTPUT_RESUMEN
)

print(
    "\nGráficos:"
)

for archivo in sorted(
    OUTPUT_DIR.glob("*.png")
):

    print(
        f"  {archivo}"
    )


# ============================================================
# FIN
# ============================================================

print()
print("=" * 70)
print(
    "VALIDACIÓN DE CENTRALIDADES FINALIZADA"
)
print("=" * 70)

print(
    f"\nNodos analizados: "
    f"{cantidad_nodos:,}"
)

print(
    f"Operaciones analizadas: "
    f"{operaciones_total:,.0f}"
)

print(
    f"Centralidad principal: "
    f"Nodo {principal_id}"
)

print(
    f"Índice principal: "
    f"{principal_indice:.2f}/100"
)

print(
    "\nSiguiente etapa:"
)

print(
    "Cruzar las centralidades con la red "
    "de transporte y la infraestructura "
    "intermodal para validar la centralidad "
    "estructural."
)