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
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CENTRALIDADES_PATH = (
    BASE_DIR / "data" / "processed"
    / "sube_2025_centralidades_movilidad.parquet"
)

H3_CENTRALIDADES_PATH = (
    BASE_DIR / "data" / "processed"
    / "sube_2025_h3_centralidades.parquet"
)

RESUMEN_CENTRALIDADES_PATH = (
    BASE_DIR / "data" / "processed"
    / "sube_2025_centralidades_resumen.json"
)

OUTPUT_VALIDACION = (
    BASE_DIR / "data" / "processed"
    / "sube_2025_validacion_centralidades.parquet"
)

OUTPUT_RESUMEN = (
    BASE_DIR / "data" / "processed"
    / "sube_2025_validacion_centralidades_resumen.json"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed"
    / "validacion_centralidades"
)

OUTPUT_MAPA_CENTRALIDADES = OUTPUT_DIR / "01_mapa_centralidades.png"
OUTPUT_MAPA_DEMANDA = OUTPUT_DIR / "02_demanda_vs_centralidad.png"
OUTPUT_MAPA_H3 = OUTPUT_DIR / "03_mapa_h3_centralidad.png"
OUTPUT_RANKING = OUTPUT_DIR / "04_ranking_operaciones_vs_centralidad.png"
OUTPUT_CONCENTRACION = OUTPUT_DIR / "05_concentracion_operaciones.png"
OUTPUT_COMPONENTES = OUTPUT_DIR / "06_componentes_indice.png"
OUTPUT_DISCREPANCIAS = OUTPUT_DIR / "07_discrepancias_ranking.png"

CRS_GEOGRAFICO = "EPSG:4326"


# ============================================================
# AUXILIARES
# ============================================================

def validar_archivo(path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido:\n{path}")


def convertir_numerico(df, columnas):
    for columna in columnas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")


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
    return float(valor / total * 100)


def geometria_es_valida(geom):
    """Verifica si un objeto geométrico de Shapely es válido y no nulo."""
    if geom is None:
        return False
    try:
        return not geom.is_empty
    except Exception:
        return False


def correlacion_segura(a, b):
    datos = pd.DataFrame({
        "a": pd.to_numeric(a, errors="coerce"),
        "b": pd.to_numeric(b, errors="coerce"),
    }).dropna()

    if len(datos) < 2:
        return None
    if datos["a"].nunique() <= 1 or datos["b"].nunique() <= 1:
        return None

    try:
        valor = np.corrcoef(
            datos["a"].to_numpy(dtype=float),
            datos["b"].to_numpy(dtype=float),
        )[0, 1]
        return float(valor) if np.isfinite(valor) else None
    except Exception:
        return None


def spearman_segura(a, b):
    datos = pd.DataFrame({
        "a": pd.to_numeric(a, errors="coerce"),
        "b": pd.to_numeric(b, errors="coerce"),
    }).dropna()

    if len(datos) < 2:
        return None
    if datos["a"].nunique() <= 1 or datos["b"].nunique() <= 1:
        return None

    try:
        ra = datos["a"].rank(method="average")
        rb = datos["b"].rank(method="average")
        valor = np.corrcoef(
            ra.to_numpy(dtype=float),
            rb.to_numpy(dtype=float),
        )[0, 1]
        return float(valor) if np.isfinite(valor) else None
    except Exception:
        return None


def guardar_figura(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


# ============================================================
# H3 / GEOMETRÍA
# ============================================================

def convertir_geometry(valor):
    if valor is None:
        return None

    try:
        if pd.isna(valor):
            return None
    except Exception:
        pass

    if hasattr(valor, "geom_type"):
        return valor

    if isinstance(valor, (bytes, bytearray, memoryview)):
        try:
            return wkb.loads(bytes(valor))
        except Exception:
            return None

    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            return None
        try:
            return wkt.loads(texto)
        except Exception:
            return None

    return None


def h3_es_valido(h3_id):
    if h3_id is None:
        return False

    try:
        h3_id = str(h3_id).strip()
        return bool(h3_id) and bool(h3.is_valid_cell(h3_id))
    except Exception:
        return False


def construir_poligono_h3(h3_id):
    try:
        h3_id = str(h3_id).strip()

        if not h3_es_valido(h3_id):
            return None

        boundary = h3.cell_to_boundary(h3_id)

        if not boundary or len(boundary) < 3:
            return None

        coords = [(float(lon), float(lat)) for lat, lon in boundary]
        poligono = Polygon(coords)

        if poligono.is_empty:
            return None

        if not poligono.is_valid:
            poligono = poligono.buffer(0)

        return poligono if geometria_es_valida(poligono) else None

    except Exception:
        return None


# ============================================================
# RECONSTRUCCIÓN GEOMÉTRICA
# ============================================================

def reconstruir_geometria_desde_h3(centralidades, h3_centralidades):
    print()
    print("Reconstruyendo geometrías de nodos desde los H3 reales...")

    for columna in ("nodo_id",):
        if columna not in centralidades.columns:
            raise ValueError(f"centralidades no contiene {columna}.")
        if columna not in h3_centralidades.columns:
            raise ValueError(f"h3_centralidades no contiene {columna}.")

    if "id_h3" not in h3_centralidades.columns:
        raise ValueError("h3_centralidades no contiene id_h3.")

    nodos = centralidades.copy()
    h3_datos = h3_centralidades.copy()

    nodos["_nodo_id"] = pd.to_numeric(nodos["nodo_id"], errors="coerce")
    h3_datos["_nodo_id"] = pd.to_numeric(h3_datos["nodo_id"], errors="coerce")

    sin_nodo = int(h3_datos["_nodo_id"].isna().sum())
    if sin_nodo:
        print(f"ADVERTENCIA: H3 sin nodo_id válido: {sin_nodo:,}")

    nodos = nodos[nodos["_nodo_id"].notna()].copy()
    h3_datos = h3_datos[h3_datos["_nodo_id"].notna()].copy()

    h3_datos["id_h3"] = (
        h3_datos["id_h3"].astype("string").str.strip()
    )
    h3_datos = h3_datos[
        h3_datos["id_h3"].notna()
        & (h3_datos["id_h3"] != "")
    ].copy()

    print(f"H3 disponibles: {len(h3_datos):,}")

    print("Validando identificadores H3...")
    h3_datos["_h3_valido"] = h3_datos["id_h3"].apply(h3_es_valido)

    invalidos = int((~h3_datos["_h3_valido"]).sum())
    print(f"H3 inválidos: {invalidos:,}")

    h3_datos = h3_datos[h3_datos["_h3_valido"]].copy()

    if h3_datos.empty:
        raise ValueError("No quedaron H3 válidos.")

    duplicados = int(h3_datos["id_h3"].duplicated().sum())
    print(f"H3 duplicados: {duplicados:,}")

    if duplicados:
        print(
            h3_datos.loc[
                h3_datos["id_h3"].duplicated(keep=False),
                ["id_h3", "nodo_id"],
            ].head(20).to_string(index=False)
        )
        raise ValueError("Se encontraron H3 duplicados.")

    nodos_totales = int(nodos["_nodo_id"].nunique())
    nodos_con_h3 = int(h3_datos["_nodo_id"].nunique())

    print()
    print("COBERTURA H3 → NODOS")
    print(f"  Nodos totales: {nodos_totales:,}")
    print(f"  Nodos con al menos un H3: {nodos_con_h3:,}")
    print(f"  Nodos sin H3: {nodos_totales - nodos_con_h3:,}")
    print(f"  H3 utilizados: {len(h3_datos):,}")

    print()
    print("Construyendo polígonos H3...")

    h3_datos["geometry"] = h3_datos["id_h3"].apply(construir_poligono_h3)

    # Renombrado a mascara_geometrias_h3 para no eclipsar la función 'geometria_es_valida'
    mascara_geometrias_h3 = h3_datos["geometry"].apply(geometria_es_valida)

    invalidas = int((~mascara_geometrias_h3).sum())
    print(f"Geometrías H3 inválidas: {invalidas:,}")

    h3_datos = h3_datos[mascara_geometrias_h3].copy()

    if h3_datos.empty:
        raise ValueError("No fue posible construir ninguna geometría H3.")

    print(f"Polígonos H3 construidos: {len(h3_datos):,}")

    h3_geo = gpd.GeoDataFrame(
        h3_datos,
        geometry="geometry",
        crs=CRS_GEOGRAFICO,
    )

    print(f"Nodos con H3 geométrico: {h3_geo['_nodo_id'].nunique():,}")

    cantidad_h3_por_nodo = (
        h3_geo.groupby("_nodo_id")
        .size()
        .rename("cantidad_h3_calculada")
    )

    print()
    print("Uniendo geometrías H3 por nodo...")

    geometrias_nodos = []
    errores_union = 0

    for nodo_id, grupo in h3_geo.groupby("_nodo_id"):
        try:
            geometria = unary_union(grupo.geometry.tolist())

            if not geometria_es_valida(geometria):
                continue

            if not geometria.is_valid:
                geometria = geometria.buffer(0)

            if geometria_es_valida(geometria):
                geometrias_nodos.append({
                    "_nodo_id": nodo_id,
                    "geometry": geometria,
                })

        except Exception as error:
            errores_union += 1
            print(
                f"ADVERTENCIA: no se pudo unir nodo "
                f"{nodo_id}: {error}"
            )

    print(f"Errores de unión geométrica: {errores_union:,}")

    if not geometrias_nodos:
        raise ValueError("No fue posible construir geometrías de nodos.")

    geometria_nodos = gpd.GeoDataFrame(
        geometrias_nodos,
        geometry="geometry",
        crs=CRS_GEOGRAFICO,
    )

    print(
        f"Geometrías de nodos reconstruidas: "
        f"{len(geometria_nodos):,}"
    )

    resultado = nodos.merge(
        geometria_nodos[["_nodo_id", "geometry"]],
        on="_nodo_id",
        how="left",
        validate="one_to_one",
    )

    resultado = resultado.drop(columns=["_nodo_id"], errors="ignore")

    resultado = gpd.GeoDataFrame(
        resultado,
        geometry="geometry",
        crs=CRS_GEOGRAFICO,
    )

    mascara_geometrias_finales = (
        resultado["geometry"].apply(geometria_es_valida)
    )

    nodos_con_geometria = int(mascara_geometrias_finales.sum())
    nodos_sin_geometria = int((~mascara_geometrias_finales).sum())

    print()
    print("VALIDACIÓN GEOMÉTRICA FINAL")
    print(f"  Nodos totales: {len(resultado):,}")
    print(f"  Nodos con geometría: {nodos_con_geometria:,}")
    print(f"  Nodos sin geometría: {nodos_sin_geometria:,}")

    if nodos_con_geometria == 0:
        raise ValueError("No se reconstruyó ninguna geometría de nodo.")

    if "h3" in resultado.columns:
        resultado["_cantidad_h3_calculada"] = (
            resultado["nodo_id"]
            .map(cantidad_h3_por_nodo)
            .fillna(0)
        )

        resultado["_h3_original"] = pd.to_numeric(
            resultado["h3"], errors="coerce"
        )

        resultado["_diferencia_h3"] = (
            resultado["_h3_original"]
            - resultado["_cantidad_h3_calculada"]
        )

        comparables = resultado["_h3_original"].notna()
        coincidencias = int(
            (
                resultado.loc[comparables, "_diferencia_h3"] == 0
            ).sum()
        )
        discrepancias = int(
            (
                resultado.loc[comparables, "_diferencia_h3"] != 0
            ).sum()
        )

        print()
        print("VALIDACIÓN CANTIDAD H3 POR NODO")
        print(f"  Nodos comparables: {int(comparables.sum()):,}")
        print(f"  Coincidencias: {coincidencias:,}")
        print(f"  Discrepancias: {discrepancias:,}")

        if discrepancias:
            print("\nPrimeras discrepancias:")
            print(
                resultado.loc[
                    comparables & (resultado["_diferencia_h3"] != 0),
                    [
                        "nodo_id",
                        "_h3_original",
                        "_cantidad_h3_calculada",
                        "_diferencia_h3",
                    ],
                ].head(20).to_string(index=False)
            )

        resultado = resultado.drop(
            columns=[
                "_cantidad_h3_calculada",
                "_h3_original",
                "_diferencia_h3",
            ],
            errors="ignore",
        )

    return resultado


# ============================================================
# CARGAR CENTRALIDADES
# ============================================================

def cargar_centralidades(path, h3_centralidades):
    print("Intentando cargar como GeoParquet...")

    try:
        datos = gpd.read_parquet(path)

        print("Archivo leído como GeoParquet.")

        if "geometry" in datos.columns and datos.geometry.notna().any():
            if datos.crs is None:
                datos = datos.set_crs(CRS_GEOGRAFICO)

            mascara = datos["geometry"].apply(geometria_es_valida)

            if mascara.any():
                print("Geometría GeoParquet válida encontrada.")
                return datos

    except Exception as error:
        print("No se pudo utilizar directamente como GeoParquet.")
        print(f"Motivo: {error}")

    print("Cargando centralidades mediante pandas...")

    datos = pd.read_parquet(path)

    print(f"Registros cargados: {len(datos):,}")

    return reconstruir_geometria_desde_h3(datos, h3_centralidades)


# ============================================================
# CUARTILES / CLASIFICACIONES
# ============================================================

def calcular_cuartiles(serie):
    datos = pd.to_numeric(serie, errors="coerce")
    ranking = datos.rank(method="first")

    resultado = pd.Series(
        pd.NA,
        index=serie.index,
        dtype="string",
    )

    validos = ranking.notna()
    cantidad = int(validos.sum())

    if cantidad == 0:
        return resultado

    if cantidad < 4:
        resultado.loc[validos] = "Q4_ALTO"
        return resultado

    try:
        resultado.loc[validos] = pd.qcut(
            ranking.loc[validos],
            4,
            labels=[
                "Q1_BAJO",
                "Q2_MEDIO_BAJO",
                "Q3_MEDIO_ALTO",
                "Q4_ALTO",
            ],
        ).astype("string")
    except Exception:
        percentiles = ranking.loc[validos].rank(pct=True)
        resultado.loc[validos] = np.select(
            [
                percentiles <= 0.25,
                percentiles <= 0.50,
                percentiles <= 0.75,
            ],
            [
                "Q1_BAJO",
                "Q2_MEDIO_BAJO",
                "Q3_MEDIO_ALTO",
            ],
            default="Q4_ALTO",
        )

    return resultado


def clasificar_discrepancia(row):
    diferencia = row["diferencia_ranking"]

    if diferencia >= 20:
        return "CENTRALIDAD_MUCHO_MAYOR_QUE_DEMANDA"
    if diferencia >= 10:
        return "CENTRALIDAD_MAYOR_QUE_DEMANDA"
    if diferencia <= -20:
        return "DEMANDA_MUCHO_MAYOR_QUE_CENTRALIDAD"
    if diferencia <= -10:
        return "DEMANDA_MAYOR_QUE_CENTRALIDAD"
    return "ALINEADO"


def clasificar_matriz(row):
    demanda = row["cuartil_operaciones"]
    centralidad = row["cuartil_centralidad"]

    demanda_alta = demanda in ["Q3_MEDIO_ALTO", "Q4_ALTO"]
    centralidad_alta = centralidad in ["Q3_MEDIO_ALTO", "Q4_ALTO"]

    if demanda_alta and centralidad_alta:
        return "ALTA_DEMANDA_ALTA_CENTRALIDAD"
    if demanda_alta and not centralidad_alta:
        return "ALTA_DEMANDA_BAJA_CENTRALIDAD"
    if not demanda_alta and centralidad_alta:
        return "BAJA_DEMANDA_ALTA_CENTRALIDAD"
    return "BAJA_DEMANDA_BAJA_CENTRALIDAD"


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("VALIDACIÓN DE CENTRALIDADES DE MOVILIDAD SUBE 2025")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nValidando archivos de entrada...")

for archivo in (
    CENTRALIDADES_PATH,
    H3_CENTRALIDADES_PATH,
    RESUMEN_CENTRALIDADES_PATH,
):
    validar_archivo(archivo)

print("Archivos de entrada encontrados correctamente.")


# ============================================================
# CARGAR H3
# ============================================================

print("\nCargando H3 → centralidad...")

h3_centralidades = pd.read_parquet(H3_CENTRALIDADES_PATH)

print(f"H3 cargados: {len(h3_centralidades):,}")
print("Columnas:")
print(h3_centralidades.columns.tolist())

columnas_h3_necesarias = [
    "id_h3",
    "nodo_id",
    "indice_centralidad",
    "categoria_centralidad",
]

faltantes_h3 = [
    c for c in columnas_h3_necesarias
    if c not in h3_centralidades.columns
]

if faltantes_h3:
    raise ValueError(
        "Faltan columnas en "
        "sube_2025_h3_centralidades.parquet:\n"
        + "\n".join(f" - {c}" for c in faltantes_h3)
    )


# ============================================================
# CARGAR CENTRALIDADES
# ============================================================

print("\nCargando centralidades...")

centralidades = cargar_centralidades(
    CENTRALIDADES_PATH,
    h3_centralidades,
)

print(f"Centralidades cargadas: {len(centralidades):,}")
print("Columnas:")
print(centralidades.columns.tolist())


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
]

faltantes = [
    c for c in columnas_obligatorias
    if c not in centralidades.columns
]

if faltantes:
    raise ValueError(
        "Faltan columnas obligatorias:\n"
        + "\n".join(f" - {c}" for c in faltantes)
    )


# ============================================================
# GEOMETRÍA
# ============================================================

print("\nValidando geometrías...")

if "geometry" not in centralidades.columns:
    raise ValueError("No existe columna geometry.")

if centralidades.crs is None:
    centralidades = centralidades.set_crs(CRS_GEOGRAFICO)

print(f"CRS: {centralidades.crs}")

mascara_geometrias = centralidades["geometry"].apply(geometria_es_valida)
cantidad_geometrias_validas = int(mascara_geometrias.sum())

print(f"Geometrías válidas: {cantidad_geometrias_validas:,}")

if cantidad_geometrias_validas == 0:
    raise ValueError("No existen geometrías válidas.")


# ============================================================
# CONVERSIONES
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

convertir_numerico(centralidades, columnas_numericas)


# ============================================================
# RESUMEN ORIGINAL
# ============================================================

print("\nCargando resumen original...")

with open(
    RESUMEN_CENTRALIDADES_PATH,
    "r",
    encoding="utf-8",
) as archivo:
    resumen_original = json.load(archivo)


# ============================================================
# IDENTIFICADORES
# ============================================================

print("\nValidando identificadores...")

duplicados_nodo = int(centralidades["nodo_id"].duplicated().sum())
print(f"Nodos duplicados: {duplicados_nodo:,}")

if duplicados_nodo:
    raise ValueError("Existen nodos duplicados.")

h3_ids = h3_centralidades["id_h3"].astype("string").str.strip()
duplicados_h3 = int(h3_ids.duplicated().sum())

print(f"H3 duplicados: {duplicados_h3:,}")

if duplicados_h3:
    raise ValueError("Existen H3 duplicados.")


# ============================================================
# COPIA
# ============================================================

validacion = centralidades.copy()

cantidad_h3_nodo = (
    h3_centralidades.groupby("nodo_id")
    .size()
    .rename("_cantidad_h3_calculada")
)

if "h3" not in validacion.columns:
    validacion["h3"] = (
        validacion["nodo_id"]
        .map(cantidad_h3_nodo)
        .fillna(0)
    )

if "operaciones_por_km2" not in validacion.columns:
    if "superficie_km2" in validacion.columns:
        superficie = pd.to_numeric(
            validacion["superficie_km2"],
            errors="coerce",
        )
        validacion["operaciones_por_km2"] = np.where(
            superficie > 0,
            validacion["operaciones"] / superficie,
            np.nan,
        )


# ============================================================
# RANKING OPERACIONES
# ============================================================

print("\nCalculando ranking por operaciones...")

ranking_operaciones = (
    validacion.sort_values(
        ["operaciones", "indice_centralidad", "nodo_id"],
        ascending=[False, False, True],
    )
    .reset_index(drop=True)
)

ranking_operaciones["ranking_operaciones_validacion"] = (
    np.arange(1, len(ranking_operaciones) + 1)
)


# ============================================================
# RANKING CENTRALIDAD
# ============================================================

print("Calculando ranking por centralidad...")

ranking_centralidad = (
    validacion.sort_values(
        ["indice_centralidad", "operaciones", "nodo_id"],
        ascending=[False, False, True],
    )
    .reset_index(drop=True)
)

ranking_centralidad["ranking_centralidad_validacion"] = (
    np.arange(1, len(ranking_centralidad) + 1)
)


# ============================================================
# UNIFICAR
# ============================================================

validacion = (
    validacion
    .merge(
        ranking_operaciones[
            ["nodo_id", "ranking_operaciones_validacion"]
        ],
        on="nodo_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        ranking_centralidad[
            ["nodo_id", "ranking_centralidad_validacion"]
        ],
        on="nodo_id",
        how="left",
        validate="one_to_one",
    )
)


# ============================================================
# PCT OPERACIONES
# ============================================================

operaciones_total = (
    pd.to_numeric(validacion["operaciones"], errors="coerce")
    .fillna(0)
    .sum()
)

if operaciones_total > 0:
    validacion["pct_operaciones"] = (
        validacion["operaciones"].fillna(0)
        / operaciones_total
        * 100
    )
else:
    validacion["pct_operaciones"] = 0.0

orden_demanda = (
    validacion.sort_values(
        ["operaciones", "nodo_id"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

orden_demanda["pct_operaciones_validacion"] = (
    orden_demanda["operaciones"].fillna(0)
    / operaciones_total
    * 100
    if operaciones_total > 0
    else 0
)

orden_demanda["pct_acumulado_validacion"] = (
    orden_demanda["pct_operaciones_validacion"].cumsum()
)

acumulado_map = orden_demanda.set_index(
    "nodo_id"
)["pct_acumulado_validacion"]

validacion["pct_operaciones_acumulado"] = (
    validacion["nodo_id"].map(acumulado_map)
)


# ============================================================
# DIFERENCIAS
# ============================================================

validacion["diferencia_ranking"] = (
    validacion["ranking_operaciones_validacion"]
    - validacion["ranking_centralidad_validacion"]
)

validacion["diferencia_ranking_abs"] = (
    validacion["diferencia_ranking"].abs()
)

cantidad_nodos = len(validacion)

if cantidad_nodos:
    validacion["percentil_operaciones"] = (
        100
        * (cantidad_nodos - validacion["ranking_operaciones_validacion"] + 1)
        / cantidad_nodos
    )

    validacion["percentil_centralidad"] = (
        100
        * (cantidad_nodos - validacion["ranking_centralidad_validacion"] + 1)
        / cantidad_nodos
    )

validacion["tipo_discrepancia"] = validacion.apply(
    clasificar_discrepancia,
    axis=1,
)

validacion["cuartil_operaciones"] = calcular_cuartiles(
    validacion["operaciones"]
)

validacion["cuartil_centralidad"] = calcular_cuartiles(
    validacion["indice_centralidad"]
)

validacion["matriz_demanda_centralidad"] = validacion.apply(
    clasificar_matriz,
    axis=1,
)


# ============================================================
# CORRELACIONES
# ============================================================

print("\nCalculando correlaciones...")

correlaciones = {}

variables_correlacion = [
    ("operaciones", "Operaciones"),
    ("operaciones_por_km2", "Operaciones por km²"),
    ("cantidad_corredores", "Cantidad de corredores"),
    ("cantidad_clusters", "Cantidad de clusters"),
    ("cantidad_jurisdicciones", "Cantidad de jurisdicciones"),
    ("score_intermodalidad", "Score de intermodalidad"),
    ("score_demanda", "Score de demanda"),
    ("score_densidad", "Score de densidad"),
    ("score_conectividad", "Score de conectividad"),
    ("score_alcance", "Score de alcance"),
    ("score_integracion", "Score de integración"),
]

for columna, nombre in variables_correlacion:
    if columna not in validacion.columns:
        continue

    correlaciones[columna] = {
        "nombre": nombre,
        "pearson": correlacion_segura(
            validacion[columna],
            validacion["indice_centralidad"],
        ),
        "spearman": spearman_segura(
            validacion[columna],
            validacion["indice_centralidad"],
        ),
    }


# ============================================================
# CONCENTRACIÓN
# ============================================================

print("\nAnalizando concentración...")

def concentracion_top(n):
    if operaciones_total == 0:
        return 0.0

    return porcentaje(
        orden_demanda["operaciones"].fillna(0).head(n).sum(),
        operaciones_total,
    )


concentracion = {
    "top_1": concentracion_top(1),
    "top_5": concentracion_top(5),
    "top_10": concentracion_top(10),
    "top_20": concentracion_top(20),
    "top_30": concentracion_top(30),
    "top_50": concentracion_top(50),
}


# ============================================================
# NODOS PRINCIPALES
# ============================================================

if validacion.empty:
    raise ValueError("No existen nodos para analizar.")

principal = (
    validacion.sort_values(
        ["indice_centralidad", "operaciones", "nodo_id"],
        ascending=[False, False, True],
    ).iloc[0]
)

nodo_mayor_demanda = (
    validacion.sort_values(
        ["operaciones", "indice_centralidad", "nodo_id"],
        ascending=[False, False, True],
    ).iloc[0]
)

nodo_mayor_discrepancia = (
    validacion.sort_values(
        ["diferencia_ranking_abs", "nodo_id"],
        ascending=[False, True],
    ).iloc[0]
)


# ============================================================
# NODOS 1 Y 6
# ============================================================

print()
print("=" * 70)
print("VALIDACIÓN DE NODOS DESTACADOS")
print("=" * 70)

for nodo_id in [1, 6]:
    datos = validacion[validacion["nodo_id"] == nodo_id]

    if datos.empty:
        print(f"\nNodo {nodo_id}: no encontrado.")
        continue

    fila = datos.iloc[0]

    print(f"\nNodo {nodo_id}")
    print(
        f"  Ranking operaciones: "
        f"{int(fila['ranking_operaciones_validacion'])}"
    )
    print(
        f"  Ranking centralidad: "
        f"{int(fila['ranking_centralidad_validacion'])}"
    )
    print(f"  Operaciones: {fila['operaciones']:,.0f}")
    print(f"  Índice: {fila['indice_centralidad']:.2f}")
    print(f"  Diferencia ranking: {int(fila['diferencia_ranking'])}")
    print(f"  Tipo centralidad: {fila['tipo_centralidad']}")
    print(f"  Matriz: {fila['matriz_demanda_centralidad']}")


# ============================================================
# TOP DISCREPANCIAS
# ============================================================

print()
print("=" * 70)
print("MAYORES DISCREPANCIAS ENTRE DEMANDA Y CENTRALIDAD")
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
    validacion.sort_values(
        "diferencia_ranking_abs",
        ascending=False,
    )[columnas_discrepancias]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# MAPA 1
# ============================================================

print("\nGenerando mapa de centralidades...")

mapa = validacion[
    validacion["geometry"].apply(geometria_es_valida)
].copy()

fig, ax = plt.subplots(figsize=(14, 12))

categorias = [
    "CENTRALIDAD_BAJA",
    "CENTRALIDAD_MEDIA",
    "CENTRALIDAD_ALTA",
    "CENTRALIDAD_CRITICA",
]

for categoria in categorias:
    grupo = mapa[mapa["categoria_centralidad"] == categoria]

    if grupo.empty:
        continue

    grupo.plot(
        ax=ax,
        alpha=0.60,
        label=categoria,
        edgecolor="none",
    )

top15 = (
    mapa.sort_values(
        ["indice_centralidad", "operaciones"],
        ascending=[False, False],
    ).head(15)
)

for _, fila in top15.iterrows():
    try:
        punto = fila.geometry.representative_point()
        ax.annotate(
            str(int(fila["nodo_id"])),
            (punto.x, punto.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    except Exception:
        pass

ax.set_title("Centralidades de movilidad SUBE 2025", fontsize=15)
ax.set_axis_off()

if ax.get_legend_handles_labels()[0]:
    ax.legend(title="Categoría", loc="best")

guardar_figura(OUTPUT_MAPA_CENTRALIDADES)


# ============================================================
# GRÁFICO 2
# ============================================================

print("Generando gráfico demanda vs centralidad...")

fig, ax = plt.subplots(figsize=(12, 9))

for categoria in categorias:
    grupo = validacion[
        validacion["categoria_centralidad"] == categoria
    ].copy()

    grupo = grupo[grupo["operaciones"] > 0]

    if grupo.empty:
        continue

    if "cantidad_corredores" in grupo.columns:
        tamanio = (
            pd.to_numeric(
                grupo["cantidad_corredores"],
                errors="coerce",
            ).fillna(0) + 1
        ) * 20
    else:
        tamanio = 40

    ax.scatter(
        grupo["operaciones"],
        grupo["indice_centralidad"],
        s=tamanio,
        alpha=0.70,
        label=categoria,
    )

ax.set_xscale("log")
ax.set_xlabel("Operaciones (escala logarítmica)")
ax.set_ylabel("Índice de centralidad")
ax.set_title("Demanda vs índice de centralidad")
ax.grid(alpha=0.25)

if ax.get_legend_handles_labels()[0]:
    ax.legend()

for _, fila in (
    validacion.sort_values(
        "indice_centralidad",
        ascending=False,
    ).head(10).iterrows()
):
    if pd.notna(fila["operaciones"]) and fila["operaciones"] > 0:
        ax.annotate(
            str(int(fila["nodo_id"])),
            (fila["operaciones"], fila["indice_centralidad"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

guardar_figura(OUTPUT_MAPA_DEMANDA)


# ============================================================
# MAPA 3 — H3
# ============================================================

print("Generando mapa H3...")

try:
    h3_mapa = h3_centralidades.copy()

    h3_mapa["geometry"] = (
        h3_mapa["id_h3"]
        .astype(str)
        .apply(construir_poligono_h3)
    )

    h3_mapa = gpd.GeoDataFrame(
        h3_mapa,
        geometry="geometry",
        crs=CRS_GEOGRAFICO,
    )

    mascara_h3_mapa = h3_mapa["geometry"].apply(geometria_es_valida)
    h3_mapa = h3_mapa[mascara_h3_mapa].copy()

    if not h3_mapa.empty:
        fig, ax = plt.subplots(figsize=(15, 12))

        h3_mapa.plot(
            ax=ax,
            column="indice_centralidad",
            legend=True,
            alpha=0.70,
            edgecolor="none",
        )

        ax.set_title(
            "Distribución H3 según índice de centralidad",
            fontsize=15,
        )
        ax.set_axis_off()

        guardar_figura(OUTPUT_MAPA_H3)
    else:
        print("ADVERTENCIA: no existen geometrías H3.")

except Exception as error:
    print("ADVERTENCIA: no se pudo generar el mapa H3.")
    print(f"Motivo: {error}")


# ============================================================
# GRÁFICO 4
# ============================================================

print("Generando comparación de rankings...")

ranking_plot = (
    validacion.sort_values(
        "ranking_centralidad_validacion"
    ).head(30)
)

fig, ax = plt.subplots(figsize=(14, 9))

ax.scatter(
    ranking_plot["ranking_operaciones_validacion"],
    ranking_plot["ranking_centralidad_validacion"],
    alpha=0.75,
)

max_ranking = max(cantidad_nodos, 1)

ax.plot(
    [1, max_ranking],
    [1, max_ranking],
    linestyle="--",
    linewidth=1,
)

ax.set_xlabel("Ranking por operaciones")
ax.set_ylabel("Ranking por centralidad")
ax.set_title("Ranking por demanda vs ranking por centralidad")
ax.grid(alpha=0.25)

for _, fila in ranking_plot.iterrows():
    ax.annotate(
        str(int(fila["nodo_id"])),
        (
            fila["ranking_operaciones_validacion"],
            fila["ranking_centralidad_validacion"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
    )

guardar_figura(OUTPUT_RANKING)


# ============================================================
# GRÁFICO 5
# ============================================================

print("Generando curva de concentración...")

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(1, len(orden_demanda) + 1)
y = orden_demanda["pct_acumulado_validacion"]

ax.plot(x, y, linewidth=2)

for n in [1, 5, 10, 20, 30, 50]:
    if n <= len(orden_demanda):
        valor = orden_demanda["pct_acumulado_validacion"].iloc[n - 1]

        ax.scatter([n], [valor])
        ax.annotate(
            f"Top {n}: {valor:.1f}%",
            (n, valor),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

ax.set_xlabel("Cantidad acumulada de nodos")
ax.set_ylabel("% acumulado de operaciones")
ax.set_title("Concentración de operaciones por nodo")
ax.grid(alpha=0.25)

guardar_figura(OUTPUT_CONCENTRACION)


# ============================================================
# GRÁFICO 6
# ============================================================

print("Generando análisis de componentes...")

componentes = [
    ("score_demanda", "Demanda"),
    ("score_densidad", "Densidad"),
    ("score_conectividad", "Conectividad"),
    ("score_intermodalidad", "Intermodalidad"),
    ("score_alcance", "Alcance"),
    ("score_integracion", "Integración"),
]

componentes_disponibles = [
    item for item in componentes if item[0] in validacion.columns
]

if componentes_disponibles:
    promedios = [
        pd.to_numeric(
            validacion[columna],
            errors="coerce",
        ).mean()
        for columna, _ in componentes_disponibles
    ]

    nombres = [
        nombre for _, nombre in componentes_disponibles
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.bar(nombres, promedios)
    ax.set_ylabel("Promedio del score")
    ax.set_title(
        "Componentes utilizados en el índice de centralidad"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    guardar_figura(OUTPUT_COMPONENTES)
else:
    print("ADVERTENCIA: no hay componentes disponibles.")


# ============================================================
# GRÁFICO 7
# ============================================================

print("Generando gráfico de discrepancias...")

discrepancias_plot = (
    validacion.sort_values(
        "diferencia_ranking_abs",
        ascending=False,
    )
    .head(20)
    .sort_values("diferencia_ranking")
)

fig, ax = plt.subplots(figsize=(13, 9))

ax.barh(
    discrepancias_plot["nodo_id"].astype(str),
    discrepancias_plot["diferencia_ranking"],
)

ax.axvline(0, linewidth=1)
ax.set_xlabel(
    "Ranking operaciones - ranking centralidad"
)
ax.set_ylabel("Nodo")
ax.set_title("Discrepancia entre demanda y centralidad")
ax.grid(axis="x", alpha=0.25)

guardar_figura(OUTPUT_DISCREPANCIAS)


# ============================================================
# DISTRIBUCIONES
# ============================================================

def distribucion_columna(df, columna):
    return {
        str(k): int(v)
        for k, v in df[columna].value_counts().items()
    }


distribucion_categorias = distribucion_columna(
    validacion, "categoria_centralidad"
)

distribucion_tipos = distribucion_columna(
    validacion, "tipo_centralidad"
)

distribucion_matriz = distribucion_columna(
    validacion, "matriz_demanda_centralidad"
)

distribucion_discrepancias = distribucion_columna(
    validacion, "tipo_discrepancia"
)


# ============================================================
# LISTAS JSON
# ============================================================

def fila_top_centralidad(fila):
    return {
        "nodo_id": safe_int(fila["nodo_id"]),
        "ranking": safe_int(
            fila["ranking_centralidad_validacion"]
        ),
        "operaciones": safe_float(fila["operaciones"]),
        "indice_centralidad": safe_float(
            fila["indice_centralidad"]
        ),
        "categoria": str(fila["categoria_centralidad"]),
        "tipo": str(fila["tipo_centralidad"]),
        "ranking_operaciones": safe_int(
            fila["ranking_operaciones_validacion"]
        ),
        "diferencia_ranking": safe_int(
            fila["diferencia_ranking"]
        ),
    }


top10_centralidad = [
    fila_top_centralidad(fila)
    for _, fila in (
        validacion.sort_values(
            ["indice_centralidad", "operaciones"],
            ascending=[False, False],
        ).head(10).iterrows()
    )
]


def fila_top_demanda(fila):
    return {
        "nodo_id": safe_int(fila["nodo_id"]),
        "ranking_operaciones": safe_int(
            fila["ranking_operaciones_validacion"]
        ),
        "operaciones": safe_float(fila["operaciones"]),
        "pct_operaciones": safe_float(fila["pct_operaciones"]),
        "indice_centralidad": safe_float(
            fila["indice_centralidad"]
        ),
        "ranking_centralidad": safe_int(
            fila["ranking_centralidad_validacion"]
        ),
        "categoria": str(fila["categoria_centralidad"]),
    }


top10_demanda = [
    fila_top_demanda(fila)
    for _, fila in (
        validacion.sort_values(
            ["operaciones", "indice_centralidad"],
            ascending=[False, False],
        ).head(10).iterrows()
    )
]


estructurales = (
    validacion[
        validacion["matriz_demanda_centralidad"]
        == "BAJA_DEMANDA_ALTA_CENTRALIDAD"
    ]
    .sort_values(
        ["indice_centralidad", "operaciones"],
        ascending=[False, False],
    )
)

estructurales_lista = []

for _, fila in estructurales.head(20).iterrows():
    estructurales_lista.append({
        "nodo_id": safe_int(fila["nodo_id"]),
        "operaciones": safe_float(fila["operaciones"]),
        "indice_centralidad": safe_float(
            fila["indice_centralidad"]
        ),
        "ranking_operaciones": safe_int(
            fila["ranking_operaciones_validacion"]
        ),
        "ranking_centralidad": safe_int(
            fila["ranking_centralidad_validacion"]
        ),
        "tipo": str(fila["tipo_centralidad"]),
        "categoria": str(fila["categoria_centralidad"]),
    })


demanda_alta = (
    validacion[
        validacion["matriz_demanda_centralidad"]
        == "ALTA_DEMANDA_BAJA_CENTRALIDAD"
    ]
    .sort_values(
        ["operaciones", "indice_centralidad"],
        ascending=[False, False],
    )
)

demanda_alta_lista = []

for _, fila in demanda_alta.head(20).iterrows():
    demanda_alta_lista.append({
        "nodo_id": safe_int(fila["nodo_id"]),
        "operaciones": safe_float(fila["operaciones"]),
        "indice_centralidad": safe_float(
            fila["indice_centralidad"]
        ),
        "ranking_operaciones": safe_int(
            fila["ranking_operaciones_validacion"]
        ),
        "ranking_centralidad": safe_int(
            fila["ranking_centralidad_validacion"]
        ),
        "tipo": str(fila["tipo_centralidad"]),
        "categoria": str(fila["categoria_centralidad"]),
    })


# ============================================================
# RESUMEN
# ============================================================

print()
print("=" * 70)
print("RESUMEN DE VALIDACIÓN")
print("=" * 70)

print(f"\nNodos analizados: {cantidad_nodos:,}")
print(f"Operaciones totales: {operaciones_total:,.0f}")
print(
    f"\nNodo principal por centralidad: "
    f"{int(principal['nodo_id'])}"
)
print(
    f"Índice principal: "
    f"{principal['indice_centralidad']:.2f}"
)
print(
    f"Nodo principal por demanda: "
    f"{int(nodo_mayor_demanda['nodo_id'])}"
)
print(
    f"Operaciones del nodo de mayor demanda: "
    f"{nodo_mayor_demanda['operaciones']:,.0f}"
)
print(
    f"\nNodo con mayor discrepancia: "
    f"{int(nodo_mayor_discrepancia['nodo_id'])}"
)
print(
    f"Diferencia absoluta: "
    f"{int(nodo_mayor_discrepancia['diferencia_ranking_abs'])}"
)

print("\nCorrelaciones:")

for _, datos in correlaciones.items():
    print(
        f"  {datos['nombre']}: "
        f"Pearson={datos['pearson']}, "
        f"Spearman={datos['spearman']}"
    )

print("\nConcentración:")

for clave, valor in concentracion.items():
    print(f"  {clave}: {valor:.2f}%")

print("\nMatriz demanda-centralidad:")

for clave, valor in distribucion_matriz.items():
    print(f"  {clave}: {valor:,}")


# ============================================================
# INTERPRETACIÓN NODO PRINCIPAL
# ============================================================

print()
print("=" * 70)
print("INTERPRETACIÓN DEL NODO PRINCIPAL")
print("=" * 70)

print(f"\nNodo {int(principal['nodo_id'])}:")
print(
    f"  Índice de centralidad: "
    f"{principal['indice_centralidad']:.2f}/100"
)
print(
    f"  Ranking centralidad: "
    f"{int(principal['ranking_centralidad_validacion'])}"
)
print(
    f"  Ranking operaciones: "
    f"{int(principal['ranking_operaciones_validacion'])}"
)
print(f"  Operaciones: {principal['operaciones']:,.0f}")
print(
    f"  Diferencia ranking: "
    f"{int(principal['diferencia_ranking'])}"
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
        "centralidades": str(CENTRALIDADES_PATH),
        "h3_centralidades": str(H3_CENTRALIDADES_PATH),
        "resumen_original": str(RESUMEN_CENTRALIDADES_PATH),
    },
    "analisis": {
        "nodos": int(cantidad_nodos),
        "operaciones": float(operaciones_total),
        "h3": int(len(h3_centralidades)),
    },
    "correlaciones": correlaciones,
    "concentracion": concentracion,
    "distribucion_categorias": distribucion_categorias,
    "distribucion_tipos": distribucion_tipos,
    "distribucion_matriz": distribucion_matriz,
    "distribucion_discrepancias": distribucion_discrepancias,
    "nodo_principal_centralidad": {
        "nodo_id": safe_int(principal["nodo_id"]),
        "indice": safe_float(principal["indice_centralidad"]),
        "ranking_centralidad": safe_int(
            principal["ranking_centralidad_validacion"]
        ),
        "ranking_operaciones": safe_int(
            principal["ranking_operaciones_validacion"]
        ),
        "operaciones": safe_float(principal["operaciones"]),
        "diferencia_ranking": safe_int(
            principal["diferencia_ranking"]
        ),
        "categoria": str(principal["categoria_centralidad"]),
        "tipo": str(principal["tipo_centralidad"]),
    },
    "nodo_mayor_demanda": {
        "nodo_id": safe_int(nodo_mayor_demanda["nodo_id"]),
        "operaciones": safe_float(nodo_mayor_demanda["operaciones"]),
        "ranking_operaciones": safe_int(
            nodo_mayor_demanda["ranking_operaciones_validacion"]
        ),
        "ranking_centralidad": safe_int(
            nodo_mayor_demanda["ranking_centralidad_validacion"]
        ),
        "indice_centralidad": safe_float(
            nodo_mayor_demanda["indice_centralidad"]
        ),
    },
    "mayor_discrepancia": {
        "nodo_id": safe_int(
            nodo_mayor_discrepancia["nodo_id"]
        ),
        "diferencia_ranking": safe_int(
            nodo_mayor_discrepancia["diferencia_ranking"]
        ),
        "diferencia_abs": safe_int(
            nodo_mayor_discrepancia["diferencia_ranking_abs"]
        ),
        "ranking_operaciones": safe_int(
            nodo_mayor_discrepancia[
                "ranking_operaciones_validacion"
            ]
        ),
        "ranking_centralidad": safe_int(
            nodo_mayor_discrepancia[
                "ranking_centralidad_validacion"
            ]
        ),
    },
    "nodos_baja_demanda_alta_centralidad": estructurales_lista,
    "nodos_alta_demanda_baja_centralidad": demanda_alta_lista,
    "top10_centralidad": top10_centralidad,
    "top10_demanda": top10_demanda,
    "graficos": {
        "mapa_centralidades": str(OUTPUT_MAPA_CENTRALIDADES),
        "demanda_vs_centralidad": str(OUTPUT_MAPA_DEMANDA),
        "mapa_h3": str(OUTPUT_MAPA_H3),
        "ranking": str(OUTPUT_RANKING),
        "concentracion": str(OUTPUT_CONCENTRACION),
        "componentes": str(OUTPUT_COMPONENTES),
        "discrepancias": str(OUTPUT_DISCREPANCIAS),
    },
}


# ============================================================
# GUARDAR
# ============================================================

print("\nGuardando validación...")

validacion = gpd.GeoDataFrame(
    validacion,
    geometry="geometry",
    crs=CRS_GEOGRAFICO,
)

validacion.to_parquet(
    OUTPUT_VALIDACION,
    index=False,
)

print(f"Validación guardada:\n{OUTPUT_VALIDACION}")

print("Guardando resumen...")

with open(
    OUTPUT_RESUMEN,
    "w",
    encoding="utf-8",
) as archivo:
    json.dump(
        resumen,
        archivo,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

print(f"Resumen guardado:\n{OUTPUT_RESUMEN}")


# ============================================================
# ARCHIVOS GENERADOS
# ============================================================

print()
print("=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)

print("\nValidación:")
print(OUTPUT_VALIDACION)

print("\nResumen:")
print(OUTPUT_RESUMEN)

print("\nGráficos:")

for archivo in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {archivo}")


# ============================================================
# FIN
# ============================================================

print()
print("=" * 70)
print("VALIDACIÓN DE CENTRALIDADES FINALIZADA")
print("=" * 70)

print(f"\nNodos analizados: {cantidad_nodos:,}")
print(f"Operaciones analizadas: {operaciones_total:,.0f}")
print(
    f"Centralidad principal: "
    f"Nodo {int(principal['nodo_id'])}"
)
print(
    f"Índice principal: "
    f"{principal['indice_centralidad']:.2f}/100"
)

print("\nSiguiente etapa:")
print(
    "Cruzar las centralidades con la red de transporte "
    "y la infraestructura intermodal para validar la "
    "centralidad estructural."
)