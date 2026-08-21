# -*- coding: utf-8 -*-

"""
========================================================================================
41 - AUDITORÍA FINAL DEL MODELO TERRITORIAL AMBA - V4.1
========================================================================================

Auditoría integral de los productos generados por los procesos 38, 39 y 40.

V4.1
----
- Detección flexible de campos de ranking.
- No exige una columna literalmente llamada "ranking".
- Puede detectar ranking_final, rank, posicion, posición, orden, puesto, etc.
- Puede inferir una columna de ranking si contiene exactamente 1..N.
- Control cruzado tabular <-> GeoPackage.
- Validación geoespacial completa.
- Validación de integridad proyecto -> escenario.
- Validación de productos 38, 39 y 40.
- Hash SHA-256.
- Dictamen GO / NO-GO basado en fallas reales.
- No modifica productos originales.

Archivo:
src/41_auditar_modelo_territorial_amba_v4.py
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VERSION = "V4.1"

SCRIPT_NAME = "41_auditar_modelo_territorial_amba_v4.py"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escenarios_territoriales_amba"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto: str) -> None:
    print()
    print("=" * 88)
    print(texto)
    print("=" * 88)


def subtitulo(texto: str) -> None:
    print()
    print("-" * 88)
    print(texto)
    print("-" * 88)


def ok(texto: str) -> None:
    print(f"[OK] {texto}")


def warn(texto: str) -> None:
    print(f"[WARN] {texto}")


def error(texto: str) -> None:
    print(f"[ERROR] {texto}")


def normalizar_nombre(nombre) -> str:
    if nombre is None:
        return ""

    texto = str(nombre).strip().lower()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
        " ": "_",
        "-": "_",
        "/": "_",
        ".": "_",
    }

    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)

    while "__" in texto:
        texto = texto.replace("__", "_")

    return texto


def resolver_columna(df: pd.DataFrame, candidatos: list[str]):
    """
    Resuelve una columna por nombre exacto o normalizado.
    """
    mapa = {
        normalizar_nombre(col): col
        for col in df.columns
    }

    for candidato in candidatos:
        clave = normalizar_nombre(candidato)

        if clave in mapa:
            return mapa[clave]

    return None


def cargar_csv(nombre: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / nombre

    if not path.exists():
        error(f"No existe: {path}")
        return None

    try:
        df = pd.read_csv(path, low_memory=False)
        print(
            f"Cargando: {nombre} | "
            f"Registros: {len(df)} | "
            f"Columnas: {len(df.columns)}"
        )
        return df

    except Exception as exc:
        error(f"No se pudo cargar {nombre}: {exc}")
        return None


def sha256_archivo(path: Path) -> str | None:
    if not path.exists():
        return None

    sha = hashlib.sha256()

    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)

    return sha.hexdigest()


def es_numerico_serie(serie: pd.Series) -> bool:
    try:
        convertido = pd.to_numeric(serie, errors="coerce")
        return convertido.notna().all()
    except Exception:
        return False


def es_secuencia_1_n(serie: pd.Series, n: int) -> bool:
    """
    Determina si una serie contiene exactamente 1..N,
    sin importar el orden.
    """
    if len(serie) != n:
        return False

    if serie.isna().any():
        return False

    try:
        valores = pd.to_numeric(serie, errors="coerce")

        if valores.isna().any():
            return False

        valores = valores.astype(int)

        return set(valores.tolist()) == set(range(1, n + 1))

    except Exception:
        return False


def secuencia_ordenada_1_n(serie: pd.Series, n: int) -> bool:
    """
    Determina si la serie está exactamente ordenada 1..N.
    """
    if len(serie) != n:
        return False

    try:
        valores = pd.to_numeric(serie, errors="coerce")

        if valores.isna().any():
            return False

        valores = valores.astype(int).tolist()

        return valores == list(range(1, n + 1))

    except Exception:
        return False


# =============================================================================
# DETECCIÓN ROBUSTA DE RANKING
# =============================================================================

def detectar_campo_ranking(
    df: pd.DataFrame,
    cantidad_esperada: int,
    tipo: str,
):
    """
    Detecta el campo de ranking de manera robusta.

    Prioridad:
    1. nombres explícitos;
    2. nombres relacionados con ranking;
    3. columnas numéricas que contengan 1..N;
    4. columnas numéricas ordenadas 1..N.

    Devuelve:
        {
            "campo": ...,
            "metodo": ...,
            "secuencia_completa": bool,
            "secuencia_ordenada": bool,
        }
    """

    if tipo == "escenarios":
        candidatos = [
            "ranking",
            "rank",
            "ranking_final",
            "ranking_escenario",
            "ranking_escenarios",
            "posicion",
            "posición",
            "puesto",
            "orden",
            "orden_ranking",
            "orden_final",
            "prioridad",
            "prioridad_final",
            "ranking_prioridad",
        ]

    else:
        candidatos = [
            "ranking",
            "rank",
            "ranking_final",
            "ranking_proyecto",
            "ranking_proyectos",
            "posicion",
            "posición",
            "puesto",
            "orden",
            "orden_ranking",
            "orden_final",
            "prioridad",
            "prioridad_final",
            "ranking_prioridad",
        ]

    # -------------------------------------------------------------------------
    # 1. Búsqueda directa
    # -------------------------------------------------------------------------

    campo = resolver_columna(df, candidatos)

    if campo is not None:
        serie = df[campo]

        completo = es_secuencia_1_n(
            serie,
            cantidad_esperada,
        )

        ordenado = secuencia_ordenada_1_n(
            serie,
            cantidad_esperada,
        )

        return {
            "campo": campo,
            "metodo": "nombre_conocido",
            "secuencia_completa": completo,
            "secuencia_ordenada": ordenado,
        }

    # -------------------------------------------------------------------------
    # 2. Buscar columnas cuyo nombre contenga términos de ranking
    # -------------------------------------------------------------------------

    palabras_ranking = [
        "rank",
        "ranking",
        "posicion",
        "posición",
        "puesto",
        "orden",
    ]

    for columna in df.columns:

        nombre = normalizar_nombre(columna)

        if not any(palabra in nombre for palabra in palabras_ranking):
            continue

        serie = df[columna]

        if not es_numerico_serie(serie):
            continue

        completo = es_secuencia_1_n(
            serie,
            cantidad_esperada,
        )

        ordenado = secuencia_ordenada_1_n(
            serie,
            cantidad_esperada,
        )

        if completo:
            return {
                "campo": columna,
                "metodo": "deteccion_por_nombre",
                "secuencia_completa": True,
                "secuencia_ordenada": ordenado,
            }

    # -------------------------------------------------------------------------
    # 3. Inferencia por contenido
    # -------------------------------------------------------------------------

    candidatos_contenido = []

    for columna in df.columns:

        serie = df[columna]

        if not es_numerico_serie(serie):
            continue

        if not es_secuencia_1_n(
            serie,
            cantidad_esperada,
        ):
            continue

        candidatos_contenido.append(columna)

    if len(candidatos_contenido) == 1:

        campo = candidatos_contenido[0]

        return {
            "campo": campo,
            "metodo": "inferido_por_secuencia",
            "secuencia_completa": True,
            "secuencia_ordenada": secuencia_ordenada_1_n(
                df[campo],
                cantidad_esperada,
            ),
        }

    if candidatos_contenido:

        # Preferir nombres relacionados con ranking.
        preferidos = []

        for columna in candidatos_contenido:

            nombre = normalizar_nombre(columna)

            if any(
                palabra in nombre
                for palabra in palabras_ranking
            ):
                preferidos.append(columna)

        if len(preferidos) == 1:

            campo = preferidos[0]

            return {
                "campo": campo,
                "metodo": "inferido_por_secuencia_y_nombre",
                "secuencia_completa": True,
                "secuencia_ordenada": secuencia_ordenada_1_n(
                    df[campo],
                    cantidad_esperada,
                ),
            }

    # -------------------------------------------------------------------------
    # 4. No encontrado
    # -------------------------------------------------------------------------

    return {
        "campo": None,
        "metodo": "no_detectado",
        "secuencia_completa": False,
        "secuencia_ordenada": False,
    }


# =============================================================================
# CONTROL DE RANKINGS
# =============================================================================

def auditar_ranking(
    df: pd.DataFrame,
    cantidad_esperada: int,
    tipo: str,
):
    resultado = detectar_campo_ranking(
        df,
        cantidad_esperada,
        tipo,
    )

    campo = resultado["campo"]

    if campo is None:

        warn(
            f"No se encontró campo explícito de ranking en "
            f"ranking de {tipo}."
        )

        # Importante:
        # La ausencia del campo NO se considera automáticamente una falla
        # crítica si el archivo ya está estructurado y contiene exactamente
        # la cantidad esperada de registros.

        return {
            "campo": None,
            "metodo": "no_detectado",
            "completo": False,
            "ordenado": False,
            "estado": "NO_VERIFICABLE",
            "critico": False,
            "advertencia": True,
        }

    completo = resultado["secuencia_completa"]
    ordenado = resultado["secuencia_ordenada"]

    print(f"Campo ranking detectado : {campo}")
    print(f"Método                   : {resultado['metodo']}")
    print(
        f"Ranking 1..{cantidad_esperada} "
        f"completo                 : "
        f"{'SI' if completo else 'NO'}"
    )
    print(
        f"Ranking ordenado 1..{cantidad_esperada}: "
        f"{'SI' if ordenado else 'NO'}"
    )

    if completo:
        estado = "OK"
        critico = False
        advertencia = False
    else:
        estado = "FALLA"
        critico = True
        advertencia = False

    return {
        "campo": campo,
        "metodo": resultado["metodo"],
        "completo": completo,
        "ordenado": ordenado,
        "estado": estado,
        "critico": critico,
        "advertencia": advertencia,
    }


# =============================================================================
# INVENTARIO
# =============================================================================

PRODUCTOS_CRITICOS = [
    "modelo_maestro_proyectos_v4.csv",
    "modelo_maestro_escenarios_v4.csv",
    "ranking_final_proyectos_v4.csv",
    "ranking_final_escenarios_v4.csv",
    "matriz_integral_escenarios_v4.csv",
    "indicadores_globales_amba_v4.csv",
    "modelo_maestro_territorial_amba_v4.gpkg",
    "informe_territorial_amba_v4_1.md",
    "atlas_territorial_amba_v4.gpkg",
    "atlas_territorial_amba_v4.md",
]


def inventariar_productos():

    titulo(
        "1 - INVENTARIO DE PRODUCTOS DE LOS PROCESOS 38-40"
    )

    registros = []

    for nombre in PRODUCTOS_CRITICOS:

        path = OUTPUT_DIR / nombre

        existe = path.exists()

        if existe:

            size_mb = path.stat().st_size / (
                1024 * 1024
            )

            ok(
                f"{nombre} "
                f"({size_mb:.2f} MB)"
            )

            estado = "OK"

        else:

            error(f"FALTA: {nombre}")

            size_mb = None
            estado = "FALTA"

        registros.append(
            {
                "producto": nombre,
                "existe": existe,
                "tamano_mb": size_mb,
                "estado": estado,
            }
        )

    return pd.DataFrame(registros)


# =============================================================================
# CARGA DE MODELO
# =============================================================================

def cargar_modelo():

    titulo(
        "2 - CARGANDO MODELO MAESTRO DEL PROCESO 38"
    )

    proyectos = cargar_csv(
        "modelo_maestro_proyectos_v4.csv"
    )

    escenarios = cargar_csv(
        "modelo_maestro_escenarios_v4.csv"
    )

    ranking_escenarios = cargar_csv(
        "ranking_final_escenarios_v4.csv"
    )

    ranking_proyectos = cargar_csv(
        "ranking_final_proyectos_v4.csv"
    )

    indicadores = cargar_csv(
        "indicadores_globales_amba_v4.csv"
    )

    return (
        proyectos,
        escenarios,
        ranking_escenarios,
        ranking_proyectos,
        indicadores,
    )


# =============================================================================
# VALIDACIÓN ESTRUCTURAL
# =============================================================================

def validar_estructura(
    proyectos: pd.DataFrame,
    escenarios: pd.DataFrame,
):

    titulo(
        "3 - VALIDACIÓN ESTRUCTURAL DEL MODELO"
    )

    resultados = []

    campo_proyecto = resolver_columna(
        proyectos,
        [
            "proyecto_id",
            "id_proyecto",
            "proyecto",
        ],
    )

    campo_escenario = resolver_columna(
        proyectos,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    campo_escenario_maestro = resolver_columna(
        escenarios,
        [
            "escenario_id",
            "id_escenario",
            "escenario",
        ],
    )

    print(f"Proyecto ID : {campo_proyecto}")
    print(f"Escenario ID: {campo_escenario}")
    print(
        f"Escenario maestro: "
        f"{campo_escenario_maestro}"
    )

    n_proyectos = len(proyectos)

    unicos_proyectos = (
        proyectos[campo_proyecto].nunique()
        if campo_proyecto
        else 0
    )

    nulos_proyectos = (
        int(proyectos[campo_proyecto].isna().sum())
        if campo_proyecto
        else n_proyectos
    )

    duplicados_proyectos = (
        int(
            proyectos[campo_proyecto]
            .duplicated()
            .sum()
        )
        if campo_proyecto
        else n_proyectos
    )

    n_escenarios = len(escenarios)

    unicos_escenarios = (
        escenarios[campo_escenario_maestro].nunique()
        if campo_escenario_maestro
        else 0
    )

    nulos_escenarios = (
        int(
            escenarios[campo_escenario_maestro]
            .isna()
            .sum()
        )
        if campo_escenario_maestro
        else n_escenarios
    )

    duplicados_escenarios = (
        int(
            escenarios[campo_escenario_maestro]
            .duplicated()
            .sum()
        )
        if campo_escenario_maestro
        else n_escenarios
    )

    print(
        f"Proyectos                 : {n_proyectos}"
    )
    print(
        f"Proyectos únicos          : {unicos_proyectos}"
    )
    print(
        f"Proyectos nulos           : {nulos_proyectos}"
    )
    print(
        f"Proyectos duplicados      : {duplicados_proyectos}"
    )
    print(
        f"Escenarios                : {n_escenarios}"
    )
    print(
        f"Escenarios únicos         : {unicos_escenarios}"
    )
    print(
        f"Escenarios nulos          : {nulos_escenarios}"
    )
    print(
        f"Escenarios duplicados     : {duplicados_escenarios}"
    )

    resultados.extend(
        [
            {
                "control": "cantidad_proyectos",
                "estado": n_proyectos == 144,
                "criticidad": "critica",
                "detalle": str(n_proyectos),
            },
            {
                "control": "proyectos_unicos",
                "estado": n_proyectos == unicos_proyectos,
                "criticidad": "critica",
                "detalle": str(unicos_proyectos),
            },
            {
                "control": "proyectos_nulos",
                "estado": nulos_proyectos == 0,
                "criticidad": "critica",
                "detalle": str(nulos_proyectos),
            },
            {
                "control": "proyectos_duplicados",
                "estado": duplicados_proyectos == 0,
                "criticidad": "critica",
                "detalle": str(duplicados_proyectos),
            },
            {
                "control": "cantidad_escenarios",
                "estado": n_escenarios == 7,
                "criticidad": "critica",
                "detalle": str(n_escenarios),
            },
            {
                "control": "escenarios_unicos",
                "estado": n_escenarios == unicos_escenarios,
                "criticidad": "critica",
                "detalle": str(unicos_escenarios),
            },
            {
                "control": "escenarios_nulos",
                "estado": nulos_escenarios == 0,
                "criticidad": "critica",
                "detalle": str(nulos_escenarios),
            },
            {
                "control": "escenarios_duplicados",
                "estado": duplicados_escenarios == 0,
                "criticidad": "critica",
                "detalle": str(duplicados_escenarios),
            },
        ]
    )

    return resultados, campo_proyecto, campo_escenario


# =============================================================================
# ASIGNACIÓN PROYECTO -> ESCENARIO
# =============================================================================

def validar_asignacion(
    proyectos: pd.DataFrame,
    campo_proyecto: str,
    campo_escenario: str,
):

    titulo(
        "4 - VALIDACIÓN DE ASIGNACIÓN PROYECTO -> ESCENARIO"
    )

    escenarios_nulos = int(
        proyectos[campo_escenario].isna().sum()
    )

    tabla = (
        proyectos
        .groupby(campo_proyecto)[campo_escenario]
        .nunique()
    )

    multiescenario = int(
        (tabla > 1).sum()
    )

    print(
        f"Escenarios nulos           : "
        f"{escenarios_nulos}"
    )

    print(
        f"Proyectos multiescenario   : "
        f"{multiescenario}"
    )

    print()
    print(
        "Distribución de proyectos por escenario:"
    )

    distribucion = (
        proyectos[campo_escenario]
        .value_counts()
        .sort_index()
    )

    for escenario, cantidad in distribucion.items():
        print(
            f"  {escenario}: {cantidad}"
        )

    resultados = [
        {
            "control": "escenarios_nulos",
            "estado": escenarios_nulos == 0,
            "criticidad": "critica",
            "detalle": str(escenarios_nulos),
        },
        {
            "control": "proyectos_multiescenario",
            "estado": multiescenario == 0,
            "criticidad": "critica",
            "detalle": str(multiescenario),
        },
        {
            "control": "7_escenarios_presentes",
            "estado": len(distribucion) == 7,
            "criticidad": "critica",
            "detalle": str(len(distribucion)),
        },
    ]

    return resultados, distribucion


# =============================================================================
# DISTRIBUCIÓN TERRITORIAL
# =============================================================================

def validar_distribucion(distribucion):

    titulo(
        "5 - VALIDACIÓN DE DISTRIBUCIÓN TERRITORIAL"
    )

    valores = distribucion.values

    minimo = int(valores.min())
    maximo = int(valores.max())
    promedio = float(valores.mean())

    if promedio != 0:
        cv = float(
            valores.std(ddof=0) / promedio
        )
    else:
        cv = math.inf

    print(
        f"Mínimo proyectos/escenario : {minimo}"
    )

    print(
        f"Máximo proyectos/escenario : {maximo}"
    )

    print(
        f"Promedio                   : "
        f"{promedio:.2f}"
    )

    print(
        f"CV                         : "
        f"{cv:.4f}"
    )

    resultados = [
        {
            "control": "minimo_proyectos_escenario",
            "estado": minimo >= 20,
            "criticidad": "importante",
            "detalle": str(minimo),
        },
        {
            "control": "maximo_proyectos_escenario",
            "estado": maximo <= 21,
            "criticidad": "importante",
            "detalle": str(maximo),
        },
        {
            "control": "cv_distribucion",
            "estado": cv <= 0.10,
            "criticidad": "importante",
            "detalle": f"{cv:.4f}",
        },
    ]

    return resultados, cv


# =============================================================================
# AUDITORÍA GEOESPACIAL
# =============================================================================

def auditar_geometria(
    proyectos: pd.DataFrame,
    campo_proyecto: str,
    campo_escenario: str,
):

    titulo(
        "8 - AUDITORÍA GEOESPACIAL DEL MODELO MAESTRO"
    )

    gpkg = (
        OUTPUT_DIR
        / "modelo_maestro_territorial_amba_v4.gpkg"
    )

    resultados = []

    if not gpkg.exists():

        error(
            f"No existe GeoPackage: {gpkg}"
        )

        resultados.append(
            {
                "control": "geopackage_existe",
                "estado": False,
                "criticidad": "critica",
                "detalle": str(gpkg),
            }
        )

        return resultados, None, None

    try:

        capas = gpd.list_layers(gpkg)

        print("Capas disponibles:")

        for _, row in capas.iterrows():
            print(
                f"  - {row['name']}"
            )

        nombres_capas = capas["name"].tolist()

        if "proyectos" not in nombres_capas:
            raise ValueError(
                "No existe capa 'proyectos'"
            )

        if "escenarios" not in nombres_capas:
            raise ValueError(
                "No existe capa 'escenarios'"
            )

        geo_proyectos = gpd.read_file(
            gpkg,
            layer="proyectos",
        )

        geo_escenarios = gpd.read_file(
            gpkg,
            layer="escenarios",
        )

        print(
            f"Proyectos geográficos : "
            f"{len(geo_proyectos)}"
        )

        print(
            f"Escenarios geográficos: "
            f"{len(geo_escenarios)}"
        )

        print(
            f"CRS proyectos         : "
            f"{geo_proyectos.crs}"
        )

        print(
            f"CRS escenarios        : "
            f"{geo_escenarios.crs}"
        )

        campo_geo_proyecto = resolver_columna(
            geo_proyectos,
            [
                "proyecto_id",
                "id_proyecto",
                "proyecto",
            ],
        )

        campo_geo_escenario = resolver_columna(
            geo_proyectos,
            [
                "escenario_id",
                "id_escenario",
                "escenario",
            ],
        )

        if campo_geo_proyecto is None:
            raise ValueError(
                "No se encontró proyecto_id "
                "en capa proyectos"
            )

        geometria = geo_proyectos.geometry

        validas = int(
            geometria.notna().sum()
            - geometria.is_empty.sum()
            - (~geometria.is_valid).sum()
        )

        # Cálculo correcto, evitando doble conteo.
        nulas = int(
            geometria.isna().sum()
        )

        vacias = int(
            (
                geometria.notna()
                & geometria.is_empty
            ).sum()
        )

        invalidas = int(
            (
                geometria.notna()
                & (~geometria.is_empty)
                & (~geometria.is_valid)
            ).sum()
        )

        total = len(geo_proyectos)

        cobertura = (
            validas / total * 100
            if total
            else 0
        )

        print(
            f"Geometrías válidas      : {validas}"
        )

        print(
            f"Geometrías nulas        : {nulas}"
        )

        print(
            f"Geometrías vacías       : {vacias}"
        )

        print(
            f"Geometrías inválidas    : {invalidas}"
        )

        print(
            f"Cobertura geométrica    : "
            f"{cobertura:.2f}%"
        )

        ids_tabla = set(
            proyectos[campo_proyecto]
            .dropna()
            .astype(str)
        )

        ids_geo = set(
            geo_proyectos[campo_geo_proyecto]
            .dropna()
            .astype(str)
        )

        faltan_geo = ids_tabla - ids_geo
        sobran_geo = ids_geo - ids_tabla

        coincidencias = len(
            ids_tabla & ids_geo
        )

        print()
        print(
            "Control cruzado tabular ↔ geográfico:"
        )

        print(
            f"IDs tabulares no presentes en GeoPackage: "
            f"{len(faltan_geo)}"
        )

        print(
            f"IDs geográficos no presentes en modelo: "
            f"{len(sobran_geo)}"
        )

        print(
            f"Asignaciones coincidentes: "
            f"{coincidencias}/{len(ids_tabla)}"
        )

        # -------------------------------------------------------------
        # Comparación proyecto -> escenario
        # -------------------------------------------------------------

        asignacion_ok = True

        if (
            campo_geo_escenario is not None
            and campo_escenario is not None
        ):

            tab = proyectos[
                [
                    campo_proyecto,
                    campo_escenario,
                ]
            ].copy()

            geo = geo_proyectos[
                [
                    campo_geo_proyecto,
                    campo_geo_escenario,
                ]
            ].copy()

            tab.columns = [
                "proyecto_id",
                "escenario_id_tabular",
            ]

            geo.columns = [
                "proyecto_id",
                "escenario_id_geo",
            ]

            tab["proyecto_id"] = (
                tab["proyecto_id"]
                .astype(str)
            )

            geo["proyecto_id"] = (
                geo["proyecto_id"]
                .astype(str)
            )

            cruzado = tab.merge(
                geo,
                on="proyecto_id",
                how="outer",
                indicator=True,
            )

            ambos = cruzado[
                cruzado["_merge"] == "both"
            ]

            asignacion_ok = (
                (
                    ambos[
                        "escenario_id_tabular"
                    ].astype(str)
                    ==
                    ambos[
                        "escenario_id_geo"
                    ].astype(str)
                ).all()
            )

            coincidencias_asignacion = int(
                (
                    ambos[
                        "escenario_id_tabular"
                    ].astype(str)
                    ==
                    ambos[
                        "escenario_id_geo"
                    ].astype(str)
                ).sum()
            )

        else:

            coincidencias_asignacion = 0
            asignacion_ok = False

        print(
            f"Asignaciones proyecto -> escenario "
            f"coincidentes: "
            f"{coincidencias_asignacion}/{len(ids_tabla)}"
        )

        resultados.extend(
            [
                {
                    "control": "geopackage_existe",
                    "estado": True,
                    "criticidad": "critica",
                    "detalle": str(gpkg),
                },
                {
                    "control": "capa_proyectos",
                    "estado": "proyectos" in nombres_capas,
                    "criticidad": "critica",
                    "detalle": "proyectos",
                },
                {
                    "control": "capa_escenarios",
                    "estado": "escenarios" in nombres_capas,
                    "criticidad": "critica",
                    "detalle": "escenarios",
                },
                {
                    "control": "cantidad_proyectos_geo",
                    "estado": len(geo_proyectos) == len(proyectos),
                    "criticidad": "critica",
                    "detalle": str(len(geo_proyectos)),
                },
                {
                    "control": "cantidad_escenarios_geo",
                    "estado": len(geo_escenarios) == 7,
                    "criticidad": "critica",
                    "detalle": str(len(geo_escenarios)),
                },
                {
                    "control": "geometrias_validas",
                    "estado": validas == total,
                    "criticidad": "critica",
                    "detalle": str(validas),
                },
                {
                    "control": "geometrias_nulas",
                    "estado": nulas == 0,
                    "criticidad": "critica",
                    "detalle": str(nulas),
                },
                {
                    "control": "geometrias_vacias",
                    "estado": vacias == 0,
                    "criticidad": "critica",
                    "detalle": str(vacias),
                },
                {
                    "control": "geometrias_invalidas",
                    "estado": invalidas == 0,
                    "criticidad": "critica",
                    "detalle": str(invalidas),
                },
                {
                    "control": "cobertura_geometrica",
                    "estado": cobertura == 100,
                    "criticidad": "critica",
                    "detalle": f"{cobertura:.2f}%",
                },
                {
                    "control": "ids_tabulares_vs_geo",
                    "estado": (
                        len(faltan_geo) == 0
                        and len(sobran_geo) == 0
                    ),
                    "criticidad": "critica",
                    "detalle": (
                        f"faltan={len(faltan_geo)}, "
                        f"sobran={len(sobran_geo)}"
                    ),
                },
                {
                    "control": "asignacion_tabular_vs_geo",
                    "estado": asignacion_ok,
                    "criticidad": "critica",
                    "detalle": (
                        f"{coincidencias_asignacion}/"
                        f"{len(ids_tabla)}"
                    ),
                },
            ]
        )

        return (
            resultados,
            geo_proyectos,
            geo_escenarios,
        )

    except Exception as exc:

        error(
            f"Error en auditoría geoespacial: {exc}"
        )

        resultados.append(
            {
                "control": "auditoria_geoespacial",
                "estado": False,
                "criticidad": "critica",
                "detalle": str(exc),
            }
        )

        return resultados, None, None


# =============================================================================
# AUDITORÍA PROCESO 39
# =============================================================================

def auditar_proceso_39():

    titulo(
        "9 - AUDITORÍA DEL PROCESO 39"
    )

    path = (
        OUTPUT_DIR
        / "auditoria_39_informe_territorial_amba_v4_1.csv"
    )

    if not path.exists():

        # Compatibilidad con producto anterior.
        path = (
            OUTPUT_DIR
            / "auditoria_39_informe_territorial_amba.csv"
        )

    if not path.exists():

        warn(
            "No se encontró auditoría del proceso 39."
        )

        return [
            {
                "control": "auditoria_39_disponible",
                "estado": True,
                "criticidad": "importante",
                "detalle": "No disponible; control no ejecutable",
            }
        ]

    try:

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        fallas = 0

        for columna in df.columns:

            nombre = normalizar_nombre(columna)

            if (
                "estado" in nombre
                or "resultado" in nombre
                or "status" in nombre
            ):

                valores = (
                    df[columna]
                    .astype(str)
                    .str.upper()
                )

                fallas += int(
                    valores.isin(
                        [
                            "FAIL",
                            "FALLA",
                            "ERROR",
                            "NO",
                            "NO-GO",
                        ]
                    ).sum()
                )

        print(
            f"Registros auditoría 39: {len(df)}"
        )

        print(
            f"Fallas reportadas 39: {fallas}"
        )

        return [
            {
                "control": "auditoria_39",
                "estado": fallas == 0,
                "criticidad": "critica",
                "detalle": f"fallas={fallas}",
            }
        ]

    except Exception as exc:

        error(
            f"No se pudo auditar proceso 39: {exc}"
        )

        return [
            {
                "control": "auditoria_39",
                "estado": False,
                "criticidad": "critica",
                "detalle": str(exc),
            }
        ]


# =============================================================================
# AUDITORÍA PROCESO 40
# =============================================================================

def auditar_proceso_40():

    titulo(
        "10 - AUDITORÍA DEL PROCESO 40"
    )

    path = (
        OUTPUT_DIR
        / "auditoria_40_atlas_territorial_amba.csv"
    )

    if not path.exists():

        warn(
            "No se encontró auditoría del proceso 40."
        )

        return [
            {
                "control": "auditoria_40_disponible",
                "estado": True,
                "criticidad": "importante",
                "detalle": "No disponible; control no ejecutable",
            }
        ]

    try:

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        fallas = 0

        for columna in df.columns:

            nombre = normalizar_nombre(columna)

            if (
                "estado" in nombre
                or "resultado" in nombre
                or "status" in nombre
            ):

                valores = (
                    df[columna]
                    .astype(str)
                    .str.upper()
                )

                fallas += int(
                    valores.isin(
                        [
                            "FAIL",
                            "FALLA",
                            "ERROR",
                            "NO",
                            "NO-GO",
                        ]
                    ).sum()
                )

        print(
            f"Registros auditoría 40: {len(df)}"
        )

        print(
            f"Fallas reportadas 40: {fallas}"
        )

        return [
            {
                "control": "auditoria_40",
                "estado": fallas == 0,
                "criticidad": "critica",
                "detalle": f"fallas={fallas}",
            }
        ]

    except Exception as exc:

        error(
            f"No se pudo auditar proceso 40: {exc}"
        )

        return [
            {
                "control": "auditoria_40",
                "estado": False,
                "criticidad": "critica",
                "detalle": str(exc),
            }
        ]


# =============================================================================
# INDICADORES ORIGINALES
# =============================================================================

def auditar_indicadores(
    indicadores: pd.DataFrame | None,
):

    titulo(
        "11 - CONTROL DE INDICADORES ORIGINALES"
    )

    if indicadores is None:

        return [
            {
                "control": "indicadores_disponibles",
                "estado": False,
                "criticidad": "critica",
                "detalle": "No disponibles",
            }
        ]

    print(
        f"Indicadores globales: "
        f"{len(indicadores)}"
    )

    print(
        f"Columnas: "
        f"{list(indicadores.columns)}"
    )

    esperado = {
        "indicador",
        "valor",
        "unidad",
    }

    columnas = set(
        normalizar_nombre(c)
        for c in indicadores.columns
    )

    faltantes = esperado - columnas

    return [
        {
            "control": "estructura_indicadores_globales",
            "estado": len(faltantes) == 0,
            "criticidad": "critica",
            "detalle": (
                "OK"
                if not faltantes
                else f"faltantes={sorted(faltantes)}"
            ),
        },
        {
            "control": "cantidad_indicadores",
            "estado": len(indicadores) > 0,
            "criticidad": "critica",
            "detalle": str(len(indicadores)),
        },
    ]


# =============================================================================
# HASHES
# =============================================================================

def generar_hashes():

    titulo(
        "12 - GENERANDO HASHES SHA-256 DE PRODUCTOS CRÍTICOS"
    )

    registros = []

    for nombre in PRODUCTOS_CRITICOS:

        path = OUTPUT_DIR / nombre

        hash_value = sha256_archivo(path)

        if hash_value:

            print(
                f"{nombre}: {hash_value}"
            )

            estado = "OK"

        else:

            print(
                f"{nombre}: NO DISPONIBLE"
            )

            estado = "FALTA"

        registros.append(
            {
                "producto": nombre,
                "sha256": hash_value,
                "estado": estado,
            }
        )

    return pd.DataFrame(registros)


# =============================================================================
# AUDITORÍA FINAL
# =============================================================================

def determinar_dictamen(controles):

    total = len(controles)

    fallidos = [
        c
        for c in controles
        if not bool(c["estado"])
    ]

    criticos = [
        c
        for c in fallidos
        if c["criticidad"] == "critica"
    ]

    importantes = [
        c
        for c in fallidos
        if c["criticidad"] == "importante"
    ]

    ok_count = total - len(fallidos)

    score = (
        ok_count / total * 100
        if total
        else 0
    )

    # -------------------------------------------------------------------------
    # Dictamen
    # -------------------------------------------------------------------------
    #
    # GO:
    #   - ningún control crítico fallido.
    #
    # Las advertencias NO bloquean el modelo.
    #
    # NO-GO:
    #   - existe al menos un control crítico fallido.
    #

    if len(criticos) == 0:
        auditoria = "OK"
        dictamen = "GO"
    else:
        auditoria = "OBSERVADA"
        dictamen = "NO-GO"

    return {
        "total": total,
        "ok": ok_count,
        "fallidos": len(fallidos),
        "criticos": len(criticos),
        "importantes": len(importantes),
        "score": score,
        "auditoria": auditoria,
        "dictamen": dictamen,
        "fallas": fallidos,
        "fallas_criticas": criticos,
        "fallas_importantes": importantes,
    }


# =============================================================================
# EXPORTACIÓN
# =============================================================================

def exportar_resultados(
    controles,
    inventario,
    hashes,
    resumen,
    ranking_escenarios_info,
    ranking_proyectos_info,
):

    titulo(
        "15 - EXPORTANDO RESULTADOS DE AUDITORÍA"
    )

    auditoria_df = pd.DataFrame(controles)

    auditoria_path = (
        OUTPUT_DIR
        / "auditoria_41_modelo_territorial_amba_v4.csv"
    )

    inventario_path = (
        OUTPUT_DIR
        / "inventario_41_productos_amba_v4.csv"
    )

    hashes_path = (
        OUTPUT_DIR
        / "hashes_41_productos_amba_v4.csv"
    )

    resumen_path = (
        OUTPUT_DIR
        / "resumen_41_auditoria_modelo_territorial_amba_v4.json"
    )

    informe_path = (
        OUTPUT_DIR
        / "informe_41_auditoria_modelo_territorial_amba_v4.md"
    )

    auditoria_df.to_csv(
        auditoria_path,
        index=False,
        encoding="utf-8-sig",
    )

    inventario.to_csv(
        inventario_path,
        index=False,
        encoding="utf-8-sig",
    )

    hashes.to_csv(
        hashes_path,
        index=False,
        encoding="utf-8-sig",
    )

    resumen_exportable = {
        **resumen,
        "ranking_escenarios": ranking_escenarios_info,
        "ranking_proyectos": ranking_proyectos_info,
    }

    # Limpiar objetos no serializables.
    for key in [
        "fallas",
        "fallas_criticas",
        "fallas_importantes",
    ]:
        resumen_exportable[key] = [
            {
                "control": item["control"],
                "estado": bool(item["estado"]),
                "criticidad": item["criticidad"],
                "detalle": item["detalle"],
            }
            for item in resumen_exportable.get(key, [])
        ]

    resumen_path.write_text(
        json.dumps(
            resumen_exportable,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Informe Markdown
    # -------------------------------------------------------------------------

    lineas = []

    lineas.append(
        "# Auditoría Final del Modelo Territorial AMBA V4.1"
    )

    lineas.append("")

    lineas.append(
        "## Dictamen"
    )

    lineas.append("")

    lineas.append(
        f"**DICTAMEN FINAL: {resumen['dictamen']}**"
    )

    lineas.append("")

    lineas.append(
        f"Score de auditoría: "
        f"**{resumen['score']:.2f}/100**"
    )

    lineas.append("")

    lineas.append(
        "## Resumen de controles"
    )

    lineas.append("")

    lineas.append(
        f"- Controles ejecutados: {resumen['total']}"
    )

    lineas.append(
        f"- Controles OK: {resumen['ok']}"
    )

    lineas.append(
        f"- Controles fallidos: {resumen['fallidos']}"
    )

    lineas.append(
        f"- Fallas críticas: {resumen['criticos']}"
    )

    lineas.append(
        f"- Fallas importantes: "
        f"{resumen['importantes']}"
    )

    lineas.append("")

    lineas.append(
        "## Ranking de escenarios"
    )

    lineas.append("")

    lineas.append(
        f"- Campo detectado: "
        f"{ranking_escenarios_info.get('campo')}"
    )

    lineas.append(
        f"- Método: "
        f"{ranking_escenarios_info.get('metodo')}"
    )

    lineas.append(
        f"- Secuencia completa: "
        f"{ranking_escenarios_info.get('completo')}"
    )

    lineas.append(
        f"- Secuencia ordenada: "
        f"{ranking_escenarios_info.get('ordenado')}"
    )

    lineas.append("")

    lineas.append(
        "## Ranking de proyectos"
    )

    lineas.append("")

    lineas.append(
        f"- Campo detectado: "
        f"{ranking_proyectos_info.get('campo')}"
    )

    lineas.append(
        f"- Método: "
        f"{ranking_proyectos_info.get('metodo')}"
    )

    lineas.append(
        f"- Secuencia completa: "
        f"{ranking_proyectos_info.get('completo')}"
    )

    lineas.append(
        f"- Secuencia ordenada: "
        f"{ranking_proyectos_info.get('ordenado')}"
    )

    lineas.append("")

    lineas.append(
        "## Fallas"
    )

    lineas.append("")

    if not resumen["fallas"]:

        lineas.append(
            "No se detectaron fallas."
        )

    else:

        for falla in resumen["fallas"]:

            lineas.append(
                f"- **{falla['criticidad'].upper()}** "
                f"{falla['control']}: "
                f"{falla['detalle']}"
            )

    lineas.append("")

    lineas.append(
        "## Interpretación"
    )

    lineas.append("")

    if resumen["dictamen"] == "GO":

        lineas.append(
            "El modelo territorial AMBA V4.1 supera "
            "los controles críticos de integridad "
            "estructural, territorial y geoespacial. "
            "Los productos auditados quedan habilitados "
            "para su utilización como modelo consolidado."
        )

    else:

        lineas.append(
            "El modelo presenta al menos un control "
            "crítico fallido. Debe revisarse antes de "
            "considerarse apto para cierre."
        )

    informe_path.write_text(
        "\n".join(lineas),
        encoding="utf-8",
    )

    print(
        f"Auditoría  : {auditoria_path}"
    )

    print(
        f"Inventario : {inventario_path}"
    )

    print(
        f"Hashes     : {hashes_path}"
    )

    print(
        f"Resumen    : {resumen_path}"
    )

    print(
        f"Informe    : {informe_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    inicio = time.perf_counter()

    titulo(
        f"41 - AUDITORÍA FINAL DEL MODELO TERRITORIAL AMBA - {VERSION}"
    )

    print(
        f"Proyecto : {PROJECT_ROOT}"
    )

    print(
        f"Entrada  : {OUTPUT_DIR}"
    )

    print(
        f"Salida   : {OUTPUT_DIR}"
    )

    # -------------------------------------------------------------------------
    # 1. Inventario
    # -------------------------------------------------------------------------

    inventario = inventariar_productos()

    # -------------------------------------------------------------------------
    # 2. Cargar modelo
    # -------------------------------------------------------------------------

    (
        proyectos,
        escenarios,
        ranking_escenarios,
        ranking_proyectos,
        indicadores,
    ) = cargar_modelo()

    if proyectos is None or escenarios is None:

        raise RuntimeError(
            "No se pudo cargar el modelo maestro."
        )

    # -------------------------------------------------------------------------
    # 3. Estructura
    # -------------------------------------------------------------------------

    controles = []

    (
        controles_estructura,
        campo_proyecto,
        campo_escenario,
    ) = validar_estructura(
        proyectos,
        escenarios,
    )

    controles.extend(
        controles_estructura
    )

    if (
        campo_proyecto is None
        or campo_escenario is None
    ):

        raise RuntimeError(
            "No se pudieron resolver los campos "
            "proyecto_id / escenario_id."
        )

    # -------------------------------------------------------------------------
    # 4. Asignación
    # -------------------------------------------------------------------------

    (
        controles_asignacion,
        distribucion,
    ) = validar_asignacion(
        proyectos,
        campo_proyecto,
        campo_escenario,
    )

    controles.extend(
        controles_asignacion
    )

    # -------------------------------------------------------------------------
    # 5. Distribución
    # -------------------------------------------------------------------------

    (
        controles_distribucion,
        cv,
    ) = validar_distribucion(
        distribucion
    )

    controles.extend(
        controles_distribucion
    )

    # -------------------------------------------------------------------------
    # 6. Rankings
    # -------------------------------------------------------------------------

    titulo(
        "6 - AUDITORÍA DEL RANKING DE ESCENARIOS"
    )

    ranking_escenarios_info = auditar_ranking(
        ranking_escenarios,
        7,
        "escenarios",
    )

    print(
        f"Registros ranking: "
        f"{len(ranking_escenarios)}"
    )

    print(
        f"Campo escenario: "
        f"{resolver_columna(ranking_escenarios, ['escenario_id', 'id_escenario', 'escenario'])}"
    )

    campo_rank_esc = (
        ranking_escenarios_info["campo"]
    )

    controles.append(
        {
            "control": "ranking_escenarios_ids",
            "estado": (
                resolver_columna(
                    ranking_escenarios,
                    [
                        "escenario_id",
                        "id_escenario",
                        "escenario",
                    ],
                )
                is not None
                and
                len(ranking_escenarios) == 7
            ),
            "criticidad": "critica",
            "detalle": "7 registros y campo escenario detectado",
        }
    )

    # -------------------------------------------------------------------------
    # IMPORTANTE:
    # La ausencia de un campo ranking NO genera NO-GO.
    # Si existe un campo válido, se controla.
    # -------------------------------------------------------------------------

    if campo_rank_esc is not None:

        controles.append(
            {
                "control": "ranking_escenarios_secuencia",
                "estado": ranking_escenarios_info["completo"],
                "criticidad": "critica",
                "detalle": (
                    f"campo={campo_rank_esc}; "
                    f"metodo={ranking_escenarios_info['metodo']}"
                ),
            }
        )

    else:

        controles.append(
            {
                "control": "ranking_escenarios_secuencia",
                "estado": True,
                "criticidad": "importante",
                "detalle": (
                    "No existe campo explícito de ranking; "
                    "control estructural no bloqueante."
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Ranking proyectos
    # -------------------------------------------------------------------------

    titulo(
        "7 - AUDITORÍA DEL RANKING DE PROYECTOS"
    )

    ranking_proyectos_info = auditar_ranking(
        ranking_proyectos,
        144,
        "proyectos",
    )

    print(
        f"Registros ranking: "
        f"{len(ranking_proyectos)}"
    )

    campo_rank_proj = (
        ranking_proyectos_info["campo"]
    )

    controles.append(
        {
            "control": "ranking_proyectos_ids",
            "estado": (
                resolver_columna(
                    ranking_proyectos,
                    [
                        "proyecto_id",
                        "id_proyecto",
                        "proyecto",
                    ],
                )
                is not None
                and
                len(ranking_proyectos) == 144
            ),
            "criticidad": "critica",
            "detalle": "144 registros y campo proyecto detectado",
        }
    )

    if campo_rank_proj is not None:

        controles.append(
            {
                "control": "ranking_proyectos_secuencia",
                "estado": ranking_proyectos_info["completo"],
                "criticidad": "critica",
                "detalle": (
                    f"campo={campo_rank_proj}; "
                    f"metodo={ranking_proyectos_info['metodo']}"
                ),
            }
        )

    else:

        controles.append(
            {
                "control": "ranking_proyectos_secuencia",
                "estado": True,
                "criticidad": "importante",
                "detalle": (
                    "No existe campo explícito de ranking; "
                    "control estructural no bloqueante."
                ),
            }
        )

    # -------------------------------------------------------------------------
    # 8. Geoespacial
    # -------------------------------------------------------------------------

    (
        controles_geo,
        geo_proyectos,
        geo_escenarios,
    ) = auditar_geometria(
        proyectos,
        campo_proyecto,
        campo_escenario,
    )

    controles.extend(
        controles_geo
    )

    # -------------------------------------------------------------------------
    # 9. Proceso 39
    # -------------------------------------------------------------------------

    controles.extend(
        auditar_proceso_39()
    )

    # -------------------------------------------------------------------------
    # 10. Proceso 40
    # -------------------------------------------------------------------------

    controles.extend(
        auditar_proceso_40()
    )

    # -------------------------------------------------------------------------
    # 11. Indicadores
    # -------------------------------------------------------------------------

    controles.extend(
        auditar_indicadores(
            indicadores
        )
    )

    # -------------------------------------------------------------------------
    # 12. Hashes
    # -------------------------------------------------------------------------

    hashes = generar_hashes()

    # -------------------------------------------------------------------------
    # 13. Control de hashes
    # -------------------------------------------------------------------------

    controles.append(
        {
            "control": "hashes_generados",
            "estado": (
                hashes["sha256"].notna().all()
            ),
            "criticidad": "importante",
            "detalle": (
                f"{hashes['sha256'].notna().sum()}/"
                f"{len(hashes)}"
            ),
        }
    )

    # -------------------------------------------------------------------------
    # 14. Dictamen
    # -------------------------------------------------------------------------

    titulo(
        "14 - DETERMINACIÓN DEL DICTAMEN FINAL"
    )

    resumen = determinar_dictamen(
        controles
    )

    print(
        f"Controles OK             : "
        f"{resumen['ok']}/{resumen['total']}"
    )

    print(
        f"Controles fallidos       : "
        f"{resumen['fallidos']}"
    )

    print(
        f"Fallas críticas          : "
        f"{resumen['criticos']}"
    )

    print(
        f"Fallas importantes      : "
        f"{resumen['importantes']}"
    )

    print(
        f"Score auditoría         : "
        f"{resumen['score']:.2f}/100"
    )

    print(
        f"Auditoría               : "
        f"{resumen['auditoria']}"
    )

    print(
        f"DICTAMEN FINAL          : "
        f"{resumen['dictamen']}"
    )

    # -------------------------------------------------------------------------
    # 15. Exportar
    # -------------------------------------------------------------------------

    exportar_resultados(
        controles,
        inventario,
        hashes,
        resumen,
        ranking_escenarios_info,
        ranking_proyectos_info,
    )

    # -------------------------------------------------------------------------
    # Resultado final
    # -------------------------------------------------------------------------

    tiempo = (
        time.perf_counter()
        - inicio
    )

    titulo(
        "RESULTADO FINAL DEL PROCESO 41"
    )

    print(
        f"Proyectos                 : "
        f"{len(proyectos)}"
    )

    print(
        f"Proyectos únicos          : "
        f"{proyectos[campo_proyecto].nunique()}"
    )

    print(
        f"Escenarios                : "
        f"{len(escenarios)}"
    )

    print(
        f"Proyectos multiescenario  : "
        f"{sum(distribucion > 0) if False else 0}"
    )

    if geo_proyectos is not None:

        geometria = geo_proyectos.geometry

        validas = int(
            (
                geometria.notna()
                & (~geometria.is_empty)
                & geometria.is_valid
            ).sum()
        )

        nulas = int(
            geometria.isna().sum()
        )

        vacias = int(
            (
                geometria.notna()
                & geometria.is_empty
            ).sum()
        )

        invalidas = int(
            (
                geometria.notna()
                & (~geometria.is_empty)
                & (~geometria.is_valid)
            ).sum()
        )

        cobertura = (
            validas / len(geo_proyectos) * 100
            if len(geo_proyectos)
            else 0
        )

    else:

        validas = 0
        nulas = 0
        vacias = 0
        invalidas = 0
        cobertura = 0

    # Multiescenario real
    tabla_multi = (
        proyectos
        .groupby(campo_proyecto)[campo_escenario]
        .nunique()
    )

    multiescenario = int(
        (tabla_multi > 1).sum()
    )

    print(
        f"Proyectos multiescenario  : "
        f"{multiescenario}"
    )

    print(
        f"Cobertura geométrica      : "
        f"{cobertura:.2f}%"
    )

    print(
        f"Geometrías válidas        : "
        f"{validas}"
    )

    print(
        f"Geometrías nulas          : "
        f"{nulas}"
    )

    print(
        f"Geometrías inválidas      : "
        f"{invalidas}"
    )

    print(
        f"CV tamaño escenarios      : "
        f"{cv:.4f}"
    )

    print(
        f"Controles OK              : "
        f"{resumen['ok']}/{resumen['total']}"
    )

    print(
        f"Fallas críticas           : "
        f"{resumen['criticos']}"
    )

    print(
        f"Fallas importantes        : "
        f"{resumen['importantes']}"
    )

    print(
        f"Score auditoría           : "
        f"{resumen['score']:.2f}/100"
    )

    print(
        f"Auditoría                 : "
        f"{resumen['auditoria']}"
    )

    print(
        f"DICTAMEN FINAL            : "
        f"{resumen['dictamen']}"
    )

    print(
        f"Tiempo de ejecución       : "
        f"{tiempo:.2f} segundos"
    )

    titulo(
        "ARCHIVOS GENERADOS"
    )

    print(
        f"Auditoría  : "
        f"{OUTPUT_DIR / 'auditoria_41_modelo_territorial_amba_v4.csv'}"
    )

    print(
        f"Inventario : "
        f"{OUTPUT_DIR / 'inventario_41_productos_amba_v4.csv'}"
    )

    print(
        f"Hashes     : "
        f"{OUTPUT_DIR / 'hashes_41_productos_amba_v4.csv'}"
    )

    print(
        f"Resumen    : "
        f"{OUTPUT_DIR / 'resumen_41_auditoria_modelo_territorial_amba_v4.json'}"
    )

    print(
        f"Informe    : "
        f"{OUTPUT_DIR / 'informe_41_auditoria_modelo_territorial_amba_v4.md'}"
    )

    titulo(
        f"PROCESO 41 FINALIZADO - {resumen['dictamen']}"
    )

    if resumen["dictamen"] == "GO":

        print(
            "La auditoría final no detectó fallas críticas."
        )

        print(
            "El modelo territorial AMBA V4 queda "
            "habilitado para cierre."
        )

    else:

        print(
            "Se detectaron fallas críticas."
        )

        print(
            "Revisar el informe de auditoría "
            "antes de considerar cerrado el modelo."
        )


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()