# Informe Territorial AMBA V4.1

**Proceso:** 39 - Generación del Informe Territorial AMBA  
**Versión:** V4.1  
**Proyecto:** movilidad  
**Estado:** **VALIDADO**

---

# 1. Resumen ejecutivo

El presente documento constituye el informe territorial consolidado del
Área Metropolitana de Buenos Aires (AMBA), elaborado sobre la base del
modelo maestro producido por el proceso 38.

El modelo integra información de proyectos, escenarios territoriales,
priorización, indicadores estructurales, cartera de intervención y
geometrías geográficas.

## 1.1 Magnitud del modelo

| Indicador | Resultado |
|---|---:|
| Proyectos | 144 |
| Proyectos únicos | 144 |
| Escenarios | 7 |
| Cobertura geométrica | 100.00% |
| Geometrías válidas | 144 |
| Geometrías nulas | 0 |
| Geometrías inválidas | 0 |
| Proyectos multiescenario | 0 |
| CV tamaño escenarios | 0.0241 |
| Score medio de escenarios | 49,33 |

---

# 2. Validación integral

La validación del modelo produjo los siguientes resultados:

- Proyectos únicos: **144**
- Escenarios: **7**
- Proyectos duplicados: **0**
- Proyectos sin identificador: **0**
- Escenarios sin identificador: **0**
- Proyectos multiescenario: **0**
- Cobertura geométrica: **100.00%**
- Geometrías inválidas: **0**

El resultado general del control es:

**VALIDADO**

---

# 3. Modelo geográfico

La geometría utilizada por este proceso se recupera del GeoPackage maestro:

`modelo_maestro_territorial_amba_v4.gpkg`

El GeoPackage contiene las capas:

- `proyectos`
- `escenarios`

La utilización del GeoPackage como fuente geométrica evita depender de la
presencia de geometrías serializadas dentro de los archivos CSV.

## 3.1 Cobertura espacial

La cobertura geométrica del modelo es:

**100.00%**

Geometrías válidas:

**144**

Geometrías inválidas:

**0**

---

# 4. Estructura territorial

El modelo se encuentra distribuido en:

**7 escenarios territoriales.**

La distribución de proyectos presenta:

- mínimo: **20**
- máximo: **21**
- promedio: **20.57**
- coeficiente de variación: **0.0241**

La baja variabilidad relativa indica una distribución territorial
relativamente equilibrada en términos del número de proyectos por escenario.

---

# 5. Ranking final de escenarios

| ranking_informe_v4_1 | escenario_id | cantidad_proyectos | tipo_escenario |
| --- | --- | --- | --- |
| 1 | AMBA-E001 | 20 | ESCENARIO_DE_NECESIDAD |
| 2 | AMBA-E003 | 20 | ESCENARIO_DE_NECESIDAD |
| 3 | AMBA-E002 | 21 | ESCENARIO_ESTRATEGICO |
| 4 | AMBA-E004 | 21 | ESCENARIO_DE_NECESIDAD |
| 5 | AMBA-E005 | 21 | ESCENARIO_DE_NECESIDAD |
| 6 | AMBA-E006 | 20 | ESCENARIO_DE_NECESIDAD |
| 7 | AMBA-E007 | 21 | ESCENARIO_DE_NECESIDAD |

---

# 6. Ranking final de proyectos

| ranking_informe_v4_1 | proyecto_id | escenario_id | tipo_escenario | dimension_dominante |
| --- | --- | --- | --- | --- |
| 1 | AMBA-P005 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 2 | AMBA-P008 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 3 | AMBA-P012 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 4 | AMBA-P013 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 5 | AMBA-P016 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 6 | AMBA-P019 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 7 | AMBA-P021 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 8 | AMBA-P023 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 9 | AMBA-P024 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 10 | AMBA-P036 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 11 | AMBA-P054 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 12 | AMBA-P055 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 13 | AMBA-P058 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 14 | AMBA-P059 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 15 | AMBA-P077 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 16 | AMBA-P090 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 17 | AMBA-P097 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 18 | AMBA-P116 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 19 | AMBA-P142 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |
| 20 | AMBA-P143 | AMBA-E001 | ESCENARIO_DE_NECESIDAD | INTEGRACION |

---

# 7. Matriz integral de escenarios

| ranking_integral_v4 | escenario_id | cantidad_proyectos | tipo_escenario | dimension_dominante | prioridad_escenario | prioridad_territorial_v4 | categoria_cartera_v4 | linea_estrategica_v4 | horizonte_intervencion_v4 | score_integral_v4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | AMBA-E001 | 20 | ESCENARIO_DE_NECESIDAD | INTEGRACION | PRIORIDAD_3_MEDIA | PRIORIDAD_2_ALTA | P2_PRIORITARIA | INTEGRACION_TERRITORIAL | CORTO_MEDIANO_PLAZO | 76,96 |
| 2 | AMBA-E003 | 20 | ESCENARIO_DE_NECESIDAD | IMPACTO | PRIORIDAD_3_MEDIA | PRIORIDAD_3_MEDIA | P3_PROGRAMADA | IMPACTO_TERRITORIAL | MEDIANO_PLAZO | 54,35 |
| 3 | AMBA-E002 | 21 | ESCENARIO_ESTRATEGICO | CONECTIVIDAD | PRIORIDAD_3_MEDIA | PRIORIDAD_3_MEDIA | P3_PROGRAMADA | CONECTIVIDAD_METROPOLITANA | MEDIANO_PLAZO | 52,70 |
| 4 | AMBA-E004 | 21 | ESCENARIO_DE_NECESIDAD | DEFICIT | PRIORIDAD_3_MEDIA | PRIORIDAD_3_MEDIA | P3_PROGRAMADA | REDUCCION_DE_BRECHAS | MEDIANO_PLAZO | 46,66 |
| 5 | AMBA-E005 | 21 | ESCENARIO_DE_NECESIDAD | DEFICIT | PRIORIDAD_3_MEDIA | PRIORIDAD_3_MEDIA | P3_PROGRAMADA | REDUCCION_DE_BRECHAS | MEDIANO_PLAZO | 45,49 |
| 6 | AMBA-E006 | 20 | ESCENARIO_DE_NECESIDAD | DEFICIT | PRIORIDAD_4_MEDIA_BAJA | PRIORIDAD_3_MEDIA | P3_PROGRAMADA | REDUCCION_DE_BRECHAS | MEDIANO_PLAZO | 41,62 |
| 7 | AMBA-E007 | 21 | ESCENARIO_DE_NECESIDAD | CONECTIVIDAD | PRIORIDAD_4_MEDIA_BAJA | PRIORIDAD_4_MEDIA_BAJA | P4_SEGUIMIENTO | CONECTIVIDAD_METROPOLITANA | MEDIANO_LARGO_PLAZO | 27,56 |

---

# 8. Indicadores globales AMBA

| indicador | valor | unidad |
| --- | --- | --- |
| proyectos_totales | 144,00 | proyectos |
| proyectos_unicos | 144,00 | proyectos |
| escenarios_totales | 7,00 | escenarios |
| cobertura_geometrica | 100,00 | % |
| geometrias_validas | 144,00 | geometrías |
| proyectos_multiescenario | 0,00 | proyectos |
| demanda_promedio | 50,35 | índice |
| demanda_maximo | 100,00 | índice |
| deficit_promedio | 68,84 | índice |
| deficit_maximo | 100,00 | índice |
| conectividad_promedio | 57,38 | índice |
| conectividad_maximo | 87,15 | índice |
| intermodalidad_promedio | 39,15 | índice |
| intermodalidad_maximo | 98,18 | índice |
| integracion_promedio | 50,35 | índice |
| integracion_maximo | 79,98 | índice |
| centralidad_promedio | 44,37 | índice |
| centralidad_maximo | 93,74 | índice |
| impacto_promedio | 51,38 | índice |
| impacto_maximo | 86,81 | índice |
| urgencia_promedio | 58,72 | índice |
| urgencia_maximo | 85,07 | índice |

---

# 9. Escenario prioritario


El escenario ubicado en la primera posición del ranking es:

**AMBA-E001**

Este escenario constituye la principal referencia para la programación
territorial derivada del modelo consolidado.


---

# 10. Trazabilidad del modelo

La cadena de procesamiento utilizada para este informe es:

1. Construcción y validación de indicadores territoriales.
2. Construcción de escenarios.
3. Priorización territorial.
4. Construcción de cartera.
5. Validación geoespacial.
6. Integración territorial.
7. Consolidación del modelo maestro.
8. Generación del presente informe.

El proceso 39 no recalcula ni modifica los indicadores originales.

Su función es consolidar, validar, documentar y presentar los resultados
producidos previamente.

---

# 11. Dictamen final

## VALIDADO

El modelo territorial AMBA V4.1 presenta:

- integridad de identificadores;
- consistencia proyecto → escenario;
- cobertura geométrica completa;
- geometrías válidas;
- estructura territorial consistente;
- ranking final disponible;
- matriz integral disponible;
- modelo geográfico consolidado disponible.

El modelo queda preparado para las siguientes etapas de:

- programación de inversiones;
- definición de cronogramas;
- análisis de cartera;
- evaluación territorial;
- elaboración de documentación institucional;
- presentación final del modelo AMBA.

---

**Fin del Informe Territorial AMBA V4.1.**
