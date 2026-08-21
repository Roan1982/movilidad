# Auditoría independiente de cierre del modelo territorial AMBA V4.1

**Proceso:** 45
**Versión:** V4.1-FINAL
**Fecha:** 2026-08-21T17:32:57

## Resultado

- Controles: 29
- Controles OK: 29
- Fallas: 0
- Fallas críticas: 0
- Fallas importantes: 0
- Score: 100.00/100
- Auditoría: **OK**
- Dictamen: **GO**

## Resumen estructural

- Proyectos: 144
- Escenarios: 7
- Proceso 42: GO
- Proceso 43: GO
- Proceso 44: GO

## Controles

| Control | Resultado | Crítico | Detalle |
|---|---|---|---|
| Cierre proceso 42 | OK | SI | Registros=13 OK=13 ERROR=0 DESCONOCIDOS=0 Dictamen=GO Score=None |
| Paquete proceso 43 | OK | SI | Controles=13 OK=13 ERROR=0 DESCONOCIDOS=0 Dictamen=GO Score=100.0 |
| Auditoría proceso 44 | OK | SI | Controles=23 OK=23 ERROR=0 DESCONOCIDOS=0 Dictamen=GO Score=100.0 |
| Inventario y estructura física | OK | SI | Directorios faltantes=0 Archivos obligatorios faltantes=0 Archivos obligatorios vacíos=0 Archivos físicos=16 |
| Carga independiente de productos maestros | OK | SI | Productos cargados=10/10 |
| Estructura modelo maestro | OK | SI | Proyectos=144 Proyectos únicos=144 Proyectos nulos=0 Proyectos duplicados=0 Escenarios=7 Escenarios únicos=7 Escenarios nulos=0 Escenarios duplicados=0 |
| Asignación proyecto-escenario | OK | SI | Escenarios nulos=0 Proyectos multiescenario=0 Escenarios extra=0 |
| Ranking proyectos completo | OK | SI | Campo ranking=ranking_final_proyecto_v4 Registros=144 Esperados=144 |
| Ranking proyectos ordenado | OK | SI | Secuencia 1..144 |
| Ranking escenarios completo | OK | SI | Campo ranking=ranking_integral_v4 Registros=7 Esperados=7 |
| Ranking escenarios ordenado | OK | SI | Secuencia 1..7 |
| Distribución territorial | OK | SI | Escenarios=7 Mínimo=20 Máximo=21 Promedio=20.57 CV=0.0241 |
| Tabla ejecutiva proyectos | OK | SI | Registros=144 Esperados=144 ID=proyecto_id |
| Tabla ejecutiva escenarios | OK | SI | Registros=7 Esperados=7 ID=escenario_id |
| Top proyectos | OK | SI | Registros=20 Esperados=20 ID=proyecto_id |
| Ranking ejecutivo escenarios | OK | SI | Registros=7 Esperados=7 ID=escenario_id |
| Indicadores ejecutivos | OK | SI | Registros=35 Esperados=35 |
| Cruce maestro-proyectos ejecutivos | OK | SI | Faltantes=0 Extras=0 |
| Cruce maestro-escenarios ejecutivos | OK | SI | Faltantes=0 Extras=0 |
| Síntesis ejecutiva | OK | SI | Caracteres=1047 |
| Informe ejecutivo | OK | SI | Caracteres=3781 |
| Manifiesto paquete 43 | OK | SI | Registros=10 Referencias físicas=9 Referencias lógicas=1 Archivos no encontrados=0 |
| Integridad SHA-256 paquete | OK | SI | Archivos SHA-256 validados=9 Errores SHA-256=0 |
| Inventario productos obligatorios | OK | SI | Productos obligatorios=15 Faltantes=0 Vacíos=0 |
| Indicadores globales | OK | SI | Indicadores=27 Campo indicador=indicador Campo valor=valor |
| Coherencia numérica | OK | NO | Campos evaluados=10 Anomalías=0 |
| Correspondencia top proyectos | OK | SI | Registros=20 IDs extra=0 |
| Correspondencia ranking escenarios | OK | SI | Registros=7 IDs extra=0 |
| Completitud paquete final | OK | SI | Directorios faltantes=0 Archivos faltantes=0 |

## Conclusión

La auditoría independiente de cierre no detectó inconsistencias.

El modelo territorial AMBA V4.1 y su paquete ejecutivo superaron los controles estructurales, documentales, de integridad y coherencia definidos para el proceso 45.

**DICTAMEN FINAL: GO**
