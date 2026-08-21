# Auditoría final del paquete territorial AMBA V4.1

**Proceso:** 44
**Versión auditor:** V4.1-FINAL-AUDIT

## Resultado

- Controles: 23
- Controles OK: 23
- Fallas: 0
- Fallas críticas: 0
- Fallas importantes: 0
- Score: 100.0/100
- Auditoría: OK
- Dictamen: **GO**

## Controles

| Control | Resultado | Crítico | Detalle |
|---|---|---|---|
| Cierre proceso 42 | OK | SI | Dictamen=GO Score=100.0 OK=13 ERROR=0 DESCONOCIDOS=0 |
| Paquete proceso 43 | OK | SI | Dictamen=GO Score=100.0 OK=13 ERROR=0 |
| Estructura modelo maestro | OK | SI | Proyectos=144 Escenarios=7 IDs únicos y sin nulos |
| Asignación proyecto-escenario | OK | SI | Nulos=0 Multiescenario=0 Escenarios extra=0 |
| Ranking proyectos completo | OK | SI | Campo=ranking_final_proyecto_v4 Registros=144 |
| Ranking proyectos ordenado | OK | SI | Secuencia 1..144 |
| Ranking escenarios completo | OK | SI | Campo=ranking_integral_v4 Registros=7 |
| Ranking escenarios ordenado | OK | SI | Secuencia 1..7 |
| Tabla ejecutiva proyectos | OK | SI | Registros=144 Esperados=144 ID=proyecto_id |
| Tabla ejecutiva escenarios | OK | SI | Registros=7 Esperados=7 ID=escenario_id |
| Top proyectos | OK | SI | Registros=20 Esperados=20 ID=proyecto_id |
| Ranking ejecutivo escenarios | OK | SI | Registros=7 Esperados=7 ID=escenario_id |
| Indicadores ejecutivos | OK | SI | Registros=35 Esperados=35 |
| Cruce maestro-proyectos ejecutivos | OK | SI | Todos los IDs coinciden. |
| Cruce maestro-escenarios ejecutivos | OK | SI | Todos los IDs coinciden. |
| Síntesis ejecutiva | OK | SI | Caracteres=1047 |
| Informe ejecutivo | OK | SI | Caracteres=3781 |
| Manifiesto paquete 43 | OK | SI | Archivos no encontrados=0 Referencias_lógicas=1 |
| Integridad SHA-256 paquete | OK | SI | Archivos validados=9 Errores=0 |
| Inventario productos obligatorios | OK | SI | Productos=15 Faltantes=0 |
| Completitud paquete final | OK | SI | Todos los directorios y archivos obligatorios existen. |
| Indicadores globales | OK | SI | Indicadores=27 Campo indicador=indicador Campo valor=valor |
| Coherencia numérica | OK | NO | Campos evaluados=12 Anomalías=0 |

## Observaciones

No se detectaron observaciones.

## Criterio de auditoría

El dictamen final es GO únicamente cuando todos los controles obligatorios resultan OK. Las referencias lógicas del manifiesto, como `MODELO`, no se consideran archivos físicos faltantes. El campo `prioridad_territorial_v4` se valida como variable categórica y no como indicador numérico.
