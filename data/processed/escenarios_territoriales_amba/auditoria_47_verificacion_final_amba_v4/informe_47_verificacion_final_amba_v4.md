# Auditoría 47 — Verificación final del artefacto ZIP

## Modelo Territorial AMBA V4.1

- Proyecto: Modelo Territorial AMBA
- Versión: V4.1
- Proceso: 47
- ZIP: `MODELO_TERRITORIAL_AMBA_V4_FINAL.zip`
- SHA-256 ZIP: `a059736db56c4eafcf95fb1cf8e2cdfd4bd97a568538517dc6b6e194dfa706fb`

## Procesos previos

| Proceso | Dictamen |
|---:|:---|
| 42 | GO |
| 43 | GO |
| 44 | GO |
| 45 | GO |

## Resultado de controles

| Control | Resultado | Descripción |
|---|:---:|---|
| 01 | OK | Existencia del ZIP definitivo |
| 02 | OK | Integridad física del ZIP |
| 03 | OK | Inventario independiente del ZIP |
| 04 | ERROR | Estructura definitiva |
| 05 | OK | Archivos físicos no vacíos |
| 06 | OK | Productos obligatorios |
| 07 | OK | SHA-256 del ZIP definitivo |
| 08 | OK | SHA-256 de archivos internos |
| 09 | OK | MANIFIESTO.md |
| 10 | ERROR | README.md |
| 11_42 | OK | Evidencia proceso 42 |
| 11_43 | OK | Evidencia proceso 43 |
| 11_44 | OK | Evidencia proceso 44 |
| 11_45 | OK | Evidencia proceso 45 |
| 12 | OK | Resultado independiente proceso 45 |
| 13 | ERROR | Correspondencia directorio definitivo versus ZIP |
| 14 | OK | Equivalencia SHA-256 directorio versus ZIP |
| 15 | OK | Manifiesto CSV |
| 16 | ERROR | Metadata del paquete |
| 17 | OK | Consistencia básica del modelo |

## Dictamen

- Controles OK: 16/20
- Controles fallidos: 4
- Fallas críticas: 4
- Fallas importantes: 0
- Score: 80.00/100
- Auditoría: **OBSERVADA**
- Dictamen final: **NO-GO**

## Cierre

La auditoría 47 verifica de forma independiente la existencia, integridad, estructura, contenido, hashes y evidencia de cierre del artefacto ZIP definitivo del Modelo Territorial AMBA V4.1.
