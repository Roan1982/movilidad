# Auditoría 47 — Verificación final

**Proyecto:** Modelo Territorial AMBA
**Versión:** V4.1
**Proceso:** 47

## Artefacto

- ZIP: `MODELO_TERRITORIAL_AMBA_V4_FINAL.zip`
- Tamaño: 615,200 bytes
- SHA-256: `a059736db56c4eafcf95fb1cf8e2cdfd4bd97a568538517dc6b6e194dfa706fb`
- Archivos ZIP: 40

## Resultado

- Controles OK: 11/13
- Controles fallidos: 2
- Fallas críticas: 2
- Score: 84.62/100
- Auditoría: **OBSERVADA**
- Dictamen final: **NO-GO**

## Controles

| Control | Resultado |
|---|---|
| ZIP íntegro | OK |
| Estructura | OK |
| Obligatorios | OK |
| No vacíos | OK |
| Hashes internos | OK |
| MANIFIESTO | ERROR |
| README | ERROR |
| Procesos 42-45 | OK |
| Directorio-ZIP | OK |
| SHA directorio-ZIP | OK |
| Manifiesto CSV | OK |
| Metadata | OK |
| Modelo | OK |

## Procesos 42-45

- Proceso 42: **OK**
- Proceso 43: **OK**
- Proceso 44: **OK**
- Proceso 45: **OK**

## Fallas críticas

- MANIFIESTO.md inválido
- README.md inválido

## Conclusión

El artefacto presenta controles críticos sin conformidad y no puede considerarse GO.
