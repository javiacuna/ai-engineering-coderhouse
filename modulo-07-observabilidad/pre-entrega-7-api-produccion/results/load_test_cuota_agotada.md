# Prueba de carga · 5 peticiones concurrentes · 2026-11-16T23:10:51+00:00

POST /tasks: 119.4 ms la más lenta (el API solo encola). Estados finales: {'DONE': 5}.

| Job | Estado | Flujo | Espera en cola | Ejecución | Total | Tokens (in/out) | Costo (Phoenix) |
|---|---|---|---|---|---|---|---|
| `3d2ba9fc` | DONE | researcher → analyst → synthesizer → FINISH | 0.03 s | 309.72 s | 309.75 s | 6866/1522 | US$ 0.00587 |
| `bf9812f8` | DONE | researcher → analyst → synthesizer → FINISH | 0.04 s | 285.89 s | 285.95 s | 5470/1024 | US$ 0.00420 |
| `86d35c60` | DONE | researcher → analyst → synthesizer → FINISH | 0.05 s | 436.72 s | 436.78 s | 5327/1026 | US$ 0.00416 |
| `dddc46ac` | DONE | researcher → analyst → synthesizer → FINISH | 0.05 s | 366.29 s | 366.35 s | 6745/1554 | US$ 0.00591 |
| `76250df3` | DONE | researcher → analyst → synthesizer → FINISH | 0.05 s | 354.71 s | 354.78 s | 5466/1198 | US$ 0.00525 |

## Latencia

| Medición | p50 | p95 |
|---|---|---|
| Total del job, medido por la API (cola + ejecución) | 354.78 s | 422.69 s |
| Ejecución del grafo (run_seconds) | 354.71 s | 422.63 s |
| Trazas en Phoenix (latencia de cada ejecución) | 354.7 s | 422.62 s |

Costo total de la corrida según Phoenix: US$ 0.02539 · promedio por ejecución: US$ 0.00508
