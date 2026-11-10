# Análisis de trazas · proyecto `orquestador-m7-20261109-2002`

7 consultas · p50 12.23 s · p95 39.77 s · llamadas al LLM: p50 1.13 s, p95 2.36 s · 74.2% del tiempo en llamadas externas · 24.4% en esperas (p95 sin esperas: 21.37 s)

## Latencia por consulta (externa vs. procesamiento local)

| Consulta | Total | LLM (API) | Fuentes de datos | Procesamiento local | Espera | Span más lento | Llamadas LLM | Tokens (in/out) | Costo est. |
|---|---|---|---|---|---|---|---|---|---|
| q1-comparacion | 12.23 s | 11.92 s | 0.05 s | 0.25 s | 0 s | ChatGoogleGenerativeAI (analyst, 2.83 s) | 9 | 6724/1476 | $0.00571 |
| q2-ranking | 46.81 s | 16.29 s | 0.11 s | 0.35 s | 30.06 s | ChatGoogleGenerativeAI (analyst, 2.55 s) | 12 | 9306/1817 | $0.00733 |
| q3-quejas | 23.35 s | 23.09 s | 0.05 s | 0.2 s | 0 s | ChatGoogleGenerativeAI (synthesizer, 12.56 s) | 9 | 5504/1033 | $0.00423 |
| q4-presupuesto | 11.08 s | 10.83 s | 0.05 s | 0.2 s | 0 s | ChatGoogleGenerativeAI (supervisor, 1.63 s) | 9 | 5276/944 | $0.00394 |
| q5-uso | 12.72 s | 12.34 s | 0.1 s | 0.27 s | 0 s | ChatGoogleGenerativeAI (analyst, 2.25 s) | 10 | 7762/1549 | $0.00620 |
| f1-fuera-de-alcance | 7.42 s | 7.25 s | 0.0 s | 0.17 s | 0 s | ChatGoogleGenerativeAI (researcher, 1.7 s) | 7 | 3346/408 | $0.00202 |
| f2-fuente-caida | 9.42 s | 9.21 s | 0.03 s | 0.17 s | 0 s | ChatGoogleGenerativeAI (researcher, 2.11 s) | 7 | 3002/448 | $0.00202 |

## Groundedness y errores

| Consulta | ¿Grounded? | Números sin fuente | Usó datos de herramientas | Reconoce falta de datos | Errores en spans |
|---|---|---|---|---|---|
| q1-comparacion | sí | — | sí | no | — |
| q2-ranking | sí | — | sí | no | ChatGoogleGenerativeAI (analyst): GoogleRateLimitError("Error calling model 'gemini-flash-lite-latest' ( · propagado a 4 spans superiores |
| q3-quejas | sí | — | sí | no | — |
| q4-presupuesto | sí | — | sí | no | — |
| q5-uso | sí | — | sí | sí | — |
| f1-fuera-de-alcance | sí | — | no | sí | — |
| f2-fuente-caida | sí | — | no | sí | db.reviews.search (researcher): ConnectionError: fuente de reseñas no disponible (simulación de caída); search_reviews (researcher): ConnectionError('fuente de reseñas no disponible (simulación de caída) · propagado a 6 spans superiores |

## Razonamiento más largo: `q2-ranking`

12 llamadas al LLM · 9306 tokens de entrada · 1817 de salida · costo estimado $0.00733

| Agente | Llamadas | Tokens de entrada | Tokens de salida | Segundos en el LLM |
|---|---|---|---|---|
| researcher | 3 | 3172 | 194 | 2.87 |
| supervisor | 4 | 2667 | 335 | 5.06 |
| analyst | 4 | 1804 | 1036 | 6.89 |
| synthesizer | 1 | 1663 | 252 | 1.47 |

Total de la sesión: 40920 tokens de entrada, 7675 de salida, costo estimado $0.03146 (en el plan pago; el plan gratuito no cobra).
