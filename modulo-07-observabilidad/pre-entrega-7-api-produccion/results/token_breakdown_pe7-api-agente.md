# Tokens y costo por agente · proyecto `pe7-api-agente`

76 llamadas al LLM · 36.198 tokens · costo estimado US$ 0.02539 (precios de la tabla de Phoenix)

| Agente | Llamadas | Fallidas (429/503) | Tokens entrada | Tokens salida | % de los tokens | Costo est. | % del costo | Segundos en el LLM | de ellos, en llamadas fallidas |
|---|---|---|---|---|---|---|---|---|---|
| analyst | 10 | 0 | 5.350 | 3.065 | 23 % | US$ 0.00927 | 37 % | 184.1 s | 0.0 s |
| supervisor | 51 | 31 | 12.434 | 1.741 | 39 % | US$ 0.00870 | 34 % | 951.3 s | 740.4 s |
| synthesizer | 5 | 0 | 5.693 | 874 | 18 % | US$ 0.00389 | 15 % | 58.9 s | 0.0 s |
| researcher | 10 | 0 | 6.397 | 644 | 19 % | US$ 0.00353 | 14 % | 196.2 s | 0.0 s |

Llamadas por modelo: `gemini-3.8-flash` 32 (31 fallidas), `gemini-flash-lite-latest` 44 (0 fallidas)
