# Módulo 5 · Agentes con LangGraph

## Ejercicio: Diseño de un grafo de soporte técnico multinivel

📁 [`ejercicio-diseno-grafo/`](ejercicio-diseno-grafo/)

Diagrama y esquema del estado (`TypedDict`) de un asistente de soporte. Clasifica la duda (Triage), busca una solución (Search_Knowledge), la valida (Evaluator) y, si no sirve, vuelve a buscar: ese es el **ciclo**. Después de 3 intentos fallidos, o si el caso es crítico, lo escala a una persona (Human_Support). Incluye una implementación ejecutable con LangGraph que recorre los tres caminos posibles.

![Diagrama del grafo](ejercicio-diseno-grafo/diagrama.png)
