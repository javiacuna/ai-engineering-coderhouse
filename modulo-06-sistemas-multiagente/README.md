# Módulo 6 · Sistemas multi-agente

## Ejercicio: Topología jerárquica para respuesta automatizada a quejas

📁 [`ejercicio-topologia-jerarquica/`](ejercicio-topologia-jerarquica/)

Propuesta arquitectónica de un sistema que recibe quejas de clientes por Twitter, analiza el sentimiento, verifica el pedido en la base de datos y responde de forma personalizada. Lo resuelven un **supervisor** y cuatro especialistas: Sentimiento, Pedidos, Redacción y Cumplimiento. El documento incluye:

- las responsabilidades, herramientas y contratos de cada agente;
- el estado compartido;
- el flujo paso a paso, con un ejemplo;
- la justificación de la topología jerárquica frente a la cooperativa, y cómo se mitigan sus riesgos.

![Diagrama](ejercicio-topologia-jerarquica/diagrama.png)
