# Arquitectura de la plataforma Nimbus Envíos

> Documento interno del equipo de ingeniería. Nimbus Envíos es una empresa ficticia de envíos de última milla creada para este ejercicio.

## Visión general

Nimbus Envíos conecta comercios online con repartidores independientes para entregar paquetes en el mismo día. La plataforma recibe pedidos desde la API pública y desde el panel web de comercios, calcula la ruta óptima para cada repartidor y permite seguir el paquete en tiempo real hasta la entrega.

La plataforma está formada por cinco servicios principales que corren en Kubernetes. Cada servicio tiene su propio repositorio, su propio pipeline de despliegue y un equipo dueño responsable de su operación.

## Servicios

**gateway**: es la puerta de entrada de todo el tráfico externo. Valida los tokens de los comercios, aplica límites de tasa (600 solicitudes por minuto por comercio) y enruta cada solicitud al servicio correspondiente. Lo mantiene el equipo de Plataforma.

**orders-service**: gestiona el ciclo de vida de los pedidos, desde la creación hasta la entrega. Está escrito en Python 3.12 con FastAPI y guarda los datos en PostgreSQL 16, en la base `orders`. Es el servicio más crítico: si no funciona, ningún comercio puede crear envíos. Lo mantiene el equipo de Pedidos.

**routing-worker**: calcula las rutas de los repartidores. Está escrito en Go y consume mensajes de la cola `route.requests` de RabbitMQ. Procesa hasta 1.200 rutas por minuto; por encima de ese volumen los mensajes se acumulan en la cola y las asignaciones se demoran. Lo mantiene el equipo de Logística.

**tracking-service**: recibe la posición GPS de los repartidores cada 10 segundos. Está escrito en Node.js. La última posición de cada repartidor se guarda en Redis con un TTL de 15 minutos y el historial completo se guarda en TimescaleDB durante 90 días. Lo mantiene el equipo de Logística.

**notifications-service**: envía los avisos al destinatario del paquete por SMS, WhatsApp y correo electrónico (pedido despachado, repartidor cerca, paquete entregado). Lo mantiene el equipo de Pedidos.

## Comunicación entre servicios

Las consultas que necesitan una respuesta inmediata se hacen por REST interno dentro del clúster. Todo lo demás se comunica de forma asíncrona mediante eventos publicados en RabbitMQ. Los eventos principales son `order.created`, `order.dispatched` y `order.delivered`. Por ejemplo, cuando orders-service publica `order.created`, routing-worker calcula la ruta y notifications-service avisa al destinatario.

Todos los eventos llevan un `correlation_id` que se propaga entre servicios. Ese identificador permite seguir un pedido en los logs de todos los servicios a la vez.

## Infraestructura

La plataforma corre en Amazon EKS, en la región sa-east-1 (São Paulo). Hay tres entornos: desarrollo, staging y producción. Staging tiene la misma configuración que producción, pero con datos de destinatarios enmascarados y una sola réplica por servicio.

En producción, orders-service y gateway corren con un mínimo de tres réplicas distribuidas en tres zonas de disponibilidad. El escalado horizontal se activa cuando el uso de CPU supera el 70 % durante dos minutos.

## Objetivos de servicio (SLO)

- Disponibilidad mensual de la API de pedidos: 99,9 %.
- Latencia de creación de un pedido: p95 menor a 300 ms.
- Actualización de la posición en el seguimiento: cada 10 segundos.
- Tiempo máximo para asignar un repartidor a un pedido: 3 minutos.

Si un servicio consume más del 50 % de su presupuesto de error mensual antes del día 15, el equipo dueño congela las funcionalidades nuevas y prioriza el trabajo de confiabilidad hasta recuperarlo.

## Datos y respaldos

PostgreSQL tiene respaldos completos cada 6 horas, con una retención de 30 días. Además, la recuperación a un punto en el tiempo (PITR) permite restaurar la base a cualquier minuto de los últimos 7 días. Una vez por trimestre, el equipo de Plataforma realiza una prueba de restauración en un entorno aislado y registra cuánto tardó.

## Limitaciones conocidas

En los picos de demanda, como Black Friday, el routing-worker es el cuello de botella de la plataforma. La mitigación actual es escalar manualmente el worker a 12 réplicas el día anterior al evento. Está planificado reemplazar ese escalado manual por un escalado automático basado en el largo de la cola de RabbitMQ.
