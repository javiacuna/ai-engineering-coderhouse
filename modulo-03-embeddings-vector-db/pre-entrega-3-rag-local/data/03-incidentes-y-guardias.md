# Gestión de incidentes y guardias en Nimbus Envíos

> Documento interno del equipo de ingeniería. Nimbus Envíos es una empresa ficticia creada para este ejercicio.

## Niveles de severidad

Todo incidente se clasifica en uno de tres niveles. Ante la duda, se elige el nivel más alto: bajar la severidad después es fácil, pero reaccionar tarde no.

**SEV1 (crítico)**: la plataforma no funciona para la mayoría de los comercios, hay pérdida de datos o hay un problema de seguridad confirmado. Ejemplos: no se pueden crear pedidos, el gateway rechaza todo el tráfico o se filtraron datos de destinatarios.

**SEV2 (alto)**: una funcionalidad importante está degradada o falla para una parte de los usuarios, por ejemplo más del 5 % de errores en un servicio, o asignaciones de repartidores que tardan más de 10 minutos.

**SEV3 (menor)**: un problema con impacto limitado o con una solución alternativa disponible. Ejemplo: un tipo de notificación por correo que no se envía, mientras el SMS sigue funcionando.

## Guardias

Cada equipo dueño de un servicio tiene una rotación de guardia semanal. El turno cambia los lunes a las 10:00 y siempre hay dos personas: una guardia primaria y una guardia secundaria. Las alertas llegan por PagerDuty a la guardia primaria.

Después de una semana de guardia con incidentes fuera del horario laboral, la persona tiene un día libre compensatorio, que debe tomar dentro de los 15 días siguientes.

## Tiempos de respuesta

| Severidad | Acuse de la alerta | Escalamiento si no hay respuesta | Actualizaciones de estado |
|---|---|---|---|
| SEV1 | 5 minutos | A la guardia secundaria a los 10 minutos | Cada 30 minutos |
| SEV2 | 15 minutos | A la guardia secundaria a los 30 minutos | Cada 2 horas |
| SEV3 | Siguiente día hábil | No aplica | No aplica |

## Qué hacer ante un SEV1

1. La guardia primaria reconoce la alerta en PagerDuty dentro de los 5 minutos.
2. Abre un canal de Slack con el formato `#inc-AAAAMMDD-descripcion`, por ejemplo `#inc-20261124-pedidos-caidos`.
3. Se designa una persona como Incident Commander. Coordina la respuesta y decide, pero no tiene que arreglar el problema: su trabajo es que el equipo no pierda el foco.
4. Se avisa a la Dirección de Ingeniería y al equipo de Soporte al Cliente, para que Soporte pueda informar a los comercios afectados.
5. Se publica una actualización de estado en el canal cada 30 minutos, aunque no haya novedades.
6. Si el incidente lo causó un despliegue reciente, la primera acción es el rollback (ver la guía de despliegues).

Un SEV2 sigue los mismos pasos, salvo el aviso a la Dirección de Ingeniería, que solo es necesario si el incidente dura más de 2 horas.

## Cierre y postmortem

Un incidente se cierra cuando el servicio volvió a funcionar normalmente durante al menos 30 minutos. El Incident Commander anuncia el cierre en el canal del incidente.

Los incidentes SEV1 y SEV2 requieren un postmortem escrito dentro de los 5 días hábiles posteriores al cierre. El postmortem es sin culpables (blameless): describe qué pasó, la línea de tiempo, el impacto, la causa raíz y qué se va a hacer para que no se repita. Cada acción de mejora tiene una persona responsable y una fecha límite, y se sigue en la reunión semanal de confiabilidad.

Para los SEV3 alcanza con una nota breve en el ticket del problema.

## Métricas de la gestión de incidentes

Todos los meses, el equipo de Plataforma publica el tiempo medio hasta el acuse (MTTA), el tiempo medio de resolución (MTTR) y la cantidad de incidentes por severidad y por servicio. El objetivo vigente es mantener el MTTR de los SEV1 por debajo de 60 minutos.
