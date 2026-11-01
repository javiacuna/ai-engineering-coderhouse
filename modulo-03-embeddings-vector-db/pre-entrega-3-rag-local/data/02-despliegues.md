# Guía de despliegues de Nimbus Envíos

> Documento interno del equipo de ingeniería. Nimbus Envíos es una empresa ficticia creada para este ejercicio.

## Flujo de trabajo

Usamos desarrollo basado en trunk: las ramas viven poco tiempo y se integran a `main` mediante un pull request. Todo pull request necesita al menos una aprobación. Los cambios en orders-service y en gateway, que son los servicios críticos, necesitan dos aprobaciones, y una de ellas tiene que ser de alguien del equipo dueño del servicio.

Cada pull request dispara el pipeline de integración continua, que ejecuta el linter, los tests unitarios y los tests de contrato entre servicios, y construye la imagen de contenedor. Un pull request no se puede integrar si alguno de esos pasos falla.

## Del merge a producción

Los despliegues se hacen con Argo CD. Cuando un cambio se integra a `main`, se despliega automáticamente en staging. El paso a producción es manual: lo dispara la persona autora del cambio desde Argo CD, después de verificar el cambio en staging.

En producción usamos despliegue canary en tres etapas:

1. El 10 % del tráfico va a la versión nueva durante 15 minutos.
2. Si las métricas están bien, se pasa al 50 % durante otros 15 minutos.
3. Por último, se pasa al 100 %.

Durante cada etapa se observan la tasa de errores 5xx, que debe mantenerse por debajo del 1 %, y la latencia p95 del servicio. Si la tasa de errores 5xx supera el 2 % durante 5 minutos seguidos, Argo CD hace un rollback automático a la versión anterior.

## Ventanas de despliegue

Los despliegues a producción se hacen de lunes a jueves, entre las 10:00 y las 16:00 (hora de Argentina), para que siempre haya gente disponible si algo sale mal.

Los viernes, los fines de semana y los feriados no se despliega a producción. La única excepción son los hotfixes que corrigen un incidente SEV1 o SEV2 en curso; en ese caso hace falta la aprobación de quien esté como Incident Commander del incidente.

Además, hay períodos de congelamiento en los que solo se permiten hotfixes:

- Del 20 de noviembre al 5 de diciembre, por Black Friday y Cyber Monday.
- Del 22 de diciembre al 2 de enero, por las fiestas de fin de año.

## Rollback

Si un despliegue causa problemas, la prioridad es volver a la versión anterior y después investigar. El rollback se hace desde Argo CD con el comando `argocd app rollback <servicio>` o revirtiendo el commit en `main`. El objetivo es completar un rollback en menos de 10 minutos desde que se detecta el problema.

Nunca se corrige "hacia adelante" un problema en producción con un cambio nuevo sin probar: primero rollback, después la corrección por el flujo normal.

## Migraciones de base de datos

Las migraciones de PostgreSQL se escriben con Alembic y se ejecutan como un Job de Kubernetes antes de desplegar la versión nueva del servicio. Seguimos el patrón expand/contract:

- **Expand**: primero se agregan las columnas o tablas nuevas, sin romper la versión que está corriendo.
- **Contract**: las columnas viejas se eliminan en un release posterior, cuando ya ninguna versión del código las usa.

Está prohibido renombrar o borrar una columna en el mismo release que cambia el código que la usa. Las migraciones que bloquean tablas grandes (más de un millón de filas) necesitan la revisión del equipo de Plataforma.

## Feature flags

Las funcionalidades nuevas que cambian el comportamiento para los comercios se publican detrás de un feature flag gestionado en Unleash. Así se pueden activar de forma gradual o desactivar sin un nuevo despliegue.

Los flags temporales se eliminan del código como máximo 30 días después de activarse para el 100 % de los comercios. Cada flag tiene una persona responsable de eliminarlo.
