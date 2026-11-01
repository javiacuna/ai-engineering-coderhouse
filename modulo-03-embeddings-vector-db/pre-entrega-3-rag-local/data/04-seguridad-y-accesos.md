# Seguridad y accesos en Nimbus Envíos

> Documento interno del equipo de ingeniería. Nimbus Envíos es una empresa ficticia creada para este ejercicio.

## Gestión de secretos

Las contraseñas, tokens y claves de API nunca se guardan en el código ni en los repositorios. Todos los secretos viven en AWS Secrets Manager, y los servicios los leen al iniciar mediante el controlador External Secrets de Kubernetes.

Todos los repositorios tienen un hook de pre-commit con gitleaks, que bloquea el commit si detecta algo que parece un secreto. El pipeline de integración continua vuelve a ejecutar el mismo análisis, por si alguien saltó el hook.

Los secretos se rotan cada 90 días. Si un secreto se filtra (por ejemplo, si se subió a un repositorio o apareció en un log), hay que rotarlo dentro de la primera hora y declarar un incidente SEV2. Si el secreto filtrado da acceso a datos de destinatarios, el incidente es SEV1.

## Acceso a producción

Aplicamos el principio de mínimo privilegio: nadie tiene acceso permanente a producción, ni siquiera las personas líderes de equipo.

El acceso se pide "justo a tiempo" en el canal `#access-requests`, indicando el motivo y el servicio. Lo aprueba la persona líder del equipo dueño del servicio, y el acceso dura 4 horas; después se revoca automáticamente. Durante una guardia, la guardia primaria puede aprobarse a sí misma un acceso de emergencia, pero tiene que justificarlo en el canal del incidente.

Para conectarse a cualquier entorno hace falta estar en la VPN corporativa (WireGuard) y usar autenticación multifactor. Todas las sesiones en producción quedan registradas y se auditan una vez por mes.

## Datos personales de destinatarios

El nombre, la dirección y el teléfono de los destinatarios son datos personales y se tratan como información sensible:

- En staging y en desarrollo se usan siempre datos enmascarados; está prohibido copiar datos reales de producción a otro entorno.
- Los logs nunca deben incluir teléfonos ni direcciones completas. La librería interna de logging enmascara esos campos automáticamente, y no se puede desactivar.
- Las direcciones de entrega se conservan durante 18 meses y después se anonimizan.
- Solo el equipo de Soporte al Cliente puede ver los datos completos de un destinatario, y únicamente desde el panel de soporte, que registra cada consulta.

## Dependencias y vulnerabilidades

Dependabot revisa las dependencias de todos los repositorios una vez por semana y abre pull requests con las actualizaciones. Las imágenes de contenedor se analizan con Trivy en cada build.

Los plazos para corregir una vulnerabilidad dependen de su gravedad:

- Crítica: 7 días.
- Alta: 30 días.
- Media y baja: se planifican dentro del trimestre.

Si una vulnerabilidad crítica se puede explotar desde internet, se trata como un incidente SEV1 y se corrige en el día.

## Altas y bajas de personas

Cuando alguien se incorpora, recibe acceso solo a los repositorios y herramientas de su equipo; el resto se pide a medida que hace falta. Cuando alguien deja la empresa, todos sus accesos se revocan el mismo día, incluidas las claves de la VPN y los tokens personales. El equipo de Plataforma revisa cada trimestre que no queden cuentas sin dueño.

## Reportar un problema de seguridad

Cualquier persona que detecte un posible problema de seguridad debe reportarlo en el canal privado `#security` o por correo a la dirección del equipo de seguridad, aunque no esté segura de que sea un problema real. Es preferible un falso positivo a un problema que nadie reportó.
