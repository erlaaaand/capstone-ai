# core/audit.py
# Structured audit logging untuk security events.
# Dipisah dari core/middleware.py karena tidak ada hubungannya dengan ASGI middleware.

from core.logger import get_logger

_audit_logger = get_logger("audit")


class AuditLogger:
    """Log security-relevant events ke named logger 'audit'."""

    @classmethod
    def auth_success(
        cls,
        request_id: str,
        key_prefix: str,
        key_name:   str,
        client_ip:  str,
        path:       str,
    ) -> None:
        _audit_logger.info(
            f"AUTH_SUCCESS | req={request_id} | key={key_prefix} "
            f"| name='{key_name}' | ip={client_ip} | path={path}"
        )

    @classmethod
    def auth_failure(
        cls,
        request_id: str,
        reason:     str,
        client_ip:  str,
        path:       str,
        key_hint:   str = "",
    ) -> None:
        _audit_logger.warning(
            f"AUTH_FAILURE | req={request_id} | reason='{reason}' "
            f"| key_hint={key_hint} | ip={client_ip} | path={path}"
        )

    @classmethod
    def rate_limit_exceeded(
        cls,
        request_id: str,
        identifier: str,
        limit:      int,
        client_ip:  str,
    ) -> None:
        _audit_logger.warning(
            f"RATE_LIMIT_EXCEEDED | req={request_id} | id={identifier} "
            f"| limit={limit}/min | ip={client_ip}"
        )

    @classmethod
    def suspicious_file(
        cls,
        request_id: str,
        reason:     str,
        filename:   str,
        client_ip:  str,
    ) -> None:
        _audit_logger.warning(
            f"SUSPICIOUS_FILE | req={request_id} | reason='{reason}' "
            f"| file='{filename}' | ip={client_ip}"
        )

    @classmethod
    def deprecated_key_used(
        cls,
        request_id: str,
        key_prefix: str,
        client_ip:  str,
    ) -> None:
        _audit_logger.warning(
            f"DEPRECATED_KEY | req={request_id} | key={key_prefix} "
            f"| ip={client_ip} | action=ROTATE_KEY_IMMEDIATELY"
        )