import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):

    SECURITY_HEADERS = {
        "Strict-Transport-Security":
            "max-age=31536000; includeSubDomains; preload",
        "X-Content-Type-Options":
            "nosniff",
        "X-Frame-Options":
            "DENY",
        "X-XSS-Protection":
            "1; mode=block",
        "Referrer-Policy":
            "strict-origin-when-cross-origin",
        "Permissions-Policy":
            "camera=(), microphone=(), geolocation=(), payment=()",
        "Content-Security-Policy":
            "default-src 'none'; frame-ancestors 'none'",
        "Cache-Control":
            "no-store, no-cache, must-revalidate, private",
        "Pragma":
            "no-cache",
        "Server":
            "Durian-API",
        "Cross-Origin-Opener-Policy":
            "same-origin",
        "Cross-Origin-Resource-Policy":
            "same-origin",
    }

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)

        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value

        if hasattr(request.state, "request_id"):
            response.headers["X-Request-ID"] = request.state.request_id

        response.headers["X-API-Version"] = settings.APP_VERSION

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    SKIP_PATHS = {"/docs", "/redoc", "/openapi.json", "/favicon.ico"}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        client_ip  = self._get_client_ip(request)
        user_agent = (request.headers.get("user-agent", "")[:100])
        start_time = time.perf_counter()

        logger.info(
            f"[{request_id}] → {request.method} {request.url.path} "
            f"| IP={client_ip} | UA={user_agent}"
        )

        try:
            response   = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            log_fn = logger.warning if response.status_code >= 400 else logger.info
            log_fn(
                f"[{request_id}] ← {response.status_code} "
                f"| {elapsed_ms:.1f}ms "
                f"| {request.method} {request.url.path}"
            )
            return response

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] ✗ UNHANDLED {type(exc).__name__}: {str(exc)} "
                f"| {elapsed_ms:.1f}ms "
                f"| {request.method} {request.url.path}"
            )
            raise

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


class PayloadSizeLimitMiddleware:

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app       = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            content_length_raw = headers.get(b"content-length")

            if content_length_raw:
                try:
                    content_length = int(content_length_raw)
                    if content_length > self.max_bytes:
                        max_mb = self.max_bytes / (1024 * 1024)
                        body   = (
                            f'{{"success":false,"error":"PayloadTooLarge",'
                            f'"detail":"Payload melebihi batas {max_mb:.0f}MB."}}'
                        ).encode()
                        response = Response(
                            content     = body,
                            status_code = 413,
                            media_type  = "application/json",
                        )
                        await response(scope, receive, send)
                        return
                except (ValueError, TypeError):
                    pass

        await self.app(scope, receive, send)


class AuditLogger:

    _audit_logger = get_logger("audit")

    @classmethod
    def auth_success(
        cls,
        request_id: str,
        key_prefix: str,
        key_name:   str,
        client_ip:  str,
        path:       str,
    ) -> None:
        cls._audit_logger.info(
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
        cls._audit_logger.warning(
            f"AUTH_FAILURE | req={request_id} | reason='{reason}' "
            f"| key_hint={key_hint} | ip={client_ip} | path={path}"
        )

    @classmethod
    def rate_limit_exceeded(
        cls,
        request_id:  str,
        identifier:  str,
        limit:       int,
        client_ip:   str,
    ) -> None:
        cls._audit_logger.warning(
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
        cls._audit_logger.warning(
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
        cls._audit_logger.warning(
            f"DEPRECATED_KEY | req={request_id} | key={key_prefix} "
            f"| ip={client_ip} | action=ROTATE_KEY_IMMEDIATELY"
        )