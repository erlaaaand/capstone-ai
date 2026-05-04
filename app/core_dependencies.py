# app/core_dependencies.py

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from core.middleware import AuditLogger
from core.rate_limiter import build_rate_limit_headers, get_rate_limiter
from core.security import AuthResult, KeyScope, get_key_manager
from core.logger import get_logger

logger = get_logger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key",      auto_error=False)
BEARER_HEADER  = APIKeyHeader(name="Authorization",  auto_error=False)


def _extract_key(api_key_header: str | None, auth_header: str | None) -> str | None:
    if api_key_header and api_key_header.strip():
        return api_key_header.strip()

    if auth_header and auth_header.strip():
        h = auth_header.strip()
        return h[7:].strip() if h.lower().startswith("bearer ") else h

    return None


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def _check_rate_limit(request: Request, auth_result: AuthResult) -> dict:
    limiter    = get_rate_limiter()
    manager    = get_key_manager()
    client_ip  = _get_client_ip(request)
    request_id = _get_request_id(request)

    if auth_result.valid:
        identifier = f"key:{auth_result.key_prefix}"
        limit      = manager.get_tier_limit(auth_result.tier)
    else:
        identifier = f"ip:{client_ip}"
        limit      = 30

    result = await limiter.check(identifier, limit)

    if not result.allowed:
        AuditLogger.rate_limit_exceeded(request_id, identifier, limit, client_ip)
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            detail      = result.reason,
            headers     = build_rate_limit_headers(result),
        )

    return build_rate_limit_headers(result)


async def verify_api_key(
    request:        Request,
    api_key_header: str | None = Security(API_KEY_HEADER),
    auth_header:    str | None = Security(BEARER_HEADER),
) -> AuthResult:
    client_ip  = _get_client_ip(request)
    request_id = _get_request_id(request)
    path       = request.url.path

    raw_key = _extract_key(api_key_header, auth_header)

    if not raw_key:
        AuditLogger.auth_failure(request_id, "Key tidak ada", client_ip, path)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "API key diperlukan. Sertakan header 'X-API-Key: dk_live_xxx'.",
            headers     = {"WWW-Authenticate": "ApiKey"},
        )

    manager     = get_key_manager()
    auth_result = await manager.validate_async(raw_key)

    if not auth_result.valid:
        AuditLogger.auth_failure(
            request_id,
            auth_result.error,
            client_ip,
            path,
            key_hint=raw_key[:8] + "..." if len(raw_key) >= 8 else "***",
        )

        if "banyak percobaan" in auth_result.error:
            raise HTTPException(
                status_code = status.HTTP_429_TOO_MANY_REQUESTS,
                detail      = auth_result.error,
                headers     = {"Retry-After": "30"},
            )

        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = f"Autentikasi gagal: {auth_result.error}",
        )

    rate_headers = await _check_rate_limit(request, auth_result)

    AuditLogger.auth_success(
        request_id, auth_result.key_prefix, auth_result.key_name, client_ip, path
    )

    if auth_result.deprecated:
        AuditLogger.deprecated_key_used(request_id, auth_result.key_prefix, client_ip)

    request.state.auth         = auth_result
    request.state.rate_headers = rate_headers

    return auth_result


def require_scope(required_scope: KeyScope):
    async def _scope_check(
        request: Request,
        auth:    AuthResult = Security(verify_api_key),
    ) -> AuthResult:
        if required_scope not in auth.scopes:
            logger.warning(
                f"[{_get_request_id(request)}] Scope tidak cukup: "
                f"butuh={required_scope.value} "
                f"| punya={[s.value for s in auth.scopes]} "
                f"| key={auth.key_prefix} "
                f"| ip={_get_client_ip(request)}"
            )
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail      = (
                    f"API key ini tidak punya akses scope '{required_scope.value}'. "
                    "Hubungi administrator untuk upgrade key."
                ),
            )
        return auth

    return _scope_check