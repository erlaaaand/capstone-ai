"""
Core dependencies for FastAPI routes.

Provides reusable dependency injection components, such as API key
authentication, to secure endpoints.
"""

import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from core.logger import get_logger

logger = get_logger(__name__)

# Define the API Key header scheme.
# auto_error=False allows us to handle the missing header with a custom exception.
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """Verify the provided API key against the environment configuration.

    Args:
        api_key_header: The API key provided in the X-API-Key header.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 403 Forbidden if the API key is missing or invalid.
    """
    # Note: For production, we load this directly from environment variables
    # rather than settings.py if it's considered highly sensitive, or we can
    # add it to settings.py. Assuming an environment variable `API_KEY`.
    expected_api_key = os.getenv("API_KEY")

    if not expected_api_key:
        logger.warning("API_KEY environment variable is not set. Authentication is disabled (DANGEROUS).")
        # In a strict production environment, you might want to raise an error here
        # to prevent accidental open access. For development/testing, we'll allow it.
        return ""

    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing 'X-API-Key' header.",
        )

    # Use constant-time comparison to prevent timing attacks
    import hmac
    
    # hmac.compare_digest requires bytes or ASCII strings
    if not hmac.compare_digest(api_key_header.encode(), expected_api_key.encode()):
        logger.warning("Failed authentication attempt with invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key.",
        )

    return api_key_header
