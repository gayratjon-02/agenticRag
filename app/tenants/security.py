import hashlib
import secrets

_API_KEY_PREFIX = "rag_"
_API_KEY_BYTES = 32


def generate_api_key() -> str:
    """Create a new high-entropy API key. The plaintext is returned only once."""
    return _API_KEY_PREFIX + secrets.token_urlsafe(_API_KEY_BYTES)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage/lookup. Keys are high-entropy, so SHA-256 is enough."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
