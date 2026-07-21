"""Bounded HTTP downloads that cannot target local or reserved networks."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import requests


MAX_REDIRECTS = 5


class UnsafeURL(ValueError):
    """Raised when an outbound URL could reach a non-public destination."""


def validate_public_http_url(url: str) -> None:
    """Require HTTP(S) and a hostname whose every resolved address is public."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeURL("only http(s) URLs are allowed")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise UnsafeURL("URL must contain a hostname and no credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeURL("URL contains an invalid port") from exc
    try:
        answers = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeURL("hostname could not be resolved") from exc
    addresses = {answer[4][0].split("%", 1)[0] for answer in answers}
    if not addresses:
        raise UnsafeURL("hostname did not resolve to an address")
    for address in addresses:
        try:
            resolved = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeURL("hostname resolved to an invalid address") from exc
        if not resolved.is_global:
            raise UnsafeURL(f"hostname resolves to non-public address {resolved}")


def fetch_public_bytes(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Download a bounded public resource, validating every redirect target."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    session = requests.Session()
    session.trust_env = False
    current_url = url
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            validate_public_http_url(current_url)
            response = session.get(
                current_url,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
            try:
                if response.is_redirect or response.is_permanent_redirect:
                    if redirect_count >= MAX_REDIRECTS:
                        raise UnsafeURL("too many redirects")
                    location = response.headers.get("Location", "").strip()
                    if not location:
                        raise UnsafeURL("redirect response has no destination")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        declared_length = int(content_length)
                    except ValueError:
                        declared_length = 0
                    if declared_length > max_bytes:
                        raise ValueError("response exceeds size limit")
                downloaded = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    downloaded.extend(chunk)
                    if len(downloaded) > max_bytes:
                        raise ValueError("response exceeds size limit")
                return bytes(downloaded)
            finally:
                response.close()
    finally:
        session.close()
    raise UnsafeURL("download did not reach a final response")
