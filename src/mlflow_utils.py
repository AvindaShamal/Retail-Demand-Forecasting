import os
from pathlib import Path
import socket
from urllib.parse import urlparse, urlunparse


def _running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def _resolve_host(hostname: str) -> str:
    try:
        return socket.gethostbyname(hostname)
    except OSError:
        return hostname


def resolve_tracking_uri(tracking_uri: str | None) -> str | None:
    if not tracking_uri:
        return tracking_uri

    parsed = urlparse(tracking_uri)
    if not _running_in_docker():
        return tracking_uri

    if parsed.scheme not in {"http", "https"}:
        return tracking_uri

    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        return tracking_uri

    docker_host = os.getenv("MLFLOW_DOCKER_HOST", "host.docker.internal")
    docker_host = _resolve_host(docker_host)
    credentials = ""
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials = f"{credentials}:{parsed.password}"
        credentials = f"{credentials}@"

    port = f":{parsed.port}" if parsed.port else ""
    return urlunparse(parsed._replace(netloc=f"{credentials}{docker_host}{port}"))


def get_tracking_uri(default_tracking_uri: str | None = None) -> str | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    return resolve_tracking_uri(tracking_uri)
