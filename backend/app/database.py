from __future__ import annotations

import socket
import sys
from collections.abc import Generator

# Python 3.14 typing compatibility for SQLAlchemy 2.0
if sys.version_info >= (3, 14):
    try:
        import typing
        import sqlalchemy.util.typing as _sut
        _sut.make_union_type = lambda *types: typing.Union[tuple(types)]
    except Exception:
        pass

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .settings import settings


class Base(DeclarativeBase):
    pass



def _resolve_ipv4_hostaddr(database_url: str) -> str | None:
    """Best-effort: resolve the DB host to a single IPv4 address.

    DNS for the DB host (e.g. Neon) returns AAAA and A records; libpq-style
    drivers try IPv6 first, and on networks with a broken IPv6 route every
    connection stalls for tens of seconds. Pinning `hostaddr` to a working
    IPv4 avoids that while keeping `host` for TLS certificate verification.
    Returns None (normal driver behaviour) on any resolution problem.
    """
    try:
        hostname = make_url(database_url).host
        if not hostname:
            return None
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if not infos:
            return None
        addr = infos[0][4][0]
        return addr if isinstance(addr, str) else None
    except Exception:  # best effort only; fall back to normal resolution
        return None


def _engine():
    settings.validate_runtime_secrets()
    connect_args: dict[str, object] = {"connect_timeout": 10}
    hostaddr = _resolve_ipv4_hostaddr(settings.database_url)
    if hostaddr:
        connect_args["hostaddr"] = hostaddr
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=10,
        max_overflow=20,
        pool_timeout=15.0,
        connect_args=connect_args,
    )


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine())


def get_db() -> Generator[Session, None, None]:
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
