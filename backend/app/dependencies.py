from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .models import SessionRecord, User
from .security import hash_session_token


def current_user(
    session_token: str | None = Cookie(default=None), database: Session = Depends(get_db),
) -> User:
    if not session_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required.")
    record = database.scalar(select(SessionRecord).where(
        SessionRecord.token_hash == hash_session_token(session_token),
        SessionRecord.revoked_at.is_(None),
        SessionRecord.expires_at > datetime.now(timezone.utc),
    ))
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is invalid or expired.")
    user = database.get(User, record.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account not found.")
    return user
