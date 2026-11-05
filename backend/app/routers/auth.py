from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import SessionRecord, User
from ..security import hash_password, hash_session_token, new_session_token, verify_password
from ..settings import settings

router = APIRouter(prefix="/auth", tags=["authentication"])


class RegisterPayload(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)


class LoginPayload(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


def _set_session(response: Response, database: Session, user_id: object) -> None:
    token = new_session_token()
    database.add(SessionRecord(
        id=uuid.uuid4(),
        user_id=user_id,
        token_hash=hash_session_token(token),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    ))
    response.set_cookie(
        "session_token", token, httponly=True,
        secure=settings.environment == "production", samesite="lax", max_age=604800,
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterPayload, response: Response, database: Session = Depends(get_db)) -> dict[str, str]:
    email = str(payload.email).lower()
    if database.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in instead.")
    user = User(id=uuid.uuid4(), name=payload.name.strip(), email=email, password_hash=hash_password(payload.password))
    database.add(user)
    database.flush()
    _set_session(response, database, user.id)
    database.commit()
    return {"message": "Account created.", "user_id": str(user.id), "name": user.name, "email": user.email}


@router.post("/login")
def login(payload: LoginPayload, response: Response, database: Session = Depends(get_db)) -> dict[str, str]:
    user = database.scalar(select(User).where(User.email == str(payload.email).lower()))
    if user is None or not verify_password(user.password_hash, payload.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    _set_session(response, database, user.id)
    database.commit()
    return {"message": "Signed in.", "user_id": str(user.id), "name": user.name, "email": user.email}


@router.post("/logout")
def logout(
    response: Response,
    session_token: str | None = Cookie(default=None),
    database: Session = Depends(get_db),
) -> dict[str, str]:
    if session_token:
        record = database.scalar(select(SessionRecord).where(
            SessionRecord.token_hash == hash_session_token(session_token),
            SessionRecord.revoked_at.is_(None),
        ))
        if record:
            record.revoked_at = datetime.now(timezone.utc)
            database.commit()
    response.delete_cookie("session_token")
    return {"message": "Signed out."}


@router.get("/me")
def me(
    session_token: str | None = Cookie(default=None),
    database: Session = Depends(get_db),
) -> dict[str, object]:
    if not session_token:
        raise HTTPException(status_code=401, detail="Not signed in.")
    record = database.scalar(select(SessionRecord).where(
        SessionRecord.token_hash == hash_session_token(session_token),
        SessionRecord.revoked_at.is_(None),
        SessionRecord.expires_at > datetime.now(timezone.utc),
    ))
    if record is None:
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    user = database.get(User, record.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Account not found.")
    return {"id": str(user.id), "name": user.name, "email": user.email}
