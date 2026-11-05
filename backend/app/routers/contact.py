from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends
from ..database import get_db

router = APIRouter(prefix="/contact", tags=["contact"])

# Simple in-memory rate limit: max 3 submissions per IP per 10 minutes
_rate: dict[str, list[float]] = {}
_WINDOW = 600.0
_MAX = 3


class ContactPayload(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    message: str = Field(min_length=10, max_length=2000)


@router.post("", status_code=201)
def submit_contact(
    payload: ContactPayload,
    request: Request,
    database: Session = Depends(get_db),
) -> dict[str, str]:
    # Rate limiting by client IP
    ip = request.client.host if request.client else "unknown"
    now = datetime.now(timezone.utc).timestamp()
    timestamps = [t for t in _rate.get(ip, []) if now - t < _WINDOW]
    if len(timestamps) >= _MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many contact submissions. Please wait before sending another message.",
        )
    timestamps.append(now)
    _rate[ip] = timestamps

    # Insert into contact_messages
    database.execute(text(
        "INSERT INTO contact_messages (id, name, email, message) "
        "VALUES (:id, :name, :email, :message)"
    ), {
        "id": str(uuid.uuid4()),
        "name": payload.name.strip(),
        "email": str(payload.email).lower(),
        "message": payload.message.strip(),
    })
    database.commit()
    return {"message": "Your message has been received. Thank you."}
