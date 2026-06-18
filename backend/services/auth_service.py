from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from backend.models.site import User
from backend.repositories import user_repo
from backend.security.security import Token, create_access_token, get_password_hash, verify_password


def login(db: Session, username: str, password: str) -> Token:
    user = user_repo.get_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(data={"sub": user.username})
    return Token(access_token=token, token_type="bearer")


def register(db: Session, username: str, email: str, password: str) -> User:
    if user_repo.get_by_username(db, username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
    if user_repo.get_by_email(db, email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
    return user_repo.create(db, username, email, get_password_hash(password))
