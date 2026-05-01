from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.models.site import User
from backend.schemas import schemas
from backend.security.security import verify_password, create_access_token, Token, get_password_hash
from backend.core.database import save_to_db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(data = {"sub": user.username})
    return Token(access_token=token, token_type="bearer")


@router.post("/register", response_model=schemas.UserResponse)
def register(register_data: schemas.UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == register_data.username).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
    if db.query(User).filter(User.email == register_data.email).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")

    hashed_password = get_password_hash(register_data.password)
    user_in_db = User(username=register_data.username, email=register_data.email, hashed_password=hashed_password)
    save_to_db(db, user_in_db)
    return user_in_db