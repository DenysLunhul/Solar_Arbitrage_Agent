from fastapi import FastAPI, Depends
from backend.models import site
from backend.core.database import engine, Base
from backend.routers import config, auth

Base.metadata.create_all(bind=engine)
app = FastAPI()

app.include_router(config.router)
app.include_router(auth.router)

@app.get("/")
def home_page():
    return {"message": "Welcome page!"}