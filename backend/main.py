from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from backend.models import site
from backend.core.database import engine, Base
from backend.routers import config, auth, predictions, strategy

Base.metadata.create_all(bind=engine)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(config.router)
app.include_router(predictions.router)
app.include_router(strategy.router)

@app.get("/")
def home_page():
    return {"message": "Welcome page!"}