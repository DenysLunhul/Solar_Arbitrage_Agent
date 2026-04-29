from fastapi import FastAPI, Depends
from backend.models import site
from backend.core.database import engine, Base


Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
def home_page():
    return {"message": "Hello World!"}