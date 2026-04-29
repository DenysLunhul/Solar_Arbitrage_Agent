import pandas as pd
from sqlalchemy.orm import Session
from backend.core.database import SessionLocal
from backend.models.site import History

#TODO
def upload_csv_to_history(file_path: str):
    return