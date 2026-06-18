import pandas as pd
from sqlalchemy.orm import Session
from backend.models.site import History


def upload_history_to_db(db: Session, df: pd.DataFrame):
    if df.empty or df.isna().any().any():
        raise Exception("Not valid dataframe!")
    first_ts = df.iloc[0]["timestamp"]
    exists = db.query(History).filter(History.timestamp == first_ts).first()
    if exists:
        print("Data already exists in db, skipped uploading")
        return
    records = []
    for _, row in df.iterrows():
        records.append(
            History(
                timestamp=row['timestamp'],
                data=row.drop(labels=['timestamp']).to_dict()
            )
        )
    db.bulk_save_objects(records)
    db.commit()
    print("History data uploaded to db successfully!")
