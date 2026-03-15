from sqlalchemy import create_engine, text
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_URI = f"""postgresql://{config["database"]["user"]}:
            {config["database"]["password"]}
            @localhost:5432/forecast_db"""
engine = create_engine(DB_URI)


def get_forecast(store: int, dept: int, week: str):
    with engine.connect() as connection:
        query = text(
            """SELECT predicted_sales 
            FROM forecasts 
            WHERE store = :store 
            AND dept = :dept 
            AND date = :date"""
        )
        result = connection.execute(
            query, {"store": store, "dept": dept, "date": week}
        ).fetchone()
        return result[0] if result else None
