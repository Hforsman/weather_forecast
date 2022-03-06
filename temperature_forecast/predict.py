import datetime
from datetime import timedelta
from prefect import task, Flow
from prefect.schedules import IntervalSchedule
import pandas as pd
import joblib
from utils import get_historical_temperature, pickle_path, store_latest_data, store_prediction


@task(log_stdout=True)
def predict_latest():
    """
    In order to create the model input features, 24 hours of historical temperatures are needed.
    For that reason this method takes 2 days of API history and filters to the last 24 records.
    This script only prints the prediction (an array of length 1) and does not implement any storage.
    """

    # Fet input for prediction
    prediction_range = pd.date_range(end=pd.Timestamp.now().date(), periods=2, freq="d")
    df_pred = get_historical_temperature(prediction_range).iloc[-25:]

    # Store latest data to database
    store_latest_data(df_pred)

    # Load model and predict temperature next hour
    pipeline = joblib.load(pickle_path)
    prediction = pipeline.predict(df_pred)

    # Store the prediction
    store_prediction(df_pred, prediction)

    print(prediction)


# Schedule object for prefect
schedule = IntervalSchedule(
    start_date=datetime.datetime.utcnow() + timedelta(seconds=5),
    interval=timedelta(hours=1),
)


with Flow("Predict weather", schedule=schedule) as flow:
    prediction = predict_latest()


if __name__ == "__main__":
    # execute the flow
    flow.run()
