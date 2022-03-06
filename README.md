# Introduction
This is the solution for _Option 2. Deploying a temperature forecast_ of the data/ml engineering assessment of Sensorfact.

## Architecture
* **Code repository**: The code is stored on Github - https://github.com/Hforsman/weather_forecast
* **Scheduler**: Prefect schedules the predict and train scripts
* **Compute**: The scripts (currently) run locally
* **Storage**: All data is stored in a sqlite3 database
  * historical data: This table contains two columns: 
    * timestamps - hourly datetimes (TEXT)
    * temp - actual temperature (REAL)
  * predictions: Contains three columns:
    * timestamps - hourly datetimes (TEXT)
    * predicted_temp - predicted temperature (REAL)
    * mse - mean squared error over last 24 predictions (REAL)
* **Dashboard**: Streamlit is used to show a simple dashboard with metrics

## Code overview

    ├── temperature_forecast (scripts only needed for assignment option 2)
    │   ├── train.py        <- Script that can be used to retrain the model on the last ~5 days of API data
    │   ├── predict.py      <- Script that makes a prediction using the latest 24 hours of API data
    │   ├── utils.py        <- Helper functions & API parameters
    │   └── pipeline.pkl    <- Stored model pickle to be used for predictions
    ├── app.py              <- Script to populate and show the streamlit dashboard
    ├── assignment_sf.db    <- sqlite3 database containing historical and predicted data

## How to start
1. Clone the repo 
2. install the requirements in a venv
3. In a terminal start the workflow for the predictions: `python temperature_forecast/predict.py`
4. In a second terminal start the workflow for the model training: `python temperature_forecast/train.py`
5. In a third terminal start the dashboard: `streamlit run app.py`