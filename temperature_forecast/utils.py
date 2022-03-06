from typing import List

import requests
import sqlite3
import datetime
import pandas as pd
import os
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import mean_squared_error


lat = 52.084516
lon = 5.115539
api_key = "ace49a766053c083b15a916b5fed71d9"
pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.pkl")
database_file = os.path.join("..", "assignment_sf.db")


def get_conn():
    """
    This method creates a connection to a sqlite database and then checks whether the database we call is new
    by querying the sqlite_master table. Creating a connection to a non-existing sqlite database creates that database
    without any tables or data in there.

    returns:
        sqlite Connection object
    """
    conn = sqlite3.connect(database_file)

    # query the sqlite_master to determine whether any tables are already there. We pick just one table as we create
    # all if it doesn't exist
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_temperature'")
    rows = c.fetchall()

    # Check if the tables in the database are already created
    if len(rows) == 0:
        # Create tables because the database is empty
        create_historical_temperature = """CREATE TABLE historical_temperature (
                                            timestamp TEXT PRIMARY KEY,
                                            temp REAL NOT NULL
                                            );"""
        create_predictions = """CREATE TABLE predictions (
                                timestamp TEXT PRIMARY KEY,
                                predicted_temp REAL NOT NULL,
                                mse REAL
                                );"""
        try:
            c.execute(create_historical_temperature)
            c.execute(create_predictions)
        except sqlite3.Error as e:
            print(e)

    c.close()

    return conn


def store_latest_data(latest_data: pd.DataFrame):
    """
    This method stores the latest data temperature data that is not yet stored in the database
    """
    conn = get_conn()
    c = conn.cursor()
    get_latest_timestamp = """SELECT max(timestamp) FROM historical_temperature"""
    c.execute(get_latest_timestamp)
    rows = list(c.fetchall())

    # if the query returned an entry then the table already exists, and we only add missing data
    if rows[0][0] is not None:
        max_timestamp = datetime.datetime.fromisoformat(rows[0][0])
        # only store data that is newer than what is currently stored in the database
        latest_data[latest_data.index > max_timestamp].\
            to_sql("historical_temperature", conn, index=True, index_label="timestamp", if_exists="append")
    else:
        # the table did not exist and we add all data to it
        latest_data.to_sql("historical_temperature", conn, index=True, index_label="timestamp", if_exists="append")

    c.close()

    # Since we now have the latest data stored with the previous prediction we immediately compute a new mse
    compute_mse()


def store_prediction(historical_data: pd.DataFrame, prediction: List):
    conn = get_conn()
    c = conn.cursor()
    max_hist_ts = historical_data.index.max()
    predicted_ts = max_hist_ts + datetime.timedelta(hours=1)
    insert_data = f"""INSERT OR REPLACE INTO predictions (timestamp, predicted_temp) 
                      VALUES ('{predicted_ts}', {prediction[0]});"""
    c.execute(insert_data)
    conn.commit()
    c.close()


def compute_mse():
    """
    This method computes and stores the Mean Squared Error over the last 24 predictions
    """
    conn = get_conn()

    # get both predictions and real values
    pred = pd.read_sql("SELECT * FROM predictions", conn)
    temp = pd.read_sql("SELECT * FROM historical_temperature", conn)

    # merge dataframes and fill missing predictions
    data = temp.merge(pred, on="timestamp", how="left").iloc[-25:]
    data.predicted_temp.fillna(value=temp.temp, inplace=True)

    # compute mse over last 24 hours
    mse = mean_squared_error(y_true=data.temp, y_pred=data.predicted_temp)

    # insert mse into database next to prediction
    insert_mse = f"""UPDATE predictions 
                      SET mse = {mse}
                      WHERE timestamp = '{temp.timestamp.max()}';"""
    c = conn.cursor()
    c.execute(insert_mse)
    conn.commit()

    c.close()


def get_historical_temperature(date_range: pd.date_range):
    """
    This method queries the historical API and parses timestamps and temperatures to a DataFrame.

    Args:
        date_range (pandas.date_range): a range of dates to collect weather data for (max 5 days ago)
    returns:
        DataFrame with single temp column, indexed with timestamps
    """
    hourlies = []
    for t in date_range:
        time = int(t.timestamp())
        api_call = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={time}&units=metric&appid={api_key}"
        hourlies.append(requests.get(api_call).json()["hourly"])
    df = (
        pd.concat([pd.DataFrame(hourly) for hourly in hourlies])
        .assign(dt=lambda df: pd.to_datetime(df["dt"], unit="s"))
        .set_index("dt")[["temp"]]
    )
    return df


class SmoothedVarCreator(BaseEstimator, TransformerMixin):
    """
    This transformer allows one to calculate an exponentially weighted moving average on the target temperatures.

    Args:
        var (str): name of the column to create moving average features for
        alpha_list (List[float]): list of alpha values to pass to pandas.series.ewm
    returns:
        DataFrame with ewm columns added
    """

    def __init__(self, var, alpha_list):
        self.var = var
        self.alpha_list = alpha_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for alpha in self.alpha_list:
            func = {
                "%s_sm%s"
                % (self.var, int(alpha * 10)): lambda df: X[self.var]
                .ewm(alpha=alpha, min_periods=0)
                .mean()
            }
            X = X.assign(**func)
        return X


class LagCreator(BaseEstimator, TransformerMixin):
    """
    This transformer allows one to calculate an exponentially weighted moving average on the target temperatures.

    Args:
        var (str): name of the column to create lagged features for
        lag_list (List[float]): list of lags to create
        drop_var (bool): whether to drop the original var column at the end
    returns:
        DataFrame with lagged columns added
    """

    def __init__(self, var, lag_list, drop_var=True):
        self.var = var
        self.lag_list = lag_list
        self.drop_var = drop_var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for lag in self.lag_list:
            func = {"%s_lag%s" % (self.var, lag): lambda df: X[self.var].shift(lag)}
            X = X.assign(**func)
        if self.drop_var:
            X = X.drop(self.var, axis=1)
        return X


class NanDropper(TransformerMixin, BaseEstimator):
    """
    This transformer drops any rows that contain NaNs.

    returns:
        DataFrame with NaN rows dropped
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna()
