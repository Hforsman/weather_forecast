import pandas as pd
import streamlit as st
import sqlite3

# create database connection
conn = sqlite3.connect("assignment_sf.db")

# Get all historical and predicted data
hist_temp_df = pd.read_sql("SELECT * FROM historical_temperature", conn)
pred_temp_df = pd.read_sql("SELECT * FROM predictions", conn)

# join historical and predicted data
data = hist_temp_df.merge(pred_temp_df, on="timestamp", how="outer")

# plot historical temp vs predicted temp
st.line_chart(data[["temp", "predicted_temp"]])

# plot the mse
st.line_chart(data["mse"])
