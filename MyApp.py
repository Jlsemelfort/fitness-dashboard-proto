import pandas as pd
import streamlit as st

st.title("Hyrox Dashboard")

weekly = pd.read_csv("data/processed/weekly_running_summary.csv")

st.subheader("Raw weekly data")
st.dataframe(weekly)

weekly["week"] = pd.to_datetime(weekly["week"])
weekly["distance"] = pd.to_numeric(weekly["distance"], errors="coerce")

st.subheader("Weekly Running Distance")
st.line_chart(weekly.set_index("week")["distance"])