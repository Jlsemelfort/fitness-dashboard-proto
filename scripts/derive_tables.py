import pandas as pd

workouts = pd.read_csv("data/processed/workouts.csv")

# Filter only runs
runs = workouts[
    workouts["workout_type"].str.contains("Running", na=False)
].copy()

# Convert types
runs["duration"] = pd.to_numeric(runs["duration"], errors="coerce")
runs["distance"] = pd.to_numeric(runs["total_distance"], errors="coerce")

# Calculate pace (min per mile)
runs["pace"] = runs["duration"] / runs["distance"].replace(0, pd.NA)

# Clean date
runs["start"] = pd.to_datetime(runs["start"])

# Save
runs.to_csv("data/processed/runs_clean.csv", index=False)

# weekly summary
runs["week"] = runs["start"].dt.to_period("W").apply(lambda r: r.start_time)

weekly = runs.groupby("week").agg({
    "distance": "sum",
    "duration": "sum",
    "pace": "mean"
}).reset_index()

weekly.to_csv("data/processed/weekly_running_summary.csv", index=False)

#Daily Aggregation
runs["date"] = runs["start"].dt.date

daily = runs.groupby("date").agg({
    "distance": "sum",
    "duration": "sum"
}).reset_index()

daily.to_csv("data/processed/daily_runs.csv", index=False)