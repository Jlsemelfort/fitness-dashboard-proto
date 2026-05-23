from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="HYROX Training Dashboard", page_icon="🏃", layout="wide")


PROCESSED_DIR = Path("data/processed")
MANUAL_DIR = Path("data/manual")
MANUAL_LOG_PATH = MANUAL_DIR / "daily_log.csv"
WORKOUTS_PATH = PROCESSED_DIR / "workouts.csv"
RECORDS_PATH = PROCESSED_DIR / "records.csv"
RUNS_PATH = PROCESSED_DIR / "runs_clean.csv"

PLOT_TEMPLATE = "plotly_white"
MANUAL_LOG_COLUMNS = [
    "date",
    "day",
    "split",
    "lift_done",
    "z2_done",
    "speed_done",
    "long_done",
    "diet_done",
    "weight",
    "notes",
]
BOOLEAN_COLUMNS = ["lift_done", "z2_done", "speed_done", "long_done", "diet_done"]
LIFT_SPLITS = {"Upper", "Lower", "Push", "Pull", "Legs"}
REST_SPLITS = {"Rest", "Recovery", "No Lift"}
SPLIT_OPTIONS = ["Upper", "Lower", "Push", "Pull", "Legs", "Rest", "Recovery", "No Lift"]


def empty_df(columns=None):
    return pd.DataFrame(columns=columns or [])


@st.cache_data(show_spinner=False)
def load_csv(path_str):
    path = Path(path_str)
    if not path.exists():
        return empty_df(), f"Missing file: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return empty_df(), f"Could not read {path.name}: {exc}"


def normalize_bool_series(series):
    if series is None:
        return pd.Series(dtype=bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
            "t": True,
            "f": False,
        }
    )
    return mapped.fillna(False).astype(bool)


def normalize_datetime_series(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def normalize_manual_log(df):
    manual = df.copy()
    for column in MANUAL_LOG_COLUMNS:
        if column not in manual.columns:
            manual[column] = np.nan
    manual = manual[MANUAL_LOG_COLUMNS]

    manual["date"] = normalize_datetime_series(manual["date"])
    manual = manual.dropna(subset=["date"]).copy()
    manual["date"] = manual["date"].dt.normalize()
    manual["day"] = manual["date"].dt.day_name()
    manual["split"] = manual["split"].fillna("No Lift").replace("", "No Lift")
    manual["weight"] = pd.to_numeric(manual["weight"], errors="coerce")
    manual["notes"] = manual["notes"].fillna("").astype(str)

    for column in BOOLEAN_COLUMNS:
        manual[column] = normalize_bool_series(manual[column])

    manual = manual.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    manual["week_start"] = manual["date"].dt.normalize() - pd.to_timedelta(manual["date"].dt.weekday, unit="D")
    manual["week_end"] = manual["week_start"] + pd.Timedelta(days=6)
    manual["date_only"] = manual["date"].dt.date
    return manual


def ensure_manual_log_file():
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    if not MANUAL_LOG_PATH.exists():
        pd.DataFrame(columns=MANUAL_LOG_COLUMNS).to_csv(MANUAL_LOG_PATH, index=False)


def load_manual_log():
    ensure_manual_log_file()
    try:
        manual = pd.read_csv(MANUAL_LOG_PATH)
    except Exception:
        manual = empty_df(MANUAL_LOG_COLUMNS)
    return normalize_manual_log(manual)


def save_manual_log(df):
    ensure_manual_log_file()
    export_df = df.copy()
    export_df["date"] = normalize_datetime_series(export_df["date"]).dt.strftime("%Y-%m-%d")
    export_df = export_df[MANUAL_LOG_COLUMNS]
    export_df.to_csv(MANUAL_LOG_PATH, index=False)


def get_week_bounds(target_date):
    target_ts = pd.Timestamp(target_date).normalize()
    week_start = target_ts - pd.Timedelta(days=target_ts.weekday())
    return week_start.date(), (week_start + pd.Timedelta(days=6)).date()


def filter_by_date(df, start, end, date_col="date"):
    if df.empty or date_col not in df.columns:
        return df.copy()
    filtered = df.copy()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    series = normalize_datetime_series(filtered[date_col])
    return filtered[series.between(start_ts, end_ts)].copy()


def coerce_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = normalize_datetime_series(df[col])
    return df


def coerce_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_pace(minutes_per_mile):
    if pd.isna(minutes_per_mile) or minutes_per_mile <= 0:
        return "N/A"
    whole = int(minutes_per_mile)
    seconds = int(round((minutes_per_mile - whole) * 60))
    if seconds == 60:
        whole += 1
        seconds = 0
    return f"{whole}:{seconds:02d} /mi"


def format_weight_delta(value):
    if pd.isna(value):
        return "N/A"
    prefix = "+" if value > 0 else ""
    return f"{prefix}{value:.1f} lb"


def format_workout_type(raw):
    if pd.isna(raw):
        return "Unknown"
    name = str(raw).replace("HKWorkoutActivityType", "")
    readable = []
    current = ""
    for char in name:
        if char.isupper() and current:
            readable.append(current)
            current = char
        else:
            current += char
    if current:
        readable.append(current)
    return " ".join(readable) or "Unknown"


def classify_workout(workout_type):
    label = str(workout_type or "").lower()
    if "running" in label:
        return "Run"
    if "strength" in label or "cross" in label:
        return "Strength / Hybrid"
    if "walking" in label or "hiking" in label:
        return "Walking / Recovery"
    return "Other"


def enrich_workouts(workouts):
    if workouts.empty:
        return workouts.copy()
    enriched = workouts.copy()
    enriched = coerce_datetime(enriched, ["start", "end"])
    enriched = coerce_numeric(enriched, ["duration", "total_distance", "total_energy"])
    enriched["workout_label"] = enriched["workout_type"].map(format_workout_type)
    enriched["category"] = enriched["workout_type"].map(classify_workout)
    enriched["duration_minutes"] = enriched.get("duration", np.nan)
    enriched["active_calories"] = enriched.get("total_energy", np.nan)
    enriched["date"] = enriched["start"].dt.normalize()
    enriched["week_start"] = enriched["date"] - pd.to_timedelta(enriched["date"].dt.weekday, unit="D")
    return enriched


def prepare_runs(runs, workouts):
    if runs.empty:
        if workouts.empty:
            return runs.copy()
        fallback = workouts[workouts["category"] == "Run"].copy()
        if fallback.empty:
            return fallback
        fallback["distance_miles"] = fallback.get("total_distance", np.nan)
        fallback["duration_minutes"] = fallback.get("duration_minutes", np.nan)
        fallback["pace"] = fallback["duration_minutes"] / fallback["distance_miles"].replace(0, np.nan)
        fallback["run_role"] = "Run"
        return fallback.sort_values("start")

    prepared = runs.copy()
    prepared = coerce_datetime(prepared, ["start", "end"])
    prepared = coerce_numeric(prepared, ["duration", "distance", "total_distance", "pace", "total_energy"])
    prepared["distance_miles"] = prepared.get("distance", prepared.get("total_distance", np.nan))
    prepared["duration_minutes"] = prepared.get("duration", np.nan)
    if "pace" not in prepared.columns or prepared["pace"].isna().all():
        prepared["pace"] = prepared["duration_minutes"] / prepared["distance_miles"].replace(0, np.nan)
    prepared["date"] = prepared["start"].dt.normalize()
    prepared["week_start"] = prepared["date"] - pd.to_timedelta(prepared["date"].dt.weekday, unit="D")
    prepared["run_role"] = "Easy / Steady"

    for _, group in prepared.groupby("week_start"):
        if group.empty:
            continue
        longest_idx = group["distance_miles"].idxmax()
        prepared.loc[longest_idx, "run_role"] = "Long Run"
        faster = group.drop(index=longest_idx, errors="ignore")
        if not faster.empty:
            fast_idx = faster["pace"].idxmin()
            prepared.loc[fast_idx, "run_role"] = "Faster Session"

    return prepared.sort_values("start")


def prepare_weight(records, manual_log):
    frames = []

    if not records.empty and {"type", "start", "value", "unit"}.issubset(records.columns):
        bodymass = records[records["type"] == "HKQuantityTypeIdentifierBodyMass"].copy()
        if not bodymass.empty:
            bodymass = coerce_datetime(bodymass, ["start"])
            bodymass = coerce_numeric(bodymass, ["value"])
            bodymass["weight_lb"] = np.where(bodymass["unit"].eq("lb"), bodymass["value"], np.nan)
            bodymass = bodymass[["start", "weight_lb"]].dropna()
            frames.append(bodymass)

    if not manual_log.empty:
        manual_weights = manual_log.dropna(subset=["weight"])[["date", "weight"]].copy()
        if not manual_weights.empty:
            manual_weights["start"] = pd.to_datetime(manual_weights["date"])
            manual_weights["weight_lb"] = manual_weights["weight"]
            frames.append(manual_weights[["start", "weight_lb"]])

    if not frames:
        return empty_df(["start", "weight_lb", "date", "week_start", "weight_7d_avg"])

    weights = pd.concat(frames, ignore_index=True)
    weights = weights.dropna(subset=["start", "weight_lb"]).sort_values("start")
    weights["date"] = weights["start"].dt.normalize()
    weights = weights.drop_duplicates(subset=["date"], keep="last")
    weights["week_start"] = weights["date"] - pd.to_timedelta(weights["date"].dt.weekday, unit="D")
    weights["weight_7d_avg"] = weights.set_index("start")["weight_lb"].rolling("7D", min_periods=1).mean().values
    return weights


def get_row_for_date(df, target_date):
    if df.empty:
        return None
    target_ts = pd.Timestamp(target_date).normalize()
    matches = df[df["date"] == target_ts]
    if matches.empty:
        return None
    return matches.iloc[-1]


def calculate_weekly_progress(df, target_date):
    target_ts = pd.Timestamp(target_date).normalize()
    today_ts = pd.Timestamp.today().normalize()
    week_start, week_end = get_week_bounds(target_ts.date())
    week_start_ts = pd.Timestamp(week_start)
    week_end_ts = pd.Timestamp(week_end)
    effective_end = min(week_end_ts, target_ts, today_ts)

    week_df = filter_by_date(df, week_start_ts, week_end_ts, "date")
    elapsed_df = filter_by_date(df, week_start_ts, effective_end, "date")

    lift_planned = int(elapsed_df["split"].isin(LIFT_SPLITS).sum()) if not elapsed_df.empty else 0
    lift_completed = int(elapsed_df.loc[elapsed_df["split"].isin(LIFT_SPLITS), "lift_done"].sum()) if not elapsed_df.empty else 0

    z2_planned = 1 if effective_end >= week_start_ts else 0
    z2_completed = int(min(1, elapsed_df["z2_done"].sum())) if not elapsed_df.empty else 0

    speed_planned = 1 if effective_end >= week_start_ts + pd.Timedelta(days=2) else 0
    speed_completed = int(min(1, elapsed_df["speed_done"].sum())) if not elapsed_df.empty else 0

    long_planned = 1 if effective_end >= week_start_ts + pd.Timedelta(days=5) else 0
    long_completed = int(min(1, elapsed_df["long_done"].sum())) if not elapsed_df.empty else 0

    diet_planned = int(((effective_end - week_start_ts).days) + 1) if effective_end >= week_start_ts else 0
    diet_completed = int(elapsed_df["diet_done"].sum()) if not elapsed_df.empty else 0

    planned_required = lift_planned + z2_planned + speed_planned + long_planned + diet_planned
    completed_required = lift_completed + z2_completed + speed_completed + long_completed + diet_completed
    progress_pct = (completed_required / planned_required * 100) if planned_required else 0.0

    return {
        "week_start": week_start_ts,
        "week_end": week_end_ts,
        "effective_end": effective_end,
        "week_df": week_df,
        "elapsed_df": elapsed_df,
        "planned_required": planned_required,
        "completed_required": completed_required,
        "remaining_required": max(planned_required - completed_required, 0),
        "progress_pct": progress_pct,
        "lift_planned": lift_planned,
        "lift_completed": lift_completed,
        "z2_planned": z2_planned,
        "z2_completed": z2_completed,
        "speed_planned": speed_planned,
        "speed_completed": speed_completed,
        "long_planned": long_planned,
        "long_completed": long_completed,
        "diet_planned": diet_planned,
        "diet_completed": diet_completed,
        "diet_misses": max(diet_planned - diet_completed, 0),
        "lift_adherence": (lift_completed / lift_planned * 100) if lift_planned else np.nan,
    }


def generate_todo_list(df, target_date, planned_split=None):
    target_ts = pd.Timestamp(target_date).normalize()
    row = get_row_for_date(df, target_ts)
    split = row["split"] if row is not None else planned_split
    weekday = target_ts.day_name()
    week_start, _ = get_week_bounds(target_ts.date())
    week_df = filter_by_date(df, week_start, target_ts, "date")
    z2_done_this_week = bool(week_df["z2_done"].sum()) if not week_df.empty else False

    tasks = []

    if split in LIFT_SPLITS:
        tasks.append({"label": "Complete lift", "done": bool(row["lift_done"]) if row is not None else False})

    if weekday == "Wednesday":
        tasks.append({"label": "Complete speed/tempo run", "done": bool(row["speed_done"]) if row is not None else False})

    if weekday == "Saturday":
        tasks.append({"label": "Complete long run", "done": bool(row["long_done"]) if row is not None else False})

    if not z2_done_this_week:
        tasks.append({"label": "Complete Zone 2 run", "done": bool(row["z2_done"]) if row is not None else False})

    weight_logged = bool(row is not None and pd.notna(row["weight"]) and row["weight"] > 0)
    tasks.append({"label": "Log weight", "done": weight_logged})
    tasks.append({"label": "Diet on track", "done": bool(row["diet_done"]) if row is not None else False})
    return tasks


def build_weight_chart(df):
    if df.empty:
        return None
    chart_df = df.dropna(subset=["weight_lb"]).sort_values("start")
    if chart_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["start"],
            y=chart_df["weight_lb"],
            mode="lines+markers",
            name="Daily Weight",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["start"],
            y=chart_df["weight_7d_avg"],
            mode="lines",
            name="7-Day Avg",
            line=dict(color="#ff7f0e", width=3),
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Bodyweight Trend",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date",
        yaxis_title="Weight (lb)",
        legend_title_text="",
    )
    return fig


def build_current_week_donut(progress):
    values = [progress["completed_required"], progress["remaining_required"]]
    if sum(values) == 0:
        values = [1, 0]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Completed", "Remaining"],
                values=values,
                hole=0.65,
                marker=dict(colors=["#2ca02c", "#d9d9d9"]),
                sort=False,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=f"Current Week On Track: {progress['progress_pct']:.0f}%",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def build_training_breakdown_chart(progress):
    breakdown = pd.DataFrame(
        {
            "category": ["Lift", "Zone 2", "Speed", "Long Run", "Diet"],
            "completed": [
                progress["lift_completed"],
                progress["z2_completed"],
                progress["speed_completed"],
                progress["long_completed"],
                progress["diet_completed"],
            ],
        }
    )
    fig = px.bar(
        breakdown,
        x="category",
        y="completed",
        color="category",
        template=PLOT_TEMPLATE,
        title="Current Week Training Breakdown",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), showlegend=False, xaxis_title="", yaxis_title="Completed")
    return fig


def build_split_breakdown_chart(df):
    if df.empty:
        return None
    lifts = df[df["lift_done"] & df["split"].isin(LIFT_SPLITS)].copy()
    if lifts.empty:
        return None

    def map_split(split):
        if split in {"Lower", "Legs"}:
            return "Lower / Legs"
        if split in {"Upper", "Push", "Pull"}:
            return split
        return "Other"

    lifts["split_bucket"] = lifts["split"].map(map_split)
    split_counts = lifts["split_bucket"].value_counts().rename_axis("split").reset_index(name="count")
    fig = px.pie(
        split_counts,
        names="split",
        values="count",
        hole=0.55,
        template=PLOT_TEMPLATE,
        title="Completed Lifts by Split",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def build_weekly_adherence_chart(df):
    if df.empty:
        return None
    weekly_rows = []
    for week_start in sorted(df["week_start"].dropna().unique()):
        progress = calculate_weekly_progress(df, pd.Timestamp(week_start) + pd.Timedelta(days=6))
        weekly_rows.append({"week_start": pd.Timestamp(week_start), "progress_pct": progress["progress_pct"]})
    weekly = pd.DataFrame(weekly_rows)
    if weekly.empty:
        return None
    fig = px.line(
        weekly,
        x="week_start",
        y="progress_pct",
        markers=True,
        template=PLOT_TEMPLATE,
        title="Weekly Adherence Over Time",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Week", yaxis_title="Adherence %")
    return fig


def build_running_volume_chart(runs):
    if runs.empty:
        return None
    weekly = (
        runs.groupby("week_start", as_index=False)
        .agg(distance_miles=("distance_miles", "sum"), runs=("distance_miles", "size"))
        .sort_values("week_start")
    )
    fig = px.bar(
        weekly,
        x="week_start",
        y="distance_miles",
        hover_data=["runs"],
        template=PLOT_TEMPLATE,
        title="Weekly Running Volume",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Week", yaxis_title="Miles")
    return fig


def build_running_pace_chart(runs):
    if runs.empty:
        return None
    chart_df = runs.dropna(subset=["start", "pace"]).sort_values("start")
    if chart_df.empty:
        return None
    fig = px.line(
        chart_df,
        x="start",
        y="pace",
        color="run_role",
        markers=True,
        hover_data=["distance_miles", "duration_minutes"],
        template=PLOT_TEMPLATE,
        title="Running Pace Over Time",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Date", yaxis_title="Min / mile")
    return fig


def get_period_bounds(period, custom_range, manual_log, workouts, runs, weights):
    today = pd.Timestamp.today().normalize().date()
    datasets = []
    for df, col in [(manual_log, "date"), (workouts, "date"), (runs, "date"), (weights, "date")]:
        if not df.empty and col in df.columns:
            values = pd.to_datetime(df[col], errors="coerce").dropna()
            if not values.empty:
                datasets.extend(values.dt.date.tolist())

    earliest = min(datasets) if datasets else today
    latest = max(datasets) if datasets else today

    if period == "Current Week":
        start, end = get_week_bounds(today)
        return start, min(end, today)
    if period == "Last 4 Weeks":
        return today - timedelta(days=27), today
    if period == "Custom":
        if isinstance(custom_range, tuple) and len(custom_range) == 2:
            return custom_range[0], custom_range[1]
        return today, today
    return earliest, latest


def build_insights(manual_log, filtered_weights, current_progress):
    insights = []
    today = pd.Timestamp.today().normalize()
    current_week = current_progress["week_df"]
    week_started = current_progress["effective_end"] >= current_progress["week_start"]

    if current_progress["progress_pct"] >= 85:
        insights.append("You are on track this week.")
    if current_progress["diet_misses"] >= 2:
        insights.append("Diet is slipping this week.")
    if week_started and current_progress["z2_completed"] == 0:
        insights.append("You have not completed a Zone 2 run this week.")
    if today >= current_progress["week_start"] + pd.Timedelta(days=5) and current_progress["long_completed"] == 0:
        insights.append("You missed your long run this week.")
    if pd.notna(current_progress["lift_adherence"]) and current_progress["lift_adherence"] >= 90:
        insights.append("Lift consistency is strong.")

    weight_df = filtered_weights.dropna(subset=["weight_7d_avg"]).sort_values("start")
    if len(weight_df) >= 2:
        rolling_delta = weight_df["weight_7d_avg"].iloc[-1] - weight_df["weight_7d_avg"].iloc[0]
        if rolling_delta <= -0.5:
            insights.append("Weight is trending down.")
        elif abs(rolling_delta) < 0.5:
            insights.append("Weight trend is flat.")

    if not insights:
        if manual_log.empty:
            insights.append("Start logging daily check-ins to unlock coach-style insights.")
        elif current_week.empty:
            insights.append("No entries logged for the current week yet.")
        else:
            insights.append("This week is in motion, but there is not enough variance yet for stronger insights.")

    return insights


def get_weight_metrics(weights, start_date, end_date):
    if weights.empty:
        return {"current": np.nan, "starting": np.nan, "total_change": np.nan, "range_change": np.nan}

    valid = weights.dropna(subset=["weight_lb"]).sort_values("start")
    if valid.empty:
        return {"current": np.nan, "starting": np.nan, "total_change": np.nan, "range_change": np.nan}

    ranged = filter_by_date(valid, start_date, end_date, "date")
    current = valid["weight_lb"].iloc[-1]
    starting = valid["weight_lb"].iloc[0]
    total_change = current - starting

    if ranged.empty:
        range_change = np.nan
    else:
        range_change = ranged["weight_lb"].iloc[-1] - ranged["weight_lb"].iloc[0]

    return {
        "current": current,
        "starting": starting,
        "total_change": total_change,
        "range_change": range_change,
    }


def render_todo_list(manual_log, target_date, planned_split=None):
    st.subheader("Today's To-Do List")
    todos = generate_todo_list(manual_log, target_date, planned_split=planned_split)
    if not todos:
        st.caption("No required items for today.")
        return
    for item in todos:
        prefix = "✅" if item["done"] else "⬜"
        st.write(f"{prefix} {item['label']}")


def render_daily_log_tab(manual_log):
    st.subheader("Daily Check-In")
    selected_date = st.date_input("Log date", value=date.today(), key="daily_log_date")
    existing_row = get_row_for_date(manual_log, selected_date)
    selected_day = pd.Timestamp(selected_date).day_name()

    default_split = existing_row["split"] if existing_row is not None else "No Lift"
    default_weight = float(existing_row["weight"]) if existing_row is not None and pd.notna(existing_row["weight"]) else 0.0

    with st.form("daily_log_form"):
        st.caption(f"Day: {selected_day}")
        split = st.selectbox("Split", SPLIT_OPTIONS, index=SPLIT_OPTIONS.index(default_split) if default_split in SPLIT_OPTIONS else len(SPLIT_OPTIONS) - 1)
        weight = st.number_input("Bodyweight (lb)", min_value=0.0, step=0.1, value=default_weight)
        lift_done = st.checkbox("Lift Done", value=bool(existing_row["lift_done"]) if existing_row is not None else False)
        z2_done = st.checkbox("Zone 2 Done", value=bool(existing_row["z2_done"]) if existing_row is not None else False)
        speed_done = st.checkbox("Speed/Tempo Done", value=bool(existing_row["speed_done"]) if existing_row is not None else False)
        long_done = st.checkbox("Long Run Done", value=bool(existing_row["long_done"]) if existing_row is not None else False)
        diet_done = st.checkbox("Diet On Track", value=bool(existing_row["diet_done"]) if existing_row is not None else False)
        notes = st.text_area("Notes", value=existing_row["notes"] if existing_row is not None else "", height=120)
        submitted = st.form_submit_button("Save Entry", type="primary")

    if submitted:
        entry = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(selected_date),
                    "day": selected_day,
                    "split": split,
                    "lift_done": lift_done,
                    "z2_done": z2_done,
                    "speed_done": speed_done,
                    "long_done": long_done,
                    "diet_done": diet_done,
                    "weight": np.nan if weight <= 0 else weight,
                    "notes": notes.strip(),
                }
            ]
        )
        updated = pd.concat([manual_log[manual_log["date"] != pd.Timestamp(selected_date)], entry], ignore_index=True)
        manual_log = normalize_manual_log(updated)
        save_manual_log(manual_log)
        st.success(f"Saved entry for {selected_date.isoformat()}.")

    saved_row = get_row_for_date(manual_log, selected_date)
    if saved_row is not None:
        display_row = saved_row[MANUAL_LOG_COLUMNS].to_frame().T.copy()
        display_row["date"] = pd.to_datetime(display_row["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(display_row, use_container_width=True, hide_index=True)
    else:
        st.info("No saved entry for that date yet.")

    return manual_log


def render_overview_tab(manual_log, filtered_manual, filtered_weights, current_progress):
    metrics = get_weight_metrics(filtered_weights, current_progress["week_start"], current_progress["effective_end"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Week On Track", f"{current_progress['progress_pct']:.0f}%")
    col2.metric("Completed Tasks", f"{current_progress['completed_required']}/{current_progress['planned_required']}")
    col3.metric("Current Weight", f"{metrics['current']:.1f} lb" if pd.notna(metrics["current"]) else "N/A")
    col4.metric("Total Weight Change", format_weight_delta(metrics["total_change"]))

    week_today = pd.Timestamp.today().normalize().date()
    today_row = get_row_for_date(manual_log, week_today)
    temporary_split = None
    if today_row is None:
        temporary_split = st.selectbox("Today's planned split", SPLIT_OPTIONS, index=SPLIT_OPTIONS.index("No Lift"), key="overview_planned_split")

    render_todo_list(manual_log, week_today, planned_split=temporary_split)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_current_week_donut(current_progress), use_container_width=True, key="overview_current_week_donut")
    with right:
        st.plotly_chart(
            build_training_breakdown_chart(current_progress),
            use_container_width=True,
            key="overview_training_breakdown",
        )

    adherence_fig = build_weekly_adherence_chart(filtered_manual if not filtered_manual.empty else manual_log)
    if adherence_fig is not None:
        st.plotly_chart(adherence_fig, use_container_width=True, key="overview_weekly_adherence")
    else:
        st.info("Weekly adherence history will appear after you log more than one week.")


def render_running_tab(filtered_runs, manual_log, current_progress):
    if filtered_runs.empty:
        st.info("Processed running files are missing or filtered out. Manual weekly run adherence is still tracked in the dashboard.")
    else:
        top_left, top_right = st.columns(2)
        with top_left:
            volume_fig = build_running_volume_chart(filtered_runs)
            if volume_fig is not None:
                st.plotly_chart(volume_fig, use_container_width=True, key="running_volume")
        with top_right:
            pace_fig = build_running_pace_chart(filtered_runs)
            if pace_fig is not None:
                st.plotly_chart(pace_fig, use_container_width=True, key="running_pace")

    week_df = current_progress["week_df"]
    summary = pd.DataFrame(
        [
            {"Run Type": "Zone 2", "Completed": current_progress["z2_completed"], "Planned": current_progress["z2_planned"]},
            {"Run Type": "Speed / Tempo", "Completed": current_progress["speed_completed"], "Planned": current_progress["speed_planned"]},
            {"Run Type": "Long Run", "Completed": current_progress["long_completed"], "Planned": current_progress["long_planned"]},
        ]
    )
    st.subheader("Manual Running Adherence")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if not week_df.empty:
        logged_runs = week_df[["date", "z2_done", "speed_done", "long_done"]].copy()
        logged_runs["date"] = logged_runs["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(logged_runs, use_container_width=True, hide_index=True)


def render_training_tab(filtered_manual, current_progress):
    st.subheader("Split Tracking")
    split_fig = build_split_breakdown_chart(filtered_manual)
    if split_fig is not None:
        st.plotly_chart(split_fig, use_container_width=True, key="training_split_breakdown")
    else:
        st.info("Completed lift split data will appear after you log lift sessions.")

    st.subheader("Current Week Status")
    st.plotly_chart(
        build_training_breakdown_chart(current_progress),
        use_container_width=True,
        key="training_current_week_breakdown",
    )

    if not filtered_manual.empty:
        log_view = filtered_manual[MANUAL_LOG_COLUMNS].copy()
        log_view["date"] = pd.to_datetime(log_view["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(log_view.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No manual log entries match the current date filter.")


def render_weight_tab(filtered_weights, start_date, end_date):
    st.subheader("Weight Trend")
    metrics = get_weight_metrics(filtered_weights, start_date, end_date)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Weight", f"{metrics['current']:.1f} lb" if pd.notna(metrics["current"]) else "N/A")
    col2.metric("Starting Weight", f"{metrics['starting']:.1f} lb" if pd.notna(metrics["starting"]) else "N/A")
    col3.metric("Total Change", format_weight_delta(metrics["total_change"]))
    col4.metric("Selected Range Change", format_weight_delta(metrics["range_change"]))

    fig = build_weight_chart(filtered_weights)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key="weight_trend")
    else:
        st.info("No bodyweight values are available for the current filter.")


def render_insights_tab(insights):
    st.subheader("Coach Notes")
    for insight in insights:
        st.write(f"- {insight}")


def main():
    st.title("HYROX Fitness Dashboard")
    st.caption("Local-first training dashboard for manual check-ins, weight, running, lifting, and weekly adherence.")

    workouts_raw, workouts_warning = load_csv(str(WORKOUTS_PATH))
    records_raw, records_warning = load_csv(str(RECORDS_PATH))
    runs_raw, runs_warning = load_csv(str(RUNS_PATH))
    manual_log = load_manual_log()

    workouts = enrich_workouts(workouts_raw)
    runs = prepare_runs(runs_raw, workouts)
    weights = prepare_weight(records_raw, manual_log)

    warnings = [msg for msg in [workouts_warning, records_warning, runs_warning] if msg]
    for warning in warnings:
        st.info(warning)

    st.sidebar.header("Filters")
    period = st.sidebar.selectbox("Selected period", ["Current Week", "Last 4 Weeks", "All Time", "Custom"])
    custom_range = st.sidebar.date_input("Custom date range", value=(date.today() - timedelta(days=27), date.today())) if period == "Custom" else None
    start_date, end_date = get_period_bounds(period, custom_range, manual_log, workouts, runs, weights)
    st.sidebar.caption(f"Showing {start_date.isoformat()} to {end_date.isoformat()}")

    filtered_manual = filter_by_date(manual_log, start_date, end_date, "date")
    filtered_workouts = filter_by_date(workouts, start_date, end_date, "date")
    filtered_runs = filter_by_date(runs, start_date, end_date, "date")
    filtered_weights = filter_by_date(weights, start_date, end_date, "date")

    current_progress = calculate_weekly_progress(manual_log, date.today())
    insights = build_insights(manual_log, filtered_weights, current_progress)

    tabs = st.tabs(["Overview", "Daily Log", "Running", "Training Breakdown", "Weight", "Insights"])
    with tabs[0]:
        render_overview_tab(manual_log, filtered_manual, filtered_weights, current_progress)
    with tabs[1]:
        manual_log = render_daily_log_tab(manual_log)
        filtered_manual = filter_by_date(manual_log, start_date, end_date, "date")
        filtered_weights = filter_by_date(prepare_weight(records_raw, manual_log), start_date, end_date, "date")
        current_progress = calculate_weekly_progress(manual_log, date.today())
        insights = build_insights(manual_log, filtered_weights, current_progress)
    with tabs[2]:
        render_running_tab(filtered_runs, manual_log, current_progress)
    with tabs[3]:
        render_training_tab(filtered_manual, current_progress)
    with tabs[4]:
        render_weight_tab(filtered_weights, start_date, end_date)
    with tabs[5]:
        render_insights_tab(insights)


if __name__ == "__main__":
    main()
