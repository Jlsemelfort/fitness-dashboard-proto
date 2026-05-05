import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

RAW_XML = Path("data/raw/export.xml")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_workouts(root):
    rows = []

    for workout in root.findall("Workout"):
        distance = workout.get("totalDistance")
        distance_unit = workout.get("totalDistanceUnit")
        energy = workout.get("totalEnergyBurned")
        energy_unit = workout.get("totalEnergyBurnedUnit")

        for stat in workout.findall("WorkoutStatistics"):
            stat_type = stat.get("type")

            if stat_type == "HKQuantityTypeIdentifierDistanceWalkingRunning":
                distance = stat.get("sum")
                distance_unit = stat.get("unit")

            if stat_type == "HKQuantityTypeIdentifierActiveEnergyBurned":
                energy = stat.get("sum")
                energy_unit = stat.get("unit")

        rows.append({
            "workout_type": workout.get("workoutActivityType"),
            "start": workout.get("startDate"),
            "end": workout.get("endDate"),
            "duration": workout.get("duration"),
            "duration_unit": workout.get("durationUnit"),
            "total_distance": distance,
            "distance_unit": distance_unit,
            "total_energy": energy,
            "energy_unit": energy_unit,
            "source": workout.get("sourceName"),
        })

    return pd.DataFrame(rows)


def parse_records(root):
    rows = []

    wanted_types = {
        "HKQuantityTypeIdentifierHeartRate",
        "HKQuantityTypeIdentifierBodyMass",
        "HKQuantityTypeIdentifierActiveEnergyBurned",
        "HKQuantityTypeIdentifierStepCount",
        "HKQuantityTypeIdentifierDistanceWalkingRunning",
    }

    for record in root.findall("Record"):
        record_type = record.get("type")

        if record_type in wanted_types:
            rows.append({
                "type": record_type,
                "source": record.get("sourceName"),
                "start": record.get("startDate"),
                "end": record.get("endDate"),
                "value": record.get("value"),
                "unit": record.get("unit"),
            })

    return pd.DataFrame(rows)


def clean_dates(df):
    for col in ["start", "end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def main():
    print("Parsing XML...")
    tree = ET.parse(RAW_XML)
    root = tree.getroot()

    workouts = clean_dates(parse_workouts(root))
    records = clean_dates(parse_records(root))

    workouts.to_csv(OUT_DIR / "workouts.csv", index=False)
    records.to_csv(OUT_DIR / "records.csv", index=False)

    print(f"Saved {len(workouts)} workouts")
    print(f"Saved {len(records)} records")


if __name__ == "__main__":
    main()