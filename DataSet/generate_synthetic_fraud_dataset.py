from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT_DIR / "DataSet" / "synthetic_fraud_merged.csv"

ROW_COUNT = 100_000
FRAUD_RATE = 0.05
RANDOM_STATE = 42
CUSTOMER_COUNT = 5_000
MERCHANTS_PER_CATEGORY = 45

CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]
NORMAL_CATEGORY_PROBS = np.array(
    [0.08, 0.12, 0.13, 0.04, 0.12, 0.07, 0.10, 0.08, 0.04, 0.07, 0.08, 0.04, 0.08, 0.05]
)
FRAUD_CATEGORY_PROBS = np.array(
    [0.04, 0.04, 0.06, 0.05, 0.16, 0.03, 0.04, 0.03, 0.18, 0.05, 0.04, 0.20, 0.06, 0.02]
)
NORMAL_CATEGORY_PROBS = NORMAL_CATEGORY_PROBS / NORMAL_CATEGORY_PROBS.sum()
FRAUD_CATEGORY_PROBS = FRAUD_CATEGORY_PROBS / FRAUD_CATEGORY_PROBS.sum()

NIGHT_HOURS = np.array([22, 23, 0, 1, 2, 3, 4, 5])
DAY_HOURS = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
RISK_CATEGORIES = np.array(["grocery_pos", "misc_net", "shopping_net", "shopping_pos"])

NORMAL_NIGHT_RATE = 0.14
FRAUD_NIGHT_RATE = 0.68
NORMAL_HIGH_AMOUNT_RATE = 0.08
FRAUD_STEALTH_AMOUNT_RATE = 0.32
NORMAL_FAR_MERCHANT_RATE = 0.07
FRAUD_CLOSE_MERCHANT_RATE = 0.30
NORMAL_RISK_CATEGORY_NOISE_RATE = 0.10
FRAUD_LOW_RISK_CATEGORY_RATE = 0.22

MISSING_RATES = {
    "job": 0.01,
    "city_pop": 0.005,
    "zip": 0.005,
    "merchant": 0.005,
    "dob": 0.003,
}

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Faker-based synthetic fraud dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--rows", type=int, default=ROW_COUNT)
    parser.add_argument("--fraud-rate", type=float, default=FRAUD_RATE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def build_customers(fake: Faker, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for customer_id in range(CUSTOMER_COUNT):
        first = fake.first_name()
        last = fake.last_name()
        gender = rng.choice(["F", "M"], p=[0.52, 0.48])
        lat = rng.uniform(26.0, 48.5)
        long = rng.uniform(-123.0, -69.0)
        rows.append(
            {
                "customer_id": customer_id,
                "cc_num": str(fake.credit_card_number()),
                "first": first,
                "last": last,
                "gender": gender,
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip": fake.postcode(),
                "lat": round(float(lat), 6),
                "long": round(float(long), 6),
                "city_pop": int(rng.lognormal(mean=10.6, sigma=1.0)),
                "job": fake.job(),
                "dob": fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
            }
        )

    return pd.DataFrame(rows)


def build_merchants(fake: Faker, rng: np.random.Generator) -> dict[str, list[str]]:
    merchants: dict[str, list[str]] = {}
    for category in CATEGORIES:
        merchants[category] = [
            f"fraud_{category}_{fake.company().replace(',', '')}_{index}"
            for index in range(MERCHANTS_PER_CATEGORY)
        ]
        rng.shuffle(merchants[category])
    return merchants


def generate_timestamps(
    rng: np.random.Generator,
    rows: int,
    fraud_mask: np.ndarray,
) -> pd.Series:
    base_date = np.datetime64("2024-01-01")
    days = rng.integers(0, 365, size=rows)
    hours = np.empty(rows, dtype="int64")
    normal_count = int((~fraud_mask).sum())
    fraud_count = int(fraud_mask.sum())
    normal_night = rng.random(normal_count) < NORMAL_NIGHT_RATE
    fraud_night = rng.random(fraud_count) < FRAUD_NIGHT_RATE

    normal_hours = np.where(
        normal_night,
        rng.choice(NIGHT_HOURS, size=normal_count),
        rng.choice(DAY_HOURS, size=normal_count),
    )
    fraud_hours = np.where(
        fraud_night,
        rng.choice(NIGHT_HOURS, size=fraud_count),
        rng.choice(DAY_HOURS, size=fraud_count),
    )
    hours[~fraud_mask] = normal_hours
    hours[fraud_mask] = fraud_hours
    minutes = rng.integers(0, 60, size=rows)
    seconds = rng.integers(0, 60, size=rows)

    timestamps = (
        base_date
        + days.astype("timedelta64[D]")
        + hours.astype("timedelta64[h]")
        + minutes.astype("timedelta64[m]")
        + seconds.astype("timedelta64[s]")
    )
    return pd.Series(pd.to_datetime(timestamps))


def generate_amounts(
    rng: np.random.Generator,
    rows: int,
    fraud_mask: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    category_multipliers = {
        "gas_transport": 1.15,
        "grocery_pos": 1.3,
        "shopping_net": 1.8,
        "shopping_pos": 1.5,
        "travel": 2.0,
        "misc_net": 1.7,
    }
    multipliers = np.array([category_multipliers.get(category, 1.0) for category in categories])
    amounts = rng.lognormal(mean=3.65, sigma=0.65, size=rows) * multipliers

    normal_index = np.where(~fraud_mask)[0]
    fraud_index = np.where(fraud_mask)[0]
    normal_high_index = normal_index[
        rng.random(len(normal_index)) < NORMAL_HIGH_AMOUNT_RATE
    ]
    fraud_stealth_index = fraud_index[
        rng.random(len(fraud_index)) < FRAUD_STEALTH_AMOUNT_RATE
    ]
    fraud_obvious_index = np.setdiff1d(fraud_index, fraud_stealth_index, assume_unique=True)

    amounts[normal_high_index] = rng.lognormal(
        mean=5.65,
        sigma=0.55,
        size=len(normal_high_index),
    )
    amounts[fraud_obvious_index] = rng.lognormal(
        mean=5.85,
        sigma=0.55,
        size=len(fraud_obvious_index),
    )
    amounts[fraud_stealth_index] = rng.lognormal(
        mean=4.25,
        sigma=0.75,
        size=len(fraud_stealth_index),
    )
    return np.round(np.clip(amounts, 1.0, 5_000.0), 2)


def generate_dataset(rows: int, fraud_rate: float, random_state: int) -> pd.DataFrame:
    fake = Faker("en_US")
    Faker.seed(random_state)
    rng = np.random.default_rng(random_state)

    fraud_count = int(round(rows * fraud_rate))
    fraud_mask = np.array([True] * fraud_count + [False] * (rows - fraud_count))
    rng.shuffle(fraud_mask)

    customers = build_customers(fake, rng)
    merchants = build_merchants(fake, rng)
    customer_ids = rng.integers(0, CUSTOMER_COUNT, size=rows)
    customer_frame = customers.iloc[customer_ids].reset_index(drop=True)

    categories = np.empty(rows, dtype=object)
    categories[~fraud_mask] = rng.choice(
        CATEGORIES,
        size=int((~fraud_mask).sum()),
        p=NORMAL_CATEGORY_PROBS,
    )
    categories[fraud_mask] = rng.choice(
        CATEGORIES,
        size=fraud_count,
        p=FRAUD_CATEGORY_PROBS,
    )
    normal_index = np.where(~fraud_mask)[0]
    fraud_index = np.where(fraud_mask)[0]
    normal_risk_index = normal_index[
        rng.random(len(normal_index)) < NORMAL_RISK_CATEGORY_NOISE_RATE
    ]
    fraud_low_risk_index = fraud_index[
        rng.random(len(fraud_index)) < FRAUD_LOW_RISK_CATEGORY_RATE
    ]
    categories[normal_risk_index] = rng.choice(RISK_CATEGORIES, size=len(normal_risk_index))
    categories[fraud_low_risk_index] = rng.choice(CATEGORIES, size=len(fraud_low_risk_index), p=NORMAL_CATEGORY_PROBS)

    merchant_values = [
        rng.choice(merchants[category])
        for category in categories
    ]
    timestamps = generate_timestamps(rng, rows, fraud_mask)
    amounts = generate_amounts(rng, rows, fraud_mask, categories)

    merch_lat = customer_frame["lat"].astype(float).to_numpy().copy()
    merch_long = customer_frame["long"].astype(float).to_numpy().copy()
    normal_index = np.where(~fraud_mask)[0]
    fraud_index = np.where(fraud_mask)[0]
    normal_far_index = normal_index[
        rng.random(len(normal_index)) < NORMAL_FAR_MERCHANT_RATE
    ]
    normal_close_index = np.setdiff1d(normal_index, normal_far_index, assume_unique=True)
    fraud_close_index = fraud_index[
        rng.random(len(fraud_index)) < FRAUD_CLOSE_MERCHANT_RATE
    ]
    fraud_far_index = np.setdiff1d(fraud_index, fraud_close_index, assume_unique=True)

    merch_lat[normal_close_index] += rng.normal(0, 0.10, size=len(normal_close_index))
    merch_long[normal_close_index] += rng.normal(0, 0.10, size=len(normal_close_index))
    merch_lat[fraud_close_index] += rng.normal(0, 0.18, size=len(fraud_close_index))
    merch_long[fraud_close_index] += rng.normal(0, 0.18, size=len(fraud_close_index))

    for index_set, low, high in [
        (normal_far_index, 1.2, 5.0),
        (fraud_far_index, 1.8, 7.0),
    ]:
        distance = rng.uniform(low, high, size=len(index_set))
        lat_sign = rng.choice([-1.0, 1.0], size=len(index_set))
        long_sign = rng.choice([-1.0, 1.0], size=len(index_set))
        merch_lat[index_set] += lat_sign * distance
        merch_long[index_set] += long_sign * distance
    merch_lat = np.clip(merch_lat, 24.5, 49.5)
    merch_long = np.clip(merch_long, -124.5, -66.5)

    dataframe = pd.DataFrame(
        {
            "trans_date_trans_time": timestamps.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": customer_frame["cc_num"].astype(str),
            "merchant": merchant_values,
            "category": categories,
            "amt": amounts,
            "first": customer_frame["first"],
            "last": customer_frame["last"],
            "gender": customer_frame["gender"],
            "street": customer_frame["street"],
            "city": customer_frame["city"],
            "state": customer_frame["state"],
            "zip": customer_frame["zip"].astype(str),
            "lat": customer_frame["lat"].astype(float),
            "long": customer_frame["long"].astype(float),
            "city_pop": customer_frame["city_pop"].astype(int),
            "job": customer_frame["job"],
            "dob": customer_frame["dob"],
            "trans_num": [f"syn_{index:06d}_{rng.integers(0, 10**10):010d}" for index in range(rows)],
            "unix_time": timestamps.astype("int64").astype(int),
            "merch_lat": np.round(merch_lat, 6),
            "merch_long": np.round(merch_long, 6),
            "is_fraud": fraud_mask,
            "source_dataset": "synthetic_faker_noisy_anomaly",
        }
    )

    for column, missing_rate in MISSING_RATES.items():
        missing_mask = rng.random(rows) < missing_rate
        dataframe.loc[missing_mask, column] = pd.NA

    return dataframe.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def print_summary(dataframe: pd.DataFrame, output_path: Path) -> None:
    fraud_count = int(dataframe["is_fraud"].sum())
    summary = Table(title="Synthetic Fraud Dataset Summary", header_style="bold magenta")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("rows", f"{len(dataframe):,}")
    summary.add_row("fraud_rows", f"{fraud_count:,}")
    summary.add_row("non_fraud_rows", f"{len(dataframe) - fraud_count:,}")
    summary.add_row("fraud_rate", f"{dataframe['is_fraud'].mean() * 100:.2f}%")
    summary.add_row("columns", f"{len(dataframe.columns):,}")
    summary.add_row("output", str(output_path))
    console.print(summary)


def main() -> None:
    args = parse_args()
    dataframe = generate_dataset(args.rows, args.fraud_rate, args.random_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)

    console.print(
        Panel.fit(
            "Faker-based synthetic fraud data\n"
            "Pattern: noisy overlapping anomaly for One-Class SVM\n"
            "Schema: compatible with fraud_merged.csv",
            title="Synthetic Dataset Generator",
            border_style="cyan",
        )
    )
    print_summary(dataframe, args.output)


if __name__ == "__main__":
    main()
