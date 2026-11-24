"""Write the synthetic example data in data/: prices, forecasts and reserve activation shares.

All numbers are made up (seeded random numbers on a daily shape), none is a market price. The files only show the
formats multimarket_model.py reads and let every stage run; README.md describes the columns.

Run: python create_example_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
START, DAYS = "2026-03-07", 9  # two days of history, then one week of delivery from a Monday
FORECAST_ERROR = {"dayahead": 5.0, "ida1": 10.0, "ida2": 8.0, "ida3": 6.0, "idc": 5.0}  # EUR/MWh, standard deviation
# forecast issue time relative to 00:00 of the delivery day (local time), before the gate of its market; the
# day-ahead forecast is out before 08:00 so that a joint decision (stage s1f) can use it
ISSUED = {"reserve": -17, "dayahead": -17, "ida1": -11, "ida2": -4, "ida3": 8}
IDC_AHEAD = 8  # quarter hours each continuous intraday forecast covers
RESERVE_LEVEL = {"price_fcr": 12.0, "price_afrr_pos": 8.0, "price_afrr_neg": 6.0, "price_mfrr_pos": 2.0,
                 "price_mfrr_neg": 3.0}  # (EUR/MW)/h
Q90 = 1.2816  # standard deviations between the median and the 90 % quantile of a normal distribution


def create_prices(rng, index):
    local = index.tz_convert(TZ)
    hour = local.hour + local.minute / 60
    # midday dip, evening peak
    shape = 80 - 45 * np.exp(-((hour - 13) / 3.5) ** 2) + 50 * np.exp(-((hour - 19.5) / 1.8) ** 2)
    day = pd.factorize(local.normalize())[0]
    level = rng.normal(0, 12, day.max() + 1)[day]  # one price level per day
    p = pd.DataFrame(index=index)
    p["price_dayahead"] = shape + level + rng.normal(0, 6, len(index))
    p["price_ida1"] = p.price_dayahead + rng.normal(0, 12, len(index))
    p["price_ida2"] = p.price_ida1 + rng.normal(0, 9, len(index))
    p["price_ida3"] = (p.price_ida2 + rng.normal(0, 7, len(index))).where(local.hour >= 12)  # held for 12-24 h only
    p["price_id1"] = p.price_ida3.fillna(p.price_ida2) + rng.normal(0, 14, len(index))
    block = pd.factorize(local.strftime("%Y-%m-%d") + (local.hour // 4).astype(str))[0]  # 4-h blocks, local time
    for col, mean in RESERVE_LEVEL.items():
        p[col] = (mean * rng.lognormal(0, 0.5, block.max() + 1))[block]
        if col != "price_fcr":  # pay-as-bid products: highest accepted capacity price
            p[f"{col}_marginal"] = p[col] * (1 + rng.uniform(0.05, 0.4, block.max() + 1))[block]
    return p.round(2)


def create_forecasts(rng, prices):
    """One row per market, issue time and target quarter hour: the real price plus an error, with a 10 to 90 % band."""
    local = prices.index.tz_convert(TZ)
    day = local.normalize()
    parts = []

    def add(market, issue, targets, real, sigma, block=None):
        error = rng.normal(0, sigma, len(real)) if block is None else rng.normal(0, sigma, block.max() + 1)[block]
        value = real + error
        parts.append(pd.DataFrame({"market": market, "issue_time": issue, "target_time": targets, "value": value,
                                   "q10": value - Q90 * sigma, "q90": value + Q90 * sigma}).dropna())

    for d in day.unique()[1:]:
        on_day = day == d
        for key, hours in ISSUED.items():
            issue = (d + pd.Timedelta(hours=hours)).tz_convert("UTC")
            if key != "reserve":
                add(key, issue, prices.index[on_day], prices.loc[on_day, f"price_{key}"].values, FORECAST_ERROR[key])
                continue
            block = pd.factorize(local[on_day].hour // 4)[0]  # one forecast per 4-h product
            for col, level in RESERVE_LEVEL.items():
                add(col, issue, prices.index[on_day], prices.loc[on_day, col].values, 0.3 * level, block)
    for i in range(96, len(prices) - IDC_AHEAD):  # every quarter hour for the next two hours
        targets = prices.index[i + 1:i + 1 + IDC_AHEAD]
        add("idc", prices.index[i], targets, prices.loc[targets, "price_id1"].values, FORECAST_ERROR["idc"])
    return pd.concat(parts, ignore_index=True).round(2)


def create_activation(rng):
    """Activated share of the reserved aFRR and mFRR per local quarter hour of the day."""
    quarter = np.arange(96)
    wave = 1 + 0.5 * np.sin(2 * np.pi * quarter / 96)
    level = {"aFRR_pos": 0.10, "aFRR_neg": 0.12, "mFRR_pos": 0.01, "mFRR_neg": 0.02}
    return pd.DataFrame({k: (v * wave * rng.uniform(0.7, 1.3, 96)).round(4) for k, v in level.items()},
                        index=pd.Index(quarter, name="quarter_of_day"))


def main():
    rng = np.random.default_rng(1)
    data = Path(__file__).resolve().parent / "data"
    data.mkdir(exist_ok=True)
    start = pd.Timestamp(START, tz=TZ)
    index = pd.date_range(start, start + pd.Timedelta(days=DAYS), freq="15min", inclusive="left").tz_convert("UTC")
    prices = create_prices(rng, index.rename("timestamp_utc"))
    prices.to_parquet(data / "example_prices.parquet")
    create_forecasts(rng, prices).to_parquet(data / "example_forecasts.parquet", index=False)
    create_activation(rng).to_parquet(data / "example_activation.parquet")
    print(f"written to {data}: {len(prices)} quarter hours from {START} ({TZ})")


if __name__ == "__main__":
    main()
