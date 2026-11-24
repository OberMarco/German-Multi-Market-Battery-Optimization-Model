# German Multi-Market Battery Optimization Model

A battery trades on the German electricity markets: reserve capacity (FCR, aFRR, mFRR), day-ahead, the intraday
auctions IDA1 to IDA3 and the continuous intraday (IDC), in quarter hours. The model walks along the time line of the
markets and takes each decision with what is known at that moment. It is built with PyPSA and solved with HiGHS.

Stages lead from one decision with perfect foresight to bids at every gate under near-real conditions. Each stage
removes one simplification of the stage before, so the difference between two stages is the value of exactly that
simplification.

The repository contains no market prices. `data/` holds synthetic example files that show the formats and let every
stage run. Results on them say nothing about real revenues.

## Run

```
conda env create -f environment.yaml
conda activate battery-multimarket
python multimarket_model.py                        # defaults of config.yaml (stage s4)
python multimarket_model.py +stage=max run.days=2  # a stage, first two delivery days only
python multimarket_model.py +stage=max costs.marketer_share=0 physics.availability=1.0   # any switch by hand
```

One delivery day takes about one minute. A run writes `results.csv` (one row per quarter hour), `summary.csv`
(revenue per market) and `schedule.pdf` to `run.output_dir`.

| File | Content |
|---|---|
| `multimarket_model.py` | the model: events, planning prices, optimisation per event, settlement |
| `config.yaml` | battery, data files and all switches, grouped by the axes markets, knowledge, decision, physics, costs |
| `stage/*.yaml` | one file per stage with the switches that differ from `config.yaml` |
| `create_example_data.py` | writes the synthetic files in `data/` |

## Model

Events per delivery day D, local time:

| Event | Decides | Time |
|---|---|---|
| reserve | FCR, aFRR, mFRR capacity of D in 4-h blocks | D-1 08:00 |
| dayahead | day-ahead auction of D | D-1 12:00 |
| ida1, ida2 | intraday auctions of D | D-1 15:00, D-1 22:00 |
| ida3 | intraday auction of D 12:00 to 24:00 | D 10:00 |
| idc | continuous intraday, one quarter hour q | q minus 15 min |

Every event optimises from the first quarter hour not yet delivered to the end of its day plus
`decision.lookahead_days`. It fixes only its own decision and leaves the rest as a plan. Later events take earlier
positions as given and may trade them back.

```text
soc = initial state of charge
for each event (time t, market, quarter hours) in time order:
    window = first quarter hour not delivered ... end of the delivery day + lookahead_days
    p = planning prices at t      # perfect: real prices; known: published prices, else the newest forecast
                                  # issued before t, else the day before (reserve, day-ahead) or the market before
    solve a MILP over the window:
        battery: power, state of charge from soc, efficiencies, charge or discharge per quarter hour
        reserve: 4-h blocks, power next to the schedule, state of charge covers the delivery time
                 (FCR 0.25 h, aFRR and mFRR 1 h), FCR at most 0.8 of the power
        earlier positions fixed, intraday may trade them back (up to twice the power)
        objective: revenue at p minus fee, on the continuous intraday also minus half the spread
    fix the decision of this event only
    idc: quarter hour delivered, soc carried on
settlement: all positions at the real prices, minus fee and half the spread
```

Revenues are always settled at the real prices; the planning prices only drive the decision. Revenue from buying in
one market and selling in another in the same quarter hour needs no battery and is reported separately
(`between_markets` in `summary.csv`).

## Stages

| Stage | Decision | Knowledge | Further switches |
|---|---|---|---|
| `s1` | one per day | perfect foresight | |
| `s1b` | one per day | naive | |
| `s1f` | one per day | day-ahead forecast | |
| `s3` | every gate | perfect foresight | |
| `s3o` | every gate | perfect foresight | no trading between markets |
| `s4` | every gate | naive | |
| `s4o` | every gate | naive | no trading between markets |
| `s5` | every gate | day-ahead forecast | |
| `s6n` | every gate | forecast per auction, naive continuous intraday | |
| `s6` | every gate | forecast per market | |
| `s7` | bids at every gate | forecast per market | an auction trade needs an awarded bid |
| `max` | bids at every gate | forecast per market | liquidity limit, balancing energy, price impact, reserve activation, outages, ageing, marketer share |

Naive: reserve and day-ahead prices as on the last known day, a later market at the price of the market before it.
Published prices are used once they are out. With bids, a buy is limited at the 90 % quantile of the forecast and a
sell at the 10 % quantile; what is not awarded is closed on the continuous intraday.

## Data formats

All times are UTC, quarter hours without gaps. To use other data, write files in these formats and set `data.*`.

`data.prices`, index `timestamp_utc`. The first two days are history for the planning and are not settled.

| Column | Unit | Content |
|---|---|---|
| `price_dayahead`, `price_ida1`, `price_ida2`, `price_ida3` | EUR/MWh | auction prices; IDA3 empty outside 12:00 to 24:00 |
| `price_id1` | EUR/MWh | continuous intraday price a trade is settled at |
| `price_fcr`, `price_afrr_pos`, `price_afrr_neg`, `price_mfrr_pos`, `price_mfrr_neg` | (EUR/MW)/h | capacity price received, constant per 4-h block |
| `price_afrr_pos_marginal` and the three alike | (EUR/MW)/h | highest accepted capacity price (pay-as-bid products), used for bids |

`data.forecasts`, one row per market, issue time and target quarter hour: `market` (`dayahead`, `ida1`, `ida2`,
`ida3`, `idc` or a reserve price column), `issue_time`, `target_time`, `value` (median), `q10`, `q90`. A reserve forecast
is the same for all quarter hours of a 4-h block.

`data.activation`, index `quarter_of_day` (0 to 95, local time): activated share of the reserved power in the columns
`aFRR_pos`, `aFRR_neg`, `mFRR_pos`, `mFRR_neg`.

## Assumptions

- A price is known from 30 min (intraday auctions) or 1 h (reserve, day-ahead) after its gate closes.
- Reserve follows the German prequalification rules (PQ-Bedingungen 2024): delivery time per product, FCR at most
  0.8 of the power. The three reserve auctions are decided together at 08:00.
- Activated reserve energy is worth the continuous intraday price; outages cost revenue in proportion to time.
- Balancing energy costs the continuous intraday price plus a surcharge.
- Values in `stage/max.yaml` and `config.yaml` marked as assumption have no source.
