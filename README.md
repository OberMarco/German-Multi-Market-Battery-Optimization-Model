# German Multi-Market Battery Optimization Model

A battery trades on the German electricity markets: reserve capacity (FCR, aFRR, mFRR), day-ahead, the intraday
auctions IDA1 to IDA3 and the continuous intraday (IDC), in quarter hours. The model walks along the time line of the
markets and takes each decision with what is known at that moment. It is built with PyPSA and solved with HiGHS.

Stages lead from one decision with perfect foresight to bids at every gate under near-real conditions. Neighbouring
stages differ in few switches, so the difference between two stages is the value of those switches (see Stages).

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

One delivery day takes one to two minutes with a decision at every gate and a few seconds with one decision per day;
the default call computes the seven delivery days of the example data (about nine minutes). Start the model from the
repository folder. A run writes `results.csv` (one row per quarter hour), `summary.csv`
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

Terms of the table:

- One per day: all markets of day D are decided together at D-1 08:00 (`decision.gates: joint`). Every gate: the events
  of the table in Model, each at its own time (`decision.gates: real`).
- Perfect foresight: every decision plans with the real prices (`knowledge.decision_prices: perfect`). In all other
  stages a price is used once it is published; before that it is planned as follows.
- Naive: reserve and day-ahead prices as on the last known day, a later market at the price of the market before it.
- Forecast: the newest forecast of `data.forecasts` issued before the decision, for the markets listed in
  `knowledge.forecasts`; the other markets stay naive.
- Bids (`decision.bids: q90`): an auction buy is limited at the 90 % quantile of the forecast and a sell at the 10 %
  quantile, a reserve offer at the 10 % quantile. A bid is awarded if the real clearing price meets its limit; what
  is not awarded is closed on the continuous intraday.

What each stage sets, relative to `config.yaml`, and what the step shows:

| Stage | Settings | Compared with | The difference is |
|---|---|---|---|
| `s1` | `gates: joint`, `decision_prices: perfect` | | upper bound: all prices known, one decision |
| `s1b` | `gates: joint` | `s1` | the value of knowing the prices |
| `s1f` | `gates: joint`, `forecasts: [dayahead]` | `s1b` | the value of a day-ahead forecast in one decision |
| `s3` | `decision_prices: perfect` | `s1` | the effect of deciding at the gates instead of once |
| `s3o` | `decision_prices: perfect`, `spread_trading: false` | `s3` | the revenue from trading between markets |
| `s4` | none (the defaults) | `s3` | the value of knowing the prices at every gate |
| `s4o` | `spread_trading: false` | `s4` | trading between markets without foresight |
| `s5` | `forecasts: [dayahead]` | `s4` | the value of a day-ahead forecast |
| `s6n` | `forecasts: [dayahead, ida1, ida2, ida3, reserve]` | `s5` | the value of forecasts for the auctions and the reserve |
| `s6` | as `s6n` plus `idc` | `s6n` | the value of a continuous intraday forecast |
| `s7` | as `s6` plus `bids: q90` | `s6` | the cost of needing an awarded bid |
| `max` | as `s7` plus the switches below | `s7` | the cost of the remaining conditions |

`max` adds: a liquidity limit of the continuous intraday (`markets.idc_liquidity_mw`), balancing energy for what the
continuous intraday cannot close (`markets.imbalance`), a price impact per traded MWh, activation of aFRR and mFRR
(`physics.activation`), outages (`physics.availability`), ageing cost per MWh discharged and the marketer share. Each
of them can be set alone on any stage with gates, for example `+stage=s7 physics.activation=true`.

## Results on real prices

Net revenue of this model on real market prices of September and October 2026, for a battery of 1 MW and 2 MWh
(0.5 C), in k EUR per MW and year, extrapolated from the month. "Between markets" is the part of the net revenue
that comes from buying and selling the same quarter hour on two markets, which needs no battery.

| Stage | Sep 2026: net revenue | Sep 2026: between markets | Oct 2026: net revenue | Oct 2026: between markets |
|---|---|---|---|---|
| `s1` | 1001 | 724 | 786 | 632 |
| `s1b` | 255 | 0 | 166 | 0 |
| `s1f` | 274 | 0 | 174 | 0 |
| `s3` | 1001 | 724 | 786 | 632 |
| `s3o` | 419 | 60 | 280 | 67 |
| `s4` | 258 | -1 | 167 | 0 |
| `s4o` | 258 | 0 | 171 | 1 |
| `s5` | 274 | 0 | 174 | 0 |
| `s6n` | 273 | 16 | 161 | 6 |
| `s6` | 274 | 28 | 176 | 32 |
| `s7` | 245 | 41 | 171 | 33 |
| `max` | 200 | 28 | 132 | 18 |

The prices and the forecasts of these runs (a time series foundation model) are not part of the repository, so the
table cannot be reproduced with the example data. On 25 October 2026 (clock change, 25 hours) these runs planned
without the look-ahead to the following day; the model in this repository plans with it, so a new run of October can
differ slightly.

## Data formats

All times are UTC, quarter hours without gaps, starting at 00:00 local time (Europe/Berlin). To use other data,
write files in these formats and set `data.*`.

`data.prices`, index `timestamp_utc`. The first two days are history for the planning and are not settled.

| Column | Unit | Content |
|---|---|---|
| `price_dayahead`, `price_ida1`, `price_ida2`, `price_ida3` | EUR/MWh | auction prices; IDA3 empty outside 12:00 to 24:00 |
| `price_id1` | EUR/MWh | continuous intraday price a trade is settled at |
| `price_fcr`, `price_afrr_pos`, `price_afrr_neg`, `price_mfrr_pos`, `price_mfrr_neg` | (EUR/MW)/h | capacity price received, constant per 4-h block |
| `price_afrr_pos_marginal` and the three alike | (EUR/MW)/h | highest accepted capacity price (pay-as-bid products), needed only with `decision.bids` |

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
- Balancing energy costs the continuous intraday price plus half the spread and a surcharge.
- With reserve activation the plan does not know the activation. The state of charge can then leave the band the
  reserve needs; this is reported (`soc_band_breach`, `reserve_short` in `summary.csv`) but not penalised in the
  settlement, the capacity price is paid in full.
- With bids the liquidity limit of the continuous intraday rises by the open auction position that has to be closed.
- Ageing counts the scheduled discharge, not the activated reserve energy.
- Values in `stage/max.yaml` and `config.yaml` marked as assumption have no source.

## Licence

Copyright (c) 2026 Marco Obermeier. Code, documentation and example data are licensed under the MIT License, see
`LICENSE`.
