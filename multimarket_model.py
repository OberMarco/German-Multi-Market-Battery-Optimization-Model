"""Multi-market model of a battery in Germany with PyPSA: the model walks along the time line of the markets.

Each event is one decision:
  reserve     FCR, aFRR and mFRR capacity of day D            at D-1 08:00 (first of the three auctions)
  dayahead    day-ahead auction of D                          at D-1 12:00
  ida1, ida2  intraday auctions of D                          at D-1 15:00 and D-1 22:00
  ida3        intraday auction of D 12:00-24:00               at D 10:00
  idc         continuous intraday, one quarter hour q         at q - 15 min
With decision.gates: joint all markets of day D are decided in one event at D-1 08:00 instead.

Every event optimises from the first quarter hour not yet delivered to the end of its day plus
decision.lookahead_days, fixes only its own decision and leaves the rest as a plan. Later events take earlier
positions as given and may trade them back on their own market, so intraday markets may trade up to twice the battery
power. The state of charge carries over from quarter hour to quarter hour.

Reserve capacity is held back, not sold as energy: revenue = capacity price (EUR/MW)/h x reserved MW. Reserved power
must fit next to the schedule, and the state of charge must cover physics.reserve_hours of delivery per reserved MW,
in both directions. Reserve products are 4-h blocks in local time.

Planning prices (knowledge.decision_prices): perfect = all real prices; known = a price counts as known once it is
published. Unknown prices come from the newest forecast issued before the event (knowledge.forecasts); without one,
reserve and day-ahead repeat the last known day by time of day, and each later market is planned at the price of the
market before. Revenues are always settled at the real prices.

Run: python multimarket_model.py [+stage=s4] [run.days=2] [<axis>.<switch>=<value>]
"""
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from omegaconf import DictConfig

H = pd.Timedelta("1h")
TZ = "Europe/Berlin"
# energy market -> (real price column, gate closure relative to 00:00 of the delivery day, local time); planned in this
# order, each unknown price at the price of the market before it. idc closes per quarter hour. imbalance: what is left
# open at delivery is settled as balancing energy, taken as the continuous intraday price plus a surcharge.
MARKETS = {"dayahead": ("price_dayahead", -12 * H), "ida1": ("price_ida1", -9 * H), "ida2": ("price_ida2", -2 * H),
           "ida3": ("price_ida3", 10 * H), "idc": ("price_id1", None), "imbalance": ("price_id1", None)}
# reserve market -> (price column in (EUR/MW)/h, direction)
RESERVE_MARKETS = {"FCR": ("price_fcr", "sym"), "aFRR_pos": ("price_afrr_pos", "pos"),
                   "aFRR_neg": ("price_afrr_neg", "neg"), "mFRR_pos": ("price_mfrr_pos", "pos"),
                   "mFRR_neg": ("price_mfrr_neg", "neg")}
RESERVE_COLUMNS = [col for col, _ in RESERVE_MARKETS.values()]
PAY_AS_BID = {"aFRR_pos", "aFRR_neg", "mFRR_pos", "mFRR_neg"}  # FCR is pay-as-cleared
RESERVE_CLOSE = -16 * H  # D-1 08:00, FCR; aFRR (09:00) and mFRR (10:00) are decided at the same time
BLOCK_HOURS = 4  # reserve products are 4-h blocks (local time)
# assumption: results are known this long after the gate closes
PUBLISHED = {"dayahead": 1 * H, "ida1": H / 2, "ida2": H / 2, "ida3": H / 2, "reserve": 1 * H}
IDC_DECISION = pd.Timedelta("15min")  # last decision before the quarter hour
EVENT_ORDER = {"reserve": 0, "dayahead": 1, "ida1": 2, "ida2": 3, "ida3": 4, "joint": 5, "idc": 6}  # at the same time
HISTORY_DAYS = 2  # first days of the price file: known to the planning, not delivered and not settled
BID_LIMITS = {"q90": ("q90", "q10"), "median": ("value", "value")}  # forecast column of the buy and the sell limit
# assumption: penalty per MWh outside the reserve state-of-charge band, far above any energy price, so the band is
# only left when activated reserve forces it
SOC_BREACH_EUR_PER_MWH = 1000.0
# joint decision: where the plan prices of several markets are equal the solver would pick one at random. This tiny
# cost per later market (EUR/MWh, not settled) keeps the energy in the earliest market.
TIE_BREAK_EUR_PER_MWH = 0.01


def read_prices(raw):
    """Real price per energy market (EUR/MWh) and reserve market ((EUR/MW)/h), UTC index without time zone."""
    assert (raw.index.to_series().diff().dropna() == pd.Timedelta("15min")).all(), "price file is not 15 min without gaps"
    prices = raw.set_axis(raw.index.tz_convert("UTC").tz_localize(None))
    out = prices[RESERVE_COLUMNS].copy()
    for name, (col, _) in MARKETS.items():
        out[name] = prices[col]
    assert not out.drop(columns="ida3").isna().any().any(), "gaps in the prices"  # IDA3 is held only 12-24 h
    return out


def read_reserve_marginal(raw):
    """Highest accepted capacity price per reserve column ((EUR/MW)/h); FCR: its settlement price."""
    prices = raw.set_axis(raw.index.tz_convert("UTC").tz_localize(None))
    return pd.DataFrame({col: prices[f"{col}_marginal" if k in PAY_AS_BID else col]
                         for k, (col, _) in RESERVE_MARKETS.items()})


def read_forecasts(path, keys):
    """Forecast table per price column for the markets in keys (reserve = all reserve columns), indexed by issue time."""
    f = pd.read_parquet(path)
    columns = [c for k in keys for c in (RESERVE_COLUMNS if k == "reserve" else [k])]
    utc = lambda s: pd.to_datetime(s, utc=True).dt.tz_localize(None).values
    f = pd.DataFrame({"market": f.market.values, "target": utc(f.target_time), "value": f.value.values,
                      "q10": f.q10.values, "q90": f.q90.values},
                     index=pd.DatetimeIndex(utc(f.issue_time), name="issue")).sort_index()
    return {c: f[f.market == c].drop(columns="market") for c in columns}


def read_events(local, gates):
    """Decisions in time order: (time UTC, kind, quarter hours the event decides, as bool array)."""
    utc = local.tz_convert("UTC").tz_localize(None)
    day = local.normalize()
    events = []
    for d in day.unique():
        on_day, start = (day == d), d.tz_convert("UTC").tz_localize(None)
        if gates == "joint":
            events.append((start + RESERVE_CLOSE, "joint", on_day))
            continue
        events.append((start + RESERVE_CLOSE, "reserve", on_day))
        events += [(start + close, name, on_day) for name, (_, close) in MARKETS.items() if close is not None]
    if gates != "joint":
        events += [(t - IDC_DECISION, "idc", utc == t) for t in utc]
    return sorted(events, key=lambda e: (e[0], EVENT_ORDER[e[1]]))


def read_published(local):
    """Time (UTC) from which each real price is known, per column; continuous intraday: once the quarter hour starts."""
    utc = local.tz_convert("UTC").tz_localize(None)
    start = local.normalize().tz_convert("UTC").tz_localize(None)
    out = pd.DataFrame({name: start + close + PUBLISHED[name]
                        for name, (_, close) in MARKETS.items() if close is not None}, index=utc)
    out["idc"] = out["imbalance"] = utc
    for col in RESERVE_COLUMNS:
        out[col] = start + RESERVE_CLOSE + PUBLISHED["reserve"]
    return out


def read_day_profile(values, local, source_day, steps):
    """Values of source_day, placed on the given steps by local time of day (DST days filled from the neighbour)."""
    minute = local.hour * 60 + local.minute
    source = values[local.normalize() == source_day].set_axis(minute[local.normalize() == source_day])
    source = source[~source.index.duplicated()]  # the hour that repeats when the clocks go back
    target = pd.Index(minute[steps])
    return source.reindex(source.index.union(target.unique())).ffill().bfill().loc[target].values


def read_forecast(forecast, t, index, column="value"):
    """Newest forecast issued before t, on the given UTC steps (NaN where it has no value)."""
    if forecast is None:
        return np.full(len(index), np.nan)
    issues = forecast.index.unique()
    k = issues.searchsorted(t, side="right") - 1
    if k < 0:
        return np.full(len(index), np.nan)
    return forecast.loc[[issues[k]]].set_index("target")[column].reindex(index).values


def read_plan_known(prices, local, published, window, t, forecasts):
    """Prices an event at time t plans on: real where published, else the newest forecast, else the last known day
    (reserve, day-ahead) or the market before (intraday)."""
    p, real = prices[window].copy(), prices[window]
    known = published[window][list(p.columns)].le(t).values
    before = None
    for i, col in enumerate(p.columns):
        gap = ~known[:, i]
        p.loc[gap, col] = read_forecast(forecasts.get(col), t, p.index)[gap]
        gap &= p[col].isna().values
        seen = published[col].le(t).values & prices[col].notna().values
        if gap.any() and col in RESERVE_COLUMNS + ["dayahead"] and seen.any():
            p.loc[gap, col] = read_day_profile(prices[col], local, local[seen].normalize().max(),
                                               np.flatnonzero(window)[gap])
        elif gap.any() and before is not None:
            p.loc[gap, col] = p.loc[gap, before].values
        assert (p[col].notna() | real[col].isna()).all(), f"{col}: no price known before {t}"
        if col in MARKETS:  # IDA3 is held only 12-24 h, elsewhere the next market follows IDA2
            before = "ida3_or_ida2" if col == "ida3" else col
            if col == "ida3":
                p["ida3_or_ida2"] = p.ida3.where(real.ida3.notna(), p.ida2)
    p.loc[real.ida3.isna().values, "ida3"] = np.nan  # no auction, no trade
    return p.drop(columns="ida3_or_ida2")


def read_plan_perfect(prices, local, published, window, t, forecasts):
    """Perfect foresight: every event knows all real prices."""
    return prices[window].copy()


DECISION_PRICES = {"perfect": read_plan_perfect, "known": read_plan_known}


def read_costs(index, fee, spread, price_impact, imbalance_surcharge):
    """EUR per traded MWh and market: fee and price impact everywhere, plus half the bid-ask spread on continuous
    trades (a buy pays the ask, a sell gets the bid); imbalance: half the spread plus the surcharge."""
    costs = pd.DataFrame(fee + price_impact, index=index, columns=list(MARKETS))
    costs["idc"] += spread / 2
    costs["imbalance"] = spread / 2 + imbalance_surcharge
    return costs


def read_bid_limits(forecast, t, index, plan, limits):
    """Buy and sell limit per step: the forecast quantiles issued before t, else the planned price."""
    return [np.where(np.isnan(v), plan, v) for v in (read_forecast(forecast, t, index, c) for c in limits)]


def create_awards(kind, t, own_steps, x, prices, plan, forecasts, limits, marginal):
    """Bids of an auction event and what of them is awarded at the real price. x: {generator or reserve market: MW}
    as optimised. Energy: a buy is awarded if the clearing price is at most its limit, a sell if at least its limit.
    Reserve (an offer to sell capacity at the sell limit): awarded if the limit is at most the marginal price;
    pay-as-bid products are paid the limit. Returns awarded MW per key and, for reserve, the price paid ((EUR/MW)/h)."""
    awarded, paid = {}, {}
    if kind == "reserve":
        for k, (col, _) in RESERVE_MARKETS.items():
            _, sell = read_bid_limits(forecasts.get(col), t, own_steps, plan[col].values, limits)
            ok = sell <= marginal.loc[own_steps, col].values
            awarded[k] = np.where(ok, x[k], 0.0)
            paid[k] = np.where(ok, sell if k in PAY_AS_BID else prices.loc[own_steps, col].values, 0.0)
        return awarded, paid
    real = prices.loc[own_steps, kind].values
    buy, sell = read_bid_limits(forecasts.get(kind), t, own_steps, plan[kind].values, limits)
    awarded[f"{kind}_buy"] = np.where(real <= buy, x[f"{kind}_buy"], 0.0)
    awarded[f"{kind}_sell"] = np.where(real >= sell, x[f"{kind}_sell"], 0.0)
    return awarded, paid


def create_network(snapshots, prices, battery, soc_initial, fixed, costs, idc_cover, tie_break):
    """Battery and one buy and one sell generator per market; fixed: MW per step and side, NaN = open. idc_cover: the
    continuous intraday may also close what earlier markets left open (bids not awarded on one side)."""
    P = battery["power_mw"]
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.snapshot_weightings.loc[:, :] = battery["dt_h"]
    n.add("Bus", "bus")
    n.add("StorageUnit", "battery", bus="bus", p_nom=P, max_hours=battery["capacity_mwh"] / P,
          efficiency_store=battery["eff_charge"], efficiency_dispatch=battery["eff_discharge"],
          cyclic_state_of_charge=False, state_of_charge_initial=soc_initial,
          marginal_cost=battery["degradation_eur_per_mwh"])  # per MWh discharged
    # assumption: the continuous intraday is deep enough to close an open position on top of its usual limit
    open_position = np.abs(fixed.loc[snapshots].fillna(0).sum(axis=1).values) if idc_cover else 0.0
    base = {"dayahead": P, "idc": battery["idc_liquidity_mw"] or 2 * P,
            "imbalance": 4 * P if battery["imbalance"] else 0}  # imbalance: no real limit
    for name in MARKETS:
        limit = np.full(len(snapshots), float(base.get(name, 2 * P)))  # intraday may reverse a full position
        limit = limit + (open_position if name == "idc" else 0.0)
        p_nom = max(limit.max(), 1e-6)
        share = np.where(prices[name].notna().values, limit / p_nom, 0.0)  # IDA3 only 12-24 h
        for side, lo, hi, sign in (("buy", 0 * share, share, 1), ("sell", -share, 0 * share, -1)):
            lo, hi = lo.copy(), hi.copy()
            f = fixed.loc[snapshots, f"{name}_{side}"].values
            lo[~np.isnan(f)] = hi[~np.isnan(f)] = f[~np.isnan(f)] / p_nom
            n.add("Generator", f"{name}_{side}", bus="bus", p_nom=p_nom,
                  p_min_pu=pd.Series(lo, snapshots), p_max_pu=pd.Series(hi, snapshots),
                  marginal_cost=prices[name].fillna(0).values
                  + sign * (costs.loc[snapshots, name].values + tie_break * list(MARKETS).index(name)))
    return n


def read_battery_variable(m, name):
    """One StorageUnit variable of the battery, indexed by snapshot only."""
    v = m[f"StorageUnit-{name}"]
    return v.sel({[d for d in v.dims if d != "snapshot"][0]: "battery"})


def create_reserve(m, snapshots, prices, battery, blocks, fixed_reserve):
    """Reserve capacity per market with headroom and state-of-charge limits; its revenue goes into the objective.
    fixed_reserve: MW per market and step from an earlier event, NaN = still open."""
    P, E, h = battery["power_mw"], battery["capacity_mwh"], battery["reserve_hours"]
    p_net = read_battery_variable(m, "p_dispatch") - read_battery_variable(m, "p_store")
    soc = read_battery_variable(m, "state_of_charge")
    same_block = xr.DataArray(np.r_[False, blocks[1:] == blocks[:-1]], coords=[snapshots])
    pos, neg, energy_pos, energy_neg, revenue = 0, 0, 0, 0, 0
    for name, (col, direction) in RESERVE_MARKETS.items():
        lo = np.zeros(len(snapshots))
        hi = np.full(len(snapshots), P * (battery["fcr_power_share"] if name == "FCR" else 1.0))
        held = ~np.isnan(fixed_reserve[name])
        lo[held] = hi[held] = fixed_reserve[name][held]
        lo, hi = (xr.DataArray(a, coords=[snapshots]) for a in (lo, hi))
        r = m.add_variables(lower=lo, upper=hi, coords=[snapshots], name=f"reserve_{name}")
        revenue = revenue + (prices[col].values * battery["dt_h"] * r).sum()
        if direction in ("pos", "sym"):
            pos, energy_pos = pos + r, energy_pos + h[name] * r
        if direction in ("neg", "sym"):
            neg, energy_neg = neg + r, energy_neg + h[name] * r
        m.add_constraints(r - r.shift(snapshot=1) == 0, name=f"block_{name}", mask=same_block)  # same MW per block
        if battery["min_bid_mw"]:  # per product either 0 or at least the minimum bid
            z = m.add_variables(binary=True, coords=[snapshots], name=f"offered_{name}")
            m.add_constraints(r >= battery["min_bid_mw"] * z, name=f"min_bid_{name}")
            m.add_constraints(r <= P * z, name=f"bid_only_if_offered_{name}")
    # with activation the state of charge may leave the reserve band, and the battery power the headroom of the
    # reserved MW, at a penalty: activated reserve can push the state of charge to a limit where the reserved MW
    # cannot be delivered. Without activation both limits are hard.
    breach = short = 0
    if battery["activation"] is not None:
        breach = m.add_variables(lower=0, coords=[snapshots], name="soc_breach")
        short = m.add_variables(lower=0, coords=[snapshots], name="headroom_short")
        revenue = revenue - SOC_BREACH_EUR_PER_MWH * (breach.sum() + battery["dt_h"] * short.sum())
    m.add_constraints(p_net + pos - short <= P, name="headroom_pos")
    m.add_constraints(p_net - neg + short >= -P, name="headroom_neg")
    m.add_constraints(soc + breach >= energy_pos, name="energy_pos")
    m.add_constraints(soc - breach + energy_neg <= E, name="energy_neg")
    m.objective = m.objective - revenue


def create_one_side(m, snapshots, P, fixed):
    """Per step the open trades of all markets either buy or sell, at most the battery power (earlier positions do not
    count, so a later market may trade them back)."""
    p = m["Generator-p"]
    dim = [d for d in p.dims if d != "snapshot"][0]
    w = m.add_variables(binary=True, coords=[snapshots], name="buy_mode")
    for side, sign, bound in (("buy", 1, P * w), ("sell", -1, P * (1 - w))):
        names = [f"{k}_{side}" for k in MARKETS]
        is_open = xr.DataArray(fixed.loc[snapshots, names].isna().values.astype(float),
                               coords=[snapshots, names], dims=["snapshot", dim])
        m.add_constraints(sign * (p.sel({dim: names}) * is_open).sum(dim) <= bound, name=f"open_{side}_one_side")


def create_constraints(prices, battery, blocks, fixed_reserve, fixed, spread_trading):
    """Constraints beyond the PyPSA network: charge or discharge per step, reserve, trades on one side only."""
    def extra(n, snapshots):
        m, P = n.model, battery["power_mw"]
        if not spread_trading:
            create_one_side(m, snapshots, P, fixed)
        u = m.add_variables(binary=True, coords=[snapshots], name="charge_mode")
        m.add_constraints(read_battery_variable(m, "p_store") <= P * u, name="charge_only_in_charge_mode")
        m.add_constraints(read_battery_variable(m, "p_dispatch") <= P * (1 - u),
                          name="discharge_only_in_discharge_mode")
        create_reserve(m, snapshots, prices, battery, blocks, fixed_reserve)
    return extra


def create_activation(row, i, soc, battery, local, fixed_reserve):
    """Activated aFRR and mFRR in step i: share of the reserved MW by local quarter hour of day, on top of the
    schedule; the state of charge moves, the energy is settled in render_summary."""
    share = battery["activation"].loc[local[i].hour * 4 + local[i].minute // 15]
    dt = battery["dt_h"]
    pos = dt * sum(share[f"{p}_pos"] * fixed_reserve[f"{p}_pos"][i] for p in ("aFRR", "mFRR"))
    neg = dt * sum(share[f"{p}_neg"] * fixed_reserve[f"{p}_neg"][i] for p in ("aFRR", "mFRR"))
    pos = min(pos, soc * battery["eff_discharge"])
    neg = min(neg, (battery["capacity_mwh"] - soc) / battery["eff_charge"])
    row |= {"activated_pos_mwh": pos, "activated_neg_mwh": neg}
    row["soc"] = soc - pos / battery["eff_discharge"] + neg * battery["eff_charge"]
    return row["soc"]


def create_schedule(prices, local, battery, costs, decision_prices, forecasts, marginal, cfg):
    """Walk through all events in time order. Returns one row per delivered quarter hour with positions, reserve and
    battery. With decision.bids the auctions get bids, awarded by create_awards; else every optimised trade is done."""
    gates, bids = cfg.decision.gates, cfg.decision.bids
    first_day = local[0].normalize() + pd.DateOffset(days=HISTORY_DAYS)  # calendar days: a day has 23 to 25 h
    if cfg.run.days is not None:
        keep = local.normalize() < first_day + pd.DateOffset(days=cfg.run.days)
        prices, local, costs = prices[keep], local[keep], costs[keep]
    day = local.normalize()
    blocks = np.array([f"{t:%Y-%m-%d}_{t.hour // BLOCK_HOURS}" for t in local])
    published = read_published(local)
    fixed = pd.DataFrame(np.nan, prices.index, [f"{k}_{s}" for k in MARKETS for s in ("buy", "sell")])
    fixed_reserve = {k: np.full(len(prices), np.nan) for k in RESERVE_MARKETS}
    paid = pd.DataFrame({k: prices[col] for k, (col, _) in RESERVE_MARKETS.items()})  # capacity price received
    done = np.asarray(local < first_day)  # quarter hours delivered or history: every position final
    soc = battery["soc_initial"] * battery["capacity_mwh"]
    rows, not_optimal = {}, []
    for t, kind, target in read_events(local, gates):
        if done[target].all():
            continue
        window = ~done & (day <= day[target].max() + pd.DateOffset(days=cfg.decision.lookahead_days))
        snapshots, own = prices.index[window], target[window]
        p = decision_prices(prices, local, published, window, t, forecasts)
        n = create_network(snapshots, p, battery, soc, fixed, costs, idc_cover=bids is not None,
                           tie_break=TIE_BREAK_EUR_PER_MWH if kind == "joint" else 0.0)
        status = n.optimize(solver_name=cfg.solver.name, **dict(cfg.solver.options),
                            extra_functionality=create_constraints(
                                p, battery, blocks[window], {k: v[window] for k, v in fixed_reserve.items()}, fixed,
                                cfg.markets.spread_trading))
        if "infeasible" in status[1] or "unbounded" in status[1]:
            raise RuntimeError(f"event {kind} at {t} has no solution: {status}")
        if status != ("ok", "optimal"):
            not_optimal.append((str(t), kind, status))
        x = {k: n.model.variables[f"reserve_{k}"].solution.values[own] for k in RESERVE_MARKETS}
        x |= {g: n.generators_t.p[g].values[own] for g in n.generators.index}
        if bids is not None and kind in ("reserve", "dayahead", "ida1", "ida2", "ida3"):
            x, price = create_awards(kind, t, snapshots[own], x, prices, p[own], forecasts, BID_LIMITS[bids], marginal)
            for k, v in price.items():
                paid.loc[snapshots[own], k] = v
        if kind in ("reserve", "joint"):
            for k in RESERVE_MARKETS:
                fixed_reserve[k][np.flatnonzero(window)[own]] = x[k]
        decided = {"joint": list(MARKETS), "idc": ["idc", "imbalance"], "reserve": []}.get(kind, [kind])
        for name in decided:
            for s in ("buy", "sell"):
                fixed.loc[snapshots[own], f"{name}_{s}"] = x[f"{name}_{s}"]
        if kind in ("joint", "idc"):  # these quarter hours are delivered now
            for q in snapshots[own]:
                rows[q] = {"p_battery": n.storage_units_t.p.at[q, "battery"],
                           "soc": n.storage_units_t.state_of_charge.at[q, "battery"]}
                if battery["activation"] is not None:
                    rows[q]["soc_breach_mwh"] = float(n.model.variables["soc_breach"].solution.sel(snapshot=q))
                    rows[q]["headroom_short_mw"] = float(n.model.variables["headroom_short"].solution.sel(snapshot=q))
            done |= target
            soc = rows[snapshots[own][-1]]["soc"]
            if battery["activation"] is not None:
                for q in snapshots[own]:
                    soc = create_activation(rows[q], prices.index.get_loc(q), soc, battery, local, fixed_reserve)
    print(f"events not solved to optimality (time limit, best solution found is used): {not_optimal}")
    res = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    for name in MARKETS:
        res[f"{name}_buy_mw"] = fixed.loc[res.index, f"{name}_buy"]
        res[f"{name}_sell_mw"] = fixed.loc[res.index, f"{name}_sell"]
        res[f"{name}_mw"] = res[f"{name}_buy_mw"] + res[f"{name}_sell_mw"]
    for k in RESERVE_MARKETS:
        res[f"reserve_{k}_mw"] = pd.Series(fixed_reserve[k], prices.index).loc[res.index]
        res[f"reserve_{k}_price"] = paid.loc[res.index, k]
    return res


def read_between_markets(prices, res, dt_h):
    """MWh bought and sold in the same quarter hour over all energy markets (matched = min of both sides) and their
    revenue at the volume-weighted sell price minus the volume-weighted buy price of that quarter hour."""
    buy = pd.concat([res[f"{k}_buy_mw"] for k in MARKETS], axis=1).fillna(0).values
    sell = -pd.concat([res[f"{k}_sell_mw"] for k in MARKETS], axis=1).fillna(0).values
    p = prices[list(MARKETS)].fillna(0).values
    b, s = buy.sum(1), sell.sum(1)
    matched = np.minimum(b, s)
    price_buy = np.divide((buy * p).sum(1), b, out=np.zeros_like(b), where=b > 0)
    price_sell = np.divide((sell * p).sum(1), s, out=np.zeros_like(s), where=s > 0)
    return dt_h * matched.sum(), dt_h * (matched * (price_sell - price_buy)).sum()


def render_summary(prices, res, battery, costs):
    """Revenue per market at the real prices, after fees and spread (costs: EUR per traded MWh)."""
    dt_h = battery["dt_h"]
    rows = {}
    for k in MARKETS:
        traded = dt_h * (res[f"{k}_buy_mw"] - res[f"{k}_sell_mw"])
        rows[k] = {"revenue_eur": -dt_h * (res[f"{k}_mw"] * prices[k].fillna(0)).sum(),
                   "cost_eur": (costs.loc[res.index, k] * traded).sum(), "volume_mwh": traded.sum()}
    for k in RESERVE_MARKETS:
        rows[f"reserve_{k}"] = {"revenue_eur": dt_h * (res[f"reserve_{k}_mw"] * res[f"reserve_{k}_price"]).sum(),
                                "cost_eur": 0.0, "volume_mwh": 0.0}
    if battery["activation"] is not None:  # assumption: activated energy is worth the continuous intraday price
        energy = (res.activated_pos_mwh - res.activated_neg_mwh) * prices.idc
        rows["activation"] = {"revenue_eur": energy.sum(), "cost_eur": 0.0,
                              "volume_mwh": (res.activated_pos_mwh + res.activated_neg_mwh).sum()}
        rows["soc_band_breach"] = {"revenue_eur": 0.0, "cost_eur": 0.0, "volume_mwh": res.soc_breach_mwh.sum()}
        rows["reserve_short"] = {"revenue_eur": 0.0, "cost_eur": 0.0, "volume_mwh": dt_h * res.headroom_short_mw.sum()}
    discharged = dt_h * res.p_battery.clip(lower=0).sum()
    rows["degradation"] = {"revenue_eur": 0.0, "cost_eur": battery["degradation_eur_per_mwh"] * discharged,
                           "volume_mwh": discharged}
    summary = pd.DataFrame(rows).T
    summary["net_eur"] = summary.revenue_eur - summary.cost_eur
    gross = summary.net_eur.sum()
    lost = (1 - battery["availability"]) * gross  # assumption: outages cost revenue in proportion to time
    share = battery["marketer_share"] * (gross - lost)
    summary.loc["availability"] = {"revenue_eur": -lost, "cost_eur": 0.0, "volume_mwh": 0.0, "net_eur": -lost}
    summary.loc["marketer"] = {"revenue_eur": 0.0, "cost_eur": share, "volume_mwh": 0.0, "net_eur": -share}
    total = summary.net_eur.sum()
    # part of the energy revenue above that comes from buying in one market and selling in another in the same
    # quarter hour, without the battery: reported, not added to the total
    matched, between = read_between_markets(prices, res, dt_h)
    summary.loc["between_markets"] = {"revenue_eur": between, "cost_eur": 0.0, "volume_mwh": matched, "net_eur": between}
    hours = len(res) * dt_h
    per_mw_year = 8760 / hours / battery["power_mw"] / 1000
    print(summary.round(2).to_string())
    print(f"\nNet revenue in {hours:.0f} h: {total:.2f} EUR (state of charge at the end {res.soc.iloc[-1]:.2f} MWh, "
          f"not valued), cycles per day {discharged / battery['capacity_mwh'] / (hours / 24):.2f}, annualised "
          f"{total * per_mw_year:.0f} k EUR/MW/year, of which between markets {between * per_mw_year:.0f}")
    return summary


def render_schedule(prices, res, path, days=7):
    """Plot of the first days in local time: prices, sold MW per market, battery power, state of charge, reserve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    colors = {"dayahead": "#2a78d6", "ida": "#eb6834", "idc": "#1baf7a", "battery": "#52514e",
              "FCR": "#6250d6", "aFRR": "#e87ba4", "mFRR": "#008300"}
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e1", "grid.linewidth": 0.5})
    r, p = res.iloc[:days * 96], prices.loc[res.index[:days * 96]]
    x = r.index.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    sold = {"dayahead": -r.dayahead_mw, "ida": -r[["ida1_mw", "ida2_mw", "ida3_mw"]].fillna(0).sum(axis=1),
            "idc": -r.idc_mw}
    names = ["price", "dayahead", "ida", "idc", "battery", "soc", "reserve"]
    fig, axes = plt.subplots(len(names), 1, figsize=(7.2, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 0.8, 0.8, 0.8, 0.8, 0.8, 1], "hspace": 0.35})
    ax = dict(zip(names, axes))
    step = dict(where="post", linewidth=0.9)
    ax["price"].step(x, p.dayahead, color=colors["dayahead"], label="day-ahead", **step)
    for k, style in (("ida1", "-"), ("ida2", "--"), ("ida3", ":")):
        ax["price"].step(x, p[k], color=colors["ida"], linestyle=style, label=k.upper(), **step)
    ax["price"].step(x, p.idc, color=colors["idc"], label="IDC", alpha=0.8, **step)
    ax["price"].legend(ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    for k, v in sold.items():
        ax[k].fill_between(x, v, step="post", color=colors[k], linewidth=0, alpha=0.85)
    ax["battery"].fill_between(x, r.p_battery, step="post", color=colors["battery"], linewidth=0)
    ax["soc"].plot(x, r.soc, color=colors["battery"], linewidth=1.2)
    for k, v in (("FCR", r.reserve_FCR_mw), ("aFRR", r.reserve_aFRR_pos_mw), ("mFRR", r.reserve_mFRR_pos_mw)):
        ax["reserve"].step(x, v, color=colors[k], label=k, **step)  # positive products and FCR up, negative down
    for k in ("aFRR", "mFRR"):
        ax["reserve"].step(x, -r[f"reserve_{k}_neg_mw"], color=colors[k], **step)
    ax["reserve"].legend(ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=6)
    labels = {"price": "EUR/MWh", "dayahead": "day-ahead\nsold MW", "ida": "IDA1-3\nsold MW", "idc": "IDC\nsold MW",
              "battery": "battery MW\n(+ discharge)", "soc": "state of\ncharge MWh", "reserve": "reserve MW\n(+ pos, - neg)"}
    for k, label in labels.items():
        ax[k].set_ylabel(label, rotation=0, ha="right", va="center")
    ax["reserve"].xaxis.set_major_locator(mdates.DayLocator())
    ax["reserve"].xaxis.set_major_formatter(mdates.DateFormatter("%a %d.%m."))
    ax["reserve"].set_xlim(x[0], x[-1] + pd.Timedelta("15min"))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    root = Path(hydra.utils.get_original_cwd())
    out = root / cfg.run.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(root / cfg.data.prices)
    prices, local = read_prices(raw), raw.index.tz_convert(TZ)
    b, mk, ph, co = cfg.battery, cfg.markets, cfg.physics, cfg.costs
    if cfg.decision.gates == "joint" and (ph.activation or cfg.decision.bids):
        raise ValueError("physics.activation and decision.bids need decision.gates=real")
    battery = dict(power_mw=b.power_mw, capacity_mwh=b.capacity_mwh, soc_initial=b.initial_soc,
                   eff_charge=b.charge_efficiency, eff_discharge=b.discharge_efficiency, dt_h=0.25,
                   reserve_hours=dict(ph.reserve_hours), fcr_power_share=ph.fcr_power_share, min_bid_mw=ph.min_bid_mw,
                   availability=ph.availability, idc_liquidity_mw=mk.idc_liquidity_mw, imbalance=mk.imbalance,
                   activation=pd.read_parquet(root / cfg.data.activation) if ph.activation else None,
                   degradation_eur_per_mwh=co.degradation_eur_per_mwh, marketer_share=co.marketer_share)
    costs = read_costs(prices.index, co.fee_eur_per_mwh, co.spread_eur_per_mwh, mk.price_impact_eur_per_mwh,
                       mk.imbalance_surcharge_eur_per_mwh)
    keys = list(cfg.knowledge.forecasts)
    forecasts = read_forecasts(root / cfg.data.forecasts, keys) if keys else {}
    res = create_schedule(prices, local, battery, costs, DECISION_PRICES[cfg.knowledge.decision_prices], forecasts,
                          read_reserve_marginal(raw) if cfg.decision.bids else None, cfg)
    summary = render_summary(prices.loc[res.index], res, battery, costs)
    res.to_csv(out / "results.csv")
    summary.to_csv(out / "summary.csv")
    render_schedule(prices, res, out / "schedule.pdf")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
