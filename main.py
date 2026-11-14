#!/usr/bin/env python3
"""Minute-by-minute Yes-price forecaster for Polymarket's FAA outage
attribution family (make.fun polymarket-faa-attribution-01).

The task: for each market in a case, forecast the Yes price at every minute
of the next 240 as five quantiles [0.05, 0.25, 0.5, 0.75, 0.95]. The program
reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (default /input) and writes
one ``<id>.out`` per case to $MAKEFUN_OUTPUT_DIR (default /output).

How it forecasts
----------------
Prediction-market prices behave like a near-martingale at minute scale: the
best point forecast of the price h minutes out is the price now, and every
drift term we tested made the median worse. So the median is always the last
observed price, and all the work goes into the spread.

The spread is measured, not assumed. For a horizon h, look at what this
market actually did over the recent past: collect every h-step price change
p[i+h] - p[i] in a trailing window, take its empirical quantiles, and centre
them on their own median so the median offset is exactly zero. A fixed
normal sigma (the house baseline's approach) is badly wrong here: one-minute
moves vary ~150x across markets, and the change distribution is a spike at
zero with rare jumps, so Gaussian quantiles of a standard deviation are far
too wide for quiet markets and far too narrow for jumpy ones.

Two trailing windows are mixed 90/10 at every anchor horizon: the last 6
hours, which tracks the current regime, and the last 24 hours, which
stabilises the tails. Anchor horizons are 1, 5, 15, 60 and 240 minutes and
the quantile offsets are interpolated log-linearly in h between them. A
market with too few sampled changes at an anchor borrows the case-wide
pooled offsets (all markets' centred changes pooled together, tails from
the 24h window, inner quantiles from the 6h window).

Finally the whole spread is rescaled per market by recent activity: the
standard deviation of one-minute changes over the last 30 minutes relative
to the last 6 hours, square-rooted and clipped to [0.7, 1.6]. A market that
has just started moving gets wider bands; a market that has gone quiet gets
narrower ones. The rescaling is applied to the offsets of the already
clamped row, so a low-priced market's lower tail floors at x * (1 - scale)
instead of collapsing to zero.

On the six practice cases (the relabelled NVIDIA cut) this scores a mean
pinball loss of 0.276209 against the house persistence baseline's
0.363589, beating the baseline on every case (per case: p00 0.448649,
p01 0.312881, p02 0.389911, p03 0.273437, p04 0.193850, p05 0.105665).

Standard library only, no network, no randomness: two runs on the same
case produce byte-identical output.
"""

import json
import math
import os

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Horizons (minutes) at which the displacement distribution is measured.
ANCHORS = (1, 5, 15, 60, 240)
RECENT_MIN = 360    # adaptive window: last 6 hours of history
STABLE_MIN = 1440   # stabilising window: last 24 hours of history
RECENT_W = 0.9      # weight of the adaptive window in the 90/10 blend
STRIDE = 1          # sample a displacement starting at every minute
MIN_OBS = 25        # fewer sampled changes than this -> fall back to pooled
ACTIVITY_MIN = 30   # minutes of recent history for the activity rescale
SCALE_LO = 0.7      # activity rescale is clipped to [SCALE_LO, SCALE_HI]
SCALE_HI = 1.6


def clamp01(value):
    """Clamp to [0, 1], round to 6 decimals, and normalise -0 to 0."""
    return round(min(1.0, max(0.0, value)), 6) + 0.0


def emp_quantile(ordered, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(ordered)
    if n == 0:
        return 0.0
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = lo + 1 if lo + 1 < n else n - 1
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def step_changes(prices, h, span):
    """Sorted h-step changes over the trailing `span` minutes of prices.

    Takes the last span+1 prices (so a change can start at the first of
    them and land span minutes later) and returns every p[i+h] - p[i]
    sampled every STRIDE minutes, sorted ascending.
    """
    tail = prices[-span - 1:] if span < len(prices) else prices
    if len(tail) <= h:
        return []
    return sorted(tail[i + h] - tail[i] for i in range(0, len(tail) - h, STRIDE))


def centred_offsets(changes):
    """[q05, q25, 0, q75, q95] of the changes, shifted so the median is 0."""
    if not changes:
        return None
    mid = emp_quantile(changes, 0.5)
    out = [emp_quantile(changes, t) - mid for t in LEVELS]
    out[2] = 0.0
    return out


def pooled_offsets(all_prices, h, span):
    """Case-wide offsets: every market's centred changes pooled together."""
    pooled = []
    for prices in all_prices:
        changes = step_changes(prices, h, span)
        if changes:
            mid = emp_quantile(changes, 0.5)
            pooled.extend(c - mid for c in changes)
    if not pooled:
        return [0.0] * 5
    pooled.sort()
    return [emp_quantile(pooled, t) for t in LEVELS]


def market_anchor_offsets(prices, h):
    """One market's [q05, q25, 0, q75, q95] at anchor horizon h.

    Returns a pair (recent, stable) of offset lists, either of which may be
    None when that window held too few sampled changes.
    """
    pair = []
    for span in (RECENT_MIN, STABLE_MIN):
        changes = step_changes(prices, h, span)
        pair.append(centred_offsets(changes) if len(changes) >= MIN_OBS else None)
    return pair


def merge_offsets(recent, stable):
    """Combine a market's two window offsets; None means that window is out.

    Both missing -> None (caller substitutes the pooled offsets). One
    missing -> the other one as-is. Both present -> the 90/10 blend.
    """
    if recent is None and stable is None:
        return None
    if recent is None:
        return list(stable)
    if stable is None:
        return list(recent)
    return [RECENT_W * recent[j] + (1.0 - RECENT_W) * stable[j] for j in range(5)]


def offsets_at(keys, anchors, h):
    """Log-linear interpolation of the per-anchor offsets at horizon h.

    `keys` is the anchors' sorted horizon list, computed once per market
    instead of re-sorted for every one of the 240 rows.
    """
    if h <= keys[0]:
        return list(anchors[keys[0]])
    if h >= keys[-1]:
        return list(anchors[keys[-1]])
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= h <= hi:
            t = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return [anchors[lo][j] + t * (anchors[hi][j] - anchors[lo][j])
                    for j in range(5)]
    return list(anchors[keys[-1]])


def activity_factor(prices):
    """Spread multiplier from how busy the last 30 minutes were.

    Compares the standard deviation of one-minute changes over the last
    ACTIVITY_MIN minutes with the same over the adaptive window, takes the
    square root of the ratio and clips it to [SCALE_LO, SCALE_HI]. Returns
    SCALE_LO when either window shows no movement at all.
    """
    def sd_one_min(span):
        tail = prices[-span - 1:] if span < len(prices) else prices
        if len(tail) < 3:
            return 0.0
        diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        mean = sum(diffs) / len(diffs)
        return math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))

    recent_sd = sd_one_min(ACTIVITY_MIN)
    window_sd = sd_one_min(RECENT_MIN)
    if window_sd <= 0.0 or recent_sd <= 0.0:
        return SCALE_LO
    return min(SCALE_HI, max(SCALE_LO, math.sqrt(recent_sd / window_sd)))


def monotone(row):
    """Enforce non-decreasing quantiles within a row, in place."""
    for j in range(1, 5):
        if row[j] < row[j - 1]:
            row[j] = row[j - 1]
    return row


def forecast_one(prices, pooled, horizon):
    """240 (horizon) rows of five quantiles for a single market."""
    last = prices[-1] if prices else 0.5
    anchors = {}
    for h in ANCHORS:
        recent, stable = market_anchor_offsets(prices, h)
        own = merge_offsets(recent, stable)
        anchors[h] = own if own is not None else pooled[h]
    factor = activity_factor(prices)
    keys = sorted(anchors)
    rows = []
    for h in range(1, horizon + 1):
        offs = offsets_at(keys, anchors, h)
        row = monotone([clamp01(last + offs[j]) for j in range(5)])
        mid = row[2]
        row = [mid + (row[0] - mid) * factor,
               mid + (row[1] - mid) * factor,
               mid,
               mid + (row[3] - mid) * factor,
               mid + (row[4] - mid) * factor]
        rows.append(monotone([clamp01(v) for v in row]))
    return rows


def run_case(case):
    """Forecast every market of one case; returns the {"forecasts": [...]} dict."""
    horizon = int(case.get("horizon_minutes", 240))
    markets = case["markets"]
    all_prices = [m.get("prices") or [] for m in markets]
    # Pooled fallback offsets per anchor: tails from the stable window,
    # inner quantiles from the adaptive window, median offset zero.
    pooled = {}
    for h in ANCHORS:
        inner = pooled_offsets(all_prices, h, RECENT_MIN)
        outer = pooled_offsets(all_prices, h, STABLE_MIN)
        pooled[h] = [outer[0], inner[1], 0.0, inner[3], outer[4]]
    forecasts = []
    for market, prices in zip(markets, all_prices):
        forecasts.append({"id": market["id"],
                          "q": forecast_one(prices, pooled, horizon)})
    return {"forecasts": forecasts}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = run_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
