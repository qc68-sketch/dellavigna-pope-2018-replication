#%% Setup
"""
Clean MTurk data for DellaVigna and Pope (2018), "What Motivates Effort?"
Sample restrictions follow Section 3.1.2.

Notes
- The paper does not report the outage boundaries. I infer one interval with elevated missing scores (Bernoulli LR scan, min 20 consecutive rows).
- Incomplete = finished != 1. The 30-minute rule from the pre-registration is not applied.
- Re-entry is proxied by MTurk IDs repeated among assigned records (all copies dropped).
- Unapproved = approved != 1 or missing MTurk ID.
- Treatment codes rounded to 1 decimal; only the 18 listed codes kept.

Outputs and run log go to ~/Desktop/replication/results/01_data_cleaning
(overwritten on each run).
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from scipy.special import xlogy

DATA_FILE = Path.home() / "Downloads/supplementary/CodeForPosting/input/MTurkData.dta"
OUT = Path.home() / "Desktop" / "replication" / "results" / "01_data_cleaning"
LOG_FILE = OUT / "mturk_run_log.txt"

VALID_TREATMENTS = [
    1.3, 1.1, 1.2, 1.4, 2.0, 3.1, 3.2, 10.0, 4.1,
    4.2, 5.1, 5.2, 5.3, 6.1, 6.2, 7.0, 8.0, 9.0,
]
MIN_WINDOW = 20

# Exclusion counts reported in the paper, keyed by our step names.
PAPER_DROPS = {
    "Inferred outage interval": 721,
    "Score above 4000": 48,
    "Not completed (finished != 1)": 1543,
    "Repeated ID among assigned records": 364,
    "Unapproved HIT or missing worker ID": 187,
    "Missing or nonpositive score": 114,
}
PAPER_FINAL_N = 9861
PAPER_RAW_N = PAPER_FINAL_N + sum(PAPER_DROPS.values())  # 12,838


#%% Helper functions
def check(condition, message):
    if not condition:
        raise ValueError(message)


def loglik(k, n):
    # Bernoulli log-likelihood at p = k/n
    p = k / n
    return xlogy(k, p) + xlogy(n - k, 1 - p)


def find_outage(is_missing, min_window=MIN_WINDOW):
    """Find one interval with an unusually high missing-score rate.
    Returns (start, end) positions in time order, end exclusive."""
    n = len(is_missing)
    cum = np.concatenate(([0], np.cumsum(is_missing)))
    k_all = cum[-1]
    base = loglik(k_all, n)

    best_gain, best_s, best_e = -np.inf, None, None
    for s in range(n - min_window + 1):
        e = np.arange(s + min_window, n + 1)
        e = e[e - s < n]  # keep at least one row outside
        n_in = e - s
        k_in = cum[e] - cum[s]
        n_out, k_out = n - n_in, k_all - k_in
        ok = k_in / n_in > k_out / n_out
        if not ok.any():
            continue

        gain = loglik(k_in[ok], n_in[ok]) + loglik(k_out[ok], n_out[ok]) - base
        i = gain.argmax()
        if gain[i] > best_gain:
            best_gain, best_s, best_e = gain[i], s, int(e[ok][i])

    check(best_s is not None, "No elevated-missingness interval found")
    return best_s, best_e


#%% Set up logging
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")],
    force=True,
)
log = logging.info


#%% Load data
raw = pd.read_stata(DATA_FILE)
needed = ["mturkid_code", "startdate", "buttonpresses", "finished", "approved", "treatment"]
missing_cols = [c for c in needed if c not in raw.columns]
check(not missing_cols, f"Missing columns: {missing_cols}")

raw["startdate"] = pd.to_datetime(raw["startdate"])
check(raw["startdate"].notna().all(), "Missing start dates")  # NaT breaks the scan

raw["raw_row"] = np.arange(1, len(raw) + 1)
raw["treatment_code"] = raw["treatment"].astype(float).round(1)

log("1. Raw data")
log(f"Rows: {len(raw):,} (implied by paper: {PAPER_RAW_N:,})")
log(f"Missing scores: {raw['buttonpresses'].isna().sum():,}")
log(f"Finished == 1: {(raw['finished'] == 1).sum():,}")
log(f"Approved == 1: {(raw['approved'] == 1).sum():,}")
log(f"Recognized treatments: {raw['treatment_code'].isin(VALID_TREATMENTS).sum():,}")
log(f"Date range: {raw['startdate'].min()} to {raw['startdate'].max()}")
log("")

audit = [("Raw sample", 0, len(raw))]
reason = pd.Series(pd.NA, index=raw.index, dtype="object")


#%% Infer the outage interval
srt = raw.sort_values("startdate", kind="stable")
s, e = find_outage(srt["buttonpresses"].isna().to_numpy(dtype=int))
outage_start = srt["startdate"].iloc[s]
outage_end = srt["startdate"].iloc[e - 1]

# timestamp ties can pull in a few rows beyond the scan window
in_outage = raw["startdate"].between(outage_start, outage_end)
df = raw.loc[~in_outage].copy()
reason[in_outage] = "Inferred outage interval"
audit.append(("Inferred outage interval", int(in_outage.sum()), len(df)))

log("2. Outage window")
log(f"Window: {outage_start} to {outage_end}")
log(f"Rows in scan window: {e - s:,}")
log(f"Rows removed: {in_outage.sum():,}")
log(f"Missing-score rate inside / outside: "
    f"{raw.loc[in_outage, 'buttonpresses'].isna().mean():.1%} / "
    f"{df['buttonpresses'].isna().mean():.1%}")
log("")


#%% Identify repeated IDs
assigned = df["treatment_code"].isin(VALID_TREATMENTS) & df["mturkid_code"].notna()
ids = df.loc[assigned, "mturkid_code"]
repeated_ids = ids[ids.duplicated(keep=False)].unique()
df["repeated_id"] = df["mturkid_code"].isin(repeated_ids)

log("3. Repeated IDs")
log(f"IDs: {len(repeated_ids):,}, rows flagged: {df['repeated_id'].sum():,}")
log("")


#%% Apply sample restrictions
# The order affects the sequential exclusion counts.
rules = [
    ("Score above 4000", df["buttonpresses"] > 4000),
    ("Not completed (finished != 1)", df["finished"] != 1),
    ("Repeated ID among assigned records", df["repeated_id"]),
    ("Unapproved HIT or missing worker ID",
     (df["approved"] != 1) | df["mturkid_code"].isna()),
    ("Missing or nonpositive score",
     df["buttonpresses"].isna() | (df["buttonpresses"] <= 0)),
    ("Missing/unrecognized treatment",
     ~df["treatment_code"].isin(VALID_TREATMENTS)),
]

keep = pd.Series(True, index=df.index)
for name, drop in rules:
    selected = keep & drop
    reason.loc[selected[selected].index] = name
    keep &= ~drop
    audit.append((name, int(selected.sum()), int(keep.sum())))

clean = df.loc[keep].copy()
audit = pd.DataFrame(audit, columns=["Step", "Dropped", "Remaining"])
audit["Paper"] = audit["Step"].map(PAPER_DROPS).astype("Int64")
audit["Diff"] = audit["Dropped"] - audit["Paper"]

n_excluded = len(raw) - len(clean)
check(audit["Dropped"].sum() == n_excluded, "Audit counts do not add up")
check(reason.notna().sum() == n_excluded, "Some excluded rows have no reason")
check(clean["treatment_code"].isin(VALID_TREATMENTS).all(), "Unrecognized treatment kept")

log("4. Exclusions")
log(audit.to_string(index=False))
log(f"\nFinal sample: {len(clean):,} (paper: {PAPER_FINAL_N:,}, "
    f"diff: {len(clean) - PAPER_FINAL_N:+,})")
log("")


#%% Save cleaned data and the exclusion audit
out = clean[["buttonpresses", "treatment_code"]].rename(columns={"treatment_code": "treatment"})
out.to_csv(OUT / "mturk_cleaned_independent.csv", index=False)
out.to_stata(OUT / "mturk_cleaned_independent.dta", write_index=False, version=118)

rows = raw[["raw_row"]].copy()
rows["excluded_inferred_outage"] = in_outage
rows["retained"] = raw.index.isin(clean.index)
rows["first_exclusion_reason"] = reason
rows.to_csv(OUT / "row_selection.csv", index=False)
audit.to_csv(OUT / "cleaning_audit.csv", index=False)

#%% Treatment summary
summary = (out.groupby("treatment")["buttonpresses"]
           .agg(["count", "mean", "std", "sem"])
           .sort_index())
summary.to_csv(OUT / "treatment_summary.csv")
check(summary["count"].sum() == len(clean), "Treatment counts do not sum to N")

log("5. By treatment")
log(summary.round(2).to_string())
log(f"\n{len(summary)} treatments, {summary['count'].sum():,} obs")

notes = {
    "source": "DellaVigna and Pope (2018), Section 3.1.2",
    "raw_file": str(DATA_FILE),
    "n_raw": len(raw),
    "n_clean": len(clean),
    "paper_final_n": PAPER_FINAL_N,
    "outage_start": str(outage_start),
    "outage_end": str(outage_end),
    "outage_method": f"single-interval Bernoulli LR scan, min {MIN_WINDOW} rows",
    "repeat_id_method": "all copies of IDs repeated among assigned records, after outage drop",
    "completion_method": "finished != 1; no 30-minute check",
    "approval_method": "approved != 1 or missing MTurk ID",
    "paper_reported_exclusions": PAPER_DROPS,
}
with (OUT / "cleaning_notes.json").open("w", encoding="utf-8") as f:
    json.dump(notes, f, indent=2)

log(f"\nOutputs saved to {OUT}")

