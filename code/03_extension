#%% Setup
"""Extension to DellaVigna and Pope (2018): out-of-sample test of the cost function.

The benchmark model has three parameters and is fit to three group means, so
its in-sample fit is exact and says nothing about whether the cost function is
right. Here the model is fit on the 0-, 1- and 4-cent groups only, and its
prediction for the 10-cent group, left out of estimation, is compared with
observed effort. Uses the authors' cleaned data.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

DATA_FOLDER = Path.home() / "Downloads/supplementary/CodeForPosting/dtafiles"
OUTPUT_FOLDER = Path.home() / "Desktop" / "replication" / "extension"
LOG_FILE = OUTPUT_FOLDER / "extension.log"

BOOTSTRAP_DRAWS = 1000
RANDOM_SEED = 20261026
METHODS = ["Power MD", "Exponential MD"]

# Treatment codes and piece rates (dollars per 100 points)
TRAINING_CODES = [1.3, 1.1, 1.4]               # no pay, 1 cent, 4 cents
TRAINING_RATES = np.array([0.0, 0.01, 0.04])
HOLDOUT_CODE = 1.2                             # 10 cents
HOLDOUT_RATE = 0.10

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_FILE.write_text("DellaVigna and Pope (2018): high-pay holdout exercise\n",
                    encoding="utf-8")


def log_print(message=""):
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(str(message) + "\n")


def start_table(name):
    log_print("\n" + "=" * 72)
    log_print(f"{name} | {datetime.now():%Y-%m-%d %H:%M:%S}")
    log_print("=" * 72)


def save_table(table, name, title):
    log_print(f"\n{title}")
    log_print(table.to_string(index=False))
    table.to_csv(OUTPUT_FOLDER / f"{name}.csv", index=False)


#%% Model functions
def fit_md(mean_effort, rates, method):
    """Solve for (log gamma, log k, log s) from three group means.

    Power:       gamma * log(mean effort) = log(s + p) - log(k)
    Exponential: gamma * mean effort      = log(s + p) - log(k)
    The power version uses log(mean effort), not mean(log effort).
    """
    moments = np.log(mean_effort) if method == "Power MD" else np.asarray(mean_effort)
    first_change = moments[1] - moments[0]
    if first_change <= 0:
        raise ValueError("Mean effort must rise from zero to the first positive rate.")
    # With s > 0, the model implies this ratio lies strictly between 1 and
    # high rate / low rate; outside that range there is no interior solution.
    observed_ratio = (moments[2] - moments[0]) / first_change
    if not 1 < observed_ratio < rates[2] / rates[1]:
        raise ValueError("Training means imply no positive interior solution.")

    # Eliminating gamma and k leaves one equation in log(s).
    def ratio_gap(log_s):
        low = np.logaddexp(log_s, np.log(rates[1])) - log_s
        high = np.logaddexp(log_s, np.log(rates[2])) - log_s
        return high / low - observed_ratio

    log_s = brentq(ratio_gap, -700.0, 15.0, xtol=1e-12)
    gamma = (np.logaddexp(log_s, np.log(rates[1])) - log_s) / first_change
    log_k = log_s - gamma * moments[0]
    return np.array([np.log(gamma), log_k, log_s])


def predict_effort(parameters, rate, method):
    log_gamma, log_k, log_s = parameters
    log_return = log_s if rate == 0 else np.logaddexp(log_s, np.log(rate))
    moment = (log_return - log_k) / np.exp(log_gamma)
    return float(np.exp(moment) if method == "Power MD" else moment)


def bootstrap_predictions(training_groups, holdout_effort, draws, seed):
    """Resample workers within all four groups; refit on the training groups only.

    Both methods use the same draws. Draws without an interior solution are
    left as NaN rather than redrawn.
    """
    rng = np.random.default_rng(seed)
    predictions = np.full((draws, len(METHODS)), np.nan)
    holdout_means = np.empty(draws)

    for draw in range(draws):
        training_means = [rng.choice(y, size=len(y), replace=True).mean()
                          for y in training_groups]
        holdout_means[draw] = rng.choice(holdout_effort, size=len(holdout_effort),
                                         replace=True).mean()
        for column, method in enumerate(METHODS):
            try:
                parameters = fit_md(training_means, TRAINING_RATES, method)
            except ValueError:
                continue
            predictions[draw, column] = predict_effort(parameters, HOLDOUT_RATE, method)

    return predictions, holdout_means


#%% Load data
start_table("Data checks")
workers = pd.read_stata(DATA_FOLDER / "MTurkCleanedDataShort.dta")
workers["code"] = pd.to_numeric(workers["treatment"].astype(str)).round(1)
if workers["buttonpresses"].isna().any() or (workers["buttonpresses"] <= 0).any():
    raise ValueError("Expected positive, nonmissing effort in the cleaned sample.")

training_groups = [workers.loc[workers["code"] == code, "buttonpresses"].to_numpy(float)
                   for code in TRAINING_CODES]
holdout_effort = workers.loc[workers["code"] == HOLDOUT_CODE, "buttonpresses"].to_numpy(float)
training_n = sum(map(len, training_groups))
holdout_n = len(holdout_effort)

log_print(f"Workers in cleaned sample: {len(workers):,}")
log_print(f"Training: {training_n:,}; holdout: {holdout_n:,}; "
          f"other treatments (not used): {len(workers) - training_n - holdout_n:,}")
log_print(f"Bootstrap draws: {BOOTSTRAP_DRAWS}; seed: {RANDOM_SEED}")

sample_rows = []
groups = training_groups + [holdout_effort]
codes = TRAINING_CODES + [HOLDOUT_CODE]
rates = list(TRAINING_RATES) + [HOLDOUT_RATE]
for code, rate, effort in zip(codes, rates, groups):
    sample_rows.append({"Treatment code": code,
                        "Pay (cents per 100)": round(100 * rate),
                        "Role": "Holdout" if code == HOLDOUT_CODE else "Training",
                        "N": len(effort),
                        "Mean effort": effort.mean(),
                        "SD effort": effort.std(ddof=1),
                        "SE mean": effort.std(ddof=1) / np.sqrt(len(effort))})
log_print("")
log_print(pd.DataFrame(sample_rows).to_string(index=False))


#%% Parameter estimates
start_table("Parameter estimates (training groups)")
training_means = np.array([y.mean() for y in training_groups])

fits = {}
parameter_rows = []
for method in METHODS:
    fits[method] = fit_md(training_means, TRAINING_RATES, method)
    fitted = np.array([predict_effort(fits[method], rate, method) for rate in TRAINING_RATES])
    max_error = np.abs(fitted - training_means).max()
    gamma, k, s = np.exp(fits[method])
    parameter_rows.append({"Method": method, "gamma": gamma, "k": k, "s": s,
                           "Training N": training_n, "Max fit error": max_error})
log_print(pd.DataFrame(parameter_rows).to_string(index=False))


#%% Table E1: predictions for the held-out 10-cent group
start_table("Table E1: predictions for the 10-cent group")
observed_mean = holdout_effort.mean()
point_predictions = np.array([predict_effort(fits[method], HOLDOUT_RATE, method)
                              for method in METHODS])
log_print(f"Observed holdout mean: {observed_mean:.3f} points (N = {holdout_n})")

boot_predictions, boot_observed = bootstrap_predictions(
    training_groups, holdout_effort, BOOTSTRAP_DRAWS, RANDOM_SEED)

# Prediction errors: predicted minus observed (positive = overprediction)
rows = []
for column, method in enumerate(METHODS):
    valid = np.isfinite(boot_predictions[:, column])
    log_print(f"{method} bootstrap: {valid.sum()}/{BOOTSTRAP_DRAWS} draws "
              "with an interior solution")

    predicted = boot_predictions[valid, column]
    errors = predicted - boot_observed[valid]
    pred_low, pred_high = np.quantile(predicted, [0.025, 0.975])
    error_low, error_high = np.quantile(errors, [0.025, 0.975])
    error = point_predictions[column] - observed_mean
    rows.append({"Method": method,
                 "Predicted mean": point_predictions[column],
                 "Observed mean": observed_mean,
                 "Error": error,
                 "Error (%)": 100 * error / observed_mean,
                 "Pred. CI low": pred_low, "Pred. CI high": pred_high,
                 "Error CI low": error_low, "Error CI high": error_high})
save_table(pd.DataFrame(rows), "table_e1_predictions", "Table E1: predictions and 95% CIs")

# Difference between the two methods, paired over the same draws
both_valid = np.isfinite(boot_predictions).all(axis=1)
difference = boot_predictions[both_valid, 0] - boot_predictions[both_valid, 1]
diff_low, diff_high = np.quantile(difference, [0.025, 0.975])
log_print(f"\nPower minus exponential: {point_predictions[0] - point_predictions[1]:.3f} points, "
          f"95% CI [{diff_low:.3f}, {diff_high:.3f}] ({both_valid.sum()} paired draws)")

log_print(f"\nOutputs saved to {OUTPUT_FOLDER}")
