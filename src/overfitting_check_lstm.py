"""
Hyperparameter-sensitivity / overfitting check for the LSTM in robust_check_lstm.py.

robust_check_lstm.py selects hyperparameters with Optuna on the TRAINING set and
then reports test scores for that one configuration. That leaves open the
question of whether the reported test performance is an artefact of the
selection itself.

This script answers it by drawing N_RANDOM hyperparameter combinations uniformly
from the *same* search space Optuna explores, training each on the same training
set and scoring each on the same held-out test set. It reports

  * the distribution of test ROC AUC, sensitivity @ 80% specificity and
    sensitivity @ 90% specificity across those random combinations, and
  * where the Bayes-selected configuration falls inside each distribution.

If the Bayes-selected point sits within the bulk of the random distribution
rather than out in its tail, the selection bought little, so the reported test
performance is not a selection artefact. The per-run train-vs-test gaps, also
recorded, make the same point for the model fitting itself.

Usage:
    python overfitting_check_lstm.py               # one fixed split (default)
    python overfitting_check_lstm.py --vary-split  # a different split per combination
"""

import argparse
import os

import pandas as pd
import numpy as np
from rnn_class import VectorizeData, LSTM, sort_batch
from make_input import make_input
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import torch.nn.functional as F
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_RANDOM = 100          # number of random hyperparameter combinations
N_BAYES_TRIALS = 30     # matches robust_check_lstm.py
SPLIT_SEED = 0          # matches seed 0 of robust_check_lstm.py's simulation
RESULTS_DIR = "../results"

dir = "../data/LungCA_smoking_diabetes_upd.xlsx"
data = pd.read_excel(dir)

df = data.sort_values(['VOL IDs FOR EXPERIMENTAL PURPOSES','Age @ sample (yrs, 2 decimal)']).groupby('VOL IDs FOR EXPERIMENTAL PURPOSES').head(10)
df = df.reset_index(drop=True)

nrow = df.shape[0]

markers = df.columns.to_list()[28:]

df.drop(columns=['CA19-9', 'CEA'], inplace=True)

"""row check"""
NA_list = []
rows_to_remove = []
for i in range(nrow):
    marker_epr = df.iloc[i].tolist()[28:]
    num_NA = np.sum([pd.isna(elt) for elt in marker_epr])
    if num_NA > 50:
        rows_to_remove.append(i)
    NA_list.append(num_NA)

df.drop(rows_to_remove, axis=0, inplace=True)

IDs = list(df["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i = df.loc[df["VOL IDs FOR EXPERIMENTAL PURPOSES"] == idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df["Age @ sample (yrs, 2 decimal)"] = times

markers = df.columns.to_list()[28:]
data_to_impute = df[markers]

# Create and fit the imputer
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data_array = imputer.fit_transform(data_to_impute)

# Convert back to DataFrame to preserve column names
imputed_data = pd.DataFrame(imputed_data_array, columns=markers, index=df.index)

# Replace original columns with imputed data
df[markers] = imputed_data

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/10*7)

selected_features = ['CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14', 'CPE']
selected_features = ['CEACAM5', 'MUC-16']   # as in robust_check_lstm.py


def make_split(seed_num):
    """Reproduce robust_check_lstm.py's train/test split for a given seed."""
    np.random.seed(seed_num)
    train_idx = list(np.random.choice(IDs_new, train_size, replace=False))
    test_idx = list(set(IDs_new) - set(train_idx))

    columns = ["VOL IDs FOR EXPERIMENTAL PURPOSES",
               "Age @ sample (yrs, 2 decimal)",
               "CASE CONTROL STATUS"] + selected_features

    df_train = df[df['VOL IDs FOR EXPERIMENTAL PURPOSES'].isin(train_idx)][columns]
    df_test = df[df['VOL IDs FOR EXPERIMENTAL PURPOSES'].isin(test_idx)][columns]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_data, train_times, train_targets = make_input(df_train, selected_features)
    test_data, test_times, test_targets = make_input(df_test, selected_features)

    ds_train = VectorizeData(train_data, train_times, train_targets)
    ds_test = VectorizeData(test_data, test_times, test_targets)
    return ds_train, ds_test, len(train_data), len(test_data)


"""Search space -- identical to the one Optuna explores in robust_check_lstm.py."""
LSTM_UNITS = [8, 12, 16, 20, 24, 28, 32]
BATCH_SIZES = [8, 12, 16, 20]


def sample_params(rng):
    """Draw one hyperparameter combination uniformly from the Optuna search space."""
    return {
        "lstm_units": int(rng.choice(LSTM_UNITS)),
        "dropout_rate": float(rng.uniform(0.0, 0.4)),
        "batch_size": int(rng.choice(BATCH_SIZES)),
        # log-uniform on [1e-3, 1e-2], matching suggest_float(..., log=True)
        "learning_rate": float(10 ** rng.uniform(np.log10(1e-3), np.log10(1e-2))),
        "epochs": int(rng.integers(1, 21)),   # suggest_int(1, 20) is inclusive
    }


def score(y_true, y_prob, target_specificity=0.8, target_specificity2=0.9):
    """ROC AUC and sensitivity at two target specificities (same convention as robust_check_lstm.py)."""
    roc_auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - target_specificity))
    idx2 = np.argmin(np.abs(specificity - target_specificity2))
    return roc_auc, tpr[idx], tpr[idx2]


def fit_and_score(params, ds_train, ds_test, n_train, n_test, torch_seed):
    """Train one LSTM with `params` and score it on both the training and test sets."""
    torch.manual_seed(torch_seed)

    model = LSTM(input_size=len(selected_features),
                 hidden_size=params["lstm_units"],
                 num_layers=1,
                 output_size=2,
                 dropout=params["dropout_rate"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    train_dl = DataLoader(ds_train, batch_size=params["batch_size"])

    model.train()
    for epoch in range(params["epochs"]):
        for batch_X, batch_y, lengths in train_dl:
            batch_X, batch_y, lengths = sort_batch(batch_X, batch_y, lengths)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X, lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for tag, ds, n in (("train", ds_train, n_train), ("test", ds_test, n_test)):
            for batch_X, batch_y, lengths in DataLoader(ds, n):
                batch_X, batch_y, lengths = sort_batch(batch_X, batch_y, lengths)
                outputs = model(batch_X.to(device), lengths)
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs.cpu().numpy().squeeze()
                auc, sens80, sens90 = score(batch_y, outputs[:, 1])
                out[f"{tag}_roc_auc"] = auc
                out[f"{tag}_sens_spec80"] = sens80
                out[f"{tag}_sens_spec90"] = sens90
    return out


def run_random_combo(i, vary_split):
    """One random hyperparameter combination, trained and scored end to end."""
    split_seed = i if vary_split else SPLIT_SEED
    ds_train, ds_test, n_train, n_test = make_split(split_seed)

    rng = np.random.default_rng(1000 + i)
    params = sample_params(rng)
    res = fit_and_score(params, ds_train, ds_test, n_train, n_test, torch_seed=1000 + i)

    return {"combo": i, "split_seed": split_seed, **params, **res}


def run_bayes_reference(split_seed):
    """Reproduce robust_check_lstm.py: tune on the training set, then score on the test set."""
    ds_train, ds_test, n_train, n_test = make_split(split_seed)

    def objective(trial):
        params = {
            "lstm_units": trial.suggest_categorical("lstm_units", LSTM_UNITS),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.4),
            "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 1, 20),
        }
        res = fit_and_score(params, ds_train, ds_test, n_train, n_test,
                            torch_seed=trial.number)
        return res["train_roc_auc"]   # selection uses the training set only

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=split_seed))
    study.optimize(objective, n_trials=N_BAYES_TRIALS)

    best_params = study.best_params
    res = fit_and_score(best_params, ds_train, ds_test, n_train, n_test,
                        torch_seed=split_seed)
    return {"combo": "bayes", "split_seed": split_seed, **best_params, **res}


METRICS = [
    ("roc_auc", "ROC AUC"),
    ("sens_spec80", "Sensitivity @ 80% specificity"),
    ("sens_spec90", "Sensitivity @ 90% specificity"),
]


def summarise(results, bayes):
    """Print the distribution of each test metric and locate the tuned configuration in it."""
    paired = len(bayes) > 1   # one Bayes run per split, so runs can be compared pairwise

    print(f"\n{'='*76}")
    print(f"Test-set score distribution over {len(results)} random hyperparameter combinations")
    print(f"{'='*76}")

    for key, label in METRICS:
        vals = results[f"test_{key}"].values
        bvals = bayes[f"test_{key}"].values
        print(f"\n{label}")
        print(f"  random combos : mean {vals.mean():.3f}  sd {vals.std(ddof=1):.3f}  "
              f"median {np.median(vals):.3f}")
        print(f"                  min {vals.min():.3f}  "
              f"2.5% {np.percentile(vals, 2.5):.3f}  "
              f"97.5% {np.percentile(vals, 97.5):.3f}  max {vals.max():.3f}")
        if paired:
            diff = bvals - vals
            print(f"  Bayes-tuned   : mean {bvals.mean():.3f}  sd {bvals.std(ddof=1):.3f}  "
                  f"median {np.median(bvals):.3f}")
            print(f"  paired difference (Bayes - random), same split: "
                  f"mean {diff.mean():+.3f}  sd {diff.std(ddof=1):.3f}  "
                  f"median {np.median(diff):+.3f}")
            print(f"  Bayes beats a random draw on {100*(diff > 0).mean():.0f}% of splits")
        else:
            b = bvals[0]
            pct = (vals < b).mean() * 100
            print(f"  Bayes-selected: {b:.3f}   (percentile {pct:.1f} of the random distribution)")
            print(f"  gain over the random median: {b - np.median(vals):+.3f}")

    print(f"\n{'-'*76}")
    print("Train-vs-test gap (train minus test)")
    print(f"{'-'*76}")
    for key, label in METRICS:
        gap = results[f"train_{key}"].values - results[f"test_{key}"].values
        bgap = bayes[f"train_{key}"].values - bayes[f"test_{key}"].values
        bayes_col = (f"Bayes mean {bgap.mean():+.3f} (sd {bgap.std(ddof=1):.3f})" if paired
                     else f"Bayes {bgap[0]:+.3f}")
        print(f"  {label:<32} random mean {gap.mean():+.3f} "
              f"(sd {gap.std(ddof=1):.3f})   {bayes_col}")


def plot(results, bayes, out_path):
    """One panel per metric: the random-combination distribution, with the tuned model marked."""
    paired = len(bayes) > 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, label) in zip(axes, METRICS):
        vals = results[f"test_{key}"].values
        bvals = bayes[f"test_{key}"].values
        bins = np.histogram_bin_edges(np.concatenate([vals, bvals]), bins=20)
        ax.hist(vals, bins=bins, color="#7fa8d1", edgecolor="white",
                label=f"random combinations (median {np.median(vals):.3f})")
        if paired:
            ax.hist(bvals, bins=bins, color="#c0392b", alpha=0.45, edgecolor="white",
                    label=f"Bayes-tuned (median {np.median(bvals):.3f})")
        else:
            ax.axvline(np.median(vals), color="#33475b", linewidth=1.6,
                       label=f"random median = {np.median(vals):.3f}")
            ax.axvline(bvals[0], color="#c0392b", linestyle="--", linewidth=1.8,
                       label=f"Bayes-selected = {bvals[0]:.3f}")
        ax.set_xlabel(f"test {label}")
        ax.set_ylabel("count")
        ax.set_title(label)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Test scores over {len(results)} random hyperparameter combinations", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vary-split", action="store_true",
                        help="use a different train/test split per combination "
                             "(default: one fixed split, so only the hyperparameters vary)")
    parser.add_argument("--n-random", type=int, default=N_RANDOM,
                        help=f"number of random hyperparameter combinations (default {N_RANDOM})")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "varysplit" if args.vary_split else f"split{SPLIT_SEED}"

    print(f"Features: {selected_features}")
    print(f"Split: {'one per combination' if args.vary_split else f'fixed (seed {SPLIT_SEED})'}")
    print(f"Drawing {args.n_random} random hyperparameter combinations...")

    rows = Parallel(n_jobs=-1, verbose=1)(
        delayed(run_random_combo)(i, args.vary_split) for i in range(args.n_random)
    )
    results = pd.DataFrame(rows)

    # The Bayesian reference has to span the same splits as the random arm, otherwise
    # the two distributions are not comparable.
    split_seeds = sorted(results["split_seed"].unique())
    print(f"Running the Bayesian reference ({N_BAYES_TRIALS} trials on "
          f"{len(split_seeds)} split{'s' if len(split_seeds) > 1 else ''})...")
    bayes = pd.DataFrame(Parallel(n_jobs=-1, verbose=1)(
        delayed(run_bayes_reference)(s) for s in split_seeds
    ))

    out_csv = f"{RESULTS_DIR}/overfitting_check_lstm_{suffix}.csv"
    pd.concat([results, bayes], ignore_index=True).to_csv(out_csv, index=False)
    print(f"Per-run scores written to {out_csv}")

    summarise(results, bayes)
    plot(results, bayes, f"{RESULTS_DIR}/overfitting_check_lstm_{suffix}.png")
